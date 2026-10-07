"""
AlGaNDevice: user-facing class for heterostructure simulation -- wurtzite
nitrides (the original scope, hence the name) and zincblende arsenides /
phosphides. `Device` is the same class under a material-neutral name.

Usage example:
    from devices.layer import AbruptLayer, GradedLayer, Contact
    from devices.device import AlGaNDevice

    layers = [
        AbruptLayer(x_Al=0.0, thickness_nm=200, n_doping=1e17),
        AbruptLayer(x_Al=0.15, thickness_nm=10),
        AbruptLayer(x_Al=0.0, thickness_nm=3),
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        AbruptLayer(x_Al=0.0, thickness_nm=100, p_doping=3e17),
    ]
    contacts = [
        Contact(position='bottom', contact_type='ohmic',    metal='Ti'),
        Contact(position='top',    contact_type='schottky', metal='Ni'),
    ]
    device = AlGaNDevice(layers=layers, contacts=contacts, T=300)
    result = device.solve(V_applied=0.0)
    result_bias = device.solve(V_applied=3.0)
"""

from __future__ import annotations

from typing import Callable, List, Optional
import numpy as np

from devices.layer import AbruptLayer, GradedLayer, Contact
from devices.grid_builder import build_grid, GridData
from physics.self_consistent import solve_self_consistent, SolverResult


class AlGaNDevice:
    """
    III-V heterostructure device simulator.

    The device is defined entirely by an ordered list of layers (bottom → top,
    i.e. substrate side first) and two metal contacts. The layers are all
    wurtzite nitrides (AlGaN, InGaN, InAlGaN) or all zincblende arsenides /
    phosphides (AlGaAs, InGaAs, InGaP, InGaAsP, ...) -- see devices.layer.

    Parameters
    ----------
    layers   : ordered list of AbruptLayer or GradedLayer
    contacts : list of exactly two Contact objects
    T        : lattice temperature [K]
    dx_nm    : spatial grid resolution [nm]
    """

    def __init__(
        self,
        layers: List,
        contacts: List[Contact],
        T: float = 300.0,
        dx_nm: float = 0.1,
        include_spontaneous_polarization: bool = True,
        include_strain_band_shift: bool = True,
        polarity: str = 'metal',
        polarization_model: str = 'ambacher2002',
        recombination: Optional[dict] = None,
    ) -> None:
        self.layers   = layers
        self.contacts = contacts
        self.T        = T
        self.dx_nm    = dx_nm
        self.include_spontaneous_polarization = include_spontaneous_polarization
        self.include_strain_band_shift = include_strain_band_shift
        self.polarity = polarity                      # 'metal' | 'N'
        self.polarization_model = polarization_model  # 'ambacher2002' | 'dreyer2016'
        # Optional overrides of the material recombination coefficients:
        # {'tau_n', 'tau_p' [s], 'B_rad' [cm^3/s], 'C_n', 'C_p' [cm^6/s]}
        self.recombination = recombination
        self._grid: Optional[GridData] = None
        self.last_sweep_failed_voltages: List[float] = []

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def build_grid(self) -> GridData:
        """Build (or rebuild) the spatial grid.  Cached after first call."""
        self._grid = build_grid(
            self.layers, self.contacts, T=self.T, dx_nm=self.dx_nm,
            include_spontaneous_polarization=self.include_spontaneous_polarization,
            include_strain_band_shift=self.include_strain_band_shift,
            polarity=self.polarity,
            polarization_model=self.polarization_model,
            recombination=self.recombination,
        )
        return self._grid

    @property
    def grid(self) -> GridData:
        if self._grid is None:
            self.build_grid()
        return self._grid

    @property
    def total_series_resistance(self) -> float:
        """Sum of every physical layer's own `series_resistance` [Ohm*cm^2]
        -- resistances of layers stacked in series add, so this replaces a
        single whole-device R_series knob. A layer with no value set
        (default 0.0) contributes nothing."""
        return sum(getattr(l, 'series_resistance', 0.0) for l in self.layers
                   if isinstance(l, (AbruptLayer, GradedLayer)))

    # ------------------------------------------------------------------
    # Single-bias solve
    # ------------------------------------------------------------------

    def solve(
        self,
        V_applied: float = 0.0,
        quantum: bool = True,
        method: str = 'newton',
        n_states_e: int = 6,
        n_states_h: int = 6,
        max_iter: int = 400,
        tol: float = 1e-6,
        alpha: float = 0.15,
        verbose: bool = False,
        **kwargs
    ):
        """
        Run the self-consistent Schrödinger-Poisson solver.

        Parameters
        ----------
        V_applied  : applied voltage [V] (positive = forward bias)
        quantum    : use Schrödinger equation for carrier density
        method     : 'newton' (default) or 'pinn' (PyTorch PINN solver)
        n_states_e : number of electron subbands (quantum mode)
        n_states_h : number of hole subbands (quantum mode)
        max_iter   : max SP iterations
        tol        : convergence threshold on phi [V]
        alpha      : mixing damping factor (0 < alpha ≤ 1)
        verbose    : print iteration convergence
        log_fn     : callable(str) used for verbose output instead of the
                     built-in print when given (see solve_self_consistent);
                     forwarded via **kwargs

        Returns
        -------
        SolverResult or PINNResult with all band diagram quantities
        """
        if method.lower() == 'pinn':
            from physics.pinn_solver import solve_pinn
            pinn_kwargs = {k: v for k, v in kwargs.items() if k in [
                'n_colloc', 'n_hidden', 'n_neurons', 'n_epochs_adam', 
                'n_epochs_lbfgs', 'lr_adam', 'lr_lbfgs', 'log_fn',
                'w_poisson', 'w_cont_n', 'w_cont_p'
            ]}
            return solve_pinn(self.grid, V_applied=V_applied, verbose=verbose, **pinn_kwargs)

        # A caller-supplied R_series (e.g. a test isolating boundary-condition
        # behavior from the series-resistance IR-drop mechanism) wins over
        # the device's own total_series_resistance -- previously this was
        # always passed positionally, so any R_series in **kwargs collided
        # with it (TypeError: got multiple values for keyword argument).
        kwargs.setdefault('R_series', self.total_series_resistance)
        from physics.self_consistent import solve_self_consistent
        return solve_self_consistent(
            self.grid,
            V_applied=V_applied,
            quantum=quantum,
            n_states_e=n_states_e,
            n_states_h=n_states_h,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Voltage sweep
    # ------------------------------------------------------------------

    def sweep_voltage(
        self,
        V_start: float,
        V_stop: float,
        n_steps: int = 10,
        quantum: bool = True,
        n_states_e: int = 6,
        n_states_h: int = 6,
        tol: float = 1e-6,
        verbose: bool = False,
        alpha: float = 0.3,
        max_iter: int = 200,
        log_fn: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        flat_qfl: bool = False,
    ) -> List[SolverResult]:
        """
        Solve at multiple bias voltages, each one independently from
        scratch (no continuation/carried-over initial guess between
        points, no bisection retry) -- see solve_ramped's docstring for
        why: continuation was found to converge coupled biased solves to
        spurious, masked-but-"converged" quasi-Fermi splits instead of the
        correct one. Every point here gets solve()'s own fresh
        charge-neutral initial guess and PTC-direct's pseudo-time
        continuation for high-bias robustness instead of bias-value
        ramping. Voltages that fail to converge are recorded in
        `self.last_sweep_failed_voltages` (and their entry in the returned
        list still carries whatever that attempt produced, converged=False)
        rather than retried.

        log_fn : callable(str) used for verbose output instead of the
                 built-in print when given (see solve_self_consistent).
        """
        _log = log_fn if log_fn is not None else print
        voltages = np.linspace(V_start, V_stop, n_steps)

        results: List[SolverResult] = []
        failed_voltages: List[float] = []

        for V_target in voltages:
            V_target = float(V_target)
            if verbose:
                _log(f"Solving V = {V_target:.3f} V (fresh, no continuation) ...")
            res = self.solve(
                V_applied=V_target,
                quantum=quantum,
                n_states_e=n_states_e,
                n_states_h=n_states_h,
                max_iter=max_iter,
                tol=tol,
                alpha=alpha,
                verbose=verbose,
                log_fn=log_fn,
                cancel_check=cancel_check,
                flat_qfl=flat_qfl,
            )
            if not res.converged:
                failed_voltages.append(V_target)
            results.append(res)

        self.last_sweep_failed_voltages = failed_voltages
        if failed_voltages and verbose:
            _log(f"Warning: sweep_voltage failed to converge at: {failed_voltages}")

        return results

    # ------------------------------------------------------------------
    # Single-bias solve, direct (no continuation)
    # ------------------------------------------------------------------

    def solve_ramped(
        self,
        V_applied: float,
        quantum: bool = True,
        n_states_e: int = 6,
        n_states_h: int = 6,
        max_iter: int = 300,
        tol: float = 1e-6,
        alpha: float = 0.15,
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        flat_qfl: bool = False,
        **_ignored,
    ) -> SolverResult:
        """
        Solve at V_applied directly, from scratch, in one shot -- kept as a
        thin alias of solve() (name retained for existing callers) now that
        bias-value ramping/continuation has been removed. Continuation used
        to exist because a one-shot jump was thought to be what defeated
        convergence at high bias; it was actually the opposite problem: the
        coupled Newton-Krylov solve, when seeded from a carried-forward
        continuation state, was converging to a spurious nearby fixed point
        with a genuinely small *scaled* residual that masks an unphysical
        quasi-Fermi split (confirmed empirically -- a direct one-shot solve
        at a given V_applied recovers the correct textbook split, while
        reaching the same V_applied via multi-step ramping did not). See
        physics.coupled_solver's module docstring and
        [[bug-quasi-fermi-pinning]]. High-bias robustness now comes from
        PTC-direct's own pseudo-time continuation (see solve_coupled_dd),
        not from bias-value ramping.

        `**_ignored` absorbs now-meaningless legacy kwargs (e.g.
        `ramp_steps`) from existing callers without erroring.
        """
        return self.solve(
            V_applied=V_applied, quantum=quantum,
            n_states_e=n_states_e, n_states_h=n_states_h,
            max_iter=max_iter, tol=tol, alpha=alpha, verbose=verbose,
            log_fn=log_fn, cancel_check=cancel_check, flat_qfl=flat_qfl,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_thickness_nm(self) -> float:
        return sum(getattr(l, 'thickness_nm', 0.0) for l in self.layers)

    def __repr__(self) -> str:
        n_layers = len(self.layers)
        total_nm = self.total_thickness_nm
        return (f"AlGaNDevice({n_layers} layers, "
                f"{total_nm:.1f} nm total, T={self.T} K)")


# Material-neutral name for the same class.
Device = AlGaNDevice

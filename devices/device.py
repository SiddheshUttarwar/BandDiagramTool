"""
AlGaNDevice: user-facing class for AlGaN heterostructure simulation.

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

from typing import List, Optional
import numpy as np

from devices.layer import AbruptLayer, GradedLayer, Contact
from devices.grid_builder import build_grid, GridData
from physics.self_consistent import solve_self_consistent, SolverResult


class AlGaNDevice:
    """
    AlxGa(1-x)N heterostructure device simulator.

    The device is defined entirely by an ordered list of layers (bottom → top,
    i.e. substrate side first) and two metal contacts.  All layers are AlGaN
    with x_Al ∈ [0, 1]; x_Al = 0 is GaN, x_Al = 1 is AlN.

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
    ) -> None:
        self.layers   = layers
        self.contacts = contacts
        self.T        = T
        self.dx_nm    = dx_nm
        self._grid: Optional[GridData] = None
        self.last_sweep_failed_voltages: List[float] = []

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def build_grid(self) -> GridData:
        """Build (or rebuild) the spatial grid.  Cached after first call."""
        self._grid = build_grid(self.layers, self.contacts,
                                T=self.T, dx_nm=self.dx_nm)
        return self._grid

    @property
    def grid(self) -> GridData:
        if self._grid is None:
            self.build_grid()
        return self._grid

    # ------------------------------------------------------------------
    # Single-bias solve
    # ------------------------------------------------------------------

    def solve(
        self,
        V_applied: float = 0.0,
        R_series: float = 1e-3,
        quantum: bool = True,
        n_states_e: int = 6,
        n_states_h: int = 6,
        max_iter: int = 300,
        tol: float = 1e-6,
        alpha: float = 0.15,
        verbose: bool = False,
        **kwargs
    ) -> SolverResult:
        """
        Run the self-consistent Schrödinger-Poisson solver.

        Parameters
        ----------
        V_applied  : applied voltage [V] (positive = forward bias)
        R_series   : series resistance [Ohm*cm^2]
        quantum    : use Schrödinger equation for carrier density
        n_states_e : number of electron subbands (quantum mode)
        n_states_h : number of hole subbands (quantum mode)
        max_iter   : max SP iterations
        tol        : convergence threshold on phi [V]
        alpha      : mixing damping factor (0 < alpha ≤ 1)
        verbose    : print iteration convergence

        Returns
        -------
        SolverResult with all band diagram quantities
        """
        return solve_self_consistent(
            self.grid,
            V_applied=V_applied,
            R_series=R_series,
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
        R_series: float = 0.0,
        quantum: bool = True,
        verbose: bool = False,
        alpha: float = 0.3,
        max_iter: int = 200,
        min_step_fraction: float = 1.0 / 16.0,
    ) -> List[SolverResult]:
        """
        Solve at multiple bias voltages and return the list of results.

        Each requested voltage is reached by continuation from the last
        converged voltage. If a step fails to converge, it is bisected and
        retried (down to `min_step_fraction` of the nominal step) before the
        requested voltage is given up on and recorded as failed — this
        replaces having to hand-pick step sizes for devices that are hard
        to converge at high bias. Voltages that could not be reached are
        listed in `self.last_sweep_failed_voltages`, and their entry in the
        returned list carries whatever the last (non-converged) attempt at
        that target voltage produced (`result.converged is False`).
        """
        voltages = np.linspace(V_start, V_stop, n_steps)
        nominal_step = abs(voltages[1] - voltages[0]) if n_steps > 1 else abs(V_stop - V_start)
        min_step = max(nominal_step * min_step_fraction, 1e-6)

        results: List[SolverResult] = []
        failed_voltages: List[float] = []
        phi_init = None
        Efn_init = None
        Efp_init = None
        V_current = float(voltages[0])

        for V_target in voltages:
            V_target = float(V_target)
            step = V_target - V_current
            phi_i, Efn_i, Efp_i = phi_init, Efn_init, Efp_init
            res = None

            while True:
                V_try = V_current + step
                if verbose:
                    print(f"Solving V = {V_try:.3f} V (step = {step:+.3f}) ...")
                res = self.solve(
                    V_applied=V_try,
                    R_series=R_series,
                    quantum=quantum,
                    max_iter=max_iter,
                    tol=1e-6,
                    alpha=alpha,
                    phi_init=phi_i,
                    Efn_init=Efn_i,
                    Efp_init=Efp_i,
                    verbose=verbose,
                )

                if res.converged:
                    V_current = V_try
                    phi_i, Efn_i, Efp_i = res.phi, res.Efn, res.Efp
                    if abs(V_try - V_target) < 1e-9:
                        break   # reached this sweep target
                    # Advance toward the target without overshooting, keeping
                    # the step size that just worked.
                    remaining = V_target - V_current
                    step = (1.0 if remaining >= 0 else -1.0) * min(abs(step), abs(remaining))
                    continue

                # Failed: bisect the step, unless already at the floor.
                if abs(step) / 2.0 < min_step:
                    failed_voltages.append(V_target)
                    break
                step = step / 2.0

            phi_init, Efn_init, Efp_init = phi_i, Efn_i, Efp_i
            results.append(res)

        self.last_sweep_failed_voltages = failed_voltages
        if failed_voltages and verbose:
            print(f"Warning: sweep_voltage failed to converge at: {failed_voltages}")

        return results

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_thickness_nm(self) -> float:
        return sum(l.thickness_nm for l in self.layers)

    def __repr__(self) -> str:
        n_layers = len(self.layers)
        total_nm = self.total_thickness_nm
        return (f"AlGaNDevice({n_layers} layers, "
                f"{total_nm:.1f} nm total, T={self.T} K)")

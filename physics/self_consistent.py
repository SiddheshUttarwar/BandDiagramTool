"""
Self-consistent Schrödinger-Poisson solver.

Algorithm:
  1. Build grid (caller supplies GridData)
  2. Initialise phi from depletion-approximation Poisson (fixed ionised charges
     + polarisation; no free carriers assumed).  This gives the correct
     qualitative band-bending shape so the Newton iterations need far fewer
     steps than starting from a flat/linear phi.
  3. Gummel loop (outer):
     a. Ec(x) = Ec0 - phi,  Ev(x) = Ev0 - phi
     b. [quantum=True]  Solve Schrödinger → (E_e, psi_e), (E_h, psi_h),
        restricted to the auto-detected "quantum region" (the undoped
        span of the device, e.g. QW/barriers/EBL — see
        `_detect_quantum_region`). Solving over the *entire* device
        including thick doped bulk contact/buffer layers would let those
        layers act as enormous, physically irrelevant low-energy "wells"
        of their own (confinement energy ~ 1/width²), swamping the
        actually-intended confined states.
     c. n(x), p(x): classical Fermi-Dirac everywhere outside the quantum
        region (bulk contacts are reservoirs, not confined), quantum
        (subband) density inside it, when quantum=True; purely classical
        when quantum=False.
     d. Newton-Raphson Poisson step (one linearised Poisson solve):
            [A_pos + D] * phi_new = D*phi + rho/eps0
        where D_i = q*(n_i+p_i)/(eps0*kBT)   [carrier screening in Jacobian]
        The matrix is positive-definite for any n,p ≥ 0.
     e. Safeguarded, adaptive Anderson AA-I mixing on (phi, phi_new):
        the residual computed this iteration reflects how well the
        *previous* step did, so if it grew, alpha is shrunk, the Anderson
        history is dropped, and this step falls back to plain damping;
        otherwise alpha grows (capped at alpha_max) and Anderson mixing is
        used. This lets AA accelerate well-behaved regions while never
        letting a bad extrapolation compound in the historically fragile
        high-polarization / high-bias cases.
     f. Check  max|phi_new - phi| < tol
  4. Return SolverResult

Two modes:
  quantum=False  →  classical Fermi-Dirac only        (fast)
  quantum=True   →  full Schrödinger-Poisson           (slower, captures quantisation)
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import Optional

from devices.grid_builder import GridData
from physics.fermi_dirac import (
    electron_density, hole_density,
    quantum_electron_density, quantum_hole_density,
    Efn_from_n, Efp_from_p,
)
from physics.poisson import solve_poisson, solve_poisson_newton, electric_field
from physics.schrodinger import solve_schrodinger, hole_potential
from physics.drift_diffusion import solve_continuity_electron, solve_continuity_hole, compute_recombination
from physics.optical import ground_state_transition, dominant_transition
from physics.constants import q as _q, kB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anderson AA-I mixer
# ---------------------------------------------------------------------------

# An Anderson-extrapolated step is rejected (falls back to plain damping)
# if its magnitude exceeds this multiple of what the plain damped step
# would have been — the signature of a runaway extrapolation.
_AA_STEP_SAFEGUARD = 5.0


class _AndersonMixer:
    """
    Anderson Acceleration (type-I) for fixed-point iteration  x = g(x).

    Reference: Walker & Ni, SINUM 49(4), 2011.

    At each step the mixer stores the history of (x_k, f_k) pairs where
    f_k = g(x_k) - x_k (the Newton-step correction).  It then solves the
    small least-squares problem

        min_c  ||f_k - ΔF · c||²

    and returns the extrapolated next iterate

        x_{k+1} = x_k + f_k - (ΔX + ΔF) · c

    The damping factor alpha scales f_k before use (alpha=1 → standard AA).
    """

    def __init__(self, m: int = 5, alpha: float = 0.5):
        self.m     = m
        self.alpha = alpha
        self._x_hist: list[np.ndarray] = []
        self._f_hist: list[np.ndarray] = []

    def reset(self) -> None:
        self._x_hist.clear()
        self._f_hist.clear()

    def mix(self, phi: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
        """
        Given the current iterate phi and the Newton-step result phi_new,
        return the Anderson-mixed next iterate.
        """
        f = phi_new - phi            # raw Newton correction
        f_damp = self.alpha * f      # damped correction

        self._x_hist.append(phi.copy())
        self._f_hist.append(f_damp.copy())

        k = len(self._x_hist)
        m_k = min(k - 1, self.m)

        if m_k == 0:
            # No history: plain damped Newton step
            return phi + f_damp

        # Build difference matrices ΔF and ΔX  (shape N × m_k)
        # Column j = (most-recent) - (j+1 steps ago)
        dF = np.column_stack([
            self._f_hist[-1] - self._f_hist[-(j + 2)]
            for j in range(m_k)
        ])
        dX = np.column_stack([
            self._x_hist[-1] - self._x_hist[-(j + 2)]
            for j in range(m_k)
        ])

        # Tikhonov-regularised least-squares:  min ||f_damp - dF·c||² + reg·||c||²
        # Consecutive corrections become nearly collinear as the outer loop
        # stalls or oscillates, which makes the raw normal equations
        # ill-conditioned and can produce huge coefficients (and therefore
        # a huge, wrong extrapolated step) even though the residual looks
        # unremarkable. Regularising is the standard fix for AA in stiff
        # nonlinear systems; reg is scaled to dF's own magnitude so it's
        # negligible when dF is well-conditioned.
        reg = 1e-8 * max(float(np.sum(dF * dF)), 1e-300)
        dFtdF = dF.T @ dF + reg * np.eye(m_k)
        c = np.linalg.solve(dFtdF, dF.T @ f_damp)

        # AA-I update
        x_next = phi + f_damp - (dX + dF) @ c

        # Trim history to m+1 entries
        if len(self._x_hist) > self.m + 1:
            self._x_hist.pop(0)
            self._f_hist.pop(0)

        return x_next


# ---------------------------------------------------------------------------
# Quantum-region detection
# ---------------------------------------------------------------------------

# Doping below this (donors or acceptors) counts as "undoped" for the
# purpose of finding the active/quantum region — well below any
# intentional doping level, comfortably above zero/numerical noise.
_UNDOPED_FLOOR_CM3 = 1e15


def _detect_quantum_region(g: GridData, margin_nm: float = 10.0) -> tuple[int, int]:
    """
    Heuristically restrict the Schrödinger solve to the device's undoped
    span (QW/barriers/EBL — the layers a designer leaves undoped precisely
    because that's the active region), rather than the entire device.

    Solving Schrödinger over the whole device — including thick doped bulk
    contact/buffer layers, which are reservoirs, not confinement regions —
    lets those layers act as enormous, physically irrelevant low-energy
    "wells" of their own (confinement energy scales as 1/width², so a
    200nm buffer swamps a 3nm quantum well's states by many orders of
    magnitude). This finds the bounding box of undoped grid points and
    pads it by `margin_nm` on each side, so the artificial hard walls this
    introduces sit a little way into the confining barriers rather than
    exactly at the doping step.

    Returns (i0, i1) grid indices (i1 exclusive). Falls back to the full
    grid [0, N) if there's no clearly undoped region (e.g. every layer is
    doped) — the same behaviour as before this restriction existed.
    """
    undoped = (g.ND < _UNDOPED_FLOOR_CM3) & (g.NA < _UNDOPED_FLOOR_CM3)
    idx = np.where(undoped)[0]
    if len(idx) < 2:
        return 0, g.N

    dx_nm = g.dx * 1e9
    margin_pts = max(1, int(round(margin_nm / dx_nm)))
    i0 = max(0, int(idx[0]) - margin_pts)
    i1 = min(g.N, int(idx[-1]) + 1 + margin_pts)
    return i0, i1


def _solve_confined_states(V_eV_full: np.ndarray, m_full: np.ndarray, dx: float,
                            n_states: int, i0: int, i1: int, N: int):
    """
    Solve the Schrödinger equation restricted to grid indices [i0, i1) and
    embed the resulting wavefunctions into full-length (N-point),
    zero-padded arrays, so downstream code (quantum_electron_density etc.)
    can treat them exactly like a full-domain solve.
    """
    E, psi_sub = solve_schrodinger(V_eV_full[i0:i1], m_full[i0:i1], dx, n_states=n_states)
    psi_full = np.zeros((len(E), N))
    psi_full[:, i0:i1] = psi_sub
    return E, psi_full


def _blended_density(quantum_density: np.ndarray, classical_density: np.ndarray,
                      region_mask: np.ndarray) -> np.ndarray:
    """Quantum (subband) density inside the quantum region, classical
    Fermi-Dirac density outside it (bulk contacts are reservoirs, not
    confined states)."""
    return np.where(region_mask, quantum_density, classical_density)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """All outputs from one solve() call."""
    # Grid
    x_nm: np.ndarray          # position [nm]
    x_Al: np.ndarray          # Al composition

    # Band edges [eV]
    Ec: np.ndarray
    Ev: np.ndarray
    Ei: np.ndarray            # intrinsic level = (Ec + Ev)/2 + kBT/2*ln(Nv/Nc)

    # Fermi levels [eV]
    Efn: np.ndarray           # electron quasi-Fermi level
    Efp: np.ndarray           # hole quasi-Fermi level

    # Electrostatic
    phi: np.ndarray           # electrostatic potential [V]
    E_field: np.ndarray       # electric field [V/m]
    F_quasi: np.ndarray       # quasi-electric field from composition gradient [V/m]

    # Carrier densities [cm^-3]
    n: np.ndarray
    p: np.ndarray

    # Polarization
    Psp: np.ndarray
    Ppz: np.ndarray
    P_total: np.ndarray

    # Strain
    eps_xx: np.ndarray
    eps_zz: np.ndarray

    # Wavefunctions (quantum mode)
    E_e: Optional[np.ndarray] = None
    psi_e: Optional[np.ndarray] = None
    E_h: Optional[np.ndarray] = None
    psi_h: Optional[np.ndarray] = None

    # Quantum-Confined Stark Effect (e1-h1 ground-state transition; see
    # physics/optical.py). None when quantum=False or no confined pair
    # was found.
    qcse_transition_eV: Optional[float] = None
    qcse_overlap: Optional[float] = None

    # Highest-overlap (electron subband, hole subband) pair among *all*
    # solved states -- the pair actually most likely to dominate emission.
    # A polarization-induced interface notch (see _detect_quantum_region)
    # can sometimes pull the lowest-energy e1/h1 states to opposite ends of
    # the device rather than into the intended quantum well, in which case
    # this differs from (0, 0) and is the more physically meaningful number.
    qcse_dominant_transition_eV: Optional[float] = None
    qcse_dominant_overlap: Optional[float] = None
    qcse_dominant_pair: Optional[tuple] = None   # (ie, ih)

    # Metadata
    V_applied: float = 0.0
    T: float = 300.0
    converged: bool = True
    n_iterations: int = 0
    interface_indices: list = field(default_factory=list)
    interface_sigmas: list = field(default_factory=list)
    residual_history: list = field(default_factory=list)   # per-iteration max|dphi| [V]
    alpha_history: list = field(default_factory=list)      # per-iteration mixing damping factor

    # Material
    Ec0: np.ndarray = field(default_factory=lambda: np.array([]))
    Ev0: np.ndarray = field(default_factory=lambda: np.array([]))
    ND: np.ndarray  = field(default_factory=lambda: np.array([]))
    NA: np.ndarray  = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_self_consistent(
    grid: GridData,
    V_applied: float = 0.0,
    R_series: float = 0.0,
    quantum: bool = True,
    n_states_e: int = 6,
    n_states_h: int = 6,
    max_iter: int = 200,
    tol: float = 1e-6,
    alpha: float = 0.1,
    alpha_min: float = 0.01,
    alpha_max: Optional[float] = None,
    anderson_m: int = 5,
    verbose: bool = False,
    phi_init: Optional[np.ndarray] = None,
    Efn_init: Optional[np.ndarray] = None,
    Efp_init: Optional[np.ndarray] = None,
) -> SolverResult:
    """
    Run the self-consistent Schrödinger-Poisson solver.

    Uses Newton-Raphson linearisation for the Poisson step:
        [A_pos + D] * phi_new = D*phi + rho/eps0
    where D_ii = q*(n+p)/(eps0*kBT) adds carrier screening to the Jacobian,
    keeping the matrix positive-definite regardless of doping.

    Anderson AA-I mixing (history length anderson_m) accelerates the outer
    Gummel convergence from linear to super-linear. The mixing is
    safeguarded and the damping factor alpha is adaptive: alpha grows
    (capped at alpha_max) after an iteration whose residual improved, and
    shrinks (floored at alpha_min) whenever the residual grows, which also
    resets the Anderson history and falls back to plain damping for that
    step. This removes the need to hand-tune alpha/anderson_m per device.

    Parameters
    ----------
    grid       : GridData from build_grid()
    V_applied  : applied voltage [V] (positive = forward bias)
    quantum    : if True, use Schrödinger equation for carrier density
    n_states_e : electron subbands (quantum mode)
    n_states_h : hole subbands (quantum mode)
    max_iter   : maximum SP iterations
    tol        : convergence threshold on max|Δphi| [V]
    alpha      : initial damping factor applied inside Anderson mixer (0 < alpha ≤ 1)
    alpha_min  : lower bound the adaptive damping factor can shrink to
    alpha_max  : upper bound the adaptive damping factor can grow to. Defaults
                 to `alpha` itself (i.e. adaptation can only recover a
                 damping factor that was shrunk for stability, never exceed
                 the caller's chosen starting point) — this stiff nonlinear
                 system reliably loses stability once alpha grows much past
                 what a device actually needs, so growing beyond the
                 starting alpha is opt-in only.
    anderson_m : Anderson history length (0 = plain damped iteration)
    verbose    : print iteration residuals

    Note on scope: adaptive alpha and safeguarded Anderson mixing are only
    applied to the pure-electrostatic equilibrium loop (V_applied == 0).
    Empirically, once V_applied != 0 the drift-diffusion / quasi-Fermi
    update (Scharfetter-Gummel + Efn_from_n/Efp_from_p) is sensitive to
    iteration-to-iteration *changes* in phi's step size, not just its
    magnitude: even a bounded, patience-gated adaptive alpha (never
    exceeding the fixed value that is known to be stable) reliably drove
    Efn/Efp to unphysical values (tens to hundreds of eV) where the
    original constant-alpha damping stayed bounded. So biased solves use
    plain fixed-alpha damping exactly as before — alpha/alpha_min/alpha_max
    and anderson_m are accepted but not varied mid-solve in that regime.
    Robust bias-point convergence (beyond the voltage-step bisection in
    AlGaNDevice.sweep_voltage) is a known gap left for future work.
    """
    g      = grid
    T      = g.T
    kBT_eV = kB * T / _q
    if alpha_max is None:
        alpha_max = alpha
    # See "Note on scope" above: adaptive alpha / safeguarded AA are only
    # used for the unbiased electrostatic loop; biased solves keep the
    # original fixed-alpha damping because the drift-diffusion coupling is
    # unstable under a varying step size.
    bias_coupled = (V_applied != 0.0)

    # --- Boundary potentials ---
    phi_left  = g.phi_bottom_eq
    V_internal = V_applied
    phi_right = g.phi_top_eq + V_internal

    # --- Quasi-Fermi levels:    # Initial guess for electrostatics
    if phi_init is not None:
        phi = phi_init.copy()
        # MUST update boundaries to current V_applied!
        phi[0] = phi_left
        phi[-1] = phi_right
    else:
        phi = np.zeros(g.N)
        
    if Efn_init is not None:
        Efn = Efn_init.copy()
    else:
        Efn = np.zeros(g.N)
        
    if Efp_init is not None:
        Efp = Efp_init.copy()
    else:
        Efp = np.full(g.N, -V_applied)
        
    # Enforce contact boundary conditions immediately
    Efn[0] = 0.0
    Efn[-1] = -V_applied
    Efp[0] = 0.0
    Efp[-1] = -V_applied

    # --- Initial phi: Hybrid Anchor-and-Smooth Initialization ---
    if phi_init is not None:
        phi = phi_init.copy()
        phi[0] = phi_left
        phi[-1] = phi_right
    else:
        phi_bulk = np.zeros(g.N)
        for i in range(g.N):
            def charge_imbalance(phi_guess):
                Ec_guess = g.Ec0[i] - phi_guess
                Ev_guess = g.Ev0[i] - phi_guess
                # Composition-dependent incomplete ionization
                Ed_i = 0.02 + 0.28 * g.x_Al[i]   # Si donor activation energy
                Nd_plus = g.ND[i] / (1.0 + 2.0 * np.exp(np.clip((Efn[i] - (Ec_guess - Ed_i)) / kBT_eV, -500, 500)))
                Na_minus = g.NA[i] / (1.0 + 4.0 * np.exp(np.clip((Ev_guess + 0.17 - Efp[i]) / kBT_eV, -500, 500)))
                n_guess = electron_density(Ec_guess, Efn[i], g.Nc[i], T)
                p_guess = hole_density(Ev_guess, Efp[i], g.Nv[i], T)
                return (Nd_plus - Na_minus) + p_guess - n_guess
            try:
                phi_bulk[i] = brentq(charge_imbalance, -10.0, 10.0)
            except ValueError:
                logger.warning(
                    "Charge-imbalance brentq failed at grid index %d during "
                    "initial-guess construction; falling back to midgap estimate", i
                )
                phi_bulk[i] = 0.5 * (g.Ec0[i] + g.Ev0[i]) - 0.5 * (Efn[i] + Efp[i])
    
        phi = np.copy(phi_bulk)
    
        # Pin boundaries
        phi[0] = phi_left
        phi[-1] = phi_right
    
        # Pin interfaces dynamically based on polarization
        if hasattr(g, 'interface_indices') and hasattr(g, 'interface_sigmas'):
            for idx, sigma in zip(g.interface_indices, g.interface_sigmas):
                if sigma > 0:
                    phi[idx] = g.Ec0[idx] - Efn[idx] + 0.05
                elif sigma < 0:
                    phi[idx] = g.Ev0[idx] - Efp[idx] - 0.05
    
        # Smooth aggressively
        sigma_pts = max(2, int(5.0e-9 / g.dx))  # ~5 nm screening length
        phi = gaussian_filter1d(phi, sigma=sigma_pts)
    
        # Re-enforce boundaries exactly
        phi[0] = phi_left
        phi[-1] = phi_right

    # --- Quantum region (static for the whole solve; doesn't depend on phi) ---
    if quantum:
        q_i0, q_i1 = _detect_quantum_region(g)
        region_mask = np.zeros(g.N, dtype=bool)
        region_mask[q_i0:q_i1] = True
        if verbose:
            print(f"  Quantum region: grid [{q_i0}:{q_i1}] "
                  f"= [{g.x_nm[q_i0]:.1f}, {g.x_nm[q_i1 - 1]:.1f}] nm "
                  f"(of {g.N} points spanning [0, {g.x_nm[-1]:.1f}] nm)")

    # --- Anderson mixer (safeguarded: reset + fall back to damping whenever
    # the residual grows from one iteration to the next) ---
    mixer = _AndersonMixer(m=anderson_m, alpha=alpha)

    E_e = E_h = psi_e = psi_h = None
    converged = False
    n_iter    = 0
    residual  = np.inf
    prev_residual = np.inf
    residual_history: list = []
    alpha_history: list = []
    good_streak = 0   # consecutive non-worsening iterations (growth needs a streak, not one lucky step)

    for iteration in range(max_iter):

        # a. Band edges
        Ec = g.Ec0 - phi
        Ev = g.Ev0 - phi

        # b/c. Carrier densities
        if quantum:
            E_e, psi_e = _solve_confined_states(Ec, g.m_e, g.dx, n_states_e, q_i0, q_i1, g.N)
            n = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T),
                                  electron_density(Ec, Efn, g.Nc, T), region_mask)

            V_hole = hole_potential(Ev)
            E_h, psi_h = _solve_confined_states(V_hole, g.m_hh, g.dx, n_states_h, q_i0, q_i1, g.N)
            p = _blended_density(quantum_hole_density(psi_h, E_h, Efp, g.m_hh, T),
                                  hole_density(Ev, Efp, g.Nv, T), region_mask)
        else:
            n = electron_density(Ec, Efn, g.Nc, T)
            p = hole_density(Ev, Efp, g.Nv, T)

        # Drift-Diffusion Step
        if V_applied != 0.0:
            ni = np.sqrt(g.Nc * g.Nv) * np.exp(-(Ec - Ev) / (2.0 * kBT_eV))
            R_total = compute_recombination(n, p, ni)
            
            # Calculate Fermi-Dirac Activity Coefficients (gamma = n_FD / n_Boltzmann)
            # This suppresses unphysical artificial diffusion at degenerate carrier densities (e.g. >10^19)
            # n_boltz = Nc * exp((Efn - Ec)/kT)
            n_boltz = g.Nc * np.exp(np.clip((Efn - Ec)/kBT_eV, -200, 200))
            p_boltz = g.Nv * np.exp(np.clip((Ev - Efp)/kBT_eV, -200, 200))
            
            # Prevent division by zero or extreme values
            n_boltz = np.maximum(n_boltz, 1e-30)
            p_boltz = np.maximum(p_boltz, 1e-30)
            n_safe = np.maximum(n, 1e-30)
            p_safe = np.maximum(p, 1e-30)
            
            gamma_n = n_safe / n_boltz
            gamma_p = p_safe / p_boltz
            
            # Generalized Scharfetter-Gummel effective potentials
            # SG assumes n ∝ exp(ψ/Vt), p ∝ exp(ψ/Vt)
            # n = Nc exp((Efn - Ec)/kT) gamma_n -> ψ_n = -Ec + kT ln(Nc/Nc0) + kT ln(gamma_n)
            # p = Nv exp((Ev - Efp)/kT) gamma_p -> ψ_p = Ev + kT ln(Nv/Nv0) + kT ln(gamma_p)
            psi_n_eff = -Ec + kBT_eV * np.log(g.Nc / g.Nc[0]) + kBT_eV * np.log(np.maximum(gamma_n, 1e-10))
            psi_p_eff = Ev + kBT_eV * np.log(g.Nv / g.Nv[0]) + kBT_eV * np.log(np.maximum(gamma_p, 1e-10))

            
            # Ohmic BCs: equilibrium carrier density at each contact
            n_left  = electron_density(Ec[0], 0.0, g.Nc[0], T)
            n_right = electron_density(Ec[-1], -V_internal, g.Nc[-1], T)
            p_left  = hole_density(Ev[0], 0.0, g.Nv[0], T)
            p_right = hole_density(Ev[-1], -V_internal, g.Nv[-1], T)
            
            # BUG 5 FIX: Composition-dependent mobility for high-Al AlGaN
            # mu_n(x) ~ 300*(1-x) + 25*x with alloy scattering reduction
            # mu_p(x) ~ 10*(1-x) + 2*x
            mu_n_avg = float(np.mean(300.0 * (1.0 - g.x_Al) + 25.0 * g.x_Al))
            mu_p_avg = float(np.mean(10.0 * (1.0 - g.x_Al) + 2.0 * g.x_Al))
            
            n_new = solve_continuity_electron(n, psi_n_eff, R_total, g.dx,
                                             mu_n=mu_n_avg, T=T,
                                             n_left=n_left, n_right=n_right)
            p_new = solve_continuity_hole(p, psi_p_eff, R_total, g.dx,
                                         mu_p=mu_p_avg, T=T,
                                         p_left=p_left, p_right=p_right)
            # Update quasi-Fermi levels
            Efn_new = Efn_from_n(n_new, Ec, g.Nc, T)
            Efp_new = Efp_from_p(p_new, Ev, g.Nv, T)
            
            # Series Resistance (Lumped model)
            # Calculate internal voltage drop
            if R_series > 0.0:
                from physics.drift_diffusion import compute_current_density
                J_total = compute_current_density(
                    n_new, p_new, psi_n_eff, psi_p_eff, g.dx, mu_n_avg, mu_p_avg, T
                )
                # J_total can be positive or negative depending on direction. 
                # V_applied is forward bias (positive). J_total is positive for forward bias.
                V_internal_target = V_applied - abs(J_total) * R_series
                # Ensure V_internal does not go negative during forward bias
                V_internal_target = max(0.0, V_internal_target)
                
                # Smooth update of V_internal to prevent oscillations
                # If first iteration, V_internal doesn't exist yet, we initialize before loop
                phi_right = g.phi_top_eq + V_internal
                V_internal = 0.9 * V_internal + 0.1 * V_internal_target
            else:
                V_internal = V_applied
                phi_right = g.phi_top_eq + V_applied

            # Enforce contact boundary conditions strictly
            Efn_new[0] = 0.0
            Efn_new[-1] = -V_internal
            Efp_new[0] = 0.0
            Efp_new[-1] = -V_internal
            
            # Clip maximum quasi-Fermi update per iteration to 0.5V to prevent wild swings
            dEfn = np.clip(Efn_new - Efn, -0.5, 0.5)
            dEfp = np.clip(Efp_new - Efp, -0.5, 0.5)
            
            # Update quasi-Fermi levels
            Efn = Efn + dEfn
            Efp = Efp + dEfp
            
            # Recompute n, p with updated quasi-Fermi levels
            if quantum:
                n = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T),
                                      electron_density(Ec, Efn, g.Nc, T), region_mask)
                p = _blended_density(quantum_hole_density(psi_h, E_h, Efp, g.m_hh, T),
                                      hole_density(Ev, Efp, g.Nv, T), region_mask)
            else:
                n = electron_density(Ec, Efn, g.Nc, T)
                p = hole_density(Ev, Efp, g.Nv, T)

        # d. Newton-Raphson Poisson step
        # Make Si a shallow donor (20 meV) everywhere to prevent total depletion of the n-layer
        # (Nextnano standard default for AlGaN LED base layers)
        Ed_x = np.full(g.N, 0.02)      # eV, Si donor
        Ea_x = np.full(g.N, 0.17)      # eV, Mg acceptor
        
        # Clamp exponent arguments to prevent overflow
        exp_arg_d = np.clip((Efn - (Ec - Ed_x)) / kBT_eV, -500.0, 500.0)
        exp_arg_a = np.clip((Ev + Ea_x - Efp) / kBT_eV, -500.0, 500.0)
        
        # Ionized densities
        exp_d = np.exp(exp_arg_d)
        exp_a = np.exp(exp_arg_a)
        Nd_plus = g.ND / (1.0 + 2.0 * exp_d)
        Na_minus = g.NA / (1.0 + 4.0 * exp_a)
        
        # Derivatives w.r.t phi (Ec = Ec0 - phi -> dEc/dphi = -1)
        # d(exp_arg_d)/dphi = 1 / kBT_eV
        dNd_dphi = -g.ND * (2.0 * exp_d) / (1.0 + 2.0 * exp_d)**2 / kBT_eV
        # d(exp_arg_a)/dphi = -1 / kBT_eV
        dNa_dphi = g.NA * (4.0 * exp_a) / (1.0 + 4.0 * exp_a)**2 / kBT_eV
        
        # Dynamic relaxation for stability
        # If D_int gets too small (e.g. in depleted regions), force a minimum D
        # to limit the max potential step
        min_D_equivalent_n = 1e16  # cm^-3 pseudo-carrier density for damping
        dNd_dphi = np.minimum(dNd_dphi, -min_D_equivalent_n / kBT_eV)
        
        phi_new = solve_poisson_newton(
            phi, g.eps_r, n, p, Nd_plus, Na_minus, dNd_dphi, dNa_dphi, g.pol_rho,
            g.dx, phi_left, phi_right, T
        )

        # Convergence residual: max change the Newton step wants to make
        residual = np.max(np.abs(phi_new - phi))
        residual_history.append(float(residual))

        # e. Mixing.
        if bias_coupled:
            # Biased solves: keep the original plain fixed-alpha damping.
            # See the "Note on scope" in this function's docstring — the
            # drift-diffusion / quasi-Fermi coupling that only runs when
            # V_applied != 0 was empirically found to destabilise under any
            # iteration-to-iteration variation in step size.
            alpha_history.append(float(alpha))
            use_aa = False
            phi_candidate = (1.0 - alpha) * phi + alpha * phi_new
        else:
            # Unbiased (equilibrium) solves: safeguarded, adaptive mixing.
            # Adapt alpha for *this* step from how the previous step's
            # residual trend looked: shrink immediately (floored at
            # alpha_min) on any residual growth (reset the good streak),
            # but only grow (capped at alpha_max) after several
            # *consecutive* non-worsening iterations — a single lucky
            # iteration is not reliable evidence, and growing off one data
            # point is what let alpha run away into instability.
            if residual > prev_residual:
                alpha = max(alpha_min, alpha * 0.5)
                good_streak = 0
            else:
                good_streak += 1
                if good_streak >= 3:
                    alpha = min(alpha_max, alpha * 1.05)
            mixer.alpha = alpha
            alpha_history.append(float(alpha))

            # Try Anderson extrapolation, but reject it *before* committing
            # to phi if its step is disproportionately larger than the
            # plain damped step would be. A runaway extrapolation is
            # exactly the failure mode the old code sidestepped by
            # disabling AA outright; checking after the fact (i.e. only via
            # next iteration's residual) is too late, since phi has already
            # been corrupted by then. This check catches it before phi is
            # ever updated.
            phi_damped  = (1.0 - alpha) * phi + alpha * phi_new
            step_damped = alpha * residual
            use_aa = anderson_m > 0
            phi_candidate = phi_damped
            if use_aa:
                phi_aa  = mixer.mix(phi, phi_new)
                step_aa = np.max(np.abs(phi_aa - phi))
                if step_aa > _AA_STEP_SAFEGUARD * max(step_damped, 1e-8):
                    mixer.reset()   # extrapolation blew up; discard history
                    use_aa = False
                else:
                    phi_candidate = phi_aa

        if verbose:
            print(f"  iter {iteration+1:4d}  |dphi|_max = {residual:.3e} V  "
                  f"alpha = {alpha:.3e}  {'AA' if use_aa else 'damped'}")

        phi = phi_candidate

        # Enforce BCs exactly (mixing can slightly drift boundary values)
        phi[0]  = phi_left
        phi[-1] = phi_right

        prev_residual = residual
        n_iter = iteration + 1
        if residual < tol:
            converged = True
            break

    if not converged and verbose:
        print(f"  Warning: did not converge after {max_iter} iterations "
              f"(|dphi|={residual:.2e} V)")

    # --- Final band edges and carrier densities ---
    Ec_final = g.Ec0 - phi
    Ev_final = g.Ev0 - phi
    Ei_final = (0.5 * (Ec_final + Ev_final)
                + 0.5 * kBT_eV * np.log(g.Nv / g.Nc))
    E_field  = electric_field(phi, g.dx)

    if quantum and psi_e is not None:
        n_final = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T),
                                    electron_density(Ec_final, Efn, g.Nc, T), region_mask)
        p_final = _blended_density(quantum_hole_density(psi_h, E_h, Efp, g.m_hh, T),
                                    hole_density(Ev_final, Efp, g.Nv, T), region_mask)
    else:
        n_final = electron_density(Ec_final, Efn, g.Nc, T)
        p_final = hole_density(Ev_final, Efp, g.Nv, T)

    qcse_transition_eV = None
    qcse_overlap = None
    qcse_dominant_transition_eV = None
    qcse_dominant_overlap = None
    qcse_dominant_pair = None
    if quantum:
        e1h1 = ground_state_transition(E_e, psi_e, E_h, psi_h, g.dx)
        if e1h1 is not None:
            qcse_transition_eV = e1h1.energy_eV
            qcse_overlap = e1h1.overlap
        dom = dominant_transition(E_e, psi_e, E_h, psi_h, g.dx,
                                   n_e=len(E_e), n_h=len(E_h))
        if dom is not None:
            qcse_dominant_transition_eV = dom.energy_eV
            qcse_dominant_overlap = dom.overlap
            qcse_dominant_pair = (dom.ie, dom.ih)

    return SolverResult(
        x_nm=g.x_nm, x_Al=g.x_Al,
        Ec=Ec_final, Ev=Ev_final, Ei=Ei_final,
        Efn=Efn, Efp=Efp,
        phi=phi, E_field=E_field, F_quasi=g.F_quasi,
        n=n_final, p=p_final,
        Psp=g.Psp, Ppz=g.Ppz, P_total=g.P_total,
        eps_xx=g.eps_xx, eps_zz=g.eps_zz,
        E_e=E_e, psi_e=psi_e, E_h=E_h, psi_h=psi_h,
        qcse_transition_eV=qcse_transition_eV, qcse_overlap=qcse_overlap,
        qcse_dominant_transition_eV=qcse_dominant_transition_eV,
        qcse_dominant_overlap=qcse_dominant_overlap,
        qcse_dominant_pair=qcse_dominant_pair,
        V_applied=V_applied, T=T,
        converged=converged, n_iterations=n_iter,
        interface_indices=g.interface_indices,
        interface_sigmas=g.interface_sigmas,
        residual_history=residual_history, alpha_history=alpha_history,
        Ec0=g.Ec0, Ev0=g.Ev0, ND=g.ND, NA=g.NA,
    )

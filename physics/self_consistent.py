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
     e. Pseudo-transient continuation: the Newton step of (d) is taken
        with a pseudo-time term 1/dt added to the Jacobian diagonal
        (physics.poisson.solve_poisson_newton_ptc). The step is accepted
        only if the true nonlinear Poisson residual does not grow; dt then
        grows (up to 3x), otherwise dt shrinks (x0.3) and the step is
        retried. dt -> infinity is the plain Newton step. (Earlier
        versions mixed a full Newton step with adaptive damping and
        Anderson acceleration; that scheme entered limit cycles on
        short-period, strongly polarized stacks. The alpha / anderson_m
        arguments are still accepted for backward compatibility.)
        Biased solves use the coupled Newton solver in physics.dd_newton.
     f. Check  max|phi_new - phi| < tol
  4. Return SolverResult

Two modes:
  quantum=False  →  classical Fermi-Dirac only        (fast)
  quantum=True   →  full Schrödinger-Poisson           (slower, captures quantisation)
"""

from __future__ import annotations

import logging
import weakref
import numpy as np
from physics.materials.algan import donor_ionization_energy, acceptor_ionization_energy
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Callable, Optional

from devices.grid_builder import GridData
from physics.fermi_dirac import (
    electron_density, hole_density,
    quantum_electron_density, quantum_hole_density,
)
from physics.poisson import (
    solve_poisson, solve_poisson_newton, solve_poisson_newton_ptc,
    electric_field, _assemble_laplacian,
)
from physics.schrodinger import solve_schrodinger, hole_potential
from physics.drift_diffusion import compute_current_density, compute_recombination_components
from physics.gain import compute_gain_spectrum
from physics.coupled_solver import QuantumState, SolveCancelled, CoupledResult
from physics import dd_newton
from physics.optical import (
    ground_state_transition, dominant_transition,
    overlap_squared, hole_subband_energy,
)
from physics.grid_utils import node_spacings
from physics.constants import q as _q, kB, eps0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anderson AA-I mixer
# ---------------------------------------------------------------------------

# An Anderson-extrapolated step is rejected (falls back to plain damping)
# if its magnitude exceeds this multiple of what the plain damped step
# would have been — the signature of a runaway extrapolation.
_AA_STEP_SAFEGUARD = 5.0

# Biased-solve backtracking line search (see _nonlinear_poisson_residual_norm
# and the "e. Mixing" bias_coupled branch below): Armijo sufficient-decrease
# constant, max halvings tried before giving up and taking the smallest step,
# and the floor on the step itself (never fully stall the outer loop).
_ARMIJO_C = 1e-4
_MAX_BACKTRACKS = 20
_MIN_LINE_SEARCH_STEP = 1e-4

# Bias-coupled outer loop fail-fast (see its use below, near dt_ptc's
# initialization): number of consecutive non-improving outer iterations,
# with the inner coupled solve also failing to converge, before giving up
# rather than continuing to max_iter. A single stalled iteration is normal
# noise; this many in a row with no inner convergence either is not.
_OUTER_STAGNATION_PATIENCE = 15
# An outer-iteration |dphi| residual must shrink by at least this fraction
# relative to the best seen so far to count as "real" improvement (not
# floating-point/PTC-retry jitter) and reset the stagnation counter.
_OUTER_STAGNATION_MIN_IMPROVEMENT = 0.99

# R_series fast path (see _solve_bias_with_series_resistance): number of
# damped fixed-point iterations tried before giving up and falling back to
# the slower, more robust bisection-with-jitter. nextnano++ doesn't even
# solve this inside the device physics at all -- it sweeps V_internal and
# applies R_series as post-hoc arithmetic on the resulting J-V curve (see
# this session's nextnano++ documentation research) -- so a handful of
# fixed-point iterations to hit one specific target V_applied is already
# more work than nextnano's own approach, not less.
_R_SERIES_FIXED_POINT_MAXITER = 8


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

    # Padding in physical nm, converted to grid indices via g.x_nm rather
    # than a single scalar dx -- correct regardless of whether the grid is
    # uniform or has per-layer spacing (devices.grid_builder's qw_window /
    # per-layer dx_nm).
    x_lo = g.x_nm[idx[0]] - margin_nm
    x_hi = g.x_nm[idx[-1]] + margin_nm
    i0 = max(0, int(np.searchsorted(g.x_nm, x_lo, side='left')))
    i1 = min(g.N, int(np.searchsorted(g.x_nm, x_hi, side='right')))
    return i0, i1


def _solve_confined_states(V_eV_full: np.ndarray, m_full: np.ndarray, dx,
                            n_states: int, i0: int, i1: int, N: int,
                            E_cut: Optional[float] = None):
    """
    Solve the Schrödinger equation restricted to grid indices [i0, i1) and
    embed the resulting wavefunctions into full-length (N-point),
    zero-padded arrays, so downstream code (quantum_electron_density etc.)
    can treat them exactly like a full-domain solve.

    dx : scalar (uniform grid; used unchanged) or the full-device length-
    (N-1) per-edge spacing array (non-uniform grid), which must be sliced
    down to the subdomain's own edges (length i1-i0-1) before being handed
    to solve_schrodinger -- passing the full-device array unsliced would
    both be the wrong length and describe the wrong edges.

    E_cut : if given, n_states is only a minimum -- it is doubled until the
    highest returned state lies above E_cut (or the region runs out of
    states). A fixed state count silently drops the charge of every well
    whose states rank past it: on a 9-well MQW with n_states=6, the last
    wells got NO quantum electrons at all (gamma pinned at 1e-8), the
    potential there collapsed by ~1 V, and the next Schrodinger solve moved
    all states into those wells instead -- the sloshing behind
    [[bug-quantum-bias-electron-divergence]].
    """
    dx_sub = dx if np.ndim(dx) == 0 else np.asarray(dx)[i0:i1 - 1]
    n_max = i1 - i0 - 2
    while True:
        E, psi_sub = solve_schrodinger(V_eV_full[i0:i1], m_full[i0:i1], dx_sub, n_states=n_states)
        if E_cut is None or len(E) == 0 or E[-1] >= E_cut or n_states >= n_max:
            break
        n_states = min(2 * n_states, n_max)
    psi_full = np.zeros((len(E), N))
    psi_full[:, i0:i1] = psi_sub
    return E, psi_full


def _local_well_transition(g: GridData, Ec: np.ndarray, Ev: np.ndarray, i0: int, i1: int,
                           dx_cell: np.ndarray, pad_nm: float = 4.0):
    """(E_e1 - E_h1 [eV], |<psi_e|psi_h>|^2) of the well [i0, i1) from a
    Schrodinger solve restricted to the well plus pad_nm of barrier on each
    side (hard walls beyond). The hole is the topmost of the HH/LH/SO
    ground states. None if no confined pair is found."""
    x = g.x_nm
    j0 = int(np.searchsorted(x, x[i0] - pad_nm))
    j1 = int(min(g.N, np.searchsorted(x, x[i1 - 1] + pad_nm) + 1))
    if j1 - j0 < 8:
        return None
    try:
        E_e, psi_e = _solve_confined_states(Ec, g.m_e, g.dx, 2, j0, j1, g.N)
        bands = _solve_hole_bands(Ev, g, g.dx, j0, j1, 2)
    except Exception:  # noqa: BLE001 - diagnostic only, never break a solve
        return None
    if len(E_e) == 0:
        return None
    best = None
    for E_h, psi_h in bands.values():
        if len(E_h) and (best is None or E_h[0] < best[0]):
            best = (E_h[0], psi_h[0])
    if best is None:
        return None
    E_h1, psi_h1 = best
    ov = overlap_squared(psi_e[0], psi_h1, dx_cell)
    return float(E_e[0] + E_h1), float(ov)     # E_e - (-E_h)


def _first_state_in_window(E: Optional[np.ndarray], psi: Optional[np.ndarray],
                            window: tuple[int, int], dx_cell: np.ndarray,
                            min_frac: float = 0.5) -> Optional[int]:
    """
    Index of the lowest-energy solved state whose probability is mostly
    (>= min_frac) inside `window` = (i0, i1) grid indices.

    Used to pick out "the" quantum well's own e1/h1 states for the QCSE
    diagnostic, since the *global* lowest-energy state (index 0) can
    instead be a polarization/doping-induced notch elsewhere in the
    device -- see the qw_window docstring in devices.grid_builder.

    Weighted by dx_cell (the per-node finite-volume quadrature weight, see
    physics.grid_utils) rather than a raw sum of psi^2: on a non-uniform
    grid (e.g. a finely-meshed well next to a coarsely-meshed buffer) an
    unweighted point count would over-count whichever region happens to
    have more grid points, independent of its actual physical extent.
    """
    if E is None or psi is None or len(E) == 0:
        return None
    i0, i1 = window
    for k in range(len(E)):
        weighted = psi[k] ** 2 * dx_cell
        total = float(np.sum(weighted))
        if total <= 0:
            continue
        inside = float(np.sum(weighted[i0:i1]))
        if inside / total >= min_frac:
            return k
    return None


def _blended_density(quantum_density: np.ndarray, classical_density: np.ndarray,
                      region_mask: np.ndarray) -> np.ndarray:
    """Quantum (subband) density inside the quantum region, classical
    Fermi-Dirac density outside it (bulk contacts are reservoirs, not
    confined states)."""
    return np.where(region_mask, quantum_density, classical_density)


# Hole band name -> mass array attr on GridData, for the decoupled 3-band
# effective-mass valence model (see physics.materials.algan.AlGaNParams.
# valence_offsets_from_top). Ev0 is the TOPMOST valence band; each band sits
# g.dEv_<band> (>= 0) below it.
_HOLE_BAND_MASS_ATTR = {'hh': 'm_hh', 'lh': 'm_lh', 'so': 'm_so'}


# Quantum correction factor bounds: gamma = n_quantum/n_classical can be
# enormous/tiny where both densities are negligible (deep depletion, or
# above the highest computed subband); clipping keeps kT*ln(gamma) (the
# quantum potential it adds to the SG flux) within ~+-0.5 eV there without
# affecting any node that actually carries charge.
_QUANTUM_GAMMA_MIN = 1e-8
_QUANTUM_GAMMA_MAX = 1e8
_QUANTUM_GAMMA_BETA_INIT = 0.5
_SUBBAND_CUTOFF_KT = 10.0
# Quantum bias convergence also requires |ln(gamma_new/gamma_used)| below
# this (i.e. quantum and modelled densities agree to ~1%) wherever the
# density exceeds _GAMMA_MISMATCH_MIN_DENSITY [cm^-3].
_GAMMA_MISMATCH_TOL = 1e-2
_GAMMA_MISMATCH_MIN_DENSITY = 1e10
_QUANTUM_GAMMA_BETA_MIN = 0.02


def _quantum_gamma(quantum_blended: np.ndarray, classical: np.ndarray,
                   region_mask: np.ndarray) -> np.ndarray:
    """Per-node n_quantum/n_classical inside the quantum region, exactly 1
    outside it -- the correction factor physics.coupled_solver folds into
    its residual (quantum_gamma_n/p)."""
    ratio = quantum_blended / np.maximum(classical, 1e-300)
    gamma = np.clip(ratio, _QUANTUM_GAMMA_MIN, _QUANTUM_GAMMA_MAX)
    return np.where(region_mask, gamma, 1.0)


def _hole_band_edges(Ev: np.ndarray, g: GridData) -> dict:
    """HH/LH/SO valence band-edge profiles [eV] at the current phi, on the
    same absolute scale as Ev = g.Ev0 - phi (the topmost band). The
    per-band depths are static (devices.grid_builder's dEv_hh/lh/so), so
    they just subtract straight off Ev like the phi shift already applied."""
    return {'hh': Ev - g.dEv_hh, 'lh': Ev - g.dEv_lh, 'so': Ev - g.dEv_so}


def _solve_hole_bands(Ev: np.ndarray, g: GridData, dx, q_i0: int, q_i1: int,
                       n_states_h: int, E_cut: Optional[float] = None) -> dict:
    """Solve HH/LH/SO confined states as three independent single-band
    Schrodinger problems sharing the same quantum-region window, each with
    its own band edge and effective mass. Returns {'hh'/'lh'/'so': (E, psi)}."""
    edges = _hole_band_edges(Ev, g)
    bands = {}
    for name, Ev_band in edges.items():
        m_band = getattr(g, _HOLE_BAND_MASS_ATTR[name])
        V_hole = hole_potential(Ev_band)
        bands[name] = _solve_confined_states(V_hole, m_band, dx, n_states_h, q_i0, q_i1, g.N, E_cut)
    return bands


def _total_quantum_hole_density(bands: dict, Efp, T: float, g: GridData,
                                local_qfl: bool = False) -> np.ndarray:
    """Sum quantum_hole_density across HH/LH/SO -- position-space density
    is additive across independent (decoupled, unmixed) bands."""
    total = np.zeros(g.N)
    for name, (E, psi) in bands.items():
        m_band = getattr(g, _HOLE_BAND_MASS_ATTR[name])
        total = total + quantum_hole_density(psi, E, Efp, m_band, T, local_qfl)
    return total


def _nonlinear_poisson_residual_norm(
    phi_trial: np.ndarray, g: GridData, Efn: np.ndarray, Efp: np.ndarray, T: float,
    kBT_eV: float, Ed_x: np.ndarray, Ea_x: np.ndarray,
    surf_idx: np.ndarray, surf_density_cm3: np.ndarray,
    surf_energy_eV: np.ndarray, surf_is_donor: np.ndarray,
) -> float:
    """
    RMS of the TRUE nonlinear Poisson residual F(phi) = A_pos@phi - rho(phi)/eps0
    at a trial potential -- cheap (no linear solve) merit function for the
    biased-solve backtracking line search below. Reuses classical
    (non-quantum) Fermi-Dirac carrier densities even when quantum=True, and
    holds the quasi-Fermi levels fixed at their current Gummel-iteration
    values: both standard, cheap approximations for a line-search trial
    evaluation (re-solving Schrodinger at every trial step would defeat the
    point of a *cheap* merit function) that are still good enough to
    reliably reject a step that overshoots into a much worse nonlinear
    regime -- exactly the failure mode fixed-alpha damping couldn't catch.
    """
    Ec_t = g.Ec0 - phi_trial
    Ev_t = g.Ev0 - phi_trial
    n_t = electron_density(Ec_t, Efn, g.Nc, T)
    p_t = hole_density(Ev_t, Efp, g.Nv, T)

    exp_arg_d = np.clip((Efn - (Ec_t - Ed_x)) / kBT_eV, -340.0, 340.0)
    exp_arg_a = np.clip((Ev_t + Ea_x - Efp) / kBT_eV, -340.0, 340.0)
    Nd_plus_t = g.ND / (1.0 + 2.0 * np.exp(exp_arg_d))
    Na_minus_t = g.NA / (1.0 + 4.0 * np.exp(exp_arg_a))

    if len(surf_idx) > 0:
        donor_mask = surf_is_donor
        if np.any(donor_mask):
            idx_d = surf_idx[donor_mask]
            exp_arg_sd = np.clip(
                (Efn[idx_d] - (Ec_t[idx_d] - surf_energy_eV[donor_mask])) / kBT_eV,
                -340.0, 340.0)
            np.add.at(Nd_plus_t, idx_d, surf_density_cm3[donor_mask] / (1.0 + 2.0 * np.exp(exp_arg_sd)))
        acceptor_mask = ~surf_is_donor
        if np.any(acceptor_mask):
            idx_a = surf_idx[acceptor_mask]
            exp_arg_sa = np.clip(
                (Ev_t[idx_a] + surf_energy_eV[acceptor_mask] - Efp[idx_a]) / kBT_eV,
                -340.0, 340.0)
            np.add.at(Na_minus_t, idx_a, surf_density_cm3[acceptor_mask] / (1.0 + 4.0 * np.exp(exp_arg_sa)))

    rho = _q * (p_t * 1e6 - n_t * 1e6 + Nd_plus_t * 1e6 - Na_minus_t * 1e6) + g.pol_rho

    eps_m, eps_p, h1, h2, cell_width = _assemble_laplacian(g.eps_r, g.dx)
    cw = cell_width[1:-1]
    lap = (eps_m / h1) * (phi_trial[1:-1] - phi_trial[:-2]) / cw \
        + (eps_p / h2) * (phi_trial[1:-1] - phi_trial[2:]) / cw
    resid_interior = lap - rho[1:-1] / eps0
    return float(np.sqrt(np.mean(resid_interior**2)))


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

    # Drift-diffusion current (biased solves only; see physics.dd_newton).
    # J_n/J_p are per-EDGE current densities [A/cm^2] (length N-1);
    # current_conservation_error = (max J - min J)/max|J| over the device,
    # ~0 for a physically valid steady state. It is only meaningful when
    # |J| is above the double-precision resolution of the quasi-Fermi
    # gradients (J below ~1e-6 A/cm^2 on typical devices reads as ~1).
    J_n: Optional[np.ndarray] = None
    J_p: Optional[np.ndarray] = None
    J_total: float = float('nan')
    current_conservation_error: float = float('nan')
    # Minimum carrier density [cm^-3] the current equation needed to
    # converge (physics.dd_newton.DDProblem.transport_floor); 0 = none, the
    # exact equations. Non-zero means a layer was cut off from both contacts:
    # currents of the order of the floor's own leakage are not physical.
    transport_floor_cm3: float = 0.0
    # How the bias was applied: 'equilibrium', 'current' (drift-diffusion),
    # 'gate' (no current, one Fermi level in the semiconductor),
    # 'interpolated' (nextnano++-style Fermi level ramp, no current) or
    # 'flat' (flat quasi-Fermi levels split by qV).
    bias_mode: str = 'equilibrium'
    bias_note: str = ''        # anything the user should know about how the bias was applied
    # Relative permittivity profile, the polarization field -P/(eps0 eps_r)
    # [V/m] (the field the polarization alone would set up in a slab with no
    # free charge) and the electric displacement D = eps0 eps_r F + P [C/m^2],
    # whose slope is the free charge density.
    eps_r: Optional[np.ndarray] = None
    E_polarization: Optional[np.ndarray] = None
    D_field: Optional[np.ndarray] = None

    # Wavefunctions (quantum mode)
    E_e: Optional[np.ndarray] = None
    psi_e: Optional[np.ndarray] = None
    E_h: Optional[np.ndarray] = None    # heavy-hole (HH) band -- "the" hole
    psi_h: Optional[np.ndarray] = None  # band for QCSE/backward compatibility

    # Decoupled 3-band effective-mass valence model (see physics.materials.
    # algan.AlGaNParams.valence_band_structure): light-hole (LH) and
    # split-off (SO) confined states, solved independently of HH (no
    # HH-LH-SO mixing). Ev_hh/Ev_lh/Ev_so are the position-dependent band
    # edges. Ev_hh/Ev_lh/Ev_so are always populated (equal to Ev when
    # quantum=False, since there's no confinement to split them further);
    # E_h_lh/psi_h_lh/E_h_so/psi_h_so (the confined *states*) are None
    # unless quantum=True.
    Ev_hh: Optional[np.ndarray] = None
    Ev_lh: Optional[np.ndarray] = None
    Ev_so: Optional[np.ndarray] = None
    # Electron affinity actually used per grid point [eV], including the
    # strain (deformation-potential) shift of Ec -- the vacuum level is
    # Ec + chi. None for results built without it.
    chi: Optional[np.ndarray] = None
    # In composition profile (InGaN / InAlGaN); None or zeros for AlGaN.
    x_In: Optional[np.ndarray] = None
    # Crystal system ('wurtzite' | 'zincblende') and, for zincblende, the P
    # fraction of the group-V sublattice.
    crystal: str = 'wurtzite'
    x_P: Optional[np.ndarray] = None
    E_h_lh: Optional[np.ndarray] = None
    psi_h_lh: Optional[np.ndarray] = None
    E_h_so: Optional[np.ndarray] = None
    psi_h_so: Optional[np.ndarray] = None

    # Quantum-Confined Stark Effect: the e-h transition energy and overlap
    # for the layer stack's own quantum well (see qw_window below), falling
    # back to the plain global e1-h1 pair if no such layer is found. None
    # when quantum=False or no confined pair was found. See physics/optical.py.
    qcse_transition_eV: Optional[float] = None
    qcse_overlap: Optional[float] = None
    qcse_pair: Optional[tuple] = None       # (ie, ih) subband indices actually used
    qcse_in_well: bool = False              # True if qcse_pair was well-restricted
    # True when qcse_transition_eV came from a local single-well
    # Schrodinger solve (no globally-solved subband was confined in the
    # detected well); the e1/h1 wavefunctions are then not in psi_e/psi_h.
    qcse_local_solve: bool = False

    # Grid index / nm window of the auto-detected quantum well (devices.
    # grid_builder.build_grid), used to focus the QCSE calc above and to
    # let the GUI auto-zoom the plot onto it. None if no such layer exists.
    qw_window_nm: Optional[tuple] = None

    # Highest-overlap (electron subband, hole subband) pair among *all*
    # solved states -- the pair actually most likely to dominate emission.
    # A polarization-induced interface notch (see _detect_quantum_region)
    # can sometimes pull the lowest-energy e1/h1 states to opposite ends of
    # the device rather than into the intended quantum well, in which case
    # this differs from (0, 0) and is the more physically meaningful number.
    qcse_dominant_transition_eV: Optional[float] = None
    qcse_dominant_overlap: Optional[float] = None
    qcse_dominant_pair: Optional[tuple] = None   # (ie, ih)

    # Recombination rates [cm^-3 s^-1], one value per grid point, computed
    # from the converged final n/p/ni (see physics.drift_diffusion.
    # compute_recombination_components -- same coefficients the solver's
    # own continuity-equation R term uses internally, just split apart by
    # mechanism here for diagnostics/plotting).
    R_srh: Optional[np.ndarray] = None
    R_rad: Optional[np.ndarray] = None
    R_aug: Optional[np.ndarray] = None

    # Optical gain spectrum (quantum mode only -- see physics.gain): material
    # gain g(hv) [cm^-1] vs photon energy, from the quantum well's own
    # confined subbands (E_e/psi_e, hole bands) and quasi-Fermi occupation,
    # Lorentzian-broadened across subband-pair transitions. None when
    # quantum=False or no confined states were found in the design's own
    # quantum well (qw_window).
    gain_energy_eV: Optional[np.ndarray] = None
    gain_wavelength_nm: Optional[np.ndarray] = None
    gain_spectrum_cm1: Optional[np.ndarray] = None
    peak_gain_cm1: Optional[float] = None
    peak_gain_wavelength_nm: Optional[float] = None

    # Surface/interface charge states (devices.layer.SurfaceCharge), for
    # plotting the trap energy level(s) directly on the band diagram so
    # Fermi-level pinning (Efn/Efp sitting at the trap level once its areal
    # density is high enough to dominate local charge balance) is visible
    # rather than something you have to trust happened. One entry per
    # state: (x_nm, trap_level_eV, state_type, density_cm2). trap_level_eV
    # is in the same Ec/Ev reference frame as everything else plotted --
    # Ec[i]-energy_eV for a donor state, Ev[i]+energy_eV for an acceptor
    # state, evaluated at the *converged* band edges.
    surface_charge_markers: list = field(default_factory=list)

    # Metadata
    V_applied: float = 0.0
    # Voltage actually applied across the *intrinsic* device (i.e. the
    # Efn/Efp boundary condition the solve actually used), after the lumped
    # series-resistance IR drop: V_internal = V_applied - |J_total|*R_series,
    # relaxed over the outer loop -- see the R_series branch below. Equal to
    # V_applied whenever R_series == 0. At high bias with R_series > 0, a
    # large idealized current can drive V_internal well below V_applied
    # (even toward 0), so the quasi-Fermi levels split by far less than
    # V_applied alone would suggest -- this is what makes that visible
    # instead of a silent mismatch between the labeled bias and the
    # solved one. See visualization.plotter's band diagram title.
    V_internal: float = 0.0
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


def _mobility_profiles(g: GridData):
    """Per-node low-field mobilities [cm^2/Vs] from the grid (see
    physics.materials.alloys.mobilities). A GridData built without them
    gets the nitride values: linear in Al fraction between GaN (300 / 10)
    and AlN (25 / 2)."""
    if getattr(g, 'mu_n', None) is not None and getattr(g, 'mu_p', None) is not None:
        return np.asarray(g.mu_n, dtype=float), np.asarray(g.mu_p, dtype=float)
    return (300.0 * (1.0 - g.x_Al) + 25.0 * g.x_Al,
            10.0 * (1.0 - g.x_Al) + 2.0 * g.x_Al)


def dopant_energies(g: GridData):
    """Per-node (Ed, Ea) [eV] from the grid; the nitride formulas (Si, Mg)
    for a GridData built without them."""
    if getattr(g, 'Ed', None) is not None and getattr(g, 'Ea', None) is not None:
        return np.asarray(g.Ed, dtype=float), np.asarray(g.Ea, dtype=float)
    x_in = getattr(g, 'x_In', None)
    if x_in is not None and np.any(x_in > 0):
        return donor_ionization_energy(g.x_Al, x_in), acceptor_ionization_energy(g.x_Al, x_in)
    return donor_ionization_energy(g.x_Al), acceptor_ionization_energy(g.x_Al)


def recombination_coefficients(g: GridData) -> dict:
    """{'tau_n', 'tau_p', 'B_rad', 'C_n', 'C_p'}: the grid's per-node
    recombination coefficients (None where the grid has none, which selects
    the defaults of physics.drift_diffusion)."""
    return {k: getattr(g, k, None) for k in ('tau_n', 'tau_p', 'B_rad', 'C_n', 'C_p')}


def _dd_problem(g: GridData, Ed_x, Ea_x, surf_idx, surf_density_cm3,
                surf_energy_eV, surf_is_donor, gamma_n=None, gamma_p=None) -> "dd_newton.DDProblem":
    mu_n, mu_p = _mobility_profiles(g)
    return dd_newton.DDProblem(
        g.Ec0, g.Ev0, g.Nc, g.Nv, g.ND, g.NA, Ed_x, Ea_x, g.pol_rho, g.eps_r, g.dx,
        g.T, mu_n, mu_p, surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
        gamma_n, gamma_p, **recombination_coefficients(g))


def _current_density_from_result(result: "SolverResult", grid: GridData) -> float:
    if np.isfinite(result.J_total):
        return float(result.J_total)
    return _legacy_current_density_from_result(result, grid)


def _legacy_current_density_from_result(result: "SolverResult", grid: GridData) -> float:
    """Gamma-corrected (degeneracy-aware) Scharfetter-Gummel current density
    [A/cm^2] at a converged bias solve -- same psi_n_eff/psi_p_eff
    construction used by the R_series IR-drop calculation elsewhere in this
    module, factored out so _solve_bias_with_series_resistance can reuse it
    without duplicating the degeneracy-correction algebra."""
    g = grid
    T = g.T
    kBT_eV = kB * T / _q
    mu_n_avg, mu_p_avg = (float(np.mean(m)) for m in _mobility_profiles(g))
    n_boltz = np.maximum(g.Nc * np.exp(np.clip((result.Efn - result.Ec) / kBT_eV, -200, 200)), 1e-30)
    gamma_n = np.maximum(result.n, 1e-30) / n_boltz
    psi_n_eff = -result.Ec + kBT_eV * np.log(g.Nc / g.Nc[0]) + kBT_eV * np.log(np.maximum(gamma_n, 1e-10))
    p_boltz = np.maximum(g.Nv * np.exp(np.clip((result.Ev - result.Efp) / kBT_eV, -200, 200)), 1e-30)
    gamma_p = np.maximum(result.p, 1e-30) / p_boltz
    psi_p_eff = result.Ev + kBT_eV * np.log(g.Nv / g.Nv[0]) + kBT_eV * np.log(np.maximum(gamma_p, 1e-10))
    return compute_current_density(result.n, result.p, psi_n_eff, psi_p_eff, g.dx, mu_n_avg, mu_p_avg, T)


def _solve_bias_with_series_resistance(
    grid: GridData, V_applied: float, R_series: float, quantum: bool,
    n_states_e: int, n_states_h: int, max_iter: int, tol: float,
    alpha: float, alpha_min: float, alpha_max: Optional[float], anderson_m: int,
    verbose: bool, log_fn: Optional[Callable[[str], None]],
    cancel_check: Optional[Callable[[], bool]],
) -> "SolverResult":
    """
    R_series > 0 handling, decoupled entirely from the coupled-Newton bias
    solve itself: find V_internal such that
        V_internal + |J(V_internal)| * R_series == V_applied
    where every trial is an independent, from-scratch call to
    solve_self_consistent(V_applied=V_internal, R_series=0.0) -- the
    already-robust, already-validated fresh-solve path.

    Tries a fast damped fixed-point iteration on V_internal first (a
    handful of direct solves for a smooth I-V curve), falling back to the
    slower bisection-with-jitter below only if that doesn't converge. Worth
    noting nextnano++ itself doesn't solve this inside the device physics
    at all -- it sweeps V_internal and applies R_series as post-hoc
    arithmetic on the resulting J-V curve (see this session's nextnano++
    documentation research) -- so even the fast path here, which targets
    one specific V_applied directly, is doing strictly more work than
    nextnano's own approach ever needs to.

    This replaces an earlier version that relaxed V_internal *inside* the
    same outer Gummel loop driving the coupled solve, carrying phi/Efn/Efp
    forward from one V_internal guess to the next. That is exactly the
    continuation-from-a-mismatched-state fragility this whole session spent
    fixing for voltage ramping, just hidden inside the R_series branch
    instead: since phi_right changes every outer iteration as V_internal
    moves, each iteration is effectively a *different* bias point, solved
    by continuing from a state that converged at the *previous* iteration's
    different boundary -- and the outer loop's fail-fast check (needed to
    catch [[bug-quasi-fermi-pinning]]'s false-convergence failure mode)
    aborted the entire solve the moment any single one of these mismatched
    attempts failed to converge. Confirmed empirically: enabling R_series
    made an already-marginal device (a 43-layer UV-LED replica) fail at
    *every* tested voltage, where R_series=0 on the identical device at
    least succeeded at a few. Bisection with independent fresh trials has
    no such coupling between steps: a single failed trial just narrows the
    bracket, exactly like sweep_voltage's own bisection-on-failure already
    does for voltage steps.
    """
    _log = log_fn if log_fn is not None else print

    def eval_at(V_int: float) -> "SolverResult":
        # Forward the caller's own verbose flag: each trial here can be a
        # full, possibly-long-running solve (this device's known patchy
        # cold-start convergence applies here too), and silencing it
        # entirely left the GUI/CLI showing nothing but a sparse
        # "bisection iter N" line every so often -- indistinguishable from
        # a hang. Prefixed so per-iteration lines are traceable to which
        # V_internal trial they belong to.
        def _prefixed_log(msg: str, _V=V_int) -> None:
            _log(f"    [V_internal={_V:.4f}] {msg}")
        return solve_self_consistent(
            grid, V_applied=V_int, R_series=0.0, quantum=quantum,
            n_states_e=n_states_e, n_states_h=n_states_h,
            max_iter=max_iter, tol=tol, alpha=alpha, alpha_min=alpha_min,
            alpha_max=alpha_max, anderson_m=anderson_m, verbose=verbose,
            log_fn=(_prefixed_log if verbose else log_fn), cancel_check=cancel_check,
        )

    gap_tol = max(tol * 10.0, 1e-4)
    best: Optional[tuple] = None   # (abs_gap, result, V_internal)

    # The IR drop always opposes the applied bias (it pulls V_internal
    # toward 0 whatever the polarity), so everything below works with
    # s = sign(V_applied): V_applied = V_internal + s*|J|*R_series, and
    # V_internal lies between 0 and V_applied. Using +|J|*R and clamping to
    # [0, V_applied] was only right for V_applied > 0 -- for reverse bias
    # that interval is empty and V_internal was pinned at V_applied.
    s = 1.0 if V_applied >= 0.0 else -1.0
    V_lo_b, V_hi_b = min(0.0, V_applied), max(0.0, V_applied)

    def _clamp(v: float, a: float, b: float) -> float:
        return min(max(v, min(a, b)), max(a, b))

    # Fast path: damped fixed-point iteration directly on
    #   V_internal = V_applied - |J(V_internal)| * R_series
    # instead of bisecting a search bracket. This is the natural fast
    # analogue of nextnano++'s own approach (see this function's docstring
    # update above): each iteration is one direct solve, converging in a
    # handful of steps for a smooth I-V curve, versus bisection's up to 30
    # iterations x 5 jittered retries = up to 150 solves. Falls straight
    # through to the existing bisection below (seeded with whatever partial
    # progress this made) the moment a trial fails to converge or the gap
    # stops improving -- never trusted past that point, so it can't make
    # a hard device any less robust than before, only faster on easy ones.
    V_internal = V_applied
    damping = 0.5
    prev_gap = None
    for it in range(_R_SERIES_FIXED_POINT_MAXITER):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")
        res = eval_at(V_internal)
        if not res.converged:
            if verbose:
                _log(f"  R_series fixed-point iter {it+1}: V_internal={V_internal:.4f} "
                     f"did not converge -- falling back to bisection")
            break

        J = _current_density_from_result(res, grid)
        implied_V_applied = V_internal + s * abs(J) * R_series
        gap = implied_V_applied - V_applied
        if verbose:
            _log(f"  R_series fixed-point iter {it+1}: V_internal={V_internal:.4f}  "
                 f"J={J:.4e} A/cm^2  implied V_applied={implied_V_applied:.4f} "
                 f"(target {V_applied:.4f})")
        if best is None or abs(gap) < best[0]:
            best = (abs(gap), res, V_internal)
        if abs(gap) < gap_tol:
            return _dc_replace(res, V_applied=V_applied, V_internal=V_internal, converged=True)

        if prev_gap is not None and abs(gap) >= abs(prev_gap):
            # Not improving -- shrink damping (adaptive, same principle as
            # the Gummel solver's own auto-shrinking alpha) rather than
            # trusting a step that isn't working.
            damping = max(damping * 0.5, 0.05)
        prev_gap = gap

        V_internal_target = V_applied - s * abs(J) * R_series
        V_internal = _clamp(V_internal + damping * (V_internal_target - V_internal), V_lo_b, V_hi_b)
    else:
        if verbose:
            _log(f"  R_series fixed-point path used its full budget "
                 f"({_R_SERIES_FIXED_POINT_MAXITER} iters) without converging "
                 f"-- falling back to bisection")

    lo, hi = 0.0, V_applied
    for it in range(30):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")
        V_mid = 0.5 * (lo + hi)
        # A failed trial is patchy-solver noise, not evidence V_mid itself
        # is on the wrong side of the root (this device's fresh-solve
        # convergence is known to be sparse/unpredictable across bias, not
        # systematically worse at higher V -- see today's MQW-device
        # findings). Nudge by small jittered offsets before ever touching
        # the bracket, so an isolated non-convergent point can't wrongly
        # collapse lo/hi the way treating it as "too high" did.
        res = None
        for jitter in (0.0, 0.01, -0.01, 0.03, -0.03):
            V_try = _clamp(V_mid + jitter * (hi - lo), lo, hi)
            res = eval_at(V_try)
            if res.converged:
                break
        if res is None or not res.converged:
            if verbose:
                _log(f"  R_series bisection iter {it+1:2d}: V_internal~{V_mid:.4f} "
                     f"did not converge (incl. jittered retries); skipping")
            continue

        J = _current_density_from_result(res, grid)
        implied_V_applied = V_try + s * abs(J) * R_series
        gap = implied_V_applied - V_applied
        if verbose:
            _log(f"  R_series bisection iter {it+1:2d}: V_internal={V_try:.4f}  "
                 f"J={J:.4e} A/cm^2  implied V_applied={implied_V_applied:.4f} "
                 f"(target {V_applied:.4f})")
        if best is None or abs(gap) < best[0]:
            best = (abs(gap), res, V_try)
        if abs(gap) < max(tol * 10.0, 1e-4):
            break
        if s * implied_V_applied > s * V_applied:
            hi = V_try
        else:
            lo = V_try

    if best is None:
        # Every single trial failed to converge -- return an honest failure
        # rather than raising, shaped like the V=0 equilibrium result so
        # callers always get a well-formed SolverResult.
        res0 = eval_at(0.0)
        return _dc_replace(res0, V_applied=V_applied, V_internal=0.0, converged=False)

    best_gap, res, V_internal_found = best
    return _dc_replace(
        res, V_applied=V_applied, V_internal=V_internal_found,
        converged=bool(res.converged and best_gap < max(tol * 10.0, 1e-4)),
    )


# Per-grid cache of the equilibrium (V=0) phi profile, used to seed every
# fresh bias solve's initial guess (see _get_equilibrium_phi_init below).
# Keyed by id(grid) with a weakref finalizer to evict the entry when that
# grid object is actually garbage-collected -- GridData has no __hash__
# (dataclass default __eq__ makes it unhashable), so it can't be a
# WeakKeyDictionary key directly, and a plain id()-keyed dict alone risks a
# stale hit if a *different*, later grid object happens to reuse a freed id.
_equilibrium_phi_cache: dict = {}


# Largest gate-voltage step taken in one Poisson solve (see the gate ramp in
# solve_self_consistent).
_GATE_RAMP_STEP_V = 0.5
# A node belongs to the channel (rather than the depleted barrier above it)
# when its mobile carrier density exceeds this fraction of the device's peak.
_GATE_CHANNEL_FRACTION = 1e-3
# A Schottky top contact counts as a gate only on a layer doped below this [cm^-3].
_GATE_MAX_SURFACE_DOPING = 1e18
# ... and only when the stack is not a p-n diode: no n-type and p-type
# layers both doped above this [cm^-3].
_GATE_PN_DOPING = 1e16
# A gate is taken to have turned on this far below its barrier height.
_GATE_TURN_ON_MARGIN_V = 0.1


def _get_equilibrium_phi_init(
    grid: GridData, quantum: bool, n_states_e: int, n_states_h: int,
    log_fn: Optional[Callable[[str], None]], cancel_check: Optional[Callable[[], bool]],
) -> Optional[np.ndarray]:
    """
    The equilibrium band profile is a far better starting guess for a
    *fresh* bias solve's phi than the generic per-point charge-neutral
    approximation (solve_self_consistent's "no phi_init" default): it's the
    exact, fully self-consistent solution at V=0 (now reliably fast via PTC
    -- see the equilibrium loop's own docstring), capturing quantum
    confinement, polarization, and composition effects the charge-neutral
    approximation only estimates. Unlike continuation across *biased*
    points (the mechanism that caused [[bug-quasi-fermi-pinning]]'s masked
    quasi-Fermi splits), this seeds from the *same* trusted reference point
    every time, regardless of which bias is being solved or whether some
    other nearby bias point converged -- confirmed empirically to matter
    for devices with many densely-packed heterointerfaces, where a generic
    flat/charge-neutral guess is a poor approximation of the true
    equilibrium shape and the coupled solve struggles to find it from cold.

    Efn/Efp keep their own generic fresh defaults (flat 0 / flat -V_applied)
    regardless -- only phi is seeded this way.

    Returns None (falls back to the generic default) if the equilibrium
    solve itself doesn't converge, rather than seeding from a known-bad
    profile.
    """
    # quantum is part of the key: the quantum and classical equilibria
    # differ, and one grid is solved both ways (e.g. a quantum bias solve's
    # classical warm start).
    key = (id(grid), bool(quantum))
    if key not in _equilibrium_phi_cache:
        eq_result = solve_self_consistent(
            grid, V_applied=0.0, quantum=quantum,
            n_states_e=n_states_e, n_states_h=n_states_h,
            max_iter=800, tol=1e-6, verbose=False,
            log_fn=log_fn, cancel_check=cancel_check,
        )
        _equilibrium_phi_cache[key] = eq_result.phi.copy() if eq_result.converged else None
        weakref.finalize(grid, _equilibrium_phi_cache.pop, key, None)
    return _equilibrium_phi_cache[key]


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
    log_fn: Optional[Callable[[str], None]] = None,
    phi_init: Optional[np.ndarray] = None,
    Efn_init: Optional[np.ndarray] = None,
    Efp_init: Optional[np.ndarray] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    flat_qfl: bool = False,
    gate_bias: Optional[bool] = None,
    interpolated_qfl: Optional[float] = None,
) -> SolverResult:
    """
    Run the self-consistent Schrödinger-Poisson solver.

    gate_bias : treat the biased (top) contact as a GATE: no current flows,
               the whole semiconductor keeps the Fermi level of the bottom
               contact (Efn = Efp = 0) and only the gate's own level moves,
               to -V_applied. This is the usual one-dimensional picture of a
               HEMT or a MOS-like stack under gate voltage: in the real device
               the channel is held by the source and drain, not by a vertical
               current through the buffer. The semiconductor is therefore in
               equilibrium with the source: E_F = 0 from the bottom up to the
               uppermost layer that holds carriers (the channel), and across
               the depleted barrier above it the level goes linearly to the
               gate's value; no carriers are there, so that stretch carries
               no charge. None (default) = automatic: on when the top contact
               is a Schottky contact on a layer doped below 1e18 cm^-3 and
               the stack is not a p-n diode; off otherwise. False = solve the
               vertical drift-diffusion current through the Schottky contact.
    interpolated_qfl : nextnano++-style Fermi level under bias, without the
               current equation. Give the "insulator band gap" E_ib in eV
               (nextnano++'s default is 1.0). One Fermi level E_F = Efn = Efp
               is interpolated between the two contact values by solving
                   d/dz [ exp(-Eg(z) / E_ib) dE_F/dz ] = 0,
               E_F = 0 at the bottom, -V_applied at the top, and Poisson (and
               Schrodinger) are solved with it. With E_ib = 1 eV the level
               ramps almost linearly across the whole structure, a little
               steeper in the wider-gap layers; a small E_ib (0.05 eV) keeps
               it flat in the narrow-gap layers and puts the whole drop in
               the barriers. It is what nextnano++ shows for a Poisson-only
               bias sweep (currents{ insulator_bandgap }). It is a
               prescription, not a transport result. None (default) = off.
    flat_qfl : constant-quasi-Fermi-level bias mode (no current solved,
               nextnano-style). Efn is held at the n-contact value (0) and
               Efp at the p-contact value (-V_applied) at every interior
               grid node, so Efn - Efp = qV_applied everywhere except the
               two contact nodes, and only (Schrodinger-)Poisson is solved
               for phi. Good for band diagrams / QCSE / wavefunctions under
               bias; carries no current information (J = 0), and R_series
               is ignored (no current, no IR drop).

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
    alpha_max  : upper bound the adaptive damping factor can grow to.
                 Defaults to `min(4*alpha, 0.6)` for the equilibrium loop
                 this actually governs (see "Note on scope" below) --
                 raised from the previous default of `alpha` itself (no
                 growth allowed at all) after confirming empirically
                 (2026-09-22) across three representative devices,
                 including the specific historically-fragile one the old
                 "reliably loses stability" caution named, that growth up
                 to this range is safe and gives a real speedup (4x fewer
                 iterations on all three, exact same converged answer).
                 That caution turned out to describe the *biased* solve
                 path below, which doesn't even use alpha_max any more
                 (superseded by the coupled JFNK solver) -- not this
                 equilibrium path, which was never actually the case the
                 instability was observed in. A HEMT-style undoped
                 heterojunction (thin AlGaN barrier on a GaN buffer, 2DEG
                 forming purely from the polarization discontinuity) is a
                 good example of a device that benefits substantially:
                 its much sharper/stronger interface transient made the
                 old fixed low ceiling converge far slower (1405
                 iterations for a device whose LED-style counterparts
                 need ~150-400) without ever being unstable at a higher
                 one.
    anderson_m : Anderson history length (0 = plain damped iteration)
    verbose    : emit iteration residuals via log_fn (default print)
    log_fn     : callable(str) used for verbose output instead of the
                 built-in print when given -- e.g. a GUI wiring iteration
                 progress into an on-screen log without touching the
                 process-wide sys.stdout (which a background solver thread
                 must never redirect: doing so would also hijack unrelated
                 output from other threads, such as the main GUI thread's
                 own prints, for as long as the solve runs)

    Note on scope: adaptive alpha and safeguarded Anderson mixing are only
    applied to the pure-electrostatic equilibrium loop (V_applied == 0).
    Empirically, once V_applied != 0 the drift-diffusion / quasi-Fermi
    update (Scharfetter-Gummel + Efn_from_n/Efp_from_p) is sensitive to
    iteration-to-iteration *changes* in phi's step size, not just its
    magnitude: a bounded, patience-gated adaptive alpha that only ever
    *grows* (never exceeding the fixed value known to be stable) reliably
    drove Efn/Efp to unphysical values (tens to hundreds of eV) where the
    original constant-alpha damping stayed bounded. Biased solves instead
    use an Armijo backtracking line search along the raw Newton direction
    (see _nonlinear_poisson_residual_norm and the bias_coupled branch of
    "e. Mixing"): the step size only ever *shrinks* from 1.0 within a given
    Gummel iteration, chosen so the true nonlinear Poisson residual
    actually decreases, which is what lets it self-correct for
    grid-induced Jacobian stiffness (the counter-intuitive finer-mesh-
    fails-at-bias failure mode) without the growth behaviour that broke the
    drift-diffusion coupling above. alpha/alpha_min/alpha_max/anderson_m
    are accepted but unused in this regime (alpha_history instead records
    the line search's chosen step size each iteration).
    """
    g      = grid
    T      = g.T
    kBT_eV = kB * T / _q
    # Composition-dependent dopant ionization energies (Si donor, Mg
    # acceptor vs Al fraction -- physics.materials.algan), used by the
    # initial guess and every solve path below.
    Ed_x, Ea_x = dopant_energies(g)
    _log   = log_fn if log_fn is not None else print
    if alpha_max is None:
        alpha_max = alpha
    # alpha/alpha_min/alpha_max/anderson_m are accepted for backward
    # compatibility but no longer drive either regime: equilibrium now uses
    # pseudo-transient continuation (see "e." below, dt_ptc) instead of
    # adaptive-alpha damping + Anderson mixing, and biased solves use the
    # fully-coupled Newton solve in physics.coupled_solver. Neither has a
    # user-tunable step-size knob any more.
    if gate_bias is None:
        # A Schottky top contact is a gate. A transistor's channel is held at
        # the source potential by its lateral contacts, which a 1D stack does
        # not contain; a vertical current solve would instead let a channel
        # that is cut off from the bottom contact (AlN/GaN/AlN) drift up to
        # the gate's level, with no depletion at all.
        # A Schottky contact on a heavily doped layer is a (poor) contact of
        # a diode, not a gate: an LED with a Schottky p-contact still carries
        # a vertical current.
        # Nor is it a gate when the stack is a p-n diode (both doping types
        # present): that is an LED or a diode with a Schottky contact.
        net = np.asarray(grid.ND, dtype=float) - np.asarray(grid.NA, dtype=float)
        is_pn_diode = net.max() >= _GATE_PN_DOPING and (-net).max() >= _GATE_PN_DOPING
        gate_bias = (grid.top_contact.contact_type == 'schottky'
                     and max(float(grid.ND[-1]), float(grid.NA[-1])) < _GATE_MAX_SURFACE_DOPING
                     and not is_pn_diode
                     and interpolated_qfl is None and not flat_qfl)
    gate_bias = bool(gate_bias) and V_applied != 0.0
    interpolate = interpolated_qfl is not None and V_applied != 0.0 and not gate_bias
    if interpolate and interpolated_qfl <= 0.0:
        raise ValueError("interpolated_qfl (the insulator band gap) must be positive, in eV.")
    if gate_bias or interpolate:
        flat_qfl = True            # same Poisson(+Schrodinger) loop, see the QFL set-up below
    bias_coupled = (V_applied != 0.0) and not flat_qfl

    if gate_bias and grid.top_contact.contact_type == 'schottky':
        # A gate forward-biased beyond its own barrier height is a diode that
        # has turned on: it conducts, and no current-free band diagram exists
        # for the excess voltage (it drops across series resistance). The
        # structure is solved at the turn-on limit and the result says so.
        barrier_eV = float(grid.Ec0[-1] - grid.phi_top_eq)          # Ec - E_F at the gate, at equilibrium
        V_on = barrier_eV - _GATE_TURN_ON_MARGIN_V
        if V_applied > V_on > 0.0:
            res = solve_self_consistent(
                grid, V_applied=V_on, R_series=0.0, quantum=quantum, n_states_e=n_states_e,
                n_states_h=n_states_h, max_iter=max_iter, tol=tol, alpha=alpha, verbose=verbose,
                log_fn=log_fn, cancel_check=cancel_check, gate_bias=True)
            note = (f"The gate is forward-biased beyond its barrier ({barrier_eV:.2f} eV): it conducts. "
                    f"Shown at the turn-on limit, {V_on:.2f} V.")
            if verbose:
                (log_fn if log_fn is not None else print)("  " + note)
            return _dc_replace(res, V_applied=V_applied, V_internal=V_on, bias_note=note)

    if (gate_bias or interpolate) and phi_init is None and abs(V_applied) > _GATE_RAMP_STEP_V:
        # Walk the gate voltage up in steps, each classical solve starting
        # from the previous potential: a single jump to several volts leaves
        # the damped Poisson loop hundreds of iterations from its answer.
        n_steps = int(np.ceil(abs(V_applied) / _GATE_RAMP_STEP_V))
        for V_step in np.linspace(0.0, V_applied, n_steps + 1)[1:-1]:
            if cancel_check is not None and cancel_check():
                raise SolveCancelled("Solve cancelled by user")
            step = solve_self_consistent(
                grid, V_applied=float(V_step), R_series=0.0, quantum=False, max_iter=max_iter, tol=tol,
                verbose=False, log_fn=log_fn, cancel_check=cancel_check, gate_bias=gate_bias,
                interpolated_qfl=interpolated_qfl, phi_init=phi_init)
            phi_init = step.phi
        if verbose and phi_init is not None:
            (log_fn if log_fn is not None else print)(
                f"  Gate bias: ramped to {V_applied:+.3f} V in {n_steps} steps")

    if bias_coupled and R_series > 0.0:
        return _solve_bias_with_series_resistance(
            grid, V_applied, R_series, quantum, n_states_e, n_states_h,
            max_iter, tol, alpha, alpha_min, alpha_max, anderson_m,
            verbose, log_fn, cancel_check,
        )

    if bias_coupled and quantum and phi_init is None and Efn_init is None and Efp_init is None:
        # Quantum bias solves warm-start from the CLASSICAL solution at the
        # same bias (fast and robust -- see [[bug-efn-efp-not-splitting]]):
        # the first Schrodinger solve then sees a physically biased
        # potential instead of the equilibrium-plus-linear-ramp guess, which
        # was confirmed to make even the first classical inner solve fail
        # and to seed gamma with values pinned at its clip bounds. Falls
        # back to the usual cold start if the classical solve fails.
        classical = solve_self_consistent(
            grid, V_applied=V_applied, R_series=0.0, quantum=False,
            max_iter=max_iter, tol=tol, verbose=False,
            log_fn=log_fn, cancel_check=cancel_check,
        )
        if classical.converged:
            phi_init, Efn_init, Efp_init = classical.phi, classical.Efn, classical.Efp
            if verbose:
                _log("  Quantum bias solve: warm-started from the converged classical solution.")

    warm_started = phi_init is not None and Efn_init is not None
    if V_applied != 0.0 and phi_init is None:
        phi_init = _get_equilibrium_phi_init(
            grid, quantum, n_states_e, n_states_h, log_fn, cancel_check)

    # --- Boundary potentials ---
    phi_left  = g.phi_bottom_eq
    V_internal = V_applied
    phi_right = g.phi_top_eq + V_internal

    # --- Initial guess linear drop for bias ---
    # To prevent huge unphysical delta-function fields at the boundary or 
    # massive inversion in Efp, we distribute the applied bias linearly.
    linear_bias_drop = V_applied * (g.x_nm / g.x_nm[-1])

    # --- Electrostatics ---
    if phi_init is not None:
        phi = phi_init.copy()
        # MUST update boundaries to current V_applied! (A warm start from a
        # biased solution at this same V_applied already includes it.)
        if not warm_started:
            phi += linear_bias_drop
        phi[0] = phi_left
        phi[-1] = phi_right
    else:
        phi = np.zeros(g.N)
        
    # --- Quasi-Fermi levels ---
    # 2026-09-27: default initial guess changed from a naive SHARED linear
    # ramp (Efn = Efp = -V*x/L, identical for both carriers) to a
    # physically-motivated, contact-localized shape: each quasi-Fermi
    # level stays near its OWN contact's boundary value through most of
    # the device, transitioning only near the OTHER (opposite-type)
    # contact -- the standard textbook low-injection diode shape (Efn
    # stays ~0 through the n-region AND deep into the p-region whenever
    # the electron diffusion length exceeds the p-region's length, which
    # is the common case; Efp mirrors this for holes in the n-region).
    #
    # This was not a cosmetic choice: the coupled Newton/Gummel system has
    # multiple fixed points, and the OLD shared-linear-ramp guess reliably
    # converged to a spurious-but-numerically-stable one where Efn stayed
    # pinned equal to Efp everywhere (max|Efn-Efp| ~ 1e-4, near zero
    # regardless of V_applied -- confirmed on every device tested, from a
    # trivial 100nm/100nm GaN diode to the 45-layer UV-LED, at 1V through
    # 10V) instead of the physically correct injected solution. Seeding
    # from a guess already shaped like the correct textbook answer lands
    # the solver in the right basin: confirmed empirically, the same
    # simple diode at V=1V now converges to max|Efn-Efp| = 1.0000 (exactly
    # V_applied) with this initial guess, versus ~0.0006 with the old one
    # -- same equations, same solver, only the starting point changed.
    # See project memory bug_efn_efp_not_splitting.md for the full
    # investigation (this was on top of two other real, independently-
    # confirmed bugs fixed the same day: a residual-scale masking floor
    # in physics.coupled_solver, and a sign error in the hole continuity
    # equation's recombination term).
    _contact_transition_frac = 0.15  # fraction of device length near each
    # contact over which that contact's OWN quasi-Fermi level relaxes
    # toward the other boundary value -- not a physical device parameter,
    # just how localized this initial-guess heuristic is; the solver
    # refines from here, so this doesn't need to match any real diffusion
    # length exactly.
    _frac_x = g.x_nm / g.x_nm[-1]
    if Efn_init is not None:
        Efn = Efn_init.copy()
    else:
        Efn = -V_applied * np.clip(
            (_frac_x - (1.0 - _contact_transition_frac)) / _contact_transition_frac,
            0.0, 1.0)

    if Efp_init is not None:
        Efp = Efp_init.copy()
    else:
        Efp = -V_applied + V_applied * np.clip(
            (_contact_transition_frac - _frac_x) / _contact_transition_frac,
            0.0, 1.0)

    # Enforce contact boundary conditions immediately
    Efn[0] = 0.0
    Efn[-1] = -V_applied
    Efp[0] = 0.0
    Efp[-1] = -V_applied

    # Smooth the boundary-to-interior transition of the *initial guess*
    # (an initialization heuristic, not physics -- same idea as phi's own
    # smoothing just below) so the coupled Newton-Krylov solve starts from
    # a profile with no artificial single-cell jump at the contact.
    # Without this, only the two array endpoints above get touched -- the
    # entire interior is left at its old value (e.g. the previous,
    # lower-bias converged profile during a voltage ramp/continuation
    # step) -- creating exactly a single-cell step discontinuity right
    # where the new bias condition is applied. That discontinuity is what
    # produced the quasi-Fermi pinning bug ([[bug-quasi-fermi-pinning]]):
    # JFNK's own scaled convergence check can be satisfied by a small
    # local correction to that one cell alone -- especially in a heavily
    # one-sided-doped region, where the local carrier-scale normalization
    # is largest -- without ever needing to propagate the new bias into
    # the rest of the device, reproducing "the entire voltage drop
    # landing in one boundary cell" even though solve_coupled_dd reports
    # converged=True. Smoothing gives JFNK a physically reasonable
    # starting *gradient* to work from instead of a delta function.
    dx_typical_ef = float(np.median(np.atleast_1d(g.dx)))
    sigma_pts_ef = max(2, int(5.0e-9 / dx_typical_ef))  # same ~5nm scale as phi's smoothing below
    Efn = gaussian_filter1d(Efn, sigma=sigma_pts_ef)
    Efp = gaussian_filter1d(Efp, sigma=sigma_pts_ef)
    Efn[0] = 0.0
    Efn[-1] = -V_applied
    Efp[0] = 0.0
    Efp[-1] = -V_applied

    if flat_qfl:
        # Constant quasi-Fermi levels: each carrier's QFL sits at its own
        # contact's Fermi level at every interior node, so the split is
        # exactly qV_applied everywhere but the (ohmic, split = 0) contact
        # nodes themselves. These stay fixed for the whole solve -- the
        # V_applied != 0 path below then runs the same Poisson(+Schrodinger)
        # loop as equilibrium, just with these QFLs and biased phi BCs.
        # Which contact is the n-side is read from the net doping at the
        # two contact nodes: hard-coding n = bottom gave a split of -qV at
        # forward bias on a p-down device (the DD solve gives +qV there).
        n_at_bottom = (g.ND[0] - g.NA[0]) >= (g.ND[-1] - g.NA[-1])
        Ef_n_contact, Ef_p_contact = (0.0, -V_applied) if n_at_bottom else (-V_applied, 0.0)
        if gate_bias:
            # Gate voltage: one Fermi level, that of the source (= the bottom
            # contact's, 0), for both carriers throughout the semiconductor.
            Ef_n_contact = Ef_p_contact = 0.0
        Efn = np.full(g.N, Ef_n_contact)
        Efp = np.full(g.N, Ef_p_contact)
        if interpolate:
            # d/dz [ w dE_F/dz ] = 0 with w = exp(-Eg / E_ib): E_F is the
            # running integral of 1/w, scaled to go from 0 to -V_applied.
            inv_w = np.exp((np.asarray(g.Eg) - float(np.min(g.Eg))) / float(interpolated_qfl))
            steps = 0.5 * (inv_w[1:] + inv_w[:-1]) * np.asarray(g.dx)
            ramp = np.concatenate([[0.0], np.cumsum(steps)])
            Efn = -V_applied * ramp / ramp[-1]
            Efp = Efn.copy()
        phi_eq_ref = (_get_equilibrium_phi_init(grid, False, n_states_e, n_states_h, log_fn, cancel_check)
                      if gate_bias else None)
        if gate_bias and phi_eq_ref is not None:
            # Between the channel and the gate metal there are no carriers
            # and the level has to get from 0 to -V: a straight line across
            # that depleted barrier. The channel edge is the uppermost node
            # holding, AT EQUILIBRIUM, a mobile density above
            # _GATE_CHANNEL_FRACTION of the peak. (Not in the biased state:
            # with the level held at 0 up to the surface, a strongly negative
            # gate lifts the barrier's valence band above it and a spurious
            # surface hole layer forms, which would then count as "channel".)
            n_ref = electron_density(g.Ec0 - phi_eq_ref, 0.0, g.Nc, T)
            p_ref = hole_density(g.Ev0 - phi_eq_ref, 0.0, g.Nv, T)
            mobile = np.maximum(n_ref, p_ref)
            holds = np.where(mobile >= _GATE_CHANNEL_FRACTION * float(mobile.max()))[0]
            i_c = int(holds[-1]) if len(holds) else 0
            if i_c < g.N - 1:
                z = np.asarray(g.x_nm, dtype=float)
                ramp = -V_applied * (z[i_c:] - z[i_c]) / (z[-1] - z[i_c])
                Efn[i_c:] = ramp
                Efp[i_c:] = ramp
        Efn[0] = Efp[0] = 0.0
        Efn[-1] = Efp[-1] = -V_applied
        if verbose:
            if interpolate:
                _log(f"  Interpolated Fermi level (nextnano++ style, insulator band gap "
                     f"{interpolated_qfl:g} eV): Efn = Efp, no current.")
            if gate_bias:
                _log(f"  Gate bias: the top contact is a Schottky gate at {V_applied:+.4f} V. No current; "
                     f"the channel and everything below it keep the source Fermi level (0), and the "
                     f"level goes to the gate's across the depleted barrier.")
            _log(f"  Flat quasi-Fermi levels ({'n' if n_at_bottom else 'p'}-side at bottom): "
                 f"Efn = {Ef_n_contact:.4f}, Efp = {Ef_p_contact:.4f} eV at all interior "
                 f"nodes (split = {Ef_n_contact - Ef_p_contact:.4f} eV), Poisson-only, no current.")

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
                Ed_i = Ed_x[i]
                Nd_plus = g.ND[i] / (1.0 + 2.0 * np.exp(np.clip((Efn[i] - (Ec_guess - Ed_i)) / kBT_eV, -500, 500)))
                Na_minus = g.NA[i] / (1.0 + 4.0 * np.exp(np.clip((Ev_guess + Ea_x[i] - Efp[i]) / kBT_eV, -500, 500)))
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
    
        # Smooth aggressively. This is only an initial-guess heuristic (not
        # part of the physics), so a single representative spacing is fine
        # even on a non-uniform grid -- the median reduces to the exact
        # value on a uniform grid, leaving that case unchanged.
        dx_typical = float(np.median(np.atleast_1d(g.dx)))
        sigma_pts = max(2, int(5.0e-9 / dx_typical))  # ~5 nm screening length
        phi = gaussian_filter1d(phi, sigma=sigma_pts)
    
        # Re-enforce boundaries exactly
        phi[0] = phi_left
        phi[-1] = phi_right

    # --- Surface/interface charge states (devices.layer.SurfaceCharge) ---
    # Zero-thickness donor-/acceptor-like trap states at specific grid
    # points (see devices.grid_builder's surface_charge_sites), ionizing
    # self-consistently with the local Fermi level exactly like bulk ND/NA
    # (see the Newton-Raphson step below), but as an areal density [cm^-2]
    # converted to an equivalent volume density via the local finite-volume
    # cell width, so it plugs into the same Nd_plus/Na_minus machinery.
    surface_sites = getattr(g, 'surface_charge_sites', None) or []
    surf_idx = np.array([], dtype=int)
    surf_density_cm3 = surf_energy_eV = surf_is_donor = np.array([])
    if surface_sites:
        dx_cell_surface = node_spacings(g.dx, g.N)[2]
        idx_l, dens_l, energy_l, donor_l = [], [], [], []
        for i0, states in surface_sites:
            cw_cm = dx_cell_surface[i0] * 100.0   # m -> cm
            for state in states:
                idx_l.append(i0)
                dens_l.append(state.density_cm2 / cw_cm)
                energy_l.append(state.energy_eV)
                donor_l.append(state.state_type == 'donor')
        surf_idx = np.array(idx_l, dtype=int)
        surf_density_cm3 = np.array(dens_l)
        surf_energy_eV = np.array(energy_l)
        surf_is_donor = np.array(donor_l, dtype=bool)

    # --- Quantum region (static for the whole solve; doesn't depend on phi) ---
    if quantum:
        # Prefer an explicit user-placed region (devices.layer.
        # QuantumRegionMarker, see devices.grid_builder) over the
        # automatic undoped-span heuristic: the heuristic can sweep a large,
        # not-actually-quantum layer (e.g. an undoped graded transport
        # region) in alongside the real MQW/barrier/EBL stack, which is both
        # physically wrong and numerically fragile (a needlessly wide,
        # densely-subbanded Schrodinger solve) especially under bias.
        manual_region = getattr(g, 'manual_quantum_region', None)
        region_source = "manual"
        if manual_region is not None:
            q_i0, q_i1 = manual_region
        else:
            q_i0, q_i1 = _detect_quantum_region(g)
            region_source = "auto-detected"
        region_mask = np.zeros(g.N, dtype=bool)
        region_mask[q_i0:q_i1] = True
        if verbose:
            _log(f"  Quantum region ({region_source}): grid [{q_i0}:{q_i1}] "
                  f"= [{g.x_nm[q_i0]:.1f}, {g.x_nm[q_i1 - 1]:.1f}] nm "
                  f"(of {g.N} points spanning [0, {g.x_nm[-1]:.1f}] nm)")

    E_e = E_h = psi_e = psi_h = None
    E_h_lh = psi_h_lh = E_h_so = psi_h_so = None
    converged = False
    n_iter    = 0
    residual  = np.inf
    prev_residual = np.inf
    residual_history: list = []
    alpha_history: list = []
    good_streak = 0   # consecutive non-worsening iterations (growth needs a streak, not one lucky step)
    dt_ptc = 1e-3     # equilibrium-only PTC pseudo-time step (see "d/e" below)

    # Bias-coupled outer-loop fail-fast (nextnano++-style: its own
    # convergence tutorial explicitly advises against waiting out a
    # max_iter budget once it's clear a solve isn't converging -- see the
    # nextnano++ documentation research this session did). Before this,
    # a device/voltage combination whose inner coupled Newton solve
    # (solve_coupled_dd) never converges would still burn the *entire*
    # max_iter outer-loop budget (which the GUI defaults to 5000) doing
    # so, each outer iteration re-running a ~40-step inner PTC-direct
    # solve from scratch -- this is what produced the hour-long hangs on
    # non-convergent voltages. Tracks the best (smallest) outer |dphi|
    # residual seen so far; if it hasn't meaningfully improved for
    # _OUTER_STAGNATION_PATIENCE consecutive iterations *and* the inner
    # coupled solve is also failing to converge on its own terms, the
    # outer loop is not making progress and won't suddenly start --
    # abort early rather than continue for possibly thousands more
    # iterations of the same expensive, fruitless work.
    best_outer_residual = np.inf
    stagnant_outer_iters = 0

    # Quantum bias solves: log-space under-relaxation of the correction
    # factors between outer iterations (see _quantum_gamma's caller below).
    # Undamped, the Schrodinger <-> transport outer loop was confirmed to
    # fall into a 2-cycle on the UV-LED MQW (|dphi| ~4 V alternating, gamma
    # flipping between two states) -- classic Schrodinger-Poisson charge
    # sloshing. beta adapts like the rest of this module: grows on an
    # improving outer step, halves when |dphi| gets worse.
    log_gamma_n_prev = log_gamma_p_prev = None
    gamma_beta = _QUANTUM_GAMMA_BETA_INIT
    gamma_mismatch = np.inf

    # Drift-diffusion state (physics.dd_newton): whether (phi, Efn, Efp)
    # already hold a converged biased solution to re-solve from, and the
    # classical equilibrium potential the bias ramp starts from.
    have_dd_state = warm_started
    phi_eq_dd = None
    dd_prob = None
    dd_floor = 0.0      # transport floor a previous drift-diffusion solve of this run needed

    for iteration in range(max_iter):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")

        # a. Band edges
        Ec = g.Ec0 - phi
        Ev = g.Ev0 - phi

        # b/c. Carrier densities
        if quantum:
            # Energy cutoff _SUBBAND_CUTOFF_KT kT past the quasi-Fermi level
            # (in the hole solver's inverted -Ev frame for holes): states
            # beyond it carry negligible charge, states below it must all be
            # present -- see _solve_confined_states.
            E_cut_e = float(np.max(Efn[region_mask])) + _SUBBAND_CUTOFF_KT * kBT_eV
            E_cut_h = float(-np.min(Efp[region_mask])) + _SUBBAND_CUTOFF_KT * kBT_eV
            E_e, psi_e = _solve_confined_states(Ec, g.m_e, g.dx, n_states_e, q_i0, q_i1, g.N, E_cut_e)
            n = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T, bias_coupled),
                                  electron_density(Ec, Efn, g.Nc, T), region_mask)

            # Decoupled 3-band effective-mass valence model (HH/LH/SO) --
            # see _solve_hole_bands and physics.materials.algan.
            # AlGaNParams.valence_band_structure. E_h/psi_h stays the HH band
            # for backward compatibility (QCSE optics, existing plots).
            hole_bands = _solve_hole_bands(Ev, g, g.dx, q_i0, q_i1, n_states_h, E_cut_h)
            E_h, psi_h = hole_bands['hh']
            E_h_lh, psi_h_lh = hole_bands['lh']
            E_h_so, psi_h_so = hole_bands['so']
            p_quantum = _total_quantum_hole_density(hole_bands, Efp, T, g, bias_coupled)
            p = _blended_density(p_quantum, hole_density(Ev, Efp, g.Nv, T), region_mask)
        else:
            n = electron_density(Ec, Efn, g.Nc, T)
            p = hole_density(Ev, Efp, g.Nv, T)


        if bias_coupled:
            # Biased solves: one fully-coupled Newton-Krylov solve of
            # Poisson + electron continuity + hole continuity together
            # (see physics.coupled_solver), replacing the historical
            # sequential Gummel step (a single linear continuity solve
            # given the *previous* iteration's potential, then a separate
            # linearised Poisson step, then damped mixing between the two).
            # That sequential scheme's continuity half had no line search
            # or step-size control of its own; adding one only to the
            # Poisson half (an Armijo search, still used below for
            # V_applied==0) left the continuity half as the actual
            # bottleneck -- confirmed empirically, the high-bias wall
            # barely moved after that fix alone. JFNK needs no analytic
            # Jacobian and includes its own Armijo line search by default.
            # Quantum-corrected drift-diffusion (predictor-corrector): the
            # Schrodinger solution from step b is frozen into per-node
            # correction factors gamma = n_quantum / n_classical (1 outside
            # the quantum region), and the inner coupled solve uses
            # gamma * n_classical(phi, Efn) as THE carrier density in
            # Poisson, continuity and recombination alike. The classical
            # exp((Efn-Ec)/kT) response to phi/Efn is an excellent predictor
            # of how the subband charge responds (subbands ride along with
            # Ec), so the outer loop only has to correct gamma's slow
            # residual dependence on well shape. Replaces the old scheme
            # (classical inner solve + a damped post-hoc quantum Poisson
            # patch on phi), which solved for two different electron
            # populations in the well and never reconciled them -- see
            # [[bug-quantum-bias-electron-divergence]].
            # The first outer iteration stays classical: its Schrodinger
            # solve saw the bias-less initial-guess potential, and a gamma
            # built from that (confirmed: pinned at the clip bounds across
            # most of the region) throws the inner solve far from the
            # biased solution. The classical bias solution is fast and
            # robust, and a physically sensible potential for the first
            # real Schrodinger solve.
            quantum_gamma_n = quantum_gamma_p = None
            if quantum and (iteration > 0 or warm_started):
                n_cl_now = electron_density(Ec, Efn, g.Nc, T)
                p_cl_now = hole_density(Ev, Efp, g.Nv, T)
                lg_n = np.log(_quantum_gamma(n, n_cl_now, region_mask))
                lg_p = np.log(_quantum_gamma(p, p_cl_now, region_mask))
                if log_gamma_n_prev is not None:
                    # Self-consistency of the quantum correction itself: how
                    # far the gamma used last solve is from the one the
                    # fresh Schrodinger solution implies, in kT units, at
                    # nodes carrying non-negligible charge. Needed alongside
                    # |dphi| because under-relaxation can make |dphi| small
                    # just by shrinking gamma_beta.
                    carries_n = n_cl_now * np.exp(log_gamma_n_prev) > _GAMMA_MISMATCH_MIN_DENSITY
                    carries_p = p_cl_now * np.exp(log_gamma_p_prev) > _GAMMA_MISMATCH_MIN_DENSITY
                    gamma_mismatch = max(
                        float(np.max(np.abs(lg_n - log_gamma_n_prev)[carries_n], initial=0.0)),
                        float(np.max(np.abs(lg_p - log_gamma_p_prev)[carries_p], initial=0.0)))
                    if len(residual_history) >= 2 and residual_history[-1] > residual_history[-2]:
                        gamma_beta = max(gamma_beta * 0.5, _QUANTUM_GAMMA_BETA_MIN)
                    else:
                        gamma_beta = min(gamma_beta * 1.2, 1.0)
                    lg_n = log_gamma_n_prev + gamma_beta * (lg_n - log_gamma_n_prev)
                    lg_p = log_gamma_p_prev + gamma_beta * (lg_p - log_gamma_p_prev)
                log_gamma_n_prev, log_gamma_p_prev = lg_n, lg_p
                quantum_gamma_n = np.exp(lg_n)
                quantum_gamma_p = np.exp(lg_p)

            # Fully-coupled analytic-Jacobian Newton (physics.dd_newton),
            # converged on the TRUE equations (see [[bug-current-not-conserved]]
            # for why the previous PTC/Gummel solver was replaced). The
            # first biased solve ramps the bias up from the converged
            # equilibrium; later outer iterations (quantum gamma updates)
            # re-solve from the current state, falling back to a ramp.
            dd_prob = _dd_problem(g, Ed_x, Ea_x, surf_idx, surf_density_cm3,
                                  surf_energy_eV, surf_is_donor,
                                  quantum_gamma_n, quantum_gamma_p)
            dd_prob.transport_floor = dd_floor
            dd_log = (_log if verbose else None)
            cd_result = None
            if have_dd_state:
                cd_result = dd_newton.newton_solve(
                    dd_prob, phi, Efn, Efp,
                    (phi_left, phi_right, 0.0, -V_internal, 0.0, -V_internal),
                    cancel_check=cancel_check)
            if cd_result is None or not cd_result.converged:
                if phi_eq_dd is None:
                    phi_eq_dd = _get_equilibrium_phi_init(
                        grid, False, n_states_e, n_states_h, log_fn, cancel_check)
                if phi_eq_dd is None:
                    raise RuntimeError("Classical equilibrium solve did not converge; "
                                       "cannot start the bias ramp.")
                cd_result = dd_newton.solve_bias_ramp(
                    dd_prob, phi_eq_dd, V_internal, phi_left, g.phi_top_eq,
                    log_fn=dd_log, cancel_check=cancel_check)
            have_dd_state = have_dd_state or cd_result.converged
            dd_floor = max(dd_floor, dd_prob.transport_floor)
            phi_candidate = cd_result.phi
            Efn = cd_result.Efn
            Efp = cd_result.Efp

            # Reuses alpha_history's slot to record the NK iteration count
            # for this step (not a damping factor -- there isn't one here).
            alpha = float(cd_result.n_iter)
            alpha_history.append(alpha)

            residual = np.max(np.abs(phi_candidate - phi))
            residual_history.append(float(residual))

            # R_series > 0 is handled entirely by _solve_bias_with_series_
            # resistance before this loop ever runs (see the top of this
            # function) -- this call always has R_series == 0, so V_internal
            # is just V_applied, fixed for the whole solve.
            Ec_cd = g.Ec0 - phi_candidate
            Ev_cd = g.Ev0 - phi_candidate
            if quantum:
                n = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T, bias_coupled),
                                      electron_density(Ec_cd, Efn, g.Nc, T), region_mask)
                p_quantum = _total_quantum_hole_density(hole_bands, Efp, T, g, bias_coupled)
                p = _blended_density(p_quantum,
                                      hole_density(Ev_cd, Efp, g.Nv, T), region_mask)
            else:
                n = electron_density(Ec_cd, Efn, g.Nc, T)
                p = hole_density(Ev_cd, Efp, g.Nv, T)

            V_internal = V_applied
            phi_right = g.phi_top_eq + V_applied
        else:
            # d. Newton-Raphson Poisson step (equilibrium only -- biased
            # solves use the coupled solver above instead).
            #
            # Clamp exponent arguments to prevent overflow. +-500 alone isn't
            # tight enough here: dNd_dphi/dNa_dphi below square (1+2*exp_d) /
            # (1+4*exp_a), so exp(500) (~1.4e217) squares to ~1e434 and
            # overflows float64 (max ~1.8e308) -- silently underflowing the
            # derivative to 0.0 (large-finite / inf) rather than raising,
            # exactly in the deep-saturation regions high bias drives phi
            # into. +-340 keeps the squared term safely finite (physically
            # inconsequential: exp_arg=340 is already far past full
            # ionization saturation, reached by exp_arg ~ 50-100).
            exp_arg_d = np.clip((Efn - (Ec - Ed_x)) / kBT_eV, -340.0, 340.0)
            exp_arg_a = np.clip((Ev + Ea_x - Efp) / kBT_eV, -340.0, 340.0)

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

            # Surface/interface charge states: same Fermi-Dirac ionization form
            # as the bulk terms just above, added at each state's grid index
            # (np.add.at so multiple states sharing one interface accumulate
            # rather than overwrite).
            if len(surf_idx) > 0:
                donor_mask = surf_is_donor
                if np.any(donor_mask):
                    idx_d = surf_idx[donor_mask]
                    exp_arg_sd = np.clip(
                        (Efn[idx_d] - (Ec[idx_d] - surf_energy_eV[donor_mask])) / kBT_eV,
                        -340.0, 340.0)
                    exp_sd = np.exp(exp_arg_sd)
                    Nsd_plus = surf_density_cm3[donor_mask] / (1.0 + 2.0 * exp_sd)
                    dNsd_dphi = -surf_density_cm3[donor_mask] * (2.0 * exp_sd) / (1.0 + 2.0 * exp_sd)**2 / kBT_eV
                    np.add.at(Nd_plus, idx_d, Nsd_plus)
                    np.add.at(dNd_dphi, idx_d, dNsd_dphi)

                acceptor_mask = ~surf_is_donor
                if np.any(acceptor_mask):
                    idx_a = surf_idx[acceptor_mask]
                    exp_arg_sa = np.clip(
                        (Ev[idx_a] + surf_energy_eV[acceptor_mask] - Efp[idx_a]) / kBT_eV,
                        -340.0, 340.0)
                    exp_sa = np.exp(exp_arg_sa)
                    Nsa_minus = surf_density_cm3[acceptor_mask] / (1.0 + 4.0 * exp_sa)
                    dNsa_dphi = surf_density_cm3[acceptor_mask] * (4.0 * exp_sa) / (1.0 + 4.0 * exp_sa)**2 / kBT_eV
                    np.add.at(Na_minus, idx_a, Nsa_minus)
                    np.add.at(dNa_dphi, idx_a, dNsa_dphi)

            # Dynamic relaxation for stability
            # If D_int gets too small (e.g. in depleted regions), force a minimum D
            # to limit the max potential step.
            #
            # This floor is intentionally donor-only, matching the solver's
            # existing tuning: it's unconditional (applies even where ND=0, i.e.
            # it isn't really "only where donors are present"), and mirroring it
            # onto dNa_dphi -- tried during debugging -- doubles that same
            # artificial forcing at every single grid point and measurably
            # *hurts* convergence on other devices (regressed the UV-LED MQW
            # robustness regression test from >=6 converged sweep points to 2).
            # The overflow-safe exponent clip above already fixes the real bug
            # (dNa_dphi silently collapsing to 0.0 instead of its correct
            # saturated value); it doesn't also need a floor to be correct.
            min_D_equivalent_n = 1e16  # cm^-3 pseudo-carrier density for damping
            dNd_dphi = np.minimum(dNd_dphi, -min_D_equivalent_n / kBT_eV)

            # e. Pseudo-transient continuation (equilibrium only).
            #
            # Previously: a plain Newton step (solve_poisson_newton) mixed
            # into phi afterward via safeguarded, adaptive-alpha damping +
            # Anderson acceleration. That reliably converges ordinary
            # devices, but was found to settle into a genuine bounded limit
            # cycle (residual wandering in a narrow band indefinitely,
            # never shrinking below it even after 3000+ iterations, alpha
            # permanently floored) on a device with many densely-packed
            # (~2nm period) heterointerfaces with large polarization
            # discontinuities -- confirmed not a slow-but-real convergence
            # (residual genuinely stops shrinking, doesn't just shrink
            # slowly) via a from-scratch grid-point-level trace. Post-hoc
            # linear damping/extrapolation of an already-computed Newton
            # step can't fix that: it still fully trusts the same
            # undamped, potentially-oversized step direction every time.
            #
            # PTC instead damps the *Jacobian* directly (see
            # solve_poisson_newton_ptc), the same technique already proven
            # for exactly this "stiff/ill-conditioned, damped-mixing
            # oscillates or stalls" failure mode on the fully-coupled bias
            # system (physics.coupled_solver._solve_ptc_direct -- see
            # [[bug-quasi-fermi-pinning]]): start from a small, heavily-
            # damped pseudo-time step (dt_ptc), accept it only if the TRUE
            # nonlinear Poisson residual (_nonlinear_poisson_residual_norm,
            # previously dead code from an abandoned bias-solve line search
            # -- reused here) actually improves, growing dt (SER) on
            # acceptance and shrinking it (retrying from the same phi) on
            # rejection. dt -> infinity recovers the plain Newton step
            # exactly, so this never gives up robustness compared to before,
            # only adds it.
            res_current = _nonlinear_poisson_residual_norm(
                phi, g, Efn, Efp, T, kBT_eV, Ed_x, Ea_x,
                surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
            )
            n_ptc_tries = 0
            while True:
                n_ptc_tries += 1
                phi_trial = solve_poisson_newton_ptc(
                    phi, g.eps_r, n, p, Nd_plus, Na_minus, dNd_dphi, dNa_dphi,
                    g.pol_rho, g.dx, phi_left, phi_right, T, dt_ptc,
                )
                res_trial = _nonlinear_poisson_residual_norm(
                    phi_trial, g, Efn, Efp, T, kBT_eV, Ed_x, Ea_x,
                    surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
                )
                if res_trial <= res_current or n_ptc_tries >= 30:
                    growth = min(res_current / max(res_trial, 1e-300), 3.0)
                    dt_ptc *= max(growth, 1.0)
                    break
                dt_ptc *= 0.3

            phi_candidate = phi_trial
            residual = float(np.max(np.abs(phi_candidate - phi)))
            residual_history.append(residual)
            alpha = dt_ptc   # reuse alpha_history's slot to record dt_ptc
            alpha_history.append(float(alpha))
            use_aa = False

        if verbose:
            if bias_coupled:
                _log(f"  outer iter {iteration+1:4d}  |dphi|_max = {residual:.3e} V  "
                      f"DD Newton {'converged' if cd_result.converged else 'DID NOT CONVERGE'} "
                      f"in {cd_result.n_iter} iters (max update={cd_result.max_update:.3e} V)"
                      + (f"  |dln(gamma)|={gamma_mismatch:.2e} beta={gamma_beta:.2f}"
                         if quantum else ""))
            else:
                _log(f"  iter {iteration+1:4d}  |dphi|_max = {residual:.3e} V  "
                      f"PTC dt = {alpha:.3e}")

        phi = phi_candidate

        # Enforce BCs exactly (mixing can slightly drift boundary values)
        phi[0]  = phi_left
        phi[-1] = phi_right

        prev_residual = residual
        n_iter = iteration + 1
        if bias_coupled:
            # The inner drift-diffusion solve must itself have converged:
            # a failed solve that hands back its input unchanged gives
            # |dphi| = 0 and used to be reported as converged (see
            # [[bug-current-not-conserved]]). The DD solver already ramps
            # and retries internally, so repeating it here is pointless --
            # fail honestly instead.
            if not cd_result.converged:
                # The returned state is the last converged ramp point, so
                # report the bias it actually corresponds to (shown next to
                # V_applied in the result/plot title) rather than V_applied.
                V_internal = float(cd_result.V_reached)
                # The generic BC enforcement above just wrote the V_applied
                # contact potential into phi[-1]; restore the reached one or
                # the last edge's current is garbage (confirmed: that alone
                # made a current-conserving 5.14 V state report a
                # conservation error of 1.0).
                # (phi aliases cd_result.phi, so it was overwritten in place
                # too -- set the contact value explicitly.)
                phi_right = g.phi_top_eq + V_internal
                phi[-1] = phi_right
                if verbose:
                    _log(f"  Drift-diffusion solve failed ({cd_result.message}) -- "
                         f"reporting converged=False; state shown is at "
                         f"V = {V_internal:.4f} V.")
                break
            if residual < tol and (not quantum or gamma_mismatch < _GAMMA_MISMATCH_TOL):
                converged = True
                break

            # Fail-fast stagnation check (see best_outer_residual's
            # initialization above for the rationale). Only the inner
            # solve's own convergence counts as "not converging" -- a
            # converged inner solve with a slowly-shrinking outer residual
            # is normal, legitimate progress, not stagnation.
            if residual < best_outer_residual * _OUTER_STAGNATION_MIN_IMPROVEMENT:
                best_outer_residual = float(residual)
                stagnant_outer_iters = 0
            elif not cd_result.converged:
                stagnant_outer_iters += 1
            if stagnant_outer_iters >= _OUTER_STAGNATION_PATIENCE:
                if verbose:
                    _log(f"  Outer loop stagnant for {stagnant_outer_iters} iterations "
                         f"(inner coupled solve not converging, |dphi| stuck near "
                         f"{best_outer_residual:.3e} V) -- aborting early instead of "
                         f"continuing to max_iter={max_iter} (nextnano++-style fail-fast).")
                break
        elif residual < tol:
            converged = True
            break

    if not converged and verbose and n_iter >= max_iter:
        _log(f"  Warning: did not converge after {max_iter} iterations "
              f"(|dphi|={residual:.2e} V)")

    # --- Final band edges and carrier densities ---
    Ec_final = g.Ec0 - phi
    Ev_final = g.Ev0 - phi
    Ei_final = (0.5 * (Ec_final + Ev_final)
                + 0.5 * kBT_eV * np.log(g.Nv / g.Nc))
    E_field  = electric_field(phi, g.dx)

    if quantum and psi_e is not None:
        n_final = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T, bias_coupled),
                                    electron_density(Ec_final, Efn, g.Nc, T), region_mask)
        p_final_quantum = _total_quantum_hole_density(hole_bands, Efp, T, g, bias_coupled)
        p_final = _blended_density(p_final_quantum,
                                    hole_density(Ev_final, Efp, g.Nv, T), region_mask)
        Ev_hh_final, Ev_lh_final, Ev_so_final = (
            _hole_band_edges(Ev_final, g)[k] for k in ('hh', 'lh', 'so')
        )
    else:
        n_final = electron_density(Ec_final, Efn, g.Nc, T)
        p_final = hole_density(Ev_final, Efp, g.Nv, T)
        Ev_hh_final, Ev_lh_final, Ev_so_final = (
            _hole_band_edges(Ev_final, g)[k] for k in ('hh', 'lh', 'so')
        )

    qcse_transition_eV = None
    qcse_overlap = None
    qcse_pair = None
    qcse_in_well = False
    qcse_local_solve = False
    qcse_dominant_transition_eV = None
    qcse_dominant_overlap = None
    qcse_dominant_pair = None
    qw_window_nm = None

    if getattr(g, 'qw_window', None) is not None:
        qw_i0, qw_i1 = g.qw_window
        qw_window_nm = (float(g.x_nm[qw_i0]), float(g.x_nm[qw_i1 - 1]))

    surface_charge_markers = []
    for i0, states in surface_sites:
        for state in states:
            if state.state_type == 'donor':
                level = float(Ec_final[i0] - state.energy_eV)
            else:
                level = float(Ev_final[i0] + state.energy_eV)
            surface_charge_markers.append(
                (float(g.x_nm[i0]), level, state.state_type, state.density_cm2))

    if quantum:
        # physics.optical's overlap/transition helpers integrate over the
        # domain point-by-point, so they need the per-*node* quadrature
        # weight (cell_width, length N) -- not g.dx, which is the per-*edge*
        # spacing array (length N-1) the stencil solvers use.
        dx_cell = node_spacings(g.dx, g.N)[2]

        # Prefer the layer stack's own quantum well (devices.grid_builder's
        # qw_window) over the global lowest-energy state: the latter can be
        # a polarization/doping-induced notch elsewhere in the device
        # rather than the well the user actually designed.
        if g.qw_window is not None:
            ie_sel = _first_state_in_window(E_e, psi_e, g.qw_window, dx_cell)
            ih_sel = _first_state_in_window(E_h, psi_h, g.qw_window, dx_cell)
            if ie_sel is not None and ih_sel is not None:
                qcse_overlap = overlap_squared(psi_e[ie_sel], psi_h[ih_sel], dx_cell)
                Ev_sub = hole_subband_energy(E_h)
                qcse_transition_eV = float(E_e[ie_sel] - Ev_sub[ih_sel])
                qcse_pair = (ie_sel, ih_sel)
                qcse_in_well = True

            # No globally-solved state sits in the designed well (typical at
            # V=0 in p-i-n LEDs: the quantum region's lowest states are in a
            # depletion-edge notch, so the well's own subbands rank past
            # n_states). Solve Schrodinger locally on the well plus up to
            # 4 nm of barrier each side for the well's own e1/h1 -- the
            # physically meaningful QCSE transition -- instead of pairing an
            # n-side electron with a p-side hole.
            if qcse_transition_eV is None:
                loc = _local_well_transition(g, Ec_final, Ev_final, qw_i0, qw_i1, dx_cell)
                if loc is not None:
                    qcse_transition_eV, qcse_overlap = loc
                    qcse_pair = (0, 0)
                    qcse_in_well = True
                    qcse_local_solve = True

        if qcse_transition_eV is None:
            e1h1 = ground_state_transition(E_e, psi_e, E_h, psi_h, dx_cell)
            if e1h1 is not None:
                qcse_transition_eV = e1h1.energy_eV
                qcse_overlap = e1h1.overlap
                qcse_pair = (e1h1.ie, e1h1.ih)
                qcse_in_well = False

        dom = dominant_transition(E_e, psi_e, E_h, psi_h, dx_cell,
                                   n_e=len(E_e), n_h=len(E_h))
        if dom is not None:
            qcse_dominant_transition_eV = dom.energy_eV
            qcse_dominant_overlap = dom.overlap
            qcse_dominant_pair = (dom.ie, dom.ih)

    # --- Recombination rates [cm^-3 s^-1], one value per grid point ---
    ni_final = np.sqrt(g.Nc * g.Nv) * np.exp(-(Ec_final - Ev_final) / (2.0 * kBT_eV))
    rec = compute_recombination_components(n_final, p_final, ni_final,
                                           split_eV=np.asarray(Efn) - np.asarray(Efp), kT_eV=kBT_eV,
                                           **recombination_coefficients(g))

    # --- Optical gain spectrum (quantum mode, only if a design quantum
    # well was found -- see physics.gain for the model and its caveats) ---
    gain_energy_eV = gain_wavelength_nm = gain_spectrum_cm1 = None
    peak_gain_cm1 = peak_gain_wavelength_nm = None
    if quantum and E_e is not None and g.qw_window is not None:
        qw_i0, qw_i1 = g.qw_window
        Lz_m = max(float(g.x_nm[qw_i1 - 1] - g.x_nm[qw_i0]) * 1e-9, 1e-10)
        x_Al_well = float(np.mean(g.x_Al[qw_i0:qw_i1]))
        Efn_well = float(np.mean(Efn[qw_i0:qw_i1]))
        Efp_well = float(np.mean(Efp[qw_i0:qw_i1]))
        gain_kw = {}
        if getattr(g, 'crystal', 'wurtzite') == 'zincblende':
            from physics.materials.zincblende import get_zincblende_params
            well = get_zincblende_params(x_Al_well, float(np.mean(g.x_In[qw_i0:qw_i1])),
                                         float(np.mean(g.x_P[qw_i0:qw_i1])), T)
            gain_kw = {'Ep_eV': well.Ep, 'n_r': 3.5}   # arsenide/phosphide refractive index
        spectrum = compute_gain_spectrum(
            E_e, psi_e, g.m_e, hole_bands,
            {'hh': g.m_hh, 'lh': g.m_lh, 'so': g.m_so},
            Efn_well, Efp_well, T, dx_cell, Lz_m, x_Al_well, **gain_kw,
        )
        if spectrum is not None:
            gain_energy_eV = spectrum.energy_eV
            gain_wavelength_nm = spectrum.wavelength_nm
            gain_spectrum_cm1 = spectrum.gain_cm1
            peak_gain_cm1 = spectrum.peak_gain_cm1
            peak_gain_wavelength_nm = spectrum.peak_wavelength_nm

    J_n_edges = J_p_edges = None
    J_total = float('nan')
    current_err = float('nan')
    if bias_coupled and dd_prob is not None:
        J_n_edges, J_p_edges, J_t_edges, current_err = dd_newton.current_profile(dd_prob, phi, Efn, Efp)
        J_total = float(np.mean(J_t_edges))
        if dd_floor > 0.0 and verbose:
            _log(f"  NOTE: a layer is cut off from both contacts, so its quasi-Fermi level was fixed with a "
                 f"minimum density of {dd_floor:.1e} cm^-3 in the current equation. The band diagram and "
                 f"densities are reliable; a current this small is the leakage of that floor, not a prediction.")
        if verbose:
            _log(f"  Current density J = {J_total:.4e} A/cm^2, conservation error "
                 f"(max-min)/max|J| = {current_err:.2e}")

    return SolverResult(
        x_nm=g.x_nm, x_Al=g.x_Al,
        J_n=J_n_edges, J_p=J_p_edges, J_total=J_total,
        current_conservation_error=current_err,
        Ec=Ec_final, Ev=Ev_final, Ei=Ei_final,
        Efn=Efn, Efp=Efp,
        phi=phi, E_field=E_field, F_quasi=g.F_quasi,
        n=n_final, p=p_final,
        Psp=g.Psp, Ppz=g.Ppz, P_total=g.P_total,
        eps_xx=g.eps_xx, eps_zz=g.eps_zz,
        E_e=E_e, psi_e=psi_e, E_h=E_h, psi_h=psi_h,
        Ev_hh=Ev_hh_final, Ev_lh=Ev_lh_final, Ev_so=Ev_so_final,
        chi=np.asarray(g.chi).copy(),
        x_In=(np.asarray(g.x_In).copy() if getattr(g, 'x_In', None) is not None
              else np.zeros_like(np.asarray(g.x_Al))),
        crystal=getattr(g, 'crystal', 'wurtzite'),
        transport_floor_cm3=float(dd_floor),
        bias_mode=('equilibrium' if V_applied == 0.0 else 'gate' if gate_bias
                   else 'interpolated' if interpolate else 'flat' if flat_qfl else 'current'),
        eps_r=np.asarray(g.eps_r).copy(),
        E_polarization=-np.asarray(g.P_total) / (eps0 * np.asarray(g.eps_r)),
        D_field=eps0 * np.asarray(g.eps_r) * np.asarray(E_field) + np.asarray(g.P_total),
        x_P=(np.asarray(g.x_P).copy() if getattr(g, 'x_P', None) is not None
             else np.zeros_like(np.asarray(g.x_Al))),
        E_h_lh=E_h_lh, psi_h_lh=psi_h_lh, E_h_so=E_h_so, psi_h_so=psi_h_so,
        qcse_transition_eV=qcse_transition_eV, qcse_overlap=qcse_overlap,
        qcse_pair=qcse_pair, qcse_in_well=qcse_in_well, qcse_local_solve=qcse_local_solve,
        qw_window_nm=qw_window_nm,
        qcse_dominant_transition_eV=qcse_dominant_transition_eV,
        qcse_dominant_overlap=qcse_dominant_overlap,
        qcse_dominant_pair=qcse_dominant_pair,
        # With flat quasi-Fermi levels under bias no current equation is
        # solved, so the splitting is imposed rather than computed and the
        # rates it would imply mean nothing: they are not reported.
        R_srh=None if (flat_qfl and V_applied != 0.0) else rec.R_srh,
        R_rad=None if (flat_qfl and V_applied != 0.0) else rec.R_rad,
        R_aug=None if (flat_qfl and V_applied != 0.0) else rec.R_aug,
        gain_energy_eV=gain_energy_eV, gain_wavelength_nm=gain_wavelength_nm,
        gain_spectrum_cm1=gain_spectrum_cm1, peak_gain_cm1=peak_gain_cm1,
        peak_gain_wavelength_nm=peak_gain_wavelength_nm,
        surface_charge_markers=surface_charge_markers,
        V_applied=V_applied, V_internal=V_internal, T=T,
        converged=converged, n_iterations=n_iter,
        interface_indices=g.interface_indices,
        interface_sigmas=g.interface_sigmas,
        residual_history=residual_history, alpha_history=alpha_history,
        Ec0=g.Ec0, Ev0=g.Ev0, ND=g.ND, NA=g.NA,
    )

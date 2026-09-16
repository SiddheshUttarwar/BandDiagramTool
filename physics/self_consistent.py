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
from typing import Callable, Optional

from devices.grid_builder import GridData
from physics.fermi_dirac import (
    electron_density, hole_density,
    quantum_electron_density, quantum_hole_density,
    Efn_from_n, Efp_from_p,
)
from physics.poisson import solve_poisson, solve_poisson_newton, electric_field, _assemble_laplacian
from physics.schrodinger import solve_schrodinger, hole_potential
from physics.drift_diffusion import (
    solve_continuity_electron, solve_continuity_hole, compute_recombination,
    compute_current_density,
)
from physics.coupled_solver import solve_coupled_dd, QuantumState, SolveCancelled
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
                            n_states: int, i0: int, i1: int, N: int):
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
    """
    dx_sub = dx if np.ndim(dx) == 0 else np.asarray(dx)[i0:i1 - 1]
    E, psi_sub = solve_schrodinger(V_eV_full[i0:i1], m_full[i0:i1], dx_sub, n_states=n_states)
    psi_full = np.zeros((len(E), N))
    psi_full[:, i0:i1] = psi_sub
    return E, psi_full


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


# Hole band name -> (band-edge-below-Ev0 array attr, mass array attr) on
# GridData, for the decoupled 3-band effective-mass valence model (see
# physics.materials.algan.AlGaNParams.valence_band_structure). 'hh' has no
# offset (Ev0 *is* the HH edge in this module's convention).
_HOLE_BAND_MASS_ATTR = {'hh': 'm_hh', 'lh': 'm_lh', 'so': 'm_so'}


def _hole_band_edges(Ev: np.ndarray, g: GridData) -> dict:
    """HH/LH/SO valence band-edge profiles [eV] at the current phi, on the
    same absolute scale as Ev = g.Ev0 - phi. LH/SO offsets are static (see
    devices.grid_builder's dEv_lh/dEv_so), so they just subtract straight
    off Ev like the phi shift already applied to it."""
    return {'hh': Ev, 'lh': Ev - g.dEv_lh, 'so': Ev - g.dEv_so}


def _solve_hole_bands(Ev: np.ndarray, g: GridData, dx, q_i0: int, q_i1: int,
                       n_states_h: int) -> dict:
    """Solve HH/LH/SO confined states as three independent single-band
    Schrodinger problems sharing the same quantum-region window, each with
    its own band edge and effective mass. Returns {'hh'/'lh'/'so': (E, psi)}."""
    edges = _hole_band_edges(Ev, g)
    bands = {}
    for name, Ev_band in edges.items():
        m_band = getattr(g, _HOLE_BAND_MASS_ATTR[name])
        V_hole = hole_potential(Ev_band)
        bands[name] = _solve_confined_states(V_hole, m_band, dx, n_states_h, q_i0, q_i1, g.N)
    return bands


def _total_quantum_hole_density(bands: dict, Efp, T: float, g: GridData) -> np.ndarray:
    """Sum quantum_hole_density across HH/LH/SO -- position-space density
    is additive across independent (decoupled, unmixed) bands."""
    total = np.zeros(g.N)
    for name, (E, psi) in bands.items():
        m_band = getattr(g, _HOLE_BAND_MASS_ATTR[name])
        total = total + quantum_hole_density(psi, E, Efp, m_band, T)
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
    _log   = log_fn if log_fn is not None else print
    if alpha_max is None:
        alpha_max = alpha
    # See "Note on scope" above: adaptive alpha / safeguarded AA are only
    # used for the unbiased electrostatic loop; biased solves use the
    # Armijo backtracking line search instead.
    bias_coupled = (V_applied != 0.0)
    # Starting step size for the biased-solve line search below: the
    # caller's alpha, captured once here so it stays fixed as a ceiling
    # across iterations (the `alpha` name itself gets reused per-iteration
    # for logging/alpha_history). See the bias_coupled branch of "e. Mixing".
    bias_alpha_ceiling = alpha

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

    # --- Anderson mixer (safeguarded: reset + fall back to damping whenever
    # the residual grows from one iteration to the next) ---
    mixer = _AndersonMixer(m=anderson_m, alpha=alpha)

    E_e = E_h = psi_e = psi_h = None
    E_h_lh = psi_h_lh = E_h_so = psi_h_so = None
    converged = False
    n_iter    = 0
    residual  = np.inf
    prev_residual = np.inf
    residual_history: list = []
    alpha_history: list = []
    good_streak = 0   # consecutive non-worsening iterations (growth needs a streak, not one lucky step)

    for iteration in range(max_iter):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")

        # a. Band edges
        Ec = g.Ec0 - phi
        Ev = g.Ev0 - phi

        # b/c. Carrier densities
        if quantum:
            E_e, psi_e = _solve_confined_states(Ec, g.m_e, g.dx, n_states_e, q_i0, q_i1, g.N)
            n = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T),
                                  electron_density(Ec, Efn, g.Nc, T), region_mask)

            # Decoupled 3-band effective-mass valence model (HH/LH/SO) --
            # see _solve_hole_bands and physics.materials.algan.
            # AlGaNParams.valence_band_structure. E_h/psi_h stays the HH band
            # for backward compatibility (QCSE optics, existing plots).
            hole_bands = _solve_hole_bands(Ev, g, g.dx, q_i0, q_i1, n_states_h)
            E_h, psi_h = hole_bands['hh']
            E_h_lh, psi_h_lh = hole_bands['lh']
            E_h_so, psi_h_so = hole_bands['so']
            p_quantum = _total_quantum_hole_density(hole_bands, Efp, T, g)
            p = _blended_density(p_quantum, hole_density(Ev, Efp, g.Nv, T), region_mask)
        else:
            n = electron_density(Ec, Efn, g.Nc, T)
            p = hole_density(Ev, Efp, g.Nv, T)

        # Make Si a shallow donor (20 meV) everywhere to prevent total depletion of the n-layer
        # (Nextnano standard default for AlGaN LED base layers)
        Ed_x = np.full(g.N, 0.02)      # eV, Si donor
        Ea_x = np.full(g.N, 0.17)      # eV, Mg acceptor

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
            mu_n_avg = float(np.mean(300.0 * (1.0 - g.x_Al) + 25.0 * g.x_Al))
            mu_p_avg = float(np.mean(10.0 * (1.0 - g.x_Al) + 2.0 * g.x_Al))

            quantum_state_cd = None
            if quantum:
                quantum_state_cd = QuantumState(
                    psi_e=psi_e, E_e=E_e, m_e=g.m_e,
                    psi_h=psi_h, E_h=E_h, m_hh=g.m_hh,
                    psi_h_lh=psi_h_lh, E_h_lh=E_h_lh, m_lh=g.m_lh,
                    psi_h_so=psi_h_so, E_h_so=E_h_so, m_so=g.m_so,
                    region_mask=region_mask,
                )

            cd_result = solve_coupled_dd(
                phi, Efn, Efp, g.Ec0, g.Ev0, g.Nc, g.Nv, g.ND, g.NA, Ed_x, Ea_x,
                g.pol_rho, g.eps_r, g.dx, T, mu_n_avg, mu_p_avg,
                phi_left, phi_right, 0.0, -V_internal, 0.0, -V_internal,
                surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
                quantum_state=quantum_state_cd,
                tol=1e-3, maxiter=80,
                log_fn=(_log if verbose else None),
                cancel_check=cancel_check,
            )
            phi_candidate = cd_result.phi
            Efn = cd_result.Efn
            Efp = cd_result.Efp
            use_aa = False
            # Reuses alpha_history's slot to record the NK iteration count
            # for this step (not a damping factor -- there isn't one here).
            alpha = float(cd_result.n_iter)
            alpha_history.append(alpha)

            residual = np.max(np.abs(phi_candidate - phi))
            residual_history.append(float(residual))

            # Series resistance (lumped model): recompute J_total from the
            # fully-converged n,p and relax V_internal toward its
            # self-consistent value, same smoothing as the old code. This
            # lags by one outer iteration (V_internal used *above* was from
            # the previous pass) -- that's what this outer loop now mainly
            # exists to settle, along with the quantum Schrodinger update
            # below, since the coupled PDE solve itself is already fully
            # converged internally on every call.
            Ec_cd = g.Ec0 - phi_candidate
            Ev_cd = g.Ev0 - phi_candidate
            if quantum:
                n = _blended_density(quantum_electron_density(psi_e, E_e, Efn, g.m_e, T),
                                      electron_density(Ec_cd, Efn, g.Nc, T), region_mask)
                p_quantum = _total_quantum_hole_density(hole_bands, Efp, T, g)
                p = _blended_density(p_quantum,
                                      hole_density(Ev_cd, Efp, g.Nv, T), region_mask)
            else:
                n = electron_density(Ec_cd, Efn, g.Nc, T)
                p = hole_density(Ev_cd, Efp, g.Nv, T)

            if R_series > 0.0:
                n_boltz_cd = np.maximum(g.Nc * np.exp(np.clip((Efn - Ec_cd) / kBT_eV, -200, 200)), 1e-30)
                gamma_n_cd = np.maximum(n, 1e-30) / n_boltz_cd
                psi_n_eff_cd = -Ec_cd + kBT_eV * np.log(g.Nc / g.Nc[0]) + kBT_eV * np.log(np.maximum(gamma_n_cd, 1e-10))
                p_boltz_cd = np.maximum(g.Nv * np.exp(np.clip((Ev_cd - Efp) / kBT_eV, -200, 200)), 1e-30)
                gamma_p_cd = np.maximum(p, 1e-30) / p_boltz_cd
                psi_p_eff_cd = Ev_cd + kBT_eV * np.log(g.Nv / g.Nv[0]) + kBT_eV * np.log(np.maximum(gamma_p_cd, 1e-10))
                J_total = compute_current_density(n, p, psi_n_eff_cd, psi_p_eff_cd,
                                                   g.dx, mu_n_avg, mu_p_avg, T)
                V_internal_target = max(0.0, V_applied - abs(J_total) * R_series)
                V_internal = 0.9 * V_internal + 0.1 * V_internal_target
                phi_right = g.phi_top_eq + V_internal
            else:
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

            phi_new = solve_poisson_newton(
                phi, g.eps_r, n, p, Nd_plus, Na_minus, dNd_dphi, dNa_dphi, g.pol_rho,
                g.dx, phi_left, phi_right, T
            )

            # Convergence residual: max change the Newton step wants to make
            residual = np.max(np.abs(phi_new - phi))
            residual_history.append(float(residual))

            # e. Mixing (equilibrium only).
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
            if bias_coupled:
                _log(f"  outer iter {iteration+1:4d}  |dphi|_max = {residual:.3e} V  "
                      f"coupled-NK {'converged' if cd_result.converged else 'DID NOT CONVERGE'} "
                      f"in {cd_result.n_iter} iters (residual={cd_result.final_residual:.3e})")
            else:
                _log(f"  iter {iteration+1:4d}  |dphi|_max = {residual:.3e} V  "
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
        _log(f"  Warning: did not converge after {max_iter} iterations "
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
        p_final_quantum = _total_quantum_hole_density(hole_bands, Efp, T, g)
        p_final = _blended_density(p_final_quantum,
                                    hole_density(Ev_final, Efp, g.Nv, T), region_mask)
        Ev_hh_final, Ev_lh_final, Ev_so_final = (
            _hole_band_edges(Ev_final, g)[k] for k in ('hh', 'lh', 'so')
        )
    else:
        n_final = electron_density(Ec_final, Efn, g.Nc, T)
        p_final = hole_density(Ev_final, Efp, g.Nv, T)
        Ev_hh_final = Ev_lh_final = Ev_so_final = Ev_final

    qcse_transition_eV = None
    qcse_overlap = None
    qcse_pair = None
    qcse_in_well = False
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

    return SolverResult(
        x_nm=g.x_nm, x_Al=g.x_Al,
        Ec=Ec_final, Ev=Ev_final, Ei=Ei_final,
        Efn=Efn, Efp=Efp,
        phi=phi, E_field=E_field, F_quasi=g.F_quasi,
        n=n_final, p=p_final,
        Psp=g.Psp, Ppz=g.Ppz, P_total=g.P_total,
        eps_xx=g.eps_xx, eps_zz=g.eps_zz,
        E_e=E_e, psi_e=psi_e, E_h=E_h, psi_h=psi_h,
        Ev_hh=Ev_hh_final, Ev_lh=Ev_lh_final, Ev_so=Ev_so_final,
        E_h_lh=E_h_lh, psi_h_lh=psi_h_lh, E_h_so=E_h_so, psi_h_so=psi_h_so,
        qcse_transition_eV=qcse_transition_eV, qcse_overlap=qcse_overlap,
        qcse_pair=qcse_pair, qcse_in_well=qcse_in_well,
        qw_window_nm=qw_window_nm,
        qcse_dominant_transition_eV=qcse_dominant_transition_eV,
        qcse_dominant_overlap=qcse_dominant_overlap,
        qcse_dominant_pair=qcse_dominant_pair,
        surface_charge_markers=surface_charge_markers,
        V_applied=V_applied, V_internal=V_internal, T=T,
        converged=converged, n_iterations=n_iter,
        interface_indices=g.interface_indices,
        interface_sigmas=g.interface_sigmas,
        residual_history=residual_history, alpha_history=alpha_history,
        Ec0=g.Ec0, Ev0=g.Ev0, ND=g.ND, NA=g.NA,
    )

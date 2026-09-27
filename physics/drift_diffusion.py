from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from physics.constants import q, kB
from physics.grid_utils import node_spacings

def bernoulli(x: np.ndarray) -> np.ndarray:
    """
    Compute Bernoulli function B(x) = x / (exp(x) - 1)
    safely handling x -> 0 limits to avoid division by zero.
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    # For very small x, B(x) ~ 1 - x/2 + x^2/12
    small = np.abs(x) < 1e-4
    out[small] = 1.0 - 0.5 * x[small] + x[small]**2 / 12.0
    
    # For moderate to large x
    large = ~small
    x_large = x[large]
    out[large] = x_large / (np.exp(x_large) - 1.0)

    return out


def dbernoulli(x: np.ndarray) -> np.ndarray:
    """
    Analytic derivative B'(x) of the Bernoulli function, for the
    Scharfetter-Gummel flux stencil's contribution to an *analytic*
    Jacobian (see physics.coupled_solver's direct solver,
    [[bug-quasi-fermi-pinning]]).

    B(x) = x/(exp(x)-1); naive differentiation gives B'(x) =
    [(e-1) - x*e] / (e-1)^2 with e=exp(x), which overflows for large
    positive x (e^2 exceeds float64 range well before e itself does --
    exactly the overflow already seen from bernoulli()'s own e=exp(x)
    call, see physics.self_consistent's known warning). Reformulated to
    divide through by e first: (1 - 1/e - x) / (e*(1-1/e)^2), stable for
    x >= 0 since 1/e <= 1 there and nothing larger than e itself appears.
    For x < 0, uses the identity B(x) = B(-x) - x (already relied on
    implicitly by callers using bernoulli(dpsi) and bernoulli(-dpsi)
    together), giving B'(x) = -B'(-x) - 1, which reduces the negative
    case to the same x>=0-safe formula via y=-x>0.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-4
    # B(x) ~ 1 - x/2 + x^2/12 - x^4/720  =>  B'(x) ~ -1/2 + x/6 - x^3/180
    out[small] = -0.5 + x[small] / 6.0 - x[small]**3 / 180.0

    large = ~small
    x_large = x[large]
    pos = x_large >= 0.0

    def _dB_nonneg(y: np.ndarray) -> np.ndarray:
        # y >= 0 here, so e = exp(y) >= 1 and 1/e in (0, 1] -- no overflow.
        e = np.exp(y)
        inv_e = 1.0 / e
        return (1.0 - inv_e - y) / (e * (1.0 - inv_e)**2)

    result = np.empty_like(x_large)
    if np.any(pos):
        result[pos] = _dB_nonneg(x_large[pos])
    if np.any(~pos):
        y = -x_large[~pos]  # y > 0
        result[~pos] = -_dB_nonneg(y) - 1.0
    out[large] = result

    return out


@dataclass
class RecombinationComponents:
    """Per-grid-point recombination rates [cm^-3 s^-1], split by mechanism."""
    R_srh: np.ndarray
    R_rad: np.ndarray
    R_aug: np.ndarray

    @property
    def R_total(self) -> np.ndarray:
        return self.R_srh + self.R_rad + self.R_aug


def compute_recombination_components(n: np.ndarray, p: np.ndarray, ni: np.ndarray) -> RecombinationComponents:
    """
    SRH, radiative, and Auger recombination rates [cm^-3 s^-1], each
    computed separately (see compute_recombination for the combined total
    this replaces internally -- same coefficients, same standard III-Nitride
    parameters, factored apart so callers that need the breakdown, e.g. for
    a recombination-mechanism or optical gain plot, don't have to
    re-derive it).
    """
    tau_n = 1e-9  # 1 ns
    tau_p = 1e-9  # 1 ns
    B_rad = 1e-11 # cm^3/s
    C_aug = 1e-30 # cm^6/s

    np2 = n * p
    ni2 = ni**2

    R_srh = (np2 - ni2) / (tau_p * (n + ni) + tau_n * (p + ni))
    R_rad = B_rad * (np2 - ni2)
    R_aug = C_aug * (n + p) * (np2 - ni2)

    return RecombinationComponents(R_srh=R_srh, R_rad=R_rad, R_aug=R_aug)


def compute_recombination(n: np.ndarray, p: np.ndarray, ni: np.ndarray) -> np.ndarray:
    """
    Compute total recombination rate [cm^-3 s^-1].
    R = R_SRH + R_rad + R_Auger
    Assumes standard III-Nitride parameters.
    """
    return compute_recombination_components(n, p, ni).R_total

def solve_continuity_electron(
    n: np.ndarray,
    psi_n_eff: np.ndarray,
    R_total: np.ndarray,
    dx,
    mu_n: float,
    T: float,
    n_left: float,
    n_right: float
) -> np.ndarray:
    """
    Solve 1D steady-state electron continuity equation via Scharfetter-Gummel.
    dJ_n/dx = q * R
    J_n = q * mu_n * n * (-grad(psi_n_eff)) + q * D_n * grad(n)

    dx : scalar (uniform) or length-(N-1) array of per-edge spacings [m]
    (non-uniform grid; see physics.grid_utils). The finite-volume form
    below reduces exactly to the original uniform-grid formula (a single
    `coeff = D_n/dx^2` for every edge) when dx is a scalar.

    Returns: new electron density n [cm^-3]
    """
    N = len(n)
    kBT_eV = kB * T / q

    D_n = mu_n * kBT_eV  # cm^2 / s

    # Per-edge/per-node spacing, in cm (node_spacings is unit-agnostic).
    dx_cm = np.asarray(dx, dtype=float) * 100.0
    h1, h2, cell_width, _edges = node_spacings(dx_cm, N)
    cw = cell_width[1:-1]

    # \Delta \psi / kBT at half-integer grid points
    dpsi = (psi_n_eff[1:] - psi_n_eff[:-1]) / kBT_eV

    B_pos = bernoulli(dpsi)   # B(\Delta \psi / kT)
    B_neg = bernoulli(-dpsi)  # B(-\Delta \psi / kT)

    # coeff_r[i] = D_n/(right-edge-of-node-i * cell_width[i])
    # coeff_l[i] = D_n/(left-edge-of-node-i  * cell_width[i])
    coeff_r = D_n / (h2 * cw)
    coeff_l = D_n / (h1 * cw)

    diag_main = np.zeros(N)
    diag_lower = np.zeros(N-1)
    diag_upper = np.zeros(N-1)
    rhs = np.zeros(N)

    # Interior nodes i = 1 ... N-2
    # Eq: coeff * ( n_{i+1} B_{pos, i} - n_i B_{neg, i} - n_i B_{pos, i-1} + n_{i-1} B_{neg, i-1} ) = R_i
    # Note: J_{i+1/2} = (q D_n / dx) * [ n_{i+1} B_{pos, i} - n_i B_{neg, i} ]
    # div J = (J_{i+1/2} - J_{i-1/2}) / cell_width = q R_i

    diag_main[1:-1] = -(coeff_r * B_neg[1:] + coeff_l * B_pos[:-1])
    diag_upper[1:] = coeff_r * B_pos[1:]
    diag_lower[:-1] = coeff_l * B_neg[:-1]

    rhs[1:-1] = R_total[1:-1]
    
    # Dirichlet BCs
    diag_main[0] = 1.0
    rhs[0] = n_left
    diag_main[-1] = 1.0
    rhs[-1] = n_right
    
    A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csr')
    n_new = spsolve(A, rhs)
    return np.maximum(n_new, 1e-10)

def solve_continuity_hole(
    p: np.ndarray,
    psi_p_eff: np.ndarray,
    R_total: np.ndarray,
    dx,
    mu_p: float,
    T: float,
    p_left: float,
    p_right: float
) -> np.ndarray:
    """
    Solve 1D steady-state hole continuity equation via Scharfetter-Gummel.
    dJ_p/dx = -q * R

    dx : scalar (uniform) or length-(N-1) array of per-edge spacings [m]
    (non-uniform grid; see physics.grid_utils and solve_continuity_electron).

    Returns: new hole density p [cm^-3]
    """
    N = len(p)
    dx_cm = np.asarray(dx, dtype=float) * 100.0
    h1, h2, cell_width, _edges = node_spacings(dx_cm, N)
    cw = cell_width[1:-1]
    kBT_eV = kB * T / q

    D_p = mu_p * kBT_eV  # cm^2 / s

    # \Delta \psi / kBT at half-integer grid points
    dpsi = (psi_p_eff[1:] - psi_p_eff[:-1]) / kBT_eV

    B_pos = bernoulli(dpsi)   # B(\Delta \psi / kT)
    B_neg = bernoulli(-dpsi)  # B(-\Delta \psi / kT)

    coeff_r = D_p / (h2 * cw)
    coeff_l = D_p / (h1 * cw)

    diag_main = np.zeros(N)
    diag_lower = np.zeros(N-1)
    diag_upper = np.zeros(N-1)
    rhs = np.zeros(N)

    # Interior nodes i = 1 ... N-2
    # Correct SG for holes: J_p = -q D_p (\nabla p - p \nabla \psi_p)
    # J_{i+1/2} = (q D_p / dx) * [ p_i B_{neg, i} - p_{i+1} B_{pos, i} ]
    # div J = (J_{i+1/2} - J_{i-1/2}) / cell_width = -q R_i
    # coeff * [ p_i B_{neg, i} - p_{i+1} B_{pos, i} - p_{i-1} B_{neg, i-1} + p_i B_{pos, i-1} ] = -R_i
    # -coeff * [ p_{i+1} B_{pos, i} - p_i (B_{neg, i} + B_{pos, i-1}) + p_{i-1} B_{neg, i-1} ] = -R_i
    # coeff * [ p_{i-1} B_{neg, i-1} - p_i (B_{neg, i} + B_{pos, i-1}) + p_{i+1} B_{pos, i} ] = -R_i

    diag_main[1:-1] = -(coeff_r * B_neg[1:] + coeff_l * B_pos[:-1])
    diag_upper[1:] = coeff_r * B_pos[1:]
    diag_lower[:-1] = coeff_l * B_neg[:-1]

    rhs[1:-1] = R_total[1:-1]
    
    # Dirichlet BCs
    diag_main[0] = 1.0
    rhs[0] = p_left
    diag_main[-1] = 1.0
    rhs[-1] = p_right
    
    A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csr')
    p_new = spsolve(A, rhs)
    return np.maximum(p_new, 1e-10)


def compute_current_density(
    n: np.ndarray,
    p: np.ndarray,
    psi_n_eff: np.ndarray,
    psi_p_eff: np.ndarray,
    dx,
    mu_n: float,
    mu_p: float,
    T: float
) -> float:
    """
    Compute total current density J = J_n + J_p in A/cm^2.
    Uses Scharfetter-Gummel expressions at the center of the device.

    dx : scalar (uniform) or length-(N-1) array of per-edge spacings [m]
    (non-uniform grid; see physics.grid_utils). Uses the local spacing of
    the specific edge at the device center.
    """
    N = len(n)
    dx_cm_arr = np.asarray(dx, dtype=float) * 100.0
    edges_cm = node_spacings(dx_cm_arr, N)[3]
    kBT_eV = kB * T / q

    # Calculate J at the center index
    mid = len(n) // 2
    dx_cm = edges_cm[mid]   # local spacing of the edge between mid, mid+1

    # J_n_{i+1/2} = (q D_n / dx) * [ n_{i+1} B_{pos, i} - n_i B_{neg, i} ]
    dpsi_n = (psi_n_eff[mid+1] - psi_n_eff[mid]) / kBT_eV
    B_pos_n = dpsi_n / (np.exp(dpsi_n) - 1.0 + 1e-15) if abs(dpsi_n) > 1e-6 else 1.0 - dpsi_n/2
    B_neg_n = -dpsi_n / (np.exp(-dpsi_n) - 1.0 + 1e-15) if abs(dpsi_n) > 1e-6 else 1.0 + dpsi_n/2
    
    D_n = mu_n * kBT_eV
    Jn = (q * D_n / dx_cm) * (n[mid+1] * B_pos_n - n[mid] * B_neg_n)
    
    # J_p_{i+1/2} = (q D_p / dx) * [ p_i B_{neg, i} - p_{i+1} B_{pos, i} ]
    dpsi_p = (psi_p_eff[mid+1] - psi_p_eff[mid]) / kBT_eV
    B_pos_p = dpsi_p / (np.exp(dpsi_p) - 1.0 + 1e-15) if abs(dpsi_p) > 1e-6 else 1.0 - dpsi_p/2
    B_neg_p = -dpsi_p / (np.exp(-dpsi_p) - 1.0 + 1e-15) if abs(dpsi_p) > 1e-6 else 1.0 + dpsi_p/2
    
    D_p = mu_p * kBT_eV
    Jp = (q * D_p / dx_cm) * (p[mid] * B_neg_p - p[mid+1] * B_pos_p)
    
    return float(Jn + Jp)

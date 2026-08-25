"""
1D Poisson equation solver (finite difference method).

Solves:
    d/dx [ eps_r(x) * eps0 * dphi/dx ] = -rho(x)

Two entry points:
  solve_poisson       - linear solve for fixed rho (depletion approximation step)
  solve_poisson_newton - Newton-Raphson step that linearises the carrier response.
                         Jacobian diagonal = A_pos + q(n+p)/(eps0*kBT), making
                         the system positive-definite even with large doping.

Non-uniform grids: `dx` may be a scalar (uniform spacing) or a length-(N-1)
array of per-edge spacings (see physics.grid_utils and devices.grid_builder's
per-layer dx_nm). Every stencil below is the finite-volume generalisation of
the uniform-grid formula, dividing each node's flux balance by that node's
own finite-volume cell width; this reduces exactly to the original uniform
formula when dx is a scalar.

Units:
    phi   [V]
    eps_r [dimensionless]
    rho   [C/m^3]
    n, p  [cm^-3]   (converted internally to m^-3)
    dx    [m]  (scalar or length-(N-1) array)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from physics.constants import eps0, q as _q, kB
from physics.grid_utils import node_spacings, central_difference


def _assemble_laplacian(eps_r: np.ndarray, dx):
    """
    Build the tridiagonal Poisson matrix for d/dz[eps_r dphi/dz] = rho/eps0.

    Matrix convention (positive main diagonal):
        A_pos * phi = rho/eps0
    which is equivalent to solving:
        -d/dz[eps_r dphi/dz] = -rho/eps0
        => d/dz[eps_r dphi/dz] = rho/eps0   (will be corrected at RHS by caller)

    Off-diagonal entries at boundary rows are left zero so Dirichlet BCs
    can be imposed cleanly by setting diag_main[0]=diag_main[-1]=1.

    Returns
    -------
    eps_m      : eps_{i-1/2} for interior points  (length N-2)
    eps_p      : eps_{i+1/2} for interior points  (length N-2)
    h1, h2     : left/right spacing at interior points (length N-2)
    cell_width : finite-volume cell width at every node (length N)
    """
    N = len(eps_r)
    h1, h2, cell_width, _edges = node_spacings(dx, N)
    eps_half = 0.5 * (eps_r[:-1] + eps_r[1:])   # length N-1
    eps_m = eps_half[:-1]                         # length N-2, eps_{i-1/2}
    eps_p = eps_half[1:]                          # length N-2, eps_{i+1/2}
    return eps_m, eps_p, h1, h2, cell_width


def solve_poisson(
    phi_init: np.ndarray,
    eps_r: np.ndarray,
    rho: np.ndarray,
    dx,
    phi_left: float,
    phi_right: float,
) -> np.ndarray:
    """
    Solve 1D Poisson equation with Dirichlet BCs for a fixed charge density.

    Parameters
    ----------
    phi_init  : initial potential [V] — unused (kept for API compatibility)
    eps_r     : relative permittivity, shape (N,)
    rho       : total charge density [C/m^3], shape (N,)
    dx        : grid spacing [m] -- scalar (uniform) or length-(N-1) array
                of per-edge spacings (non-uniform; see physics.grid_utils)
    phi_left  : left (bottom) boundary potential [V]
    phi_right : right (top) boundary potential [V]

    Returns
    -------
    phi : electrostatic potential [V], shape (N,)
    """
    N = len(eps_r)
    eps_m, eps_p, h1, h2, cell_width = _assemble_laplacian(eps_r, dx)
    cw = cell_width[1:-1]

    diag_main  = np.zeros(N)
    diag_lower = np.zeros(N - 1)
    diag_upper = np.zeros(N - 1)
    rhs        = np.zeros(N)

    # Interior rows  i = 1 .. N-2 (finite-volume form; reduces to the
    # uniform-grid (eps_m+eps_p)/dx² etc. exactly when h1=h2=cw=dx).
    # Equivalent to: d/dz[eps_r dphi/dz] = -rho/eps0  (correct Poisson)
    diag_main[1:-1]  =  (eps_m / h1 + eps_p / h2) / cw
    diag_lower[:-1]  = -(eps_m / h1) / cw        # A[i, i-1] for i=1..N-2
    diag_upper[1:]   = -(eps_p / h2) / cw        # A[i, i+1] for i=1..N-2
    rhs[1:-1]        = rho[1:-1] / eps0      # RHS = +rho/eps0 to satisfy: Lap_pos*phi = rho/eps0

    # Dirichlet BCs (off-diagonal entries at boundary rows stay zero)
    diag_main[0]  = 1.0;  rhs[0]  = phi_left
    diag_main[-1] = 1.0;  rhs[-1] = phi_right

    A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csr')
    return spsolve(A, rhs)


def solve_poisson_newton(
    phi: np.ndarray,
    eps_r: np.ndarray,
    n_cm3: np.ndarray,
    p_cm3: np.ndarray,
    ND: np.ndarray,
    NA: np.ndarray,
    dNd_dphi_cm3: np.ndarray,
    dNa_dphi_cm3: np.ndarray,
    pol_rho: np.ndarray,
    dx,
    phi_left: float,
    phi_right: float,
    T: float,
) -> np.ndarray:
    """
    Solve 1D Poisson equation for \delta \phi (Newton-Raphson update).p for the nonlinear Poisson equation.

    Linearises the carrier response around the current phi:
        drho/dphi ≈ -q*(n+p)*1e6 / kBT   [C/m^3 per V]

    The linearised system is:
        [A_pos + D] * phi_new = D * phi + rho/eps0
    where D_ii = q*(n_i+p_i)*1e6 / (eps0 * kBT_V)

    This keeps the matrix positive-definite for all carrier densities and
    converges quadratically near the solution.

    Parameters
    ----------
    phi     : current electrostatic potential [V], shape (N,)
    eps_r   : relative permittivity, shape (N,)
    n_cm3   : electron density [cm^-3], shape (N,)
    p_cm3   : hole density [cm^-3], shape (N,)
    ND, NA  : donor / acceptor density [cm^-3], shape (N,)
    pol_rho : polarization charge density [C/m^3], shape (N,)
    dx      : grid spacing [m]
    phi_left, phi_right : Dirichlet BCs [V]
    T       : temperature [K]

    Returns
    -------
    phi_new : updated potential [V], shape (N,)
    """
    N = len(phi)
    kBT_V = kB * T / _q          # thermal voltage [V]

    n_m3  = np.asarray(n_cm3) * 1e6   # m^-3
    p_m3  = np.asarray(p_cm3) * 1e6
    ND_m3 = np.asarray(ND) * 1e6
    NA_m3 = np.asarray(NA) * 1e6

    rho = _q * (p_m3 - n_m3 + ND_m3 - NA_m3) + pol_rho   # C/m^3

    eps_m, eps_p, h1, h2, cell_width = _assemble_laplacian(eps_r, dx)
    cw = cell_width[1:-1]

    # Newton diagonal correction: D_i = - d(rho)/d(phi) / eps0  [1/m^2]
    # drho/dphi = q*(-n/kBT - p/kBT + dNd_dphi - dNa_dphi)
    # Note: dNd_dphi and dNa_dphi are already derivatives w.r.t phi (negative)
    # This is a local (pointwise) reaction term, not a discretised
    # derivative, so it is not divided by the cell width -- unchanged from
    # the uniform-grid formula regardless of grid spacing.
    dNd_m3 = np.asarray(dNd_dphi_cm3) * 1e6
    dNa_m3 = np.asarray(dNa_dphi_cm3) * 1e6

    D_int = _q * (n_m3[1:-1] / kBT_V + p_m3[1:-1] / kBT_V - dNd_m3[1:-1] + dNa_m3[1:-1]) / eps0

    diag_main  = np.zeros(N)
    diag_lower = np.zeros(N - 1)
    diag_upper = np.zeros(N - 1)
    rhs        = np.zeros(N)

    # Interior: [A_pos + D] * phi_new = D * phi + rho/eps0 (finite-volume
    # form; reduces to the uniform-grid (eps_m+eps_p)/dx² etc. exactly when
    # h1=h2=cw=dx)
    diag_main[1:-1]  = (eps_m / h1 + eps_p / h2) / cw + D_int
    diag_lower[:-1]  = -(eps_m / h1) / cw
    diag_upper[1:]   = -(eps_p / h2) / cw
    rhs[1:-1]        = D_int * phi[1:-1] + rho[1:-1] / eps0

    # Dirichlet BCs
    diag_main[0]  = 1.0;  rhs[0]  = phi_left
    diag_main[-1] = 1.0;  rhs[-1] = phi_right

    A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csr')
    return spsolve(A, rhs)


def electric_field(phi: np.ndarray, dx) -> np.ndarray:
    """
    Compute electric field E = -dphi/dx [V/m].

    dx : scalar (uniform) or length-(N-1) array of per-edge spacings
    (non-uniform; see physics.grid_utils).
    """
    return -central_difference(phi, dx)

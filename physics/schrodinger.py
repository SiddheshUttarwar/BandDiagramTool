"""
1D Schrödinger equation solver (finite difference, BenDaniel-Duke form).

Solves the effective-mass Schrödinger equation:

    [-hbar^2/2 * d/dx * (1/m*(x)) * d/dx + V(x)] * psi = E * psi

using the BenDaniel-Duke discretisation, which correctly handles
position-dependent effective mass at heterointerfaces.

Hamiltonian matrix elements:
    H[i,i]   = hbar^2/(2*dx^2) * (1/m_{i-1/2} + 1/m_{i+1/2}) + V[i]
    H[i,i+1] = -hbar^2/(2*dx^2 * m_{i+1/2})
    H[i,i-1] = -hbar^2/(2*dx^2 * m_{i-1/2})

where m_{i+1/2} = (m[i] + m[i+1])/2.

Non-uniform grids: `dx` may be a scalar (uniform spacing) or a length-(N-1)
array of per-edge spacings (see physics.grid_utils and devices.grid_builder's
per-layer dx_nm). The discretisation below is the finite-volume
generalisation of the BenDaniel-Duke form, which reduces exactly to it when
dx is uniform (see physics.grid_utils.node_spacings' docstring for the
finite-volume cell convention that makes this reduction exact).

Units:
    V     [J]  (convert from eV by multiplying by q)
    m_eff [kg] (m_eff in m0 units * m0)
    dx    [m]  (scalar or length-(N-1) array)
    E     [eV] (returned, converted from J)
    psi   [m^{-1/2}] (normalised so that sum(|psi|^2 * cell_width) = 1,
           which is exactly integral |psi|^2 dx = 1 for a uniform grid)
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from physics.constants import hbar, m0, q as _q
from physics.grid_utils import node_spacings


def solve_schrodinger(
    V_eV: np.ndarray,
    m_eff_m0: np.ndarray,
    dx,
    n_states: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 1D Schrödinger equation with hard-wall BCs (psi=0 at boundaries).

    Discretised (BenDaniel-Duke, finite-volume form) as the generalised
    eigenvalue problem A*psi = E*W*psi, where W = diag(cell_width) is the
    finite-volume cell-width weight and A is symmetric tridiagonal by
    construction (A[i,i+1] = A[i+1,i] = -hbar^2/(2*m_{i+1/2}*edge[i]), the
    same physical edge viewed from either side):

        A[i,i]   = -A[i,i-1] - A[i,i+1] + V[i]*cell_width[i]
        A[i,i+1] = -hbar^2 / (2 * m_{i+1/2} * edge[i])

    Converted to a standard *symmetric tridiagonal* eigenproblem via
    phi = sqrt(W)*psi (H_sym = D^-1 A D^-1, D = diag(sqrt(W))), which stays
    tridiagonal since D is diagonal -- so it's still solved directly via
    scipy.linalg.eigh_tridiagonal, with none of the cost or fragility of a
    general sparse generalised eigensolver. eigh_tridiagonal's own
    normalisation of phi (sum(phi^2)=1) is *exactly* the desired weighted
    normalisation sum(psi^2 * cell_width) = 1, since phi = sqrt(W)*psi.

    On a uniform grid this reduces algebraically to the original
    BenDaniel-Duke formulas (H[i,i] = hbar^2/(2dx^2)*(1/m_{i-1/2}+1/m_{i+1/2})
    + V[i], H[i,i+1] = -hbar^2/(2dx^2*m_{i+1/2})) and produces the same
    matrix, so results are unchanged for every existing (uniform-dx) caller.

    Parameters
    ----------
    V_eV      : potential energy profile [eV], shape (N,)
    m_eff_m0  : effective mass profile [units of m0], shape (N,)
    dx        : grid spacing [m] -- scalar (uniform) or length-(N-1) array
                of per-edge spacings (non-uniform; see physics.grid_utils)
    n_states  : number of lowest eigenstates to return

    Returns
    -------
    E_eV   : eigenvalues [eV], shape (n_states,)
    psi    : normalised eigenfunctions [m^{-1/2}], shape (n_states, N)
             satisfying sum(|psi_n|^2 * cell_width) = 1
    """
    N    = len(V_eV)
    V_J  = np.asarray(V_eV, dtype=float) * _q       # eV → J
    m_kg = np.asarray(m_eff_m0, dtype=float) * m0   # m0 → kg

    h1, h2, cell_width, edges = node_spacings(dx, N)

    # Half-point (edge) masses: m_{i+1/2} = (m[i] + m[i+1]) / 2
    m_half = 0.5 * (m_kg[:-1] + m_kg[1:])           # length N-1, per edge

    # A_offdiag[i] = A[i,i+1] = A[i+1,i] = -hbar^2/(2*m_{i+1/2}*edge[i])
    # -- symmetric by construction: the same physical edge, same mass,
    # viewed identically from either endpoint.
    a_offdiag = -hbar**2 / (2.0 * m_half * edges)   # length N-1

    # Diagonal: interior nodes reuse a_offdiag[i-1] and a_offdiag[i]
    # (the "hbar^2/(2*m*edge)" magnitude with sign flipped) plus the
    # potential integrated over the node's cell.
    a_diag = np.empty(N)
    a_diag[1:-1] = -a_offdiag[:-1] - a_offdiag[1:] + V_J[1:-1] * cell_width[1:-1]

    # Hard-wall boundaries: a fictitious ghost edge of the same length as
    # the real adjacent edge, with the boundary point's own mass (matches
    # the original m_left[0]=m[0], m_right[-1]=m[-1] self-referential
    # convention exactly in the uniform-dx case).
    a_diag[0]  = hbar**2 / (2.0 * m_kg[0]  * edges[0])  - a_offdiag[0]  + V_J[0]  * cell_width[0]
    a_diag[-1] = -a_offdiag[-1] + hbar**2 / (2.0 * m_kg[-1] * edges[-1]) + V_J[-1] * cell_width[-1]

    # Symmetrise: H_sym = D^-1 A D^-1, D = diag(sqrt(cell_width)). Stays
    # tridiagonal (diagonal congruence preserves sparsity pattern).
    inv_sqrt_w = 1.0 / np.sqrt(cell_width)
    d_main = a_diag * inv_sqrt_w * inv_sqrt_w                      # = a_diag / cell_width
    e_off  = a_offdiag * inv_sqrt_w[:-1] * inv_sqrt_w[1:]           # length N-1

    n_req = min(n_states, N - 2)   # hard-wall: at most N-2 interior states

    eigenvalues, phi = eigh_tridiagonal(
        d_main, e_off,
        eigvals_only=False,
        select='i',
        select_range=(0, n_req - 1),
    )

    # phi (from eigh_tridiagonal) already satisfies sum(phi_k^2) = 1; since
    # phi = sqrt(cell_width)*psi, that's exactly sum(psi_k^2*cell_width)=1
    # -- no separate normalisation step needed. phi has shape (N, n_req);
    # transpose to this function's (n_req, N) convention.
    psi_out = (phi * inv_sqrt_w[:, None]).T

    return eigenvalues[:n_req] / _q, np.ascontiguousarray(psi_out)


def electron_potential(Ec_eV: np.ndarray, phi_V: np.ndarray) -> np.ndarray:
    """
    Conduction band potential for electrons [eV].
    V_e(x) = Ec(x) = Ec0(x) - phi(x)   (already computed in the SP loop)
    """
    return np.asarray(Ec_eV, dtype=float)


def hole_potential(Ev_eV: np.ndarray) -> np.ndarray:
    """
    Valence band potential for holes [eV].
    For holes, the Schrödinger equation uses V_h = -Ev(x) so that the
    ground state is the highest valence band energy.
    Returns -Ev [eV] (positive values, holes sitting at the top).
    """
    return -np.asarray(Ev_eV, dtype=float)

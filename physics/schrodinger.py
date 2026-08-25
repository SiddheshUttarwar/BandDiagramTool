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

Units:
    V     [J]  (convert from eV by multiplying by q)
    m_eff [kg] (m_eff in m0 units * m0)
    dx    [m]
    E     [eV] (returned, converted from J)
    psi   [m^{-1/2}] (normalised so that sum(|psi|^2)*dx = 1)
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from physics.constants import hbar, m0, q as _q


def solve_schrodinger(
    V_eV: np.ndarray,
    m_eff_m0: np.ndarray,
    dx: float,
    n_states: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 1D Schrödinger equation with hard-wall BCs (psi=0 at boundaries).

    Uses the BenDaniel-Duke form to correctly handle position-dependent mass:
        H[i,i]   = hbar²/(2dx²) * (1/m_{i-1/2} + 1/m_{i+1/2}) + V[i]
        H[i,i±1] = -hbar²/(2dx² * m_{i±1/2})

    Solved as a symmetric tridiagonal eigenvalue problem via
    scipy.linalg.eigh_tridiagonal.

    Parameters
    ----------
    V_eV      : potential energy profile [eV], shape (N,)
    m_eff_m0  : effective mass profile [units of m0], shape (N,)
    dx        : grid spacing [m]
    n_states  : number of lowest eigenstates to return

    Returns
    -------
    E_eV   : eigenvalues [eV], shape (n_states,)
    psi    : normalised eigenfunctions [m^{-1/2}], shape (n_states, N)
             satisfying integral |psi_n|^2 dx = 1
    """
    N    = len(V_eV)
    V_J  = np.asarray(V_eV, dtype=float) * _q       # eV → J
    m_kg = np.asarray(m_eff_m0, dtype=float) * m0   # m0 → kg

    # Half-point masses: m_{i+1/2} = (m[i] + m[i+1]) / 2
    m_half = 0.5 * (m_kg[:-1] + m_kg[1:])           # length N-1

    # Off-diagonal: e[i] = H[i, i+1] = -hbar²/(2dx² * m_{i+1/2})
    e_off = -hbar**2 / (2.0 * dx**2 * m_half)       # length N-1

    # Main diagonal: d[i] = hbar²/(2dx²)*(1/m_{i-1/2} + 1/m_{i+1/2}) + V[i]
    # Edge handling: use m[0] as the left half-mass at i=0 and m[N-1] at i=N-1
    m_left  = np.concatenate([[m_kg[0]],  m_half])   # m_{i-1/2}, length N
    m_right = np.concatenate([m_half, [m_kg[-1]]])   # m_{i+1/2}, length N
    d_main  = hbar**2 / (2.0 * dx**2) * (1.0/m_left + 1.0/m_right) + V_J

    n_req = min(n_states, N - 2)   # hard-wall: at most N-2 interior states

    eigenvalues, eigenvectors = eigh_tridiagonal(
        d_main, e_off,
        eigvals_only=False,
        select='i',
        select_range=(0, n_req - 1),
    )

    # Normalise: ∫|ψ|² dx = 1  →  ψ in m^{-1/2}
    psi_out = np.empty((n_req, N))
    for k in range(n_req):
        psi  = eigenvectors[:, k]
        norm = np.sqrt(np.sum(psi**2) * dx)
        psi_out[k] = psi / norm

    return eigenvalues[:n_req] / _q, psi_out


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

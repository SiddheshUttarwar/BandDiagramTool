"""
Analytic-limit tests for physics/schrodinger.py: the BenDaniel-Duke
finite-difference solver should reproduce the infinite square well
(particle-in-a-box) spectrum and orthonormal eigenfunctions when given a
flat potential and constant effective mass.
"""

import numpy as np

from physics.schrodinger import solve_schrodinger


def test_infinite_square_well_energy_ratios():
    """
    E_n scales as n^2 for the infinite square well. Checking the *ratio*
    E_n/E_1 (rather than absolute energies against a specific box-width
    convention) is a robust, convention-independent signature: it would
    fail for e.g. a harmonic oscillator (ratio ~ n) or a free particle.
    """
    N = 400
    dx = 0.2e-9  # m
    V_eV = np.zeros(N)
    m_eff = np.ones(N)  # 1 m0 everywhere

    E, psi = solve_schrodinger(V_eV, m_eff, dx, n_states=4)

    assert np.all(np.diff(E) > 0)
    ratios = E[1:] / E[0]
    expected = np.array([4.0, 9.0, 16.0])  # (2/1)^2, (3/1)^2, (4/1)^2
    assert np.allclose(ratios, expected, rtol=0.02)


def test_infinite_square_well_orthonormal():
    N = 300
    dx = 0.25e-9
    V_eV = np.zeros(N)
    m_eff = np.ones(N)

    E, psi = solve_schrodinger(V_eV, m_eff, dx, n_states=3)

    # Normalisation: integral |psi_n|^2 dx = 1
    norms = np.sum(psi**2, axis=1) * dx
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Orthogonality between distinct eigenstates
    for i in range(psi.shape[0]):
        for j in range(i + 1, psi.shape[0]):
            overlap = np.sum(psi[i] * psi[j]) * dx
            assert abs(overlap) < 1e-6


def test_deeper_well_has_lower_ground_state_energy():
    """Sanity check on the potential term: adding a uniform offset to V_eV
    shifts every eigenvalue by exactly that offset."""
    N = 200
    dx = 0.2e-9
    m_eff = np.ones(N)

    E0, _ = solve_schrodinger(np.zeros(N), m_eff, dx, n_states=3)
    E_shifted, _ = solve_schrodinger(np.full(N, 0.5), m_eff, dx, n_states=3)

    assert np.allclose(E_shifted - E0, 0.5, atol=1e-9)

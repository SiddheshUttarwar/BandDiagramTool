"""
Tests for physics/optical.py: the QCSE (transition energy / e-h overlap)
diagnostics built on top of physics/schrodinger.py.
"""

import numpy as np

from physics.schrodinger import solve_schrodinger
from physics.optical import (
    hole_subband_energy, overlap_squared, transition_matrix,
    ground_state_transition, dominant_transition,
)


def test_hole_subband_energy_sign_flip():
    E_h = np.array([0.1, 0.3, 0.5])
    assert np.allclose(hole_subband_energy(E_h), [-0.1, -0.3, -0.5])


def test_overlap_of_identical_state_is_one():
    N = 300
    dx = 0.2e-9
    V_eV = np.zeros(N)
    m_eff = np.ones(N)
    E, psi = solve_schrodinger(V_eV, m_eff, dx, n_states=1)

    assert np.isclose(overlap_squared(psi[0], psi[0], dx), 1.0, atol=1e-6)


def test_overlap_of_orthogonal_states_is_zero():
    N = 300
    dx = 0.2e-9
    V_eV = np.zeros(N)
    m_eff = np.ones(N)
    E, psi = solve_schrodinger(V_eV, m_eff, dx, n_states=2)

    assert overlap_squared(psi[0], psi[1], dx) < 1e-6


def test_flat_well_ground_state_transition_equals_confinement_sum():
    """
    In a flat (untilted) symmetric well, the electron and hole ground
    states are identical in shape (same box, same effective mass here), so
    the e1-h1 transition energy should equal the sum of the confinement
    energies referenced to Ec/Ev, and the overlap should be ~1 (no field to
    separate them).
    """
    N = 300
    dx = 0.2e-9
    m_eff = np.ones(N)

    Ec = np.zeros(N)          # electron sits in a well at 0 eV
    Ev = np.full(N, -3.0)     # hole "well" (after -Ev flip) at +3.0 eV depth

    E_e, psi_e = solve_schrodinger(Ec, m_eff, dx, n_states=2)
    E_h, psi_h = solve_schrodinger(-Ev, m_eff, dx, n_states=2)  # hole_potential(Ev) = -Ev

    t = ground_state_transition(E_e, psi_e, E_h, psi_h, dx)

    assert t is not None
    assert np.isclose(t.energy_eV, E_e[0] + E_h[0], atol=1e-9)
    assert np.isclose(t.overlap, 1.0, atol=1e-6)


def test_transition_matrix_shape_and_dominant_selection():
    N = 300
    dx = 0.2e-9
    m_eff = np.ones(N)
    E_e, psi_e = solve_schrodinger(np.zeros(N), m_eff, dx, n_states=3)
    E_h, psi_h = solve_schrodinger(np.zeros(N), m_eff, dx, n_states=3)

    matrix = transition_matrix(E_e, psi_e, E_h, psi_h, dx, n_e=2, n_h=2)
    assert len(matrix) == 4  # 2x2 pairs

    # Same-index pairs (e1-h1, e2-h2, ...) have full overlap in a symmetric
    # well with matched masses; cross pairs are orthogonal.
    same_index = [t for t in matrix if t.ie == t.ih]
    cross = [t for t in matrix if t.ie != t.ih]
    assert all(np.isclose(t.overlap, 1.0, atol=1e-6) for t in same_index)
    assert all(t.overlap < 1e-6 for t in cross)

    dom = dominant_transition(E_e, psi_e, E_h, psi_h, dx, n_e=2, n_h=2)
    assert dom is not None
    assert dom.ie == dom.ih


def test_no_confined_states_returns_none():
    assert ground_state_transition(None, None, None, None, 0.2e-9) is None
    assert transition_matrix(np.array([]), np.zeros((0, 10)),
                              np.array([]), np.zeros((0, 10)), 0.2e-9) == []

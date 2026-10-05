"""
Tests for physics/mott.py.
"""

import numpy as np

from physics.mott import donor_mott_density_cm3, acceptor_mott_density_cm3


def test_donor_mott_density_increases_with_x_Al():
    """Effective mass grows with Al content (electron mass rises from GaN
    to AlN), shrinking the donor Bohr radius, so N_Mott should increase
    monotonically with x_Al."""
    xs = np.linspace(0.0, 1.0, 11)
    n_mott = donor_mott_density_cm3(xs)
    assert np.all(np.diff(n_mott) > 0)


def test_donor_mott_density_matches_literature_order_of_magnitude():
    """Si:AlGaN Mott density is reported (Zhu et al., APL 85, 4669 (2004);
    see this project's TestPrograms/UnderstandingAlGaNDoping/
    AlGaN_Mott_transition_correction.py) in the low-to-mid 1e19 cm^-3
    range around x~0.7-0.85. Loose bound: just the order of magnitude."""
    n_mott = donor_mott_density_cm3(0.82)
    assert 1e18 < n_mott < 1e21


def test_acceptor_mott_density_much_higher_than_donor():
    """Holes are much heavier than electrons in AlGaN, so the acceptor
    Bohr radius is much smaller and N_Mott much higher -- consistent with
    Mg:GaN/AlGaN essentially never reaching the Mott transition in
    practice at achievable doping levels. (Threshold 30x: since the
    2026-10-01 switch to anisotropic per-band DOS masses, m_h_dos near the
    x~0.5 HH/crystal-field crossover is ~1.2 m0, giving ~65x here.)"""
    x = 0.5
    assert acceptor_mott_density_cm3(x) > 30.0 * donor_mott_density_cm3(x)


def test_scalar_and_array_input_consistent():
    xs = np.array([0.1, 0.5, 0.9])
    arr = donor_mott_density_cm3(xs)
    scalars = np.array([donor_mott_density_cm3(float(x)) for x in xs])
    assert np.allclose(arr, scalars)
    assert isinstance(donor_mott_density_cm3(0.5), float)

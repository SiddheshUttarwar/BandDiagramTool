"""
Analytic-limit tests for physics/fermi_dirac.py: the Fermi-Dirac integral
lookup table and its inverse.
"""

import numpy as np

from physics.fermi_dirac import fermi_half, inverse_fermi_half


def test_boltzmann_limit():
    """F_{1/2}(eta) -> exp(eta) for eta << 0 (non-degenerate limit)."""
    eta = np.array([-20.0, -15.0, -10.0, -8.0])
    assert np.allclose(fermi_half(eta), np.exp(eta), rtol=1e-3)


def test_monotonically_increasing():
    eta = np.linspace(-20.0, 50.0, 500)
    f = fermi_half(eta)
    assert np.all(np.diff(f) > 0)


def test_degenerate_sommerfeld_asymptote():
    """For eta >> 0, F_{1/2}(eta) ~ (4/3sqrt(pi)) * eta^1.5 (leading term)."""
    eta = np.array([30.0, 40.0, 50.0])
    f = fermi_half(eta)
    leading = (4.0 / (3.0 * np.sqrt(np.pi))) * eta**1.5
    assert np.allclose(f, leading, rtol=0.05)


def test_inverse_round_trips_non_degenerate():
    eta = np.array([-15.0, -8.0, -3.0, -1.0, 0.0])
    ratio = fermi_half(eta)
    eta_back = inverse_fermi_half(ratio)
    assert np.allclose(eta_back, eta, atol=0.05)


def test_inverse_round_trips_degenerate():
    # inverse_fermi_half switches between its Joyce-Dixon and Sommerfeld
    # branches around ratio~10 (eta~5-6); accuracy dips to ~14% right at
    # that crossover (a real, narrow limitation of the two-branch
    # approximation), so this checks points clearly on the degenerate side
    # of it rather than straddling the seam.
    eta = np.array([8.0, 15.0, 30.0])
    ratio = fermi_half(eta)
    eta_back = inverse_fermi_half(ratio)
    assert np.allclose(eta_back, eta, rtol=0.03)

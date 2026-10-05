"""
Analytic-limit tests for physics/drift_diffusion.py.
"""

import numpy as np

from physics.drift_diffusion import solve_continuity_electron, bernoulli
from physics.constants import kB, q as _q


def test_bernoulli_matches_definition_away_from_zero():
    x = np.array([-5.0, -1.0, 1.0, 5.0, 10.0])
    expected = x / (np.exp(x) - 1.0)
    assert np.allclose(bernoulli(x), expected, rtol=1e-6)


def test_bernoulli_small_x_series_matches_exact_near_zero():
    x = np.array([1e-6, -1e-6, 1e-5])
    exact = x / np.expm1(x)
    assert np.allclose(bernoulli(x), exact, atol=1e-10)


def test_scharfetter_gummel_exact_for_zero_current_zero_recombination():
    """
    The Scharfetter-Gummel discretisation is exact (independent of mesh
    coarseness) for the zero-current, zero-recombination steady state:
        n(x) = C * exp(psi_eff(x) / kT)   for ANY potential profile psi_eff(x)
    This is the defining property of the scheme (Scharfetter & Gummel,
    1969) and is checked here against a deliberately nonlinear (quadratic)
    psi_eff, not just the trivial constant-field case.
    """
    N = 40
    dx = 1e-9  # m
    T = 300.0
    kBT_eV = kB * T / _q

    idx = np.arange(N, dtype=float)
    psi_eff = 4e-4 * (idx - N / 2.0) ** 2   # nonlinear potential [eV]

    C = 1e16  # arbitrary reference density [cm^-3]
    n_exact = C * np.exp(psi_eff / kBT_eV)

    R_total = np.zeros(N)
    n_new = solve_continuity_electron(
        n_exact.copy(), psi_eff, R_total, dx, mu_n=300.0, T=T,
        n_left=n_exact[0], n_right=n_exact[-1],
    )

    assert np.allclose(n_new, n_exact, rtol=1e-6)

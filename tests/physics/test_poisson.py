"""
Analytic-limit tests for physics/poisson.py.

Sign convention (verified empirically against the implementation, since the
module docstring and an inline comment in solve_poisson disagree about the
sign): for constant eps_r and rho, solve_poisson satisfies
    eps_r * phi''(x) = -rho / eps0
i.e. the standard physical Poisson equation. A constant rho therefore has
the closed-form quadratic solution used below.
"""

import numpy as np

from physics.poisson import solve_poisson, solve_poisson_newton, electric_field
from physics.constants import eps0


def _quadratic_solution(x, L, phi_left, phi_right, eps_r, rho):
    a = -rho / (2.0 * eps_r * eps0)
    return phi_left + (phi_right - phi_left) * x / L + a * x * (x - L)


def test_constant_rho_matches_quadratic_analytic_solution():
    N = 101
    dx = 1e-9
    x = np.arange(N) * dx
    L = x[-1]
    eps_r = np.full(N, 10.0)
    rho = np.full(N, 1e18)  # C/m^3

    phi_left, phi_right = 0.0, 1.0
    phi_num = solve_poisson(np.zeros(N), eps_r, rho, dx, phi_left, phi_right)
    phi_analytic = _quadratic_solution(x, L, phi_left, phi_right, eps_r[0], rho[0])

    assert np.allclose(phi_num, phi_analytic, rtol=1e-9, atol=1e-6)


def test_zero_rho_gives_linear_solution():
    """With no charge, phi must be the straight line between the boundaries."""
    N = 51
    dx = 2e-9
    x = np.arange(N) * dx
    eps_r = np.full(N, 9.5)
    rho = np.zeros(N)

    phi_left, phi_right = -0.5, 1.3
    phi_num = solve_poisson(np.zeros(N), eps_r, rho, dx, phi_left, phi_right)
    phi_linear = phi_left + (phi_right - phi_left) * x / x[-1]

    assert np.allclose(phi_num, phi_linear, atol=1e-10)


def test_electric_field_matches_finite_difference_derivative():
    N = 60
    dx = 0.5e-9
    x = np.arange(N) * dx
    phi = 0.001 * x**2 + 0.2 * x  # smooth synthetic potential [V]

    E = electric_field(phi, dx)
    E_expected = -np.gradient(phi, dx)

    assert np.allclose(E, E_expected, rtol=1e-6, atol=1e-9)


def test_constant_rho_matches_quadratic_analytic_solution_nonuniform_grid():
    """Same closed-form quadratic solution as the uniform-grid version
    above, but on a genuinely non-uniform grid (coarse then fine), passing
    dx as a length-(N-1) per-edge array. Validates physics.grid_utils'
    finite-volume generalisation directly against an analytic solution,
    independent of the uniform-grid reduction argument."""
    x1 = np.linspace(0.0, 50e-9, 40)
    x2 = np.linspace(50e-9, 100e-9, 300)[1:]
    x = np.concatenate([x1, x2])
    edges = np.diff(x)
    N = len(x)
    L = x[-1]
    eps_r = np.full(N, 10.0)
    rho = np.full(N, 1e18)  # C/m^3

    phi_left, phi_right = 0.0, 1.0
    phi_num = solve_poisson(np.zeros(N), eps_r, rho, edges, phi_left, phi_right)
    phi_analytic = _quadratic_solution(x, L, phi_left, phi_right, eps_r[0], rho[0])

    assert np.allclose(phi_num, phi_analytic, rtol=1e-4, atol=1e-6)


def test_newton_poisson_matches_linear_solve_at_fixed_carrier_density():
    """
    solve_poisson_newton linearises the carrier response around the given
    n, p. If dNd_dphi = dNa_dphi = 0 (i.e. treat n, p, ND, NA as truly fixed,
    not functions of phi), the Newton step should reduce to solving the
    linear Poisson equation for the corresponding total rho — checked by
    comparing its result to plain solve_poisson with that rho, iterated to
    a fixed point starting from the same initial guess.
    """
    from physics.constants import q as _q, kB

    N = 81
    dx = 1e-9
    T = 300.0
    eps_r = np.full(N, 10.0)
    n_cm3 = np.full(N, 1e16)
    p_cm3 = np.full(N, 1e10)
    ND = np.full(N, 1e17)
    NA = np.zeros(N)
    pol_rho = np.zeros(N)
    phi_left, phi_right = 0.0, 0.0

    rho = _q * ((p_cm3 - n_cm3 + ND - NA) * 1e6) + pol_rho

    phi_lin = solve_poisson(np.zeros(N), eps_r, rho, dx, phi_left, phi_right)

    # Newton step with zero Jacobian correction (dNd_dphi=dNa_dphi=0) should
    # land on exactly the same linear solution in one step, since D_int then
    # only contains the (fixed) n+p screening term which both formulations
    # use identically when starting from phi = phi_lin itself (a fixed point).
    phi_newton = solve_poisson_newton(
        phi_lin, eps_r, n_cm3, p_cm3, ND, NA,
        np.zeros(N), np.zeros(N), pol_rho, dx, phi_left, phi_right, T,
    )

    assert np.allclose(phi_newton, phi_lin, atol=1e-9)

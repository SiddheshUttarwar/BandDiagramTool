"""
Burton-Cabrera-Frank (BCF) steady-state adatom surface density profile
across a single atomic terrace during step-flow MOCVD/MBE growth -- the
n(x) from the very first equation of this whole growth-kinetics
discussion.

Solves the steady-state continuity equation on a terrace of width L
(0 <= x <= L), with perfect-sink boundary conditions at both step edges
(n(0) = n(L) = n_eq):

    D * d^2n/dx^2 - n/tau_d + F = 0

Analytical (hyperbolic-cosine) solution:

    n(x) = F*tau_d - (F*tau_d - n_eq) * cosh((x - L/2)/x_s) / cosh(L/(2*x_s))

where x_s = sqrt(D*tau_d) is the surface diffusion length. D and tau_d are
themselves thermally activated:

    D(T)     = D0   * exp(-E_diff / kT)      (surface diffusion coefficient)
    tau_d(T) = tau0 * exp(+E_des  / kT)       (mean residence time before desorption)

NOTE ON x vs. x_Al: this module's "x" is POSITION ALONG A TERRACE [cm],
unrelated to the Al composition "x_Al" used throughout the rest of this
project (physics/materials/algan.py etc.). They are unfortunately the same
letter in the growth-kinetics literature and in semiconductor-alloy
notation; do not confuse them.

CALIBRATION CONSTANTS -- no specific literature source pins these down for
a particular MOCVD system; the defaults below are order-of-magnitude,
illustrative placeholders (typical ranges quoted for adatom diffusion and
desorption on III-N surfaces), not tabulated values for any specific
precursor chemistry or reactor. Treat quantitative outputs with
appropriate skepticism -- this reproduces the qualitative BCF profile
shape correctly, not a specific growth run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.constants import kB, q as _q_charge


@dataclass
class BCFGrowthParams:
    F_cm2_s: float = 1.0e15        # incoming precursor flux [cm^-2 s^-1]
    D0_cm2_s: float = 1.0e-2       # diffusion prefactor [cm^2/s]
    E_diff_eV: float = 1.8         # surface diffusion activation energy [eV]
    tau0_s: float = 1.0e-13        # desorption attempt-time prefactor [s] (~1/phonon frequency)
    E_des_eV: float = 2.8          # desorption activation energy [eV]
    n_eq_cm2: float = 1.0e10       # equilibrium (flux-free) adatom density at the step edge [cm^-2]
    L_cm: float = 100.0e-7         # terrace width [cm] (default: 100 nm)
    T: float = 1323.15             # growth temperature [K] (~1050 C, typical MOCVD GaN growth)


def diffusion_coefficient_cm2_s(params: BCFGrowthParams) -> float:
    """D(T) = D0 * exp(-E_diff / kT)  [cm^2/s]."""
    kBT_eV = kB * params.T / _q_charge
    return params.D0_cm2_s * np.exp(-params.E_diff_eV / kBT_eV)


def residence_time_s(params: BCFGrowthParams) -> float:
    """tau_d(T) = tau0 * exp(+E_des / kT)  [s]."""
    kBT_eV = kB * params.T / _q_charge
    return params.tau0_s * np.exp(params.E_des_eV / kBT_eV)


def diffusion_length_cm(params: BCFGrowthParams) -> float:
    """x_s = sqrt(D * tau_d)  [cm], the mean distance an adatom travels before desorbing."""
    D = diffusion_coefficient_cm2_s(params)
    tau_d = residence_time_s(params)
    return np.sqrt(D * tau_d)


def adatom_density_profile_cm2(x_cm, params: BCFGrowthParams) -> np.ndarray:
    """
    BCF steady-state adatom density n(x) [cm^-2] across a terrace of width
    params.L_cm, x measured from one step edge (x=0) to the other (x=L).
    """
    x_cm = np.asarray(x_cm, dtype=float)
    D = diffusion_coefficient_cm2_s(params)
    tau_d = residence_time_s(params)
    x_s = np.sqrt(D * tau_d)
    L = params.L_cm
    F_tau = params.F_cm2_s * tau_d
    return F_tau - (F_tau - params.n_eq_cm2) * (
        np.cosh((x_cm - L / 2.0) / x_s) / np.cosh(L / (2.0 * x_s))
    )

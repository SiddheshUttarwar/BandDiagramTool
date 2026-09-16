"""
Mott transition: the critical dopant density above which a discrete
donor/acceptor level delocalizes into an impurity band merged with the host
band, so essentially all dopants contribute a free carrier regardless of
their nominal thermal activation energy.

Model (matches this project's own prior validation in TestPrograms/
UnderstandingAlGaNDoping/AlGaN_Mott_transition_correction.py, compared there
against measured Si:AlGaN conductivity data -- Zhu, Nakarmi, Kim, Lin & Jiang,
"Silicon doping dependence of highly conductive n-type Al0.7Ga0.3N,"
Appl. Phys. Lett. 85, 4669 (2004)):

    N_Mott = 1 / (4.2 * aB^3)          Mott & Twose, Adv. Phys. 10, 107 (1961)
    aB     = a_H * eps_r / (m*/m0)     hydrogenic effective Bohr radius

Unlike the TestPrograms script (which re-typed a standalone set of NSM-Archive
material constants for a one-off analysis), this module draws eps_r and the
DOS effective masses from physics.materials.algan -- the same composition
model used everywhere else in this tool -- so N_Mott(x_Al) stays consistent
with the Nc/Nv/Poisson physics rather than being a second, disconnected
parameter set.

DIAGNOSTIC / VISUALIZATION ONLY: this module computes where N_Mott(x_Al)
sits relative to a device's actual doping, for display on the band diagram.
It does NOT feed back into physics.self_consistent's carrier-density
physics, which still uses a single fixed donor/acceptor activation energy
regardless of doping level (see that module's docstring for the
composition-dependent estimate it *does* use, but only for the initial
Newton guess, not the converged solve).
"""

from __future__ import annotations

import numpy as np

from physics.materials.algan import get_AlGaN_params

_A_H = 0.529e-10       # hydrogen Bohr radius [m]
_MOTT_TWOSE_C = 4.2    # Mott & Twose (1961) criterion constant: N_Mott*aB^3 = 1/C


def _mott_density_cm3(m_dos_m0: float, eps_r: float) -> float:
    aB = _A_H * eps_r / m_dos_m0            # hydrogenic effective Bohr radius [m]
    N_mott_m3 = 1.0 / (_MOTT_TWOSE_C * aB ** 3)
    return N_mott_m3 * 1e-6                 # m^-3 -> cm^-3


def donor_mott_density_cm3(x_Al):
    """
    N_Mott [cm^-3] for a hydrogenic donor (e.g. Si) at composition x_Al,
    using the conduction-band DOS effective mass. Accepts a scalar or array.
    """
    scalar = np.ndim(x_Al) == 0
    x_arr = np.atleast_1d(np.asarray(x_Al, dtype=float))
    out = np.empty_like(x_arr)
    for i, xi in enumerate(x_arr):
        p = get_AlGaN_params(xi)
        out[i] = _mott_density_cm3(p.m_e_dos, p.eps_r)
    return float(out[0]) if scalar else out


def acceptor_mott_density_cm3(x_Al):
    """
    N_Mott [cm^-3] for a hydrogenic acceptor (e.g. Mg) at composition x_Al,
    using the valence-band DOS effective mass. Accepts a scalar or array.
    """
    scalar = np.ndim(x_Al) == 0
    x_arr = np.atleast_1d(np.asarray(x_Al, dtype=float))
    out = np.empty_like(x_arr)
    for i, xi in enumerate(x_arr):
        p = get_AlGaN_params(xi)
        out[i] = _mott_density_cm3(p.m_h_dos, p.eps_r)
    return float(out[0]) if scalar else out

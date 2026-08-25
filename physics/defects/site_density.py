"""
Cation (Ga+Al) / anion (N) site density in wurtzite Al_xGa_(1-x)N, used to
convert a defect formation energy into a predicted equilibrium
concentration: c = N_sites * exp(-E^f / kT).

The wurtzite conventional hexagonal cell contains 2 formula units (2
cations + 2 anions), giving:

    N_sites = 2 / V0(x),   V0(x) = (sqrt(3)/2) * a0(x)^2 * c0(x)

Sanity check (not calibrated -- a genuine independent check): at x=0
(GaN, a0=3.189 A, c0=5.186 A) this gives N_sites ~ 4.4e22 cm^-3, matching
the well-known GaN atomic density.
"""

import numpy as np

from physics.materials.algan import get_AlGaN_params_T

_ANGSTROM_TO_CM = 1.0e-8


def wurtzite_primitive_cell_volume_cm3(a0_angstrom: float, c0_angstrom: float) -> float:
    """Conventional wurtzite hexagonal-cell volume [cm^3], from lattice constants [Angstrom]."""
    a0_cm = a0_angstrom * _ANGSTROM_TO_CM
    c0_cm = c0_angstrom * _ANGSTROM_TO_CM
    return (np.sqrt(3.0) / 2.0) * a0_cm**2 * c0_cm


def cation_site_density_cm3(x: float, T: float = 300.0) -> float:
    """
    Cation (equivalently anion) site density [cm^-3] for Al_xGa_(1-x)N at
    composition x and temperature T.
    """
    params = get_AlGaN_params_T(x, T)
    V0 = wurtzite_primitive_cell_volume_cm3(params.a0, params.c0)
    return 2.0 / V0


def site_density_profile_cm3(x_Al: np.ndarray, T: float = 300.0) -> np.ndarray:
    """Vectorised cation site density [cm^-3] over an array of compositions."""
    x_Al = np.asarray(x_Al, dtype=float)
    return np.array([cation_site_density_cm3(float(x), T) for x in x_Al])


def monolayer_areal_density_cm2(x: float = 0.0, T: float = 300.0) -> float:
    """
    Areal cation-site density [cm^-2] of a single (0001) Al_xGa_(1-x)N
    monolayer: N_2D = 2 / (sqrt(3) * a0^2), the natural upper bound on a
    physically meaningful BCF adatom surface density n -- beyond ~1
    monolayer of coverage, the "adatom gas on a terrace" picture no longer
    applies (it is just the next completed atomic layer).
    """
    params = get_AlGaN_params_T(x, T)
    a0_cm = params.a0 * 1.0e-8
    return 2.0 / (np.sqrt(3.0) * a0_cm**2)

"""
Strain and polarization physics for wurtzite AlGaN on a c-plane substrate.

Implements:
  - Biaxial strain from lattice mismatch (pseudomorphic growth)
  - Piezoelectric polarization Ppz (Bernardini et al. 1997)
  - Spontaneous polarization Psp with temperature correction (pyroelectric effect)
  - Total polarization divergence → volume charge density
  - Interface sheet charges at abrupt composition steps
  - Quasi-electric field from composition gradient

Sign convention: Ga-face [0001] growth. Polarization points along -c direction
for both GaN (Psp < 0) and strained AlGaN on GaN (Ppz < 0 for tensile in-plane).
Interface charge σ = P_total(top of interface) - P_total(bottom of interface).
A positive σ creates a 2DEG (positive fixed charge attracts electrons).

References:
  Bernardini, Fiorentini & Vanderbilt, PRB 56:R10024 (1997)
  Ambacher et al., JAP 85:3222 (1999)
  Ambacher et al., JAP 87:334 (2000)
"""

import numpy as np
from physics.materials.algan import get_AlGaN_params
from physics.grid_utils import central_difference


# Pyroelectric coefficients [C/(m²·K)], PRB 93:081205 (2016)
_P_PYRO_GaN = 6.0e-5
_P_PYRO_AlN = 4.5e-5


def compute_Psp(x_Al: np.ndarray, T: float = 300.0) -> np.ndarray:
    """
    Spontaneous polarization profile [C/m²] including temperature correction.

    Parameters
    ----------
    x_Al : array of Al compositions
    T    : temperature [K]

    Returns
    -------
    Psp  : array [C/m²], negative values (Ga-face convention)
    """
    x = np.asarray(x_Al, dtype=float)
    Psp300 = x * (-0.081) + (1.0 - x) * (-0.034) - 0.021 * x * (1.0 - x)
    p_x    = x * _P_PYRO_AlN + (1.0 - x) * _P_PYRO_GaN
    return Psp300 + (T - 300.0) * p_x


def compute_strain(x_Al: np.ndarray, x_sub: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Biaxial strain for pseudomorphic AlGaN on a substrate with Al composition x_sub.

    Assumes full pseudomorphic coherence (no relaxation).
    For relaxed layers, the caller should override eps_xx with zeros.

    Returns
    -------
    eps_xx : in-plane biaxial strain (dimensionless), positive = tensile
    eps_zz : out-of-plane strain (dimensionless)
    """
    x = np.asarray(x_Al, dtype=float)

    # Substrate lattice constant (Vegard's law, Angstrom)
    a_sub = x_sub * 3.112 + (1.0 - x_sub) * 3.189

    # Layer lattice constant (unstrained)
    a_layer = x * 3.112 + (1.0 - x) * 3.189

    # Elastic constants [Pa] (linear interpolation)
    C13 = (x * 108.0 + (1.0 - x) * 103.0) * 1e9
    C33 = (x * 373.0 + (1.0 - x) * 405.0) * 1e9

    eps_xx = (a_sub - a_layer) / a_layer           # positive when substrate is larger
    eps_zz = -2.0 * (C13 / C33) * eps_xx          # out-of-plane (Poisson relaxation)

    return eps_xx, eps_zz


def compute_Ppz(x_Al: np.ndarray,
                eps_xx: np.ndarray,
                eps_zz: np.ndarray) -> np.ndarray:
    """
    Piezoelectric polarization [C/m²].

    Ppz = e33 * eps_zz + 2 * e31 * eps_xx

    where e33 and e31 are composition-dependent (linear interpolation).
    """
    x   = np.asarray(x_Al,  dtype=float)
    exx = np.asarray(eps_xx, dtype=float)
    ezz = np.asarray(eps_zz, dtype=float)

    e33 = x * 1.56  + (1.0 - x) * 0.87
    e31 = x * (-0.62) + (1.0 - x) * (-0.50)

    return e33 * ezz + 2.0 * e31 * exx


def compute_pol_charge(P_total: np.ndarray,
                       dx,
                       relaxed_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Convert total polarization profile to volume charge density [C/m³].

    Uses the divergence of polarization:
        rho_pol = -dP/dz

    For abrupt interfaces (large dP/dz), the finite-difference naturally
    captures the sheet charge as a concentrated volume charge over one cell.

    Parameters
    ----------
    P_total      : total polarization P_sp + P_pz [C/m²], length N
    dx           : grid spacing [m] -- scalar (uniform) or length-(N-1)
                   array of per-edge spacings (non-uniform; see
                   physics.grid_utils)
    relaxed_mask : boolean array; if True at index i, the interface charge
                   between i-1 and i is zeroed (relaxed layer boundary)

    Returns
    -------
    rho_pol : volume charge density [C/m³], length N
              (positive = donor-like, negative = acceptor-like)
    """
    P = np.asarray(P_total, dtype=float)
    return -central_difference(P, dx)


def compute_quasi_field(x_Al: np.ndarray,
                        dx) -> np.ndarray:
    """
    Quasi-electric field from the composition gradient [V/m].

    In a graded AlGaN layer the conduction band edge shifts with composition
    independently of any electrostatic potential:
        F_quasi = (1/q) * d(chi(x))/dz

    where chi(x) = 4.1 - 2.2*x [eV] is the electron affinity.
    This is separate from and additive to the Poisson-solved field F = -dphi/dz.

    dx : scalar (uniform) or length-(N-1) array of per-edge spacings
    (non-uniform; see physics.grid_utils).

    Returns
    -------
    F_quasi [V/m] (positive = pointing in +z direction, i.e., toward surface)
    """
    x   = np.asarray(x_Al, dtype=float)
    # Use 65:35 offset ratio consistent with algan.py
    Eg = x * 6.12 + (1.0 - x) * 3.44 - 0.7 * x * (1.0 - x)
    Eg_GaN = 3.44
    chi = 4.1 - 0.65 * (Eg - Eg_GaN)   # electron affinity [eV]

    # dchi/dz in eV/m → V/m (chi is in eV, dz in m)
    return central_difference(chi, dx)


def interface_sheet_charges(P_total: np.ndarray,
                            x_Al: np.ndarray,
                            x_threshold: float = 0.005) -> list[tuple[int, float]]:
    """
    Identify abrupt interfaces and return their sheet charges.

    An abrupt interface is a grid point where |dx_Al| > x_threshold.

    Returns
    -------
    List of (grid_index, sigma [C/m²]) for each interface.
    sigma = P_total[i] - P_total[i-1]  (positive → 2DEG-forming charge)
    """
    x   = np.asarray(x_Al,  dtype=float)
    P   = np.asarray(P_total, dtype=float)
    dx_Al = np.abs(np.diff(x))
    interfaces = []
    for i in np.where(dx_Al > x_threshold)[0]:
        sigma = P[i] - P[i + 1]   # sheet charge [C/m²] (correct: P_below - P_above)
        interfaces.append((int(i), float(sigma)))
    return interfaces

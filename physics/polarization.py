"""
Strain and polarization physics for wurtzite Al(In)GaN on a c-plane substrate.
Every function takes an optional x_In array (None = AlGaN, unchanged).

Implements:
  - Biaxial strain from lattice mismatch (pseudomorphic growth)
  - Piezoelectric polarization Ppz (Bernardini et al. 1997)
  - Spontaneous polarization Psp (Ambacher 2002) with temperature correction (pyroelectric effect)
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
from physics.materials.algan import get_AlGaN_params, get_nitride_params, spontaneous_polarization


def _x_in(x_In, like):
    """x_In array matching `like` (zeros when None -> pure AlGaN)."""
    if x_In is None:
        return np.zeros_like(np.asarray(like, dtype=float))
    return np.broadcast_to(np.asarray(x_In, dtype=float), np.shape(like)).astype(float)


def _par(xa, xi):
    """Material parameters at one point. Pure AlGaN goes through
    get_AlGaN_params so its results (and any monkeypatching of it) are
    unchanged; In-containing points use the quaternary model."""
    return get_AlGaN_params(xa) if xi == 0 else get_nitride_params(xa, xi)
from physics.grid_utils import central_difference, node_spacings


# Pyroelectric coefficients [C/(m²·K)], PRB 93:081205 (2016)
_P_PYRO_GaN = 6.0e-5
_P_PYRO_AlN = 4.5e-5

# Polarization parameter sets (the `model` argument below).
#   'ambacher2002' (default): Psp referenced to zincblende with bowing
#       (Ambacher et al. 2002) and the proper piezoelectric constants of
#       Bernardini et al. 1997 -- physics.materials.algan.
#   'dreyer2016': Dreyer, Janotti, Van de Walle & Vanderbilt, PRX 6,
#       021038 (2016). Effective Psp referenced to the layered-hexagonal
#       structure and IMPROPER e31 (= proper e31 - Psp), which is the
#       consistent pair for interface bound charges; linear interpolation
#       for alloys, as in that paper. (GaN, AlN, InN):
POLARIZATION_MODELS = ('ambacher2002', 'dreyer2016')
_DREYER_PSP = (1.312, 1.351, 1.026)        # C/m^2, H reference
_DREYER_E31 = (-1.863, -2.027, -1.630)     # C/m^2, improper
_DREYER_E33 = (1.020, 1.569, 1.238)        # C/m^2


def _check_model(model):
    if model not in POLARIZATION_MODELS:
        raise ValueError(f"Unknown polarization model {model!r}; choose from {POLARIZATION_MODELS}")


def _dreyer(vals, x, y):
    g, a, i = vals
    return x * a + (1.0 - x - y) * g + y * i


def compute_Psp(x_Al: np.ndarray, T: float = 300.0, x_In=None,
                model: str = 'ambacher2002') -> np.ndarray:
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
    y = _x_in(x_In, x)
    _check_model(model)
    if model == 'dreyer2016':
        Psp300 = _dreyer(_DREYER_PSP, x, y)
    else:
        Psp300 = (spontaneous_polarization(x) if not np.any(y) else
                  spontaneous_polarization(x, y))   # Ambacher 2002 (single source of truth)
    # pyroelectric: InN assumed equal to GaN (no data)
    p_x    = x * _P_PYRO_AlN + (1.0 - x) * _P_PYRO_GaN
    return Psp300 + (T - 300.0) * p_x


def _elastic_constants(x_Al: np.ndarray, x_In=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-point C13/C33 [Pa] from physics.materials.algan -- the single
    source of truth for composition-dependent material constants (avoids
    the elastic/piezoelectric constants here silently drifting out of sync
    with the ones get_AlGaN_params actually returns, as they previously
    had after algan.py was updated to NSM Archive values but this module's
    own hardcoded duplicates weren't)."""
    x = np.atleast_1d(np.asarray(x_Al, dtype=float))
    y = np.atleast_1d(_x_in(x_In, x))
    C13 = np.array([_par(a, b).C13 for a, b in zip(x, y)])
    C33 = np.array([_par(a, b).C33 for a, b in zip(x, y)])
    return C13, C33


def eps_zz_from_eps_xx(x_Al: np.ndarray, eps_xx: np.ndarray, x_In=None) -> np.ndarray:
    """
    Out-of-plane strain implied by a given in-plane biaxial strain, via the
    elastic (Poisson relaxation) relation eps_zz = -2*(C13/C33)*eps_xx.
    Used both for pseudomorphic strain (below) and for a layer's
    user-specified custom eps_xx (devices.layer.*.custom_strain_xx,
    consumed in devices.grid_builder) -- eps_zz is never an independent
    free parameter under the biaxial-strain approximation this tool uses,
    so a custom strain only ever sets eps_xx and this derives eps_zz.
    """
    C13, C33 = _elastic_constants(x_Al, x_In)
    return -2.0 * (C13 / C33) * np.asarray(eps_xx, dtype=float)


def compute_strain(x_Al: np.ndarray, x_sub: float, x_In=None,
                   x_In_sub: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
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

    y = _x_in(x_In, x)
    a_sub = _par(x_sub, float(x_In_sub)).a0
    a_layer = np.array([_par(a, b).a0 for a, b in zip(x, y)])   # unstrained

    eps_xx = (a_sub - a_layer) / a_layer           # positive when substrate is larger
    eps_zz = eps_zz_from_eps_xx(x, eps_xx, y)       # out-of-plane (Poisson relaxation)

    return eps_xx, eps_zz


def compute_Ppz(x_Al: np.ndarray,
                eps_xx: np.ndarray,
                eps_zz: np.ndarray,
                x_In=None,
                model: str = 'ambacher2002') -> np.ndarray:
    """
    Piezoelectric polarization [C/m²].

    Ppz = e33 * eps_zz + 2 * e31 * eps_xx

    where e33 and e31 are composition-dependent (physics.materials.algan).
    """
    x   = np.asarray(x_Al,  dtype=float)
    exx = np.asarray(eps_xx, dtype=float)
    ezz = np.asarray(eps_zz, dtype=float)

    y   = _x_in(x_In, x)
    _check_model(model)
    if model == 'dreyer2016':
        e33 = _dreyer(_DREYER_E33, np.atleast_1d(x), np.atleast_1d(y))
        e31 = _dreyer(_DREYER_E31, np.atleast_1d(x), np.atleast_1d(y))
    else:
        e33 = np.array([_par(a, b).e33 for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))])
        e31 = np.array([_par(a, b).e31 for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))])
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
    # Conservative (finite-volume) divergence: rho_i * cell_width_i =
    # -(P_{i+1} - P_{i-1})/2, which telescopes so the integrated charge is
    # exactly the polarization step on ANY grid. The non-uniform 3-point
    # central_difference is second-order for smooth P but not conservative:
    # at a step where the spacing changes h1 -> h2 it scales the sheet
    # charge by h1/h2 (e.g. 4x at a 0.2 -> 0.05 nm per-layer dx_nm change),
    # which made 2DEG densities depend on per-layer grid spacing.
    N = len(P)
    _h1, _h2, cell_width, edges = node_spacings(dx, N)
    rho = np.empty(N)
    rho[1:-1] = -0.5 * (P[2:] - P[:-2]) / cell_width[1:-1]
    rho[0] = -0.5 * (P[1] - P[0]) / cell_width[0]
    rho[-1] = -0.5 * (P[-1] - P[-2]) / cell_width[-1]
    return rho


def compute_quasi_field(x_Al: np.ndarray,
                        dx, x_In=None) -> np.ndarray:
    """
    Quasi-electric field from the composition gradient [V/m].

    In a graded AlGaN layer the conduction band edge shifts with composition
    independently of any electrostatic potential:
        F_quasi = (1/q) * d(chi(x))/dz

    where chi(x) is the composition-dependent electron affinity
    (physics.materials.algan.get_AlGaN_params). This is separate from and
    additive to the Poisson-solved field F = -dphi/dz.

    dx : scalar (uniform) or length-(N-1) array of per-edge spacings
    (non-uniform; see physics.grid_utils).

    Returns
    -------
    F_quasi [V/m] (positive = pointing in +z direction, i.e., toward surface)
    """
    x   = np.asarray(x_Al, dtype=float)
    y   = _x_in(x_In, x)
    chi = np.array([_par(a, b).chi for a, b in zip(x, y)])   # electron affinity [eV]

    # dchi/dz in eV/m → V/m (chi is in eV, dz in m)
    return central_difference(chi, dx)


def interface_sheet_charges(P_total: np.ndarray,
                            x_Al: np.ndarray,
                            x_threshold: float = 0.005,
                            x_In=None) -> list[tuple[int, float]]:
    """
    Identify abrupt interfaces and return their sheet charges.

    An abrupt interface is a grid point where the Al or In fraction jumps
    by more than x_threshold.

    Returns
    -------
    List of (grid_index, sigma [C/m²]) for each interface.
    sigma = P_total[i] - P_total[i-1]  (positive → 2DEG-forming charge)
    """
    x   = np.asarray(x_Al,  dtype=float)
    P   = np.asarray(P_total, dtype=float)
    dx_Al = np.maximum(np.abs(np.diff(x)), np.abs(np.diff(_x_in(x_In, x))))
    interfaces = []
    for i in np.where(dx_Al > x_threshold)[0]:
        sigma = P[i] - P[i + 1]   # sheet charge [C/m²] (correct: P_below - P_above)
        interfaces.append((int(i), float(sigma)))
    return interfaces

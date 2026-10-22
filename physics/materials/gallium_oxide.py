"""
Monoclinic beta-gallium oxide and its aluminium alloy,

    beta-(Al(x) Ga(1-x))2 O3          x = x_Al,  0 <= x <= ~0.7

grown on (010) beta-Ga2O3, the orientation of the (AlGa)2O3/Ga2O3
modulation-doped FETs and of most homoepitaxial power devices. The growth
axis is b; the in-plane axes are a and c.

Sources
  - Gap of beta-Ga2O3, 4.85 eV at 300 K, and electron mass 0.28 m0: the
    common device-simulation set (e.g. Wong et al., ECS J. Solid State Sci.
    Technol. 12, 2023, TCAD parameters calibrated to experiment).
    Temperature dependence: Varshni alpha = 4.45 meV/K, beta = 2000 K
    (about -1.1 meV/K at 300 K), anchored to 4.85 eV at 300 K.
  - End member Al2O3 in the beta (theta-alumina) structure, Eg = 7.24 eV,
    and the alloy gap bowing b = 1.0 eV (first-principles, arXiv:1806.03360;
    Eg(x = 0.5) = 5.80 eV, in line with Phys. Chem. Chem. Phys. 2026).
  - Band offsets: almost all of the gap difference goes to the conduction
    band; the valence-band maximum falls by only ~0.1 eV per unit x
    (DFT and XPS agree the VBM barely moves, the CBO carries the bowing).
    Delta_Ec(x = 0.2) = 0.30 eV. Measured CBOs at x ~ 0.2 range 0.2-0.5 eV.
  - Electron affinity of beta-Ga2O3: 4.00 eV (reported 3.5-4.0 eV).
  - Static permittivity along b: 10.2 (field normal to (010)); theta-Al2O3
    is taken as 8.8. Electron mobility 150 cm^2/Vs (phonon-limited ~200).
  - Holes: the O-2p valence band is almost flat and holes self-trap.
    Density-of-states mass 3.6 m0 (Nv = 1.7e20 cm^-3 at 300 K, from
    computed band masses, arXiv:2501.17373); band hole mobility taken as
    5 cm^2/Vs. p-type doping is not achieved in practice: Mg and N are deep
    acceptors (~1.05 eV and ~1.3 eV), here Ea = 1.05 eV, so an acceptor-doped
    layer acts as compensation (current-blocking), as in real devices.
  - Donors: Si / Ge / Sn on the tetrahedral Ga site, hydrogenic,
    Ed = 13.6 eV m_e / eps_r^2 = 37 meV at x = 0 (measured ~30-36 meV).
  - Lattice (Angstrom): beta-Ga2O3 a = 12.214, b = 3.037, c = 5.798;
    theta-Al2O3 a = 11.854, b = 2.904, c = 5.622 (Vegard). The two in-plane
    mismatches are nearly equal (a: -2.95 %, c: -3.04 % for Al2O3), so the
    in-plane strain is treated as isotropic with a0 = sqrt(a c).
  - Elastic constants of beta-Ga2O3 (GPa): C12 = 128.0, C22 = 343.8,
    C23 = 70.9, giving for (010) growth eps_bb = -(C12 + C23) / C22 * eps_xx.

Model simplifications, stated so the results are read accordingly:
  - beta-Ga2O3 is centrosymmetric (C2/m): no spontaneous and no
    piezoelectric polarization, so a 2DEG needs modulation doping.
  - Band edges do not move with strain (no reliable deformation potentials);
    strain enters the lattice only.
  - One isotropic electron valley and three degenerate isotropic hole bands
    whose densities of states add up to the density-of-states mass; the
    Schrodinger solve is the uncoupled (single-band) limit of k.p, so no
    interband optical matrix element (no gain spectrum).

The parameter object has the same interface as ZincblendeParams, so the
grid builder and the solvers treat it like the other crystal systems.
"""

from dataclasses import dataclass, field
import numpy as np
from physics.constants import kB, h, m0

_Q = 1.602176634e-19
_RYDBERG_EV = 13.605693

# End members. Gaps at 300 K; everything else at 300 K.
_GA2O3 = dict(Eg=4.85, eps=10.2, me=0.28, mh_dos=3.6, a=12.214, b=3.037, c=5.798,
              mu_n=150.0, mu_p=5.0, Ea=1.05)
_AL2O3 = dict(Eg=7.24, eps=8.8, me=0.40, mh_dos=4.0, a=11.854, b=2.904, c=5.622,
              mu_n=20.0, mu_p=1.0, Ea=1.50)
_BOW_EG = 1.0             # gap bowing [eV]
_DEV_DX = -0.10           # VBM shift per unit x [eV]
_CHI_GA2O3 = 4.00         # electron affinity of beta-Ga2O3 at 300 K [eV]
_VARSHNI = (4.45e-3, 2000.0)   # alpha [eV/K], beta [K]
_C12, _C22, _C23 = 128.0, 343.8, 70.9   # GPa, (010) growth
_X_MAX = 0.7              # beyond ~0.7 the beta phase is not stable

# Recombination: no reliable data. The gap is indirect by ~40 meV, so the
# radiative coefficient is taken small; Auger as for the other wide-gap
# materials; SRH lifetime 1 ns (sample-dependent, overridable per device).
_B_RAD = 1e-12
_C_AUG = 1e-31
_TAU_SRH = 1e-9


def _check_x(x_Al):
    x = float(x_Al)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x_Al must be in [0, 1], got {x_Al}")
    return x


def _lin(key, x):
    return (1.0 - x) * _GA2O3[key] + x * _AL2O3[key]


def _varshni_shift(T: float) -> float:
    """Eg(T) - Eg(300 K) [eV]."""
    a, b = _VARSHNI
    return -a * (T * T / (T + b) - 300.0 ** 2 / (300.0 + b))


def gallium_oxide_name(x_Al: float = 0.0, digits: int = 2) -> str:
    """Ga2O3, (Al0.20Ga0.80)2O3, ... (the beta phase is the only one modelled)."""
    x = _check_x(x_Al)
    tol = 10 ** (-digits - 1)
    if x <= tol:
        return 'Ga2O3'
    if x >= 1.0 - tol:
        return 'Al2O3'
    fmt = '{:.%df}' % digits
    return f'(Al{fmt.format(x)}Ga{fmt.format(1.0 - x)})2O3'


@dataclass
class GalliumOxideParams:
    """Material parameters of beta-(Al(x)Ga(1-x))2O3 at one composition and
    temperature. Field names follow ZincblendeParams."""
    x_Al: float
    Eg: float
    chi: float
    eps_r: float
    m_e_par: float
    m_e_perp: float
    m_e_dos: float
    m_hh: float          # per-band hole mass of the three degenerate bands [m0]
    m_lh: float
    m_so: float
    m_h_dos: float       # total valence density-of-states mass [m0]
    a0: float            # effective in-plane lattice constant sqrt(a c) [Angstrom]
    c0: float            # b, along the growth axis [Angstrom]
    C13: float           # (C12 + C23) / 2 [Pa]: eps_zz = -2 (C13/C33) eps_xx holds for (010)
    C33: float           # C22 [Pa]
    lattice_abc: tuple   # (a, b, c) [Angstrom]
    mu_n: float
    mu_p: float
    Ed: float
    Ea: float
    B_rad: float = _B_RAD
    C_aug: float = _C_AUG
    tau_srh: float = _TAU_SRH
    Delta_so: float = 0.0
    Delta_cr: float = 0.0
    Psp: float = 0.0
    e33: float = 0.0
    e31: float = 0.0
    valley: str = 'G'
    Eg_valleys: dict = field(default_factory=dict)

    @property
    def direct(self) -> bool:
        return True

    @property
    def m_hh_dos(self) -> float:
        return self.m_hh

    @property
    def m_lh_dos(self) -> float:
        return self.m_lh

    @property
    def m_so_dos(self) -> float:
        return self.m_so

    def Nc(self, T: float, eps_xx: float = 0.0, eps_zz: float = 0.0) -> float:
        return 2.0 * (2.0 * np.pi * self.m_e_dos * m0 * kB * T / h**2) ** 1.5 * 1e-6

    def Nv(self, T: float, eps_xx: float = 0.0, eps_zz: float = 0.0) -> float:
        return 2.0 * (2.0 * np.pi * self.m_h_dos * m0 * kB * T / h**2) ** 1.5 * 1e-6

    def ni(self, T: float) -> float:
        kT = kB * T / _Q
        return np.sqrt(self.Nc(T) * self.Nv(T)) * np.exp(-self.Eg / (2.0 * kT))

    def valence_band_energies(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        return 0.0, 0.0, 0.0, float(self.m_lh), float(self.m_so)

    def valence_offsets_from_top(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        return 0.0, 0.0, 0.0, float(self.m_lh), float(self.m_so)

    def valence_band_structure(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        return 0.0, 0.0, float(self.m_lh), float(self.m_so)

    def strain_band_shifts(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        return 0.0, 0.0


def get_gallium_oxide_params(x_Al: float = 0.0, T: float = 300.0) -> GalliumOxideParams:
    """Parameters of beta-(Al(x)Ga(1-x))2O3 at temperature T [K]. The
    temperature enters through the gap only, placed in the conduction band
    as for the other crystal systems."""
    x = _check_x(x_Al)
    Eg300 = _lin('Eg', x) - _BOW_EG * x * (1.0 - x)
    Eg = Eg300 + _varshni_shift(T)
    Ev = -(_CHI_GA2O3 + _GA2O3['Eg']) + _DEV_DX * x
    chi = -(Ev + Eg)
    eps_r = _lin('eps', x)
    me = _lin('me', x)
    mh_dos = _lin('mh_dos', x)
    m_band = mh_dos / 3.0 ** (2.0 / 3.0)      # three bands whose DOS add to mh_dos
    a, b, c = (_lin(k, x) for k in ('a', 'b', 'c'))
    mu_n = 1.0 / ((1.0 - x) / _GA2O3['mu_n'] + x / _AL2O3['mu_n'])
    mu_p = 1.0 / ((1.0 - x) / _GA2O3['mu_p'] + x / _AL2O3['mu_p'])
    return GalliumOxideParams(
        x_Al=x, Eg=Eg, chi=chi, eps_r=eps_r,
        m_e_par=me, m_e_perp=me, m_e_dos=me,
        m_hh=m_band, m_lh=m_band, m_so=m_band, m_h_dos=mh_dos,
        a0=float(np.sqrt(a * c)), c0=b,
        C13=0.5 * (_C12 + _C23) * 1e9, C33=_C22 * 1e9, lattice_abc=(a, b, c),
        mu_n=mu_n, mu_p=mu_p,
        Ed=_RYDBERG_EV * me / eps_r ** 2, Ea=_lin('Ea', x),
        Eg_valleys={'G': Eg},
    )

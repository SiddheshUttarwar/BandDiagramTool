"""
Zincblende (cubic) III-V material parameters: the arsenides and phosphides
of Al, Ga and In and all their alloys,

    Al(x) In(y) Ga(1-x-y) As(1-v) P(v)        x = x_Al, y = x_In, v = x_P

i.e. the binaries GaAs, AlAs, InAs, GaP, AlP, InP, the ternaries (AlGaAs,
InGaAs, InAlAs, InGaP, AlInP, AlGaP, GaAsP, InAsP, AlAsP) and the
quaternaries (InGaAsP, AlGaInAs, AlGaInP, ...). Growth along [001].

Sources
  - Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001) ("V&M"):
    lattice constants, band gaps at Gamma / X / L with their Varshni
    parameters, spin-orbit splitting, effective masses, Luttinger
    parameters, Kane energies, valence-band offsets, deformation
    potentials a_c, a_v, b, elastic constants, and every bowing parameter.
    GaP gamma3 = 1.25 (2.93 in the printed table is a known misprint).
  - Gap deformation potentials of the X and L valleys: Wei & Zunger, PRB 60,
    5404 (1999), as tabulated in the nextnano++ database.
  - Static dielectric constants: Landolt-Boernstein, as in the nextnano++
    database. Low-field mobilities at 300 K: NSM Archive (Ioffe Institute).
  - Electron affinity anchor: chi(GaAs) = 4.07 eV at 300 K. Every other
    band edge follows from the V&M valence-band offsets.

Differences from the wurtzite nitrides (physics.materials.algan):
  - no spontaneous polarization, and no piezoelectric polarization for
    biaxial strain on (001) (only the shear constant e14 is non-zero);
  - the conduction-band minimum can sit at Gamma, X or L. Eg, chi and the
    band-edge mass refer to the LOWEST valley; Nc() sums all three;
  - heavy and light holes are degenerate at the zone centre when unstrained
    and split under biaxial strain; the split-off band lies Delta_so below.

The parameter object has the same interface as AlGaNParams (Eg, chi, eps_r,
masses, a0, C13/C33, Nc, Nv, valence_offsets_from_top, strain_band_shifts),
so the grid builder and the solvers treat both crystal systems alike.
"""

from dataclasses import dataclass
import numpy as np
from physics.constants import kB, h, m0

_Q = 1.602176634e-19
_RYDBERG_EV = 13.605693

_III = ('Ga', 'Al', 'In')
_V = ('As', 'P')

# ---------------------------------------------------------------------------
# Binaries. Gaps at 0 K with Varshni (alpha [eV/K], beta [K]); everything
# else at 300 K. c11/c12 in GPa, energies in eV, masses in m0.
# ---------------------------------------------------------------------------
_BIN = {
    'GaAs': dict(a=5.65325, eps=12.93, c11=122.1, c12=56.6,
                 EgG=(1.519, 0.5405e-3, 204.0), EgX=(1.981, 0.460e-3, 204.0), EgL=(1.815, 0.605e-3, 204.0),
                 Dso=0.341, meG=0.067, mX=(1.3, 0.23), mL=(1.9, 0.0754),
                 lutt=(6.98, 2.06, 2.93), mso=0.172, Ep=28.8, VBO=-0.80,
                 ac=-7.17, av=-1.16, b=-2.0, agapX=1.05, agapL=-3.70,
                 mu_n=8500.0, mu_p=400.0, Ea=0.027),
    'AlAs': dict(a=5.6611, eps=10.064, c11=125.0, c12=53.4,
                 EgG=(3.099, 0.885e-3, 530.0), EgX=(2.24, 0.70e-3, 530.0), EgL=(2.46, 0.605e-3, 204.0),
                 Dso=0.28, meG=0.15, mX=(0.97, 0.22), mL=(1.32, 0.15),
                 lutt=(3.76, 0.82, 1.42), mso=0.28, Ep=21.1, VBO=-1.33,
                 ac=-5.64, av=-2.47, b=-2.3, agapX=1.01, agapL=-4.60,
                 mu_n=200.0, mu_p=100.0, Ea=0.060),
    'InAs': dict(a=6.0583, eps=15.15, c11=83.29, c12=45.26,
                 EgG=(0.417, 0.276e-3, 93.0), EgX=(1.433, 0.276e-3, 93.0), EgL=(1.133, 0.276e-3, 93.0),
                 Dso=0.39, meG=0.026, mX=(1.13, 0.16), mL=(0.64, 0.05),
                 lutt=(20.0, 8.5, 9.2), mso=0.14, Ep=21.5, VBO=-0.59,
                 ac=-5.08, av=-1.00, b=-1.8, agapX=0.92, agapL=-2.89,
                 mu_n=40000.0, mu_p=500.0, Ea=0.017),
    'GaP':  dict(a=5.4505, eps=11.1, c11=140.5, c12=62.03,
                 EgG='GaP_G', EgX=(2.35, 0.5771e-3, 372.0), EgL=(2.72, 0.5771e-3, 372.0),
                 Dso=0.08, meG=0.13, mX=(2.0, 0.253), mL=(1.2, 0.15),
                 lutt=(4.05, 0.49, 1.25), mso=0.25, Ep=31.4, VBO=-1.27,
                 ac=-8.2, av=-1.7, b=-1.6, agapX=1.27, agapL=-3.83,
                 mu_n=250.0, mu_p=150.0, Ea=0.055),
    'AlP':  dict(a=5.4672, eps=9.8, c11=133.0, c12=63.0,
                 EgG=(3.63, 0.5771e-3, 372.0), EgX=(2.52, 0.318e-3, 588.0), EgL=(3.57, 0.318e-3, 588.0),
                 Dso=0.07, meG=0.22, mX=(2.68, 0.155), mL=(1.0, 0.1),
                 lutt=(3.35, 0.71, 1.23), mso=0.30, Ep=17.7, VBO=-1.74,
                 ac=-5.7, av=-3.0, b=-1.5, agapX=1.34, agapL=-4.38,
                 mu_n=60.0, mu_p=450.0, Ea=0.070),
    'InP':  dict(a=5.8697, eps=12.61, c11=101.1, c12=56.1,
                 EgG=(1.4236, 0.363e-3, 162.0), EgX='InP_X', EgL=(2.014, 0.363e-3, 162.0),
                 Dso=0.108, meG=0.0795, mX=(0.88, 0.88), mL=(0.47, 0.47),   # V&M give DOS masses only
                 lutt=(5.08, 1.60, 2.10), mso=0.21, Ep=20.7, VBO=-0.94,
                 ac=-6.0, av=-0.6, b=-2.0, agapX=1.00, agapL=-3.00,
                 mu_n=5400.0, mu_p=200.0, Ea=0.035),
}
# Ea: typical shallow-acceptor ionization energy [eV] (Be/C/Zn in GaAs,
# Zn/Be in InP and InAs, Zn/Mg in GaP). AlAs and AlP are estimates.

# Recombination at 300 K (NSM Archive, Ioffe Institute): radiative
# coefficient B [cm^3/s] and Auger coefficient C [cm^6/s], taken equal for
# the electron- and hole-initiated processes. The indirect binaries have a
# phonon-assisted B about three orders of magnitude smaller; the AlAs and
# AlP values are estimates. Alloys interpolate log-linearly. The SRH
# lifetime depends on the sample, not the material: 1 ns unless overridden.
_RECOMB = {
    'GaAs': (7.2e-10, 1.0e-30),
    'InP':  (1.2e-10, 9.0e-31),
    'InAs': (1.1e-10, 2.2e-27),
    'GaP':  (1.0e-13, 1.0e-30),
    'AlAs': (1.0e-13, 1.0e-30),
    'AlP':  (1.0e-13, 1.0e-30),
}
_TAU_SRH = 1e-9

# ---------------------------------------------------------------------------
# Bowing, Q = linear - C u (1 - u) (V&M 2001). Keys: the mixed pair and the
# common element. Missing entries are zero.
# ---------------------------------------------------------------------------
_BOW_III = {   # (cation, cation, anion)
    ('Al', 'Ga', 'As'): dict(EgG='AlGaAs', EgX=0.055),
    ('Ga', 'In', 'As'): dict(EgG=0.477, EgX=1.4, EgL=0.33, Dso=0.15, meG=0.0091, VBO=-0.38, ac=2.61),
    ('Al', 'In', 'As'): dict(EgG=0.70, Dso=0.15, meG=0.049, VBO=-0.64, ac=-1.4),
    ('Al', 'Ga', 'P'):  dict(EgX=0.13),
    ('Ga', 'In', 'P'):  dict(EgG=0.65, EgX=0.20, EgL=1.03, meG=0.051),
    ('Al', 'In', 'P'):  dict(EgG=-0.48, EgX=0.38, Dso=-0.19, meG=0.22),
}
_BOW_V = {     # (cation,) for the As-P pair
    'Ga': dict(EgG=0.19, EgX=0.24, EgL=0.16),
    'In': dict(EgG=0.10, EgX=0.27, EgL=0.27, Dso=0.16),
    'Al': dict(EgG=0.22, EgX=0.22, EgL=0.22),
}

_CHI_GAAS = 4.07          # electron affinity of GaAs at 300 K [eV]
_VALLEY_DEGENERACY = {'G': 1, 'X': 3, 'L': 4}


def _gap(binary: str, valley: str, T: float) -> float:
    spec = _BIN[binary]['Eg' + valley]
    if spec == 'GaP_G':       # V&M: Bose-Einstein form for the direct gap of GaP
        return 2.886 + 0.1081 * (1.0 - 1.0 / np.tanh(164.0 / max(T, 1e-6)))
    if spec == 'InP_X':       # V&M: linear in T
        return 2.384 - 3.7e-4 * T
    e0, alpha, beta = spec
    return e0 - alpha * T * T / (T + beta)


def _hole_masses(lutt):
    """(m_hh, m_lh) along [001] and their density-of-states values in the
    spherical approximation gamma_bar = (2 gamma2 + 3 gamma3) / 5."""
    g1, g2, g3 = lutt
    gb = (2.0 * g2 + 3.0 * g3) / 5.0
    return 1.0 / (g1 - 2.0 * g2), 1.0 / (g1 + 2.0 * g2), 1.0 / (g1 - 2.0 * gb), 1.0 / (g1 + 2.0 * gb)


def _check_comp(x_Al, x_In, x_P):
    x_Al = float(np.clip(x_Al, 0.0, 1.0))
    x_In = float(np.clip(x_In, 0.0, 1.0))
    x_P = float(np.clip(x_P, 0.0, 1.0))
    if x_Al + x_In > 1.0 + 1e-9:
        raise ValueError(f"x_Al + x_In = {x_Al + x_In:.3f} > 1")
    return x_Al, x_In, x_P


def _weights(x_Al, x_In, x_P):
    f = {'Al': x_Al, 'In': x_In, 'Ga': max(0.0, 1.0 - x_Al - x_In)}
    g = {'P': x_P, 'As': 1.0 - x_P}
    return f, g


def _interp(binary_value, key, f, g):
    """Bilinear interpolation over the six binaries, minus the ternary
    bowing terms weighted by the fraction of the common element. Reduces to
    the V&M ternary formula on every edge of the composition space."""
    q = sum(f[c] * g[a] * binary_value(c + a) for c in _III for a in _V)
    for (c1, c2, a), bow in _BOW_III.items():
        C = bow.get(key, 0.0)
        if C == 'AlGaAs':     # composition-dependent: -0.127 + 1.310 u, u = Al share of the pair
            pair = f['Al'] + f['Ga']
            C = -0.127 + 1.310 * (f['Al'] / pair if pair > 0 else 0.0)
        if C:
            q -= g[a] * f[c1] * f[c2] * C
    for c, bow in _BOW_V.items():
        C = bow.get(key, 0.0)
        if C:
            q -= f[c] * g['As'] * g['P'] * C
    return q


def zincblende_name(x_Al: float = 0.0, x_In: float = 0.0, x_P: float = 0.0, digits: int = 2) -> str:
    """GaAs, InP, Al0.30Ga0.70As, In0.53Ga0.47As, In0.49Ga0.51P,
    In0.58Ga0.42As0.90P0.10, ..."""
    x_Al, x_In, x_P = _check_comp(x_Al, x_In, x_P)
    fmt = '{:.%df}' % digits
    tol = 10 ** (-digits - 1)

    def part(items):
        present = [(el, v) for el, v in items if v > tol]
        if len(present) == 1:
            return present[0][0]
        return ''.join(el + fmt.format(v) for el, v in present)
    return (part([('Al', x_Al), ('In', x_In), ('Ga', 1.0 - x_Al - x_In)])
            + part([('As', 1.0 - x_P), ('P', x_P)]))


@dataclass
class ZincblendeParams:
    """Material parameters of Al(x)In(y)Ga(1-x-y)As(1-v)P(v) at one
    composition and temperature. Field names follow AlGaNParams."""
    x_Al: float
    x_In: float
    x_P: float
    Eg: float            # gap from the top valence band to the lowest valley [eV]
    chi: float           # electron affinity of the lowest valley [eV]
    eps_r: float
    m_e_par: float       # confinement (growth-direction) mass of the lowest valley [m0]
    m_e_perp: float
    m_e_dos: float       # single-mass equivalent of Nc() [m0]
    m_hh: float          # [001] masses
    m_lh: float
    m_so: float
    m_h_dos: float
    Delta_so: float
    a0: float            # lattice constant [Angstrom]
    c0: float            # = a0 (cubic)
    C13: float           # = c12 [Pa], so eps_zz = -2 (C13/C33) eps_xx holds for (001)
    C33: float           # = c11 [Pa]
    m_hh_dos: float
    m_lh_dos: float
    m_so_dos: float
    Eg_valleys: dict     # {'G': , 'X': , 'L': } gaps [eV]
    m_valley_dos: dict   # per-valley density-of-states mass (one valley) [m0]
    valley: str          # lowest valley: 'G' | 'X' | 'L'
    a_c: dict            # per-valley absolute conduction-band deformation potential [eV]
    a_v: float           # valence-band hydrostatic potential, V&M sign [eV]
    b: float             # valence-band shear potential [eV]
    Ep: float            # Kane energy [eV]
    mu_n: float          # low-field mobilities [cm^2/Vs]
    mu_p: float
    Ed: float            # shallow donor ionization energy [eV]
    Ea: float            # shallow acceptor ionization energy [eV]
    B_rad: float = 0.0   # radiative recombination coefficient [cm^3/s]
    C_aug: float = 0.0   # Auger coefficient [cm^6/s]
    tau_srh: float = _TAU_SRH   # SRH lifetime of electrons and holes [s]
    Delta_cr: float = 0.0
    Psp: float = 0.0
    e33: float = 0.0
    e31: float = 0.0

    @property
    def direct(self) -> bool:
        return self.valley == 'G'

    def _valley_edges(self, eps_xx: float = 0.0, eps_zz: float = 0.0) -> dict:
        tr = 2.0 * eps_xx + eps_zz
        return {v: self.Eg_valleys[v] + self.a_c[v] * tr for v in self.Eg_valleys}

    def Nc(self, T: float, eps_xx: float = 0.0, eps_zz: float = 0.0) -> float:
        """Effective conduction-band density of states [cm^-3], referenced to
        the lowest valley: the Gamma, X (x3) and L (x4) valleys, each
        Boltzmann-weighted by its distance above the lowest one."""
        kT = kB * T / _Q
        edges = self._valley_edges(eps_xx, eps_zz)
        e_min = min(edges.values())
        m_sum = sum(_VALLEY_DEGENERACY[v] * self.m_valley_dos[v] ** 1.5 * np.exp(-(edges[v] - e_min) / kT)
                    for v in edges)
        return 2.0 * (2.0 * np.pi * m0 * kB * T / h**2) ** 1.5 * 1e-6 * m_sum

    def Nv(self, T: float, eps_xx: float = 0.0, eps_zz: float = 0.0) -> float:
        """Effective valence-band density of states [cm^-3], referenced to
        the topmost band (heavy plus light holes when unstrained)."""
        kT = kB * T / _Q
        d_hh, d_lh, d_so, _, _ = self.valence_offsets_from_top(eps_xx, eps_zz)
        m_sum = (self.m_hh_dos ** 1.5 * np.exp(-d_hh / kT)
                 + self.m_lh_dos ** 1.5 * np.exp(-d_lh / kT)
                 + self.m_so_dos ** 1.5 * np.exp(-d_so / kT))
        return 2.0 * (2.0 * np.pi * m0 * kB * T / h**2) ** 1.5 * 1e-6 * m_sum

    def ni(self, T: float) -> float:
        kT = kB * T / _Q
        return np.sqrt(self.Nc(T) * self.Nv(T)) * np.exp(-self.Eg / (2.0 * kT))

    def valence_band_energies(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        """Zone-centre HH / LH / SO energies [eV] under biaxial (001) strain,
        zero at the unstrained valence-band top (Bir-Pikus; Chuang, Physics
        of Optoelectronic Devices, ch. 4):

            P = a_v (2 eps_xx + eps_zz)         (V&M sign of a_v)
            Q = -b (eps_xx - eps_zz)
            E_HH = -P - Q
            E_LH = -P + (Q - D + sqrt(D^2 + 2 D Q + 9 Q^2)) / 2
            E_SO = -P + (Q - D - sqrt(D^2 + 2 D Q + 9 Q^2)) / 2,  D = Delta_so

        Compressive strain puts the heavy hole on top, tensile the light hole."""
        P = self.a_v * (2.0 * eps_xx + eps_zz)
        Q = -self.b * (eps_xx - eps_zz)
        D = self.Delta_so
        root = np.sqrt(D * D + 2.0 * D * Q + 9.0 * Q * Q)
        e_hh = -P - Q
        e_lh = -P + 0.5 * (Q - D + root)
        e_so = -P + 0.5 * (Q - D - root)
        return float(e_hh), float(e_lh), float(e_so), float(self.m_lh), float(self.m_so)

    def valence_offsets_from_top(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        """(dE_hh, dE_lh, dE_so, m_lh, m_so): depth of each band below the
        topmost one [eV, all >= 0]."""
        e_hh, e_lh, e_so, m_lh, m_so = self.valence_band_energies(eps_xx, eps_zz)
        top = max(e_hh, e_lh, e_so)
        return float(top - e_hh), float(top - e_lh), float(top - e_so), m_lh, m_so

    def valence_band_structure(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        e_hh, e_lh, e_so, m_lh, m_so = self.valence_band_energies(eps_xx, eps_zz)
        return float(e_hh - e_lh), float(e_hh - e_so), m_lh, m_so

    def strain_band_shifts(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        """(dEc, dEv_top) [eV]: shift of the lowest conduction valley and of
        the topmost valence band under biaxial strain, on an absolute scale.
        Only the hydrostatic shift of each valley is included (no splitting
        of the X valleys)."""
        dEc = min(self._valley_edges(eps_xx, eps_zz).values()) - min(self.Eg_valleys.values())
        return float(dEc), float(max(self.valence_band_energies(eps_xx, eps_zz)[:3]))


def get_zincblende_params(x_Al: float = 0.0, x_In: float = 0.0, x_P: float = 0.0,
                          T: float = 300.0) -> ZincblendeParams:
    """Parameters of zincblende Al(x)In(y)Ga(1-x-y)As(1-v)P(v) at
    temperature T [K]. The temperature enters through the band gaps only
    (Varshni), placed in the conduction band as for the nitrides."""
    x, y, v = _check_comp(x_Al, x_In, x_P)
    f, g = _weights(x, y, v)

    def L(key):
        return _interp(lambda b: _BIN[b][key], key, f, g)

    gaps = {vl: _interp(lambda b, vl=vl: _gap(b, vl, T), 'Eg' + vl, f, g) for vl in ('G', 'X', 'L')}
    valley = min(gaps, key=gaps.get)
    Eg = gaps[valley]

    # Band alignment from the valence-band offsets; chi anchored at GaAs, 300 K.
    Ev = -(_CHI_GAAS + _gap('GaAs', 'G', 300.0)) + (L('VBO') - _BIN['GaAs']['VBO'])
    chi = -(Ev + Eg)

    meG = L('meG')
    mXl = _interp(lambda b: _BIN[b]['mX'][0], 'mXl', f, g)
    mXt = _interp(lambda b: _BIN[b]['mX'][1], 'mXt', f, g)
    mLl = _interp(lambda b: _BIN[b]['mL'][0], 'mLl', f, g)
    mLt = _interp(lambda b: _BIN[b]['mL'][1], 'mLt', f, g)
    m_valley_dos = {'G': meG, 'X': (mXl * mXt ** 2) ** (1.0 / 3.0), 'L': (mLl * mLt ** 2) ** (1.0 / 3.0)}
    # Growth-direction mass of the lowest subband: Gamma is isotropic; of the
    # X valleys the two along z (mass m_l) lie lowest; every L valley is
    # tilted the same way to [001], 1/m_z = (1/m_l + 2/m_t) / 3.
    m_conf = {'G': meG, 'X': mXl, 'L': 3.0 * mLl * mLt / (mLt + 2.0 * mLl)}[valley]
    m_inplane = {'G': meG, 'X': mXt, 'L': m_valley_dos['L']}[valley]

    hole = {b: _hole_masses(_BIN[b]['lutt']) for b in _BIN}
    m_hh, m_lh, m_hh_dos, m_lh_dos = (_interp(lambda b, i=i: hole[b][i], 'mh', f, g) for i in range(4))
    m_so = L('mso')

    a_v = L('av')
    a_c = {'G': L('ac'), 'X': L('agapX') - a_v, 'L': L('agapL') - a_v}

    eps_r = L('eps')
    # Shallow hydrogenic donor on the lowest valley; tabulated acceptors.
    Ed = _RYDBERG_EV * m_valley_dos[valley] / eps_r ** 2
    Ea = L('Ea')
    # Mobility: reciprocal (Matthiessen-like) average of the binaries.
    mu_n = 1.0 / sum(f[c] * g[a] / _BIN[c + a]['mu_n'] for c in _III for a in _V)
    mu_p = 1.0 / sum(f[c] * g[a] / _BIN[c + a]['mu_p'] for c in _III for a in _V)

    B_rad, C_aug = (float(np.exp(sum(f[c] * g[a] * np.log(_RECOMB[c + a][i]) for c in _III for a in _V)))
                    for i in range(2))

    a0 = L('a')
    p = ZincblendeParams(
        x_Al=x, x_In=y, x_P=v, Eg=Eg, chi=chi, eps_r=eps_r,
        m_e_par=m_conf, m_e_perp=m_inplane, m_e_dos=m_valley_dos[valley],
        m_hh=m_hh, m_lh=m_lh, m_so=m_so, m_h_dos=m_hh_dos,
        Delta_so=L('Dso'), a0=a0, c0=a0, C13=L('c12') * 1e9, C33=L('c11') * 1e9,
        m_hh_dos=m_hh_dos, m_lh_dos=m_lh_dos, m_so_dos=m_so,
        Eg_valleys=gaps, m_valley_dos=m_valley_dos, valley=valley,
        a_c=a_c, a_v=a_v, b=L('b'), Ep=L('Ep'), mu_n=mu_n, mu_p=mu_p, Ed=Ed, Ea=Ea,
        B_rad=B_rad, C_aug=C_aug,
    )
    # Single-mass equivalents of the multi-valley / multi-band sums at 300 K
    # (used e.g. by physics.mott).
    n0 = 2.0 * (2.0 * np.pi * m0 * kB * 300.0 / h**2) ** 1.5 * 1e-6
    p.m_e_dos = (p.Nc(300.0) / n0) ** (2.0 / 3.0)
    p.m_h_dos = (p.Nv(300.0) / n0) ** (2.0 / 3.0)
    return p

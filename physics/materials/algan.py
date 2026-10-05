"""
AlxGa(1-x)N material parameters for all compositions x in [0, 1].

Sources (NSM Archive -- Ioffe Institute "New Semiconductor Materials,
Characteristics and Properties", https://www.ioffe.ru/SVA/NSM/Semicond/,
== Bougrov, Levinshtein, Rumyantsev & Zubrilov, "Properties of Advanced
Semiconductor Materials: GaN, AlN, InN, BN, SiC, SiGe" (Wiley, 2001) for
most values; individual NSM table entries cite their own original papers,
noted per parameter below):
  - GaN: ioffe.ru/SVA/NSM/Semicond/GaN/  (basic, ebasic, bandstr, mechanic,
    magnetic pages)
  - AlN: ioffe.ru/SVA/NSM/Semicond/AlN/  (same page set)

Values NOT taken from NSM (changed 2026-10-01 after benchmarking against
nextnano++ and the literature -- see benchmarks/REPORT.md):
  - Band alignment: valence-band offset GaN/AlN = 0.80 eV, type-I
    (Vurgaftman & Meyer, JAP 94:3675 (2003)); chi is derived from it with
    the NSM GaN anchor chi = 4.1 eV, giving chi(AlN) ~= 2.26 eV. Literal NSM
    affinities (4.1 -> 0.6 eV) gave a type-II GaN/AlN alignment with
    dEc = 3.5 eV, contradicting the type-I literature.
  - Psp: Ambacher et al., J. Phys.: Condens. Matter 14:3399 (2002),
    -0.090x - 0.034(1-x) + 0.021x(1-x) (spontaneous_polarization());
    pyroelectric T-correction: Ambacher et al., JAP 87:334 (2000).
  - e31/e33: Bernardini et al. PRB 56:R10024 (1997) as used by Ambacher 2002
    (GaN -0.49/0.73, AlN -0.60/1.46 C/m2).
  - eps_r along c: GaN 10.1 (Tsai 1999), AlN 8.57 (Fonoberov 2003).
  - AlN C13/C33 = 108/373 GPa (Vurgaftman & Meyer 2003).
  - AlN crystal-field splitting Delta_cr = -0.169 eV (Vurgaftman & Meyer).
  - Dopant ionization energies vs x: donor_ionization_energy() (Si) and
    acceptor_ionization_energy() (Mg, Nam et al. APL 83:878 (2003)).

Valence-band reference: Eg and Ev0 refer to the TOPMOST valence band (HH in
GaN, the crystal-field/SO-character band in Al-rich AlGaN);
AlGaNParams.valence_offsets_from_top() gives each band's depth below it.

Remaining NSM endpoint values (300 K): Eg GaN 3.39 / AlN 6.026 eV (bowing
1.0 eV, see _B_EG_ALGA; Varshni Eg(T) via get_nitride_params_T), m_e 0.20 / 0.40, m_hh 1.4 / 3.53, m_lh 0.3 / 3.53,
m_so 0.6 / 0.25, Delta_so 0.008 / 0.019 eV, a0 3.189 / 3.112 A,
c0 5.186 / 4.982 A, GaN C13/C33 106/398 GPa.
"""

from dataclasses import dataclass
import numpy as np
from physics.constants import kB, h, m0


@dataclass
class AlGaNParams:
    """All material parameters for AlxGa(1-x)N at a given composition."""
    x_Al: float         # Al composition [0, 1]
    Eg: float           # Bandgap [eV]
    chi: float          # Electron affinity [eV]
    eps_r: float        # Relative permittivity (static, parallel to c-axis)
    m_e_par: float      # Electron effective mass parallel to c [units of m0]
    m_e_perp: float     # Electron effective mass perpendicular to c [units of m0]
    m_e_dos: float      # Electron DOS effective mass (m_par^2 * m_perp)^(1/3) [m0]
    m_hh: float         # Heavy-hole effective mass [m0]
    m_lh: float         # Light-hole effective mass [m0]
    m_so: float         # Split-off hole effective mass [m0]
    m_h_dos: float      # Hole DOS effective mass (m_hh^1.5 + m_lh^1.5)^(2/3) [m0]
    Delta_cr: float     # Crystal-field split energy [eV]
    Delta_so: float     # Spin-orbit split energy [eV]
    a0: float           # In-plane lattice constant [Angstrom]
    c0: float           # Out-of-plane lattice constant [Angstrom]
    Psp: float          # Spontaneous polarization at 300 K [C/m²]
    e33: float          # Piezoelectric constant [C/m²]
    e31: float          # Piezoelectric constant [C/m²]
    C13: float          # Elastic constant [Pa]
    C33: float          # Elastic constant [Pa]
    # Per-band density-of-states masses (m_par * m_perp^2)^(1/3) [m0] for
    # the HH / LH-character / SO-character bands, used by Nv(). Distinct
    # from m_hh/m_lh/m_so above, which are growth-direction (kz) masses for
    # the Schrodinger solve: wurtzite valence bands are strongly
    # anisotropic (e.g. the AlN crystal-field band: 0.25 along c, 3.81
    # in-plane), so kz masses badly underestimate the DOS.
    m_hh_dos: float = 0.0
    m_lh_dos: float = 0.0
    m_so_dos: float = 0.0
    # Deformation potentials [eV] (Vurgaftman & Meyer, JAP 94:3675 (2003),
    # same values as the nextnano++ database). a_cz / a_ct: ABSOLUTE
    # conduction-band potentials along / perpendicular to c (a1 + D1,
    # a2 + D2); D1..D4: valence-band potentials of the Chuang-Chang
    # Hamiltonian (see valence_band_energies).
    a_cz: float = 0.0
    a_ct: float = 0.0
    D1: float = 0.0
    D2: float = 0.0
    D3: float = 0.0
    D4: float = 0.0
    x_In: float = 0.0   # In fraction (Al(x)In(y)Ga(1-x-y)N); 0 = AlGaN

    def Nc(self, T: float) -> float:
        """Effective conduction band DOS [cm^-3] at temperature T [K]."""
        return 2.0 * (2.0 * np.pi * self.m_e_dos * m0 * kB * T / h**2)**1.5 * 1e-6

    def Nv(self, T: float, eps_xx: float = 0.0, eps_zz: float = 0.0) -> float:
        """Effective valence band DOS [cm^-3] at temperature T [K], referenced
        to the TOPMOST valence band edge (Ev0 in this module's convention):
        sum over HH/LH/SO of each band's DOS, Boltzmann-weighted by how far
        that band sits below the top (valence_offsets_from_top). Exact in
        the non-degenerate limit, and correctly follows the high-Al
        reordering where the crystal-field (SO-character) band is on top."""
        kBT_eV = kB * T / 1.602176634e-19
        d_hh, d_lh, d_so, m_lh_eff, m_so_eff = self.valence_offsets_from_top(eps_xx, eps_zz)
        m_sum = (self.m_hh_dos ** 1.5 * np.exp(-d_hh / kBT_eV)
                 + self.m_lh_dos ** 1.5 * np.exp(-d_lh / kBT_eV)
                 + self.m_so_dos ** 1.5 * np.exp(-d_so / kBT_eV))
        return 2.0 * (2.0 * np.pi * m0 * kB * T / h**2)**1.5 * 1e-6 * m_sum

    def ni(self, T: float) -> float:
        """Intrinsic carrier concentration [cm^-3] at temperature T [K]."""
        Nc_ = self.Nc(T)
        Nv_ = self.Nv(T)
        kBT_eV = kB * T / 1.602176634e-19
        return np.sqrt(Nc_ * Nv_) * np.exp(-self.Eg / (2.0 * kBT_eV))

    def valence_band_structure(self, eps_xx: float = 0.0,
                               eps_zz: float = 0.0) -> tuple[float, float, float, float]:
        """
        Zone-center (k_t=0) heavy-hole / light-hole / split-off splitting
        and masses, as (E_hh - E_lh, E_hh - E_so, m_lh_eff, m_so_eff) --
        how far *below* the HH band edge (this module's Eg/Ev0 convention
        -- see get_AlGaN_params) the LH and SO bands sit, and which of the
        two raw NSM masses (self.m_lh, self.m_so) actually belongs to each.

        Rashba-Sheka-Pikus / Chuang & Chang wurtzite valence-band k.p
        Hamiltonian (Chuang & Chang, PRB 54:2491 (1996); Vurgaftman &
        Meyer, JAP 94:3675 (2003)), crystal-field split Delta_cr = Delta1
        and spin-orbit split Delta_so = 3*Delta2 = 3*Delta3. At k_t=kz=0
        this 6x6 (3 bands x 2 spins) block-diagonalises exactly into a 1x1
        HH (Gamma_9) block and a 2x2 Gamma_7 (LH/SO) block, each spin pair
        degenerate:

            E_hh = Delta1 + Delta2      (Gamma_9, decoupled)

            H_27 = [[ (Delta1-Delta2)/2,   sqrt(2)*Delta3        ],
                    [ sqrt(2)*Delta3,     -(Delta1-Delta2)/2     ]]

        diagonalised numerically below (np.linalg.eigh) in the zero-order
        {|LH>, |SO>} basis, rather than via a hand-derived closed form, so
        that the eigenVECTORS -- not just the eigenvalues -- are available.
        They're needed because which basis state (LH-like or SO-like)
        dominates the upper vs. lower Gamma_7 eigenvalue is composition-
        dependent, not fixed:

        - GaN limit (Delta_cr > 0 large): the upper Gamma_7 eigenvector
          is ~(1, 0) -- almost pure LH character -- so LH sits just below
          HH and SO sits well below it: the textbook GaN A > B > C
          ordering (HH > LH > SO).
        - AlN / high-Al-AlGaN limit (Delta_cr < 0 large, i.e. negative
          crystal field): the SAME upper Gamma_7 eigenvector rotates to
          ~(0, 1) -- almost pure SO (crystal-field split-off) character.
          The *split-off* band ends up topmost, not light-hole -- it can
          even rise above HH. This is the well-known AlN-rich valence-
          band reordering. Character swaps continuously through 50/50 at
          Delta_cr = Delta_so/3 (x_Al ~ 0.176 with the values below), so
          each grid point's eigenvector is classified independently by
          its own dominant component -- not by energy-proximity to HH
          (which gets the crossover composition wrong and can jump
          discontinuously between adjacent grid points) and not by a
          fixed "+sqrt is always LH" formula slot (which was tried and
          is wrong for the same reason: it locks the topmost Gamma_7
          state to "LH" even deep into AlN, contradicting the character
          it actually has).

        Because LH and SO are the two eigenvalues of the same 2x2 block,
        they can never cross each other regardless of labeling convention
        (gap = 2*sqrt(a^2+b^2) is zero only if Delta_so = 0, never true for
        physical AlGaN, where Delta_so stays in [0.008, 0.019] eV across
        the whole composition range). HH (a separate 1x1 block, different
        irrep at k_t=0) IS allowed to cross into the Gamma_7 sector -- and
        with character-based labeling, whichever of LH/SO is on top will
        smoothly cross it once, not jump.

        self.m_lh/self.m_so are the raw per-endpoint NSM masses, assigned
        to whichever eigenvalue's eigenvector is actually LH-dominant or
        SO-dominant. At the AlN endpoint the SO-dominant eigenvalue is the
        lower one there (m_so_eff = 0.25), while the LH-dominant (upper,
        topmost) eigenvalue correctly picks up m_lh_eff = m_hh = 3.53 --
        the expected wurtzite k.p kz-mass degeneracy noted in
        get_AlGaN_params().
        """
        e_hh, e_lh, e_so, m_lh_eff, m_so_eff = self.valence_band_energies(eps_xx, eps_zz)
        return float(e_hh - e_lh), float(e_hh - e_so), float(m_lh_eff), float(m_so_eff)

    def valence_band_energies(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        """Zone-center HH / LH-character / SO(CH)-character valence band
        energies [eV] (+ the LH/SO masses), on a strain-independent
        reference, from the Chuang & Chang wurtzite Hamiltonian (PRB
        54:2491 (1996)) at k = 0 with biaxial strain (eps_yy = eps_xx):

            lambda = D1 eps_zz + D2 (eps_xx + eps_yy)
            theta  = D3 eps_zz + D4 (eps_xx + eps_yy)
            E_HH   = Delta1 + Delta2 + lambda + theta          (Gamma_9)
            Gamma_7 block, basis {(X+-iY)-like, Z-like}:
              [[Delta1 - Delta2 + lambda + theta, sqrt(2) Delta3],
               [sqrt(2) Delta3,                   lambda        ]]

        with Delta1 = Delta_cr, Delta2 = Delta3 = Delta_so/3. Unstrained this
        gives E_B,C = (Delta1-Delta2)/2 +- sqrt(((Delta1-Delta2)/2)^2 +
        2 Delta3^2) -- GaN A-B ~5 meV, A-C ~43 meV; AlN crystal-field band
        ~0.16 eV ABOVE HH. (The previous version omitted the (Delta1-Delta2)/2
        centre of this block, overstating GaN's A-B splitting ~5x.)
        The XY-like eigenvector is labelled "LH", the Z-like (crystal-field)
        one "SO", each by its own dominant character.
        """
        d1 = self.Delta_cr
        d2 = d3 = self.Delta_so / 3.0
        lam = self.D1 * eps_zz + 2.0 * self.D2 * eps_xx
        th = self.D3 * eps_zz + 2.0 * self.D4 * eps_xx
        e_hh = d1 + d2 + lam + th
        b = np.sqrt(2.0) * d3
        evals, evecs = np.linalg.eigh(np.array([[d1 - d2 + lam + th, b], [b, lam]]))
        e_lh = e_so = m_lh_eff = m_so_eff = None
        for e, v in zip(evals, evecs.T):
            if v[0] ** 2 >= v[1] ** 2 and e_lh is None:
                e_lh, m_lh_eff = float(e), self.m_lh
            else:
                e_so, m_so_eff = float(e), self.m_so
        if e_so is None:   # both eigenvectors XY-dominant (cannot happen for
            e_so, m_so_eff = float(evals[0]), self.m_so   # a 2x2 Hermitian, guard only)
        return float(e_hh), float(e_lh), float(e_so), float(m_lh_eff), float(m_so_eff)

    def strain_band_shifts(self, eps_xx: float = 0.0, eps_zz: float = 0.0):
        """(dEc, dEv_top) [eV]: shift of the conduction band edge and of the
        TOPMOST valence band edge under biaxial strain, both on an absolute
        scale (dEc = a_cz eps_zz + 2 a_ct eps_xx; dEv_top = strained minus
        unstrained top valence-band energy). The gap to the top band
        changes by dEc - dEv_top."""
        dEc = self.a_cz * eps_zz + 2.0 * self.a_ct * eps_xx
        e_s = self.valence_band_energies(eps_xx, eps_zz)[:3]
        e_0 = self.valence_band_energies(0.0, 0.0)[:3]
        return float(dEc), float(max(e_s) - max(e_0))

    def valence_offsets_from_top(self, eps_xx: float = 0.0,
                                 eps_zz: float = 0.0) -> tuple[float, float, float, float, float]:
        """(dE_hh, dE_lh, dE_so, m_lh_eff, m_so_eff): how far each zone-center
        valence band sits BELOW the topmost one [eV, all >= 0]. Ev0 / Eg in
        this module refer to that topmost band (the measured optical gap is
        to the top band -- in GaN that is HH, in AlN the crystal-field,
        SO-character band; see valence_band_structure)."""
        e_hh, e_lh, e_so, m_lh_eff, m_so_eff = self.valence_band_energies(eps_xx, eps_zz)
        e = np.array([e_hh, e_lh, e_so])
        top = float(e.max())
        d_hh, d_lh, d_so = (float(top - ei) for ei in e)
        return d_hh, d_lh, d_so, m_lh_eff, m_so_eff


# ---------------------------------------------------------------------------
# Binary endpoints.  Wurtzite, 300 K.  GaN/AlN values as documented above;
# InN (added 2026-10-01 for InGaN / AlInN / AlInGaN):
#   Eg 0.69 eV (room-temperature value, Wu et al. / Davydov et al.; V&M's
#   Varshni form gives 0.756 at 300 K), valence-band offset InN above GaN
#   0.58 eV (King et al., PRB 78, 033308 (2008), XPS), eps_c 14.4
#   (NSM/Ioffe), m_e 0.07 (Wu et al. PRB 66 201403), holes HH/LH/SO
#   1.63/0.27/0.65 isotropic (NSM/Ioffe), Delta_cr 0.040, Delta_so 0.005,
#   a 3.545, c 5.703 A, C13/C33 92/224 GPa, e31/e33 -0.57/0.97 C/m^2
#   (Bernardini 1997 / V&M), Psp -0.042 C/m^2 (Ambacher 2002),
#   a1 = a2 = -3.5 eV, D1..D4 = -3.7, 4.5, 8.2, -4.1 eV (V&M 2003).
# ---------------------------------------------------------------------------
_BIN = {
    #            GaN        AlN        InN
    'Eg':      (3.39,      6.026,     0.69),
    'Ev_off':  (0.0,      -0.80,      0.58),     # top-VB offset vs GaN [eV]
    'eps_r':   (10.1,      8.57,      14.4),
    'm_e':     (0.20,      0.40,      0.07),
    'm_hh':    (1.4,       3.53,      1.63),
    'm_lh':    (0.3,       3.53,      0.27),
    'm_so':    (0.6,       0.25,      0.65),
    'Dcr':     (0.04,     -0.169,     0.040),
    'Dso':     (0.008,     0.019,     0.005),
    'a0':      (3.189,     3.112,     3.545),
    'c0':      (5.186,     4.982,     5.703),
    'e33':     (0.73,      1.46,      0.97),
    'e31':     (-0.49,    -0.60,     -0.57),
    'C13':     (106.0,     108.0,     92.0),
    'C33':     (398.0,     373.0,     224.0),
    'hh_pp':   ((1.1, 1.6),   (3.53, 10.42), (1.63, 1.63)),   # (m_par, m_perp)
    'lh_pp':   ((1.1, 0.15),  (3.53, 0.24),  (0.27, 0.27)),
    'so_pp':   ((0.15, 1.1),  (0.25, 3.81),  (0.65, 0.65)),
    'D1':      (-3.7,     -17.1,     -3.7),
    'D2':      (4.5,       7.9,       4.5),
    'D3':      (8.2,       8.8,       8.2),
    'D4':      (-4.1,     -3.9,      -4.1),
    'a1':      (-4.9,     -3.4,      -3.5),
    'a2':      (-11.3,    -11.8,     -3.5),
    'p_pyro':  (6.0e-5,    4.5e-5,    6.0e-5),   # InN: GaN value (no data)
    # Varshni parameters (Vurgaftman & Meyer 2003): Eg(T) = Eg(0) - alpha T^2 / (T + beta)
    'varshni_alpha': (0.909e-3, 1.799e-3, 0.245e-3),   # [eV/K]
    'varshni_beta':  (830.0,    1462.0,   624.0),      # [K]
}


def varshni_shift(x_Al: float, x_In: float, T: float) -> float:
    """Band-gap change Eg(T) - Eg(300 K) [eV] from the Varshni relation,
    linearly interpolated between the binaries. Zero at 300 K, so the 300 K
    gaps in _BIN (and every 300 K result) are untouched. Positive below
    300 K: +0.072 eV (GaN), +0.092 eV (AlN), +0.024 eV (InN) at T = 0."""
    x, y, z = _check_comp(x_Al, x_In)

    def dv(i):
        a, b = _BIN['varshni_alpha'][i], _BIN['varshni_beta'][i]
        return -a * T * T / (T + b) + a * 300.0 ** 2 / (300.0 + b)
    return z * dv(0) + x * dv(1) + y * dv(2)

# Band-gap bowing [eV], pairwise form Eg = linear - b_AlGa x z - b_InGa y z
# - b_AlIn x y (x = Al, y = In, z = Ga): InGaN 1.4 (V&M 2003). AlGaN 1.0:
# raised from V&M's 0.7 on 2026-10-04 after the 102-paper benchmark --
# Brunner et al. JAP 82:5090 (1997) measure 1.3, Nepal et al. APL 87:242104
# (2005) 1.0, and Al0.7GaN emits at 4.78 eV (Agrawal 2023) where b = 0.7
# gives 5.09 eV and b = 1.0 gives 5.03 eV.
# AlInN composition dependent, b(u) = 6.43 / (1 + 1.21 u^2) with u the In
# fraction of the Al-In pair (Sakalauskas et al. 2010) -> ~6.2 eV near the
# GaN lattice match, giving Eg(In0.17Al0.83N) ~4.2 eV (measured 4.0-4.5).
_B_EG_ALGA = 1.0
_B_EG_INGA = 1.4

# Valence-band (offset) bowing for the Al-In pair [eV], Ev_top += b x y.
# With a linear VBO, lattice-matched Al0.83In0.17N would sit 0.57 eV below
# GaN, but XPS measures only 0.15-0.2 (+/-0.3) eV with dEc ~0.9-1.0 eV
# (Akazawa et al., APL 96, 132104 (2010); angle-resolved XPS 0.15 eV).
# b = 2.6 eV reproduces dEv = 0.2 eV there. Zero for AlGaN and InGaN.
_B_EV_ALIN = 2.6


def _b_eg_alin(x_Al, x_In):
    u = x_In / (x_Al + x_In) if (x_Al + x_In) > 0 else 0.0
    return 6.43 / (1.0 + 1.21 * u * u)


def _check_comp(x_Al, x_In):
    x_Al = float(np.clip(x_Al, 0.0, 1.0))
    x_In = float(np.clip(x_In, 0.0, 1.0))
    if x_Al + x_In > 1.0 + 1e-9:
        raise ValueError(f"x_Al + x_In = {x_Al + x_In:.3f} > 1")
    return x_Al, x_In, max(0.0, 1.0 - x_Al - x_In)


def nitride_name(x_Al: float = 0.0, x_In: float = 0.0, digits: int = 2) -> str:
    """Human-readable alloy name: GaN, AlN, InN, Al0.30Ga0.70N,
    In0.15Ga0.85N, Al0.83In0.17N, Al0.10In0.05Ga0.85N."""
    x_Al, x_In, x_Ga = _check_comp(x_Al, x_In)
    fmt = '{:.%df}' % digits
    parts = [('Al', x_Al), ('In', x_In), ('Ga', x_Ga)]
    present = [(el, v) for el, v in parts if v > 10 ** (-digits - 1)]
    if len(present) == 1:
        return present[0][0] + 'N'
    return ''.join(el + fmt.format(v) for el, v in present) + 'N'


def spontaneous_polarization(x, x_In=0.0):
    """Spontaneous polarization Psp at 300 K [C/m^2], Ga-face sign, for
    Al(x)In(y)Ga(1-x-y)N (Ambacher et al., J. Phys.: Condens. Matter
    14:3399 (2002)): linear GaN/AlN/InN (-0.034/-0.090/-0.042) plus
    positive pairwise bowing +0.021 x z (AlGaN), +0.037 y z (InGaN),
    +0.070 x y (AlInN). For y = 0 this is exactly the AlGaN formula
    -0.090 x - 0.034 (1-x) + 0.021 x (1-x). nextnano++ uses the same
    constants. Accepts scalars or arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(x_In, dtype=float)
    z = 1.0 - x - y
    out = (-0.090 * x - 0.034 * z - 0.042 * y
           + 0.021 * x * z + 0.037 * y * z + 0.070 * x * y)
    return float(out) if out.ndim == 0 else out


# Si donor ionization energy vs Al fraction [eV]: 18 meV for x <= 0.1,
# ~50 meV at x = 0.4, <= 90 meV at x = 0.6 (Si-doped AlGaN Hall/PPC data,
# e.g. Penn State "Properties of Si donors and persistent photoconductivity
# in AlGaN"); held at 85 meV above x = 0.6, where reported values scatter
# widely (60 meV shallow Si in AlN up to >200 meV DX-like).
_SI_ED_TABLE_X = np.array([0.0, 0.1, 0.4, 0.6, 1.0])
_SI_ED_TABLE_E = np.array([0.018, 0.018, 0.050, 0.085, 0.085])


def donor_ionization_energy(x, x_In=0.0):
    """Si donor ionization energy Ed [eV] below Ec. Al dependence from the
    table above; Si stays shallow in InGaN, so x_In is accepted for API
    symmetry but does not change Ed."""
    out = np.interp(np.asarray(x, dtype=float), _SI_ED_TABLE_X, _SI_ED_TABLE_E)
    return float(out) if np.ndim(out) == 0 else out


def acceptor_ionization_energy(x, x_In=0.0):
    """Mg acceptor ionization energy Ea [eV] above the top valence band.
    Al: linear 0.17 (GaN) -> 0.51 eV (AlN), Nam et al., APL 83:878 (2003).
    In: the Mg activation energy falls as the gap shrinks; modelled as the
    Al value scaled by Eg(x, y) / Eg(x, 0) (heuristic: Ea roughly
    proportional to Eg). Identical to the AlGaN formula for x_In = 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(x_In, dtype=float)
    if np.all(y == 0):
        out = 0.17 + 0.34 * x
    else:
        xb, yb = np.broadcast_arrays(x, y)
        ratio = np.array([get_nitride_params(float(a), float(b)).Eg
                          / get_nitride_params(float(a), 0.0).Eg
                          for a, b in zip(xb.ravel(), yb.ravel())]).reshape(xb.shape)
        out = (0.17 + 0.34 * xb) * ratio
    return float(out) if np.ndim(out) == 0 else out


def get_nitride_params(x_Al: float, x_In: float = 0.0) -> AlGaNParams:
    """
    Material parameters for wurtzite Al(x)In(y)Ga(1-x-y)N, x = x_Al,
    y = x_In. Linear (Vegard) interpolation between the GaN/AlN/InN
    endpoints in _BIN, except the band gap and spontaneous polarization,
    which use pairwise bowing. For x_In = 0 every value is arithmetically
    identical to the former AlGaN-only model.
    """
    x, y, z = _check_comp(x_Al, x_In)

    def L(key):
        g, a, i = _BIN[key]
        return x * a + z * g + y * i

    # --- Band gap with pairwise bowing
    Eg = L('Eg') - _B_EG_ALGA * x * z - _B_EG_INGA * y * z - _b_eg_alin(x, y) * x * y

    # --- Band alignment from the valence-band offset (linear, no VBO
    # bowing): AlN 0.80 eV below GaN (V&M 2003), InN 0.58 eV above (King
    # 2008). chi is derived from Ec with the NSM GaN anchor chi(GaN) = 4.1.
    chi_GaN = 4.1
    Ev_top = -(chi_GaN + _BIN['Eg'][0]) + L('Ev_off') + _B_EV_ALIN * x * y
    chi = -(Ev_top + Eg)

    eps_r = L('eps_r')
    m_e_par = L('m_e')
    m_e_perp = m_e_par
    m_e_dos = (m_e_par**2 * m_e_perp)**(1.0 / 3.0)
    m_hh, m_lh, m_so = L('m_hh'), L('m_lh'), L('m_so')
    m_h_dos = (m_hh**1.5 + m_lh**1.5)**(2.0 / 3.0)
    Delta_cr, Delta_so = L('Dcr'), L('Dso')
    a0, c0 = L('a0'), L('c0')
    Psp = spontaneous_polarization(x, y)
    e33, e31 = L('e33'), L('e31')
    C13, C33 = L('C13') * 1e9, L('C33') * 1e9

    def _dos(key):
        g, a, i = (((pp[0] * pp[1] ** 2) ** (1.0 / 3.0)) for pp in _BIN[key])
        return x * a + z * g + y * i
    m_hh_dos, m_lh_dos, m_so_dos = _dos('hh_pp'), _dos('lh_pp'), _dos('so_pp')

    D1, D2, D3, D4 = L('D1'), L('D2'), L('D3'), L('D4')
    a_cz = L('a1') + D1
    a_ct = L('a2') + D2

    params = AlGaNParams(
        x_Al=x,
        Eg=Eg, chi=chi, eps_r=eps_r,
        m_e_par=m_e_par, m_e_perp=m_e_perp, m_e_dos=m_e_dos,
        m_hh=m_hh, m_lh=m_lh, m_so=m_so, m_h_dos=m_h_dos,
        Delta_cr=Delta_cr, Delta_so=Delta_so,
        a0=a0, c0=c0,
        Psp=Psp, e33=e33, e31=e31, C13=C13, C33=C33,
        m_hh_dos=m_hh_dos, m_lh_dos=m_lh_dos, m_so_dos=m_so_dos,
        a_cz=a_cz, a_ct=a_ct, D1=D1, D2=D2, D3=D3, D4=D4,
        x_In=y,
    )
    # DOS-equivalent single hole mass at 300 K matching Nv() (used e.g. by
    # physics.mott); Nv() itself evaluates the multi-band sum at any T.
    params.m_h_dos = (params.Nv(300.0) / (2.0 * (2.0 * np.pi * m0 * kB * 300.0
                                                 / h**2)**1.5 * 1e-6)) ** (2.0 / 3.0)
    return params


def get_nitride_params_T(x_Al: float, x_In: float, T: float) -> AlGaNParams:
    """get_nitride_params at temperature T: Varshni band gap (see
    varshni_shift; the whole shift is put in the conduction band, i.e. the
    valence-band offsets are kept temperature independent) and
    temperature-corrected Psp (pyroelectric, PRB 93:081205 (2016): GaN
    +6e-5, AlN +4.5e-5 C/(m^2 K); InN assumed equal to GaN). Identical to
    get_nitride_params at T = 300 K."""
    params = get_nitride_params(x_Al, x_In)
    x, y, z = _check_comp(x_Al, x_In)
    g, a, i = _BIN['p_pyro']
    params.Psp += (T - 300.0) * (x * a + z * g + y * i)
    dEg = varshni_shift(x_Al, x_In, T)
    params.Eg += dEg
    params.chi -= dEg
    return params


def get_AlGaN_params(x: float) -> AlGaNParams:
    """Al(x)Ga(1-x)N parameters (x_In = 0); see get_nitride_params."""
    return get_nitride_params(x, 0.0)


def get_AlGaN_params_T(x: float, T: float) -> AlGaNParams:
    """Al(x)Ga(1-x)N parameters with T-corrected Psp; see get_nitride_params_T."""
    return get_nitride_params_T(x, 0.0, T)


# Generic alias: the parameter set now covers the full AlInGaN system.
NitrideParams = AlGaNParams

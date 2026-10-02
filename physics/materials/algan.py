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
0.7 eV, Vurgaftman), m_e 0.20 / 0.40, m_hh 1.4 / 3.53, m_lh 0.3 / 3.53,
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


def spontaneous_polarization(x):
    """Spontaneous polarization Psp(x) at 300 K [C/m^2], Ga-face sign:
    Ambacher et al., J. Phys.: Condens. Matter 14:3399 (2002),
        Psp = -0.090 x - 0.034 (1-x) + 0.021 x (1-x).
    Note the bowing term is POSITIVE (less negative Psp) -- the previous
    -0.021 (and AlN -0.081) overstated the AlGaN/GaN sheet charge by
    ~2.8e12 cm^-2 at x = 0.3. nextnano++'s own pyroelectric charge agrees
    with this formula. Accepts scalars or arrays; the single source of
    truth for physics.polarization.compute_Psp too."""
    x = np.asarray(x, dtype=float)
    out = -0.090 * x - 0.034 * (1.0 - x) + 0.021 * x * (1.0 - x)
    return float(out) if out.ndim == 0 else out


# Si donor ionization energy vs Al fraction [eV]: 18 meV for x <= 0.1,
# ~50 meV at x = 0.4, <= 90 meV at x = 0.6 (Si-doped AlGaN Hall/PPC data,
# e.g. Penn State "Properties of Si donors and persistent photoconductivity
# in AlGaN"); held at 85 meV above x = 0.6, where reported values scatter
# widely (60 meV shallow Si in AlN up to >200 meV DX-like).
_SI_ED_TABLE_X = np.array([0.0, 0.1, 0.4, 0.6, 1.0])
_SI_ED_TABLE_E = np.array([0.018, 0.018, 0.050, 0.085, 0.085])


def donor_ionization_energy(x):
    """Si donor ionization energy Ed(x) [eV] below Ec (see table above)."""
    out = np.interp(np.asarray(x, dtype=float), _SI_ED_TABLE_X, _SI_ED_TABLE_E)
    return float(out) if np.ndim(out) == 0 else out


def acceptor_ionization_energy(x):
    """Mg acceptor ionization energy Ea(x) [eV] above the top valence band:
    linear from 0.17 eV (GaN) to 0.51 eV (AlN), Nam et al., APL 83:878
    (2003)."""
    out = 0.17 + 0.34 * np.asarray(x, dtype=float)
    return float(out) if np.ndim(out) == 0 else out


def get_AlGaN_params(x: float) -> AlGaNParams:
    """
    Return material parameters for AlxGa(1-x)N.

    x = 0.0  →  GaN
    x = 1.0  →  AlN
    All intermediate values use linear interpolation (Vegard's law) unless
    a bowing correction is known.
    """
    x = float(np.clip(x, 0.0, 1.0))

    # --- Bandgap [eV] (NSM: GaN 3.39 eV, AlN 6.026 eV), bowing b = 0.7 eV
    # (Vurgaftman 2003 -- NSM doesn't tabulate a bowing parameter)
    Eg_GaN = 3.39
    Eg_AlN = 6.026
    Eg = x * Eg_AlN + (1.0 - x) * Eg_GaN - 0.7 * x * (1.0 - x)

    # --- Band alignment: valence-band-offset rule, not electron-affinity
    # interpolation. GaN/AlN is type-I with the AlN valence band 0.80 eV
    # BELOW GaN's (Vurgaftman & Meyer, JAP 94:3675 (2003); XPS 0.7-0.8 eV),
    # linear in x (no VBO bowing). Ev_top(x) = Ev_GaN - 0.80*x, Ec = Ev + Eg,
    # and chi is derived from Ec with the NSM GaN anchor chi(GaN) = 4.1 eV.
    # (Interpolating the NSM affinities 4.1 -> 0.6 eV directly made GaN/AlN
    # type-II with dEc = 3.5 eV -- found by the 2026-10 nextnano++
    # benchmark, see benchmarks/REPORT.md.)
    chi_GaN = 4.1
    dEv_GaN_AlN = 0.80
    Ev_top = -(chi_GaN + Eg_GaN) - dEv_GaN_AlN * x
    chi = -(Ev_top + Eg)

    # --- Static relative permittivity PARALLEL to c (the 1D growth axis):
    # GaN 10.1 (Tsai et al., JAP 85:1475 (1999)), AlN 8.57 (Fonoberov &
    # Balandin, JAP 94:7178 (2003)) -- the c-axis values nextnano uses. The
    # NSM 8.9/8.5 are the in-plane/averaged values, too low for Poisson
    # along [0001].
    eps_r = x * 8.57 + (1.0 - x) * 10.1

    # --- Electron effective mass [m0]: NSM gives one isotropic value per
    # endpoint for wurtzite (no separate par/perp component), so both
    # components below are equal -- kept as two fields for interface
    # compatibility with the rest of the codebase.
    m_e_par  = x * 0.40 + (1.0 - x) * 0.20
    m_e_perp = m_e_par
    m_e_dos  = (m_e_par**2 * m_e_perp)**(1.0 / 3.0)

    # --- Hole effective masses [m0] (NSM): GaN mhh/mlh/mso = 1.4/0.3/0.6
    # (Leszczynski et al./Fan et al. 1996, isotropic); AlN mhh/mlh/mso =
    # 3.53/3.53/0.25 (Suzuki & Uenoyama 1996 kz/growth-direction masses --
    # mhh==mlh there isn't a typo, it's the expected wurtzite k.p
    # degeneracy of HH/LH along kz at k_t=0). These are the "unswapped"
    # values valence_band_structure() consumes; that method may pair m_lh
    # with the physically-SO branch (and vice versa) at high Al -- see
    # its docstring.
    m_hh    = x * 3.53 + (1.0 - x) * 1.4
    m_lh    = x * 3.53 + (1.0 - x) * 0.3
    m_so    = x * 0.25 + (1.0 - x) * 0.6
    m_h_dos = (m_hh**1.5 + m_lh**1.5)**(2.0 / 3.0)

    # --- Crystal-field / spin-orbit splitting [eV]: NSM gives GaN
    # Delta_cr=0.04, Delta_so=0.008 (Bougrov et al. 2001). NSM's AlN table
    # states crystal-field splitting is "not separately listed", so
    # Delta_cr(AlN) keeps Vurgaftman & Meyer 2003's -0.169 eV; Delta_so(AlN)
    # is NSM's own 0.019 eV (Goldberg 2001) -- see
    # AlGaNParams.valence_band_structure() for how these become the
    # HH/LH/SO band-edge splitting.
    Delta_cr = x * (-0.169) + (1.0 - x) * 0.04
    Delta_so = x * 0.019    + (1.0 - x) * 0.008

    # --- Lattice constants [Angstrom], Vegard's law (NSM)
    a0 = x * 3.112 + (1.0 - x) * 3.189
    c0 = x * 4.982 + (1.0 - x) * 5.186

    # --- Spontaneous polarization [C/m²]: not in NSM (see module
    # docstring) -- kept from Bernardini 1997, with bowing.
    Psp = spontaneous_polarization(x)

    # --- Piezoelectric constants [C/m²]: Bernardini, Fiorentini &
    # Vanderbilt PRB 56:R10024 (1997) as used by Ambacher et al., J. Phys.:
    # Condens. Matter 14:3399 (2002) -- the same source as Psp above, so
    # piezo and spontaneous terms are mutually consistent. (NSM's GaN
    # e33 = 0.65 underestimates the piezo charge.)
    e33 = x * 1.46  + (1.0 - x) * 0.73
    e31 = x * (-0.60) + (1.0 - x) * (-0.49)

    # --- Elastic constants [Pa] (NSM, converted from GPa)
    # GaN: NSM/Polian 106/398 GPa; AlN: Vurgaftman & Meyer 2003 108/373 GPa
    C13 = (x * 108.0 + (1.0 - x) * 106.0) * 1e9
    C33 = (x * 373.0 + (1.0 - x) * 398.0) * 1e9

    # Valence DOS masses from NSM/Ioffe par/perp masses (GaN: HH 1.1/1.6,
    # LH 1.1/0.15, SO 0.15/1.1; AlN: HH 3.53/10.42, LH 3.53/0.24,
    # SO 0.25/3.81 m0 -- ioffe.ru/SVA/NSM/Semicond/{GaN,AlN}/bandstr.html),
    # linear in x. "LH"/"SO" follow the character-based labeling of
    # valence_band_structure (the AlN crystal-field band is SO-character).
    def _dos(par_perp_GaN, par_perp_AlN):
        dg = (par_perp_GaN[0] * par_perp_GaN[1] ** 2) ** (1.0 / 3.0)
        da = (par_perp_AlN[0] * par_perp_AlN[1] ** 2) ** (1.0 / 3.0)
        return x * da + (1.0 - x) * dg
    m_hh_dos = _dos((1.1, 1.6), (3.53, 10.42))
    m_lh_dos = _dos((1.1, 0.15), (3.53, 0.24))
    m_so_dos = _dos((0.15, 1.1), (0.25, 3.81))

    # Deformation potentials [eV], Vurgaftman & Meyer 2003 (linear in x):
    # GaN a1=-4.9 a2=-11.3 D1..D4 = -3.7, 4.5, 8.2, -4.1;
    # AlN a1=-3.4 a2=-11.8 D1..D4 = -17.1, 7.9, 8.8, -3.9.
    def _l(g, a):
        return x * a + (1.0 - x) * g
    D1, D2, D3, D4 = _l(-3.7, -17.1), _l(4.5, 7.9), _l(8.2, 8.8), _l(-4.1, -3.9)
    a_cz = _l(-4.9, -3.4) + D1
    a_ct = _l(-11.3, -11.8) + D2

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
    )
    # DOS-equivalent single hole mass at 300 K matching Nv() (used e.g. by
    # physics.mott); Nv() itself evaluates the multi-band sum at any T.
    params.m_h_dos = (params.Nv(300.0) / (2.0 * (2.0 * np.pi * m0 * kB * 300.0
                                                 / h**2)**1.5 * 1e-6)) ** (2.0 / 3.0)
    return params


def get_AlGaN_params_T(x: float, T: float) -> AlGaNParams:
    """
    Return AlGaN parameters with temperature-corrected Psp.
    Pyroelectric correction from PRB 93:081205 (2016):
      p_GaN ≈ +6e-5 C/(m²·K),  p_AlN ≈ +4.5e-5 C/(m²·K)
    Sign convention: Psp becomes less negative as T increases (Ga-face).
    """
    params = get_AlGaN_params(x)
    p_GaN = 6.0e-5   # C/(m²·K)
    p_AlN = 4.5e-5   # C/(m²·K)
    p_x   = x * p_AlN + (1.0 - x) * p_GaN
    params.Psp += (T - 300.0) * p_x
    return params

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

Two parameters are NOT in the NSM Archive (it has no piezoelectric-nitride-
specific or crystal-field-splitting-for-AlN entries) and are kept from
their original sources, noted where used below:
  - Spontaneous polarization Psp: Bernardini, Fiorentini & Vanderbilt,
    PRB 56:R10024 (1997); pyroelectric T-correction: Ambacher et al.,
    JAP 87:334 (2000).
  - AlN crystal-field splitting Delta_cr: Vurgaftman & Meyer, JAP 94:3675
    (2003) -- NSM's AlN band-structure table states this is "not
    separately listed".

GaN endpoint (x=0), all NSM/wurtzite/300 K unless noted:
  Eg=3.39 eV [Chow & Ghezzo 1996], chi=4.1 eV [Goldberg 2001-style NSM
  basic-parameters entry], eps_r=8.9, m_e=0.20 m0 (isotropic --  NSM gives
  no separate par/perp electron mass for wurtzite)
  m_hh=1.4 m0, m_lh=0.3 m0, m_so=0.6 m0 [Leszczynski et al./Fan et al. 1996]
  Delta_cr=0.04 eV, Delta_so=0.008 eV [Bougrov et al. 2001]
  a0=3.189 A, c0=5.186 A
  e33=0.65 C/m2, e31=-0.33 C/m2 [NSM piezoelectric table]
  C13=106 GPa, C33=398 GPa [Polian et al. 1996]
  Psp=-0.034 C/m2 [Bernardini 1997 -- not in NSM]

AlN endpoint (x=1), all NSM/wurtzite/300 K unless noted:
  Eg=6.026 eV [Guo & Yoshida 1994; Teisseyre et al. 1994], chi=0.6 eV
  [Goldberg 2001], eps_r=8.5 [Goldberg 2001], m_e=0.40 m0 (isotropic DOS
  mass [Xu & Ching 1993])
  m_hh=3.53 m0, m_lh=3.53 m0, m_so=0.25 m0 (kz/growth-direction masses
  [Suzuki & Uenoyama 1996] -- HH/LH sharing the same z-mass is the real,
  expected wurtzite k.p degeneracy at k_t=0, not a coincidence)
  Delta_so=0.019 eV [Goldberg 2001]; Delta_cr=-0.169 eV [Vurgaftman &
  Meyer 2003 -- not in NSM, see module docstring above]
  a0=3.112 A, c0=4.982 A
  e33=1.55 C/m2, e31=-0.58 C/m2 [Xinjiao et al. 1986, via NSM]
  C13=99 GPa, C33=389 GPa [McNeil et al. 1993]
  Psp=-0.081 C/m2 [Bernardini 1997 -- not in NSM]

chi(x) is a straight NSM-endpoint interpolation (4.1 eV -> 0.6 eV), not the
65:35 delta-Ec:delta-Eg band-offset rule used previously. That rule gave a
physically well-behaved type-I offset; literal NSM electron affinities
imply delta-Ec/delta-Eg ~= 1.35 (the valence edge would *rise* going toward
AlN), which contradicts the standard GaN/AlN type-I alignment literature --
electron affinity is a surface-sensitive quantity, and independently
measured GaN/AlN values aren't generally safe to difference for a bulk
heterojunction offset. Used here anyway per explicit instruction to source
every value from NSM; if AlGaN band diagrams start looking physically
wrong at the conduction/valence split, this is the first place to check.
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

    def Nc(self, T: float) -> float:
        """Effective conduction band DOS [cm^-3] at temperature T [K]."""
        return 2.0 * (2.0 * np.pi * self.m_e_dos * m0 * kB * T / h**2)**1.5 * 1e-6

    def Nv(self, T: float) -> float:
        """Effective valence band DOS [cm^-3] at temperature T [K]."""
        return 2.0 * (2.0 * np.pi * self.m_h_dos * m0 * kB * T / h**2)**1.5 * 1e-6

    def ni(self, T: float) -> float:
        """Intrinsic carrier concentration [cm^-3] at temperature T [K]."""
        Nc_ = self.Nc(T)
        Nv_ = self.Nv(T)
        kBT_eV = kB * T / 1.602176634e-19
        return np.sqrt(Nc_ * Nv_) * np.exp(-self.Eg / (2.0 * kBT_eV))

    def valence_band_structure(self) -> tuple[float, float, float, float]:
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
        d1 = self.Delta_cr
        d2 = d3 = self.Delta_so / 3.0
        e_hh = d1 + d2
        a = (d1 - d2) / 2.0
        b = np.sqrt(2.0) * d3
        evals, evecs = np.linalg.eigh(np.array([[a, b], [b, -a]]))
        e_lh = e_so = m_lh_eff = m_so_eff = None
        for e, v in zip(evals, evecs.T):
            if v[0] ** 2 >= v[1] ** 2:
                e_lh, m_lh_eff = float(e), self.m_lh
            else:
                e_so, m_so_eff = float(e), self.m_so
        return float(e_hh - e_lh), float(e_hh - e_so), float(m_lh_eff), float(m_so_eff)


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

    # --- Electron affinity [eV]: straight NSM-endpoint interpolation
    # (GaN 4.1 eV, AlN 0.6 eV) -- see the physically-questionable-but-
    # requested caveat in the module docstring.
    chi = x * 0.6 + (1.0 - x) * 4.1

    # --- Static relative permittivity (parallel to c-axis)
    eps_r = x * 8.5 + (1.0 - x) * 8.9

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
    Psp = x * (-0.081) + (1.0 - x) * (-0.034) - 0.021 * x * (1.0 - x)

    # --- Piezoelectric constants [C/m²] (NSM piezoelectric table)
    e33 = x * 1.55  + (1.0 - x) * 0.65
    e31 = x * (-0.58) + (1.0 - x) * (-0.33)

    # --- Elastic constants [Pa] (NSM, converted from GPa)
    C13 = (x * 99.0  + (1.0 - x) * 106.0) * 1e9
    C33 = (x * 389.0 + (1.0 - x) * 398.0) * 1e9

    return AlGaNParams(
        x_Al=x,
        Eg=Eg, chi=chi, eps_r=eps_r,
        m_e_par=m_e_par, m_e_perp=m_e_perp, m_e_dos=m_e_dos,
        m_hh=m_hh, m_lh=m_lh, m_so=m_so, m_h_dos=m_h_dos,
        Delta_cr=Delta_cr, Delta_so=Delta_so,
        a0=a0, c0=c0,
        Psp=Psp, e33=e33, e31=e31, C13=C13, C33=C33,
    )


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

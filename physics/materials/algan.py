"""
AlxGa(1-x)N material parameters for all compositions x in [0, 1].

Sources (all values from or consistent with):
  - Vurgaftman & Meyer, JAP 94:3675 (2003)          [bandgap, masses, lattice]
  - Bernardini, Fiorentini & Vanderbilt, PRB 56:R10024 (1997) [Psp, e33, e31, elastic]
  - Ambacher et al., JAP 85:3222 (1999)              [piezoelectric, 2DEG physics]
  - Ambacher et al., JAP 87:334 (2000)               [pyroelectric coefficients]

GaN endpoint (x=0):
  Eg=3.44 eV, chi=4.1 eV, eps_r=10.7, m_e_par=0.20 m0, m_e_perp=0.21 m0
  m_hh=1.87 m0, m_lh=0.14 m0, a0=3.189 Å, c0=5.186 Å
  Psp=-0.034 C/m², e33=0.87 C/m², e31=-0.50 C/m², C13=103 GPa, C33=405 GPa

AlN endpoint (x=1):
  Eg=6.12 eV, chi=1.9 eV, eps_r=8.5, m_e_par=0.30 m0, m_e_perp=0.32 m0
  m_hh=2.99 m0, m_lh=1.0 m0, a0=3.112 Å, c0=4.982 Å
  Psp=-0.081 C/m², e33=1.56 C/m², e31=-0.62 C/m², C13=108 GPa, C33=373 GPa
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
    m_h_dos: float      # Hole DOS effective mass (m_hh^1.5 + m_lh^1.5)^(2/3) [m0]
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


def get_AlGaN_params(x: float) -> AlGaNParams:
    """
    Return material parameters for AlxGa(1-x)N.

    x = 0.0  →  GaN
    x = 1.0  →  AlN
    All intermediate values use linear interpolation (Vegard's law) unless
    a bowing correction is known.
    """
    x = float(np.clip(x, 0.0, 1.0))

    # --- Bandgap [eV], bowing b = 0.7 eV (Vurgaftman 2003)
    Eg = x * 6.12 + (1.0 - x) * 3.44 - 0.7 * x * (1.0 - x)
    Eg_GaN = 3.44

    # --- Effective Electron affinity [eV] using 65:35 band offset ratio
    # Delta Ec = 0.65 * Delta Eg. Since Ec = -chi (vacuum ref), chi(x) = chi(0) - 0.65 * (Eg(x) - Eg(0))
    chi = 4.1 - 0.65 * (Eg - Eg_GaN)

    # --- Static relative permittivity (parallel to c-axis)
    eps_r = x * 8.5 + (1.0 - x) * 10.7

    # --- Electron effective masses [m0]
    m_e_par  = x * 0.30 + (1.0 - x) * 0.20   # parallel to c
    m_e_perp = x * 0.32 + (1.0 - x) * 0.21   # perpendicular to c
    m_e_dos  = (m_e_par**2 * m_e_perp)**(1.0 / 3.0)

    # --- Hole effective masses [m0]
    m_hh    = x * 2.99 + (1.0 - x) * 1.87
    m_lh    = x * 1.0  + (1.0 - x) * 0.14
    m_h_dos = (m_hh**1.5 + m_lh**1.5)**(2.0 / 3.0)

    # --- Lattice constants [Angstrom], Vegard's law
    a0 = x * 3.112 + (1.0 - x) * 3.189
    c0 = x * 4.982 + (1.0 - x) * 5.186

    # --- Spontaneous polarization [C/m²] (Bernardini 1997, with bowing)
    Psp = x * (-0.081) + (1.0 - x) * (-0.034) - 0.021 * x * (1.0 - x)

    # --- Piezoelectric constants [C/m²], linear interpolation
    e33 = x * 1.56  + (1.0 - x) * 0.87
    e31 = x * (-0.62) + (1.0 - x) * (-0.50)

    # --- Elastic constants [Pa] (converted from GPa)
    C13 = (x * 108.0 + (1.0 - x) * 103.0) * 1e9
    C33 = (x * 373.0 + (1.0 - x) * 405.0) * 1e9

    return AlGaNParams(
        x_Al=x,
        Eg=Eg, chi=chi, eps_r=eps_r,
        m_e_par=m_e_par, m_e_perp=m_e_perp, m_e_dos=m_e_dos,
        m_hh=m_hh, m_lh=m_lh, m_h_dos=m_h_dos,
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

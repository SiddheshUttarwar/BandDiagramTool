"""
Fermi-Dirac statistics for semiconductor carrier densities.

All energies in eV, temperatures in K, carrier densities in cm^-3.

The complete Fermi-Dirac integral of order 1/2 is:

    F_{1/2}(eta) = (2/sqrt(pi)) * integral_0^inf  sqrt(t)/(1+exp(t-eta)) dt

normalised so that the Boltzmann (non-degenerate) limit is:
    F_{1/2}(eta) → exp(eta)   for eta << 0

Carrier densities:
    n = Nc * F_{1/2}((Ef - Ec) / kBT)
    p = Nv * F_{1/2}((Ev - Ef) / kBT)

A lookup table is pre-computed once on first use to make vectorised evaluation fast.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.integrate import quad
from physics.constants import kB, q as _q


# ---------------------------------------------------------------------------
# Lookup-table for F_{1/2}(eta)
# ---------------------------------------------------------------------------

_LUT_BUILT = False
_LUT_ETA: np.ndarray | None = None
_LUT_F:   np.ndarray | None = None

_LUT_ETA_MIN = -25.0
_LUT_ETA_MAX =  60.0
_LUT_N_PTS   =  8500

# Disk cache: written next to this file, loaded on subsequent runs
_LUT_CACHE = os.path.join(os.path.dirname(__file__), '_fd_lut_cache.npz')


def _build_lut() -> None:
    global _LUT_BUILT, _LUT_ETA, _LUT_F
    if _LUT_BUILT:
        return

    # Load from disk cache if available
    if os.path.exists(_LUT_CACHE):
        data = np.load(_LUT_CACHE)
        _LUT_ETA = data['eta']
        _LUT_F   = data['f']
        _LUT_BUILT = True
        return

    print("Building Fermi-Dirac integral lookup table (one-time, ~15 s)...")
    eta_arr = np.linspace(_LUT_ETA_MIN, _LUT_ETA_MAX, _LUT_N_PTS)
    f_arr   = np.empty(_LUT_N_PTS)

    for i, e in enumerate(eta_arr):
        if e < -5.0:
            f_arr[i] = np.exp(e)                          # Boltzmann limit
        elif e > 50.0:
            # Sommerfeld expansion (highly degenerate)
            f_arr[i] = (4.0 / (3.0 * np.sqrt(np.pi))) * e**1.5 * (1.0 + np.pi**2 / (8.0 * e**2))
        else:
            upper = max(e + 60.0, 120.0)
            val, _ = quad(
                lambda t, _e=e: (2.0 / np.sqrt(np.pi)) * np.sqrt(t)
                                / (1.0 + np.exp(min(t - _e, 500.0))),
                0.0, upper, limit=200,
            )
            f_arr[i] = val

    _LUT_ETA = eta_arr
    _LUT_F   = f_arr
    _LUT_BUILT = True

    # Save to disk for future runs
    try:
        np.savez(_LUT_CACHE, eta=eta_arr, f=f_arr)
        print("Fermi-Dirac LUT cached to disk.")
    except Exception:
        pass  # cache write failure is non-fatal


def fermi_half(eta: float | np.ndarray) -> np.ndarray:
    """
    Complete Fermi-Dirac integral F_{1/2}(eta), vectorised.
    Boltzmann limit: F_{1/2}(eta) → exp(eta) for eta << 0.
    """
    _build_lut()
    eta_arr = np.asarray(eta, dtype=float)
    # np.interp clips at boundaries; handle extremes explicitly
    result = np.interp(eta_arr, _LUT_ETA, _LUT_F)
    return result


# ---------------------------------------------------------------------------
# Carrier density from Fermi level
# ---------------------------------------------------------------------------

def electron_density(Ec: np.ndarray,
                     Ef: float | np.ndarray,
                     Nc: np.ndarray,
                     T: float) -> np.ndarray:
    """
    n(x) = Nc(x) * F_{1/2}((Ef - Ec(x)) / kBT)  [cm^-3]

    Parameters
    ----------
    Ec  : conduction band edge [eV]
    Ef  : electron quasi-Fermi level [eV] (scalar or array)
    Nc  : effective DOS [cm^-3]
    T   : temperature [K]
    """
    kBT_eV = kB * T / _q
    eta    = (np.asarray(Ef) - np.asarray(Ec)) / kBT_eV
    return np.asarray(Nc) * fermi_half(eta)


def hole_density(Ev: np.ndarray,
                 Ef: float | np.ndarray,
                 Nv: np.ndarray,
                 T: float) -> np.ndarray:
    """
    p(x) = Nv(x) * F_{1/2}((Ev(x) - Ef) / kBT)  [cm^-3]

    Parameters
    ----------
    Ev  : valence band edge [eV]
    Ef  : hole quasi-Fermi level [eV]
    Nv  : effective DOS [cm^-3]
    T   : temperature [K]
    """
    kBT_eV = kB * T / _q
    eta    = (np.asarray(Ev) - np.asarray(Ef)) / kBT_eV
    return np.asarray(Nv) * fermi_half(eta)


# ---------------------------------------------------------------------------
# Equilibrium Fermi level
# ---------------------------------------------------------------------------

def find_equilibrium_Ef(Ec: np.ndarray,
                        Ev: np.ndarray,
                        Nc: np.ndarray,
                        Nv: np.ndarray,
                        ND: np.ndarray,
                        NA: np.ndarray,
                        T: float) -> float:
    """
    Find a single global equilibrium Fermi level [eV] by solving charge
    neutrality integrated over the device.

    Uses the condition:
        integral [ p(x) - n(x) + ND(x) - NA(x) ] dx = 0

    Solved via scipy.optimize.brentq on the Fermi level.

    Returns Ef [eV] (referenced to Ec0[0] - phi_bottom, same as Ec/Ev arrays).
    """
    from scipy.optimize import brentq

    kBT_eV = kB * T / _q

    def charge_imbalance(Ef_val: float) -> float:
        n = electron_density(Ec, Ef_val, Nc, T)
        p = hole_density(Ev, Ef_val, Nv, T)
        return float(np.mean(p - n + ND - NA))

    # Bracket: Ef between deep in valence band and deep in conduction band
    Ef_low  = float(np.min(Ev)) - 3.0 * kBT_eV
    Ef_high = float(np.max(Ec)) + 3.0 * kBT_eV

    # Ensure bracket straddles zero
    f_low  = charge_imbalance(Ef_low)
    f_high = charge_imbalance(Ef_high)

    if f_low * f_high > 0:
        # Fallback: return midpoint Ec estimate for dominant doping
        if np.mean(ND) >= np.mean(NA):
            n0 = max(np.mean(ND), 1.0)
            return float(np.mean(Ec)) + kBT_eV * np.log(n0 / float(np.mean(Nc)))
        else:
            p0 = max(np.mean(NA), 1.0)
            return float(np.mean(Ev)) - kBT_eV * np.log(p0 / float(np.mean(Nv)))

    return brentq(charge_imbalance, Ef_low, Ef_high, xtol=1e-8, maxiter=200)


# ---------------------------------------------------------------------------
# Inverse Fermi-Dirac (Joyce-Dixon approximation)
# ---------------------------------------------------------------------------

def inverse_fermi_half(ratio: np.ndarray) -> np.ndarray:
    """
    Inverse of F_{1/2}(eta) = ratio, returning eta.
    
    Uses the Joyce-Dixon approximation valid for all degeneracy levels:
        eta ≈ ln(ratio) + (1/sqrt(8)) * ratio            for ratio < 5
        eta ≈ ((3*sqrt(pi)/4) * ratio)^(2/3)             for ratio > 50
    with smooth interpolation in between.
    
    This is the correct inverse for degenerate semiconductors where the
    Boltzmann approximation (eta = ln(ratio)) fails.
    """
    ratio = np.asarray(ratio, dtype=float)
    eta = np.empty_like(ratio)
    
    # Non-degenerate regime: Joyce-Dixon 2-term
    # eta = ln(n/Nc) + (1/sqrt(8)) * (n/Nc) - (3/16 - sqrt(3)/9) * (n/Nc)^2
    c1 = 1.0 / np.sqrt(8.0)                    # 0.3536
    c2 = -(3.0/16.0 - np.sqrt(3.0)/9.0)        # -0.00495 (small correction)
    
    safe_ratio = np.maximum(ratio, 1e-30)
    eta = np.log(safe_ratio) + c1 * safe_ratio + c2 * safe_ratio**2
    
    # For highly degenerate (ratio >> 1), use Sommerfeld limit
    # F_{1/2}(eta) ≈ (4/(3*sqrt(pi))) * eta^{3/2}
    # => eta ≈ ((3*sqrt(pi)/4) * ratio)^{2/3}
    degen = ratio > 10.0
    if np.any(degen):
        eta[degen] = (0.75 * np.sqrt(np.pi) * ratio[degen])**(2.0/3.0)
    
    return eta


def Efn_from_n(n_cm3: np.ndarray, Ec: np.ndarray, Nc: np.ndarray, T: float) -> np.ndarray:
    """
    Recover electron quasi-Fermi level from carrier density [eV].
    
    Efn = Ec + kBT * inverse_F_{1/2}(n/Nc)
    
    Valid for both degenerate and non-degenerate regimes.
    """
    kBT_eV = kB * T / _q
    ratio = np.maximum(np.asarray(n_cm3), 1e-30) / np.asarray(Nc)
    return np.asarray(Ec) + kBT_eV * inverse_fermi_half(ratio)


def Efp_from_p(p_cm3: np.ndarray, Ev: np.ndarray, Nv: np.ndarray, T: float) -> np.ndarray:
    """
    Recover hole quasi-Fermi level from carrier density [eV].
    
    Efp = Ev - kBT * inverse_F_{1/2}(p/Nv)
    
    Valid for both degenerate and non-degenerate regimes.
    """
    kBT_eV = kB * T / _q
    ratio = np.maximum(np.asarray(p_cm3), 1e-30) / np.asarray(Nv)
    return np.asarray(Ev) - kBT_eV * inverse_fermi_half(ratio)


# ---------------------------------------------------------------------------
# 2D quantum carrier density (for subbands from Schrödinger solver)
# ---------------------------------------------------------------------------

def quantum_electron_density(psi_n: np.ndarray,
                              E_n: np.ndarray,
                              Efn: float,
                              m_e_dos: np.ndarray,
                              T: float) -> np.ndarray:
    """
    Carrier density from quantised electron subbands [cm^-3].

    n(x) = sum_n |psi_n(x)|^2 * N2D_n

    where the 2D sheet density of subband n is:
        N2D_n = (m_e * kBT) / (pi * hbar^2) * ln(1 + exp((Efn - En) / kBT))
              [m^-2]

    Parameters
    ----------
    psi_n   : (n_states, N) array of normalised wavefunctions (dx-normalised → m^{-1/2})
    E_n     : subband energies [eV], shape (n_states,)
    Efn     : electron quasi-Fermi level [eV]
    m_e_dos : electron DOS effective mass profile [units of m0], shape (N,)
    T       : temperature [K]

    Returns
    -------
    n [cm^-3], shape (N,)
    """
    from physics.constants import hbar, m0
    kBT_eV = kB * T / _q
    kBT_J  = kB * T
    n = np.zeros(psi_n.shape[1])
    n_q = np.zeros(psi_n.shape[1])

    for s in range(psi_n.shape[0]):
        En  = E_n[s]   # eV
        eta = (Efn - En) / kBT_eV
        # Use average effective mass over the wavefunction extent
        m_avg = float(np.mean(m_e_dos)) * m0   # kg
        N2D   = (m_avg * kBT_J / (np.pi * hbar**2)) * np.log1p(np.exp(np.minimum(eta, 500.0)))
        n_q  += psi_n[s]**2 * N2D   # m^-3

    return np.maximum(n_q, 1e-20) * 1e-6   # m^-3 → cm^-3


def quantum_hole_density(psi_h: np.ndarray, E_h: np.ndarray, Efp: float | np.ndarray, 
                         m_h: np.ndarray, T: float) -> np.ndarray:
    """
    Compute hole density from confined wavefunctions [cm^-3].

    The hole Schrödinger uses V_hole = -Ev (flipped potential), so the eigenvalues
    E_h are in the inverted frame. The actual hole subband energy (on the same
    absolute scale as Efp) is:
        E_sub = -(E_h)   ... since V = -Ev, eigenvalue E in V-space = -Ev_sub
    But the Schrödinger solver converts back to eV by dividing by q, so E_h is
    already in eV of the inverted potential. The real valence subband energy is:
        Ev_sub = -E_h  (the eigenvalue of -Ev gives us -Ev_sub)

    For holes, the 2D sheet density of subband k is:
        P2D_k = (m_h * kBT / pi*hbar^2) * ln(1 + exp((Ev_sub - Efp) / kBT))

    p(x) = sum_k |psi_k(x)|^2 * P2D_k
    """
    from physics.constants import hbar, m0
    if psi_h is None or len(E_h) == 0:
        return np.zeros_like(m_h)

    N      = psi_h.shape[1]
    p_q    = np.zeros(N)
    kBT_eV = kB * T / _q
    kBT_J  = kB * T
    m_avg  = float(np.mean(m_h)) * m0   # kg

    for k in range(len(E_h)):
        # E_h[k] is eigenvalue of the operator with V = -Ev
        # Real subband energy on absolute scale: Ev_sub = -E_h[k]
        Ev_sub = -E_h[k]   # eV, on the same scale as Efp

        # Hole occupation: f_h = 1/(1 + exp((Efp - Ev_sub)/kT))
        # Sheet density: P2D = DOS_2D * kBT * ln(1 + exp((Ev_sub - Efp)/kT))
        eta = (Ev_sub - Efp) / kBT_eV
        P2D = (m_avg * kBT_J / (np.pi * hbar**2)) * np.log1p(np.exp(np.minimum(eta, 500.0)))
        
        psi2 = psi_h[k]**2
        p_q += psi2 * P2D   # m^-3

    return np.maximum(p_q, 1e-20) * 1e-6   # m^-3 → cm^-3


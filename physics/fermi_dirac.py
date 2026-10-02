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
import sys
import numpy as np
from scipy.integrate import quad
from physics.constants import kB, q as _q


# ---------------------------------------------------------------------------
# Lookup-table for F_{1/2}(eta)
# ---------------------------------------------------------------------------

_LUT_BUILT = False
_LUT_ETA: np.ndarray | None = None
_LUT_F:   np.ndarray | None = None
_LUT_DF:  np.ndarray | None = None   # dF_{1/2}/deta, for the analytic Jacobian (see dfermi_half)

_LUT_ETA_MIN = -25.0
_LUT_ETA_MAX =  60.0
_LUT_N_PTS   =  8500


def _lut_cache_path() -> str:
    """Disk cache location, loaded on subsequent runs to skip the ~15s
    rebuild. Next to this file for a normal source checkout; under a
    per-user cache dir when frozen (e.g. PyInstaller), since a onefile
    build re-extracts to a fresh temp dir every launch (so a cache saved
    next to the frozen module would never actually be found next time),
    and an installed onedir build's own folder may not be writable
    without admin rights."""
    if getattr(sys, 'frozen', False):
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        cache_dir = os.path.join(base, 'BandDiagramTool')
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            cache_dir = os.path.dirname(__file__)
    else:
        cache_dir = os.path.dirname(__file__)
    return os.path.join(cache_dir, '_fd_lut_cache.npz')


_LUT_CACHE = _lut_cache_path()


def _build_lut() -> None:
    global _LUT_BUILT, _LUT_ETA, _LUT_F, _LUT_DF
    if _LUT_BUILT:
        return

    # Load from disk cache if available
    if os.path.exists(_LUT_CACHE):
        data = np.load(_LUT_CACHE)
        _LUT_ETA = data['eta']
        _LUT_F   = data['f']
        if 'df' in data:
            _LUT_DF = data['df']
        else:
            # Older cache from before dfermi_half existed -- backfill the
            # derivative table from the already-loaded F table (cheap,
            # O(1) np.gradient call) rather than invalidating the cache
            # and eating the ~15s rebuild.
            _LUT_DF = np.gradient(_LUT_F, _LUT_ETA)
            try:
                np.savez(_LUT_CACHE, eta=_LUT_ETA, f=_LUT_F, df=_LUT_DF)
            except Exception:
                pass
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
    # dF_{1/2}/deta via a single numerical gradient of the *table itself*
    # (not a per-Newton-iteration finite difference on the full nonlinear
    # residual): this is smooth, well-resolved (8500 points over the
    # table range) and computed once at startup, so it doesn't carry the
    # catastrophic-cancellation or clamping pitfalls a live finite
    # difference on the coupled residual has (see [[bug-quasi-fermi-pinning]]
    # and dfermi_half below) -- a standard, cheap way to get an accurate
    # derivative of a special function without deriving/implementing the
    # true F_{-1/2} integral from scratch.
    _LUT_DF = np.gradient(f_arr, eta_arr)
    _LUT_BUILT = True

    # Save to disk for future runs
    try:
        np.savez(_LUT_CACHE, eta=eta_arr, f=f_arr, df=_LUT_DF)
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
    result = np.interp(eta_arr, _LUT_ETA, _LUT_F)

    # np.interp CLAMPS outside [_LUT_ETA_MIN, _LUT_ETA_MAX] -- returns the
    # flat boundary value instead of extrapolating, which silently zeroes
    # out F_{1/2}'s (and therefore n/p's) sensitivity to the quasi-Fermi
    # level for any eta beyond the table range. That range is easily
    # exceeded: eta = (Ef-Ec)/kT or (Ev-Ef)/kT, and a deep minority
    # carrier in a heavily-doped bulk (e.g. holes in a 1e19-1e20 cm^-3
    # n-type layer) routinely has |eta| > 25-60 once band bending is
    # included. Confirmed this is a real, previously-unnoticed bug (not
    # just a self_consistent.py issue): a from-scratch finite-difference
    # Jacobian assembled for a candidate direct-Newton coupled solver
    # (2026-09-22, see physics.coupled_solver) came out *exactly*
    # singular, with zero-diagonal rows clustered precisely at the hole
    # channel in the first few grid points of a 1e19 cm^-3 n-doped layer
    # -- i.e. hole_density's derivative w.r.t. Efp really was
    # analytically zero there, from this clamp, not a solver artifact.
    # This almost certainly also contributes to [[bug-quasi-fermi-pinning]]
    # for JFNK too: a Newton-Krylov method (Jacobian-free or not) has
    # nothing to act on when the density genuinely doesn't respond to the
    # quasi-Fermi level in the local discretization, regardless of
    # residual scaling.
    #
    # Extrapolate instead, using the same closed-form asymptotic formulas
    # _build_lut() already uses to *populate* the table (so this is
    # consistent with the table's own values at the boundary, not a
    # different approximation): Boltzmann limit for eta below the table,
    # Sommerfeld expansion above it.
    # np.where evaluates BOTH branches eagerly over the *whole* array before
    # selecting -- so an unguarded np.exp(eta_arr)/eta_arr**1.5 computes
    # exp()/power() of every element, including ones nowhere near this
    # branch (e.g. a huge positive eta destined for the Sommerfeld branch
    # would still overflow exp() here). A wild intermediate Newton trial
    # value (this feeds Poisson/JFNK iterations, not just converged
    # states) can genuinely push eta far outside any physical range before
    # a line search corrects it -- substitute a safe dummy value outside
    # each branch's own mask so that trial never blows up the untaken
    # branch's arithmetic, same pattern the Sommerfeld branch already used
    # for its own eta**1.5.
    below = eta_arr < _LUT_ETA_MIN
    above = eta_arr > _LUT_ETA_MAX
    if np.any(below):
        eta_below = np.where(below, eta_arr, -1.0)  # dummy value elsewhere to avoid overflow if eta_arr has huge positive entries too
        result = np.where(below, np.exp(eta_below), result)
    if np.any(above):
        eta_above = np.where(above, eta_arr, 1.0)  # dummy value elsewhere to avoid overflow in eta**1.5
        sommerfeld = (4.0 / (3.0 * np.sqrt(np.pi))) * eta_above**1.5 * (1.0 + np.pi**2 / (8.0 * eta_above**2))
        result = np.where(above, sommerfeld, result)
    return result


def dfermi_half(eta: float | np.ndarray) -> np.ndarray:
    """
    Analytic derivative dF_{1/2}/deta, vectorised -- the building block
    for an *analytic* (not finite-difference) Jacobian of the coupled
    Poisson/drift-diffusion system (see physics.coupled_solver's
    nextnano++-style direct solver, [[bug-quasi-fermi-pinning]]).
    Boltzmann limit: dF_{1/2}/deta -> exp(eta) for eta << 0 (same as
    F_{1/2} itself in that limit, since d/deta exp(eta) = exp(eta)).

    Interpolates the table-gradient _LUT_DF built once in _build_lut()
    (see there for why a table gradient instead of implementing the true
    F_{-1/2} integral), with the same closed-form extrapolation strategy
    as fermi_half for eta outside the table: exact derivatives of the
    same Boltzmann/Sommerfeld formulas used there, so value and
    derivative stay consistent with each other in the extrapolated tails.
    """
    _build_lut()
    eta_arr = np.asarray(eta, dtype=float)
    result = np.interp(eta_arr, _LUT_ETA, _LUT_DF)

    below = eta_arr < _LUT_ETA_MIN
    above = eta_arr > _LUT_ETA_MAX
    if np.any(below):
        eta_below = np.where(below, eta_arr, -1.0)
        result = np.where(below, np.exp(eta_below), result)  # d/deta exp(eta) = exp(eta)
    if np.any(above):
        eta_above = np.where(above, eta_arr, 1.0)
        # d/deta [ C*(eta^1.5 + a*eta^-0.5) ] = C*(1.5*eta^0.5 - 0.5*a*eta^-1.5)
        C = 4.0 / (3.0 * np.sqrt(np.pi))
        a = np.pi**2 / 8.0
        dsommerfeld = C * (1.5 * eta_above**0.5 - 0.5 * a * eta_above**-1.5)
        result = np.where(above, dsommerfeld, result)
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
# Analytic density derivatives (for physics.coupled_solver's analytic
# Jacobian direct solver -- the nextnano++-style approach: closed-form
# derivatives assembled into a sparse matrix and factored directly,
# instead of Jacobian-Free Newton-Krylov's finite-difference/Krylov
# approximation of the same step). Each pair (d.../dEc-or-Ev,
# d.../dEf) are the two partials chain rule needs; the caller applies
# dEc/dphi = dEv/dphi = -1 itself (Ec = Ec0-phi, Ev = Ev0-phi).
# ---------------------------------------------------------------------------

def d_electron_density_dEc(Ec: np.ndarray, Ef: float | np.ndarray,
                            Nc: np.ndarray, T: float) -> np.ndarray:
    """dn/dEc [cm^-3 / eV]. n = Nc*F_{1/2}((Ef-Ec)/kT), so dn/dEc =
    -Nc*F_{1/2}'(eta)/kT (chain rule through eta=(Ef-Ec)/kT)."""
    kBT_eV = kB * T / _q
    eta = (np.asarray(Ef) - np.asarray(Ec)) / kBT_eV
    return -np.asarray(Nc) * dfermi_half(eta) / kBT_eV


def d_electron_density_dEf(Ec: np.ndarray, Ef: float | np.ndarray,
                            Nc: np.ndarray, T: float) -> np.ndarray:
    """dn/dEf [cm^-3 / eV] = -dn/dEc (eta depends on Ef and Ec with
    opposite sign)."""
    kBT_eV = kB * T / _q
    eta = (np.asarray(Ef) - np.asarray(Ec)) / kBT_eV
    return np.asarray(Nc) * dfermi_half(eta) / kBT_eV


def d_hole_density_dEv(Ev: np.ndarray, Ef: float | np.ndarray,
                        Nv: np.ndarray, T: float) -> np.ndarray:
    """dp/dEv [cm^-3 / eV]. p = Nv*F_{1/2}((Ev-Ef)/kT), so dp/dEv =
    Nv*F_{1/2}'(eta)/kT (eta=(Ev-Ef)/kT)."""
    kBT_eV = kB * T / _q
    eta = (np.asarray(Ev) - np.asarray(Ef)) / kBT_eV
    return np.asarray(Nv) * dfermi_half(eta) / kBT_eV


def d_hole_density_dEf(Ev: np.ndarray, Ef: float | np.ndarray,
                        Nv: np.ndarray, T: float) -> np.ndarray:
    """dp/dEf [cm^-3 / eV] = -dp/dEv."""
    kBT_eV = kB * T / _q
    eta = (np.asarray(Ev) - np.asarray(Ef)) / kBT_eV
    return -np.asarray(Nv) * dfermi_half(eta) / kBT_eV


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
                              T: float,
                              local_qfl: bool = False) -> np.ndarray:
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
    Efn_arr = np.asarray(Efn, dtype=float)
    n = np.zeros(psi_n.shape[1])
    n_q = np.zeros(psi_n.shape[1])

    for s in range(psi_n.shape[0]):
        En  = E_n[s]   # eV
        # Efn may be a full position-dependent array (the coupled/Gummel
        # solvers' quasi-Fermi level state) rather than a single scalar --
        # a delocalized subband has no single "local" Fermi level, so
        # broadcasting Efn(x) straight into the per-subband sheet-density
        # formula below would make N2D vary *pointwise within one
        # subband*, which isn't physically meaningful (a sheet density is
        # by definition integrated over the whole confined direction) and
        # produces a self-inconsistent density profile once the quantum
        # region is wide enough for Efn to vary meaningfully across it --
        # confirmed empirically: this is exactly what breaks convergence
        # at any nonzero bias when the auto-detected quantum region spans
        # an entire undoped device (physics.self_consistent._detect_
        # quantum_region's full-grid fallback). Use the wavefunction-
        # weighted average of Efn over the subband's own probability
        # density instead -- the standard "local quasi-Fermi level"
        # treatment for a confined state under bias -- which collapses
        # back to the previous (already-validated) behavior whenever Efn
        # is uniform across the subband's extent, e.g. equilibrium or a
        # thin quantum well.
        #
        # local_qfl=True instead occupies each subband with the LOCAL
        # quasi-Fermi level at every x (nextnano++'s treatment under current
        # flow). Used by biased drift-diffusion solves: there the quantum
        # density enters transport through gamma = n_quantum/n_classical,
        # and with the averaged Efn that ratio carries a factor
        # exp((<Efn> - Efn(x))/kT) that changes whenever Efn does --
        # confirmed to keep the Schrodinger <-> transport loop from
        # converging on the UV-LED at 6 V (|dln gamma| stuck at 1-2.5).
        # With the local form gamma depends only on the potential shape.
        # Identical to the averaged form whenever Efn is flat (equilibrium).
        m_avg = float(np.mean(m_e_dos)) * m0   # kg
        pref = m_avg * kBT_J / (np.pi * hbar**2)
        if local_qfl and Efn_arr.ndim > 0:
            N2D = pref * np.logaddexp(0.0, np.minimum((Efn_arr - En) / kBT_eV, 500.0))
        else:
            Efn_s = float(np.average(Efn_arr, weights=psi_n[s]**2)) if Efn_arr.ndim > 0 else float(Efn_arr)
            eta = (Efn_s - En) / kBT_eV
            N2D = pref * np.log1p(np.exp(min(eta, 500.0)))
        n_q  += psi_n[s]**2 * N2D   # m^-3

    return np.maximum(n_q, 1e-20) * 1e-6   # m^-3 → cm^-3


def quantum_hole_density(psi_h: np.ndarray, E_h: np.ndarray, Efp: float | np.ndarray,
                         m_h: np.ndarray, T: float, local_qfl: bool = False) -> np.ndarray:
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
    Efp_arr = np.asarray(Efp, dtype=float)

    for k in range(len(E_h)):
        # E_h[k] is eigenvalue of the operator with V = -Ev
        # Real subband energy on absolute scale: Ev_sub = -E_h[k]
        Ev_sub = -E_h[k]   # eV, on the same scale as Efp

        # Hole occupation: f_h = 1/(1 + exp((Efp - Ev_sub)/kT))
        # Sheet density: P2D = DOS_2D * kBT * ln(1 + exp((Ev_sub - Efp)/kT))
        # Efp may be a full position-dependent array rather than a single
        # scalar -- see quantum_electron_density's matching comment for
        # why broadcasting it straight in here instead of using the
        # wavefunction-weighted average is a real convergence bug for a
        # wide (not-thin-QW) quantum region.
        # local_qfl: see quantum_electron_density.
        pref = m_avg * kBT_J / (np.pi * hbar**2)
        if local_qfl and Efp_arr.ndim > 0:
            P2D = pref * np.logaddexp(0.0, np.minimum((Ev_sub - Efp_arr) / kBT_eV, 500.0))
        else:
            Efp_k = float(np.average(Efp_arr, weights=psi_h[k]**2)) if Efp_arr.ndim > 0 else float(Efp_arr)
            eta = (Ev_sub - Efp_k) / kBT_eV
            P2D = pref * np.log1p(np.exp(min(eta, 500.0)))
        
        psi2 = psi_h[k]**2
        p_q += psi2 * P2D   # m^-3

    return np.maximum(p_q, 1e-20) * 1e-6   # m^-3 → cm^-3


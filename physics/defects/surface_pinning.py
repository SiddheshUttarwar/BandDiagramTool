"""
Surface-state Fermi-level pinning model: connects the BCF adatom surface
density n [cm^-2] (the MOCVD growth-kinetics quantity from the surface
continuity equation, n(x) = F*tau_d - (F*tau_d - n_eq)*cosh(..)/cosh(..))
to the LOCAL Fermi level at the growing surface, and from there to the
defect-formation-energy machinery in physics/defects/formation_energy.py.

This is a standalone, deliberately simplified model -- NOT a re-derivation
of the full nonlinear Schrodinger-Poisson solver in
physics/self_consistent.py. It captures the same qualitative physics
(surface states -> charge transfer -> band bending -> pinned surface
Fermi level) with a single self-consistent nonlinear equation (the classic
depletion-approximation surface-state-pinning problem), appropriate for a
0-D "what adatom surface density n minimizes defect formation" sweep
rather than a full 1-D device solve.

Physical picture (all energies referenced to the LOCAL valence-band
maximum of the growing surface; composition x_Al is fixed -- default pure
GaN, matching the original BCF/MOCVD discussion):

  * The bulk semiconductor (doping ND) has a flat equilibrium Fermi level
    E_F_bulk above the bulk VBM (reused from
    physics.fermi_dirac.find_equilibrium_Ef).
  * A single, lumped, amphoteric adatom-induced surface-state level sits
    at a FIXED energy E_D above the local VBM, with areal density
        N_ss(n) = c*n + N_vac                              [cm^-2]
    (c, N_vac, E_D are calibration constants -- no literature values were
    available; see the dataclass docstring below).
  * Band bending V_s >= 0 (surface bands bend UP relative to the deep
    bulk -- the standard n-type depletion picture). Charge neutrality
    between the bulk depletion charge (ionized donors) and the surface
    charge (electrons captured from the bulk -- literally the mechanism
    described earlier: "this charge transfer leaves behind a depletion
    region of ionized dopants... and creates a sheet of charge at the
    surface") gives:
        Q_bulk(V_s) = +sqrt(2*q*eps_s*ND*V_s)               [C/cm^2]
        Q_ss(V_s)   = -q*N_ss*f(E_D + V_s, E_F_bulk, T)     [C/cm^2]
        Q_bulk(V_s) + Q_ss(V_s) = 0          <- solved for V_s
    (f is the ordinary Fermi-Dirac occupation function; E_D+V_s is the
    surface state's energy on the SAME absolute scale as the flat bulk
    E_F_bulk, since the local VBM itself sits at +V_s relative to the deep
    bulk VBM reference).
  * Local surface Fermi level above the LOCAL (surface) VBM:
        E_F_local_surface = E_F_bulk - V_s
    (bands bend up, so the gap between the flat E_F and the now-higher
    local VBM shrinks -- the local Fermi level moves down, away from Ec,
    exactly the n-type depletion picture).

CALIBRATION CONSTANTS -- no literature source for these was available;
flagged exactly as such (matching the honesty standard set in
physics/materials/defect_constants.py):
  * c, N_vac_cm2    : relate adatom density n to surface trap density
  * E_D_above_Ev_eV : illustrative surface-trap energy (default: near
                      mid-gap for GaN)
  * N_D_bulk_cm3    : bulk doping, used only for the reference bulk Fermi
                      level and depletion charge
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import brentq

from physics.constants import kB, q as _q_charge, eps0
from physics.materials.algan import get_AlGaN_params_T
from physics.fermi_dirac import find_equilibrium_Ef
from physics.materials.defect_constants import DefectSpecies
from physics.defects.formation_energy import compute_defect_formation_profile


@dataclass
class SurfaceStateModel:
    c: float = 1.0                  # N_ss = c*n + N_vac (fraction of adatoms that trap)
    N_vac_cm2: float = 1.0e11       # background trap density [cm^-2] -- calibration constant
    E_D_above_Ev_eV: float = 1.7    # surface trap energy, illustrative mid-gap-ish placement
    N_D_bulk_cm3: float = 1.0e17    # bulk n-type doping [cm^-3], for the reference bulk E_F
    x_Al: float = 0.0               # composition (default: pure GaN)


def N_ss_cm2(n_adatom_cm2, model: SurfaceStateModel) -> np.ndarray:
    """Surface trap density [cm^-2] from adatom density n [cm^-2]."""
    return model.c * np.asarray(n_adatom_cm2, dtype=float) + model.N_vac_cm2


def _occupation(E_level_eV, E_F_eV: float, T: float) -> float:
    kBT_eV = kB * T / _q_charge
    return 1.0 / (1.0 + np.exp(np.clip((E_level_eV - E_F_eV) / kBT_eV, -500.0, 500.0)))


def bulk_fermi_level_eV(model: SurfaceStateModel, T: float = 300.0) -> float:
    """Flat bulk equilibrium Fermi level [eV] above the bulk VBM."""
    params = get_AlGaN_params_T(model.x_Al, T)
    Nc = params.Nc(T)
    Nv = params.Nv(T)
    return find_equilibrium_Ef(Ec=params.Eg, Ev=0.0, Nc=Nc, Nv=Nv,
                                ND=model.N_D_bulk_cm3, NA=0.0, T=T)


def solve_band_bending_eV(n_adatom_cm2: float, model: SurfaceStateModel,
                            EF_bulk_eV: float, T: float = 300.0) -> float:
    """
    Solve the self-consistent depletion-approximation band bending V_s [eV]
    (>= 0) for one adatom surface density n [cm^-2].
    """
    params = get_AlGaN_params_T(model.x_Al, T)
    eps_s_Fcm = params.eps_r * eps0 * 1e-2   # F/m -> F/cm
    N_ss = float(N_ss_cm2(n_adatom_cm2, model))

    def residual(V_s: float) -> float:
        Q_bulk = np.sqrt(2.0 * _q_charge * eps_s_Fcm * model.N_D_bulk_cm3 * max(V_s, 0.0))
        f_occ = _occupation(model.E_D_above_Ev_eV + V_s, EF_bulk_eV, T)
        Q_ss = -_q_charge * N_ss * f_occ
        return Q_bulk + Q_ss

    V_hi = max(3.0 * params.Eg, 3.0)
    lo, hi = 1e-6, V_hi
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo * r_hi > 0:
        # No sign change across the whole bracket (e.g. N_ss too small to
        # require meaningful bending): return whichever end is closer to
        # neutrality rather than letting brentq raise.
        return 0.0 if abs(r_lo) < abs(r_hi) else V_hi
    return brentq(residual, lo, hi, xtol=1e-9, maxiter=200)


def local_surface_EF_eV(n_adatom_cm2, model: SurfaceStateModel, T: float = 300.0
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Vectorised local surface Fermi level above the LOCAL VBM [eV] vs. adatom density n."""
    EF_bulk = bulk_fermi_level_eV(model, T)
    n_arr = np.atleast_1d(np.asarray(n_adatom_cm2, dtype=float))
    V_s = np.array([solve_band_bending_eV(n, model, EF_bulk, T) for n in n_arr])
    EF_local = EF_bulk - V_s
    return EF_local, V_s, EF_bulk


def sweep_defect_formation_vs_n(n_array: np.ndarray, model: SurfaceStateModel,
                                  species_list: List[DefectSpecies],
                                  condition: str = "N_rich", T: float = 300.0) -> Dict:
    """
    For each n in n_array, compute every species' formation energy at the
    resulting pinned local surface Fermi level, plus the worst-case
    (minimum-over-species) formation energy -- the quantity that actually
    controls total surface defect density, since whichever species has the
    lowest E^f dominates thermodynamically.
    """
    n_array = np.asarray(n_array, dtype=float)
    EF_local, V_s, EF_bulk = local_surface_EF_eV(n_array, model, T)

    params = get_AlGaN_params_T(model.x_Al, T)
    fake_result = SimpleNamespace(
        x_Al=np.full_like(n_array, model.x_Al),
        Ev=np.zeros_like(n_array),
        Ec=np.full_like(n_array, params.Eg),
        Efn=EF_local, Efp=EF_local,
        x_nm=n_array, T=T,
    )

    per_species = {}
    for sp in species_list:
        prof = compute_defect_formation_profile(fake_result, sp, condition=condition)
        per_species[sp.name] = prof.E_formation_min_eV

    worst_case = np.min(np.stack(list(per_species.values()), axis=0), axis=0)

    return {
        "n": n_array, "V_s": V_s, "EF_local": EF_local, "EF_bulk": EF_bulk,
        "per_species": per_species, "worst_case": worst_case,
    }


def find_optimal_n(model: SurfaceStateModel, species_list: List[DefectSpecies],
                     n_bounds: Tuple[float, float] = (1e9, 1e14), n_points: int = 400,
                     condition: str = "N_rich", T: float = 300.0) -> Dict:
    """
    Sweep adatom surface density n over n_bounds (log-spaced) and return the
    n* that MAXIMIZES the worst-case (minimum-over-species) formation
    energy -- the surface density that minimizes overall surface defect
    density, given the competing (opposite-sign-slope) demands of the
    donor-like and acceptor-like species tracked here.
    """
    n_array = np.geomspace(n_bounds[0], n_bounds[1], n_points)
    result = sweep_defect_formation_vs_n(n_array, model, species_list, condition, T)
    idx = int(np.argmax(result["worst_case"]))
    result["n_star"] = n_array[idx]
    result["idx_star"] = idx
    binding = min(result["per_species"].items(), key=lambda kv: kv[1][idx])[0]
    result["binding_species_at_optimum"] = binding
    return result

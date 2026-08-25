"""
Core engine: point-defect / dopant formation energy through a graded
Al_xGa_(1-x)N structure, following Van de Walle & Neugebauer (2004):

    E^f[X^q](z) = E_bare(q, x(z)) + q*E_F_local(z) + sum_i n_i*dmu_i(x(z))

where E_F_local(z) = E_F(z) - Ev(z) is the Fermi level measured above the
LOCAL valence-band maximum (the convention the literature transition
levels in physics/materials/defect_constants.py are anchored to).

This module is a downstream CONSUMER of physics/self_consistent.py's
SolverResult -- it does not solve any electrostatics itself. The real
Poisson/Schrodinger solve (including polarization charge as a genuine
source term) already gives position-resolved Ec(z), Ev(z), Efn(z), Efp(z)
for arbitrary graded x(z); this module only evaluates defect thermodynamics
on top of that.

See physics/materials/defect_constants.py's module docstring for the
honest accounting of what is/isn't literature-exact in this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from physics.constants import kB, q as _q_charge
from physics.materials.algan import get_AlGaN_params
from physics.materials.chemical_potentials import get_mu_offsets_profile
from physics.materials.defect_constants import DefectChargeState, DefectSpecies
from physics.defects.site_density import site_density_profile_cm3
from physics.defects.dx_center import dx_branch_E_bare_minus

# Reference bandgaps (GaN, AlN endpoints), used only for the "bandgap_ratio"
# AlN E_bare scaling fallback and for the DX-branch target-level construction.
_EG_GAN_REF = get_AlGaN_params(0.0).Eg
_EG_ALN_REF = get_AlGaN_params(1.0).Eg


@dataclass
class ChargeStateProfile:
    q: int
    branch: str                  # "normal" | "DX"
    E_formation_eV: np.ndarray   # (N,) formation energy vs. position


@dataclass
class DefectProfileResult:
    species: str
    x_nm: np.ndarray
    x_Al: np.ndarray
    condition: str
    T: float
    charge_states: List[ChargeStateProfile]
    E_formation_min_eV: np.ndarray   # convex-hull (min over q) formation energy per z
    q_stable: np.ndarray              # dominant charge state per z
    N_sites_cm3: np.ndarray
    concentration_cm3: np.ndarray     # N_sites(z) * exp(-E_formation_min_eV(z)/kT)
    dx_crossover_nm: Optional[float]  # z where stable branch flips normal -> DX (donors only)
    notes: List[str] = field(default_factory=list)


def _interp_E_bare(cs: DefectChargeState, x_Al: np.ndarray, e_bare_scaling: str,
                     absolute_offset_eV: float = 0.0) -> np.ndarray:
    """Linearly interpolate E_bare(q, x) between the GaN (x=0) and AlN (x=1) endpoints,
    plus the species' uniform absolute_offset_eV (see DefectSpecies docstring --
    this only moves the whole species' zero point, it does not change relative
    transition levels within the species)."""
    E_GaN = cs.E_bare_GaN_eV
    if cs.E_bare_AlN_eV is not None:
        E_AlN = cs.E_bare_AlN_eV
    elif e_bare_scaling == "bandgap_ratio":
        # Weakest-justified approximation in this module -- see
        # defect_constants.py docstring. No literature AlN endpoint value
        # was available, so we crudely scale by the bandgap ratio.
        E_AlN = E_GaN * (_EG_ALN_REF / _EG_GAN_REF)
    else:
        E_AlN = E_GaN
    return (1.0 - x_Al) * E_GaN + x_Al * E_AlN + absolute_offset_eV


def _chem_pot_term(stoichiometry, mu_map, impurity_mu_eV, x_Al) -> np.ndarray:
    """- sum_i n_i * dmu_i(x), pulling host-atom (Ga/Al/N) offsets from the
    alloy chemical-potential profile and impurity (O/Si/...) offsets from
    the species' fixed, user-set impurity_mu_eV table (default 0.0)."""
    term = np.zeros_like(x_Al)
    for element, n_i in stoichiometry.items():
        if element in mu_map:
            dmu = mu_map[element]
        else:
            dmu = impurity_mu_eV.get(element, 0.0)
        term = term - n_i * dmu
    return term


def compute_defect_formation_profile(
    result,
    species: DefectSpecies,
    condition: str = "N_rich",
    ef_mode: str = "midgap",
) -> DefectProfileResult:
    """
    Compute the position-resolved formation-energy profile for `species`
    through a graded structure already solved by
    physics.self_consistent.solve_self_consistent().

    Parameters
    ----------
    result    : SolverResult (from AlGaNDevice.solve() / solve_self_consistent())
    species   : DefectSpecies (e.g. V_GA, V_N, O_N, SI_GA from defect_constants.py)
    condition : "N_rich" | "metal_rich" growth-condition boundary
    ef_mode   : which Fermi level to use -- "midgap" (0.5*(Efn+Efp), the
                equilibrium choice), "Efn", or "Efp"
    """
    x_Al = np.asarray(result.x_Al, dtype=float)
    Ev = np.asarray(result.Ev, dtype=float)
    Ec = np.asarray(result.Ec, dtype=float)
    Eg_z = Ec - Ev

    if ef_mode == "midgap":
        E_F = 0.5 * (np.asarray(result.Efn) + np.asarray(result.Efp))
    elif ef_mode == "Efn":
        E_F = np.asarray(result.Efn, dtype=float)
    elif ef_mode == "Efp":
        E_F = np.asarray(result.Efp, dtype=float)
    else:
        raise ValueError(f"Unknown ef_mode '{ef_mode}'. Use 'midgap', 'Efn', or 'Efp'.")

    E_F_local = E_F - Ev   # energy above LOCAL valence-band maximum

    mu = get_mu_offsets_profile(x_Al, condition)
    mu_map = {"Ga": mu.dmu_Ga, "Al": mu.dmu_Al, "N": mu.dmu_N}

    charge_state_profiles: List[ChargeStateProfile] = []
    for cs in species.charge_states:
        E_bare_x = _interp_E_bare(cs, x_Al, species.e_bare_scaling, species.absolute_offset_eV)
        mu_term = _chem_pot_term(cs.stoichiometry, mu_map, species.impurity_mu_eV, x_Al)
        E_f = E_bare_x + cs.q * E_F_local + mu_term
        charge_state_profiles.append(ChargeStateProfile(q=cs.q, branch=cs.branch, E_formation_eV=E_f))

    if species.absolute_offset_eV != 0.0:
        notes = [
            f"absolute_offset_eV={species.absolute_offset_eV} eV applied: an "
            "approximate, order-of-magnitude visual calibration against Fig. 5 of "
            "Van de Walle & Neugebauer (2004), NOT a precise tabulated DFT value. "
            "Cross-species comparison against other offset-calibrated species is "
            "directionally meaningful; comparison against species still at the "
            "default 0.0 offset (uncalibrated) is not.",
        ]
    else:
        notes = [
            "E_bare zero point is uncalibrated (absolute_offset_eV=0.0, the "
            "default): absolute cross-species comparisons and absolute "
            "concentration magnitudes involving this species are illustrative "
            "only, not first-principles (see physics/materials/defect_constants.py "
            "docstring).",
        ]

    if species.dx_params is not None:
        shallow_cs = next(cs for cs in species.charge_states if cs.branch == "normal")
        E_bare_shallow_x = _interp_E_bare(shallow_cs, x_Al, species.e_bare_scaling, species.absolute_offset_eV)
        E_bare_DX_minus_x = dx_branch_E_bare_minus(x_Al, Eg_z, E_bare_shallow_x, species.dx_params)
        mu_term_dx = _chem_pot_term(shallow_cs.stoichiometry, mu_map, species.impurity_mu_eV, x_Al)
        E_f_dx = E_bare_DX_minus_x + (-1) * E_F_local + mu_term_dx
        charge_state_profiles.append(ChargeStateProfile(q=-1, branch="DX", E_formation_eV=E_f_dx))
        notes.append(
            f"DX-center crossover calibrated to x_c={species.dx_params.x_c} "
            f"(literature range {species.dx_params.x_c_lit_range}); deep-level depth "
            f"{species.dx_params.E_DX_deep_below_Ec_eV} eV below Ec is an illustrative "
            "calibration constant, not a first-principles DFT value."
        )
        notes.append(
            "The ACTUAL dx_crossover_nm reported below depends on where this "
            "device's self-consistent Fermi level sits relative to Ec, not on "
            "x_c alone: x_c is only the composition at which the model's DX "
            "target level equals its own shallow/deep logistic midpoint. A "
            "more heavily n-type (Fermi level closer to Ec) structure will "
            "show the crossover shift to a different composition than x_c."
        )

    E_stack = np.stack([cs.E_formation_eV for cs in charge_state_profiles], axis=0)
    q_array = np.array([cs.q for cs in charge_state_profiles])

    idx_min = np.argmin(E_stack, axis=0)
    E_formation_min_eV = E_stack[idx_min, np.arange(E_stack.shape[1])]
    q_stable = q_array[idx_min]

    N_sites = site_density_profile_cm3(x_Al, result.T)
    kBT_eV = kB * result.T / _q_charge
    # Saturating (Fermi-like) form rather than a raw Boltzmann exp(): reduces
    # to the usual dilute-limit c = N_sites*exp(-E_f/kT) when E_f >> kT, but
    # correctly saturates at N_sites (100% site occupation, the physical
    # ceiling) instead of overflowing when a species' arbitrarily-anchored
    # E_bare zero point (see module docstring) makes E_f swing very negative.
    exponent = np.clip(E_formation_min_eV / kBT_eV, -700.0, 700.0)
    concentration_cm3 = N_sites / (1.0 + np.exp(exponent))

    dx_crossover_nm = None
    if species.dx_params is not None:
        normal_idx = [i for i, cs in enumerate(charge_state_profiles) if cs.branch == "normal"]
        dx_idx = [i for i, cs in enumerate(charge_state_profiles) if cs.branch == "DX"]
        E_normal_min = np.min(E_stack[normal_idx, :], axis=0)
        E_dx_min = np.min(E_stack[dx_idx, :], axis=0)
        diff = E_normal_min - E_dx_min   # > 0 once the DX branch becomes favorable
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_changes) > 0:
            i0 = int(sign_changes[0])
            x0, x1 = result.x_nm[i0], result.x_nm[i0 + 1]
            d0, d1 = diff[i0], diff[i0 + 1]
            frac = -d0 / (d1 - d0) if d1 != d0 else 0.0
            dx_crossover_nm = float(x0 + frac * (x1 - x0))

    return DefectProfileResult(
        species=species.name,
        x_nm=result.x_nm,
        x_Al=x_Al,
        condition=condition,
        T=result.T,
        charge_states=charge_state_profiles,
        E_formation_min_eV=E_formation_min_eV,
        q_stable=q_stable,
        N_sites_cm3=N_sites,
        concentration_cm3=concentration_cm3,
        dx_crossover_nm=dx_crossover_nm,
        notes=notes,
    )

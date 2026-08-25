"""
Reference data for point-defect / dopant formation-energy calculations in
Al_xGa_(1-x)N, following the Van de Walle & Neugebauer (2004) formalism
(J. Appl. Phys. 95, 3851):

    E^f[X^q](x) = E_bare(q, x) + q*E_F_local + sum_i n_i * dmu_i(x)

IMPORTANT — read before trusting any absolute number out of this module:

  * Each defect species below is calibrated ONLY to reproduce its own
    literature-reported RELATIVE charge-state transition level (e.g. V_N's
    (3+/+) level at 0.59 eV above the valence-band maximum, V_Ga's (2-/3-)
    level at 1.1 eV — both from Van de Walle & Neugebauer 2004, GaN,
    Ga-rich conditions). The E_bare "zero point" is chosen independently
    per species. Relative transition levels WITHIN one species are
    literature-accurate; ABSOLUTE formation energies are NOT comparable
    ACROSS species, and predicted absolute concentrations are illustrative
    only. Real cross-species numbers require actual DFT total energies
    (e.g. Table I/II of the source paper), which are not incorporated here.

  * AlN-endpoint E_bare values have no literature source in this project —
    they default to a crude bandgap-ratio scaling
    (E_bare_AlN = E_bare_GaN * Eg_AlN/Eg_GaN) unless explicitly overridden.
    This is the single weakest-justified approximation in this module,
    mirroring how the source paper itself treats AlN ("as a first
    approximation... by interpolating between AlN and GaN").

  * The DX-center parameters (x_c, transition depths) are literature-
    informed calibration knobs, not first-principles results. Oxygen's
    x_c = 0.30 is a real reported threshold (Van de Walle & Neugebauer,
    Sec. IV A 1). Silicon's x_c is disputed in the literature (reported
    anywhere from 0.24 to 0.6 depending on the study); the default here is
    a documented, user-adjustable midpoint guess, not a consensus value.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --- Alloy heat of formation endpoints [eV / formula unit], used by
#     chemical_potentials.py to bound the Ga/Al/N reservoirs.
DHF_GAN_EV = -1.15
DHF_ALN_EV = -3.05


@dataclass
class DefectChargeState:
    """One charge state of a native point defect or substitutional dopant."""
    q: int                                     # signed charge state
    E_bare_GaN_eV: float                       # calibrated intercept at x=0
    E_bare_AlN_eV: Optional[float] = None      # None -> bandgap-ratio fallback
    stoichiometry: Dict[str, int] = field(default_factory=dict)
    branch: str = "normal"                     # "normal" | "DX"


@dataclass
class DXCenterParams:
    """Phenomenological two-branch DX-center model parameters for a donor."""
    x_c: float                                 # calibrated crossover composition
    x_c_lit_range: Tuple[float, float]         # documented literature spread
    E_DX_shallow_below_Ec_eV: float            # target level depth, x << x_c
    E_DX_deep_below_Ec_eV: float               # target level depth, x >> x_c (calibration constant)
    transition_width_x: float = 0.03           # logistic blend width in x-space


@dataclass
class DefectSpecies:
    """A defect/dopant species: its charge states, plus an optional DX model."""
    name: str
    charge_states: List[DefectChargeState]
    e_bare_scaling: str = "bandgap_ratio"      # "bandgap_ratio" | "fixed"
    dx_params: Optional[DXCenterParams] = None
    # Fixed chemical-potential offset [eV] for impurity elements (O, Si, ...)
    # that are not part of the Ga/Al/N alloy equilibrium. Default 0.0 means
    # "reservoir at its own reference state" -- a simplification, since we
    # have no oxide/nitride secondary-phase equilibrium data for O or Si.
    impurity_mu_eV: Dict[str, float] = field(default_factory=dict)
    # Uniform absolute shift [eV] applied to EVERY charge state of this
    # species (does not change relative transition levels -- only moves
    # the whole species' zero point). Needed because each species'
    # E_bare_GaN_eV values above are anchored independently (see module
    # docstring): without this, cross-species "which defect dominates"
    # comparisons are meaningless. Default 0.0 (no correction available).
    # Set for V_GA/V_N below from an approximate visual read of Fig. 5 of
    # Van de Walle & Neugebauer (2004) (Ga-rich GaN, E_F=0 at the VBM) --
    # this is an order-of-magnitude calibration, NOT a precise tabulated
    # DFT value, since the paper's actual numeric table was not available.
    # O_N/Si_Ga have no such reference and are left at 0.0 -- their
    # absolute scale remains uncalibrated; if either turns out to be the
    # "binding" (worst-case) species in a cross-species comparison, treat
    # that specific conclusion as provisional.
    absolute_offset_eV: float = 0.0


def _solve_two_state_E_bare(q_anchor: int, q_other: int, transition_level_eV: float) -> float:
    """
    Given one charge state anchored at E_bare=0 and a literature transition
    level (energy above local VBM at which the two states become
    degenerate), solve the other state's E_bare:

        E_bare(q_anchor) + q_anchor*eps == E_bare(q_other) + q_other*eps
        E_bare(q_other) = (q_anchor - q_other) * eps
    """
    return (q_anchor - q_other) * transition_level_eV


# ---------------------------------------------------------------------------
# Nitrogen vacancy (donor). (3+/+) transition at 0.59 eV above VBM
# (GaN, Ga-rich conditions; Van de Walle & Neugebauer 2004, Sec. III B).
# ---------------------------------------------------------------------------
_VN_TRANSITION_EV = 0.59
# absolute_offset_eV chosen so the TOTAL formation energy (including the
# Ga-rich/metal_rich chemical-potential term, since that is the condition
# Fig. 5 itself was computed under) of the dominant (3+) branch reads ~0 eV
# at E_F=VBM -- an approximate visual read of Fig. 5 (see DefectSpecies
# docstring above for the honesty caveat on this number). Without the mu
# term folded in, the required offset would be 1.18 eV; the extra +1.15 eV
# cancels the metal_rich dmu_N contribution at x=0 so the calibration point
# lands where Fig. 5 actually shows it.
V_N = DefectSpecies(
    name="V_N",
    charge_states=[
        DefectChargeState(q=1, E_bare_GaN_eV=0.0, stoichiometry={"N": -1}),
        DefectChargeState(
            q=3,
            E_bare_GaN_eV=_solve_two_state_E_bare(1, 3, _VN_TRANSITION_EV),
            stoichiometry={"N": -1},
        ),
    ],
    absolute_offset_eV=2.33,
)

# ---------------------------------------------------------------------------
# Gallium vacancy (triple acceptor). (2-/3-) transition at 1.1 eV above VBM
# (GaN; Van de Walle & Neugebauer 2004, Sec. III C).
# ---------------------------------------------------------------------------
_VGA_TRANSITION_EV = 1.1
# absolute_offset_eV chosen so the dominant (2-) branch reads ~2 eV at
# E_F=VBM, Ga-rich -- an approximate visual read of Fig. 5 (see DefectSpecies
# docstring above for the honesty caveat on this number).
V_GA = DefectSpecies(
    name="V_Ga",
    charge_states=[
        DefectChargeState(q=-2, E_bare_GaN_eV=0.0, stoichiometry={"Ga": -1}),
        DefectChargeState(
            q=-3,
            E_bare_GaN_eV=_solve_two_state_E_bare(-2, -3, _VGA_TRANSITION_EV),
            stoichiometry={"Ga": -1},
        ),
    ],
    absolute_offset_eV=2.0,
)

# ---------------------------------------------------------------------------
# Oxygen on the N site. Shallow donor in GaN; well-documented DX transition
# above x_Al ~ 0.30 (Van de Walle & Neugebauer 2004, Sec. IV A 1).
# The deep-level depth below Ec (0.40 eV) is an illustrative calibration
# constant -- no first-principles value was available for this project.
# ---------------------------------------------------------------------------
O_N = DefectSpecies(
    name="O_N",
    charge_states=[
        DefectChargeState(q=1, E_bare_GaN_eV=0.0,
                           stoichiometry={"N": -1, "O": 1}, branch="normal"),
    ],
    dx_params=DXCenterParams(
        x_c=0.30,
        x_c_lit_range=(0.30, 0.30),
        E_DX_shallow_below_Ec_eV=0.0,
        E_DX_deep_below_Ec_eV=0.40,
        transition_width_x=0.03,
    ),
)

# ---------------------------------------------------------------------------
# Silicon on the Ga site. Confirmed shallow donor up to x_Al=0.44
# experimentally; DX-like transition reported anywhere between x=0.24 and
# x=0.6 depending on the study (Park & Chadi vs. Boguslawski & Bernholc vs.
# Van de Walle's own conservative assessment). x_c=0.45 and the shallower
# 0.15 eV deep-trap depth (vs. oxygen's 0.40 eV) are both illustrative,
# user-adjustable calibration choices reflecting Si's reportedly milder
# DX behavior -- not a consensus literature value.
# ---------------------------------------------------------------------------
SI_GA = DefectSpecies(
    name="Si_Ga",
    charge_states=[
        DefectChargeState(q=1, E_bare_GaN_eV=0.0,
                           stoichiometry={"Ga": -1, "Si": 1}, branch="normal"),
    ],
    dx_params=DXCenterParams(
        x_c=0.45,
        x_c_lit_range=(0.24, 0.60),
        E_DX_shallow_below_Ec_eV=0.0,
        E_DX_deep_below_Ec_eV=0.15,
        transition_width_x=0.05,
    ),
)

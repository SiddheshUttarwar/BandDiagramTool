"""
Phenomenological DX-center model for donors (oxygen, silicon) in graded
Al_xGa_(1-x)N.

This is NOT a first-principles (lattice-relaxation / negative-U DFT)
calculation -- that is out of scope for this project. Instead, a donor is
modeled with two competing formation-energy branches:

  * "normal" (shallow) branch: ordinary ionized donor, charge q=+1,
        E_shallow(x,z) = E_bare_shallow(x) + E_F_local(z)
  * "DX" (deep, negative-U) branch: the donor captures a second electron
    and relaxes into a deep, negatively charged state, skipping the
    neutral charge state, charge q=-1,
        E_DX(x,z) = E_bare_DX_minus(x) - E_F_local(z)

The physically meaningful, composition-dependent knob is the TARGET
transition level (where the two branches would cross), expressed as a
depth below the local conduction band minimum. That depth is smoothly
blended, as a function of Al content x, from a "shallow" value (near Ec,
x << x_c) to a "deep" value (calibration constant, x >> x_c) using a
logistic function -- the same blending pattern used for the (unrelated)
Mott-transition model in
TestPrograms/UnderstandingAlGaNDoping/AlGaN_Mott_transition_correction.py,
here applied to the INPUT target level rather than the output energy, so
that the actual normal-vs-DX crossover in z falls out of ordinary
convex-hull (minimum-energy) comparison downstream in formation_energy.py,
rather than being double-smoothed.

Calibration note: E_DX_deep_below_Ec_eV and x_c are literature-informed
but not first-principles values (see physics/materials/defect_constants.py
docstring for the honest accounting of what is and isn't literature-exact).
"""

import numpy as np

from physics.materials.defect_constants import DXCenterParams


def dx_blend_weight(x_Al, x_c: float, width: float):
    """Logistic blend weight, 0 far below x_c, 1 far above x_c."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(x_Al, dtype=float) - x_c) / width))


def dx_deep_level_below_Ec_eV(x_Al, dx_params: DXCenterParams):
    """
    Composition-blended target DX transition-level depth below Ec [eV].
    Approaches E_DX_shallow_below_Ec_eV for x << x_c and
    E_DX_deep_below_Ec_eV for x >> x_c.
    """
    w = dx_blend_weight(x_Al, dx_params.x_c, dx_params.transition_width_x)
    return (1.0 - w) * dx_params.E_DX_shallow_below_Ec_eV + w * dx_params.E_DX_deep_below_Ec_eV


def dx_branch_E_bare_minus(x_Al, Eg_x, E_bare_shallow_plus_x, dx_params: DXCenterParams):
    """
    Solve the DX (q=-1) branch's E_bare(x) so that the shallow (q=+1) and
    DX (q=-1) formation-energy lines cross exactly at
    E_F_local = Eg(x) - depth(x), i.e. at the blended target level measured
    from the local VBM:

        E_bare_shallow(x) + target = E_bare_DX_minus(x) - target
        E_bare_DX_minus(x) = E_bare_shallow(x) + 2*target
    """
    depth = dx_deep_level_below_Ec_eV(x_Al, dx_params)
    target_level_above_VBM = np.asarray(Eg_x, dtype=float) - depth
    return np.asarray(E_bare_shallow_plus_x, dtype=float) + 2.0 * target_level_above_VBM

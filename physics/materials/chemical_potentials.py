"""
Chemical potentials for the Ga/Al/N reservoirs in Al_xGa_(1-x)N growth,
used to evaluate the "sum_i n_i*mu_i" term in defect formation energies.

Convention: mu_i = mu_i^0 (bulk metal / half an N2 molecule) + dmu_i, with
mu_i^0 == 0 eV as the reference. Only the offsets dmu_i (<= 0) are ever
computed or used here.

Growth-condition boundary construction (standard two-limit approach,
Van de Walle & Neugebauer 2004, Sec. II E):

    metal_rich: dmu_Ga = dmu_Al = 0,           dmu_N = dHf_alloy(x)
    N_rich:     dmu_N = 0,                     dmu_Ga = dmu_Al = dHf_alloy(x)

where dHf_alloy(x) is the (negative) alloy heat of formation per formula
unit, linearly interpolated between the GaN and AlN endpoints.

APPROXIMATION: both cations (Ga, Al) are assumed to share one reservoir
offset at a given growth condition -- there is no explicit modeling of
AlN/GaN phase competition or partial segregation at intermediate x. This
is a simplification, not a rigorous ternary phase-boundary construction.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

from physics.materials.defect_constants import DHF_GAN_EV, DHF_ALN_EV

ArrayOrFloat = Union[float, np.ndarray]


@dataclass
class ChemPotResult:
    """Chemical-potential offsets [eV]. Fields hold scalars or arrays,
    depending on whether get_mu_offsets() or get_mu_offsets_profile()
    produced them."""
    x: ArrayOrFloat
    condition: str
    dmu_Ga: ArrayOrFloat
    dmu_Al: ArrayOrFloat
    dmu_N: ArrayOrFloat


def alloy_heat_of_formation_eV(x: ArrayOrFloat) -> ArrayOrFloat:
    """Linearly interpolated alloy heat of formation dHf_alloy(x) [eV/formula unit]."""
    return x * DHF_ALN_EV + (1.0 - x) * DHF_GAN_EV


def _boundary_offsets(x: ArrayOrFloat, condition: str, zeros):
    dHf = alloy_heat_of_formation_eV(x)
    if condition == "metal_rich":
        return zeros, zeros, dHf              # dmu_Ga, dmu_Al, dmu_N
    elif condition == "N_rich":
        return dHf, dHf, zeros
    raise ValueError(f"Unknown condition '{condition}'. Use 'metal_rich' or 'N_rich'.")


def get_mu_offsets(x: float, condition: str = "N_rich") -> ChemPotResult:
    """
    Chemical-potential offsets [eV] for Al_xGa_(1-x)N at one boundary of the
    thermodynamically allowed growth window.

    condition : "metal_rich" | "N_rich"
    """
    dmu_Ga, dmu_Al, dmu_N = _boundary_offsets(float(x), condition, 0.0)
    return ChemPotResult(x=x, condition=condition, dmu_Ga=dmu_Ga, dmu_Al=dmu_Al, dmu_N=dmu_N)


def get_mu_offsets_profile(x_Al: np.ndarray, condition: str = "N_rich") -> ChemPotResult:
    """Vectorised version of get_mu_offsets() for an array of compositions."""
    x_Al = np.asarray(x_Al, dtype=float)
    dmu_Ga, dmu_Al, dmu_N = _boundary_offsets(x_Al, condition, np.zeros_like(x_Al))
    return ChemPotResult(x=x_Al, condition=condition, dmu_Ga=dmu_Ga, dmu_Al=dmu_Al, dmu_N=dmu_N)

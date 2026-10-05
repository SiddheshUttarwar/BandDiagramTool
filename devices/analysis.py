"""
Post-processing helpers that work on a solved device.

sheet_density            integrated carrier density in a depth window
extract_surface_barrier  the surface barrier height that reproduces a
                         measured sheet density (inverse problem)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from devices.device import AlGaNDevice
from devices.layer import Contact


def sheet_density(result, z0_nm: float, z1_nm: float, carrier: str = 'n') -> float:
    """Sheet density [cm^-2] of electrons ('n') or holes ('p') between two
    depths [nm, measured from the bottom of the stack]."""
    x = np.asarray(result.x_nm)
    c = np.asarray(result.n if carrier == 'n' else result.p)
    m = (x >= z0_nm) & (x <= z1_nm)
    return float(np.trapezoid(c[m], x[m] * 1e-7))


def extract_surface_barrier(
    layers: List,
    n_s_measured: float,
    window_nm: Tuple[float, float],
    carrier: str = 'n',
    bottom_contact: Optional[Contact] = None,
    bounds_eV: Tuple[float, float] = (0.05, 5.5),
    T: float = 300.0,
    dx_nm: float = 0.2,
    quantum: bool = False,
    xtol_eV: float = 5e-3,
    **device_kwargs,
) -> float:
    """
    Surface barrier height [eV] (Ec - Ef at the top surface) for which the
    computed sheet density in `window_nm` equals `n_s_measured` [cm^-2].

    Why this exists: the sheet density of a polarization-induced 2DEG under a
    thin barrier of thickness d depends on the surface barrier Phi_B as
    d n_s / d Phi_B = -eps0 eps_r / (q^2 d), about -2.4e13 cm^-2 per eV for
    2 nm of AlN. Phi_B of a free nitride surface is not known to better than
    several tenths of an eV and is not constant: Li et al., APL 97, 222110
    (2010) find that a fixed Phi_B cannot describe Al0.72Ga0.28N/GaN over a
    range of barrier thicknesses, the extracted Phi_B falling from 3.5 to
    2.2 eV as the 2DEG density rises. For thin barriers the meaningful
    comparison with experiment is therefore this inverse problem.

    Raises ValueError if no barrier inside `bounds_eV` reproduces the target
    (for example a measured density above the polarization charge).
    """
    bottom = bottom_contact or Contact('bottom', 'ohmic', 'Ti')

    def mismatch(phi_b: float) -> float:
        top = Contact('top', 'schottky', 'Ni', barrier_eV=float(phi_b))
        dev = AlGaNDevice(layers, [bottom, top], T=T, dx_nm=dx_nm, **device_kwargs)
        r = dev.solve(V_applied=0.0, quantum=quantum)
        if not r.converged:
            raise RuntimeError(f'solve did not converge at barrier {phi_b:.3f} eV')
        return sheet_density(r, window_nm[0], window_nm[1], carrier) - n_s_measured

    lo, hi = bounds_eV
    # A very low barrier on a strongly polarized stack can fail to converge;
    # raise the lower bound until it solves.
    f_lo = None
    while f_lo is None:
        try:
            f_lo = mismatch(lo)
        except RuntimeError:
            lo += 0.5
            if lo >= hi:
                raise
    f_hi = mismatch(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f'measured density {n_s_measured:.3g} cm^-2 is outside the range reachable with a '
            f'barrier of {lo}-{hi} eV ({f_lo + n_s_measured:.3g} to {f_hi + n_s_measured:.3g} cm^-2)')
    return float(brentq(mismatch, lo, hi, xtol=xtol_eV))

"""
Optical gain spectrum for a quantum-confined active region.

Standard 2D quantum-well gain model -- parabolic subbands, a k-independent
bulk (k.p) momentum matrix element, quasi-Fermi occupation factors evaluated
at each subband's own band edge, and Lorentzian homogeneous broadening
across every (electron subband, hole subband) transition. This is the
textbook band-edge (k=0) approximation (e.g. Coldren, Corzine & Masanovic,
"Diode Lasers and Photonic Integrated Circuits", 2nd ed., Ch. 4; Chuang,
"Physics of Photonic Devices", Ch. 9): it reproduces the qualitatively
correct spectrum shape (peak position tracking the confined transition
energies, the expected sign flip from absorption to gain as injection
increases, roughly correct linewidth) without a full k-resolved in-plane
dispersion/joint-DOS integral, which would need subband curvature this
solver doesn't track.

    g(hv) = sum over (electron subband i, hole subband j, hole band b in
            {HH, LH, SO}) of:

        g0_ij(hv) = (pi q^2 hbar) / (n_r eps0 c m0^2 * E_ij_J)
                    * |M_T|^2 * |overlap_ij|^2
                    * m_r_ij / (pi hbar^2 Lz)
                    * [f_c(E_e,i) - f_v(E_h,j)]
                    * L(hv - E_ij, Gamma)

    |M_T|^2 = (m0/6) * Ep         -- bulk momentum matrix element, Ep =
                                      Kane energy (literature-typical for
                                      III-nitrides, NOT fit to any specific
                                      device -- see kane_energy_eV)
    m_r_ij  = 1 / (1/m_e,i + 1/m_h,j)   subband-averaged reduced mass [m0]
    f_c(E)  = 1 / (1 + exp((E-Efn)/kT))   electron occupation, conduction
    f_v(E)  = 1 / (1 + exp((E-Efp)/kT))   electron occupation, valence
              (near 1 = state filled with an electron = no hole there)
    L(x, Gamma) = (Gamma/pi) / (x^2 + Gamma^2)   normalised Lorentzian

The 2D density-of-states prefactor is m_r/(pi hbar^2 Lz) -- NOT 2*m_r/(...)
-- because m_r/(pi hbar^2) is already the *spin-degenerate* (both spins
included) 2D areal density of states; an earlier version of this module
multiplied by an extra factor of 2 here, double-counting spin. Fixed after
cross-checking this module's overall prefactor structure (matrix element +
Kane energy convention) against a case with an independent, well-documented
answer: bulk GaN's near-band-edge absorption coefficient (~1e3-1e4 cm^-1
within a few kT of the gap) via the same |M_T|^2/Ep convention with the 3D
joint density of states in place of the 2D one above -- that reproduced the
expected experimental range, giving confidence in the matrix-element/Kane-
energy part specifically.

CAVEAT: absolute gain magnitude from this module should be read as
order-of-magnitude / qualitative, not a validated, metrology-grade
prediction -- the bulk cross-check above validates the matrix-element
convention, but not this specific 2D extension for a physically extreme
(sub-2nm) quantum well, where confinement-enhanced density of states can
legitimately push material gain far above typical (2-5nm well) literature
values, and this hasn't been independently cross-checked against a trusted
QW-specific reference. The spectral *shape* (peak position tracking the
confined transition energies, sign flip from absorption to gain as
injection increases, relative broadening) is on much firmer footing than
the absolute cm^-1 scale.

All physics-formula energies are converted to Joules internally for SI
consistency; only the returned spectrum's axes are in eV/nm for plotting.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from physics.constants import q as _q, kB, eps0, hbar, m0, c as _c
from physics.optical import hole_subband_energy, overlap_squared

# Kane energy [eV] endpoints, literature-typical range for III-nitrides
# (comparable order of magnitude to values reported across Vurgaftman &
# Meyer-style band-parameter compilations for GaN/AlN) -- linearly
# interpolated by composition. Not fit to any specific device or paper:
# absolute gain magnitude from this module should be read as
# order-of-magnitude/qualitative, not a metrology-grade prediction.
_EP_GAN_EV = 14.0
_EP_ALN_EV = 18.0

DEFAULT_LINEWIDTH_EV = 0.010   # 10 meV homogeneous broadening, typical QW value
DEFAULT_N_R = 2.5              # refractive index, typical AlGaN in the near-UV


def kane_energy_eV(x_Al: float) -> float:
    x = float(np.clip(x_Al, 0.0, 1.0))
    return _EP_GAN_EV * (1.0 - x) + _EP_ALN_EV * x


def _fermi(E_eV: float, Ef_eV: float, kT_eV: float) -> float:
    return 1.0 / (1.0 + np.exp(np.clip((E_eV - Ef_eV) / kT_eV, -200.0, 200.0)))


@dataclass
class GainSpectrum:
    energy_eV: np.ndarray
    wavelength_nm: np.ndarray
    gain_cm1: np.ndarray
    peak_gain_cm1: float
    peak_wavelength_nm: float
    n_transitions: int   # how many (subband, hole-band) pairs contributed


def compute_gain_spectrum(
    E_e: Optional[np.ndarray], psi_e: Optional[np.ndarray], m_e_conf: np.ndarray,
    hole_bands: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    m_hole_conf: Dict[str, np.ndarray],
    Efn: float, Efp: float, T: float,
    dx_cell: np.ndarray, Lz_m: float, x_Al_well: float,
    n_r: float = DEFAULT_N_R,
    linewidth_eV: float = DEFAULT_LINEWIDTH_EV,
    energy_range_eV: Optional[Tuple[float, float]] = None,
    n_points: int = 400,
    Ep_eV: Optional[float] = None,
) -> Optional[GainSpectrum]:
    """
    hole_bands / m_hole_conf: {'hh': (E_h, psi_h), 'lh': ..., 'so': ...} and
    {'hh': m_array, 'lh': ..., 'so': ...} -- as built by
    physics.self_consistent._solve_hole_bands and grid.m_hh/m_lh/m_so.
    Efn/Efp: quasi-Fermi levels [eV], same reference frame as E_e/hole
    subband energies. Lz_m: quantum well width [m] (the 2D-DOS-per-volume
    prefactor's length scale -- use the well's own physical thickness).
    x_Al_well: representative Al composition of the well, for the Kane
    energy estimate (nitrides); Ep_eV, when given, is used instead.

    Returns None if there are no confined states to build a spectrum from.
    """
    if E_e is None or len(E_e) == 0:
        return None
    kT_eV = kB * T / _q
    if Ep_eV is None:
        Ep_eV = kane_energy_eV(x_Al_well)
    M_T2 = (m0 / 6.0) * (Ep_eV * _q)   # kg * J

    transitions = []  # (E_ij_eV, overlap, m_r_kg)
    f_pairs = []       # (f_c, f_v) per transition, same order

    for band_name, (E_h, psi_h) in hole_bands.items():
        if E_h is None or psi_h is None or len(E_h) == 0:
            continue
        m_conf = m_hole_conf.get(band_name)
        if m_conf is None:
            continue
        Ev_sub = hole_subband_energy(E_h)
        for ie in range(len(E_e)):
            for ih in range(len(E_h)):
                overlap = overlap_squared(psi_e[ie], psi_h[ih], dx_cell)
                if overlap < 1e-6:
                    continue
                E_ij = float(E_e[ie]) - float(Ev_sub[ih])
                if E_ij <= 0.0:
                    continue
                w_e = psi_e[ie] ** 2 * dx_cell
                m_e_i = float(np.sum(w_e * m_e_conf) / max(np.sum(w_e), 1e-30))
                w_h = psi_h[ih] ** 2 * dx_cell
                m_h_j = float(np.sum(w_h * m_conf) / max(np.sum(w_h), 1e-30))
                m_r = 1.0 / (1.0 / max(m_e_i, 1e-6) + 1.0 / max(m_h_j, 1e-6))
                f_c = _fermi(float(E_e[ie]), Efn, kT_eV)
                f_v = _fermi(float(Ev_sub[ih]), Efp, kT_eV)
                transitions.append((E_ij, overlap, m_r * m0))
                f_pairs.append((f_c, f_v))

    if not transitions:
        return None

    if energy_range_eV is None:
        Es = [t[0] for t in transitions]
        lo = max(0.01, min(Es) - 15.0 * linewidth_eV)
        hi = max(Es) + 15.0 * linewidth_eV
    else:
        lo, hi = energy_range_eV

    hv_eV = np.linspace(lo, hi, n_points)
    hv_J = hv_eV * _q
    gain_m1 = np.zeros_like(hv_eV)

    for (E_ij_eV, overlap, m_r_kg), (f_c, f_v) in zip(transitions, f_pairs):
        E_ij_J = E_ij_eV * _q
        prefactor = (
            (np.pi * _q ** 2 * hbar) / (n_r * eps0 * _c * m0 ** 2 * E_ij_J)
            * M_T2 * overlap
            * m_r_kg / (np.pi * hbar ** 2 * Lz_m)
            * (f_c - f_v)
        )
        lorentzian = (linewidth_eV / np.pi) / ((hv_eV - E_ij_eV) ** 2 + linewidth_eV ** 2)
        # lorentzian is normalised in eV^-1; convert hv_J's own implicit
        # eV-vs-J bookkeeping by evaluating the whole prefactor at E_ij_J
        # (energies) while broadening stays in eV (axis units) -- consistent
        # since L(x,Gamma) integrates to 1 over the eV axis either way.
        gain_m1 += prefactor * lorentzian

    gain_cm1 = gain_m1 * 1e-2   # m^-1 -> cm^-1
    wavelength_nm = 1239.8419 / np.maximum(hv_eV, 1e-6)
    i_peak = int(np.argmax(gain_cm1))

    return GainSpectrum(
        energy_eV=hv_eV, wavelength_nm=wavelength_nm, gain_cm1=gain_cm1,
        peak_gain_cm1=float(gain_cm1[i_peak]),
        peak_wavelength_nm=float(wavelength_nm[i_peak]),
        n_transitions=len(transitions),
    )

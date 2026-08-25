"""
Quantum-Confined Stark Effect (QCSE) diagnostics.

The self-consistent Schrodinger-Poisson solver already models the QCSE
*physically*: the built-in polarization field (plus any applied bias) tilts
the quantum-well potential, the Schrodinger solve sees that tilted well
directly, and the resulting electron/hole ground states separate toward
opposite sides of the well. This module turns those wavefunctions into the
two standard experimental signatures of QCSE:

  - the interband transition energy, which redshifts as the internal field
    grows (the "Stark shift")
  - the electron-hole wavefunction overlap, which falls as the electron and
    hole separate spatially -- the direct proxy for the reduction in
    oscillator strength / radiative recombination rate that accompanies the
    redshift.

Reference: Miller et al., "Band-edge electroabsorption in quantum well
structures: The quantum-confined Stark effect", PRL 53, 2173 (1984).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Transition:
    ie: int          # electron subband index
    ih: int          # hole subband index
    energy_eV: float  # e-h transition (photon) energy [eV]
    overlap: float    # |integral psi_e * psi_h dx|^2, in [0, 1]


def hole_subband_energy(E_h: np.ndarray) -> np.ndarray:
    """
    Convert Schrodinger-solver hole eigenvalues to the absolute energy
    scale shared with Ec/Ev/E_e.

    physics.schrodinger.hole_potential() flips the sign of the valence
    band (V_hole = -Ev) so that the solver's lowest eigenvalue is the
    *highest* (least negative) hole energy. E_h comes back positive; the
    real subband energy on the same scale as Ec/Ev/E_e is -E_h.
    """
    return -np.asarray(E_h, dtype=float)


def overlap_squared(psi_e: np.ndarray, psi_h: np.ndarray, dx) -> float:
    """
    |integral psi_e(x) psi_h(x) dx|^2 for one electron/hole state pair.

    dx : scalar (uniform grid) or a length-N array of per-node quadrature
    weights (a non-uniform grid's finite-volume cell widths -- see
    physics.grid_utils.node_spacings; NOT the length-(N-1) per-edge
    spacing array the stencil solvers use). Weighting per-point *before*
    summing is what makes this correct for a non-uniform grid; scaling the
    unweighted sum by a scalar dx afterwards (equivalent only when dx is
    constant) is the uniform-grid special case of the same expression.

    Both wavefunctions are dx-normalised (integral |psi|^2 dx = 1, see
    physics.schrodinger.solve_schrodinger), so by Cauchy-Schwarz this is
    dimensionless and bounded in [0, 1]: 1 for perfectly coincident
    envelopes (flat-band limit), falling toward 0 as an internal field
    pulls the electron and hole to opposite sides of the well.
    """
    integral = np.sum(psi_e * psi_h * dx)
    return float(integral * integral)


def transition_matrix(
    E_e: Optional[np.ndarray], psi_e: Optional[np.ndarray],
    E_h: Optional[np.ndarray], psi_h: Optional[np.ndarray],
    dx: float,
    n_e: Optional[int] = None, n_h: Optional[int] = None,
) -> List[Transition]:
    """
    All (electron subband, hole subband) transition energies and overlaps,
    for the lowest `n_e` electron and `n_h` hole subbands (default: every
    solved state).
    """
    if E_e is None or E_h is None or len(E_e) == 0 or len(E_h) == 0:
        return []
    n_e = len(E_e) if n_e is None else min(n_e, len(E_e))
    n_h = len(E_h) if n_h is None else min(n_h, len(E_h))
    Ev_sub = hole_subband_energy(E_h)

    out: List[Transition] = []
    for ie in range(n_e):
        for ih in range(n_h):
            overlap = overlap_squared(psi_e[ie], psi_h[ih], dx)
            energy = float(E_e[ie]) - float(Ev_sub[ih])
            out.append(Transition(ie=ie, ih=ih, energy_eV=energy, overlap=overlap))
    return out


def ground_state_transition(
    E_e: Optional[np.ndarray], psi_e: Optional[np.ndarray],
    E_h: Optional[np.ndarray], psi_h: Optional[np.ndarray],
    dx: float,
) -> Optional[Transition]:
    """The e1-h1 transition: the headline QCSE number (redshift + overlap)."""
    matrix = transition_matrix(E_e, psi_e, E_h, psi_h, dx, n_e=1, n_h=1)
    return matrix[0] if matrix else None


def dominant_transition(
    E_e: Optional[np.ndarray], psi_e: Optional[np.ndarray],
    E_h: Optional[np.ndarray], psi_h: Optional[np.ndarray],
    dx: float, n_e: int = 4, n_h: int = 4,
) -> Optional[Transition]:
    """
    The highest-overlap transition among the lowest few subbands. Under
    strong field tilt the nominal e1-h1 pair can lose so much overlap that
    a higher-index pair actually carries more oscillator strength; this is
    the pair most likely to dominate emission/absorption in that regime.
    """
    matrix = transition_matrix(E_e, psi_e, E_h, psi_h, dx, n_e=n_e, n_h=n_h)
    if not matrix:
        return None
    return max(matrix, key=lambda t: t.overlap)

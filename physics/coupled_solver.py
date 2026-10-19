"""
Shared types of the biased drift-diffusion solve.

This module used to hold the first fully-coupled (phi, Efn, Efp) Newton
solver: a pseudo-transient-continuation direct solve (_solve_ptc_direct),
preceded by an adaptive Gummel stage, with a finite-difference block-
tridiagonal Jacobian. It was replaced by the analytic-Jacobian Newton
solver in physics.dd_newton (see that module's docstring for why) and is no
longer called; it remains in the git history. An earlier version of it
tried Jacobian-Free Newton-Krylov first and was found to converge to
spurious fixed points from carried-forward continuation states -- see
[[bug-quasi-fermi-pinning]].

What is left are the types other modules still import: SolveCancelled
(raised by the solvers, caught by the GUI worker), and the QuantumState /
CoupledResult containers kept for existing imports.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


class SolveCancelled(Exception):
    """Raised (from physics.dd_newton's Newton loop, or the outer loop in
    physics.self_consistent) to unwind a solve early when the user clicks
    Stop -- distinct from a genuine solver failure."""


@dataclass
class QuantumState:
    """Frozen (not re-solved) Schrodinger states, held fixed for one coupled
    solve -- physics.self_consistent's existing pattern of solving
    Schrodinger with the band profile from the *previous* outer iteration.

    Holes are the decoupled 3-band effective-mass model (HH/LH/SO, see
    physics.materials.algan.AlGaNParams.valence_band_structure): three
    independent single-band confined-state solutions, whose quantum
    densities are simply summed (no HH-LH-SO mixing)."""
    psi_e: np.ndarray
    E_e: np.ndarray
    m_e: np.ndarray
    psi_h: np.ndarray
    E_h: np.ndarray
    m_hh: np.ndarray
    psi_h_lh: np.ndarray
    E_h_lh: np.ndarray
    m_lh: np.ndarray
    psi_h_so: np.ndarray
    E_h_so: np.ndarray
    m_so: np.ndarray
    region_mask: np.ndarray


@dataclass
class CoupledResult:
    phi: np.ndarray
    Efn: np.ndarray
    Efp: np.ndarray
    converged: bool
    n_iter: int
    final_residual: float

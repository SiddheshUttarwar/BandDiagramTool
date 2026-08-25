"""
Shared helpers for the 1D finite-difference/finite-volume grid used by
poisson.py, schrodinger.py, and drift_diffusion.py.

Every one of those solvers accepts its grid-spacing argument (conventionally
named `dx`) as either:

  - a scalar float: a uniform grid (dx is the same everywhere). This is the
    original, pre-non-uniform-grid behaviour and every formula below reduces
    exactly to the solver's original uniform-grid formula in this case.

  - a 1D array of per-edge spacings, length N-1, where edges[i] = x[i+1] -
    x[i]: a non-uniform grid (see devices.grid_builder's per-layer dx_nm),
    using the finite-volume generalisation of the same formula.

This module derives the per-node quantities every non-uniform stencil needs
from that single `dx` argument: the left/right spacing at each interior
node, and each node's finite-volume "cell width" (the quadrature weight
used for normalisation/integration, e.g. physics.schrodinger's
integral|psi|^2 dx = 1 and physics.optical's overlap integral).
"""

from __future__ import annotations

import numpy as np


def node_spacings(dx, N: int):
    """
    Parameters
    ----------
    dx : float or array-like of length N-1
    N  : number of grid points

    Returns
    -------
    h1, h2     : left/right spacing at each interior node i=1..N-2 [m],
                 each length N-2 (h1[k] = edges[k], h2[k] = edges[k+1] for
                 interior node i=k+1)
    cell_width : finite-volume cell width at every node, length N.
                 Interior: 0.5*(h1+h2) (the standard finite-volume cell).
                 Boundary (i=0, i=N-1): the single adjacent edge, unhalved
                 -- this is what makes the uniform-dx case reduce exactly
                 to the original "every point gets weight dx" convention
                 used throughout this codebase before non-uniform grids
                 existed (e.g. physics.schrodinger's old `sum(psi**2)*dx`).
    edges      : per-edge spacing, length N-1 (dx broadcast to an array if
                 it was passed as a scalar).
    """
    if np.ndim(dx) == 0:
        edges = np.full(N - 1, float(dx))
    else:
        edges = np.asarray(dx, dtype=float)
        if len(edges) != N - 1:
            raise ValueError(f"dx array must have length N-1={N - 1}, got {len(edges)}")

    h1 = edges[:-1]   # length N-2: left spacing of interior nodes 1..N-2
    h2 = edges[1:]    # length N-2: right spacing of interior nodes 1..N-2

    cell_width = np.empty(N)
    cell_width[1:-1] = 0.5 * (h1 + h2)
    cell_width[0] = edges[0]
    cell_width[-1] = edges[-1]

    return h1, h2, cell_width, edges


def central_difference(f: np.ndarray, dx) -> np.ndarray:
    """
    df/dx via a 3-point central difference, generalised to a non-uniform
    grid (dx: scalar or length-(N-1) per-edge array; see node_spacings).

    Interior points use the standard non-uniform 3-point formula
        f'(x_i) ~= [h1^2 f_{i+1} - h2^2 f_{i-1} - (h1^2-h2^2) f_i] / (h1 h2 (h1+h2))
    (h1 = left spacing, h2 = right spacing), which reduces exactly to
    (f[2:]-f[:-2])/(2*dx) when h1=h2=dx. Boundary points use their single
    adjacent edge, as in every other stencil in this codebase.

    Used by physics.poisson.electric_field (E = -dphi/dx) and
    physics.polarization's charge/quasi-field derivatives (both are plain
    df/dx of a scalar profile).
    """
    N = len(f)
    _h1, _h2, _cw, edges = node_spacings(dx, N)
    h1 = edges[:-1]   # length N-2: left spacing of interior nodes
    h2 = edges[1:]    # length N-2: right spacing of interior nodes

    df = np.empty_like(f)
    df[1:-1] = (h1**2 * f[2:] - h2**2 * f[:-2] - (h1**2 - h2**2) * f[1:-1]) \
               / (h1 * h2 * (h1 + h2))
    df[0]  = (f[1]  - f[0])  / edges[0]
    df[-1] = (f[-1] - f[-2]) / edges[-1]
    return df

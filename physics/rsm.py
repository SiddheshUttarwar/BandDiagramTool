"""
X-ray reciprocal space map of a solved structure, in the kinematical
approximation.

The strain profile fixes the lattice of every grid cell,

    a(z) = a0(x) [1 + eps_xx(z)]        in-plane
    c(z) = c0(x) [1 + eps_zz(z)]        along the growth axis

with a0, c0 the relaxed constants of the local alloy. For a wurtzite
reflection (h k i l) a cell scatters around the reciprocal lattice point

    Q_x = 4 pi sqrt(h^2 + h k + k^2) / (sqrt(3) a),      Q_z = 2 pi l / c.

Cells sharing the same in-plane constant scatter coherently along the rod
at their Q_x. Each cell is a slab of uniform lattice, so its contribution is
integrated exactly; the amplitude along the rod is

    A(Q_z) = sum_k f_k dz_k sinc(delta_k dz_k / 2) exp(i Phi_k),
    delta_k = Q_z - 2 pi l / c_k,
    Phi_k   = sum_{j<k} delta_j dz_j + delta_k dz_k / 2,

which reproduces layer peaks, thickness fringes and superlattice
satellites. Groups with different in-plane constants (relaxed layers) add in
intensity, each on its own rod. The map is |A|^2 with a Gaussian resolution
function in both directions.

Not included: dynamical diffraction (the peak of a thick, perfect substrate
is too tall and too narrow here), absorption, mosaic tilt and twist,
diffuse scattering, and the substrate below the simulated stack. The
scattering factor f is taken as the mean atomic number of the cell, which
sets relative layer intensities only approximately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from physics.materials.algan import get_nitride_params, nitride_name

# (h, k, l) of the reflections offered; i = -(h + k) is implied
REFLECTIONS = {
    "(0002)": (0, 0, 2),
    "(0004)": (0, 0, 4),
    "(0006)": (0, 0, 6),
    "(10-15)": (1, 0, 5),
    "(10-14)": (1, 0, 4),
    "(11-24)": (1, 1, 4),
    "(20-25)": (2, 0, 5),
}
_Z = {"Al": 13.0, "Ga": 31.0, "In": 49.0, "N": 7.0}


@dataclass
class RSMResult:
    hkl: Tuple[int, int, int]
    label: str
    qx: np.ndarray                 # [1/Angstrom], Q = 2 pi / d
    qz: np.ndarray                 # [1/Angstrom]
    intensity: np.ndarray          # (len(qz), len(qx)), normalised to its maximum
    symmetric: bool                # h = k = 0: every layer lies on the Q_x = 0 rod
    qx_substrate: float            # rod of the bottom layer (the pseudomorphic line)
    # (name, Q_x, Q_z) of each alloy as grown, and fully relaxed
    strained_points: List[Tuple[str, float, float]] = field(default_factory=list)
    relaxed_points: List[Tuple[str, float, float]] = field(default_factory=list)


def _q_inplane(h: int, k: int, a: np.ndarray) -> np.ndarray:
    return 4.0 * np.pi * np.sqrt(h * h + h * k + k * k) / (np.sqrt(3.0) * a)


def simulate_rsm(result, hkl=(1, 0, 5), n_qx: int = 241, n_qz: int = 801,
                 sigma_qx: float = 1.5e-3, sigma_qz: float = 4e-4, alloys=None) -> RSMResult:
    """Reciprocal space map of `result` (a SolverResult) around reflection
    `hkl` = (h, k, l). sigma_qx / sigma_qz are the Gaussian resolution widths
    in 1/Angstrom. `alloys`, a list of (x_Al, x_In), names the compositions
    to mark on the map (the layers of the device); by default they are read
    off the grid."""
    h, k, l = (int(v) for v in hkl)
    if l == 0:
        raise ValueError("The reflection needs l != 0 (the map is taken along the growth axis).")

    z = np.asarray(result.x_nm, dtype=float) * 10.0            # Angstrom
    if len(z) < 3:
        raise ValueError("The structure has too few grid points.")
    x_al = np.asarray(result.x_Al, dtype=float)
    x_in = getattr(result, "x_In", None)
    x_in = np.zeros_like(x_al) if x_in is None else np.asarray(x_in, dtype=float)
    exx = np.asarray(result.eps_xx, dtype=float)
    ezz = np.asarray(result.eps_zz, dtype=float)

    # relaxed lattice constants per distinct composition
    comps = np.round(np.stack([x_al, x_in], axis=1), 6)
    uniq, inverse = np.unique(comps, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    a0_u = np.empty(len(uniq))
    c0_u = np.empty(len(uniq))
    for i, (xa, xi) in enumerate(uniq):
        prm = get_nitride_params(float(xa), float(xi))
        a0_u[i], c0_u[i] = prm.a0, prm.c0
    a = a0_u[inverse] * (1.0 + exx)
    c = c0_u[inverse] * (1.0 + ezz)
    f = x_al * _Z["Al"] + x_in * _Z["In"] + (1.0 - x_al - x_in) * _Z["Ga"] + _Z["N"]
    dz = np.gradient(z)

    qx_cell = _q_inplane(h, k, a)
    qz_cell = 2.0 * np.pi * l / c
    symmetric = (h == 0 and k == 0)

    span_z = float(qz_cell.max() - qz_cell.min())
    pad_z = max(0.05, 0.25 * span_z)
    qz = np.linspace(qz_cell.min() - pad_z, qz_cell.max() + pad_z, n_qz)
    if symmetric:
        qx = np.linspace(-0.03, 0.03, n_qx)
    else:
        # wide enough to hold the fully relaxed positions as well
        qx_all = np.concatenate([qx_cell, _q_inplane(h, k, a0_u)])
        pad_x = max(0.012, 0.15 * float(qx_all.max() - qx_all.min()))
        qx = np.linspace(qx_all.min() - pad_x, qx_all.max() + pad_x, n_qx)

    # cells sharing an in-plane constant scatter coherently on one rod
    keys = np.round(a, 4)
    intensity = np.zeros((n_qz, n_qx))
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        delta = qz[:, None] - qz_cell[idx][None, :]                 # (n_qz, n_cells)
        step = delta * dz[idx][None, :]
        phase = np.cumsum(step, axis=1) - 0.5 * step
        amp = (f[idx] * dz[idx])[None, :] * np.sinc(step / (2.0 * np.pi)) * np.exp(1j * phase)
        rod = np.abs(amp.sum(axis=1)) ** 2                          # (n_qz,)
        if sigma_qz > 0:
            dq = qz[1] - qz[0]
            half = max(1, int(np.ceil(4 * sigma_qz / dq)))
            kern = np.exp(-0.5 * (np.arange(-half, half + 1) * dq / sigma_qz) ** 2)
            rod = np.convolve(rod, kern / kern.sum(), mode="same")
        q_rod = 0.0 if symmetric else float(np.mean(qx_cell[idx]))
        profile = np.exp(-0.5 * ((qx - q_rod) / sigma_qx) ** 2)
        intensity += rod[:, None] * profile[None, :]
    peak = float(intensity.max())
    if peak > 0:
        intensity /= peak

    # Mark the alloys the stack is made of. The grid blends composition over
    # a node or two at each interface, so read off the grid only compositions
    # that fill at least two cells -- or use the caller's list of layers.
    marks = np.round(comps, 3)
    if alloys is None:
        alloys = [(xa, xi) for xa, xi in np.unique(marks, axis=0)
                  if np.count_nonzero((marks[:, 0] == xa) & (marks[:, 1] == xi)) >= 2]
    strained, relaxed, seen = [], [], set()
    for xa, xi in alloys:
        key = (round(float(xa), 3), round(float(xi), 3))
        if key in seen:
            continue
        seen.add(key)
        cells = np.where((np.abs(marks[:, 0] - key[0]) < 2e-3) & (np.abs(marks[:, 1] - key[1]) < 2e-3))[0]
        if len(cells) == 0:
            continue
        prm = get_nitride_params(*key)
        name = nitride_name(*key)
        strained.append((name, 0.0 if symmetric else float(np.mean(qx_cell[cells])),
                         float(np.mean(qz_cell[cells]))))
        relaxed.append((name, 0.0 if symmetric else float(_q_inplane(h, k, prm.a0)),
                        float(2.0 * np.pi * l / prm.c0)))
    order = np.argsort([pt[2] for pt in relaxed])
    strained = [strained[i] for i in order]
    relaxed = [relaxed[i] for i in order]

    label = next((name for name, v in REFLECTIONS.items() if v == (h, k, l)), f"({h} {k} {l})")
    return RSMResult(hkl=(h, k, l), label=label, qx=qx, qz=qz, intensity=intensity,
                     symmetric=symmetric, qx_substrate=0.0 if symmetric else float(qx_cell[0]),
                     strained_points=strained, relaxed_points=relaxed)

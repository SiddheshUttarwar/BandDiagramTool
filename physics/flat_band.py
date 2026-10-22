"""
Flat-band voltage of the quantum wells: the bias at which the field inside
the wells vanishes.

In a metal-polar p-up nitride LED the polarization field in the wells points
against the junction field, so a reverse bias cancels it (the classic QCSE
experiment). The bias spreads over the whole depleted region -- wells,
barriers, depleted cladding -- so the well field moves by about
1 / W_depleted per volt (0.05 MV/cm per V over 200 nm), and a well with a
large polarization step can need tens of volts.

The well field is close to linear in V once the cladding is depleted, so a
secant / false-position search on F_well(V) takes a handful of solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# A well is a band-gap minimum at least this much below the lower of the
# two maxima beside it [eV], no wider than _MAX_WELL_NM. Its extent is the
# nodes within half that depth, and within _CORE_EV, of its bottom: a well
# grown straight on a lower-gap layer (no barrier between) would otherwise
# take that layer in.
_MIN_DEPTH_EV = 0.05
_CORE_EV = 0.1
_MAX_WELL_NM = 30.0


def quantum_region_nm(layers) -> Optional[Tuple[float, float]]:
    """(start, end) [nm] of the Quantum Region Markers in a layer list, or None."""
    z, start, end = 0.0, None, None
    for layer in layers:
        kind = getattr(layer, "boundary", None)
        if kind == "start" and start is None:
            start = z
        elif kind == "end":
            end = z
        z += float(getattr(layer, "thickness_nm", 0.0) or 0.0)
    return (start, end) if start is not None and end is not None and end > start else None


def find_wells(r, window_nm: Optional[Tuple[float, float]] = None) -> List[Tuple[int, int]]:
    """Node ranges [i0, i1) of the quantum wells of a solved structure.

    Composition is graded over ~1 nm at interfaces (devices.grid_builder), so
    a thin well has no stretch of constant gap: a well is a minimum of the
    band gap, at least _MIN_DEPTH_EV below the lower of the maxima on either
    side, spanning the nodes below half its depth (at most _CORE_EV). window_nm (e.g. the
    Quantum Region Markers, quantum_region_nm) keeps the wells whose centre
    lies in it, so a short-period superlattice outside is not counted.
    """
    x = np.asarray(r.x_nm, dtype=float)
    gap = np.asarray(r.Ec0, dtype=float) - np.asarray(r.Ev0, dtype=float)
    N = len(gap)
    if N < 3:
        return []
    # minima, a flat bottom counted once (its first node)
    d = np.sign(np.diff(gap))
    for k in range(1, len(d)):           # carry the last slope over flat steps
        if d[k] == 0:
            d[k] = d[k - 1]
    mins = [k + 1 for k in range(len(d) - 1) if d[k] < 0 and d[k + 1] > 0]
    wells = []
    for j, m in enumerate(mins):
        lo = mins[j - 1] if j > 0 else 0
        hi = mins[j + 1] if j + 1 < len(mins) else N - 1
        left, right = gap[lo:m + 1].max(), gap[m:hi + 1].max()
        depth = min(left, right) - gap[m]
        if depth < _MIN_DEPTH_EV:
            continue
        half = gap[m] + min(0.5 * depth, _CORE_EV)
        i0 = m
        while i0 > lo and gap[i0 - 1] < half:
            i0 -= 1
        i1 = m + 1
        while i1 < hi and gap[i1] < half:
            i1 += 1
        if x[i1 - 1] - x[i0] > _MAX_WELL_NM:
            continue
        centre = 0.5 * (x[i0] + x[i1 - 1])
        if window_nm is not None and not window_nm[0] <= centre <= window_nm[1]:
            continue
        wells.append((int(i0), int(i1)))
    return wells


def well_fields(r, wells: Optional[List[Tuple[int, int]]] = None) -> List[float]:
    """Mean electric field in each well [MV/cm], over its extent (find_wells)."""
    wells = find_wells(r) if wells is None else wells
    F = np.asarray(r.E_field, dtype=float) * 1e-8
    out = []
    for i0, i1 in wells:
        out.append(float(np.mean(F[i0:i1])))
    return out


@dataclass
class FlatBandResult:
    V_fb: float                     # flat-band voltage [V]
    converged: bool
    message: str
    result: object                  # SolverResult at V_fb (the last solve)
    wells_nm: List[Tuple[float, float]]
    target: str                     # 'mean of N wells' or 'well k'
    history: List[Tuple[float, List[float]]] = field(default_factory=list)   # (V, field per well [MV/cm])
    # (V_fb, converged) of each well on its own (flat_band_each_well); None until run
    per_well: Optional[List[Tuple[float, bool]]] = None


def find_flat_band(device, solve_kwargs: Optional[dict] = None, well: Optional[int] = None,
                   V_first: float = -5.0, V_limit: float = 150.0, max_step: float = 40.0,
                   tol_MV_cm: float = 0.005, max_solves: int = 14,
                   log_fn: Optional[Callable[[str], None]] = None) -> FlatBandResult:
    """Bias at which the field in the wells is zero.

    well: index of the well to flatten (0 = the one nearest the bottom
    contact); None flattens their mean. The wells are those of the quantum
    region (Quantum Region Markers; all wells without them). Each bias point is a fresh device.solve(V_applied=V,
    **solve_kwargs). Stops when |F| < tol_MV_cm or the bias step is below
    1 mV; |V| is kept within V_limit.
    """
    kw = dict(solve_kwargs or {})
    log = log_fn or (lambda s: None)
    history: List[Tuple[float, List[float]]] = []
    pts: List[Tuple[float, float]] = []
    wells_idx: List[Tuple[int, int]] = []
    last = None

    def solve(V):
        nonlocal wells_idx, last
        log(f"Flat band: solving V = {V:+.4f} V ...")
        r = device.solve(V_applied=V, **kw)
        if not wells_idx:
            wells_idx = find_wells(r, quantum_region_nm(getattr(device, "layers", [])))
            if not wells_idx:
                raise ValueError("no quantum well found: a well is a layer of lower band gap than both "
                                 "neighbours (inside the quantum region markers, if set)")
            if well is not None and not 0 <= well < len(wells_idx):
                raise ValueError(f"well {well} does not exist: {len(wells_idx)} wells found")
        fw = well_fields(r, wells_idx)
        f = float(np.mean(fw)) if well is None else fw[well]
        history.append((float(V), fw))
        pts.append((float(V), f))
        last = r
        log(f"Flat band: V = {V:+.4f} V   well field {f:+.4f} MV/cm"
            + ("" if r.converged else "   (solve NOT converged)"))
        return f

    def done(V, ok, msg):
        x = np.asarray(last.x_nm, dtype=float)
        return FlatBandResult(V_fb=float(V), converged=ok, message=msg, result=last,
                              wells_nm=[(float(x[i0]), float(x[i1 - 1])) for i0, i1 in wells_idx],
                              target=(f"mean of {len(wells_idx)} wells" if well is None else f"well {well + 1}"),
                              history=history)

    f = solve(0.0)
    if abs(f) < tol_MV_cm:
        return done(0.0, True, "the wells are flat without bias")
    V = V_first
    for _ in range(max_solves - 1):
        f = solve(V)
        if abs(f) < tol_MV_cm:
            return done(V, last.converged, "")
        lo = [p for p in pts if p[1] < 0]
        hi = [p for p in pts if p[1] > 0]
        if lo and hi:
            # bracketed: false position between the closest points either side
            (Va, fa) = min(lo, key=lambda p: abs(p[1]))
            (Vb, fb) = min(hi, key=lambda p: abs(p[1]))
            V_new = Va - fa * (Vb - Va) / (fb - fa)
        else:
            (V0, f0), (V1, f1) = pts[-2], pts[-1]
            if f1 == f0:
                return done(V, False, "the well field does not change with bias")
            V_new = V1 - f1 * (V1 - V0) / (f1 - f0)
            V_new = V1 + float(np.clip(V_new - V1, -max_step, max_step))
        V_new = float(np.clip(V_new, -V_limit, V_limit))
        if abs(V_new - V) < 1e-3:
            return done(V, last.converged, "")
        if abs(V_new) >= V_limit and abs(V) >= V_limit:
            return done(V, False, f"no flat band within |V| <= {V_limit:g} V")
        V = V_new
    return done(V, False, f"not found to {tol_MV_cm} MV/cm in {max_solves} solves")


def flat_band_each_well(device, fb: FlatBandResult, solve_kwargs: Optional[dict] = None,
                        tol_MV_cm: float = 0.005, max_solves: int = 6, V_limit: float = 150.0,
                        log_fn: Optional[Callable[[str], None]] = None) -> List[Tuple[float, bool]]:
    """Flat-band voltage of every well on its own, (V_fb, converged) per well.

    The wells of a stack do not flatten together: the outer wells sit next to
    cladding of another composition and the depletion edges, so each has its
    own field at a given bias. Starts from the solves of a find_flat_band
    search (fb.history, which every new solve is added to, so later wells
    reuse them) and refines each well's zero by false position; F(V) is close
    to linear, so this usually takes zero to two solves per well.
    """
    kw = dict(solve_kwargs or {})
    log = log_fn or (lambda s: None)
    n = len(fb.history[0][1]) if fb.history else 0
    wells_idx = find_wells(fb.result, quantum_region_nm(getattr(device, "layers", [])))
    out: List[Tuple[float, bool]] = []
    for k in range(n):
        ok, V_best = False, fb.history[-1][0]
        for _ in range(max_solves + 1):
            pts = [(V, fw[k]) for V, fw in fb.history]
            V_best, f_best = min(pts, key=lambda p: abs(p[1]))
            if abs(f_best) < tol_MV_cm:
                ok = True
                break
            lo = [p for p in pts if p[1] < 0]
            hi = [p for p in pts if p[1] > 0]
            if lo and hi:
                (Va, fa) = min(lo, key=lambda p: abs(p[1]))
                (Vb, fb_) = min(hi, key=lambda p: abs(p[1]))
            else:
                (Va, fa), (Vb, fb_) = sorted(pts, key=lambda p: abs(p[1]))[:2]
            if fb_ == fa:
                break
            V_new = float(np.clip(Va - fa * (Vb - Va) / (fb_ - fa), -V_limit, V_limit))
            if min(abs(V_new - V) for V, _ in pts) < 1e-3:
                ok = abs(f_best) < 10 * tol_MV_cm
                break
            log(f"Flat band, well {k + 1}: solving V = {V_new:+.4f} V ...")
            r = device.solve(V_applied=V_new, **kw)
            fw = well_fields(r, wells_idx)
            fb.history.append((V_new, fw))
            log(f"Flat band, well {k + 1}: V = {V_new:+.4f} V   well field {fw[k]:+.4f} MV/cm"
                + ("" if r.converged else "   (solve NOT converged)"))
        out.append((float(V_best), ok))
    return out

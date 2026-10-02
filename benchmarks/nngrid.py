"""Clustered 1D grid generator that fits nextnano++ free-edition's 100-point cap."""
import numpy as np

def clustered_points(interfaces, fine, coarse, growth=1.35, max_pts=100):
    """Points fine at each interface, growing geometrically to `coarse` between."""
    interfaces = sorted(set(float(v) for v in interfaces))
    pts = set(interfaces)
    for a, b in zip(interfaces[:-1], interfaces[1:]):
        L = b - a
        # grow from both ends toward middle
        left, right = [a], [b]
        h = fine
        while True:
            nl = left[-1] + h
            nr = right[-1] - h
            if nl >= nr - 0.5 * h:
                break
            left.append(nl); right.append(nr)
            h = min(h * growth, coarse)
        seg = sorted(left + right)
        pts.update(seg)
    pts = np.array(sorted(pts))
    # merge points closer than fine/2
    keep = [pts[0]]
    for p in pts[1:]:
        if p - keep[-1] > 0.45 * fine:
            keep.append(p)
    keep[-1] = pts[-1]
    return np.array(keep)

def fit_grid(interfaces, fine=0.5, coarse=50.0, growth=1.35, max_pts=100):
    """Coarsen `fine`/`growth` until the grid fits max_pts."""
    f, g = fine, growth
    for _ in range(200):
        p = clustered_points(interfaces, f, coarse, g)
        if len(p) <= max_pts:
            return p
        f *= 1.08; g = min(g * 1.03, 2.2)
    raise RuntimeError('cannot fit grid')

def xgrid_block(pts):
    lines = []
    for i, p in enumerate(pts):
        h = (pts[i + 1] - p) if i + 1 < len(pts) else (p - pts[i - 1])
        lines.append(f"        line{{ pos = {p:.6g} spacing = {h:.6g} }}")
    return "    xgrid{\n" + "\n".join(lines) + "\n    }"

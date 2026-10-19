"""
Indirect conduction valleys (X, L) of zincblende alloys, which the 8-band
k.p model (Gamma only) does not contain. As in nextnano++, each valley set
is a single-band effective-mass problem (physics.schrodinger) on its own
band edge, added to the k.p electron density.

For [001] growth:
  X_z   1 valley,  growth-direction mass m_l,  in-plane DOS mass m_t
  X_xy  2 valleys, growth-direction mass m_t,  in-plane DOS mass sqrt(m_l m_t)
  L     4 valleys, growth-direction mass 3 m_l m_t / (m_t + 2 m_l),
                   in-plane DOS mass sqrt(m_t (2 m_l + m_t) / 3)
(one valley = both halves at the zone boundary). Band edges: the grid's
(lowest-valley) Ec plus the strained distance from the lowest valley to the
valley in question (ZincblendeParams._valley_edges). A valley set is solved
only where it comes within the occupation window of Efn in the region.
"""
from __future__ import annotations

import numpy as np

from physics.constants import kB, q as _q, hbar, m0
from physics.schrodinger import solve_schrodinger


def valley_tables(g, i0: int, i1: int):
    """{name: (offset [eV], m_conf, m_dos, degeneracy)} on nodes [i0, i1)
    of a zincblende grid; empty for wurtzite."""
    if getattr(g, 'crystal', 'wurtzite') != 'zincblende':
        return {}
    from physics.materials.zincblende import get_zincblende_params, _BIN, _interp, _weights, _check_comp
    out = {k: ([], [], [], d) for k, d in (('X_z', 1), ('X_xy', 2), ('L', 4))}
    cache = {}
    for i in range(i0, i1):
        key = (float(g.x_Al[i]), float(g.x_In[i]), float(g.x_P[i]), float(g.eps_xx[i]), float(g.eps_zz[i]))
        if key not in cache:
            p = get_zincblende_params(key[0], key[1], key[2], g.T)
            e = p._valley_edges(key[3], key[4])
            low = min(e.values())
            x, y, v = _check_comp(key[0], key[1], key[2])
            f, gg = _weights(x, y, v)
            ml_x, mt_x = (_interp(lambda b, j=j: _BIN[b]['mX'][j], 'mX', f, gg) for j in (0, 1))
            ml_l, mt_l = (_interp(lambda b, j=j: _BIN[b]['mL'][j], 'mL', f, gg) for j in (0, 1))
            cache[key] = {
                'X_z': (e['X'] - low, ml_x, mt_x),
                'X_xy': (e['X'] - low, mt_x, np.sqrt(ml_x * mt_x)),
                'L': (e['L'] - low, 3.0 * ml_l * mt_l / (mt_l + 2.0 * ml_l), np.sqrt(mt_l * (2.0 * ml_l + mt_l) / 3.0)),
            }
        for k in out:
            for lst, val in zip(out[k][:3], cache[key][k]):
                lst.append(val)
    return {k: (np.array(a), np.array(b), np.array(c), d) for k, (a, b, c, d) in out.items()}


def valley_density(tables, Ec_region: np.ndarray, dx_region, Efn_region: np.ndarray, T: float,
                   local_qfl: bool = False, tail_kT: float = 10.0) -> np.ndarray:
    """Electron density [cm^-3] in the X and L valleys on the region."""
    kT = kB * T / _q
    n = np.zeros(len(Ec_region))
    if not tables:
        return n
    Efn_region = np.broadcast_to(np.asarray(Efn_region, float), Ec_region.shape)
    E_hi = float(Efn_region.max()) + tail_kT * kT
    from physics.grid_utils import node_spacings
    w = node_spacings(dx_region, len(Ec_region))[2]
    for name, (offset, m_conf, m_dos, deg) in tables.items():
        Ev_band = Ec_region + offset
        if float(Ev_band.min()) > E_hi:
            continue
        n_states = 8
        while True:
            E, psi = solve_schrodinger(Ev_band, m_conf, dx_region, n_states=n_states)
            if len(E) == 0 or E[-1] >= E_hi or n_states >= len(Ec_region) - 2:
                break
            n_states *= 2
        for Es, ps in zip(E, psi):
            if Es > E_hi:
                break
            prob = ps ** 2 * w
            m_s = float(np.sum(prob * m_dos)) * m0
            pref = deg * m_s * kB * T / (np.pi * hbar ** 2)          # m^-2, spin included
            if local_qfl:
                sheet = pref * np.logaddexp(0.0, np.minimum((Efn_region - Es) / kT, 500.0))
            else:
                F = float(np.sum(prob * Efn_region))
                sheet = pref * np.logaddexp(0.0, min((F - Es) / kT, 500.0))
            n += ps ** 2 * sheet * 1e-6                             # m^-3 -> cm^-3
    return n

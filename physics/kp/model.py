"""
The quantum model the self-consistent solver uses: 8-band k.p for the Gamma
conduction band and the valence bands (physics.kp.solver), plus single-band
X and L valleys in indirect zincblende alloys (physics.kp.valleys), on one
quantum region [i0, i1) of a grid.

Also the optics helpers that need the multi-component states:

  spinor_overlap   |<F_e,S | F_h,v>|^2 of an electron level and a hole
                   level, summed over the hole's Kramers pair and over the
                   valence components, normalised by the electron's |S>
                   weight and the hole's valence weight. For pure states it
                   is the familiar envelope overlap |<f_e|f_h>|^2.
  levels           Kramers pairs grouped into levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from physics.kp.parameters import build_kp_static
from physics.kp.solver import KPSolver, KPStates
from physics.kp.valleys import valley_tables, valley_density

_DEGENERATE_EV = 1e-6
_VB = [1, 2, 3, 5, 6, 7]


def levels(E: np.ndarray) -> list:
    """Indices grouped into (Kramers-) degenerate levels, in order."""
    out = []
    for i, e in enumerate(E):
        if out and abs(e - E[out[-1][0]]) < _DEGENERATE_EV:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def spinor_overlap(spin_e: np.ndarray, spin_h_pair: np.ndarray, w_nm: np.ndarray) -> float:
    """Overlap of one electron spinor (N, 8) with a hole level given as its
    Kramers pair (m, N, 8)."""
    eS = spin_e[:, [0, 4]]                                  # (N, 2) S up/down
    wS = float(np.sum(w_nm[:, None] * np.abs(eS) ** 2))
    tot = 0.0
    wV = 0.0
    for h in spin_h_pair:
        for sigma, comps in ((0, (1, 2, 3)), (1, (5, 6, 7))):
            for c in comps:
                tot += abs(np.sum(w_nm * np.conj(eS[:, sigma]) * h[:, c])) ** 2
        wV += float(np.sum(w_nm[:, None] * np.abs(h[:, _VB]) ** 2))
    wV /= max(len(spin_h_pair), 1)
    return float(tot / max(wS * wV, 1e-300))


# A carrier type whose classical density stays below this everywhere in the
# region carries no charge that matters (the correction factors are only
# checked above 1e10 cm^-3): it is not k.p-integrated and keeps its
# classical density -- e.g. the holes ~3 eV below the Fermi level in a HEMT
# channel, which otherwise cost half of every solve.
_NEGLIGIBLE_CM3 = 1.0


@dataclass
class QuantumSolution:
    """One k.p (+ valleys) solve on the region."""
    states: KPStates
    n_valley: np.ndarray            # X/L electrons on the region [cm^-3]
    skipped: tuple = ()             # carrier types left classical ('e', 'h')


class KPModel:
    def __init__(self, g, i0: int, i1: int):
        self.g, self.i0, self.i1 = g, i0, i1
        self.par = build_kp_static(g, i0, i1)
        self.solver = KPSolver(self.par, g.dx, i0, i1, g.N)
        self.valleys = valley_tables(g, i0, i1)
        dx = np.asarray(g.dx)
        self.dx_region = dx if dx.ndim == 0 else dx[i0:i1 - 1]
        self.w_nm = self.solver.w_nm

    @property
    def ep_rescaled(self):
        return self.par.ep_rescaled

    def _sl(self, a):
        a = np.asarray(a, float)
        return a[self.i0:self.i1] if a.ndim else np.full(self.i1 - self.i0, float(a))

    def solve(self, Ec, Ev, Efn, Efp, T: float, local_qfl: bool = False,
              n_display: int = 8) -> QuantumSolution:
        """Ec, Ev, Efn, Efp: full-grid arrays (or scalars for the levels)."""
        Ec_r, Ev_r = self._sl(Ec), self._sl(Ev)
        Efn_r, Efp_r = self._sl(Efn), self._sl(Efp)
        skip = tuple(k for k, cl in (('e', self._classical(Ec, Ev, Efn, Efp, T)[0]),
                                     ('h', self._classical(Ec, Ev, Efn, Efp, T)[1]))
                     if float(np.max(cl)) < _NEGLIGIBLE_CM3)
        st = self.solver.solve(Ec_r, Ev_r, Efn_r, Efp_r, T, local_qfl, n_display=n_display, skip=skip)
        nv = valley_density(self.valleys, Ec_r, self.dx_region, Efn_r, T, local_qfl)
        return QuantumSolution(st, nv, skip)

    def _classical(self, Ec, Ev, Efn, Efp, T):
        """Classical (n, p) on the region [cm^-3]."""
        from physics.fermi_dirac import electron_density, hole_density
        sl = slice(self.i0, self.i1)
        n = electron_density(self._sl(Ec), self._sl(Efn), np.asarray(self.g.Nc)[sl], T)
        p = hole_density(self._sl(Ev), self._sl(Efp), np.asarray(self.g.Nv)[sl], T)
        return n, p

    def densities(self, sol: QuantumSolution, Efn, Efp, T: float, local_qfl: bool = False,
                  Ec=None, Ev=None):
        """Full-grid (n, p) [cm^-3], zero outside the region (callers blend
        with the classical density there). With Efn/Efp other than the
        solve's, the k.p states are frozen (the bias predictor); the
        valley part is re-solved on Ec when given."""
        Efn_r, Efp_r = self._sl(Efn), self._sl(Efp)
        n_r, p_r = sol.states.densities(Efn_r, Efp_r, local_qfl)
        if Ec is not None and self.valleys:
            n_r = n_r + valley_density(self.valleys, self._sl(Ec), self.dx_region, Efn_r, T, local_qfl)
        else:
            n_r = n_r + sol.n_valley
        if sol.skipped:
            if Ec is None or Ev is None:
                raise ValueError("densities() needs Ec and Ev when a carrier type was left classical")
            n_cl, p_cl = self._classical(Ec, Ev, Efn, Efp, T)
            if 'e' in sol.skipped:
                n_r = n_cl          # (the classical density already counts every valley)
            if 'h' in sol.skipped:
                p_r = p_cl
        n = np.zeros(self.g.N)
        p = np.zeros(self.g.N)
        n[self.i0:self.i1] = np.maximum(n_r, 1e-20)
        p[self.i0:self.i1] = np.maximum(p_r, 1e-20)
        return n, p

    # ---- k = 0 states in the solver's legacy single-band layout ----------
    def display_states(self, sol: QuantumSolution):
        """(E_e, psi_e, E_h, psi_h, hole_character, e_levels, h_levels):
        one entry per Kramers level, psi = sqrt(sum of |component|^2)
        on the full grid in m^-1/2 (sum psi^2 * cell_width_m = 1), E_h in
        the legacy inverted frame (valence energy = -E_h)."""
        st = self.solver.display(sol.states)
        N = self.g.N

        def pack(E, rho):
            lv = levels(E)
            Eu = np.array([E[ix[0]] for ix in lv])
            psi = np.zeros((len(lv), N))
            for k, ix in enumerate(lv):
                psi[k, self.i0:self.i1] = np.sqrt(np.maximum(rho[ix[0]], 0.0) * 1e9)
            return Eu, psi, lv
        E_e, psi_e, lv_e = pack(st.E_e0, st.rho_e0) if len(st.E_e0) else (None, None, [])
        if len(st.E_h0):
            E_h, psi_h, lv_h = pack(st.E_h0, st.rho_h0)
            E_h = -E_h
            char = np.array([st.char_h0[ix[0]] for ix in lv_h])
        else:
            E_h = psi_h = char = None
            lv_h = []
        return E_e, psi_e, E_h, psi_h, char, lv_e, lv_h

    def overlap(self, sol: QuantumSolution, lv_e: list, lv_h: list, ie: int, ih: int) -> float:
        st = self.solver.display(sol.states)
        return spinor_overlap(st.spin_e0[lv_e[ie][0]], st.spin_h0[lv_h[ih]], self.w_nm)


def local_well_transition(g, Ec: np.ndarray, Ev: np.ndarray, i0: int, i1: int, T: float,
                          pad_nm: float = 4.0) -> Optional[tuple]:
    """(E_e1 - E_h1 [eV], overlap) of the well [i0, i1) from a k = 0 k.p
    solve restricted to the well plus pad_nm of barrier on each side (hard
    walls beyond). None if no confined pair is found."""
    x = g.x_nm
    j0 = int(np.searchsorted(x, x[i0] - pad_nm))
    j1 = int(min(g.N, np.searchsorted(x, x[i1 - 1] + pad_nm) + 1))
    if j1 - j0 < 8:
        return None
    try:
        m = KPModel(g, j0, j1)
        Ec_r, Ev_r = np.asarray(Ec)[j0:j1], np.asarray(Ev)[j0:j1]
        lo_e = float((Ec_r + m.par.Ec).min()) - 0.05
        Ee, pe = m.solver.eigen(0.0, Ec_r, Ev_r, lo_e, lo_e + 1.5, side='up')
        cb = m.solver.s_weight(pe) > 0.5
        hi_h = float(Ev_r.max()) + 0.05
        Eh, ph = m.solver.eigen(0.0, Ec_r, Ev_r, hi_h - 1.5, hi_h, side='down')
        vb = m.solver.s_weight(ph) <= 0.5
        Ee, pe, Eh, ph = Ee[cb], pe[cb], Eh[vb][::-1], ph[vb][::-1]
    except Exception:  # noqa: BLE001 - diagnostic only, never break a solve
        return None
    if len(Ee) == 0 or len(Eh) == 0:
        return None
    lh = levels(Eh)
    ov = spinor_overlap(pe[0], ph[lh[0]], m.w_nm)
    return float(Ee[0] - Eh[0]), ov

"""
Reverse-bias breakdown and leakage, evaluated on a solved band diagram.

Three mechanisms, each from the band edges, field and quasi-Fermi levels of
one SolverResult (post-processing: the leakage they predict is far below the
currents that would change the electrostatics, until breakdown itself):

  * Impact ionization (avalanche). Chynoweth coefficients
        alpha(F) = a exp(-b / |F|)      [cm^-1, F in V/cm]
    with the GaN values of Cao et al., APL 112, 262103 (2018), and b scaled
    with the local band gap as (Eg / Eg_GaN)^2 for AlGaN / InGaN (critical
    field ~ Eg^2: GaN ~3.3 MV/cm, AlN ~10 MV/cm over a 1 um drift region).
    There are no measured coefficients for high-Al AlGaN; the scaling is the
    weakest number in this module. alpha is evaluated at the effective
    field: the field along the carrier path averaged over a dead space of
    DEAD_SPACE_NM, and zero where it decelerates the carriers. The ionization integral along the
    electron path
        I_n = int alpha_n exp(-int_0^x (alpha_n - alpha_p) dx') dx
    gives the multiplication M_n = 1 / (1 - I_n); avalanche breakdown is
    I = 1 (Sze, Physics of Semiconductor Devices, ch. 2.4).

  * Interband (Zener) tunnelling, nonlocal WKB. For every energy E between
    the lowest conduction-band edge and the highest valence-band edge, each
    stretch of the gap with a valence band on one side (Ev >= E) and a
    conduction band on the other (Ec <= E) is a tunnelling path. Two-band
    (Kane) decay constant
        kappa(x) = sqrt(2 m_r (Ec - E)(E - Ev) / Eg) / hbar,
    m_r the reduced mass of the electron and the light hole. Transmission
    T = exp(-2 int kappa dx); the transverse momentum lowers it as
    exp(-k_perp^2 L), L = int dx / kappa, which integrates over k_perp to
    1 / (4 pi L) per unit area. Current (Landauer, spin 2):
        J = (2 q / h) int dE (f_v - f_c) T(E) / (4 pi L(E)),
    f_v, f_c the occupations at the valence and conduction ends of the path.
    For a uniform field this is Kane's result, T = exp(-pi sqrt(m_r)
    Eg^1.5 / (2 sqrt(2) q hbar F)). Unlike a local Kane model it gives no
    current in a thin high-field layer (a 1 nm polarization-doped
    superlattice period) whose potential drop is below the gap.

  * Trap-assisted tunnelling (Hurkx et al., IEEE TED 39, 331 (1992)). SRH
    through midgap traps with field-enhanced emission:
        R = (n p - ni^2) / (tau_p (n + ni)/(1 + Gamma_p) + tau_n (p + ni)/(1 + Gamma_n)),
        Gamma = (dE/kT) int_0^1 exp((dE/kT) u - K u^1.5) du,
        K = 4 sqrt(2 m dE^3) / (3 q hbar |F|),  dE = Eg / 2.
    The generation (R < 0) integrated over the device is the TAT current.
    With the default 1 ns lifetimes this is an upper estimate: real
    leakage in AlGaN UV LEDs runs along dislocations, a 3D path this 1D
    model does not have.

All currents are magnitudes in A/cm^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from physics.constants import q as _q, hbar as _hbar, h as _h, m0 as _m0, kB as _kB

# GaN impact ionization, Cao et al. APL 112, 262103 (2018)
A_N, B_N = 4.48e8, 3.39e7      # electrons [cm^-1], [V/cm]
A_P, B_P = 7.13e6, 1.46e7      # holes
EG_GAN = 3.39                  # gap the coefficients belong to [eV]
B_GAP_EXPONENT = 2.0           # b ~ Eg^2 for other alloys

# Dead space: a carrier must travel ~E_th/(qF) (~10 nm at the fields of
# interest, E_th ~ 1.5 Eg) before it can ionize, so impact ionization sees
# the field averaged over this length, not the local one. Without it the
# +-4 MV/cm polarization fields of a 1 nm-period AlGaN superlattice, which
# cancel over a few periods, would count as avalanche regions.
DEAD_SPACE_NM = 10.0

_DE_BTBT = 0.005               # energy step of the tunnelling integral [eV]
_HURKX_U = np.linspace(0.0, 1.0, 401)


@dataclass
class BreakdownAnalysis:
    V: float
    ion_integral_n: float          # electron-initiated ionization integral (1 = avalanche)
    ion_integral_p: float          # hole-initiated
    M_n: float                     # multiplication of injected electrons (inf at/after breakdown)
    M_p: float
    J_btbt: float                  # interband tunnelling current [A/cm^2]
    J_tat: float                   # trap-assisted tunnelling (Hurkx SRH generation) [A/cm^2]
    J_leak: float                  # (J_btbt + J_tat) multiplied by the avalanche gain
    peak_field_MV_cm: float
    peak_field_nm: float
    alpha_n: np.ndarray            # [cm^-1] per node
    alpha_p: np.ndarray
    G_tat: np.ndarray              # TAT generation [cm^-3 s^-1] per node
    btbt_path: Optional[Tuple[float, float]] = None   # (valence end, conduction end) [nm] of the strongest path

    @property
    def ion_integral(self) -> float:
        return max(self.ion_integral_n, self.ion_integral_p)


def ionization_coefficients(F_V_cm, Eg):
    """alpha_n, alpha_p [cm^-1] at field |F| [V/cm] in a gap Eg [eV]."""
    F = np.maximum(np.abs(np.asarray(F_V_cm, dtype=float)), 1.0)
    s = (np.asarray(Eg, dtype=float) / EG_GAN) ** B_GAP_EXPONENT
    return A_N * np.exp(-B_N * s / F), A_P * np.exp(-B_P * s / F)


def ionization_integrals(x_cm, alpha_n, alpha_p, electrons_forward: bool):
    """(I_n, I_p): electron- and hole-initiated ionization integrals on the
    node grid. electrons_forward: electrons drift towards +x."""
    x = np.asarray(x_cm, dtype=float)
    an, ap = np.asarray(alpha_n, dtype=float), np.asarray(alpha_p, dtype=float)
    if not electrons_forward:                 # walk along the electron path
        x, an, ap = x[-1] - x[::-1], an[::-1], ap[::-1]
    dx = np.diff(x)
    d = an - ap
    # cumulative int_0^x (alpha_n - alpha_p), trapezoid
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * dx)])
    fn = an * np.exp(-cum)
    I_n = float(np.sum(0.5 * (fn[1:] + fn[:-1]) * dx))
    # holes enter at the far end: int alpha_p exp(+int_x^W (alpha_n - alpha_p)) ...
    # written as exp(-(cum_W - cum_x) * -1)
    fp = ap * np.exp(cum - cum[-1])
    I_p = float(np.sum(0.5 * (fp[1:] + fp[:-1]) * dx))
    return I_n, I_p


def _hurkx_gamma(dE_eV, F_V_m, m_rel, kT_eV):
    """Hurkx field-enhancement factor Gamma (array), computed in log space."""
    dE = np.maximum(np.asarray(dE_eV, dtype=float), 1e-3)
    F = np.maximum(np.abs(np.asarray(F_V_m, dtype=float)), 1.0)
    K = 4.0 * np.sqrt(2.0 * m_rel * _m0 * (dE * _q) ** 3) / (3.0 * _q * _hbar * F)
    a = dE / kT_eV
    u = _HURKX_U
    expo = a[:, None] * u[None, :] - K[:, None] * u[None, :] ** 1.5
    m = expo.max(axis=1)
    w = np.full(len(u), u[1] - u[0]); w[[0, -1]] *= 0.5
    ln_int = m + np.log(np.sum(w[None, :] * np.exp(expo - m[:, None]), axis=1))
    return np.exp(np.minimum(np.log(a) + ln_int, 700.0))


def tat_generation(r, Nc, Nv, tau_n, tau_p, m_e, m_h):
    """Trap-assisted tunnelling generation rate [cm^-3 s^-1] per node (>= 0)."""
    kT = _kB * r.T / _q
    Ec, Ev = np.asarray(r.Ec, dtype=float), np.asarray(r.Ev, dtype=float)
    Eg = Ec - Ev
    F = np.asarray(r.E_field, dtype=float)
    ln_ni = 0.5 * (np.log(Nc) + np.log(Nv)) - Eg / (2.0 * kT)
    ni = np.exp(ln_ni)
    n = np.maximum(np.asarray(r.n, dtype=float), 0.0)
    p = np.maximum(np.asarray(r.p, dtype=float), 0.0)
    gn = _hurkx_gamma(0.5 * Eg, F, m_e, kT)
    gp = _hurkx_gamma(0.5 * Eg, F, m_h, kT)
    # n p - ni_eff^2 with the solver's quasi-Fermi split (Boltzmann form)
    a = np.clip((np.asarray(r.Efn) - np.asarray(r.Efp)) / kT, -600.0, 600.0)
    excess = n * p * (-np.expm1(-a)) if np.all(n * p > 0) else n * p - ni ** 2
    den = tau_p * (n + ni) / (1.0 + gp) + tau_n * (p + ni) / (1.0 + gn)
    R = excess / den
    return np.maximum(-R, 0.0)


def _wkb_segments(k2a, k2b, dx):
    """int kappa dx and int dx / kappa over segments on which kappa^2 is
    linear between its node values k2a, k2b [m^-2] (exact for a uniform
    field, and exact at the turning points, where the 1/kappa integrand is
    singular but integrable). The part of a segment with kappa^2 < 0 (beyond
    a turning point) is left out."""
    k2a = np.asarray(k2a, dtype=float); k2b = np.asarray(k2b, dtype=float)
    lo_, hi_ = np.minimum(k2a, k2b), np.maximum(k2a, k2b)
    pos = hi_ > 0.0
    # length of the positive part, and kappa^2 at its two ends
    l = np.where(lo_ >= 0.0, dx, dx * hi_ / np.where(hi_ - lo_ > 0, hi_ - lo_, 1.0))
    k_lo = np.where(lo_ >= 0.0, lo_, 0.0)
    slope = (hi_ - k_lo) / np.maximum(l, 1e-300)
    flat = slope * l < 1e-9 * np.maximum(hi_, 1e-300)
    sh, sl = np.sqrt(np.maximum(hi_, 0.0)), np.sqrt(k_lo)
    S_seg = np.where(flat, sh * l, (2.0 / 3.0) * (sh ** 3 - sl ** 3) / np.where(flat, 1.0, slope))
    L_seg = np.where(flat, l / np.maximum(sh, 1e-300), 2.0 * (sh - sl) / np.where(flat, 1.0, slope))
    return float(np.sum(np.where(pos, S_seg, 0.0))), float(np.sum(np.where(pos, L_seg, 0.0)))


def btbt_current(r, m_e, m_h, dE: float = _DE_BTBT):
    """Interband tunnelling current [A/cm^2] (magnitude) and the
    (valence end, conduction end) [nm] of the path carrying the most."""
    x = np.asarray(r.x_nm, dtype=float) * 1e-9
    Ec, Ev = np.asarray(r.Ec, dtype=float), np.asarray(r.Ev, dtype=float)
    Efn, Efp = np.asarray(r.Efn, dtype=float), np.asarray(r.Efp, dtype=float)
    kT = _kB * r.T / _q
    lo, hi = float(Ec.min()), float(Ev.max())
    if hi <= lo:
        return 0.0, None                      # no energy has both bands available
    mr = 1.0 / (1.0 / np.asarray(m_e, dtype=float) + 1.0 / np.asarray(m_h, dtype=float))
    xm_d = np.diff(x)
    Egn = np.maximum(Ec - Ev, 1e-3)
    pref = 2.0 * _q / _h / (4.0 * np.pi)       # A/J per m^2 ... times dE[J] gives A/m^2
    J, best, best_path = 0.0, 0.0, None
    for E in np.arange(lo + 0.5 * dE, hi, dE):
        val = Ev >= E
        con = Ec <= E
        gap = ~(val | con)
        # runs of gap nodes
        g = np.concatenate([[False], gap, [False]]).astype(np.int8)
        starts = np.where(np.diff(g) == 1)[0]
        ends = np.where(np.diff(g) == -1)[0]          # exclusive
        for a, b in zip(starts, ends):
            if a == 0 or b == len(x):
                continue
            left_v, right_v = val[a - 1], val[b]
            left_c, right_c = con[a - 1], con[b]
            if left_v and right_c:
                iv, ic = a - 1, b
            elif left_c and right_v:
                iv, ic = b, a - 1
            else:
                continue
            nodes = slice(a - 1, b + 1)                  # the gap stretch and the two nodes outside it
            k2 = 2.0 * mr[nodes] * _m0 * (Ec[nodes] - E) * (E - Ev[nodes]) / Egn[nodes] * _q / _hbar ** 2
            S, L = _wkb_segments(k2[:-1], k2[1:], xm_d[a - 1:b])
            fv = 1.0 / (1.0 + np.exp(np.clip((E - Efp[iv]) / kT, -700, 700)))
            fc = 1.0 / (1.0 + np.exp(np.clip((E - Efn[ic]) / kT, -700, 700)))
            dJ = pref * (fv - fc) * np.exp(-2.0 * S) / L * dE * _q
            J += dJ
            if abs(dJ) > best:
                best, best_path = abs(dJ), (float(x[iv] * 1e9), float(x[ic] * 1e9))
    return abs(J) * 1e-4, best_path


def effective_field(x_nm, F, electrons_forward: bool, dead_nm: float = DEAD_SPACE_NM):
    """Field that heats the carriers [same units as F]: F along the carrier
    path averaged over dead_nm (centred, shortened at the device ends), and
    0 where it points the wrong way. Electrons and holes move in opposite
    directions with opposite charge, so the same sign of field heats both."""
    x = np.asarray(x_nm, dtype=float)
    F = np.asarray(F, dtype=float)
    U = np.concatenate([[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(x))])   # int F dx
    lo = np.clip(x - 0.5 * dead_nm, x[0], x[-1])
    hi = np.clip(x + 0.5 * dead_nm, x[0], x[-1])
    Favg = (np.interp(hi, x, U) - np.interp(lo, x, U)) / np.maximum(hi - lo, 1e-12)
    heat = -Favg if electrons_forward else Favg        # electrons move against F
    return np.maximum(heat, 0.0)


def analyze(r, g) -> BreakdownAnalysis:
    """Breakdown and leakage of one solved state r (SolverResult) on its grid g."""
    x_cm = np.asarray(r.x_nm, dtype=float) * 1e-7
    F = np.asarray(r.E_field, dtype=float)                 # V/m
    Eg = np.asarray(r.Ec, dtype=float) - np.asarray(r.Ev, dtype=float)
    # electrons drift against the field; take the direction of the net drop
    dxm = np.diff(np.asarray(r.x_nm, dtype=float))
    drop = float(np.sum(0.5 * (F[1:] + F[:-1]) * dxm))
    fwd = drop < 0
    an, ap = ionization_coefficients(effective_field(r.x_nm, F, fwd) * 1e-2, Eg)
    an = np.where(effective_field(r.x_nm, F, fwd) > 0, an, 0.0)
    ap = np.where(effective_field(r.x_nm, F, fwd) > 0, ap, 0.0)
    I_n, I_p = ionization_integrals(x_cm, an, ap, electrons_forward=fwd)
    M_n = 1.0 / (1.0 - I_n) if I_n < 1.0 else float('inf')
    M_p = 1.0 / (1.0 - I_p) if I_p < 1.0 else float('inf')
    m_e = np.asarray(g.m_e, dtype=float)
    m_h = np.asarray(g.m_lh, dtype=float)
    tau_n = np.asarray(g.tau_n if getattr(g, 'tau_n', None) is not None else 1e-9, dtype=float)
    tau_p = np.asarray(g.tau_p if getattr(g, 'tau_p', None) is not None else 1e-9, dtype=float)
    G = tat_generation(r, np.asarray(g.Nc, dtype=float), np.asarray(g.Nv, dtype=float),
                       tau_n, tau_p, m_e, m_h)
    xc = x_cm
    J_tat = float(_q * np.sum(0.5 * (G[1:] + G[:-1]) * np.diff(xc)))
    J_btbt, path = btbt_current(r, m_e, m_h)
    M = max(M_n, M_p)
    i = int(np.argmax(np.abs(F)))
    return BreakdownAnalysis(
        V=float(r.V_applied), ion_integral_n=I_n, ion_integral_p=I_p, M_n=M_n, M_p=M_p,
        J_btbt=J_btbt, J_tat=J_tat, J_leak=(J_btbt + J_tat) * M,
        peak_field_MV_cm=float(abs(F[i]) * 1e-8), peak_field_nm=float(r.x_nm[i]),
        alpha_n=an, alpha_p=ap, G_tat=G, btbt_path=path)


@dataclass
class BreakdownResult:
    V_br: float                      # avalanche breakdown voltage (ionization integral = 1) [V]
    converged: bool
    message: str
    result: object                   # SolverResult at the last bias
    analysis: BreakdownAnalysis      # at the last bias
    history: List[BreakdownAnalysis] = field(default_factory=list)
    V_leak: Optional[float] = None   # bias where the leakage first reaches J_leak_limit
    J_leak_limit: float = 1.0        # [A/cm^2]


def find_breakdown_voltage(device, solve_kwargs: Optional[dict] = None, V_first: float = -5.0,
                           V_limit: float = 300.0, max_step: float = 40.0, tol: float = 0.01,
                           max_solves: int = 16, J_leak_limit: float = 1.0,
                           log_fn: Optional[Callable[[str], None]] = None) -> BreakdownResult:
    """Reverse bias at which the ionization integral reaches 1 (avalanche),
    and on the way the bias at which the tunnelling leakage passes
    J_leak_limit [A/cm^2]. ln(I) is close to linear in V, so a secant /
    false-position search on ln I takes a handful of solves. The sign of
    V_first sets which polarity is 'reverse'."""
    kw = dict(solve_kwargs or {})
    log = log_fn or (lambda s: None)
    g = device.grid
    hist: List[BreakdownAnalysis] = []
    pts: List[Tuple[float, float]] = []      # (V, ln I)
    last = {}

    def solve(V):
        log(f"Breakdown: solving V = {V:+.3f} V ...")
        r = device.solve(V_applied=V, **kw)
        a = analyze(r, g)
        hist.append(a)
        last['r'], last['a'] = r, a
        I = a.ion_integral
        pts.append((float(V), float(np.log(max(I, 1e-300)))))
        log(f"Breakdown: V = {V:+.3f} V   ionization integral {I:.4g}   J_BTBT {a.J_btbt:.3g}  "
            f"J_TAT {a.J_tat:.3g} A/cm^2   peak field {a.peak_field_MV_cm:.2f} MV/cm"
            + ("" if r.converged else "   (solve NOT converged)"))
        return pts[-1][1]

    def done(V, ok, msg):
        V_leak = None
        hs = sorted(hist, key=lambda h: abs(h.V))
        for h0, h1 in zip(hs, hs[1:]):
            if h0.J_leak < J_leak_limit <= h1.J_leak:
                l0, l1 = np.log(max(h0.J_leak, 1e-300)), np.log(h1.J_leak)
                V_leak = h0.V + (np.log(J_leak_limit) - l0) * (h1.V - h0.V) / (l1 - l0)
                break
        return BreakdownResult(V_br=float(V), converged=ok, message=msg, result=last['r'],
                               analysis=last['a'], history=hist, V_leak=V_leak,
                               J_leak_limit=J_leak_limit)

    sgn = -1.0 if V_first < 0 else 1.0
    f = solve(0.0)
    V = V_first
    for _ in range(max_solves - 1):
        f = solve(V)
        if abs(f) < tol:                    # |ln I| < tol: I within 1 %
            return done(V, last['r'].converged, "")
        below = [p for p in pts if p[1] < 0]
        above = [p for p in pts if p[1] > 0]
        if below and above:
            Va, fa = max(below, key=lambda p: sgn * p[0])           # closest to breakdown from below
            Vb, fb = min(above, key=lambda p: sgn * p[0])
            V_new = Va - fa * (Vb - Va) / (fb - fa)
        else:
            (V0, f0), (V1, f1) = pts[-2], pts[-1]
            if f1 == f0:
                V_new = V1 + sgn * max_step
            else:
                V_new = V1 - f1 * (V1 - V0) / (f1 - f0)
            step = float(np.clip(V_new - V1, -max_step, max_step))
            if sgn * step <= 0:            # ln I falling the wrong way: keep going in reverse
                step = sgn * max_step
            V_new = V1 + step
        V_new = float(np.clip(V_new, -V_limit, V_limit))
        if abs(V_new - V) < 1e-3:
            return done(V, last['r'].converged, "")
        if abs(V_new) >= V_limit and abs(V) >= V_limit:
            return done(V, False, f"no avalanche breakdown within |V| <= {V_limit:g} V")
        V = V_new
    return done(V, False, f"not found to {tol} in ln(I) within {max_solves} solves")

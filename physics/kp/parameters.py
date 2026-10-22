"""
k.p parameters for the 8-band Hamiltonian (physics.kp.hamiltonian), per grid
node, in one Cartesian form shared by both crystal systems.

Basis (per spin): |S>, |X>, |Y>, |Z>. Units: energies eV, lengths nm,
HB = hbar^2 / 2 m0 = 0.0381 eV nm^2. The bulk Hamiltonian is H1 + H2 of
Rinke et al., PRB 77, 075202 (2008), Appendix A:

  H1 (per spin)
      S-S  Ec + HB k^2
      S-X  i P2 kx,  S-Y  i P2 ky,  S-Z  i P1 kz
      X-X  XY0 + HB k^2,  Y-Y  XY0 + HB k^2,  Z-Z  Z0 + HB k^2
  H2 (remote bands)
      S-S  A2' (kx^2 + ky^2) + A1' kz^2
      X-X  L1' kx^2 + M1 ky^2 + M2 kz^2      (Y-Y: kx <-> ky)
      Z-Z  M3 (kx^2 + ky^2) + L2' kz^2
      X-Y  N1' kx ky
      X-Z  N2' kx kz - N3' kx,   Y-Z  N2' ky kz - N3' ky
  plus spin-orbit (Delta_so / 3) L.sigma on X, Y, Z.

with P_i^2 = HB Ep_i. The primed parameters are the six-band (Luttinger-
like) values with the free-electron term and the conduction-band coupling
taken out (Loewdin partitioning onto |S> at energy Eg above the valence
band):

  wurtzite (Chuang & Chang, PRB 54, 2491 (1996) notation, A_i in HB units)
      A1' = HB (1/m_e_par  - 1) - P1^2 / Eg
      A2' = HB (1/m_e_perp - 1) - P2^2 / Eg
      L1' = HB (A2 + A4 + A5 - 1) + P2^2 / Eg
      L2' = HB (A1 - 1)           + P1^2 / Eg
      M1  = HB (A2 + A4 - A5 - 1),  M2 = HB (A1 + A3 - 1),  M3 = HB (A2 - 1)
      N1' = HB 2 A5         + P2^2 / Eg
      N2' = HB sqrt(2) A6   + P1 P2 / Eg
      N3' = i sqrt(2) A7
    Rinke et al. print P1^2 in L1' and N1' and P2^2 in L2'. Loewdin
    partitioning gives the reverse: kx couples |S> to |X> through P2, so the
    kx^2 term of X-X and the kx ky term of X-Y take P2^2, and kz (|S>-|Z>,
    through P1) takes P1^2 in L2'. Check on bulk GaN: the in-plane A-band
    mass comes out 0.331 m0 with this form against -1/(A2+A4) = 0.329, and
    0.349 with the printed one (tests/physics/test_kp.py).

  zincblende (Luttinger gamma_i; P1 = P2 = P, Delta_cr = 0, N3' = 0)
      A'  = HB (1/m_e - 1) - P^2 (Eg + 2 D/3) / (Eg (Eg + D))
      L'  = -HB (gamma1 + 4 gamma2 + 1) + P^2 / Eg
      M   = -HB (gamma1 - 2 gamma2 + 1)
      N'  = -HB 6 gamma3 + P^2 / Eg
    (the familiar gamma1' = gamma1 - Ep / 3Eg, gamma2,3' = gamma2,3 - Ep / 6Eg.)

Spurious solutions. A negative coefficient of k^2 on |S| in the full 8x8
(S = 1/m_e - [P^2 term]/Eg < 0 in HB units) puts unphysical states in the
gap. Following Foreman, PRB 56, R12748 (1997), Ep is lowered until
S = S_MIN wherever it falls below; the primed valence parameters are then
recomputed from the same six-band values, so the band-edge masses are
unchanged. Each rescaling is recorded in KPParams.ep_rescaled.

Wurtzite dispersion parameters (m_e, A1-A7, Ep): Rinke et al., PRB 77,
075202 (2008), Table V (from quasiparticle OEPx(cLDA)+G0W0 band
structures), linear in composition. The crystal-field and spin-orbit
splittings, gaps and deformation potentials are the existing material
model's (physics.materials.algan), so the k = 0 band edges equal the ones
the classical solver uses. Zincblende: physics.materials.zincblende
(Vurgaftman, Meyer & Ram-Mohan 2001), interpolated with its own bowing.

Band edges. The |S> diagonal is the Gamma conduction edge Ec_G (= the grid
Ec, plus the distance from the lowest valley to Gamma in an indirect
zincblende alloy). XY0 and Z0 are chosen so that at k = 0 the three
valence doublets are exactly the material model's strained HH/LH/SO(CH)
energies (valence_band_energies), shifted so that the topmost one equals
the grid Ev:
  wurtzite    XY0 = Dcr + lambda + theta,  Z0 = lambda
              lambda = D1 ezz + 2 D2 exx, theta = D3 ezz + 2 D4 exx
  zincblende  XY0 = -P - Q - D/3,  Z0 = -P + 2Q - D/3
              P = a_v (2 exx + ezz), Q = -b (exx - ezz)
Biaxial strain has no shear component, so it enters the diagonal only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from physics.materials.alloys import MONOCLINIC, ZINCBLENDE, get_params

HB = 0.0380998212        # hbar^2 / 2 m0 [eV nm^2]
S_MIN = 1.0              # Foreman: smallest allowed |S> k^2 coefficient [HB units]

# Rinke et al., PRB 77, 075202 (2008), Table V.  (GaN, AlN, InN)
_WZ = {
    'me_par':  (0.186, 0.322, 0.065),
    'me_perp': (0.209, 0.329, 0.068),
    'A1': (-5.947, -3.991, -15.803),
    'A2': (-0.528, -0.311, -0.497),
    'A3': (5.414, 3.671, 15.251),
    'A4': (-2.512, -1.147, -7.151),
    'A5': (-2.510, -1.329, -7.060),
    'A6': (-3.202, -1.952, -10.078),
    'A7': (0.046, 0.026, 0.175),       # [eV Angstrom]
    'Ep_par':  (17.292, 16.972, 8.742),
    'Ep_perp': (16.265, 18.165, 8.809),
}


def _wz(key, x_Al, x_In):
    g, a, i = _WZ[key]
    return (1.0 - x_Al - x_In) * g + x_Al * a + x_In * i


@dataclass
class KPParams:
    """Per-node arrays (length N) of the 8-band parameters. Energies eV,
    k.p coefficients eV nm^2 (P: eV nm, N3: eV nm)."""
    crystal: str
    Ec: np.ndarray           # Gamma edge minus the grid conduction edge [eV]
    XY0: np.ndarray          # X, Y diagonal at k = 0, relative to the top valence band [eV]
    Z0: np.ndarray           # Z diagonal at k = 0, relative to the top valence band [eV]
    Dso: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    A1p: np.ndarray
    A2p: np.ndarray
    L1p: np.ndarray
    L2p: np.ndarray
    M1: np.ndarray
    M2: np.ndarray
    M3: np.ndarray
    N1p: np.ndarray
    N2p: np.ndarray
    N3: np.ndarray           # imaginary part of N3' [eV nm]
    ep_rescaled: list = field(default_factory=list)   # (node, Ep_old, Ep_new)

    def node(self, j: int) -> dict:
        return {k: (getattr(self, k)[j]) for k in _NODE_KEYS}


_NODE_KEYS = ('Ec', 'XY0', 'Z0', 'Dso', 'P1', 'P2', 'A1p', 'A2p', 'L1p', 'L2p',
              'M1', 'M2', 'M3', 'N1p', 'N2p', 'N3')


def _foreman_ep(Ep, inv_me, so_factor):
    """Ep lowered (if needed) so that inv_me - Ep * so_factor >= S_MIN."""
    s = inv_me - Ep * so_factor
    return Ep if s >= S_MIN else (inv_me - S_MIN) / so_factor


def node_params(crystal: str, x_Al: float, x_In: float, x_P: float, T: float,
                eps_xx: float, eps_zz: float) -> dict:
    """k.p parameters of one material at one strain, with the valence
    diagonal on the material model's own (strain-independent) energy scale
    and the conduction edge as an offset above the TOP valence band:
    returns XY0_rel, Z0_rel (valence diagonal relative to the topmost
    k = 0 valence level) and dEcG (Gamma edge minus the lowest conduction
    edge, 0 except in an indirect zincblende alloy)."""
    p = get_params(crystal, x_Al, x_In, x_P, T)
    out = {}
    if crystal == MONOCLINIC:
        # beta-Ga2O3: the uncoupled limit (P = 0) -- a parabolic conduction
        # band of mass m_e and three degenerate isotropic hole bands of mass
        # m_hh (gamma1 = 1/m_hh, gamma2 = gamma3 = 0), no spin-orbit split.
        g1 = 1.0 / float(p.m_hh)
        Lp = -HB * (g1 + 1.0)
        out.update(P1=0.0, P2=0.0, A1p=HB * (1.0 / p.m_e_par - 1.0), A2p=HB * (1.0 / p.m_e_perp - 1.0),
                   L1p=Lp, L2p=Lp, M1=Lp, M2=Lp, M3=Lp, N1p=0.0, N2p=0.0, N3=0.0)
        D = 0.0
        XY0 = Z0 = 0.0
        out['dEcG'] = 0.0
        out['Ep_used'], out['Ep_table'] = 0.0, 0.0
    elif crystal == ZINCBLENDE:
        from physics.materials.zincblende import _BIN, _interp, _weights, _check_comp
        x, y, v = _check_comp(x_Al, x_In, x_P)
        f, g = _weights(x, y, v)
        g1, g2, g3 = (_interp(lambda b, i=i: _BIN[b]['lutt'][i], 'lutt', f, g) for i in range(3))
        D = float(p.Delta_so)
        edges = p._valley_edges(eps_xx, eps_zz)
        EgG = float(p.Eg_valleys['G'])                  # unstrained Gamma gap at T
        meG = float(_interp(lambda b: _BIN[b]['meG'], 'meG', f, g))
        Ep0 = float(_interp(lambda b: _BIN[b]['Ep'], 'Ep', f, g))
        so = (EgG + 2.0 * D / 3.0) / (EgG * (EgG + D))
        Ep = _foreman_ep(Ep0, 1.0 / meG, so)
        P2 = HB * Ep
        P = np.sqrt(P2)
        out.update(P1=P, P2=P, A1p=HB * (1.0 / meG - 1.0) - P2 * so)
        out['A2p'] = out['A1p']
        Lp = -HB * (g1 + 4.0 * g2 + 1.0) + P2 / EgG
        M = -HB * (g1 - 2.0 * g2 + 1.0)
        Np = -HB * 6.0 * g3 + P2 / EgG
        out.update(L1p=Lp, L2p=Lp, M1=M, M2=M, M3=M, N1p=Np, N2p=Np, N3=0.0)
        Pst = p.a_v * (2.0 * eps_xx + eps_zz)
        Q = -p.b * (eps_xx - eps_zz)
        XY0, Z0 = -Pst - Q - D / 3.0, -Pst + 2.0 * Q - D / 3.0
        out['dEcG'] = float(edges['G'] - min(edges.values()))
        out['Ep_used'], out['Ep_table'] = Ep, Ep0
    else:
        D = float(p.Delta_so)
        Eg = float(p.Eg)
        mpar, mperp = _wz('me_par', x_Al, x_In), _wz('me_perp', x_Al, x_In)
        Ep1_0, Ep2_0 = _wz('Ep_par', x_Al, x_In), _wz('Ep_perp', x_Al, x_In)
        Ep1 = _foreman_ep(Ep1_0, 1.0 / mpar, 1.0 / Eg)
        Ep2 = _foreman_ep(Ep2_0, 1.0 / mperp, 1.0 / Eg)
        P1s, P2s = HB * Ep1, HB * Ep2
        A = {k: _wz(k, x_Al, x_In) for k in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')}
        out.update(
            P1=np.sqrt(P1s), P2=np.sqrt(P2s),
            A1p=HB * (1.0 / mpar - 1.0) - P1s / Eg,
            A2p=HB * (1.0 / mperp - 1.0) - P2s / Eg,
            L1p=HB * (A['A2'] + A['A4'] + A['A5'] - 1.0) + P2s / Eg,
            L2p=HB * (A['A1'] - 1.0) + P1s / Eg,
            M1=HB * (A['A2'] + A['A4'] - A['A5'] - 1.0),
            M2=HB * (A['A1'] + A['A3'] - 1.0),
            M3=HB * (A['A2'] - 1.0),
            N1p=HB * 2.0 * A['A5'] + P2s / Eg,
            N2p=HB * np.sqrt(2.0) * A['A6'] + np.sqrt(P1s * P2s) / Eg,
            N3=np.sqrt(2.0) * A['A7'] * 0.1,          # eV A -> eV nm
        )
        lam = p.D1 * eps_zz + 2.0 * p.D2 * eps_xx
        th = p.D3 * eps_zz + 2.0 * p.D4 * eps_xx
        XY0, Z0 = p.Delta_cr + lam + th, lam
        out['dEcG'] = 0.0
        out['Ep_used'], out['Ep_table'] = (Ep1, Ep2), (Ep1_0, Ep2_0)
    top = max(p.valence_band_energies(eps_xx, eps_zz)[:3])
    out.update(Dso=D, XY0_rel=float(XY0 - top), Z0_rel=float(Z0 - top))
    return out


def build_kp_static(g, i0: int = 0, i1: Optional[int] = None, T: Optional[float] = None) -> Optional[KPParams]:
    """KPParams for grid nodes [i0, i1) of GridData g, WITHOUT the
    electrostatic potential: Ec holds the distance from the grid's (lowest)
    conduction edge to the Gamma edge (0 except in indirect zincblende),
    XY0 and Z0 the valence diagonal relative to the topmost valence band.
    physics.kp.solver adds the grid Ec, Ev of the current iteration."""
    i1 = g.N if i1 is None else i1
    T = g.T if T is None else T
    crystal = getattr(g, 'crystal', 'wurtzite')
    nodes = getattr(g, 'crystal_nodes', None)
    xin = getattr(g, 'x_In', None)
    xp = getattr(g, 'x_P', None)
    cache = {}
    rows = []
    for i in range(i0, i1):
        key = (round(float(g.x_Al[i]), 6), round(float(xin[i]) if xin is not None else 0.0, 6),
               round(float(xp[i]) if xp is not None else 0.0, 6),
               round(float(g.eps_xx[i]), 9), round(float(g.eps_zz[i]), 9))
        cr = crystal if nodes is None else str(nodes[i])
        if (cr,) + key not in cache:
            cache[(cr,) + key] = node_params(cr, key[0], key[1], key[2], T, key[3], key[4])
        rows.append(cache[(cr,) + key])
    if not rows:
        return None

    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)
    rescaled = [(i0 + j, r['Ep_table'], r['Ep_used']) for j, r in enumerate(rows)
                if np.any(np.asarray(r['Ep_used']) != np.asarray(r['Ep_table']))]
    return KPParams(
        crystal=crystal, Ec=arr('dEcG'), XY0=arr('XY0_rel'), Z0=arr('Z0_rel'),
        Dso=arr('Dso'), P1=arr('P1'), P2=arr('P2'), A1p=arr('A1p'), A2p=arr('A2p'),
        L1p=arr('L1p'), L2p=arr('L2p'), M1=arr('M1'), M2=arr('M2'), M3=arr('M3'),
        N1p=arr('N1p'), N2p=arr('N2p'), N3=arr('N3'), ep_rescaled=rescaled,
    )

"""
Literature benchmark: 20 AlN/GaN/AlGaN papers (1969-2024) vs BandDiagramTool.

Each paper's published structure is rebuilt with the tool's DEFAULT physics and
the predicted observable is compared with the paper's MEASURED value.

Rules fixed before any simulation (see README in this folder):
  * Surface/Schottky barrier: the value stated in the paper if it gives one,
    else Ambacher et al., JAP 87, 334 (2000): e*phi_B = 0.84 + 1.3x eV.
  * Insulating AlN substrate/buffer: Fermi level mid-gap (barrier 3.0 eV).
  * UID GaN buffers: N_D = 1e16 cm^-3 background (ohmic bottom contact).
  * 300 K classical drift-diffusion/Poisson at V = 0 unless stated.
Scoring rubric: see score() below.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from devices.layer import AbruptLayer, GradedLayer, Contact  # noqa: E402
from devices.device import AlGaNDevice  # noqa: E402
from physics.materials.algan import get_AlGaN_params  # noqa: E402
from physics.polarization import compute_Psp, compute_strain, compute_Ppz  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ambacher_barrier(x):
    return 0.84 + 1.3 * x


def A(x, t, ND=0.0, NA=0.0, dx=None, strain=None):
    return AbruptLayer(x_Al=x, thickness_nm=t, n_doping=ND, p_doping=NA, dx_nm=dx,
                       custom_strain_xx=strain)


def G(x0, x1, t, ND=0.0, NA=0.0, dx=None):
    return GradedLayer(x_Al_start=x0, x_Al_end=x1, thickness_nm=t, n_doping=ND,
                       p_doping=NA, profile='linear', dx_nm=dx)


def solve(layers, top_barrier=None, bottom_barrier=None, T=300.0, dx=0.2, quantum=False):
    bot = (Contact('bottom', 'ohmic', 'Ti') if bottom_barrier is None
           else Contact('bottom', 'schottky', 'Ni', barrier_eV=bottom_barrier))
    top = (Contact('top', 'ohmic', 'Ni') if top_barrier is None
           else Contact('top', 'schottky', 'Ni', barrier_eV=top_barrier))
    d = AlGaNDevice(layers, [bot, top], T=T, dx_nm=dx)
    r = d.solve(V_applied=0.0, quantum=quantum)
    if not r.converged:
        raise RuntimeError('solver did not converge')
    return r


def sheet(r, z0, z1, carrier='n'):
    x = np.asarray(r.x_nm)
    c = np.asarray(r.n if carrier == 'n' else r.p)
    m = (x >= z0) & (x <= z1)
    return float(np.trapezoid(c[m], x[m] * 1e-7))


def zsum(layers):
    return sum(l.thickness_nm for l in layers)


# ---------------------------------------------------------------------------
# scoring (fixed before simulating)
# ---------------------------------------------------------------------------
def score(kind, pred, meas):
    if kind == 'energy':           # eV
        d = abs(pred - meas) * 1e3
        for lim, s in [(30, 10), (60, 9), (100, 8), (150, 7), (250, 5), (400, 3)]:
            if d <= lim:
                return s, f'{(pred - meas) * 1e3:+.0f} meV'
        return 1, f'{(pred - meas) * 1e3:+.0f} meV'
    if kind == 'thickness':        # nm
        d = abs(pred - meas)
        for lim, s in [(0.5, 10), (1.0, 8), (2.0, 6)]:
            if d <= lim:
                return s, f'{pred - meas:+.2f} nm'
        return 3, f'{pred - meas:+.2f} nm'
    # ratio quantities (densities, fields, charges)
    if pred <= 0 or meas <= 0:
        return 1, 'n/a'
    r = max(pred / meas, meas / pred) - 1
    sign = '+' if pred >= meas else '-'
    for lim, s in [(0.10, 10), (0.20, 9), (0.30, 8), (0.50, 6.5), (1.0, 5), (2.0, 3)]:
        if r <= lim:
            return s, f'{sign}{r:.0%}' if pred >= meas else f'{(pred / meas - 1):+.0%}'
    return 1, f'x{pred / meas:.2g}'


# ---------------------------------------------------------------------------
# the 20 papers
# ---------------------------------------------------------------------------
PAPERS = []


def paper(year, cite, device, confidence, note=''):
    def deco(fn):
        PAPERS.append(dict(year=year, cite=cite, device=device, confidence=confidence,
                           note=note, fn=fn))
        return fn
    return deco


@paper(1969, 'Maruska & Tietjen, APL 15, 327 (1969)', 'Vapor-grown GaN: room-temp band gap', 'High')
def p_maruska():
    return [('Eg(GaN) 300 K [eV]', get_AlGaN_params(0.0).Eg, 3.39, 'energy')]


@paper(1973, 'Yim et al., JAP 44, 292 (1973)', 'Epitaxial AlN: optical band gap', 'High',
       'early absorption edge; modern RT value ~6.0-6.1 eV')
def p_yim():
    return [('Eg(AlN) 300 K [eV]', get_AlGaN_params(1.0).Eg, 6.2, 'energy')]


@paper(1974, 'Monemar, PRB 10, 676 (1974)', 'GaN fundamental gap at 1.6 K (PLE)', 'High',
       'tests temperature dependence of Eg')
def p_monemar():
    # get_AlGaN_params_T only corrects Psp; Eg has no T dependence in the tool
    from physics.materials.algan import get_AlGaN_params_T
    return [('Eg(GaN) 1.6 K [eV]', get_AlGaN_params_T(0.0, 1.6).Eg, 3.503, 'energy')]


@paper(1992, 'Khan et al., APL 60, 3027 (1992)', 'First GaN/Al0.13GaN 2DEG (SdH)', 'Low',
       'barrier thickness/polarity not public: assumed 30 nm Ga-face Al0.13GaN on GaN')
def p_khan92():
    L = [A(0.0, 300, ND=1e16), A(0.13, 30)]
    r = solve(L, top_barrier=ambacher_barrier(0.13))
    return [('2DEG n_s [cm^-2]', sheet(r, 260, 330), 1e11, 'ratio')]


@paper(1997, 'Brunner et al., JAP 82, 5090 (1997)', 'AlGaN band-gap bowing (MBE, 0<=x<=1)', 'High',
       'compared as Eg(x=0.5) shift implied by bowing b: tool 0.7 eV vs measured 1.3+/-0.2 eV')
def p_brunner():
    e0, e1, emid = (get_AlGaN_params(x).Eg for x in (0.0, 1.0, 0.5))
    b_tool = 4 * (0.5 * (e0 + e1) - emid)
    pred = 0.5 * (e0 + e1) - b_tool / 4
    meas = 0.5 * (e0 + e1) - 1.3 / 4
    return [(f'Eg(x=0.5) via bowing (b_tool={b_tool:.2f}) [eV]', pred, meas, 'energy')]


@paper(1997, 'Li, Lin, Jiang, Khan & Chen, JVST B 15, 1117 (1997)',
       'n-Al0.1GaN(25nm)/n-GaN(25nm) 2DEG, dark Hall', 'Medium',
       'measured at 10 K (dark, read from Fig. 4); simulated at 77 K')
def p_li97():
    L = [A(0.0, 300, ND=1e15), A(0.0, 25, ND=1e17), A(0.1, 25, ND=5e17)]
    r = solve(L, top_barrier=ambacher_barrier(0.1), T=77.0)
    return [('2DEG n_s [cm^-2]', sheet(r, 300, 350), 0.85e12, 'ratio')]


@paper(1999, 'Ambacher et al., JAP 85, 3222 (1999)', 'Ga-face AlGaN/GaN 2DEG vs Al fraction', 'Medium',
       'barriers 20-65 nm in paper; 30 nm assumed')
def p_ambacher():
    out = []
    for x, meas in [(0.15, 6e12), (0.31, 2e13)]:
        r = solve([A(0.0, 300, ND=1e16), A(x, 30)], top_barrier=ambacher_barrier(x))
        out.append((f'2DEG n_s x={x} [cm^-2]', sheet(r, 260, 330), meas, 'ratio'))
    return out


@paper(1999, 'Grandjean et al., JAP 86, 3714 (1999)', 'GaN/Al0.27GaN QW built-in field', 'Medium',
       '~1 MV/cm reported at x=0.27; single 3 nm QW between thick barriers assumed')
def p_grandjean():
    L = [A(0.0, 200, ND=1e16), A(0.27, 50), A(0.0, 3, dx=0.05), A(0.27, 50)]
    r = solve(L, top_barrier=ambacher_barrier(0.27))
    x = np.asarray(r.x_nm)
    m = (x > 250.5) & (x < 252.5)
    F = float(np.mean(np.abs(np.asarray(r.E_field)[m]))) / 1e8  # MV/cm
    return [('QW field [MV/cm]', F, 1.0, 'ratio')]


@paper(2000, 'Ibbetson et al., APL 77, 250 (2000)', 'Al0.34GaN/GaN: critical barrier thickness', 'High',
       'surface donors at 1.65 eV (paper); onset defined as n_s > 1e11 cm^-2')
def p_ibbetson():
    tc = None
    for t in np.arange(1.0, 8.01, 0.25):
        r = solve([A(0.0, 300, ND=1e16), A(0.34, float(t), dx=0.05)], top_barrier=1.65)
        if sheet(r, 260, 300 + t) > 1e11:
            tc = float(t)
            break
    return [('critical thickness [nm]', tc if tc is not None else 99.0, 3.5, 'thickness')]


@paper(2000, 'Kaufmann et al., PRB 62, 10867 (2000)', 'MOCVD GaN:Mg, max hole density', 'High',
       'p_max ~6e17 at [Mg]~2e19; tool has no self-compensation model')
def p_kaufmann():
    r = solve([A(0.0, 600, NA=2e19)])
    x = np.asarray(r.x_nm)
    p = float(np.interp(300.0, x, np.asarray(r.p)))
    return [('hole density [cm^-3]', p, 6e17, 'ratio')]


@paper(2002, 'Jena et al., APL 81, 4395 (2002)', 'Polarization bulk-doped graded AlGaN + 2DEG control', 'High',
       '300 K Hall, Table I; surface at graded-layer top (Ambacher barrier)')
def p_jena02():
    out = []
    for xm, meas in [(0.1, 1.7e12), (0.2, 7.8e12), (0.3, 8.9e12)]:
        r = solve([A(0.0, 300, ND=1e16), G(0.0, xm, 100)], top_barrier=ambacher_barrier(xm))
        out.append((f'3DES n_s 0-{int(xm*100)}% [cm^-2]', sheet(r, 280, 400), meas, 'ratio'))
    r = solve([A(0.0, 300, ND=1e16), A(0.2, 20)], top_barrier=ambacher_barrier(0.2))
    out.append(('2DEG n_s Al0.2 20nm [cm^-2]', sheet(r, 260, 320), 7.8e12, 'ratio'))
    return out


@paper(2003, 'Heikman et al., JAP 93, 10114 (2003)', 'Al0.32GaN/GaN interface polarization charge', 'High',
       'paper extracts sigma = 1.6-1.7e13 cm^-2 from thickness-dependent Hall')
def p_heikman():
    x = np.array([0.0, 0.32])
    exx, ezz = compute_strain(x, 0.0)
    P = compute_Psp(x) + compute_Ppz(x, exx, ezz)
    sigma = (P[0] - P[1]) / 1.602176634e-19 * 1e-4
    return [('sigma_pol [cm^-2]', float(sigma), 1.65e13, 'ratio')]


@paper(2003, 'Adelmann et al., APL (2003), cond-mat/0304124', 'GaN/AlN QWs: internal field + PL energies', 'Medium',
       '9.2+/-1.0 MV/cm; PL 4.2 eV (0.7 nm) - 2.3 eV (2.6 nm); thick AlN barriers on AlN assumed')
def p_adelmann():
    L = [A(1.0, 100), A(0.0, 2.0, dx=0.02), A(1.0, 30)]
    r = solve(L, top_barrier=3.0, bottom_barrier=3.0)
    x = np.asarray(r.x_nm)
    m = (x > 100.4) & (x < 101.6)
    F = float(np.mean(np.abs(np.asarray(r.E_field)[m]))) / 1e8
    out = [('QW field [MV/cm]', F, 9.2, 'ratio')]
    # PL energies at the ends of the reported range (0.7 nm -> 4.2 eV,
    # 2.6 nm -> 2.3 eV): E1H1 from the Schrodinger-Poisson solve with the
    # strain-shifted GaN gap (GaN coherent on AlN). Added 2026-10-01 together
    # with the deformation-potential model; no exciton correction applied.
    from devices.layer import QuantumRegionMarker as Q
    C = [Contact('bottom', 'schottky', 'Ni', barrier_eV=3.0),
         Contact('top', 'schottky', 'Ni', barrier_eV=3.0)]
    for tw, meas in [(0.7, 4.2), (2.6, 2.3)]:
        L = [A(1.0, 60), Q('start'), A(1.0, 8, dx=0.05), A(0.0, tw, dx=0.02),
             A(1.0, 8, dx=0.05), Q('end'), A(1.0, 30)]
        rq = AlGaNDevice(L, C, dx_nm=0.2).solve(V_applied=0.0, quantum=True)
        out.append((f'PL E1H1 {tw} nm well [eV]', float(rq.qcse_transition_eV), meas, 'energy'))
    return out


@paper(2006, 'Taniyasu, Kasu & Makimoto, Nature 441, 325 (2006)', 'AlN p-i-n LED near-band-edge emission', 'High',
       'EL 210 nm = 5.90 eV (excitonic NBE); tool predicts ~Eg')
def p_taniyasu():
    return [('emission energy [eV]', get_AlGaN_params(1.0).Eg, 1239.84 / 210.0, 'energy')]


@paper(2007, 'Cao & Jena, APL 90, 182112 (2007)', 'AlN/GaN 2DEG vs AlN thickness', 'Medium',
       'RT n_s read from Fig. 2(a); free-surface barrier 3.0 eV (paper)')
def p_cao():
    out = []
    for t, meas in [(2.0, 5e12), (4.0, 3.5e13), (6.0, 5.0e13)]:
        r = solve([A(0.0, 143, ND=1e16), A(1.0, t, dx=0.05)], top_barrier=3.0)
        out.append((f'n_s AlN {t:.0f} nm [cm^-2]', sheet(r, 100, 143 + t), meas, 'ratio'))
    return out


@paper(2019, 'Chaudhuri et al., Science 365, 1454 (2019)', 'Undoped GaN(13nm)/AlN 2D hole gas', 'High',
       'p ~4e13 cm^-2; GaN surface barrier from Ambacher rule (0.84 eV)')
def p_chaudhuri():
    L = [A(1.0, 200), A(0.0, 13, dx=0.1)]
    r = solve(L, top_barrier=ambacher_barrier(0.0), bottom_barrier=3.0)
    return [('2DHG p_s [cm^-2]', sheet(r, 180, 213, 'p'), 4e13, 'ratio')]


@paper(2023, 'Mukhopadhyay et al., Crystals 13 (2023), arXiv:2304.05593',
       'Al0.36GaN/AlN/GaN HEMTs (5 samples)', 'High',
       'Fe-buffer effect of UID thickness t1 not modelled')
def p_mukho():
    out = []
    for name, t2, t3, meas in [('S1', 0.7, 21, 1.20e13), ('S2', 1.2, 21, 1.12e13),
                               ('S3', 0.7, 21, 1.33e13), ('S4', 0.7, 31, 1.46e13),
                               ('S5', 1.2, 31, 1.63e13)]:
        L = [A(0.0, 300, ND=1e16), A(0.0, 40), A(1.0, t2, dx=0.05), A(0.36, t3)]
        r = solve(L, top_barrier=ambacher_barrier(0.36))
        out.append((f'{name} (AlN {t2} nm, AlGaN {t3} nm) n_s', sheet(r, 300, 340 + t2 + t3), meas, 'ratio'))
    return out


@paper(2023, 'Knight et al., JAP 134, 185701 (2023)', 'AlxGaN(30nm)/GaN on SiC, x=0.07-0.42', 'Medium',
       'THz optical-Hall n_s read from Fig. 4 (+/-15%)')
def p_knight():
    out = []
    for x, meas in [(0.07, 2.3e12), (0.21, 6.5e12), (0.30, 9.6e12), (0.42, 1.45e13)]:
        r = solve([A(0.0, 300, ND=1e16), A(x, 30)], top_barrier=ambacher_barrier(x))
        out.append((f'n_s x={x} [cm^-2]', sheet(r, 260, 330), meas, 'ratio'))
    return out


@paper(2024, 'Chen et al., APL 124, 152111 (2024)', 'AlN/GaN/AlN QW-HEMTs on bulk AlN (4 samples)', 'High',
       'surface barrier 0.3 eV (paper); measured strain used for relaxed wells C, D')
def p_chen():
    out = []
    a_relaxed = 3.17  # measured GaN a for C, D
    s_gan = (a_relaxed - 3.189) / 3.189
    s_aln = (a_relaxed - 3.112) / 3.112
    for name, tb, tw, relaxed, meas in [('A', 6, 14, False, 1.99e13), ('B', 6, 20, False, 2.47e13),
                                        ('C', 6, 250, True, 3.68e13), ('D', 3, 250, True, 2.53e13)]:
        if relaxed:
            L = [A(1.0, 200), A(0.0, tw, strain=s_gan), A(1.0, tb, dx=0.05, strain=s_aln),
                 A(0.0, 1.0, dx=0.05, strain=s_gan)]
        else:
            L = [A(1.0, 200), A(0.0, tw), A(1.0, tb, dx=0.05), A(0.0, 1.0, dx=0.05)]
        r = solve(L, top_barrier=0.3, bottom_barrier=3.0)
        z = 200 + tw
        out.append((f'{name} (tb={tb}, tw={tw}) n_s', sheet(r, z - 30, z + tb), meas, 'ratio'))
    return out


@paper(2017, 'Zhu et al., APL (2017), arXiv:1704.03001', 'Polarization-doped graded Al0->0.2GaN, 600 nm', 'Medium',
       'measured n ~1e17 cm^-3 (with ~1e17 compensating centres reported)')
def p_zhu():
    r = solve([A(0.0, 300, ND=1e16), G(0.0, 0.2, 600)], top_barrier=ambacher_barrier(0.2), dx=0.5)
    x = np.asarray(r.x_nm)
    m = (x > 450) & (x < 750)
    return [('3D electron density [cm^-3]', float(np.mean(np.asarray(r.n)[m])), 1e17, 'ratio')]



@paper(2022, 'Rathkanthiwar et al., APL 120, 202105 (2022)',
       'Pseudomorphic Al0.6Ga0.4N (0.95-3.5 um) on bulk AlN: lattice + stress', 'High',
       'c from (00.2) XRD/RSM; stress from wafer curvature (Stoney). Tool has no C11/C12, '
       'so stress uses the tool strain with Vurgaftman C11/C12; tool has no relaxation model')
def p_rathkanthiwar():
    x = 0.6
    exx, ezz = compute_strain(np.array([x]), 1.0)
    exx, ezz = float(exx[0]), float(ezz[0])
    p = get_AlGaN_params(x)
    c_pred = p.c0 * (1.0 + ezz)
    c_meas = 4.982 * 3093.0 / 3030.0          # Fig. 6(a) RSM Qz ratio to AlN
    C11 = x * 396 + (1 - x) * 390
    C12 = x * 137 + (1 - x) * 145
    M = C11 + C12 - 2 * (p.C13 / 1e9) ** 2 / (p.C33 / 1e9)
    sig = abs(M * exx)
    out = [('strained c-lattice [A]', c_pred, c_meas, 'ratio')]
    for h, meas in [(0.95, 3.8), (1.8, 3.6), (3.5, 3.3)]:
        out.append((f'|stress| {h} um [GPa]', sig, meas, 'ratio'))
    return out

# ---------------------------------------------------------------------------
def main():
    PAPERS.sort(key=lambda p: p['year'])
    rows = []
    for p in PAPERS:
        t0 = time.time()
        try:
            pts = p['fn']()
            scored = []
            for lab, pred, meas, kind in pts:
                s, err = score(kind, pred, meas)
                scored.append(dict(label=lab, pred=pred, meas=meas, kind=kind, score=s, err=err))
            ps = float(np.mean([s['score'] for s in scored]))
            rows.append(dict({k: v for k, v in p.items() if k != 'fn'}, points=scored,
                             score=round(ps, 1), time=time.time() - t0))
            print(f"[{p['year']}] {p['cite']}: {ps:.1f}/10  ({time.time()-t0:.0f}s)", flush=True)
            for s in scored:
                print(f"     {s['label']}: pred={s['pred']:.4g} meas={s['meas']:.4g} {s['err']} -> {s['score']}",
                      flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rows.append(dict({k: v for k, v in p.items() if k != 'fn'}, error=repr(e), score=None))
            print(f"[{p['year']}] {p['cite']}: ERROR {e!r}", flush=True)
    json.dump(rows, open(os.path.join(HERE, 'lit_results.json'), 'w'), indent=1, default=float)
    sc = [r['score'] for r in rows if r.get('score') is not None]
    print(f'\nPapers scored: {len(sc)}/{len(rows)}   mean score {np.mean(sc):.1f}/10')


if __name__ == '__main__':
    main()

"""
Benchmark: BandDiagramTool vs nextnano++ (free edition) at V = 0.

Free-edition limits and how they are handled
--------------------------------------------
* max 100 grid points -> clustered non-uniform grid (nngrid.fit_grid)
* no strain module    -> pseudomorphic polarization charge is computed here
  from nextnano's OWN database values (Vurgaftman & Meyer 2003 / Ambacher
  2002, copied from database_free.nnp) with the standard linear-piezo
  model, and injected as fixed positive/negative "charge" impurities
  per grid control volume. Band edges are therefore unstrained in both
  tools (our tool has no deformation potentials either).
* no database edits   -> nextnano always runs with its own parameters.

Runs per device
---------------
NN        : nextnano++ reference (its database + Psp + Ppz)
OURS_def  : our tool with its defaults (Psp ON since 2026-10-01)
OURS_nnp  : our tool with nextnano's material parameters injected
            -> isolates solver numerics from material-parameter choices
All runs share contact barrier heights and dopant models (our composition-
dependent Ed(x)/Ea(x), g=2/4, passed to nextnano per layer).
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from nngrid import fit_grid, xgrid_block  # noqa: E402
from physics.materials.algan import (  # noqa: E402
    donor_ionization_energy, acceptor_ionization_energy)

kT = 0.025852  # eV at 300 K

# ---------------------------------------------------------------------------
# nextnano database values (database_free.nnp, nextnano++ 2025_12_17)
# ---------------------------------------------------------------------------
NN_DB = {
    'GaN': dict(a=3.189, C13=106.0, C33=398.0, e31=-0.35, e33=1.27, Psp=-0.034,
                eps=10.10, me=(0.206, 0.202), hh=(1.1, 1.6), lh=(1.1, 0.15), so=(0.15, 1.1)),
    'AlN': dict(a=3.112, C13=108.0, C33=373.0, e31=-0.50, e33=1.79, Psp=-0.090,
                eps=8.57, me=(0.32, 0.30), hh=(3.53, 10.42), lh=(3.53, 0.24), so=(0.25, 3.81)),
}
PSP_BOW = 0.021  # Ambacher 2002: Psp(x) = -0.090x - 0.034(1-x) + 0.021x(1-x)
_EDGES = np.loadtxt(os.path.join(HERE, 'nn_intrinsic_edges.txt'))  # x Ec HH LH SO


def _lin(key, x):
    g, a = NN_DB['GaN'][key], NN_DB['AlN'][key]
    if isinstance(g, tuple):
        return tuple(x * ai + (1 - x) * gi for gi, ai in zip(g, a))
    return x * a + (1 - x) * g


def nn_edges(x):
    """nextnano intrinsic Ec, HH, LH, SO [eV] (unstrained, 300 K)."""
    return [float(np.interp(x, _EDGES[:, 0], _EDGES[:, k])) for k in range(1, 5)]


def nn_polarization(x, x_sub, psp=True):
    """Polarization [C/m^2], pseudomorphic to x_sub (nextnano DB): Ppz (+ Psp if psp)."""
    x = np.asarray(x, float)
    a_sub = _lin('a', x_sub)
    a = _lin('a', x)
    exx = (a_sub - a) / a
    ezz = -2 * _lin('C13', x) / _lin('C33', x) * exx
    ppz = _lin('e33', x) * ezz + 2 * _lin('e31', x) * exx
    psp_ = _lin('Psp', x) + PSP_BOW * x * (1 - x)
    return (psp_ + ppz) if psp else ppz


def _mdos(ml_mt):
    ml, mt = ml_mt
    return (ml * mt * mt) ** (1 / 3)


def _NcNv_m0_300():
    from physics.constants import kB, h, m0
    return 2.0 * (2.0 * np.pi * m0 * kB * 300.0 / h ** 2) ** 1.5 * 1e-6  # cm^-3 for m=1


# ---------------------------------------------------------------------------
# Device specification
# ---------------------------------------------------------------------------
@dataclass
class L:
    x0: float
    t: float
    ND: float = 0.0
    NA: float = 0.0
    x1: float | None = None  # graded end composition (linear)
    dx: float | None = None  # our-tool per-layer spacing [nm]

    def x_at(self, s):  # s in [0,1]
        return self.x0 if self.x1 is None else self.x0 + (self.x1 - self.x0) * s


@dataclass
class Dev:
    key: str
    name: str
    ref: str
    layers: list
    top: tuple  # ('ohmic',) or ('schottky', barrier_eV)
    bottom: tuple = ('ohmic',)
    notes: str = ''
    fine: float = 0.15
    dx_default: float = 0.2
    sheet_windows: list = field(default_factory=list)  # [(label, z0, z1, 'n'|'p')]
    expected: dict = field(default_factory=dict)  # literature numbers


def interfaces(dev):
    z = [0.0]
    for l in dev.layers:
        z.append(z[-1] + l.t)
    return z


def x_profile(dev, z):
    """Al fraction at positions z (layer boundaries belong to the upper layer)."""
    zi = interfaces(dev)
    out = np.empty_like(np.asarray(z, float))
    for k, zz in enumerate(np.atleast_1d(z)):
        j = min(np.searchsorted(zi, zz, side='right') - 1, len(dev.layers) - 1)
        j = max(j, 0)
        l = dev.layers[j]
        out[k] = l.x_at((zz - zi[j]) / l.t)
    return out


# ---------------------------------------------------------------------------
# nextnano input generation
# ---------------------------------------------------------------------------
def _material(x0, x1=None):
    if x1 is not None and abs(x1 - x0) > 1e-9:
        return (f'ternary_linear{{ name = "Al(x)Ga(1-x)N" alloy_x = [{x0:.6g}, {x1:.6g}] '
                f'}}'), True
    if x0 <= 1e-9:
        return 'binary{ name = "GaN" }', False
    if x0 >= 1 - 1e-9:
        return 'binary{ name = "AlN" }', False
    return f'ternary_constant{{ name = "Al(x)Ga(1-x)N" alloy_x = {x0:.6g} }}', False


def nn_grid(dev):
    zi = interfaces(dev)
    return fit_grid(zi, fine=dev.fine, coarse=60.0, growth=1.3, max_pts=99)


def make_nn_input(dev, pts, with_pol=True):
    zi = interfaces(dev)
    x_sub = dev.layers[0].x0
    regs, imps = [], []
    for j, l in enumerate(dev.layers):
        a, b = zi[j], zi[j + 1]
        mat, graded = _material(l.x0, l.x1)
        if graded:
            mat = mat.replace('alloy_x = [', 'alloy_x = [').replace('] }', '] }')
            mat = (f'ternary_linear{{ name = "Al(x)Ga(1-x)N" alloy_x = [{l.x0:.6g}, {l.x1:.6g}] '
                   f'x = [{a:.6g}, {b:.6g}] }}')
        dop = ''
        xm = l.x0 if l.x1 is None else 0.5 * (l.x0 + l.x1)
        if l.ND > 0:
            imps.append(f'    donor{{ name = "nd{j}" energy = {donor_ionization_energy(xm):.5g} degeneracy = 2 }}')
            dop += f' doping{{ constant{{ name = "nd{j}" conc = {l.ND:.4g} }} }}'
        if l.NA > 0:
            imps.append(f'    acceptor{{ name = "na{j}" energy = {acceptor_ionization_energy(xm):.5g} degeneracy = 4 }}')
            dop += f' doping{{ constant{{ name = "na{j}" conc = {l.NA:.4g} }} }}'
        regs.append(f'    region{{ {mat} line{{ x = [{a:.6g}, {b:.6g}] }}{dop} }}')

    pol_info = []
    if with_pol:
        # Only the PIEZOELECTRIC part is injected: the free edition still computes
        # spontaneous (pyro) polarization itself (see total_charges.txt).
        # Cell-based deposition (nextnano assigns impurity densities per cell):
        #  - graded bulk charge in cell k: P(x_k+) - P(x_k+1 -)
        #  - abrupt sheet at node i: split half/half into cells i-1 and i
        ncell = len(pts) - 1
        Qc = np.zeros(ncell)
        tiny = 1e-6
        for k in range(ncell):
            a, b = pts[k], pts[k + 1]
            Pa = nn_polarization(x_profile(dev, [a + tiny]), x_sub, psp=False)[0]
            Pb = nn_polarization(x_profile(dev, [b - tiny]), x_sub, psp=False)[0]
            Qc[k] += Pa - Pb
        for i in range(1, len(pts) - 1):
            Pl = nn_polarization(x_profile(dev, [pts[i] - tiny]), x_sub, psp=False)[0]
            Pr = nn_polarization(x_profile(dev, [pts[i] + tiny]), x_sub, psp=False)[0]
            sig = Pl - Pr
            if abs(sig) > 1e-9:
                Qc[i - 1] += 0.5 * sig
                Qc[i] += 0.5 * sig
                pol_info.append((float(pts[i]), float(sig)))
        for k in range(ncell):
            w = (pts[k + 1] - pts[k]) * 1e-7  # cm
            dens = Qc[k] / 1.602176634e-19 * 1e-4 / w
            if abs(dens) < 1e12:
                continue
            nm = 'qpos' if dens > 0 else 'qneg'
            regs.append(f'    region{{ line{{ x = [{pts[k]:.8g}, {pts[k + 1]:.8g}] }} '
                        f'doping{{ constant{{ name = "{nm}" conc = {abs(dens):.6g} }} }} }}')

    L0, L1 = pts[0], pts[-1]
    regs.append(f'    region{{ contact{{ name = "bottom" }} line{{ x = [{L0:.6g}, {pts[1]:.6g}] }} }}')
    regs.append(f'    region{{ contact{{ name = "top" }} line{{ x = [{pts[-2]:.6g}, {L1:.6g}] }} }}')

    def contact(name, spec):
        if spec[0] == 'ohmic':
            return f'    ohmic{{ name = "{name}" bias = 0.0 }}'
        return f'    schottky{{ name = "{name}" bias = 0.0 barrier = {spec[1]:.6g} }}'

    return f'''global{{
    simulate1D{{}}
    crystal_wz{{ x_hkl = [0, 0, 1] y_hkl = [1, 0, 0] }}
    substrate{{ name = "GaN" }}
    temperature = 300
}}
impurities{{
{chr(10).join(imps)}
    charge{{ name = "qpos" type = positive }}
    charge{{ name = "qneg" type = negative }}
}}
contacts{{
{contact("bottom", dev.bottom)}
{contact("top", dev.top)}
}}
structure{{
{chr(10).join(regs)}
}}
grid{{
{xgrid_block(pts)}
}}
classical{{
    Gamma{{}} HH{{}} LH{{}} SO{{}}
    output_bandedges{{ averaged = no }}
    output_carrier_densities{{}}
}}
poisson{{ output_potential{{}} output_electric_field{{}} }}
run{{ poisson{{}} }}
''', pol_info


def run_nextnano(dev, pts=None):
    import nextnanopy as nn
    pts = nn_grid(dev) if pts is None else pts
    txt, pol = make_nn_input(dev, pts)
    d = os.path.join(HERE, 'nextnano_inputs')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{dev.key}.in')
    open(path, 'w').write(txt)
    t0 = time.time()
    nn.InputFile(path).execute(show_log=False, convergenceCheck=False)
    dt = time.time() - t0
    od = os.path.join(r'C:\nextnano_opt\output', dev.key, 'bias_00000')
    logp = os.path.join(r'C:\nextnano_opt\output', dev.key, f'{dev.key}.log')
    if not os.path.exists(os.path.join(od, 'bandedges.dat')):
        raise RuntimeError('nextnano failed: ' + open(logp, errors='ignore').read()[-1500:])
    log = open(logp, errors='ignore').read()
    be = np.loadtxt(os.path.join(od, 'bandedges.dat'), skiprows=1)
    ne = np.loadtxt(os.path.join(od, 'density_electron.dat'), skiprows=1)
    nh = np.loadtxt(os.path.join(od, 'density_hole.dat'), skiprows=1)
    # contact (metal) nodes are written as all-zero rows -> drop them
    be = be[np.any(np.abs(be[:, 1:5]) > 0, axis=1)]
    return dict(x=be[:, 0], Ec=be[:, 1], Ev=be[:, 2:5].max(axis=1),
                Ef=be[:, 5], xn=ne[:, 0], n=ne[:, 1] * 1e18, p=nh[:, 1] * 1e18,
                pts=pts, time=dt, pol=pol,
                converged=('not converged' not in log.lower()))


# ---------------------------------------------------------------------------
# Our tool
# ---------------------------------------------------------------------------
@contextmanager
def nextnano_params():
    """Temporarily replace our material database with nextnano's values."""
    import physics.materials.algan as A
    import physics.polarization as P
    import devices.grid_builder as G
    orig = (A.get_AlGaN_params, A.get_AlGaN_params_T, P.get_AlGaN_params,
            G.get_AlGaN_params, G.get_AlGaN_params_T, P.compute_Psp, G.compute_Psp)
    base = orig[0]
    unit = _NcNv_m0_300()

    def params(x):
        x = float(np.clip(x, 0, 1))
        p = base(x)
        Ec, hh, lh, so = nn_edges(x)
        Etop = max(hh, lh, so)
        me = _mdos(_lin('me', x))
        # nextnano delta = [D1, D2, D3]: GaN [0.010, 0.00567, 0.00567],
        # AlN [-0.169, 0.00633, 0.00633]; our model uses Delta_so = 3*D2.
        d_cr = x * (-0.169) + (1 - x) * 0.010
        d_so = 3 * (x * 0.00633 + (1 - x) * 0.00567)
        return replace(p, Eg=Ec - Etop, chi=-Ec, eps_r=_lin('eps', x),
                       Delta_cr=d_cr, Delta_so=d_so,
                       m_hh_dos=_mdos(_lin('hh', x)), m_lh_dos=_mdos(_lin('lh', x)),
                       m_so_dos=_mdos(_lin('so', x)),
                       m_e_par=me, m_e_perp=me, m_e_dos=me,
                       a0=_lin('a', x), Psp=_lin('Psp', x) + PSP_BOW * x * (1 - x),
                       e33=_lin('e33', x), e31=_lin('e31', x),
                       C13=_lin('C13', x) * 1e9, C33=_lin('C33', x) * 1e9)

    def params_T(x, T):
        return params(x)

    def psp(x, T=300.0):
        x = np.asarray(x, float)
        return _lin('Psp', x) + PSP_BOW * x * (1 - x)

    A.get_AlGaN_params = params; A.get_AlGaN_params_T = params_T
    P.get_AlGaN_params = params; G.get_AlGaN_params = params; G.get_AlGaN_params_T = params_T
    P.compute_Psp = psp; G.compute_Psp = psp
    try:
        yield
    finally:
        (A.get_AlGaN_params, A.get_AlGaN_params_T, P.get_AlGaN_params,
         G.get_AlGaN_params, G.get_AlGaN_params_T, P.compute_Psp, G.compute_Psp) = orig


def build_ours(dev, psp):
    from devices.layer import AbruptLayer, GradedLayer, Contact
    from devices.device import AlGaNDevice
    layers = []
    for l in dev.layers:
        if l.x1 is None:
            layers.append(AbruptLayer(x_Al=l.x0, thickness_nm=l.t, n_doping=l.ND,
                                      p_doping=l.NA, dx_nm=l.dx))
        else:
            layers.append(GradedLayer(x_Al_start=l.x0, x_Al_end=l.x1, thickness_nm=l.t,
                                      n_doping=l.ND, p_doping=l.NA, profile='linear',
                                      dx_nm=l.dx))
    ct = lambda pos, spec: Contact(position=pos,
                                   contact_type='ohmic' if spec[0] == 'ohmic' else 'schottky',
                                   metal='Ni',
                                   barrier_eV=None if spec[0] == 'ohmic' else spec[1])
    # strain band shifts off: nextnano++ free edition cannot apply strain to
    # its band edges, so both codes compare unstrained band structures
    d = AlGaNDevice(layers, [ct('bottom', dev.bottom), ct('top', dev.top)], T=300.0,
                    dx_nm=dev.dx_default, include_spontaneous_polarization=psp,
                    include_strain_band_shift=False)
    g = d.grid
    native_barrier = None
    if dev.top[0] == 'schottky':
        native_barrier = 5.10 - float(g.chi[-1])  # what Ni (Schottky-Mott) would give
    return d, native_barrier


def run_ours(dev, mode):
    """mode: 'def' (tool defaults), 'nopsp' (Psp off), 'nnp' (nextnano params)."""
    t0 = time.time()
    if mode == 'nnp':
        with nextnano_params():
            d, nb = build_ours(dev, psp=True)
            r = d.solve(V_applied=0.0, quantum=False)
    else:
        d, nb = build_ours(dev, psp=(mode != 'nopsp'))
        r = d.solve(V_applied=0.0, quantum=False)
    return dict(x=np.asarray(r.x_nm), Ec=np.asarray(r.Ec), Ev=np.asarray(r.Ev),
                Ef=np.asarray(r.Efn), n=np.asarray(r.n), p=np.asarray(r.p),
                converged=bool(r.converged), time=time.time() - t0,
                native_barrier=nb)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _uniq(x, y):
    x = np.asarray(x); y = np.asarray(y)
    o = np.argsort(x, kind='stable')
    x, y = x[o], y[o]
    keep = np.concatenate([[True], np.diff(x) > 1e-9])
    return x[keep], y[keep]


def sheet(x, c, z0, z1):
    x, c = _uniq(x, c)
    m = (x >= z0) & (x <= z1)
    if m.sum() < 2:
        return 0.0
    return float(np.trapezoid(c[m], x[m] * 1e-7))


def compare(dev, nn, us):
    """Band-edge errors evaluated at nextnano nodes (excluding exact interface nodes,
    where both codes have a discontinuity and the value is convention-dependent)."""
    zi = np.array(interfaces(dev))
    x = np.unique(nn['x'])
    far = np.min(np.abs(x[:, None] - zi[None, :]), axis=1) > 0.3
    xe = x[far]
    xn, Ecn = _uniq(nn['x'], nn['Ec']); _, Evn = _uniq(nn['x'], nn['Ev'])
    xu, Ecu = _uniq(us['x'], us['Ec']); _, Evu = _uniq(us['x'], us['Ev'])
    dEc = np.interp(xe, xu, Ecu) - np.interp(xe, xn, Ecn)
    dEv = np.interp(xe, xu, Evu) - np.interp(xe, xn, Evn)
    out = dict(rms_Ec=float(np.sqrt(np.mean(dEc ** 2))), max_Ec=float(np.max(np.abs(dEc))),
               p95_Ec=float(np.percentile(np.abs(dEc), 95)),
               rms_Ev=float(np.sqrt(np.mean(dEv ** 2))), max_Ev=float(np.max(np.abs(dEv))),
               p95_Ev=float(np.percentile(np.abs(dEv), 95)))
    out['sheets'] = {}
    for lab, z0, z1, car in dev.sheet_windows:
        s_nn = sheet(nn['xn'], nn[car], z0, z1)
        s_us = sheet(us['x'], us[car], z0, z1)
        out['sheets'][lab] = dict(nn=s_nn, ours=s_us,
                                  rel=(s_us - s_nn) / s_nn if s_nn > 1e9 else float('nan'))
    return out

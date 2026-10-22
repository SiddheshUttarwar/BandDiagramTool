"""
Tests for physics.breakdown: nonlocal interband tunnelling against Kane's
uniform-field result, the dead-space effective field, and the avalanche
breakdown of a GaN p+-n diode against the depletion approximation.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from devices.device import AlGaNDevice
from devices.layer import AbruptLayer, Contact
from physics.breakdown import (analyze, btbt_current, effective_field, ionization_coefficients,
                               ionization_integrals)
from physics.constants import h, hbar, m0, q
from physics.materials.alloys import get_params


@pytest.mark.parametrize("Eg, F, dx_nm", [(3.39, 3e8, 0.05), (5.3, 6e8, 0.2)])
def test_btbt_matches_kane_uniform_field(Eg, F, dx_nm):
    me, mh = 0.2, 0.3
    mr = 1.0 / (1.0 / me + 1.0 / mh)
    dV = Eg + 2.0
    x = np.arange(0.0, 80e-9, dx_nm * 1e-9)
    Ec = 1.0 - F * np.clip(x - 10e-9, 0.0, dV / F)
    r = SimpleNamespace(x_nm=x * 1e9, Ec=Ec, Ev=Ec - Eg, Efn=np.full_like(x, -20.0),
                        Efp=np.full_like(x, 20.0), T=300.0)
    J, _ = btbt_current(r, np.full_like(x, me), np.full_like(x, mh), dE=0.002)
    T0 = np.exp(-np.pi * np.sqrt(2 * mr * m0) * (Eg * q) ** 1.5 / (4 * q * hbar * F))
    L = hbar * np.pi * np.sqrt(Eg * q / (2 * mr * m0)) / (q * F)
    J_kane = 2 * q / h * (dV - Eg) * q * T0 / (4 * np.pi * L) * 1e-4
    assert J == pytest.approx(J_kane, rel=0.02)


def test_no_tunnelling_through_a_thin_high_field_layer():
    """1 nm at 5 MV/cm drops 0.5 V, far below the gap: no path."""
    x = np.arange(0.0, 50.0, 0.1)
    Ec = 3.0 - 0.5 * np.clip(x - 20.0, 0.0, 1.0)
    r = SimpleNamespace(x_nm=x, Ec=Ec, Ev=Ec - 3.39, Efn=np.zeros_like(x), Efp=np.zeros_like(x), T=300.0)
    assert btbt_current(r, np.full_like(x, 0.2), np.full_like(x, 0.3))[0] == 0.0


def test_dead_space_averages_out_a_superlattice():
    x = np.arange(0.0, 100.0, 0.1)
    F = np.where((x // 1.0) % 2 == 0, 4.5e8, -4.5e8)          # +-4.5 MV/cm, 1 nm layers
    inner = (x > 10) & (x < 90)                                 # window shortened at the ends
    assert np.max(effective_field(x, F, electrons_forward=False)[inner]) < 0.1e8


def test_gan_pn_avalanche_matches_depletion_approximation():
    Nd, Fm_target = 1e17, None
    p = get_params('wurtzite', 0.0)
    eps = p.eps_r * 8.8541878128e-14

    def I_dep(Fm):
        W = eps * Fm / (q * Nd)
        xx = np.linspace(0.0, W, 20001)
        an, ap = ionization_coefficients(Fm * (1 - xx / W), p.Eg)
        return max(ionization_integrals(xx, an, ap, True)), W

    lo, hi = 1e6, 1e7
    for _ in range(60):
        m = np.sqrt(lo * hi)
        lo, hi = (m, hi) if I_dep(m)[0] < 1 else (lo, m)
    Fm_target = lo
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=500, n_doping=1e19, dx_nm=10),
              AbruptLayer(x_Al=0.0, thickness_nm=2000, n_doping=Nd, dx_nm=5),
              AbruptLayer(x_Al=0.0, thickness_nm=300, p_doping=3e19, dx_nm=2)]
    d = AlGaNDevice(layers, [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Ni')], dx_nm=10)
    r = d.solve(V_applied=-204.0, quantum=False, verbose=False)
    a = analyze(r, d.grid)
    assert a.peak_field_MV_cm == pytest.approx(Fm_target * 1e-6, rel=0.01)
    assert a.ion_integral == pytest.approx(1.0, abs=0.03)

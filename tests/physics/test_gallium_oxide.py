"""
Tests for the monoclinic beta-(AlGa)2O3 material model
(physics.materials.gallium_oxide, .alloys) and its use in the grid builder
and the solvers: no polarization, a modulation-doped 2DEG, and a Schottky
diode that matches the depletion approximation.
"""
import numpy as np
import pytest

from devices.device import Device
from devices.grid_builder import build_grid
from devices.layer import AbruptLayer, Contact
from physics.materials.alloys import crystal_of, get_params, make_material, normalize_composition
from physics.materials.gallium_oxide import gallium_oxide_name, get_gallium_oxide_params
from physics.rsm import reflections_for

_Q = 1.602176634e-19
_EPS0 = 8.8541878128e-12


def test_ga2o3_reference_values():
    p = get_gallium_oxide_params(0.0)
    assert gallium_oxide_name(0.0) == 'Ga2O3'
    assert p.Eg == pytest.approx(4.85)
    assert p.chi == pytest.approx(4.00)
    assert p.eps_r == pytest.approx(10.2)
    assert p.Nc(300.0) == pytest.approx(3.7e18, rel=0.03)
    assert p.Nv(300.0) == pytest.approx(1.7e20, rel=0.03)
    assert p.Ed == pytest.approx(0.037, abs=0.003)       # hydrogenic Si donor
    assert p.Ea > 1.0                                    # Mg: deep, no p-type
    assert p.Psp == 0.0 and p.e31 == 0.0 and p.e33 == 0.0


def test_alloy_gap_and_offsets():
    g0, g2, g5 = (get_gallium_oxide_params(x) for x in (0.0, 0.2, 0.5))
    assert gallium_oxide_name(0.2) == '(Al0.20Ga0.80)2O3'
    assert g5.Eg == pytest.approx(5.80, abs=0.01)        # bowing 1.0 eV
    dEc, dEv = g0.chi - g2.chi, (g2.chi + g2.Eg) - (g0.chi + g0.Eg)
    assert dEc == pytest.approx(0.30, abs=0.02)          # CBO carries most of dEg
    assert 0.0 < dEv < 0.05
    assert g2.a0 < g0.a0                                 # Al shrinks the lattice


def test_temperature_lowers_the_gap():
    assert get_gallium_oxide_params(0.0, T=400.0).Eg == pytest.approx(4.85 - 0.11, abs=0.02)


def test_material_class():
    assert crystal_of('Ga2O3') == 'monoclinic'
    assert make_material('Ga2O3', 0.2).name == '(Al0.20Ga0.80)2O3'
    assert normalize_composition('Ga2O3', 0.3, 0.4, 0.5) == (0.3, 0.0, 0.0)
    with pytest.raises(ValueError):
        make_material('Ga2O3', 0.2, x_In=0.1)
    with pytest.raises(ValueError):
        AbruptLayer(x_Al=0.2, x_In=0.1, thickness_nm=10, material='Ga2O3')
    assert get_params('monoclinic', 0.2).Eg == make_material('Ga2O3', 0.2).band_gap()
    assert reflections_for('monoclinic') == {}


def test_ga2o3_on_gan_is_allowed():
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=20, material='Ga2O3'),
              AbruptLayer(x_Al=0.0, thickness_nm=20)]
    g = build_grid(layers, [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Au')], dx_nm=1.0)
    assert list(g.crystal_nodes[[0, -1]]) == ['monoclinic', 'wurtzite']


def test_grid_has_no_polarization_and_strained_barrier():
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=40, material='Ga2O3'),
              AbruptLayer(x_Al=0.2, thickness_nm=20, material='Ga2O3')]
    g = build_grid(layers, [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Au')], dx_nm=1.0)
    assert g.crystal == 'monoclinic'
    assert np.all(g.pol_rho == 0.0)
    assert np.all(g.Psp == 0.0) and np.all(g.Ppz == 0.0)
    assert g.eps_xx[-1] > 0.0                            # barrier in tension on Ga2O3
    assert g.eps_zz[-1] < 0.0


def test_modfet_two_dimensional_electron_gas():
    """Si delta-doping behind a spacer gives a 2DEG of ~1e12 cm^-2 with
    quantized subbands, as in (010) AlGaO/GaO MODFETs."""
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=200, material='Ga2O3', n_doping=1e15),
              AbruptLayer(x_Al=0.0, thickness_nm=30, material='Ga2O3', dx_nm=0.2),
              AbruptLayer(x_Al=0.2, thickness_nm=4, material='Ga2O3', dx_nm=0.2),
              AbruptLayer(x_Al=0.2, thickness_nm=2, material='Ga2O3', n_doping=3e19, dx_nm=0.2),
              AbruptLayer(x_Al=0.2, thickness_nm=24, material='Ga2O3', dx_nm=0.5)]
    dev = Device(layers=layers, contacts=[Contact('bottom', 'ohmic', 'Ti'),
                                         Contact('top', 'schottky', 'Ni', barrier_eV=1.4)], dx_nm=1.0)
    r = dev.solve(quantum=False, verbose=False)
    assert r.converged
    x, n = np.asarray(r.x_nm), np.asarray(r.n)
    well = (x > 200) & (x < 231)
    ns = np.trapezoid(n[well], x[well] * 1e-7)
    assert 0.8e12 < ns < 3e12
    assert x[well][np.argmax(n[well])] > 225             # piled up at the interface


def test_schottky_diode_matches_depletion_approximation():
    """Ni / 2e16 cm^-3 Ga2O3: depletion width and peak field at -10 V
    against the abrupt depletion approximation."""
    Nd, barrier, V = 2e16, 1.1, -10.0
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=300, material='Ga2O3', n_doping=5e18),
              AbruptLayer(x_Al=0.0, thickness_nm=1500, material='Ga2O3', n_doping=Nd)]
    dev = Device(layers=layers, contacts=[Contact('bottom', 'ohmic', 'Ti'),
                                         Contact('top', 'schottky', 'Ni', barrier_eV=barrier)], dx_nm=5.0)
    r = dev.solve(V_applied=V, quantum=False, verbose=False, gate_bias=False)
    assert r.converged
    eps = 10.2 * _EPS0
    W = np.sqrt(2.0 * eps * (r.V_bi - V - 0.0259) / (_Q * Nd * 1e6))
    F_max = _Q * Nd * 1e6 * W / eps
    assert np.max(np.abs(r.E_field)) == pytest.approx(F_max, rel=0.03)
    assert np.trapezoid(r.p, np.asarray(r.x_nm) * 1e-7) < 1.0     # deep depletion, no inversion


def test_ga2o3_inside_algan():
    """Ga2O3 grown in Al0.6Ga0.4N: one stack, two crystal systems. The
    nitride polarization ends at each oxide interface (sheet charge -/+P),
    the band offsets follow the electron affinities (Ga2O3 conduction band
    0.76 eV below Al0.6Ga0.4N), composition is not blended across the
    interface, and the AlGaN above the oxide is relaxed on it."""
    from physics.materials.alloys import get_params
    layers = [AbruptLayer(x_Al=0.3, thickness_nm=100, n_doping=1e18),
              AbruptLayer(x_Al=0.6, thickness_nm=20, dx_nm=0.2),
              AbruptLayer(x_Al=0.0, thickness_nm=10, material='Ga2O3', dx_nm=0.2),
              AbruptLayer(x_Al=0.6, thickness_nm=20, dx_nm=0.2),
              AbruptLayer(x_Al=0.6, thickness_nm=50, n_doping=1e18)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Ti')]
    g = build_grid(layers, contacts, dx_nm=1.0)
    ox = g.crystal_nodes == 'monoclinic'
    assert g.crystal == 'wurtzite' and ox.sum() > 0
    assert np.all(g.x_Al[ox] == 0.0) and np.all(g.Psp[ox] == 0.0) and np.all(g.Ppz[ox] == 0.0)
    algan = get_params('wurtzite', 0.6)
    assert g.chi[ox].mean() - g.chi[(g.x_nm > 105) & (g.x_nm < 118)].mean() == pytest.approx(
        get_gallium_oxide_params(0.0).chi - algan.chi + 0.0, abs=0.05)
    assert g.eps_xx[(g.x_nm > 105) & (g.x_nm < 118)].mean() > 0.0       # Al0.6 on Al0.3: tensile
    assert np.all(np.abs(g.eps_xx[g.x_nm > 132]) < 1e-12)                # relaxed on the oxide
    P = (algan.Psp + 0.0)                                                # bottom AlGaN strained, top relaxed
    sig = dict(zip([round(float(g.x_nm[i])) for i in g.interface_indices], g.interface_sigmas))
    assert len([k for k in sig if 119 <= k <= 131]) == 2
    assert sig[min(k for k in sig if k >= 119)] < 0 < sig[max(k for k in sig if k <= 131)]
    assert abs(sig[max(k for k in sig if k <= 131)]) == pytest.approx(abs(P), rel=0.02)
    r = Device(layers, contacts, dx_nm=1.0).solve(quantum=False, verbose=False)
    assert r.converged
    assert r.crystal_nodes is not None


def test_zincblende_still_cannot_mix():
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=20, material='Ga2O3'),
              AbruptLayer(x_Al=0.0, thickness_nm=20, material='AlGaAs')]
    with pytest.raises(ValueError, match='zincblende'):
        build_grid(layers, [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Au')], dx_nm=1.0)

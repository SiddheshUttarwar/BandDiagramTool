"""
Tests for the zincblende arsenide / phosphide material model
(physics.materials.zincblende, .alloys), its use in the grid builder, and
the SRH / radiative / Auger recombination split.
"""
import numpy as np
import pytest

from devices.device import Device
from devices.grid_builder import build_grid
from devices.layer import AbruptLayer, GradedLayer, Contact
from physics.drift_diffusion import compute_recombination_components
from physics.materials.alloys import get_params, make_material, normalize_composition
from physics.materials.zincblende import get_zincblende_params, zincblende_name
from physics.rsm import simulate_rsm

OHMIC = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Au')]


def test_binaries_at_room_temperature():
    # (x_Al, x_In, x_P) -> name, gap [eV], lowest valley, lattice constant [A]
    for comp, name, gap, valley, a in [
            ((0, 0, 0), 'GaAs', 1.42, 'G', 5.6533), ((1, 0, 0), 'AlAs', 2.16, 'X', 5.6611),
            ((0, 1, 0), 'InAs', 0.354, 'G', 6.0583), ((0, 0, 1), 'GaP', 2.27, 'X', 5.4505),
            ((1, 0, 1), 'AlP', 2.49, 'X', 5.4672), ((0, 1, 1), 'InP', 1.353, 'G', 5.8697)]:
        p = get_zincblende_params(*comp)
        assert zincblende_name(*comp) == name
        assert p.Eg == pytest.approx(gap, abs=0.01)
        assert p.valley == valley
        assert p.a0 == pytest.approx(a, abs=1e-4)
        assert p.Psp == 0.0 and p.e31 == 0.0 and p.e33 == 0.0


def test_gaas_reference_values():
    p = get_zincblende_params(0, 0, 0)
    assert p.chi == pytest.approx(4.07, abs=1e-6)
    assert p.Nc(300.0) == pytest.approx(4.4e17, rel=0.05)
    assert p.Nv(300.0) == pytest.approx(9.5e18, rel=0.2)
    assert p.ni(300.0) == pytest.approx(2.1e6, rel=0.3)
    assert get_zincblende_params(0, 0, 0, T=0.0).Eg == pytest.approx(1.519)


def test_alloys_and_band_offsets():
    ingaas = get_zincblende_params(0, 0.53, 0)
    inp = get_zincblende_params(0, 1, 1)
    assert zincblende_name(0, 0.53, 0) == 'In0.53Ga0.47As'
    assert ingaas.Eg == pytest.approx(0.74, abs=0.01)
    assert ingaas.a0 == pytest.approx(inp.a0, rel=5e-4)              # lattice matched to InP
    assert ingaas.chi - inp.chi == pytest.approx(0.25, abs=0.04)     # conduction-band offset
    gaas, algaas = get_zincblende_params(0, 0, 0), get_zincblende_params(0.3, 0, 0)
    dEc = gaas.chi - algaas.chi
    dEv = (algaas.chi + algaas.Eg) - (gaas.chi + gaas.Eg)
    assert dEc / (dEc + dEv) == pytest.approx(0.62, abs=0.04)        # the ~60:40 split
    assert get_zincblende_params(0, 0.49, 1).Eg == pytest.approx(1.9, abs=0.03)   # InGaP on GaAs
    assert get_zincblende_params(0.6, 0, 0).valley != 'G'            # AlGaAs is indirect at high Al


def test_strain_splits_heavy_and_light_holes():
    p = get_zincblende_params(0, 0.2, 0)
    exx = (5.65325 - p.a0) / p.a0
    ezz = -2.0 * p.C13 / p.C33 * exx
    assert exx < 0
    d_hh, d_lh, d_so, _, _ = p.valence_offsets_from_top(exx, ezz)
    assert d_hh == 0.0 and d_lh > 0.05                               # compressive: heavy hole on top
    d_hh_t, d_lh_t, _, _, _ = p.valence_offsets_from_top(-exx, -ezz)
    assert d_lh_t == 0.0 and d_hh_t > 0.05                           # tensile: light hole on top
    assert p.valence_offsets_from_top()[:3] == pytest.approx((0.0, 0.0, p.Delta_so))
    dEc, dEv = p.strain_band_shifts(exx, ezz)
    assert dEc - dEv > 0                                             # compression widens the gap


def test_material_classes():
    assert make_material('InGaP', x_In=1.0, x_P=1.0).name == 'InP'
    assert normalize_composition('InGaP', 0.3, 0.5, 0.2) == (0.0, 0.5, 1.0)
    assert AbruptLayer(0.0, 10, material='InGaP', x_In=0.49).x_P == 1.0
    assert AbruptLayer(0.3, 10, material='AlGaAs').crystal == 'zincblende'
    assert AbruptLayer(0.3, 10).crystal == 'wurtzite'
    with pytest.raises(ValueError):
        AbruptLayer(0.0, 10, material='AlGaAs', x_In=0.2)
    assert get_params('wurtzite', 0.3).Psp != 0.0


def test_grid_has_no_polarization_and_rejects_mixed_stacks():
    layers = [AbruptLayer(0.0, 50, material='AlGaAs'),
              GradedLayer(0.0, 0.3, 20, material='AlGaAs'),
              AbruptLayer(0.0, 8, material='InGaAs', x_In=0.2),
              AbruptLayer(0.3, 20, material='AlGaAs', n_doping=1e18)]
    g = build_grid(layers, OHMIC, dx_nm=0.5)
    assert g.crystal == 'zincblende'
    assert not np.any(g.P_total) and not np.any(g.pol_rho) and g.interface_sigmas == []
    assert g.eps_xx.min() == pytest.approx(-0.014, abs=0.002)        # the InGaAs well
    assert np.all(g.eps_zz * g.eps_xx <= 0)
    assert g.mu_n[0] == pytest.approx(8500.0) and g.B_rad[0] == pytest.approx(7.2e-10)
    with pytest.raises(ValueError, match='one crystal system'):
        build_grid([AbruptLayer(0.0, 50), AbruptLayer(0.3, 20, material='AlGaAs')], OHMIC)


def test_recombination_components_and_overrides():
    n, p, ni = np.array([1e17]), np.array([1e17]), np.array([1e6])
    rec = compute_recombination_components(n, p, ni, tau_n=2e-9, tau_p=1e-9, B_rad=1e-10, C_n=1e-30, C_p=3e-30)
    assert rec.R_srh[0] == pytest.approx(1e34 / (1e-9 * 1e17 + 2e-9 * 1e17), rel=1e-6)
    assert rec.R_rad[0] == pytest.approx(1e-10 * 1e34, rel=1e-6)
    assert rec.R_aug[0] == pytest.approx((1e-30 + 3e-30) * 1e17 * 1e34, rel=1e-6)
    default = compute_recombination_components(n, p, ni)             # nitride defaults, as before
    assert default.R_rad[0] == pytest.approx(1e-11 * 1e34, rel=1e-6)

    layers = [AbruptLayer(0.0, 50, material='AlGaAs')]
    g = build_grid(layers, OHMIC, dx_nm=1.0, recombination={'tau_n': 5e-9, 'B_rad': 2e-10})
    assert np.all(g.tau_n == 5e-9) and np.all(g.B_rad == 2e-10) and np.all(g.tau_p == 1e-9)
    with pytest.raises(ValueError):
        build_grid(layers, OHMIC, recombination={'tau': 1e-9})


def test_double_heterostructure_under_bias():
    """Carriers injected into a GaAs layer between AlGaAs barriers recombine
    there: the recombination integrated over the device equals the current."""
    device = Device(layers=[AbruptLayer(0.3, 150, material='AlGaAs', n_doping=1e18),
                            AbruptLayer(0.0, 80, material='AlGaAs'),
                            AbruptLayer(0.3, 150, material='AlGaAs', p_doping=1e18)],
                    contacts=OHMIC, dx_nm=2.0)
    r = device.solve(V_applied=1.2, quantum=False)
    assert r.converged and r.crystal == 'zincblende'
    x_cm = r.x_nm * 1e-7
    integ = getattr(np, 'trapezoid', None) or np.trapz
    j_rec = 1.602176634e-19 * integ(r.R_srh + r.R_rad + r.R_aug, x_cm)
    assert j_rec == pytest.approx(abs(r.J_total), rel=0.05)
    assert r.R_rad.min() > -1e-9 * r.R_rad.max()                      # no generation (contact nodes: roundoff)
    mid = len(r.x_nm) // 2
    assert r.Efn[mid] - r.Efp[mid] == pytest.approx(1.2, abs=0.05)

    slow = Device(layers=device.layers, contacts=OHMIC, dx_nm=2.0, recombination={'tau_n': 1e-7, 'tau_p': 1e-7})
    r2 = slow.solve(V_applied=1.2, quantum=False)
    assert abs(r2.J_total) < abs(r.J_total)                          # longer SRH lifetime, less current


def test_quantum_well_and_reciprocal_space_map():
    device = Device(layers=[AbruptLayer(0.0, 40, material='InGaP', x_In=1.0),
                            AbruptLayer(0.0, 8, material='InGaAs', x_In=0.53),
                            AbruptLayer(0.0, 40, material='InGaP', x_In=1.0)],
                    contacts=OHMIC, dx_nm=0.25)
    r = device.solve(quantum=True)
    assert r.converged
    assert r.qcse_transition_eV == pytest.approx(0.80, abs=0.03)     # ~1.55 um
    assert r.qcse_overlap > 0.8                                      # no built-in field
    rsm = simulate_rsm(r, (2, 2, 4), alloys=[(0.0, 1.0, 1.0), (0.0, 0.53, 0.0)])
    assert not rsm.symmetric
    assert {name for name, _, _ in rsm.relaxed_points} == {'InP', 'In0.53Ga0.47As'}
    assert rsm.qx_substrate == pytest.approx(2.0 * np.pi * np.sqrt(8.0) / 5.8697, rel=1e-4)


def test_no_recombination_at_equilibrium():
    """Detailed balance: with coinciding quasi-Fermi levels the three rates
    are exactly zero, also for degenerate and quantum-mechanical densities
    (where n p differs from Nc Nv exp(-Eg/kT))."""
    well = Device(layers=[AbruptLayer(0.0, 40, material='InGaP', x_In=1.0, n_doping=2e18),
                          AbruptLayer(0.0, 8, material='InGaAs', x_In=0.53),
                          AbruptLayer(0.0, 40, material='InGaP', x_In=1.0, n_doping=2e18)],
                  contacts=OHMIC, dx_nm=0.25)
    for quantum in (False, True):
        r = well.solve(quantum=quantum)
        assert r.converged
        for rate in (r.R_srh, r.R_rad, r.R_aug):
            assert not np.any(rate)
    # the same form, away from equilibrium, is the textbook n p - ni^2
    n, p, kT = np.array([3e16]), np.array([2e15]), 0.025852
    ni = np.array([2e6])
    split = kT * np.log(n * p / ni ** 2)
    a = compute_recombination_components(n, p, ni)
    b = compute_recombination_components(n, p, None, split_eV=split, kT_eV=kT)
    assert b.R_srh[0] == pytest.approx(a.R_srh[0], rel=1e-9)
    assert b.R_rad[0] == pytest.approx(a.R_rad[0], rel=1e-9)
    assert b.R_aug[0] == pytest.approx(a.R_aug[0], rel=1e-9)
    # reverse bias: net generation
    c = compute_recombination_components(n, p, None, split_eV=np.array([-0.2]), kT_eV=kT)
    assert c.R_srh[0] < 0 and c.R_rad[0] < 0


def test_flat_quasi_fermi_levels_report_no_rates():
    device = Device(layers=[AbruptLayer(0.0, 100, material='AlGaAs', n_doping=1e17),
                            AbruptLayer(0.0, 100, material='AlGaAs', p_doping=1e17)], contacts=OHMIC, dx_nm=2.0)
    r = device.solve(V_applied=0.5, quantum=False, flat_qfl=True)
    assert r.R_srh is None and r.R_rad is None and r.R_aug is None
    assert device.solve(V_applied=0.5, quantum=False).R_srh is not None

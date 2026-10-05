"""Growth under UV illumination (physics.dqfl): the limits the model must reproduce."""
import numpy as np
import pytest

from devices.layer import AbruptLayer, QuantumRegionMarker
from physics.dqfl import simulate_illumination, surface_excess


def _one(**kw):
    return simulate_illumination([AbruptLayer(x_Al=0.0, thickness_nm=500, **kw)]).layers[0]


def test_no_light_changes_nothing():
    r = simulate_illumination([AbruptLayer(x_Al=0.0, thickness_nm=500, p_doping=1e19)], power_W_cm2=0.0)
    la = r.layers[0]
    assert la.Ep_eV == pytest.approx(0.0, abs=1e-9)
    assert all(d.factor == pytest.approx(1.0) for d in la.defects)


def test_photovoltage_is_kT_ln_np_over_ni2():
    la = _one(n_doping=1e18)
    kT = 8.617333262e-5 * (1040 + 273.15)
    assert la.Ep_eV == pytest.approx(kT * np.log((la.n0 + la.excess) * (la.p0 + la.excess) / (la.n0 * la.p0)))
    assert la.n0 * la.p0 == pytest.approx(la.ni ** 2, rel=1e-6)


def test_p_type_shallow_donor_is_suppressed_deep_one_is_not():
    """Mg:GaN (Klump et al. 2020): H, shallow, gains most of the
    photovoltage; V_N, deep near the valence band, gains nothing."""
    la = _one(p_doping=6e18)
    h, vn = la.defects
    assert h.dE_eV > 0.5 * la.Ep_eV and h.factor < 0.05
    assert vn.dE_eV == pytest.approx(0.0, abs=5e-3) and vn.factor == pytest.approx(1.0, abs=0.05)


def test_carbon_in_n_gan_is_reduced_most_at_low_doping():
    """C_N in Si:GaN: a several-fold reduction at 1e16 (cf. about 4.5-fold
    in Reddy/Szymanski, Appl. Phys. Express), fading at high doping."""
    low, high = _one(n_doping=1e16).defects[0], _one(n_doping=1e19).defects[0]
    assert 0.05 < low.factor < 0.6
    assert high.factor > 0.6


def test_undoped_and_marker_layers():
    r = simulate_illumination([AbruptLayer(x_Al=1.0, thickness_nm=100), QuantumRegionMarker('start'),
                               AbruptLayer(x_Al=0.6, thickness_nm=50, n_doping=1e18)])
    assert [la.doping for la in r.layers] == ['undoped', 'n']
    assert r.layers[0].defects == [] and len(r.layers[1].defects) == 2
    assert (r.layers[1].z0_nm, r.layers[1].z1_nm) == (100.0, 150.0)


def test_capture_asymmetry_only_moves_levels_far_from_the_minority_band():
    layer = [AbruptLayer(x_Al=0.65, thickness_nm=600, n_doping=2e18)]
    sym = simulate_illumination(layer, T_growth_C=1100, capture_asymmetry=1.0).layers[0].defects[0]
    asym = simulate_illumination(layer, T_growth_C=1100, capture_asymmetry=100.0).layers[0].defects[0]
    assert asym.w_p > sym.w_p and asym.factor < sym.factor <= 1.0


def test_light_below_the_gap_does_nothing_and_above_it_does():
    """405 nm (3.06 eV) is above the gap of GaN at growth temperature and
    below that of Al0.6Ga0.4N: one lamp, two outcomes."""
    stack = [AbruptLayer(x_Al=0.0, thickness_nm=200, n_doping=1e16),
             AbruptLayer(x_Al=0.6, thickness_nm=200, n_doping=1e16)]
    gan, algan = simulate_illumination(stack, wavelength_nm=405.0).layers
    assert gan.absorption_cm > 0 and gan.Ep_eV > 0.05 and gan.defects[0].factor < 1.0
    assert algan.absorption_cm == 0.0 and algan.excess == 0.0
    assert algan.Ep_eV == pytest.approx(0.0, abs=1e-9)
    assert all(d.factor == pytest.approx(1.0) for d in algan.defects)


def test_same_power_gives_more_photons_at_longer_wavelength():
    layer = [AbruptLayer(x_Al=0.0, thickness_nm=200, n_doping=1e18)]
    uv = simulate_illumination(layer, wavelength_nm=250.0).layers[0]
    blue = simulate_illumination(layer, wavelength_nm=350.0).layers[0]
    assert blue.excess == pytest.approx(uv.excess * 350.0 / 250.0, rel=1e-6)


def test_diffusion_spreads_the_carriers_over_the_diffusion_length():
    """S = 0: the surface density is the local value divided by (1 + alpha L);
    surface recombination lowers it further; no diffusion recovers the local value."""
    layer = [AbruptLayer(x_Al=0.0, thickness_nm=500, n_doping=1e17)]
    la = simulate_illumination(layer).layers[0]
    alpha_L = la.absorption_cm * la.diffusion_length_nm * 1e-7
    assert alpha_L > 0.5
    assert la.excess == pytest.approx(la.excess_local / (1.0 + alpha_L), rel=1e-9)
    assert simulate_illumination(layer, surface_velocity_cm_s=1e5).layers[0].excess < la.excess
    assert simulate_illumination(layer, diffusion=False).layers[0].excess == pytest.approx(la.excess_local)


def test_electrons_in_p_type_diffuse_further_than_holes_in_n_type():
    n_type = simulate_illumination([AbruptLayer(x_Al=0.0, thickness_nm=500, n_doping=1e18)]).layers[0]
    p_type = simulate_illumination([AbruptLayer(x_Al=0.0, thickness_nm=500, p_doping=1e19)]).layers[0]
    assert p_type.diffusion_length_nm > 3 * n_type.diffusion_length_nm
    assert p_type.excess < n_type.excess


def test_surface_excess_reduces_to_local_for_short_diffusion_length():
    with_diff, local, L = surface_excess(alpha=1e3, flux=1e18, tau=1e-12, T=1300.0, x_al=0.0, doping='n')
    assert 1e3 * L < 1e-2 and with_diff == pytest.approx(local, rel=2e-2)

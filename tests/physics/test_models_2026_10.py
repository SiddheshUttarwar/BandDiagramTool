"""
Tests for the model additions of 2026-10-04: AlGaN bowing 1.0 eV, Varshni
Eg(T), the Dreyer 2016 polarization set, N-polar growth, and the
surface-barrier extraction helper.
"""

import numpy as np
import pytest

from devices.analysis import extract_surface_barrier, sheet_density
from devices.device import AlGaNDevice
from devices.layer import AbruptLayer, Contact
from physics.materials.algan import (
    get_AlGaN_params, get_nitride_params, get_nitride_params_T, varshni_shift,
)
from physics.polarization import compute_Ppz, compute_Psp, compute_strain

Q = 1.602176634e-19


def _sigma_cm2(x_low, x_up, model, y_low=0.0, y_up=0.0):
    """Interface polarization charge [cm^-2] of (x_up, y_up) grown on a
    relaxed (x_low, y_low) substrate."""
    x = np.array([x_low, x_up])
    y = np.array([y_low, y_up])
    exx, ezz = compute_strain(x, x_low, y, y_low)
    P = compute_Psp(x, x_In=y, model=model) + compute_Ppz(x, exx, ezz, y, model=model)
    return (P[0] - P[1]) / Q * 1e-4


# --------------------------------------------------------------- band gap
def test_algan_bowing_is_one_ev():
    e0, e1, emid = (get_AlGaN_params(x).Eg for x in (0.0, 1.0, 0.5))
    assert 4.0 * (0.5 * (e0 + e1) - emid) == pytest.approx(1.0, abs=1e-9)
    assert emid == pytest.approx(4.458, abs=1e-3)


def test_bowing_leaves_binaries_and_valence_offset_unchanged():
    assert get_AlGaN_params(0.0).Eg == pytest.approx(3.39)
    assert get_AlGaN_params(1.0).Eg == pytest.approx(6.026)
    # the offset model is defined on the valence band, so bowing goes to Ec
    for x in (0.2, 0.5, 0.8):
        p, g = get_AlGaN_params(x), get_AlGaN_params(0.0)
        dEv = -(p.chi + p.Eg) + (g.chi + g.Eg)
        assert dEv == pytest.approx(-0.80 * x, abs=1e-9)


def test_varshni_is_zero_at_300K():
    for comp in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.3, 0.1)]:
        assert varshni_shift(*comp, 300.0) == pytest.approx(0.0, abs=1e-12)
        a, b = get_nitride_params(*comp), get_nitride_params_T(*comp, 300.0)
        assert b.Eg == pytest.approx(a.Eg) and b.chi == pytest.approx(a.chi)


def test_varshni_low_temperature_gaps():
    # GaN and AlN low-temperature gaps: measured 3.50 and 6.12 eV
    assert get_nitride_params_T(0.0, 0.0, 10.0).Eg == pytest.approx(3.462, abs=2e-3)
    assert get_nitride_params_T(1.0, 0.0, 10.0).Eg == pytest.approx(6.118, abs=2e-3)
    # the gap shrinks monotonically with temperature
    gaps = [get_nitride_params_T(0.0, 0.0, T).Eg for T in (10, 77, 300, 500)]
    assert all(a > b for a, b in zip(gaps, gaps[1:]))


def test_varshni_keeps_valence_band_fixed():
    for T in (10.0, 500.0):
        a, b = get_nitride_params(0.4, 0.0), get_nitride_params_T(0.4, 0.0, T)
        assert (a.chi + a.Eg) == pytest.approx(b.chi + b.Eg, abs=1e-12)


# --------------------------------------------------------------- polarization
def test_default_model_is_ambacher():
    x = np.array([0.0, 0.3])
    assert np.allclose(compute_Psp(x), compute_Psp(x, model='ambacher2002'))


def test_dreyer_interface_charges():
    # hand values from the PRX 6, 021038 constants (H reference, improper e31)
    assert _sigma_cm2(0.0, 0.3, 'dreyer2016') == pytest.approx(1.306e13, rel=5e-3)
    assert _sigma_cm2(0.0, 1.0, 'dreyer2016') == pytest.approx(5.23e13, rel=5e-3)
    assert _sigma_cm2(1.0, 0.0, 'dreyer2016') == pytest.approx(-4.00e13, rel=5e-3)
    # lower than the Ambacher set for Al-containing interfaces
    assert _sigma_cm2(0.0, 0.3, 'dreyer2016') < _sigma_cm2(0.0, 0.3, 'ambacher2002')


def test_unknown_polarization_model_raises():
    with pytest.raises(ValueError):
        compute_Psp(np.array([0.0]), model='nope')


# --------------------------------------------------------------- polarity
def _gan_algan_gan(polarity):
    layers = [AbruptLayer(0.0, 200, n_doping=1e16), AbruptLayer(0.3, 25), AbruptLayer(0.0, 20)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni', barrier_eV=0.84)]
    return AlGaNDevice(layers, contacts, dx_nm=0.2, polarity=polarity).solve(V_applied=0.0, quantum=False)


@pytest.mark.slow
def test_n_polar_moves_the_2deg_above_the_barrier():
    metal, npol = _gan_algan_gan('metal'), _gan_algan_gan('N')
    assert metal.converged and npol.converged
    # metal-polar: electrons under the barrier; N-polar: on top of it
    assert sheet_density(metal, 150, 200) > 1e12 > sheet_density(metal, 225, 245)
    assert sheet_density(npol, 225, 245) > 1e12 > sheet_density(npol, 150, 200)
    # the polarization profile is exactly mirrored
    assert np.allclose(np.asarray(metal.P_total), -np.asarray(npol.P_total))


def test_bad_polarity_raises():
    layers = [AbruptLayer(0.0, 50, n_doping=1e17)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Ti')]
    with pytest.raises(ValueError):
        AlGaNDevice(layers, contacts, polarity='Ga-ish').build_grid()


# --------------------------------------------------------------- inverse problem
@pytest.mark.slow
def test_extract_surface_barrier_round_trip():
    layers = [AbruptLayer(0.0, 200, n_doping=1e16), AbruptLayer(0.3, 20)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni', barrier_eV=1.4)]
    r = AlGaNDevice(layers, contacts, dx_nm=0.2).solve(V_applied=0.0, quantum=False)
    n_s = sheet_density(r, 150, 220)
    assert extract_surface_barrier(layers, n_s, (150, 220)) == pytest.approx(1.4, abs=0.02)


@pytest.mark.slow
def test_extract_surface_barrier_rejects_unreachable_density():
    layers = [AbruptLayer(0.0, 200, n_doping=1e16), AbruptLayer(0.3, 20)]
    with pytest.raises(ValueError):
        extract_surface_barrier(layers, 5e13, (150, 220))

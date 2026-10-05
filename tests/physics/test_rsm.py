"""Reciprocal space map (physics.rsm): peak positions follow the lattice."""
import numpy as np
import pytest

from devices.device import AlGaNDevice
from devices.layer import AbruptLayer, Contact
from physics.materials.algan import get_nitride_params
from physics.rsm import simulate_rsm


@pytest.fixture(scope="module")
def result():
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=200, n_doping=1e16),
              AbruptLayer(x_Al=0.3, thickness_nm=25)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni', barrier_eV=1.2)]
    return AlGaNDevice(layers, contacts, dx_nm=0.5).solve(V_applied=0.0, quantum=False)


def test_symmetric_peak_at_gan_0002(result):
    rsm = simulate_rsm(result, (0, 0, 2))
    assert rsm.symmetric
    scan = rsm.intensity[:, rsm.intensity.shape[1] // 2]
    qz_peak = rsm.qz[int(np.argmax(scan))]
    assert qz_peak == pytest.approx(2 * np.pi * 2 / get_nitride_params(0.0).c0, abs=2e-3)


def test_pseudomorphic_barrier_shares_the_substrate_rod(result):
    rsm = simulate_rsm(result, (1, 0, 5))
    gan = get_nitride_params(0.0)
    assert rsm.qx_substrate == pytest.approx(4 * np.pi / (np.sqrt(3) * gan.a0), rel=1e-6)
    by_name = {name: (qx, qz) for name, qx, qz in rsm.strained_points}
    qx_b, qz_b = by_name["Al0.30Ga0.70N"]
    assert qx_b == pytest.approx(rsm.qx_substrate, rel=1e-6)      # coherent: same in-plane constant
    assert qz_b > by_name["GaN"][1]                               # tensile in-plane -> smaller c
    relaxed = {name: (qx, qz) for name, qx, qz in rsm.relaxed_points}
    assert relaxed["Al0.30Ga0.70N"][0] > rsm.qx_substrate          # relaxed AlGaN has the smaller a
    assert relaxed["Al0.30Ga0.70N"][1] < qz_b                      # strained c is below the relaxed c


def test_relaxed_layer_moves_to_its_own_rod():
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=100, n_doping=1e16),
              AbruptLayer(x_Al=0.3, thickness_nm=50, relaxed=True)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni', barrier_eV=1.2)]
    r = AlGaNDevice(layers, contacts, dx_nm=0.5).solve(V_applied=0.0, quantum=False)
    rsm = simulate_rsm(r, (1, 0, 5))
    strained = {name: (qx, qz) for name, qx, qz in rsm.strained_points}
    relaxed = {name: (qx, qz) for name, qx, qz in rsm.relaxed_points}
    assert strained["Al0.30Ga0.70N"] == pytest.approx(relaxed["Al0.30Ga0.70N"], rel=1e-6)

"""Bands under illumination through the top surface (physics.illumination)."""
import numpy as np
import pytest

from devices.device import AlGaNDevice
from devices.layer import AbruptLayer, Contact
from physics.illumination import generation_profile, solve_illuminated

_PINNED = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni', barrier_eV=1.0)]


def _n_gan(T=300.0):
    return AlGaNDevice([AbruptLayer(x_Al=0.0, thickness_nm=500, n_doping=1e17)], _PINNED, T=T, dx_nm=1.0)


def test_generation_integrates_to_the_absorbed_photon_flux():
    g = _n_gan().build_grid()
    G, absorbed, photon_eV = generation_profile(g, 300.0, 1.0)
    from physics.grid_utils import node_spacings
    cw_cm = node_spacings(g.dx, g.N)[2] * 100.0
    flux0 = 1.0 / (photon_eV * 1.602176634e-19)
    assert float(np.sum(G * cw_cm)) == pytest.approx(absorbed * flux0, rel=1e-9)
    assert absorbed > 0.99 and G[-1] > G[0]                 # absorbed near the top surface


def test_light_below_the_gap_changes_nothing():
    r = solve_illuminated(_n_gan(), wavelength_nm=450.0)     # 2.76 eV < Eg(GaN, 300 K)
    assert r.absorbed_fraction == 0.0 and r.photovoltage_V == 0.0
    assert np.allclose(r.light.Ec, r.dark.Ec) and np.allclose(r.light.Efn, r.light.Efp)


def test_surface_depletion_flattens_at_open_circuit():
    """n-GaN with the surface pinned 1 eV below Ec: light reduces the band
    bending, no current flows, and the dark state is recovered far below."""
    r = solve_illuminated(_n_gan())
    assert r.converged
    bend_dark = float(r.dark.Ec[-1] - r.dark.Ec[0])
    bend_light = float(r.light.Ec[-1] - r.light.Ec[0])
    assert 0.0 < bend_light < bend_dark - 0.2
    assert r.photovoltage_V == pytest.approx(bend_dark - bend_light, abs=1e-6)
    assert abs(r.J_residual_A_cm2) < 1e-4 * abs(r.J_short_A_cm2)
    assert float(np.max(r.light.Efn - r.light.Efp)) > 1.0     # quasi-Fermi levels split where absorbed
    assert float(r.light.Ec[0]) == pytest.approx(float(r.dark.Ec[0]), abs=1e-6)


def test_photovoltage_collapses_at_growth_temperature():
    """At 1040 C the same surface passes so much thermal current that the
    photovoltage is negligible: the bands do not move."""
    r = solve_illuminated(_n_gan(T=1313.15))
    assert r.converged and abs(r.photovoltage_V) < 1e-3
    assert abs(r.J_residual_A_cm2) < 1e-3 * abs(r.J_short_A_cm2)

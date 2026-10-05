"""
Tests for the AlGaN / InGaN / InAlGaN material classes and the quaternary
parameter model behind them (physics.materials.nitrides, .algan).
"""
import numpy as np
import pytest

from physics.materials.nitrides import AlGaN, InGaN, InAlGaN, make_material
from physics.materials.algan import get_nitride_params, get_AlGaN_params, nitride_name


def test_classes_are_views_of_one_parameter_set():
    assert InAlGaN(0.3, 0.0).params() == AlGaN(0.3).params()
    assert InAlGaN(0.0, 0.15).params() == InGaN(0.15).params()
    assert AlGaN(0.3).params() == get_AlGaN_params(0.3)


def test_names_and_validation():
    assert AlGaN(0.0).name == 'GaN'
    assert InGaN(1.0).name == 'InN'
    assert InGaN(0.15).name == 'In0.15Ga0.85N'
    assert InAlGaN(0.83, 0.17).name == 'Al0.83In0.17N'
    assert nitride_name(0.1, 0.05) == 'Al0.10In0.05Ga0.85N'
    with pytest.raises(ValueError):
        InAlGaN(0.7, 0.4)
    with pytest.raises(ValueError):
        make_material('AlGaN', 0.2, 0.1)
    with pytest.raises(ValueError):
        make_material('InGaN', 0.2, 0.1)


def test_binary_endpoints_and_bowing():
    assert np.isclose(InGaN(1.0).band_gap(), 0.69)
    # InGaN bowing 1.4 eV: Eg(In0.15) = 0.15*0.69 + 0.85*3.39 - 1.4*0.1275
    assert np.isclose(InGaN(0.15).band_gap(), 0.15 * 0.69 + 0.85 * 3.39 - 1.4 * 0.15 * 0.85)


def test_alinn_lattice_matched_to_gan():
    a_gan = AlGaN(0.0).lattice_constant()[0]
    a_alinn = InAlGaN(0.83, 0.17).lattice_constant()[0]
    assert abs(a_alinn - a_gan) / a_gan < 0.002


def test_band_alignment_type_I():
    """InGaN on GaN: both band edges inside GaN's gap (type I, VB above by
    ~0.087 eV at 15% In from the 0.58 eV InN/GaN VBO). Lattice-matched AlInN:
    VB 0.2 eV below GaN (XPS-anchored), CB ~0.65 eV above."""
    def edges(p):                     # absolute (Ec, Ev_top) [eV]
        return -p.chi, -(p.chi + p.Eg)
    Ec_g, Ev_g = edges(get_nitride_params(0.0, 0.0))
    Ec_i, Ev_i = edges(get_nitride_params(0.0, 0.15))
    assert Ec_i < Ec_g and Ev_i > Ev_g                    # type I well
    assert np.isclose(Ev_i - Ev_g, 0.087, atol=0.002)
    Ec_a, Ev_a = edges(get_nitride_params(0.83, 0.17))
    assert np.isclose(Ev_g - Ev_a, 0.199, atol=0.005)     # AlInN VB 0.2 eV below
    assert 0.5 < Ec_a - Ec_g < 0.8


def test_ingan_qw_detected_and_algan_unchanged():
    from devices.layer import AbruptLayer, Contact
    from devices.grid_builder import build_grid
    C = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Ni')]
    L = [AbruptLayer(0.0, 50, n_doping=1e18), AbruptLayer(0.0, 10),
         AbruptLayer.from_material(InGaN(0.15), 3.0), AbruptLayer(0.0, 10),
         AbruptLayer(0.0, 50, p_doping=1e19)]
    g = build_grid(L, C, dx_nm=0.2)
    assert g.qw_window is not None
    i0, i1 = g.qw_window
    assert np.all(g.x_In[i0 + 2:i1 - 2] > 0.14)
    assert g.eps_xx[(i0 + i1) // 2] < 0          # compressive InGaN on GaN
    g_alg = build_grid([AbruptLayer(0.0, 50, n_doping=1e18), AbruptLayer(0.3, 25)], C, dx_nm=0.2)
    assert not np.any(g_alg.x_In)

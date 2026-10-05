"""
Analytic-limit tests for physics/polarization.py.
"""

import numpy as np

from physics.polarization import (
    compute_Psp, compute_strain, compute_Ppz,
    compute_pol_charge, interface_sheet_charges,
)
from physics.constants import q as _q


def test_interface_sheet_charge_order_of_magnitude_2deg():
    """
    A GaN/Al0.25Ga0.75N heterojunction (pseudomorphic on GaN) should produce
    an interface sheet charge on the order of the ~1e13 cm^-2 2DEG sheet
    densities routinely reported for AlGaN/GaN HEMTs (Ambacher et al., JAP
    85:3222, 1999). This checks the model is self-consistent with that
    well-known order of magnitude (using the same physics.materials.algan
    parameters the solver uses), not an independently hard-coded value.
    """
    N = 200
    x_Al = np.concatenate([np.zeros(N // 2), np.full(N // 2, 0.25)])
    x_sub = 0.0  # GaN substrate

    eps_xx, eps_zz = compute_strain(x_Al, x_sub)
    P_total = compute_Psp(x_Al, T=300.0) + compute_Ppz(x_Al, eps_xx, eps_zz)

    interfaces = interface_sheet_charges(P_total, x_Al)
    assert len(interfaces) == 1
    _, sigma = interfaces[0]

    sheet_density_cm2 = abs(sigma) / _q * 1e-4  # C/m^2 -> cm^-2
    assert 1e12 < sheet_density_cm2 < 1e14


def test_uniform_composition_has_no_polarization_charge():
    N = 100
    x_Al = np.full(N, 0.2)
    x_sub = 0.2

    eps_xx, eps_zz = compute_strain(x_Al, x_sub)
    P_total = compute_Psp(x_Al, T=300.0) + compute_Ppz(x_Al, eps_xx, eps_zz)

    assert len(interface_sheet_charges(P_total, x_Al)) == 0

    dx = 1e-9
    rho = compute_pol_charge(P_total, dx)
    assert np.allclose(rho[2:-2], 0.0, atol=1e-20)  # dP/dz = 0 in the interior


def test_strain_vanishes_when_layer_matches_substrate():
    x_Al = np.full(50, 0.3)
    eps_xx, eps_zz = compute_strain(x_Al, x_sub=0.3)
    assert np.allclose(eps_xx, 0.0, atol=1e-12)
    assert np.allclose(eps_zz, 0.0, atol=1e-12)


def test_pol_charge_conserved_on_nonuniform_grid():
    """
    The integrated polarization charge across an abrupt step must equal the
    polarization jump regardless of grid spacing. Regression test: the
    non-uniform 3-point central difference scaled the sheet charge by h1/h2
    where per-layer dx_nm changed at the interface (4x for 0.2 -> 0.05 nm),
    making 2DEG densities depend on mesh choice.
    """
    from physics.grid_utils import node_spacings
    P_lo, P_hi = -0.034, -0.060
    for h_left, h_right in [(0.2e-9, 0.05e-9), (0.05e-9, 0.2e-9), (0.1e-9, 0.1e-9)]:
        edges = np.concatenate([np.full(50, h_left), np.full(50, h_right)])
        P = np.where(np.arange(101) <= 50, P_lo, P_hi)
        rho = compute_pol_charge(P, edges)
        _, _, cw, _ = node_spacings(edges, 101)
        sigma = np.sum(rho * cw)
        assert np.isclose(sigma, P_lo - P_hi, rtol=1e-12), (h_left, h_right, sigma)


def test_unstrained_valence_splittings_match_vurgaftman():
    """Chuang-Chang zone-centre energies: GaN A-B ~5 meV, A-C ~43 meV
    (Vurgaftman & Meyer 2003 parameters); AlN crystal-field band on top,
    HH ~0.16 eV below it."""
    from physics.materials.algan import get_AlGaN_params
    d_hh, d_lh, d_so, _, _ = get_AlGaN_params(0.0).valence_offsets_from_top()
    assert d_hh == 0.0 and 0.003 < d_lh < 0.008 and 0.035 < d_so < 0.050
    d_hh, d_lh, d_so, _, _ = get_AlGaN_params(1.0).valence_offsets_from_top()
    assert d_so == 0.0 and 0.14 < d_hh < 0.19


def test_strain_band_shift_gan_on_aln():
    """GaN coherently strained on AlN (eps_xx ~ -2.4%) widens its gap by
    ~0.18 eV with Vurgaftman deformation potentials; zero strain = no shift."""
    from physics.materials.algan import get_AlGaN_params
    from physics.polarization import eps_zz_from_eps_xx
    p = get_AlGaN_params(0.0)
    exx = (3.112 - 3.189) / 3.189
    ezz = float(eps_zz_from_eps_xx(np.array([0.0]), np.array([exx]))[0])
    dEc, dEv = p.strain_band_shifts(exx, ezz)
    assert 0.15 < dEc - dEv < 0.21
    assert p.strain_band_shifts(0.0, 0.0) == (0.0, 0.0)


def test_custom_strain_equal_to_pseudomorphic_reproduces_default():
    """A layer's custom_strain_xx set to its pseudomorphic value must give
    the same bulk band edges and interface polarization charge; relaxed=True
    removes both the strain band shift and the piezo charge."""
    from devices.layer import AbruptLayer, Contact
    from devices.grid_builder import build_grid
    from physics.grid_utils import node_spacings
    from physics.materials.algan import get_AlGaN_params
    C = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni', barrier_eV=1.2)]
    exx = (get_AlGaN_params(0.0).a0 - get_AlGaN_params(0.3).a0) / get_AlGaN_params(0.3).a0
    g0 = build_grid([AbruptLayer(0.0, 100, n_doping=1e17), AbruptLayer(0.3, 25)], C, dx_nm=0.2)
    g1 = build_grid([AbruptLayer(0.0, 100, n_doping=1e17),
                     AbruptLayer(0.3, 25, custom_strain_xx=exx)], C, dx_nm=0.2)
    g2 = build_grid([AbruptLayer(0.0, 100, n_doping=1e17),
                     AbruptLayer(0.3, 25, relaxed=True)], C, dx_nm=0.2)
    i = int(np.argmin(abs(g0.x_nm - 112.0)))
    assert np.isclose(g0.Eg[i], g1.Eg[i]) and np.isclose(g0.chi[i], g1.chi[i])
    _, _, cw, _ = node_spacings(g0.dx, g0.N)
    m = (g0.x_nm > 90) & (g0.x_nm < 110)
    assert np.isclose(np.sum(g0.pol_rho[m] * cw[m]), np.sum(g1.pol_rho[m] * cw[m]), rtol=1e-9)
    assert np.isclose(g2.Eg[i], get_AlGaN_params(0.3).Eg) and g2.Ppz[i] == 0.0

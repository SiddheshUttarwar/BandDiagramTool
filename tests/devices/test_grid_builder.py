"""
Tests for devices/grid_builder.py.
"""

import numpy as np
import pytest

from devices.layer import (
    AbruptLayer, Contact, QuantumRegionMarker, SurfaceCharge, SurfaceState, InterfaceDipole,
)
from devices.grid_builder import build_grid
from physics.materials.algan import get_AlGaN_params_T


def _simple_device():
    layers = [AbruptLayer(x_Al=0.1, thickness_nm=100, n_doping=1e17, p_doping=0.0)]
    contacts = [
        Contact(position='bottom', contact_type='ohmic', metal='Ti'),
        Contact(position='top',    contact_type='ohmic', metal='Ti'),
    ]
    return layers, contacts


def test_uniform_layer_composition_and_doping():
    layers, contacts = _simple_device()
    grid = build_grid(layers, contacts, T=300.0, dx_nm=1.0)

    # Away from the (Gaussian-smoothed) edges, composition and doping
    # should sit at the layer's nominal values.
    interior = slice(10, -10)
    assert np.allclose(grid.x_Al[interior], 0.1, atol=1e-3)
    assert np.allclose(grid.ND[interior], 1e17)
    assert np.allclose(grid.NA[interior], 0.0)


def test_effective_dos_matches_material_model():
    layers, contacts = _simple_device()
    grid = build_grid(layers, contacts, T=300.0, dx_nm=1.0)

    expected = get_AlGaN_params_T(0.1, 300.0)
    mid = grid.N // 2
    assert np.isclose(grid.Nc[mid], expected.Nc(300.0), rtol=1e-6)
    assert np.isclose(grid.Nv[mid], expected.Nv(300.0), rtol=1e-6)


def test_build_grid_requires_exactly_two_contacts():
    layers, _ = _simple_device()
    only_one = [Contact(position='bottom', contact_type='ohmic', metal='Ti')]
    with pytest.raises(ValueError):
        build_grid(layers, only_one, T=300.0, dx_nm=1.0)


# ---------------------------------------------------------------------------
# qw_window: auto-detection of the layer stack's "textbook" quantum well
# (used by physics.self_consistent for QCSE diagnostics).
# ---------------------------------------------------------------------------

_STD_CONTACTS = [
    Contact(position='bottom', contact_type='ohmic', metal='Ti'),
    Contact(position='top',    contact_type='ohmic', metal='Ti'),
]


def test_qw_window_detects_barrier_well_barrier_stack():
    """An undoped low-x_Al layer sandwiched by higher-x_Al barriers is the
    textbook quantum well; qw_window should span exactly that layer."""
    layers = [
        AbruptLayer(x_Al=0.15, thickness_nm=20),   # barrier
        AbruptLayer(x_Al=0.0,  thickness_nm=3),    # well
        AbruptLayer(x_Al=0.15, thickness_nm=20),   # barrier
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)

    assert grid.qw_window is not None
    i0, i1 = grid.qw_window
    assert grid.x_nm[i0] == pytest.approx(20.0, abs=1.0)
    assert grid.x_nm[i1 - 1] == pytest.approx(23.0, abs=1.0)
    assert np.all(grid.x_Al[i0:i1] < 0.05)   # inside the low-x_Al well


def test_qw_window_none_when_composition_monotonic():
    """No local minimum -> no well (a graded/step-up profile, not a QW)."""
    layers = [
        AbruptLayer(x_Al=0.0,  thickness_nm=50),
        AbruptLayer(x_Al=0.15, thickness_nm=50),
        AbruptLayer(x_Al=0.30, thickness_nm=50),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=1.0)
    assert grid.qw_window is None


def test_qw_window_ignores_doped_candidate():
    """A locally-narrower-bandgap layer that's doped isn't 'the' quantum
    well (a real device's active region is left undoped on purpose)."""
    layers = [
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        AbruptLayer(x_Al=0.0,  thickness_nm=3, n_doping=1e18),   # doped -> excluded
        AbruptLayer(x_Al=0.15, thickness_nm=20),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    assert grid.qw_window is None


# ---------------------------------------------------------------------------
# Per-layer grid spacing (layer.dx_nm): non-uniform grids
# ---------------------------------------------------------------------------

def test_uniform_when_no_layer_overrides_spacing():
    """dx should be a constant-valued array (still per-edge, but every edge
    equal) when no layer requests its own spacing -- the pre-existing
    uniform-grid behaviour, now just expressed as an array."""
    layers = [
        AbruptLayer(x_Al=0.0,  thickness_nm=50, n_doping=1e17),
        AbruptLayer(x_Al=0.15, thickness_nm=50),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    assert np.allclose(grid.dx, 0.5e-9)
    assert np.allclose(np.diff(grid.x_nm), 0.5, atol=1e-6)


def test_per_layer_dx_nm_produces_non_uniform_spacing():
    """A layer with its own dx_nm should be locally finer/coarser than its
    neighbours, without affecting their spacing."""
    layers = [
        AbruptLayer(x_Al=0.0,  thickness_nm=50, n_doping=1e17, dx_nm=1.0),   # 50 pts
        AbruptLayer(x_Al=0.0,  thickness_nm=3,  dx_nm=0.1),   # fine QW mesh, 30 pts
        AbruptLayer(x_Al=0.0,  thickness_nm=50, n_doping=1e17, dx_nm=1.0),   # 50 pts
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=1.0)
    assert grid.N == 130

    diffs = np.diff(grid.x_nm)
    # Interior of layer 1 (indices 0..48): its own dx=1.0
    assert np.allclose(diffs[0:49], 1.0, atol=1e-6)
    # Interior of the fine QW layer (indices 50..78): its own dx=0.1
    assert np.allclose(diffs[50:79], 0.1, atol=1e-6)
    # Interior of layer 3 (indices 80..128): its own dx=1.0
    assert np.allclose(diffs[80:129], 1.0, atol=1e-6)

    # The saving that motivates this feature: far fewer points than meshing
    # the whole device at the QW's fine spacing (no per-layer override, so
    # every layer takes the device-wide default of 0.1nm).
    layers_no_override = [
        AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17),
        AbruptLayer(x_Al=0.0, thickness_nm=3),
        AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17),
    ]
    grid_all_fine = build_grid(layers_no_override, _STD_CONTACTS, T=300.0, dx_nm=0.1)
    assert grid.N < grid_all_fine.N / 3


def test_per_layer_dx_nm_defaults_to_device_dx_when_unset():
    """A layer with dx_nm=None should be spaced identically to a device
    whose default dx_nm equals that value -- i.e. dx_nm=None really is
    'use the device default', not some other fallback value."""
    layers_explicit = [AbruptLayer(x_Al=0.0, thickness_nm=40, dx_nm=0.5)]
    layers_default   = [AbruptLayer(x_Al=0.0, thickness_nm=40)]

    grid_explicit = build_grid(layers_explicit, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    grid_default  = build_grid(layers_default,  _STD_CONTACTS, T=300.0, dx_nm=0.5)

    assert grid_explicit.N == grid_default.N
    assert np.allclose(grid_explicit.x_nm, grid_default.x_nm)


def test_qw_window_picks_thinnest_candidate():
    """With multiple undoped local-minimum layers, the thinnest (most
    confined -- what a designer would call "the" well) wins."""
    layers = [
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        AbruptLayer(x_Al=0.0,  thickness_nm=10),   # wider candidate well
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        AbruptLayer(x_Al=0.0,  thickness_nm=3),    # narrower candidate well
        AbruptLayer(x_Al=0.15, thickness_nm=20),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)

    assert grid.qw_window is not None
    i0, i1 = grid.qw_window
    width_nm = grid.x_nm[i1 - 1] - grid.x_nm[i0]
    assert width_nm == pytest.approx(3.0, abs=1.0)


# ---------------------------------------------------------------------------
# manual_quantum_region: explicit QuantumRegionMarker interface layers, an
# alternative to physics.self_consistent's automatic undoped-span heuristic.
# ---------------------------------------------------------------------------

def test_manual_quantum_region_none_when_unmarked():
    layers = [
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        AbruptLayer(x_Al=0.0,  thickness_nm=3),
        AbruptLayer(x_Al=0.15, thickness_nm=20),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    assert grid.manual_quantum_region is None


def test_manual_quantum_region_spans_marked_layers_plus_padding():
    layers = [
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        QuantumRegionMarker(boundary='start'),
        AbruptLayer(x_Al=0.0,  thickness_nm=3),
        AbruptLayer(x_Al=0.15, thickness_nm=5),
        AbruptLayer(x_Al=0.0,  thickness_nm=3),
        QuantumRegionMarker(boundary='end'),
        AbruptLayer(x_Al=0.15, thickness_nm=20),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)

    assert grid.manual_quantum_region is not None
    i0, i1 = grid.manual_quantum_region
    # Markers sit at physical [20, 31] nm; expect ~2nm padding each side.
    assert grid.x_nm[i0] == pytest.approx(18.0, abs=1.0)
    assert grid.x_nm[i1 - 1] == pytest.approx(33.0, abs=1.0)


def test_manual_quantum_region_ignores_single_marker():
    """Only a start with no end (or vice versa) shouldn't produce a region --
    both markers must be present."""
    layers = [
        AbruptLayer(x_Al=0.15, thickness_nm=20),
        QuantumRegionMarker(boundary='start'),
        AbruptLayer(x_Al=0.0,  thickness_nm=3),
        AbruptLayer(x_Al=0.15, thickness_nm=20),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    assert grid.manual_quantum_region is None


def test_manual_quantum_region_excludes_wide_undoped_layer():
    """The scenario that motivated this feature: a wide undoped layer next
    to a real MQW stack should NOT be swept in when explicit markers are
    used, unlike the automatic undoped-span heuristic."""
    layers = [
        AbruptLayer(x_Al=0.82, thickness_nm=50, n_doping=1e19),
        QuantumRegionMarker(boundary='start'),
        AbruptLayer(x_Al=0.78, thickness_nm=2),
        AbruptLayer(x_Al=0.82, thickness_nm=2),
        QuantumRegionMarker(boundary='end'),
        AbruptLayer(x_Al=0.70, thickness_nm=100),   # wide undoped layer, NOT marked
        AbruptLayer(x_Al=0.0, thickness_nm=10, p_doping=1e19),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.2)

    assert grid.manual_quantum_region is not None
    i0, i1 = grid.manual_quantum_region
    width_nm = grid.x_nm[i1 - 1] - grid.x_nm[i0]
    assert width_nm < 20.0   # marked stack (4nm) + padding, not the 100nm layer too


# ---------------------------------------------------------------------------
# Zero-thickness interface layers: QuantumRegionMarker / SurfaceCharge /
# InterfaceDipole don't consume grid points of their own.
# ---------------------------------------------------------------------------

def test_interface_layers_consume_no_grid_points():
    layers_with_markers = [
        AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17),
        QuantumRegionMarker(boundary='start'),
        SurfaceCharge(states=[SurfaceState(density_cm2=1e12, energy_eV=0.1, state_type='donor')]),
        InterfaceDipole(sheet_charge_C_m2=1e-3, separation_nm=0.5),
        QuantumRegionMarker(boundary='end'),
        AbruptLayer(x_Al=0.15, thickness_nm=50),
    ]
    layers_bare = [
        AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17),
        AbruptLayer(x_Al=0.15, thickness_nm=50),
    ]
    grid_marked = build_grid(layers_with_markers, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    grid_bare = build_grid(layers_bare, _STD_CONTACTS, T=300.0, dx_nm=0.5)
    assert grid_marked.N == grid_bare.N


def test_surface_charge_site_recorded_at_correct_position():
    layers = [
        AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17),
        SurfaceCharge(states=[
            SurfaceState(density_cm2=1e12, energy_eV=0.1, state_type='donor'),
            SurfaceState(density_cm2=5e11, energy_eV=0.3, state_type='acceptor'),
        ]),
        AbruptLayer(x_Al=0.15, thickness_nm=50),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)

    assert len(grid.surface_charge_sites) == 1
    i0, states = grid.surface_charge_sites[0]
    assert grid.x_nm[i0] == pytest.approx(50.0, abs=1.0)
    assert len(states) == 2


def test_interface_dipole_injects_opposite_sheet_charges_into_pol_rho():
    layers = [
        AbruptLayer(x_Al=0.0, thickness_nm=50),
        InterfaceDipole(sheet_charge_C_m2=1e-2, separation_nm=2.0),
        AbruptLayer(x_Al=0.0, thickness_nm=50),
    ]
    grid = build_grid(layers, _STD_CONTACTS, T=300.0, dx_nm=0.5)

    i_pos = int(np.searchsorted(grid.x_nm, 50.0))
    i_neg = int(np.searchsorted(grid.x_nm, 52.0))
    assert grid.pol_rho[i_pos] > 0
    assert grid.pol_rho[i_neg] < 0

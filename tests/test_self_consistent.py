"""
Solver-level regression and robustness tests, run on real device stacks
rather than synthetic profiles.

These are marked `slow` (full self-consistent Schrödinger-Poisson /
drift-diffusion solves) — run the fast unit-test suite with
`pytest -m "not slow"` and the full suite (including these) with `pytest`.
"""

import numpy as np
import pytest

from devices.layer import (
    AbruptLayer, GradedLayer, Contact, QuantumRegionMarker, SurfaceCharge, SurfaceState,
)
from devices.device import AlGaNDevice


def _led_device():
    """The example_led.py / main.py device: n-GaN / AlGaN barrier / GaN QW
    / AlGaN EBL / p-GaN, ohmic n-contact + Schottky p-contact."""
    layers = [
        AbruptLayer(x_Al=0.0,  thickness_nm=200, n_doping=1e17, p_doping=0.0),
        AbruptLayer(x_Al=0.15, thickness_nm=10,  n_doping=0.0,  p_doping=0.0),
        AbruptLayer(x_Al=0.0,  thickness_nm=3,   n_doping=0.0,  p_doping=0.0),
        AbruptLayer(x_Al=0.15, thickness_nm=20,  n_doping=0.0,  p_doping=0.0),
        AbruptLayer(x_Al=0.0,  thickness_nm=100, n_doping=0.0,  p_doping=3e17),
    ]
    contacts = [
        Contact(position='bottom', contact_type='ohmic',    metal='Ti'),
        Contact(position='top',    contact_type='schottky', metal='Ni'),
    ]
    return AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=0.2)


def _uv_led_mqw_device():
    """The examples/debug_convergence.py device: a deep-UV AlGaN MQW that
    was historically hard to converge at forward bias (the reason that
    debug script exists)."""
    layers = [
        AbruptLayer(x_Al=0.82, thickness_nm=400, n_doping=1e19),
        AbruptLayer(x_Al=0.78, thickness_nm=1.6),
        AbruptLayer(x_Al=0.82, thickness_nm=1.1),
        AbruptLayer(x_Al=0.78, thickness_nm=1.6),
        AbruptLayer(x_Al=0.82, thickness_nm=1.1),
        AbruptLayer(x_Al=0.78, thickness_nm=1.6),
        AbruptLayer(x_Al=0.82, thickness_nm=1.1),
        AbruptLayer(x_Al=0.95, thickness_nm=2.0),
        GradedLayer(x_Al_start=0.90, x_Al_end=0.65, thickness_nm=150, profile='linear'),
        GradedLayer(x_Al_start=0.65, x_Al_end=0.25, thickness_nm=40, profile='linear', p_doping=1e19),
        AbruptLayer(x_Al=0.0, thickness_nm=10, p_doping=1e20),
    ]
    contacts = [
        Contact(position='bottom', contact_type='ohmic', metal='Ti'),
        Contact(position='top',    contact_type='ohmic', metal='Ni'),
    ]
    return AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=0.5)


@pytest.mark.slow
def test_led_equilibrium_converges_with_default_settings():
    """
    Regression test for the equilibrium (V=0) solve, which is the case the
    adaptive/safeguarded mixing in physics.self_consistent targets. This
    should converge cleanly with *default* solve() kwargs — no hand-tuned
    alpha/anderson_m required.
    """
    device = _led_device()
    r = device.solve(V_applied=0.0, quantum=True, max_iter=400)
    assert r.converged


@pytest.mark.slow
def test_led_equilibrium_band_bending_and_fermi_pinning():
    device = _led_device()
    r = device.solve(V_applied=0.0, quantum=True, max_iter=400)
    assert r.converged

    # Single global equilibrium Fermi level: with V_applied=0 the
    # drift-diffusion update never runs, so both quasi-Fermi levels stay
    # at exactly 0 eV everywhere.
    assert np.allclose(r.Efn, 0.0, atol=1e-9)
    assert np.allclose(r.Efp, 0.0, atol=1e-9)

    # n-side (bottom, ohmic n-GaN) sits closer to Ec than the p-side (top,
    # Schottky p-GaN) — the expected p-n heterojunction band-bending
    # direction.
    assert r.Ec[0] < r.Ec[-1]
    assert r.Ev[0] < r.Ev[-1]


@pytest.mark.slow
def test_led_equilibrium_golden_values():
    """
    Golden-value regression: catches accidental changes to the physics or
    solver by comparing key scalars against values recorded from a known
    run, with a tolerance band loose enough to survive minor numerical
    changes but tight enough to catch a real regression.

    Re-recorded 2026-09-22: the previous values (peak_field=1.226e8,
    max_sigma=6.11e-3) were stale and failing well outside their own 10%
    tolerance band (peak_field ~1.8x off, max_sigma ~3.7x off) on
    unmodified code too -- confirmed via a clean git-stash baseline
    comparison, so this was not a regression from any change made this
    session. Consistent with this file's own
    test_uv_led_mqw_sweep_fails_safely_without_false_convergence docstring,
    which separately documents that devices/grid_builder.py's interface
    smoothing was changed from a fixed 1.0nm width to 1 grid point at some
    point -- sharper interfaces concentrate the same total polarization
    charge into fewer grid points and change both these values -- and that
    change was never reflected here. New values confirmed reproducible
    across multiple independent runs (peak_field: 6.746207e7 twice to 6
    significant figures).

    Re-recorded 2026-10-01 after the nextnano++-benchmarked material update
    (type-I VBO band alignment, Ambacher 2002 Psp + Bernardini piezo,
    c-axis permittivity, Psp on by default, conservative non-uniform-grid
    polarization charge): peak_field 6.746e7 -> 1.126381e8 V/m, max_sigma
    1.668e-3 -> 4.408101e-3 C/m^2, both reproduced to 7 significant figures
    across two runs. See benchmarks/REPORT.md.
    """
    device = _led_device()
    r = device.solve(V_applied=0.0, quantum=True, max_iter=400)
    assert r.converged

    peak_field = max(abs(r.E_field.min()), abs(r.E_field.max()))
    assert peak_field == pytest.approx(1.126e8, rel=0.1)  # V/m

    # Baseline reflects devices/grid_builder.py's interface smoothing,
    # fixed at 1 grid point (not a fixed nm width) — sharper interfaces
    # concentrate the same total polarization charge into fewer grid
    # points, raising this peak; it will shift if dx_nm changes.
    max_sigma = max(abs(s) for s in r.interface_sigmas)
    assert max_sigma == pytest.approx(4.408e-3, rel=0.1)  # C/m^2


@pytest.mark.slow
def test_uv_led_mqw_sweep_reaches_high_bias_via_direct_ptc():
    """
    Robustness regression for the historically fragile device in
    examples/debug_convergence.py (7 barrier/well layers 1.1-2.0nm wide,
    0.78-0.95 Al composition contrast, at dx_nm=0.5 -- 2-4 grid points per
    layer).

    History: this test originally asserted `n_converged >= 6` and passed,
    but that was a false positive from a since-fixed outer-loop bug (a
    failed coupled solve's unchanged-state fallback read as "converged" --
    see [[bug-quasi-fermi-pinning]]). Once that was fixed (2026-09-22),
    honest logging showed this device's JFNK solve genuinely never
    reaching tolerance at *any* bias, including the smallest bisected step
    -- so the test was rewritten to assert universal, safe failure
    instead (`test_uv_led_mqw_sweep_fails_safely_without_false_convergence`,
    since renamed/superseded by this one).

    That in turn stopped being true once physics.coupled_solver gained a
    direct (non-Krylov-approximated) pseudo-transient-continuation
    fallback for when JFNK plateaus (_solve_ptc_direct -- a nextnano++-
    style direct sparse Newton step per pseudo-time increment, instead of
    JFNK's Krylov-approximated one, which is what was plateauing
    regardless of how much Krylov capacity it was given -- confirmed
    empirically before this fix). With that in place this device now
    genuinely converges at 9 of 10 sweep points and reaches the full 4.5V
    target, in under a minute -- confirmed reproducible. Only V=0.0
    (equilibrium) fails; that's a different code path (the AA-mixing
    equilibrium loop, not the coupled-bias solver this fix targets) and
    not addressed here.
    """
    device = _uv_led_mqw_device()
    results = device.sweep_voltage(0.0, 4.5, n_steps=10, quantum=False,
                                    alpha=0.1, max_iter=200)

    assert len(results) == 10
    n_converged = sum(r.converged for r in results)
    assert n_converged >= 8
    assert results[-1].V_applied >= 4.0
    assert results[-1].converged

    for r in results:
        assert np.all(np.isfinite(r.phi))
        assert np.all(np.isfinite(r.Efn))
        assert np.all(np.isfinite(r.Efp))
        # Exactly the failure mode a masked/false convergence would hide: an
        # unconverged/unstable solve producing a nonsensical quasi-Fermi
        # split (tens to hundreds of eV) instead of a value bounded near
        # the applied bias.
        splitting = r.Efn - r.Efp
        assert splitting.max() < 20.0


@pytest.mark.slow
def test_led_equilibrium_qcse_fields_populated():
    """
    A quantum-mode solve should always populate the QCSE diagnostics
    (physics/optical.py) alongside E_e/E_h/psi_e/psi_h: an e1-h1 transition
    energy and a bounded [0, 1] overlap, plus the highest-overlap pair
    among all solved subbands. quantum=False should leave them unset.
    """
    device = _led_device()
    r = device.solve(V_applied=0.0, quantum=True, max_iter=400)
    assert r.converged

    assert r.qcse_transition_eV is not None
    assert 0.0 <= r.qcse_overlap <= 1.0 + 1e-9

    assert r.qcse_dominant_pair is not None
    if r.qcse_local_solve:
        # e1/h1 came from a local single-well solve (no globally-solved
        # subband was confined in the detected well), so it is not one of
        # the pairs the dominant-overlap max runs over. It must be a real
        # well transition (> 2 eV) -- the old fallback paired an n-side
        # electron with a p-side hole and reported ~0.6 eV.
        assert r.qcse_in_well and r.qcse_transition_eV > 2.0
    else:
        assert r.qcse_dominant_overlap >= r.qcse_overlap - 1e-9  # dominant is a max over all pairs

    r_classical = device.solve(V_applied=0.0, quantum=False, max_iter=400)
    assert r_classical.qcse_transition_eV is None
    assert r_classical.qcse_overlap is None


@pytest.mark.slow
def test_manual_quantum_region_used_when_marked():
    """
    devices.layer.QuantumRegionMarker (see devices/grid_builder.py's
    manual_quantum_region) should be used directly instead of the automatic
    undoped-span heuristic when placed. Verified by confirming the ground
    state wavefunction stays localized within the marked window rather than
    spreading across an adjacent wide undoped layer that the heuristic
    would otherwise sweep in alongside it (the scenario that motivated this
    feature: a real device with a large undoped graded transport layer next
    to the actual MQW stack).
    """
    layers = [
        AbruptLayer(x_Al=0.82, thickness_nm=50, n_doping=1e19),
        QuantumRegionMarker(boundary='start'),
        AbruptLayer(x_Al=0.78, thickness_nm=2),
        AbruptLayer(x_Al=0.82, thickness_nm=2),
        QuantumRegionMarker(boundary='end'),
        AbruptLayer(x_Al=0.70, thickness_nm=60),   # wide undoped layer, unmarked
        AbruptLayer(x_Al=0.0,  thickness_nm=10, p_doping=1e19),
    ]
    contacts = [
        Contact(position='bottom', contact_type='ohmic', metal='Ti'),
        Contact(position='top',    contact_type='ohmic', metal='Ni'),
    ]
    # Region restriction is a static property of the quantum region setup
    # (computed once before the Gummel loop, independent of phi -- see
    # physics.self_consistent's "Quantum region (static for the whole
    # solve...)" comment), so this doesn't require full Newton-Poisson
    # convergence to check; only that the solve runs and produces psi_e.
    device = AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=0.5)
    r = device.solve(V_applied=0.0, quantum=True, n_states_e=4, n_states_h=4, max_iter=400)

    marked_lo, marked_hi = 50.0 - 2.0, 54.0 + 2.0  # marked layers span [50,54]nm + 2nm padding
    psi2 = r.psi_e[0] ** 2
    mask = (r.x_nm >= marked_lo) & (r.x_nm <= marked_hi)
    frac_inside = np.sum(psi2[mask]) / np.sum(psi2)
    assert frac_inside > 0.5


@pytest.mark.slow
def test_surface_charge_donor_state_shifts_band_bending():
    """
    devices.layer.SurfaceCharge should ionize self-consistently with the
    local Fermi level (see physics.self_consistent's surf_idx/Nd_plus
    injection) and measurably perturb the solved potential relative to an
    otherwise-identical device with no surface charge -- a coarse sanity
    check on the physics, not a golden-value regression.
    """
    def _device(with_surface_charge: bool):
        layers = [
            AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17),
        ]
        if with_surface_charge:
            layers.append(SurfaceCharge(states=[
                SurfaceState(density_cm2=5e12, energy_eV=0.1, state_type='donor'),
            ]))
        layers.append(AbruptLayer(x_Al=0.0, thickness_nm=50, n_doping=1e17))
        contacts = [
            Contact(position='bottom', contact_type='ohmic', metal='Ti'),
            Contact(position='top', contact_type='ohmic', metal='Ti'),
        ]
        return AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=0.5)

    r_plain = _device(False).solve(V_applied=0.0, quantum=False, max_iter=400)
    r_surf = _device(True).solve(V_applied=0.0, quantum=False, max_iter=400)

    assert r_plain.converged
    assert r_surf.converged
    assert not np.allclose(r_plain.phi, r_surf.phi, atol=1e-4)


@pytest.mark.slow
def test_boundary_conditions_enforced_under_bias():
    """
    Contact boundary conditions under bias (Efn=Efp=0 at the grounded
    contact, Efn=Efp=-V at the biased one). Uses the LED with an OHMIC
    p-contact: with the original Schottky Ni p-contact (2.4 eV hole
    barrier) the p-region is cut off from its contact under forward bias
    and the drift-diffusion solve honestly reports converged=False -- see
    test_blocking_contact_fails_honestly.
    """
    device = _led_device()
    device.contacts[1].contact_type = 'ohmic'
    device.build_grid()
    V_applied = 1.5
    r = device.solve(V_applied=V_applied, quantum=False, max_iter=200, R_series=0.0)

    assert r.converged
    assert r.Efn[0] == pytest.approx(0.0, abs=1e-9)
    assert r.Efp[0] == pytest.approx(0.0, abs=1e-9)
    assert r.Efn[-1] == pytest.approx(-V_applied, abs=1e-9)
    assert r.Efp[-1] == pytest.approx(-V_applied, abs=1e-9)


@pytest.mark.slow
def test_blocking_contact_fails_honestly():
    """
    Schottky Ni p-contact: forward LED bias reverse-biases the contact, the
    p-region floats and the solve cannot resolve it in double precision.
    It must NOT report converged with fake boundary conditions: either it
    converges at V_applied, or it reports converged=False and V_internal
    equal to the bias its returned state actually corresponds to.
    """
    device = _led_device()
    V_applied = 1.5
    r = device.solve(V_applied=V_applied, quantum=False, max_iter=200, R_series=0.0)
    V_state = -r.Efn[-1]
    if r.converged:
        assert V_state == pytest.approx(V_applied, abs=1e-9)
    else:
        assert r.V_internal == pytest.approx(V_state, abs=1e-9)
        assert r.V_internal < V_applied


def _pn_diode():
    layers = [
        AbruptLayer(x_Al=0.0, thickness_nm=100, n_doping=1e18, p_doping=0.0),
        AbruptLayer(x_Al=0.0, thickness_nm=100, n_doping=0.0, p_doping=1e18),
    ]
    contacts = [
        Contact(position='bottom', contact_type='ohmic', metal='Ti'),
        Contact(position='top', contact_type='ohmic', metal='Ni'),
    ]
    return AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=1.0)


@pytest.mark.slow
def test_pn_diode_current_conserved_and_ideality():
    """
    Drift-diffusion physics check (see [[bug-current-not-conserved]]):
    a GaN p-n diode must conserve current (J_n + J_p constant through the
    device) wherever J is above double-precision resolution, show an
    SRH-dominated ideality factor ~2 (~118 mV/decade) at low bias, and
    above turn-on clamp the junction quasi-Fermi split below the bandgap
    (excess voltage drops along QFL gradients in the neutral regions).
    """
    device = _pn_diode()
    Vs = [1.5, 2.0, 3.0, 4.0]
    res = {V: device.solve(V_applied=V, quantum=False, max_iter=200, R_series=0.0) for V in Vs}
    for V, r in res.items():
        assert r.converged, V
    assert res[3.0].current_conservation_error < 1e-6
    assert res[4.0].current_conservation_error < 1e-6
    slope_mV = 1000 * 0.5 / np.log10(abs(res[2.0].J_total) / abs(res[1.5].J_total))
    assert 100 < slope_mV < 130
    mid = len(res[4.0].x_nm) // 2
    split = res[4.0].Efn[mid] - res[4.0].Efp[mid]
    assert split < 3.4          # GaN gap
    assert abs(res[4.0].J_total) > abs(res[3.0].J_total)


@pytest.mark.slow
@pytest.mark.parametrize("quantum", [False, True])
def test_flat_qfl_split_equals_bias_everywhere_but_contacts(quantum):
    """
    flat_qfl mode: Efn - Efp must equal qV_applied at every interior grid
    node (not just in the quantum region), and drop to 0 only at the two
    ohmic contact nodes, with phi carrying the full applied bias.
    """
    device = _led_device()
    V_applied = 2.0
    r = device.solve(V_applied=V_applied, quantum=quantum, max_iter=400,
                     R_series=0.0, flat_qfl=True)

    assert r.converged
    split = r.Efn - r.Efp
    np.testing.assert_allclose(split[1:-1], V_applied, atol=1e-12)
    assert split[0] == pytest.approx(0.0, abs=1e-12)
    assert split[-1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(r.Efn[:-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(r.Efp[1:], -V_applied, atol=1e-12)


@pytest.mark.slow
def test_flat_qfl_p_down_device_forward_split_positive():
    """
    flat_qfl on a p-down diode (p at the bottom contact): forward bias is
    V_applied < 0 under the contact convention, and each QFL must sit at
    its OWN side's contact level, giving split = -V_applied > 0 -- the same
    sign the drift-diffusion solve gives. Hard-coding the n-side at the
    bottom used to produce split = V_applied < 0 here.
    """
    layers = [
        AbruptLayer(x_Al=0.0, thickness_nm=100, n_doping=0.0, p_doping=1e18),
        AbruptLayer(x_Al=0.0, thickness_nm=100, n_doping=1e18, p_doping=0.0),
    ]
    contacts = [
        Contact(position='bottom', contact_type='ohmic', metal='Ni'),
        Contact(position='top', contact_type='ohmic', metal='Ti'),
    ]
    device = AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=1.0)
    V_applied = -3.0
    r = device.solve(V_applied=V_applied, quantum=False, max_iter=400,
                     R_series=0.0, flat_qfl=True)
    assert r.converged
    split = r.Efn - r.Efp
    np.testing.assert_allclose(split[1:-1], -V_applied, atol=1e-12)
    assert split[0] == pytest.approx(0.0, abs=1e-12)
    assert split[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("V_applied", [4.0, -4.0])
def test_series_resistance_ir_drop_both_polarities(monkeypatch, V_applied):
    """
    R_series logic in isolation, on a stand-in ohmic device J = V/R_dev:
    with R_dev = R_series the IR drop must take exactly half the bias at
    EITHER polarity (V_internal = V_applied / 2). Previously a +|J|*R drop
    and a [0, V_applied] clamp pinned V_internal at V_applied for V < 0.
    """
    import dataclasses
    import physics.self_consistent as sc

    @dataclasses.dataclass
    class _FakeResult:
        V_applied: float
        V_internal: float
        J_total: float
        converged: bool = True

    R_dev = 1.0
    monkeypatch.setattr(sc, "solve_self_consistent",
                        lambda grid, V_applied, **kw: _FakeResult(V_applied, V_applied, V_applied / R_dev))
    r = sc._solve_bias_with_series_resistance(
        None, V_applied, R_dev, False, 4, 4, 100, 1e-6, 0.1, 0.01, None, 0,
        False, None, None)
    assert r.converged
    assert r.V_internal == pytest.approx(V_applied / 2, abs=1e-3)

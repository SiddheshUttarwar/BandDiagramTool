"""
Bias must work on every template device, up to +/-10 V, solved the way the
GUI's Run button solves it (template settings, quantum on where the
template has it on).
"""
import numpy as np
import pytest

from devices.device import Device
from devices.layer import AbruptLayer, Contact
from gui import templates


@pytest.mark.parametrize("key", [t.key for t in templates.TEMPLATES])
def test_template_converges_up_to_ten_volts(key):
    m = templates.get(key).build()
    s = m.settings
    for V in (1.5, 10.0, -10.0):
        device = Device(list(m.layers), list(m.contacts), T=s.T, dx_nm=s.dx_nm)
        r = device.solve_ramped(V_applied=V, quantum=s.quantum, n_states_e=s.n_states_e,
                                n_states_h=s.n_states_h, max_iter=s.max_iter, tol=s.tol, alpha=s.alpha)
        assert r.converged, f"{key} at {V} V"
        assert np.all(np.isfinite(r.Ec)) and np.all(np.isfinite(r.n)) and np.all(np.isfinite(r.p))
        # the contacts carry the bias: the applied one, or for a gate driven
        # beyond its barrier the turn-on limit, which the result reports
        if r.bias_note:
            assert r.bias_mode == 'gate' and 0.0 < r.V_internal < V
        else:
            assert r.V_internal == pytest.approx(V)
        assert r.Efn[0] == pytest.approx(0.0, abs=1e-9)
        assert r.Efn[-1] == pytest.approx(-r.V_internal, abs=1e-9)


def test_cut_off_electron_gas_is_reported():
    """A 2D electron gas between a p-type buffer and a Schottky barrier has
    no path to either contact. The exact equations cannot fix its
    quasi-Fermi level; the solver falls back to a minimum density in the
    current equation, converges, and says so."""
    layers = [AbruptLayer(0.0, 300, material='AlGaAs', p_doping=1e14),
              AbruptLayer(0.0, 30, material='AlGaAs'),
              AbruptLayer(0.3, 10, material='AlGaAs'),
              AbruptLayer(0.3, 2, material='AlGaAs', n_doping=2.5e19, dx_nm=0.2),
              AbruptLayer(0.3, 25, material='AlGaAs'),
              AbruptLayer(0.0, 5, material='AlGaAs')]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Au', barrier_eV=0.8)]
    for V in (0.5, -1.0):
        # gate_bias=False: solve the vertical current through the Schottky contact
        r = Device(layers, contacts, dx_nm=1.0).solve(V_applied=V, quantum=False, gate_bias=False)
        assert r.converged and r.bias_mode == 'current'
        assert r.transport_floor_cm3 > 0.0
        # the channel sits between the two contact levels
        channel = (r.x_nm > 305) & (r.x_nm < 325)
        lo, hi = sorted((0.0, -V))
        assert lo - 1e-6 <= r.Efn[channel].mean() <= hi + 1e-6
    # a device with a current path needs no such help
    pn = Device([AbruptLayer(0.0, 200, material='AlGaAs', n_doping=1e17),
                 AbruptLayer(0.0, 200, material='AlGaAs', p_doping=1e17)],
                [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Au')], dx_nm=2.0)
    assert pn.solve(V_applied=1.0, quantum=False).transport_floor_cm3 == 0.0


def test_displacement_field():
    """D = eps F + P: its slope is the free charge density, and with no
    polarization it reduces to eps F."""
    m = templates.get("gaas_hemt").build()
    r = Device(list(m.layers), list(m.contacts), dx_nm=m.settings.dx_nm).solve(quantum=False)
    eps0 = 8.8541878128e-12
    assert np.allclose(r.D_field, eps0 * r.eps_r * r.E_field)            # zincblende: P = 0
    assert not np.any(r.E_polarization)
    rho = 1.602176634e-19 * 1e6 * (r.p - r.n)
    slope = np.gradient(r.D_field, r.x_nm * 1e-9)
    undoped = (r.ND == 0) & (r.NA == 0)
    undoped[:3] = undoped[-3:] = False
    k = int(np.argmax(np.abs(rho) * undoped))
    assert slope[k] == pytest.approx(rho[k], rel=0.1)

    n = templates.get("stark").build()
    w = Device(list(n.layers), list(n.contacts), dx_nm=n.settings.dx_nm).solve(quantum=False)
    assert np.allclose(w.D_field, eps0 * w.eps_r * w.E_field + w.P_total)
    assert np.allclose(w.E_polarization, -w.P_total / (eps0 * w.eps_r))
    # across the undoped well the polarization changes by ~0.09 C/m^2 but D stays nearly constant
    assert np.ptp(w.D_field) < 0.2 * np.ptp(w.P_total)


def test_gate_bias_option():
    """Gate voltage: no current; buffer and channel keep the source Fermi
    level, which goes linearly to the gate's across the barrier; the electron
    gas is depleted by a negative gate voltage. It is the automatic choice
    for a Schottky contact on an undoped layer, not on a doped contact layer."""
    m = templates.get("hemt").build()
    s = m.settings

    def solve(V):
        return Device(list(m.layers), list(m.contacts), dx_nm=s.dx_nm).solve(
            V_applied=V, quantum=False, gate_bias=True)

    def ns(r):
        return float(np.trapezoid(r.n, r.x_nm * 1e-7))

    r0, rn, rp = solve(0.0), solve(-2.0), solve(0.5)
    for r, V in ((rn, -2.0), (rp, 0.5)):
        assert r.converged and r.bias_mode == 'gate'
        below = r.x_nm <= 195.0                      # buffer and channel: the source level
        assert np.allclose(r.Efn[below], 0.0) and np.allclose(r.Efp[below], 0.0)
        assert np.allclose(r.Efn, r.Efp)
        barrier = r.Efn[r.x_nm >= 205.0]             # a straight line to the gate's level
        assert np.all(np.diff(barrier) * (-V) > 0)
        assert r.Efn[-1] == pytest.approx(-V)
        assert not np.isfinite(r.J_total) and r.R_srh is None
    assert ns(rn) < 0.8 * ns(r0) < ns(r0) < ns(rp)
    assert r0.bias_mode == 'equilibrium'
    # the default is the current solve
    pn = templates.get("gaas_led").build()
    r = Device(list(pn.layers), list(pn.contacts), dx_nm=2.0).solve(V_applied=1.0, quantum=False)
    assert r.bias_mode == 'current' and np.isfinite(r.J_total)


def test_automatic_bias_model():
    hemt = templates.get("hemt").build()
    r = Device(list(hemt.layers), list(hemt.contacts), dx_nm=hemt.settings.dx_nm).solve(V_applied=-1.0, quantum=False)
    assert r.bias_mode == 'gate'
    # a Schottky contact on a p-n diode is a diode contact, not a gate
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'schottky', 'Ni')]
    for p_doping in (3e17, 2e19):
        led = [AbruptLayer(0.0, 150, n_doping=2e18), AbruptLayer(0.0, 150, p_doping=p_doping)]
        assert Device(led, contacts, dx_nm=2.0).solve(V_applied=0.5, quantum=False).bias_mode == 'current'
    # the turn-on limit of a forward-biased gate is reported, not hidden
    r = Device(list(hemt.layers), list(hemt.contacts), dx_nm=hemt.settings.dx_nm).solve(V_applied=5.0, quantum=False)
    assert r.converged and r.bias_note and r.V_internal == pytest.approx(1.13, abs=0.01)

"""
Band diagram of a structure illuminated through its top surface.

Light of a given wavelength enters at the surface (the last grid node) and
is absorbed on its way down wherever the photon energy exceeds the local
gap:

    flux(z) = flux_0 exp(-int_z^surface alpha dz'),      G(z) = alpha(z) flux(z)

with alpha from physics.dqfl.absorption (zero below the gap). G is added to
the drift-diffusion continuity equations (physics.dd_newton) and Poisson and
both continuity equations are solved together, classically, starting from
the dark equilibrium and raising the light in steps.

No current is drawn. A layer that is being grown, or a wafer lying under a
lamp, has no external circuit, so after the light is on the bias between
the two ends is adjusted until the total current is zero. The bias found is
the photovoltage of the structure, and what moves in the band diagram is
the band bending the carriers screen: a surface depletion layer or a
junction flattens, and the quasi-Fermi levels separate where the light is
absorbed.

Boundaries:
  * bottom: the neutral substrate (the device's bottom contact);
  * top: the free surface (physics.dd_newton.DDProblem.surface). Its dark
    band bending is the top contact's: a Schottky barrier is a surface with
    its Fermi level pinned that far below the conduction band. Under light
    the surface states recombine electrons and holes with velocities S_n,
    S_p (Shockley-Read-Hall through states at the pinning level), and their
    occupancy -- dense states, so the pinning -- stays at its dark value:
    S_n (n - n1) = S_p (p - p1). The surface potential then follows from
    the solution: no current flows anywhere and the surface photovoltage is
    the shift of the surface potential. surface_velocity_cm_s=None keeps the
    old limit, both quasi-Fermi levels pinned together at a contact
    (infinitely fast surface recombination) with the open-circuit bias
    searched for.

Carriers are classical unless quantum=True (8-band k.p in the quantum
region, quantum-corrected drift-diffusion as in the bias solve).

Other limits: one wavelength,
no reflection or interference, no photon recycling, the solver's fixed
recombination constants (SRH 1 ns, radiative 1e-11 cm^3/s, Auger 1e-30
cm^6/s), and no light-induced change of surface-state occupation beyond
what the drift-diffusion statistics give.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from physics import dd_newton
from physics.dqfl import absorption
from physics.grid_utils import node_spacings

_Q = 1.602176634e-19
# Start of the continuation in the surface recombination velocity [log10 cm/s]:
# fast enough that the free surface is the pinned limit (the UV-LED template's
# first step failed from 1e9, its surface split there still ~3 eV)
_S_START_LOG10 = 14.0


@dataclass
class IlluminatedBands:
    dark: object                 # SolverResult at equilibrium
    light: object                # SolverResult under illumination, open circuit
    wavelength_nm: float
    photon_eV: float
    power_W_cm2: float
    generation: np.ndarray       # G(z) [cm^-3 s^-1]
    absorbed_fraction: float     # of the incident photons, within the structure
    photovoltage_V: float        # open-circuit bias, top relative to bottom
    J_residual_A_cm2: float      # total current left at the open-circuit point
    J_short_A_cm2: float         # photocurrent at zero bias
    converged: bool
    message: str = ""
    surface_velocity_cm_s: Optional[float] = None   # free top surface's S; None = pinned contact limit


def generation_profile(grid, wavelength_nm: float, power_W_cm2: float, above_gap_cm: float = 3e5):
    """G(z) [cm^-3 s^-1] for light entering at the last node, and the
    fraction of the incident photons absorbed in the structure."""
    photon_eV = 1239.841984 / float(wavelength_nm)
    flux0 = power_W_cm2 / (photon_eV * _Q)                         # photons / cm^2 / s
    Eg = np.asarray(grid.Ec0 - grid.Ev0, dtype=float)
    alpha = np.array([absorption(photon_eV, e, above_gap_cm) for e in Eg])   # 1/cm
    cw_cm = node_spacings(grid.dx, grid.N)[2] * 100.0
    tau = alpha * cw_cm                                            # optical thickness of each cell
    above = np.concatenate([np.cumsum(tau[::-1])[::-1][1:], [0.0]])   # cells between this one and the surface
    # photons absorbed in the cell / cell width: exact for a uniform cell
    G = flux0 * np.exp(-above) * (-np.expm1(-tau)) / cw_cm
    return G, float(1.0 - np.exp(-tau.sum())), photon_eV


def _surface_arrays(g):
    sites = getattr(g, 'surface_charge_sites', None) or []
    idx, dens, energy, donor = [], [], [], []
    if sites:
        cw = node_spacings(g.dx, g.N)[2]
        for i0, states in sites:
            for state in states:
                idx.append(i0)
                dens.append(state.density_cm2 / (cw[i0] * 100.0))
                energy.append(state.energy_eV)
                donor.append(state.state_type == 'donor')
    return (np.array(idx, dtype=int), np.array(dens), np.array(energy), np.array(donor, dtype=bool))


def solve_illuminated(device, wavelength_nm: float = 300.0, power_W_cm2: float = 1.0,
                      above_gap_cm: float = 3e5, dark=None,
                      log_fn: Optional[Callable[[str], None]] = None,
                      surface_velocity_cm_s: Optional[float] = 1e4,
                      quantum: bool = False) -> IlluminatedBands:
    """Bands of `device` (an AlGaNDevice, at its own temperature) under
    light through the top surface, at open circuit. `dark` is its classical
    equilibrium SolverResult; it is computed if not given.
    surface_velocity_cm_s: recombination velocity of the free top surface
    (S_n = S_p); None = the top contact's Dirichlet limit (see module doc).
    quantum: 8-band k.p carriers in the device's quantum region."""
    log = log_fn or (lambda _s: None)
    if dark is None:
        dark = device.solve(V_applied=0.0, quantum=False)
    g = device.build_grid()
    from physics.self_consistent import _dd_problem, dopant_energies
    Ed, Ea = dopant_energies(g)
    prob = _dd_problem(g, Ed, Ea, *_surface_arrays(g))
    G, absorbed, photon_eV = generation_profile(g, wavelength_nm, power_W_cm2, above_gap_cm)
    # Free surface: solved last, by continuation in S from the pinned-contact
    # limit at full light (below). Ramping the light with the free surface
    # from the start fails on a p-n diode: its top layer then floats, and at
    # 1e-9 of the light its potential is set by currents ~1e-30 A/cm^2 that
    # double precision does not resolve (confirmed on the GaN p-n template).
    free_surface = surface_velocity_cm_s is not None
    ln_n0, ln_p0, _, _ = prob.carriers(np.asarray(dark.phi, dtype=float), np.zeros(g.N), np.zeros(g.N))
    n1_s, p1_s = float(np.exp(ln_n0[-1])), float(np.exp(ln_p0[-1]))

    def bc(v):
        return (g.phi_bottom_eq, g.phi_top_eq + v, 0.0, -v, 0.0, -v)

    def current(state):
        return float(np.mean(dd_newton.current_profile(prob, state.phi, state.Efn, state.Efp, state.lo)[2]))

    zeros = np.zeros(g.N)
    state = dd_newton.DDResult(np.asarray(dark.phi, dtype=float).copy(), zeros.copy(), zeros.copy(), True, 0, 0.0)

    # ---- turn the light on in steps (continuation in intensity) ----
    # The first step must be tiny: a minority density of ~1e-37 cm^-3 (wide
    # gap, either side of a p-n junction) rises by tens of decades at even
    # 1e-20 of the light, and Newton from the dark state diverges (confirmed
    # on a GaN p-n diode, which failed at 1e-12; it converges from 1e-40).
    ok, message = True, ""
    if np.max(G) > 0:
        level, step = -40.0, 2.0                     # log10 of the fraction of full intensity
        while level < 0.0:
            trial = min(0.0, level + step)
            prob.G_opt = G * 10.0 ** trial
            res = dd_newton.newton_solve(prob, state.phi, state.Efn, state.Efp, bc(0.0), maxiter=60, lo=state.lo)
            if res.converged:
                state, level = res, trial
                step = min(step * 1.5, 3.0)
            else:
                step *= 0.5
                if step < 0.02:
                    ok, message = False, f"stopped at 10^{level:.1f} of the full intensity"
                    break
        log(f"light on: {'full intensity' if ok else message}")
    prob.G_opt = G if ok else G * 10.0 ** level
    j_short = current(state)

    # ---- open circuit: move the top bias until the total current is zero ----
    def solve_at(v_new, start):
        return dd_newton.newton_solve(prob, start.phi, start.Efn, start.Efp, bc(v_new), maxiter=60, lo=start.lo)

    def open_circuit(start, v0, j0):
        """(state, v, j, message) at zero total current, searched from the
        converged state `start` at top bias v0 carrying current j0."""
        msg = ""
        # First probe: a small step either way. At high temperature the
        # contacts pass so much thermal current that even this overshoots
        # the zero, in which case it already brackets it.
        direction, bracket = None, None
        for sign in (+1.0, -1.0):
            res = solve_at(v0 + sign * 0.02, start)
            if not res.converged:
                continue
            jn = current(res)
            if jn * j0 <= 0.0:
                bracket = (v0, start, j0, v0 + sign * 0.02, res, jn)
                break
            if abs(jn) < abs(j0):
                direction = sign
                break
        if direction is None and bracket is None:
            return start, v0, j0, "open-circuit search did not start; shown at zero bias"
        dv, lo_state, lo_v, lo_j = 0.05, start, v0, j0
        while bracket is None and abs(lo_v) < 8.0:
            res = solve_at(lo_v + direction * dv, lo_state)
            if not res.converged:
                dv *= 0.5
                if dv < 1e-4:
                    msg = f"open-circuit search stalled at {lo_v:+.3f} V"
                    break
                continue
            jn = current(res)
            if jn * lo_j <= 0.0:
                bracket = (lo_v, lo_state, lo_j, lo_v + direction * dv, res, jn)
                break
            lo_state, lo_v, lo_j = res, lo_v + direction * dv, jn
            dv = min(dv * 1.3, 0.2)
        st, v, jj = lo_state, lo_v, lo_j
        if bracket is not None:
            a_v, a_s, a_j, b_v, b_s, b_j = bracket
            for _ in range(80):
                m_v = 0.5 * (a_v + b_v)
                res = solve_at(m_v, a_s)
                if not res.converged:
                    break
                m_j = current(res)
                if m_j * a_j <= 0.0:
                    b_v, b_s, b_j = m_v, res, m_j
                else:
                    a_v, a_s, a_j = m_v, res, m_j
                if abs(b_v - a_v) < 1e-12 or min(abs(a_j), abs(b_j)) < 1e-6 * abs(j_short):
                    break
            st, v, jj = (a_s, a_v, a_j) if abs(a_j) < abs(b_j) else (b_s, b_v, b_j)
        return st, v, jj, msg

    state_oc, j_oc, v_oc = state, j_short, 0.0
    # (a photocurrent below 1e-12 A/cm^2 is zero for this purpose: nothing to balance)
    if ok and abs(j_short) > 1e-12:
        state_oc, v_oc, j_oc, message = open_circuit(state, 0.0, j_short)
        log(f"open circuit: V = {v_oc:+.4f} V, residual current {j_oc:.2e} A/cm^2 "
            f"(short-circuit {j_short:.2e} A/cm^2)")

    # ---- free surface: continuation in S from the pinned limit ----
    if ok and free_surface and np.max(G) > 0:      # (no light absorbed: the dark state is the answer)
        S_target = float(surface_velocity_cm_s)
        # the pinned solution holds the surface node's QFLs at the contact
        # value, ~1e-60 cm^-3 minority electrons under strong generation
        # there: as a free-surface start that node is far off (UV-LED: Newton
        # diverged at once even at 1e14 cm/s). Its neighbour holds the
        # generation-balanced density; the surface starts from it.
        Efn0, Efp0 = state_oc.Efn.copy(), state_oc.Efp.copy()
        Efn0[-1], Efp0[-1] = Efn0[-2], Efp0[-2]
        lo0 = None if state_oc.lo is None else tuple(np.array(a, dtype=float) for a in state_oc.lo)
        if lo0 is not None:
            for arr in lo0:
                arr[-1] = arr[-2]
        st = dd_newton.DDResult(state_oc.phi.copy(), Efn0, Efp0, True, 0, 0.0,
                                Efn_lo=None if lo0 is None else lo0[0], Efp_lo=None if lo0 is None else lo0[1])
        S_now = None
        lgS, lg_target = _S_START_LOG10, np.log10(S_target)
        step = 1.0
        bc_oc = bc(v_oc)
        while True:
            trial = max(lg_target, lgS) if S_now is None else max(lg_target, lgS - step)
            prob.surface = (10.0 ** trial, 10.0 ** trial, n1_s, p1_s)
            res = dd_newton.newton_solve(prob, st.phi, st.Efn, st.Efp, bc_oc, maxiter=60, lo=st.lo)
            if res.converged:
                st, S_now, lgS = res, 10.0 ** trial, trial
                if trial <= lg_target:
                    break
                step = min(step * 1.5, 2.0)
            else:
                step *= 0.5
                if step < 0.02 or S_now is None:
                    ok, message = False, (f"free surface: stopped at S = {S_now:.3g} cm/s" if S_now
                                          else f"free surface: no solution at S = 1e{_S_START_LOG10:.0f} cm/s")
                    break
        state_oc = st
        v_oc = float(st.phi[-1] - g.phi_top_eq)
        j_oc = current(st)
        j_short = float('nan')          # no contact at the top: nothing to short
        log(f"free surface, S = {S_now if S_now else float('nan'):.3g} cm/s: surface photovoltage {v_oc:+.4f} V")

    # ---- quantum carriers: 8-band k.p in the quantum region ----
    # Quantum-corrected drift-diffusion as in the bias solve: the k.p
    # densities of the illuminated state (each state filled with the local
    # quasi-Fermi levels) give gamma = n_kp / n_classical per node, folded
    # into the carrier densities of the equations; the state is re-solved
    # with gamma frozen (and, in the pinned limit, brought back to zero
    # current) until gamma no longer changes.
    kp_display = None
    if ok and quantum and np.max(G) > 0:
        from physics.fermi_dirac import electron_density, hole_density
        from physics.kp.model import KPModel
        from physics.self_consistent import (_detect_quantum_region, _quantum_gamma, _blended_density,
                                             _GAMMA_MISMATCH_TOL, _GAMMA_MISMATCH_MIN_DENSITY,
                                             _QUANTUM_GAMMA_BETA_MIN)
        T = float(device.T)
        region = getattr(g, 'manual_quantum_region', None) or _detect_quantum_region(g)
        mask = np.zeros(g.N, dtype=bool)
        mask[region[0]:region[1]] = True
        kp_model = KPModel(g, region[0], region[1])
        lg_n = np.zeros(g.N)
        lg_p = np.zeros(g.N)
        beta, mism_prev, q_ok = 1.0, np.inf, False
        for it in range(40):
            Ec, Ev = g.Ec0 - state_oc.phi, g.Ev0 - state_oc.phi
            sol = kp_model.solve(Ec, Ev, state_oc.Efn, state_oc.Efp, T, True)
            n_q, p_q = kp_model.densities(sol, state_oc.Efn, state_oc.Efp, T, True, Ec=Ec, Ev=Ev)
            n_cl = electron_density(Ec, state_oc.Efn, g.Nc, T)
            p_cl = hole_density(Ev, state_oc.Efp, g.Nv, T)
            n_qb, p_qb = _blended_density(n_q, n_cl, mask), _blended_density(p_q, p_cl, mask)
            new_n = np.log(_quantum_gamma(n_qb, n_cl, mask))
            new_p = np.log(_quantum_gamma(p_qb, p_cl, mask))
            mism = max(float(np.max(np.abs(new_n - lg_n)[n_qb > _GAMMA_MISMATCH_MIN_DENSITY], initial=0.0)),
                       float(np.max(np.abs(new_p - lg_p)[p_qb > _GAMMA_MISMATCH_MIN_DENSITY], initial=0.0)))
            log(f"k.p under light {it + 1}: |dln(gamma)| = {mism:.2e}")
            kp_display = (kp_model, sol)
            if mism < _GAMMA_MISMATCH_TOL:
                q_ok = True
                break
            beta = min(1.0, beta * 1.5) if mism < mism_prev else max(_QUANTUM_GAMMA_BETA_MIN, beta * 0.5)
            mism_prev = mism
            lg_n = lg_n + beta * (new_n - lg_n)
            lg_p = lg_p + beta * (new_p - lg_p)
            prob._ln_gn, prob._ln_gp = lg_n.copy(), lg_p.copy()
            res = solve_at(v_oc, state_oc)
            if not res.converged:
                ok, message = False, f"k.p under light: drift-diffusion did not converge at outer step {it + 1}"
                break
            state_oc, j_oc = res, current(res)
            if free_surface:
                v_oc = float(state_oc.phi[-1] - g.phi_top_eq)
            elif abs(j_short) > 1e-12:
                state_oc, v_oc, j_oc, message = open_circuit(state_oc, v_oc, j_oc)
        if ok and not q_ok:
            ok, message = False, "k.p under light: quantum correction did not settle in 40 steps"

    phi = state_oc.phi
    ln_n, ln_p, _dn, _dp = prob.carriers(phi, state_oc.Efn, state_oc.Efp)
    shift = np.asarray(dark.phi, dtype=float) - phi           # every band edge moves by -(phi - phi_dark)
    x_m = np.asarray(dark.x_nm, dtype=float) * 1e-9
    field_dark = -np.gradient(np.asarray(dark.phi, dtype=float), x_m)
    sign = 1.0 if float(np.dot(field_dark, np.asarray(dark.E_field, dtype=float))) >= 0 else -1.0

    def moved(arr):
        return None if arr is None else np.asarray(arr, dtype=float) + shift

    light = replace(
        dark, phi=phi, Ec=g.Ec0 - phi, Ev=g.Ev0 - phi, Ei=moved(dark.Ei),
        Ev_hh=moved(dark.Ev_hh), Ev_lh=moved(dark.Ev_lh), Ev_so=moved(dark.Ev_so),
        Efn=state_oc.Efn, Efp=state_oc.Efp, n=np.exp(ln_n), p=np.exp(ln_p),
        E_field=sign * -np.gradient(phi, x_m), V_applied=float(v_oc), V_internal=float(v_oc),
        J_total=float(j_oc), converged=bool(ok), qcse_transition_eV=None, qcse_overlap=None,
        E_e=None, psi_e=None, E_h=None, psi_h=None)
    if kp_display is not None:
        E_e, psi_e, E_h, psi_h, _char, _lv_e, _lv_h = kp_display[0].display_states(kp_display[1])
        light = replace(light, E_e=E_e, psi_e=psi_e, E_h=E_h, psi_h=psi_h)
    return IlluminatedBands(dark, light, float(wavelength_nm), photon_eV, float(power_W_cm2), G, absorbed,
                            float(v_oc), float(j_oc), float('nan') if free_surface else float(j_short), bool(ok), message,
                            surface_velocity_cm_s if free_surface else None)

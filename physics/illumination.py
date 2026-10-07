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

Boundaries. The solver knows contacts only, so the two ends are:
  * bottom: the neutral substrate (the device's bottom contact);
  * top: the free surface, represented by the device's top contact. A
    Schottky top contact with a barrier is a surface with its Fermi level
    pinned that far below the conduction band; an ohmic one is a flat-band
    surface. Either way both quasi-Fermi levels are pinned together there,
    i.e. the surface recombines carriers infinitely fast. A real surface
    has a finite recombination velocity, so the splitting right at the
    surface is underestimated.

Other limits: classical carriers (no Schrodinger equation), one wavelength,
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
                      log_fn: Optional[Callable[[str], None]] = None) -> IlluminatedBands:
    """Bands of `device` (an AlGaNDevice, at its own temperature) under
    light through the top surface, at open circuit. `dark` is its classical
    equilibrium SolverResult; it is computed if not given."""
    log = log_fn or (lambda _s: None)
    if dark is None:
        dark = device.solve(V_applied=0.0, quantum=False)
    g = device.build_grid()
    from physics.self_consistent import _dd_problem, dopant_energies
    Ed, Ea = dopant_energies(g)
    prob = _dd_problem(g, Ed, Ea, *_surface_arrays(g))
    G, absorbed, photon_eV = generation_profile(g, wavelength_nm, power_W_cm2, above_gap_cm)

    def bc(v):
        return (g.phi_bottom_eq, g.phi_top_eq + v, 0.0, -v, 0.0, -v)

    def current(state):
        return float(np.mean(dd_newton.current_profile(prob, state.phi, state.Efn, state.Efp)[2]))

    zeros = np.zeros(g.N)
    state = dd_newton.DDResult(np.asarray(dark.phi, dtype=float).copy(), zeros.copy(), zeros.copy(), True, 0, 0.0)

    # ---- turn the light on in steps (continuation in intensity) ----
    ok, message = True, ""
    if np.max(G) > 0:
        level, step = -12.0, 2.0                     # log10 of the fraction of full intensity
        while level < 0.0:
            trial = min(0.0, level + step)
            prob.G_opt = G * 10.0 ** trial
            res = dd_newton.newton_solve(prob, state.phi, state.Efn, state.Efp, bc(0.0), maxiter=60)
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
    v, j = 0.0, j_short
    state_oc, j_oc, v_oc = state, j_short, 0.0
    # (a photocurrent below 1e-12 A/cm^2 is zero for this purpose: nothing to balance)
    if ok and abs(j_short) > 1e-12:
        def solve_at(v_new, start):
            return dd_newton.newton_solve(prob, start.phi, start.Efn, start.Efp, bc(v_new), maxiter=60)

        # First probe: a small step either way. At high temperature the
        # contacts pass so much thermal current that even this overshoots
        # the zero, in which case it already brackets it.
        direction, bracket = None, None
        for sign in (+1.0, -1.0):
            res = solve_at(sign * 0.02, state)
            if not res.converged:
                continue
            jn = current(res)
            if jn * j_short <= 0.0:
                bracket = (0.0, state, j_short, sign * 0.02, res, jn)
                break
            if abs(jn) < abs(j_short):
                direction = sign
                break
        if direction is None and bracket is None:
            message = "open-circuit search did not start; shown at zero bias"
        else:
            dv, lo_state, lo_v, lo_j = 0.05, state, 0.0, j_short
            while bracket is None and abs(lo_v) < 8.0:
                res = solve_at(lo_v + direction * dv, lo_state)
                if not res.converged:
                    dv *= 0.5
                    if dv < 1e-4:
                        message = f"open-circuit search stalled at {lo_v:+.3f} V"
                        break
                    continue
                jn = current(res)
                if jn * lo_j <= 0.0:
                    bracket = (lo_v, lo_state, lo_j, lo_v + direction * dv, res, jn)
                    break
                lo_state, lo_v, lo_j = res, lo_v + direction * dv, jn
                dv = min(dv * 1.3, 0.2)
            state_oc, v_oc, j_oc = lo_state, lo_v, lo_j
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
                state_oc, v_oc, j_oc = (a_s, a_v, a_j) if abs(a_j) < abs(b_j) else (b_s, b_v, b_j)
        log(f"open circuit: V = {v_oc:+.4f} V, residual current {j_oc:.2e} A/cm^2 "
            f"(short-circuit {j_short:.2e} A/cm^2)")

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
    return IlluminatedBands(dark, light, float(wavelength_nm), photon_eV, float(power_W_cm2), G, absorbed,
                            float(v_oc), float(j_oc), float(j_short), bool(ok), message)

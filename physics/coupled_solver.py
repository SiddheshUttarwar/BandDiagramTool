"""
Fully-coupled Newton solver for the biased (V_applied != 0) drift-diffusion
system: phi, Efn, Efp solved *simultaneously* as one 3N-unknown nonlinear
system, instead of physics.self_consistent's historical Gummel-style
sequential update (drift-diffusion continuity step, then a separate
linearised Poisson step, then damped mixing between them).

That sequential scheme has no globalization of its own on the
drift-diffusion half: the continuity solve is a single linear solve given
the *previous* iteration's potential, with no line search or step-size
control. physics.self_consistent's own Poisson sub-step got an Armijo line
search (see _nonlinear_poisson_residual_norm there), but that only helped
the Poisson half -- the continuity half remained the actual bottleneck,
confirmed empirically (the historical high-bias wall persisted almost
unchanged after the Poisson-only line search).

This module solves all three equations together via Jacobian-Free
Newton-Krylov (scipy.optimize.newton_krylov): the residual function below
is assembled directly from the same, already-validated building blocks
used elsewhere (physics.fermi_dirac's Fermi-Dirac densities, physics.
drift_diffusion's Scharfetter-Gummel Bernoulli-function flux stencil,
physics.poisson's finite-volume Laplacian assembly) -- re-derived here only
as *residuals* of those exact discretizations (matrix @ unknown - rhs),
not re-derived physics. JFNK needs no explicit Jacobian (it approximates
Krylov-subspace directional derivatives numerically), and scipy's
implementation includes an Armijo line search by default, which is exactly
the missing piece: global convergence robustness for the coupled system,
without hand-deriving and debugging a 3N x 3N analytic block Jacobian.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

from scipy.optimize import newton_krylov, NoConvergence

from physics.constants import q as _q, eps0
from physics.grid_utils import node_spacings
from physics.poisson import _assemble_laplacian
from physics.fermi_dirac import (
    electron_density, hole_density,
    quantum_electron_density, quantum_hole_density,
)
from physics.drift_diffusion import bernoulli, compute_recombination


class SolveCancelled(Exception):
    """Raised (from the newton_krylov callback, or the outer Gummel loop in
    physics.self_consistent) to unwind a solve early when the user clicks
    Stop -- distinct from NoConvergence/a genuine solver failure."""


@dataclass
class QuantumState:
    """Frozen (not re-solved) Schrodinger states, held fixed for one coupled
    solve -- physics.self_consistent's existing pattern of solving
    Schrodinger with the band profile from the *previous* outer iteration.

    Holes are the decoupled 3-band effective-mass model (HH/LH/SO, see
    physics.materials.algan.AlGaNParams.valence_band_structure): three
    independent single-band confined-state solutions, whose quantum
    densities are simply summed (no HH-LH-SO mixing) -- see p_of() below."""
    psi_e: np.ndarray
    E_e: np.ndarray
    m_e: np.ndarray
    psi_h: np.ndarray
    E_h: np.ndarray
    m_hh: np.ndarray
    psi_h_lh: np.ndarray
    E_h_lh: np.ndarray
    m_lh: np.ndarray
    psi_h_so: np.ndarray
    E_h_so: np.ndarray
    m_so: np.ndarray
    region_mask: np.ndarray


@dataclass
class CoupledResult:
    phi: np.ndarray
    Efn: np.ndarray
    Efp: np.ndarray
    converged: bool
    n_iter: int
    final_residual: float


def _make_residual_fn(
    N: int, Ec0: np.ndarray, Ev0: np.ndarray, Nc: np.ndarray, Nv: np.ndarray,
    ND: np.ndarray, NA: np.ndarray, Ed_x: np.ndarray, Ea_x: np.ndarray,
    pol_rho: np.ndarray, eps_r: np.ndarray, dx, T: float, kBT_eV: float,
    mu_n: float, mu_p: float,
    phi_left: float, phi_right: float,
    Efn_left: float, Efn_right: float, Efp_left: float, Efp_right: float,
    surf_idx: np.ndarray, surf_density_cm3: np.ndarray,
    surf_energy_eV: np.ndarray, surf_is_donor: np.ndarray,
    quantum_state: Optional[QuantumState],
) -> Callable[[np.ndarray], np.ndarray]:
    eps_m, eps_p, h1, h2, cell_width = _assemble_laplacian(eps_r, dx)
    cw = cell_width[1:-1]

    dx_cm = np.asarray(dx, dtype=float) * 1e2
    h1_cm, h2_cm, cell_width_cm, _edges_cm = node_spacings(dx_cm, N)
    cw_cm = cell_width_cm[1:-1]
    D_n = mu_n * kBT_eV
    D_p = mu_p * kBT_eV
    coeff_r_n = D_n / (h2_cm * cw_cm)
    coeff_l_n = D_n / (h1_cm * cw_cm)
    coeff_r_p = D_p / (h2_cm * cw_cm)
    coeff_l_p = D_p / (h1_cm * cw_cm)

    # Fixed (state-independent) per-block scale references, computed once
    # from problem data. This is deliberately *not* a per-point
    # self-normalizing scale (max(|term|,...) recomputed at every trial
    # state): dividing rho by eps0 alone inflates even a genuinely
    # negligible charge imbalance into a huge number (eps0 ~ 1e-11), so a
    # per-point scale built from that same inflated quantity can't tell
    # "converged" from "not" -- verified empirically: at a fully converged
    # equilibrium solution (independently validated elsewhere, golden-value
    # tested), the old per-point scheme still reported an O(1) normalized
    # residual regardless of doping level. Using one fixed, physically
    # meaningful reference per block (the charge scale set by the device's
    # own peak doping; the flux/recombination scale set by its own peak
    # diffusive flux) fixes that, and being state-independent it also can't
    # introduce state-dependent kinks that would confuse JFNK's
    # finite-difference Jacobian approximation.
    # Per-point, not a single device-wide scalar: a device with one very
    # heavily doped layer (e.g. a 1e20 cm^-3 contact next to 1e19/1e17
    # neighbors) would otherwise set carrier_scale from that one outlier
    # layer's doping alone, making the *normalized* residual in every
    # lower-doped region artificially tiny (falsely "already converged")
    # long before those regions reach their true biased solution -- JFNK
    # then stops with the rest of the device essentially frozen near its
    # previous/equilibrium state while all the adjustment concentrates in
    # the one layer the scale was actually calibrated to (confirmed: this
    # produced exactly that symptom -- the whole applied bias landing in
    # one heavily-doped layer, everything else showing equilibrium-like
    # quasi-Fermi levels). Built once from ND/NA (state-independent, same
    # as before) so it still can't introduce state-dependent kinks that
    # would confuse JFNK's finite-difference Jacobian approximation.
    #
    # A single shared carrier-density scale (not tied separately to ND vs
    # NA) for both flux references: in a one-sided-doped region the
    # minority carrier can be many orders of magnitude below the *other*
    # dopant's density. Tried splitting this into a per-carrier ND-only /
    # NA-only scale (2026-08-26): it does correctly reveal that the
    # quasi-Fermi pinning bug's flat-except-boundary state has an enormous
    # *raw* (unscaled) residual at the pinned boundary edge (~1e24, not
    # small) -- so the pinning is a genuine masking-by-scale artifact,
    # confirmed empirically, not a fundamentally-degenerate discretization.
    # But using ND/NA directly (undoing the shared max(ND,NA) here) also
    # collapses the *other* carrier's scale to the bare 1.0 floor at every
    # ordinary one-sided-doped node throughout the WHOLE device (not just
    # the extreme repro region) -- confirmed empirically: this broke JFNK
    # globally, failing to converge even the first small ramp step
    # ("Jacobian inversion yielded zero vector"). So the fix needs a scale
    # that shrinks specifically where the flux-vs-scale mismatch is
    # extreme (as here) without collapsing to the floor for every routine
    # one-sided-doped node -- not yet found; reverted to the known-working
    # shared scale below. See [[bug-quasi-fermi-pinning]] project memory.
    carrier_scale = np.maximum(np.maximum(ND, NA), 1e10)[1:-1]
    poisson_scale = np.maximum(1.0, _q * carrier_scale * 1e6)
    n_flux_scale = np.maximum(1.0, coeff_r_n * carrier_scale * 1e6)
    p_flux_scale = np.maximum(1.0, coeff_r_p * carrier_scale * 1e6)

    def n_of(Ec, Efn):
        n_cl = electron_density(Ec, Efn, Nc, T)
        if quantum_state is None:
            return n_cl
        n_q = quantum_electron_density(quantum_state.psi_e, quantum_state.E_e,
                                        Efn, quantum_state.m_e, T)
        return np.where(quantum_state.region_mask, n_q, n_cl)

    def p_of(Ev, Efp):
        p_cl = hole_density(Ev, Efp, Nv, T)
        if quantum_state is None:
            return p_cl
        qs = quantum_state
        p_q = (quantum_hole_density(qs.psi_h, qs.E_h, Efp, qs.m_hh, T)
               + quantum_hole_density(qs.psi_h_lh, qs.E_h_lh, Efp, qs.m_lh, T)
               + quantum_hole_density(qs.psi_h_so, qs.E_h_so, Efp, qs.m_so, T))
        return np.where(quantum_state.region_mask, p_q, p_cl)

    def residual(state: np.ndarray) -> np.ndarray:
        phi = state[:N]
        Efn = state[N:2 * N]
        Efp = state[2 * N:3 * N]
        Ec = Ec0 - phi
        Ev = Ev0 - phi
        n = n_of(Ec, Efn)
        p = p_of(Ev, Efp)

        # --- Poisson: F1 = A_pos@phi - rho/eps0 (see physics.poisson /
        # physics.self_consistent._nonlinear_poisson_residual_norm) ---
        exp_arg_d = np.clip((Efn - (Ec - Ed_x)) / kBT_eV, -340.0, 340.0)
        exp_arg_a = np.clip((Ev + Ea_x - Efp) / kBT_eV, -340.0, 340.0)
        Nd_plus = ND / (1.0 + 2.0 * np.exp(exp_arg_d))
        Na_minus = NA / (1.0 + 4.0 * np.exp(exp_arg_a))
        if len(surf_idx) > 0:
            donor_mask = surf_is_donor
            if np.any(donor_mask):
                idx_d = surf_idx[donor_mask]
                exp_arg_sd = np.clip(
                    (Efn[idx_d] - (Ec[idx_d] - surf_energy_eV[donor_mask])) / kBT_eV,
                    -340.0, 340.0)
                np.add.at(Nd_plus, idx_d,
                          surf_density_cm3[donor_mask] / (1.0 + 2.0 * np.exp(exp_arg_sd)))
            acceptor_mask = ~surf_is_donor
            if np.any(acceptor_mask):
                idx_a = surf_idx[acceptor_mask]
                exp_arg_sa = np.clip(
                    (Ev[idx_a] + surf_energy_eV[acceptor_mask] - Efp[idx_a]) / kBT_eV,
                    -340.0, 340.0)
                np.add.at(Na_minus, idx_a,
                          surf_density_cm3[acceptor_mask] / (1.0 + 4.0 * np.exp(exp_arg_sa)))

        rho = _q * (p * 1e6 - n * 1e6 + Nd_plus * 1e6 - Na_minus * 1e6) + pol_rho
        lap = (eps_m / h1) * (phi[1:-1] - phi[:-2]) / cw \
            + (eps_p / h2) * (phi[1:-1] - phi[2:]) / cw
        # Residual multiplied through by eps0 (equivalent equation: A_pos@phi
        # = rho/eps0  <=>  eps0*A_pos@phi = rho) so both terms stay in
        # ordinary charge-density units (~C/m^3) instead of the ~1/eps0 ~
        # 1e11-inflated scale that made a negligible imbalance look huge --
        # see poisson_scale above.
        F1_interior = eps0 * lap - rho[1:-1]
        F1 = np.empty(N)
        F1[1:-1] = F1_interior / poisson_scale
        F1[0] = phi[0] - phi_left
        F1[-1] = phi[-1] - phi_right

        # --- Electron / hole continuity (Scharfetter-Gummel), residuals of
        # the exact same matrix physics.drift_diffusion.solve_continuity_*
        # builds (div(J)/q - R = 0 for electrons; identical row structure
        # for holes, see solve_continuity_hole) ---
        ni = np.sqrt(Nc * Nv) * np.exp(-(Ec - Ev) / (2.0 * kBT_eV))
        R = compute_recombination(n, p, ni)

        # gamma_n/gamma_p (Fermi-Dirac vs. Boltzmann degeneracy correction)
        # and psi_n_eff/psi_p_eff were investigated as the suspected cause
        # of the quasi-Fermi pinning bug (see project memory,
        # [[bug-quasi-fermi-pinning]]) and RULED OUT: psi_n_eff
        # algebraically simplifies to exactly kBT*ln(n) - Efn + const
        # regardless of gamma_n's value (n and n_boltz's Efn/Ec dependence
        # cancels), which makes the resulting flux formula an exact
        # discretization of J_n = mu_n*n*dEfn/dx for small edge-to-edge
        # changes (verified by Taylor expansion) -- correct, not buggy.
        # n_flux_scale/p_flux_scale below ARE part of the real mechanism
        # (confirmed: the pinned solution's raw, unscaled residual at the
        # boundary edge is enormous, ~1e24, not small -- so it genuinely is
        # a scale-masking artifact, not a fundamentally degenerate fixed
        # point) but the fix isn't as simple as scaling each carrier off
        # its own dopant -- that breaks JFNK globally. See the scale
        # comment above this function for what was tried and why it
        # didn't work; still unresolved.
        n_boltz = np.maximum(Nc * np.exp(np.clip((Efn - Ec) / kBT_eV, -200, 200)), 1e-30)
        gamma_n = np.maximum(n, 1e-30) / n_boltz
        psi_n_eff = -Ec + kBT_eV * np.log(Nc / Nc[0]) + kBT_eV * np.log(np.maximum(gamma_n, 1e-10))

        p_boltz = np.maximum(Nv * np.exp(np.clip((Ev - Efp) / kBT_eV, -200, 200)), 1e-30)
        gamma_p = np.maximum(p, 1e-30) / p_boltz
        psi_p_eff = Ev + kBT_eV * np.log(Nv / Nv[0]) + kBT_eV * np.log(np.maximum(gamma_p, 1e-10))

        dpsi_n = (psi_n_eff[1:] - psi_n_eff[:-1]) / kBT_eV
        Bpos_n = bernoulli(dpsi_n)
        # Flux brackets use the Bernoulli identity B(x)-B(-x) = -x to avoid
        # catastrophic cancellation: n[i+1]*Bpos-n[i]*Bneg, computed
        # directly, subtracts two products that are each ~coeff*n (huge in
        # heavily-doped regions, since coeff ~ D/dx^2 can reach ~1e15 and
        # n ~1e18-1e19) to recover a net flux divergence that should be
        # comparable to R (~1e16-1e20) -- a difference of ~30-orders-of-
        # magnitude terms representing an ~18-order-of-magnitude result
        # blows straight through double precision. Rewriting the bracket as
        # Bpos*(n[i+1]-n[i]) - n[i]*dpsi instead subtracts the *densities*
        # first (an accurate, unamplified difference of neighbouring values)
        # before multiplying by the coefficient, which is the standard
        # numerically-stable form of the Scharfetter-Gummel flux used in
        # TCAD Newton solvers for exactly this reason. Algebraically
        # identical to the direct form -- same equation, stable arithmetic.
        dpsi_n_r, dpsi_n_l = dpsi_n[1:], dpsi_n[:-1]
        right_n = Bpos_n[1:] * (n[2:] - n[1:-1]) - n[1:-1] * dpsi_n_r
        left_n = Bpos_n[:-1] * (n[1:-1] - n[:-2]) - n[:-2] * dpsi_n_l
        Jdiv_n = coeff_r_n * right_n - coeff_l_n * left_n
        F2_interior = Jdiv_n - R[1:-1]
        F2 = np.empty(N)
        F2[1:-1] = F2_interior / n_flux_scale
        F2[0] = Efn[0] - Efn_left
        F2[-1] = Efn[-1] - Efn_right

        dpsi_p = (psi_p_eff[1:] - psi_p_eff[:-1]) / kBT_eV
        Bpos_p = bernoulli(dpsi_p)
        dpsi_p_r, dpsi_p_l = dpsi_p[1:], dpsi_p[:-1]
        right_p = Bpos_p[1:] * (p[2:] - p[1:-1]) - p[1:-1] * dpsi_p_r
        left_p = Bpos_p[:-1] * (p[1:-1] - p[:-2]) - p[:-2] * dpsi_p_l
        Jdiv_p = coeff_r_p * right_p - coeff_l_p * left_p
        F3_interior = Jdiv_p - R[1:-1]
        F3 = np.empty(N)
        F3[1:-1] = F3_interior / p_flux_scale
        F3[0] = Efp[0] - Efp_left
        F3[-1] = Efp[-1] - Efp_right

        return np.concatenate([F1, F2, F3])

    return residual


def solve_coupled_dd(
    phi0: np.ndarray, Efn0: np.ndarray, Efp0: np.ndarray,
    Ec0: np.ndarray, Ev0: np.ndarray, Nc: np.ndarray, Nv: np.ndarray,
    ND: np.ndarray, NA: np.ndarray, Ed_x: np.ndarray, Ea_x: np.ndarray,
    pol_rho: np.ndarray, eps_r: np.ndarray, dx, T: float,
    mu_n: float, mu_p: float,
    phi_left: float, phi_right: float,
    Efn_left: float, Efn_right: float, Efp_left: float, Efp_right: float,
    surf_idx: np.ndarray, surf_density_cm3: np.ndarray,
    surf_energy_eV: np.ndarray, surf_is_donor: np.ndarray,
    quantum_state: Optional[QuantumState] = None,
    tol: float = 1e-3,
    maxiter: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CoupledResult:
    """
    Jointly solve Poisson + electron continuity + hole continuity for a
    fixed set of Dirichlet boundary conditions, via Jacobian-Free
    Newton-Krylov (see module docstring). Returns the converged (or
    best-effort, if NoConvergence) state.

    tol here is f_tol on the self-normalized combined residual (each of the
    three physics blocks is scaled by its own local magnitude before
    concatenation -- see _make_residual_fn -- so a single O(1e-3) tolerance
    is meaningful across all three regardless of their very different
    physical units/scales).
    """
    from physics.constants import kB
    kBT_eV = kB * T / _q
    N = len(phi0)

    residual_fn = _make_residual_fn(
        N, Ec0, Ev0, Nc, Nv, ND, NA, Ed_x, Ea_x, pol_rho, eps_r, dx, T, kBT_eV,
        mu_n, mu_p, phi_left, phi_right, Efn_left, Efn_right, Efp_left, Efp_right,
        surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor, quantum_state,
    )

    state0 = np.concatenate([phi0, Efn0, Efp0])
    state0[0] = phi_left; state0[N - 1] = phi_right
    state0[N] = Efn_left; state0[2 * N - 1] = Efn_right
    state0[2 * N] = Efp_left; state0[3 * N - 1] = Efp_right

    n_iter_seen = [0]

    def _callback(x, f):
        n_iter_seen[0] += 1
        if log_fn is not None:
            log_fn(f"    coupled-NK iter {n_iter_seen[0]:3d}  "
                    f"|F|_max = {np.max(np.abs(f)):.3e}")
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")

    try:
        state = newton_krylov(
            residual_fn, state0, f_tol=tol, maxiter=maxiter,
            method='lgmres', line_search='armijo', callback=_callback,
        )
        converged = True
        final_residual = float(np.max(np.abs(residual_fn(state))))
    except NoConvergence as exc:
        state = np.asarray(exc.args[0]) if exc.args else state0
        converged = False
        final_residual = float(np.max(np.abs(residual_fn(state))))
    except SolveCancelled:
        raise
    except Exception as exc:  # pragma: no cover - defensive: never crash the outer solve
        if log_fn is not None:
            log_fn(f"    coupled-NK raised {type(exc).__name__}: {exc}; "
                   f"falling back to the pre-step state")
        state = state0
        converged = False
        final_residual = float(np.max(np.abs(residual_fn(state0))))

    phi = state[:N].copy()
    Efn = state[N:2 * N].copy()
    Efp = state[2 * N:3 * N].copy()
    phi[0], phi[-1] = phi_left, phi_right
    Efn[0], Efn[-1] = Efn_left, Efn_right
    Efp[0], Efp[-1] = Efp_left, Efp_right

    return CoupledResult(phi=phi, Efn=Efn, Efp=Efp, converged=converged,
                          n_iter=n_iter_seen[0], final_residual=final_residual)

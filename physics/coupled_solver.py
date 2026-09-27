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

This module solves all three equations together as one 3N-unknown
nonlinear system via a direct (non-Krylov) pseudo-transient-continuation
Newton solve (_solve_ptc_direct/_assemble_block_tridiag_jacobian): the
residual function below is assembled directly from the same,
already-validated building blocks used elsewhere (physics.fermi_dirac's
Fermi-Dirac densities, physics.drift_diffusion's Scharfetter-Gummel
Bernoulli-function flux stencil, physics.poisson's finite-volume Laplacian
assembly) -- re-derived here only as *residuals* of those exact
discretizations (matrix @ unknown - rhs), not re-derived physics.

An earlier version of this module tried Jacobian-Free Newton-Krylov
(scipy.optimize.newton_krylov) first, falling back to PTC-direct only on
failure. That was found to actively cause wrong answers, not just slower
ones: JFNK's Krylov-approximated Newton step, when seeded from a
carried-forward (continuation) state, would converge to a spurious nearby
fixed point -- a genuinely small *scaled* residual that masks an
unphysical quasi-Fermi split (confirmed: a direct one-shot solve at a
given V_applied recovers the correct textbook split, while reaching the
same V_applied via multi-step bias ramping converged "successfully" to a
wrong, masked root instead). PTC-direct is now used unconditionally --
see solve_coupled_dd and [[bug-quasi-fermi-pinning]].
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass

from scipy.sparse import coo_matrix as sp_coo_matrix, identity as sp_identity
from scipy.sparse.linalg import spsolve as spla_spsolve

from physics.constants import q as _q, eps0
from physics.grid_utils import node_spacings
from physics.poisson import _assemble_laplacian, solve_poisson_newton_ptc
from physics.fermi_dirac import (
    electron_density, hole_density, Efn_from_n, Efp_from_p,
    quantum_electron_density, quantum_hole_density,
)
from physics.drift_diffusion import (
    bernoulli, compute_recombination,
    solve_continuity_electron, solve_continuity_hole,
)


# nextnano++-style carrier density floor (see its currents{} keyword's
# minimum_density_electrons/minimum_density_holes) -- see the comment where
# this is applied in _make_residual_fn's residual() for why.
_MIN_CARRIER_DENSITY_CM3 = 1.0


class SolveCancelled(Exception):
    """Raised (from the PTC-direct loop, or the outer Gummel loop in
    physics.self_consistent) to unwind a solve early when the user clicks
    Stop -- distinct from a genuine solver failure."""


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
    quantum_gamma_n: Optional[np.ndarray] = None,
    quantum_gamma_p: Optional[np.ndarray] = None,
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
    # 2026-09-22: tried making n_flux_scale/p_flux_scale self-referential
    # (each carrier scaled by its OWN trial-state n/p instead of shared
    # max(ND,NA)) to stop the minority carrier's residual being masked by
    # the majority dopant's scale. Confirmed it does un-mask the pinning
    # (a real, if partial, Efn/Efp gradient appeared near the boundary
    # instead of a single-cell jump) -- but it also reproduced the exact
    # "broke JFNK globally" failure the ND-alone/NA-alone split already
    # ruled out (see below), just as widespread non-convergence/timeouts
    # (|F|_max stalling around the 80-iter cap) instead of an outright
    # "Jacobian inversion yielded zero vector" crash: a sweep that used to
    # complete now stalls around V~0.94 of a 2V target even with
    # bisection engaged. Root cause turned out to be upstream of this
    # scale at all -- see [[bug-quasi-fermi-pinning]] and
    # physics.self_consistent's Efn/Efp initial-guess smoothing -- so
    # reverted to the known-working shared scale here.
    # 2026-09-22, third attempt: freeze n_flux_scale/p_flux_scale from the
    # incoming (pre-Newton-solve) n/p -- physics.self_consistent's own
    # carrier-density estimate for this outer Gummel iteration, itself now
    # built from the smoothed Efn/Efp initial guess below -- instead of
    # ND/NA, updating it once per outer iteration rather than per Newton
    # trial step (unlike the second attempt above, this keeps the scale
    # state-independent *within* one JFNK solve, so it can't confuse the
    # finite-difference Jacobian approximation mid-solve). Confirmed this
    # does NOT fix it either: it made things worse than either previous
    # attempt (non-convergence partway through the ramp, and where it did
    # move, an unphysical split -- e.g. Efp momentarily *exceeding*
    # V_applied). Root cause: immediately after a new bias step, the
    # incoming n/p reference is itself still the *previous*, lower-bias
    # (near-equilibrium) estimate -- smallest and least representative
    # exactly on the iteration that most needs a good scale -- so freezing
    # against it just relocates the same masking-by-bad-reference problem
    # rather than removing it. Reverted to the shared ND/NA-based
    # carrier_scale below, which is the only scale formula that has held
    # up under a real bisected ramp without global JFNK breakage or
    # instability. See [[bug-quasi-fermi-pinning]] -- the physical fix
    # that actually resolved the *reported* symptom (flat Efn/Efp except a
    # single boundary cell, falsely reported as converged) ended up being
    # upstream of this scale entirely: physics.self_consistent's Efn/Efp
    # initial-guess smoothing plus honoring solve_coupled_dd's own
    # `converged` flag in the outer loop (previously ignored, letting a
    # failed inner solve's unchanged-state fallback read as "no change
    # needed" instead of "failed"). What remains open here is more subtle
    # than the original report: on the *most* extreme repro (1e19-1e20
    # doping beside 1e17-1e18), the now-honestly-converged solution still
    # comes out with a genuine but tiny (~1e-6 eV) Efn/Efp split instead of
    # a fully resolved one -- the masking mechanism this scale was meant
    # to fix is still real, just far less severe than before. A normal
    # (non-extreme) device already splits correctly (verified: a
    # symmetric 1e17/1e17 GaN p-n diode shows Efn flat near 0 through the
    # n-bulk, Efp settling at the correct injected value through the
    # p-bulk, both tapering toward the boundary condition near each
    # contact -- genuine textbook behavior, not degenerate tracking).
    # 2026-09-26: switched to a self-referential scale -- each carrier's
    # flux/Poisson residual normalized by its OWN current trial-state
    # density (n or p, floored), recomputed every residual evaluation --
    # replacing the static per-node max(ND,NA) scale above (see the long
    # comment history above this line for everything already tried: a
    # global scalar masks low-doped regions, a static per-node ND/NA-only
    # split collapses the minority carrier's scale to the floor, and this
    # exact self-referential idea was tried once before and "broke JFNK
    # globally" -- but that was diagnosed afterward as a downstream
    # symptom of two OTHER bugs fixed since then (the Efn/Efp initial-guess
    # single-cell discontinuity, and fermi_half's LUT-clamping silently
    # zeroing minority-carrier sensitivity), and JFNK itself has since been
    # replaced entirely by PTC-direct (a direct sparse Newton step per
    # pseudo-time increment, not GMRES's Krylov approximation) -- retried
    # now on that cleaner foundation.
    #
    # This is the same principle real TCAD/FEM semiconductor solvers use
    # (the Slotboom/logarithmic-variable reformulation's e^{-n̄}/e^{-p̄}
    # premultiplication of the continuity equations -- see the literature
    # review linked from [[bug-quasi-fermi-pinning]]): a carrier's own
    # residual is only ever compared against a reference *of that same
    # carrier's own local magnitude*, so it can never be masked by an
    # unrelated node's or an unrelated dopant's doping level the way a
    # static, position-independent-of-the-actual-carrier scale can.
    # Recomputed from the CURRENT trial state (not frozen from the
    # previous outer iteration, which was tried and also failed -- see
    # above): a scale that only updates once per outer Gummel iteration is
    # stale exactly when it matters most (right after a bias step, before
    poisson_scale = np.maximum(1.0, _q * np.maximum(np.maximum(ND, NA), 1e15)[1:-1] * 1e6)

    def residual(state: np.ndarray) -> np.ndarray:
        phi = state[:N]
        Efn = state[N:2 * N]
        Efp = state[2 * N:3 * N]
        Ec = Ec0 - phi
        Ev = Ev0 - phi

        # nextnano++-style density floor (its currents{} keyword exposes
        # this directly as minimum_density_electrons/holes): in a fully
        # depleted region the Boltzmann tail can underflow to ~1e-30 cm^-3
        # or below, and that literal near-zero value then propagates into
        # gamma_n/gamma_p's log() and gets divided into flux/scale terms
        # below, amplifying floating-point noise into spurious Jacobian
        # entries. 1 cm^-3 is physically negligible next to any doping
        # level or transport-relevant density in this device (>=1e15), so
        # this changes nothing about the converged physics -- it only
        # keeps the algebra numerically sane in already-negligible regions.
        #
        # Quantum correction (quantum_gamma_n/p = n_quantum/n_classical,
        # frozen for this solve by physics.self_consistent from the current
        # Schrodinger solution -- the standard predictor-corrector
        # quantum-corrected drift-diffusion scheme): ONE density per carrier,
        # used identically in Poisson, continuity and recombination. An
        # earlier version fed the quantum density to Poisson only and kept
        # continuity classical, which made the two blocks solve for two
        # different electron populations in the well -- the root cause of
        # [[bug-quantum-bias-electron-divergence]]. Using gamma*n_cl in the
        # SG flux is safe: psi_n_eff below picks up kT*ln(gamma) (the
        # quantum potential) via gamma_n = n/n_boltz, and the SG flux of a
        # flat-Efn state is exactly zero for ANY gamma profile (including
        # the step at the quantum-region edge), so no spurious driving
        # field is introduced.
        n_cl = electron_density(Ec, Efn, Nc, T)
        p_cl = hole_density(Ev, Efp, Nv, T)
        if quantum_gamma_n is not None:
            n_cl = n_cl * quantum_gamma_n
        if quantum_gamma_p is not None:
            p_cl = p_cl * quantum_gamma_p
        n = np.maximum(n_cl, _MIN_CARRIER_DENSITY_CM3)
        p = np.maximum(p_cl, _MIN_CARRIER_DENSITY_CM3)
        n_pois = n
        p_pois = p

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

        rho = _q * (p_pois * 1e6 - n_pois * 1e6 + Nd_plus * 1e6 - Na_minus * 1e6) + pol_rho
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
        # Self-referential scale: this carrier's own current trial-state
        # density, not a fixed doping-based reference -- see the comment
        # above n_of/p_of's caller (where poisson_scale is built) for why.
        #
        # 2026-09-27: floor lowered from 1e10 to _MIN_CARRIER_DENSITY_CM3
        # (1.0) after finding this masked minority-carrier injection
        # entirely, on EVERY device tested (not just hard MQW ones) --
        # confirmed on a plain 100nm/100nm 1e18/1e18 GaN p-n diode at 1V
        # forward bias: minority hole density at the junction edge should
        # increase by ~exp(qV/kT) ~ 1e17x (law of the junction) but came
        # out essentially unchanged (ratio ~0.999) because wherever the
        # true n dropped below the old 1e10 floor -- which is most of the
        # opposite-doping-type region, i.e. exactly where injection
        # physics lives -- the flux_scale saturated at that floor
        # regardless of n's real (much smaller) value. That made a raw,
        # wildly-unsatisfied continuity residual (e.g. +1.25e5 at the
        # junction) normalize down to ~1e-26 and read as "converged" well
        # inside tol=1e-3, while the true minority-carrier profile was
        # frozen at its equilibrium shape. Exactly the same masking
        # mechanism as [[bug-quasi-fermi-pinning]] (a scale disconnected
        # from the true local magnitude hides a real error), just
        # resurfacing via a hardcoded floor instead of a doping-based
        # scale. See project memory bug_efn_efp_not_splitting.md.
        n_flux_scale = np.maximum(1.0, coeff_r_n * np.maximum(n[1:-1], _MIN_CARRIER_DENSITY_CM3) * 1e6)
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
        # 2026-09-27: sign fixed -- hole continuity is dJ_p/dx = -q*R
        # (opposite sign from electron continuity's dJ_n/dx = +q*R; see
        # physics.drift_diffusion.solve_continuity_hole's matching fix and
        # its comment for the full derivation/rationale). Jdiv_p here is
        # built with the identical algebraic structure as Jdiv_n just
        # above (same Bernoulli-identity stencil, n->p/mu_n->mu_p/
        # psi_n_eff->psi_p_eff), so it's the same "flux divergence"
        # quantity and needs the same sign flip relative to electrons'
        # F2_interior = Jdiv_n - R.
        F3_interior = Jdiv_p + R[1:-1]
        # See n_flux_scale's comment above -- same 2026-09-27 floor fix.
        p_flux_scale = np.maximum(1.0, coeff_r_p * np.maximum(p[1:-1], _MIN_CARRIER_DENSITY_CM3) * 1e6)
        F3 = np.empty(N)
        F3[1:-1] = F3_interior / p_flux_scale
        F3[0] = Efp[0] - Efp_left
        F3[-1] = Efp[-1] - Efp_right

        return np.concatenate([F1, F2, F3])

    return residual


def _ionized_dopants(
    Efn: np.ndarray, Efp: np.ndarray, Ec: np.ndarray, Ev: np.ndarray,
    ND: np.ndarray, NA: np.ndarray, Ed_x: np.ndarray, Ea_x: np.ndarray,
    kBT_eV: float,
    surf_idx: np.ndarray, surf_density_cm3: np.ndarray,
    surf_energy_eV: np.ndarray, surf_is_donor: np.ndarray,
):
    """
    Ionized donor/acceptor density [cm^-3] and its derivative w.r.t. phi,
    given the current quasi-Fermi levels -- the same incomplete-ionization
    formula physics.self_consistent's equilibrium branch and this module's
    _make_residual_fn already use (Si donor / Mg acceptor activation
    energies via Ed_x/Ea_x), factored out here so _solve_gummel_adaptive
    below can call it once per outer iteration instead of duplicating the
    algebra. dphi sign convention: Ec = Ec0 - phi, so d(exp_arg_d)/dphi =
    +1/kBT_eV and d(exp_arg_a)/dphi = -1/kBT_eV, exactly matching
    physics.self_consistent's own dNd_dphi/dNa_dphi.
    """
    exp_arg_d = np.clip((Efn - (Ec - Ed_x)) / kBT_eV, -340.0, 340.0)
    exp_arg_a = np.clip((Ev + Ea_x - Efp) / kBT_eV, -340.0, 340.0)
    exp_d = np.exp(exp_arg_d)
    exp_a = np.exp(exp_arg_a)
    Nd_plus = ND / (1.0 + 2.0 * exp_d)
    Na_minus = NA / (1.0 + 4.0 * exp_a)
    dNd_dphi = -ND * (2.0 * exp_d) / (1.0 + 2.0 * exp_d) ** 2 / kBT_eV
    dNa_dphi = NA * (4.0 * exp_a) / (1.0 + 4.0 * exp_a) ** 2 / kBT_eV

    if len(surf_idx) > 0:
        donor_mask = surf_is_donor
        if np.any(donor_mask):
            idx_d = surf_idx[donor_mask]
            exp_arg_sd = np.clip(
                (Efn[idx_d] - (Ec[idx_d] - surf_energy_eV[donor_mask])) / kBT_eV,
                -340.0, 340.0)
            exp_sd = np.exp(exp_arg_sd)
            np.add.at(Nd_plus, idx_d, surf_density_cm3[donor_mask] / (1.0 + 2.0 * exp_sd))
            np.add.at(dNd_dphi, idx_d,
                      -surf_density_cm3[donor_mask] * (2.0 * exp_sd) / (1.0 + 2.0 * exp_sd) ** 2 / kBT_eV)
        acceptor_mask = ~surf_is_donor
        if np.any(acceptor_mask):
            idx_a = surf_idx[acceptor_mask]
            exp_arg_sa = np.clip(
                (Ev[idx_a] + surf_energy_eV[acceptor_mask] - Efp[idx_a]) / kBT_eV,
                -340.0, 340.0)
            exp_sa = np.exp(exp_arg_sa)
            np.add.at(Na_minus, idx_a, surf_density_cm3[acceptor_mask] / (1.0 + 4.0 * exp_sa))
            np.add.at(dNa_dphi, idx_a,
                      surf_density_cm3[acceptor_mask] * (4.0 * exp_sa) / (1.0 + 4.0 * exp_sa) ** 2 / kBT_eV)

    return Nd_plus, Na_minus, dNd_dphi, dNa_dphi


# nextnano++-style adaptive Gummel relaxation (see this session's nextnano++
# documentation research): alpha shrinks whenever the outer residual grows
# (the auto-shrink-on-oscillation behaviour of nextnano's own alpha_fermi/
# alpha_iterations) and grows slowly on sustained improvement, so it never
# needs per-device hand-tuning.
_GUMMEL_ALPHA_INIT = 0.3
_GUMMEL_ALPHA_MIN = 0.01
_GUMMEL_ALPHA_MAX = 1.0
_GUMMEL_ALPHA_GROWTH = 1.1
_GUMMEL_ALPHA_SHRINK = 0.5
# Gummel iterations are cheap (two linear solves + one linearised Poisson
# solve, no Jacobian assembly) compared to the monolithic solver's, so this
# is deliberately generous and independent of solve_coupled_dd's own
# (much more expensive-per-iteration) maxiter parameter.
_GUMMEL_MAXITER = 200


def _solve_gummel_adaptive(
    phi0: np.ndarray, Efn0: np.ndarray, Efp0: np.ndarray,
    Ec0: np.ndarray, Ev0: np.ndarray, Nc: np.ndarray, Nv: np.ndarray,
    ND: np.ndarray, NA: np.ndarray, Ed_x: np.ndarray, Ea_x: np.ndarray,
    pol_rho: np.ndarray, eps_r: np.ndarray, dx, T: float, kBT_eV: float,
    mu_n: float, mu_p: float,
    phi_left: float, phi_right: float,
    Efn_left: float, Efn_right: float, Efp_left: float, Efp_right: float,
    surf_idx: np.ndarray, surf_density_cm3: np.ndarray,
    surf_energy_eV: np.ndarray, surf_is_donor: np.ndarray,
    tol: float, maxiter: int,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    quantum_gamma_n: Optional[np.ndarray] = None,
    quantum_gamma_p: Optional[np.ndarray] = None,
) -> CoupledResult:
    """
    nextnano++-style decoupled Gummel map: solve the (linear, given phi)
    electron and hole continuity equations, then the (nonlinear, given n/p)
    Poisson equation, and relax between the two with an adaptive damping
    factor that auto-shrinks on oscillation -- instead of physics.
    coupled_solver's monolithic 3N-unknown Newton solve (phi, Efn, Efp
    together via a finite-difference Jacobian + sparse LU every single
    pseudo-time step). Each outer iteration here costs two linear solves
    (continuity) plus one linearised Poisson solve, all with matrices
    assembled analytically -- no finite-difference Jacobian, no repeated
    residual evaluations -- which is the actual source of nextnano++'s
    speed advantage on devices this solver used to take 3000+ seconds on
    (see this session's nextnano++ documentation research).

    This function's result is NEVER trusted on its own: solve_coupled_dd
    (its only caller) always re-checks a "converged" Gummel result against
    the true fully-coupled nonlinear residual (_make_residual_fn) before
    accepting it, and falls back to the monolithic PTC-direct solve
    otherwise. That gate exists because an earlier, simpler Gummel
    implementation this session (since removed) was found to oscillate
    without ever properly diverging -- i.e. a plausible-looking but WRONG
    quasi-Fermi split, the same "small residual masks a wrong answer"
    failure mode [[bug-quasi-fermi-pinning]] was about, just via sequential
    coupling instead of continuation. Whether adaptive relaxation alone
    fully fixes that is unproven; the residual cross-check is what makes
    this safe to ship regardless.
    """
    N = len(phi0)
    phi = phi0.copy()
    phi[0], phi[-1] = phi_left, phi_right

    # Quantum correction factors (see _make_residual_fn): n = gamma_n *
    # n_classical everywhere, and the SG driving potential gains the
    # quantum potential kT*ln(gamma) so a flat-Efn state stays current-free.
    gamma_n = np.ones(N) if quantum_gamma_n is None else np.asarray(quantum_gamma_n, dtype=float)
    gamma_p = np.ones(N) if quantum_gamma_p is None else np.asarray(quantum_gamma_p, dtype=float)
    qpot_n = kBT_eV * np.log(gamma_n)
    qpot_p = kBT_eV * np.log(gamma_p)

    Ec = Ec0 - phi
    Ev = Ev0 - phi
    n = np.maximum(gamma_n * electron_density(Ec, Efn0, Nc, T), _MIN_CARRIER_DENSITY_CM3)
    p = np.maximum(gamma_p * hole_density(Ev, Efp0, Nv, T), _MIN_CARRIER_DENSITY_CM3)

    n_left = float(gamma_n[0] * electron_density(Ec0[0] - phi_left, Efn_left, Nc[0], T))
    n_right = float(gamma_n[-1] * electron_density(Ec0[-1] - phi_right, Efn_right, Nc[-1], T))
    p_left = float(gamma_p[0] * hole_density(Ev0[0] - phi_left, Efp_left, Nv[0], T))
    p_right = float(gamma_p[-1] * hole_density(Ev0[-1] - phi_right, Efp_right, Nv[-1], T))

    alpha = _GUMMEL_ALPHA_INIT
    dt_ptc = 1e-1
    best_residual = np.inf
    residual = np.inf
    n_iter = 0

    for n_iter in range(1, maxiter + 1):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")

        Ec = Ec0 - phi
        Ev = Ev0 - phi
        ni = np.sqrt(Nc * Nv) * np.exp(-(Ec - Ev) / (2.0 * kBT_eV))
        R = compute_recombination(n, p, ni)

        # Boltzmann-form driving potentials for the SG flux (no explicit
        # degeneracy/gamma correction -- see physics.coupled_solver's own
        # residual() docstring: that correction was investigated as a
        # suspect for the quasi-Fermi masking bug and ruled out, so
        # skipping it here for speed loses no accuracy this codebase
        # doesn't already forgo elsewhere at this level of the model).
        psi_n_eff = -Ec + qpot_n
        psi_p_eff = Ev + qpot_p

        n_trial = solve_continuity_electron(n, psi_n_eff, R, dx, mu_n, T, n_left, n_right)
        p_trial = solve_continuity_hole(p, psi_p_eff, R, dx, mu_p, T, p_left, p_right)
        n_trial = np.maximum(n_trial, _MIN_CARRIER_DENSITY_CM3)
        p_trial = np.maximum(p_trial, _MIN_CARRIER_DENSITY_CM3)

        Efn_trial = Efn_from_n(n_trial / gamma_n, Ec, Nc, T)
        Efp_trial = Efp_from_p(p_trial / gamma_p, Ev, Nv, T)
        Efn_trial[0], Efn_trial[-1] = Efn_left, Efn_right
        Efp_trial[0], Efp_trial[-1] = Efp_left, Efp_right

        Nd_plus, Na_minus, dNd_dphi, dNa_dphi = _ionized_dopants(
            Efn_trial, Efp_trial, Ec, Ev, ND, NA, Ed_x, Ea_x, kBT_eV,
            surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
        )
        phi_trial = solve_poisson_newton_ptc(
            phi, eps_r, n_trial, p_trial, Nd_plus, Na_minus, dNd_dphi, dNa_dphi,
            pol_rho, dx, phi_left, phi_right, T, dt_ptc,
        )

        if not (np.all(np.isfinite(phi_trial)) and np.all(np.isfinite(n_trial))
                and np.all(np.isfinite(p_trial))):
            # A singular/near-singular linear solve (seen on the hardest
            # stress devices, e.g. a fully-depleted region driving Bpos/Bneg
            # to under/overflow) can silently hand back NaN/Inf rather than
            # raising. NaN comparisons are always False, so without this
            # check the loop would never trip its own "diverging" branch and
            # would instead burn its entire iteration budget doing nothing
            # useful before reporting not-converged anyway -- exit now
            # instead, same fail-fast principle as the PTC dt floor above.
            if log_fn is not None:
                log_fn(f"    Gummel-adaptive iter {n_iter:3d}: non-finite trial state "
                       f"-- diverging, failing fast instead of exhausting maxiter")
            break

        residual = float(np.max(np.abs(phi_trial - phi)))
        if residual > best_residual * 1.01:
            alpha = max(alpha * _GUMMEL_ALPHA_SHRINK, _GUMMEL_ALPHA_MIN)
        else:
            alpha = min(alpha * _GUMMEL_ALPHA_GROWTH, _GUMMEL_ALPHA_MAX)
        best_residual = min(best_residual, residual)

        phi = phi + alpha * (phi_trial - phi)
        phi[0], phi[-1] = phi_left, phi_right
        n = np.maximum(n + alpha * (n_trial - n), _MIN_CARRIER_DENSITY_CM3)
        p = np.maximum(p + alpha * (p_trial - p), _MIN_CARRIER_DENSITY_CM3)

        if log_fn is not None:
            log_fn(f"    Gummel-adaptive iter {n_iter:3d}  alpha={alpha:.3e}  "
                   f"|dphi|_max = {residual:.3e} V")

        if residual < tol:
            Efn = Efn_from_n(n / gamma_n, Ec0 - phi, Nc, T)
            Efp = Efp_from_p(p / gamma_p, Ev0 - phi, Nv, T)
            Efn[0], Efn[-1] = Efn_left, Efn_right
            Efp[0], Efp[-1] = Efp_left, Efp_right
            return CoupledResult(phi=phi, Efn=Efn, Efp=Efp, converged=True,
                                  n_iter=n_iter, final_residual=residual)

    Efn = Efn_from_n(n / gamma_n, Ec0 - phi, Nc, T)
    Efp = Efp_from_p(p / gamma_p, Ev0 - phi, Nv, T)
    Efn[0], Efn[-1] = Efn_left, Efn_right
    Efp[0], Efp[-1] = Efp_left, Efp_right
    return CoupledResult(phi=phi, Efn=Efn, Efp=Efp, converged=False,
                          n_iter=n_iter, final_residual=residual)


def _assemble_block_tridiag_jacobian(residual_fn, state: np.ndarray, N: int,
                                      h_rel: float = 1e-6):
    """
    Sparse Jacobian of `residual_fn` at `state`, exploiting that this is a
    1D problem: residual row i (any of the 3 equation blocks) depends only
    on unknowns at grid indices i-1, i, i+1 (any of the 3 channels
    phi/Efn/Efp) -- a block-tridiagonal structure with 3x3 blocks, not a
    dense 3N x 3N matrix.

    Assembled via grouped ("colored") finite differences instead of one
    column at a time: for a fixed channel, grid points spaced 3 apart
    can't have overlapping row-effects (point i affects only rows
    [i-1,i+1], point i+3 only [i+2,i+4]), so perturbing every third grid
    point *simultaneously* and reading off which rows moved recovers all
    of those columns from a single residual evaluation. 3 channels x 3
    colors = 9 residual evaluations total to assemble the *entire* sparse
    Jacobian, regardless of N.

    First attempted 2026-09-22 and abandoned: the assembled Jacobian came
    out exactly singular (zero-diagonal rows clustered at the hole
    channel in a heavily n-doped layer). Root-caused to physics.
    fermi_dirac.fermi_half's LUT clamping silently zeroing carrier-density
    sensitivity outside eta in [-25,60] -- now fixed there, so this is
    being retried. See solve_coupled_dd's module-level rationale for why
    a *direct* solve (this or a fully analytic Jacobian) is worth having
    alongside JFNK: it gives the exact Newton step every time instead of
    GMRES's Krylov-subspace approximation, which is what plateaus on
    stiff/ill-conditioned steps regardless of how much Krylov capacity is
    given (confirmed empirically raising inner_maxiter/outer_k on the
    JFNK call -- see there).
    """
    F0 = residual_fn(state)
    rows: list = []
    cols: list = []
    vals: list = []
    for c in range(3):
        base = c * N
        for color in range(3):
            idx = np.arange(color, N, 3)
            if idx.size == 0:
                continue
            h = h_rel * np.maximum(np.abs(state[base + idx]), 1.0)
            pert = state.copy()
            pert[base + idx] += h
            dF = residual_fn(pert) - F0
            for k in range(idx.size):
                i = idx[k]
                hk = h[k]
                for row_block in range(3):
                    for di in (-1, 0, 1):
                        r = i + di
                        if 0 <= r < N:
                            row = row_block * N + r
                            d = dF[row] / hk
                            if d != 0.0:
                                rows.append(row)
                                cols.append(base + i)
                                vals.append(d)
    J = sp_coo_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N)).tocsc()
    return F0, J



# Below this pseudo-time step, a PTC step is contributing a numerically
# negligible correction -- continuing to reassemble the (expensive: 9
# residual evaluations) Jacobian and factor a fresh sparse LU every
# remaining iteration just to keep rejecting is wasted work, not honest
# extra effort toward convergence. This is the direct analogue of
# nextnano++'s own documented advice ("there is no reason to wait for
# simulation to finish, if you already know that it will not converge
# enough") -- see the nextnano++ convergence-tutorial research this
# session did. Chosen four decades below dt_init's floor value (1e-2) --
# far enough that a step this size can never represent a real recovering
# trend, only a stalled one.
_PTC_DT_MIN = 1e-6


def _solve_ptc_direct(
    residual_fn, state0: np.ndarray, N: int,
    tol: float, maxiter: int, dt_init: float, dt_growth: float,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    dt_min: float = _PTC_DT_MIN,
) -> CoupledResult:
    """
    Pseudo-transient continuation (PTC) using a *direct* sparse Newton
    step at each pseudo-time increment, instead of nesting a full JFNK
    solve inside every PTC step (see _solve_ptc, the first version of
    this, which worked but was slow -- 30-60 outer steps each running an
    inner Krylov solve). The standard, efficient PTC scheme needs only
    ONE linear solve per step: solve (J(x) + I/dt)*delta = -F(x) directly
    (implicit-Euler damping folded straight into the Jacobian's diagonal,
    not into the residual the way _solve_ptc does it), accept/reject
    based on whether the residual actually improved, and grow dt (SER,
    Kelley & Keyes 1998) as it does. This is the nextnano++-style
    approach: a direct solve gives the exact (up to the FD-approximated
    Jacobian) Newton step every time, so far fewer pseudo-time steps are
    needed overall than the nested-JFNK version, and each one is cheap
    (one Jacobian assembly + one sparse LU, no inner Krylov iteration
    loop) -- see [[bug-quasi-fermi-pinning]].
    """
    x = state0.copy()
    dt = dt_init
    F = residual_fn(x)
    res_norm = float(np.max(np.abs(F))) if np.all(np.isfinite(F)) else np.inf
    n_iter = 0

    if log_fn is not None:
        log_fn(f"    PTC-direct iter {n_iter:3d}  dt={dt:.3e}  |F|_max = {res_norm:.3e}")

    if res_norm < tol:
        return CoupledResult(phi=x[:N].copy(), Efn=x[N:2 * N].copy(), Efp=x[2 * N:3 * N].copy(),
                              converged=True, n_iter=0, final_residual=res_norm)

    identity_3n = sp_identity(3 * N, format='csc')

    for n_iter in range(1, maxiter + 1):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")

        if dt < dt_min:
            if log_fn is not None:
                log_fn(f"    PTC-direct iter {n_iter:3d}: dt={dt:.3e} below floor "
                       f"{dt_min:.3e} -- stagnant, failing fast instead of "
                       f"burning the remaining {maxiter - n_iter + 1} iterations")
            break

        F, J = _assemble_block_tridiag_jacobian(residual_fn, x, N)
        J_ptc = J + identity_3n / dt
        try:
            delta = spla_spsolve(J_ptc, -F)
        except Exception as exc:
            dt *= 0.3
            if log_fn is not None:
                log_fn(f"    PTC-direct iter {n_iter:3d}: spsolve raised "
                       f"{type(exc).__name__}, shrinking dt to {dt:.3e}")
            continue
        if not np.all(np.isfinite(delta)):
            dt *= 0.3
            if log_fn is not None:
                log_fn(f"    PTC-direct iter {n_iter:3d}: non-finite step, shrinking dt to {dt:.3e}")
            continue

        x_trial = x + delta
        F_trial = residual_fn(x_trial)
        res_trial = float(np.max(np.abs(F_trial))) if np.all(np.isfinite(F_trial)) else np.inf

        if res_trial > res_norm * 2.0:
            dt *= 0.3
            if log_fn is not None:
                log_fn(f"    PTC-direct iter {n_iter:3d}: rejected (|F| {res_trial:.3e} > "
                       f"2x{res_norm:.3e}), shrinking dt to {dt:.3e}")
            continue

        x = x_trial
        growth = min(res_norm / max(res_trial, 1e-300), dt_growth)
        dt *= max(growth, 1.0)
        res_norm = res_trial

        if log_fn is not None:
            # Recompute F_trial blocks for logging
            F_trial = residual_fn(x)
            F1_max = float(np.max(np.abs(F_trial[:N])))
            F2_max = float(np.max(np.abs(F_trial[N:2*N])))
            F3_max = float(np.max(np.abs(F_trial[2*N:])))
            log_fn(f"    PTC-direct iter {n_iter:3d}  dt={dt:.3e}  |F|_max = {res_norm:.3e}  (F1={F1_max:.1e}, F2={F2_max:.1e}, F3={F3_max:.1e})")

        if res_norm < tol:
            return CoupledResult(phi=x[:N].copy(), Efn=x[N:2 * N].copy(), Efp=x[2 * N:3 * N].copy(),
                                  converged=True, n_iter=n_iter, final_residual=res_norm)

    return CoupledResult(phi=x[:N].copy(), Efn=x[N:2 * N].copy(), Efp=x[2 * N:3 * N].copy(),
                          converged=False, n_iter=n_iter, final_residual=res_norm)


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
    quantum_gamma_n: Optional[np.ndarray] = None,
    quantum_gamma_p: Optional[np.ndarray] = None,
    tol: float = 1e-3,
    maxiter: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CoupledResult:
    """
    Jointly solve Poisson + electron continuity + hole continuity for a
    fixed set of Dirichlet boundary conditions, via a direct (non-Krylov)
    pseudo-transient-continuation Newton solve (see _solve_ptc_direct).
    Returns the converged (or best-effort) state.

    PTC-direct only, no JFNK attempt: JFNK's Krylov-approximated Newton
    step was found to converge to spurious nearby fixed points when seeded
    from a carried-forward (continuation) state -- a genuinely small
    scaled residual that masks an unphysical quasi-Fermi split (see
    [[bug-quasi-fermi-pinning]] and the ramped-bias investigation that
    found it: a direct one-shot solve at a given V_applied recovers the
    correct textbook split, while reaching the same V_applied via a
    multi-step continuation converged to a wrong, masked root instead).
    Every solve is now a single direct shot at its target V_applied (see
    AlGaNDevice.solve_ramped/sweep_voltage -- both call solve() directly
    per point, no continuation), and PTC-direct's own pseudo-time
    continuation (small, heavily-damped steps growing via SER) is what
    handles convergence robustness at high bias instead.

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
        surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
        quantum_gamma_n, quantum_gamma_p,
    )

    # Cheap already-converged check: the outer Gummel loop in physics.
    # self_consistent calls this function again every outer iteration,
    # re-passing whatever (phi, Efn, Efp) the *previous* iteration already
    # converged to. Without this check, that incoming already-good state
    # still had to survive a full, expensive Gummel attempt (up to
    # _GUMMEL_MAXITER=200 iterations) before falling through to the
    # monolithic solver's own cheap initial-residual check -- confirmed
    # empirically: a state that had already converged burned 200 wasted
    # Gummel iterations on the very next outer call purely because Gummel
    # doesn't know to check this first. One residual_fn evaluation here
    # (cheap: no linear solve) avoids that entirely.
    state0_check = np.concatenate([phi0, Efn0, Efp0])
    F0 = residual_fn(state0_check)
    if np.all(np.isfinite(F0)):
        res0 = float(np.max(np.abs(F0)))
        if res0 < tol:
            if log_fn is not None:
                log_fn(f"    Already converged (|F|_max={res0:.3e} < tol={tol:.3e}) "
                       f"-- skipping both Gummel and monolithic solves.")
            phi = phi0.copy(); Efn = Efn0.copy(); Efp = Efp0.copy()
            phi[0], phi[-1] = phi_left, phi_right
            Efn[0], Efn[-1] = Efn_left, Efn_right
            Efp[0], Efp[-1] = Efp_left, Efp_right
            return CoupledResult(phi=phi, Efn=Efn, Efp=Efp, converged=True,
                                  n_iter=0, final_residual=res0)

    # Fast path: nextnano++-style decoupled Gummel with adaptive relaxation
    # (see _solve_gummel_adaptive) -- roughly 10-100x cheaper per iteration
    # than the monolithic PTC-direct solve below (analytic-matrix linear
    # solves instead of a finite-difference Jacobian assembly + sparse LU
    # every pseudo-time step). Handles the quantum correction factors
    # (quantum_gamma_n/p) the same way the residual does.
    # A Gummel "converged" result is NEVER trusted on its own -- it's
    # always re-checked against the exact same fully-coupled residual_fn
    # the monolithic solver itself converges against, and only accepted if
    # it independently passes that check. See _solve_gummel_adaptive's
    # docstring for why this gate is non-negotiable.
    gummel_result = _solve_gummel_adaptive(
        phi0, Efn0, Efp0, Ec0, Ev0, Nc, Nv, ND, NA, Ed_x, Ea_x,
        pol_rho, eps_r, dx, T, kBT_eV, mu_n, mu_p,
        phi_left, phi_right, Efn_left, Efn_right, Efp_left, Efp_right,
        surf_idx, surf_density_cm3, surf_energy_eV, surf_is_donor,
        tol=tol, maxiter=_GUMMEL_MAXITER,
        log_fn=log_fn, cancel_check=cancel_check,
        quantum_gamma_n=quantum_gamma_n, quantum_gamma_p=quantum_gamma_p,
    )
    if gummel_result.converged:
        state = np.concatenate([gummel_result.phi, gummel_result.Efn, gummel_result.Efp])
        F = residual_fn(state)
        verify_residual = float(np.max(np.abs(F))) if np.all(np.isfinite(F)) else np.inf
        if verify_residual < tol:
            if log_fn is not None:
                log_fn(f"    Gummel-adaptive converged in {gummel_result.n_iter} iters "
                       f"and passed the fully-coupled residual check "
                       f"(|F|_max={verify_residual:.3e} < tol={tol:.3e}) -- accepted, "
                       f"skipping the monolithic PTC-direct solve.")
            phi = gummel_result.phi.copy()
            Efn = gummel_result.Efn.copy()
            Efp = gummel_result.Efp.copy()
            phi[0], phi[-1] = phi_left, phi_right
            Efn[0], Efn[-1] = Efn_left, Efn_right
            Efp[0], Efp[-1] = Efp_left, Efp_right
            return CoupledResult(phi=phi, Efn=Efn, Efp=Efp, converged=True,
                                  n_iter=gummel_result.n_iter, final_residual=verify_residual)
        elif log_fn is not None:
            log_fn(f"    Gummel-adaptive reported converged in {gummel_result.n_iter} "
                   f"iters but FAILED the fully-coupled residual check "
                   f"(|F|_max={verify_residual:.3e} >= tol={tol:.3e}) -- falling back "
                   f"to the monolithic PTC-direct solver.")
    elif log_fn is not None:
        log_fn(f"    Gummel-adaptive did not converge in {gummel_result.n_iter} iters "
               f"(|dphi|={gummel_result.final_residual:.3e}) -- falling back to the "
               f"monolithic PTC-direct solver.")

    state0 = np.concatenate([phi0, Efn0, Efp0])
    state0[0] = phi_left; state0[N - 1] = phi_right
    state0[N] = Efn_left; state0[2 * N - 1] = Efn_right
    state0[2 * N] = Efp_left; state0[3 * N - 1] = Efp_right

    ptc_result = _solve_ptc_direct(
        residual_fn, state0, N, tol=tol, maxiter=maxiter,
        dt_init=1e-2, dt_growth=3.0,
        log_fn=log_fn, cancel_check=cancel_check,
    )
    state = np.concatenate([ptc_result.phi, ptc_result.Efn, ptc_result.Efp])
    converged = ptc_result.converged
    final_residual = ptc_result.final_residual
    n_iter_seen = [ptc_result.n_iter]

    phi = state[:N].copy()
    Efn = state[N:2 * N].copy()
    Efp = state[2 * N:3 * N].copy()
    phi[0], phi[-1] = phi_left, phi_right
    Efn[0], Efn[-1] = Efn_left, Efn_right
    Efp[0], Efp[-1] = Efp_left, Efp_right

    return CoupledResult(phi=phi, Efn=Efn, Efp=Efp, converged=converged,
                          n_iter=n_iter_seen[0], final_residual=final_residual)

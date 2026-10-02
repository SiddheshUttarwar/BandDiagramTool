"""
Fully-coupled drift-diffusion Newton solver in (phi, Efn, Efp) with an
analytic Jacobian -- the biased-solve engine behind physics.self_consistent.

Why this replaced physics.coupled_solver's PTC-direct/Gummel pair
(see [[bug-current-not-conserved]]): those converged against a residual
whose continuity block was normalized by a scale 1e6 too large, so a
"converged" solution only satisfied current continuity to ~1000x the local
diffusive flux -- J_n + J_p varied by tens of orders of magnitude across the
device. With the scale corrected they could not converge at all (finite-
difference Jacobian across a 1e-50..1e20 cm^-3 dynamic range).

Design:
  * Unknowns phi [V], Efn, Efp [eV] at every node, interleaved per node
    (block-tridiagonal Jacobian, direct sparse LU).
  * Carrier densities n = gamma_n * Nc * F_1/2((Efn-Ec)/kT), same for p
    (gamma = frozen quantum correction factor from physics.self_consistent,
    1 when classical). Handled in log space so densities far below 1e-300
    never underflow into singular rows.
  * Generalized Scharfetter-Gummel flux in quasi-Fermi form. For an edge
    a -> c with L = ln(n_c/n_a), b = (Efn_c - Efn_a)/kT, d = L - b:
        G/C = B(d) n_c (1 - e^-b) = B(-d) n_a (e^b - 1),   C = D/h
    identical to standard SG in the Boltzmann limit (d is the usual
    normalized potential step), exactly zero for flat Efn with ANY
    Fermi-Dirac / quantum-gamma density profile, and -> mu n dEfn/dx for
    small steps. The second form is used when d > 0 so neither form's
    exponentials overflow/underflow. J_n = q G. Holes are the mirror image
    (ln p, -Efp).
  * Equations (cm units for continuity, SI for Poisson):
        eps0 * A_pos phi - rho = 0
        (G_{i+1/2} - G_{i-1/2}) / cw_i - R_i = 0     (dJn/dx =  qR)
        (H_{i+1/2} - H_{i-1/2}) / cw_i + R_i = 0     (dJp/dx = -qR)
    with ohmic (Dirichlet) contacts.
  * Convergence is judged on the Newton UPDATE (|dphi|, |dEfn|, |dEfp|
    below tol_update, in volts) -- a scale-free criterion that cannot be
    masked by residual normalization. Rows are equilibrated only for the
    linear solve's pivoting.
  * Globalization: per-step update limiting, plus adaptive bias
    continuation from equilibrium (solve_bias_ramp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.sparse import coo_matrix, diags as sp_diags
from scipy.sparse.linalg import spsolve

from physics.constants import q as _q, eps0, kB
from physics.grid_utils import node_spacings
from physics.poisson import _assemble_laplacian
from physics.fermi_dirac import fermi_half, dfermi_half
from physics.drift_diffusion import bernoulli, dbernoulli

# Recombination coefficients: same standard III-Nitride values as
# physics.drift_diffusion.compute_recombination_components.
_TAU_N = 1e-9
_TAU_P = 1e-9
_B_RAD = 1e-11
_C_AUG = 1e-30

# eta is clipped here before building densities: Ec - Efn > ~17 eV only
# happens in absurd intermediate Newton states, and keeps ln(n) finite.
_ETA_MIN = -650.0

# Newton acceptance (see newton_solve): the primary test is the undamped
# update < tol_update; the secondary test accepts a row-equilibrated
# residual at roundoff level (volts) as long as the update is only
# conditioning noise.
_RES_VOLTS_TOL = 1e-10
# Bias-ramp leaps over a steep/folded I-V stretch (see solve_bias_ramp).
_LEAP_STEPS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
_LEAP_MAXITER = 80
_LEAP_BACKOFF_MIN = 0.015   # start states at least this far behind the fold [V]
_LEAP_N_STARTS = 3
_RAMP_RETRY_PATTERNS = ((0.25, 1.0), (0.4, 1.0), (0.15, 0.6))
_NOISE_UPDATE_MAX = 1e-4


from physics.coupled_solver import SolveCancelled  # same class the GUI worker catches


_LNF_SPLINE = None
_ETA_SWITCH = -5.0


def _lnF_spline():
    """C2 cubic spline of ln F_1/2 through physics.fermi_dirac's exact
    quadrature table values for eta >= -5 (built once)."""
    global _LNF_SPLINE
    if _LNF_SPLINE is None:
        from scipy.interpolate import CubicSpline
        import physics.fermi_dirac as fd
        fd._build_lut()
        m = fd._LUT_ETA >= _ETA_SWITCH
        _LNF_SPLINE = CubicSpline(fd._LUT_ETA[m], np.log(fd._LUT_F[m]))
    return _LNF_SPLINE


def _ln_fermi_half(eta: np.ndarray):
    """
    ln F_1/2(eta) and d ln F_1/2 / d eta: underflow-free for eta << 0 and
    SMOOTH with a consistent derivative -- Newton needs the exact
    derivative of a differentiable function. physics.fermi_dirac's
    fermi_half/dfermi_half linearly interpolate two separate tables:
    value and slope disagree by ~0.3% at eta ~ -20 (verified by finite
    differences), the value has a slope kink every 0.01 in eta, and below
    eta = -5 it is pure e^eta (no degeneracy correction) -- together these
    trapped Newton in a ~1e-7 V limit cycle on the UV-LED at ~3.47 V.
      eta <  -5: non-degenerate series ln[e^eta (1 - e^eta/2^1.5 + e^2eta/3^1.5)]
                 (truncation ~e^{3 eta}/8 < 4e-8), analytic derivative;
      eta >= -5: cubic spline of ln F through the table's exact quadrature
                 values (spacing 0.01), matching the series at -5 to ~4e-8.
      eta >  60: Sommerfeld expansion (table limit).
    """
    eta = np.maximum(eta, _ETA_MIN)
    deep = eta < _ETA_SWITCH
    ed = np.minimum(eta, _ETA_SWITCH)
    e = np.exp(ed)
    s = 1.0 - e / 2.0 ** 1.5 + e ** 2 / 3.0 ** 1.5
    ds = -e / 2.0 ** 1.5 + 2.0 * e ** 2 / 3.0 ** 1.5
    lnF = ed + np.log(s)
    dlnF = 1.0 + ds / s

    sp = _lnF_spline()
    mid = (~deep) & (eta <= sp.x[-1])
    if np.any(mid):
        lnF = np.where(mid, sp(np.where(mid, eta, 0.0)), lnF)
        dlnF = np.where(mid, sp(np.where(mid, eta, 0.0), 1), dlnF)
    hi = eta > sp.x[-1]
    if np.any(hi):
        eh = np.where(hi, eta, 100.0)
        C = 4.0 / (3.0 * np.sqrt(np.pi))
        a = np.pi ** 2 / 8.0
        F = C * (eh ** 1.5 + a * eh ** -0.5)
        dF = C * (1.5 * eh ** 0.5 - 0.5 * a * eh ** -1.5)
        lnF = np.where(hi, np.log(F), lnF)
        dlnF = np.where(hi, dF / F, dlnF)
    return lnF, dlnF


def _sg_flux(la, lc, b, C):
    """
    Generalized SG flux G = C*B(d)*e^lc*(1-e^-b) = C*B(-d)*e^la*(e^b-1),
    d = lc - la - b, and its partials w.r.t. (la, lc, b). Branch chosen per
    edge so B is always evaluated at a non-positive argument (B(x<=0) ~ |x|,
    never underflows) and the density factor is the one it multiplies.
    """
    d = lc - la - b
    neg = d <= 0.0
    # Form A (d <= 0): C * B(d) * e^lc * (1 - e^-b)
    # Form B (d >  0): C * B(s) * e^la * (e^b - 1),  s = -d
    x = np.where(neg, d, -d)
    Bx = bernoulli(x)
    dBx = dbernoulli(x)
    em1_neg = -np.expm1(-b)   # 1 - e^-b
    em1_pos = np.expm1(b)     # e^b - 1
    eb = np.exp(np.minimum(b, 700.0))
    emb = np.exp(np.minimum(-b, 700.0))

    ec = np.exp(lc)
    ea = np.exp(la)

    G_A = C * Bx * ec * em1_neg
    dA_la = -C * em1_neg * ec * dBx
    dA_lc = C * em1_neg * ec * (Bx + dBx)
    dA_b = C * ec * (Bx * emb - dBx * em1_neg)

    G_B = C * Bx * ea * em1_pos
    dB_la = C * em1_pos * ea * (Bx + dBx)
    dB_lc = -C * em1_pos * ea * dBx
    dB_b = C * ea * (dBx * em1_pos + Bx * eb)

    G = np.where(neg, G_A, G_B)
    dla = np.where(neg, dA_la, dB_la)
    dlc = np.where(neg, dA_lc, dB_lc)
    db = np.where(neg, dA_b, dB_b)
    return G, dla, dlc, db


@dataclass
class DDProblem:
    """Static device data for one drift-diffusion solve."""
    Ec0: np.ndarray
    Ev0: np.ndarray
    Nc: np.ndarray
    Nv: np.ndarray
    ND: np.ndarray
    NA: np.ndarray
    Ed: np.ndarray
    Ea: np.ndarray
    pol_rho: np.ndarray
    eps_r: np.ndarray
    dx: object
    T: float
    mu_n: object            # scalar or per-node array [cm^2/Vs]
    mu_p: object
    surf_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    surf_density_cm3: np.ndarray = field(default_factory=lambda: np.array([]))
    surf_energy_eV: np.ndarray = field(default_factory=lambda: np.array([]))
    surf_is_donor: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    gamma_n: Optional[np.ndarray] = None
    gamma_p: Optional[np.ndarray] = None

    def __post_init__(self):
        self.N = len(self.Ec0)
        self.kT = kB * self.T / _q
        eps_m, eps_p, h1, h2, cw = _assemble_laplacian(self.eps_r, self.dx)
        self._eps_m, self._eps_p, self._h1, self._h2, self._cw = eps_m, eps_p, h1, h2, cw
        dx_cm = np.asarray(self.dx, dtype=float) * 1e2
        _, _, cw_cm, edges_cm = node_spacings(dx_cm, self.N)
        self._cw_cm = cw_cm
        self._edges_cm = edges_cm
        mu_n = np.broadcast_to(np.asarray(self.mu_n, dtype=float), (self.N,))
        mu_p = np.broadcast_to(np.asarray(self.mu_p, dtype=float), (self.N,))
        # Edge mobility: harmonic mean (series resistance of the two half-cells).
        mun_e = 2.0 * mu_n[:-1] * mu_n[1:] / (mu_n[:-1] + mu_n[1:])
        mup_e = 2.0 * mu_p[:-1] * mu_p[1:] / (mu_p[:-1] + mu_p[1:])
        self._Cn = mun_e * self.kT / edges_cm     # D/h [cm/s]
        self._Cp = mup_e * self.kT / edges_cm
        self._ln_gn = np.zeros(self.N) if self.gamma_n is None else np.log(self.gamma_n)
        self._ln_gp = np.zeros(self.N) if self.gamma_p is None else np.log(self.gamma_p)
        self._ni2 = (self.Nc * self.Nv * np.exp(-(self.Ec0 - self.Ev0) / self.kT))

    # ------------------------------------------------------------------
    def carriers(self, phi, Efn, Efp):
        """ln n, ln p and their derivatives w.r.t. their own QFL (d ln n /
        d Efn = d ln n / d phi; d ln p / d Efp = d ln p / d phi)."""
        kT = self.kT
        eta_n = (Efn - (self.Ec0 - phi)) / kT
        eta_p = ((self.Ev0 - phi) - Efp) / kT
        lFn, dlFn = _ln_fermi_half(eta_n)
        lFp, dlFp = _ln_fermi_half(eta_p)
        ln_n = np.log(self.Nc) + lFn + self._ln_gn
        ln_p = np.log(self.Nv) + lFp + self._ln_gp
        dln_n = dlFn / kT          # d ln n / d Efn  (= d/dphi)
        dln_p = -dlFp / kT         # d ln p / d Efp  (= d/dphi)
        return ln_n, ln_p, dln_n, dln_p

    def _ionization(self, phi, Efn, Efp):
        kT = self.kT
        Ec = self.Ec0 - phi
        Ev = self.Ev0 - phi
        ad = np.clip((Efn - (Ec - self.Ed)) / kT, -340.0, 340.0)
        aa = np.clip((Ev + self.Ea - Efp) / kT, -340.0, 340.0)
        ed = np.exp(ad)
        ea = np.exp(aa)
        Ndp = self.ND / (1.0 + 2.0 * ed)
        Nam = self.NA / (1.0 + 4.0 * ea)
        # dNd+/dphi = dNd+/dEfn ; dNa-/dphi = dNa-/dEfp
        dNd = -self.ND * 2.0 * ed / (1.0 + 2.0 * ed) ** 2 / kT
        dNa = self.NA * 4.0 * ea / (1.0 + 4.0 * ea) ** 2 / kT
        if len(self.surf_idx) > 0:
            for is_donor in (True, False):
                m = self.surf_is_donor == is_donor
                if not np.any(m):
                    continue
                idx = self.surf_idx[m]
                dens = self.surf_density_cm3[m]
                E = self.surf_energy_eV[m]
                if is_donor:
                    a = np.clip((Efn[idx] - (Ec[idx] - E)) / kT, -340.0, 340.0)
                    e = np.exp(a)
                    np.add.at(Ndp, idx, dens / (1.0 + 2.0 * e))
                    np.add.at(dNd, idx, -dens * 2.0 * e / (1.0 + 2.0 * e) ** 2 / kT)
                else:
                    a = np.clip((Ev[idx] + E - Efp[idx]) / kT, -340.0, 340.0)
                    e = np.exp(a)
                    np.add.at(Nam, idx, dens / (1.0 + 4.0 * e))
                    np.add.at(dNa, idx, dens * 4.0 * e / (1.0 + 4.0 * e) ** 2 / kT)
        return Ndp, Nam, dNd, dNa

    def _recombination(self, n, p):
        ni2 = self._ni2
        ni = np.sqrt(ni2)
        np_ = n * p
        excess = np_ - ni2
        den = _TAU_P * (n + ni) + _TAU_N * (p + ni)
        R = excess / den + _B_RAD * excess + _C_AUG * (n + p) * excess
        dR_dn = (p / den - excess * _TAU_P / den ** 2) + _B_RAD * p \
            + _C_AUG * (excess + (n + p) * p)
        dR_dp = (n / den - excess * _TAU_N / den ** 2) + _B_RAD * n \
            + _C_AUG * (excess + (n + p) * n)
        return R, dR_dn, dR_dp

    # ------------------------------------------------------------------
    def fluxes(self, phi, Efn, Efp):
        """Edge particle fluxes G = Jn/q, H = Jp/q [cm^-2 s^-1] (length N-1)."""
        ln_n, ln_p, _, _ = self.carriers(phi, Efn, Efp)
        kT = self.kT
        G, *_ = _sg_flux(ln_n[:-1], ln_n[1:], (Efn[1:] - Efn[:-1]) / kT, self._Cn)
        # Holes: mirror image -- density ln p, "quasi-Fermi" -Efp, and the
        # sign of the resulting flux flipped so H -> mu p dEfp/dx.
        Hm, *_ = _sg_flux(ln_p[:-1], ln_p[1:], -(Efp[1:] - Efp[:-1]) / kT, self._Cp)
        return G, -Hm

    def residual_and_jacobian(self, phi, Efn, Efp, bc, want_jac=True):
        N, kT = self.N, self.kT
        ln_n, ln_p, dln_n, dln_p = self.carriers(phi, Efn, Efp)
        n = np.exp(ln_n)
        p = np.exp(ln_p)
        dn = n * dln_n            # dn/dEfn = dn/dphi
        dp = p * dln_p            # dp/dEfp = dp/dphi
        Ndp, Nam, dNd, dNa = self._ionization(phi, Efn, Efp)

        F = np.zeros(3 * N)
        iP = 3 * np.arange(N)       # phi rows/cols
        iN = iP + 1                 # Efn
        iH = iP + 2                 # Efp
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        I = np.arange(1, N - 1)
        # ---------------- Poisson ----------------
        cw = self._cw[1:-1]
        am = self._eps_m / self._h1 / cw
        ap = self._eps_p / self._h2 / cw
        rho = _q * 1e6 * (p - n + Ndp - Nam) + self.pol_rho
        F[iP[I]] = eps0 * (am * (phi[I] - phi[I - 1]) + ap * (phi[I] - phi[I + 1])) - rho[I]
        if want_jac:
            add(iP[I], iP[I - 1], -eps0 * am)
            add(iP[I], iP[I + 1], -eps0 * ap)
            add(iP[I], iP[I], eps0 * (am + ap) - _q * 1e6 * (dp[I] - dn[I] + dNd[I] - dNa[I]))
            add(iP[I], iN[I], -_q * 1e6 * (-dn[I] + dNd[I]))
            add(iP[I], iH[I], -_q * 1e6 * (dp[I] - dNa[I]))

        # ---------------- recombination ----------------
        R, dR_dn, dR_dp = self._recombination(n, p)

        # ---------------- electrons ----------------
        bn = (Efn[1:] - Efn[:-1]) / kT
        G, g_la, g_lc, g_b = _sg_flux(ln_n[:-1], ln_n[1:], bn, self._Cn)
        cwc = self._cw_cm[1:-1]
        # edges: k = i-1/2 -> index i-1 ; i+1/2 -> index i
        F[iN[I]] = (G[I] - G[I - 1]) / cwc - R[I]
        # ---------------- holes (mirror) ----------------
        bp = -(Efp[1:] - Efp[:-1]) / kT
        Hm, h_la, h_lc, h_b = _sg_flux(ln_p[:-1], ln_p[1:], bp, self._Cp)
        H = -Hm
        F[iH[I]] = (H[I] - H[I - 1]) / cwc + R[I]

        if want_jac:
            # Electron row i: +G[i]/cw (edge i -> i+1, node a=i, c=i+1)
            #                  -G[i-1]/cw (edge i-1 -> i, node a=i-1, c=i)
            s = 1.0 / cwc
            # edge i (right): la = ln n_i, lc = ln n_{i+1}
            dGr_la, dGr_lc, dGr_b = g_la[I], g_lc[I], g_b[I]
            dGl_la, dGl_lc, dGl_b = g_la[I - 1], g_lc[I - 1], g_b[I - 1]
            # w.r.t. node i (la of right edge, lc of left edge)
            d_lni = s * (dGr_la - dGl_lc)
            d_bi = s * (-dGr_b / kT - dGl_b / kT)          # b_r = (Efn_{i+1}-Efn_i)/kT, b_l = (Efn_i-Efn_{i-1})/kT
            d_lnip1 = s * dGr_lc
            d_bip1 = s * dGr_b / kT
            d_lnim1 = -s * dGl_la
            d_bim1 = s * dGl_b / kT
            # chain: ln n_j depends on phi_j and Efn_j with the same slope dln_n[j]
            add(iN[I], iP[I], d_lni * dln_n[I] - dR_dn[I] * dn[I] - dR_dp[I] * dp[I])
            add(iN[I], iN[I], d_lni * dln_n[I] + d_bi - dR_dn[I] * dn[I])
            add(iN[I], iH[I], -dR_dp[I] * dp[I])
            add(iN[I], iP[I + 1], d_lnip1 * dln_n[I + 1])
            add(iN[I], iN[I + 1], d_lnip1 * dln_n[I + 1] + d_bip1)
            add(iN[I], iP[I - 1], d_lnim1 * dln_n[I - 1])
            add(iN[I], iN[I - 1], d_lnim1 * dln_n[I - 1] + d_bim1)

            # Hole row: F = (H[i]-H[i-1])/cw + R, H = -Hm(ln p, bp), bp = -(dEfp)/kT
            dHr_la, dHr_lc, dHr_b = -h_la[I], -h_lc[I], -h_b[I]
            dHl_la, dHl_lc, dHl_b = -h_la[I - 1], -h_lc[I - 1], -h_b[I - 1]
            # d bp_right / d Efp_i = +1/kT ; d bp_right / d Efp_{i+1} = -1/kT
            # d bp_left  / d Efp_i = -1/kT ; d bp_left  / d Efp_{i-1} = +1/kT
            d_lpi = s * (dHr_la - dHl_lc)
            d_ci = s * (dHr_b / kT + dHl_b / kT)
            d_lpip1 = s * dHr_lc
            d_cip1 = -s * dHr_b / kT
            d_lpim1 = -s * dHl_la
            d_cim1 = -s * dHl_b / kT
            add(iH[I], iP[I], d_lpi * dln_p[I] + dR_dn[I] * dn[I] + dR_dp[I] * dp[I])
            add(iH[I], iH[I], d_lpi * dln_p[I] + d_ci + dR_dp[I] * dp[I])
            add(iH[I], iN[I], dR_dn[I] * dn[I])
            add(iH[I], iP[I + 1], d_lpip1 * dln_p[I + 1])
            add(iH[I], iH[I + 1], d_lpip1 * dln_p[I + 1] + d_cip1)
            add(iH[I], iP[I - 1], d_lpim1 * dln_p[I - 1])
            add(iH[I], iH[I - 1], d_lpim1 * dln_p[I - 1] + d_cim1)

        # ---------------- Dirichlet contacts ----------------
        phiL, phiR, EfnL, EfnR, EfpL, EfpR = bc
        for node, vals_bc in ((0, (phiL, EfnL, EfpL)), (N - 1, (phiR, EfnR, EfpR))):
            F[iP[node]] = phi[node] - vals_bc[0]
            F[iN[node]] = Efn[node] - vals_bc[1]
            F[iH[node]] = Efp[node] - vals_bc[2]
            if want_jac:
                for rr in (iP[node], iN[node], iH[node]):
                    add(np.array([rr]), np.array([rr]), np.array([1.0]))

        if not want_jac:
            return F, None
        r = np.concatenate([np.atleast_1d(x) for x in rows])
        c = np.concatenate([np.atleast_1d(x) for x in cols])
        v = np.concatenate([np.atleast_1d(x) for x in vals])
        J = coo_matrix((v, (r, c)), shape=(3 * N, 3 * N)).tocsr()
        return F, J


@dataclass
class DDResult:
    phi: np.ndarray
    Efn: np.ndarray
    Efp: np.ndarray
    converged: bool
    n_iter: int
    max_update: float
    V_reached: float = 0.0
    message: str = ""


def newton_solve(prob: DDProblem, phi, Efn, Efp, bc,
                 tol_update: float = 1e-7, maxiter: int = 40,
                 max_step: float = 0.5,
                 log_fn: Optional[Callable[[str], None]] = None,
                 cancel_check: Optional[Callable[[], bool]] = None) -> DDResult:
    """Damped Newton at fixed contact BCs. Converged when the full
    (undamped) Newton update is below tol_update volts in every unknown."""
    N = prob.N
    phi, Efn, Efp = phi.copy(), Efn.copy(), Efp.copy()
    phi[0], Efn[0], Efp[0] = bc[0], bc[2], bc[4]
    phi[-1], Efn[-1], Efp[-1] = bc[1], bc[3], bc[5]
    max_upd = np.inf
    for it in range(1, maxiter + 1):
        if cancel_check is not None and cancel_check():
            raise SolveCancelled("Solve cancelled by user")
        F, J = prob.residual_and_jacobian(phi, Efn, Efp, (bc[0], bc[1], bc[2], bc[3], bc[4], bc[5]))
        if not np.all(np.isfinite(F)):
            return DDResult(phi, Efn, Efp, False, it, np.inf, message="non-finite residual")
        # Row equilibration (pivoting quality only; doesn't change the step).
        row_max = np.abs(J).max(axis=1).toarray().ravel()
        row_max[row_max == 0] = 1.0
        S = sp_diags(1.0 / row_max)
        try:
            delta = -spsolve((S @ J).tocsc(), S @ F)
        except Exception as exc:  # noqa: BLE001
            return DDResult(phi, Efn, Efp, False, it, np.inf, message=f"linear solve failed: {exc}")
        if not np.all(np.isfinite(delta)):
            return DDResult(phi, Efn, Efp, False, it, np.inf, message="non-finite Newton step")
        # Convergence measure (undamped update): phi everywhere, and each
        # QFL weighted by min(1, density / 1 cm^-3). A QFL is only defined
        # to roundoff where its carrier is essentially absent (confirmed on
        # the UV-LED: n ~ e^-85 at the p-contact, p ~ e^-61 in the undoped
        # AlN layer -- their updates bounce at 1e-7..1e-6 V forever), and an
        # error dEf there changes that density by only n*dEf/kT, i.e.
        # nothing. Where n, p >= 1 cm^-3 the full tol_update applies.
        # current_profile's conservation check stays as the independent,
        # unmaskable physics check.
        ln_n, ln_p, _, _ = prob.carriers(phi, Efn, Efp)
        w_n = np.exp(np.minimum(ln_n, 0.0))
        w_p = np.exp(np.minimum(ln_p, 0.0))
        max_upd = float(max(np.abs(delta[0::3]).max(),
                            np.abs(delta[1::3] * w_n).max(),
                            np.abs(delta[2::3] * w_p).max()))
        # Per-component logarithmic damping (standard TCAD update limiting):
        # |delta| << kT passes through unchanged, large updates are
        # compressed to ~kT*ln(|delta|/kT). Scaling the WHOLE step by the
        # largest component instead was confirmed to stall the UV-LED at
        # ~3.5 V: one barely-constrained minority QFL (n ~ 1e-12 cm^-3 in
        # the p-GaN contact layer) asked for a 1e4-1e23 V update and froze
        # every other node at step x0.00.
        kT = prob.kT
        damped = np.sign(delta) * kT * np.log1p(np.abs(delta) / kT)
        damped = np.clip(damped, -max_step, max_step)
        phi += damped[0::3]
        Efn += damped[1::3]
        Efp += damped[2::3]
        # Row-equilibrated residual: each row divided by its largest
        # Jacobian entry, i.e. "how many volts the dominant unknown of that
        # row is off by" -- a scale taken from the equations themselves, not
        # an arbitrary normalization. Used as a second acceptance test: a
        # nearly-decoupled carrier reservoir (e.g. an n-region separated
        # from its contact by an undoped high-Al layer: cond(J) ~ 4.5e16
        # measured on the UV-LED) leaves the update bouncing at cond*eps
        # ~1e-7 V along that one near-null direction even once the
        # residual is at machine precision.
        res_volts = float(np.abs(S @ F).max())
        if log_fn is not None:
            log_fn(f"      Newton {it:2d}: max|update| = {max_upd:.3e} V  "
                   f"row-scaled residual = {res_volts:.3e} V")
        if max_upd < tol_update or (res_volts < _RES_VOLTS_TOL and max_upd < _NOISE_UPDATE_MAX):
            return DDResult(phi, Efn, Efp, True, it, max_upd)
    return DDResult(phi, Efn, Efp, False, maxiter, max_upd, message="maxiter")


def solve_bias_ramp(prob: DDProblem, phi_eq, V_target: float,
                    phi_left: float, phi_right_eq: float, *args,
                    log_fn: Optional[Callable[[str], None]] = None, **kwargs) -> DDResult:
    """
    solve_bias_ramp with fallbacks: if the adaptive ramp stalls (in
    practice at an S-shaped I-V fold it cannot leap from the stalled side),
    retry fresh ramps from the same start with coarser step patterns, which
    cross the fold in one step from well behind it (confirmed: direct
    ramps to 5.2-5.3 V converge on the UV-LED where the fine ramp stalls at
    the 5.06 V fold). Returns the first success, else the furthest-reaching
    failure.
    """
    res = _solve_bias_ramp(prob, phi_eq, V_target, phi_left, phi_right_eq, *args,
                           log_fn=log_fn, **kwargs)
    if res.converged:
        return res
    best = res
    for dV_init, dV_max in _RAMP_RETRY_PATTERNS:
        if log_fn is not None:
            log_fn(f"    DD ramp: retrying with coarser steps (dV_init={dV_init}, dV_max={dV_max})")
        kw = dict(kwargs, dV_init=dV_init, dV_max=dV_max)
        r = _solve_bias_ramp(prob, phi_eq, V_target, phi_left, phi_right_eq, *args,
                             log_fn=log_fn, **kw)
        if r.converged:
            return r
        if abs(r.V_reached) > abs(best.V_reached):
            best = r
    return best


def _solve_bias_ramp(prob: DDProblem, phi_eq, V_target: float,
                    phi_left: float, phi_right_eq: float,
                    phi_start=None, Efn_start=None, Efp_start=None, V_start: float = 0.0,
                    dV_init: float = 0.1, dV_max: float = 0.5, dV_min: float = 1e-3,
                    tol_update: float = 1e-7,
                    log_fn: Optional[Callable[[str], None]] = None,
                    cancel_check: Optional[Callable[[], bool]] = None) -> DDResult:
    """
    Adaptive bias continuation from a converged state at V_start (default:
    equilibrium, Efn = Efp = 0) to V_target. Each step is a full Newton
    solve at the new contact BCs, seeded with the previous converged state;
    a failed step is retried at half the increment. Safe because every
    step converges on the true (unmasked) equations.

    Contacts: left (x=0) is grounded (phi = phi_left, Efn = Efp = 0);
    right is biased (phi = phi_right_eq + V, Efn = Efp = -V).
    """
    phi = (phi_eq if phi_start is None else phi_start).copy()
    Efn = np.zeros(prob.N) if Efn_start is None else Efn_start.copy()
    Efp = np.zeros(prob.N) if Efp_start is None else Efp_start.copy()
    V = V_start
    sign = 1.0 if V_target >= V_start else -1.0
    dV = min(dV_init, abs(V_target - V_start))
    total_iter = 0
    if phi_start is None and V_target != V_start:
        # Converge the starting point for THIS problem first (a quantum
        # gamma shifts the equilibrium away from the classical phi_eq).
        bc0 = (phi_left, phi_right_eq + V, 0.0, -V, 0.0, -V)
        res0 = newton_solve(prob, phi, Efn, Efp, bc0, tol_update=tol_update,
                            cancel_check=cancel_check)
        total_iter += res0.n_iter
        if not res0.converged:
            return DDResult(phi, Efn, Efp, False, total_iter, res0.max_update, V_reached=V,
                            message=f"could not converge the starting point at V = {V:.4f} V")
        phi, Efn, Efp = res0.phi, res0.Efn, res0.Efp
    if abs(V_target - V_start) == 0.0:
        bc = (phi_left, phi_right_eq + V, 0.0, -V, 0.0, -V)
        res = newton_solve(prob, phi, Efn, Efp, bc, tol_update=tol_update,
                           log_fn=log_fn, cancel_check=cancel_check)
        res.V_reached = V
        return res
    history = [(V, phi, Efn, Efp)]
    while abs(V_target - V) > 1e-12:
        V_try = V + sign * min(dV, abs(V_target - V))
        bc = (phi_left, phi_right_eq + V_try, 0.0, -V_try, 0.0, -V_try)
        res = newton_solve(prob, phi, Efn, Efp, bc, tol_update=tol_update,
                           log_fn=None, cancel_check=cancel_check)
        total_iter += res.n_iter
        if res.converged:
            phi, Efn, Efp, V = res.phi, res.Efn, res.Efp, V_try
            history.append((V, phi, Efn, Efp))
            if log_fn is not None:
                log_fn(f"    DD ramp: V = {V:.4f} V converged in {res.n_iter} Newton iters")
            dV = min(dV * (1.5 if res.n_iter <= 12 else 1.0), dV_max)
        else:
            dV *= 0.5
            if log_fn is not None:
                log_fn(f"    DD ramp: step to V = {V_try:.4f} V failed ({res.message}); "
                       f"halving step to {dV:.4f} V")
            if dV < dV_min:
                # Small steps can't get through: a fold of an S-shaped
                # (bistable) I-V. Confirmed on the UV-LED (dx=0.5 nm): for
                # 5.04-5.06 V a lower branch (J ~60-140 A/cm^2) and an upper
                # branch (J ~1800-2300 A/cm^2) coexist (up- vs down-sweeps
                # agree everywhere else) -- carrier injection over the
                # polarization-heavy EBL. The Jacobian is singular AT the
                # fold, so Newton diverges from the last converged state
                # (which sits on it). Leap over to the other branch the way
                # a voltage-driven device does, starting from converged
                # states slightly BEHIND the fold (where the Jacobian is
                # regular) and progressively larger jumps.
                leapt = False
                remaining = abs(V_target - V)
                starts = [h for h in reversed(history)
                          if abs(V - h[0]) in (0.0,) or abs(V - h[0]) >= _LEAP_BACKOFF_MIN][:_LEAP_N_STARTS]
                for jump in _LEAP_STEPS:
                    jump = min(jump, remaining)
                    V_try = V + sign * jump
                    bc = (phi_left, phi_right_eq + V_try, 0.0, -V_try, 0.0, -V_try)
                    for (V_s, phi_s, Efn_s, Efp_s) in starts:
                        res = newton_solve(prob, phi_s, Efn_s, Efp_s, bc, tol_update=tol_update,
                                           maxiter=_LEAP_MAXITER, cancel_check=cancel_check)
                        total_iter += res.n_iter
                        if res.converged:
                            break
                    if res.converged:
                        if log_fn is not None:
                            J_old = float(np.mean(current_profile(prob, phi, Efn, Efp)[2]))
                            J_new = float(np.mean(current_profile(prob, res.phi, res.Efn, res.Efp)[2]))
                            log_fn(f"    DD ramp: I-V fold near V = {V:.4f} V (J = {J_old:.3e} A/cm^2); "
                                   f"jumped to the other branch at V = {V_try:.4f} V "
                                   f"(J = {J_new:.3e} A/cm^2)")
                        phi, Efn, Efp, V = res.phi, res.Efn, res.Efp, V_try
                        history.append((V, phi, Efn, Efp))
                        dV = dV_init
                        leapt = True
                        break
                    if jump == remaining:
                        break
                if not leapt:
                    return DDResult(phi, Efn, Efp, False, total_iter, res.max_update, V_reached=V,
                                    message=f"bias ramp stalled at V = {V:.4f} V")
    return DDResult(phi, Efn, Efp, True, total_iter, 0.0, V_reached=V)


def current_profile(prob: DDProblem, phi, Efn, Efp):
    """
    Edge current densities [A/cm^2] and a current-conservation diagnostic.

    Returns (Jn, Jp, J_total, conservation_error) where conservation_error =
    (max J_total - min J_total) / max|J_total| over all edges -- ~0 for a
    physically valid steady state (dJ/dx = 0).
    """
    G, H = prob.fluxes(phi, Efn, Efp)
    Jn = _q * G
    Jp = _q * H
    Jt = Jn + Jp
    scale = float(np.max(np.abs(Jt)))
    err = float((Jt.max() - Jt.min()) / scale) if scale > 0 else 0.0
    return Jn, Jp, Jt, err

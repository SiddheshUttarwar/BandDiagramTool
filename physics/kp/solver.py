"""
8-band k.p eigenstates of a 1D (growth direction z) heterostructure and the
electron and hole densities they carry.

Discretisation. The bulk Hamiltonian of each node (physics.kp.hamiltonian)
is quadratic in kz and in the in-plane wavevector k_t, so it is split
exactly into coefficient matrices D[a][b] (8x8, Hermitian) of k_t^a kz^b.
With kz -> -i d/dz, the finite-volume (Galerkin) forms on the grid's own
nodes and cell widths w_j (physics.grid_utils.node_spacings) are

    D[.][0] psi            ->  sum_j w_j phi_j^+ D0_j psi_j
    (D1 kz + kz D1) / 2    ->  -(i/2) sum_e phi_j^+ D1_e psi_j+1 - phi_j+1^+ D1_e psi_j
    kz D2 kz               ->  sum_e (phi_j+1 - phi_j)^+ D2_e (psi_j+1 - psi_j) / h_e

with D_e the mean of the two nodes of edge e: the symmetrised operator
ordering, Hermitian by construction. Hard walls (psi = 0) one edge beyond
each end of the region, as in physics.schrodinger. The generalised problem
K psi = E W psi is made standard with phi = W^(1/2) psi. Ordering the
unknowns node by node (8 per node) makes the matrix block-tridiagonal; the
eigenpairs inside an energy window come from shift-invert Lanczos (ARPACK,
scipy.sparse.linalg.eigsh) anchored at the window edge on the side of the
band gap, with the number of requested pairs doubled until the window is
covered. (LAPACK's banded selective solver forms a dense transformation
and was ~100x slower at 500 nodes.)

Mirror split (exact). For an in-plane wavevector along the unit vector
(cos t, sin t), the vertical plane containing it is a mirror plane of the
(axial) Hamiltonian; its operator M = R_orbital (x) (-i n.sigma), n the
plane normal, commutes with H(k_t, kz) and has eigenvalues +-i. In M's
eigenbasis (a constant 8x8 unitary U) the Hamiltonian is two 4x4 blocks at
every node and every k, so each block is solved as a 4-component problem
(confirmed: off-block elements 1e-16, eigenvalues equal to the 8x8 ones
to 1e-15). Without k-linear terms the blocks are degenerate (Kramers
partners), so zincblende solves one block and counts it twice; the
wurtzite A7 term splits them by up to ~0.2 meV, so both are solved.

Axial approximation: the in-plane dispersion is taken as isotropic. For
zincblende the bulk Hamiltonian is averaged over in-plane rotations,
H_ax(k) = <Rot(phi)^+ H(R_phi k) Rot(phi)>_phi, which removes the cubic
(cos 4 phi) warping; for these quadratic terms the average over phi = 0 and
pi/4 is exact. Wurtzite is isotropic in-plane already.

Real blocks. After the mirror split a constant diagonal phase per
component (found by a spanning-tree walk over the couplings and then
checked element by element) makes every 4x4 block real, so the
eigenproblem is real symmetric: symmetric Lanczos (ARPACK dsaupd) instead
of the complex non-Hermitian Arnoldi scipy uses for complex matrices,
which was ~110 iterations per call. (The [110] zincblende block cannot be
made real; the axial average removes it.)

Densities:

    n(z) = (1 / 2 pi) int k dk  sum_{s in CB} |psi_s(z; k)|^2 f(E_s(k) - Efn)
    p(z) = (1 / 2 pi) int k dk  sum_{s in VB} |psi_s(z; k)|^2 (1 - f(E_s(k) - Efp))

Spin is explicit in the basis, so there is no factor 2. A state is a
conduction state when more than half of its weight is on |S>. The k
integral uses Gauss-Legendre points on [0, k_max], k_max raised until no
appreciably occupied state is left at k_max.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from physics.constants import kB, q as _q
from physics.grid_utils import node_spacings
from physics.kp.hamiltonian import bulk_hamiltonian
from physics.kp.parameters import KPParams, HB

_S_IDX = (0, 4)               # |S> up / down
_N_K = 16                     # Gauss-Legendre points of the k_t integral (HEMT: -0.04 % vs 48)
_TAIL_KT = 10.0               # occupation window beyond the quasi-Fermi level [kT]
_OCC_TOL = 1e-6               # an occupation below this at k_max ends the k range


def _mirror_unitary(cos_t: float, sin_t: float) -> np.ndarray:
    """8x8 unitary whose first/last four columns span the +i / -i
    eigenspaces of the mirror through the plane (k_t direction, z)."""
    n = np.array([-sin_t, cos_t, 0.0])
    orb = np.eye(4)
    orb[1:, 1:] = np.eye(3) - 2.0 * np.outer(n, n)
    sig = (np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]))
    M = np.kron(-1j * (n[0] * sig[0] + n[1] * sig[1]), orb)
    w, V = np.linalg.eig(M)
    cols = [np.linalg.qr(V[:, np.abs(w - lam) < 1e-8])[0] for lam in (1j, -1j)]
    return np.hstack(cols)


def _rotation(phi: float) -> np.ndarray:
    """8x8 rotation by phi about z on (S, X, Y, Z) x spin."""
    c, s_ = np.cos(phi), np.sin(phi)
    orb = np.eye(4)
    orb[1:3, 1:3] = [[c, -s_], [s_, c]]
    spin = np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])
    return np.kron(spin, orb)


def _real_phases(mats, nc: int = 4):
    """Diagonal unitary P making every P^+ M P real, or None."""
    agg = np.zeros((nc, nc))
    for M in mats:
        agg = np.maximum(agg, np.abs(M))
    th = np.full(nc, np.nan)
    th[0] = 0.0
    changed = True
    while changed:
        changed = False
        for a in range(nc):
            if np.isnan(th[a]):
                continue
            for b in range(nc):
                if a != b and np.isnan(th[b]) and agg[a, b] > 1e-12:
                    z = max(mats, key=lambda M: abs(M[a, b]))[a, b]
                    th[b] = th[a] - np.angle(z)
                    changed = True
    P = np.diag(np.exp(1j * np.nan_to_num(th)))
    scale = max(float(agg.max()), 1e-30)
    ok = all(np.abs((P.conj().T @ M @ P).imag).max() <= 1e-10 * scale for M in mats)
    return P if ok else None


def _mirror_blocks(D: np.ndarray):
    """(U^+ D U, U): the coefficients in the mirror eigenbasis, where they
    are block diagonal (4 + 4)."""
    U = _mirror_unitary(1.0, 0.0)
    return np.matmul(np.matmul(U.conj().T[None, None, None], D), U[None, None, None]), U


def _coefficients_no_a7(par: KPParams, D1: np.ndarray) -> np.ndarray:
    """D[1] (the k_t-linear coefficients) without the wurtzite A7 term
    (N3' = i sqrt2 A7 on X-Z, Y-Z: kx-linear, kz-independent)."""
    if not np.any(par.N3):
        return D1
    D1 = D1.copy()
    n3 = 1j * par.N3[:, None, None] if False else 1j * par.N3
    for off in (0, 4):
        x, z = 1 + off, 3 + off
        D1[0, :, x, z] += n3            # remove -N3' kx (X-Z) ...
        D1[0, :, z, x] -= n3            # ... and its conjugate (Z-X)
    return D1


def _coefficients(par: KPParams, cos_t: float, sin_t: float) -> np.ndarray:
    """D[a, b, j] (shape (3, 3, N, 8, 8)): coefficient of k_t^a kz^b at
    node j, from the bulk Hamiltonian evaluated at k_t, kz in {-1, 0, 1}."""
    N = len(par.Ec)
    D = np.zeros((3, 3, N, 8, 8), complex)
    for j in range(N):
        p = par.node(j)
        H = {(a, b): bulk_hamiltonian(p, a * cos_t, a * sin_t, b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
        # quadratic in each variable: c0 = H(0), c1 = (H(1) - H(-1))/2, c2 = (H(1) + H(-1))/2 - H(0)
        G = np.zeros((3, 3, 8, 8), complex)        # G[a, b] over kz for fixed k_t = a - 1
        for ia, a in enumerate((-1, 0, 1)):
            G[ia, 0] = H[(a, 0)]
            G[ia, 1] = 0.5 * (H[(a, 1)] - H[(a, -1)])
            G[ia, 2] = 0.5 * (H[(a, 1)] + H[(a, -1)]) - H[(a, 0)]
        for b in range(3):
            D[0, b, j] = G[1, b]
            D[1, b, j] = 0.5 * (G[2, b] - G[0, b])
            D[2, b, j] = 0.5 * (G[2, b] + G[0, b]) - G[1, b]
    return D


@dataclass
class _Assembled:
    """Block-tridiagonal pieces of W^-1/2 K_a W^-1/2 (a = k_t power):
    diagonal blocks Bd[a] (N,8,8) and upper blocks Bo[a] (N-1,8,8)."""
    Bd: np.ndarray
    Bo: np.ndarray
    inv_sqrt_w: np.ndarray     # per node [nm^-1/2]
    U: Optional[np.ndarray] = None     # (8, nc): block components -> S,X,Y,Z basis
    weight: float = 1.0                # share of this block in the k integral


def _assemble(D: np.ndarray, h_nm: np.ndarray, w_nm: np.ndarray) -> _Assembled:
    N, nc = D.shape[2], D.shape[3]
    Bd = np.zeros((3, N, nc, nc), complex)
    Bo = np.zeros((3, max(N - 1, 0), nc, nc), complex)
    for a in range(3):
        Bd[a] += w_nm[:, None, None] * D[a, 0]
        if N > 1:
            D1e = 0.5 * (D[a, 1, :-1] + D[a, 1, 1:])
            D2e = 0.5 * (D[a, 2, :-1] + D[a, 2, 1:]) / h_nm[:, None, None]
            Bo[a] += -0.5j * D1e - D2e
            Bd[a, :-1] += D2e
            Bd[a, 1:] += D2e
        # hard walls one edge beyond each end
        Bd[a, 0] += D[a, 2, 0] / (h_nm[0] if N > 1 else 1.0)
        Bd[a, -1] += D[a, 2, -1] / (h_nm[-1] if N > 1 else 1.0)
    s = 1.0 / np.sqrt(w_nm)
    Bd *= (s[:, None, None] ** 2)[None]
    if N > 1:
        Bo *= (s[:-1] * s[1:])[None, :, None, None]
    return _Assembled(Bd, Bo, s)


import os as _os
import time as _time
_TRACE = bool(_os.environ.get('EPIBAND_KP_TRACE'))    # per-batch timing to stderr
_CNL_KT = 2.0                 # width of the electron/hole switch at the mid-gap [kT]
_SLICE_STATES = 30           # eigenpairs per shift-invert slice
_SLICE_TOL = 1e-10           # eV; near-degenerate clusters at a slice boundary go to the next slice


def _sliced_eigsh(H, lo: float, hi: float, side: str, m: int):
    """All eigenpairs of the sparse Hermitian H with lo <= E <= hi, by
    spectrum slicing: shift-invert requests of m pairs, swept away from the
    band gap ('down' from hi, 'up' from lo).

    A request centred at sigma whose m nearest eigenvalues reach out to
    distance d contains EVERY eigenvalue in [sigma - d, sigma + d]. The
    first slice is centred at the window edge; each next one half a
    slice-spread beyond the region covered so far, and is moved back if it
    fails to overlap it. Each slice keeps only its new range; its outermost
    eigenvalue, and anything within _SLICE_TOL of it, is left to the next
    slice, which overlaps it and so holds that whole near-degenerate cluster
    (the 7 wells of a superlattice give distinct states closer than 1e-9
    eV, so de-duplicating by value would merge them). Checked: identical
    eigenvalue set to one large request for slices of 20-60."""
    sgn = -1.0 if side == 'down' else 1.0
    end = lo if side == 'down' else hi
    edge = hi if side == 'down' else lo
    sigma = edge
    Es, Vs = [], []
    spread = None
    for _ in range(400):
        try:
            E, v = eigsh(H, k=m, sigma=sigma, which='LM')
        except ArpackNoConvergence as exc:
            E, v = exc.eigenvalues, exc.eigenvectors
        d = float(np.max(np.abs(E - sigma)))
        near, far = (sigma + d, sigma - d) if side == 'down' else (sigma - d, sigma + d)
        if spread is not None and sgn * (near - edge) > 0:      # gap to the covered region
            sigma = 0.5 * (sigma + edge)
            continue
        last = sgn * (far - end) >= 0
        keep = np.ones(E.shape, bool) if last else (sgn * (E - far) < -_SLICE_TOL)
        if spread is not None:
            keep &= sgn * (E - edge) >= -_SLICE_TOL
        Es.append(E[keep])
        Vs.append(v[:, keep])
        n_slices = len(Es)
        spread = max(float(E.max() - E.min()), 1e-6)
        edge = far
        if last:
            break
        sigma = edge + sgn * 0.5 * spread
    if _TRACE:
        import sys as _sys
        print(f'      sliced: {len(Es)} slices, {sum(len(e) for e in Es)} states', file=_sys.stderr, flush=True)
    return np.concatenate(Es), np.concatenate(Vs, axis=1)


def _sparse_pattern(N: int, nc: int):
    """Row/column indices of the (nc N) x (nc N) block-tridiagonal matrix,
    in the order (diagonal blocks, upper blocks, lower blocks)."""
    r, c = np.meshgrid(np.arange(nc), np.arange(nc), indexing='ij')
    jd = np.arange(N)[:, None, None]
    rows_d, cols_d = (nc * jd + r).ravel(), (nc * jd + c).ravel()
    jo = np.arange(N - 1)[:, None, None]
    rows_u, cols_u = (nc * jo + r).ravel(), (nc * (jo + 1) + c).ravel()
    return (np.concatenate([rows_d, rows_u, cols_u]), np.concatenate([cols_d, cols_u, rows_u]))


def _block_matrix(Bd3, Bo3, U, pattern, k: float, Ec: np.ndarray, Ev: np.ndarray):
    """Sparse 4N x 4N matrix of one mirror block at in-plane k."""
    Bd = Bd3[0] + k * Bd3[1] + k * k * Bd3[2]
    Bo = Bo3[0] + k * Bo3[1] + k * k * Bo3[2]
    # (the Gamma offset and the valence splittings are already in D[0][0]; the
    # potential, W^-1/2 (w V) W^-1/2 = V, is Ec on |S> and Ev on X, Y, Z, i.e.
    # Ev + (Ec - Ev) |S><S| -> U^+ |S><S| U per block)
    Ps = U[list(_S_IDX)].conj().T @ U[list(_S_IDX)]
    if np.isrealobj(Bd):
        Ps = Ps.real
    Bd = Bd + Ev[:, None, None] * np.eye(Bd.shape[1])[None] + (Ec - Ev)[:, None, None] * Ps[None]
    vals = np.concatenate([Bd.ravel(), Bo.ravel(), Bo.conj().ravel()])
    n = Bd.shape[1] * len(Ec)
    return sp.csc_matrix((vals, pattern), shape=(n, n))


def _eigen_core(Bd3, Bo3, U, inv_sqrt_w, pattern, k, Ec, Ev, lo, hi, side, n_hint):
    """(E ascending, spinors (n, N, 8)) of one block in [lo, hi] at k."""
    N = len(Ec)
    if hi <= lo:
        return np.zeros(0), np.zeros((0, N, 8), complex)
    H = _block_matrix(Bd3, Bo3, U, pattern, k, Ec, Ev)
    n8 = H.shape[0]
    if n_hint is not None and n_hint + 3 <= _SLICE_STATES:
        sigma = lo if side == 'up' else hi
        n_req = min(max(n_hint + 3, 4), n8 - 2)
        while True:
            try:
                E, v = eigsh(H, k=n_req, sigma=sigma, which='LM')
            except ArpackNoConvergence as exc:
                E, v = exc.eigenvalues, exc.eigenvectors
            covered = (E.max() >= hi) if side == 'up' else (E.min() <= lo)
            if covered or n_req >= n8 - 2:
                break
            n_req = min(2 * n_req, n8 - 2)
    else:
        E, v = _sliced_eigsh(H, lo, hi, side, min(_SLICE_STATES, n8 - 2))
    sel = (E >= lo) & (E <= hi)
    E, v = E[sel], v[:, sel]
    order = np.argsort(E)
    E, v = E[order], v[:, order]
    psi4 = (v.T.reshape(len(E), N, U.shape[1])) * inv_sqrt_w[None, :, None]
    return E, psi4 @ U.T


def _k_states(args):
    """Worker task: the states of one carrier kind at one k, as (E, rho,
    n_window) with rho = sum over the 8 components of |psi|^2 [nm^-1] and
    n_window the number of eigenpairs in [lo, hi] before the character
    filter (what the next request must be sized for)."""
    (Bd3, Bo3, U, inv_sqrt_w, pattern, w_nm, k, Ec, Ev, lo, hi, side, n_hint, kind) = args
    E, psi = _eigen_core(Bd3, Bo3, U, inv_sqrt_w, pattern, k, Ec, Ev, lo, hi, side, n_hint)
    N = len(Ec)
    if len(E) == 0:
        return E, np.zeros((0, N)), 0
    # every state of the window is returned: which carrier its density is,
    # node by node, is decided in _occupy against the local mid-gap
    return E, np.sum(np.abs(psi) ** 2, axis=2), len(E)


_POOL = None
_PARALLEL_MIN_NODES = 250     # smaller regions (HEMT channels, single wells) stay sequential


def _worker_init():
    import os
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # noqa: BLE001 - optional
        pass


def _pool():
    """Persistent process pool for the in-plane k points, one BLAS thread per
    worker. Threads do not help: ARPACK's reverse communication holds the GIL
    (8 threads ran at 0.9x the sequential speed on the superlattice)."""
    global _POOL
    if _POOL is None:
        import os
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1)),
                                    initializer=_worker_init)
    return _POOL


@dataclass
class KPStates:
    """Result of one k.p solve on the quantum region [i0, i1)."""
    n: np.ndarray                      # electron density on the region [cm^-3]
    p: np.ndarray                      # hole density on the region [cm^-3]
    # k_t = 0 states (for display and optics): energies, |psi|^2 summed over
    # the 8 components [nm^-1], full spinors (n_states, N, 8) and the
    # weight on |S> (conduction character)
    E_e0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_e0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    spin_e0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 8)))
    E_h0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_h0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    spin_h0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 8)))
    char_h0: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # HH-like, LH-like, Z-like weight
    k_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    E_e_k: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (n_k, n_sub) lowest CB subbands
    E_h_k: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (n_k, n_sub) highest VB subbands
    # k-sampled states behind n and p: (kind 'e'|'h', weight [nm^-2], E (s,), rho (s, N) [nm^-1])
    samples: list = field(default_factory=list)
    w_nm: Optional[np.ndarray] = None
    T: float = 300.0
    mid: Optional[np.ndarray] = None   # local mid-gap [eV] (charge-neutrality level, see _occupy)
    # arguments of the k = 0 display solve, run on first use (KPSolver.display):
    # only the converged solve's states are shown, and in a broken gap that
    # solve cost 2 s per self-consistent iteration (AlN/InN/AlN, 12 iterations)
    display_args: Optional[tuple] = None

    def densities(self, Efn, Efp, local_qfl: bool = False):
        """(n, p) [cm^-3] of the stored states for new quasi-Fermi levels
        (frozen wavefunctions: the bias path's predictor step)."""
        return _occupy(self.samples, self.w_nm, Efn, Efp, self.T, local_qfl, self.mid)


class KPSolver:
    """k.p problem of one quantum region of one grid: the k-dependent
    matrices are assembled once; each solve adds the band-edge potential."""

    def __init__(self, par: KPParams, dx_m, i0: int, i1: int, N_grid: int):
        self.par = par
        self.i0, self.i1 = i0, i1
        n = i1 - i0
        _h1, _h2, w_m, edges_m = node_spacings(dx_m, N_grid)
        self.w_nm = np.asarray(w_m)[i0:i1] * 1e9
        self.h_nm = (np.asarray(edges_m)[i0:i1 - 1] * 1e9) if n > 1 else np.ones(0)
        zb = par.crystal == 'zincblende'
        D = _coefficients(par, 1.0, 0.0)
        if zb:
            # axial average: phi = 0 and pi/4, the latter rotated back onto x
            R = _rotation(np.pi / 4)
            D45 = _coefficients(par, np.sqrt(0.5), np.sqrt(0.5))
            D = 0.5 * (D + np.matmul(np.matmul(R.conj().T[None, None, None], D45), R[None, None, None]))
        # One block, counted twice: the two mirror blocks are Kramers partners,
        # degenerate except for wurtzite's k-linear A7 term, which splits them
        # by <= 0.2 meV and changes the AlGaN/GaN HEMT electron sheet density
        # by 0.014 % (local 0.03 %). It has no effect at k_t = 0, so it is
        # dropped from the in-plane integral only.
        D = D.copy()
        D[1] = _coefficients_no_a7(par, D[1])
        self._Dp, self._U = _mirror_blocks(D)
        part = self._block(slice(0, 4))
        part.weight = 2.0
        self.parts = [part]
        self._part2 = None
        self._pattern = _sparse_pattern(n, 4)
        self._kmax_rung = {}        # (kind, part) -> k_max ladder rung of the previous solve

    def _block(self, b: slice) -> _Assembled:
        """Assembled real 4-component block b of the mirror split."""
        n = len(self.par.Ec)
        part = _assemble(np.ascontiguousarray(self._Dp[:, :, :, b, b]), self.h_nm, self.w_nm)
        Ub = self._U[:, b]
        Ps = Ub[list(_S_IDX)].conj().T @ Ub[list(_S_IDX)]
        mats = ([part.Bd[a_][j] for a_ in range(3) for j in range(n)]
                + [part.Bo[a_][j] for a_ in range(3) for j in range(n - 1)] + [Ps])
        P = _real_phases(mats)
        if P is not None:
            part.Bd = np.real(np.matmul(np.matmul(P.conj().T[None, None], part.Bd), P[None, None]))
            part.Bo = np.real(np.matmul(np.matmul(P.conj().T[None, None], part.Bo), P[None, None]))
            Ub = Ub @ P
        part.U = Ub
        return part

    def block2(self) -> int:
        """Index in self.parts of the second mirror block (built on first
        use; only the optics need it -- p_y couples the two blocks). Its
        k_t != 0 levels differ from the first block's by the Rashba
        splitting only (0.001-0.004 meV in the AlGaN/GaN HEMT, 0.002 % in
        the sheet density), so the densities count the first block twice."""
        if self._part2 is None:
            part = self._block(slice(4, 8))
            part.weight = 0.0
            self.parts.append(part)
            self._part2 = len(self.parts) - 1
        return self._part2

    # ------------------------------------------------------------------
    def _matrix(self, part: _Assembled, k: float, Ec: np.ndarray, Ev: np.ndarray):
        return _block_matrix(part.Bd, part.Bo, part.U, self._pattern, k, Ec, Ev)

    def eigen(self, k: float, Ec: np.ndarray, Ev: np.ndarray, lo: float, hi: float,
              part: int = 0, side: str = 'up', n_hint: Optional[int] = None):
        """All eigenpairs with lo <= E <= hi at in-plane k [nm^-1]:
        energies (ascending) and spinors psi (n, N, 8) normalised to
        sum_j w_j |psi_j|^2 = 1. side='up' anchors the shift at lo (a
        conduction window above the gap), 'down' at hi (a valence window).
        n_hint: expected number of pairs in the window (e.g. the count at
        k = 0). With few states one shift-invert request of that size
        (+ margin) is used; otherwise (or when unknown) the window is
        spectrum-sliced (_sliced_eigsh). ARPACK's cost grows steeply with
        the number of pairs requested: on the p-type superlattice of
        Krishna et al. (190 hole states per k) one request took 3.8 s, slices
        of 30 0.70 s for the identical set."""
        pt = self.parts[part]
        return _eigen_core(pt.Bd, pt.Bo, pt.U, pt.inv_sqrt_w, self._pattern, k, Ec, Ev, lo, hi, side, n_hint)

    def _states_at(self, ks, Ec, Ev, lo, hi, ip, side, n_hint, kind):
        """[(E, rho)] at every k in ks: in the process pool for large
        regions, sequentially otherwise (the same function either way)."""
        pt = self.parts[ip]
        args = [(pt.Bd, pt.Bo, pt.U, pt.inv_sqrt_w, self._pattern, self.w_nm, float(k), Ec, Ev, lo, hi,
                 side, n_hint, kind) for k in ks]
        res = None
        if len(Ec) >= _PARALLEL_MIN_NODES and len(ks) > 1:
            try:
                res = list(_pool().map(_k_states, args))
            except Exception:  # noqa: BLE001 - no process support here: identical results sequentially
                res = None
        if res is None:
            res = [_k_states(x) for x in args]
        self._n_window = [r[2] for r in res]
        return [(r[0], r[1]) for r in res]

    def s_weight(self, psi: np.ndarray) -> np.ndarray:
        """Weight of each state on |S| (conduction character)."""
        return np.einsum('j,sjc->s', self.w_nm, np.abs(psi[:, :, list(_S_IDX)]) ** 2)

    def character(self, psi: np.ndarray) -> np.ndarray:
        """(n, 3): weight on (X, Y) [heavy/light-hole-like], on Z
        [crystal-field / light-hole along z], on S."""
        a2 = np.abs(psi) ** 2 * self.w_nm[None, :, None]
        xy = a2[:, :, [1, 2, 5, 6]].sum(axis=(1, 2))
        z = a2[:, :, [3, 7]].sum(axis=(1, 2))
        s = a2[:, :, [0, 4]].sum(axis=(1, 2))
        return np.stack([xy, z, s], axis=1)

    # ------------------------------------------------------------------
    def solve(self, Ec: np.ndarray, Ev: np.ndarray, Efn, Efp, T: float,
              local_qfl: bool = False, n_display: int = 8,
              skip: tuple = ()) -> KPStates:
        """Densities and k = 0 states for band edges Ec, Ev (grid Ec0 - phi,
        Ev0 - phi on the region) and quasi-Fermi levels Efn, Efp (scalars or
        region arrays)."""
        kT = kB * T / _q
        N = len(Ec)
        Efn = np.broadcast_to(np.asarray(Efn, float), (N,))
        Efp = np.broadcast_to(np.asarray(Efp, float), (N,))
        EcG = Ec + self.par.Ec
        # occupation window: _TAIL_KT kT past the quasi-Fermi level or past
        # the band edge, whichever is further in (non-degenerate carriers
        # occupy the band bottom with exp(-(E - Ef)/kT) relative weight; a
        # window ending at Ef + 10 kT cut through them -- confirmed 49-69 %
        # low with the quasi-Fermi level 0.2 eV inside the gap)
        lo_e = float(EcG.min()) - 0.05
        hi_h = float(Ev.max()) + 0.05
        if local_qfl:
            # Biased: the window covers every node's own band edge. The
            # current crosses every node, and a node whose local edge lies
            # outside the window gets ~no quantum density, i.e. a gamma at its
            # 1e-8 clip that blocks transport. Confirmed on an InGaN/GaN
            # p-i-n at 2.6 V: the band slope put the p-side 4 nm of the region
            # below a window set by the well's Ev, the hole gamma there sat at
            # 1e-8, the well was cut off from the p contact and the exact
            # drift-diffusion ramp stalled at 0.007 V.
            hi_e = float(np.max(np.maximum(Efn, EcG))) + _TAIL_KT * kT
            lo_h = float(np.min(np.minimum(Efp, Ev))) - _TAIL_KT * kT
        else:
            hi_e = max(float(Efn.max()), float(EcG.min())) + _TAIL_KT * kT
            lo_h = min(float(Efp.min()), float(Ev.max())) - _TAIL_KT * kT
        mid = 0.5 * (EcG + Ev)
        samples = []
        self._n0 = {}
        xg, wg = np.polynomial.legendre.leggauss(_N_K)
        windows = [('e', lo_e, hi_e, 'up'), ('h', lo_h, hi_h, 'down')]
        if lo_e <= hi_h:
            # Broken gap: Ec somewhere below Ev elsewhere, the two windows
            # overlap and a state can be conduction-like at one end of the
            # region and valence-like at the other. One window then holds
            # every state once; _occupy splits each one by position.
            windows = [('eh', min(lo_e, lo_h), max(hi_e, hi_h), 'up')]
            if 'e' in skip and 'h' in skip:
                windows = []
        for kind, lo, hi, side in windows:
            if hi <= lo or kind in skip:
                continue
            for ip in range(len(self.parts)):
                if self.parts[ip].weight == 0.0:
                    continue
                (E0, rho0), = self._states_at([0.0], Ec, Ev, lo, hi, ip, side, None, kind)
                occ0 = _pick(_occupy([(kind, 1.0, E0, rho0)], self.w_nm, Efn, Efp, T, local_qfl, mid, raw=True), kind)
                if occ0 <= 0.0:
                    continue
                # sized by every eigenpair in the window, not only the kept
                # ones: in a broken gap (AlN/InN/AlN) the electron window held
                # 67 states at k = 0, 4 of them conduction-like; a hint of 4
                # made every k point re-request 7, 14, 28, ... pairs
                hint = self._n_window[0]
                self._n0[kind] = hint
                if _TRACE:
                    import sys as _sys
                    print(f'    k.p {kind}: {hint} states at k=0, window [{lo:.3f}, {hi:.3f}] eV, N={len(Ec)}',
                          file=_sys.stderr, flush=True)
                    _t0 = _time.time()
                # k_max: the first k on the ladder 0.5 * 1.6^j where the occupation
                # left is below _OCC_TOL of the largest at k = 0 (relative: with
                # non-degenerate carriers the occupation is ~1e-4 even at k = 0, and an
                # absolute threshold cut the integral off far too early -- confirmed
                # 49 % low for holes with Efp in the gap). Evaluated 6 rungs at a time.
                # The previous solve's k_max (the potential changes little between
                # self-consistent iterations) is tried first, with the rung below it:
                # if it passes, only that pair is evaluated.
                ladder = [0.5 * 1.6 ** j for j in range(16)]
                k_max = ladder[-1]
                j_prev = self._kmax_rung.get((kind, ip))
                batches = [ladder[j:j + 6] for j in range(0, len(ladder), 6)]
                if j_prev is not None:
                    batches = [ladder[max(j_prev - 1, 0):j_prev + 1]] + batches
                for batch in batches:
                    found = None
                    for kk, (E, rho) in zip(batch, self._states_at(batch, Ec, Ev, lo, hi, ip, side, hint, kind)):
                        occ = _occupy([(kind, 1.0, E, rho)], self.w_nm, Efn, Efp, T, local_qfl, mid, raw=True)
                        if _pick(occ, kind) < _OCC_TOL * occ0:
                            found = kk
                            break
                    if found is not None:
                        if found == batch[0] and batch is batches[0] and j_prev is not None and j_prev > 0:
                            # the rung below the previous k_max passes too: keep stepping down
                            j = ladder.index(found)
                            while j > 0:
                                (E, rho), = self._states_at([ladder[j - 1]], Ec, Ev, lo, hi, ip, side, hint, kind)
                                occ = _occupy([(kind, 1.0, E, rho)], self.w_nm, Efn, Efp, T, local_qfl, mid, raw=True)
                                if _pick(occ, kind) >= _OCC_TOL * occ0:
                                    break
                                j -= 1
                            found = ladder[j]
                        k_max = found
                        self._kmax_rung[(kind, ip)] = ladder.index(found)
                        break
                if _TRACE:
                    print(f'    k.p {kind}: k_max {k_max:.3f} nm^-1 found in {_time.time() - _t0:.1f} s',
                          file=_sys.stderr, flush=True)
                    _t0 = _time.time()
                ks = 0.5 * k_max * (xg + 1.0)
                for k, w, (E, rho) in zip(ks, wg, self._states_at(ks, Ec, Ev, lo, hi, ip, side, hint, kind)):
                    if len(E):
                        samples.append((kind, 0.5 * k_max * w * k / (2.0 * np.pi) * self.parts[ip].weight, E, rho))
                if _TRACE:
                    print(f'    k.p {kind}: {len(ks)} k points in {_time.time() - _t0:.1f} s', file=_sys.stderr, flush=True)
        n, p = _occupy(samples, self.w_nm, Efn, Efp, T, local_qfl, mid)
        out = KPStates(n=n, p=p, samples=samples, w_nm=self.w_nm, T=T, mid=mid)

        out.display_args = (Ec, Ev, lo_e, hi_e, lo_h, hi_h, n_display, dict(self._n0))
        return out

    def display(self, out: KPStates) -> KPStates:
        """Fill the k = 0 display states of `out` (once)."""
        if out.display_args is not None:
            args, out.display_args = out.display_args, None
            self._display(out, *args)
        return out


    def _lowest(self, Ec, Ev, sigma: float, count: int, side: str):
        """The `count` k = 0 eigenpairs nearest sigma on the given side."""
        pt = self.parts[0]
        H = self._matrix(pt, 0.0, Ec, Ev)
        m = int(min(count, H.shape[0] - 2))
        try:
            E, v = eigsh(H, k=m, sigma=sigma, which='LM')
        except ArpackNoConvergence as exc:
            E, v = exc.eigenvalues, exc.eigenvectors
        keep = (E >= sigma) if side == 'up' else (E <= sigma)
        E, v = E[keep], v[:, keep]
        o = np.argsort(E) if side == 'up' else np.argsort(E)[::-1]
        E, v = E[o], v[:, o]
        psi4 = v.T.reshape(len(E), len(Ec), pt.U.shape[1]) * pt.inv_sqrt_w[None, :, None]
        return E, psi4 @ pt.U.T

    def _display(self, out, Ec, Ev, lo_e, hi_e, lo_h, hi_h, n_display, n0):
        """k = 0 levels for display / optics: the n_display lowest
        conduction and highest valence levels, each Kramers pair complete
        (one mirror block holds one partner of each pair; the other is its
        time-reversed image, built with T = i sigma_y K)."""
        n8 = self._matrix(self.parts[0], 0.0, Ec, Ev).shape[0]
        for side, sigma in (('up', lo_e), ('down', hi_h)):
            # Broken gap (Ec on one side of the region below Ev on the other):
            # the states nearest the conduction edge are valence states, so
            # the request grows until n_display of the wanted character are
            # in it. Confirmed on AlN/InN(15 nm)/AlN: all 20 states above
            # min Ec were InN valence states and no electron state was shown.
            # first request: every pair of the density window at k = 0 (the
            # other character included) plus margin, so it rarely has to grow
            count = min(n0.get('e' if side == 'up' else 'h', 0) + 2 * n_display + 4, n8 - 2)
            while True:
                E, psi = self._lowest(Ec, Ev, sigma, count, side)
                cb = self.s_weight(psi) > 0.5 if len(E) else np.zeros(0, bool)
                keep = cb if side == 'up' else ~cb
                if keep.sum() >= n_display or count >= n8 - 2:
                    break
                count = min(2 * count, n8 - 2)
            if not keep.any():
                continue
            E, psi = E[keep][:n_display], psi[keep][:n_display]
            partner = _time_reverse(psi)
            E2 = np.repeat(E, 2)
            psi2 = np.empty((2 * len(E),) + psi.shape[1:], complex)
            psi2[0::2], psi2[1::2] = psi, partner
            if side == 'up':
                out.E_e0, out.spin_e0 = E2, psi2
                out.rho_e0 = np.sum(np.abs(psi2) ** 2, axis=2)
            else:
                out.E_h0, out.spin_h0 = E2, psi2
                out.rho_h0 = np.sum(np.abs(psi2) ** 2, axis=2)
                out.char_h0 = self.character(psi2)
        return out


    # ------------------------------------------------------------------
    def momentum(self, C: np.ndarray, V: np.ndarray):
        """Interband momentum matrix elements (Mx, My, Mz) [eV nm] between
        conduction spinors C (nc, N, 8) and valence spinors V (nv, N, 8):
        M_a = sum_j w_j sum_spin [ c_S^* (i P_a) v_a + c_a^* (-i P_a) v_S ],
        P_x = P_y = P2, P_z = P1 -- the Kane coupling of the H1 block, i.e.
        <S|p_a|a> = (m0/hbar) i P_a. (The remote-band parts of the velocity
        operator are intraband-dominated and neglected.) p = (m0/hbar) M."""
        w = self.w_nm
        out = []
        for comp, P in ((1, self.par.P2), (2, self.par.P2), (3, self.par.P1)):
            wp = w * P
            m = np.zeros((C.shape[0], V.shape[0]), complex)
            for off in (0, 4):
                cS, ca = np.conj(C[:, :, off]), np.conj(C[:, :, off + comp])
                vS, va = V[:, :, off], V[:, :, off + comp]
                m += 1j * (cS * wp) @ va.T - 1j * (ca * wp) @ vS.T
            out.append(m)
        return out

    def gain(self, Ec: np.ndarray, Ev: np.ndarray, Efn: float, Efp: float, T: float,
             Lz_nm: float, n_r: float, linewidth_eV: float = 0.010, n_points: int = 400):
        """Material gain [cm^-1] for TE (in-plane polarisation, averaged over
        its angle to k_t) and TM (along z):

            g(hv) = pi q^2 hbar / (n_r eps0 c m0^2 hv) (1/Lz)
                    int d^2k/(2pi)^2 sum_{c,v} |e.p_cv(k)|^2 (f_c - f_v) L(E_c - E_v - hv)

        every conduction/valence pair of both mirror blocks at each k_t
        (spin explicit), quasi-Fermi occupations Efn/Efp, Lorentzian L of
        half-width linewidth_eV. Returns (hv [eV], g_TE, g_TM) or None."""
        from physics.constants import hbar, m0, eps0, c as c_light
        kT = kB * T / _q
        EcG = Ec + self.par.Ec
        lo_e = float(EcG.min()) - 0.05
        hi_e = max(Efn, float(EcG.min())) + _TAIL_KT * kT
        hi_h = float(Ev.max()) + 0.05
        lo_h = min(Efp, float(Ev.max())) - _TAIL_KT * kT
        ib = self.block2()
        parts = (0, ib)

        def states(k, ip, lo, hi, side, want_cb):
            E, psi = self.eigen(k, Ec, Ev, lo, hi, ip, side)
            if len(E) == 0:
                return E, psi
            cb = self.s_weight(psi) > 0.5
            keep = cb if want_cb else ~cb
            return E[keep], psi[keep]

        # k range: until both carriers' occupation at k_max is negligible
        def f(E, F):
            return 1.0 / (1.0 + np.exp(np.clip((E - F) / kT, -500.0, 500.0)))
        k_max = 0.5
        for _ in range(16):
            Ee, _p = states(k_max, 0, lo_e, hi_e, 'up', True)
            Eh, _p = states(k_max, 0, lo_h, hi_h, 'down', False)
            occ_e = float(f(Ee, Efn).max()) if len(Ee) else 0.0
            occ_h = float((1.0 - f(Eh, Efp)).max()) if len(Eh) else 0.0
            if occ_e < 1e-4 and occ_h < 1e-4:
                break
            k_max *= 1.6
        trans = []        # (E_cv, TE, TM, fc - fv, weight)
        xg, wg = np.polynomial.legendre.leggauss(_N_K)
        for x, wq in zip(xg, wg):
            k = 0.5 * k_max * (x + 1.0)
            Cs, Ces, Vs, Ves = [], [], [], []
            for ip in parts:
                E, P = states(k, ip, lo_e, hi_e, 'up', True)
                Cs.append(P); Ces.append(E)
                E, P = states(k, ip, lo_h, hi_h, 'down', False)
                Vs.append(P); Ves.append(E)
            C, Ecs = np.concatenate(Cs), np.concatenate(Ces)
            V, Evs = np.concatenate(Vs), np.concatenate(Ves)
            if len(Ecs) == 0 or len(Evs) == 0:
                continue
            Mx, My, Mz = self.momentum(C, V)
            te = 0.5 * (np.abs(Mx) ** 2 + np.abs(My) ** 2)
            tm = np.abs(Mz) ** 2
            occ = f(Ecs, Efn)[:, None] - f(Evs, Efp)[None, :]
            Ecv = Ecs[:, None] - Evs[None, :]
            wk = 0.5 * k_max * wq * k / (2.0 * np.pi)          # nm^-2
            trans.append((Ecv.ravel(), te.ravel(), tm.ravel(), occ.ravel(), np.full(Ecv.size, wk)))
        if not trans:
            return None
        Ecv, te, tm, occ, wk = (np.concatenate(z) for z in zip(*trans))
        lo = max(0.01, float(Ecv.min()) - 15 * linewidth_eV)
        hi = float(np.percentile(Ecv, 50)) + 15 * linewidth_eV
        hv = np.linspace(lo, hi, n_points)
        L = (linewidth_eV / np.pi) / ((hv[:, None] - Ecv[None, :]) ** 2 + linewidth_eV ** 2)   # eV^-1
        pref = np.pi * _q ** 2 * hbar / (n_r * eps0 * c_light * m0 ** 2 * hv * _q)            # / J
        mom2 = (m0 / hbar) ** 2 * (_q * 1e-9) ** 2          # (eV nm)^2 -> (kg m/s)^2
        common = wk * occ * 1e18 / (Lz_nm * 1e-9)           # m^-3
        g_te = pref * mom2 * ((L / _q) @ (common * te))     # m^-1
        g_tm = pref * mom2 * ((L / _q) @ (common * tm))
        return hv, g_te * 1e-2, g_tm * 1e-2

def _time_reverse(psi: np.ndarray) -> np.ndarray:
    """Kramers partner T psi = i sigma_y K psi of spinors (n, N, 8) in the
    (S, X, Y, Z) x (up, down) basis: (up, down) -> (down*, -up*)."""
    out = np.empty_like(psi)
    out[..., :4] = np.conj(psi[..., 4:])
    out[..., 4:] = -np.conj(psi[..., :4])
    return out

def _pick(occ, kind):
    """The occupation that matters for a window of this kind."""
    return occ[0] if kind == 'e' else occ[1] if kind == 'h' else max(occ)


def _occupy(samples, w_nm, Efn, Efp, T, local_qfl, mid, raw=False):
    """Electron and hole densities [cm^-3] of k-sampled states. raw=True
    returns instead the largest occupation found (for the k_max test).

    Carrier assignment (Andlauer & Vogl, PRB 79, 045307 (2009)): at each
    node a state's density is electron density where its energy lies above
    the local charge-neutrality level, taken as the local mid-gap
    (Ec + Ev) / 2, and hole density where it lies below, switched smoothly
    over _CNL_KT kT. In an ordinary well every conduction state lies above
    and every valence state below the mid-gap of each node it occupies, and
    this is the per-state assignment it replaces. In a broken gap it is not:
    labelling whole states by their |S> weight put the valence-like end of
    mixed states into n and the conduction-like end into p (confirmed on
    N-polar AlN/InN(15 nm)/AlN: "holes" of 3e14 cm^-3 where Ec lay below Ef,
    gamma at its 1e8 clip, the self-consistent loop drifting for 126 k.p
    solves).

    Occupation: with local_qfl each state is filled with the local
    quasi-Fermi level at every z (nextnano++'s treatment under current
    flow), otherwise with its probability-weighted average (as
    the single-band subband density it replaced)."""
    kT = kB * T / _q
    N = len(w_nm)
    Efn = np.broadcast_to(np.asarray(Efn, float), (N,))
    Efp = np.broadcast_to(np.asarray(Efp, float), (N,))
    n = np.zeros(N)
    p = np.zeros(N)
    occ_max = [0.0, 0.0]
    for _kind, weight, E, rho in samples:
        if len(E) == 0:
            continue
        we = 0.5 * (1.0 + np.tanh((E[:, None] - mid[None, :]) / (2.0 * _CNL_KT * kT)))
        rho_e, rho_h = we * rho, (1.0 - we) * rho
        if local_qfl:
            fe = 1.0 / (1.0 + np.exp(np.clip((E[:, None] - Efn[None, :]) / kT, -500.0, 500.0)))
            fh = 1.0 / (1.0 + np.exp(np.clip((Efp[None, :] - E[:, None]) / kT, -500.0, 500.0)))
            ce, ch = np.sum(fe * rho_e, axis=0), np.sum(fh * rho_h, axis=0)
            oe = float(np.max(fe * (rho_e @ w_nm)[:, None])) if raw else 0.0
            oh = float(np.max(fh * (rho_h @ w_nm)[:, None])) if raw else 0.0
        else:
            Fn = (rho * w_nm[None, :]) @ Efn
            Fp = (rho * w_nm[None, :]) @ Efp
            fe = 1.0 / (1.0 + np.exp(np.clip((E - Fn) / kT, -500.0, 500.0)))
            fh = 1.0 / (1.0 + np.exp(np.clip((Fp - E) / kT, -500.0, 500.0)))
            ce, ch = fe @ rho_e, fh @ rho_h
            oe = float(np.max(fe * (rho_e @ w_nm))) if raw else 0.0
            oh = float(np.max(fh * (rho_h @ w_nm))) if raw else 0.0
        n += weight * ce
        p += weight * ch
        occ_max[0] = max(occ_max[0], oe)
        occ_max[1] = max(occ_max[1], oh)
    if raw:
        return occ_max[0], occ_max[1]
    return n * 1e21, p * 1e21       # nm^-3 -> cm^-3

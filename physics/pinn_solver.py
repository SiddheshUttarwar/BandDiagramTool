"""
Physics-Informed Neural Network (PINN) solver for 1D semiconductor
drift-diffusion equations.

Instead of discretizing PDEs on a grid and inverting Jacobian matrices,
this module trains a neural network whose outputs inherently satisfy
the semiconductor Poisson + Continuity equations.

Architecture
------------
  Input:  x_norm ∈ [0, 1]  (normalised spatial coordinate)
  Output: [phi, Efn, Efp]  in physical units [V]

The network is trained by minimising a physics loss:
  L = w1*L_poisson + w2*L_cont_n + w3*L_cont_p

All spatial derivatives are computed exactly via torch.autograd.grad
(no finite-difference discretization errors).

Normalisation (De Mari scaling)
-------------------------------
To handle the extreme multi-scale nature of semiconductor physics
(carrier densities spanning 10^4 to 10^19 cm^-3), we use:
  - Potential scale: V_T = kT/q  (~26 mV at 300K)
  - Length scale:    L_D = sqrt(eps * V_T / (q * C_ref))  (Debye length)
  - Density scale:   C_ref = max(ND, NA)  (peak doping)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

from physics.constants import q as Q_E, kB, eps0

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Compute device selection (CPU / NVIDIA CUDA / Intel XPU)
# ═══════════════════════════════════════════════════════════════════════

def available_devices() -> list:
    """
    Compute backends this machine can actually use right now, as
    (value, label) pairs for a GUI dropdown -- 'cpu' is always present;
    'cuda'/'xpu' are only listed if torch reports them truly available
    (not just present in the API), so a CPU-only torch build or missing
    GPU drivers never show an option that would just fail at train time.
    """
    options = [('cpu', 'CPU')]
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            options.append(('cuda', f'CUDA ({name})'))
    except Exception:
        pass
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            name = torch.xpu.get_device_name(0)
            options.append(('xpu', f'Intel XPU ({name})'))
    except Exception:
        pass
    return options


def resolve_torch_device(preference: str = 'cpu',
                          log_fn: Optional[Callable[[str], None]] = None) -> torch.device:
    """
    Map a GUI device choice ('cpu' | 'cuda' | 'xpu') to an actual
    torch.device, falling back to CPU with a logged reason if the
    requested backend isn't really available (wrong torch build, no
    drivers, no hardware) rather than raising -- training should still
    run, just not accelerated, if the preference can't be honoured.
    """
    _log = log_fn if log_fn is not None else print
    preference = (preference or 'cpu').lower()

    if preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        _log("  Requested CUDA but it isn't available on this torch install "
             "(needs an NVIDIA GPU + CUDA-enabled torch build) -- using CPU instead.")
        return torch.device('cpu')

    if preference == 'xpu':
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')
        _log("  Requested Intel XPU but it isn't available on this torch install "
             "(needs the XPU-enabled torch build + Intel GPU drivers) -- using CPU instead.")
        return torch.device('cpu')

    return torch.device('cpu')


# ═══════════════════════════════════════════════════════════════════════
# De Mari Scaling (Normalisation)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeVariScaling:
    """De Mari normalisation constants for semiconductor equations."""
    V_T: float       # Thermal voltage kT/q [V]
    L_D: float       # Debye length [m]
    C_ref: float     # Reference concentration [cm^-3]
    C_ref_m3: float  # Reference concentration [m^-3]
    L_device: float  # Total device length [m]
    mu_n: float      # Electron mobility [cm^2/V/s]
    mu_p: float      # Hole mobility [cm^2/V/s]
    D_n: float       # Electron diffusivity [cm^2/s]
    D_p: float       # Hole diffusivity [cm^2/s]

    @staticmethod
    def from_device(eps_r_avg: float, T: float, C_ref_cm3: float,
                    L_device_m: float, mu_n: float, mu_p: float) -> 'DeVariScaling':
        V_T = kB * T / Q_E
        C_ref_m3 = C_ref_cm3 * 1e6
        L_D = np.sqrt(eps0 * eps_r_avg * V_T / (Q_E * C_ref_m3))
        D_n = mu_n * V_T   # Einstein relation: D = mu * kT/q
        D_p = mu_p * V_T
        return DeVariScaling(
            V_T=V_T, L_D=L_D, C_ref=C_ref_cm3, C_ref_m3=C_ref_m3,
            L_device=L_device_m,
            mu_n=mu_n, mu_p=mu_p, D_n=D_n, D_p=D_p,
        )


# ═══════════════════════════════════════════════════════════════════════
# Neural Network Architecture
# ═══════════════════════════════════════════════════════════════════════

class PINNNetwork(nn.Module):
    """
    Multi-Layer Perceptron for the semiconductor PINN.

    Input:  x_norm ∈ [0, 1]  (1D)
    Output: [phi, Efn, Efp]  in physical units [V]

    Uses Tanh activations (C^∞ smooth, required for second derivatives
    in the Poisson equation via autograd).

    Hard boundary enforcement: the network output is multiplied by
    x*(1-x) and the boundary values are added, so the BCs are
    exactly satisfied for any network weights.
    """

    def __init__(self, n_hidden: int = 5, n_neurons: int = 64):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, n_neurons))
        layers.append(nn.Tanh())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_neurons, 3))  # [phi, Efn, Efp]
        self.net = nn.Sequential(*layers)

        # Boundary values (set before training)
        self.phi_left = 0.0
        self.phi_right = 0.0
        self.Efn_left = 0.0
        self.Efn_right = 0.0
        self.Efp_left = 0.0
        self.Efp_right = 0.0

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for stable training."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with hard boundary enforcement.

        x: shape (N, 1), values in [0, 1]
        Returns: phi, Efn, Efp each of shape (N, 1)
        """
        raw = self.net(x)  # (N, 3)

        # Hard BC enforcement: output = BC_left + x*(BC_right - BC_left) + x*(1-x)*NN
        # This guarantees exact BC satisfaction for any network weights.
        phi_raw = raw[:, 0:1]
        efn_raw = raw[:, 1:2]
        efp_raw = raw[:, 2:3]

        window = x * (1.0 - x)  # zero at x=0 and x=1

        phi = self.phi_left + x * (self.phi_right - self.phi_left) + window * phi_raw
        Efn = self.Efn_left + x * (self.Efn_right - self.Efn_left) + window * efn_raw
        Efp = self.Efp_left + x * (self.Efp_right - self.Efp_left) + window * efp_raw

        return phi, Efn, Efp


# ═══════════════════════════════════════════════════════════════════════
# Material Property Interpolators (continuous functions of x for autograd)
# ═══════════════════════════════════════════════════════════════════════

class MaterialInterpolator:
    """
    Convert discrete grid arrays into smooth PyTorch-compatible functions
    of the normalised coordinate x ∈ [0, 1].

    Uses simple linear interpolation (torch-compatible, differentiable
    w.r.t. x via straight-through estimator on the floor index).
    """

    def __init__(self, x_norm_np: np.ndarray, device: torch.device = torch.device('cpu')):
        self.x_norm = torch.tensor(x_norm_np, dtype=torch.float64, device=device)
        self._profiles = {}
        self.device = device
        self.N = len(x_norm_np)

    def register(self, name: str, values: np.ndarray):
        """Register a material profile array."""
        self._profiles[name] = torch.tensor(
            values, dtype=torch.float64, device=self.device
        )

    def __call__(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate profile `name` at normalised positions x ∈ [0, 1].

        Uses differentiable linear interpolation via detached floor index
        (straight-through estimator: gradients pass through the fractional
        part only, which is correct for piecewise-linear interpolation).
        """
        profile = self._profiles[name]
        N = self.N

        # Map x to fractional grid index
        idx_float = x.squeeze(-1) * (N - 1)
        idx_floor = torch.clamp(idx_float.detach().long(), 0, N - 2)
        frac = idx_float - idx_floor.float()

        # Linear interpolation (differentiable w.r.t. frac → w.r.t. x)
        v0 = profile[idx_floor]
        v1 = profile[torch.clamp(idx_floor + 1, max=N - 1)]
        return (v0 + frac * (v1 - v0)).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# Collocation point sampling
# ═══════════════════════════════════════════════════════════════════════

def _build_collocation_points(
    x_norm_np: np.ndarray, ND: np.ndarray, NA: np.ndarray, x_Al: np.ndarray,
    n_colloc: int, L_D_norm: float, uniform_frac: float = 0.3,
) -> np.ndarray:
    """
    Non-uniform collocation points, clustered near doping and composition
    transitions (p-n junctions, heterojunction interfaces) instead of
    uniformly spaced across the whole device.

    Why: a depletion region is typically only nm-to-tens-of-nm wide inside
    a device that can be hundreds of nm long. Tanh-MLP PINNs have a strong
    spectral bias toward smooth, low-frequency solutions -- with uniform
    collocation points, only a handful ever land inside the depletion
    region, giving the network almost no training signal to represent the
    real, sharp junction physics there. It instead converges to a smooth,
    low-loss-looking but physically wrong solution: no flat neutral-bulk
    plateau, quasi-Fermi levels barely splitting -- confirmed empirically
    (see examples/pinn_test.py's first result, physics/self_consistent.py's
    already-validated classical solve on the same device shows the full
    applied bias as quasi-Fermi separation, not the ~0.1-0.2V the PINN
    found). Same principle as the non-uniform grid refinement already used
    for the classical finite-volume solver (devices/grid_builder.py's
    per-layer dx_nm / auto-refinement), applied here to collocation
    sampling instead of a discretization grid.

    Density = uniform_frac (baseline coverage, so nothing is ever
    completely unsampled) + (1-uniform_frac) * sum of Gaussians centered at
    each detected transition, width L_D_norm (the Debye length in
    normalised [0,1] coordinates -- the natural physical length scale for
    how wide a depletion/interface transition actually is, already
    computed for the De Mari scaling elsewhere in this module, not an
    arbitrary tuning constant).

    Transitions are detected from the *net* ionized doping (ND-NA, so a
    same-type doping step, e.g. n+ -> n, still counts) and from x_Al
    (heterojunction composition steps), via where the finite-difference
    gradient magnitude is a large fraction of its own peak -- catches every
    interface in a multi-layer device (MQW barriers/wells, EBL, etc.), not
    just a single p-n junction.
    """
    N = len(x_norm_np)
    net_doping = ND - NA
    d_doping = np.abs(np.gradient(net_doping, x_norm_np))
    d_xal = np.abs(np.gradient(x_Al, x_norm_np))

    def _peaks(d: np.ndarray, rel_thresh: float = 0.1) -> np.ndarray:
        peak = d.max()
        if peak <= 0:
            return np.array([])
        return x_norm_np[d > rel_thresh * peak]

    centers = np.concatenate([_peaks(d_doping), _peaks(d_xal)])
    if len(centers) == 0:
        # No detected transitions (e.g. a uniform slab) -- fall back to
        # plain uniform sampling, nothing to cluster around.
        return np.linspace(0.0, 1.0, n_colloc)

    # Density on a fine reference grid, then sample n_colloc points from it
    # via inverse-CDF (standard technique for sampling an arbitrary 1D
    # density without rejection sampling's wasted evaluations).
    x_ref = np.linspace(0.0, 1.0, 5000)
    density = np.full_like(x_ref, uniform_frac)
    sigma = max(L_D_norm, 1.0 / N)   # floor so it's never narrower than one grid cell
    for c in centers:
        density += (1.0 - uniform_frac) * np.exp(-0.5 * ((x_ref - c) / sigma) ** 2)
    density /= np.sum(density) * (x_ref[1] - x_ref[0])

    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    u = np.linspace(0.0, 1.0, n_colloc)
    x_colloc_np = np.interp(u, cdf, x_ref)
    x_colloc_np[0], x_colloc_np[-1] = 0.0, 1.0
    return np.sort(x_colloc_np)


# ═══════════════════════════════════════════════════════════════════════
# Autograd derivative helpers
# ═══════════════════════════════════════════════════════════════════════

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx using autograd. Both y and x must require grad."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


# ═══════════════════════════════════════════════════════════════════════
# Physics Loss Functions
# ═══════════════════════════════════════════════════════════════════════

class SemiconductorPINNLoss:
    """
    Computes the physics-informed loss for the 1D semiconductor equations.

    All three PDEs (Poisson, electron continuity, hole continuity) are
    evaluated in physical units and then normalised by appropriate scales
    to make all loss terms O(1).

    Equations (physical units):
    ─────────────────────────────────────────
    Poisson:
        d/dx [ε_r * ε_0 * dφ/dx] = -q*(p - n + N_D - N_A)*1e6 - ρ_pol

    Electron continuity (steady-state, quasi-Fermi formulation):
        d/dx [μ_n * n * dE_fn/dx] = R    (in cm^-3/s after unit adjustment)

    Hole continuity (steady-state, quasi-Fermi formulation):
        -d/dx [μ_p * p * dE_fp/dx] = R

    Carrier densities (Boltzmann approximation):
        n = N_c * exp((E_fn - E_c) / kT)   where E_c = E_c0 - φ
        p = N_v * exp((E_v - E_fp) / kT)   where E_v = E_v0 - φ
    """

    def __init__(self, scaling: DeVariScaling, mat: MaterialInterpolator,
                 w_poisson: float = 1.0,
                 w_cont_n: float = 1.0,
                 w_cont_p: float = 1.0):
        self.s = scaling
        self.mat = mat
        self.w_poisson = w_poisson
        self.w_cont_n = w_cont_n
        self.w_cont_p = w_cont_p

        # Recombination parameters (III-Nitride)
        self.tau_n = 1e-9   # SRH electron lifetime [s]
        self.tau_p = 1e-9   # SRH hole lifetime [s]
        self.B_rad = 1e-11  # Radiative coefficient [cm^3/s]
        self.C_aug = 1e-30  # Auger coefficient [cm^6/s]

    def carrier_densities(self, phi: torch.Tensor, Efn: torch.Tensor,
                          Efp: torch.Tensor, x: torch.Tensor):
        """
        Compute n, p from phi, Efn, Efp using Boltzmann statistics.

        All inputs are in PHYSICAL units (V for potentials).
        Returns n, p in [cm^-3], Ec, Ev in [eV].
        """
        Ec0 = self.mat('Ec0', x)   # [eV]
        Ev0 = self.mat('Ev0', x)   # [eV]
        Nc = self.mat('Nc', x)     # [cm^-3]
        Nv = self.mat('Nv', x)     # [cm^-3]

        Ec = Ec0 - phi   # [eV]
        Ev = Ev0 - phi   # [eV]

        V_T = self.s.V_T

        # Boltzmann approximation (smooth, differentiable everywhere)
        eta_n = torch.clamp((Efn - Ec) / V_T, min=-80.0, max=80.0)
        eta_p = torch.clamp((Ev - Efp) / V_T, min=-80.0, max=80.0)

        n = Nc * torch.exp(eta_n)  # [cm^-3]
        p = Nv * torch.exp(eta_p)  # [cm^-3]

        return n, p, Ec, Ev

    def recombination(self, n: torch.Tensor, p: torch.Tensor,
                      x: torch.Tensor) -> torch.Tensor:
        """Total recombination rate R [cm^-3/s]."""
        Ec0 = self.mat('Ec0', x)
        Ev0 = self.mat('Ev0', x)
        Nc = self.mat('Nc', x)
        Nv = self.mat('Nv', x)

        Eg = Ec0 - Ev0
        ni = torch.sqrt(Nc * Nv) * torch.exp(-Eg / (2.0 * self.s.V_T))

        np_prod = n * p
        ni2 = ni * ni

        R_srh = (np_prod - ni2) / (self.tau_p * (n + ni) + self.tau_n * (p + ni) + 1e-30)
        R_rad = self.B_rad * (np_prod - ni2)
        R_aug = self.C_aug * (n + p) * (np_prod - ni2)

        return R_srh + R_rad + R_aug

    def compute_loss(self, model: PINNNetwork, x_colloc: torch.Tensor):
        """
        Compute the total physics-informed loss.

        x_colloc: collocation points, shape (N_c, 1), values in [0, 1],
                  with requires_grad=True.

        Returns: total_loss, loss_dict
        """
        s = self.s
        L = s.L_device  # [m]

        # ── Network forward pass ──
        phi_V, Efn_V, Efp_V = model(x_colloc)

        # ── Carrier densities ──
        n, p, Ec, Ev = self.carrier_densities(phi_V, Efn_V, Efp_V, x_colloc)

        # ── Material properties ──
        eps_r = self.mat('eps_r', x_colloc)
        ND = self.mat('ND', x_colloc)
        NA = self.mat('NA', x_colloc)
        pol_rho = self.mat('pol_rho', x_colloc)

        # ══════════════════════════════════════════════════════════════
        # 1. POISSON EQUATION LOSS
        # ══════════════════════════════════════════════════════════════
        # d/dx[ε_r * ε_0 * dφ/dx] = -q*(p - n + ND - NA)*1e6 - ρ_pol
        #
        # x_colloc is in [0,1], x_phys = L * x_norm
        # dφ/dx_phys = (1/L) * dφ/dx_norm
        # d²φ/dx_phys² = (1/L²) * d²φ/dx_norm²
        # (assuming ε_r varies slowly: d/dx[ε_r dφ/dx] ≈ ε_r d²φ/dx²)

        dphi = _grad(phi_V, x_colloc)
        d2phi = _grad(dphi, x_colloc)

        charge = Q_E * (p - n + ND - NA) * 1e6 + pol_rho  # [C/m³]
        poisson_res = eps_r * eps0 * d2phi / (L * L) + charge

        rho_scale = Q_E * s.C_ref * 1e6
        poisson_loss = torch.mean((poisson_res / rho_scale) ** 2)

        # ══════════════════════════════════════════════════════════════
        # 2. ELECTRON CONTINUITY LOSS
        # ══════════════════════════════════════════════════════════════
        # Quasi-Fermi formulation:
        #   J_n = μ_n * n * dE_fn/dx  [A/cm² if μ in cm²/Vs, n in cm⁻³]
        #   (1/q) * dJ_n/dx = R  =>  d/dx[μ_n * n * dE_fn/dx] = R  [cm⁻³s⁻¹]
        # With x in [0,1]:  d/dx_phys = (1/L) d/dx_norm
        # => (μ_n / L²) * d/dx_norm[n * dE_fn/dx_norm] = R
        # We need d/dx_norm of (n * dE_fn/dx_norm):
        # = dn/dx_norm * dE_fn/dx_norm + n * d²E_fn/dx_norm²
        # But L is in [m] and μ_n in [cm²/Vs], so L² needs *1e4 to get cm².

        R = self.recombination(n, p, x_colloc)

        dn = _grad(n, x_colloc)
        dEfn = _grad(Efn_V, x_colloc)
        d2Efn = _grad(dEfn, x_colloc)

        # μ_n [cm²/Vs], L [m], L_cm = L*100 [cm], L_cm² = L²*1e4
        # (μ_n / L_cm²) * (...) = R
        L_cm2 = L * L * 1e4
        cont_n_res = (s.mu_n / L_cm2) * (dn * dEfn + n * d2Efn) - R

        R_scale = max(s.C_ref / self.tau_n, 1.0)
        cont_n_loss = torch.mean((cont_n_res / R_scale) ** 2)

        # ══════════════════════════════════════════════════════════════
        # 3. HOLE CONTINUITY LOSS
        # ══════════════════════════════════════════════════════════════
        # -d/dx[μ_p * p * dE_fp/dx] = R
        # => -(μ_p / L_cm²) * (dp/dx * dEfp/dx + p * d²Efp/dx²) = R

        dp = _grad(p, x_colloc)
        dEfp = _grad(Efp_V, x_colloc)
        d2Efp = _grad(dEfp, x_colloc)

        cont_p_res = -(s.mu_p / L_cm2) * (dp * dEfp + p * d2Efp) - R

        cont_p_loss = torch.mean((cont_p_res / R_scale) ** 2)

        # ══════════════════════════════════════════════════════════════
        # Total loss
        # ══════════════════════════════════════════════════════════════
        total_loss = (self.w_poisson * poisson_loss
                      + self.w_cont_n * cont_n_loss
                      + self.w_cont_p * cont_p_loss)

        loss_dict = {
            'total': total_loss.item(),
            'poisson': poisson_loss.item(),
            'cont_n': cont_n_loss.item(),
            'cont_p': cont_p_loss.item(),
        }
        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════════════
# PINN Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PINNResult:
    """Result of PINN training."""
    phi: np.ndarray       # Electrostatic potential [V]
    Efn: np.ndarray       # Electron quasi-Fermi level [V]
    Efp: np.ndarray       # Hole quasi-Fermi level [V]
    n: np.ndarray         # Electron density [cm^-3]
    p: np.ndarray         # Hole density [cm^-3]
    Ec: np.ndarray        # Conduction band edge [eV]
    Ev: np.ndarray        # Valence band edge [eV]
    x_nm: np.ndarray      # Grid positions [nm]
    loss_history: list     # Training loss per epoch
    final_loss: float
    n_epochs: int
    training_time_s: float


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def solve_pinn(
    grid,  # GridData
    V_applied: float = 0.0,
    n_colloc: int = 500,
    n_hidden: int = 5,
    n_neurons: int = 64,
    n_epochs_adam: int = 5000,
    n_epochs_lbfgs: int = 2000,
    lr_adam: float = 1e-3,
    lr_lbfgs: float = 1.0,
    w_poisson: float = 1.0,
    w_cont_n: float = 0.1,
    w_cont_p: float = 0.1,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> PINNResult:
    """
    Solve the 1D semiconductor equations using a Physics-Informed Neural Network.

    Parameters
    ----------
    grid : GridData
        Device grid with all material profiles.
    V_applied : float
        Applied voltage [V] (positive = forward bias).
    n_colloc : int
        Number of collocation points for PDE evaluation.
    n_hidden : int
        Number of hidden layers in the MLP.
    n_neurons : int
        Number of neurons per hidden layer.
    n_epochs_adam : int
        Number of Adam optimiser epochs (phase 1).
    n_epochs_lbfgs : int
        Number of L-BFGS optimiser epochs (phase 2 refinement).
    lr_adam : float
        Learning rate for Adam.
    lr_lbfgs : float
        Learning rate for L-BFGS.
    w_poisson, w_cont_n, w_cont_p : float
        Loss weights for each PDE.
    verbose : bool
        Print training progress.
    log_fn : callable
        Optional logging function.

    Returns
    -------
    PINNResult
        Solution profiles and training metadata.
    """
    _log = log_fn if log_fn is not None else print
    g = grid
    T = g.T
    device = torch.device('cpu')
    dtype = torch.float64

    # ── Material property averages for scaling ──
    eps_r_avg = float(np.mean(g.eps_r))
    C_ref = max(float(np.max(g.ND)), float(np.max(g.NA)), 1e15)
    mu_n_avg = float(np.mean(300.0 * (1.0 - g.x_Al) + 25.0 * g.x_Al))
    mu_p_avg = float(np.mean(10.0 * (1.0 - g.x_Al) + 2.0 * g.x_Al))
    L_device = float(g.x_m[-1] - g.x_m[0])

    scaling = DeVariScaling.from_device(eps_r_avg, T, C_ref, L_device, mu_n_avg, mu_p_avg)

    if verbose:
        _log(f"PINN Solver: De Mari scaling")
        _log(f"  V_T = {scaling.V_T*1e3:.2f} mV,  L_D = {scaling.L_D*1e9:.2f} nm")
        _log(f"  C_ref = {scaling.C_ref:.2e} cm^-3,  L_device = {L_device*1e9:.1f} nm")
        _log(f"  mu_n = {mu_n_avg:.1f} cm^2/Vs,  mu_p = {mu_p_avg:.1f} cm^2/Vs")

    # ── Boundary conditions (physical units) ──
    phi_left = float(g.phi_bottom_eq)
    phi_right = float(g.phi_top_eq + V_applied)
    Efn_left = 0.0
    Efn_right = -V_applied
    Efp_left = 0.0
    Efp_right = -V_applied

    if verbose:
        _log(f"  BCs: phi=[{phi_left:.3f}, {phi_right:.3f}] V")
        _log(f"        Efn=[{Efn_left:.3f}, {Efn_right:.3f}] V")
        _log(f"        Efp=[{Efp_left:.3f}, {Efp_right:.3f}] V")

    # ── Build material interpolator ──
    x_norm_np = (g.x_nm - g.x_nm[0]) / (g.x_nm[-1] - g.x_nm[0])
    mat = MaterialInterpolator(x_norm_np, device=device)
    mat.register('Ec0', g.Ec0)
    mat.register('Ev0', g.Ev0)
    mat.register('Nc', g.Nc)
    mat.register('Nv', g.Nv)
    mat.register('ND', g.ND)
    mat.register('NA', g.NA)
    mat.register('eps_r', g.eps_r)
    mat.register('pol_rho', g.pol_rho)

    # ── Build network ──
    model = PINNNetwork(n_hidden=n_hidden, n_neurons=n_neurons).to(device).to(dtype)
    model.phi_left = phi_left
    model.phi_right = phi_right
    model.Efn_left = Efn_left
    model.Efn_right = Efn_right
    model.Efp_left = Efp_left
    model.Efp_right = Efp_right

    # ── Loss function ──
    loss_fn = SemiconductorPINNLoss(scaling, mat,
                                     w_poisson=w_poisson,
                                     w_cont_n=w_cont_n,
                                     w_cont_p=w_cont_p)

    # ── Collocation points: clustered near doping/composition transitions
    # (depletion regions, heterojunction interfaces) instead of uniform --
    # see _build_collocation_points for why this matters. Width set by the
    # Debye length in normalised coordinates, the natural physical scale
    # for how wide a real transition actually is. ──
    L_D_norm = scaling.L_D / scaling.L_device
    x_colloc_np = _build_collocation_points(
        x_norm_np, g.ND, g.NA, g.x_Al, n_colloc, L_D_norm)
    if verbose:
        _log(f"  Collocation: {n_colloc} points, clustered (L_D_norm={L_D_norm:.4f})")
    x_colloc = torch.tensor(x_colloc_np, dtype=dtype, device=device).unsqueeze(-1)
    x_colloc.requires_grad_(True)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Adam optimiser (exploration)
    # ══════════════════════════════════════════════════════════════════
    loss_history = []
    t_start = time.time()

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, patience=500, factor=0.5, min_lr=1e-6
    )

    if verbose:
        _log(f"\nPhase 1: Adam optimiser ({n_epochs_adam} epochs, lr={lr_adam})")

    for epoch in range(n_epochs_adam):
        optimizer_adam.zero_grad()
        total_loss, ld = loss_fn.compute_loss(model, x_colloc)
        total_loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer_adam.step()
        scheduler.step(total_loss.item())

        loss_history.append(ld)

        if verbose and (epoch % 500 == 0 or epoch == n_epochs_adam - 1):
            _log(f"  epoch {epoch:5d}  loss = {ld['total']:.4e}  "
                 f"(Poisson={ld['poisson']:.3e}, Jn={ld['cont_n']:.3e}, Jp={ld['cont_p']:.3e})")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: L-BFGS optimiser (refinement)
    # ══════════════════════════════════════════════════════════════════
    if n_epochs_lbfgs > 0:
        if verbose:
            _log(f"\nPhase 2: L-BFGS optimiser ({n_epochs_lbfgs} epochs)")

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=lr_lbfgs,
            max_iter=20, history_size=50,
            line_search_fn='strong_wolfe',
        )

        lbfgs_step = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, ld = loss_fn.compute_loss(model, x_colloc)
            total_loss.backward()
            loss_history.append(ld)
            lbfgs_step[0] += 1
            if verbose and lbfgs_step[0] % 100 == 0:
                _log(f"  L-BFGS step {lbfgs_step[0]:5d}  loss = {ld['total']:.4e}  "
                     f"(Poisson={ld['poisson']:.3e}, Jn={ld['cont_n']:.3e}, Jp={ld['cont_p']:.3e})")
            return total_loss

        for _ in range(n_epochs_lbfgs):
            optimizer_lbfgs.step(closure)
            if lbfgs_step[0] >= n_epochs_lbfgs:
                break

    training_time = time.time() - t_start

    # ══════════════════════════════════════════════════════════════════
    # Extract solution on the full device grid
    # ══════════════════════════════════════════════════════════════════
    model.eval()
    with torch.no_grad():
        x_eval = torch.tensor(x_norm_np, dtype=dtype, device=device).unsqueeze(-1)
        phi_out, Efn_out, Efp_out = model(x_eval)
        phi_np = phi_out.squeeze().numpy()
        Efn_np = Efn_out.squeeze().numpy()
        Efp_np = Efp_out.squeeze().numpy()

    # Compute carrier densities from the solved potentials
    Ec_np = g.Ec0 - phi_np
    Ev_np = g.Ev0 - phi_np
    kBT_eV = kB * T / Q_E
    n_np = g.Nc * np.exp(np.clip((Efn_np - Ec_np) / kBT_eV, -80, 80))
    p_np = g.Nv * np.exp(np.clip((Ev_np - Efp_np) / kBT_eV, -80, 80))

    final_loss = loss_history[-1]['total'] if loss_history else float('inf')

    if verbose:
        _log(f"\nPINN training complete in {training_time:.1f}s")
        _log(f"  Final loss: {final_loss:.4e}")
        _log(f"  Total epochs: {len(loss_history)}")
        _log(f"  Efn - Efp range: [{(Efn_np - Efp_np).min():.4f}, {(Efn_np - Efp_np).max():.4f}] eV")

    return PINNResult(
        phi=phi_np, Efn=Efn_np, Efp=Efp_np,
        n=n_np, p=p_np,
        Ec=Ec_np, Ev=Ev_np,
        x_nm=g.x_nm,
        loss_history=loss_history,
        final_loss=final_loss,
        n_epochs=len(loss_history),
        training_time_s=training_time,
    )


# ═══════════════════════════════════════════════════════════════════════
# V-parameterized PINN: train once over V in [0, V_max], evaluate any bias
# afterward with a single forward pass (no retraining per voltage).
# ═══════════════════════════════════════════════════════════════════════
#
# The single-voltage solve_pinn above bakes V_applied into the boundary
# conditions at construction time, so a new voltage means a new network,
# retrained from scratch (minutes each). Here V_applied is a second network
# input instead: the network learns phi/Efn/Efp as a function of (x, V)
# jointly, trained on collocation points spanning the whole [0, V_max]
# range at once. One training run (still minutes, same cost as ONE
# single-voltage solve_pinn call) then makes any V in that range an
# instant query -- a single forward pass, no gradient descent -- enabling
# a live "train once, then slide through any bias" GUI workflow instead of
# "wait minutes per voltage".

class PINNNetworkV(nn.Module):
    """
    Like PINNNetwork, but with V_applied as a second input instead of a
    fixed boundary condition. Hard BC enforcement now depends on V:
        phi_right(V)  = phi_top_eq + V
        Efn_right(V)  = Efp_right(V) = -V
    (phi_left, Efn_left, Efp_left are always 0 / the equilibrium bottom
    value, independent of V, same as the single-voltage network.)
    """

    def __init__(self, n_hidden: int = 6, n_neurons: int = 96):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_neurons, 3))
        self.net = nn.Sequential(*layers)

        self.phi_left = 0.0
        self.phi_top_eq = 0.0   # phi_right(V) = phi_top_eq + V

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, V: torch.Tensor):
        """x, V: shape (N, 1) each. V in physical Volts (not normalised --
        small enough range that raw volts train fine; avoids an extra
        rescale/unscale round trip at evaluation time)."""
        raw = self.net(torch.cat([x, V], dim=1))
        phi_raw, efn_raw, efp_raw = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]

        window = x * (1.0 - x)
        phi_right = self.phi_top_eq + V
        Efn_right = -V
        Efp_right = -V

        phi = self.phi_left + x * (phi_right - self.phi_left) + window * phi_raw
        Efn = x * Efn_right + window * efn_raw
        Efp = x * Efp_right + window * efp_raw
        return phi, Efn, Efp


def _compute_loss_v(loss_fn: SemiconductorPINNLoss, model: PINNNetworkV,
                     x_colloc: torch.Tensor, V_colloc: torch.Tensor):
    """Same physics loss as SemiconductorPINNLoss.compute_loss, but for the
    2-input (x, V) network -- reuses carrier_densities/recombination
    unchanged (they only depend on phi/Efn/Efp, not on how those were
    produced) and only changes the model call and derivative w.r.t. x
    (still x only: V is a parameter the PDE is solved *for*, not a
    coordinate the PDE differentiates with respect to)."""
    s = loss_fn.s
    L = s.L_device

    phi_V, Efn_V, Efp_V = model(x_colloc, V_colloc)
    n, p, Ec, Ev = loss_fn.carrier_densities(phi_V, Efn_V, Efp_V, x_colloc)

    eps_r = loss_fn.mat('eps_r', x_colloc)
    ND = loss_fn.mat('ND', x_colloc)
    NA = loss_fn.mat('NA', x_colloc)
    pol_rho = loss_fn.mat('pol_rho', x_colloc)

    dphi = _grad(phi_V, x_colloc)
    d2phi = _grad(dphi, x_colloc)
    charge = Q_E * (p - n + ND - NA) * 1e6 + pol_rho
    poisson_res = eps_r * eps0 * d2phi / (L * L) + charge
    rho_scale = Q_E * s.C_ref * 1e6
    poisson_loss = torch.mean((poisson_res / rho_scale) ** 2)

    R = loss_fn.recombination(n, p, x_colloc)
    L_cm2 = L * L * 1e4

    dn = _grad(n, x_colloc)
    dEfn = _grad(Efn_V, x_colloc)
    d2Efn = _grad(dEfn, x_colloc)
    cont_n_res = (s.mu_n / L_cm2) * (dn * dEfn + n * d2Efn) - R
    R_scale = max(s.C_ref / loss_fn.tau_n, 1.0)
    cont_n_loss = torch.mean((cont_n_res / R_scale) ** 2)

    dp = _grad(p, x_colloc)
    dEfp = _grad(Efp_V, x_colloc)
    d2Efp = _grad(dEfp, x_colloc)
    cont_p_res = -(s.mu_p / L_cm2) * (dp * dEfp + p * d2Efp) - R
    cont_p_loss = torch.mean((cont_p_res / R_scale) ** 2)

    total_loss = (loss_fn.w_poisson * poisson_loss
                  + loss_fn.w_cont_n * cont_n_loss
                  + loss_fn.w_cont_p * cont_p_loss)
    loss_dict = {
        'total': total_loss.item(), 'poisson': poisson_loss.item(),
        'cont_n': cont_n_loss.item(), 'cont_p': cont_p_loss.item(),
    }
    return total_loss, loss_dict


@dataclass
class PINNTrainedModel:
    """
    A network trained once over V in [0, V_max]. `.evaluate(V)` is an
    instant (single forward pass) query for any V in that range -- no
    retraining -- returning a physics.self_consistent.SolverResult so it
    slots directly into the existing band-diagram plotting/GUI code the
    same way a classical solve() result does.
    """
    model: PINNNetworkV
    grid: object            # devices.grid_builder.GridData
    x_norm_np: np.ndarray
    V_max: float
    loss_history: list
    final_loss: float
    training_time_s: float
    torch_device: torch.device = torch.device('cpu')

    def evaluate(self, V_applied: float):
        from physics.self_consistent import SolverResult
        from physics.poisson import electric_field

        g = self.grid
        T = g.T
        dtype = torch.float64
        dev = self.torch_device
        self.model.eval()
        with torch.no_grad():
            x_eval = torch.tensor(self.x_norm_np, dtype=dtype, device=dev).unsqueeze(-1)
            V_eval = torch.full_like(x_eval, float(V_applied))
            phi_out, Efn_out, Efp_out = self.model(x_eval, V_eval)
            phi_np = phi_out.squeeze().cpu().numpy()
            Efn_np = Efn_out.squeeze().cpu().numpy()
            Efp_np = Efp_out.squeeze().cpu().numpy()

        Ec_np = g.Ec0 - phi_np
        Ev_np = g.Ev0 - phi_np
        Ei_np = 0.5 * (Ec_np + Ev_np) + 0.5 * (kB * T / Q_E) * np.log(g.Nv / g.Nc)
        kBT_eV = kB * T / Q_E
        n_np = g.Nc * np.exp(np.clip((Efn_np - Ec_np) / kBT_eV, -80, 80))
        p_np = g.Nv * np.exp(np.clip((Ev_np - Efp_np) / kBT_eV, -80, 80))
        E_field_np = electric_field(phi_np, g.dx)

        return SolverResult(
            x_nm=g.x_nm, x_Al=g.x_Al,
            Ec=Ec_np, Ev=Ev_np, Ei=Ei_np,
            Efn=Efn_np, Efp=Efp_np,
            phi=phi_np, E_field=E_field_np, F_quasi=g.F_quasi,
            n=n_np, p=p_np,
            Psp=g.Psp, Ppz=g.Ppz, P_total=g.P_total,
            eps_xx=g.eps_xx, eps_zz=g.eps_zz,
            surface_charge_markers=[],
            V_applied=float(V_applied), V_internal=float(V_applied), T=T,
            converged=True, n_iterations=0,
            interface_indices=g.interface_indices,
            interface_sigmas=g.interface_sigmas,
            residual_history=[], alpha_history=[],
            Ec0=g.Ec0, Ev0=g.Ev0, ND=g.ND, NA=g.NA,
        )


def train_pinn_v(
    grid,
    V_max: float,
    n_colloc_x: int = 300,
    n_colloc_v: int = 20,
    n_hidden: int = 6,
    n_neurons: int = 96,
    n_epochs_adam: int = 5000,
    n_epochs_lbfgs: int = 1000,
    lr_adam: float = 1e-3,
    lr_lbfgs: float = 1.0,
    w_poisson: float = 1.0,
    w_cont_n: float = 0.1,
    w_cont_p: float = 0.1,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    device: str = 'cpu',
) -> PINNTrainedModel:
    """
    Train a single network to solve the device for ANY V_applied in
    [0, V_max] at once (see PINNTrainedModel). Collocation points are a
    fixed grid of n_colloc_x (clustered, see _build_collocation_points) by
    n_colloc_v (uniform in V) points, re-evaluated together every epoch --
    n_colloc_x * n_colloc_v total residual evaluations per step, so this
    costs roughly n_colloc_v times one single-voltage solve_pinn epoch;
    the trade is paying that once instead of once per voltage queried
    afterward.

    device : 'cpu' | 'cuda' | 'xpu' -- see resolve_torch_device. Falls back
    to CPU (with a logged reason) if the requested backend isn't actually
    available on this machine/torch build.
    """
    _log = log_fn if log_fn is not None else print
    g = grid
    T = g.T
    torch_device = resolve_torch_device(device, log_fn=_log)
    if verbose:
        _log(f"  Compute device: {torch_device}")
    device = torch_device
    dtype = torch.float64

    eps_r_avg = float(np.mean(g.eps_r))
    C_ref = max(float(np.max(g.ND)), float(np.max(g.NA)), 1e15)
    mu_n_avg = float(np.mean(300.0 * (1.0 - g.x_Al) + 25.0 * g.x_Al))
    mu_p_avg = float(np.mean(10.0 * (1.0 - g.x_Al) + 2.0 * g.x_Al))
    L_device = float(g.x_m[-1] - g.x_m[0])
    scaling = DeVariScaling.from_device(eps_r_avg, T, C_ref, L_device, mu_n_avg, mu_p_avg)

    if verbose:
        _log(f"PINN Solver (V-parameterized, train once for V in [0, {V_max}]V)")
        _log(f"  V_T = {scaling.V_T*1e3:.2f} mV,  L_D = {scaling.L_D*1e9:.2f} nm")
        _log(f"  C_ref = {scaling.C_ref:.2e} cm^-3,  L_device = {L_device*1e9:.1f} nm")

    x_norm_np = (g.x_nm - g.x_nm[0]) / (g.x_nm[-1] - g.x_nm[0])
    mat = MaterialInterpolator(x_norm_np, device=device)
    mat.register('Ec0', g.Ec0)
    mat.register('Ev0', g.Ev0)
    mat.register('Nc', g.Nc)
    mat.register('Nv', g.Nv)
    mat.register('ND', g.ND)
    mat.register('NA', g.NA)
    mat.register('eps_r', g.eps_r)
    mat.register('pol_rho', g.pol_rho)

    model = PINNNetworkV(n_hidden=n_hidden, n_neurons=n_neurons).to(device).to(dtype)
    model.phi_left = float(g.phi_bottom_eq)
    model.phi_top_eq = float(g.phi_top_eq)

    loss_fn = SemiconductorPINNLoss(scaling, mat, w_poisson=w_poisson,
                                     w_cont_n=w_cont_n, w_cont_p=w_cont_p)

    L_D_norm = scaling.L_D / scaling.L_device
    x_colloc_1d = _build_collocation_points(
        x_norm_np, g.ND, g.NA, g.x_Al, n_colloc_x, L_D_norm)
    V_colloc_1d = np.linspace(0.0, V_max, n_colloc_v)
    x_grid, V_grid = np.meshgrid(x_colloc_1d, V_colloc_1d, indexing='ij')
    x_colloc = torch.tensor(x_grid.ravel(), dtype=dtype, device=device).unsqueeze(-1)
    V_colloc = torch.tensor(V_grid.ravel(), dtype=dtype, device=device).unsqueeze(-1)
    x_colloc.requires_grad_(True)
    if verbose:
        _log(f"  Collocation: {n_colloc_x} x-points (clustered) x {n_colloc_v} "
             f"V-points = {x_colloc.shape[0]} total")

    loss_history: list = []
    t_start = time.time()

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam, patience=500, factor=0.5, min_lr=1e-6)

    if verbose:
        _log(f"\nPhase 1: Adam optimiser ({n_epochs_adam} epochs, lr={lr_adam})")

    for epoch in range(n_epochs_adam):
        if cancel_check is not None and cancel_check():
            _log("  PINN training cancelled by user")
            break
        optimizer_adam.zero_grad()
        total_loss, ld = _compute_loss_v(loss_fn, model, x_colloc, V_colloc)
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step(total_loss.item())
        loss_history.append(ld)
        if verbose and (epoch % 200 == 0 or epoch == n_epochs_adam - 1):
            _log(f"  epoch {epoch:5d}  loss = {ld['total']:.4e}  "
                 f"(Poisson={ld['poisson']:.3e}, Jn={ld['cont_n']:.3e}, Jp={ld['cont_p']:.3e})")

    if n_epochs_lbfgs > 0:
        if verbose:
            _log(f"\nPhase 2: L-BFGS optimiser ({n_epochs_lbfgs} epochs)")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=lr_lbfgs, max_iter=20, history_size=50,
            line_search_fn='strong_wolfe')
        lbfgs_step = [0]
        cancelled = [False]

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, ld = _compute_loss_v(loss_fn, model, x_colloc, V_colloc)
            total_loss.backward()
            loss_history.append(ld)
            lbfgs_step[0] += 1
            if verbose and lbfgs_step[0] % 50 == 0:
                _log(f"  L-BFGS step {lbfgs_step[0]:5d}  loss = {ld['total']:.4e}  "
                     f"(Poisson={ld['poisson']:.3e}, Jn={ld['cont_n']:.3e}, Jp={ld['cont_p']:.3e})")
            return total_loss

        for _ in range(n_epochs_lbfgs):
            if cancel_check is not None and cancel_check():
                _log("  PINN training cancelled by user")
                cancelled[0] = True
                break
            optimizer_lbfgs.step(closure)
            if lbfgs_step[0] >= n_epochs_lbfgs:
                break

    training_time = time.time() - t_start
    final_loss = loss_history[-1]['total'] if loss_history else float('inf')

    if verbose:
        _log(f"\nPINN training complete in {training_time:.1f}s "
             f"({len(loss_history)} total epochs, final loss={final_loss:.4e})")
        _log(f"  Ready: call .evaluate(V) for any V in [0, {V_max}] -- instant, no retraining")

    return PINNTrainedModel(
        model=model, grid=g, x_norm_np=x_norm_np, V_max=V_max,
        loss_history=loss_history, final_loss=final_loss,
        training_time_s=training_time, torch_device=torch_device,
    )

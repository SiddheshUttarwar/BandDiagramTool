"""
Build the 1D spatial grid from a list of AlGaN layers.

GridData holds all spatially resolved material profiles needed by the solvers.
All internal arrays use SI units (m, J, C, F/m, etc.).
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from devices.layer import AbruptLayer, GradedLayer, Contact
from physics.materials.algan import get_AlGaN_params, get_AlGaN_params_T
from physics.polarization import (
    compute_Psp, compute_strain, compute_Ppz,
    compute_pol_charge, compute_quasi_field, interface_sheet_charges,
)
from physics.constants import q
from scipy.optimize import brentq
from physics.fermi_dirac import electron_density, hole_density

logger = logging.getLogger(__name__)


@dataclass
class GridData:
    """All spatially-resolved profiles on the 1D simulation grid."""

    # --- Grid ---
    x_m: np.ndarray          # position [m]
    x_nm: np.ndarray         # position [nm] (for plotting)
    dx: float                # uniform grid spacing [m]
    N: int                   # number of grid points

    # --- Composition ---
    x_Al: np.ndarray         # Al composition profile

    # --- Band structure (equilibrium, zero field) ---
    Ec0: np.ndarray          # conduction band edge [eV], Ef-referenced
    Ev0: np.ndarray          # valence band edge [eV], Ef-referenced
    Eg: np.ndarray           # bandgap [eV]
    chi: np.ndarray          # electron affinity [eV]

    # --- Material profiles ---
    eps_r: np.ndarray        # relative permittivity
    m_e: np.ndarray          # electron DOS eff. mass [units of m0]
    m_hh: np.ndarray         # heavy-hole eff. mass [units of m0]
    m_lh: np.ndarray         # light-hole eff. mass [units of m0]

    # --- Doping [cm^-3] ---
    ND: np.ndarray           # ionised donor concentration
    NA: np.ndarray           # ionised acceptor concentration

    # --- Effective DOS [cm^-3] ---
    Nc: np.ndarray
    Nv: np.ndarray

    # --- Strain ---
    eps_xx: np.ndarray       # biaxial in-plane strain (dimensionless)
    eps_zz: np.ndarray       # out-of-plane strain

    # --- Polarization ---
    Psp: np.ndarray          # spontaneous polarization [C/m²]
    Ppz: np.ndarray          # piezoelectric polarization [C/m²]
    P_total: np.ndarray      # Psp + Ppz [C/m²]
    pol_rho: np.ndarray      # polarization volume charge [C/m³]
    F_quasi: np.ndarray      # quasi-electric field from composition gradient [V/m]

    # --- Interface info ---
    interface_indices: List[int]    # grid indices of abrupt interfaces
    interface_sigmas: List[float]   # sheet charges [C/m²]

    # --- Boundary potentials (equilibrium, V=0) ---
    phi_bottom_eq: float     # equilibrium electrostatic potential at bottom contact [V]
    phi_top_eq: float        # equilibrium electrostatic potential at top contact [V]

    # --- Contact info (for bias calculations) ---
    bottom_contact: Contact
    top_contact: Contact

    # --- Temperature ---
    T: float                 # simulation temperature [K]


def _x_Al_profile(layer: AbruptLayer | GradedLayer, n_pts: int) -> np.ndarray:
    """Return the x_Al array for a layer with n_pts grid points."""
    if isinstance(layer, AbruptLayer):
        return np.full(n_pts, layer.x_Al)

    x0, x1 = layer.x_Al_start, layer.x_Al_end
    z = np.linspace(0.0, 1.0, n_pts)  # normalised position within the layer

    if layer.profile == 'abrupt':
        return np.full(n_pts, x0)
    elif layer.profile == 'linear':
        return x0 + (x1 - x0) * z
    elif layer.profile == 'parabolic':
        return x0 + (x1 - x0) * z**2
    elif layer.profile == 'stepped':
        steps = np.linspace(x0, x1, layer.n_steps)
        idx   = np.floor(z * layer.n_steps).astype(int).clip(0, layer.n_steps - 1)
        return steps[idx]
    else:
        raise ValueError(f"Unknown grading profile '{layer.profile}'. "
                         f"Choose 'linear', 'parabolic', 'stepped', or 'abrupt'.")


def build_grid(
    layers: list,
    contacts: list,
    T: float = 300.0,
    dx_nm: float = 0.1,
) -> GridData:
    """
    Build the 1D simulation grid from an ordered list of layers (bottom → top).

    Parameters
    ----------
    layers   : list of AbruptLayer or GradedLayer, ordered bottom to top
    contacts : list of two Contact objects (one 'bottom', one 'top')
    T        : temperature [K]
    dx_nm    : grid spacing [nm]

    Returns
    -------
    GridData with all profiles populated.
    """
    if len(contacts) != 2:
        raise ValueError("Exactly 2 contacts required (bottom and top).")

    # Sort contacts
    bottom_contact = next(c for c in contacts if c.position == 'bottom')
    top_contact    = next(c for c in contacts if c.position == 'top')

    dx_m = dx_nm * 1e-9   # nm → m

    # --- Build x_Al, doping, and relaxation profiles from layers ---
    x_Al_list  = []
    ND_list    = []
    NA_list    = []
    relaxed_list = []

    for layer in layers:
        n_pts = max(2, int(round(layer.thickness_nm / dx_nm)))
        x_Al_segment = _x_Al_profile(layer, n_pts)
        x_Al_list.append(x_Al_segment)
        ND_list.append(np.full(n_pts, layer.n_doping))
        NA_list.append(np.full(n_pts, layer.p_doping))
        relaxed_list.append(
            np.full(n_pts, getattr(layer, 'relaxed', False))
        )

    x_Al_raw = np.concatenate(x_Al_list)
    ND    = np.concatenate(ND_list)
    NA    = np.concatenate(NA_list)
    relaxed = np.concatenate(relaxed_list).astype(bool)
    N     = len(x_Al_raw)
    x_nm  = np.arange(N) * dx_nm
    x_m   = x_nm * 1e-9

    # --- Nextnano Nanosmoothing ---
    # Real heterojunctions are not perfectly abrupt. We smooth the composition
    # profile by 1 grid point to prevent numerical delta-spikes in
    # polarization charge — the minimum smoothing that still avoids a
    # literal single-cell discontinuity. Fixed in grid units (not physical
    # nm), so the physical smoothing width scales with dx_nm: finer grids
    # give physically sharper junctions, not the same nm-wide ramp resolved
    # more finely.
    from scipy.ndimage import gaussian_filter1d
    x_Al = gaussian_filter1d(x_Al_raw, sigma=1.0)
    x_Al = np.clip(x_Al, 0.0, 1.0)

    # --- Material parameters at each grid point ---
    Eg    = np.empty(N)
    chi   = np.empty(N)
    eps_r = np.empty(N)
    m_e   = np.empty(N)
    m_hh  = np.empty(N)
    m_lh  = np.empty(N)
    Nc_arr = np.empty(N)
    Nv_arr = np.empty(N)

    for i in range(N):
        p = get_AlGaN_params_T(x_Al[i], T)
        Eg[i]    = p.Eg
        chi[i]   = p.chi
        eps_r[i] = p.eps_r
        m_e[i]   = p.m_e_dos
        m_hh[i]  = p.m_hh
        m_lh[i]  = p.m_lh
        Nc_arr[i] = p.Nc(T)
        Nv_arr[i] = p.Nv(T)

    # --- Strain ---
    x_sub = float(x_Al[0])   # substrate = bottom layer composition
    eps_xx, eps_zz = compute_strain(x_Al, x_sub)
    # Zero strain where layer is relaxed
    eps_xx[relaxed] = 0.0
    eps_zz[relaxed] = 0.0

    # --- Polarization ---
    Psp_arr = compute_Psp(x_Al, T)
    Ppz_arr = compute_Ppz(x_Al, eps_xx, eps_zz)
    P_total = Psp_arr + Ppz_arr
    pol_rho = compute_pol_charge(P_total, dx_m)
    F_quasi = compute_quasi_field(x_Al, dx_m)

    ifaces  = interface_sheet_charges(P_total, x_Al)
    iface_idx   = [idx for idx, _ in ifaces]
    iface_sigma = [sig for _, sig in ifaces]

    # --- Band edges (Fermi level referenced at 0) ---
    # Reference: Ef = 0 everywhere in equilibrium.
    # At bottom contact (ohmic or Schottky), set Ec0 such that charge
    # neutrality holds at x=0.
    # For the rest of the device, use Anderson's rule: Ec0(x) = chi[0] - chi(x) + Ec0[0]
    kBT_eV = 1.380649e-23 * T / 1.602176634e-19

    # Ec0 at bottom boundary from charge neutrality (Fermi-Dirac + Incomplete Ionization)
    if bottom_contact.contact_type == 'ohmic':
        def charge_imbalance_bottom(Ec0_guess: float) -> float:
            Ev0_guess = Ec0_guess - Eg[0]
            Ed_0 = 0.02
            Nd_plus = ND[0] / (1.0 + 2.0 * np.exp((0.0 - (Ec0_guess - Ed_0)) / kBT_eV))
            Na_minus = NA[0] / (1.0 + 4.0 * np.exp((Ev0_guess + 0.17 - 0.0) / kBT_eV))
            n = float(electron_density(Ec0_guess, 0.0, Nc_arr[0], T))
            p = float(hole_density(Ev0_guess, 0.0, Nv_arr[0], T))
            return (Nd_plus - Na_minus) + p - n
            
        try:
            Ec0_bottom = brentq(charge_imbalance_bottom, -10.0, 10.0)
        except ValueError:
            logger.warning(
                "Charge-neutrality brentq failed at bottom ohmic contact; "
                "falling back to Ec0_bottom = 0.0 eV"
            )
            Ec0_bottom = 0.0
    else:
        # Schottky: Ec = Ef + phi_B at contact
        from physics.materials.metals import get_work_function
        phi_M = get_work_function(bottom_contact.metal)
        phi_B = phi_M - chi[0]   # Schottky barrier height [eV]
        Ec0_bottom = phi_B       # Ec measured from Ef=0

    # Ec0 profile via Anderson's rule: Ec0(x) = Ec0_bottom + (chi[0] - chi(x))
    Ec0 = Ec0_bottom + (chi[0] - chi)
    Ev0 = Ec0 - Eg

    # --- Equilibrium boundary potentials ---
    # phi = 0 at bottom (ground reference); phi satisfies Ec(x)=Ec0(x)-phi(x)
    # At bottom: phi_bottom = 0 → Ec(0) = Ec0(0) ✓
    phi_bottom_eq = 0.0

    # At top contact: find phi_top such that charge neutrality holds there
    if top_contact.contact_type == 'ohmic':
        def charge_imbalance_top(Ec_top_guess: float) -> float:
            Ev_top_guess = Ec_top_guess - Eg[-1]
            Ed_top = 0.02
            Nd_plus = ND[-1] / (1.0 + 2.0 * np.exp((0.0 - (Ec_top_guess - Ed_top)) / kBT_eV))
            Na_minus = NA[-1] / (1.0 + 4.0 * np.exp((Ev_top_guess + 0.17 - 0.0) / kBT_eV))
            n = float(electron_density(Ec_top_guess, 0.0, Nc_arr[-1], T))
            p = float(hole_density(Ev_top_guess, 0.0, Nv_arr[-1], T))
            return (Nd_plus - Na_minus) + p - n
            
        try:
            Ec_top = brentq(charge_imbalance_top, -10.0, 10.0)
        except ValueError:
            logger.warning(
                "Charge-neutrality brentq failed at top ohmic contact; "
                "falling back to Ec_top = 0.0 eV"
            )
            Ec_top = 0.0
            
        phi_top_eq = Ec0[-1] - Ec_top
    else:
        from physics.materials.metals import get_work_function
        phi_M  = get_work_function(top_contact.metal)
        phi_B  = phi_M - chi[-1]
        phi_top_eq = Ec0[-1] - phi_B

    return GridData(
        x_m=x_m, x_nm=x_nm, dx=dx_m, N=N,
        x_Al=x_Al,
        Ec0=Ec0, Ev0=Ev0, Eg=Eg, chi=chi,
        eps_r=eps_r, m_e=m_e, m_hh=m_hh, m_lh=m_lh,
        ND=ND, NA=NA,
        Nc=Nc_arr, Nv=Nv_arr,
        eps_xx=eps_xx, eps_zz=eps_zz,
        Psp=Psp_arr, Ppz=Ppz_arr, P_total=P_total,
        pol_rho=pol_rho, F_quasi=F_quasi,
        interface_indices=iface_idx, interface_sigmas=iface_sigma,
        phi_bottom_eq=phi_bottom_eq, phi_top_eq=phi_top_eq,
        bottom_contact=bottom_contact, top_contact=top_contact,
        T=T,
    )

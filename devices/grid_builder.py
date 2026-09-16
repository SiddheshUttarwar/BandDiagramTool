"""
Build the 1D spatial grid from a list of AlGaN layers.

GridData holds all spatially resolved material profiles needed by the solvers.
All internal arrays use SI units (m, J, C, F/m, etc.).
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from devices.layer import (
    AbruptLayer, GradedLayer, Contact,
    QuantumRegionMarker, SurfaceCharge, SurfaceState, InterfaceDipole,
)
from physics.materials.algan import get_AlGaN_params, get_AlGaN_params_T
from physics.polarization import (
    compute_Psp, compute_strain, compute_Ppz, eps_zz_from_eps_xx,
    compute_pol_charge, compute_quasi_field, interface_sheet_charges,
)
from physics.constants import q
from physics.grid_utils import node_spacings
from scipy.optimize import brentq
from physics.fermi_dirac import electron_density, hole_density

logger = logging.getLogger(__name__)


@dataclass
class GridData:
    """All spatially-resolved profiles on the 1D simulation grid."""

    # --- Grid ---
    x_m: np.ndarray          # position [m]
    x_nm: np.ndarray         # position [nm] (for plotting)
    dx: np.ndarray           # per-edge grid spacing [m], length N-1 (may be
                              # non-uniform if any layer set its own dx_nm;
                              # see physics.grid_utils.node_spacings)
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
    m_so: np.ndarray         # split-off hole eff. mass [units of m0]

    # Zone-center (k_t=0) LH/SO band-edge offsets *below* Ev0 (the HH edge
    # -- see physics.materials.algan.AlGaNParams.valence_band_structure).
    # Position-dependent (via local Al composition), static -- doesn't
    # depend on phi, so computed once here rather than every solver
    # iteration. Ev_lh(x) = Ev0(x) - dEv_lh(x), Ev_so(x) = Ev0(x) - dEv_so(x)
    # (can be negative for high-Al compositions -- see that docstring).
    dEv_lh: np.ndarray
    dEv_so: np.ndarray

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

    # --- Primary quantum well (for QCSE diagnostics) ---
    # Grid index window (i0, i1) of the layer stack's "textbook" quantum
    # well -- an undoped layer whose composition is a strict local minimum
    # relative to both neighbours (lower bandgap sandwiched by
    # higher-bandgap barriers). None if the layer stack has no such layer.
    qw_window: Optional[Tuple[int, int]] = None

    # --- Manual quantum (Schrodinger-solve) region ---
    # Grid index window (i0, i1), padded 2nm beyond the user-placed
    # QuantumRegionMarker layers (see devices.layer.QuantumRegionMarker).
    # When set, physics.self_consistent uses this directly instead of its
    # automatic undoped-span heuristic. None if no start+end pair is placed.
    manual_quantum_region: Optional[Tuple[int, int]] = None

    # --- Surface/interface charge states (devices.layer.SurfaceCharge) ---
    # One entry per SurfaceCharge layer: (grid index, list of SurfaceState).
    # physics.self_consistent ionizes each state self-consistently with the
    # local Fermi level, the same way as bulk donor/acceptor doping.
    surface_charge_sites: List[Tuple[int, List[SurfaceState]]] = field(default_factory=list)


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
    include_spontaneous_polarization: bool = False,
) -> GridData:
    """
    Build the 1D simulation grid from an ordered list of layers (bottom → top).

    Parameters
    ----------
    layers   : list of AbruptLayer or GradedLayer, ordered bottom to top
    contacts : list of two Contact objects (one 'bottom', one 'top')
    T        : temperature [K]
    dx_nm    : grid spacing [nm]
    include_spontaneous_polarization : if False, Psp is excluded from the
        charge (pol_rho, interface sheet charges) that drives band bending
        -- only Ppz contributes. Psp/P_total on GridData are still computed
        and populated either way, for display/diagnostics.

    Returns
    -------
    GridData with all profiles populated.
    """
    if len(contacts) != 2:
        raise ValueError("Exactly 2 contacts required (bottom and top).")

    # Sort contacts
    bottom_contact = next(c for c in contacts if c.position == 'bottom')
    top_contact    = next(c for c in contacts if c.position == 'top')

    dx_m = dx_nm * 1e-9   # nm → m; the *default* spacing for layers that
                          # don't request their own (layer.dx_nm)

    # --- Build x_Al, doping, and relaxation profiles from layers ---
    # Each layer may request its own grid spacing (layer.dx_nm) -- e.g. a
    # fine mesh in a thin quantum well without paying for that resolution
    # across the whole device. Position is therefore built by concatenating
    # each layer's own local, uniformly-spaced sub-grid (n_pts points at
    # that layer's own dx) rather than a single global np.arange(N)*dx_nm;
    # the result is uniform overall only if every layer shares the same
    # (or default) spacing, and non-uniform otherwise. See physics.
    # grid_utils for how the solvers consume the resulting per-edge dx.
    x_Al_list  = []
    ND_list    = []
    NA_list    = []
    relaxed_list = []
    custom_strain_list = []   # per-point custom eps_xx, NaN where unset (see AbruptLayer.custom_strain_xx)
    x_nm_list  = []
    physical_layers = []   # layers filtered to Abrupt/Graded only, parallel to *_list above

    # Zero-thickness interface layers don't consume grid points -- their
    # position in the device is simply their position in the stack, i.e.
    # the running `pos_nm` at the moment they're encountered.
    quantum_markers = []    # list of (boundary, pos_nm)
    surface_charges = []    # list of (SurfaceCharge, pos_nm)
    interface_dipoles = []  # list of (InterfaceDipole, pos_nm)

    pos_nm = 0.0
    for layer in layers:
        if isinstance(layer, QuantumRegionMarker):
            quantum_markers.append((layer.boundary, pos_nm))
            continue
        if isinstance(layer, SurfaceCharge):
            surface_charges.append((layer, pos_nm))
            continue
        if isinstance(layer, InterfaceDipole):
            interface_dipoles.append((layer, pos_nm))
            continue

        physical_layers.append(layer)
        layer_dx_nm = layer.dx_nm if getattr(layer, 'dx_nm', None) else dx_nm
        n_pts = max(2, int(round(layer.thickness_nm / layer_dx_nm)))
        x_Al_segment = _x_Al_profile(layer, n_pts)
        x_Al_list.append(x_Al_segment)
        ND_list.append(np.full(n_pts, layer.n_doping))
        NA_list.append(np.full(n_pts, layer.p_doping))
        relaxed_list.append(
            np.full(n_pts, getattr(layer, 'relaxed', False))
        )
        custom_strain_xx = getattr(layer, 'custom_strain_xx', None)
        custom_strain_list.append(
            np.full(n_pts, custom_strain_xx if custom_strain_xx is not None else np.nan)
        )
        x_nm_list.append(pos_nm + np.arange(n_pts) * layer_dx_nm)
        pos_nm += n_pts * layer_dx_nm

    if not physical_layers:
        raise ValueError("Device stack has no physical (Abrupt/Graded) layers.")

    x_Al_raw = np.concatenate(x_Al_list)
    ND    = np.concatenate(ND_list)
    NA    = np.concatenate(NA_list)
    relaxed = np.concatenate(relaxed_list).astype(bool)
    custom_strain_xx = np.concatenate(custom_strain_list)
    x_nm  = np.concatenate(x_nm_list)
    N     = len(x_Al_raw)
    x_m   = x_nm * 1e-9
    dx_edges_m = np.diff(x_m)   # per-edge spacing [m], length N-1 (see physics.grid_utils)

    # --- Identify the primary quantum well (for QCSE diagnostics) ---
    # A "well" is an undoped layer whose composition is a strict local
    # minimum relative to both neighbours (lower bandgap sandwiched by
    # higher-bandgap barriers on both sides) -- the textbook definition of
    # a quantum well, found directly from the user's layer stack rather
    # than from wherever a Schrodinger solve's *global* ground state
    # happens to localise (which can instead be a polarization/doping
    # notch elsewhere in the device -- see
    # physics.self_consistent._detect_quantum_region, which restricts the
    # solve to the whole undoped span, QW+barriers+EBL together, not just
    # the well). Ties broken by picking the thinnest candidate (narrowest
    # = most confined = the layer a designer would call "the" well).
    def _layer_x_Al_repr(layer) -> float:
        if isinstance(layer, AbruptLayer):
            return layer.x_Al
        return 0.5 * (layer.x_Al_start + layer.x_Al_end)

    def _layer_undoped(layer) -> bool:
        return layer.n_doping <= 0.0 and layer.p_doping <= 0.0

    layer_offsets = np.cumsum([0] + [len(seg) for seg in x_Al_list])
    qw_window: Optional[Tuple[int, int]] = None
    qw_thickness = np.inf
    for li in range(1, len(physical_layers) - 1):
        layer = physical_layers[li]
        if not _layer_undoped(layer):
            continue
        x_here = _layer_x_Al_repr(layer)
        x_prev = _layer_x_Al_repr(physical_layers[li - 1])
        x_next = _layer_x_Al_repr(physical_layers[li + 1])
        if x_here < x_prev - 1e-6 and x_here < x_next - 1e-6:
            if layer.thickness_nm < qw_thickness:
                qw_thickness = layer.thickness_nm
                qw_window = (int(layer_offsets[li]), int(layer_offsets[li + 1]))

    # --- Manual quantum region markers ---
    # physics.self_consistent's automatic quantum-region detection
    # (_detect_quantum_region) sweeps in the device's *entire* contiguous
    # undoped span, which can incorrectly include a thick undoped layer
    # that isn't actually meant to be quantum-confined (e.g. a graded
    # transport/spacer region with no explicit doping set) alongside the
    # real MQW/barrier/EBL stack -- solving Schrodinger with many subbands
    # over such an oversized, physically-inappropriate span is both wrong
    # and numerically fragile at bias. If the user has placed a
    # QuantumRegionMarker('start') and QuantumRegionMarker('end') in the
    # stack, use their positions directly instead: padded 2nm on each side.
    manual_quantum_region: Optional[Tuple[int, int]] = None
    start_pos = next((pos for boundary, pos in quantum_markers if boundary == 'start'), None)
    end_pos = next((pos for boundary, pos in quantum_markers if boundary == 'end'), None)
    if start_pos is not None and end_pos is not None and start_pos <= end_pos:
        x_lo = start_pos - 2.0
        x_hi = end_pos + 2.0
        i0_padded = max(0, int(np.searchsorted(x_nm, x_lo, side='left')))
        i1_padded = min(N, int(np.searchsorted(x_nm, x_hi, side='right')))
        manual_quantum_region = (i0_padded, i1_padded)

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
    m_so  = np.empty(N)
    dEv_lh = np.empty(N)
    dEv_so = np.empty(N)
    Nc_arr = np.empty(N)
    Nv_arr = np.empty(N)

    for i in range(N):
        p = get_AlGaN_params_T(x_Al[i], T)
        Eg[i]    = p.Eg
        chi[i]   = p.chi
        eps_r[i] = p.eps_r
        m_e[i]   = p.m_e_dos
        m_hh[i]  = p.m_hh
        # dEv_lh/dEv_so and m_lh/m_so come from the same call: past the
        # LH/SO character crossover (see AlGaNParams.valence_band_structure),
        # the mass has to swap along with the energy branch, not just the
        # offset -- fetching mass and offset separately here would silently
        # decouple them.
        dEv_lh[i], dEv_so[i], m_lh[i], m_so[i] = p.valence_band_structure()
        Nc_arr[i] = p.Nc(T)
        Nv_arr[i] = p.Nv(T)

    # --- Strain ---
    x_sub = float(x_Al[0])   # substrate = bottom layer composition
    eps_xx, eps_zz = compute_strain(x_Al, x_sub)
    # Zero strain where layer is relaxed
    eps_xx[relaxed] = 0.0
    eps_zz[relaxed] = 0.0
    # Custom user-specified in-plane strain (devices.layer.*.custom_strain_xx)
    # overrides both of the above for its own layer: eps_xx is taken
    # directly from the layer, eps_zz re-derived from it via the same
    # elastic relation compute_strain itself uses (never set independently
    # -- see physics.polarization.eps_zz_from_eps_xx).
    has_custom = ~np.isnan(custom_strain_xx)
    if np.any(has_custom):
        eps_xx[has_custom] = custom_strain_xx[has_custom]
        eps_zz[has_custom] = eps_zz_from_eps_xx(x_Al[has_custom], eps_xx[has_custom])

    # --- Polarization ---
    # Psp is always computed and kept on GridData for display/diagnostics.
    # Band bending (pol_rho, interface sheet charges below) is driven by
    # P_charge, which only includes Psp when the caller opts in --
    # otherwise Ppz alone feeds the Poisson solve.
    Psp_arr = compute_Psp(x_Al, T)
    Ppz_arr = compute_Ppz(x_Al, eps_xx, eps_zz)
    P_total = Psp_arr + Ppz_arr
    P_charge = P_total if include_spontaneous_polarization else Ppz_arr
    pol_rho = compute_pol_charge(P_charge, dx_edges_m)
    F_quasi = compute_quasi_field(x_Al, dx_edges_m)

    # --- Interface dipoles (devices.layer.InterfaceDipole) ---
    # A fixed structural dipole: two equal-and-opposite sheet charges
    # separated by `separation_nm`. Injected as two concentrated volume
    # charges (areal charge / local cell width, same convention as
    # compute_pol_charge's polarization sheet charges) so the existing
    # Poisson solve produces the right potential step self-consistently --
    # no separate boundary-condition machinery needed.
    if interface_dipoles:
        _, _, cell_width_m, _ = node_spacings(dx_edges_m, N)
        for dipole, pos_nm_dip in interface_dipoles:
            i_pos = int(np.clip(np.searchsorted(x_nm, pos_nm_dip), 0, N - 1))
            i_neg = int(np.clip(np.searchsorted(x_nm, pos_nm_dip + dipole.separation_nm), 0, N - 1))
            if i_pos == i_neg:
                logger.warning(
                    "InterfaceDipole separation_nm=%.3g is finer than the local "
                    "grid spacing; both charge sheets landed on the same grid "
                    "point and cancelled out. Increase separation_nm or use a "
                    "finer dx_nm near this interface.", dipole.separation_nm
                )
                continue
            pol_rho[i_pos] += dipole.sheet_charge_C_m2 / cell_width_m[i_pos]
            pol_rho[i_neg] -= dipole.sheet_charge_C_m2 / cell_width_m[i_neg]

    # --- Surface/interface charge sites (devices.layer.SurfaceCharge) ---
    # Only the grid location + trap states are resolved here; the
    # Fermi-level-dependent ionization itself happens inside
    # physics.self_consistent's Newton-Poisson loop (it needs the local
    # quasi-Fermi level, which doesn't exist yet at grid-build time).
    surface_charge_sites: List[Tuple[int, List[SurfaceState]]] = []
    for surf, pos_nm_surf in surface_charges:
        i0 = int(np.clip(np.searchsorted(x_nm, pos_nm_surf), 0, N - 1))
        surface_charge_sites.append((i0, surf.states))

    ifaces  = interface_sheet_charges(P_charge, x_Al)
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
        x_m=x_m, x_nm=x_nm, dx=dx_edges_m, N=N,
        x_Al=x_Al,
        Ec0=Ec0, Ev0=Ev0, Eg=Eg, chi=chi,
        eps_r=eps_r, m_e=m_e, m_hh=m_hh, m_lh=m_lh, m_so=m_so,
        dEv_lh=dEv_lh, dEv_so=dEv_so,
        ND=ND, NA=NA,
        Nc=Nc_arr, Nv=Nv_arr,
        eps_xx=eps_xx, eps_zz=eps_zz,
        Psp=Psp_arr, Ppz=Ppz_arr, P_total=P_total,
        pol_rho=pol_rho, F_quasi=F_quasi,
        interface_indices=iface_idx, interface_sigmas=iface_sigma,
        phi_bottom_eq=phi_bottom_eq, phi_top_eq=phi_top_eq,
        bottom_contact=bottom_contact, top_contact=top_contact,
        T=T,
        qw_window=qw_window,
        manual_quantum_region=manual_quantum_region,
        surface_charge_sites=surface_charge_sites,
    )

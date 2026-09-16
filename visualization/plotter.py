"""
Band diagram visualization for AlGaN heterostructures.

BandDiagramPlotter wraps a SolverResult and produces publication-quality figures.

Panels available:
  plot_band_diagram()   — Ec, Ev, Efn, Efp, Ei, vacuum level + composition
                           background, and Mott-transition (physics.mott)
                           diagnostic shading (toggle: show_mott)
  plot_wavefunctions()  — |psi_n|^2 overlaid on band diagram
  plot_carriers()       — n(x) and p(x) on log scale
  plot_fields()         — electrostatic + quasi-electric + total field
  plot_polarization()   — Psp, Ppz, P_total + interface sheet charge bars
  plot_strain()         — eps_xx and eps_zz vs position
  plot_qcse()           — QCSE Stark shift + e-h overlap vs bias
  plot_all()            — 3x2 figure with all panels
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - registers the 'science' style family on import
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import List, Optional

from physics.self_consistent import SolverResult
from physics.constants import q as _q
from physics.mott import donor_mott_density_cm3, acceptor_mott_density_cm3

# 'no-latex' variant: 'science' alone requires a LaTeX installation (text.usetex),
# which most machines running this tool won't have; this keeps the same
# typography/spine/tick styling using matplotlib's built-in mathtext instead.
# Applied at import time so every panel in this module picks it up; per-plot
# colors/linewidths set explicitly below still override these style defaults.
plt.style.use(['science', 'no-latex'])


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_COLORS = {
    'Ec':   '#1f77b4',   # blue
    'Ev':   '#2ca02c',   # green (Ev == Ev_hh, the heavy-hole edge)
    'Ev_lh': '#9467bd',  # purple -- distinct from Ev/Ev_so, not a green shade
    'Ev_so': '#8c564b',  # brown -- distinct from Ev/Ev_lh, not a green shade
    'Efn':  '#d62728',   # red
    'Efp':  '#ff7f0e',   # orange
    'Ei':   '#7f7f7f',   # gray
    'vac':  '#bcbcbc',   # light gray
    'n':    '#1f77b4',
    'p':    '#d62728',
    'Psp':  '#1f77b4',
    'Ppz':  '#ff7f0e',
    'Ptot': '#d62728',
    'Felec': '#1f77b4',
    'Fquasi': '#2ca02c',
    'Ftot': '#d62728',
    'exx':  '#1f77b4',
    'ezz':  '#ff7f0e',
}


def _composition_background(ax, x_nm: np.ndarray, x_Al: np.ndarray,
                             y_min: float, y_max: float,
                             cmap: str = 'Blues', alpha: float = 0.15) -> None:
    """Shade background with Al composition colour map."""
    from matplotlib.collections import PolyCollection
    cmap_obj = plt.get_cmap(cmap)
    dx = x_nm[1] - x_nm[0] if len(x_nm) > 1 else 0.1
    for i in range(len(x_nm) - 1):
        color = cmap_obj(x_Al[i])
        rect = plt.Polygon(
            [[x_nm[i], y_min], [x_nm[i+1], y_min],
             [x_nm[i+1], y_max], [x_nm[i], y_max]],
            closed=True, facecolor=color, edgecolor='none', alpha=alpha,
        )
        ax.add_patch(rect)


def _shaded_spans(ax, x_nm: np.ndarray, mask: np.ndarray,
                   color, label: str, alpha: float = 0.16, hatch: Optional[str] = None) -> None:
    """Shade each contiguous True-run of `mask` as a full-height axvspan;
    only the first span carries the legend label so it isn't repeated."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    first = True
    for s, e in zip(starts, ends):
        ax.axvspan(x_nm[s], x_nm[min(e, len(x_nm) - 1)], color=color, alpha=alpha,
                   hatch=hatch, label=label if first else None, zorder=0)
        first = False


def _legend_right(ax, fontsize=9, ncol=1) -> None:
    """Place the legend outside the axes, to the right, so it can never
    cover any of the plotted data -- these panels plot device-spanning
    profiles (composition/doping shaded the whole width), so there is
    often no empty pocket inside the axes for loc='best' to land in
    without covering a curve. tight_layout() then shrinks the axes to
    make room, so the plot itself never gets covered by the legend box."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, fontsize=fontsize, loc='upper left',
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0., ncol=ncol,
              frameon=True)
    fig = ax.get_figure()
    if fig is not None:
        fig.tight_layout()


def _legend_above(ax, fontsize=9, ncol=3) -> None:
    """Place the legend outside the axes, above it -- for panels where a
    twin y-axis or a neighbouring subplot already occupies the right
    margin (see _legend_right). Anchored well clear of y=1.0 so it doesn't
    collide with the axes title, which sits just above the top spine."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, fontsize=fontsize, loc='lower center',
              bbox_to_anchor=(0.5, 1.14), borderaxespad=0., ncol=ncol,
              frameon=True)
    fig = ax.get_figure()
    if fig is not None:
        fig.tight_layout()


def _mott_background(ax, r: SolverResult) -> None:
    """
    Mott transition (physics.mott): shaded wherever local doping exceeds
    the composition-dependent Mott critical density -- a genuine
    threshold/delocalization criterion, so shown as an on/off region.

    Diagnostic overlay only -- see physics.mott's docstring for exactly
    what it does and doesn't model, and physics.self_consistent's docstring
    for what the solver's actual carrier-density physics uses.
    """
    N_mott_donor = donor_mott_density_cm3(r.x_Al)
    N_mott_acceptor = acceptor_mott_density_cm3(r.x_Al)
    mott_mask = (r.ND > N_mott_donor) | (r.NA > N_mott_acceptor)
    _shaded_spans(ax, r.x_nm, mott_mask, color='#9467bd',
                 label='Mott transition (doping > $N_{Mott}$)', alpha=0.16)


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def plot_band_diagram(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
    show_vacuum: bool = True,
    show_composition_bg: bool = True,
    show_mott: bool = True,
    show_valence_bands: bool = True,
    annotate: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """
    Panel 1: Ec, Ev (HH), Efn, Efp, Ei, optional vacuum level, and (if
    show_valence_bands) the LH/SO valence band edges alongside Ev -- the
    decoupled 3-band effective-mass model (see physics.materials.algan.
    AlGaNParams.valence_band_structure; physics.self_consistent solves each
    band's confined states independently, no HH-LH-SO mixing).
    Background is shaded by Al composition, and (if show_mott) by the
    Mott-transition diagnostic -- see physics.mott for what it does and
    doesn't model.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    x = r.x_nm
    Ev_lh = getattr(r, 'Ev_lh', None)
    Ev_so = getattr(r, 'Ev_so', None)
    have_valence_bands = show_valence_bands and Ev_lh is not None and Ev_so is not None

    y_min = float(np.min(r.Ev)) - 0.3
    y_max = float(np.max(r.Ec)) + 0.5
    if have_valence_bands:
        y_min = min(y_min, float(np.min(Ev_lh)) - 0.3, float(np.min(Ev_so)) - 0.3)
        y_max = max(y_max, float(np.max(Ev_lh)) + 0.3, float(np.max(Ev_so)) + 0.3)

    # Vacuum level = Ec + chi(x). chi (electron affinity) is several eV, so
    # E_vac sits well above Ec -- if it's being shown, the y-range computed
    # from Ec/Ev alone clips it off the top of the axis entirely.
    E_vac = None
    if show_vacuum:
        from physics.materials.algan import get_AlGaN_params
        chi_arr = np.array([get_AlGaN_params(xi).chi for xi in r.x_Al])
        E_vac = r.Ec + chi_arr
        y_max = max(y_max, float(np.max(E_vac)) + 0.2)

    if show_composition_bg:
        _composition_background(ax, x, r.x_Al, y_min, y_max)

    if show_mott:
        _mott_background(ax, r)

    ax.plot(x, r.Ec, color=_COLORS['Ec'],  lw=1.3, label='$E_c$')
    ax.plot(x, r.Ev, color=_COLORS['Ev'],  lw=1.3,
            label='$E_v$ (HH)' if have_valence_bands else '$E_v$')
    if have_valence_bands:
        ax.plot(x, Ev_lh, color=_COLORS['Ev_lh'], lw=1.1, label='$E_v$ (LH)')
        ax.plot(x, Ev_so, color=_COLORS['Ev_so'], lw=1.1, label='$E_v$ (SO)')
    ax.plot(x, r.Ei, color=_COLORS['Ei'],  lw=1.0, ls=':', label='$E_i$')

    # Quasi-Fermi levels (now arrays)
    # They might contain NaNs in some highly depleted regions, or just be plotted directly
    ax.plot(x, r.Efn, color=_COLORS['Efn'], lw=1.5, ls='--', label='$E_{fc}$')
    ax.plot(x, r.Efp, color=_COLORS['Efp'], lw=1.5, ls='--', label='$E_{fv}$')

    if show_vacuum:
        ax.plot(x, E_vac, color=_COLORS['vac'], lw=1.0, ls='-.',
                label='$E_{vac}$', alpha=0.7)

    # Surface/interface charge trap levels (devices.layer.SurfaceCharge):
    # a marker at each state's (x, Ec-energy or Ev+energy) so Fermi-level
    # pinning -- Efn/Efp settling onto this level once the state's areal
    # density is high enough to dominate local charge balance -- is
    # something you can see directly rather than infer.
    if annotate and getattr(r, 'surface_charge_markers', None):
        for xpos, level, state_type, density_cm2 in r.surface_charge_markers:
            color = '#1a9850' if state_type == 'donor' else '#762a83'
            ax.plot(xpos, level, marker='_', markersize=14, markeredgewidth=2.5, color=color, zorder=5)
            ax.annotate(f'{state_type[0].upper()} trap\n{density_cm2:.1e} cm$^{{-2}}$',
                        xy=(xpos, level), xytext=(6, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=6.5, color=color)

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Energy (eV)',    fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, y_max)
    _legend_right(ax, fontsize=8, ncol=1)
    if title:
        ax.set_title(title, fontsize=10)
    else:
        V_internal = getattr(r, 'V_internal', r.V_applied)
        # Series-resistance IR drop (physics.self_consistent's R_series
        # relaxation of V_internal) can leave the voltage actually applied
        # across the intrinsic device well below V_applied at high bias --
        # surfacing both here instead of just V_applied is what makes a
        # collapsed V_internal visible instead of a diagram that's
        # mysteriously flat despite being labeled at the requested bias.
        if abs(V_internal - r.V_applied) > 0.01:
            label = (f'Band Diagram  (V$_{{applied}}$ = {r.V_applied:.2f} V, '
                      f'V$_{{internal}}$ = {V_internal:.2f} V -- IR drop across '
                      f'R$_{{series}}$, T = {r.T:.0f} K)')
        else:
            label = f'Band Diagram  (V = {r.V_applied:.2f} V, T = {r.T:.0f} K)'
        ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    return ax


def plot_wavefunctions(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
    n_show_e: int = 3,
    n_show_h: int = 3,
    scale: float = 0.4,
) -> plt.Axes:
    """
    Panel 2: |ψ_n|² overlaid on band diagram, offset to their subband energy.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    # LH/SO band edges are hidden here: only the HH wavefunctions (psi_h)
    # are drawn below, so showing LH/SO curves with no matching
    # wavefunction overlay would just be visual clutter with no state to
    # anchor it to.
    ax = plot_band_diagram(r, ax=ax, show_vacuum=False, show_valence_bands=False,
                           annotate=False, title='Wavefunctions')

    if r.psi_e is None or r.E_e is None:
        ax.set_title('Wavefunctions (run with quantum=True)')
        return ax

    x = r.x_nm
    cmap_e = plt.get_cmap('Blues')
    cmap_h = plt.get_cmap('Reds')

    n_e = min(n_show_e, len(r.E_e))
    for k in range(n_e):
        psi2 = r.psi_e[k]**2
        En   = r.E_e[k]
        # Normalise for display
        psi2_norm = psi2 / (np.max(psi2) + 1e-30) * scale
        color = cmap_e(0.4 + 0.5 * k / max(n_e - 1, 1))
        ax.fill_between(x, En, En + psi2_norm, alpha=0.5, color=color)
        ax.axhline(En, color=color, lw=0.8, ls='-', alpha=0.8)
        ax.text(x[-1] * 0.98, En + 0.01, f'$e_{k+1}$={En:.3f} eV',
                ha='right', va='bottom', fontsize=7, color=color)

    n_h = min(n_show_h, len(r.E_h)) if r.E_h is not None else 0
    for k in range(n_h):
        psi2 = r.psi_h[k]**2
        # Hole energies: E_h measured downward from Ev; in band diagram: Ev - E_h
        Ev_avg = float(np.mean(r.Ev[r.Ev == np.max(r.Ev)]))
        En_plot = float(np.max(r.Ev)) - r.E_h[k]
        psi2_norm = psi2 / (np.max(psi2) + 1e-30) * scale
        color = cmap_h(0.4 + 0.5 * k / max(n_h - 1, 1))
        ax.fill_between(x, En_plot - psi2_norm, En_plot, alpha=0.5, color=color)
        ax.axhline(En_plot, color=color, lw=0.8, ls='-', alpha=0.8)
        ax.text(x[-1] * 0.98, En_plot - 0.01, f'$h_{k+1}$',
                ha='right', va='top', fontsize=7, color=color)

    if r.qcse_transition_eV is not None:
        ie, ih = r.qcse_pair if r.qcse_pair is not None else (0, 0)
        if r.qcse_in_well:
            well_tag = ' (in QW)'
        elif r.qw_window_nm is not None:
            well_tag = ' (global — no state confined in the detected QW)'
        else:
            well_tag = ' (global — no QW layer detected)'
        ax.text(
            0.02, 0.02,
            f'QCSE:  $E_{{e_{{{ie+1}}}h_{{{ih+1}}}}}$ = {r.qcse_transition_eV:.4f} eV   '
            f'|$\\int\\psi_e\\psi_h\\,dx$|$^2$ = {r.qcse_overlap * 100:.2f}%{well_tag}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
            color='#333333', bbox=dict(boxstyle='round', fc='white', alpha=0.75, ec='#cccccc'),
        )

    return ax


def plot_carriers(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Panel 3: Electron and hole carrier density on log scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    x = r.x_nm
    n_safe = np.maximum(r.n, 1.0)
    p_safe = np.maximum(r.p, 1.0)

    ax.semilogy(x, n_safe, color=_COLORS['n'], lw=1.8, label='$n$ (electrons)')
    ax.semilogy(x, p_safe, color=_COLORS['p'], lw=1.8, label='$p$ (holes)')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Carrier density (cm$^{-3}$)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    _legend_right(ax, fontsize=9)
    ax.set_title(f'Carrier Density  (V = {r.V_applied:.2f} V)', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, lw=0.5)
    return ax


def plot_fields(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
    units: str = 'MV/cm',
) -> plt.Axes:
    """Panel 4: Electrostatic field, quasi-electric field, and total field."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    x  = r.x_nm
    if units == 'MV/cm':
        scale = 1e-8   # V/m → MV/cm
        ylabel = 'Electric field (MV/cm)'
    else:
        scale = 1.0
        ylabel = 'Electric field (V/m)'

    F_elec  = r.E_field * scale
    F_quasi = r.F_quasi * scale
    F_total = (r.E_field + r.F_quasi) * scale

    ax.plot(x, F_elec,  color=_COLORS['Felec'],  lw=1.8, label='$F_{elec}$')
    ax.plot(x, F_quasi, color=_COLORS['Fquasi'], lw=1.5, ls='--', label='$F_{quasi}$')
    ax.plot(x, F_total, color=_COLORS['Ftot'],   lw=1.8, ls='-.',
            label='$F_{total}$', alpha=0.8)

    ax.axhline(0, color='black', lw=0.5, ls=':')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(x[0], x[-1])
    _legend_right(ax, fontsize=9)
    ax.set_title('Electric Fields', fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    return ax


def plot_polarization(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
    ax_bar: Optional[plt.Axes] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Panel 5: Psp, Ppz, P_total vs position + bar chart of interface sheet charges.
    Returns (ax_line, ax_bar).
    """
    if ax is None:
        fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(12, 4),
                                          gridspec_kw={'width_ratios': [3, 1]})

    x = r.x_nm
    ax.plot(x, r.Psp,     color=_COLORS['Psp'],  lw=1.8, label='$P_{sp}$')
    ax.plot(x, r.Ppz,     color=_COLORS['Ppz'],  lw=1.8, label='$P_{pz}$')
    ax.plot(x, r.P_total, color=_COLORS['Ptot'], lw=2.0, label='$P_{total}$')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Polarization (C/m²)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    # Placed above (not _legend_right) since ax_bar, when the caller
    # supplies one, sits immediately to ax's right and would collide with
    # a right-hand legend.
    _legend_above(ax, fontsize=9, ncol=3)
    ax.set_title('Polarization', fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)

    if ax_bar is not None and len(r.interface_sigmas) > 0:
        # Sheet charges in units of 10^13 cm^-2
        sigma_13 = [s / _q * 1e-13 for s in r.interface_sigmas]
        positions = [r.x_nm[i] for i in r.interface_indices]
        colors = ['#d62728' if s > 0 else '#1f77b4' for s in sigma_13]
        ax_bar.bar(range(len(sigma_13)), sigma_13, color=colors, edgecolor='black',
                   linewidth=0.7)
        ax_bar.axhline(0, color='black', lw=0.8)
        ax_bar.set_xticks(range(len(sigma_13)))
        ax_bar.set_xticklabels(
            [f'{p:.0f} nm' for p in positions], fontsize=8, rotation=45, ha='right'
        )
        ax_bar.set_ylabel('Sheet charge\n(10¹³ cm⁻²)', fontsize=9)
        ax_bar.set_title('Interface charges', fontsize=10)
        ax_bar.grid(True, axis='y', alpha=0.3)

    return ax, ax_bar


def plot_strain(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Panel 6: In-plane and out-of-plane strain vs position."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))

    x = r.x_nm
    ax.plot(x, r.eps_xx * 100, color=_COLORS['exx'], lw=1.8, label='$\\varepsilon_{xx}$ (%)')
    ax.plot(x, r.eps_zz * 100, color=_COLORS['ezz'], lw=1.8, label='$\\varepsilon_{zz}$ (%)')
    ax.axhline(0, color='black', lw=0.5, ls=':')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Strain (%)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    _legend_right(ax, fontsize=9)
    ax.set_title('Strain', fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    return ax


def plot_qcse(
    results: List[SolverResult],
    current_index: int = -1,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Quantum-Confined Stark Effect panel.

    Given a voltage sweep (multiple SolverResults), plots the e1-h1
    transition energy and electron-hole wavefunction overlap against
    applied bias -- the two standard experimental signatures of QCSE:
    as the internal field grows, the transition energy redshifts and the
    overlap (proxy for oscillator strength / radiative rate) falls; as an
    applied bias screens the built-in polarization field, both effects
    reverse.

    Given a single result (no sweep), reports the same two numbers as text
    since there's no bias axis to plot against.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    have = [r for r in results if r.qcse_transition_eV is not None]

    if len(results) <= 1 or len(have) <= 1:
        r = results[current_index] if results else None
        ax.axis('off')
        if r is None or r.qcse_transition_eV is None:
            ax.text(0.5, 0.5,
                    'No confined electron-hole pair found.\n'
                    'Run with Quantum enabled and an undoped quantum well layer.',
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
        else:
            ie, ih = r.qcse_pair if r.qcse_pair is not None else (0, 0)
            label = f'$e_{{{ie+1}}}$–$h_{{{ih+1}}}$'
            if r.qcse_in_well:
                lo, hi = r.qw_window_nm
                source = f'the quantum well at {lo:.1f}–{hi:.1f} nm'
            elif r.qw_window_nm is not None:
                lo, hi = r.qw_window_nm
                source = (f'the global ground state — a quantum well was detected at '
                          f'{lo:.1f}–{hi:.1f} nm, but no solved subband is confined '
                          f'there; the electron/hole ground states are instead trapped in a '
                          f'different polarization/doping notch elsewhere in the device')
            else:
                source = ('the global ground state — no undoped, locally-narrower-bandgap '
                          'layer was found in this device, so there is no well to restrict to')
            ax.text(0.5, 0.74, f'{label} transition energy:  {r.qcse_transition_eV:.4f} eV',
                    ha='center', fontsize=12, transform=ax.transAxes)
            ax.text(0.5, 0.60,
                    f'{label} wavefunction overlap  |$\\int\\psi_e\\psi_h\\,dx$|$^2$:  '
                    f'{r.qcse_overlap * 100:.2f}%',
                    ha='center', fontsize=12, transform=ax.transAxes)
            ax.text(0.5, 0.48, f'({source})', ha='center', fontsize=8,
                    color='#888888', transform=ax.transAxes, wrap=True)
            if (r.qcse_dominant_pair is not None
                    and r.qcse_dominant_pair != (ie, ih)
                    and r.qcse_dominant_overlap is not None
                    and r.qcse_dominant_overlap > (r.qcse_overlap or 0.0) * 1.5):
                die, dih = r.qcse_dominant_pair
                ax.text(
                    0.5, 0.32,
                    f'Note: highest-overlap pair overall is $e_{{{die+1}}}$–$h_{{{dih+1}}}$: '
                    f'{r.qcse_dominant_transition_eV:.4f} eV, '
                    f'{r.qcse_dominant_overlap * 100:.2f}% overlap.',
                    ha='center', fontsize=8.5, color='#a55a00', transform=ax.transAxes)
            ax.text(0.5, 0.12, 'Run a voltage sweep to see the Stark shift vs bias.',
                    ha='center', fontsize=9, color='#888888', transform=ax.transAxes)
        ax.set_title('Quantum-Confined Stark Effect', fontsize=10)
        return ax

    V   = np.array([r.V_applied for r in have])
    E_t = np.array([r.qcse_transition_eV for r in have])
    ov  = np.array([r.qcse_overlap * 100.0 for r in have])
    order = np.argsort(V)
    V, E_t, ov = V[order], E_t[order], ov[order]

    ax.plot(V, E_t, color=_COLORS['Ec'], lw=2.0, marker='o', ms=4,
             label='$E_{e-h}$ (QW transition energy)')
    ax.set_xlabel('Applied bias (V)', fontsize=11)
    ax.set_ylabel('Transition energy (eV)', color=_COLORS['Ec'], fontsize=11)
    ax.tick_params(axis='y', labelcolor=_COLORS['Ec'])
    ax.grid(True, alpha=0.3, lw=0.5)

    if results and 0 <= current_index < len(results):
        r_cur = results[current_index]
        if r_cur.qcse_transition_eV is not None:
            ax.axvline(r_cur.V_applied, color='#888888', lw=0.8, ls=':')

    ax2 = ax.twinx()
    ax2.plot(V, ov, color=_COLORS['p'], lw=1.8, ls='--', marker='s', ms=4,
              label='$e$-$h$ overlap')
    ax2.set_ylabel('$e$-$h$ wavefunction overlap (%)', color=_COLORS['p'], fontsize=11)
    ax2.tick_params(axis='y', labelcolor=_COLORS['p'])
    ax2.set_ylim(0, max(100.0, float(np.max(ov)) * 1.15) if len(ov) else 100.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Above, not _legend_right: ax2 (twinx) already owns the right margin
    # for its own y-axis label/ticks.
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
              loc='lower center', bbox_to_anchor=(0.5, 1.14),
              borderaxespad=0., ncol=2, frameon=True)
    ax.get_figure().tight_layout()

    ax.set_title('Quantum-Confined Stark Effect vs Bias', fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# Combined 3×2 figure
# ---------------------------------------------------------------------------

def plot_all(
    r: SolverResult,
    save_path: Optional[str] = None,
    figsize: tuple = (18, 12),
    dpi: int = 120,
) -> plt.Figure:
    """
    3×2 panel figure: band diagram, wavefunctions, carriers,
    fields, polarization, strain.

    Parameters
    ----------
    save_path : if given, save figure to this file path (PNG, PDF, etc.)
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(
        f'AlGaN Band Diagram  —  V = {r.V_applied:.2f} V,  T = {r.T:.0f} K',
        fontsize=13, fontweight='bold',
    )

    # Row 1
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    # Row 2
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    # Row 3
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    plot_band_diagram(r, ax=ax1)
    plot_wavefunctions(r, ax=ax2)
    plot_carriers(r, ax=ax3)
    plot_fields(r, ax=ax4)
    plot_polarization(r, ax=ax5, ax_bar=None)  # line only in combined view
    plot_strain(r, ax=ax6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig

"""
Band diagram visualization for AlGaN heterostructures.

BandDiagramPlotter wraps a SolverResult and produces publication-quality figures.

Panels available:
  plot_band_diagram()   — Ec, Ev, Efn, Efp, Ei, vacuum level + composition
                           background, and Mott-transition (physics.mott)
                           diagnostic shading (toggle: show_mott)
  plot_wavefunctions()  — |psi_n|^2 overlaid on band diagram
  plot_carriers()       — n(x) and p(x) on log scale
  plot_fields()         — electric field (quasi-electric field shown for reference)
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
    'Ev':   '#2ca02c',   # green (Ev = topmost valence band edge)
    'Ev_hh': '#17becf',  # cyan -- heavy-hole edge (== Ev in GaN-rich layers)
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
                             cmap: str = 'Blues', alpha: float = 0.15,
                             x_In: Optional[np.ndarray] = None) -> None:
    """Shade background by composition: Al fraction in blue; where In is
    present (InGaN / InAlGaN) an orange overlay scaled by x_In."""
    if x_In is not None and np.any(np.asarray(x_In) > 0):
        _composition_background(ax, x_nm, x_Al, y_min, y_max, cmap, alpha)
        cmap_in = plt.get_cmap('Oranges')
        xi = np.asarray(x_In)
        for i in range(len(x_nm) - 1):
            if xi[i] <= 0:
                continue
            ax.add_patch(plt.Polygon(
                [[x_nm[i], y_min], [x_nm[i + 1], y_min], [x_nm[i + 1], y_max], [x_nm[i], y_max]],
                closed=True, facecolor=cmap_in(0.35 + 0.65 * min(1.0, xi[i] / 0.4)),
                edgecolor='none', alpha=alpha * 1.6))
        return
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
    comp = (r.x_Al, getattr(r, 'x_In', None), getattr(r, 'x_P', None), getattr(r, 'crystal', 'wurtzite'))
    N_mott_donor = donor_mott_density_cm3(*comp)
    N_mott_acceptor = acceptor_mott_density_cm3(*comp)
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
    Panel 1: Ec, Ev (topmost valence band), Efn, Efp, Ei, optional vacuum
    level, and (if show_valence_bands) the HH/LH/SO valence band edges -- the
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
    Ev_hh = getattr(r, 'Ev_hh', None)
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
        chi_arr = (np.asarray(r.chi) if getattr(r, 'chi', None) is not None
                   else np.array([get_AlGaN_params(xi).chi for xi in r.x_Al]))
        E_vac = r.Ec + chi_arr
        y_max = max(y_max, float(np.max(E_vac)) + 0.2)

    if show_composition_bg:
        _composition_background(ax, x, r.x_Al, y_min, y_max, x_In=getattr(r, 'x_In', None))

    if show_mott:
        _mott_background(ax, r)

    ax.plot(x, r.Ec, color=_COLORS['Ec'],  lw=1.3, label='$E_c$')
    ax.plot(x, r.Ev, color=_COLORS['Ev'],  lw=1.3,
            label='$E_v$ (top)' if have_valence_bands else '$E_v$')
    if have_valence_bands:
        if Ev_hh is not None:
            ax.plot(x, Ev_hh, color=_COLORS['Ev_hh'], lw=1.1, label='$E_v$ (HH)')
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
        # A NON-converged result can also differ: the drift-diffusion
        # ramp stalled and V_internal is the bias the shown state reached.
        if abs(V_internal - r.V_applied) > 0.01 and not r.converged:
            label = (f'Band Diagram  (NOT CONVERGED at V$_{{applied}}$ = {r.V_applied:.2f} V; '
                      f'state shown is at V = {V_internal:.2f} V, T = {r.T:.0f} K)')
        elif abs(V_internal - r.V_applied) > 0.01:
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
    show_composition_bg: bool = True,
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
                           annotate=False, title='Wavefunctions',
                           show_composition_bg=show_composition_bg)

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
            well_tag = (' (in QW, local well solve)' if getattr(r, 'qcse_local_solve', False)
                        else ' (in QW)')
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


def recombination_currents(r: SolverResult) -> Optional[dict]:
    """Recombination integrated over the device, as current densities
    [A/cm^2]: {'srh', 'rad', 'aug', 'total'} plus 'iqe', the radiative share
    of the total. None if the result carries no rates. Positive = net
    recombination; under forward bias 'total' is the part of the current
    that recombines inside the structure rather than reaching a contact."""
    if getattr(r, 'R_srh', None) is None or r.R_rad is None or r.R_aug is None:
        return None
    x_cm = np.asarray(r.x_nm, dtype=float) * 1e-7
    q = 1.602176634e-19
    integ = getattr(np, 'trapezoid', None) or np.trapz
    out = {k: float(q * integ(np.asarray(v, dtype=float), x_cm))
           for k, v in (('srh', r.R_srh), ('rad', r.R_rad), ('aug', r.R_aug))}
    out['total'] = out['srh'] + out['rad'] + out['aug']
    out['iqe'] = out['rad'] / out['total'] if out['total'] > 0 else float('nan')
    return out


def plot_recombination(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Shockley-Read-Hall, radiative and Auger recombination rates on a log
    scale. Where a rate is negative (net generation, n p < ni^2) its
    magnitude is drawn dashed."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    x = r.x_nm
    if getattr(r, 'R_srh', None) is None:
        ax.text(0.5, 0.5, 'No recombination rates in this result', ha='center', va='center',
                transform=ax.transAxes)
        return ax
    total = r.R_srh + r.R_rad + r.R_aug
    peak = float(np.max(np.abs(total))) if len(total) else 0.0
    floor = max(peak * 1e-12, 1e-30)
    for key, label, colour, lw in (('R_srh', 'Shockley-Read-Hall', _COLORS['Ppz'], 1.6),
                                   ('R_rad', 'Radiative', _COLORS['Psp'], 1.6),
                                   ('R_aug', 'Auger', _COLORS['Ptot'], 1.6),
                                   (None, 'Total', _COLORS['Ei'], 1.0)):
        y = total if key is None else np.asarray(getattr(r, key), dtype=float)
        pos = np.where(y > floor, y, np.nan)
        neg = np.where(y < -floor, -y, np.nan)
        style = dict(color=colour, lw=lw, alpha=0.6 if key is None else 1.0)
        ax.semilogy(x, pos, label=label, **style)
        if np.any(np.isfinite(neg)):
            ax.semilogy(x, neg, ls='--', **style)
    if peak > 0:
        ax.set_ylim(max(peak * 1e-10, 1e-30), peak * 5.0)
    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Recombination rate (cm$^{-3}$ s$^{-1}$)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    _legend_right(ax, fontsize=9)
    ax.set_title(f'Recombination  (V = {r.V_applied:.2f} V; dashed = net generation)', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, lw=0.5)
    return ax


def plot_fields(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
    units: str = 'MV/cm',
) -> plt.Axes:
    """Panel 4: the electric field. The quasi-electric field of a composition
    gradient is drawn alongside for reference only; it is not added to the field."""
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

    ax.plot(x, F_elec,  color=_COLORS['Felec'],  lw=1.8, label='$F$')
    ax.plot(x, F_quasi, color=_COLORS['Fquasi'], lw=1.5, ls='--', label='$F_{quasi}$ (reference only)')

    ax.axhline(0, color='black', lw=0.5, ls=':')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(x[0], x[-1])
    # Scale the axis to the electric field. The reference curve is included
    # only where it is smooth (a graded layer): at an abrupt interface it is
    # a one-node spike that would otherwise flatten the field itself.
    lo, hi = float(np.min(F_elec)), float(np.max(F_elec))
    q_lo, q_hi = np.percentile(F_quasi, [2, 98])
    lo, hi = min(lo, float(q_lo)), max(hi, float(q_hi))
    pad = 0.08 * (hi - lo) if hi > lo else 0.5
    ax.set_ylim(lo - pad, hi + pad)
    _legend_right(ax, fontsize=9)
    ax.set_title('Electric Field', fontsize=10)
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


# ---------------------------------------------------------------------------
# Reciprocal space map (physics.rsm)
# ---------------------------------------------------------------------------

def plot_rsm(rsm, ax: Optional[plt.Axes] = None, decades: float = 6.0) -> plt.Axes:
    """
    Reciprocal space map from physics.rsm.simulate_rsm: log intensity over
    (Q_x, Q_z), the pseudomorphic line through the bottom layer's rod, and
    the fully relaxed position of every alloy in the stack, numbered and
    listed in a key. A symmetric reflection has all layers on Q_x = 0, so it
    is drawn as the line scan along Q_z instead.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    fig = ax.get_figure()
    key = '\n'.join(f'{i + 1}   {name}' for i, (name, _qx, _qz) in enumerate(rsm.relaxed_points))
    box = dict(boxstyle='round,pad=0.5', fc='white', ec='#cfd4dc', lw=0.6, alpha=0.92)

    if rsm.symmetric:
        scan = rsm.intensity[:, rsm.intensity.shape[1] // 2]
        ax.semilogy(rsm.qz, np.maximum(scan, 10.0 ** (-decades)), color=_COLORS['Ec'], lw=1.2)
        for i, (_name, _qx, qz) in enumerate(rsm.relaxed_points):
            ax.axvline(qz, color='#8892a0', lw=0.6, ls=':')
            ax.text(qz, 3.0 * (3.2 if i % 2 else 1.0), str(i + 1), ha='center', va='bottom',
                    fontsize=7.5, color='#17202c')
        ax.text(0.015, 0.97, 'Relaxed alloys\n' + key, transform=ax.transAxes, ha='left', va='top',
                fontsize=7.5, color='#17202c', bbox=box, linespacing=1.4)
        ax.set_xlim(rsm.qz[0], rsm.qz[-1])
        ax.set_ylim(10.0 ** (-decades), 60.0)
        ax.set_xlabel('$Q_z$ (Å$^{-1}$)', fontsize=11)
        ax.set_ylabel('Intensity (normalised)', fontsize=11)
        ax.set_title(f'Scan along $Q_z$, {rsm.label} reflection', fontsize=10)
        ax.grid(True, which='major', alpha=0.3, lw=0.5)
        return ax

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'rsm', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#104281', '#0d366b'])
    logi = np.log10(np.maximum(rsm.intensity, 10.0 ** (-decades)))
    mesh = ax.pcolormesh(rsm.qx, rsm.qz, logi, cmap=cmap, vmin=-decades, vmax=0.0,
                         shading='auto', rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label('log$_{10}$ intensity (normalised)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.axvline(rsm.qx_substrate, color='#566273', lw=0.7, ls='--', label='pseudomorphic to the bottom layer')
    for i, (_name, qx, qz) in enumerate(rsm.relaxed_points):
        ax.plot(qx, qz, marker='o', ms=5, mfc='none', mec='#e34948', mew=1.0, ls='none',
                label='fully relaxed alloy (numbered)' if i == 0 else None)
        ax.annotate(str(i + 1), xy=(qx, qz), xytext=(7 if i % 2 == 0 else -7, 0),
                    textcoords='offset points', ha='left' if i % 2 == 0 else 'right', va='center',
                    fontsize=7.5, color='#17202c')
    ax.text(0.015, 0.97, key, transform=ax.transAxes, ha='left', va='top',
            fontsize=7.5, color='#17202c', bbox=box, linespacing=1.4)
    ax.set_xlim(rsm.qx[0], rsm.qx[-1])
    ax.set_ylim(rsm.qz[0], rsm.qz[-1])
    ax.set_xlabel('$Q_x$ (Å$^{-1}$)', fontsize=11)
    ax.set_ylabel('$Q_z$ (Å$^{-1}$)', fontsize=11)
    ax.set_title(f'Reciprocal space map, {rsm.label} reflection', fontsize=10)
    ax.legend(fontsize=8, loc='lower left', frameon=True)
    return ax


# ---------------------------------------------------------------------------
# Growth under illumination (physics.dqfl)
# ---------------------------------------------------------------------------

def plot_illumination(res, fig) -> None:
    """
    physics.dqfl.simulate_illumination, layer by layer along the growth
    axis: the photovoltage each layer develops while it is the growth
    surface, and the factor by which each compensating defect's charged
    state is reduced. Drawn into `fig` as two panels on a shared axis.
    """
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    hues = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']
    names = []
    for layer in res.layers:
        ax1.plot([layer.z0_nm, layer.z1_nm], [layer.Ep_eV, layer.Ep_eV], color='#17202c', lw=1.4,
                 solid_capstyle='butt')
        for d in layer.defects:
            if d.compensator.name not in names:
                names.append(d.compensator.name)
    for layer in res.layers:
        for d in layer.defects:
            k = names.index(d.compensator.name)
            ax2.plot([layer.z0_nm, layer.z1_nm], [d.factor, d.factor], color=hues[k % len(hues)], lw=1.6,
                     solid_capstyle='butt', label=d.compensator.name)
    handles, labels = ax2.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    if seen:
        ax2.legend(seen.values(), seen.keys(), fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0)
        ax2.set_yscale('log')
        lo = min(d.factor for layer in res.layers for d in layer.defects)
        ax2.set_ylim(min(lo / 3.0, 0.1), 3.0)
    else:
        ax2.text(0.5, 0.5, 'No doped layers: nothing is compensated.', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=9, color='#566273')
    ax2.axhline(1.0, color='#8892a0', lw=0.6, ls=':')
    z_max = max((layer.z1_nm for layer in res.layers), default=1.0)
    ax2.set_xlim(0.0, z_max)
    ax1.set_ylim(0.0, max([layer.Ep_eV for layer in res.layers] + [0.1]) * 1.15)
    ax1.set_ylabel('Photovoltage $E_{Fn}-E_{Fp}$ (eV)', fontsize=10)
    ax2.set_ylabel('Charged defects: with light / without', fontsize=10)
    ax2.set_xlabel('Position (nm)', fontsize=11)
    ax1.set_title(f'Growth under illumination: {res.wavelength_nm:g} nm ({res.photon_eV:.2f} eV), '
                  f'{res.power_W_cm2:g} W/cm$^2$, {res.T_K - 273.15:.0f} °C'
                  '  (each layer as the free growth surface; no contacts)', fontsize=10)
    for layer in res.layers:                 # layers the lamp does not reach
        if layer.absorption_cm == 0.0:
            ax1.axvspan(layer.z0_nm, layer.z1_nm, facecolor='none', edgecolor='#b7bec9', hatch='///', lw=0.0)
    if any(layer.absorption_cm == 0.0 for layer in res.layers):
        ax1.text(0.99, 0.95, 'hatched: gap wider than the photon energy, not absorbed', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=7.5, color='#566273')
    for ax in (ax1, ax2):
        ax.grid(True, which='major', alpha=0.3, lw=0.5)


# ---------------------------------------------------------------------------
# Bands under illumination through the top surface (physics.illumination)
# ---------------------------------------------------------------------------

def plot_illuminated_bands(res, fig) -> None:
    """
    physics.illumination.solve_illuminated: the band diagram in the dark
    (dashed grey) and under light at open circuit (colour), and below it the
    carrier densities the same way. The light enters at the right-hand end,
    the top surface.
    """
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    d, l = res.dark, res.light
    x = np.asarray(l.x_nm)
    grey = '#8892a0'
    ax1.plot(x, d.Ec, color=grey, lw=0.9, ls=(0, (4, 2)), label='dark: $E_c$, $E_v$')
    ax1.plot(x, d.Ev, color=grey, lw=0.9, ls=(0, (4, 2)))
    ax1.plot(x, d.Efn, color=grey, lw=0.8, ls=(0, (1, 2)), label='dark: $E_F$')
    ax1.plot(x, l.Ec, color=_COLORS['Ec'], lw=1.3, label='light: $E_c$')
    ax1.plot(x, l.Ev, color=_COLORS['Ev'], lw=1.3, label='light: $E_v$')
    ax1.plot(x, l.Efn, color='#17202c', lw=1.0, ls=(0, (7, 3)), label='light: $E_{Fn}$')
    ax1.plot(x, l.Efp, color='#6b7483', lw=1.0, ls=(0, (2, 2)), label='light: $E_{Fp}$')
    ax1.set_ylabel('Energy (eV)', fontsize=11)
    ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax1.text(0.99, 0.96, '← light enters here (top surface)', transform=ax1.transAxes, ha='right', va='top',
             fontsize=8, color='#566273')
    state = 'open circuit' if res.converged else 'NOT CONVERGED'
    ax1.set_title(f'Bands under {res.wavelength_nm:g} nm light ({res.photon_eV:.2f} eV), {res.power_W_cm2:g} W/cm$^2$, '
                  f'{l.T - 273.15:.0f} °C: {state}, photovoltage {res.photovoltage_V:+.3g} V', fontsize=10)

    floor = 1.0
    ax2.semilogy(x, np.maximum(d.n, floor), color=grey, lw=0.9, ls=(0, (4, 2)), label='dark')
    ax2.semilogy(x, np.maximum(d.p, floor), color=grey, lw=0.9, ls=(0, (4, 2)))
    ax2.semilogy(x, np.maximum(l.n, floor), color=_COLORS['n'], lw=1.3, label='light: $n$')
    ax2.semilogy(x, np.maximum(l.p, floor), color=_COLORS['p'], lw=1.3, label='light: $p$')
    ax2.set_ylabel('Carriers (cm$^{-3}$)', fontsize=11)
    ax2.set_xlabel('Position (nm)', fontsize=11)
    ax2.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax2.set_xlim(x[0], x[-1])
    for ax in (ax1, ax2):
        ax.grid(True, which='major', alpha=0.3, lw=0.5)

"""
Band diagram visualization for AlGaN heterostructures.

BandDiagramPlotter wraps a SolverResult and produces publication-quality figures.

Panels available:
  plot_band_diagram()   — Ec, Ev, Efn, Efp, Ei, vacuum level + composition background
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
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import List, Optional

from physics.self_consistent import SolverResult
from physics.constants import q as _q


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_COLORS = {
    'Ec':   '#1f77b4',   # blue
    'Ev':   '#2ca02c',   # green
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


def _interface_lines(ax, r: SolverResult, y_min: float, y_max: float,
                     color: str = '#555555', lw: float = 0.6) -> None:
    """Draw vertical dashed lines at abrupt heterointerfaces."""
    for idx in r.interface_indices:
        xpos = r.x_nm[idx]
        ax.axvline(xpos, color=color, lw=lw, ls='--', alpha=0.6)


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def plot_band_diagram(
    r: SolverResult,
    ax: Optional[plt.Axes] = None,
    show_vacuum: bool = True,
    show_composition_bg: bool = True,
    annotate: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """
    Panel 1: Ec, Ev, Efn, Efp, Ei, and optional vacuum level.
    Background is shaded by Al composition.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    x = r.x_nm

    y_min = float(np.min(r.Ev)) - 0.3
    y_max = float(np.max(r.Ec)) + 0.5

    if show_composition_bg:
        _composition_background(ax, x, r.x_Al, y_min, y_max)

    _interface_lines(ax, r, y_min, y_max)

    ax.plot(x, r.Ec, color=_COLORS['Ec'],  lw=2.0, label='$E_c$')
    ax.plot(x, r.Ev, color=_COLORS['Ev'],  lw=2.0, label='$E_v$')
    ax.plot(x, r.Ei, color=_COLORS['Ei'],  lw=1.0, ls=':', label='$E_i$')

    # Quasi-Fermi levels (now arrays)
    # They might contain NaNs in some highly depleted regions, or just be plotted directly
    ax.plot(x, r.Efn, color=_COLORS['Efn'], lw=1.5, ls='--', label='$E_{fn}$')
    ax.plot(x, r.Efp, color=_COLORS['Efp'], lw=1.5, ls='--', label='$E_{fp}$')

    if show_vacuum:
        # Vacuum level = Ec + chi(x)
        from physics.materials.algan import get_AlGaN_params
        chi_arr = np.array([get_AlGaN_params(xi).chi for xi in r.x_Al])
        E_vac = r.Ec + chi_arr
        ax.plot(x, E_vac, color=_COLORS['vac'], lw=1.0, ls='-.',
                label='$E_{vac}$', alpha=0.7)

    if annotate and len(r.interface_indices) > 0:
        for idx, sigma in zip(r.interface_indices, r.interface_sigmas):
            if abs(sigma) > 1e-3:
                xpos = r.x_nm[idx]
                label = f'σ={sigma*1e2:.1f} mC/m²'
                ax.annotate(label, xy=(xpos, y_max - 0.2),
                            ha='center', fontsize=7, color='#333333',
                            rotation=90, va='top')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Energy (eV)',    fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.set_title(title or f'Band Diagram  (V = {r.V_applied:.2f} V, T = {r.T:.0f} K)',
                 fontsize=10)
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

    ax = plot_band_diagram(r, ax=ax, show_vacuum=False, annotate=False,
                           title='Wavefunctions')

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
        ax.text(
            0.02, 0.02,
            f'QCSE:  $E_{{e_1h_1}}$ = {r.qcse_transition_eV:.4f} eV   '
            f'|$\\int\\psi_e\\psi_h\\,dx$|$^2$ = {r.qcse_overlap * 100:.2f}%',
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

    _interface_lines(ax, r, 0, 1e25)

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Carrier density (cm$^{-3}$)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=9)
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

    _interface_lines(ax, r, min(F_total.min(), -0.01), max(F_total.max(), 0.01))
    ax.axhline(0, color='black', lw=0.5, ls=':')

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=9)
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

    _interface_lines(ax, r, r.P_total.min() - 0.002, r.P_total.max() + 0.002)
    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Polarization (C/m²)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=9)
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

    _interface_lines(ax, r, min(r.eps_zz.min(), r.eps_xx.min()) * 100 - 0.05,
                     max(r.eps_zz.max(), r.eps_xx.max()) * 100 + 0.05)

    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Strain (%)', fontsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=9)
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
            ax.text(0.5, 0.68, f'$e_1$–$h_1$ transition energy:  {r.qcse_transition_eV:.4f} eV',
                    ha='center', fontsize=12, transform=ax.transAxes)
            ax.text(0.5, 0.54,
                    f'$e_1$–$h_1$ wavefunction overlap  |$\\int\\psi_e\\psi_h\\,dx$|$^2$:  '
                    f'{r.qcse_overlap * 100:.2f}%',
                    ha='center', fontsize=12, transform=ax.transAxes)
            if (r.qcse_dominant_pair is not None
                    and r.qcse_dominant_pair != (0, 0)
                    and r.qcse_dominant_overlap is not None
                    and r.qcse_dominant_overlap > (r.qcse_overlap or 0.0) * 1.5):
                ie, ih = r.qcse_dominant_pair
                ax.text(
                    0.5, 0.38,
                    f'Note: $e_1$/$h_1$ are localised in different notches (low overlap).\n'
                    f'Best-overlap pair is $e_{{{ie+1}}}$–$h_{{{ih+1}}}$: '
                    f'{r.qcse_dominant_transition_eV:.4f} eV, '
                    f'{r.qcse_dominant_overlap * 100:.2f}% overlap.',
                    ha='center', fontsize=8.5, color='#a55a00', transform=ax.transAxes)
            ax.text(0.5, 0.16, 'Run a voltage sweep to see the Stark shift vs bias.',
                    ha='center', fontsize=9, color='#888888', transform=ax.transAxes)
        ax.set_title('Quantum-Confined Stark Effect', fontsize=10)
        return ax

    V   = np.array([r.V_applied for r in have])
    E_t = np.array([r.qcse_transition_eV for r in have])
    ov  = np.array([r.qcse_overlap * 100.0 for r in have])
    order = np.argsort(V)
    V, E_t, ov = V[order], E_t[order], ov[order]

    ax.plot(V, E_t, color=_COLORS['Ec'], lw=2.0, marker='o', ms=4,
             label='$E_{e_1h_1}$ (transition energy)')
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
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

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

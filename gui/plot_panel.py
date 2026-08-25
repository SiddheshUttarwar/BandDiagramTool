"""
PlotPanel: tabbed matplotlib views (reusing visualization.plotter panel
functions unchanged) plus a sweep-index slider for scrubbing cached
SolverResults from a voltage sweep, a "Zoom to layer" control, and PNG
export.

Every visualization.plotter panel function sets its own x-axis to the full
device width on every call, so a thin layer (e.g. a 3nm quantum well
sitting inside a 300nm device) is squeezed into a handful of pixels and its
band-bending / wavefunction shape is essentially invisible — and since the
panel functions re-set xlim on every redraw (new solve, slider move, tab
switch), a manual toolbar zoom gets wiped out immediately. The "Zoom"
control here fixes that generically for every tab: it stores the selected
layer's (xmin, xmax) and re-applies it *after* calling the panel function,
so it survives redraws until the user changes it back to "Full device".
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from physics.self_consistent import SolverResult
from visualization.plotter import (
    plot_band_diagram, plot_wavefunctions, plot_carriers,
    plot_fields, plot_polarization, plot_strain,
)
from gui.layer_stack import _layer_summary

_PANELS = [
    ("Band Diagram", plot_band_diagram),
    # plot_wavefunctions only draws its first 3 states by default. A device
    # can have a deep polarization-induced notch (e.g. at a doped/undoped
    # interface) that's a genuinely lower-energy state than an intended
    # quantum well, pushing the QW's own state well past index 3 — so draw
    # more of them by default; combine with "Zoom to layer" to inspect a
    # specific one.
    ("Wavefunctions", lambda r, ax=None: plot_wavefunctions(r, ax=ax, n_show_e=16, n_show_h=16)),
    ("Carriers", plot_carriers),
    ("Fields", plot_fields),
    ("Polarization", lambda r, ax=None: plot_polarization(r, ax=ax)[0]),
    ("Strain", plot_strain),
]

_FULL_DEVICE = "Full device"

# Which SolverResult arrays determine the y-range for each tab, used to
# tighten the y-axis to what's actually visible in a zoomed x-window
# (matplotlib's own relim()/autoscale_view() don't support the fill_between
# PolyCollections these panels use, and would autoscale from the *full*
# device's data regardless of the current xlim, not just the visible slice).
_Y_FIELDS = {
    "Band Diagram": ["Ec", "Ev", "Ei", "Efn", "Efp"],
    "Wavefunctions": ["Ec", "Ev", "Ei", "Efn", "Efp"],
    "Carriers": ["n", "p"],
    "Fields": ["E_field", "F_quasi"],
    "Polarization": ["Psp", "Ppz", "P_total"],
    "Strain": ["eps_xx", "eps_zz"],
}


class PlotPanel(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.results: List[SolverResult] = []
        self.current_index = 0
        self._layer_ranges: List[Tuple[str, float, float]] = []  # (label, xmin, xmax)

        self._status_var = tk.StringVar(value="No solve yet.")
        status_bar = ttk.Frame(self)
        status_bar.pack(fill="x", side="bottom", padx=6, pady=4)
        ttk.Label(status_bar, textvariable=self._status_var).pack(side="left")
        ttk.Button(status_bar, text="Export PNG…", command=self._export_png).pack(side="right")

        zoom_frame = ttk.Frame(self)
        zoom_frame.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(zoom_frame, text="Zoom:").pack(side="left")
        self._zoom_var = tk.StringVar(value=_FULL_DEVICE)
        self._zoom_combo = ttk.Combobox(zoom_frame, textvariable=self._zoom_var,
                                         state="readonly", values=[_FULL_DEVICE], width=48)
        self._zoom_combo.pack(side="left", padx=6)
        self._zoom_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_current())
        ttk.Label(zoom_frame, text="(pick a layer to see thin regions like a quantum well clearly)",
                  foreground="#888888").pack(side="left", padx=6)

        self._slider_frame = ttk.Frame(self)
        self._slider_label = ttk.Label(self._slider_frame, text="", width=40)
        self._slider_label.pack(side="left", padx=(6, 4))
        self._slider = ttk.Scale(self._slider_frame, from_=0, to=0, orient="horizontal",
                                  command=self._on_slider)
        self._slider.pack(side="left", fill="x", expand=True, padx=6, pady=4)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True)

        self._tabs = {}
        for name, fn in _PANELS:
            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=name)
            fig = plt.Figure(figsize=(8, 5.5), dpi=100)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._tabs[name] = (fig, ax, canvas, fn)

        self._notebook.bind("<<NotebookTabChanged>>", lambda e: self._redraw_current())

    # ------------------------------------------------------------------
    def set_status(self, text: str):
        self._status_var.set(text)

    def show_result(self, result: SolverResult, layers: Optional[list] = None):
        self.results = [result]
        self.current_index = 0
        self._slider_frame.pack_forget()
        self._update_zoom_choices(layers)
        self._redraw_current()

    def show_sweep(self, results: List[SolverResult], layers: Optional[list] = None):
        if not results:
            return
        self.results = results
        self.current_index = len(results) - 1
        self._slider.configure(from_=0, to=max(0, len(results) - 1))
        self._slider.set(self.current_index)
        self._slider_frame.pack(fill="x", before=self._notebook)
        self._update_slider_label()
        self._update_zoom_choices(layers)
        self._redraw_current()

    def _update_zoom_choices(self, layers: Optional[list]):
        """Rebuild the Zoom dropdown from the current device's layers (in
        the bottom->top order devices.device.AlGaNDevice uses), keeping the
        current selection if a layer of the same label still exists."""
        previous = self._zoom_var.get()
        self._layer_ranges = []
        if layers:
            start = 0.0
            for i, layer in enumerate(layers):
                thickness = layer.thickness_nm
                end = start + thickness
                title, _ = _layer_summary(layer)
                label = f"#{i + 1} {title} ({start:.1f}–{end:.1f} nm)"
                self._layer_ranges.append((label, start, end))
                start = end

        values = [_FULL_DEVICE] + [label for label, _, _ in self._layer_ranges]
        self._zoom_combo.configure(values=values)
        self._zoom_var.set(previous if previous in values else _FULL_DEVICE)

    def _on_slider(self, value):
        idx = int(round(float(value)))
        if idx == self.current_index:
            return
        self.current_index = idx
        self._update_slider_label()
        self._redraw_current()

    def _update_slider_label(self):
        if not self.results:
            return
        r = self.results[self.current_index]
        conv = "converged" if r.converged else "NOT converged"
        self._slider_label.configure(
            text=f"V = {r.V_applied:.3f} V  ({self.current_index + 1}/{len(self.results)}, {conv})")

    def _redraw_current(self):
        if not self.results:
            return
        result = self.results[self.current_index]
        name = self._current_tab_name()
        if name is None:
            return
        fig, ax, canvas, fn = self._tabs[name]
        ax.clear()
        try:
            fn(result, ax=ax)
            self._apply_zoom(ax, result, name)
        except Exception as exc:  # noqa: BLE001 - never let a plot crash the app
            ax.text(0.5, 0.5, f"Could not render this panel:\n{exc}",
                     ha="center", va="center", transform=ax.transAxes, wrap=True)
        canvas.draw_idle()

    def _apply_zoom(self, ax, result: SolverResult, tab_name: str):
        """Re-apply the selected layer's x-range after the panel function
        has already set (and would otherwise leave) the full-device xlim,
        then tighten the y-axis to what's actually in view there — a thin
        quantum well's band bending / confined-state wavefunction hump is
        only ~0.1-0.5 eV against a multi-eV full-device y-range, and stays
        an invisible sliver without this."""
        # Subband-energy / interface-charge text labels place themselves at
        # full-device x-coordinates; clip them so ones that fall outside a
        # zoomed view don't float in a stray position past the plot edge.
        for t in ax.texts:
            t.set_clip_on(True)

        choice = self._zoom_var.get()
        if choice == _FULL_DEVICE:
            return
        match = next(((xmin, xmax) for label, xmin, xmax in self._layer_ranges
                      if label == choice), None)
        if match is None:
            return
        xmin, xmax = match
        margin = max(10.0, xmax - xmin)
        x_lo, x_hi = xmin - margin, xmax + margin
        ax.set_xlim(x_lo, x_hi)

        fields = _Y_FIELDS.get(tab_name)
        if not fields:
            return
        log_floor = 1.0 if tab_name == "Carriers" else None
        ylim = self._windowed_ylim(result, x_lo, x_hi, fields, log_floor)
        if ylim is None:
            return
        lo, hi = ylim
        if tab_name == "Carriers":
            ax.set_ylim(max(lo * 0.3, 1e-2), hi * 3.0)  # log-scale axis: multiplicative padding
        else:
            pad = 0.15 * (hi - lo) if hi > lo else 0.5
            if tab_name == "Wavefunctions":
                pad = max(pad, 0.5)  # headroom for the psi^2 humps drawn above Ec / below Ev
            ax.set_ylim(lo - pad, hi + pad)

    @staticmethod
    def _windowed_ylim(result: SolverResult, xmin: float, xmax: float,
                        fields: List[str], log_floor: Optional[float]):
        x = np.asarray(result.x_nm)
        mask = (x >= xmin) & (x <= xmax)
        if not np.any(mask):
            return None
        chunks = []
        for f in fields:
            arr = getattr(result, f, None)
            if arr is None:
                continue
            arr = np.asarray(arr)[mask]
            if log_floor is not None:
                arr = np.maximum(arr, log_floor)
            chunks.append(arr)
        if not chunks:
            return None
        values = np.concatenate(chunks)
        lo, hi = float(np.min(values)), float(np.max(values))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return lo, hi

    def _current_tab_name(self) -> Optional[str]:
        try:
            idx = self._notebook.index(self._notebook.select())
        except tk.TclError:
            return None
        return _PANELS[idx][0]

    def _export_png(self):
        if not self.results:
            messagebox.showinfo("Export PNG", "Nothing to export yet — run a solve first.")
            return
        name = self._current_tab_name()
        if name is None:
            return
        fig, *_ = self._tabs[name]
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG image", "*.png")],
            initialfile=f"{name.lower().replace(' ', '_')}.png")
        if not path:
            return
        fig.savefig(path, dpi=150, bbox_inches="tight")
        messagebox.showinfo("Export PNG", f"Saved to {path}")

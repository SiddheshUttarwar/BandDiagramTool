"""
PlotPanel: the graphics area of the main window and everything that belongs
to it — a tool palette beside one large figure, the "Style" page of the side
panel (which curves are drawn, the region shown), and the text panes shown
under the figure (solver output, and a summary of the solution).

The figures reuse visualization.plotter's panel functions and restyle the
result (palette, typography, layer bands in the layer table's material
colours), so scripts and the GUI draw the same physics.

Every plotter panel function sets its own x-axis to the full device width on
every call, so a thin layer (e.g. a 3nm quantum well sitting inside a 300nm
device) is squeezed into a handful of pixels — and since the panel functions
re-set xlim on every redraw (new solve, slider move, figure switch), a manual
zoom gets wiped out immediately. The "Region" control fixes that generically
for every figure: it stores the selected layer's (xmin, xmax) and re-applies
it *after* calling the panel function, so it survives redraws until the user
changes it back to "Full device".

Each solved result is still written to results/ as CSV (useful as a
standalone artifact), but plotting reads directly from the in-memory
SolverResult rather than reloading and re-parsing that CSV.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt6 import QtCore, QtGui, QtWidgets

from physics.self_consistent import SolverResult
import visualization.plotter as _plotter
from visualization.plotter import (
    plot_band_diagram, plot_wavefunctions, plot_carriers,
    plot_fields, plot_polarization, plot_strain, plot_qcse, plot_rsm, plot_illuminated_bands,
    plot_recombination, recombination_currents,
)
from physics.rsm import REFLECTIONS, reflections_for, simulate_rsm
from visualization.csv_export import save_result_csv, result_filename
from gui.layer_stack import _layer_summary, layer_colors, stack_crystal
from gui import theme, widgets

# The GUI draws with the theme's palette and a sans-serif face
# (visualization.plotter's own default, for scripts, is a serif journal style).
_plotter._COLORS.update(theme.FIGURE_COLORS)
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Segoe UI",
    "mathtext.it": "Segoe UI:italic",
    "mathtext.bf": "Segoe UI:bold",
    "mathtext.fallback": "cm",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": theme.FIG_AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": theme.TEXT_MUTED,
    "text.color": theme.TEXT,
    "xtick.color": theme.TEXT_MUTED,
    "ytick.color": theme.TEXT_MUTED,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "grid.color": theme.FIG_GRID,
    "legend.frameon": False,
})

_trapz = getattr(np, "trapezoid", None) or np.trapz

# Solved results are written as CSV to a "results" folder: beside the .exe
# in the packaged program, in the repository otherwise.
_BASE_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_BASE_DIR, "results")
_DEFAULT_PROJECT_NAME = "untitled"

_PANELS = [
    ("Band Diagram", lambda r, ax=None: plot_band_diagram(r, ax=ax, show_composition_bg=False)),
    # plot_wavefunctions only draws its first 3 states by default. A device
    # can have a deep polarization-induced notch (e.g. at a doped/undoped
    # interface) that's a genuinely lower-energy state than an intended
    # quantum well, pushing the QW's own state well past index 3 — so draw
    # more of them by default; combine with "Region" to inspect a specific one.
    ("Wavefunctions", lambda r, ax=None: plot_wavefunctions(r, ax=ax, n_show_e=16, n_show_h=16,
                                                            show_composition_bg=False)),
    ("Carriers", plot_carriers),
    ("Recombination", plot_recombination),
    ("Fields", plot_fields),
    ("Polarization", lambda r, ax=None: plot_polarization(r, ax=ax)[0]),
    ("Strain", plot_strain),
    ("QCSE", plot_qcse),
    # computed on request from the strain profile (see _calculate_rsm)
    ("RSM", None),
    # band diagram under light through the top surface (see show_light_bands)
    ("LB", None),
]
_TAB_LABELS = {"Band Diagram": "Band diagram", "Fields": "Electric field", "QCSE": "Stark effect",
               "RSM": "Reciprocal space map", "LB": "Bands under illumination"}

# Panels that need the *whole* loaded result set (plus which index is
# currently selected) rather than just the currently-selected SolverResult
# — e.g. QCSE plots the Stark shift/overlap across a voltage sweep. These
# also skip the position-axis "Region" behaviour below, since their
# x-axis isn't device position.
_SWEEP_AWARE_PANELS = {"QCSE"}

_FULL_DEVICE = "Full device"

# Which SolverResult arrays determine the y-range for each tab, used to
# tighten the y-axis to what's actually visible in a zoomed x-window
# (matplotlib's own relim()/autoscale_view() don't support the fill_between
# PolyCollections these panels use, and would autoscale from the *full*
# device's data regardless of the current xlim, not just the visible slice).
_Y_FIELDS = {
    "Band Diagram": ["Ec", "Ev", "Ev_hh", "Ev_lh", "Ev_so", "Ei", "Efn", "Efp"],
    "Wavefunctions": ["Ec", "Ev", "Ei", "Efn", "Efp"],
    "Carriers": ["n", "p"],
    "Fields": ["E_field", "E_polarization"],
    "Polarization": ["Psp", "Ppz", "P_total"],
    "Strain": ["eps_xx", "eps_zz"],
}

_MAX_LINEWIDTH = 1.2

# Line restyling by legend label: (linewidth, dash pattern or None, alpha)
_LINE_STYLES = {
    "$E_c$": (1.2, None, 1.0),
    "$E_v$": (1.2, None, 1.0),
    "$E_v$ (top)": (1.2, None, 1.0),
    "$E_v$ (HH)": (1.0, (6, 2), 1.0),
    "$E_v$ (LH)": (1.0, (6, 2, 1.5, 2), 1.0),
    "$E_v$ (SO)": (1.0, (1.5, 2), 1.0),
    "$E_i$": (0.8, (1, 3), 1.0),
    "$E_{fc}$": (0.9, (7, 3), 1.0),
    "$E_{fv}$": (0.9, (2, 2), 1.0),
    "$E_{vac}$": (0.8, (6, 2, 1, 2), 1.0),
}


# Style page: (key, checkbox text, legend labels it controls, on by default)
_CURVES = [
    ("Ec", "Conduction band edge", ("$E_c$",), True),
    ("Ev", "Valence band edge", ("$E_v$", "$E_v$ (top)"), True),
    ("sub", "HH / LH / SO edges", ("$E_v$ (HH)", "$E_v$ (LH)", "$E_v$ (SO)"), True),
    ("qfl", "Quasi-Fermi levels", ("$E_{fc}$", "$E_{fv}$"), True),
    ("Ei", "Intrinsic level", ("$E_i$",), False),
    ("vac", "Vacuum level", ("$E_{vac}$",), False),
]
_DISPLAY = [
    ("shading", "Layer shading", True),
    ("grid", "Grid lines", True),
    ("legend", "Legend", True),
]


class PlotPanel(QtWidgets.QWidget):
    tab_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[SolverResult] = []
        self.current_index = 0
        self._layer_ranges: List[Tuple[str, float, float]] = []  # (label, xmin, xmax)
        self._layer_spans: List[Tuple[float, float, str]] = []   # (xmin, xmax, colour) for the backdrop
        # Auto-zoom the plot onto the device's detected quantum well the
        # first time a solve comes back (see SolverResult.qw_window_nm) --
        # otherwise a QCSE-tilted 3nm well is an invisible sliver against a
        # 300nm+ full-device x-axis. Only fires once per PlotPanel instance
        # (a new project/open rebuilds the whole layout, see App._rebind_model)
        # so it never fights a zoom choice the user made themselves.
        self._auto_zoom_applied = False
        self._alloys = []
        self._crystal = "wurtzite"   # of the stack on screen; selects the RSM reflections offered
        self._light_bands = None    # physics.illumination.IlluminatedBands, once solved
        self._rsm = None            # physics.rsm.RSMResult of the result on screen, once calculated
        self._status_callback = None
        self._status_text = "Ready."
        self._opts = {key: on for key, _text, _labels, on in _CURVES}
        self._opts.update({key: on for key, _text, on in _DISPLAY})

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # ---- tool palette, beside the graphics area ----
        palette = QtWidgets.QToolBar()
        palette.setOrientation(QtCore.Qt.Orientation.Vertical)
        palette.setIconSize(QtCore.QSize(20, 20))
        palette.setMovable(False)
        self._home_btn = widgets.icon_button("home", "Reset the view", self._redraw_current, size=20)
        self._pan_btn = widgets.icon_button("pan", "Pan: drag the figure", checkable=True, size=20)
        self._zoom_btn = widgets.icon_button("zoom", "Zoom: drag a rectangle", checkable=True, size=20)
        self._pan_btn.clicked.connect(lambda _c=False: self._set_mode("pan"))
        self._zoom_btn.clicked.connect(lambda _c=False: self._set_mode("zoom"))
        self._export_btn = widgets.icon_button("export", "Export this figure (Ctrl+E)", self._export_png, size=20)
        for b in (self._home_btn, self._pan_btn, self._zoom_btn):
            palette.addWidget(b)
        palette.addSeparator()
        palette.addWidget(self._export_btn)
        root.addWidget(palette)

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        root.addLayout(right, 1)

        self._slider_widget = QtWidgets.QWidget()
        slider_row = QtWidgets.QHBoxLayout(self._slider_widget)
        slider_row.setContentsMargins(4, 2, 4, 2)
        slider_row.setSpacing(10)
        self._slider_label = QtWidgets.QLabel("")
        self._slider_label.setMinimumWidth(250)
        slider_row.addWidget(self._slider_label)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, 1)
        right.addWidget(self._slider_widget)

        # ---- shown with the Strain figure: diffraction from the strain profile ----
        self._rsm_bar = QtWidgets.QWidget()
        rsm_row = QtWidgets.QHBoxLayout(self._rsm_bar)
        rsm_row.setContentsMargins(4, 2, 4, 2)
        rsm_row.setSpacing(6)
        rsm_row.addWidget(QtWidgets.QLabel("Reflection:"))
        self._rsm_combo = QtWidgets.QComboBox()
        self._rsm_combo.addItems(list(REFLECTIONS))
        self._rsm_combo.setCurrentText("(10-15)")
        self._rsm_combo.setMinimumWidth(92)
        self._rsm_combo.setToolTip("Asymmetric reflections show relaxation; symmetric ones give the scan along Qz")
        rsm_row.addWidget(self._rsm_combo)
        rsm_btn = QtWidgets.QPushButton("Calculate reciprocal space map")
        rsm_btn.setToolTip("Simulate the X-ray reciprocal space map of this strain profile "
                           "and show it in the RSM figure")
        rsm_btn.clicked.connect(lambda _c=False: self._calculate_rsm())
        rsm_row.addWidget(rsm_btn)
        rsm_row.addStretch(1)
        right.addWidget(self._rsm_bar)
        self._rsm_bar.setVisible(False)
        self._slider_widget.setVisible(False)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        frame_lay = QtWidgets.QVBoxLayout(frame)
        frame_lay.setContentsMargins(0, 0, 0, 0)
        self._stack = QtWidgets.QStackedWidget()
        frame_lay.addWidget(self._stack)
        right.addWidget(frame, 1)

        self._readout = QtWidgets.QLabel("")       # placed in the status bar by the main window

        self._tabs = {}
        for name, fn in _PANELS:
            fig = plt.Figure(figsize=(8, 5.5), dpi=100)
            canvas = FigureCanvasQTAgg(fig)
            canvas.mpl_connect("motion_notify_event", self._on_motion)
            canvas.mpl_connect("figure_leave_event", lambda _e: self._readout.setText(""))
            toolbar = NavigationToolbar2QT(canvas, self)   # used for its pan/zoom logic only
            toolbar.hide()
            self._stack.addWidget(canvas)
            self._tabs[name] = (fig, canvas, fn, toolbar)
            self._draw_placeholder(name)

        # ---- text panes shown under the graphics area ----
        # Output streams physics.self_consistent's verbose=True per-iteration
        # lines live while a solve runs in the background (see
        # gui.solve_worker's log_fn callback); Summary lists the solution.
        self.log_widget = QtWidgets.QPlainTextEdit()
        self.summary_widget = QtWidgets.QPlainTextEdit()
        for w in (self.log_widget, self.summary_widget):
            w.setObjectName("log")
            w.setReadOnly(True)
            w.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_widget.setMaximumBlockCount(5000)
        self._log_text = self.log_widget
        self.output_tabs = QtWidgets.QTabWidget()
        self.output_tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.South)
        self.output_tabs.setDocumentMode(True)
        self.output_tabs.addTab(self.summary_widget, "Summary")
        self.output_tabs.addTab(self.log_widget, "Solver output")

        # ---- "Style" page of the side panel ----
        self.style_panel = QtWidgets.QWidget()
        style_lay = QtWidgets.QVBoxLayout(self.style_panel)
        style_lay.setContentsMargins(0, 0, 0, 0)
        style_lay.setSpacing(8)

        region_box = widgets.Section("Region shown")
        self._zoom_combo = QtWidgets.QComboBox()
        self._zoom_combo.addItem(_FULL_DEVICE)
        self._zoom_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._zoom_combo.setMinimumContentsLength(12)
        self._zoom_combo.setToolTip("Show one layer and its surroundings, e.g. a thin quantum well")
        self._zoom_combo.currentIndexChanged.connect(lambda _i: self._redraw_current())
        region_box.body.addWidget(self._zoom_combo)
        style_lay.addWidget(region_box)

        curve_box = widgets.Section("Band diagram")
        for key, text, _labels, _on in _CURVES:
            curve_box.body.addWidget(self._option_checkbox(key, text))
        style_lay.addWidget(curve_box)

        display_box = widgets.Section("Display")
        for key, text, _on in _DISPLAY:
            display_box.body.addWidget(self._option_checkbox(key, text))
        style_lay.addWidget(display_box)
        style_lay.addStretch(1)

    def _option_checkbox(self, key: str, text: str) -> QtWidgets.QCheckBox:
        cb = QtWidgets.QCheckBox(text)
        cb.setChecked(self._opts[key])
        cb.toggled.connect(lambda state, k=key: self._set_option(k, state))
        return cb

    def _set_option(self, key: str, state: bool) -> None:
        self._opts[key] = bool(state)
        self._redraw_current()

    @property
    def readout(self) -> QtWidgets.QLabel:
        return self._readout

    # ------------------------------------------------------------------
    def set_status_callback(self, callback) -> None:
        """callback(str) receives every status message (the main window's
        status bar)."""
        self._status_callback = callback
        callback(self._status_text)

    def set_status(self, text: str):
        self._status_text = text
        if self._status_callback is not None:
            self._status_callback(text)

    def append_log(self, text: str) -> None:
        self._log_text.appendPlainText(text)

    def clear_log(self) -> None:
        self._log_text.clear()

    def set_log_visible(self, visible: bool) -> None:
        self.output_tabs.setVisible(visible)

    # ---- used by the main window ----
    def tab_names(self) -> List[str]:
        return [name for name, _fn in _PANELS]

    def show_tab(self, name: str) -> None:
        for i, (panel_name, _fn) in enumerate(_PANELS):
            if panel_name == name:
                self.set_tab(i)
                return

    def set_tab(self, index: int) -> None:
        if index == self._stack.currentIndex():
            return
        self._on_tab_changed(index)
        self.tab_changed.emit(index)

    def current_tab(self) -> int:
        return self._stack.currentIndex()

    def export_png(self) -> None:
        self._export_png()

    # ------------------------------------------------------------------
    def _draw_placeholder(self, name: str) -> None:
        fig, canvas, _fn, _tb = self._tabs[name]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.55, _TAB_LABELS.get(name, name), ha="center", va="center",
                transform=ax.transAxes, color=theme.TEXT_MUTED, fontsize=13)
        hint = ("Run a simulation, open the Strain figure and press Calculate reciprocal space map."
                if name == "RSM" else
                "On the Simulation page, set the light and press Solve bands under light."
                if name == "LB" else
                "Build a device (or File > New from Template), then Simulation > Run (F5).")
        ax.text(0.5, 0.47, hint,
                ha="center", va="center", transform=ax.transAxes,
                color=theme.TEXT_FAINT, fontsize=10)
        canvas.draw_idle()

    def _on_tab_changed(self, index: int) -> None:
        for _fig, _canvas, _fn, tb in self._tabs.values():
            mode = getattr(tb.mode, "name", str(tb.mode))
            if mode == "PAN":
                tb.pan()
            elif mode == "ZOOM":
                tb.zoom()
        self._pan_btn.setChecked(False)
        self._zoom_btn.setChecked(False)
        self._stack.setCurrentIndex(index)
        self._rsm_bar.setVisible(_PANELS[index][0] in ("Strain", "RSM"))
        self._readout.setText("")
        self._redraw_current()

    def _set_mode(self, mode: str) -> None:
        name = self._current_tab_name()
        if name is None:
            return
        tb = self._tabs[name][3]
        tb.pan() if mode == "pan" else tb.zoom()
        active = getattr(tb.mode, "name", str(tb.mode))
        self._pan_btn.setChecked(active == "PAN")
        self._zoom_btn.setChecked(active == "ZOOM")

    def _on_motion(self, event) -> None:
        if event.inaxes is None or event.xdata is None or not self.results:
            self._readout.setText("")
            return
        if self._current_tab_name() == "RSM":
            if self._rsm is not None and self._rsm.symmetric:
                self._readout.setText(f"Qz = {event.xdata:.4f} 1/Å")
            else:
                self._readout.setText(f"Qx = {event.xdata:.4f}     Qz = {event.ydata:.4f} 1/Å")
        elif self._current_tab_name() in _SWEEP_AWARE_PANELS:
            self._readout.setText(f"V = {event.xdata:.3f} V     y = {event.ydata:.4g}")
        else:
            self._readout.setText(f"z = {event.xdata:.2f} nm     y = {event.ydata:.4g}")

    # ------------------------------------------------------------------
    def _write_csv(self, result: SolverResult, project_name: str) -> None:
        """Fire-and-forget CSV export -- a convenience artifact for the user,
        not something the plots themselves depend on (see module docstring)."""
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        stem = result_filename(project_name, result.V_applied, result.T)
        path = os.path.join(_RESULTS_DIR, f"{stem}.csv")
        save_result_csv(result, path)

    def show_result(self, result: SolverResult, layers: Optional[list] = None,
                     project_name: str = _DEFAULT_PROJECT_NAME):
        self._write_csv(result, project_name)
        self.results = [result]
        self._rsm = None
        self.current_index = 0
        self._slider_widget.setVisible(False)
        self._update_zoom_choices(layers, qw_range_nm=result.qw_window_nm)
        self._update_tiles()
        self._redraw_current()

    def show_sweep(self, results: List[SolverResult], layers: Optional[list] = None,
                    project_name: str = _DEFAULT_PROJECT_NAME):
        if not results:
            return
        for r in results:
            self._write_csv(r, project_name)
        self.results = results
        self._rsm = None
        self.current_index = len(results) - 1
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, len(results) - 1))
        self._slider.setValue(self.current_index)
        self._slider.blockSignals(False)
        self._slider_widget.setVisible(True)
        self._update_slider_label()
        self._update_zoom_choices(layers, qw_range_nm=results[self.current_index].qw_window_nm)
        self._update_tiles()
        self._redraw_current()

    def _update_tiles(self) -> None:
        """Write the summary of the current solution as plain text."""
        if not self.results:
            self.summary_widget.setPlainText("\n".join(self._light_band_lines()))
            return
        r = self.results[self.current_index]
        x_cm = np.asarray(r.x_nm, dtype=float) * 1e-7
        lines = []

        def put(key, value):
            lines.append(f"  {key:<26}{value}")

        lines.append("SOLUTION")
        put("Status", ("converged" if r.converged else "NOT CONVERGED") + f" after {r.n_iterations} iterations")
        v_int = getattr(r, "V_internal", r.V_applied)
        bias = f"{r.V_applied:.4f} V"
        if abs(v_int - r.V_applied) > 0.01:
            bias += f"   (internal {v_int:.4f} V)"
        put("Applied bias", bias)
        mode = getattr(r, "bias_mode", "")
        if getattr(r, "bias_note", ""):
            lines.append("      " + r.bias_note)
        if mode == "gate":
            put("Bias mode", "gate voltage (Schottky top contact): no current")
            lines.append("      The channel is held at the source Fermi level (0); the level reaches the gate's "
                         "across the depleted barrier.")
        elif mode == "interpolated":
            put("Bias mode", "nextnano++ Fermi level (interpolated between the contacts): no current")
        elif mode == "flat":
            put("Bias mode", "flat quasi-Fermi levels: no current")
        elif mode == "current":
            put("Bias mode", "drift-diffusion current between the two contacts")
        put("Temperature", f"{r.T:.1f} K")
        j = getattr(r, "J_total", None)
        if j is not None and np.isfinite(j) and abs(r.V_applied) > 1e-9:
            err = getattr(r, "current_conservation_error", float("nan"))
            if np.isfinite(err) and err > 1e-2:
                # a current this small is lost in the arithmetic: say so
                # rather than print a number that is not conserved
                put("Current density", f"below the numerical resolution (|J| ~ {abs(float(j)):.0e} A/cm^2)")
            else:
                put("Current density", f"{float(j):.4e} A/cm^2" + (f"   (conservation error {err:.1e})"
                                                                    if np.isfinite(err) else ""))
            floor = getattr(r, "transport_floor_cm3", 0.0)
            if floor > 0:
                put("Current equation", f"minimum density {floor:.1e} cm^-3")
                lines.append("      A layer is cut off from both contacts. Bands and densities are reliable;")
                lines.append("      a current this small is the leakage of that minimum density, not a prediction.")
        ns = float(_trapz(np.asarray(r.n, dtype=float), x_cm))
        ps = float(_trapz(np.asarray(r.p, dtype=float), x_cm))
        put("Electron sheet density", (f"{ns:.4e}" if ns >= 1 else "0") + " cm^-2   (integrated over the device)")
        put("Hole sheet density", (f"{ps:.4e}" if ps >= 1 else "0") + " cm^-2")
        f = np.abs(np.asarray(r.E_field, dtype=float)) * 1e-8
        if len(f):
            k = int(np.argmax(f))
            put("Peak field", f"{f[k]:.3f} MV/cm at z = {float(r.x_nm[k]):.2f} nm")

        rec = recombination_currents(r)
        # a unipolar device has rates of order 1e-30: numerical zeros, not worth a block
        if rec is not None and abs(r.V_applied) > 1e-9 and abs(rec["total"]) > 1e-15:
            lines.append("")
            lines.append("RECOMBINATION   integrated over the device, as current density")
            total = rec["total"]
            for key, label in (("srh", "Shockley-Read-Hall"), ("rad", "Radiative"), ("aug", "Auger")):
                share = f"   ({100.0 * rec[key] / total:5.1f} %)" if total > 0 else ""
                put(label, f"{rec[key]:.4e} A/cm^2{share}")
            put("Total", f"{total:.4e} A/cm^2")
            if total > 0:
                put("Radiative efficiency", f"{rec['iqe']:.4f}   (radiative / total recombination)")

        if r.qcse_transition_eV is not None and r.qcse_transition_eV > 0:
            ie, ih = r.qcse_pair if r.qcse_pair is not None else (0, 0)
            lines.append("")
            lines.append("OPTICAL TRANSITION")
            put(f"e{ie + 1}-h{ih + 1} energy", f"{r.qcse_transition_eV:.4f} eV   ({1239.84 / r.qcse_transition_eV:.1f} nm)")
            if r.qcse_overlap is not None:
                put("Overlap squared", f"{r.qcse_overlap:.4e}")

        for title, energies in (("ELECTRON SUBBANDS", getattr(r, "E_e", None)),
                                ("HOLE SUBBANDS (HH)", getattr(r, "E_h", None))):
            if energies is None or len(energies) == 0:
                continue
            lines.append("")
            lines.append(f"{title}   [eV]")
            vals = [f"{float(e):+9.4f}" for e in list(energies)[:12]]
            for i in range(0, len(vals), 6):
                lines.append("  " + " ".join(vals[i:i + 6]))

        sig = getattr(r, "interface_sigmas", None)
        idx = getattr(r, "interface_indices", None)
        if sig is not None and idx is not None and len(sig):
            lines.append("")
            lines.append("INTERFACE POLARIZATION CHARGE")
            lines.append("      z [nm]     sigma [1e13 e/cm^2]")
            for i, sgm in zip(idx, sig):
                lines.append(f"  {float(r.x_nm[i]):10.2f}     {sgm / 1.602176634e-19 * 1e-17:+10.4f}")
        if self._rsm is not None:
            # as grown (from the strain profile) and fully relaxed, per alloy;
            # r.l.u. = lambda / (2 d) x 10^4 for Cu K-alpha1, as diffractometers report it
            rlu = 1.540598 / (4.0 * np.pi) * 1e4
            m = self._rsm
            head = [f"RECIPROCAL SPACE MAP   {m.label} reflection, Q = 2 pi / d",
                    "  alloy               Qx [1/A]   Qz [1/A]   Qz [rlu]   |  relaxed: Qx [1/A]   Qz [1/A]   Qz [rlu]"]
            for (name, qx, qz), (_n, qx0, qz0) in zip(m.strained_points, m.relaxed_points):
                head.append(f"  {name:<18} {qx:9.4f}  {qz:9.4f}  {qz * rlu:9.1f}   |"
                             f"          {qx0:9.4f}  {qz0:9.4f}  {qz0 * rlu:9.1f}")
            lines = head + [""] + lines      # shown first: it is what was just calculated
        if self._light_bands is not None:
            lines = self._light_band_lines() + [""] + lines
        self.summary_widget.setPlainText("\n".join(lines))

    def _update_zoom_choices(self, layers: Optional[list],
                              qw_range_nm: Optional[Tuple[float, float]] = None):
        """Rebuild the Region dropdown from the current device's layers (in
        the bottom->top order devices.device.AlGaNDevice uses), keeping the
        current selection if a layer of the same label still exists.

        The first time this PlotPanel sees a solve with a detected quantum
        well (qw_range_nm), it auto-selects that layer instead of leaving
        "Full device" selected -- a thin QCSE-tilted well is otherwise an
        invisible sliver against the full x-axis and easy to miss entirely.
        Never overrides a zoom choice the user has already made.
        """
        previous = self._zoom_combo.currentText()
        self._layer_ranges = []
        self._layer_spans = []
        # compositions of the device's layers, for the reciprocal space map's markers
        self._alloys = []
        for layer in layers or []:
            if hasattr(layer, "x_Al"):
                self._alloys.append((layer.x_Al, getattr(layer, "x_In", 0.0), getattr(layer, "x_P", 0.0)))
            elif hasattr(layer, "x_Al_start"):
                self._alloys.append((layer.x_Al_start, getattr(layer, "x_In_start", 0.0),
                                     getattr(layer, "x_P_start", 0.0)))
                self._alloys.append((layer.x_Al_end, getattr(layer, "x_In_end", 0.0),
                                     getattr(layer, "x_P_end", 0.0)))
        crystal = stack_crystal(layers) if layers else self._crystal
        if crystal != self._crystal:
            # wurtzite (hkil) and zincblende (hkl) offer different reflections
            self._crystal = crystal
            self._rsm = None
            self._rsm_combo.clear()
            self._rsm_combo.addItems(list(reflections_for(crystal)))
            self._rsm_combo.setCurrentText("(224)" if crystal == "zincblende" else "(10-15)")
        if layers:
            start = 0.0
            for i, layer in enumerate(layers):
                thickness = getattr(layer, 'thickness_nm', None)
                if thickness is None:
                    continue   # zero-thickness interface layer (marker/surface charge/dipole): no zoom span of its own
                end = start + thickness
                title, _ = _layer_summary(layer)
                label = f"{i + 1}  {title}  ({start:g}–{end:g} nm)"
                self._layer_ranges.append((label, start, end))
                bottom, top = layer_colors(layer)
                mid = theme._hex(theme._mix(theme._rgb(bottom), theme._rgb(top), 0.5))
                self._layer_spans.append((start, end, mid))
                start = end

        values = [_FULL_DEVICE] + [label for label, _, _ in self._layer_ranges]
        self._zoom_combo.blockSignals(True)
        self._zoom_combo.clear()
        self._zoom_combo.addItems(values)

        if not self._auto_zoom_applied and qw_range_nm is not None and self._layer_ranges:
            qw_lo, qw_hi = qw_range_nm
            match = min(
                self._layer_ranges,
                key=lambda item: abs(item[1] - qw_lo) + abs(item[2] - qw_hi),
            )
            self._zoom_combo.setCurrentText(match[0])
            self._auto_zoom_applied = True
            self._zoom_combo.blockSignals(False)
            return

        self._zoom_combo.setCurrentText(previous if previous in values else _FULL_DEVICE)
        self._zoom_combo.blockSignals(False)

    def _zoom_shared(self, fig) -> None:
        """Apply the Region choice to a figure whose panels share the position axis."""
        choice = self._zoom_combo.currentText()
        match = next(((x0, x1) for label, x0, x1 in self._layer_ranges if label == choice), None)
        if match is not None:
            margin = max(10.0, match[1] - match[0])
            fig.axes[0].set_xlim(match[0] - margin, match[1] + margin)

    def _light_band_lines(self) -> List[str]:
        m = self._light_bands
        if m is None:
            return []
        d, l = m.dark, m.light
        split = np.asarray(l.Efn) - np.asarray(l.Efp)
        k = int(np.argmax(split))
        out = [f"BANDS UNDER ILLUMINATION   {m.wavelength_nm:g} nm ({m.photon_eV:.3f} eV), {m.power_W_cm2:g} W/cm^2, "
               f"{l.T - 273.15:.0f} C, light through the top surface, open circuit",
               f"  Status                    {'converged' if m.converged else 'NOT CONVERGED'}"
               + (f"   ({m.message})" if m.message else ""),
               f"  Photons absorbed          {100 * m.absorbed_fraction:.1f} % of those incident",
               f"  Photovoltage              {m.photovoltage_V:+.4g} V   (top relative to bottom, zero current)",
               f"  Short-circuit current     {m.J_short_A_cm2:.3e} A/cm^2   (residual at open circuit {m.J_residual_A_cm2:.1e})",
               f"  Ec at the surface         {float(d.Ec[-1]):+.3f} eV dark  ->  {float(l.Ec[-1]):+.3f} eV light"
               f"   (relative to the substrate Fermi level)",
               f"  Largest E_Fn - E_Fp       {float(split[k]):.3f} eV at z = {float(l.x_nm[k]):.1f} nm",
               "  The top contact stands for the free surface; both quasi-Fermi levels are pinned together there."]
        return out

    def show_light_bands(self, result) -> None:
        """Show a physics.illumination result in its figure and the summary."""
        self._light_bands = result
        self._update_tiles()
        self.output_tabs.setCurrentWidget(self.summary_widget)
        if self._current_tab_name() == "LB":
            self._redraw_current()
        else:
            self.show_tab("LB")

    def set_layers(self, layers) -> None:
        """Layer spans for the backdrop of figures drawn before any solve."""
        self._update_zoom_choices(layers)

    def _calculate_rsm(self) -> None:
        """Simulate the reciprocal space map of the result on screen and
        show it in its own figure."""
        if not self.results:
            self.set_status("Run a simulation first: the map is calculated from its strain profile.")
            return
        result = self.results[self.current_index]
        hkl = reflections_for(getattr(result, "crystal", "wurtzite")).get(self._rsm_combo.currentText())
        if hkl is None:
            self.set_status("Choose a reflection for this crystal structure.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._rsm = simulate_rsm(self.results[self.current_index], hkl, alloys=self._alloys or None)
        except Exception as exc:  # noqa: BLE001
            self._rsm = None
            self.set_status(f"Reciprocal space map failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.set_status(f"Reciprocal space map calculated for the {self._rsm.label} reflection "
                        "(kinematical approximation).")
        self._update_tiles()
        self.output_tabs.setCurrentWidget(self.summary_widget)
        if self._current_tab_name() == "RSM":
            self._redraw_current()
        else:
            self.show_tab("RSM")

    def _on_slider(self, value: int):
        idx = int(value)
        if idx == self.current_index:
            return
        self._rsm = None
        self.current_index = idx
        self._update_slider_label()
        self._update_tiles()
        self._redraw_current()

    def _update_slider_label(self):
        if not self.results:
            return
        r = self.results[self.current_index]
        conv = "converged" if r.converged else "not converged"
        self._slider_label.setText(
            f"V = {r.V_applied:.3f} V   ·   point {self.current_index + 1} of {len(self.results)}   ·   {conv}")

    # ------------------------------------------------------------------
    def _redraw_current(self):
        name = self._current_tab_name()
        if name == "LB":                      # has its own result, not the Run solution
            fig, canvas, _fn, toolbar = self._tabs[name]
            if self._light_bands is None:
                self._draw_placeholder(name)
                return
            plot_illuminated_bands(self._light_bands, fig)   # always the whole device: the surface matters
            for ax in fig.axes:
                if self._opts["shading"]:
                    self._layer_backdrop(ax)
            self._polish(fig, legend=None)
            try:
                fig.tight_layout(pad=1.3)
            except Exception:  # noqa: BLE001
                pass
            toolbar.update()
            canvas.draw_idle()
            return
        if not self.results:
            return
        result = self.results[self.current_index]
        if name is None:
            return
        fig, canvas, fn, toolbar = self._tabs[name]
        if name == "RSM" and self._rsm is None:
            self._draw_placeholder(name)
            return
        fig.clear()
        ax = fig.add_subplot(111)
        try:
            if name == "RSM":
                plot_rsm(self._rsm, ax=ax)
            elif name == "QCSE" and sum(r.qcse_transition_eV is not None for r in self.results) > 1:
                self._draw_stark_sweep(fig)
            elif name in _SWEEP_AWARE_PANELS:
                fn(self.results, self.current_index, ax=ax)
            else:
                if name == "Band Diagram":
                    plot_band_diagram(result, ax=ax, show_composition_bg=False,
                                      show_vacuum=self._opts["vac"],
                                      show_valence_bands=self._opts["sub"])
                else:
                    fn(result, ax=ax)
                if self._opts["shading"]:
                    self._layer_backdrop(ax)
                self._apply_zoom(ax, result, name)
            self._polish(fig, legend=None if name == "RSM" else name not in _SWEEP_AWARE_PANELS)
        except Exception as exc:  # noqa: BLE001 - never let a plot crash the app
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Could not render this figure:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes, wrap=True,
                    color=theme.TEXT_MUTED)
        try:
            # the panel functions lay the figure out before the title is
            # set, so lay it out once more or the title is clipped
            fig.tight_layout(pad=1.3)
        except Exception:  # noqa: BLE001
            pass
        toolbar.update()
        canvas.draw_idle()

    def _layer_backdrop(self, ax) -> None:
        """Faint bands in each layer's material colour, the same colours as
        the cross-section, so the figure reads against the structure."""
        previous = None
        for x0, x1, color in self._layer_spans:
            ax.axvspan(x0, x1, facecolor=color, alpha=0.07, linewidth=0, zorder=0)
            if previous is not None and previous != color:
                ax.axvline(x0, color=theme.BORDER_STRONG, lw=0.6, zorder=0.5)
            previous = color

    def _draw_stark_sweep(self, fig) -> None:
        """Transition energy and electron-hole overlap against bias, as two
        panels on a shared bias axis (rather than two y-scales on one)."""
        fig.clear()
        have = [r for r in self.results if r.qcse_transition_eV is not None]
        v = np.array([r.V_applied for r in have])
        order = np.argsort(v)
        v = v[order]
        e_t = np.array([r.qcse_transition_eV for r in have])[order]
        ov = np.array([r.qcse_overlap * 100.0 for r in have])[order]
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        ax1.plot(v, e_t, color=theme.FIG_BLUE, lw=1.2, marker="o", ms=4.5, mec="white", mew=1.0)
        ax1.set_ylabel("Transition energy (eV)")
        ax1.set_title("Stark effect against bias")
        ax2.plot(v, ov, color=theme.FIG_ORANGE, lw=1.2, marker="o", ms=4.5, mec="white", mew=1.0)
        ax2.set_ylabel("e–h overlap (%)")
        ax2.set_xlabel("Applied bias (V)")
        ax2.set_ylim(bottom=0)
        cur = self.results[self.current_index]
        if cur.qcse_transition_eV is not None:
            for a in (ax1, ax2):
                a.axvline(cur.V_applied, color=theme.FIG_NEUTRAL, lw=0.8, ls=(0, (2, 3)))

    def _polish(self, fig, legend: Optional[bool] = True) -> None:
        """House style applied to whatever the panel function drew.
        legend=None leaves the panel's own legend and grid as drawn."""
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>":
                continue
            title = ax.get_title()
            if title:
                ax.set_title("")
                ax.set_title(title, loc="left", fontsize=10.5, fontweight="semibold",
                             color=theme.TEXT, pad=10)
            ax.xaxis.label.set_size(10)
            ax.yaxis.label.set_size(10)
            ax.xaxis.label.set_color(theme.TEXT_MUTED)
            ax.yaxis.label.set_color(theme.TEXT_MUTED)
            ax.tick_params(axis="both", which="both", labelsize=9, colors=theme.TEXT_MUTED,
                           direction="out", top=False, right=False)
            ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
            ax.tick_params(axis="both", which="minor", length=2.0, width=0.6)
            for side in ax.spines.values():
                side.set_color(theme.FIG_AXIS)
                side.set_linewidth(0.8)
            if ax.axison and legend is not None:
                if self._opts["grid"]:
                    ax.grid(True, which="major", color=theme.FIG_GRID, lw=0.8, alpha=1.0)
                else:
                    ax.grid(False, which="major")
                ax.grid(False, which="minor")
                ax.set_axisbelow(True)
            hidden = {label for key, _text, labels, _on in _CURVES if not self._opts[key] for label in labels}
            for line in list(ax.lines):
                if line.get_label() in hidden:
                    line.remove()
                    continue
                style = _LINE_STYLES.get(line.get_label())
                if style is None and line.get_linewidth() > _MAX_LINEWIDTH:
                    line.set_linewidth(_MAX_LINEWIDTH)      # the plotter's script defaults are heavier
                if style is not None:
                    lw, dashes, alpha = style
                    line.set_linewidth(lw)
                    line.set_alpha(alpha)
                    line.set_linestyle((0, dashes) if dashes else "-")
            if legend is None:
                continue
            if ax.get_legend() is not None and not (legend and self._opts["legend"]):
                ax.get_legend().remove()
            elif legend and ax.get_legend() is not None:
                handles, labels = ax.get_legend_handles_labels()
                # a second y-axis on the right (the field figure's D axis)
                # needs the legend moved clear of its tick labels
                twin = any(o is not ax and o.get_label() != "<colorbar>"
                           and o.get_position().bounds == ax.get_position().bounds for o in fig.axes)
                ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.17 if twin else 1.015, 1.0),
                          borderaxespad=0.0, frameon=False, fontsize=9, handlelength=3.2,
                          labelcolor=theme.TEXT_MUTED, labelspacing=0.55)

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

        choice = self._zoom_combo.currentText()
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
        shown_qfl = dict(zip(("Efn", "Efp"), _plotter.quasi_fermi_levels_where_defined(result)))
        for f in fields:
            # a quasi-Fermi level counts only where it is drawn (see
            # visualization.plotter.quasi_fermi_levels_where_defined)
            arr = shown_qfl.get(f, getattr(result, f, None))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)[mask]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
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
        idx = self._stack.currentIndex()
        if idx < 0:
            return None
        return _PANELS[idx][0]

    def _export_png(self):
        if not self.results:
            self.set_status("Nothing to export yet. Run the simulation first.")
            return
        name = self._current_tab_name()
        if name is None:
            return
        fig = self._tabs[name][0]
        default_name = f"{name.lower().replace(' ', '_')}.png"
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export figure", default_name,
            "PNG image (*.png);;PDF document (*.pdf);;SVG drawing (*.svg)")
        if not path:
            return
        fig.savefig(path, dpi=300, bbox_inches="tight")
        self.set_status(f"Figure saved to {path}")

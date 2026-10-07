"""
Main application window, laid out like established desktop scientific
software (VESTA and its kin):

    File  Edit  View  Simulation  Help                                   menu bar
    [new][open][save] | [run][stop] | Bands  Wavefunctions  Carriers ... toolbar
    +-------------------------+--+--------------------------------------+
    | Structure|Simulation|Style |  |                                   |
    |                         |  |                                      |
    | layer table   New       |to|        graphics area                 |
    |               Delete    |ol|        (one large figure)            |
    |               Up / Down |s |                                      |
    | Selected layer          |  |                                      |
    | Contacts                +--+--------------------------------------+
    |                         | Summary | Solver output   (text)        |
    +-------------------------+-----------------------------------------+
    status message                      cursor position     project  state

Wires DeviceModel, LayerStackPanel, LayerEditorPanel, ContactsPanel,
SettingsPanel, PlotPanel and SolveWorker together, and owns the file
actions (new / open / save / save as).
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from gui import icons, templates, theme, widgets
from gui.models import DeviceModel
from gui.layer_stack import LayerStackPanel
from gui.layer_editor import LayerEditorPanel
from gui.contacts_settings import ContactsPanel, SettingsPanel
from gui.plot_panel import PlotPanel, _RESULTS_DIR
from gui.solve_worker import SolveWorker, SolveDone, SolveLog
from gui.project_io import save_project, load_project

_POLL_MS = 80
_DOCS_URL = "https://siddheshuttarwar.github.io/BandDiagramTool/"
_APP_NAME = "EpiBand"

# (figure name in PlotPanel, toolbar text, menu text)
_FIGURES = [
    ("Band Diagram", "Bands", "Band Diagram"),
    ("Wavefunctions", "Wavefunctions", "Wavefunctions"),
    ("Carriers", "Carriers", "Carrier Density"),
    ("Recombination", "Recombination", "Recombination Rates"),
    ("Fields", "Field", "Electric Field"),
    ("Polarization", "Polarization", "Polarization"),
    ("Strain", "Strain", "Strain"),
    ("QCSE", "Stark", "Stark Effect"),
    ("RSM", "RSM", "Reciprocal Space Map"),
    ("LB", "Illuminated", "Bands under Illumination"),
]


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1360, 860)
        self.setMinimumSize(1000, 640)

        self.model = DeviceModel()
        self.worker = SolveWorker()
        self._current_project_path: Optional[str] = None
        self._project_label_override: Optional[str] = None   # a template's name, until saved
        self._active_request_id: Optional[int] = None
        self._busy = False

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._build_body()
        self.model.add_listener(self._on_model_changed)
        self._update_title()
        self._set_busy(False)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_worker)
        self._poll_timer.start(_POLL_MS)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _action(self, text: str, slot, shortcut: str = "", icon_name: str = "",
                tip: str = "", icon_color: str = theme.TEXT) -> QtGui.QAction:
        act = QtGui.QAction(text, self)
        if icon_name:
            act.setIcon(icons.icon(icon_name, icon_color))
        if shortcut:
            act.setShortcut(QtGui.QKeySequence(shortcut))
            act.setShortcutContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        tip = tip or text.replace("&", "").rstrip(".…")
        act.setToolTip(f"{tip} ({shortcut})" if shortcut else tip)
        act.setStatusTip(tip)
        act.triggered.connect(lambda _checked=False: slot())
        return act

    def _build_actions(self):
        a = self._action
        self.act_new = a("&New", self._new_project, "Ctrl+N", "new", "New project")
        self.act_open = a("&Open…", self._open_project, "Ctrl+O", "open", "Open a project")
        self.act_save = a("&Save", self._save_project, "Ctrl+S", "save", "Save the project")
        self.act_save_as = a("Save &As…", self._save_project_as, "Ctrl+Shift+S")
        self.act_export = a("&Export Figure…", lambda: self.plot_panel.export_png(), "Ctrl+E", "export",
                            "Export the current figure as PNG, PDF or SVG")
        self.act_results = a("Open &Results Folder", self._open_results_folder, "", "results",
                             "Open the folder where solved results are written as CSV")
        self.act_exit = a("E&xit", self.close, "Ctrl+Q")
        self.act_delete = a("&Delete Layer", lambda: self.stack_panel.delete_selected(), "",
                            "trash", "Delete the selected layer")
        self.act_run = a("&Run", self._request_solve_now, "F5", "play",
                         "Solve the device with the current settings", icon_color="#1E8E3E")
        self.act_stop = a("&Stop", self._stop_solve, "Esc", "stop",
                          "Stop the running simulation", icon_color="#C4372E")
        self.act_docs = a("&Manual", self._open_docs, "F1", "book", "Model reference and validation")
        self.act_about = a(f"&About {_APP_NAME}", self._about)

        self._figure_group = QtGui.QActionGroup(self)
        self._figure_group.setExclusive(True)
        self._figure_actions = []
        for i, (name, short, long) in enumerate(_FIGURES):
            act = QtGui.QAction(long, self)
            act.setIconText(short)
            act.setCheckable(True)
            act.setChecked(i == 0)
            key = f"Ctrl+{(i + 1) % 10}"          # the tenth figure is Ctrl+0
            act.setShortcut(QtGui.QKeySequence(key))
            act.setToolTip(f"{long} ({key})")
            act.setStatusTip(f"Show the {long.lower()} figure")
            act.triggered.connect(lambda _checked=False, n=name: self.plot_panel.show_tab(n))
            self._figure_group.addAction(act)
            self._figure_actions.append(act)

        self.act_side = QtGui.QAction("&Side Panel", self, checkable=True, checked=True)
        self.act_side.toggled.connect(lambda state: self.side_tabs.setVisible(state))
        self.act_output = QtGui.QAction("&Text Area", self, checkable=True, checked=True)
        self.act_output.toggled.connect(lambda state: self.plot_panel.set_log_visible(state))

    def _build_menus(self):
        bar = self.menuBar()

        m = bar.addMenu("&File")
        m.addAction(self.act_new)
        tpl = m.addMenu("New from &Template")
        for t in templates.TEMPLATES:
            act = tpl.addAction(t.title, lambda k=t.key: self._load_template(k))
            act.setStatusTip(t.description)
        m.addAction(self.act_open)
        m.addSeparator()
        m.addAction(self.act_save)
        m.addAction(self.act_save_as)
        m.addSeparator()
        m.addAction(self.act_export)
        m.addAction(self.act_results)
        m.addSeparator()
        m.addAction(self.act_exit)

        m = bar.addMenu("&Edit")
        self._add_menu = m.addMenu("&Add")
        for label, method in (("Layer", "_add_abrupt"), ("Graded Layer", "_add_graded"), (None, None),
                              ("Quantum Region Marker", "_add_quantum_marker"),
                              ("Surface States", "_add_surface_charge"),
                              ("Interface Dipole", "_add_interface_dipole")):
            if label is None:
                self._add_menu.addSeparator()
            else:
                # looked up at call time: the panel is rebuilt on New/Open
                self._add_menu.addAction(label, lambda mth=method: getattr(self.stack_panel, mth)())
        m.addAction(self.act_delete)

        m = bar.addMenu("&View")
        for act in self._figure_actions:
            m.addAction(act)
        m.addSeparator()
        m.addAction(self.act_side)
        m.addAction(self.act_output)

        m = bar.addMenu("&Simulation")
        m.addAction(self.act_run)
        m.addAction(self.act_stop)

        m = bar.addMenu("&Help")
        m.addAction(self.act_docs)
        m.addSeparator()
        m.addAction(self.act_about)

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        for act in (self.act_new, self.act_open, self.act_save):
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.act_run)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        for act in self._figure_actions:        # text buttons, one per figure
            tb.addAction(act)
            tb.widgetForAction(act).setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        tb.addSeparator()
        tb.addAction(self.act_export)
        tb.addAction(self.act_docs)

    def _build_statusbar(self):
        bar = self.statusBar()
        self._status_message = QtWidgets.QLabel("")
        bar.addWidget(self._status_message, 1)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 0)          # indeterminate
        self._progress.setTextVisible(False)
        self._progress.setFixedSize(120, 14)
        self._progress.setVisible(False)
        bar.addPermanentWidget(self._progress)
        self._readout_holder = QtWidgets.QLabel("")
        self._readout_holder.setMinimumWidth(210)
        bar.addPermanentWidget(self._readout_holder)
        self._state_label = QtWidgets.QLabel("Ready")
        self._state_label.setMinimumWidth(60)
        bar.addPermanentWidget(self._state_label)

    def _build_body(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        # --- graphics area + text panes (right) ---
        self.plot_panel = PlotPanel()
        self.plot_panel.set_status_callback(self._show_status)
        self.plot_panel.tab_changed.connect(lambda i: self._figure_actions[i].setChecked(True))
        self.plot_panel.readout.setParent(None)
        self.plot_panel._readout = self._readout_holder       # cursor position goes to the status bar
        self._figure_actions[0].setChecked(True)

        # --- side panel (left): Structure | Simulation | Style ---
        self.side_tabs = QtWidgets.QTabWidget()

        self.stack_panel = LayerStackPanel(self.model, on_select=self._on_select_layer)
        self.editor_panel = LayerEditorPanel(self.model)
        self.contacts_panel = ContactsPanel(self.model)
        structure = QtWidgets.QWidget()
        s_lay = QtWidgets.QVBoxLayout(structure)
        s_lay.setContentsMargins(8, 8, 8, 8)
        s_lay.setSpacing(8)
        s_lay.addWidget(self.stack_panel, 3)
        s_lay.addWidget(self.contacts_panel)
        s_lay.addWidget(self.editor_panel)
        s_lay.addStretch(1)
        self.side_tabs.addTab(self._scrolled(structure), "Structure")

        self.settings_panel = SettingsPanel(self.model)
        simulation = QtWidgets.QWidget()
        m_lay = QtWidgets.QVBoxLayout(simulation)
        m_lay.setContentsMargins(8, 8, 8, 8)
        m_lay.setSpacing(8)
        m_lay.addWidget(self.settings_panel)
        m_lay.addWidget(self._build_growth_box())
        run_row = QtWidgets.QHBoxLayout()
        run_btn = QtWidgets.QPushButton("Run")
        run_btn.setIcon(icons.icon("play", "#1E8E3E"))
        run_btn.setToolTip("Solve the device with the current settings (F5)")
        run_btn.clicked.connect(lambda _c=False: self._request_solve_now())
        stop_btn = QtWidgets.QPushButton("Stop")
        stop_btn.clicked.connect(lambda _c=False: self._stop_solve())
        self._side_run, self._side_stop = run_btn, stop_btn
        run_row.addStretch(1)
        run_row.addWidget(run_btn)
        run_row.addWidget(stop_btn)
        m_lay.addLayout(run_row)
        m_lay.addStretch(1)
        self.side_tabs.addTab(self._scrolled(simulation), "Simulation")

        style = QtWidgets.QWidget()
        y_lay = QtWidgets.QVBoxLayout(style)
        y_lay.setContentsMargins(8, 8, 8, 8)
        y_lay.addWidget(self.plot_panel.style_panel)
        self.side_tabs.addTab(self._scrolled(style), "Style")

        # combo boxes size to their longest entry by default, which makes the
        # side panel wider than it should be; let them shrink with it
        for combo in self.side_tabs.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(4)
        self.side_tabs.setMinimumWidth(330)
        split.addWidget(self.side_tabs)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.setChildrenCollapsible(False)
        right.addWidget(self.plot_panel)
        right.addWidget(self.plot_panel.output_tabs)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setSizes([560, 210])
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 980])
        holder = QtWidgets.QWidget()
        h_lay = QtWidgets.QVBoxLayout(holder)
        h_lay.setContentsMargins(4, 2, 4, 0)
        h_lay.addWidget(split)
        self.setCentralWidget(holder)
        self.side_tabs.setVisible(self.act_side.isChecked())
        self.plot_panel.set_log_visible(self.act_output.isChecked())
        self._set_busy(self._busy)

    def _build_growth_box(self) -> QtWidgets.QWidget:
        """Light through the top surface (physics.illumination)."""
        box = widgets.Section("Illumination")
        self._growth_fields = {}
        for key, label, default, unit, tip in (
                ("T_growth_C", "Temperature", "1040", "°C",
                 "Temperature of the illuminated structure, e.g. the growth temperature"),
                ("wavelength_nm", "Wavelength", "300", "nm",
                 "Wavelength of the light. A layer absorbs only if the photon energy (1239.84 / "
                 "wavelength in nm, in eV) exceeds its band gap at this temperature."),
                ("power_W_cm2", "Power", "1", "W/cm²", "Power density of the light at the surface"),
                ("absorption_cm", "Absorption", "3e5", "cm⁻¹",
                 "Absorption coefficient well above the gap; it falls to zero at the gap")):
            edit = QtWidgets.QLineEdit(default)
            self._growth_fields[key] = edit
            box.body.addLayout(widgets.form_row(label, edit, unit, tip))
        box.body.addWidget(widgets.note(
            "Light enters through the top surface and no current is drawn. The top contact stands for "
            "the free surface (a Schottky barrier is its Fermi-level pinning)."))
        band_row = QtWidgets.QHBoxLayout()
        band_btn = QtWidgets.QPushButton("Solve bands under light")
        band_btn.setToolTip("Band diagram and carriers with the light on, at open circuit, "
                            "compared with the dark")
        band_btn.clicked.connect(lambda _c=False: self._solve_light_bands())
        band_row.addStretch(1)
        band_row.addWidget(band_btn)
        box.body.addLayout(band_row)
        return box

    def _solve_light_bands(self):
        """Bands with light through the top surface, at open circuit (physics.illumination)."""
        from devices.device import AlGaNDevice
        from physics.illumination import solve_illuminated
        if not self.model.is_solvable():
            self.plot_panel.set_status("Add at least one layer first.")
            return
        self.editor_panel.flush_pending()
        self.settings_panel.flush_pending()
        s = self.model.settings
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            f = {k: float(e.text()) for k, e in self._growth_fields.items()}
            device = AlGaNDevice(layers=list(self.model.layers), contacts=list(self.model.contacts),
                                 T=f["T_growth_C"] + 273.15, dx_nm=s.dx_nm,
                                 include_spontaneous_polarization=s.include_spontaneous_polarization,
                                 polarity=s.polarity, polarization_model=s.polarization_model,
                                 recombination=s.recombination())
            result = solve_illuminated(device, wavelength_nm=f["wavelength_nm"], power_W_cm2=f["power_W_cm2"],
                                       above_gap_cm=f["absorption_cm"], log_fn=self.plot_panel.append_log)
        except Exception as exc:  # noqa: BLE001
            self.plot_panel.set_status(f"Bands under light failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.plot_panel.set_layers(self.model.layers)
        self.plot_panel.show_light_bands(result)
        self.plot_panel.set_status(
            f"Bands under light: photovoltage {result.photovoltage_V:+.4g} V at open circuit"
            + ("" if result.converged else "  (NOT CONVERGED)") + ".")

    @staticmethod
    def _scrolled(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _show_status(self, text: str) -> None:
        self._status_message.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._progress.setVisible(busy)
        self._state_label.setText("Solving" if busy else "Ready")
        self.act_run.setEnabled(not busy)
        self.act_stop.setEnabled(busy)
        if hasattr(self, "_side_run"):
            self._side_run.setEnabled(not busy)
            self._side_stop.setEnabled(busy)

    def _update_title(self) -> None:
        if self._current_project_path:
            name = os.path.basename(self._current_project_path)
        else:
            name = self._project_label_override or "Untitled"
        self.setWindowTitle(f"{name} - {_APP_NAME}")

    # ------------------------------------------------------------------
    # Model / solve
    # ------------------------------------------------------------------
    def _on_select_layer(self, index):
        self.editor_panel.show(index)

    def _on_model_changed(self):
        self.stack_panel.refresh()
        # Solving only happens on an explicit Run (see _request_solve_now) --
        # editing the device just updates the model and flags that the
        # on-screen plots are now stale, rather than re-solving
        # automatically on every keystroke/edit.
        if self.model.is_solvable() and not self._busy:
            self.plot_panel.set_status("Device changed. Press Run (F5) to update the results.")

    def _request_solve_now(self):
        if self._busy:
            return
        if not self.model.is_solvable():
            self.plot_panel.set_status("Add at least one layer before running.")
            return

        # Field edits are debounced (~400ms) before landing in the model --
        # flush anything still pending so a value just typed (e.g. Applied
        # bias) is never silently solved-over with its pre-edit value.
        self.settings_panel.flush_pending()
        self.editor_panel.flush_pending()

        s = self.model.settings
        req = dict(
            layers=list(self.model.layers),
            contacts=list(self.model.contacts),
            T=s.T, dx_nm=s.dx_nm,
            include_spontaneous_polarization=s.include_spontaneous_polarization,
            polarity=s.polarity, polarization_model=s.polarization_model,
            recombination=s.recombination(),
            quantum=s.quantum,
            flat_qfl=s.flat_qfl,
            n_states_e=s.n_states_e, n_states_h=s.n_states_h,
            max_iter=s.max_iter, tol=s.tol, alpha=s.alpha,
            sweep=False, V_applied=s.V_applied,      # single bias point only

            V_start=s.V_start, V_stop=s.V_stop, n_steps=s.n_steps,
        )
        self.plot_panel.clear_log()
        self.plot_panel.output_tabs.setCurrentWidget(self.plot_panel.log_widget)
        self._active_request_id = self.worker.submit(req)
        self._set_busy(True)
        self.plot_panel.set_status("Solving…")

    def _stop_solve(self):
        if not self._busy:
            return
        self.worker.stop()
        self.plot_panel.set_status("Stopping…")

    def _poll_worker(self):
        for msg in self.worker.poll():
            if msg.request_id != self._active_request_id:
                continue
            if isinstance(msg, SolveLog):
                self.plot_panel.append_log(msg.text)
            elif isinstance(msg, SolveDone):
                self._set_busy(False)
                self._handle_solve_done(msg)

    def _project_base_name(self) -> str:
        """Basename (no extension) of the currently open/saved project file,
        used as the '<inputFileName>' prefix for solved-result CSVs -- falls
        back to 'untitled' for a project that hasn't been saved yet."""
        if not self._current_project_path:
            return "untitled"
        stem = os.path.splitext(os.path.basename(self._current_project_path))[0]
        return stem or "untitled"

    def _handle_solve_done(self, msg: SolveDone):
        if msg.cancelled:
            self.plot_panel.set_status("Stopped.")
            return
        if msg.error:
            self.plot_panel.set_status(f"Error: {msg.error}")
            return
        project_name = self._project_base_name()
        if msg.results is not None:
            self.plot_panel.show_sweep(msg.results, layers=self.model.layers,
                                        project_name=project_name)
            n_conv = sum(r.converged for r in msg.results)
            self.plot_panel.set_status(f"Sweep finished: {n_conv} of {len(msg.results)} points converged.")
        elif msg.result is not None:
            self.plot_panel.show_result(msg.result, layers=self.model.layers,
                                         project_name=project_name)
            r = msg.result
            status = "Converged" if r.converged else "Did not converge"
            v_note = f"V = {r.V_applied:.3f} V"
            if abs(r.V_internal - r.V_applied) > 0.01:
                # Not converged: V_internal is where the bias ramp stalled,
                # not an IR drop (it differs even with R_series = 0).
                reason = "series-resistance drop" if r.converged else "the bias ramp stopped here"
                v_note = (f"applied {r.V_applied:.3f} V, "
                          f"internal {r.V_internal:.3f} V ({reason})")
            self.plot_panel.set_status(f"{status} in {r.n_iterations} iterations.  {v_note}")
        self.plot_panel.output_tabs.setCurrentWidget(self.plot_panel.summary_widget)

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------
    def _open_docs(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(_DOCS_URL))

    def _open_results_folder(self):
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(_RESULTS_DIR))

    def _about(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"About {_APP_NAME}")
        pm = theme.logo_pixmap(128)
        pm.setDevicePixelRatio(2.0)
        box.setIconPixmap(pm)
        box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        box.setText(
            f"<b>{_APP_NAME}</b><br><br>"
            "One-dimensional Schrödinger–Poisson and drift-diffusion solver<br>"
            "for III-V heterostructures: wurtzite nitrides and<br>"
            "zincblende arsenides and phosphides.<br><br>"
            f"Model reference and validation:<br><a href='{_DOCS_URL}'>{_DOCS_URL}</a>")
        box.exec()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    def _new_project(self):
        self.model = DeviceModel()
        self._current_project_path = None
        self._project_label_override = None
        self._rebind_model()

    def _load_template(self, key: str):
        tpl = templates.get(key)
        self.model = tpl.build()
        self._current_project_path = None
        self._project_label_override = tpl.title
        self._rebind_model()
        self._request_solve_now()

    def _rebind_model(self):
        # Simplest robust way to point every panel at a new DeviceModel
        # instance (on New/Open) without each panel needing its own
        # set_model() plumbing: tear down and rebuild the body. Menus and
        # toolbar stay; their actions look the panels up at call time.
        old = self.takeCentralWidget()
        if old is not None:
            old.deleteLater()
        self._build_body()
        self.model.add_listener(self._on_model_changed)
        self._update_title()
        self.plot_panel.set_status("Project loaded." if self._current_project_path else "Ready.")

    def _load_path(self, path: str) -> bool:
        try:
            self.model = load_project(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open", f"Could not load the project:\n{exc}")
            return False
        self._current_project_path = path
        self._project_label_override = None
        self._rebind_model()
        return True

    def _open_project(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open", "", "EpiBand project (*.json)")
        if path:
            self._load_path(path)

    def _save_project(self):
        if not self._current_project_path:
            self._save_project_as()
            return
        try:
            save_project(self.model, self._current_project_path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save", f"Could not save the project:\n{exc}")
            return
        self.plot_panel.set_status(f"Saved {os.path.basename(self._current_project_path)}.")

    def _save_project_as(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save As", "", "EpiBand project (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_project(self.model, path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save", f"Could not save the project:\n{exc}")
            return
        self._current_project_path = path
        self._project_label_override = None
        self._update_title()
        self.plot_panel.set_status(f"Saved {os.path.basename(path)}.")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._poll_timer.stop()
        super().closeEvent(event)

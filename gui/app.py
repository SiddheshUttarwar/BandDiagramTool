"""
Main application window.

Layout, in the manner of the MATLAB desktop:

    +--------------------------------------------------------------+
    | HOME | PLOTS | VIEW                              project.json |  toolstrip tabs
    | [New][Open][Save] | [Add Layer] | [Run][Stop] | [Export] ...  |  toolstrip
    +---------------+------------------------------+---------------+
    | Device Stack  |  Figures (tabbed plots)      |  Properties   |
    | (dock)        |                              |  (dock)       |
    |               +------------------------------+               |
    |               |  Solver Log (dock)           |               |
    +---------------+------------------------------+---------------+
    | status message                          project | Ready/Busy |  status bar

Wires DeviceModel, LayerStackPanel, LayerEditorPanel, ContactsPanel,
SettingsPanel, PlotPanel and SolveWorker together, and owns the file
actions (new / open / save / save as).
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from gui import theme
from gui.models import DeviceModel
from gui.layer_stack import LayerStackPanel
from gui.layer_editor import LayerEditorPanel
from gui.contacts_settings import ContactsPanel, SettingsPanel
from gui.plot_panel import PlotPanel, _RESULTS_DIR
from gui.solve_worker import SolveWorker, SolveDone, SolveLog
from gui.project_io import save_project, load_project
from gui.toolstrip import Toolstrip

_POLL_MS = 80
_DOCS_URL = "https://siddheshuttarwar.github.io/BandDiagramTool/"
_APP_NAME = "BandDiagramTool"

# toolstrip PLOTS tab: (figure tab name, icon, two-line label)
_PLOT_BUTTONS = [
    ("Band Diagram", "plot_bands", "Band\nDiagram"),
    ("Wavefunctions", "plot_waves", "Wave-\nfunctions"),
    ("Carriers", "plot_carriers", "Carrier\nDensity"),
    ("Fields", "plot_fields", "Electric\nField"),
    ("Polarization", "plot_polarization", "Polari-\nzation"),
    ("Strain", "plot_strain", "Strain"),
    ("QCSE", "plot_qcse", "QCSE"),
]


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1480, 920)
        self.setMinimumSize(1100, 680)
        self.setDockNestingEnabled(True)
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AnimatedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks)

        self.model = DeviceModel()
        self.worker = SolveWorker()
        self._current_project_path: Optional[str] = None
        self._active_request_id: Optional[int] = None
        self._busy = False
        self._docks = []

        self._build_statusbar()
        self._build_layout()
        self._build_toolstrip()
        self._build_shortcuts()
        self.model.add_listener(self._on_model_changed)
        self._update_title()
        self._set_busy(False)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_worker)
        self._poll_timer.start(_POLL_MS)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        bar = self.statusBar()
        bar.setSizeGripEnabled(True)
        self._status_message = QtWidgets.QLabel("")
        bar.addWidget(self._status_message, 1)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 0)          # indeterminate
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        bar.addPermanentWidget(self._progress)

        self._project_label = QtWidgets.QLabel("")
        bar.addPermanentWidget(self._project_label)
        self._state_label = QtWidgets.QLabel("Ready")
        self._state_label.setMinimumWidth(52)
        bar.addPermanentWidget(self._state_label)

    def _make_dock(self, title: str, widget: QtWidgets.QWidget, name: str) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self._docks.append(dock)
        return dock

    def _build_layout(self):
        # --- figures (central) ---
        self.plot_panel = PlotPanel()
        self.plot_panel.set_status_callback(self._show_status)
        self.setCentralWidget(self.plot_panel)

        # --- device stack (left) ---
        self.stack_panel = LayerStackPanel(self.model, on_select=self._on_select_layer)
        self.stack_dock = self._make_dock("Device Stack", self.stack_panel, "dockStack")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.stack_dock)

        # --- properties (right): layer editor, contacts, simulation setup ---
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setSpacing(2)
        inner_layout.setContentsMargins(6, 0, 6, 6)

        self.editor_panel = LayerEditorPanel(self.model)
        inner_layout.addWidget(self.editor_panel)
        self.contacts_panel = ContactsPanel(self.model)
        inner_layout.addWidget(self.contacts_panel)
        self.settings_panel = SettingsPanel(self.model)
        inner_layout.addWidget(self.settings_panel)
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        # combo boxes size to their longest entry by default, which makes the
        # inspector wider than its dock; let them shrink with the panel
        for combo in inner.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(5)
        self.props_dock = self._make_dock("Properties", scroll, "dockProperties")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)

        # --- solver log (bottom) ---
        self.log_dock = self._make_dock("Solver Log", self.plot_panel.log_widget, "dockLog")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        # side docks run the full height; the log sits under the figures only
        self.setCorner(QtCore.Qt.Corner.BottomLeftCorner, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.BottomRightCorner, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self._apply_default_sizes()

    def _apply_default_sizes(self):
        self.resizeDocks([self.stack_dock, self.props_dock], [290, 430], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.log_dock], [150], QtCore.Qt.Orientation.Vertical)

    def _build_toolstrip(self):
        strip = Toolstrip()
        self.toolstrip = strip

        # ---------------- HOME ----------------
        home = strip.add_tab("Home")
        sec = home.add_section("File")
        sec.add_large("New", "new", self._new_project, "New project", shortcut="Ctrl+N")
        sec.add_large("Open", "open", self._open_project, "Open project", shortcut="Ctrl+O")
        sec.add_large("Save", "save", self._save_project, "Save project", shortcut="Ctrl+S")
        sec.add_small("Save As", "save_as", self._save_project_as, "Save project under a new name (Ctrl+Shift+S)")

        sec = home.add_section("Device")
        self._add_layer_btn = sec.add_large("Add\nLayer", "add_layer", tooltip="Add a layer or interface to the top of the stack")
        self._add_layer_btn.setMenu(self._layer_menu())
        self._add_layer_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        sec.add_small("Delete Layer", "clear", lambda: self.stack_panel.delete_selected(),
                      "Delete the selected layer (Del)")

        sec = home.add_section("Simulate")
        self._run_btn = sec.add_large("Run", "run", self._request_solve_now,
                                      "Solve the device with the current settings", shortcut="F5")
        self._stop_btn = sec.add_large("Stop", "stop", self._stop_solve, "Stop the running simulation",
                                       shortcut="Esc")

        sec = home.add_section("Results")
        sec.add_large("Export\nFigure", "export", lambda: self.plot_panel.export_png(),
                      "Save the current figure as a PNG image", shortcut="Ctrl+E")
        sec.add_small("Results Folder", "results", self._open_results_folder,
                      "Open the folder where solved results are written as CSV")
        sec.add_small("Clear Log", "clear", lambda: self.plot_panel.clear_log(), "Clear the solver log")

        sec = home.add_section("Resources")
        sec.add_large("Documentation", "help", self._open_docs, "Open the model reference in a browser",
                      shortcut="F1")
        sec.add_small("About", "info", self._about)

        # ---------------- PLOTS ----------------
        plots = strip.add_tab("Plots")
        sec = plots.add_section("Figure")
        for tab_name, icon_name, label in _PLOT_BUTTONS:
            sec.add_large(label, icon_name, lambda n=tab_name: self.plot_panel.show_tab(n),
                          f"Show the {tab_name} figure")
        sec = plots.add_section("Export")
        sec.add_large("Export\nFigure", "export", lambda: self.plot_panel.export_png(),
                      "Save the current figure as a PNG image", shortcut="Ctrl+E")

        # ---------------- VIEW ----------------
        view = strip.add_tab("View")
        sec = view.add_section("Panels")
        self._panel_buttons = []
        for label, dock_attr in (("Device Stack", "stack_dock"), ("Properties", "props_dock"),
                                 ("Solver Log", "log_dock")):
            btn = sec.add_small(label, "panel", lambda state, a=dock_attr: getattr(self, a).setVisible(state),
                                f"Show or hide the {label} panel", checkable=True)
            btn.setChecked(True)
            self._panel_buttons.append((btn, dock_attr))
        sec = view.add_section("Layout")
        sec.add_large("Default\nLayout", "layout", self._reset_layout, "Restore the default panel arrangement")

        self.setMenuWidget(strip)
        self._sync_panel_buttons()

    def _layer_menu(self) -> QtWidgets.QMenu:
        """Add-layer menu that always targets the current stack panel (the
        panel is rebuilt on New/Open)."""
        menu = QtWidgets.QMenu(self)
        for label, method in (("Abrupt layer", "_add_abrupt"), ("Graded layer", "_add_graded"), (None, None),
                              ("Quantum region marker", "_add_quantum_marker"),
                              ("Surface charge", "_add_surface_charge"),
                              ("Interface dipole", "_add_interface_dipole")):
            if label is None:
                menu.addSeparator()
            else:
                menu.addAction(label, lambda m=method: getattr(self.stack_panel, m)())
        return menu

    def _sync_panel_buttons(self):
        """Keep the VIEW toggles in step when a dock is closed with its own
        close button."""
        for btn, dock_attr in self._panel_buttons:
            dock = getattr(self, dock_attr)
            dock.visibilityChanged.connect(
                lambda _visible, b=btn, d=dock: (b.blockSignals(True), b.setChecked(not d.isHidden()),
                                                 b.blockSignals(False)))

    def _build_shortcuts(self):
        def bind(keys, slot):
            sc = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)

        bind("Ctrl+N", self._new_project)
        bind("Ctrl+O", self._open_project)
        bind("Ctrl+S", self._save_project)
        bind("Ctrl+Shift+S", self._save_project_as)
        bind("Ctrl+E", lambda: self.plot_panel.export_png())
        bind("F5", self._request_solve_now)
        bind("Esc", self._stop_solve)
        bind("F1", self._open_docs)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _show_status(self, text: str) -> None:
        self._status_message.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._progress.setVisible(busy)
        self._state_label.setText("Busy" if busy else "Ready")
        self._state_label.setStyleSheet(
            f"color: {theme.ACCENT if busy else theme.SUCCESS}; font-weight: 600;")
        if hasattr(self, "_run_btn"):
            self._run_btn.setEnabled(not busy)
            self._stop_btn.setEnabled(busy)

    def _update_title(self) -> None:
        name = os.path.basename(self._current_project_path) if self._current_project_path else "untitled"
        self.setWindowTitle(f"{name} - {_APP_NAME}")
        self._project_label.setText(self._current_project_path or "Unsaved project")
        if hasattr(self, "toolstrip"):
            self.toolstrip.set_caption(name)

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
            self.plot_panel.set_status("Device changed. Press Run (F5) to update the figures.")

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
            quantum=s.quantum,
            flat_qfl=s.flat_qfl,
            n_states_e=s.n_states_e, n_states_h=s.n_states_h,
            max_iter=s.max_iter, tol=s.tol, alpha=s.alpha,
            sweep=s.sweep_mode, V_applied=s.V_applied,
            V_start=s.V_start, V_stop=s.V_stop, n_steps=s.n_steps,
        )
        self.plot_panel.clear_log()
        self._active_request_id = self.worker.submit(req)
        self._set_busy(True)
        self.plot_panel.set_status("Running voltage sweep…" if s.sweep_mode else "Solving…")

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
            self.plot_panel.set_status("Stopped by user.")
            return
        if msg.error:
            self.plot_panel.set_status(f"Error: {msg.error}")
            return
        project_name = self._project_base_name()
        if msg.results is not None:
            self.plot_panel.show_sweep(msg.results, layers=self.model.layers,
                                        project_name=project_name)
            n_conv = sum(r.converged for r in msg.results)
            self.plot_panel.set_status(f"Sweep done: {n_conv}/{len(msg.results)} points converged.")
        elif msg.result is not None:
            self.plot_panel.show_result(msg.result, layers=self.model.layers,
                                         project_name=project_name)
            r = msg.result
            status = "Converged" if r.converged else "Did not converge"
            v_note = f"V = {r.V_applied:.3f} V"
            if abs(r.V_internal - r.V_applied) > 0.01:
                # Not converged: V_internal is where the bias ramp stalled,
                # not an IR drop (it differs even with R_series = 0).
                reason = "R_series IR drop" if r.converged else "bias ramp stopped here"
                v_note = (f"V_applied = {r.V_applied:.3f} V, "
                          f"V_internal = {r.V_internal:.3f} V ({reason})")
            self.plot_panel.set_status(
                f"{status} in {r.n_iterations} iterations. ({v_note})")

    # ------------------------------------------------------------------
    # Toolstrip actions
    # ------------------------------------------------------------------
    def _open_docs(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(_DOCS_URL))

    def _open_results_folder(self):
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(_RESULTS_DIR))

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, f"About {_APP_NAME}",
            f"<b>{_APP_NAME}</b><br>"
            "One-dimensional Schrödinger-Poisson-current solver for wurtzite "
            "III-nitride heterostructures.<br><br>"
            f"Model reference and validation:<br><a href='{_DOCS_URL}'>{_DOCS_URL}</a>")

    def _reset_layout(self):
        for dock, area in ((self.stack_dock, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea),
                           (self.props_dock, QtCore.Qt.DockWidgetArea.RightDockWidgetArea),
                           (self.log_dock, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.setVisible(True)
        self._apply_default_sizes()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    def _new_project(self):
        self.model = DeviceModel()
        self._current_project_path = None
        self._rebind_model()

    def _rebind_model(self):
        # Simplest robust way to point every panel at a new DeviceModel
        # instance (on New/Open) without each panel needing its own
        # set_model() plumbing: tear down and rebuild the panels. The
        # toolstrip stays; its actions look the panels up at call time.
        for dock in self._docks:
            self.removeDockWidget(dock)
            dock.deleteLater()
        self._docks = []
        old_central = self.takeCentralWidget()
        if old_central is not None:
            old_central.deleteLater()
        self._build_layout()
        self._sync_panel_buttons()
        for btn, _attr in self._panel_buttons:
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
        self.model.add_listener(self._on_model_changed)
        self.stack_panel.refresh()
        self._update_title()
        self.plot_panel.set_status("Project loaded." if self._current_project_path else "New project.")

    def _load_path(self, path: str) -> bool:
        try:
            self.model = load_project(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open", f"Could not load project:\n{exc}")
            return False
        self._current_project_path = path
        self._rebind_model()
        return True

    def _open_project(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open", "", "BandDiagramTool project (*.json)")
        if path:
            self._load_path(path)

    def _save_project(self):
        if not self._current_project_path:
            self._save_project_as()
            return
        try:
            save_project(self.model, self._current_project_path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save", f"Could not save project:\n{exc}")
            return
        self.plot_panel.set_status(f"Saved {os.path.basename(self._current_project_path)}.")

    def _save_project_as(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save As", "", "BandDiagramTool project (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_project(self.model, path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save", f"Could not save project:\n{exc}")
            return
        self._current_project_path = path
        self._update_title()
        self.plot_panel.set_status(f"Saved {os.path.basename(path)}.")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._poll_timer.stop()
        super().closeEvent(event)

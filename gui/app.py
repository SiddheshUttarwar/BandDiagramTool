"""
Main application window: wires DeviceModel, LayerStackPanel,
LayerEditorPanel, ContactsPanel, SettingsPanel, PlotPanel and SolveWorker
together. Handles debounced live re-solve, sweeps, and the File menu
(new/open/save/save as).
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from gui.models import DeviceModel
from gui.layer_stack import LayerStackPanel
from gui.layer_editor import LayerEditorPanel
from gui.contacts_settings import ContactsPanel, SettingsPanel
from gui.plot_panel import PlotPanel
from gui.solve_worker import SolveWorker, SolveDone, SolveLog
from gui.project_io import save_project, load_project

_POLL_MS = 80


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BandDiagramTool")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 650)

        self.model = DeviceModel()
        self.worker = SolveWorker()
        self._current_project_path: Optional[str] = None
        self._active_request_id: Optional[int] = None

        self._build_menu()
        self._build_layout()
        self.model.add_listener(self._on_model_changed)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_worker)
        self._poll_timer.start(_POLL_MS)

    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = self.menuBar()
        menubar.clear()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New", self._new_project)
        file_menu.addAction("Open…", self._open_project)
        file_menu.addAction("Save", self._save_project)
        file_menu.addAction("Save As…", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

    def _build_layout(self):
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self.stack_panel = LayerStackPanel(self.model, on_select=self._on_select_layer)
        splitter.addWidget(self.stack_panel)

        middle_scroll = QtWidgets.QScrollArea()
        middle_scroll.setWidgetResizable(True)
        middle_inner = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_inner)
        middle_layout.setSpacing(10)
        middle_layout.setContentsMargins(6, 6, 6, 6)

        self.editor_panel = LayerEditorPanel(self.model)
        middle_layout.addWidget(self.editor_panel)

        self.contacts_panel = ContactsPanel(self.model)
        middle_layout.addWidget(self.contacts_panel)

        self.settings_panel = SettingsPanel(self.model)
        middle_layout.addWidget(self.settings_panel)

        solve_now_btn = QtWidgets.QPushButton("Solve Now")
        solve_now_btn.setObjectName("primary")
        solve_now_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        solve_now_btn.clicked.connect(self._request_solve_now)
        middle_layout.addWidget(solve_now_btn)

        stop_btn = QtWidgets.QPushButton("Stop Simulation")
        stop_btn.setObjectName("danger")
        stop_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        stop_btn.clicked.connect(self._stop_solve)
        middle_layout.addWidget(stop_btn)

        middle_layout.addStretch(1)
        middle_scroll.setWidget(middle_inner)
        splitter.addWidget(middle_scroll)

        self.plot_panel = PlotPanel()
        splitter.addWidget(self.plot_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([340, 380, 780])
        splitter.setContentsMargins(10, 10, 10, 10)
        splitter.setHandleWidth(14)

    # ------------------------------------------------------------------
    def _on_select_layer(self, index):
        self.editor_panel.show(index)

    def _on_model_changed(self):
        self.stack_panel.refresh()
        # Solving only happens on an explicit "Solve Now" click (see
        # _request_solve_now) -- editing the device just updates the model
        # and flags that the on-screen plots are now stale, rather than
        # re-solving automatically on every keystroke/edit.
        if self.model.is_solvable():
            self.plot_panel.set_status("Device changed — click \"Solve Now\" to update.")

    def _request_solve_now(self):
        if not self.model.is_solvable():
            self.plot_panel.set_status("Add at least one layer to solve.")
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
            quantum=s.quantum,
            n_states_e=s.n_states_e, n_states_h=s.n_states_h,
            max_iter=s.max_iter, tol=s.tol, alpha=s.alpha,
            sweep=s.sweep_mode, V_applied=s.V_applied,
            V_start=s.V_start, V_stop=s.V_stop, n_steps=s.n_steps,
        )
        self.plot_panel.clear_log()
        self._active_request_id = self.worker.submit(req)
        self.plot_panel.set_status("Sweeping…" if s.sweep_mode else "Solving…")

    def _stop_solve(self):
        self.worker.stop()
        self.plot_panel.set_status("Stopping…")

    def _poll_worker(self):
        for msg in self.worker.poll():
            if msg.request_id != self._active_request_id:
                continue
            if isinstance(msg, SolveLog):
                self.plot_panel.append_log(msg.text)
            elif isinstance(msg, SolveDone):
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
                v_note = (f"V_applied = {r.V_applied:.3f} V, "
                          f"V_internal = {r.V_internal:.3f} V (R_series IR drop)")
            self.plot_panel.set_status(
                f"{status} in {r.n_iterations} iterations. ({v_note})")

    # ------------------------------------------------------------------
    def _new_project(self):
        self.model = DeviceModel()
        self._current_project_path = None
        self._rebind_model()

    def _rebind_model(self):
        # Simplest robust way to point every panel at a new DeviceModel
        # instance (on New/Open) without each panel needing its own
        # set_model() plumbing: tear down and rebuild the whole layout.
        old_central = self.takeCentralWidget()
        if old_central is not None:
            old_central.deleteLater()
        self._build_menu()
        self._build_layout()
        self.model.add_listener(self._on_model_changed)
        self.stack_panel.refresh()
        self.plot_panel.set_status("Project loaded." if self._current_project_path else "New project.")

    def _open_project(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open", "", "BandDiagramTool project (*.json)")
        if not path:
            return
        try:
            self.model = load_project(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open", f"Could not load project:\n{exc}")
            return
        self._current_project_path = path
        self._rebind_model()

    def _save_project(self):
        if not self._current_project_path:
            self._save_project_as()
            return
        try:
            save_project(self.model, self._current_project_path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save", f"Could not save project:\n{exc}")

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

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._poll_timer.stop()
        super().closeEvent(event)

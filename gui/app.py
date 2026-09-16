"""
Main application window: wires DeviceModel, LayerStackPanel,
LayerEditorPanel, ContactsPanel, SettingsPanel, PlotPanel and SolveWorker
together. Handles debounced live re-solve, sweeps, and the File menu
(new/open/save/save as).
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from gui.models import DeviceModel
from gui.layer_stack import LayerStackPanel
from gui.layer_editor import LayerEditorPanel
from gui.contacts_settings import ContactsPanel, SettingsPanel
from gui.plot_panel import PlotPanel
from gui.solve_worker import SolveWorker, SolveDone, SolveLog
from gui.project_io import save_project, load_project

_POLL_MS = 80


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("BandDiagramTool")
        root.geometry("1440x900")
        root.minsize(1000, 650)

        self.model = DeviceModel()
        self.worker = SolveWorker()
        self._current_project_path: Optional[str] = None
        self._active_request_id: Optional[int] = None

        self._build_menu()
        self._build_layout()
        self.model.add_listener(self._on_model_changed)

        self.root.after(_POLL_MS, self._poll_worker)

    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="New", command=self._new_project)
        file_menu.add_command(label="Open…", command=self._open_project)
        file_menu.add_command(label="Save", command=self._save_project)
        file_menu.add_command(label="Save As…", command=self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_layout(self):
        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, width=340)
        middle = ttk.Frame(paned, width=300)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(middle, weight=0)
        paned.add(right, weight=3)

        self.stack_panel = LayerStackPanel(left, self.model, on_select=self._on_select_layer)
        self.stack_panel.pack(fill="both", expand=True)

        # `middle` scrolls if the editor + contacts + settings don't fit.
        middle_canvas = tk.Canvas(middle, highlightthickness=0)
        middle_scroll = ttk.Scrollbar(middle, orient="vertical", command=middle_canvas.yview)
        middle_canvas.configure(yscrollcommand=middle_scroll.set)
        middle_canvas.pack(side="left", fill="both", expand=True)
        middle_scroll.pack(side="right", fill="y")
        middle_inner = ttk.Frame(middle_canvas)
        middle_inner_id = middle_canvas.create_window((0, 0), window=middle_inner, anchor="nw")
        middle_inner.bind("<Configure>",
                           lambda e: middle_canvas.configure(scrollregion=middle_canvas.bbox("all")))
        middle_canvas.bind("<Configure>",
                            lambda e: middle_canvas.itemconfigure(middle_inner_id, width=e.width))

        self.editor_panel = LayerEditorPanel(middle_inner, self.model)
        self.editor_panel.pack(fill="x", pady=(6, 6), padx=4)

        self.contacts_panel = ContactsPanel(middle_inner, self.model)
        self.contacts_panel.pack(fill="x", pady=(0, 6), padx=4)

        self.settings_panel = SettingsPanel(middle_inner, self.model)
        self.settings_panel.pack(fill="x", padx=4)

        solve_bar = ttk.Frame(middle_inner)
        solve_bar.pack(fill="x", pady=10, padx=4)
        ttk.Button(solve_bar, text="Solve Now", command=self._request_solve_now).pack(fill="x")
        ttk.Button(solve_bar, text="Stop Simulation", command=self._stop_solve).pack(fill="x", pady=(4, 0))

        self.plot_panel = PlotPanel(right)
        self.plot_panel.pack(fill="both", expand=True)

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
        self.root.after(_POLL_MS, self._poll_worker)

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
        for child in self.root.winfo_children():
            child.destroy()
        self._build_menu()
        self._build_layout()
        self.model.add_listener(self._on_model_changed)
        self.stack_panel.refresh()
        self.plot_panel.set_status("Project loaded." if self._current_project_path else "New project.")

    def _open_project(self):
        path = filedialog.askopenfilename(filetypes=[("BandDiagramTool project", "*.json")])
        if not path:
            return
        try:
            self.model = load_project(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open", f"Could not load project:\n{exc}")
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
            messagebox.showerror("Save", f"Could not save project:\n{exc}")

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("BandDiagramTool project", "*.json")])
        if not path:
            return
        try:
            save_project(self.model, path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save", f"Could not save project:\n{exc}")
            return
        self._current_project_path = path

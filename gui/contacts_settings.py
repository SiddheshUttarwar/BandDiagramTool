"""
ContactsPanel: bottom/top contact type + metal selection.
SettingsPanel: temperature, grid resolution, quantum toggle, bias/sweep
controls, and a collapsible "Advanced solver settings" section.

Both panels only ever push changes into DeviceModel (which calls
model.notify() itself) — they don't rebuild their own widgets in response
to edits, so typing/selecting is never interrupted.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from devices.layer import Contact
from physics.materials.metals import METAL_WORK_FUNCTIONS
from gui.models import DeviceModel

_METALS = sorted(METAL_WORK_FUNCTIONS.keys())


class ContactsPanel(ttk.LabelFrame):
    def __init__(self, parent, model: DeviceModel, **kwargs):
        super().__init__(parent, text="Contacts", **kwargs)
        self.model = model
        self._build_row("Bottom", model.bottom_contact, model.set_bottom_contact)
        self._build_row("Top", model.top_contact, model.set_top_contact)

    def _build_row(self, label, contact: Contact, setter):
        row = ttk.Frame(self)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text=label, width=8).pack(side="left")

        type_var = tk.StringVar(value=contact.contact_type)
        metal_var = tk.StringVar(value=contact.metal)

        def apply(*_):
            setter(Contact(position=contact.position, contact_type=type_var.get(),
                            metal=metal_var.get()))

        type_combo = ttk.Combobox(row, textvariable=type_var, state="readonly",
                                   values=["ohmic", "schottky"], width=10)
        type_combo.pack(side="left", padx=4)
        type_combo.bind("<<ComboboxSelected>>", apply)

        metal_combo = ttk.Combobox(row, textvariable=metal_var, state="readonly",
                                    values=_METALS, width=6)
        metal_combo.pack(side="left", padx=4)
        metal_combo.bind("<<ComboboxSelected>>", apply)


_APPLY_DEBOUNCE_MS = 400


class SettingsPanel(ttk.LabelFrame):
    def __init__(self, parent, model: DeviceModel, **kwargs):
        super().__init__(parent, text="Settings", **kwargs)
        self.model = model
        self._apply_after_ids: dict = {}
        s = model.settings

        self._row_entry(self, "Temperature (K)", "T", s.T)
        self._row_entry(self, "Grid spacing (nm)", "dx_nm", s.dx_nm)

        quantum_var = tk.BooleanVar(value=s.quantum)
        ttk.Checkbutton(self, text="Quantum (Schrödinger-Poisson)",
                         variable=quantum_var,
                         command=lambda: self._set("quantum", quantum_var.get())
                         ).pack(anchor="w", padx=8, pady=(6, 2))

        ttk.Separator(self).pack(fill="x", padx=8, pady=6)

        self.sweep_var = tk.BooleanVar(value=s.sweep_mode)
        ttk.Checkbutton(self, text="Voltage sweep mode", variable=self.sweep_var,
                         command=self._toggle_sweep).pack(anchor="w", padx=8)

        self._single_frame = ttk.Frame(self)
        self._sweep_frame = ttk.Frame(self)
        self._row_entry(self._single_frame, "Applied bias (V)", "V_applied", s.V_applied)
        self._row_entry(self._sweep_frame, "V start", "V_start", s.V_start)
        self._row_entry(self._sweep_frame, "V stop", "V_stop", s.V_stop)
        self._row_entry(self._sweep_frame, "Steps", "n_steps", s.n_steps, cast=int)
        self._toggle_sweep()

        ttk.Separator(self).pack(fill="x", padx=8, pady=6)
        self._adv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Advanced solver settings", variable=self._adv_var,
                         command=self._toggle_advanced).pack(anchor="w", padx=8)
        self._adv_frame = ttk.Frame(self)
        self._row_entry(self._adv_frame, "Max iterations", "max_iter", s.max_iter, cast=int)
        self._row_entry(self._adv_frame, "Tolerance (V)", "tol", s.tol)
        self._row_entry(self._adv_frame, "Damping alpha", "alpha", s.alpha)
        self._row_entry(self._adv_frame, "Series R (Ω·cm²)", "R_series", s.R_series)
        self._row_entry(self._adv_frame, "Electron subbands", "n_states_e", s.n_states_e, cast=int)
        self._row_entry(self._adv_frame, "Hole subbands", "n_states_h", s.n_states_h, cast=int)

    # ------------------------------------------------------------------
    def _row_entry(self, parent, label, attr, initial, cast=float):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar(value=str(initial))
        entry = ttk.Entry(row, textvariable=var, width=10)
        entry.pack(side="left")
        var.trace_add("write", lambda *a: self._debounced_set_cast(attr, var, cast))
        return var

    def _set(self, attr, value):
        setattr(self.model.settings, attr, value)
        self.model.notify()

    def _set_cast(self, attr, text, cast):
        try:
            value = cast(text)
        except ValueError:
            return
        self._set(attr, value)

    def _debounced_set_cast(self, attr, var, cast):
        # Debounced like LayerEditorPanel's fields: applying on every
        # keystroke makes the whole UI feel sluggish, since every model
        # change rebuilds the layer stack panel and restarts the
        # solve-debounce timer.
        pending = self._apply_after_ids.get(attr)
        if pending is not None:
            self.after_cancel(pending)
        self._apply_after_ids[attr] = self.after(
            _APPLY_DEBOUNCE_MS, lambda: self._set_cast(attr, var.get(), cast))

    # ------------------------------------------------------------------
    def _toggle_sweep(self):
        sweep = self.sweep_var.get()
        self._set("sweep_mode", sweep)
        if sweep:
            self._single_frame.pack_forget()
            self._sweep_frame.pack(fill="x")
        else:
            self._sweep_frame.pack_forget()
            self._single_frame.pack(fill="x")

    def _toggle_advanced(self):
        if self._adv_var.get():
            self._adv_frame.pack(fill="x")
        else:
            self._adv_frame.pack_forget()

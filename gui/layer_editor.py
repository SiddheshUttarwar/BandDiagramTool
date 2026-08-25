"""
LayerEditorPanel: a type-aware form for editing the currently selected
layer's fields (AbruptLayer vs GradedLayer), plus a "Layer type" switcher
that recreates the layer as the other type while preserving compatible
fields.

Only rebuilds its widgets on an explicit `show(index)` call (selection
change or type switch) — never in response to DeviceModel.notify() in
general, so that typing into a field doesn't get interrupted by an
unrelated model refresh triggered elsewhere in the app.

Field edits are debounced (applied to the model ~400ms after you stop
typing, not on every keystroke): each model change makes
LayerStackPanel destroy and rebuild every card and restarts the app's
solve-debounce timer, so applying on every character while typing a
number made the whole UI feel sluggish.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from devices.layer import AbruptLayer, GradedLayer
from gui.models import DeviceModel

_APPLY_DEBOUNCE_MS = 400


class LayerEditorPanel(ttk.LabelFrame):
    def __init__(self, parent, model: DeviceModel, **kwargs):
        super().__init__(parent, text="Layer Properties", **kwargs)
        self.model = model
        self.index: Optional[int] = None
        self._vars: dict = {}
        self._apply_after_id: Optional[str] = None

        self._empty_label = ttk.Label(self, text="Select a layer to edit.",
                                       foreground="#888888")
        self._empty_label.pack(anchor="w", padx=10, pady=10)

        self._body = ttk.Frame(self)

    # ------------------------------------------------------------------
    def show(self, index: Optional[int]):
        if self._apply_after_id is not None:
            self.after_cancel(self._apply_after_id)
            self._apply_after_id = None
        self.index = index
        for w in self._body.winfo_children():
            w.destroy()
        self._vars = {}

        if index is None or index >= len(self.model.layers):
            self._body.pack_forget()
            self._empty_label.pack(anchor="w", padx=10, pady=10)
            return

        self._empty_label.pack_forget()
        self._body.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        layer = self.model.layers[index]
        kind = "Abrupt" if isinstance(layer, AbruptLayer) else "Graded"

        type_row = ttk.Frame(self._body)
        type_row.pack(fill="x", pady=(0, 10))
        ttk.Label(type_row, text=f"Layer #{index + 1} from bottom — type").pack(side="left")
        type_var = tk.StringVar(value=kind)
        type_combo = ttk.Combobox(type_row, textvariable=type_var, state="readonly",
                                   values=["Abrupt", "Graded"], width=10)
        type_combo.pack(side="left", padx=8)
        type_combo.bind("<<ComboboxSelected>>", lambda e: self._switch_type(type_var.get()))

        if isinstance(layer, AbruptLayer):
            self._build_abrupt_form(layer)
        else:
            self._build_graded_form(layer)

    # ------------------------------------------------------------------
    def _switch_type(self, new_kind: str):
        layer = self.model.layers[self.index]
        if new_kind == "Abrupt" and isinstance(layer, GradedLayer):
            new_layer = AbruptLayer(x_Al=layer.x_Al_start, thickness_nm=layer.thickness_nm,
                                     n_doping=layer.n_doping, p_doping=layer.p_doping,
                                     relaxed=layer.relaxed, dx_nm=layer.dx_nm)
        elif new_kind == "Graded" and isinstance(layer, AbruptLayer):
            new_layer = GradedLayer(x_Al_start=layer.x_Al, x_Al_end=layer.x_Al,
                                     thickness_nm=layer.thickness_nm,
                                     n_doping=layer.n_doping, p_doping=layer.p_doping,
                                     relaxed=layer.relaxed, dx_nm=layer.dx_nm)
        else:
            return
        self.model.replace_layer(self.index, new_layer)
        self.show(self.index)

    # ------------------------------------------------------------------
    def _field(self, label, initial, kind="float"):
        row = ttk.Frame(self._body)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar(value=str(initial))
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        var.trace_add("write", lambda *a: self._debounced_apply())
        self._vars[label] = (var, kind)
        return var

    def _debounced_apply(self):
        if self._apply_after_id is not None:
            self.after_cancel(self._apply_after_id)
        self._apply_after_id = self.after(_APPLY_DEBOUNCE_MS, self._apply)

    def _bool_field(self, label, initial):
        var = tk.BooleanVar(value=initial)
        ttk.Checkbutton(self._body, text=label, variable=var,
                         command=self._apply).pack(anchor="w", pady=3)
        self._vars[label] = (var, "bool")
        return var

    def _choice_field(self, label, initial, choices):
        row = ttk.Frame(self._body)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar(value=initial)
        combo = ttk.Combobox(row, textvariable=var, state="readonly", values=choices)
        combo.pack(side="left", fill="x", expand=True)
        combo.bind("<<ComboboxSelected>>", lambda e: self._apply())
        self._vars[label] = (var, "str")
        return var

    def _build_abrupt_form(self, layer: AbruptLayer):
        self._field("Al fraction (0-1)", layer.x_Al)
        self._field("Thickness (nm)", layer.thickness_nm)
        self._field("n-doping (cm^-3)", layer.n_doping)
        self._field("p-doping (cm^-3)", layer.p_doping)
        self._bool_field("Strain-relaxed", layer.relaxed)
        self._field("Grid spacing (nm, blank=default)",
                     "" if layer.dx_nm is None else layer.dx_nm, kind="optional_float")

    def _build_graded_form(self, layer: GradedLayer):
        self._field("Al start (0-1)", layer.x_Al_start)
        self._field("Al end (0-1)", layer.x_Al_end)
        self._field("Thickness (nm)", layer.thickness_nm)
        self._choice_field("Profile", layer.profile, ["linear", "parabolic", "stepped", "abrupt"])
        self._field("n-doping (cm^-3)", layer.n_doping)
        self._field("p-doping (cm^-3)", layer.p_doping)
        self._field("Steps (if stepped)", layer.n_steps, kind="int")
        self._bool_field("Strain-relaxed", layer.relaxed)
        self._field("Grid spacing (nm, blank=default)",
                     "" if layer.dx_nm is None else layer.dx_nm, kind="optional_float")

    # ------------------------------------------------------------------
    def _apply(self):
        if self.index is None:
            return
        layer = self.model.layers[self.index]

        def get(label):
            var, kind = self._vars[label]
            v = var.get()
            if kind == "float":
                return float(v)
            if kind == "int":
                return int(float(v))
            if kind == "bool":
                return bool(v)
            if kind == "optional_float":
                return None if v.strip() == "" else float(v)
            return v

        try:
            if isinstance(layer, AbruptLayer):
                new_layer = AbruptLayer(
                    x_Al=get("Al fraction (0-1)"),
                    thickness_nm=get("Thickness (nm)"),
                    n_doping=get("n-doping (cm^-3)"),
                    p_doping=get("p-doping (cm^-3)"),
                    relaxed=get("Strain-relaxed"),
                    dx_nm=get("Grid spacing (nm, blank=default)"),
                )
            else:
                new_layer = GradedLayer(
                    x_Al_start=get("Al start (0-1)"),
                    x_Al_end=get("Al end (0-1)"),
                    thickness_nm=get("Thickness (nm)"),
                    n_doping=get("n-doping (cm^-3)"),
                    p_doping=get("p-doping (cm^-3)"),
                    profile=get("Profile"),
                    n_steps=get("Steps (if stepped)"),
                    relaxed=get("Strain-relaxed"),
                    dx_nm=get("Grid spacing (nm, blank=default)"),
                )
        except (ValueError, KeyError):
            return  # invalid/incomplete input mid-edit; don't apply yet

        self.model.replace_layer(self.index, new_layer)

"""
LayerEditorPanel: a type-aware form for editing the currently selected
layer's fields, plus a "Layer type" switcher that recreates the layer as
another type (preserving compatible fields when switching between the two
physical layer types, AbruptLayer/GradedLayer).

Five layer types are supported:
  Abrupt / Graded        - physical AlGaN layers (thickness, composition, doping)
  Quantum region marker  - zero-thickness start/end flag for the Schrodinger-
                            solved region (devices.layer.QuantumRegionMarker)
  Surface charge         - zero-thickness interface trap states, one or more
                            donor-/acceptor-like energy levels (devices.layer.
                            SurfaceCharge)
  Interface dipole       - zero-thickness fixed dipole, two opposite sheet
                            charges separated by a distance (devices.layer.
                            InterfaceDipole)

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

from devices.layer import (
    AbruptLayer, GradedLayer, QuantumRegionMarker, SurfaceCharge,
    SurfaceState, InterfaceDipole,
)
from gui.models import DeviceModel

_APPLY_DEBOUNCE_MS = 400

_TYPE_NAMES = ["Abrupt", "Graded", "Quantum region marker", "Surface charge", "Interface dipole"]


def _kind_of(layer) -> str:
    if isinstance(layer, AbruptLayer):
        return "Abrupt"
    if isinstance(layer, GradedLayer):
        return "Graded"
    if isinstance(layer, QuantumRegionMarker):
        return "Quantum region marker"
    if isinstance(layer, SurfaceCharge):
        return "Surface charge"
    if isinstance(layer, InterfaceDipole):
        return "Interface dipole"
    raise TypeError(f"Unknown layer type: {type(layer)}")


class LayerEditorPanel(ttk.LabelFrame):
    def __init__(self, parent, model: DeviceModel, **kwargs):
        super().__init__(parent, text="Layer Properties", **kwargs)
        self.model = model
        self.index: Optional[int] = None
        self._vars: dict = {}
        self._surface_rows: list = []   # [(density_var, energy_var, type_var), ...]
        self._apply_after_id: Optional[str] = None

        self._empty_label = ttk.Label(self, text="Select a layer to edit.",
                                       foreground="#888888")
        self._empty_label.pack(anchor="w", padx=10, pady=10)

        self._body = ttk.Frame(self)

    # ------------------------------------------------------------------
    def flush_pending(self) -> None:
        """Immediately apply any field edit still waiting on its debounce
        timer, instead of leaving it to land up to 400ms later. Called
        before a solve starts (see App._request_solve_now) so a value just
        typed isn't silently solved-over with its pre-edit value."""
        if self._apply_after_id is not None:
            self.after_cancel(self._apply_after_id)
            self._apply_after_id = None
            self._apply()

    def show(self, index: Optional[int]):
        if self._apply_after_id is not None:
            self.after_cancel(self._apply_after_id)
            self._apply_after_id = None
        self.index = index
        for w in self._body.winfo_children():
            w.destroy()
        self._vars = {}
        self._surface_rows = []

        if index is None or index >= len(self.model.layers):
            self._body.pack_forget()
            self._empty_label.pack(anchor="w", padx=10, pady=10)
            return

        self._empty_label.pack_forget()
        self._body.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        layer = self.model.layers[index]
        kind = _kind_of(layer)

        type_row = ttk.Frame(self._body)
        type_row.pack(fill="x", pady=(0, 10))
        ttk.Label(type_row, text=f"Layer #{index + 1} from bottom — type").pack(side="left")
        type_var = tk.StringVar(value=kind)
        type_combo = ttk.Combobox(type_row, textvariable=type_var, state="readonly",
                                   values=_TYPE_NAMES, width=20)
        type_combo.pack(side="left", padx=8)
        type_combo.bind("<<ComboboxSelected>>", lambda e: self._switch_type(type_var.get()))

        if isinstance(layer, AbruptLayer):
            self._build_abrupt_form(layer)
        elif isinstance(layer, GradedLayer):
            self._build_graded_form(layer)
        elif isinstance(layer, QuantumRegionMarker):
            self._build_quantum_marker_form(layer)
        elif isinstance(layer, SurfaceCharge):
            self._build_surface_charge_form(layer)
        elif isinstance(layer, InterfaceDipole):
            self._build_interface_dipole_form(layer)

    # ------------------------------------------------------------------
    def _switch_type(self, new_kind: str):
        layer = self.model.layers[self.index]

        if new_kind == "Abrupt":
            if isinstance(layer, GradedLayer):
                new_layer = AbruptLayer(x_Al=layer.x_Al_start, thickness_nm=layer.thickness_nm,
                                         n_doping=layer.n_doping, p_doping=layer.p_doping,
                                         relaxed=layer.relaxed, custom_strain_xx=layer.custom_strain_xx,
                                         series_resistance=layer.series_resistance,
                                         dx_nm=layer.dx_nm)
            elif isinstance(layer, AbruptLayer):
                return
            else:
                new_layer = AbruptLayer(x_Al=0.1, thickness_nm=10.0)
        elif new_kind == "Graded":
            if isinstance(layer, AbruptLayer):
                new_layer = GradedLayer(x_Al_start=layer.x_Al, x_Al_end=layer.x_Al,
                                         thickness_nm=layer.thickness_nm,
                                         n_doping=layer.n_doping, p_doping=layer.p_doping,
                                         relaxed=layer.relaxed, custom_strain_xx=layer.custom_strain_xx,
                                         series_resistance=layer.series_resistance,
                                         dx_nm=layer.dx_nm)
            elif isinstance(layer, GradedLayer):
                return
            else:
                new_layer = GradedLayer(x_Al_start=0.0, x_Al_end=0.3, thickness_nm=10.0)
        elif new_kind == "Quantum region marker":
            if isinstance(layer, QuantumRegionMarker):
                return
            new_layer = QuantumRegionMarker(boundary='start')
        elif new_kind == "Surface charge":
            if isinstance(layer, SurfaceCharge):
                return
            new_layer = SurfaceCharge(states=[SurfaceState(density_cm2=1e12, energy_eV=0.1, state_type='donor')])
        elif new_kind == "Interface dipole":
            if isinstance(layer, InterfaceDipole):
                return
            new_layer = InterfaceDipole(sheet_charge_C_m2=1e-3, separation_nm=0.5)
        else:
            return

        self.model.replace_layer(self.index, new_layer)
        self.show(self.index)

    # ------------------------------------------------------------------
    def _field(self, label, initial, kind="float"):
        row = ttk.Frame(self._body)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=22).pack(side="left")
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

    def _choice_field(self, label, initial, choices, on_change=None):
        row = ttk.Frame(self._body)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=22).pack(side="left")
        var = tk.StringVar(value=initial)
        combo = ttk.Combobox(row, textvariable=var, state="readonly", values=choices)
        combo.pack(side="left", fill="x", expand=True)
        combo.bind("<<ComboboxSelected>>", lambda e: (self._apply(), on_change() if on_change else None))
        self._vars[label] = (var, "str")
        return var

    def _build_abrupt_form(self, layer: AbruptLayer):
        self._field("Al fraction (0-1)", layer.x_Al)
        self._field("Thickness (nm)", layer.thickness_nm)
        self._field("n-doping (cm^-3)", layer.n_doping)
        self._field("p-doping (cm^-3)", layer.p_doping)
        self._bool_field("Strain-relaxed", layer.relaxed)
        self._field("Custom strain εxx (blank=auto, -=compressive, +=tensile)",
                     "" if layer.custom_strain_xx is None else layer.custom_strain_xx,
                     kind="optional_float")
        self._field("Series resistance (Ω·cm², 0=none)", layer.series_resistance)
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
        self._field("Custom strain εxx (blank=auto, -=compressive, +=tensile)",
                     "" if layer.custom_strain_xx is None else layer.custom_strain_xx,
                     kind="optional_float")
        self._field("Series resistance (Ω·cm², 0=none)", layer.series_resistance)
        self._field("Grid spacing (nm, blank=default)",
                     "" if layer.dx_nm is None else layer.dx_nm, kind="optional_float")

    # ------------------------------------------------------------------
    def _build_quantum_marker_form(self, layer: QuantumRegionMarker):
        """
        A zero-thickness flag marking where the Schrodinger-solved quantum
        region starts or ends -- place one 'start' marker and one 'end'
        marker in the stack (physics.self_consistent uses their positions
        +/- 2nm padding instead of its automatic undoped-span heuristic).
        """
        ttk.Label(self._body,
                  text="Marks a boundary of the quantum (Schrodinger-solved) region.\n"
                       "Place one 'start' and one 'end' marker in the stack; the solved\n"
                       "region spans 2nm before the start to 2nm after the end.",
                  foreground="#555555", justify="left").pack(anchor="w", pady=(0, 8))
        self._choice_field("Boundary", layer.boundary, ["start", "end"])

    def _build_interface_dipole_form(self, layer: InterfaceDipole):
        """
        A fixed structural dipole at an interface: two equal-and-opposite
        sheet charges +/-sigma separated by a user-defined distance.
        """
        ttk.Label(self._body,
                  text="Fixed dipole: two opposite sheet charges of magnitude\n"
                       "'Sheet charge' separated by 'Separation'.",
                  foreground="#555555", justify="left").pack(anchor="w", pady=(0, 8))
        self._field("Sheet charge (C/m^2)", layer.sheet_charge_C_m2)
        self._field("Separation (nm)", layer.separation_nm)

    def _build_surface_charge_form(self, layer: SurfaceCharge):
        """
        One or more donor-/acceptor-like trap energy states at a single
        interface, each ionizing self-consistently with the local Fermi
        level (see physics.self_consistent) -- the areal-charge analogue of
        bulk donor/acceptor doping.
        """
        ttk.Label(self._body,
                  text="One or more trap energy states at this interface, each\n"
                       "ionizing with the local Fermi level (donor: neutral filled,\n"
                       "+q ionized; acceptor: neutral empty, -q ionized).",
                  foreground="#555555", justify="left").pack(anchor="w", pady=(0, 8))

        states_frame = ttk.Frame(self._body)
        states_frame.pack(fill="x")

        header = ttk.Frame(states_frame)
        header.pack(fill="x")
        ttk.Label(header, text="Density (cm^-2)", width=16).pack(side="left")
        ttk.Label(header, text="Energy (eV)", width=12).pack(side="left")
        ttk.Label(header, text="Type", width=10).pack(side="left")

        self._surface_rows = []
        states = layer.states if layer.states else [SurfaceState(density_cm2=1e12)]
        for i, state in enumerate(states):
            row = ttk.Frame(states_frame)
            row.pack(fill="x", pady=2)
            dvar = tk.StringVar(value=str(state.density_cm2))
            evar = tk.StringVar(value=str(state.energy_eV))
            tvar = tk.StringVar(value=state.state_type)
            ttk.Entry(row, textvariable=dvar, width=16).pack(side="left")
            ttk.Entry(row, textvariable=evar, width=12).pack(side="left")
            ttk.Combobox(row, textvariable=tvar, state="readonly",
                         values=["donor", "acceptor"], width=9).pack(side="left")
            ttk.Button(row, text="Remove", width=8,
                       command=lambda i=i: self._remove_surface_state(i)).pack(side="left", padx=(4, 0))
            dvar.trace_add("write", lambda *a: self._debounced_apply())
            evar.trace_add("write", lambda *a: self._debounced_apply())
            tvar.trace_add("write", lambda *a: self._debounced_apply())
            self._surface_rows.append((dvar, evar, tvar))

        ttk.Button(self._body, text="+ Add energy state",
                   command=self._add_surface_state).pack(anchor="w", pady=(6, 0))

    def _current_surface_states(self) -> list:
        result = []
        for dvar, evar, tvar in self._surface_rows:
            try:
                result.append(SurfaceState(
                    density_cm2=float(dvar.get()),
                    energy_eV=float(evar.get()),
                    state_type=tvar.get(),
                ))
            except ValueError:
                continue  # mid-edit/invalid; skip rather than lose the whole list
        return result

    def _add_surface_state(self):
        states = self._current_surface_states()
        states.append(SurfaceState(density_cm2=1e12, energy_eV=0.1, state_type='donor'))
        self.model.replace_layer(self.index, SurfaceCharge(states=states))
        self.show(self.index)

    def _remove_surface_state(self, i: int):
        states = self._current_surface_states()
        if 0 <= i < len(states):
            del states[i]
        self.model.replace_layer(self.index, SurfaceCharge(states=states))
        self.show(self.index)

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
                    custom_strain_xx=get("Custom strain εxx (blank=auto, -=compressive, +=tensile)"),
                    series_resistance=get("Series resistance (Ω·cm², 0=none)"),
                    dx_nm=get("Grid spacing (nm, blank=default)"),
                )
            elif isinstance(layer, GradedLayer):
                new_layer = GradedLayer(
                    x_Al_start=get("Al start (0-1)"),
                    x_Al_end=get("Al end (0-1)"),
                    thickness_nm=get("Thickness (nm)"),
                    n_doping=get("n-doping (cm^-3)"),
                    p_doping=get("p-doping (cm^-3)"),
                    profile=get("Profile"),
                    n_steps=get("Steps (if stepped)"),
                    relaxed=get("Strain-relaxed"),
                    custom_strain_xx=get("Custom strain εxx (blank=auto, -=compressive, +=tensile)"),
                    series_resistance=get("Series resistance (Ω·cm², 0=none)"),
                    dx_nm=get("Grid spacing (nm, blank=default)"),
                )
            elif isinstance(layer, QuantumRegionMarker):
                new_layer = QuantumRegionMarker(boundary=get("Boundary"))
            elif isinstance(layer, InterfaceDipole):
                new_layer = InterfaceDipole(
                    sheet_charge_C_m2=get("Sheet charge (C/m^2)"),
                    separation_nm=get("Separation (nm)"),
                )
            elif isinstance(layer, SurfaceCharge):
                new_layer = SurfaceCharge(states=self._current_surface_states())
            else:
                return
        except (ValueError, KeyError):
            return  # invalid/incomplete input mid-edit; don't apply yet

        self.model.replace_layer(self.index, new_layer)

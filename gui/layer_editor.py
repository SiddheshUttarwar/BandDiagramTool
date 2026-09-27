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

from typing import Optional

from PyQt6 import QtCore, QtWidgets

from devices.layer import (
    AbruptLayer, GradedLayer, QuantumRegionMarker, SurfaceCharge,
    SurfaceState, InterfaceDipole,
)
from gui.models import DeviceModel
from gui import theme

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


class LayerEditorPanel(QtWidgets.QGroupBox):
    def __init__(self, model: DeviceModel, parent=None):
        super().__init__("Layer Properties", parent)
        self.model = model
        self.index: Optional[int] = None
        self._vars: dict = {}
        self._surface_rows: list = []   # [(density_edit, energy_edit, type_combo), ...]
        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply)

        self._outer = QtWidgets.QVBoxLayout(self)
        self._empty_label = QtWidgets.QLabel("Select a layer to edit.")
        self._empty_label.setStyleSheet(f"color: {theme.TEXT_FAINT};")
        self._outer.addWidget(self._empty_label)

        self._body = QtWidgets.QWidget()
        self._body_layout = QtWidgets.QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._outer.addWidget(self._body)
        self._body.setVisible(False)

    # ------------------------------------------------------------------
    def flush_pending(self) -> None:
        """Immediately apply any field edit still waiting on its debounce
        timer, instead of leaving it to land up to 400ms later. Called
        before a solve starts (see App._request_solve_now) so a value just
        typed isn't silently solved-over with its pre-edit value."""
        if self._apply_timer.isActive():
            self._apply_timer.stop()
            self._apply()

    def show(self, index: Optional[int]):
        self._apply_timer.stop()
        self.index = index
        self._replace_body()
        self._vars = {}
        self._surface_rows = []

        if index is None or index >= len(self.model.layers):
            self._body.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._empty_label.setVisible(False)
        self._body.setVisible(True)

        layer = self.model.layers[index]
        kind = _kind_of(layer)

        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel(f"Layer #{index + 1} from bottom — type"))
        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(_TYPE_NAMES)
        type_combo.setCurrentText(kind)
        type_combo.currentTextChanged.connect(self._switch_type)
        type_row.addWidget(type_combo)
        type_row.addStretch(1)
        self._body_layout.addLayout(type_row)

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

    def _replace_body(self):
        """Swap in a brand-new body widget rather than trying to selectively
        clear the old one's layout: removing items via takeAt()+deleteLater()
        detaches them from the layout immediately, but the widgets themselves
        stay parented (and visible, at their last geometry) until the
        deferred deleteLater() actually runs -- which briefly left the old
        form's fields overlapping the new one's when switching layer type.
        setParent(None) below hides the old body immediately, before any
        repaint can happen."""
        old_body = self._body
        self._body = QtWidgets.QWidget()
        self._body_layout = QtWidgets.QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._outer.insertWidget(1, self._body)
        self._outer.removeWidget(old_body)
        old_body.setParent(None)
        old_body.deleteLater()

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
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        col.addWidget(lbl)
        edit = QtWidgets.QLineEdit(str(initial))
        edit.textEdited.connect(self._debounced_apply)
        col.addWidget(edit)
        self._body_layout.addLayout(col)
        self._vars[label] = (edit, kind)
        return edit

    def _debounced_apply(self):
        self._apply_timer.start(_APPLY_DEBOUNCE_MS)

    def _bool_field(self, label, initial):
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(bool(initial))
        cb.toggled.connect(self._apply)
        self._body_layout.addWidget(cb)
        self._vars[label] = (cb, "bool")
        return cb

    def _choice_field(self, label, initial, choices, on_change=None):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        col.addWidget(lbl)
        combo = QtWidgets.QComboBox()
        combo.addItems(choices)
        combo.setCurrentText(initial)

        def _changed(_text):
            self._apply()
            if on_change:
                on_change()
        combo.currentTextChanged.connect(_changed)
        col.addWidget(combo)
        self._body_layout.addLayout(col)
        self._vars[label] = (combo, "str")
        return combo

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
        note = QtWidgets.QLabel(
            "Marks a boundary of the quantum (Schrödinger-solved) region.\n"
            "Place one 'start' and one 'end' marker in the stack; the solved\n"
            "region spans 2nm before the start to 2nm after the end.")
        note.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._body_layout.addWidget(note)
        self._choice_field("Boundary", layer.boundary, ["start", "end"])

    def _build_interface_dipole_form(self, layer: InterfaceDipole):
        """
        A fixed structural dipole at an interface: two equal-and-opposite
        sheet charges +/-sigma separated by a user-defined distance.
        """
        note = QtWidgets.QLabel(
            "Fixed dipole: two opposite sheet charges of magnitude\n"
            "'Sheet charge' separated by 'Separation'.")
        note.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._body_layout.addWidget(note)
        self._field("Sheet charge (C/m^2)", layer.sheet_charge_C_m2)
        self._field("Separation (nm)", layer.separation_nm)

    def _build_surface_charge_form(self, layer: SurfaceCharge):
        """
        One or more donor-/acceptor-like trap energy states at a single
        interface, each ionizing self-consistently with the local Fermi
        level (see physics.self_consistent) -- the areal-charge analogue of
        bulk donor/acceptor doping.
        """
        note = QtWidgets.QLabel(
            "One or more trap energy states at this interface, each\n"
            "ionizing with the local Fermi level (donor: neutral filled,\n"
            "+q ionized; acceptor: neutral empty, -q ionized).")
        note.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._body_layout.addWidget(note)

        header = QtWidgets.QHBoxLayout()
        h1 = QtWidgets.QLabel("Density (cm^-2)")
        h1.setFixedWidth(90)
        header.addWidget(h1)
        h2 = QtWidgets.QLabel("Energy (eV)")
        h2.setFixedWidth(65)
        header.addWidget(h2)
        h3 = QtWidgets.QLabel("Type")
        h3.setFixedWidth(75)
        header.addWidget(h3)
        header.addStretch(1)
        self._body_layout.addLayout(header)

        self._surface_rows = []
        states = layer.states if layer.states else [SurfaceState(density_cm2=1e12)]
        for i, state in enumerate(states):
            row = QtWidgets.QHBoxLayout()
            d_edit = QtWidgets.QLineEdit(str(state.density_cm2))
            d_edit.setFixedWidth(90)
            e_edit = QtWidgets.QLineEdit(str(state.energy_eV))
            e_edit.setFixedWidth(65)
            t_combo = QtWidgets.QComboBox()
            t_combo.addItems(["donor", "acceptor"])
            t_combo.setCurrentText(state.state_type)
            t_combo.setFixedWidth(85)
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                      QtWidgets.QSizePolicy.Policy.Fixed)
            remove_btn.clicked.connect(lambda _checked, i=i: self._remove_surface_state(i))

            d_edit.textEdited.connect(self._debounced_apply)
            e_edit.textEdited.connect(self._debounced_apply)
            t_combo.currentTextChanged.connect(lambda _t: self._debounced_apply())

            row.addWidget(d_edit)
            row.addWidget(e_edit)
            row.addWidget(t_combo)
            row.addWidget(remove_btn)
            row.addStretch(1)
            self._body_layout.addLayout(row)
            self._surface_rows.append((d_edit, e_edit, t_combo))

        add_btn = QtWidgets.QPushButton("+ Add energy state")
        add_btn.clicked.connect(self._add_surface_state)
        add_row = QtWidgets.QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addStretch(1)
        self._body_layout.addLayout(add_row)

    def _current_surface_states(self) -> list:
        result = []
        for d_edit, e_edit, t_combo in self._surface_rows:
            try:
                result.append(SurfaceState(
                    density_cm2=float(d_edit.text()),
                    energy_eV=float(e_edit.text()),
                    state_type=t_combo.currentText(),
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
            widget, kind = self._vars[label]
            if kind == "bool":
                return widget.isChecked()
            v = widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()
            if kind == "float":
                return float(v)
            if kind == "int":
                return int(float(v))
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

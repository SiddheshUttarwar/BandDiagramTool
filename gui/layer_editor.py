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
from gui import widgets

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


# Form keys (unchanged, _apply reads fields by them) -> (label shown, unit,
# placeholder, tooltip).
_FIELD_DISPLAY = {
    "Al fraction (0-1)": ("Al fraction", "", "", "Aluminium mole fraction, 0 to 1"),
    "In fraction (0-1)": ("In fraction", "", "", "Indium mole fraction, 0 to 1"),
    "Al start (0-1)": ("Al at bottom", "", "", "Aluminium mole fraction at the bottom face, 0 to 1"),
    "Al end (0-1)": ("Al at top", "", "", "Aluminium mole fraction at the top face, 0 to 1"),
    "In start (0-1)": ("In at bottom", "", "", "Indium mole fraction at the bottom face, 0 to 1"),
    "In end (0-1)": ("In at top", "", "", "Indium mole fraction at the top face, 0 to 1"),
    "Thickness (nm)": ("Thickness", "nm", "", ""),
    "n-doping (cm^-3)": ("Donors", "cm⁻³", "", "Donor concentration (n-type doping)"),
    "p-doping (cm^-3)": ("Acceptors", "cm⁻³", "", "Acceptor concentration (p-type doping)"),
    "Steps (if stepped)": ("Steps", "", "", "Number of steps, for the stepped profile"),
    "Custom strain εxx (blank=auto, -=compressive, +=tensile)": (
        "Strain εxx", "", "auto",
        "In-plane strain. Leave blank to derive it from the lattice mismatch; "
        "negative is compressive, positive tensile."),
    "Series resistance (Ω·cm², 0=none)": ("Series resistance", "Ω·cm²", "", "0 = none"),
    "Grid spacing (nm, blank=default)": ("Grid spacing", "nm", "default",
                                         "Mesh spacing inside this layer. Leave blank for the global value."),
    "Sheet charge (C/m^2)": ("Sheet charge", "C/m²", "", ""),
    "Separation (nm)": ("Separation", "nm", "", ""),
}


class LayerEditorPanel(widgets.Section):
    def __init__(self, model: DeviceModel, parent=None):
        super().__init__("Selected layer", parent)
        self.model = model
        self.index: Optional[int] = None
        self._vars: dict = {}
        self._surface_rows: list = []   # [(density_edit, energy_edit, type_combo), ...]
        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply)

        self._outer = self.body
        self._empty_label = QtWidgets.QLabel("Select a layer in the table to edit it.")
        self._empty_label.setObjectName("empty")
        self._empty_label.setWordWrap(True)
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
            self.set_meta("")
            return

        self._empty_label.setVisible(False)
        self._body.setVisible(True)

        layer = self.model.layers[index]
        kind = _kind_of(layer)
        self.set_meta(f"Layer {index + 1} of {len(self.model.layers)}, counted from the substrate")

        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(_TYPE_NAMES)
        type_combo.setCurrentText(kind)
        type_combo.currentTextChanged.connect(self._switch_type)
        self._body_layout.addLayout(widgets.form_row("Type", type_combo))

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
        self._body_layout.setSpacing(7)
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
                                         x_In=getattr(layer, "x_In_start", 0.0),
                                         material=getattr(layer, "material", None),
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
                                         x_In_start=getattr(layer, "x_In", 0.0),
                                         x_In_end=getattr(layer, "x_In", 0.0),
                                         material=getattr(layer, "material", None),
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
    def _subhead(self, text):
        self._body_layout.addWidget(widgets.subhead(text))

    def _note(self, text):
        self._body_layout.addWidget(widgets.note(text))

    def _field(self, label, initial, kind="float"):
        shown, unit, placeholder, tip = _FIELD_DISPLAY.get(label, (label, "", "", ""))
        if isinstance(initial, float):
            initial = f"{initial:.10g}"
        edit = QtWidgets.QLineEdit(str(initial))
        edit.setPlaceholderText(placeholder)
        edit.textEdited.connect(self._debounced_apply)
        self._body_layout.addLayout(widgets.form_row(shown, edit, unit, tip))
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
        combo = QtWidgets.QComboBox()
        combo.addItems(choices)
        combo.setCurrentText(initial)

        def _changed(_text):
            self._apply()
            if on_change:
                on_change()
        combo.currentTextChanged.connect(_changed)
        self._body_layout.addLayout(widgets.form_row(label, combo))
        self._vars[label] = (combo, "str")
        return combo

    def _material_choice(self, layer):
        """Material class dropdown; changing it rebuilds the form so only
        the fractions that material uses are shown."""
        self._choice_field("Material", getattr(layer, "material", None) or "AlGaN",
                           ["AlGaN", "InGaN", "InAlGaN"],
                           on_change=lambda: self.show(self.index))

    def _build_abrupt_form(self, layer: AbruptLayer):
        mat = getattr(layer, "material", None) or "AlGaN"
        self._material_choice(layer)
        if mat in ("AlGaN", "InAlGaN"):
            self._field("Al fraction (0-1)", layer.x_Al)
        if mat in ("InGaN", "InAlGaN"):
            self._field("In fraction (0-1)", getattr(layer, "x_In", 0.0))
        self._field("Thickness (nm)", layer.thickness_nm)
        self._subhead("Doping")
        self._field("n-doping (cm^-3)", layer.n_doping)
        self._field("p-doping (cm^-3)", layer.p_doping)
        self._subhead("Strain and mesh")
        self._bool_field("Strain-relaxed", layer.relaxed)
        self._field("Custom strain εxx (blank=auto, -=compressive, +=tensile)",
                     "" if layer.custom_strain_xx is None else layer.custom_strain_xx,
                     kind="optional_float")
        self._field("Series resistance (Ω·cm², 0=none)", layer.series_resistance)
        self._field("Grid spacing (nm, blank=default)",
                     "" if layer.dx_nm is None else layer.dx_nm, kind="optional_float")

    def _build_graded_form(self, layer: GradedLayer):
        mat = getattr(layer, "material", None) or "AlGaN"
        self._material_choice(layer)
        if mat in ("AlGaN", "InAlGaN"):
            self._field("Al start (0-1)", layer.x_Al_start)
            self._field("Al end (0-1)", layer.x_Al_end)
        if mat in ("InGaN", "InAlGaN"):
            self._field("In start (0-1)", getattr(layer, "x_In_start", 0.0))
            self._field("In end (0-1)", getattr(layer, "x_In_end", 0.0))
        self._field("Thickness (nm)", layer.thickness_nm)
        self._choice_field("Profile", layer.profile, ["linear", "parabolic", "stepped", "abrupt"])
        self._field("Steps (if stepped)", layer.n_steps, kind="int")
        self._subhead("Doping")
        self._field("n-doping (cm^-3)", layer.n_doping)
        self._field("p-doping (cm^-3)", layer.p_doping)
        self._subhead("Strain and mesh")
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
        self._note(
            "Marks a boundary of the region where the Schrödinger equation is solved. "
            "Place one start and one end marker in the stack; the solved region runs "
            "from 2 nm before the start to 2 nm after the end.")
        self._choice_field("Boundary", layer.boundary, ["start", "end"])

    def _build_interface_dipole_form(self, layer: InterfaceDipole):
        """
        A fixed structural dipole at an interface: two equal-and-opposite
        sheet charges +/-sigma separated by a user-defined distance.
        """
        self._note("A fixed dipole: two opposite sheet charges of the given magnitude, "
                   "the given distance apart.")
        self._field("Sheet charge (C/m^2)", layer.sheet_charge_C_m2)
        self._field("Separation (nm)", layer.separation_nm)

    def _build_surface_charge_form(self, layer: SurfaceCharge):
        """
        One or more donor-/acceptor-like trap energy states at a single
        interface, each ionizing self-consistently with the local Fermi
        level (see physics.self_consistent) -- the areal-charge analogue of
        bulk donor/acceptor doping.
        """
        self._note(
            "Trap levels at this interface, each ionizing with the local Fermi level "
            "(donor: neutral when filled, +q when ionized; acceptor: neutral when empty, "
            "−q when ionized).")

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(6)
        for text, width in (("Density (cm⁻²)", 90), ("Energy (eV)", 65), ("Type", 85)):
            h = QtWidgets.QLabel(text)
            h.setObjectName("unit")
            h.setFixedWidth(width)
            header.addWidget(h)
        header.addStretch(1)
        self._body_layout.addLayout(header)

        self._surface_rows = []
        states = layer.states if layer.states else [SurfaceState(density_cm2=1e12)]
        for i, state in enumerate(states):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(6)
            d_edit = QtWidgets.QLineEdit(f"{state.density_cm2:.10g}")
            d_edit.setFixedWidth(90)
            e_edit = QtWidgets.QLineEdit(f"{state.energy_eV:.10g}")
            e_edit.setFixedWidth(65)
            t_combo = QtWidgets.QComboBox()
            t_combo.addItems(["donor", "acceptor"])
            t_combo.setCurrentText(state.state_type)
            t_combo.setFixedWidth(85)
            remove_btn = widgets.icon_button("close", "Remove this level",
                                             lambda i=i: self._remove_surface_state(i), size=14)

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

        add_btn = QtWidgets.QPushButton("+  Add level")
        add_btn.setObjectName("link")
        add_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
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

        def getf(label, default=0.0):
            return get(label) if label in self._vars else default

        def adapt(mat, x_al, x_in, default_in=0.1):
            """Composition for `mat`, converting from whatever the form held
            (In removed for AlGaN, Al removed for InGaN; a fresh InGaN gets
            10% In so it is not plain GaN)."""
            if mat == "AlGaN":
                return x_al, 0.0
            if mat == "InGaN":
                return 0.0, (x_in if x_in > 0 else default_in)
            return x_al, x_in

        try:
            if isinstance(layer, AbruptLayer):
                mat = getf("Material", "AlGaN")
                x_al, x_in = adapt(mat, getf("Al fraction (0-1)", layer.x_Al),
                                   getf("In fraction (0-1)", getattr(layer, "x_In", 0.0)))
                new_layer = AbruptLayer(
                    x_Al=x_al, x_In=x_in, material=mat,
                    thickness_nm=get("Thickness (nm)"),
                    n_doping=get("n-doping (cm^-3)"),
                    p_doping=get("p-doping (cm^-3)"),
                    relaxed=get("Strain-relaxed"),
                    custom_strain_xx=get("Custom strain εxx (blank=auto, -=compressive, +=tensile)"),
                    series_resistance=get("Series resistance (Ω·cm², 0=none)"),
                    dx_nm=get("Grid spacing (nm, blank=default)"),
                )
            elif isinstance(layer, GradedLayer):
                mat = getf("Material", "AlGaN")
                a0, i0 = adapt(mat, getf("Al start (0-1)", layer.x_Al_start),
                               getf("In start (0-1)", getattr(layer, "x_In_start", 0.0)), 0.0)
                a1, i1 = adapt(mat, getf("Al end (0-1)", layer.x_Al_end),
                               getf("In end (0-1)", getattr(layer, "x_In_end", 0.0)), 0.1)
                new_layer = GradedLayer(
                    x_Al_start=a0, x_Al_end=a1, x_In_start=i0, x_In_end=i1, material=mat,
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

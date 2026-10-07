"""
ContactsPanel: top/bottom contact type, metal and barrier.
SettingsPanel: conditions (temperature, mesh), applied bias, physical
model switches, and a collapsible solver section.

Both panels only ever push changes into DeviceModel (which calls
model.notify() itself) — they don't rebuild their own widgets in response
to edits, so typing/selecting is never interrupted.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from devices.layer import Contact
from physics.materials.metals import METAL_WORK_FUNCTIONS
from gui.models import DeviceModel
from gui import widgets

_METALS = sorted(METAL_WORK_FUNCTIONS.keys())
_APPLY_DEBOUNCE_MS = 400


class ContactsPanel(widgets.Section):
    def __init__(self, model: DeviceModel, parent=None):
        super().__init__("Contacts", parent)
        self.model = model
        self.body.addLayout(self._build_row("Top", model.top_contact, model.set_top_contact))
        self.body.addLayout(self._build_row("Bottom", model.bottom_contact, model.set_bottom_contact))

    def _build_row(self, label, contact: Contact, setter):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        lbl = QtWidgets.QLabel(label + ":")
        lbl.setFixedWidth(46)
        row.addWidget(lbl)

        type_combo = QtWidgets.QComboBox()
        type_combo.addItem("Ohmic", "ohmic")
        type_combo.addItem("Schottky", "schottky")
        type_combo.setCurrentIndex(max(0, type_combo.findData(contact.contact_type)))

        metal_combo = QtWidgets.QComboBox()
        metal_combo.addItems(_METALS)
        metal_combo.setCurrentText(contact.metal)
        metal_combo.setToolTip("Contact metal (sets the work function)")

        # Explicit Schottky barrier height (Ec - Ef at the metal); 0 shows as
        # "auto" = Schottky-Mott estimate work_function - chi.
        barrier_spin = QtWidgets.QDoubleSpinBox()
        barrier_spin.setRange(0.0, 6.0)
        barrier_spin.setDecimals(2)
        barrier_spin.setSingleStep(0.05)
        barrier_spin.setSuffix(" eV")
        barrier_spin.setSpecialValueText("auto")
        barrier_spin.setMinimumWidth(74)
        barrier_spin.setToolTip("Schottky barrier height (Ec − Ef at the contact). "
                                "'auto' is the Schottky-Mott estimate: metal work function − electron affinity.")
        barrier_spin.setValue(contact.barrier_eV if getattr(contact, 'barrier_eV', None) else 0.0)
        barrier_spin.setEnabled(contact.contact_type == 'schottky')

        def apply(_=None):
            kind = type_combo.currentData()
            barrier_spin.setEnabled(kind == 'schottky')
            barrier = barrier_spin.value()
            setter(Contact(position=contact.position, contact_type=kind,
                            metal=metal_combo.currentText(),
                            barrier_eV=barrier if barrier > 0.0 else None))

        type_combo.currentIndexChanged.connect(apply)
        metal_combo.currentTextChanged.connect(apply)
        barrier_spin.valueChanged.connect(apply)

        row.addWidget(type_combo, 3)
        row.addWidget(metal_combo, 2)
        row.addWidget(barrier_spin, 3)
        return row


def _optional_positive(text: str):
    """Blank -> None (use the material value); otherwise a positive number."""
    if text.strip() == "":
        return None
    value = float(text)
    if value <= 0:
        raise ValueError("must be positive")
    return value


class SettingsPanel(widgets.Section):
    def __init__(self, model: DeviceModel, parent=None):
        super().__init__("Simulation", parent)
        self.model = model
        self._apply_timers: dict = {}
        self._field_vars: dict = {}   # attr -> (QLineEdit, cast) for flush_pending
        s = model.settings
        layout = self.body

        layout.addLayout(self._row_entry("Temperature", "T", s.T, unit="K"))
        layout.addLayout(self._row_entry("Grid spacing", "dx_nm", s.dx_nm, unit="nm",
                                         tip="Default mesh spacing; a layer can override it."))

        # --- Bias (a single point; the bias sweep is not offered in the GUI) ---
        model.settings.sweep_mode = False
        layout.addLayout(self._row_entry("Applied bias", "V_applied", s.V_applied, unit="V"))

        # --- Physical model ---
        layout.addWidget(widgets.subhead("Model"))
        quantum_cb = QtWidgets.QCheckBox("Schrödinger–Poisson (quantum)")
        quantum_cb.setToolTip("Solve the Schrödinger equation in the quantum region and use the "
                              "subband densities in Poisson's equation.")
        quantum_cb.setChecked(s.quantum)
        quantum_cb.toggled.connect(lambda v: self._set("quantum", v))
        layout.addWidget(quantum_cb)

        psp_cb = QtWidgets.QCheckBox("Spontaneous polarization")
        psp_cb.setToolTip("Include spontaneous polarization in addition to the piezoelectric part.")
        psp_cb.setChecked(s.include_spontaneous_polarization)
        psp_cb.toggled.connect(lambda v: self._set("include_spontaneous_polarization", v))
        layout.addWidget(psp_cb)

        flat_qfl_cb = QtWidgets.QCheckBox("Flat quasi-Fermi levels (no current)")
        flat_qfl_cb.setToolTip(
            "Hold Efn at the n-contact level and Efp at the p-contact level at every "
            "interior grid point (split = qV everywhere except the contacts) and solve "
            "only (Schrödinger-)Poisson. Fast and robust for band diagrams and the Stark "
            "effect under bias; gives no current information.")
        flat_qfl_cb.setChecked(s.flat_qfl)
        flat_qfl_cb.toggled.connect(lambda v: self._set("flat_qfl", v))
        layout.addWidget(flat_qfl_cb)

        polarity_combo = QtWidgets.QComboBox()
        for label, value in (("Metal-polar [0001]", "metal"), ("N-polar [000-1]", "N")):
            polarity_combo.addItem(label, value)
        polarity_combo.setCurrentIndex(max(0, polarity_combo.findData(s.polarity)))
        polarity_combo.currentIndexChanged.connect(
            lambda _i, c=polarity_combo: self._set("polarity", c.currentData()))
        layout.addLayout(widgets.form_row(
            "Polarity", polarity_combo,
            tip="N-polar growth reverses the polarization along the growth axis."))

        model_combo = QtWidgets.QComboBox()
        for label, value in (("Ambacher 2002", "ambacher2002"), ("Dreyer 2016", "dreyer2016")):
            model_combo.addItem(label, value)
        model_combo.setCurrentIndex(max(0, model_combo.findData(s.polarization_model)))
        model_combo.currentIndexChanged.connect(
            lambda _i, c=model_combo: self._set("polarization_model", c.currentData()))
        layout.addLayout(widgets.form_row(
            "Polarization set", model_combo,
            tip="Ambacher 2002 (default): zincblende-referenced Psp with proper piezoelectric "
                "constants. Dreyer 2016: hexagonal-referenced Psp with improper e31 "
                "(Phys. Rev. X 6, 021038)."))

        # --- Solver (collapsed by default) ---
        self._adv_cb = QtWidgets.QToolButton()
        self._adv_cb.setAutoRaise(True)
        self._adv_cb.setText("Solver settings")
        self._adv_cb.setCheckable(True)
        self._adv_cb.setArrowType(QtCore.Qt.ArrowType.NoArrow)
        self._adv_cb.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._adv_cb.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._adv_cb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._adv_cb.toggled.connect(self._toggle_advanced)
        adv_row = QtWidgets.QHBoxLayout()
        adv_row.setContentsMargins(0, 4, 0, 0)
        adv_row.addWidget(self._adv_cb)
        adv_row.addStretch(1)
        layout.addLayout(adv_row)

        self._adv_widget = QtWidgets.QWidget()
        adv_layout = QtWidgets.QVBoxLayout(self._adv_widget)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(7)
        adv_layout.addLayout(self._row_entry("Max iterations", "max_iter", s.max_iter, cast=int))
        adv_layout.addLayout(self._row_entry("Tolerance", "tol", s.tol, unit="V"))
        adv_layout.addLayout(self._row_entry("Damping α", "alpha", s.alpha))
        adv_layout.addLayout(self._row_entry("Electron subbands", "n_states_e", s.n_states_e, cast=int))
        adv_layout.addLayout(self._row_entry("Hole subbands", "n_states_h", s.n_states_h, cast=int))
        adv_layout.addWidget(widgets.subhead("Recombination"))
        blank = "Leave blank to use each material's own value."
        for label, attr, unit, tip in (
                ("SRH lifetime, e", "tau_n_ns", "ns", "Shockley-Read-Hall electron lifetime. " + blank),
                ("SRH lifetime, h", "tau_p_ns", "ns", "Shockley-Read-Hall hole lifetime. " + blank),
                ("Radiative B", "B_rad", "cm³/s", "Radiative recombination coefficient. " + blank),
                ("Auger C", "C_auger", "cm⁶/s", "Auger coefficient, used for electrons and holes. " + blank)):
            row = self._row_entry(label, attr, getattr(s, attr), cast=_optional_positive, unit=unit, tip=tip)
            self._field_vars[attr][0].setPlaceholderText("material")
            adv_layout.addLayout(row)
        layout.addWidget(self._adv_widget)
        self._toggle_advanced(False)

    # ------------------------------------------------------------------
    def _row_entry(self, label, attr, initial, cast=float, unit="", tip=""):
        text = "" if initial is None else f"{initial:.10g}" if isinstance(initial, float) else str(initial)
        edit = QtWidgets.QLineEdit(text)
        edit.textEdited.connect(lambda text, attr=attr, cast=cast: self._debounced_set_cast(attr, text, cast))
        self._field_vars[attr] = (edit, cast)
        return widgets.form_row(label, edit, unit, tip)

    def _set(self, attr, value):
        setattr(self.model.settings, attr, value)
        self.model.notify()

    def _set_cast(self, attr, text, cast):
        try:
            value = cast(text)
        except ValueError:
            return
        self._set(attr, value)

    def _debounced_set_cast(self, attr, text, cast):
        # Debounced like LayerEditorPanel's fields: applying on every
        # keystroke makes the whole UI feel sluggish, since every model
        # change rebuilds the layer stack panel and restarts the
        # solve-debounce timer.
        timer = self._apply_timers.get(attr)
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda attr=attr, cast=cast: self._flush_one(attr, cast))
            self._apply_timers[attr] = timer
        timer.start(_APPLY_DEBOUNCE_MS)

    def _flush_one(self, attr, cast):
        edit, _cast = self._field_vars[attr]
        self._set_cast(attr, edit.text(), cast)

    def flush_pending(self) -> None:
        """Immediately apply any field edit still waiting on its debounce
        timer (e.g. Applied bias), instead of leaving it to land up to
        400ms later. Called before a solve starts (see
        App._request_solve_now) so a value just typed isn't silently
        solved-over with its pre-edit value -- this was letting "Run"
        clicked right after typing a new bias run at the *old* bias
        with no visible error, looking like the biased solve was ignored."""
        for attr, timer in list(self._apply_timers.items()):
            if timer.isActive():
                timer.stop()
                edit, cast = self._field_vars[attr]
                self._set_cast(attr, edit.text(), cast)

    # ------------------------------------------------------------------
    def _toggle_advanced(self, expanded: bool):
        from gui import icons
        self._adv_widget.setVisible(expanded)
        self._adv_cb.setIcon(icons.icon("chevron_down" if expanded else "chevron_right"))

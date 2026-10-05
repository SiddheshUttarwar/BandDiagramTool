"""
ContactsPanel: bottom/top contact type + metal selection.
SettingsPanel: temperature, grid resolution, quantum toggle, bias/sweep
controls, and a collapsible "Advanced solver settings" section.

Both panels only ever push changes into DeviceModel (which calls
model.notify() itself) — they don't rebuild their own widgets in response
to edits, so typing/selecting is never interrupted.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from devices.layer import Contact
from physics.materials.metals import METAL_WORK_FUNCTIONS
from gui.models import DeviceModel

_METALS = sorted(METAL_WORK_FUNCTIONS.keys())
_APPLY_DEBOUNCE_MS = 400


class ContactsPanel(QtWidgets.QGroupBox):
    def __init__(self, model: DeviceModel, parent=None):
        super().__init__("Contacts", parent)
        self.model = model
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self._build_row("Bottom", model.bottom_contact, model.set_bottom_contact))
        layout.addLayout(self._build_row("Top", model.top_contact, model.set_top_contact))

    def _build_row(self, label, contact: Contact, setter):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(50)
        row.addWidget(lbl)

        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(["ohmic", "schottky"])
        type_combo.setCurrentText(contact.contact_type)

        metal_combo = QtWidgets.QComboBox()
        metal_combo.addItems(_METALS)
        metal_combo.setCurrentText(contact.metal)

        # Explicit Schottky barrier height (Ec - Ef at the metal); 0 shows as
        # "auto" = Schottky-Mott estimate work_function - chi.
        barrier_spin = QtWidgets.QDoubleSpinBox()
        barrier_spin.setRange(0.0, 6.0)
        barrier_spin.setDecimals(2)
        barrier_spin.setSingleStep(0.05)
        barrier_spin.setSuffix(" eV")
        barrier_spin.setSpecialValueText("auto")
        barrier_spin.setMinimumWidth(78)
        barrier_spin.setToolTip("Schottky barrier height (Ec - Ef at the contact). "
                                "'auto' = Schottky-Mott estimate: metal work function - electron affinity.")
        barrier_spin.setValue(contact.barrier_eV if getattr(contact, 'barrier_eV', None) else 0.0)
        barrier_spin.setEnabled(contact.contact_type == 'schottky')

        def apply(_=None):
            barrier_spin.setEnabled(type_combo.currentText() == 'schottky')
            barrier = barrier_spin.value()
            setter(Contact(position=contact.position, contact_type=type_combo.currentText(),
                            metal=metal_combo.currentText(),
                            barrier_eV=barrier if barrier > 0.0 else None))

        type_combo.currentTextChanged.connect(apply)
        metal_combo.currentTextChanged.connect(apply)
        barrier_spin.valueChanged.connect(apply)

        row.addWidget(type_combo)
        row.addWidget(metal_combo)
        row.addWidget(barrier_spin)
        row.addStretch(1)
        return row


class SettingsPanel(QtWidgets.QGroupBox):
    def __init__(self, model: DeviceModel, parent=None):
        super().__init__("Settings", parent)
        self.model = model
        self._apply_timers: dict = {}
        self._field_vars: dict = {}   # attr -> (QLineEdit, cast) for flush_pending
        s = model.settings

        layout = QtWidgets.QVBoxLayout(self)

        layout.addLayout(self._row_entry("Temperature (K)", "T", s.T))
        layout.addLayout(self._row_entry("Grid spacing (nm)", "dx_nm", s.dx_nm))

        quantum_cb = QtWidgets.QCheckBox("Quantum (Schrödinger-Poisson)")
        quantum_cb.setChecked(s.quantum)
        quantum_cb.toggled.connect(lambda v: self._set("quantum", v))
        layout.addWidget(quantum_cb)

        flat_qfl_cb = QtWidgets.QCheckBox("Flat quasi-Fermi levels (Efn − Efp = qV, no current)")
        flat_qfl_cb.setToolTip(
            "Hold Efn at the n-contact level and Efp at the p-contact level at every "
            "interior grid point (split = qV everywhere except the contacts) and solve "
            "only (Schrödinger-)Poisson. Fast and robust for band diagrams / QCSE under "
            "bias; gives no current information.")
        flat_qfl_cb.setChecked(s.flat_qfl)
        flat_qfl_cb.toggled.connect(lambda v: self._set("flat_qfl", v))
        layout.addWidget(flat_qfl_cb)

        psp_cb = QtWidgets.QCheckBox("Add spontaneous polarization in band diagram calculations")
        psp_cb.setChecked(s.include_spontaneous_polarization)
        psp_cb.toggled.connect(lambda v: self._set("include_spontaneous_polarization", v))
        layout.addWidget(psp_cb)

        # Crystal polarity and polarization parameter set
        pol_row = QtWidgets.QHBoxLayout()
        pol_row.addWidget(QtWidgets.QLabel("Polarity"))
        polarity_combo = QtWidgets.QComboBox()
        for label, value in (("Metal-polar [0001]", "metal"), ("N-polar [000-1]", "N")):
            polarity_combo.addItem(label, value)
        polarity_combo.setCurrentIndex(max(0, polarity_combo.findData(s.polarity)))
        polarity_combo.setToolTip("N-polar growth reverses the polarization along the growth axis.")
        polarity_combo.currentIndexChanged.connect(
            lambda _i, c=polarity_combo: self._set("polarity", c.currentData()))
        pol_row.addWidget(polarity_combo, 1)
        layout.addLayout(pol_row)

        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("Polarization constants"))
        model_combo = QtWidgets.QComboBox()
        for label, value in (("Ambacher 2002 (default)", "ambacher2002"), ("Dreyer 2016", "dreyer2016")):
            model_combo.addItem(label, value)
        model_combo.setCurrentIndex(max(0, model_combo.findData(s.polarization_model)))
        model_combo.setToolTip(
            "Ambacher 2002: zincblende-referenced Psp with proper piezoelectric constants. "
            "Dreyer 2016: hexagonal-referenced Psp with improper e31 (PRX 6, 021038).")
        model_combo.currentIndexChanged.connect(
            lambda _i, c=model_combo: self._set("polarization_model", c.currentData()))
        model_row.addWidget(model_combo, 1)
        layout.addLayout(model_row)

        layout.addWidget(self._separator())

        # --- Bias controls (sweep toggle + single/sweep fields) ---
        self._classical_widget = QtWidgets.QWidget()
        classical_layout = QtWidgets.QVBoxLayout(self._classical_widget)
        classical_layout.setContentsMargins(0, 0, 0, 0)

        self.sweep_cb = QtWidgets.QCheckBox("Voltage sweep mode")
        self.sweep_cb.setChecked(s.sweep_mode)
        self.sweep_cb.toggled.connect(self._toggle_sweep)
        classical_layout.addWidget(self.sweep_cb)

        self._single_widget = QtWidgets.QWidget()
        single_layout = QtWidgets.QVBoxLayout(self._single_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addLayout(self._row_entry("Applied bias (V)", "V_applied", s.V_applied))
        classical_layout.addWidget(self._single_widget)

        self._sweep_widget = QtWidgets.QWidget()
        sweep_layout = QtWidgets.QVBoxLayout(self._sweep_widget)
        sweep_layout.setContentsMargins(0, 0, 0, 0)
        sweep_layout.addLayout(self._row_entry("V start", "V_start", s.V_start))
        sweep_layout.addLayout(self._row_entry("V stop", "V_stop", s.V_stop))
        sweep_layout.addLayout(self._row_entry("Steps", "n_steps", s.n_steps, cast=int))
        classical_layout.addWidget(self._sweep_widget)
        self._toggle_sweep(s.sweep_mode)

        layout.addWidget(self._classical_widget)

        layout.addWidget(self._separator())
        self._adv_cb = QtWidgets.QCheckBox("Advanced solver settings")
        self._adv_cb.setChecked(False)
        self._adv_cb.toggled.connect(self._toggle_advanced)
        layout.addWidget(self._adv_cb)

        self._adv_widget = QtWidgets.QWidget()
        adv_layout = QtWidgets.QVBoxLayout(self._adv_widget)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.addLayout(self._row_entry("Max iterations", "max_iter", s.max_iter, cast=int))
        adv_layout.addLayout(self._row_entry("Tolerance (V)", "tol", s.tol))
        adv_layout.addLayout(self._row_entry("Damping alpha", "alpha", s.alpha))
        adv_layout.addLayout(self._row_entry("Electron subbands", "n_states_e", s.n_states_e, cast=int))
        adv_layout.addLayout(self._row_entry("Hole subbands", "n_states_h", s.n_states_h, cast=int))
        layout.addWidget(self._adv_widget)
        self._adv_widget.setVisible(False)

    @staticmethod
    def _separator() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    # ------------------------------------------------------------------
    def _row_entry(self, label, attr, initial, cast=float):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(130)
        row.addWidget(lbl)
        edit = QtWidgets.QLineEdit(str(initial))
        edit.setMaximumWidth(100)
        edit.textEdited.connect(lambda text, attr=attr, cast=cast: self._debounced_set_cast(attr, text, cast))
        row.addWidget(edit)
        row.addStretch(1)
        self._field_vars[attr] = (edit, cast)
        return row

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
        solved-over with its pre-edit value -- this was letting "Solve
        Now" clicked right after typing a new bias run at the *old* bias
        with no visible error, looking like the biased solve was ignored."""
        for attr, timer in list(self._apply_timers.items()):
            if timer.isActive():
                timer.stop()
                edit, cast = self._field_vars[attr]
                self._set_cast(attr, edit.text(), cast)

    # ------------------------------------------------------------------
    def _toggle_sweep(self, sweep: bool):
        self._set("sweep_mode", sweep)
        self._single_widget.setVisible(not sweep)
        self._sweep_widget.setVisible(sweep)

    def _toggle_advanced(self, expanded: bool):
        self._adv_widget.setVisible(expanded)

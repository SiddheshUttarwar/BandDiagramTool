"""
LayerStackPanel: the device as a table, one row per layer, surface at the
top and substrate at the bottom — although DeviceModel.layers stays in the
codebase's bottom->top convention (index 0 = substrate, as in
devices.device.AlGaNDevice).

Columns: position from the substrate, material (with its colour from
theme.material_color), thickness and doping. Zero-thickness entries
(quantum-region markers, surface states, interface dipoles) are single
spanning rows. New / Delete / Up / Down act on the selected row.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from physics.materials.algan import nitride_name
from devices.layer import (
    AbruptLayer, GradedLayer, QuantumRegionMarker, SurfaceCharge, InterfaceDipole,
)
from gui.models import DeviceModel
from gui import theme

_INTERFACE_TYPES = (QuantumRegionMarker, SurfaceCharge, InterfaceDipole)
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_ROLE_INDEX = QtCore.Qt.ItemDataRole.UserRole


def _layer_summary(layer) -> tuple[str, str]:
    """Return (title, subtitle) describing a layer in plain text."""
    if isinstance(layer, AbruptLayer):
        title = nitride_name(layer.x_Al, getattr(layer, "x_In", 0.0))
        bits = [f"{layer.thickness_nm:g} nm"]
        if layer.n_doping > 0:
            bits.append(f"n={layer.n_doping:.1e}")
        if layer.p_doping > 0:
            bits.append(f"p={layer.p_doping:.1e}")
        return title, "  ·  ".join(bits)
    if isinstance(layer, GradedLayer):
        title = (f"{nitride_name(layer.x_Al_start, getattr(layer, 'x_In_start', 0.0))}"
                 f" → {nitride_name(layer.x_Al_end, getattr(layer, 'x_In_end', 0.0))}")
        bits = [f"{layer.thickness_nm:g} nm", layer.profile]
        if layer.n_doping > 0:
            bits.append(f"n={layer.n_doping:.1e}")
        if layer.p_doping > 0:
            bits.append(f"p={layer.p_doping:.1e}")
        return title, "  ·  ".join(bits)
    if isinstance(layer, QuantumRegionMarker):
        return ("Quantum region start" if layer.boundary == 'start' else "Quantum region end"), ""
    if isinstance(layer, SurfaceCharge):
        n = len(layer.states)
        return "Surface states", f"{n} level{'s' if n != 1 else ''}"
    if isinstance(layer, InterfaceDipole):
        return "Interface dipole", f"{theme.sci(layer.sheet_charge_C_m2)} C/m²"
    return "Layer", ""


def layer_colors(layer) -> Optional[tuple[str, str]]:
    """(colour at the bottom face, colour at the top face) of a physical
    layer; None for zero-thickness entries."""
    if isinstance(layer, AbruptLayer):
        c = theme.material_color(layer.x_Al, getattr(layer, "x_In", 0.0))
        return c, c
    if isinstance(layer, GradedLayer):
        return (theme.material_color(layer.x_Al_start, getattr(layer, "x_In_start", 0.0)),
                theme.material_color(layer.x_Al_end, getattr(layer, "x_In_end", 0.0)))
    return None


def _swatch(bottom: str, top: str) -> QtGui.QIcon:
    pm = QtGui.QPixmap(28, 28)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    grad = QtGui.QLinearGradient(0, 3, 0, 25)
    grad.setColorAt(0.0, QtGui.QColor(top))
    grad.setColorAt(1.0, QtGui.QColor(bottom))
    p.setPen(QtGui.QColor(theme.shade(bottom, 0.35)))
    p.setBrush(grad)
    p.drawRect(3, 3, 22, 22)
    p.end()
    return QtGui.QIcon(pm)


def _doping_text(layer) -> str:
    bits = []
    if layer.n_doping > 0:
        bits.append(f"n  {theme.sci(layer.n_doping, 1)}")
    if layer.p_doping > 0:
        bits.append(f"p  {theme.sci(layer.p_doping, 1)}")
    return ",  ".join(bits) if bits else "—"


class LayerStackPanel(QtWidgets.QWidget):
    """on_select(index_or_None) fires when the selected row changes (index
    is in the model's bottom->top list space)."""

    def __init__(self, model: DeviceModel,
                 on_select: Callable[[Optional[int]], None], parent=None):
        super().__init__(parent)
        self.model = model
        self.on_select = on_select
        self.selected_index: Optional[int] = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._summary = QtWidgets.QLabel("")
        root.addWidget(self._summary)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", "Material", "d (nm)", "Doping"])
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(True)
        self._table.setIconSize(QtCore.QSize(14, 14))
        self._table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self._table.horizontalHeader()
        header.setHighlightSections(False)
        self._table.horizontalHeaderItem(3).setToolTip("Donor (n) or acceptor (p) concentration, cm⁻³")
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.setMinimumHeight(232)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        row.addWidget(self._table, 1)

        buttons = QtWidgets.QVBoxLayout()
        buttons.setSpacing(4)
        self._new_btn = QtWidgets.QPushButton("New")
        self._new_btn.setToolTip("Add a layer or interface on top of the stack")
        self._new_btn.setMenu(self.build_add_menu(self._new_btn))
        self._del_btn = QtWidgets.QPushButton("Delete")
        self._del_btn.clicked.connect(lambda _c=False: self.delete_selected())
        self._up_btn = QtWidgets.QPushButton("Up")
        self._up_btn.setToolTip("Move the selected layer toward the surface")
        self._up_btn.clicked.connect(lambda _c=False: self._move(+1))
        self._down_btn = QtWidgets.QPushButton("Down")
        self._down_btn.setToolTip("Move the selected layer toward the substrate")
        self._down_btn.clicked.connect(lambda _c=False: self._move(-1))
        for b in (self._new_btn, self._del_btn, self._up_btn, self._down_btn):
            b.setFixedWidth(62)
            buttons.addWidget(b)
        buttons.addStretch(1)
        row.addLayout(buttons)
        root.addLayout(row, 1)

        self.refresh()

    # ------------------------------------------------------------------
    def build_add_menu(self, parent=None) -> QtWidgets.QMenu:
        """The "New" menu, shared with the main window's Edit menu."""
        menu = QtWidgets.QMenu(parent)
        menu.addAction("Layer", self._add_abrupt)
        menu.addAction("Graded layer", self._add_graded)
        menu.addSeparator()
        menu.addAction("Quantum region marker", self._add_quantum_marker)
        menu.addAction("Surface states", self._add_surface_charge)
        menu.addAction("Interface dipole", self._add_interface_dipole)
        return menu

    def delete_selected(self) -> None:
        if self.selected_index is not None:
            self._delete(self.selected_index)

    def _add_abrupt(self):
        self.model.add_layer(AbruptLayer(x_Al=0.0, thickness_nm=10.0))
        self.select(len(self.model.layers) - 1)

    def _add_graded(self):
        self.model.add_layer(GradedLayer(x_Al_start=0.0, x_Al_end=0.2, thickness_nm=10.0))
        self.select(len(self.model.layers) - 1)

    def _add_quantum_marker(self):
        self.model.add_layer(QuantumRegionMarker(boundary='start'))
        self.select(len(self.model.layers) - 1)

    def _add_surface_charge(self):
        from devices.layer import SurfaceState
        self.model.add_layer(SurfaceCharge(
            states=[SurfaceState(density_cm2=1e12, energy_eV=0.1, state_type='donor')]))
        self.select(len(self.model.layers) - 1)

    def _add_interface_dipole(self):
        self.model.add_layer(InterfaceDipole(sheet_charge_C_m2=1e-3, separation_nm=0.5))
        self.select(len(self.model.layers) - 1)

    # ------------------------------------------------------------------
    def select(self, index: Optional[int]):
        self.selected_index = index
        self.refresh()
        self.on_select(index)

    def _on_selection_changed(self):
        rows = self._table.selectionModel().selectedRows()
        index = None
        if rows:
            item = self._table.item(rows[0].row(), 0)
            index = item.data(_ROLE_INDEX) if item is not None else None
        if index != self.selected_index:
            self.selected_index = index
            self._update_buttons()
            self.on_select(index)

    def _update_buttons(self):
        n = len(self.model.layers)
        has = self.selected_index is not None
        self._del_btn.setEnabled(has)
        self._up_btn.setEnabled(has and self.selected_index < n - 1)
        self._down_btn.setEnabled(has and self.selected_index > 0)

    def refresh(self):
        layers = self.model.layers
        n = len(layers)
        physical = [l for l in layers if not isinstance(l, _INTERFACE_TYPES)]
        total = sum(l.thickness_nm for l in physical)
        self._summary.setText(
            f"{len(physical)} layer{'s' if len(physical) != 1 else ''}, {total:g} nm in total  "
            f"(surface at top)" if physical else "No layers. Use New to add one.")

        scroll = self._table.verticalScrollBar().value()
        self._table.blockSignals(True)
        self._table.clearSpans()
        self._table.setRowCount(n)
        right = int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # Visual order is reversed vs. the bottom->top model list: the top
        # layer (last in the list) is the first row.
        for row, model_index in enumerate(range(n - 1, -1, -1)):
            layer = layers[model_index]
            num = QtWidgets.QTableWidgetItem(str(model_index + 1))
            num.setData(_ROLE_INDEX, model_index)
            num.setTextAlignment(right)
            num.setForeground(QtGui.QColor(theme.TEXT_FAINT))
            self._table.setItem(row, 0, num)
            title, sub = _layer_summary(layer)
            if isinstance(layer, _INTERFACE_TYPES):
                item = QtWidgets.QTableWidgetItem(f"— {title}{'  (' + sub + ')' if sub else ''} —")
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setForeground(QtGui.QColor(theme.TEXT_MUTED))
                self._table.setItem(row, 1, item)
                self._table.setSpan(row, 1, 1, 3)
                continue
            name = QtWidgets.QTableWidgetItem(title.translate(_SUB))
            name.setIcon(_swatch(*layer_colors(layer)))
            self._table.setItem(row, 1, name)
            thick = QtWidgets.QTableWidgetItem(f"{layer.thickness_nm:g}")
            thick.setTextAlignment(right)
            self._table.setItem(row, 2, thick)
            self._table.setItem(row, 3, QtWidgets.QTableWidgetItem(_doping_text(layer)))

        if self.selected_index is not None and 0 <= self.selected_index < n:
            self._table.selectRow(n - 1 - self.selected_index)
        else:
            self._table.clearSelection()
        self._table.blockSignals(False)
        self._table.verticalScrollBar().setValue(scroll)
        self._update_buttons()

    def _move(self, step: int):
        i = self.selected_index
        if i is None:
            return
        j = i + step
        if not 0 <= j < len(self.model.layers):
            return
        self.selected_index = j
        self.model.move_layer(i, j)
        self.on_select(j)

    def _delete(self, model_index: int):
        self.model.remove_layer(model_index)
        if self.selected_index == model_index:
            self.selected_index = None
            self.on_select(None)
        elif self.selected_index is not None and self.selected_index > model_index:
            self.selected_index -= 1
        self.refresh()

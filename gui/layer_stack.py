"""
LayerStackPanel: a vertical, drag-to-reorder list of layer "cards" mirroring
the device's physical cross-section (top layer's card drawn at the top of
the window, bottom layer's card at the bottom) even though the underlying
DeviceModel.layers list stays in the codebase's bottom->top convention
(index 0 = substrate/bottom, matching devices.device.AlGaNDevice).

Reordering uses QListWidget's native internal-move drag & drop (dragging any
part of a card moves it) rather than hand-rolled mouse tracking.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from physics.materials.algan import nitride_name
from devices.layer import (
    AbruptLayer, GradedLayer, QuantumRegionMarker, SurfaceCharge, InterfaceDipole,
)
from gui.models import DeviceModel
from gui import theme

_CARD_BG = theme.CARD_BG
_CARD_BG_SELECTED = theme.CARD_BG_SELECTED
_CARD_BG_INTERFACE = theme.CARD_BG_INTERFACE   # zero-thickness interface layers get a tint
_CARD_BG_INTERFACE_SELECTED = theme.CARD_BG_INTERFACE_SELECTED
_CARD_BORDER = theme.CARD_BORDER
_CARD_BORDER_SELECTED = theme.CARD_BORDER_SELECTED

_INTERFACE_TYPES = (QuantumRegionMarker, SurfaceCharge, InterfaceDipole)


def _layer_summary(layer) -> tuple[str, str]:
    """Return (title, subtitle) describing a layer for its card."""
    if isinstance(layer, AbruptLayer):
        title = nitride_name(layer.x_Al, getattr(layer, "x_In", 0.0))
        bits = [f"{layer.thickness_nm:g} nm"]
        if layer.n_doping > 0:
            bits.append(f"n={layer.n_doping:.1e}")
        if layer.p_doping > 0:
            bits.append(f"p={layer.p_doping:.1e}")
        return title, "  ·  ".join(bits)
    if isinstance(layer, GradedLayer):
        title = (f"Graded {nitride_name(layer.x_Al_start, getattr(layer, 'x_In_start', 0.0))}"
                 f" → {nitride_name(layer.x_Al_end, getattr(layer, 'x_In_end', 0.0))}")
        bits = [f"{layer.thickness_nm:g} nm", layer.profile]
        if layer.n_doping > 0:
            bits.append(f"n={layer.n_doping:.1e}")
        if layer.p_doping > 0:
            bits.append(f"p={layer.p_doping:.1e}")
        return title, "  ·  ".join(bits)
    if isinstance(layer, QuantumRegionMarker):
        title = "▶ Quantum region start" if layer.boundary == 'start' else "Quantum region end ◀"
        return title, "marker · 0 nm"
    if isinstance(layer, SurfaceCharge):
        title = "Surface charge"
        n = len(layer.states)
        bits = [f"{n} state{'s' if n != 1 else ''}"]
        for s in layer.states[:3]:
            bits.append(f"{s.state_type[0].upper()} {s.density_cm2:.1e}cm⁻² @{s.energy_eV:g}eV")
        return title, "  ·  ".join(bits)
    if isinstance(layer, InterfaceDipole):
        title = "Interface dipole"
        return title, f"±{layer.sheet_charge_C_m2:.2e} C/m²  ·  {layer.separation_nm:g} nm sep."
    return "Layer", ""


class _Card(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal()
    delete_requested = QtCore.pyqtSignal()

    def __init__(self, title: str, subtitle: str, selected: bool, is_interface: bool,
                 swatch: Optional[str] = None):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        if selected:
            bg = _CARD_BG_INTERFACE_SELECTED if is_interface else _CARD_BG_SELECTED
            border = _CARD_BORDER_SELECTED
        else:
            bg = _CARD_BG_INTERFACE if is_interface else _CARD_BG
            border = _CARD_BORDER
        # A flat row: 1px outline, and a left bar in the layer's material
        # colour (see theme.material_color) so the stack reads as a cross
        # section at a glance. Marker rows have no bar.
        bar = swatch or "transparent"
        self.setStyleSheet(
            f"_Card {{ background-color: {bg}; border: 1px solid {border}; "
            f"border-left: 5px solid {bar if swatch else border}; border-radius: 0px; }}")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(6)

        handle = QtWidgets.QLabel("⋮⋮")
        handle.setToolTip("Drag to reorder")
        handle.setStyleSheet(f"color: {theme.TEXT_FAINT}; border: none; background: transparent;")
        layout.addWidget(handle)

        text_col = QtWidgets.QVBoxLayout()
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet(
            f"border: none; background: transparent; font-weight: 600; color: {theme.TEXT};")
        text_col.addWidget(title_lbl)
        if subtitle:
            sub_lbl = QtWidgets.QLabel(subtitle)
            sub_lbl.setStyleSheet(
                f"border: none; background: transparent; color: {theme.TEXT_MUTED}; font-size: 8pt;")
            text_col.addWidget(sub_lbl)
        layout.addLayout(text_col, 1)

        del_btn = QtWidgets.QToolButton()
        del_btn.setText("✕")
        del_btn.setStyleSheet(
            f"QToolButton {{ border: none; background: transparent; color: {theme.TEXT_FAINT}; }}"
            f"QToolButton:hover {{ background: {theme.DANGER_SOFT}; color: {theme.DANGER}; border-radius: 2px; }}")
        del_btn.setToolTip("Delete this layer")
        del_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        del_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(del_btn)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class LayerStackPanel(QtWidgets.QWidget):
    """on_select(index_or_None) fires when a card is clicked or dropped
    after a drag (index is in the model's bottom->top list space)."""

    def __init__(self, model: DeviceModel,
                 on_select: Callable[[Optional[int]], None], parent=None):
        super().__init__(parent)
        self.model = model
        self.on_select = on_select
        self.selected_index: Optional[int] = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Surface (top)")
        title.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        header.addWidget(title)
        header.addStretch(1)

        add_btn = QtWidgets.QToolButton()
        add_btn.setText("Add Layer")
        add_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        add_btn.setMenu(self.build_add_menu(add_btn))
        header.addWidget(add_btn)
        root.addLayout(header)

        self._list = QtWidgets.QListWidget()
        self._list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setStyleSheet(
            f"QListWidget {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER}; }}")
        self._list.setSpacing(1)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        root.addWidget(self._list, 1)

        self._empty_label = QtWidgets.QLabel('No layers yet.\nUse "Add Layer" to build the stack,\nfrom the substrate upward.')
        self._empty_label.setStyleSheet(f"color: {theme.TEXT_FAINT};")
        self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty_label, 1)

        footer = QtWidgets.QLabel("Substrate (bottom)")
        footer.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        root.addWidget(footer)

        self.refresh()

    # ------------------------------------------------------------------
    def build_add_menu(self, parent=None) -> QtWidgets.QMenu:
        """The "Add Layer" menu, shared with the toolstrip button."""
        menu = QtWidgets.QMenu(parent)
        menu.addAction("Abrupt layer", self._add_abrupt)
        menu.addAction("Graded layer", self._add_graded)
        menu.addSeparator()
        menu.addAction("Quantum region marker", self._add_quantum_marker)
        menu.addAction("Surface charge", self._add_surface_charge)
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

    def refresh(self):
        n = len(self.model.layers)
        self._empty_label.setVisible(n == 0)
        self._list.setVisible(n > 0)

        self._list.blockSignals(True)
        self._list.clear()
        # Visual order is reversed vs. the bottom->top model list: the top
        # layer (last in the list) is drawn first (at the top of the panel).
        for model_index in range(n - 1, -1, -1):
            self._build_card(model_index)
        self._list.blockSignals(False)

    def _build_card(self, model_index: int):
        layer = self.model.layers[model_index]
        selected = (model_index == self.selected_index)
        is_interface = isinstance(layer, _INTERFACE_TYPES)
        title, subtitle = _layer_summary(layer)

        swatch = None
        if isinstance(layer, AbruptLayer):
            swatch = theme.material_color(layer.x_Al, getattr(layer, "x_In", 0.0))
        elif isinstance(layer, GradedLayer):
            swatch = theme.material_color(
                0.5 * (layer.x_Al_start + layer.x_Al_end),
                0.5 * (getattr(layer, "x_In_start", 0.0) + getattr(layer, "x_In_end", 0.0)))
        card = _Card(title, subtitle, selected, is_interface, swatch)
        card.clicked.connect(lambda i=model_index: self.select(i))
        card.delete_requested.connect(lambda i=model_index: self._delete(i))

        item = QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.UserRole, model_index)
        item.setSizeHint(card.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, card)

    def _delete(self, model_index: int):
        self.model.remove_layer(model_index)
        if self.selected_index == model_index:
            self.selected_index = None
            self.on_select(None)
        elif self.selected_index is not None and self.selected_index > model_index:
            self.selected_index -= 1
        self.refresh()

    # ------------------------------------------------------------------
    def _on_rows_moved(self, *_args):
        """Fires after a drag-and-drop internal move completes. Reads the
        list's current (top-of-screen-first) visual order back out via each
        item's stored original model_index, rebuilds DeviceModel.layers to
        match (bottom->top), and re-renders so every card's stored index is
        fresh again."""
        visual_order_old_indices = [
            self._list.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(self._list.count())
        ]
        old_layers = self.model.layers
        new_layers = [old_layers[i] for i in reversed(visual_order_old_indices)]
        if new_layers == old_layers:
            return

        if self.selected_index is not None and self.selected_index < len(old_layers):
            moved_layer = old_layers[self.selected_index]
            self.selected_index = next(
                i for i, layer in enumerate(new_layers) if layer is moved_layer)

        self.model.layers = new_layers
        self.model.notify()

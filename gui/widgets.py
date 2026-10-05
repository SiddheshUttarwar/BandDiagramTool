"""
Small building blocks shared by the side-panel pages: group boxes, the
label / field / unit form row, sub-headings, a radio-button choice and icon
buttons. Native widgets throughout.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PyQt6 import QtCore, QtWidgets

from gui import icons, theme

LABEL_WIDTH = 96
UNIT_WIDTH = 34


class Section(QtWidgets.QGroupBox):
    """A titled group box. Content goes in `self.body`; `self.meta_label`
    is a line of secondary text at the top."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(4)
        self.meta_label = QtWidgets.QLabel("")
        self.meta_label.setObjectName("sectionMeta")
        self.meta_label.setVisible(False)
        outer.addWidget(self.meta_label)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(4)
        outer.addLayout(self.body)

    def set_meta(self, text: str) -> None:
        self.meta_label.setText(text)
        self.meta_label.setVisible(bool(text))


def form_row(label: str, widget: QtWidgets.QWidget, unit: str = "", tip: str = "") -> QtWidgets.QHBoxLayout:
    """[label][field][unit] on one line; the unit column is always reserved
    so fields line up down the panel."""
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(6)
    lbl = QtWidgets.QLabel(label + ":")
    lbl.setFixedWidth(LABEL_WIDTH)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    unit_lbl = QtWidgets.QLabel(unit)
    unit_lbl.setObjectName("unit")
    unit_lbl.setFixedWidth(UNIT_WIDTH)
    row.addWidget(unit_lbl)
    if tip:
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
    return row


def subhead(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("subhead")
    return lbl


def note(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("note")
    lbl.setWordWrap(True)
    return lbl


def hairline() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return line


def icon_button(icon_name: str, tooltip: str = "", callback: Optional[Callable] = None,
                checkable: bool = False, color: str = theme.TEXT, size: int = 16) -> QtWidgets.QToolButton:
    btn = QtWidgets.QToolButton()
    btn.setIcon(icons.icon(icon_name, color))
    btn.setIconSize(QtCore.QSize(size, size))
    btn.setToolTip(tooltip)
    btn.setCheckable(checkable)
    btn.setAutoRaise(True)
    btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    if callback is not None:
        if checkable:
            btn.toggled.connect(lambda state: callback(state))
        else:
            btn.clicked.connect(lambda _checked=False: callback())
    return btn


class Segmented(QtWidgets.QWidget):
    """Mutually exclusive choices as a row of radio buttons."""

    changed = QtCore.pyqtSignal(int)

    def __init__(self, options: Sequence[str], current: int = 0, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        self._group = QtWidgets.QButtonGroup(self)
        for i, text in enumerate(options):
            btn = QtWidgets.QRadioButton(text)
            btn.setChecked(i == current)
            self._group.addButton(btn, i)
            row.addWidget(btn)
        row.addStretch(1)
        self._group.idClicked.connect(self.changed.emit)

    def current(self) -> int:
        return self._group.checkedId()

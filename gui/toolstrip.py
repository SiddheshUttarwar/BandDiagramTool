"""
Toolstrip: the tabbed ribbon across the top of the main window, in the
manner of the MATLAB desktop -- a dark tab bar (HOME / PLOTS / VIEW) over a
light body divided into labelled sections of large (icon above text) and
small (icon beside text) buttons.

The widget is purely presentational: App builds the tabs by calling
add_tab / Section.add_large / add_small with its own callbacks.

Icons are drawn here with QPainter instead of shipped as image files, so
they stay sharp at any display scaling and the package has no binary assets.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from gui import theme

_INK = "#3A3A3A"
_BLUE = theme.ACCENT
_GREEN = "#3A9B35"
_RED = "#D0392B"
_AMBER = "#E0A526"
_PAPER = "#FFFFFF"


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------
def _pen(color: str, width: float = 1.6) -> QtGui.QPen:
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidthF(width)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    return pen


def _poly(points) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])


def _draw_new(p):
    page = _poly([(7, 4), (18, 4), (25, 11), (25, 28), (7, 28)])
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_PAPER)); p.drawPolygon(page)
    p.drawPolyline(_poly([(18, 4), (18, 11), (25, 11)]))
    p.setPen(_pen(_GREEN, 2.4)); p.drawLine(QtCore.QPointF(16, 16), QtCore.QPointF(16, 24))
    p.drawLine(QtCore.QPointF(12, 20), QtCore.QPointF(20, 20))


def _draw_open(p):
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_AMBER))
    p.drawPolygon(_poly([(4, 9), (13, 9), (15, 12), (27, 12), (27, 25), (4, 25)]))
    p.setBrush(QtGui.QColor("#F5D27A"))
    p.drawPolygon(_poly([(8, 15), (30, 15), (26, 25), (4, 25)]))


def _draw_save(p):
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_BLUE))
    p.drawPolygon(_poly([(5, 5), (22, 5), (27, 10), (27, 27), (5, 27)]))
    p.setBrush(QtGui.QColor(_PAPER))
    p.drawRect(QtCore.QRectF(10, 5, 10, 7)); p.drawRect(QtCore.QRectF(9, 17, 14, 10))
    p.setPen(_pen(_INK, 1.2))
    p.drawLine(QtCore.QPointF(12, 21), QtCore.QPointF(20, 21))
    p.drawLine(QtCore.QPointF(12, 24), QtCore.QPointF(20, 24))


def _draw_save_as(p):
    _draw_save(p)
    p.setPen(_pen(_INK, 1.2)); p.setBrush(QtGui.QColor(_AMBER))
    p.drawPolygon(_poly([(20, 22), (28, 14), (31, 17), (23, 25), (19, 26)]))


def _draw_layers(p, plus: bool = False):
    colors = [theme.material_color(0.0), theme.material_color(0.5), theme.material_color(1.0)]
    for k, c in enumerate(colors):
        y = 20 - 6 * k
        p.setPen(_pen(_INK, 1.2)); p.setBrush(QtGui.QColor(c))
        p.drawPolygon(_poly([(16, y - 5), (28, y), (16, y + 5), (4, y)]))
    if plus:
        p.setPen(_pen(_PAPER, 5.0)); p.drawLine(QtCore.QPointF(24, 24), QtCore.QPointF(24, 24))
        p.setPen(QtCore.Qt.PenStyle.NoPen); p.setBrush(QtGui.QColor(_GREEN))
        p.drawEllipse(QtCore.QPointF(24, 24), 6.5, 6.5)
        p.setPen(_pen(_PAPER, 2.0))
        p.drawLine(QtCore.QPointF(24, 20.5), QtCore.QPointF(24, 27.5))
        p.drawLine(QtCore.QPointF(20.5, 24), QtCore.QPointF(27.5, 24))


def _draw_run(p):
    p.setPen(_pen("#2B7A27", 1.4)); p.setBrush(QtGui.QColor(_GREEN))
    p.drawPolygon(_poly([(9, 5), (27, 16), (9, 27)]))


def _draw_stop(p):
    p.setPen(_pen("#9E2B20", 1.4)); p.setBrush(QtGui.QColor(_RED))
    p.drawRoundedRect(QtCore.QRectF(7, 7, 18, 18), 2, 2)


def _draw_axes(p):
    p.setPen(_pen(_INK, 1.4)); p.setBrush(QtGui.QColor(_PAPER))
    p.drawRect(QtCore.QRectF(5, 5, 22, 20))


def _curve(p, pts, color, width=2.0):
    path = QtGui.QPainterPath(QtCore.QPointF(*pts[0]))
    for i in range(1, len(pts) - 1, 2):
        path.quadTo(QtCore.QPointF(*pts[i]), QtCore.QPointF(*pts[i + 1]))
    p.setPen(_pen(color, width)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush); p.drawPath(path)


def _draw_plot_bands(p):
    _draw_axes(p)
    _curve(p, [(6, 10), (12, 9), (15, 13), (18, 18), (26, 12)], _BLUE)
    _curve(p, [(6, 18), (12, 17), (15, 21), (18, 25), (26, 20)], _RED)


def _draw_plot_waves(p):
    _draw_axes(p)
    _curve(p, [(6, 20), (11, 20), (13, 14), (16, 6), (19, 14), (21, 20), (26, 20)], _BLUE)
    _curve(p, [(6, 22), (10, 22), (12, 18), (14, 12), (16, 18), (18, 24), (20, 18), (23, 12), (26, 16)], _AMBER, 1.6)


def _draw_plot_carriers(p):
    _draw_axes(p)
    _curve(p, [(6, 23), (13, 23), (15, 12), (17, 7), (19, 14), (22, 22), (26, 23)], _BLUE)
    _curve(p, [(6, 9), (12, 9), (15, 16), (18, 23), (26, 23)], _RED)


def _draw_plot_fields(p):
    _draw_axes(p)
    p.setPen(_pen(_BLUE, 2.0))
    p.drawPolyline(_poly([(6, 20), (13, 20), (13, 9), (19, 9), (19, 16), (26, 16)]))


def _draw_plot_polarization(p):
    _draw_axes(p)
    p.setPen(_pen(_INK, 1.0))
    for x in (11, 16, 21):
        p.setPen(_pen(_BLUE, 1.8)); p.drawLine(QtCore.QPointF(x, 21), QtCore.QPointF(x, 10))
        p.drawPolyline(_poly([(x - 3, 13), (x, 9), (x + 3, 13)]))


def _draw_plot_strain(p):
    _draw_axes(p)
    p.setPen(_pen(_GREEN, 1.8))
    p.drawPolyline(_poly([(8, 15), (12, 15)])); p.drawPolyline(_poly([(10, 12), (7, 15), (10, 18)]))
    p.drawPolyline(_poly([(20, 15), (24, 15)])); p.drawPolyline(_poly([(22, 12), (25, 15), (22, 18)]))
    p.setPen(_pen(_INK, 1.4)); p.setBrush(QtGui.QColor("#DDE9F5")); p.drawRect(QtCore.QRectF(13, 10, 6, 10))


def _draw_plot_qcse(p):
    _draw_axes(p)
    p.setPen(_pen(_INK, 1.4))
    p.drawPolyline(_poly([(6, 9), (12, 11), (12, 20), (20, 23), (20, 14), (26, 16)]))
    _curve(p, [(12, 17), (14, 11), (17, 19)], _RED, 1.8)


def _draw_export(p):
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_PAPER)); p.drawRect(QtCore.QRectF(4, 8, 20, 17))
    _curve(p, [(6, 21), (10, 13), (13, 18), (16, 22), (22, 12)], _BLUE, 1.8)
    p.setPen(_pen(_GREEN, 2.4))
    p.drawLine(QtCore.QPointF(20, 26), QtCore.QPointF(29, 17))
    p.drawPolyline(_poly([(23, 17), (29, 17), (29, 23)]))


def _draw_folder_results(p):
    _draw_open(p)
    p.setPen(_pen(_BLUE, 1.8))
    p.drawPolyline(_poly([(11, 22), (15, 18), (18, 21), (23, 17)]))


def _draw_clear(p):
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_PAPER)); p.drawRect(QtCore.QRectF(5, 6, 22, 20))
    p.setPen(_pen(theme.TEXT_FAINT, 1.2))
    for y in (11, 15, 19):
        p.drawLine(QtCore.QPointF(8, y), QtCore.QPointF(19, y))
    p.setPen(_pen(_RED, 2.2))
    p.drawLine(QtCore.QPointF(20, 19), QtCore.QPointF(27, 26)); p.drawLine(QtCore.QPointF(27, 19), QtCore.QPointF(20, 26))


def _draw_layout(p):
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_PAPER)); p.drawRect(QtCore.QRectF(4, 6, 24, 20))
    p.setBrush(QtGui.QColor(theme.ACCENT_SOFT))
    p.drawRect(QtCore.QRectF(4, 6, 8, 20)); p.drawRect(QtCore.QRectF(12, 19, 16, 7))


def _draw_panel(p):
    p.setPen(_pen(_INK)); p.setBrush(QtGui.QColor(_PAPER)); p.drawRect(QtCore.QRectF(5, 6, 22, 20))
    p.setBrush(QtGui.QColor(theme.HEADER_BG)); p.drawRect(QtCore.QRectF(5, 6, 22, 5))


def _draw_help(p):
    p.setPen(_pen("#005A96", 1.4)); p.setBrush(QtGui.QColor(_BLUE)); p.drawEllipse(QtCore.QPointF(16, 16), 11, 11)
    f = QtGui.QFont(theme.FONT_FAMILY); f.setPixelSize(17); f.setBold(True); p.setFont(f)
    p.setPen(QtGui.QColor(_PAPER))
    p.drawText(QtCore.QRectF(5, 5, 22, 22), int(QtCore.Qt.AlignmentFlag.AlignCenter), "?")


def _draw_info(p):
    p.setPen(_pen(_INK, 1.4)); p.setBrush(QtGui.QColor(_PAPER)); p.drawEllipse(QtCore.QPointF(16, 16), 11, 11)
    f = QtGui.QFont("Georgia"); f.setPixelSize(17); f.setBold(True); f.setItalic(True); p.setFont(f)
    p.setPen(QtGui.QColor(_BLUE))
    p.drawText(QtCore.QRectF(5, 5, 22, 22), int(QtCore.Qt.AlignmentFlag.AlignCenter), "i")


_PAINTERS: Dict[str, Callable] = {
    "new": _draw_new, "open": _draw_open, "save": _draw_save, "save_as": _draw_save_as,
    "layers": _draw_layers, "add_layer": lambda p: _draw_layers(p, plus=True),
    "run": _draw_run, "stop": _draw_stop,
    "plot_bands": _draw_plot_bands, "plot_waves": _draw_plot_waves, "plot_carriers": _draw_plot_carriers,
    "plot_fields": _draw_plot_fields, "plot_polarization": _draw_plot_polarization,
    "plot_strain": _draw_plot_strain, "plot_qcse": _draw_plot_qcse,
    "export": _draw_export, "results": _draw_folder_results, "clear": _draw_clear,
    "layout": _draw_layout, "panel": _draw_panel, "help": _draw_help, "info": _draw_info,
}


def icon(name: str) -> QtGui.QIcon:
    """Vector-drawn icon on a 32x32 design grid, rendered at 1x, 2x and 3x."""
    painter_fn = _PAINTERS[name]
    ico = QtGui.QIcon()
    for scale in (1, 2, 3):
        pm = QtGui.QPixmap(32 * scale, 32 * scale)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        p.scale(scale, scale)
        painter_fn(p)
        p.end()
        ico.addPixmap(pm)
    return ico


# ---------------------------------------------------------------------------
# Toolstrip widgets
# ---------------------------------------------------------------------------
class Section(QtWidgets.QWidget):
    """One labelled group of buttons in a toolstrip tab."""

    def __init__(self, title: str):
        super().__init__()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 0)
        outer.setSpacing(1)
        self._row = QtWidgets.QHBoxLayout()
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(2)
        outer.addLayout(self._row, 1)
        label = QtWidgets.QLabel(title.upper())
        label.setObjectName("stripSectionLabel")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(label)
        self._small_col: Optional[QtWidgets.QVBoxLayout] = None

    def add_large(self, text: str, icon_name: str, callback: Optional[Callable] = None,
                  tooltip: str = "", menu: Optional[QtWidgets.QMenu] = None,
                  shortcut: str = "") -> QtWidgets.QToolButton:
        self._small_col = None
        btn = QtWidgets.QToolButton()
        btn.setObjectName("stripLarge")
        btn.setText(text)
        btn.setIcon(icon(icon_name))
        btn.setIconSize(QtCore.QSize(32, 32))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        tip = tooltip or text.replace("\n", " ")
        btn.setToolTip(f"{tip} ({shortcut})" if shortcut else tip)
        if menu is not None:
            btn.setMenu(menu)
            btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        if callback is not None:
            btn.clicked.connect(lambda _checked=False: callback())
        self._row.addWidget(btn)
        return btn

    def add_small(self, text: str, icon_name: str, callback: Optional[Callable] = None,
                  tooltip: str = "", checkable: bool = False) -> QtWidgets.QToolButton:
        """Small buttons stack three to a column."""
        if self._small_col is None or self._small_col.count() >= 3:
            self._small_col = QtWidgets.QVBoxLayout()
            self._small_col.setContentsMargins(2, 0, 2, 0)
            self._small_col.setSpacing(0)
            holder = QtWidgets.QVBoxLayout()
            holder.setContentsMargins(0, 0, 0, 0)
            holder.addLayout(self._small_col)
            holder.addStretch(1)
            self._row.addLayout(holder)
        btn = QtWidgets.QToolButton()
        btn.setObjectName("stripSmall")
        btn.setText(text)
        btn.setIcon(icon(icon_name))
        btn.setIconSize(QtCore.QSize(16, 16))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        btn.setCheckable(checkable)
        btn.setToolTip(tooltip or text)
        if callback is not None:
            if checkable:
                btn.toggled.connect(lambda state: callback(state))
            else:
                btn.clicked.connect(lambda _checked=False: callback())
        self._small_col.addWidget(btn)
        return btn


class Toolstrip(QtWidgets.QWidget):
    HEIGHT_BODY = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tab_bar = QtWidgets.QWidget()
        self._tab_bar.setObjectName("stripTabBar")
        self._tab_bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._tab_row = QtWidgets.QHBoxLayout(self._tab_bar)
        self._tab_row.setContentsMargins(6, 3, 10, 0)
        self._tab_row.setSpacing(1)
        self._tab_row.addStretch(1)
        self._caption = QtWidgets.QLabel("")
        self._tab_row.addWidget(self._caption)
        root.addWidget(self._tab_bar)

        self._body = QtWidgets.QStackedWidget()
        self._body.setObjectName("stripBody")
        self._body.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._body.setFixedHeight(self.HEIGHT_BODY)
        root.addWidget(self._body)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._tabs: List[QtWidgets.QToolButton] = []

    def set_caption(self, text: str) -> None:
        """Right-aligned text in the tab bar (the open project's name)."""
        self._caption.setText(text)

    def add_tab(self, title: str) -> "TabPage":
        btn = QtWidgets.QToolButton()
        btn.setObjectName("stripTab")
        btn.setText(title.upper())
        btn.setCheckable(True)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        index = self._body.count()
        btn.clicked.connect(lambda _checked=False, i=index: self._body.setCurrentIndex(i))
        self._group.addButton(btn)
        self._tab_row.insertWidget(len(self._tabs), btn)
        self._tabs.append(btn)
        page = TabPage()
        self._body.addWidget(page)
        if index == 0:
            btn.setChecked(True)
        return page


class TabPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._row = QtWidgets.QHBoxLayout(self)
        self._row.setContentsMargins(4, 0, 4, 0)
        self._row.setSpacing(0)
        self._row.addStretch(1)
        self._count = 0

    def add_section(self, title: str) -> Section:
        if self._count:
            sep = QtWidgets.QFrame()
            sep.setObjectName("stripSeparator")
            sep.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            sep.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
            holder = QtWidgets.QVBoxLayout()
            holder.setContentsMargins(0, 6, 0, 6)
            holder.addWidget(sep)
            self._row.insertLayout(self._row.count() - 1, holder)
        section = Section(title)
        self._row.insertWidget(self._row.count() - 1, section)
        self._count += 1
        return section

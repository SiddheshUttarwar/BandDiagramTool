"""
Line icons, drawn with QPainter on a 20-unit grid at a single stroke weight
so the whole set reads as one family. Nothing is shipped as an image file:
icons stay sharp at any display scaling and take any colour from the theme.
"""

from __future__ import annotations

from typing import Callable, Dict

from PyQt6 import QtCore, QtGui

from gui import theme

_STROKE = 1.5


def _pts(points) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])


def _line(p, x0, y0, x1, y1):
    p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))


def _new(p):
    p.drawPolygon(_pts([(5, 2.5), (11.5, 2.5), (15.5, 6.5), (15.5, 17.5), (5, 17.5)]))
    p.drawPolyline(_pts([(11.5, 2.5), (11.5, 6.5), (15.5, 6.5)]))
    _line(p, 10.25, 9.6, 10.25, 14.4)
    _line(p, 7.85, 12, 12.65, 12)


def _open(p):
    p.drawPolygon(_pts([(2.5, 4.5), (7.8, 4.5), (9.6, 6.8), (17.5, 6.8), (17.5, 15.5), (2.5, 15.5)]))


def _save(p):
    p.drawPolygon(_pts([(3.5, 3.5), (13.5, 3.5), (16.5, 6.5), (16.5, 16.5), (3.5, 16.5)]))
    p.drawPolyline(_pts([(6.5, 3.5), (6.5, 7.2), (12.2, 7.2), (12.2, 3.5)]))
    p.drawPolyline(_pts([(6.2, 16.5), (6.2, 11.2), (13.8, 11.2), (13.8, 16.5)]))


def _plus(p):
    _line(p, 10, 4.5, 10, 15.5)
    _line(p, 4.5, 10, 15.5, 10)


def _layers(p):
    p.drawPolygon(_pts([(10, 3), (17, 7), (10, 11), (3, 7)]))
    p.drawPolyline(_pts([(3, 10.5), (10, 14.5), (17, 10.5)]))
    p.drawPolyline(_pts([(3, 14), (10, 18), (17, 14)]))


def _play(p):
    p.setBrush(p.pen().color())
    p.drawPolygon(_pts([(6.8, 4.6), (15.6, 10), (6.8, 15.4)]))


def _stop(p):
    p.setBrush(p.pen().color())
    p.drawRoundedRect(QtCore.QRectF(5.8, 5.8, 8.4, 8.4), 1.4, 1.4)


def _export(p):
    _line(p, 10, 3.2, 10, 12.4)
    p.drawPolyline(_pts([(6.4, 6.8), (10, 3.2), (13.6, 6.8)]))
    p.drawPolyline(_pts([(3.5, 11.8), (3.5, 16.5), (16.5, 16.5), (16.5, 11.8)]))


def _book(p):
    p.drawRoundedRect(QtCore.QRectF(4.2, 3, 11.6, 14), 1.6, 1.6)
    _line(p, 7.6, 3, 7.6, 17)
    _line(p, 10.2, 7, 13.2, 7)
    _line(p, 10.2, 10, 13.2, 10)


def _frame(p):
    p.drawRoundedRect(QtCore.QRectF(2.5, 4, 15, 12), 2, 2)


def _sidebar_left(p):
    _frame(p)
    _line(p, 7.6, 4, 7.6, 16)


def _sidebar_right(p):
    _frame(p)
    _line(p, 12.4, 4, 12.4, 16)


def _panel_bottom(p):
    _frame(p)
    _line(p, 2.5, 11.6, 17.5, 11.6)


def _home(p):
    p.drawPolyline(_pts([(3, 9.6), (10, 3.6), (17, 9.6)]))
    p.drawPolyline(_pts([(5.2, 8.4), (5.2, 16.4), (14.8, 16.4), (14.8, 8.4)]))


def _pan(p):
    _line(p, 10, 3, 10, 17)
    _line(p, 3, 10, 17, 10)
    for head in ([(8, 5), (10, 3), (12, 5)], [(8, 15), (10, 17), (12, 15)],
                 [(5, 8), (3, 10), (5, 12)], [(15, 8), (17, 10), (15, 12)]):
        p.drawPolyline(_pts(head))


def _zoom(p):
    p.drawEllipse(QtCore.QPointF(8.8, 8.8), 5.3, 5.3)
    _line(p, 12.8, 12.8, 17, 17)


def _trash(p):
    _line(p, 3.5, 5.6, 16.5, 5.6)
    p.drawPolyline(_pts([(7.6, 5.6), (7.6, 3.4), (12.4, 3.4), (12.4, 5.6)]))
    p.drawPolyline(_pts([(5, 5.6), (5.8, 17), (14.2, 17), (15, 5.6)]))


def _close(p):
    _line(p, 5.5, 5.5, 14.5, 14.5)
    _line(p, 14.5, 5.5, 5.5, 14.5)


def _info(p):
    p.drawEllipse(QtCore.QPointF(10, 10), 7.2, 7.2)
    _line(p, 10, 9.4, 10, 13.8)
    _line(p, 10, 6.4, 10, 6.5)


def _chevron_down(p):
    p.drawPolyline(_pts([(5.5, 8), (10, 12.5), (14.5, 8)]))


def _chevron_right(p):
    p.drawPolyline(_pts([(8, 5.5), (12.5, 10), (8, 14.5)]))


def _duplicate(p):
    p.drawRoundedRect(QtCore.QRectF(6.6, 6.6, 10, 10), 1.6, 1.6)
    p.drawPolyline(_pts([(13.4, 4.4), (13.4, 3.4), (3.4, 3.4), (3.4, 13.4), (4.4, 13.4)]))


def _terminal(p):
    _frame(p)
    p.drawPolyline(_pts([(5.6, 7.6), (8.2, 10), (5.6, 12.4)]))
    _line(p, 10.2, 12.6, 14, 12.6)


def _grid(p):
    for x, y in ((3.2, 3.2), (11.2, 3.2), (3.2, 11.2), (11.2, 11.2)):
        p.drawRoundedRect(QtCore.QRectF(x, y, 5.6, 5.6), 1.2, 1.2)


def _folder_chart(p):
    _open(p)
    p.drawPolyline(_pts([(6, 13), (8.6, 10.4), (10.8, 12.2), (14, 9.2)]))


_PAINTERS: Dict[str, Callable] = {
    "new": _new, "open": _open, "save": _save, "plus": _plus, "layers": _layers,
    "play": _play, "stop": _stop, "export": _export, "book": _book,
    "sidebar_left": _sidebar_left, "sidebar_right": _sidebar_right, "panel_bottom": _panel_bottom,
    "home": _home, "pan": _pan, "zoom": _zoom, "trash": _trash, "close": _close, "info": _info,
    "chevron_down": _chevron_down, "chevron_right": _chevron_right, "duplicate": _duplicate,
    "terminal": _terminal, "grid": _grid, "results": _folder_chart,
}

_CACHE: Dict[tuple, QtGui.QIcon] = {}


def pixmap(name: str, color: str = theme.TEXT_MUTED, size: int = 20, scale: int = 1) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(size * scale, size * scale)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.scale(size * scale / 20.0, size * scale / 20.0)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidthF(_STROKE)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    _PAINTERS[name](p)
    p.end()
    return pm


def icon(name: str, color: str = theme.TEXT_MUTED, disabled: str = theme.BORDER_HOVER,
         size: int = 20) -> QtGui.QIcon:
    """Icon `name` in `color`, with a paler disabled state, at 1x-3x."""
    key = (name, color, disabled, size)
    if key not in _CACHE:
        ico = QtGui.QIcon()
        for scale in (1, 2, 3):
            ico.addPixmap(pixmap(name, color, size, scale), QtGui.QIcon.Mode.Normal)
            ico.addPixmap(pixmap(name, disabled, size, scale), QtGui.QIcon.Mode.Disabled)
        _CACHE[key] = ico
    return _CACHE[key]

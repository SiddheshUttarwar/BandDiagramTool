"""
Shared visual constants for the EpiBand GUI.

The interface itself uses the platform's native widgets (see apply()), in
the manner of established desktop scientific software: menu bar, toolbar,
tabbed side panel, one large graphics area and a text output pane. This
module holds what the native style does not: the material colour scale used
by the layer table and the figures, the figure palette, number formatting,
and the application mark.
"""

from __future__ import annotations

import os
import re

# ---- Colour tokens --------------------------------------------------------
BG = "#F3F4F7"              # application canvas
SURFACE = "#FFFFFF"         # panels, cards, inputs
SURFACE_ALT = "#F8F9FB"     # quiet fill: hover rows, read-only areas
SURFACE_SUNKEN = "#EEF0F4"  # segmented-control track, pressed

BORDER = "#E4E7EC"          # hairlines
BORDER_STRONG = "#CFD4DC"   # control outlines
BORDER_HOVER = "#AEB6C2"

TEXT = "#17202C"
TEXT_MUTED = "#566273"
TEXT_FAINT = "#8892A0"
TEXT_ON_ACCENT = "#FFFFFF"

ACCENT = "#2B5FDB"
ACCENT_HOVER = "#234FBE"
ACCENT_PRESSED = "#1C419E"
ACCENT_SOFT = "#EAF0FE"     # selection / hover wash
ACCENT_SOFT_BORDER = "#B8CBF8"

SUCCESS = "#16824B"
SUCCESS_SOFT = "#E5F4EC"
WARNING = "#A8690A"
WARNING_SOFT = "#FBF1DD"
DANGER = "#C4372E"
DANGER_SOFT = "#FCECEA"

METAL = "#2A3340"           # contact bars in the cross-section
METAL_TEXT = "#E7EBF0"

FONT_FAMILY = "Segoe UI"
FONT_SIZE_PT = 9
MONO_FAMILY = "Cascadia Mono"

RADIUS = 6
RADIUS_CARD = 10

# ---- Figure palette -------------------------------------------------------
# Categorical hues checked with the palette validator for colour-vision
# separation on a white surface (all pairs, since band-diagram lines cross):
# blue / red / aqua / violet / yellow pass together (the valence sub-bands
# also differ by dash pattern), as do blue / orange / aqua.
FIG_BLUE = "#2A78D6"
FIG_RED = "#E34948"
FIG_YELLOW = "#EDA100"
FIG_VIOLET = "#4A3AA7"
FIG_ORANGE = "#EB6834"
FIG_AQUA = "#1BAF7A"
FIG_NEUTRAL = "#8892A0"
FIG_GRID = "#ECEEF2"
FIG_AXIS = "#B7BEC9"

FIGURE_COLORS = {
    "Ec": FIG_BLUE, "Ev": FIG_RED,
    "Ev_hh": FIG_AQUA, "Ev_lh": FIG_VIOLET, "Ev_so": FIG_YELLOW,   # each its own hue and dash
    "Efn": "#17202C", "Efp": "#6B7483",                            # Fermi levels in ink, not a series hue
    "Ei": FIG_NEUTRAL, "vac": "#B7BEC9",
    "n": FIG_BLUE, "p": FIG_RED,
    "Psp": FIG_BLUE, "Ppz": FIG_ORANGE, "Ptot": FIG_AQUA,
    "Felec": FIG_BLUE, "Fquasi": FIG_ORANGE,
    "exx": FIG_BLUE, "ezz": FIG_ORANGE,
}


# ---- Materials ------------------------------------------------------------
_GAN = (0x2F, 0x6F, 0xDB)   # blue
_ALN = (0x7B, 0x45, 0xD6)   # violet  (wider gap -> shorter wavelength)
_INN = (0x14, 0xA0, 0x7C)   # green   (narrower gap)


def _mix(a, b, t: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def tint(color: str, amount: float) -> str:
    """`color` mixed toward white; amount 0 = unchanged, 1 = white."""
    return _hex(_mix(_rgb(color), (255, 255, 255), amount))


def shade(color: str, amount: float) -> str:
    """`color` mixed toward the text colour; amount 0 = unchanged."""
    return _hex(_mix(_rgb(color), _rgb(TEXT), amount))


# Arsenides and phosphides: warm hues, apart from the nitrides' cool ones.
_ZB = {
    ("Ga", "As"): (0xE0, 0x7A, 0x2C),   # orange
    ("Al", "As"): (0xC8, 0x3F, 0x5E),   # crimson
    ("In", "As"): (0x8C, 0x5E, 0x34),   # brown
    ("Ga", "P"): (0xE2, 0xB0, 0x24),    # amber
    ("Al", "P"): (0xA9, 0xB8, 0x2E),    # olive
    ("In", "P"): (0x22, 0xA3, 0xA6),    # teal
}


def material_color(x_al: float, x_in: float = 0.0, x_p: float = 0.0, crystal: str = "wurtzite") -> str:
    """Colour of a composition. Nitrides, Al(x)In(y)Ga(1-x-y)N: GaN blue,
    toward violet with Al and toward green with In. Zincblende,
    Al(x)In(y)Ga(1-x-y)As(1-v)P(v): a blend of the six binaries' colours.
    The same scale colours the device cross-section and the layer bands
    behind every figure."""
    x_al = max(0.0, min(1.0, float(x_al)))
    x_in = max(0.0, min(1.0, float(x_in)))
    z = max(0.0, 1.0 - x_al - x_in)
    if crystal == "zincblende":
        x_p = max(0.0, min(1.0, float(x_p)))
        w = {("Ga", "As"): z * (1 - x_p), ("Al", "As"): x_al * (1 - x_p), ("In", "As"): x_in * (1 - x_p),
             ("Ga", "P"): z * x_p, ("Al", "P"): x_al * x_p, ("In", "P"): x_in * x_p}
        total = sum(w.values()) or 1.0
        return _hex(tuple(int(round(sum(w[k] * _ZB[k][c] for k in w) / total)) for c in range(3)))
    return _hex(tuple(int(round(z * g + x_al * a + x_in * i)) for g, a, i in zip(_GAN, _ALN, _INN)))


# ---- Text helpers ---------------------------------------------------------
_SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def formula_html(name: str) -> str:
    """'Al0.30Ga0.70N' -> 'Al<sub>0.30</sub>Ga<sub>0.70</sub>N' (Qt rich text)."""
    return re.sub(r"(?<=[A-Za-z])(\d+(?:\.\d+)?)", r"<sub>\1</sub>", name)


def sci(value: float, digits: int = 2) -> str:
    """1.08e13 -> '1.08×10¹³'; values between 0.01 and 10^4 stay plain."""
    if value == 0 or value != value:
        return "0"
    a = abs(value)
    if 1e-2 <= a < 1e4:
        return f"{value:.{digits + 1}g}"
    exp = int(f"{a:e}".split("e")[1])
    mant = value / 10 ** exp
    mant_s = f"{mant:.{digits}f}".rstrip("0").rstrip(".")
    if mant_s in ("10", "-10"):
        mant_s, exp = mant_s[:-1], exp + 1
    power = "10" + str(exp).translate(_SUP)
    if mant_s == "1":
        return power
    return f"{mant_s}×{power}"


# ---- Drawn assets ---------------------------------------------------------
def _asset_dir() -> str:
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "epiband_ui_v3")
    os.makedirs(d, exist_ok=True)
    return d


def _glyph(name: str) -> str:
    """Path (forward slashes, for QSS url()) of a small glyph image drawn on
    first use: Qt style sheets can only take indicator and arrow artwork
    from image files. Requires a running QApplication."""
    from PyQt6 import QtCore, QtGui
    path = os.path.join(_asset_dir(), f"{name}.png")
    if not os.path.exists(path):
        scale = 4
        pm = QtGui.QPixmap(12 * scale, 12 * scale)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.scale(scale, scale)
        color = {"check": "#FFFFFF", "chev_down_disabled": BORDER_HOVER}.get(name, TEXT_MUTED)
        pen = QtGui.QPen(QtGui.QColor(color))
        pen.setWidthF(1.8 if name == "check" else 1.4)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        if name == "check":
            pts = [(2.6, 6.3), (5.0, 8.6), (9.5, 3.6)]
        elif name == "chev_up":
            pts = [(3.0, 7.6), (6.0, 4.6), (9.0, 7.6)]
        else:
            pts = [(3.0, 4.6), (6.0, 7.6), (9.0, 4.6)]
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
        p.end()
        pm.save(path)
    return path.replace(os.sep, "/")


def logo_pixmap(size: int = 64):
    """The application mark: a polarization-tilted quantum well (conduction
    and valence band edges) on a GaN-blue to AlN-violet field."""
    from PyQt6 import QtCore, QtGui
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.scale(size / 32.0, size / 32.0)
    grad = QtGui.QLinearGradient(0, 0, 32, 32)
    grad.setColorAt(0.0, QtGui.QColor("#2F7BEA"))
    grad.setColorAt(1.0, QtGui.QColor("#6B3FD9"))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(QtCore.QRectF(1, 1, 30, 30), 7.5, 7.5)
    pen = QtGui.QPen(QtGui.QColor("#FFFFFF"))
    pen.setWidthF(2.1)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    # a quantum well: band edge, bound level, and its probability density
    well = [(5.5, 11.0), (11.0, 11.0), (11.0, 23.0), (21.0, 23.0), (21.0, 11.0), (26.5, 11.0)]
    p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in well]))
    psi = QtGui.QPainterPath(QtCore.QPointF(11.0, 19.0))
    psi.cubicTo(QtCore.QPointF(13.6, 19.0), QtCore.QPointF(14.2, 13.2), QtCore.QPointF(16.0, 13.2))
    psi.cubicTo(QtCore.QPointF(17.8, 13.2), QtCore.QPointF(18.4, 19.0), QtCore.QPointF(21.0, 19.0))
    fill = QtGui.QPainterPath(psi)
    fill.closeSubpath()
    p.fillPath(fill, QtGui.QColor(255, 255, 255, 90))
    pen.setWidthF(1.6)
    p.setPen(pen)
    p.drawPath(psi)
    p.end()
    return pm


def app_icon():
    from PyQt6 import QtGui
    ico = QtGui.QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ico.addPixmap(logo_pixmap(s))
    return ico


def polish_menu(menu):
    """Menus use the platform's own frame; kept as a hook for callers."""
    return menu


def apply(app) -> None:
    """Native platform widgets (the look of established desktop scientific
    software), the system UI font, and the window icon. Only a handful of
    labels are styled; everything else is drawn by the operating system."""
    from PyQt6 import QtGui, QtWidgets
    for name in ("windowsvista", "Fusion"):
        if name in QtWidgets.QStyleFactory.keys():
            app.setStyle(name)
            break
    app.setFont(QtGui.QFont(FONT_FAMILY, FONT_SIZE_PT))
    app.setStyleSheet(stylesheet())
    app.setWindowIcon(app_icon())


def stylesheet() -> str:
    return f"""
    QLabel#unit, QLabel#sectionMeta {{ color: {TEXT_FAINT}; }}
    QLabel#note {{ color: {TEXT_MUTED}; }}
    QLabel#empty {{ color: {TEXT_FAINT}; }}
    QLabel#subhead {{ color: {TEXT_MUTED}; font-weight: 600; padding-top: 4px; }}
    QPlainTextEdit#log {{
        font-family: "Consolas", "Courier New", monospace;
        font-size: 9pt;
    }}
    QToolBar {{ spacing: 1px; }}
    """

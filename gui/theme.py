"""
Visual theme for the BandDiagramTool GUI: a neutral engineering-desktop look
in the manner of MATLAB -- a dark-blue toolstrip tab bar over a light grey
ribbon, flat grey panel headers, white work surfaces, compact square
controls and a single blue accent. Applied globally as one Qt stylesheet
from run_gui.py.

Color tokens are also imported directly by layer_stack.py and toolstrip.py,
which paint parts of their widgets outside the stylesheet.
"""

from __future__ import annotations

import os

# ---- Color tokens ---------------------------------------------------------
BG = "#F0F0F0"              # window / dock background
SURFACE = "#FFFFFF"         # work surfaces (inputs, lists, figure area)
SURFACE_MUTED = "#F7F7F7"   # secondary fill (log, alternate rows)
BORDER = "#D4D4D4"
BORDER_STRONG = "#ABABAB"   # control outlines

TEXT = "#1F1F1F"
TEXT_MUTED = "#555555"
TEXT_FAINT = "#8A8A8A"

ACCENT = "#0072BD"          # MATLAB blue
ACCENT_HOVER = "#005F9E"
ACCENT_PRESSED = "#004C7F"
ACCENT_SOFT = "#CCE4F7"     # selection / hover fill
ACCENT_SOFT_BORDER = "#7FB9E6"

DANGER = "#C0392B"
DANGER_SOFT = "#FBEDEB"
SUCCESS = "#2E8B2E"

# Toolstrip
STRIP_TAB_BG = "#0F4C81"        # tab bar
STRIP_TAB_BG_HOVER = "#1A5E9A"
STRIP_TAB_TEXT = "#FFFFFF"
STRIP_BG = "#F5F5F5"            # ribbon body
STRIP_SECTION_TEXT = "#6B6B6B"
STRIP_BTN_HOVER = "#DCEBF7"
STRIP_BTN_PRESSED = "#C2DCF2"
STRIP_BTN_HOVER_BORDER = "#9CC7EA"

# Dock / panel headers
HEADER_BG = "#E1E1E1"
HEADER_TEXT = "#1F1F1F"

BUTTON_BG = "#E9E9E9"
BUTTON_BG_HOVER = "#DCEBF7"
BUTTON_BG_PRESSED = "#C2DCF2"

# --- layer_stack.py row tokens ---
CARD_BG = SURFACE
CARD_BG_SELECTED = ACCENT_SOFT
CARD_BG_INTERFACE = "#FBF7E8"           # zero-thickness marker rows
CARD_BG_INTERFACE_SELECTED = ACCENT_SOFT
CARD_BORDER = BORDER
CARD_BORDER_SELECTED = ACCENT

FONT_FAMILY = "Segoe UI"
FONT_SIZE_PT = 9
MONO_FAMILY = "Consolas"


def material_color(x_al: float, x_in: float = 0.0) -> str:
    """Swatch colour for an Al(x)In(y)Ga(1-x-y)N composition: GaN blue,
    shifting to violet with Al (wider gap) and to green/amber with In
    (narrower gap). Used for the colour bar on layer rows."""
    x_al = max(0.0, min(1.0, float(x_al)))
    x_in = max(0.0, min(1.0, float(x_in)))
    gan, aln, inn = (0x00, 0x72, 0xBD), (0x7E, 0x2F, 0x8E), (0xD9, 0x53, 0x19)
    z = max(0.0, 1.0 - x_al - x_in)
    rgb = [int(round(z * g + x_al * a + x_in * i)) for g, a, i in zip(gan, aln, inn)]
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _asset_dir() -> str:
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "banddiagramtool_ui")
    os.makedirs(d, exist_ok=True)
    return d


def _glyph(name: str) -> str:
    """Path (forward slashes, for QSS url()) of a small glyph image drawn on
    first use: Qt style sheets can only take indicator and arrow artwork from
    image files. Requires a running QApplication."""
    from PyQt6 import QtCore, QtGui
    path = os.path.join(_asset_dir(), f"{name}.png")
    if not os.path.exists(path):
        scale = 3
        pm = QtGui.QPixmap(12 * scale, 12 * scale)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.scale(scale, scale)
        if name == "check":
            pen = QtGui.QPen(QtGui.QColor("#FFFFFF"))
            pen.setWidthF(1.9)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(2.6, 6.3), QtCore.QPointF(5.0, 8.7),
                                            QtCore.QPointF(9.6, 3.4)]))
        else:   # "arrow_down" / "arrow_down_disabled" / "arrow_up"
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(TEXT_FAINT if name.endswith("disabled") else TEXT_MUTED))
            if name == "arrow_up":
                pts = [(2.5, 8.0), (9.5, 8.0), (6.0, 4.0)]
            else:
                pts = [(2.5, 4.5), (9.5, 4.5), (6.0, 8.5)]
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
        p.end()
        pm.save(path)
    return path.replace(os.sep, "/")


def stylesheet() -> str:
    check, arrow, arrow_off = _glyph("check"), _glyph("arrow_down"), _glyph("arrow_down_disabled")
    arrow_up = _glyph("arrow_up")
    return f"""
    * {{
        font-family: "{FONT_FAMILY}";
        color: {TEXT};
    }}
    QMainWindow, QWidget {{
        background: {BG};
    }}
    QMainWindow::separator {{
        background: {BORDER};
        width: 4px;
        height: 4px;
    }}
    QMainWindow::separator:hover {{
        background: {ACCENT_SOFT_BORDER};
    }}
    QSplitter::handle {{
        background: {BORDER};
    }}

    /* ---- docks ---- */
    QDockWidget {{
        font-weight: 600;
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background: {HEADER_BG};
        border: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER_STRONG};
        padding: 5px 8px;
        text-align: left;
    }}
    QDockWidget > QWidget {{
        border: 1px solid {BORDER};
        border-top: none;
    }}

    /* ---- sections inside the property inspector ---- */
    QGroupBox {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 0px;
        margin-top: 22px;
        padding: 8px 8px 8px 8px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 0px;
        top: 0px;
        padding: 3px 8px;
        background: {HEADER_BG};
        border: 1px solid {BORDER};
        color: {HEADER_TEXT};
    }}
    QLabel {{
        background: transparent;
    }}

    /* ---- inputs ---- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        border-radius: 2px;
        padding: 2px 6px;
        min-height: 18px;
        selection-background-color: {ACCENT};
        selection-color: white;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {ACCENT};
    }}
    QLineEdit:disabled, QComboBox:disabled {{
        background: {BG};
        color: {TEXT_FAINT};
    }}
    QComboBox {{
        padding-right: 18px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: center right;
        border: none;
        border-left: 1px solid {BORDER};
        width: 16px;
    }}
    QComboBox::down-arrow {{
        image: url({arrow});
        width: 10px;
        height: 10px;
    }}
    QComboBox::down-arrow:disabled {{
        image: url({arrow_off});
    }}
    QAbstractSpinBox {{
        padding-right: 16px;
    }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        subcontrol-origin: border;
        width: 15px;
        border: none;
        border-left: 1px solid {BORDER};
        background: {BG};
    }}
    QAbstractSpinBox::up-button {{
        subcontrol-position: top right;
        border-bottom: 1px solid {BORDER};
    }}
    QAbstractSpinBox::down-button {{
        subcontrol-position: bottom right;
    }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {STRIP_BTN_HOVER};
    }}
    QAbstractSpinBox::up-arrow {{
        image: url({arrow_up});
        width: 8px;
        height: 8px;
    }}
    QAbstractSpinBox::down-arrow {{
        image: url({arrow});
        width: 8px;
        height: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
        outline: 0;
    }}

    /* ---- buttons ---- */
    QPushButton {{
        background: {BUTTON_BG};
        border: 1px solid {BORDER_STRONG};
        border-radius: 2px;
        padding: 4px 14px;
        min-height: 16px;
    }}
    QPushButton:hover {{
        background: {BUTTON_BG_HOVER};
        border-color: {ACCENT_SOFT_BORDER};
    }}
    QPushButton:pressed {{
        background: {BUTTON_BG_PRESSED};
    }}
    QPushButton:disabled {{
        color: {TEXT_FAINT};
        background: {BG};
        border-color: {BORDER};
    }}
    QPushButton#primary {{
        background: {ACCENT};
        border: 1px solid {ACCENT_HOVER};
        color: white;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background: {ACCENT_HOVER};
    }}
    QPushButton#primary:pressed {{
        background: {ACCENT_PRESSED};
    }}
    QPushButton#danger {{
        background: {SURFACE};
        color: {DANGER};
        border: 1px solid {DANGER};
    }}
    QPushButton#danger:hover {{
        background: {DANGER_SOFT};
    }}
    QCheckBox, QRadioButton {{
        spacing: 6px;
        background: transparent;
    }}
    QCheckBox::indicator {{
        width: 13px;
        height: 13px;
        border: 1px solid {BORDER_STRONG};
        border-radius: 2px;
        background: {SURFACE};
    }}
    QCheckBox::indicator:hover {{
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border-color: {ACCENT};
        image: url({check});
    }}
    QCheckBox::indicator:disabled {{
        background: {BG};
        border-color: {BORDER};
    }}
    QToolButton {{
        border: 1px solid transparent;
        background: transparent;
        padding: 3px 6px;
        border-radius: 2px;
    }}
    QToolButton:hover {{
        background: {STRIP_BTN_HOVER};
        border: 1px solid {STRIP_BTN_HOVER_BORDER};
    }}
    QToolButton:pressed, QToolButton:checked {{
        background: {STRIP_BTN_PRESSED};
        border: 1px solid {ACCENT_SOFT_BORDER};
    }}
    QToolButton:disabled {{
        color: {TEXT_FAINT};
    }}
    QToolButton::menu-indicator {{
        image: url({arrow});
        width: 9px;
        height: 9px;
        subcontrol-position: right center;
        right: 3px;
    }}
    QToolButton#stripLarge::menu-indicator {{
        subcontrol-position: bottom center;
        bottom: 1px;
        right: 0px;
    }}

    /* ---- toolstrip ---- */
    QWidget#stripTabBar {{
        background: {STRIP_TAB_BG};
    }}
    QWidget#stripTabBar QLabel {{
        color: #D6E4F2;
    }}
    QToolButton#stripTab {{
        color: {STRIP_TAB_TEXT};
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 6px 16px 5px 16px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    QToolButton#stripTab:hover {{
        background: {STRIP_TAB_BG_HOVER};
        border: none;
    }}
    QToolButton#stripTab:checked {{
        background: {STRIP_BG};
        color: {TEXT};
        border: none;
    }}
    QWidget#stripBody {{
        background: {STRIP_BG};
        border-bottom: 1px solid {BORDER_STRONG};
    }}
    QWidget#stripBody QWidget {{
        background: {STRIP_BG};
    }}
    QLabel#stripSectionLabel {{
        color: {STRIP_SECTION_TEXT};
        font-size: 7.5pt;
        letter-spacing: 0.6px;
        padding: 0px 0px 2px 0px;
    }}
    QFrame#stripSeparator {{
        background: {BORDER};
        max-width: 1px;
        min-width: 1px;
    }}
    QToolButton#stripLarge {{
        padding: 3px 6px 2px 6px;
        min-width: 54px;
    }}
    QToolButton#stripSmall {{
        padding: 1px 6px;
        text-align: left;
    }}

    /* ---- matplotlib navigation toolbar ---- */
    QWidget#mplToolbar {{
        background: {STRIP_BG};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 1px;
    }}
    QWidget#mplToolbar QLabel {{
        color: {TEXT_MUTED};
    }}

    QMenuBar {{
        background: {BG};
        border-bottom: 1px solid {BORDER};
    }}
    QMenuBar::item {{
        padding: 3px 9px;
    }}
    QMenuBar::item:selected {{
        background: {ACCENT_SOFT};
    }}
    QMenu {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        padding: 2px;
    }}
    QMenu::item {{
        padding: 4px 24px 4px 20px;
    }}
    QMenu::item:selected {{
        background: {ACCENT_SOFT};
    }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER};
        margin: 3px 6px;
    }}

    /* ---- document tabs (figures) ---- */
    QTabWidget::pane {{
        border: 1px solid {BORDER_STRONG};
        background: {SURFACE};
        top: -1px;
    }}
    QTabBar::tab {{
        background: {HEADER_BG};
        border: 1px solid {BORDER_STRONG};
        border-bottom: none;
        padding: 4px 14px;
        margin-right: -1px;
        color: {TEXT_MUTED};
    }}
    QTabBar::tab:selected {{
        background: {SURFACE};
        color: {TEXT};
        font-weight: 600;
        border-top: 2px solid {ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        background: {STRIP_BTN_HOVER};
        color: {TEXT};
    }}

    QListWidget {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        outline: 0;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: {BG};
        width: 12px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: #C2C2C2;
        min-height: 24px;
        margin: 2px;
        border-radius: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #A0A0A0;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {BG};
        height: 12px;
        margin: 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: #C2C2C2;
        min-width: 24px;
        margin: 2px;
        border-radius: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: #A0A0A0;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    QSlider::groove:horizontal {{
        background: {BORDER_STRONG};
        height: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 10px;
        height: 16px;
        margin: -7px 0;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
    }}
    QPlainTextEdit {{
        font-family: "{MONO_FAMILY}", "Courier New", monospace;
        background: {SURFACE};
        border: none;
        selection-background-color: {ACCENT};
        selection-color: white;
    }}
    QStatusBar {{
        background: {BG};
        border-top: 1px solid {BORDER_STRONG};
    }}
    QStatusBar::item {{
        border: none;
    }}
    QStatusBar QLabel {{
        padding: 0px 6px;
        color: {TEXT_MUTED};
    }}
    QProgressBar {{
        border: 1px solid {BORDER_STRONG};
        background: {SURFACE};
        max-height: 10px;
        min-width: 110px;
        max-width: 110px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {ACCENT};
    }}
    QMessageBox, QDialog {{
        background: {BG};
    }}
    QToolTip {{
        background: #FFFFE1;
        color: {TEXT};
        border: 1px solid {BORDER_STRONG};
        padding: 3px 6px;
    }}
    """

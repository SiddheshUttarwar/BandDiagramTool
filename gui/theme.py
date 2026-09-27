"""
Visual theme for the BandDiagramTool GUI: a warm, minimal palette in the
same spirit as Claude.ai's interface (cream backgrounds, terracotta accent,
soft rounded panels), applied globally as one Qt stylesheet from run_gui.py.
Color tokens are also imported directly by layer_stack.py, which paints its
own card widgets outside the stylesheet (per-instance selected/interface
state can't be expressed as static QSS).
"""

from __future__ import annotations

# ---- Color tokens ---------------------------------------------------------
BG = "#F7F6F1"              # app background
SURFACE = "#FFFFFF"         # panels / cards
SURFACE_MUTED = "#F0EEE6"   # secondary panel fill (scroll areas, list bg, log)
BORDER = "#E4E1D7"
BORDER_STRONG = "#C9C4B2"   # darkened from an earlier pass -- buttons/inputs
                             # need a border that reads against a white card,
                             # not just against the cream page background.

TEXT = "#2B2A27"
TEXT_MUTED = "#5F5D54"      # darkened for readability against white/cream
TEXT_FAINT = "#8B8879"

ACCENT = "#D97757"          # Claude terracotta
ACCENT_HOVER = "#C35F3E"
ACCENT_PRESSED = "#AE5236"
ACCENT_SOFT = "#F3E4DB"     # tinted background for selection/hover
ACCENT_SOFT_BORDER = "#E8B49B"

DANGER = "#B3261E"
DANGER_SOFT = "#FBEDEC"

# A plain QPushButton previously used the same white as its parent QGroupBox
# card, so its outline was the only thing separating it from the card body --
# easy to miss. BUTTON_BG gives it a body of its own.
BUTTON_BG = "#EDEAE0"
BUTTON_BG_HOVER = "#E4E0D2"
BUTTON_BG_PRESSED = ACCENT_SOFT

# --- layer_stack.py card tokens (kept as plain names for that module) ---
CARD_BG = SURFACE
CARD_BG_SELECTED = ACCENT_SOFT
CARD_BG_INTERFACE = "#FBF3E3"           # warm tint for zero-thickness marker layers
CARD_BG_INTERFACE_SELECTED = ACCENT_SOFT
CARD_BORDER = BORDER_STRONG
CARD_BORDER_SELECTED = ACCENT

FONT_FAMILY = "Segoe UI"
FONT_SIZE_PT = 9


def stylesheet() -> str:
    return f"""
    * {{
        font-family: "{FONT_FAMILY}";
        color: {TEXT};
    }}
    QMainWindow, QWidget {{
        background: {BG};
    }}
    QSplitter::handle {{
        background: {BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QGroupBox {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-top: 14px;
        padding: 12px 10px 10px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {TEXT};
    }}
    QLabel {{
        background: transparent;
    }}
    QLineEdit, QComboBox {{
        background: {SURFACE};
        border: 1px solid {BORDER_STRONG};
        border-radius: 6px;
        padding: 4px 8px;
        selection-background-color: {ACCENT_SOFT};
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QPushButton {{
        background: {BUTTON_BG};
        border: 1px solid {BORDER_STRONG};
        border-radius: 6px;
        padding: 7px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: {BUTTON_BG_HOVER};
        border-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background: {BUTTON_BG_PRESSED};
    }}
    QPushButton:disabled {{
        color: {TEXT_FAINT};
        background: {SURFACE_MUTED};
        border-color: {BORDER};
    }}
    QPushButton#primary {{
        background: {ACCENT};
        border: 1px solid {ACCENT};
        color: white;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background: {ACCENT_HOVER};
        border-color: {ACCENT_HOVER};
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
    QCheckBox {{
        spacing: 8px;
        background: transparent;
    }}
    QToolButton {{
        border: 1px solid transparent;
        background: transparent;
        padding: 4px 8px;
        border-radius: 6px;
    }}
    QToolButton:hover {{
        background: {ACCENT_SOFT};
        border: 1px solid {ACCENT_SOFT_BORDER};
    }}
    QToolButton:checked {{
        background: {ACCENT_SOFT};
        border: 1px solid {ACCENT};
    }}
    QToolButton::menu-indicator {{
        subcontrol-position: right center;
    }}
    /* matplotlib's NavigationToolbar2QT: its icon glyphs are faint on their
       own, so give the toolbar strip a contrasting band + bottom border to
       separate it from the plot canvas and make icons easier to place. */
    QWidget#mplToolbar {{
        background: {SURFACE_MUTED};
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}
    QMenuBar {{
        background: {BG};
        border-bottom: 1px solid {BORDER};
        padding: 2px;
    }}
    QMenuBar::item {{
        padding: 4px 10px;
        border-radius: 4px;
    }}
    QMenuBar::item:selected {{
        background: {ACCENT_SOFT};
    }}
    QMenu {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 22px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background: {ACCENT_SOFT};
    }}
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        background: {SURFACE};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        padding: 8px 18px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        color: {TEXT_MUTED};
    }}
    QTabBar::tab:selected {{
        background: {SURFACE};
        color: {TEXT};
        font-weight: 600;
        border: 1px solid {BORDER};
        border-bottom: none;
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT};
    }}
    QListWidget {{
        background: {SURFACE_MUTED};
        border: none;
        border-radius: 8px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_STRONG};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {TEXT_FAINT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_STRONG};
        border-radius: 5px;
        min-width: 24px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    QSlider::groove:horizontal {{
        background: {BORDER_STRONG};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
    }}
    QPlainTextEdit {{
        background: {SURFACE_MUTED};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QStatusBar {{
        background: {BG};
    }}
    QMessageBox {{
        background: {SURFACE};
    }}
    QToolTip {{
        background: {TEXT};
        color: {SURFACE};
        border: none;
        padding: 4px 8px;
        border-radius: 4px;
    }}
    """

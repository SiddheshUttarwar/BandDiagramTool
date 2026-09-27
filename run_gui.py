"""
BandDiagramTool GUI entry point.

Usage:
    python run_gui.py

Build a device by adding/reordering layer cards, editing their properties,
setting contacts, and either solving a single bias point or running a
voltage sweep — no Python required.
"""

import sys

import matplotlib
matplotlib.use("QtAgg")

from PyQt6 import QtGui, QtWidgets

from gui import theme
from gui.app import App


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont(theme.FONT_FAMILY, theme.FONT_SIZE_PT))
    app.setStyleSheet(theme.stylesheet())
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

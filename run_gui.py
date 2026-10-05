"""
BandDiagramTool GUI entry point.

Usage:
    python run_gui.py

Build a device by adding/reordering layers, editing their properties,
setting contacts, and either solving a single bias point or running a
voltage sweep — no Python required.
"""

import sys

import matplotlib
matplotlib.use("QtAgg")

from PyQt6 import QtWidgets

from gui import theme
from gui.app import App


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BandDiagramTool")
    theme.apply(app)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

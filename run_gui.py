"""
BandDiagramTool GUI entry point.

Usage:
    python run_gui.py

Build a device by adding/reordering layer cards, editing their properties,
setting contacts, and either solving a single bias point or running a
voltage sweep — no Python required.
"""

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk

from gui.app import App


def main():
    root = tk.Tk()
    style = ttk.Style()
    for theme in ("vista", "clam"):
        try:
            style.theme_use(theme)
            break
        except tk.TclError:
            continue
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

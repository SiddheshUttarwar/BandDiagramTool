"""
BandDiagramTool GUI entry point.

Usage:
    python run_gui.py

Build a device by adding/reordering layers, editing their properties,
setting contacts, and either solving a single bias point or running a
voltage sweep — no Python required.

Setting the environment variable BANDDIAGRAMTOOL_SELFTEST to a file path
makes the program load a template, solve it, write one line with the result
to that file and exit. It is how the packaged .exe is checked after a build.
"""

import os
import sys

import matplotlib
matplotlib.use("QtAgg")

from PyQt6 import QtCore, QtWidgets

from gui import theme
from gui.app import App


def _selftest(app, window, path: str) -> None:
    """Solve the HEMT template and report, then quit."""
    import traceback

    def report(text: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        app.quit()

    def poll() -> None:
        try:
            if window._busy:
                return
            timer.stop()
            results = window.plot_panel.results
            if not results:
                report("FAIL no result: " + window._status_message.text())
                return
            r = results[0]
            from devices.analysis import sheet_density
            report(f"OK converged={r.converged} iterations={r.n_iterations} "
                   f"ns={sheet_density(r, 150, 225):.3e}")
        except Exception:  # noqa: BLE001
            timer.stop()
            report("FAIL " + traceback.format_exc().replace("\n", " | "))

    try:
        window._load_template("hemt")
    except Exception:  # noqa: BLE001
        report("FAIL " + traceback.format_exc().replace("\n", " | "))
        return
    timer = QtCore.QTimer(window)
    timer.timeout.connect(poll)
    timer.start(300)
    QtCore.QTimer.singleShot(180000, lambda: report("FAIL timeout"))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BandDiagramTool")
    theme.apply(app)
    window = App()
    window.show()
    selftest = os.environ.get("BANDDIAGRAMTOOL_SELFTEST")
    if selftest:
        QtCore.QTimer.singleShot(500, lambda: _selftest(app, window, selftest))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

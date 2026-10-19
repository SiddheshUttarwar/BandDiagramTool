# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for the Windows program.
#
#     pip install pyinstaller
#     pyinstaller --noconfirm EpiBand.spec
#
# Output: dist/EpiBand/EpiBand.exe, in a folder with its
# libraries (start-up is much faster than a single-file .exe, which unpacks
# itself on every launch). Zip that folder to distribute it.
# On macOS the same recipe also makes dist/EpiBand.app (built on a Mac: see
# .github/workflows/macos.yml; PyInstaller does not cross-compile).
import os
import sys

from PyInstaller.utils.hooks import collect_data_files

# the application icon, drawn by the program itself
os.makedirs('build', exist_ok=True)
ICON = os.path.join('build', 'EpiBand.ico')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6 import QtGui, QtWidgets  # noqa: E402
_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
from gui import theme  # noqa: E402
from PIL import Image  # noqa: E402
_png = os.path.join('build', 'icon.png')
theme.logo_pixmap(256).save(_png)
Image.open(_png).save(ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
MACOS = sys.platform == 'darwin'
if MACOS:
    theme.logo_pixmap(1024).save(_png)
    ICON = _png                      # PyInstaller converts it to .icns (needs Pillow)

datas = collect_data_files('scienceplots')      # the plot style sheets

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Not used by the GUI. torch is pulled in only by the unused
    # physics/pinn_solver.py and would add several hundred megabytes.
    excludes=['torch', 'torchvision', 'torchaudio', 'sympy', 'pandas', 'IPython', 'jupyter', 'notebook',
              'sklearn', 'numba', 'llvmlite', 'tkinter', 'pytest', 'PyQt5', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EpiBand',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='EpiBand',
)
if MACOS:
    app = BUNDLE(
        coll,
        name='EpiBand.app',
        icon=ICON,
        bundle_identifier='io.github.siddheshuttarwar.epiband',
        info_plist={'NSHighResolutionCapable': True, 'CFBundleShortVersionString': '1.2.0'},
    )

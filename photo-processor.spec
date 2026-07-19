# PyInstaller spec — one-file Windows exe for the RMS Photo Processor.
# Build with: pyinstaller photo-processor.spec  (see build.ps1)

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

hiddenimports = (
    collect_submodules("rembg.sessions")  # sessions are looked up dynamically
    + ["app", "app.main", "app.config", "app.faces", "app.formdata",
       "app.pipeline", "app.presets"]
)

datas = (
    collect_data_files("rembg")
    + collect_data_files("cv2", includes=["**/*.xml"])  # Haar cascades fallback
    # These read their own version via importlib.metadata at import time.
    + copy_metadata("pymatting")
    + copy_metadata("rembg")
)

a = Analysis(
    ["run.py"],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="photo-processor",
    debug=False,
    strip=False,
    upx=False,
    console=True,  # visible console so the shop staff can see it is running
    disable_windowed_traceback=False,
)

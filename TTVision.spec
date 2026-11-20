# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
sys.path.insert(0, str(root / "scripts"))
from fetch_ffmpeg import native_target, verify

verify(native_target())
extension = ".exe" if sys.platform == "win32" else ""
datas = [
    (str(root / "images/output_table_horizontal.png"), "images"),
    (str(root / "images/racket.png"), "images"),
    (str(root / "weights/table_segmentation.ckpt"), "weights"),
    (str(root / "third_party/ffmpeg"), "licenses/ffmpeg"),
    (str(root / "LICENSE"), "licenses/ttvision"),
    (str(root / "packaging/THIRD_PARTY.md"), "licenses"),
]
if sys.platform.startswith("linux"):
    datas.append((str(root / "packaging/linux/org.ttvision.TTVision.desktop"), "packaging/linux"))
datas += collect_data_files("sv_ttk")
# timm scripts activation functions during import; TorchScript needs their
# Python source files even in a frozen application.
datas += collect_data_files("timm", include_py_files=True)
for package in ("lightning", "pytorch-lightning", "torchmetrics", "segmentation-models-pytorch", "timm"):
    datas += copy_metadata(package)

a = Analysis(
    [str(root / "src/TTVision.py")],
    pathex=[str(root / "src")],
    binaries=[(str(root / "bin" / (name + extension)), "bin") for name in ("ffmpeg", "ffprobe")],
    datas=datas,
    # six.moves aliases are resolved by PyInstaller's pre-safe-import hook.
    hiddenimports=["six", "six.moves"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "pytest", "tensorboard", "torchaudio"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TTVision',
    icon=str(root / "images/racket.png") if sys.platform == "win32" else None,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=os.environ.get("TTVISION_CONSOLE") == "1",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
bundle = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TTVision")
if sys.platform == "darwin":
    # PyInstaller converts PNG to ICNS using the bundled build-time Pillow dependency.
    app = BUNDLE(bundle, name="TTVision.app", bundle_identifier="org.ttvision.desktop",
                 icon=str(root / "images/racket.png"))

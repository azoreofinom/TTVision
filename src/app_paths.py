"""Paths shared by source runs and PyInstaller bundles."""
from pathlib import Path
import shutil
import subprocess
import sys


def resource_path(relative):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / relative


def media_tool(name):
    if name not in ("ffmpeg", "ffprobe"):
        raise ValueError(f"Unknown media tool: {name}")
    filename = name + (".exe" if sys.platform == "win32" else "")
    bundled = resource_path("bin") / filename
    if bundled.is_file():
        return str(bundled)
    # Developer convenience only: frozen releases must be self-contained.
    if not getattr(sys, "frozen", False):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(f"Missing {filename}. Reinstall TTVision or run scripts/fetch_ffmpeg.py before building.")


def subprocess_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

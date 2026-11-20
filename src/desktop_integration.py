"""Install a Linux application-menu entry for an extracted standalone bundle."""
import os
import sys
from pathlib import Path

from app_paths import resource_path


def desktop_value(value):
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def desktop_exec(executable):
    value = str(executable)
    if "=" in value or any(character in value for character in "\n\r\t"):
        raise ValueError("Install TTVision in a folder without equals signs, tabs, or line breaks")
    # Desktop entries apply string escaping before Exec argument unquoting.
    value = value.replace("%", "%%")
    value = "".join("\\" + c if c in '\\"`$' else c for c in value)
    return '"' + desktop_value(value) + '"'


def install_linux_launcher():
    if not sys.platform.startswith("linux") or not getattr(sys, "frozen", False):
        raise RuntimeError("Run --install-desktop from the standalone Linux application")
    executable = Path(sys.executable).resolve(strict=True)
    icon = resource_path("images/racket.png").resolve(strict=True)
    template = resource_path("packaging/linux/org.ttvision.TTVision.desktop").read_text(encoding="utf-8")
    text = template.replace("@EXEC@", desktop_exec(executable)).replace("@ICON@", desktop_value(icon))
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    if not data_home.is_absolute():
        data_home = Path.home() / ".local/share"
    destination = data_home / "applications/org.ttvision.TTVision.desktop"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o644)
    return destination

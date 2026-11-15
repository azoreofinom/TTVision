"""Fetch pinned, SHA-256 verified media tools for the native build target."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "packaging" / "ffmpeg.json"


def native_target():
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(sys.platform, sys.platform)
    return f"{system}-{arch}"


def asset_path(target, asset):
    executable = asset.startswith(("ffmpeg-", "ffprobe-"))
    name = asset.split("-", 1)[0] + (".exe" if target.startswith("windows") else "") if executable else asset
    return ROOT / ("bin" if executable else "third_party/ffmpeg") / name


def fetch(url, destination, expected):
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = hashlib.sha256(partial.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {url}: {actual}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def verify(target):
    manifest = json.loads(MANIFEST.read_text())
    for asset, digest in manifest["targets"][target].items():
        path = asset_path(target, asset)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or unverified {path}; run scripts/fetch_ffmpeg.py")


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=manifest["targets"], default=native_target())
    args = parser.parse_args()
    if args.target not in manifest["targets"]:
        parser.error(f"Unsupported target {args.target}")
    for asset, digest in manifest["targets"][args.target].items():
        destination = asset_path(args.target, asset)
        print(f"Preparing {asset}", flush=True)
        fetch(f"{manifest['release']}/{asset}", destination, digest)
        if asset.startswith(("ffmpeg-", "ffprobe-")):
            destination.chmod(destination.stat().st_mode | 0o111)
    shutil.copy2(MANIFEST, ROOT / "third_party/ffmpeg/manifest.json")
    verify(args.target)


if __name__ == "__main__":
    main()

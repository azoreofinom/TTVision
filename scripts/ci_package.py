"""Check the native frozen build, then archive it with a SHA-256 checksum."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from fetch_ffmpeg import native_target

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {
    "windows-x64": ("TTVision", "TTVision.exe"),
    "macos-arm64": ("TTVision.app", "Contents/MacOS/TTVision"),
    "linux-x64": ("TTVision", "TTVision"),
}
SUCCESS = "Offline CPU inference and bundled FFmpeg export passed"


def check_application(bundle, executable, report_dir):
    executable = (bundle / executable).resolve(strict=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = (report_dir / "frozen-self-test.txt").resolve()
    report.unlink(missing_ok=True)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    # A fresh cwd also catches accidental dependencies on the source checkout.
    with tempfile.TemporaryDirectory(prefix="TTVision packaged check ") as cwd:
        try:
            result = subprocess.run(
                [str(executable), "--self-test", str(report)],
                cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=300, **options,
            )
        except subprocess.TimeoutExpired as error:
            (report_dir / "frozen-process.log").write_text(str(error), encoding="utf-8")
            raise RuntimeError("Frozen self-test timed out; check startup errors in the build logs") from error
    (report_dir / "frozen-process.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    contents = report.read_text(encoding="utf-8") if report.is_file() else "No self-test report was written."
    print(contents, flush=True)
    if result.returncode != 0 or SUCCESS not in contents:
        raise RuntimeError(f"Frozen self-test failed (exit {result.returncode}); see {report_dir}")


def archive_application(bundle, target, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if target == "windows-x64":
        archive = output_dir / f"TTVision-{target}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in sorted(bundle.rglob("*")):
                output.write(path, path.relative_to(bundle.parent))
    else:
        archive = output_dir / f"TTVision-{target}.tar.gz"
        # tar preserves executable permissions and macOS app-bundle symlinks.
        with tarfile.open(archive, "w:gz", dereference=False) as output:
            output.add(bundle, arcname=bundle.name)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/artifacts")
    args = parser.parse_args()
    if args.target != native_target():
        parser.error(f"Cannot check {args.target} on {native_target()}")
    folder, executable = TARGETS[args.target]
    bundle = (args.dist_dir / folder).resolve()
    check_application(bundle, executable, ROOT / "build/ci")
    archive = archive_application(bundle, args.target, args.output_dir.resolve())
    print(f"Created {archive}")


if __name__ == "__main__":
    main()

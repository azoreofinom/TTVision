# Standalone builds

Initial targets: Windows x64, macOS Apple Silicon (arm64), and Linux x64.
Build natively on each target; this is not cross compilation. Intel Macs,
Windows ARM, and Linux ARM are not included in this first configuration.
Use Python 3.12.10 with Tk/Tcl available and uv 0.9.7. `.python-version`
pins the release interpreter; other Python 3.12 patch versions can be used
for development. Linux builds should use the oldest supported distribution
(initial baseline: Ubuntu 22.04); newer glibc builds do not support older systems.
Minimum supported OS versions still need native testing before publication.

## Prepare and build

From the repository root on each OS:

```sh
python -m pip install uv==0.9.7
uv sync --frozen --extra build
uv run --frozen --extra build python scripts/fetch_ffmpeg.py
uv run --frozen --extra build python -m unittest discover -s tests
uv run --frozen --extra build python -m PyInstaller --noconfirm TTVision.spec
```

The download script fetches native FFmpeg/FFprobe binaries and license/readme
files from ffmpeg-static b6.1.1, checks their pinned SHA-256 digests, and stages
them in `bin/` and `third_party/ffmpeg/`. These generated files are ignored by
Git. The spec refuses to build with missing or mismatched media binaries.
Downloads happen at build time; the packaged app does not download dependencies.
`--target` permits staging another target's files, but does not enable cross builds.

Output: `dist/TTVision/TTVision.exe` on Windows, `dist/TTVision.app` on macOS,
and `dist/TTVision/TTVision` on Linux. Distribute the entire folder or app bundle.
The executable alone is not sufficient. Keep file permissions when archiving
macOS/Linux packages. `gui.spec` is an obsolete local configuration; use
`TTVision.spec`, which is tracked in Git.

Application icons use `images/racket.png`: Windows embeds an ICO in the EXE,
and macOS builds convert it to ICNS for the application bundle. Pillow performs
both conversions during the build; no additional icon tools are required.

On Linux, extract the entire folder to its permanent location, then run
`./TTVision --install-desktop` from that folder to add an application-menu
launcher with the racket icon. This needs no root privileges or system Python.
The entry is installed under `$XDG_DATA_HOME/applications` (normally
`~/.local/share/applications`). Run the command again after moving the folder.
To remove the menu entry, delete `org.ttvision.TTVision.desktop` from that
applications directory. The launcher uses absolute paths to the executable and
bundled PNG; the desktop environment may take a moment to refresh its menu.

The build includes Python, Tk resources, CPU PyTorch, model weights, image,
theme resources, FFmpeg and FFprobe. Release inference uses CPU consistently.
Model restoration disables pretrained encoder downloads. Frozen apps never
fall back to tools on the system PATH. Source runs may use PATH as a convenience.

## Validate

```sh
uv run --frozen --extra build python scripts/smoke_test.py
```

This checks imports, checkpoint inference with network access blocked, bundled
media tool resolution with PATH emptied, and a short H.264/AAC video export.
It runs from an unrelated working directory and uses paths containing spaces
and Unicode. Unit tests cover frozen-resource paths and platform guards.
Native GUI testing on all targets remains necessary; these checks do not prove
every supported OS works. Test real analysis/editing on a clean machine with
no Python or system FFmpeg before publishing.

The frozen executable also accepts `--self-test /absolute/path/report.txt` to
run the same checks without opening a window. On macOS, invoke
`TTVision.app/Contents/MacOS/TTVision` directly for this check.

Local validation on Windows used Python 3.12.14: five portability tests passed,
the installed dependency set passed `uv pip check`, and both source and frozen
offline inference/export checks passed. The locally generated windowed build
is in `dist/standalone/TTVision/`. macOS/Linux have not yet been built or tested.

GUI editing asks for a destination; CLI editing defaults to an output beside
the input video and accepts `--output`. Neither writes into the installation
directory. The application rejects output paths matching the input video.

## Dependencies

`pyproject.toml` declares direct dependencies and the CPU-only PyTorch index.
`uv.lock` locks transitive dependencies and download hashes across all targets.
NumPy 1.26.4 is intentional for this PyTorch generation. Only one OpenCV
distribution is installed. The old Albumentations resize-to-tensor step uses
OpenCV/NumPy/PyTorch directly, avoiding its extra OpenCV distribution.

After changing dependencies, regenerate the lock and pip export:

```sh
uv lock
uv export --frozen --extra build --no-emit-project --output-file requirements.txt
```

For pip rather than uv, use Python 3.12 in a clean virtual environment:

```sh
python -m pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

The pip export includes build tools. uv is preferred for per-package index
selection and repeatable builds. Do not reuse the old `ttenv` environment.

## GitHub Actions

`.github/workflows/build.yml` runs the three native builds on Windows 2022 x64,
macOS 15 ARM64, and Ubuntu 22.04 x64. It runs on relevant pull requests, pushes
to `main`, version tags such as `v1.0.0`, and manual dispatches.

Commit and push the workflow together with the build configuration, lockfile,
scripts, source changes, weights, and image assets. The workflow cannot build
from uncommitted local files. No custom secrets are required.

To run manually, open the repository's **Actions** tab, select **Build standalone
apps**, choose **Run workflow**, select the branch, and start the run. The workflow
must first be present on the default branch for the manual run button to appear.
Open the finished run and download the matching `TTVision-<target>-<run number>`
artifact. GitHub wraps the package and checksum in an artifact download; extract
that download, then extract the contained ZIP or tar.gz. Launch the executable
inside the extracted folder (or the `.app` on macOS).

Every job installs the frozen dependency lock, verifies FFmpeg hashes, runs
unit tests and offline source inference/export, builds from a clean PyInstaller
cache, and runs the frozen executable's self-test. Packages are uploaded only
after the frozen check passes. This catches missing packaged imports such as
`six`, even when source execution works. Diagnostic artifacts are uploaded even
on failure. A failing platform does not cancel the other platform builds.

Downloads include a SHA-256 file. macOS/Linux use tar.gz to retain executable
permissions and symlinks. Build artifacts and diagnostic logs expire after
14 days. These builds remain unsigned and need real GUI testing on each target.
The workflow creates downloadable Actions artifacts; it does not publish a release.

## Distribution follow-up

Release uploading, installers, signing, and notarization are separate follow-up
steps. Review `packaging/THIRD_PARTY.md` and preserve required
corresponding FFmpeg/library sources before public distribution.

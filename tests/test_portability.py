from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import app_paths
import fetch_ffmpeg


class PortabilityTests(unittest.TestCase):
    def test_source_resource_is_independent_of_cwd(self):
        self.assertEqual(app_paths.resource_path("weights/table_segmentation.ckpt"), ROOT / "weights/table_segmentation.ckpt")

    def test_frozen_binary_wins_without_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
            binary.parent.mkdir()
            binary.touch()
            with patch.object(sys, "_MEIPASS", folder, create=True), patch.object(sys, "frozen", True, create=True), patch("shutil.which", side_effect=AssertionError("PATH consulted")):
                self.assertEqual(app_paths.media_tool("ffmpeg"), str(binary))
                with self.assertRaises(FileNotFoundError):
                    app_paths.media_tool("ffprobe")

    def test_posix_subprocess_has_no_windows_flags(self):
        for system in ("darwin", "linux"):
            with patch.object(sys, "platform", system):
                self.assertEqual(app_paths.subprocess_options(), {})

    def test_target_detection(self):
        for system, machine, expected in (("win32", "AMD64", "windows-x64"), ("darwin", "arm64", "macos-arm64"), ("linux", "x86_64", "linux-x64")):
            with patch.object(sys, "platform", system), patch("platform.machine", return_value=machine):
                self.assertEqual(fetch_ffmpeg.native_target(), expected)

    def test_corrupt_download_is_not_installed(self):
        import io
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "ffmpeg"
            with patch.object(fetch_ffmpeg, "urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaises(ValueError):
                    fetch_ffmpeg.fetch("https://example.invalid/file", destination, "0" * 64)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".part").exists())


if __name__ == "__main__":
    unittest.main()

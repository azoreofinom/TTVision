import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ci_package


class PackageTests(unittest.TestCase):
    def test_archive_contains_full_folder_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "TTVision"
            (bundle / "_internal").mkdir(parents=True)
            (bundle / "TTVision.exe").write_bytes(b"executable")
            (bundle / "_internal/model.ckpt").write_bytes(b"weights")
            archive = ci_package.archive_application(bundle, "windows-x64", root / "output")
            with zipfile.ZipFile(archive) as package:
                self.assertEqual(package.read("TTVision/_internal/model.ckpt"), b"weights")
                self.assertIn("TTVision/TTVision.exe", package.namelist())
            checksum = archive.with_name(archive.name + ".sha256").read_text()
            self.assertEqual(checksum.split()[0], hashlib.sha256(archive.read_bytes()).hexdigest())

    @unittest.skipIf(sys.platform == "win32", "Windows does not expose POSIX executable permission bits")
    def test_tar_preserves_executable_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "TTVision"
            bundle.mkdir()
            executable = bundle / "TTVision"
            executable.write_bytes(b"executable")
            executable.chmod(0o755)
            archive = ci_package.archive_application(bundle, "linux-x64", root / "output")
            with tarfile.open(archive) as package:
                self.assertTrue(package.getmember("TTVision/TTVision").mode & 0o111)

    def test_successful_exit_without_report_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "TTVision.exe").touch()
            result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch.object(ci_package.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "Frozen self-test failed"):
                    ci_package.check_application(root, "TTVision.exe", root / "logs")

    def test_failed_exit_does_not_accept_stale_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "TTVision.exe").touch()
            logs = root / "logs"
            logs.mkdir()
            (logs / "frozen-self-test.txt").write_text(ci_package.SUCCESS)
            result = subprocess.CompletedProcess([], 1, stdout="", stderr="No module named six")
            with patch.object(ci_package.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "exit 1"):
                    ci_package.check_application(root, "TTVision.exe", logs)
            self.assertIn("No module named six", (logs / "frozen-process.log").read_text())


if __name__ == "__main__":
    unittest.main()

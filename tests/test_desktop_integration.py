from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import desktop_integration as desktop


class DesktopIntegrationTests(unittest.TestCase):
    def test_install_uses_permanent_paths_and_user_data_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
            root = Path(folder).resolve()
            executable = root / 'App space 100% $test' / 'TTVision'
            executable.parent.mkdir()
            executable.touch()
            def resource(name):
                return ROOT / name
            with patch.object(desktop.sys, 'platform', 'linux'), patch.object(desktop.sys, 'frozen', True, create=True), patch.object(desktop.sys, 'executable', str(executable)), patch.object(desktop, 'resource_path', side_effect=resource), patch.dict(os.environ, {'XDG_DATA_HOME': str(root / 'user data')}):
                destination = desktop.install_linux_launcher()
                self.assertEqual(destination, root / 'user data/applications/org.ttvision.TTVision.desktop')
                text = destination.read_text(encoding='utf-8')
                self.assertIn('Exec=' + desktop.desktop_exec(executable), text)
                self.assertIn('Icon=' + desktop.desktop_value(ROOT / 'images/racket.png'), text)
                self.assertNotIn('@EXEC@', text)
                self.assertNotIn('@ICON@', text)
                self.assertIn('Terminal=false', text)

    def test_exec_escaping(self):
        self.assertEqual(desktop.desktop_exec('/opt/100% game/TTVision'), '"/opt/100%% game/TTVision"')
        self.assertEqual(desktop.desktop_exec('/opt/$app/TTVision'), '"/opt/' + '\\' * 2 + '$app/TTVision"')
        self.assertEqual(desktop.desktop_exec('/opt/"app/TTVision'), '"/opt/' + '\\' * 2 + '"app/TTVision"')
        for path in ('/opt/a=b/TTVision', '/opt/a\nb/TTVision'):
            with self.assertRaises(ValueError):
                desktop.desktop_exec(path)

    def test_macos_icon_conversion(self):
        from PyInstaller.building.icon import normalize_icon_type
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as folder:
            icon = Path(normalize_icon_type(str(ROOT / 'images/racket.png'), ('icns',), 'icns', folder))
            self.assertEqual(icon.read_bytes()[:4], b'icns')
            with Image.open(icon) as image:
                self.assertEqual(image.format, 'ICNS')
                image.load()


if __name__ == '__main__':
    unittest.main()

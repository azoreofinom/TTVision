import os
import sys
import tkinter as tk
import unittest

from PIL import Image, ImageTk


@unittest.skipIf(sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
                 "Tk image tests require a display; run with xvfb-run on Linux")
class TkImageTests(unittest.TestCase):
    def test_pillow_tk_bridge_displays_pixels_and_transparency(self):
        root = tk.Tk()
        root.withdraw()
        try:
            source = Image.new("RGBA", (3, 2), (10, 20, 30, 255))
            source.putpixel((0, 0), (0, 0, 0, 0))
            photo = ImageTk.PhotoImage(source, master=root)
            self.assertEqual((photo.width(), photo.height()), (3, 2))
            self.assertEqual(root.tk.call(str(photo), "get", 1, 1), (10, 20, 30))
            self.assertTrue(root.tk.call(str(photo), "transparency", "get", 0, 0))
            label = tk.Label(root, image=photo)
            label.pack()
            root.update_idletasks()
            self.assertEqual(label.cget("image"), str(photo))
        finally:
            root.destroy()

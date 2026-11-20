"""Exercise offline model inference and bundled video tools without a GUI."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

def run_installation_check():
    # Load test dependencies after the caller redirects output, so import
    # failures are included in the report from a windowed executable.
    import cv2
    import numpy as np
    import torch
    import TTVision
    import edit_video
    import mask_processing
    from app_paths import media_tool, resource_path, subprocess_options

    if os.environ.get("TTVISION_TEST_GUI") == "1":
        import tkinter as tk
        from PIL import Image, ImageTk
        root = tk.Tk()
        root.withdraw()
        try:
            with Image.open(resource_path("images/output_table_horizontal.png")) as image:
                photo = ImageTk.PhotoImage(image, master=root)
                label = tk.Label(root, image=photo)
                label.pack()
                root.update_idletasks()
                assert photo.width() == image.width
            print("Pillow/Tk image display passed.")
        finally:
            root.destroy()

    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="TTVision smoke ") as folder:
            os.chdir(folder)
            with patch.dict(os.environ, {"PATH": ""}), patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network access")):
                assert cv2.imread(str(resource_path("images/output_table_horizontal.png"))) is not None
                assert cv2.imread(str(resource_path("images/racket.png"))) is not None
                model = mask_processing.load_model()
                mask = mask_processing.segment_image(np.zeros((320, 640, 3), dtype=np.uint8), model)
                assert mask.shape == (320, 640), mask.shape
                assert next(model.parameters()).device.type == "cpu"
                source = str(Path(folder) / "input ü sample.mp4")
                output = str(Path(folder) / "output ü sample.mp4")
                subprocess.run([
                    media_tool("ffmpeg"), "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", source,
                ], check=True, **subprocess_options())
                assert edit_video.has_audio_stream(source)
                assert edit_video.render_sequential_clips(
                    source, [(0, 0.3), (0.6, 0.9)], "ultrafast", output, 10
                )
                assert Path(output).stat().st_size > 0
                assert edit_video.has_audio_stream(output)
            print(f"Offline CPU inference and bundled FFmpeg export passed (torch {torch.__version__}).")
            # Windows cannot remove the process's current working directory.
            os.chdir(original_cwd)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    run_installation_check()

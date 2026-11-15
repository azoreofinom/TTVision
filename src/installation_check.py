"""Exercise offline model inference and bundled video tools without a GUI."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

def main():
    import cv2
    import numpy as np
    import torch
    import TTVision
    import edit_video
    import mask_processing
    from app_paths import media_tool, resource_path, subprocess_options

    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="TTVision smoke ") as folder:
            os.chdir(folder)
            with patch.dict(os.environ, {"PATH": ""}), patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network access")):
                assert cv2.imread(str(resource_path("images/output_table_horizontal.png"))) is not None
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
                command = edit_video.build_command(source, [(0, 0.5)], "ultrafast", output)
                assert edit_video.run_ffmpeg_with_progress(command, 0.5, None, 5, 10)
                assert Path(output).stat().st_size > 0
                assert edit_video.has_audio_stream(output)
            print(f"Offline CPU inference and bundled FFmpeg export passed (torch {torch.__version__}).")
            # Windows cannot remove the process's current working directory.
            os.chdir(original_cwd)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

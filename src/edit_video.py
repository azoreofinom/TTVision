import os
import time
import cv2
import numpy as np
import subprocess
import json
import mask_processing
import argparse
import tempfile
from pathlib import Path
from app_paths import media_tool, subprocess_options

DOWNSAMPLE_ROWS = 320
DOWNSAMPLE_COLS = 640
ANALYSIS_FPS = 15
GRACE_BEFORE = 1.0
GRACE_AFTER = 0.2

INIT_SEG_FPS = 5                 # run segmentation at 5 fps
STABILITY_IOU_THRESHOLD = 0.95   # masks almost identical
STABILITY_DURATION_SEC = 0.5     # must be stable for 0.5 seconds
INIT_TIMEOUT_SEC = 30            # max time allowed for init

PIXEL_COUNT_THRESHOLD = 2
IOU_THRESHOLD = 0.0001
SECONDS_TO_TRIGGER = 1.5

def ffmpeg_installed():
    try:
        media_tool("ffmpeg")
        media_tool("ffprobe")
        return True
    except FileNotFoundError:
        return False


def has_audio_stream(video_path):
    cmd = [
        media_tool("ffprobe"),
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, **subprocess_options())
    data = json.loads(result.stdout)

    return "streams" in data and len(data["streams"]) > 0


def compute_common_pixel_count(mask1,mask2):
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()

    return intersection


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]

    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    return merged




def run_ffmpeg_with_progress(cmd, total_duration_sec, progress_callback, total_frames, fps, stop_event=None, warning_queue=None):
    """Drain progress on a reader thread so cancellation also works during stalls."""
    import queue
    import threading

    if stop_event is not None and stop_event.is_set():
        return False
    progress_cmd = [cmd[0], "-loglevel", "error", "-progress", "pipe:1", "-nostats"] + cmd[1:]
    lines = queue.Queue()
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
        process = subprocess.Popen(
            progress_cmd, stdout=subprocess.PIPE, stderr=errors,
            text=True, **subprocess_options()
        )

        def read_progress():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_progress, daemon=True)
        reader.start()
        cancelled = False
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    break
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line.startswith("out_time_ms="):
                    try:
                        seconds = int(line.strip().split("=", 1)[1]) / 1_000_000
                    except ValueError:
                        continue
                    frame = max(0, min(int(seconds * fps), total_frames))
                    if progress_callback:
                        progress_callback(frame, total_frames)
            if cancelled:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            reader.join()
            process.stdout.close()

        if cancelled or (stop_event is not None and stop_event.is_set()):
            Path(cmd[-1]).unlink(missing_ok=True)
            return False
        if process.returncode != 0 or not Path(cmd[-1]).is_file():
            errors.seek(0)
            message = errors.read() or "FFmpeg did not create the output video."
            print(message)
            if warning_queue is not None:
                warning_queue.put(message)
            return False
    return True


ENCODERS = {"cpu": "libx264", "nvidia": "h264_nvenc", "amd": "h264_amf", "intel": "h264_qsv"}

# Ordered from fastest encoding to greatest encoding effort. Hardware preset
# names describe an effort trade-off, not equivalent quality across encoders.
ENCODER_PRESETS = {
    "ultrafast": ("p1", "speed"),
    "superfast": ("p2", "speed"),
    "veryfast": ("p3", "speed"),
    "faster": ("p3", "balanced"),
    "fast": ("p4", "balanced"),
    "medium": ("p4", "balanced"),
    "slow": ("p5", "quality"),
    "slower": ("p6", "quality"),
    "veryslow": ("p7", "quality"),
}


def encoder_options(encoder, preset):
    nvenc_preset, amf_quality = ENCODER_PRESETS[preset]
    fast = preset in ("ultrafast", "superfast", "veryfast", "faster", "fast")
    if encoder == "h264_nvenc":
        options = ["-preset", nvenc_preset, "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    elif encoder == "h264_amf":
        options = ["-quality", amf_quality, "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"]
    elif encoder == "h264_qsv":
        options = ["-preset", "fast" if fast else "slow", "-global_quality", "23"]
    else:
        options = ["-preset", preset, "-crf", "23"]
    return ["-c:v", encoder] + options + ["-pix_fmt", "yuv420p"]


def select_encoder(mode, preset, work, stop_event):
    """Test real encoding rather than trusting the compiled encoder list."""
    if mode not in ("auto", *ENCODERS):
        raise ValueError(f"Unknown encoder mode: {mode}")
    if mode == "cpu":
        return "libx264"
    candidates = [ENCODERS[key] for key in ("nvidia", "amd", "intel")] if mode == "auto" else [ENCODERS[mode]]
    probe = work / "encoder-probe.mkv"
    try:
        for encoder in candidates:
            if stop_event is not None and stop_event.is_set():
                break
            cmd = [media_tool("ffmpeg"), "-y", "-f", "lavfi", "-i",
                   "color=size=128x128:rate=30:duration=0.1"]
            cmd += encoder_options(encoder, preset) + [str(probe)]
            # Bound driver initialization and keep Cancel responsive via the runner.
            import threading
            probe_stop = threading.Event()
            timer = threading.Timer(10, probe_stop.set)

            class ProbeStop:
                def is_set(self):
                    return probe_stop.is_set() or (stop_event is not None and stop_event.is_set())

            timer.start()
            try:
                if run_ffmpeg_with_progress(cmd, 0.1, None, 3, 30, ProbeStop()):
                    return encoder
            finally:
                timer.cancel()
            probe.unlink(missing_ok=True)
    finally:
        probe.unlink(missing_ok=True)
    return "libx264"


def build_clip_command(video_path, start, end, preset, output_path, audio_exists, encoder="libx264"):
    # Input seeking is accurate when transcoding: preroll before start is discarded.
    cmd = [media_tool("ffmpeg"), "-y", "-ss", str(start), "-i", str(video_path),
           "-t", str(end - start), "-map", "0:v:0"]
    if audio_exists:
        cmd += ["-map", "0:a:0"]
    cmd += encoder_options(encoder, preset)
    if audio_exists:
        # PCM intermediates avoid a separate AAC priming delay at every cut.
        cmd += ["-c:a", "pcm_s16le"]
    return cmd + [str(output_path)]


def render_sequential_clips(video_path, intervals_sec, preset, output_path, fps,
                            progress_callback=None, stop_event=None, warning_queue=None, encoder="auto"):
    """Render a single interval at a time; join video by stream copy."""
    intervals_sec = [(start, end) for start, end in intervals_sec if end > start]
    if not intervals_sec:
        if warning_queue is not None:
            warning_queue.put("No segments were found to keep.")
        return False
    if stop_event is not None and stop_event.is_set():
        return False
    audio_exists = has_audio_stream(video_path)
    total_duration = sum(end - start for start, end in intervals_sec)

    last_progress = 0

    def report(fraction):
        nonlocal last_progress
        last_progress = max(last_progress, min(999, max(0, int(fraction * 1000))))
        if progress_callback:
            # Only the caller signals completion, after the final file is published.
            progress_callback(last_progress, 1000)

    # Keep temporary media beside the destination, including a staged final file.
    # An unsuccessful export therefore leaves an existing destination untouched.
    with tempfile.TemporaryDirectory(prefix="ttvision-clips-", dir=Path(output_path).resolve().parent) as folder:
        work = Path(folder)
        selected = select_encoder(encoder, preset, work, stop_event)
        print(f"Video encoder: {selected} {preset}")
        # Retry the entire clip set on CPU: never concatenate mixed encoders.
        for attempt in range(2):
            completed = 0.0
            filenames = []
            failed = False
            for index, (start, end) in enumerate(intervals_sec):
                duration = end - start
                clip = work / f"clip_{index:06d}.mkv"
                cmd = build_clip_command(video_path, start, end, preset, clip, audio_exists, selected)

                def clip_progress(frame, total, offset=completed, length=duration):
                    report(0.95 * (offset + length * frame / max(1, total)) / total_duration)

                if not run_ffmpeg_with_progress(cmd, duration, clip_progress,
                        max(1, round(duration * fps)), fps, stop_event,
                        warning_queue if selected == "libx264" else None):
                    failed = True
                    break
                completed += duration
                filenames.append(clip.name)
                report(0.95 * completed / total_duration)
            if not failed:
                break
            if (stop_event is not None and stop_event.is_set()) or selected == "libx264":
                return False
            print(f"{selected} failed; restarting all clips with CPU encoding.")
            for clip in work.glob("clip_*.mkv"):
                clip.unlink()
            selected = "libx264"

        manifest = work / "clips.txt"
        manifest.write_text("".join(f"file '{name}'\n" for name in filenames), encoding="utf-8")
        staged = work / "edited.mp4"
        cmd = [media_tool("ffmpeg"), "-y", "-f", "concat", "-safe", "1",
               "-i", str(manifest), "-map", "0:v:0", "-c:v", "copy"]
        if audio_exists:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-movflags", "+faststart", str(staged)]

        def join_progress(frame, total):
            report(0.95 + 0.05 * frame / max(1, total))

        if not run_ffmpeg_with_progress(cmd, total_duration, join_progress,
                max(1, round(total_duration * fps)), fps, stop_event, warning_queue):
            return False
        if stop_event is not None and stop_event.is_set():
            return False
        os.replace(staged, output_path)
    return True


def remove_low_overlap_segments(
        video_path,
        stop_event=None,
        progress_callback=None,
        preset="fast",
        warning_queue =None,
        display=False,
        output_path=None,
        encoder="auto",
        ):

    if not ffmpeg_installed():
        error_msg = "FFmpeg or FFprobe is missing. Reinstall TTVision. For source runs, run scripts/fetch_ffmpeg.py."
        print(error_msg)
        if warning_queue:
            warning_queue.put(error_msg)
        return


    source = Path(video_path).resolve()
    output_path = Path(output_path).resolve() if output_path else source.with_name(source.stem + "_edited.mp4")
    if os.path.normcase(str(source)) == os.path.normcase(str(output_path)):
        raise ValueError("Output must be different from the input video")
    output_path = str(output_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps>=ANALYSIS_FPS:
        skip_rate = round(fps/ANALYSIS_FPS)
    else:
        skip_rate = 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_trigger = int(SECONDS_TO_TRIGGER * ANALYSIS_FPS)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    
    mog2 = cv2.createBackgroundSubtractorMOG2(
        varThreshold=12,
        detectShadows=True,
        history=500
    )

    intervals = []
    low_overlap_counter = 0
    in_keep_segment = False
    segment_start_frame = 0

    segmentation_mask, corners, frame_idx = mask_processing.compute_stable_segmentation_mask(cap,fps, stop_event)
    
    if segmentation_mask is not None:
        print(f"initialized, frame ID:{frame_idx}")
    else:
        if warning_queue:
            warning_queue.put("Table not found. Try using a different video or make sure the camera is stationary after the first 30s!")
        return
    
    progress_interval = max(1, int(fps * 5))
    while True:
        if stop_event is not None and stop_event.is_set():
            print("Task cancelled!")
            return

        if frame_idx % progress_interval == 0: 
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)
            else:
                print(f"Analysis progress: {frame_idx/total_frames*100:.2f}%")

        if frame_idx % skip_rate != 0:
            ret = cap.grab()
            frame_idx += 1
            continue
        

        ret, frame = cap.read()
        if not ret:
            break

        downsampled_frame = cv2.resize(
            frame,
            (DOWNSAMPLE_COLS, DOWNSAMPLE_ROWS),
            interpolation=cv2.INTER_AREA
        )

        fg_mask = mog2.apply(downsampled_frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        fg_mask = mask_processing.remove_large_blobs(fg_mask)

        common_pix = compute_common_pixel_count(segmentation_mask, fg_mask)

        if display:
            overlay = downsampled_frame.copy()

            # segmentation in green
            overlay[segmentation_mask > 0] = [0, 255, 0]

            # motion in red
            overlay[fg_mask > 0] = [0, 0, 255]

            vis = cv2.addWeighted(downsampled_frame, 0.6, overlay, 0.4, 0)
            vis = cv2.resize(vis, None, fx=0.7, fy=0.7)
            cv2.putText(
                vis,
                f"IoU: {common_pix:.5f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            time.sleep(0.1)
            cv2.imshow("Processing", vis)

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

        # if iou < overlap_threshold:
        if common_pix < PIXEL_COUNT_THRESHOLD:
            low_overlap_counter += 1

            if low_overlap_counter >= frames_to_trigger and in_keep_segment:
                segment_end_frame = frame_idx - frames_to_trigger
                intervals.append((segment_start_frame, segment_end_frame))
                in_keep_segment = False
        else:
            low_overlap_counter = 0

            if not in_keep_segment:
                segment_start_frame = frame_idx
                in_keep_segment = True

        frame_idx += 1

    if in_keep_segment:
        intervals.append((segment_start_frame, frame_idx - 1))

    cap.release()
    
    # Convert frames → seconds + add grace period
    intervals_sec = []
    for start, end in intervals:
        start_sec = max(0, (start / fps) - GRACE_BEFORE)
        end_sec = min(total_frames / fps, (end / fps) + GRACE_AFTER)
        intervals_sec.append((start_sec, end_sec))

    intervals_sec = merge_intervals(intervals_sec)
   
    finished = render_sequential_clips(
        video_path, intervals_sec, preset, output_path, fps,
        progress_callback, stop_event, warning_queue, encoder
    )
    if finished and progress_callback:
        progress_callback(1,1)
        print(f"Edited video saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Remove low-overlap segments from a video")

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video"
    )

    parser.add_argument(
        "--output", help="Destination MP4 (default: beside the input video)"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show debug visualization"
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        help="Encoding preset (default: fast)"
    )

    parser.add_argument("--encoder", choices=["auto", *ENCODERS], default="auto",
                        help="Video encoder preference; falls back to CPU (default: auto)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    remove_low_overlap_segments(
        args.video_path,
        display=args.display,
        preset=args.preset,
        output_path=args.output,
        encoder=args.encoder,
    )

import os
import time
import cv2
import numpy as np
import subprocess
import json
import mask_processing
import argparse
import shutil

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
    return shutil.which("ffmpeg") is not None


def has_audio_stream(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
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




def run_ffmpeg_with_progress(cmd, total_duration_sec, progress_callback, total_frames, fps, stop_event = None, warning_queue = None):
    """Run ffmpeg and parse progress output."""
    
    # Insert progress flags after 'ffmpeg'
    progress_cmd = [cmd[0], "-loglevel", "fatal","-progress", "pipe:1", "-nostats"] + cmd[1:]
    
    # startupinfo = subprocess.STARTUPINFO()
    # startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    process = subprocess.Popen(
        progress_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # startupinfo=startupinfo,
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    for line in process.stdout:
        if stop_event is not None and stop_event.is_set():
            print("Encoding cancelled!")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            if os.path.exists(cmd[-1]):
                print("Removed partially encoded video")
                os.remove(cmd[-1])
            return False

        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                out_time_sec = int(line.split("=")[1]) / 1_000_000
                # Map ffmpeg time progress to frame progress
                # FFmpeg starts from 0; offset by the init frames
                ffmpeg_frame = int(out_time_sec * fps)
                if progress_callback:
                    progress_callback(ffmpeg_frame, total_frames)
                else:
                    print(f"Encoding progress: {ffmpeg_frame/total_frames*100:.2f}%")
            except ValueError:
                pass
    
    process.wait()

    return_code = process.returncode

    if return_code != 0:
        error_output = process.stderr.read()
        print("FFmpeg failed with return code:", return_code)
        print(error_output)
        if warning_queue:
            warning_queue.put(error_output)
        return False

    output_file = cmd[-1]
    if not os.path.exists(output_file):
        msg = "FFmpeg finished but output file was not created!"
        print(msg)
        if warning_queue:
            warning_queue.put(msg)
        return False

    return True



def build_command(video_path, intervals_sec, preset, output_path):
    # Build filter_complex script
    audio_exists = has_audio_stream(video_path)

    filter_parts = []
    concat_inputs = []

    for i, (start, end) in enumerate(intervals_sec):
        if audio_exists:
            filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
                f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[v{i}][a{i}]")
        else:
            filter_parts.append(
                f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
            )
            concat_inputs.append(f"[v{i}]")

    filter_complex = "".join(filter_parts)
    filter_complex += "".join(concat_inputs)

    if audio_exists:
        filter_complex += f"concat=n={len(intervals_sec)}:v=1:a=1[outv][outa]"
    else:
        filter_complex += f"concat=n={len(intervals_sec)}:v=1[outv]"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
    ]

    if audio_exists:
        cmd += ["-map", "[outa]"]

    # cmd += [
    #     "-preset", preset,
    #     output_path
    # ]


    # for win media player compatibility
    cmd += [
    "-c:v", "libx264",     # ensure H.264
    "-preset", preset,     # your selectable preset still works
    # "-crf", "18",
    "-pix_fmt", "yuv420p", # WMP compatibility
    ]
    if audio_exists:
        cmd += [
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    cmd += [
        "-movflags", "+faststart",
        output_path
    ]
    return cmd


def remove_low_overlap_segments(
        video_path,
        stop_event=None,
        progress_callback=None,
        preset="fast",
        warning_queue =None,
        display=False,
        ):

    if not ffmpeg_installed():
        error_msg = "FFmpeg was not found on your system. Please make sure it is installed and listed in the system PATH."
        print(error_msg)
        if warning_queue:
            warning_queue.put(error_msg)
        return


    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base_name}_edited.mp4"
    
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

            if low_overlap_counter >= frames_to_trigger:
                if in_keep_segment:
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
    total_frames = sum(round((end - start) * fps) for start, end in intervals_sec)
   
    cmd = build_command(video_path,intervals_sec, preset, output_path)
    finished = run_ffmpeg_with_progress(cmd, total_frames / fps, progress_callback, total_frames, fps, stop_event, warning_queue)
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
        "--display",
        action="store_true",
        help="Show debug visualization"
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="ultrafast",
        help="Encoding preset (default: ultrafast)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    remove_low_overlap_segments(
        args.video_path,
        display=args.display,
        preset=args.preset
    )
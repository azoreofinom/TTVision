import cProfile
import os
import pstats
import time
import cv2
import numpy as np
import subprocess
import tempfile
import segment
import json
from utils import postprocess, preprocess
import torch
from table_segmenter import TableSegmenter
from PIL import Image

#TODO: explore presets, output fps? grace period. segmentation quality makes or breakes it... also wait until seg mask is stable(setup period)

DOWNSAMPLE_ROWS = 320
DOWNSAMPLE_COLS = 640
ANALYSIS_FPS = 15
GRACE_BEFORE = 0.0
GRACE_AFTER = 0.2

INIT_SEG_FPS = 5                 # run segmentation at 5 fps
STABILITY_IOU_THRESHOLD = 0.95   # masks almost identical
STABILITY_DURATION_SEC = 0.5       # must be stable for 0.5 seconds
INIT_TIMEOUT_SEC = 10            # max time allowed for init


def has_audio_stream(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    return "streams" in data and len(data["streams"]) > 0

def compute_iou(mask1, mask2):
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


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


def load_model():
    basedir = os.path.dirname(__file__)
    model_path = os.path.join(basedir, "weights/table_segmentation.ckpt")
    print(model_path)
   
    ckpt = torch.load(model_path, map_location="cpu")
    print(ckpt["hyper_parameters"]["loss"])

    model = TableSegmenter.load_from_checkpoint(model_path, loss="DICE")
    
    # Ensure model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def segment_image(frame, model):
    # Resize the frame to the model input size
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sizes = [pil_img.size[::-1]]

    # Run the table segmentation model
    with torch.no_grad():
        t_imgs = preprocess([pil_img],device="cpu")
        t_masks = model.infer(t_imgs)
        masks = postprocess(t_masks, sizes)

    return masks[0]


def compute_stable_segmentation_mask(cap, fps):
    """
    Runs segmentation at 5 FPS until the mask stabilizes.
    Returns the stable mask and the frame index where init ended.
    """

    frame_interval = int(fps / INIT_SEG_FPS)
    timeout_frames = int(INIT_TIMEOUT_SEC * fps)
    stable_required_frames = int(STABILITY_DURATION_SEC * INIT_SEG_FPS)

    prev_mask = None
    stable_counter = 0
    frame_idx = 0

    model = load_model()
    
    while frame_idx < timeout_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame (≈ 5 FPS)
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        downsampled = cv2.resize(
            frame,
            (DOWNSAMPLE_COLS, DOWNSAMPLE_ROWS),
            interpolation=cv2.INTER_AREA
        )

        # current_mask = segment.segment_image(downsampled)
        current_mask = segment_image(downsampled,model)

        if prev_mask is not None:
            iou = compute_iou(prev_mask, current_mask)
            print(iou)
            if iou > STABILITY_IOU_THRESHOLD:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= stable_required_frames:
                print("Segmentation stabilized.")
                return current_mask, frame_idx

        prev_mask = current_mask
        frame_idx += 1

    print("Initialization timeout reached.")
    return None, None


def remove_low_overlap_segments(
        video_path,
        stop_event=None,
        progress_callback=None,
        display=False  
        ):

    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base_name}_edited_test.mp4"

    overlap_threshold = 0.0001
    seconds_to_trigger = 2
    grace_period = 0.0  # keep a bit before/after motion

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps>=ANALYSIS_FPS:
        skip_rate = round(fps/ANALYSIS_FPS)
    else:
        skip_rate = 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_trigger = int(seconds_to_trigger * ANALYSIS_FPS)
    print(frames_to_trigger)

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

    segmentation_mask, frame_idx = compute_stable_segmentation_mask(cap,fps)
    if segmentation_mask is not None:
        print(f"initialized, frame ID:{frame_idx}")
    else:
        #???
        return
    
    progress_interval = max(1, int(fps * 5))
    

    while True:
        if stop_event is not None and stop_event.is_set():
            print("Task cancelled!")
            #is this right? 
            return

        if frame_idx % progress_interval == 0: 
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)
            else:
                print(frame_idx/total_frames)

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

        iou = compute_iou(segmentation_mask, fg_mask)

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
                f"IoU: {iou:.5f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
 
            cv2.imshow("Processing", vis)

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

        if iou < overlap_threshold:
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
        start_sec = max(0, (start / fps) - grace_period)
        end_sec = min(total_frames / fps, (end / fps) + grace_period)
        intervals_sec.append((start_sec, end_sec))

    print(intervals_sec)
    intervals_sec = merge_intervals(intervals_sec)

    print("Final intervals:", intervals_sec)


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

    cmd += [
        "-preset", "fast",
        output_path
    ]

    subprocess.run(cmd)
    print(f"Edited video saved to: {output_path}")

if __name__ == '__main__':
    
    profiler = cProfile.Profile()
    profiler.enable()
 
    
    start_t = time.time()
    remove_low_overlap_segments(
            "myvideos/wtt.webm",
            display=True
        )
    # remove_low_overlap_segments(
    #     "openData/game_4.mp4",
    #     display=True
    # )
    end_t = time.time()
    print(end_t-start_t)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)
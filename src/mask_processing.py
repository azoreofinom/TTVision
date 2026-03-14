import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import deque
from utils import postprocess, preprocess
import torch
from table_segmenter import TableSegmenter
import os
from PIL import Image
import sys

DOWNSAMPLE_ROWS = 320
DOWNSAMPLE_COLS = 640
ANALYSIS_FPS = 15
GRACE_BEFORE = 1.0
GRACE_AFTER = 0.2

INIT_SEG_FPS = 5                 # run segmentation at 5 fps
STABILITY_IOU_THRESHOLD = 0.95   # masks almost identical
STABILITY_DURATION_SEC = 0.5     # must be stable for 0.5 seconds
INIT_TIMEOUT_SEC = 30  

def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Euclidean distance in RGB space."""
    return float(np.sqrt(np.sum((c1.astype(float) - c2.astype(float)) ** 2)))

def get_most_common_color_in_mask_lab(lab_image, mask, k=2, random_state=42):
    """
    Find dominant perceptual color inside mask.
    
    Parameters:
        lab_image : (H, W, 3) OpenCV Lab image (uint8)
        mask      : (H, W) binary mask (0 or >0)
        k         : number of clusters
    
    Returns:
        dominant_lab_cie  : dominant color in true CIE Lab (float)
        dominant_lab_cv   : dominant color in OpenCV Lab format (uint8)
    """

    mask_bool = mask > 0

    if not np.any(mask_bool):
        raise ValueError("Mask contains no positive pixels.")

    # Extract masked pixels
    pixels = lab_image[mask_bool].reshape(-1, 3)

    # K-means clustering in OpenCV Lab space
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Find largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]



    dominant_lab_cv = kmeans.cluster_centers_[dominant_cluster]
    
    # Convert to uint8 format
    # dominant_lab_cv_uint8 = np.clip(dominant_lab_cv, 0, 255).astype(np.uint8)
    # print(f"uint8:{dominant_lab_cv_uint8}")
    # # Convert to true CIE Lab
    # dominant_lab_cie = opencv_lab_to_cielab(dominant_lab_cv_uint8)
    return dominant_lab_cv

def remove_large_blobs(mask, max_area=150, kernel_size=10):
    """
    Removes large blobs and small blobs that are close enough to be connected by morphological closing.

    Parameters:
        mask (np.ndarray): Binary input mask.
        max_area (int): Maximum area to keep blobs.
        kernel_size (int): Size of the morphological kernel.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
   
    closed_mask = mask

    # Remove large blobs from closed mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            cleaned[labels == i] = 255

    return cleaned


def keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    """
    Keeps only the largest connected component in a binary mask.

    Args:
        mask (np.ndarray): Binary mask (0/1 or 0/255)

    Returns:
        np.ndarray: Binary mask with only the largest blob
    """
    
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # Ignore background (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    largest_blob = np.zeros_like(mask)
    largest_blob[labels == largest_label] = 1

    return largest_blob


def sample_mask_points(mask: np.ndarray, image: np.ndarray, most_common_color, num_samples: int = 5) -> list[tuple[int, int]]:
    # Get mask coordinates
    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        raise ValueError("Mask has no foreground pixels.")

    # Extract corresponding Lab pixels
    pixels = image[ys, xs].astype(np.float32)

    # Compute vectorized distances
    diffs = pixels - most_common_color.astype(np.float32)
    dists = np.linalg.norm(diffs, axis=1)

    # Keep only pixels within threshold
    valid_indices = np.where(dists <= 15)[0]

    if len(valid_indices) == 0:
        print("No pixels within distance threshold.")
        return []

    # Evenly sample from valid ones
    chosen = np.linspace(
        0, len(valid_indices) - 1,
        num=min(num_samples, len(valid_indices)),
        dtype=int
    )

    sampled_points = [
        (int(ys[valid_indices[i]]), int(xs[valid_indices[i]]))
        for i in chosen
    ]

    return sampled_points


def expand_mask_bfs(
    image: np.ndarray,
    mask: np.ndarray,
    seed_points: list[tuple[int, int]],
    threshold: float = 20.0,
) -> np.ndarray:
    """
    Starting from seed_points, flood-fill into pixels whose color is within
    `threshold` of the seed pixel's color. Already-masked pixels are skipped.
    """
    h, w = mask.shape[:2]
    new_mask = mask.copy()
    visited = new_mask > 0  # start with existing mask as visited

    queue = deque()

    for (y, x) in seed_points:
        ref_color = image[y, x]
        queue.append((y, x, ref_color))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    added = 0
    while queue:
        cy, cx, ref_color = queue.popleft()
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                pixel_color = image[ny, nx]
                if color_distance(ref_color, pixel_color) <= threshold:
                    new_mask[ny, nx] = 255
                    added += 1
                    queue.append((ny, nx, ref_color))

    return new_mask



def get_mask_corners_robust(mask: np.ndarray):
    """
    Returns 4 ordered corners from a binary mask.

    Output order:
    [top_left, top_right, bottom_right, bottom_left]
    format: (x, y)
    """

    mask = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Take largest contour
    contour = max(contours, key=cv2.contourArea)

    # Convex hull
    hull = cv2.convexHull(contour)


    # epsilon = 0.05 * cv2.arcLength(hull, True)
    # approx = cv2.approxPolyDP(hull, epsilon, True)

    # Polygon approximation
    approx = cv2.approxPolyN(hull, nsides=4)
    pts = approx.reshape(-1, 2)
   
    # ---- VISUALIZATION ----
    vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    # contour (blue)
    cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)

    # hull (green)
    cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)

    # approx polygon (red)
    cv2.polylines(vis, [approx], True, (0, 0, 255), 2)

    # draw approx points
    for p in pts:
        cv2.circle(vis, tuple(p), 5, (0, 255, 255), -1)

    # cv2.imshow("contour / hull / approx", vis)
    # cv2.waitKey(0)
    # -----------------------

    # If more than 4 points, fallback to bounding rect corners
    # if len(pts) != 4:
    #     x, y, w, h = cv2.boundingRect(hull)
    #     pts = np.array([
    #         [x, y],
    #         [x + w, y],
    #         [x + w, y + h],
    #         [x, y + h]
    #     ])


    return pts




def quad_mask(points, shape):
    """
    points: array-like with shape (4,2) -> [(x,y), ...]
    shape: (height, width) of mask
    """

    pts = np.array(points, dtype=np.float32)

    # order points around centroid
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    pts = pts[np.argsort(angles)]

    mask = np.zeros(shape, dtype=np.uint8)

    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

    return mask

    

def order_quad_points(points):
    pts = np.array(points, dtype=float)

    # sort by x
    x_sorted = pts[np.argsort(pts[:, 0])]

    left = x_sorted[:2]
    right = x_sorted[2:]

    # left side
    left = left[np.argsort(left[:, 1])]
    top_left, bottom_left = left

    # right side
    right = right[np.argsort(right[:, 1])]
    top_right, bottom_right = right

    return np.array([top_left, top_right, bottom_right, bottom_left])


def load_model():
    basedir = os.path.dirname(__file__)
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.join(basedir, ".."))
    path = "weights/table_segmentation.ckpt"
    model_path = os.path.join(base_path, path)
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

def compute_iou(mask1, mask2):
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union

def compute_stable_segmentation_mask(cap, fps, stop_event):
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
        if stop_event is not None and stop_event.is_set():
            print("Task cancelled!")
            #is this right? 
            return
        
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
        
            if iou > STABILITY_IOU_THRESHOLD:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= stable_required_frames:
                print("Segmentation stabilized.")

                
                downsampled_lab = cv2.cvtColor(downsampled,cv2.COLOR_BGR2LAB)
                most_common_color = get_most_common_color_in_mask_lab(downsampled_lab,current_mask)
                seeds = sample_mask_points(current_mask, downsampled_lab, most_common_color, num_samples=500)
                expanded = expand_mask_bfs(downsampled_lab, current_mask, seeds, threshold=10)
                corners = get_mask_corners_robust(expanded)
                final_mask = quad_mask(corners, current_mask.shape)
                
                color=(0, 0, 255)
                for (x, y) in corners:
                    cv2.circle(downsampled_lab, (int(x), int(y)), 5, color, -1)

                # cv2.imshow("current",current_mask)
                # cv2.imshow("expanded",expanded)
                # cv2.imshow("image",downsampled_lab)
                # cv2.imshow("final", final_mask)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                


                return final_mask, corners, frame_idx

        prev_mask = current_mask
        frame_idx += 1

    print("Initialization timeout reached.")
    return None, None, None
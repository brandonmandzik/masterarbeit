#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# ---------- video loading ----------

def read_video_frames(path, max_frames=None, to_gray=True):
    """
    Read frames from a video.

    Args:
        path: path to video file
        max_frames: optional cap on number of frames
        to_gray: if True, convert to grayscale

    Returns:
        list of frames as float32 in [0, 1]
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames to compute optical flow.")

    return frames


# ---------- optical flow (TV-L1) ----------

# Create the TV-L1 optical flow object once (reuse for speed)
_tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()


def compute_flow_tvl1(f1, f2):
    """
    Compute dense optical flow from f1 to f2 using TV-L1.

    OpenCV convention for dense flow:
        flow[y, x] = (u, v) such that
        f1(y, x) ≈ f2(y + v, x + u).

    Args:
        f1, f2: HxW float32 in [0,1] (grayscale)

    Returns:
        flow: HxW x 2 float32 (u, v), mapping f1 -> f2
    """
    # TV-L1 expects CV_8UC1 or CV_32FC1; our frames are CV_32FC1 already
    flow = _tvl1.calc(f1, f2, None)
    return flow.astype(np.float32)


# ---------- metrics ----------

def forward_backward_error(flow_fw, flow_bw):
    """
    Forward-backward consistency error.

    flow_fw: forward flow from t -> t+1
    flow_bw: backward flow from t+1 -> t

    OpenCV convention:
        flow_fw[y, x] = (u, v) : (x, y) in frame_t
        maps to (x+u, y+v) in frame_{t+1}.

    For each pixel p=(x,y) at time t:
      1) Move it to p' = p + flow_fw(p) in t+1.
      2) Sample backward flow flow_bw at p'.
      3) In the consistent case: flow_fw(p) + flow_bw(p') ≈ 0.

    Returns:
        fb_l1: scalar mean L1 error over valid pixels
        fb_l2: scalar mean L2 error over valid pixels
    """
    H, W, _ = flow_fw.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    # where forward flow lands in frame t+1
    x_fw = xs + flow_fw[..., 0]
    y_fw = ys + flow_fw[..., 1]

    # sample backward flow at forward endpoints
    flow_bw_sampled = cv2.remap(
        flow_bw,
        x_fw,
        y_fw,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    fb_sum = flow_fw + flow_bw_sampled  # should be ~0 if consistent

    # L1 norm of flow inconsistency per pixel (|u| + |v|)
    fb_err_l1 = np.abs(fb_sum[..., 0]) + np.abs(fb_sum[..., 1])

    # L2 norm of flow inconsistency per pixel (sqrt(u^2 + v^2))
    fb_err_l2 = np.linalg.norm(fb_sum, axis=-1)

    # valid mask: only positions that map inside the image
    valid = (
        (x_fw >= 0)
        & (x_fw <= W - 1)
        & (y_fw >= 0)
        & (y_fw <= H - 1)
    )

    fb_err_l1[~valid] = np.nan
    fb_err_l2[~valid] = np.nan

    fb_l1 = float(np.nanmean(fb_err_l1))
    fb_l2 = float(np.nanmean(fb_err_l2))

    return fb_l1, fb_l2


def warp_next_to_prev(frame_t1, flow_fw):
    """
    Warp frame at time t+1 into the coordinates of frame t
    using forward flow defined by OpenCV.

    OpenCV convention:
        flow_fw[y, x] = (u, v) such that
        frame_t(y, x) ≈ frame_t1(y + v, x + u).

    For each pixel (x, y) in frame_t, we want a sample
    from frame_t1 at (x+u, y+v). That gives us a version
    of frame_t1 aligned to frame_t coordinates.

    Args:
        frame_t1: frame at time t+1, HxW (or HxWxC), float32
        flow_fw:  forward flow from t -> t+1, HxW x 2

    Returns:
        warped_t1_to_t: frame_t1 warped into frame_t coordinates
                        with same shape as frame_t1.
    """
    H, W = frame_t1.shape[:2]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    x_src = xs + flow_fw[..., 0]  # x + u
    y_src = ys + flow_fw[..., 1]  # y + v

    warped = cv2.remap(
        frame_t1,
        x_src,
        y_src,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def frame_warp_error(frame_t, frame_t1, flow_fw):
    """
    Temporal warping error between consecutive frames.

    Using OpenCV's forward-flow convention:
      - flow_fw[y, x] = (u, v) such that
        frame_t(y, x) ≈ frame_t1(y + v, x + u).

    We:
      - warp frame_t1 back into the coordinate system of frame_t
        using the forward flow (via warp_next_to_prev)
      - compare the warped frame_t1 to frame_t.

    Args:
        frame_t:   frame at time t (HxW, float32 in [0,1])
        frame_t1:  frame at time t+1 (HxW, float32 in [0,1])
        flow_fw:   forward flow from t -> t+1 (HxW x 2)

    Returns:
        warp_l1: scalar mean L1 error (mean |frame_t - warped|)
        warp_l2: scalar L2 error as RMSE (sqrt(mean (frame_t - warped)^2))
    """
    warped_t1_to_t = warp_next_to_prev(frame_t1, flow_fw)
    diff = frame_t - warped_t1_to_t

    # L1: mean absolute difference
    warp_l1 = float(np.mean(np.abs(diff)))

    # L2: root-mean-square error (RMSE)
    warp_l2 = float(np.sqrt(np.mean(diff ** 2)))

    return warp_l1, warp_l2


def optical_flow_no_ref_score(video_path, max_frames=None, step=1):
    """
    Compute no-reference temporal consistency metrics
    based on TV-L1 optical flow.

    Args:
        video_path: path to input video file
        max_frames: optional cap on number of frames
        step:       use every N-th frame pair (1 = all)

    Returns:
        dict with raw error metrics and transformed quality scores.
    """
    frames = read_video_frames(video_path, max_frames=max_frames, to_gray=True)

    fb_errors_l1 = []
    fb_errors_l2 = []
    warp_errors_l1 = []
    warp_errors_l2 = []
    motion_mag_means = []

    num_pairs = (len(frames) - 1 + (step - 1)) // step  # ceil
    pair_idx = 0

    for t in range(0, len(frames) - 1, step):
        f_t = frames[t]
        f_t1 = frames[t + 1]
        pair_idx += 1

        # simple progress print
        print(f"[pair {pair_idx}/{num_pairs}] computing TV-L1 flow...", end="\r")

        # TV-L1 forward and backward flows
        flow_fw = compute_flow_tvl1(f_t, f_t1)   # t -> t+1
        flow_bw = compute_flow_tvl1(f_t1, f_t)   # t+1 -> t

        # forward-backward consistency
        fb_l1, fb_l2 = forward_backward_error(flow_fw, flow_bw)

        # warp error (t vs t+1 warped back using forward flow)
        warp_l1, warp_l2 = frame_warp_error(f_t, f_t1, flow_fw)

        fb_errors_l1.append(fb_l1)
        fb_errors_l2.append(fb_l2)
        warp_errors_l1.append(warp_l1)
        warp_errors_l2.append(warp_l2)

        # motion magnitude (from forward flow)
        mag = np.linalg.norm(flow_fw, axis=-1)  # per-pixel magnitude
        motion_mag_means.append(float(np.mean(mag)))

    print()  # newline after progress

    # ---------- aggregate raw error metrics ----------

    fb_mean_l1 = float(np.nanmean(fb_errors_l1))
    fb_mean_l2 = float(np.nanmean(fb_errors_l2))
    warp_mean_l1 = float(np.nanmean(warp_errors_l1))
    warp_mean_l2 = float(np.nanmean(warp_errors_l2))

    motion_mag_mean = float(np.mean(motion_mag_means))
    motion_mag_var = float(np.var(motion_mag_means))

    # ---------- convert errors -> quality scores (Q = exp(-E)) ----------

    alpha_fb = 1.0
    alpha_warp = 1.0
    alpha_mmv = 1.0

    Q_fb_mean_l1 = math.exp(-alpha_fb * fb_mean_l1)
    Q_fb_mean_l2 = math.exp(-alpha_fb * fb_mean_l2)

    Q_warp_mean_l1 = math.exp(-alpha_warp * warp_mean_l1)
    Q_warp_mean_l2 = math.exp(-alpha_warp * warp_mean_l2)

    # Use variance as “instability error” for motion magnitude
    Q_motion_mag_var = math.exp(-alpha_mmv * motion_mag_var)

    result = {
        # raw error metrics
        "fb_mean_l1": fb_mean_l1,
        "fb_mean_l2": fb_mean_l2,
        "warp_mean_l1": warp_mean_l1,
        "warp_mean_l2": warp_mean_l2,
        "motion_mag_mean": motion_mag_mean,
        "motion_mag_var": motion_mag_var,

        # quality scores (higher = better)
        "Q_fb_mean_l1": Q_fb_mean_l1,
        "Q_fb_mean_l2": Q_fb_mean_l2,
        "Q_warp_mean_l1": Q_warp_mean_l1,
        "Q_warp_mean_l2": Q_warp_mean_l2,
        "Q_motion_mag_var": Q_motion_mag_var,

        "num_pairs": len(fb_errors_l1),
    }

    return result


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="No-reference optical-flow consistency metric for videos (TV-L1)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input video folder.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally limit number of frames for speed.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Use every N-th frame pair (1 = all).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./results/tv_l1_results.csv",
        help="Output CSV file path (default: tv_l1_results.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = Path(args.input)

    if not input_folder.exists():
        raise SystemExit(f"Input folder not found: {input_folder}")

    if not input_folder.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_folder}")

    # Find all video files in the input folder
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    video_files = [f for f in input_folder.iterdir() if f.suffix in video_extensions]

    if not video_files:
        raise SystemExit(f"No video files found in: {input_folder}")

    # Sort video files by name for consistent ordering
    video_files.sort(key=lambda x: x.name)

    print(f"Found {len(video_files)} video file(s) in {input_folder}")
    print()

    # Prepare results
    results = []

    for idx, video_path in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")

        try:
            result = optical_flow_no_ref_score(
                video_path=video_path,
                max_frames=args.max_frames,
                step=args.step,
            )

            # Add filename to result dict
            result['filename'] = video_path.name
            results.append(result)

            # Print detailed metrics to terminal (original format)
            print(f"  Raw error metrics:")
            print(f"    fb_mean_l1:        {result['fb_mean_l1']:.6f}")
            print(f"    fb_mean_l2:        {result['fb_mean_l2']:.6f}")
            print(f"    warp_mean_l1:      {result['warp_mean_l1']:.6f}")
            print(f"    warp_mean_l2:      {result['warp_mean_l2']:.6f}")
            print(f"    motion_mag_mean:   {result['motion_mag_mean']:.6f}")
            print(f"    motion_mag_var:    {result['motion_mag_var']:.6f}")
            print(f"  Quality scores (higher = better):")
            print(f"    Q_fb_mean_l1:      {result['Q_fb_mean_l1']:.6f}")
            print(f"    Q_fb_mean_l2:      {result['Q_fb_mean_l2']:.6f}")
            print(f"    Q_warp_mean_l1:    {result['Q_warp_mean_l1']:.6f}")
            print(f"    Q_warp_mean_l2:    {result['Q_warp_mean_l2']:.6f}")
            print(f"    Q_motion_mag_var:  {result['Q_motion_mag_var']:.6f}")
            print(f"  num_pairs: {result['num_pairs']}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            # Still add a row with error indication
            result = {
                'filename': video_path.name,
                'fb_mean_l1': float('nan'),
                'fb_mean_l2': float('nan'),
                'warp_mean_l1': float('nan'),
                'warp_mean_l2': float('nan'),
                'motion_mag_mean': float('nan'),
                'motion_mag_var': float('nan'),
                'Q_fb_mean_l1': float('nan'),
                'Q_fb_mean_l2': float('nan'),
                'Q_warp_mean_l1': float('nan'),
                'Q_warp_mean_l2': float('nan'),
                'Q_motion_mag_var': float('nan'),
                'num_pairs': 0,
            }
            results.append(result)

    # Write CSV file with all metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as csvfile:
        fieldnames = [
            'filename',
            'fb_mean_l1',
            'fb_mean_l2',
            'warp_mean_l1',
            'warp_mean_l2',
            'motion_mag_mean',
            'motion_mag_var',
            'Q_fb_mean_l1',
            'Q_fb_mean_l2',
            'Q_warp_mean_l1',
            'Q_warp_mean_l2',
            'Q_motion_mag_var',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()

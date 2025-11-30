import argparse
import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute LPIPS and SSIM between (video, video) or (video, image)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--videos",
        nargs=2,
        metavar=("VIDEO1", "VIDEO2"),
        help="Compare two videos: --videos video1.mp4 video2.mp4",
    )
    group.add_argument(
        "--video",
        type=str,
        help="Video file path when comparing video to image: --video video.mp4 --image image.png",
    )
    group.add_argument(
        "--input-dir",
        type=str,
        help="Batch mode: Input directory containing source videos (source1_1 to source2_5)",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Image file path when comparing video to image: --video video.mp4 --image image.png",
    )

    parser.add_argument(
        "--reference-dir",
        type=str,
        help="Batch mode: Reference directory containing ground-truth videos (original1 to original5)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="lpips_ssim_results.csv",
        help="Output CSV file path (default: lpips_ssim_results.csv)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Spatial size to which frames and images are resized (default: 256).",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every N-th frame (default: 1 = use all frames).",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to process (per video pair).",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.input_dir:
        if not args.reference_dir:
            sys.exit("ERROR: --reference-dir is required when using --input-dir.")
        if not os.path.isdir(args.input_dir):
            sys.exit(f"ERROR: Input directory not found: {args.input_dir}")
        if not os.path.isdir(args.reference_dir):
            sys.exit(f"ERROR: Reference directory not found: {args.reference_dir}")
    elif args.videos:
        v1, v2 = args.videos
        if not os.path.isfile(v1):
            sys.exit(f"ERROR: Video not found: {v1}")
        if not os.path.isfile(v2):
            sys.exit(f"ERROR: Video not found: {v2}")
    else:
        if not args.image:
            sys.exit("ERROR: --image is required when using --video.")
        if not os.path.isfile(args.video):
            sys.exit(f"ERROR: Video not found: {args.video}")
        if not os.path.isfile(args.image):
            sys.exit(f"ERROR: Image not found: {args.image}")

    return args


def compute_stats(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "count": int(len(arr)),
    }


def print_stats(name, stats):
    print(f"\n{name}:")
    print(f"  count : {stats['count']}")
    print(f"  min   : {stats['min']:.6f}")
    print(f"  max   : {stats['max']:.6f}")
    print(f"  mean  : {stats['mean']:.6f}")
    print(f"  median: {stats['median']:.6f}")
    print(f"  std   : {stats['std']:.6f}")


def tensor_to_uint8_np(img_tensor):
    """
    img_tensor: (1, 3, H, W) in [0, 1]
    returns: (H, W, 3) uint8 in [0, 255]
    """
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def prepare_transforms(image_size):
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # [0, 1]
        ]
    )
    return transform


def setup_lpips(device):
    loss_fn = lpips.LPIPS(net="alex")
    loss_fn = loss_fn.to(device)
    loss_fn.eval()
    return loss_fn


def lpips_distance(lpips_model, im1_tensor, im2_tensor):
    """
    im*_tensor: (1, 3, H, W) in [0, 1]
    LPIPS expects [-1, 1].
    """
    with torch.no_grad():
        t1 = (im1_tensor * 2.0 - 1.0)
        t2 = (im2_tensor * 2.0 - 1.0)
        d = lpips_model(t1, t2)
    return float(d.item())


def process_video_video(
    video_path_1,
    video_path_2,
    lpips_model,
    transform,
    device,
    frame_stride=1,
    max_frames=None,
):
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    if not cap1.isOpened():
        sys.exit(f"ERROR: Could not open video: {video_path_1}")
    if not cap2.isOpened():
        sys.exit(f"ERROR: Could not open video: {video_path_2}")

    frame_idx = 0
    processed_frames = 0

    lpips_scores = []
    ssim_scores = []

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            # Stop when either video ends
            break

        # frame stride
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Convert BGR->RGB
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        pil1 = Image.fromarray(frame1_rgb)
        pil2 = Image.fromarray(frame2_rgb)

        # ---- LPIPS preprocessing ----
        t1 = transform(pil1).unsqueeze(0).to(device)  # [0,1]
        t2 = transform(pil2).unsqueeze(0).to(device)

        lp = lpips_distance(lpips_model, t1, t2)
        lpips_scores.append(lp)

        # ---- SSIM preprocessing ----
        img1_np = tensor_to_uint8_np(t1.cpu())
        img2_np = tensor_to_uint8_np(t2.cpu())

        ssim_val = ssim(
            img1_np,
            img2_np,
            channel_axis=2,
            data_range=255,
        )
        ssim_scores.append(float(ssim_val))

        frame_idx += 1
        processed_frames += 1

        if max_frames is not None and processed_frames >= max_frames:
            break

    cap1.release()
    cap2.release()

    if len(lpips_scores) == 0:
        sys.exit("ERROR: No frames processed. Check your frame_stride / max_frames.")

    return lpips_scores, ssim_scores


def process_video_image(
    video_path,
    image_path,
    lpips_model,
    transform,
    device,
    frame_stride=1,
    max_frames=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open video: {video_path}")

    # Load & preprocess reference image once
    ref_pil = Image.open(image_path).convert("RGB")
    ref_tensor = transform(ref_pil).unsqueeze(0).to(device)  # [0,1]
    ref_np = tensor_to_uint8_np(ref_tensor.cpu())

    frame_idx = 0
    processed_frames = 0

    lpips_scores = []
    ssim_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # ---- LPIPS ----
        frame_tensor = transform(pil_frame).unsqueeze(0).to(device)
        lp = lpips_distance(lpips_model, ref_tensor, frame_tensor)
        lpips_scores.append(lp)

        # ---- SSIM ----
        frame_np = tensor_to_uint8_np(frame_tensor.cpu())
        ssim_val = ssim(
            ref_np,
            frame_np,
            channel_axis=2,
            data_range=255,
        )
        ssim_scores.append(float(ssim_val))

        frame_idx += 1
        processed_frames += 1

        if max_frames is not None and processed_frames >= max_frames:
            break

    cap.release()

    if len(lpips_scores) == 0:
        sys.exit("ERROR: No frames processed. Check your frame_stride / max_frames.")

    return lpips_scores, ssim_scores


def find_video_file(directory, base_name):
    """Find video file with common extensions."""
    extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for ext in extensions:
        path = os.path.join(directory, base_name + ext)
        if os.path.isfile(path):
            return path
    return None


def process_batch(
    input_dir,
    reference_dir,
    lpips_model,
    transform,
    device,
    frame_stride=1,
    max_frames=None,
):
    """Process all source videos against reference videos."""
    results = []

    # Expected naming: source1_1 to source1_5, source2_1 to source2_5
    # Reference: original1 to original5
    sources = ["source1", "source2"]
    video_ids = [1, 2, 3, 4, 5]

    for source in sources:
        for vid_id in video_ids:
            source_name = f"{source}_{vid_id}"
            reference_name = f"original{vid_id}"

            # Find video files
            source_path = find_video_file(input_dir, source_name)
            reference_path = find_video_file(reference_dir, reference_name)

            if source_path is None:
                print(f"WARNING: Source video not found: {source_name}")
                continue

            if reference_path is None:
                print(f"WARNING: Reference video not found: {reference_name}")
                continue

            print(f"\nProcessing: {os.path.basename(source_path)} vs {os.path.basename(reference_path)}")

            # Compute metrics
            lpips_vals, ssim_vals = process_video_video(
                source_path,
                reference_path,
                lpips_model,
                transform,
                device,
                frame_stride=frame_stride,
                max_frames=max_frames,
            )

            lpips_stats = compute_stats(lpips_vals)
            ssim_stats = compute_stats(ssim_vals)

            # Store results
            results.append({
                "Filename": os.path.basename(source_path),
                "LPIPS_mean": lpips_stats["mean"],
                "LPIPS_median": lpips_stats["median"],
                "LPIPS_std": lpips_stats["std"],
                "LPIPS_min": lpips_stats["min"],
                "LPIPS_max": lpips_stats["max"],
                "SSIM_mean": ssim_stats["mean"],
                "SSIM_median": ssim_stats["median"],
                "SSIM_std": ssim_stats["std"],
                "SSIM_min": ssim_stats["min"],
                "SSIM_max": ssim_stats["max"],
            })

            print(f"  LPIPS mean: {lpips_stats['mean']:.6f}")
            print(f"  SSIM mean: {ssim_stats['mean']:.6f}")

    return results


def write_csv(results, output_path):
    """Write results to CSV file in COVER-like format."""
    if not results:
        print("WARNING: No results to write")
        return

    fieldnames = [
        "Filename",
        "LPIPS_mean",
        "LPIPS_median",
        "LPIPS_std",
        "LPIPS_min",
        "LPIPS_max",
        "SSIM_mean",
        "SSIM_median",
        "SSIM_std",
        "SSIM_min",
        "SSIM_max",
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")


def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    transform = prepare_transforms(args.image_size)
    lpips_model = setup_lpips(device)

    if args.input_dir:
        # Batch mode
        print(f"Batch mode:\n  INPUT: {args.input_dir}\n  REFERENCE: {args.reference_dir}")
        results = process_batch(
            args.input_dir,
            args.reference_dir,
            lpips_model,
            transform,
            device,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        write_csv(results, args.output)
    elif args.videos:
        # Single video-to-video mode
        v1, v2 = args.videos
        print(f"Comparing videos:\n  VIDEO1: {v1}\n  VIDEO2: {v2}")
        lpips_vals, ssim_vals = process_video_video(
            v1,
            v2,
            lpips_model,
            transform,
            device,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        lpips_stats = compute_stats(lpips_vals)
        ssim_stats = compute_stats(ssim_vals)
        print_stats("LPIPS", lpips_stats)
        print_stats("SSIM", ssim_stats)
    else:
        # Video-to-image mode
        print(f"Comparing video and image:\n  VIDEO: {args.video}\n  IMAGE: {args.image}")
        lpips_vals, ssim_vals = process_video_image(
            args.video,
            args.image,
            lpips_model,
            transform,
            device,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        lpips_stats = compute_stats(lpips_vals)
        ssim_stats = compute_stats(ssim_vals)
        print_stats("LPIPS", lpips_stats)
        print_stats("SSIM", ssim_stats)


if __name__ == "__main__":
    main()

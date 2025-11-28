#!/usr/bin/env python3
"""
Minimal CLIP Score Assessment Tool
Calculates mean CLIP similarity between video frames and reference image/text.
"""
import argparse
import torch
import clip
from PIL import Image
import av
import numpy as np
import os
import csv
from pathlib import Path


def extract_frames(video_path):
    """Extract all frames from video."""
    frames = []
    container = av.open(video_path)

    for frame in container.decode(video=0):
        img = frame.to_image()
        frames.append(img)

    container.close()
    return frames


def calculate_clip_score_image(video_path, image_path, device, verbose=False):
    """Calculate mean CLIP score between video frames and reference image."""
    if verbose:
        print("\n[Loading CLIP model...]")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load reference image
    if verbose:
        print(f"[Loading reference image: {image_path}]")
    ref_image = Image.open(image_path).convert('RGB')
    if verbose:
        print(f"[Reference image size: {ref_image.size}]")
    ref_features = model.encode_image(preprocess(ref_image).unsqueeze(0).to(device))
    ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)

    # Extract and process video frames
    if verbose:
        print(f"[Extracting frames from video...]")
    frames = extract_frames(video_path)
    if verbose:
        print(f"[Extracted {len(frames)} frames]")
    scores = []

    for i, frame in enumerate(frames):
        frame_input = preprocess(frame).unsqueeze(0).to(device)
        frame_features = model.encode_image(frame_input)
        frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (frame_features @ ref_features.T).item()
        scores.append(similarity)

        if verbose and (i + 1) % 10 == 0:
            print(f"[Processed {i + 1}/{len(frames)} frames, current score: {similarity:.4f}]")

    scores_array = np.array(scores)

    if verbose:
        print(f"\n[Score statistics:]")
        print(f"  Min: {np.min(scores_array):.4f}")
        print(f"  Max: {np.max(scores_array):.4f}")
        print(f"  Mean: {np.mean(scores_array):.4f}")
        print(f"  Median: {np.median(scores_array):.4f}")
        print(f"  Std: {np.std(scores_array):.4f}")

    return np.mean(scores_array), np.median(scores_array), len(frames), scores if verbose else None


def calculate_clip_score_text(video_path, text_prompt, device, verbose=False):
    """Calculate mean CLIP score between video frames and text prompt."""
    if verbose:
        print("\n[Loading CLIP model...]")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Encode text prompt
    if verbose:
        print(f"[Encoding text prompt: '{text_prompt}']")
    text_tokens = clip.tokenize([text_prompt]).to(device)

    # Check for truncation (CLIP's context length is 77 tokens)
    token_count = (text_tokens[0] != 0).sum().item()
    if token_count >= 77:
        print(f"WARNING: Text prompt truncated (77 token limit reached, {len(text_prompt.split())} words provided)")

    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Extract and process video frames
    if verbose:
        print(f"[Extracting frames from video...]")
    frames = extract_frames(video_path)
    if verbose:
        print(f"[Extracted {len(frames)} frames]")
    scores = []

    for i, frame in enumerate(frames):
        frame_input = preprocess(frame).unsqueeze(0).to(device)
        frame_features = model.encode_image(frame_input)
        frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (frame_features @ text_features.T).item()
        scores.append(similarity)

        if verbose and (i + 1) % 10 == 0:
            print(f"[Processed {i + 1}/{len(frames)} frames, current score: {similarity:.4f}]")

    scores_array = np.array(scores)

    if verbose:
        print(f"\n[Score statistics:]")
        print(f"  Min: {np.min(scores_array):.4f}")
        print(f"  Max: {np.max(scores_array):.4f}")
        print(f"  Mean: {np.mean(scores_array):.4f}")
        print(f"  Median: {np.median(scores_array):.4f}")
        print(f"  Std: {np.std(scores_array):.4f}")

    return np.mean(scores_array), np.median(scores_array), len(frames), scores if verbose else None


def calculate_clip_score_image_to_image(image1_path, image2_path, device, verbose=False):
    """Calculate CLIP score between two images."""
    if verbose:
        print("\n[Loading CLIP model...]")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load both images
    if verbose:
        print(f"[Loading image 1: {image1_path}]")
    img1 = Image.open(image1_path).convert('RGB')
    if verbose:
        print(f"[Image 1 size: {img1.size}]")
        print(f"[Loading image 2: {image2_path}]")
    img2 = Image.open(image2_path).convert('RGB')
    if verbose:
        print(f"[Image 2 size: {img2.size}]")

    # Encode images
    if verbose:
        print("[Encoding images...]")
    img1_features = model.encode_image(preprocess(img1).unsqueeze(0).to(device))
    img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)

    img2_features = model.encode_image(preprocess(img2).unsqueeze(0).to(device))
    img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)

    # Cosine similarity
    if verbose:
        print("[Computing cosine similarity...]")
        print(f"[Feature vector norm - Image 1: {img1_features.norm():.4f}]")
        print(f"[Feature vector norm - Image 2: {img2_features.norm():.4f}]")
    similarity = (img1_features @ img2_features.T).item()

    return similarity


def calculate_clip_score_video_to_video(video1_path, video2_path, device, verbose=False):
    """Calculate mean CLIP score between corresponding frames of two videos."""
    if verbose:
        print("\n[Loading CLIP model...]")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Extract frames from both videos
    if verbose:
        print(f"[Extracting frames from video 1: {video1_path}]")
    frames1 = extract_frames(video1_path)
    if verbose:
        print(f"[Extracted {len(frames1)} frames from video 1]")
        print(f"[Extracting frames from video 2: {video2_path}]")
    frames2 = extract_frames(video2_path)
    if verbose:
        print(f"[Extracted {len(frames2)} frames from video 2]")

    # Handle videos with different frame counts
    num_frames = min(len(frames1), len(frames2))
    if len(frames1) != len(frames2):
        print(f"WARNING: Videos have different frame counts ({len(frames1)} vs {len(frames2)}). Comparing first {num_frames} frames.")

    if num_frames == 0:
        raise ValueError("One or both videos have no frames")

    scores = []

    # Compare corresponding frames
    for i in range(num_frames):
        frame1_input = preprocess(frames1[i]).unsqueeze(0).to(device)
        frame1_features = model.encode_image(frame1_input)
        frame1_features = frame1_features / frame1_features.norm(dim=-1, keepdim=True)

        frame2_input = preprocess(frames2[i]).unsqueeze(0).to(device)
        frame2_features = model.encode_image(frame2_input)
        frame2_features = frame2_features / frame2_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (frame1_features @ frame2_features.T).item()
        scores.append(similarity)

        if verbose and (i + 1) % 10 == 0:
            print(f"[Processed {i + 1}/{num_frames} frame pairs, current score: {similarity:.4f}]")

    scores_array = np.array(scores)

    if verbose:
        print(f"\n[Score statistics:]")
        print(f"  Min: {np.min(scores_array):.4f}")
        print(f"  Max: {np.max(scores_array):.4f}")
        print(f"  Mean: {np.mean(scores_array):.4f}")
        print(f"  Median: {np.median(scores_array):.4f}")
        print(f"  Std: {np.std(scores_array):.4f}")

    return np.mean(scores_array), np.median(scores_array), num_frames, scores if verbose else None


def batch_calculate_clip_scores(result_folder, input_folder, output_folder, device, verbose=False):
    """
    Calculate CLIP scores between result videos and their corresponding original videos.

    Naming convention:
    - Result videos: source{source_id}_{video_id}.mp4 (e.g., source1_1.mp4, source1_2.mp4, source2_1.mp4)
    - Input videos: original{video_id}.mp4 (e.g., original1.mp4, original2.mp4)

    Args:
        result_folder: Path to folder containing result videos (source1_1.mp4, etc.)
        input_folder: Path to folder containing original videos (original1.mp4, etc.)
        output_folder: Path to folder where CSV results will be saved
        device: Device to use (cuda/cpu)
        verbose: Enable verbose output
    """
    result_path = Path(result_folder)
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all result videos matching the pattern source{source_id}_{video_id}.mp4
    result_videos = sorted([f for f in result_path.glob("source*_*.mp4")])

    if not result_videos:
        print(f"ERROR: No result videos found in {result_folder} matching pattern 'source*_*.mp4'")
        return

    print(f"\nFound {len(result_videos)} result videos to process")
    print(f"Result folder: {result_folder}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Using device: {device}\n")

    # Store results for all videos
    results = []

    # Process each result video
    for result_video in result_videos:
        # Parse filename: source{source_id}_{video_id}.mp4
        filename = result_video.stem  # e.g., "source1_1"
        parts = filename.split('_')

        if len(parts) != 2:
            print(f"WARNING: Skipping {result_video.name} - unexpected filename format")
            continue

        # Extract video_id (the number after underscore)
        try:
            video_id = parts[1]  # e.g., "1" from "source1_1"
            original_video = input_path / f"original{video_id}.mp4"
        except (IndexError, ValueError):
            print(f"WARNING: Skipping {result_video.name} - could not parse video ID")
            continue

        # Check if original video exists
        if not original_video.exists():
            print(f"WARNING: Skipping {result_video.name} - original video not found: {original_video}")
            continue

        print(f"Processing: {result_video.name} vs {original_video.name}")

        try:
            # Calculate CLIP score (don't request frame-by-frame scores)
            mean_score, median_score, num_frames, _ = calculate_clip_score_video_to_video(
                str(result_video),
                str(original_video),
                device,
                verbose=False  # Disable verbose to avoid frame-by-frame computation
            )

            # Store result
            results.append({
                'filename': result_video.name,
                'original': original_video.name,
                'mean_score': mean_score,
                'median_score': median_score,
                'frames': num_frames
            })

            print(f"  ✓ Mean Score: {mean_score:.4f} | Median: {median_score:.4f} | Frames: {num_frames}\n")

        except Exception as e:
            print(f"  ✗ ERROR processing {result_video.name}: {str(e)}\n")
            continue

    # Write summarized CSV file
    if results:
        csv_filename = output_path / "clip_score_results.csv"

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header - only essential columns for P.1401
            writer.writerow(['Filename', 'Mean_CLIP_Score', 'Median_CLIP_Score'])

            # Write all results
            for result in results:
                writer.writerow([
                    result['filename'],
                    f"{result['mean_score']:.6f}",
                    f"{result['median_score']:.6f}"
                ])

        print(f"\n✓ Batch processing complete!")
        print(f"✓ Results saved to: {csv_filename}")
        print(f"✓ Total videos processed: {len(results)}")
    else:
        print("\n✗ No results to save. No videos were successfully processed.")


def main():
    parser = argparse.ArgumentParser(description='Calculate CLIP Score for video/image comparisons')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument('--video2', help='Path to second video (for video-to-video comparison)')
    parser.add_argument('--image', help='Path to reference image')
    parser.add_argument('--image2', help='Path to second image (for image-to-image comparison)')
    parser.add_argument('--text', help='Text prompt for comparison')
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch processing mode for folders')
    parser.add_argument('--result-folder', help='Path to folder containing result videos (for batch mode)')
    parser.add_argument('--input-folder', help='Path to folder containing original videos (for batch mode)')
    parser.add_argument('--output-folder', help='Path to folder for CSV output (for batch mode)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output with detailed processing information')

    args = parser.parse_args()

    device = args.device

    # Batch processing mode
    if args.batch:
        if not args.result_folder or not args.input_folder or not args.output_folder:
            parser.error('Batch mode requires --result-folder, --input-folder, and --output-folder')

        print(f"Using device: {device}")
        batch_calculate_clip_scores(
            args.result_folder,
            args.input_folder,
            args.output_folder,
            device,
            args.verbose
        )
        return

    print(f"Using device: {device}")

    # Image-to-image comparison
    if args.image and args.image2 and not args.video and not args.text:
        print(f"Image 1: {args.image}")
        print(f"Image 2: {args.image2}")
        score = calculate_clip_score_image_to_image(args.image, args.image2, device, args.verbose)
        print(f"\nResults:")
        print(f"  CLIP Score (image-to-image): {score:.4f}")
        return

    # Video-to-video comparison
    if args.video and args.video2 and not args.image and not args.text:
        print(f"Video 1: {args.video}")
        print(f"Video 2: {args.video2}")
        result = calculate_clip_score_video_to_video(args.video, args.video2, device, args.verbose)
        if args.verbose:
            mean_score, median_score, num_frames, scores = result
        else:
            mean_score, median_score, num_frames, _ = result
        print(f"\nResults:")
        print(f"  Frame pairs analyzed: {num_frames}")
        print(f"  Mean CLIP Score (video-to-video): {mean_score:.4f}")
        print(f"  Median CLIP Score (video-to-video): {median_score:.4f}")
        return

    # Video-based comparisons
    if not args.video:
        parser.error('Must provide --video (or use --image and --image2 for image comparison, or --video and --video2 for video comparison)')

    if not args.image and not args.text:
        parser.error('For video comparison, provide either --image or --text')

    if args.image and args.text:
        parser.error('Provide only one: --image or --text (not both)')

    print(f"Video: {args.video}")

    if args.image:
        print(f"Reference image: {args.image}")
        result = calculate_clip_score_image(args.video, args.image, device, args.verbose)
        if args.verbose:
            mean_score, median_score, num_frames, scores = result
        else:
            mean_score, median_score, num_frames, _ = result
        print(f"\nResults:")
        print(f"  Frames analyzed: {num_frames}")
        print(f"  Mean CLIP Score (video-to-image): {mean_score:.4f}")
        print(f"  Median CLIP Score (video-to-image): {median_score:.4f}")

    elif args.text:
        print(f"Text prompt: '{args.text}'")
        result = calculate_clip_score_text(args.video, args.text, device, args.verbose)
        if args.verbose:
            mean_score, median_score, num_frames, scores = result
        else:
            mean_score, median_score, num_frames, _ = result
        print(f"\nResults:")
        print(f"  Frames analyzed: {num_frames}")
        print(f"  Mean CLIP Score (video-to-text): {mean_score:.4f}")
        print(f"  Median CLIP Score (video-to-text): {median_score:.4f}")


if __name__ == '__main__':
    main()

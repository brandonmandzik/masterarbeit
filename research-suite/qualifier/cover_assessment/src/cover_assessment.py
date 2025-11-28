#!/usr/bin/env python3
"""
COVER Video Quality Assessment
Simple implementation following KISS principle with ARM64 compatibility
"""

import os
import sys
import glob
import argparse
import csv
import av
import numpy as np
import torch
from pathlib import Path

# Patch decord with PyAV before importing COVER modules
class PyAVVideoReader:
    """Wrapper to make PyAV compatible with decord.VideoReader interface"""
    def __init__(self, path, num_threads=1):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        # Pre-decode all frames
        self.frames = []
        for frame in self.container.decode(video=0):
            # Convert to torch tensor to match decord behavior
            img = torch.from_numpy(frame.to_ndarray(format='rgb24'))
            self.frames.append(img)
        self.container.close()
        self._total_frames = len(self.frames)

    def __len__(self):
        return self._total_frames

    def __getitem__(self, idx):
        """Return frame as torch tensor to match decord bridge behavior"""
        return self.frames[idx]

# Mock decord module
class MockBridge:
    def set_bridge(self, name):
        pass

class MockDecord:
    VideoReader = PyAVVideoReader
    bridge = MockBridge()
    @staticmethod
    def cpu(num=0):
        return f"cpu:{num}"
    @staticmethod
    def gpu(num=0):
        return f"gpu:{num}"

sys.modules['decord'] = MockDecord()

# Add COVER repo to path
cover_path = os.path.join(os.path.dirname(__file__), 'cover_repo')
sys.path.insert(0, cover_path)

import yaml
from cover.models import COVER
from cover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

# Normalization constants from official COVER
mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

mean_clip, std_clip = (
    torch.FloatTensor([122.77, 116.75, 104.09]),
    torch.FloatTensor([68.50, 66.63, 70.32])
)


def fuse_results(results):
    """
    Fuse semantic, technical, and aesthetic scores
    results: [semantic, technical, aesthetic]
    """
    return {
        "semantic": results[0],
        "technical": results[1],
        "aesthetic": results[2],
        "overall": results[0] + results[1] + results[2],
    }


def assess_video(model, video_path, samplers, sample_types, device):
    """
    Assess a single video using COVER
    """
    try:
        # Step 1: Load and decompose video into multi-view representations
        views, _ = spatial_temporal_view_decomposition(
            video_path, sample_types, samplers
        )

        # Step 2: Normalize each view
        for k, v in views.items():
            num_clips = sample_types[k].get("num_clips", 1)
            if k == 'technical' or k == 'aesthetic':
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(device)
                )
            elif k == 'semantic':
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean_clip) / std_clip)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(device)
                )

        # Step 3: Run inference
        with torch.no_grad():
            results = [r.mean().item() for r in model(views)]

        # Step 4: Fuse results
        scores = fuse_results(results)

        return scores

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Assess video quality using COVER')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Directory containing video files to assess (default: ../source_videos)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='File pattern for video files (default: *.mp4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results/cover_results.csv',
        help='Output CSV file (default: cover_results.csv)'
    )
    args = parser.parse_args()

    # Determine video directory
    if args.input:
        video_dir = os.path.abspath(args.input)
    else:
        video_dir = os.path.join(os.path.dirname(__file__), '..', 'source_videos')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load config
    config_path = os.path.join(cover_path, 'cover.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get sample types configuration (use val-ytugc as default)
    dopt = config['data']['val-ytugc']['args']
    sample_types = dopt['sample_types']

    # Create temporal samplers exactly as official COVER does
    samplers = {}
    for stype, sopt in sample_types.items():
        samplers[stype] = UnifiedFrameSampler(
            sopt["clip_len"] // sopt["t_frag"],
            sopt["t_frag"],
            sopt["frame_interval"],
            sopt["num_clips"],
        )

    # Load COVER model
    print("Loading COVER model...")
    model = COVER(**config["model"]["args"]).to(device)

    model_path = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'COVER.pth')
    state_dict = torch.load(model_path, map_location=device)

    # set strict=False to avoid error of missing weights
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    print("Model loaded successfully!\n")

    # Find videos
    video_files = sorted(glob.glob(os.path.join(video_dir, args.pattern)))

    if not video_files:
        print(f"No video files matching '{args.pattern}' found in {video_dir}")
        return

    print(f"Input directory: {video_dir}")
    print(f"Found {len(video_files)} videos to assess\n")
    print("=" * 80)

    # Collect results
    results = []

    # Process each video
    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"\nProcessing: {filename}")

        scores = assess_video(
            model, video_path, samplers, sample_types, device
        )

        if scores is not None:
            print(f"  Semantic Quality:   {scores['semantic']:.4f}")
            print(f"  Technical Quality:  {scores['technical']:.4f}")
            print(f"  Aesthetic Quality:  {scores['aesthetic']:.4f}")
            print(f"  Overall Quality:    {scores['overall']:.4f}")

            results.append({
                'filename': filename,
                'semantic': scores['semantic'],
                'technical': scores['technical'],
                'aesthetic': scores['aesthetic'],
                'overall': scores['overall']
            })
        else:
            print(f"  ERROR: Failed to process video")

    print("\n" + "=" * 80)

    # Write results to CSV
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Semantic', 'Technical', 'Aesthetic', 'Overall'])

            for result in results:
                writer.writerow([
                    result['filename'],
                    result['semantic'],
                    result['technical'],
                    result['aesthetic'],
                    result['overall']
                ])

        print(f"Results saved to: {output_path}")
        print(f"Processed {len(results)} videos successfully")
    else:
        print("No videos were processed successfully")

    print("Assessment complete!")


if __name__ == "__main__":
    main()

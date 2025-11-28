#!/usr/bin/env python3
"""
SI/TI Assessment Tool - Main CLI Entry Point

Calculates Spatial Information (SI) and Temporal Information (TI)
according to ITU-T Recommendation P.910 for video quality assessment.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict

from video_processor import VideoProcessor, find_video_files
from si_ti_calculator import SITICalculator
from visualizer import SITIVisualizer


def process_video(video_path: Path) -> Dict:
    """
    Process a single video file and calculate SI/TI metrics.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video info and SI/TI results
    """
    with VideoProcessor(str(video_path)) as vp:
        # Get video metadata
        video_info = vp.get_video_info()

        # Calculate SI/TI metrics (frames processed incrementally to save memory)
        results = SITICalculator.process_video_frames(vp.extract_frames(show_progress=True))

        # Combine metadata and results
        results['video_info'] = video_info

        return results


def save_csv_results(results: Dict[str, Dict], output_dir: Path):
    """
    Save frame-by-frame SI/TI values to CSV files.

    Args:
        results: Dictionary mapping video names to their SI/TI results
        output_dir: Output directory
    """
    for video_name, data in results.items():
        csv_path = output_dir / f"{Path(video_name).stem}_si_ti.csv"

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'SI', 'TI'])

            si_values = data['si_values']
            ti_values = data['ti_values']

            # First frame has SI but no TI
            writer.writerow([0, f"{si_values[0]:.4f}", "N/A"])

            # Remaining frames have both SI and TI
            for i in range(len(ti_values)):
                writer.writerow([i+1, f"{si_values[i+1]:.4f}", f"{ti_values[i]:.4f}"])

        print(f"Saved CSV: {csv_path}")


def save_json_results(results: Dict[str, Dict], output_dir: Path):
    """
    Save aggregated statistics to JSON file.

    Args:
        results: Dictionary mapping video names to their SI/TI results
        output_dir: Output directory
    """
    json_path = output_dir / "si_ti_results.json"

    # Prepare data for JSON (exclude frame-by-frame values)
    json_data = {}
    for video_name, data in results.items():
        json_data[video_name] = {
            'video_info': data['video_info'],
            'statistics': {
                'si_max': data['si_max'],
                'si_mean': data['si_mean'],
                'si_median': data['si_median'],
                'si_std': data['si_std'],
                'ti_max': data['ti_max'],
                'ti_mean': data['ti_mean'],
                'ti_median': data['ti_median'],
                'ti_std': data['ti_std'],
            }
        }

    with open(json_path, 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=2)

    print(f"Saved JSON: {json_path}")


def save_summary_report(results: Dict[str, Dict], output_dir: Path):
    """
    Save human-readable summary report.

    Args:
        results: Dictionary mapping video names to their SI/TI results
        output_dir: Output directory
    """
    report_path = output_dir / "si_ti_summary.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SI/TI ASSESSMENT REPORT (ITU-T P.910)\n")
        f.write("=" * 80 + "\n\n")

        for video_name, data in results.items():
            info = data['video_info']
            f.write(f"Video: {video_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Resolution: {info['width']}x{info['height']}\n")
            f.write(f"  Frame Rate: {info['fps']:.2f} fps\n")
            f.write(f"  Total Frames: {info['total_frames']}\n")
            f.write(f"  Duration: {info['duration_sec']:.2f} seconds\n")
            f.write("\n")
            f.write("  Spatial Information (SI):\n")
            f.write(f"    Maximum:    {data['si_max']:.2f}\n")
            f.write(f"    Mean:       {data['si_mean']:.2f}\n")
            f.write(f"    Median:     {data['si_median']:.2f}\n")
            f.write(f"    Std Dev:    {data['si_std']:.2f}\n")
            f.write("\n")
            f.write("  Temporal Information (TI):\n")
            f.write(f"    Maximum:    {data['ti_max']:.2f}\n")
            f.write(f"    Mean:       {data['ti_mean']:.2f}\n")
            f.write(f"    Median:     {data['ti_median']:.2f}\n")
            f.write(f"    Std Dev:    {data['ti_std']:.2f}\n")
            f.write("\n")
            f.write("=" * 80 + "\n\n")

    print(f"Saved summary report: {report_path}")


def main():
    """Main entry point for SI/TI assessment tool."""
    parser = argparse.ArgumentParser(
        description="SI/TI Assessment Tool - Calculate Spatial and Temporal Information according to ITU-T P.910",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input ./videos --output ./results
  python main.py -i /path/to/videos -o /path/to/output --no-csv

Reference:
  ITU-T Recommendation P.910: Subjective video quality assessment methods
  for multimedia applications
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing video files'
    )

    parser.add_argument(
        '-o', '--output',
        default='./results',
        help='Output directory for results (default: ./output)'
    )

    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Do not save CSV files with per-frame data'
    )

    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Do not save JSON file with statistics'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not generate plots'
    )

    args = parser.parse_args()

    # Find video files
    print(f"\nSearching for videos in: {args.input}")
    try:
        video_files = find_video_files(args.input)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not video_files:
        print(f"Error: No video files found in {args.input}", file=sys.stderr)
        print(f"Supported formats: {', '.join(VideoProcessor.SUPPORTED_FORMATS)}")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s):\n")
    for vf in video_files:
        print(f"  - {vf.name}")
    print()

    # Process each video
    results = {}
    for video_path in video_files:
        print(f"\n{'='*80}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*80}")

        try:
            video_results = process_video(video_path)
            results[video_path.name] = video_results

            # Print quick summary
            print(f"\nResults for {video_path.name}:")
            print(f"  SI: max={video_results['si_max']:.2f}, mean={video_results['si_mean']:.2f}")
            print(f"  TI: max={video_results['ti_max']:.2f}, mean={video_results['ti_mean']:.2f}")

        except Exception as e:
            print(f"Error processing {video_path.name}: {e}", file=sys.stderr)
            continue

    if not results:
        print("\nError: No videos were successfully processed", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")

    if not args.no_csv:
        save_csv_results(results, output_dir)

    if not args.no_json:
        save_json_results(results, output_dir)

    save_summary_report(results, output_dir)

    if not args.no_plots:
        visualizer = SITIVisualizer(str(output_dir))
        visualizer.generate_all_plots(results)

    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

def compute_mos_ci_from_votes(votes):
    """
    votes: 1D array-like of individual ratings for a single video.
    Returns: mos, std, n, ci95
    """
    votes = np.asarray(votes, dtype=float)
    n = len(votes)
    mos = votes.mean()
    std = votes.std(ddof=1) if n > 1 else 0.0

    if n > 1:
        t_val = stats.t.ppf(1 - 0.05/2, df=n-1)
        ci95 = t_val * std / np.sqrt(n)
    else:
        ci95 = 0.0

    return mos, std, n, ci95


def process_csv_folder(input_folder):
    """
    Process all CSV files in the input folder and compute MOS statistics.

    Args:
        input_folder: Path to folder containing CSV files with rating data

    Returns:
        Tuple of (results_df, combined_df) where:
        - results_df: DataFrame with MOS statistics per video
        - combined_df: Combined raw ratings from all CSV files
    """
    input_path = Path(input_folder)

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_folder}")

    print(f"Found {len(csv_files)} CSV file(s) in {input_folder}")

    all_ratings = []

    for csv_file in csv_files:
        print(f"Reading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        all_ratings.append(df)

    combined_df = pd.concat(all_ratings, ignore_index=True)

    print(f"\nTotal ratings collected: {len(combined_df)}")
    print(f"Unique videos: {combined_df['Filename'].nunique()}\n")

    results = []

    for filename in sorted(combined_df['Filename'].unique()):
        votes = combined_df[combined_df['Filename'] == filename]['Rating'].values
        mos, std, n, ci95 = compute_mos_ci_from_votes(votes)

        results.append({
            'Filename': filename,
            'MOS': mos,
            'STD': std,
            'N': n,
            'CI95': ci95
        })

    results_df = pd.DataFrame(results)
    return results_df, combined_df


def plot_results(combined_df, output_path):
    """
    Create box plot of rating distributions grouped by source.

    Args:
        combined_df: Combined DataFrame with all ratings
        output_path: Path to save the plot image
    """
    matplotlib.use('Agg')

    filenames = sorted(combined_df['Filename'].unique())

    source1_videos = [f for f in filenames if f.startswith('source1')]
    source2_videos = [f for f in filenames if f.startswith('source2')]

    all_videos = source1_videos + source2_videos

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    data_to_plot = []
    colors = []

    for i, filename in enumerate(all_videos):
        votes = combined_df[combined_df['Filename'] == filename]['Rating'].values
        data_to_plot.append(votes)
        positions.append(i + 1)

        if filename.startswith('source1'):
            colors.append('#3498db')
        else:
            colors.append('#e74c3c')

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=False,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'),
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Video Filename', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rating', fontsize=12, fontweight='bold')
    ax.set_title('MOS Rating Distributions by Video and Source', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f.replace('.mp4', '') for f in all_videos], rotation=45, ha='right')
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='Source 1'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Source 2')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {Path(output_path).absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute MOS and 95% confidence intervals from assessment CSV files'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input folder containing CSV files with rating data'
    )
    parser.add_argument(
        '-o', '--output',
        default='./results/mos/mos_results.csv',
        help='Output CSV filename (default: mos_results.csv)'
    )
    parser.add_argument(
        '-p', '--plot',
        default='./results/mos/mos_boxplot.png',
        help='Output plot filename (default: mos_boxplot.png)'
    )

    args = parser.parse_args()

    results_df, combined_df = process_csv_folder(args.input)

    print("=" * 80)
    print("MOS Statistics")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.absolute()}")

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(combined_df, plot_path)
    print()


if __name__ == "__main__":
    main()

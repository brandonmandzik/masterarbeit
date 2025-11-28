"""Visualization module for generating SI/TI plots using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


class SITIVisualizer:
    """Generate various plots for SI/TI analysis."""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style (with fallback for compatibility)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            # Fallback to default style if seaborn style not available
            pass

    def plot_time_series(self, results: Dict[str, Dict], output_name: str = "si_ti_timeseries.png"):
        """
        Plot SI and TI values over time for each video.

        Args:
            results: Dictionary mapping video names to their SI/TI results
            output_name: Output filename
        """
        num_videos = len(results)
        fig, axes = plt.subplots(num_videos, 2, figsize=(14, 4 * num_videos))

        if num_videos == 1:
            axes = axes.reshape(1, -1)

        for idx, (video_name, data) in enumerate(results.items()):
            si_values = data['si_values']
            ti_values = data['ti_values']

            # Plot SI
            axes[idx, 0].plot(si_values, linewidth=1.5, color='#2E86AB')
            axes[idx, 0].set_title(f'{video_name} - Spatial Information (SI)', fontsize=12, fontweight='bold')
            axes[idx, 0].set_xlabel('Frame Number', fontsize=10)
            axes[idx, 0].set_ylabel('SI Value', fontsize=10)
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].axhline(y=data['si_mean'], color='r', linestyle='--',
                                 label=f"Mean: {data['si_mean']:.2f}", alpha=0.7)
            axes[idx, 0].legend(loc='upper right')

            # Plot TI
            axes[idx, 1].plot(ti_values, linewidth=1.5, color='#A23B72')
            axes[idx, 1].set_title(f'{video_name} - Temporal Information (TI)', fontsize=12, fontweight='bold')
            axes[idx, 1].set_xlabel('Frame Number', fontsize=10)
            axes[idx, 1].set_ylabel('TI Value', fontsize=10)
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].axhline(y=data['ti_mean'], color='r', linestyle='--',
                                 label=f"Mean: {data['ti_mean']:.2f}", alpha=0.7)
            axes[idx, 1].legend(loc='upper right')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved time series plot: {output_path}")

    def plot_si_ti_scatter(self, results: Dict[str, Dict], output_name: str = "si_ti_scatter.png"):
        """
        Plot SI vs TI scatter plot (classic P.910 visualization).

        Args:
            results: Dictionary mapping video names to their SI/TI results
            output_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for idx, (video_name, data) in enumerate(results.items()):
            si_values = data['si_values']
            ti_values = data['ti_values']

            # Align TI values (one less than SI)
            si_aligned = si_values[1:]  # Skip first frame to match TI length

            ax.scatter(si_aligned, ti_values, alpha=0.5, s=10,
                      color=colors[idx], label=video_name)

            # Mark mean point
            ax.scatter(data['si_mean'], data['ti_mean'],
                      marker='*', s=300, color=colors[idx],
                      edgecolors='black', linewidths=1.5, zorder=10)

        ax.set_xlabel('Spatial Information (SI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temporal Information (TI)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Frame SI-TI Distribution (Detailed Analysis)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SI-TI scatter plot: {output_path}")

    def plot_p910_classification(self, results: Dict[str, Dict],
                                  output_name: str = "si_ti_p910_classification.png",
                                  si_threshold: float = 35.0,
                                  ti_threshold: float = 18.0):
        """
        Plot standard ITU-T P.910 SI vs TI classification plot.

        Shows one point per video sequence using mean SI and TI values.
        Includes classification quadrants for content complexity.

        Args:
            results: Dictionary mapping video names to their SI/TI results
            output_name: Output filename
            si_threshold: SI threshold for classification (default: 35.0)
            ti_threshold: TI threshold for classification (default: 18.0)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Plot one point per video using MEAN values (standard P.910)
        for idx, (video_name, data) in enumerate(results.items()):
            ax.scatter(data['si_mean'], data['ti_mean'],
                      s=200, color=colors[idx], label=video_name,
                      alpha=0.8, edgecolors='black', linewidths=1.5)

        # Add classification threshold lines
        ax.axvline(x=si_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=ti_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Add region labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        label_alpha = 0.5
        label_size = 10
        label_weight = 'bold'

        # Bottom-left: Low SI, Low TI
        ax.text(xlim[0] + (si_threshold - xlim[0]) * 0.5,
                ylim[0] + (ti_threshold - ylim[0]) * 0.5,
                'Simple\nStatic', ha='center', va='center',
                alpha=label_alpha, fontsize=label_size, fontweight=label_weight,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Bottom-right: High SI, Low TI
        ax.text(si_threshold + (xlim[1] - si_threshold) * 0.5,
                ylim[0] + (ti_threshold - ylim[0]) * 0.5,
                'Complex\nStatic', ha='center', va='center',
                alpha=label_alpha, fontsize=label_size, fontweight=label_weight,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # Top-left: Low SI, High TI
        ax.text(xlim[0] + (si_threshold - xlim[0]) * 0.5,
                ti_threshold + (ylim[1] - ti_threshold) * 0.5,
                'Simple\nDynamic', ha='center', va='center',
                alpha=label_alpha, fontsize=label_size, fontweight=label_weight,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        # Top-right: High SI, High TI
        ax.text(si_threshold + (xlim[1] - si_threshold) * 0.5,
                ti_threshold + (ylim[1] - ti_threshold) * 0.5,
                'Complex\nDynamic', ha='center', va='center',
                alpha=label_alpha, fontsize=label_size, fontweight=label_weight,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        ax.set_xlabel('Mean Spatial Information (SI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Temporal Information (TI)', fontsize=12, fontweight='bold')
        ax.set_title('ITU-T P.910 Content Classification', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved P.910 classification plot: {output_path}")

    def plot_histograms(self, results: Dict[str, Dict], output_name: str = "si_ti_histograms.png"):
        """
        Plot histograms of SI and TI distributions.

        Args:
            results: Dictionary mapping video names to their SI/TI results
            output_name: Output filename
        """
        num_videos = len(results)
        fig, axes = plt.subplots(num_videos, 2, figsize=(14, 4 * num_videos))

        if num_videos == 1:
            axes = axes.reshape(1, -1)

        for idx, (video_name, data) in enumerate(results.items()):
            si_values = data['si_values']
            ti_values = data['ti_values']

            # SI histogram
            axes[idx, 0].hist(si_values, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
            axes[idx, 0].set_title(f'{video_name} - SI Distribution', fontsize=12, fontweight='bold')
            axes[idx, 0].set_xlabel('SI Value', fontsize=10)
            axes[idx, 0].set_ylabel('Frequency', fontsize=10)
            axes[idx, 0].axvline(x=data['si_mean'], color='r', linestyle='--',
                                linewidth=2, label=f"Mean: {data['si_mean']:.2f}")
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3, axis='y')

            # TI histogram
            axes[idx, 1].hist(ti_values, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
            axes[idx, 1].set_title(f'{video_name} - TI Distribution', fontsize=12, fontweight='bold')
            axes[idx, 1].set_xlabel('TI Value', fontsize=10)
            axes[idx, 1].set_ylabel('Frequency', fontsize=10)
            axes[idx, 1].axvline(x=data['ti_mean'], color='r', linestyle='--',
                                linewidth=2, label=f"Mean: {data['ti_mean']:.2f}")
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram plot: {output_path}")

    def plot_summary_statistics(self, results: Dict[str, Dict], output_name: str = "si_ti_summary.png"):
        """
        Plot summary statistics (mean, max) for all videos.

        Args:
            results: Dictionary mapping video names to their SI/TI results
            output_name: Output filename
        """
        video_names = list(results.keys())
        si_means = [results[v]['si_mean'] for v in video_names]
        ti_means = [results[v]['ti_mean'] for v in video_names]
        si_maxs = [results[v]['si_max'] for v in video_names]
        ti_maxs = [results[v]['ti_max'] for v in video_names]

        x = np.arange(len(video_names))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Mean values
        ax1.bar(x - width/2, si_means, width, label='SI Mean', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, ti_means, width, label='TI Mean', color='#A23B72', alpha=0.8)
        ax1.set_xlabel('Video', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
        ax1.set_title('Mean SI/TI Values', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(video_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Max values
        ax2.bar(x - width/2, si_maxs, width, label='SI Max', color='#2E86AB', alpha=0.8)
        ax2.bar(x + width/2, ti_maxs, width, label='TI Max', color='#A23B72', alpha=0.8)
        ax2.set_xlabel('Video', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Max Value', fontsize=12, fontweight='bold')
        ax2.set_title('Maximum SI/TI Values', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(video_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary statistics plot: {output_path}")

    def generate_all_plots(self, results: Dict[str, Dict]):
        """
        Generate all plot types.

        Args:
            results: Dictionary mapping video names to their SI/TI results
        """
        print("\nGenerating visualizations...")
        self.plot_time_series(results)
        self.plot_si_ti_scatter(results)
        self.plot_p910_classification(results)
        self.plot_histograms(results)
        self.plot_summary_statistics(results)
        print(f"\nAll plots saved to: {self.output_dir}")

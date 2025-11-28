# ================================================================================
# SECTION 1: Imports and Data Structures
# ================================================================================

from dataclasses import dataclass
from typing import Dict, List
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class MetricStats:
    """
    Holds evaluation metrics for a single objective quality metric.

    Training metrics (required):
        metric_name: Name of the objective metric
        pearson_r: Pearson correlation coefficient
        pearson_p: Pearson p-value
        spearman_r: Spearman rank correlation
        spearman_p: Spearman p-value
        rmse: Root mean square error (training)
        rmse_star: Epsilon-insensitive RMSE (training)
        outlier_ratio: Proportion of outliers (training)
        n_samples: Number of samples

    Cross-validation metrics (optional, default None):
        rmse_cv: RMSE from cross-validation
        rmse_star_cv: Epsilon-insensitive RMSE from CV
        outlier_ratio_cv: Outlier ratio from CV
        generalization_gap: rmse_cv - rmse (overfitting indicator)
    """
    # Training metrics (required)
    metric_name: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    rmse: float
    rmse_star: float
    outlier_ratio: float
    n_samples: int

    # Cross-validation metrics (optional, backward compatible)
    rmse_cv: float = None
    rmse_star_cv: float = None
    outlier_ratio_cv: float = None
    generalization_gap: float = None

# ================================================================================
# SECTION 2: Core Mapping Functions
# ================================================================================

def fit_third_order_mapping(x: np.ndarray, mos: np.ndarray,
                            mos_min: float = 1.0, mos_max: float = 5.0):
    """
    Third-order polynomial mapping as per ITU-T P.1401:
    MOS_hat = a0 + a1*x + a2*x^2 + a3*x^3

    Returns callable f(x_new) and coefficients.
    """
    # Create polynomial features: [1, x, x^2, x^3]
    X = np.column_stack([np.ones_like(x), x, x**2, x**3])

    # Fit using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, mos, rcond=None)
    a0, a1, a2, a3 = coeffs

    def f(x_new):
        x_new = np.asarray(x_new)
        y_hat = a0 + a1*x_new + a2*x_new**2 + a3*x_new**3
        # Clip to MOS range
        return np.clip(y_hat, mos_min, mos_max)

    return f, coeffs

# ================================================================================
# SECTION 3: Evaluation Functions - Training
# ================================================================================

def evaluate_metric_p1401(df: pd.DataFrame,
                          metric_col: str,
                          mos_col: str = "mos",
                          ci_col: str = "mos_ci95",
                          mos_min: float = 1.0,
                          mos_max: float = 5.0) -> MetricStats:
    """
    Evaluate objective quality metric according to ITU-T P.1401 framework.

    df must have columns: mos, mos_ci95, metric_col.

    Parameters:
        df: DataFrame with subjective MOS, CI95, and objective metric
        metric_col: Name of objective metric column
        mos_col: Name of MOS column (default: "mos")
        ci_col: Name of CI95 column (default: "mos_ci95")
        mos_min: Minimum MOS value (default: 1.0)
        mos_max: Maximum MOS value (default: 5.0)
    """
    mos = df[mos_col].values.astype(float)
    ci = df[ci_col].values.astype(float)
    x = df[metric_col].values.astype(float)

    # 1) Third-order polynomial mapping (P.1401 standard)
    f, coeffs = fit_third_order_mapping(x, mos, mos_min, mos_max)
    mos_pred = f(x)

    # 2) Errors
    err = mos - mos_pred
    n = len(mos)

    # RMSE per ITU-T P.1401 Equation 7-2: uses unbiased estimator (N-1)
    rmse = np.sqrt(np.sum(err**2) / (n - 1))

    # Epsilon-insensitive RMSE* per ITU-T P.1401 Equation 7-29:
    # discount errors within MOS CI95, account for degrees of freedom (d=4 for 3rd order polynomial)
    eps = ci
    eff_err = np.maximum(np.abs(err) - eps, 0.0)
    d = 4  # degrees of freedom for third-order polynomial mapping
    rmse_star = np.sqrt(np.sum(eff_err**2) / (n - d))

    # 3) Outlier ratio: |err| > CI95
    outliers = np.abs(err) > ci
    outlier_ratio = outliers.mean()

    # 4) Correlations between objective metric (x) and subjective MOS
    # NOTE: This is the correct P.1401 approach - correlate raw metric with MOS,
    # not predicted MOS with actual MOS (which would be artificially high)
    pearson_r, pearson_p = stats.pearsonr(x, mos)
    spearman_r, spearman_p = stats.spearmanr(x, mos)

    return MetricStats(
        metric_name=metric_col,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        rmse=rmse,
        rmse_star=rmse_star,
        outlier_ratio=outlier_ratio,
        n_samples=len(df)
    )

def evaluate_all_metrics(df: pd.DataFrame,
                         metric_cols: List[str]) -> Dict[str, MetricStats]:
    """
    DEPRECATED: Use evaluate_all_metrics_with_cv() for full P.1401 compliance.

    Evaluate all metrics (training only, no cross-validation).
    Kept for backward compatibility.
    """
    results = {}
    for m in metric_cols:
        results[m] = evaluate_metric_p1401(df, metric_col=m)
    return results

# ================================================================================
# SECTION 4: Evaluation Functions - Cross-Validation
# ================================================================================

def evaluate_metric_loocv(df: pd.DataFrame,
                          metric_col: str,
                          mos_col: str = "mos",
                          ci_col: str = "mos_ci95",
                          mos_min: float = 1.0,
                          mos_max: float = 5.0) -> Dict:
    """
    Perform Leave-One-Out Cross-Validation for P.1401 metric evaluation.

    Essential for generalization validation with small datasets (N < 30).
    Trains N polynomial models (each on N-1 samples) and predicts the
    held-out sample. Aggregates out-of-sample predictions to calculate
    cross-validated RMSE, RMSE*, and outlier ratio.

    Args:
        df: DataFrame with subjective and objective data
        metric_col: Name of objective metric column
        mos_col: Name of MOS column (default: "mos")
        ci_col: Name of CI95 column (default: "mos_ci95")
        mos_min: Minimum MOS value (default: 1.0)
        mos_max: Maximum MOS value (default: 5.0)

    Returns:
        Dictionary containing:
            - rmse_cv: Cross-validated RMSE
            - rmse_star_cv: Cross-validated RMSE*
            - outlier_ratio_cv: Cross-validated outlier ratio
            - predictions: Array of out-of-sample predictions (length N)
            - errors: Array of prediction errors (length N)
            - mos_actual: Actual MOS values for plotting
    """
    mos = df[mos_col].values.astype(float)
    ci = df[ci_col].values.astype(float)
    x = df[metric_col].values.astype(float)

    # Initialize arrays for predictions
    n = len(df)
    predictions = np.zeros(n)

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x):
        # Train on N-1 samples
        x_train = x[train_idx]
        mos_train = mos[train_idx]

        # Fit third-order polynomial
        f, _ = fit_third_order_mapping(x_train, mos_train, mos_min, mos_max)

        # Predict held-out sample
        predictions[test_idx] = f(x[test_idx])

    # Calculate CV metrics
    errors = mos - predictions
    n = len(mos)

    # RMSE per ITU-T P.1401 Equation 7-2: uses unbiased estimator (N-1)
    rmse_cv = np.sqrt(np.sum(errors**2) / (n - 1))

    # Epsilon-insensitive RMSE* per ITU-T P.1401 Equation 7-29:
    # account for degrees of freedom (d=4 for 3rd order polynomial)
    eff_err = np.maximum(np.abs(errors) - ci, 0.0)
    d = 4  # degrees of freedom for third-order polynomial mapping
    rmse_star_cv = np.sqrt(np.sum(eff_err**2) / (n - d))

    # Outlier ratio: |error| > CI95
    outliers = np.abs(errors) > ci
    outlier_ratio_cv = outliers.mean()

    return {
        'rmse_cv': rmse_cv,
        'rmse_star_cv': rmse_star_cv,
        'outlier_ratio_cv': outlier_ratio_cv,
        'predictions': predictions,
        'errors': errors,
        'mos_actual': mos
    }


def evaluate_all_metrics_with_cv(df: pd.DataFrame,
                                  metric_cols: List[str]) -> Dict[str, MetricStats]:
    """
    Evaluate all metrics with both training and LOOCV validation.

    For each metric:
    1. Run standard P.1401 evaluation (training on all data)
    2. Run LOOCV evaluation (out-of-sample predictions)
    3. Combine into enhanced MetricStats object
    4. Calculate generalization gap (overfitting indicator)

    Args:
        df: DataFrame with subjective and objective data
        metric_cols: List of objective metric column names

    Returns:
        Dict[str, MetricStats]: Enhanced MetricStats with CV metrics
    """
    results = {}
    for metric in metric_cols:
        # Training evaluation
        stats = evaluate_metric_p1401(df, metric_col=metric)

        # CV evaluation
        cv_result = evaluate_metric_loocv(df, metric_col=metric)

        # Combine into enhanced MetricStats
        stats.rmse_cv = cv_result['rmse_cv']
        stats.rmse_star_cv = cv_result['rmse_star_cv']
        stats.outlier_ratio_cv = cv_result['outlier_ratio_cv']
        stats.generalization_gap = cv_result['rmse_cv'] - stats.rmse

        results[metric] = stats

    return results

# ================================================================================
# SECTION 5: Data Loading
# ================================================================================

def load_and_merge_data(mos_path: str, metric_paths: List[str]) -> tuple[pd.DataFrame, List[str]]:
    """
    Load MOS CSV and metric CSVs, merge on filename.

    Returns:
        Merged DataFrame and list of metric column names
    """
    # Load MOS data
    mos_df = pd.read_csv(mos_path)

    # Normalize column names to lowercase for consistency
    mos_df.columns = mos_df.columns.str.lower()

    # Rename columns to match expected format
    column_mapping = {
        'mos': 'mos',
        'ci95': 'mos_ci95'
    }
    mos_df = mos_df.rename(columns=column_mapping)

    # Keep only filename, mos, mos_ci95
    mos_df = mos_df[['filename', 'mos', 'mos_ci95']]

    # Merge with each metric CSV
    all_metric_cols = []
    merged_df = mos_df.copy()

    for metric_path in metric_paths:
        metric_df = pd.read_csv(metric_path)
        metric_df.columns = metric_df.columns.str.lower()

        # Auto-detect metric columns (all except filename)
        metric_cols = [col for col in metric_df.columns if col != 'filename']
        all_metric_cols.extend(metric_cols)

        # Merge on filename
        merged_df = merged_df.merge(metric_df, on='filename', how='inner')

    return merged_df, all_metric_cols

# ================================================================================
# SECTION 6: Visualization - Essential
# ================================================================================

def plot_metric_analysis(df: pd.DataFrame, metric_col: str, stats: MetricStats, output_dir: Path):
    """
    Create scatter plot with fitted third-order polynomial per P.1401.
    """
    mos = df['mos'].values
    x = df[metric_col].values
    ci = df['mos_ci95'].values

    # Fit mapping
    f, coeffs = fit_third_order_mapping(x, mos)

    # Generate smooth curve for plotting
    x_sorted = np.sort(x)
    y_pred = f(x_sorted)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with error bars
    ax.errorbar(x, mos, yerr=ci, fmt='o', alpha=0.6, capsize=5,
                label='MOS with CI95')

    # Fitted polynomial curve
    ax.plot(x_sorted, y_pred, 'r-', linewidth=2,
            label=f'Third-order polynomial fit')

    # Labels and title
    ax.set_xlabel(f'{metric_col}', fontsize=12)
    ax.set_ylabel('MOS', fontsize=12)
    ax.set_title(f'P.1401 Analysis: {metric_col}\n'
                 f'Pearson r={stats.pearson_r:.3f} (p={stats.pearson_p:.4f}), '
                 f'Spearman ρ={stats.spearman_r:.3f} (p={stats.spearman_p:.4f})\n'
                 f'RMSE={stats.rmse:.3f}, RMSE*={stats.rmse_star:.3f}, '
                 f'Outlier ratio={stats.outlier_ratio:.2%}',
                 fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save plot
    plot_path = output_dir / f'{metric_col}_p1401.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  Saved plot: {plot_path}")

# ================================================================================
# SECTION 7: Visualization - Comparisons
# ================================================================================

def plot_metrics_comparison(results: Dict[str, MetricStats], output_dir: Path):
    """
    Create bar chart comparing correlation coefficients across all metrics.
    """
    # Extract data
    metrics = []
    pearson_vals = []
    spearman_vals = []

    for metric, stats in results.items():
        # Skip metrics with NaN correlations
        if not np.isnan(stats.pearson_r):
            metrics.append(metric)
            pearson_vals.append(stats.pearson_r)
            spearman_vals.append(stats.spearman_r)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(metrics))
    width = 0.35

    # Color-code by strength
    def get_colors(vals):
        colors = []
        for v in vals:
            abs_v = abs(v)
            if abs_v < 0.3:
                colors.append('#e74c3c')  # Red - weak
            elif abs_v < 0.5:
                colors.append('#f39c12')  # Orange - moderate
            elif abs_v < 0.7:
                colors.append('#27ae60')  # Green - strong
            else:
                colors.append('#16a085')  # Teal - very strong
        return colors

    # Pearson correlation
    bars1 = ax1.bar(x, pearson_vals, width, color=get_colors(pearson_vals))
    ax1.axhline(y=0.3, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Weak/Moderate')
    ax1.axhline(y=-0.3, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Moderate/Strong')
    ax1.axhline(y=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.9, label='Strong/Very Strong')
    ax1.axhline(y=-0.7, color='gray', linestyle='--', linewidth=1, alpha=0.9)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_ylabel('Pearson r', fontsize=12)
    ax1.set_title('Metric Performance Comparison - Pearson Correlation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_ylim(-1, 1)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Spearman correlation
    bars2 = ax2.bar(x, spearman_vals, width, color=get_colors(spearman_vals))
    ax2.axhline(y=0.3, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-0.3, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.9)
    ax2.axhline(y=-0.7, color='gray', linestyle='--', linewidth=1, alpha=0.9)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Spearman ρ', fontsize=12)
    ax2.set_title('Metric Performance Comparison - Spearman Correlation', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45, ha='right')
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'metrics_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison plot: {plot_path}")

# ================================================================================
# SECTION 9: Analysis and Reporting
# ================================================================================

def analyze_results(results: Dict[str, MetricStats], df: pd.DataFrame) -> Dict:
    """
    Automated analysis of P.1401 results with warnings and recommendations.
    """
    analysis = {
        'warnings': [],
        'valid_metrics': [],
        'moderate_metrics': [],
        'weak_metrics': [],
        'invalid_metrics': [],
        'best_metric': None,
        'sample_size': len(df),
        'mos_range': (df['mos'].min(), df['mos'].max())
    }

    # Check sample size
    n = len(df)
    if n < 20:
        analysis['warnings'].append(f"⚠️  CRITICAL: Sample size N={n} is too small (recommend N≥30 for reliable validation)")
    elif n < 30:
        analysis['warnings'].append(f"⚠️  WARNING: Sample size N={n} is below recommended minimum (N≥30)")

    # Check MOS range
    mos_min, mos_max = df['mos'].min(), df['mos'].max()
    mos_span = mos_max - mos_min
    if mos_span < 2.0:
        analysis['warnings'].append(f"⚠️  WARNING: MOS range [{mos_min:.1f}, {mos_max:.1f}] is narrow (span={mos_span:.1f}). Recommend full range [1.0, 5.0]")

    # Check for overfitting
    all_rmse_star_zero = all(stats.rmse_star == 0.0 for stats in results.values())
    if all_rmse_star_zero and n < 20:
        analysis['warnings'].append("⚠️  WARNING: All RMSE*=0 with small sample size suggests overfitting")

    # Check for generalization gap issues (NEW)
    for metric, stats in results.items():
        if stats.generalization_gap is None:
            continue

        # Large positive gap = overfitting
        if stats.generalization_gap > 0.2:
            analysis['warnings'].append(
                f"⚠️  OVERFITTING: {metric} shows large generalization gap "
                f"({stats.generalization_gap:.3f}). Model may not generalize well."
            )

        # Very large gap = severe overfitting
        if stats.generalization_gap > 0.5:
            analysis['warnings'].append(
                f"🚨 CRITICAL OVERFITTING: {metric} gap={stats.generalization_gap:.3f} "
                f"suggests severe overfitting. Reconsider model or collect more data."
            )

        # Negative gap = suspicious (possible data leakage)
        if stats.generalization_gap < -0.05:
            analysis['warnings'].append(
                f"🚨 SUSPICIOUS: {metric} shows negative generalization gap "
                f"({stats.generalization_gap:.3f}). Check for data leakage or errors."
            )

    # Categorize metrics
    best_r = -1
    best_metric_name = None

    for metric, stats in results.items():
        # Skip NaN correlations
        if np.isnan(stats.pearson_r):
            analysis['invalid_metrics'].append((metric, 'Constant values (no variance)'))
            continue

        # Use absolute value for strength, but keep sign for interpretation
        abs_r = abs(stats.pearson_r)
        abs_rho = abs(stats.spearman_r)
        max_corr = max(abs_r, abs_rho)

        # Track best metric
        if max_corr > best_r:
            best_r = max_corr
            best_metric_name = metric

        # Categorize by strength and significance
        p_sig = stats.pearson_p < 0.05 or stats.spearman_p < 0.05

        if max_corr >= 0.5:
            if p_sig:
                analysis['valid_metrics'].append((metric, max_corr, 'Strong & significant'))
            else:
                analysis['moderate_metrics'].append((metric, max_corr, 'Strong but not significant (need more data)'))
        elif max_corr >= 0.3:
            if p_sig:
                analysis['moderate_metrics'].append((metric, max_corr, 'Moderate & significant'))
            else:
                analysis['moderate_metrics'].append((metric, max_corr, 'Moderate but not significant'))
        else:
            analysis['weak_metrics'].append((metric, max_corr, 'Weak correlation'))

    analysis['best_metric'] = (best_metric_name, best_r)

    return analysis

def print_summary_table(results: Dict[str, MetricStats]):
    """
    Print summary table of all metrics to stdout (with CV metrics if available).
    """
    print("\n" + "="*130)
    print("ITU-T P.1401 Framework - Summary Results (with Cross-Validation)")
    print("="*130)
    print(f"{'Metric':<25} {'Pearson':<10} {'Spearman':<10} {'RMSE':<10} "
          f"{'RMSE_CV':<10} {'RMSE*':<10} {'RMSE*_CV':<10} "
          f"{'Gap':<10} {'Outliers':<10} {'N':<6}")
    print("-"*130)

    for metric, stats in results.items():
        rmse_cv_str = f"{stats.rmse_cv:.4f}" if stats.rmse_cv is not None else "N/A"
        rmse_star_cv_str = f"{stats.rmse_star_cv:.4f}" if stats.rmse_star_cv is not None else "N/A"
        gap_str = f"{stats.generalization_gap:+.4f}" if stats.generalization_gap is not None else "N/A"

        print(f"{metric:<25} "
              f"{stats.pearson_r:<10.4f} {stats.spearman_r:<10.4f} "
              f"{stats.rmse:<10.4f} {rmse_cv_str:<10} "
              f"{stats.rmse_star:<10.4f} {rmse_star_cv_str:<10} "
              f"{gap_str:<10} {stats.outlier_ratio:<10.2%} {stats.n_samples:<6}")

    print("="*130 + "\n")

# ================================================================================
# SECTION 10: Export Functions
# ================================================================================

def export_results_csv(results: Dict[str, MetricStats], output_dir: Path):
    """
    Export summary results to CSV file (original format, backward compatibility).
    """
    data = []
    for metric, stats in results.items():
        data.append({
            'Metric': metric,
            'Pearson_r': stats.pearson_r,
            'Pearson_p': stats.pearson_p,
            'Spearman_rho': stats.spearman_r,
            'Spearman_p': stats.spearman_p,
            'RMSE': stats.rmse,
            'RMSE_star': stats.rmse_star,
            'Outlier_Ratio': stats.outlier_ratio,
            'N_Samples': stats.n_samples
        })

    df_export = pd.DataFrame(data)
    csv_path = output_dir / 'p1401_summary.csv'
    df_export.to_csv(csv_path, index=False)
    print(f"  Saved summary CSV: {csv_path}")


def export_results_csv_enhanced(results: Dict[str, MetricStats], output_dir: Path):
    """
    Enhanced CSV export with training + CV metrics.

    Exports two files:
    1. p1401_summary_enhanced.csv (new format with CV columns)
    2. p1401_summary.csv (original format, backward compatibility)
    """
    data = []
    for metric, stats in results.items():
        data.append({
            'Metric': metric,
            'Pearson_r': stats.pearson_r,
            'Pearson_p': stats.pearson_p,
            'Spearman_rho': stats.spearman_r,
            'Spearman_p': stats.spearman_p,
            'RMSE': stats.rmse,
            'RMSE_CV': stats.rmse_cv if stats.rmse_cv is not None else np.nan,
            'RMSE_star': stats.rmse_star,
            'RMSE_star_CV': stats.rmse_star_cv if stats.rmse_star_cv is not None else np.nan,
            'Generalization_Gap': stats.generalization_gap if stats.generalization_gap is not None else np.nan,
            'Outlier_Ratio': stats.outlier_ratio,
            'Outlier_Ratio_CV': stats.outlier_ratio_cv if stats.outlier_ratio_cv is not None else np.nan,
            'N_Samples': stats.n_samples
        })

    # Export enhanced format
    df_export = pd.DataFrame(data)
    csv_path_enhanced = output_dir / 'p1401_summary_enhanced.csv'
    df_export.to_csv(csv_path_enhanced, index=False)
    print(f"  Saved enhanced summary CSV: {csv_path_enhanced}")

    # Also export original format for backward compatibility
    export_results_csv(results, output_dir)

def plot_top_metrics_overlay(df: pd.DataFrame, results: Dict[str, MetricStats],
                              top_metrics: List[str], output_dir: Path):
    """
    Create overlay plot showing top 3 metrics for comparison.
    """
    n_metrics = min(3, len(top_metrics))
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(top_metrics[:n_metrics]):
        ax = axes[idx]
        stats = results[metric]

        mos = df['mos'].values
        x = df[metric].values
        ci = df['mos_ci95'].values

        # Fit mapping
        f, coeffs = fit_third_order_mapping(x, mos)
        x_sorted = np.sort(x)
        y_pred = f(x_sorted)

        # Plot
        ax.errorbar(x, mos, yerr=ci, fmt='o', alpha=0.6, capsize=5,
                   markersize=8, label='MOS ± CI95')
        ax.plot(x_sorted, y_pred, 'r-', linewidth=2.5, label='P.1401 fit')

        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_ylabel('MOS', fontsize=11)
        ax.set_title(f'{metric}\nr={stats.pearson_r:.3f}, ρ={stats.spearman_r:.3f}\nRMSE={stats.rmse:.3f}',
                    fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Top Performing Metrics - P.1401 Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = output_dir / 'top_metrics_overlay.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved top metrics overlay: {plot_path}")

def print_analysis_report(analysis: Dict):
    """
    Print automated analysis report with warnings and recommendations.
    """
    print("="*100)
    print("AUTOMATED ANALYSIS REPORT")
    print("="*100)

    # Print warnings first
    if analysis['warnings']:
        print("\n🚨 STATISTICAL WARNINGS:")
        for warning in analysis['warnings']:
            print(f"   {warning}")
        print()

    # Dataset info
    mos_min, mos_max = analysis['mos_range']
    print(f"📊 Dataset: N={analysis['sample_size']} samples, MOS range=[{mos_min:.2f}, {mos_max:.2f}]\n")

    # Valid metrics
    if analysis['valid_metrics']:
        print("✅ VALID METRICS (Strong correlation):")
        for metric, corr, note in sorted(analysis['valid_metrics'], key=lambda x: x[1], reverse=True):
            print(f"   • {metric:<30} |r|={corr:.3f} - {note}")
        print()
    else:
        print("❌ NO VALID METRICS FOUND (none achieve |r|≥0.5 with significance)\n")

    # Moderate metrics
    if analysis['moderate_metrics']:
        print("⚠️  MODERATE METRICS (Acceptable but need validation):")
        for metric, corr, note in sorted(analysis['moderate_metrics'], key=lambda x: x[1], reverse=True):
            print(f"   • {metric:<30} |r|={corr:.3f} - {note}")
        print()

    # Weak metrics
    if analysis['weak_metrics']:
        print("❌ WEAK METRICS (Poor predictors):")
        for metric, corr, note in sorted(analysis['weak_metrics'], key=lambda x: x[1], reverse=True):
            print(f"   • {metric:<30} |r|={corr:.3f} - {note}")
        print()

    # Invalid metrics
    if analysis['invalid_metrics']:
        print("🚫 INVALID METRICS (Cannot be evaluated):")
        for metric, reason in analysis['invalid_metrics']:
            print(f"   • {metric:<30} - {reason}")
        print()

    # Best metric
    if analysis['best_metric'][0]:
        best_name, best_r = analysis['best_metric']
        print(f"🏆 BEST PERFORMER: {best_name} (|r|={best_r:.3f})")
        print()

    # NEW: Cross-Validation Summary
    # Check if we have CV results by looking for any metric with rmse_cv
    cv_metrics = []
    if 'results' in analysis:  # We'll pass results through analysis
        for metric_name, (metric, stats) in analysis['results'].items():
            if hasattr(stats, 'rmse_cv') and stats.rmse_cv is not None:
                cv_metrics.append((metric, stats.rmse_cv, stats.generalization_gap))

    if cv_metrics:
        cv_metrics.sort(key=lambda x: x[1])  # Sort by CV RMSE
        print("📊 CROSS-VALIDATION SUMMARY (LOOCV):")
        print(f"   Validated metrics: {len(cv_metrics)}")
        print("   Top 5 metrics by CV RMSE:")
        for metric, rmse_cv, gap in cv_metrics[:5]:
            gap_str = f"(gap={gap:+.3f})" if gap is not None else ""
            status = ""
            if gap is not None:
                if gap > 0.2:
                    status = " ⚠️ OVERFITTING"
                elif gap < -0.05:
                    status = " 🚨 SUSPICIOUS"
            print(f"   • {metric:<30} RMSE_CV={rmse_cv:.3f} {gap_str}{status}")
        print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if analysis['sample_size'] < 30:
        print(f"   1. Increase sample size to N≥30 (currently N={analysis['sample_size']})")
    mos_span = analysis['mos_range'][1] - analysis['mos_range'][0]
    if mos_span < 3.0:
        print(f"   2. Expand MOS range to cover [1.0, 5.0] (currently [{mos_min:.1f}, {mos_max:.1f}])")
    if not analysis['valid_metrics'] and analysis['moderate_metrics']:
        print("   3. Re-evaluate with more data - moderate correlations may become significant")
    if len(analysis['weak_metrics']) > len(analysis['valid_metrics']) + len(analysis['moderate_metrics']):
        print("   4. Consider removing weak metrics or feature engineering new ones")

    print("="*100 + "\n")

def plot_prediction_scatter_combined(df: pd.DataFrame, results: Dict[str, MetricStats],
                                      output_dir: Path, max_metrics: int = 10):
    """
    Create combined scatter plot: Predicted MOS vs Actual MOS for all metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Colors for different metrics
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', 'p']

    mos_actual = df['mos'].values

    # Plot each metric
    plotted_count = 0
    for idx, (metric, stats) in enumerate(results.items()):
        # Skip NaN correlations and limit to max_metrics for readability
        if np.isnan(stats.pearson_r) or plotted_count >= max_metrics:
            continue

        # Get predictions
        x = df[metric].values
        f, coeffs = fit_third_order_mapping(x, mos_actual)
        mos_pred = f(x)

        # Plot with unique marker
        marker = markers[plotted_count % len(markers)]
        ax.scatter(mos_actual, mos_pred, c=[colors[idx]], marker=marker,
                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                  label=f'{metric} (r={stats.pearson_r:.2f})')
        plotted_count += 1

    # Perfect prediction line (diagonal)
    lims = [min(mos_actual.min(), mos_actual.min()) - 0.2,
            max(mos_actual.max(), mos_actual.max()) + 0.2]
    ax.plot(lims, lims, 'k-', linewidth=2, label='Perfect prediction', zorder=1)

    # ±0.5 MOS error bands
    ax.plot(lims, [lims[0] + 0.5, lims[1] + 0.5], 'k--', linewidth=1,
            alpha=0.5, label='±0.5 MOS error')
    ax.plot(lims, [lims[0] - 0.5, lims[1] - 0.5], 'k--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel('Actual MOS', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted MOS', fontsize=14, fontweight='bold')
    ax.set_title('Prediction Validation - All Metrics\n(P.1401 Third-Order Polynomial Mapping)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()
    plot_path = output_dir / 'prediction_validation_combined.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined prediction scatter: {plot_path}")

def plot_prediction_scatter_individual(df: pd.DataFrame, metric: str, stats: MetricStats,
                                       output_dir: Path):
    """
    Create individual prediction scatter plot for a single metric.
    """
    mos_actual = df['mos'].values
    ci = df['mos_ci95'].values
    x = df[metric].values

    # Get predictions
    f, coeffs = fit_third_order_mapping(x, mos_actual)
    mos_pred = f(x)

    # Calculate residuals
    residuals = mos_actual - mos_pred

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Prediction scatter
    ax1.errorbar(mos_actual, mos_pred, xerr=ci, fmt='o', markersize=8,
                alpha=0.7, capsize=5, color='steelblue', ecolor='gray')

    # Perfect prediction line
    lims = [min(mos_actual.min(), mos_pred.min()) - 0.3,
            max(mos_actual.max(), mos_pred.max()) + 0.3]
    ax1.plot(lims, lims, 'k-', linewidth=2, label='Perfect prediction')
    ax1.plot(lims, [lims[0] + 0.5, lims[1] + 0.5], 'k--', linewidth=1,
            alpha=0.5, label='±0.5 MOS')
    ax1.plot(lims, [lims[0] - 0.5, lims[1] - 0.5], 'k--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Actual MOS', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted MOS', fontsize=12, fontweight='bold')
    ax1.set_title(f'{metric}\nr={stats.pearson_r:.3f}, RMSE={stats.rmse:.3f}', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')

    # Right: Residual plot
    ax2.scatter(mos_pred, residuals, s=80, alpha=0.7, color='coral', edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='k', linewidth=2, label='Zero error')
    ax2.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Predicted MOS', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Residual Analysis\nRMSE*={stats.rmse_star:.3f}, Outliers={stats.outlier_ratio:.1%}',
                 fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Prediction Validation: {metric}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    plot_path = output_dir / f'{metric}_prediction_scatter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved prediction scatter: {plot_path}")

# ================================================================================
# SECTION 8: Visualization - Advanced CV & Analysis
# ================================================================================

def _create_scatter_with_fit(ax, x, mos, ci, metric_col, show_curve=True):
    """
    Internal helper: Create scatter plot with fitted polynomial.
    Reduces code duplication across multiple plotting functions.

    Args:
        ax: Matplotlib axes object
        x: Objective metric values
        mos: Subjective MOS values
        ci: CI95 values for error bars
        metric_col: Metric name for labels
        show_curve: Whether to plot polynomial fit curve (default: True)

    Returns:
        f: Mapping function
        coeffs: Polynomial coefficients
    """
    # Fit mapping
    f, coeffs = fit_third_order_mapping(x, mos)

    # Scatter with error bars
    ax.errorbar(x, mos, yerr=ci, fmt='o', alpha=0.6, capsize=5,
                label='MOS with CI95')

    # Fitted curve (optional)
    if show_curve:
        x_sorted = np.sort(x)
        y_pred = f(x_sorted)
        ax.plot(x_sorted, y_pred, 'r-', linewidth=2,
                label='Third-order polynomial fit')

    # Formatting
    ax.set_xlabel(f'{metric_col}', fontsize=12)
    ax.set_ylabel('MOS', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return f, coeffs


def plot_cv_scatter(df: pd.DataFrame,
                   metric: str,
                   stats: MetricStats,
                   cv_predictions: np.ndarray,
                   output_dir: Path):
    """
    Side-by-side scatter plots: Training vs. CV predictions.

    Layout: 1x2 subplots
    - Left: Predictions from model trained on all N samples
    - Right: LOOCV predictions (out-of-sample)

    Features: Perfect prediction line, ±0.5 error bands, CI95 bars
    """
    mos = df['mos'].values
    ci = df['mos_ci95'].values
    x = df[metric].values

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Training predictions
    f, _ = fit_third_order_mapping(x, mos)
    train_pred = f(x)

    ax1.errorbar(mos, train_pred, xerr=ci, fmt='o', markersize=8,
                alpha=0.7, capsize=5, color='steelblue', ecolor='gray')

    # Perfect prediction line
    lims = [min(mos.min(), train_pred.min()) - 0.3,
            max(mos.max(), train_pred.max()) + 0.3]
    ax1.plot(lims, lims, 'k-', linewidth=2, label='Perfect prediction')
    ax1.plot(lims, [lims[0] + 0.5, lims[1] + 0.5], 'k--', linewidth=1,
            alpha=0.5, label='±0.5 MOS')
    ax1.plot(lims, [lims[0] - 0.5, lims[1] - 0.5], 'k--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Actual MOS', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted MOS', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training: RMSE={stats.rmse:.3f}', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')

    # Right: CV predictions
    ax2.errorbar(mos, cv_predictions, xerr=ci, fmt='o', markersize=8,
                alpha=0.7, capsize=5, color='darkorange', ecolor='gray')

    lims2 = [min(mos.min(), cv_predictions.min()) - 0.3,
             max(mos.max(), cv_predictions.max()) + 0.3]
    ax2.plot(lims2, lims2, 'k-', linewidth=2, label='Perfect prediction')
    ax2.plot(lims2, [lims2[0] + 0.5, lims2[1] + 0.5], 'k--', linewidth=1,
            alpha=0.5, label='±0.5 MOS')
    ax2.plot(lims2, [lims2[0] - 0.5, lims2[1] - 0.5], 'k--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Actual MOS', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted MOS', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cross-Validation: RMSE={stats.rmse_cv:.3f}', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lims2)
    ax2.set_ylim(lims2)
    ax2.set_aspect('equal')

    plt.suptitle(f'{metric} - Training vs. Cross-Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / f'{metric}_cv_scatter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved CV scatter plot: {plot_path}")


def plot_monotonic_mapping_curve(df: pd.DataFrame,
                                 metric: str,
                                 stats: MetricStats,
                                 output_dir: Path):
    """
    Visualize third-order polynomial mapping transformation.
    Shows how raw objective metric values map to MOS scale.

    Features:
    - Smooth curve (1000 sample points)
    - Actual data points overlaid
    - Clipping regions highlighted (< 1.0, > 5.0)
    - Polynomial equation annotated
    """
    mos = df['mos'].values
    x = df[metric].values

    # Fit mapping
    f, coeffs = fit_third_order_mapping(x, mos)
    a0, a1, a2, a3 = coeffs

    # Generate smooth curve
    x_range = np.linspace(x.min() - 0.1*(x.max()-x.min()),
                          x.max() + 0.1*(x.max()-x.min()), 1000)
    y_mapped = f(x_range)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, y_mapped, 'b-', linewidth=2, label='Mapping curve')
    ax.scatter(x, mos, s=100, c='red', marker='o', edgecolors='black',
              linewidth=1, label='Actual data', zorder=5)

    # Highlight clipping zones
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.fill_between(x_range, 0, 1.0, alpha=0.1, color='red', label='Clipped region')
    ax.fill_between(x_range, 5.0, 6.0, alpha=0.1, color='red')

    # Annotate equation
    eq_text = f'MOS = {a0:.3f} + {a1:.3f}x + {a2:.3f}x² + {a3:.3f}x³'
    ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mapped MOS', fontsize=12, fontweight='bold')
    ax.set_title(f'Monotonic Mapping Curve: {metric}\n'
                f'r={stats.pearson_r:.3f}, RMSE={stats.rmse:.3f}',
                fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plot_path = output_dir / f'{metric}_monotonic_curve.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved monotonic curve: {plot_path}")


def plot_generalization_gap_analysis(results: Dict[str, MetricStats],
                                     output_dir: Path):
    """
    Grouped bar chart comparing training vs. CV RMSE for all metrics.
    Essential for detecting overfitting.

    Features:
    - Sorted by CV RMSE (best to worst)
    - Gap annotations for large differences (>0.2)
    - Reference line at RMSE=0.5
    - Color coding: Training=blue, CV=orange
    """
    # Extract data
    metrics = []
    train_rmse = []
    cv_rmse = []
    gaps = []

    for metric, stats in results.items():
        if stats.rmse_cv is not None:
            metrics.append(metric)
            train_rmse.append(stats.rmse)
            cv_rmse.append(stats.rmse_cv)
            gaps.append(stats.generalization_gap)

    if not metrics:
        print("  No CV metrics available for generalization gap analysis")
        return

    # Sort by CV RMSE
    sorted_idx = np.argsort(cv_rmse)
    metrics = [metrics[i] for i in sorted_idx]
    train_rmse = [train_rmse[i] for i in sorted_idx]
    cv_rmse = [cv_rmse[i] for i in sorted_idx]
    gaps = [gaps[i] for i in sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, train_rmse, width, label='Training RMSE', color='steelblue')
    ax.bar(x + width/2, cv_rmse, width, label='CV RMSE', color='darkorange')

    # Annotate large gaps
    for i, gap in enumerate(gaps):
        if gap > 0.2:
            ax.annotate(f'Gap: {gap:.2f}', xy=(i, cv_rmse[i]),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8, color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1,
              alpha=0.7, label='Acceptable threshold')

    ax.set_xlabel('Metrics (sorted by CV RMSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Generalization Gap Analysis: Training vs. Cross-Validation',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'generalization_gap_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved generalization gap analysis: {plot_path}")

# ================================================================================
# SECTION 11: Ranking Tables
# ================================================================================

def generate_ranking_table(results: Dict[str, MetricStats]) -> pd.DataFrame:
    """
    Generate comprehensive ranking table sorted by CV RMSE.

    Columns:
    - Rank (1, 2, 3, ...)
    - Metric name
    - Pearson_r, Spearman_rho
    - RMSE, RMSE_CV
    - RMSE_star, RMSE_star_CV
    - Generalization_Gap
    - Outlier_Ratio, Outlier_Ratio_CV
    - Performance_Category (Excellent/Good/Fair/Poor)

    Performance Categories:
    - Excellent: RMSE_CV < 0.3 AND |Pearson| > 0.7
    - Good: RMSE_CV < 0.5 AND |Pearson| > 0.5
    - Fair: RMSE_CV < 0.7 AND |Pearson| > 0.3
    - Poor: Otherwise
    """
    data = []
    for metric, stats in results.items():
        # Categorize performance
        if stats.rmse_cv is not None:
            if stats.rmse_cv < 0.3 and abs(stats.pearson_r) > 0.7:
                category = 'Excellent'
            elif stats.rmse_cv < 0.5 and abs(stats.pearson_r) > 0.5:
                category = 'Good'
            elif stats.rmse_cv < 0.7 and abs(stats.pearson_r) > 0.3:
                category = 'Fair'
            else:
                category = 'Poor'
        else:
            category = 'N/A'

        data.append({
            'Metric': metric,
            'Pearson_r': stats.pearson_r,
            'Spearman_rho': stats.spearman_r,
            'RMSE': stats.rmse,
            'RMSE_CV': stats.rmse_cv if stats.rmse_cv is not None else np.nan,
            'RMSE_star': stats.rmse_star,
            'RMSE_star_CV': stats.rmse_star_cv if stats.rmse_star_cv is not None else np.nan,
            'Generalization_Gap': stats.generalization_gap if stats.generalization_gap is not None else np.nan,
            'Outlier_Ratio': stats.outlier_ratio,
            'Outlier_Ratio_CV': stats.outlier_ratio_cv if stats.outlier_ratio_cv is not None else np.nan,
            'Performance_Category': category
        })

    df = pd.DataFrame(data)
    # Sort by RMSE_CV (ascending, NaN last)
    df = df.sort_values('RMSE_CV', na_position='last').reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df)+1))

    return df


def export_ranking_table_csv(ranking_df: pd.DataFrame,
                              output_dir: Path):
    """Export ranking table to CSV with formatted floats."""
    csv_path = output_dir / 'p1401_ranking_table.csv'
    ranking_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved ranking table: {csv_path}")


def plot_ranking_table_visual(ranking_df: pd.DataFrame,
                               output_dir: Path,
                               top_n: int = 10):
    """
    Create visual table figure with color-coded performance.

    Features:
    - Heat map coloring (green=good, red=poor)
    - Bold font for top 3 metrics
    - Alternating row colors
    - Performance category colors
    - Professional matplotlib.table styling
    """
    # Select top N rows
    display_df = ranking_df.head(top_n).copy()

    # Format floats for display
    for col in ['Pearson_r', 'Spearman_rho', 'RMSE', 'RMSE_CV', 'RMSE_star',
                'RMSE_star_CV', 'Generalization_Gap', 'Outlier_Ratio', 'Outlier_Ratio_CV']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 0.5*min(top_n, len(ranking_df)) + 1))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center')

    # Style cells
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code by performance category
    for i in range(len(display_df)):
        category = ranking_df.iloc[i]['Performance_Category']
        if category == 'Excellent':
            color = '#d4edda'  # Light green
        elif category == 'Good':
            color = '#d1ecf1'  # Light blue
        elif category == 'Fair':
            color = '#fff3cd'  # Light yellow
        else:
            color = '#f8d7da'  # Light red

        for j in range(len(display_df.columns)):
            cell = table[(i+1, j)]
            cell.set_facecolor(color)
            # Bold top 3
            if i < 3:
                cell.set_text_props(weight='bold')

    # Header styling
    for j in range(len(display_df.columns)):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(weight='bold', color='white')

    plt.title('Metric Ranking Table - Sorted by CV RMSE',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    plot_path = output_dir / 'ranking_table_visual.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Saved ranking table visual: {plot_path}")

# ================================================================================
# SECTION 12: Main Execution
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ITU-T P.1401 Framework Analysis: Evaluate objective quality metrics against MOS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python mapping.py --mos mos_results.csv --metrics tv-l1_results.csv cover_results.csv
  python mapping.py --mos mos_results.csv --metrics metric1.csv metric2.csv --output results/
        """
    )

    parser.add_argument('--mos', type=str, required=True,
                        help='Path to MOS CSV file (must contain: Filename, MOS, CI95)')
    parser.add_argument('--metrics', type=str, nargs='+', required=True,
                        help='Path(s) to metric CSV file(s) (must contain: Filename and metric columns)')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for plots and results (default: current directory)')

    args = parser.parse_args()

    # Validate input files
    mos_path = Path(args.mos)
    if not mos_path.exists():
        print(f"Error: MOS file not found: {args.mos}", file=sys.stderr)
        sys.exit(1)

    metric_paths = [Path(p) for p in args.metrics]
    for mp in metric_paths:
        if not mp.exists():
            print(f"Error: Metric file not found: {mp}", file=sys.stderr)
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("ITU-T P.1401 Framework Analysis")
    print("="*100)
    print(f"MOS file: {mos_path}")
    print(f"Metric files: {', '.join(str(p) for p in metric_paths)}")
    print(f"Output directory: {output_dir}")
    print("="*100 + "\n")

    # Load and merge data
    print("Loading and merging data...")
    df, metric_cols = load_and_merge_data(str(mos_path), [str(p) for p in metric_paths])
    print(f"  Loaded {len(df)} samples")
    print(f"  Detected {len(metric_cols)} metrics: {', '.join(metric_cols)}\n")

    # Evaluate all metrics with cross-validation
    print("Evaluating metrics using P.1401 framework with LOOCV...")
    results = evaluate_all_metrics_with_cv(df, metric_cols)

    # Store CV predictions for visualization
    cv_predictions_dict = {}
    for metric in metric_cols:
        cv_result = evaluate_metric_loocv(df, metric)
        cv_predictions_dict[metric] = cv_result['predictions']

    print(f"  Completed evaluation for {len(results)} metrics")
    print(f"  Training + cross-validation metrics calculated\n")

    # Analyze results
    print("Analyzing results...")
    analysis = analyze_results(results, df)
    # Pass results to analysis for CV summary in report
    analysis['results'] = {m: (m, stats) for m, stats in results.items()}
    print(f"  Identified {len(analysis['valid_metrics'])} valid, "
          f"{len(analysis['moderate_metrics'])} moderate, "
          f"{len(analysis['weak_metrics'])} weak metrics\n")

    # Generate individual metric plots
    print("Generating individual metric plots...")
    for metric_col in metric_cols:
        plot_metric_analysis(df, metric_col, results[metric_col], output_dir)
    print()

    # Generate comparison plots
    print("Generating comparison plots...")
    plot_metrics_comparison(results, output_dir)

    # Generate top-3 overlay plot
    top_metrics = []
    # Combine valid and moderate metrics, sorted by correlation
    all_ranked = (analysis['valid_metrics'] + analysis['moderate_metrics'])
    if all_ranked:
        top_metrics = [m[0] for m in sorted(all_ranked, key=lambda x: x[1], reverse=True)[:3]]
        plot_top_metrics_overlay(df, results, top_metrics, output_dir)
    print()

    # NEW: Generate CV scatter plots for top metrics
    print("Generating cross-validation scatter plots...")
    if top_metrics:
        for metric in top_metrics:
            plot_cv_scatter(df, metric, results[metric],
                          cv_predictions_dict[metric], output_dir)
    print()

    # NEW: Generate monotonic mapping curves for top metrics
    print("Generating monotonic mapping curves...")
    if top_metrics:
        for metric in top_metrics:
            plot_monotonic_mapping_curve(df, metric, results[metric], output_dir)
    print()

    # NEW: Generate generalization gap analysis
    print("Generating generalization gap analysis...")
    plot_generalization_gap_analysis(results, output_dir)
    print()

    # Generate prediction validation scatter plots
    print("Generating prediction validation plots...")
    plot_prediction_scatter_combined(df, results, output_dir)

    # Generate individual prediction scatter plots for top metrics
    if top_metrics:
        for metric in top_metrics:
            plot_prediction_scatter_individual(df, metric, results[metric], output_dir)
    print()

    # NEW: Generate ranking table
    print("Generating ranking table...")
    ranking_df = generate_ranking_table(results)
    export_ranking_table_csv(ranking_df, output_dir)
    plot_ranking_table_visual(ranking_df, output_dir)
    print()

    # Export CSV (enhanced version)
    print("Exporting results...")
    export_results_csv_enhanced(results, output_dir)
    print()

    # Print summary table
    print_summary_table(results)

    # Print analysis report
    print_analysis_report(analysis)

    print(f"\n✅ Full P.1401 compliance analysis complete.")
    print(f"   All results saved to: {output_dir.resolve()}\n")
    print(f"📊 Key Outputs:")
    print(f"   CSV Files:")
    print(f"   - p1401_summary_enhanced.csv (training + CV metrics)")
    print(f"   - p1401_ranking_table.csv (sorted by CV RMSE)")
    print(f"   - p1401_summary.csv (original format)")
    print(f"\n   Visualization Files:")
    print(f"   - ranking_table_visual.png")
    print(f"   - generalization_gap_analysis.png")
    if top_metrics:
        print(f"   - {{metric}}_cv_scatter.png (top {len(top_metrics)} metrics)")
        print(f"   - {{metric}}_monotonic_curve.png (top {len(top_metrics)} metrics)")
    print(f"   - metrics_comparison.png")
    print(f"   - {{metric}}_p1401.png ({len(metric_cols)} metrics)")
    print(f"\n   See README.md for interpretation guide.")

if __name__ == "__main__":
    main()

# ITU-T P.1401 Quality Metric Assessment

> Python implementation of ITU-T Recommendation P.1401 for statistical evaluation and comparison of objective quality prediction models.

## 📋 Overview

This framework validates how well **objective video quality metrics** (VMAF, PSNR, SSIM, etc.) predict **subjective Mean Opinion Scores (MOS)** from user studies. It implements the international standard methodology for:

- Non-linear polynomial regression mapping
- Correlation analysis (Pearson, Spearman)
- Cross-validation and overfitting detection
- Statistical performance quantification

**Why it exists:** Standardized validation of perceptual quality models against human judgment per ITU-T guidelines.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Third-order polynomial mapping** | Non-linear regression from objective metrics to MOS scale (ITU-T P.1401 Eq. 7-1) |
| **Pearson correlation** | Linear relationship strength |
| **Spearman correlation** | Monotonic relationship strength (rank-based) |
| **RMSE** | Training prediction error |
| **RMSE*** | Epsilon-insensitive error (excludes CI95 range) |
| **Outlier ratio** | Percentage of predictions outside confidence intervals |
| **LOOCV** | Leave-One-Out Cross-Validation for small datasets (N < 30) |
| **Generalization gap** | Overfitting detection (RMSE_CV - RMSE_train) |
| **Automated ranking** | Sorts metrics by cross-validated performance |
| **Visualization suite** | Scatter plots, mapping curves, residual analysis |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Step 1:** Compute MOS from raw votes
```bash
python src/ci95.py -i data/votes/ -o results/mos/mos_results.csv
```

**Step 2:** Evaluate metrics against MOS
```bash
python src/mapping.py \
  --mos results/mos/mos_results.csv \
  --metrics cover_results.csv tv-l1_results.csv \
  --output results/p1401/
```

> **Note:** `--metrics` accepts **n result files** (e.g., `metric1.csv metric2.csv ... metricN.csv`). Open for extension with additional objective quality metrics.

**Step 3:** Check results
```bash
cat results/p1401/p1401_ranking_table.csv
```

## 🏗️ Architecture

### High-Level Flow

```
Raw Votes (CSV)
      │
      ▼
   ci95.py ──► MOS + CI95
      │
      ▼
mos_results.csv ──┬──► cover_results.csv
                  │
                  ├──► tv-l1_results.csv
                  │
                  └──► ... (n metrics)
                          │
                          ▼
                      mapping.py
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Training        LOOCV           Plots
      Metrics        Metrics          (PNG)
          │               │               │
          └───────────────┴───────────────┘
                          ▼
              ranking_table.csv
              summary_enhanced.csv
```

### Low-Level Components

**`mapping.py` execution flow:**
```
evaluate_all_metrics_with_cv()
│
├─► evaluate_metric_p1401()
│   ├─ fit_third_order_mapping()    → MOS_hat = a₀ + a₁x + a₂x² + a₃x³
│   ├─ pearsonr(x, MOS)             → Linear correlation
│   ├─ spearmanr(x, MOS)            → Rank correlation
│   ├─ RMSE = √(Σe²/(N-1))          → Training error
│   └─ RMSE* = √(Σmax(|e|-CI,0)²/(N-4))
│
├─► evaluate_metric_loocv()
│   └─ For each sample:
│       ├─ Train on N-1 samples
│       ├─ Predict held-out sample
│       └─ Compute RMSE_CV, Gap
│
├─► analyze_results()
│   ├─ Validate sample size (N ≥ 30)
│   ├─ Detect overfitting (gap > 0.2)
│   └─ Categorize by strength
│
└─► Visualization & Export
    ├─ plot_cv_scatter()
    ├─ generate_ranking_table()
    └─ export_results_csv_enhanced()
```

**`ci95.py` execution flow:**
```
process_csv_folder(votes_dir)
│
├─► Load all CSV files (Filename, Rating)
│
└─► For each video:
    └─ compute_mos_ci_from_votes()
       ├─ MOS = mean(votes)
       ├─ STD = std(votes, ddof=1)
       ├─ t = t_dist(0.975, df=N-1)
       └─ CI95 = t × STD / √N
            │
            ▼
       mos_results.csv
```

## 🛠️ Technologies

### Core Libraries

| Library | Purpose | Key Functions |
|---------|---------|---------------|
| **NumPy** | Numerical operations | `np.linalg.lstsq` (polynomial fitting), array operations |
| **Pandas** | Data manipulation | CSV I/O, DataFrame merging |
| **SciPy** | Statistical functions | `pearsonr`, `spearmanr`, `stats.t` (CI95) |
| **scikit-learn** | Machine learning | `LeaveOneOut` (cross-validation) |
| **Matplotlib** | Visualization | Scatter plots, bar charts, PNG export |

### Theoretical Concepts

#### 📐 Third-Order Polynomial Regression

**Formula:**
```
MOS_hat = a₀ + a₁x + a₂x² + a₃x³
```

**Purpose:** Maps objective metric values `x` to the subjective MOS scale `[1.0, 5.0]`.

**Why third-order?** Human perception is non-linear. A video with PSNR=30 may not be perceived as twice as good as PSNR=15. The polynomial captures:
- **Linear term** (a₁x): Base relationship
- **Quadratic term** (a₂x²): Saturation effects (diminishing returns at high quality)
- **Cubic term** (a₃x³): Fine-grained perceptual curvature

**Fitting process:** Solve via least-squares: `Xa = y` where `X = [1, x, x², x³]` and minimize `||MOS - MOS_hat||²`.

**Clipping:** Predictions outside `[1.0, 5.0]` are clipped to valid MOS range.

**Interpretation:** A good fit means the metric captures perceptual quality variations across the full scale.

---

#### 📊 Pearson Correlation (r)

**Formula:**
```
r = cov(x, MOS) / (σ_x × σ_MOS)
    = Σ((x - x̄)(MOS - MOS̄)) / √(Σ(x - x̄)² × Σ(MOS - MOS̄)²)
```

**What it measures:** Linear relationship strength between raw objective metric and subjective MOS.

**Range:** `[-1, 1]`
- `r = 1`: Perfect positive linear relationship
- `r = 0`: No linear relationship
- `r = -1`: Perfect negative linear relationship

**Interpretation guide:**
- `|r| > 0.9`: Excellent predictor (strong linear relationship)
- `0.7 < |r| ≤ 0.9`: Good predictor
- `0.5 < |r| ≤ 0.7`: Moderate predictor
- `0.3 < |r| ≤ 0.5`: Weak predictor
- `|r| ≤ 0.3`: Poor predictor

**Critical detail:** Computed between **raw metric** and **MOS**, NOT between predicted and actual MOS. Computing `Pearson(MOS_hat, MOS)` would artificially inflate correlation due to polynomial fitting.

**P-value significance:** `p < 0.05` indicates the correlation is statistically significant (unlikely due to random chance).

---

#### 📈 Spearman Correlation (ρ)

**Formula:**
```
ρ = Pearson(rank(x), rank(MOS))
```

**What it measures:** Monotonic relationship strength using rank order instead of raw values.

**Example:**
```
Metric:  [10, 20, 35, 50]     →  Ranks: [1, 2, 3, 4]
MOS:     [2.1, 3.0, 4.2, 4.5] →  Ranks: [1, 2, 3, 4]
ρ = 1.0 (perfect monotonic agreement)
```

**Why use it?**
- Robust to outliers (uses ranks, not raw values)
- Captures non-linear but monotonic relationships
- If `ρ >> r`, relationship is monotonic but non-linear

**When to prefer Spearman:**
- Metric has non-linear scale (e.g., logarithmic)
- Data contains outliers
- Relationship is monotonic but curved

---

#### 🎯 RMSE (Root Mean Square Error)

**Formula:**
```
RMSE = √(Σ(MOS - MOS_hat)² / (N - 1))
```

**What it measures:** Average prediction error magnitude in MOS units.

**Why (N-1)?** Bessel's correction for unbiased sample variance estimator (ITU-T P.1401 Eq. 7-2).

**Interpretation:**
- `RMSE < 0.3`: Excellent prediction accuracy
- `0.3 ≤ RMSE < 0.5`: Good accuracy
- `0.5 ≤ RMSE < 0.7`: Moderate accuracy
- `RMSE ≥ 0.7`: Poor accuracy

**Example:** `RMSE = 0.4` means predictions are on average 0.4 MOS points away from actual ratings. On a 1-5 scale, this is ~8% error.

**Limitation:** Treats all errors equally. A 0.5 MOS error within CI95 is treated the same as one outside CI95.

---

#### 🎯 RMSE* (Epsilon-Insensitive RMSE)

**Formula:**
```
eff_err_i = max(|MOS_i - MOS_hat_i| - CI95_i, 0)
RMSE* = √(Σeff_err_i² / (N - d))
```
where `d = 4` (degrees of freedom for 3rd-order polynomial).

**What it measures:** Prediction error **outside** subjective uncertainty range.

**Key insight:** If `|error| < CI95`, the prediction is within subjective uncertainty → discount this error.

**Example:**
```
Sample 1: MOS=4.2, CI95=0.3, MOS_hat=4.0
  |error| = 0.2 < 0.3  →  eff_err = 0 (within uncertainty)

Sample 2: MOS=3.5, CI95=0.2, MOS_hat=4.1
  |error| = 0.6 > 0.2  →  eff_err = 0.4 (outside uncertainty)
```

**Why it matters:** RMSE* better reflects **practically significant errors**. A metric with `RMSE=0.5` but `RMSE*=0.1` makes most errors within subjective noise.

**Interpretation:**
- `RMSE* < 0.2`: Excellent (most errors within CI95)
- `0.2 ≤ RMSE* < 0.4`: Good
- `RMSE* ≥ 0.4`: Poor (many errors exceed subjective uncertainty)

---

#### 🔄 Leave-One-Out Cross-Validation (LOOCV)

**Process:**
```
For i = 1 to N:
    1. Remove sample i from dataset
    2. Train polynomial on remaining N-1 samples
    3. Predict MOS for sample i using trained model
    4. Record prediction error
Aggregate all N predictions to compute RMSE_CV
```

**Why use it?**
- Validates generalization on unseen data
- Essential for small datasets (N < 30) where train/test split wastes data
- Detects overfitting: model memorizing training data vs. learning true relationship

**Interpretation:**
- `RMSE_CV ≈ RMSE`: Good generalization
- `RMSE_CV >> RMSE`: Overfitting (model doesn't generalize)

**Computational cost:** Trains N models (expensive), but maximizes data usage.

---

#### ⚠️ Generalization Gap

**Formula:**
```
Gap = RMSE_CV - RMSE_train
```

**What it measures:** Difference between out-of-sample error and training error.

**Interpretation:**
- **Gap ≈ 0** (± 0.1): Model generalizes well
- **Gap > 0.2**: Moderate overfitting — model fits training data better than it generalizes
- **Gap > 0.5**: Severe overfitting — polynomial may be too flexible or dataset too small
- **Gap < -0.05**: Suspicious — CV error lower than training error suggests data leakage or implementation error

**Example scenario:**
```
RMSE_train = 0.25, RMSE_CV = 0.55  →  Gap = 0.30
Interpretation: Model overfits. Polynomial captures training noise.
Action: Collect more data or simplify model.
```

**When overfitting occurs:**
- Small dataset (N < 20)
- High-order polynomial with few samples
- Correlated samples (e.g., similar videos)

---

#### 📏 Student's t-Distribution (CI95)

**Formula:**
```
CI95 = t(α/2, df=N-1) × σ / √N
```
where:
- `t(0.975, N-1)`: 97.5th percentile of t-distribution (two-tailed test)
- `σ`: Sample standard deviation of votes
- `N`: Number of votes per video

**What it measures:** 95% confidence interval for MOS estimate.

**Interpretation:** "We are 95% confident the true population MOS lies within `[MOS - CI95, MOS + CI95]`."

**Example:**
```
Votes: [4, 5, 4, 3, 4, 5, 4, 4, 3, 5] (N=10)
MOS = 4.1, σ = 0.74, t(0.975, df=9) = 2.262
CI95 = 2.262 × 0.74 / √10 = 0.53

Result: MOS = 4.1 ± 0.53  →  True MOS likely in [3.57, 4.63]
```

**Why t-distribution?** For small samples (N < 30), t-distribution accounts for uncertainty in estimating σ. As N → ∞, t-distribution → normal distribution.

**Practical impact:**
- Narrow CI95 (< 0.3): High agreement among raters
- Wide CI95 (> 0.5): Low agreement, ambiguous quality
- RMSE* uses CI95 to determine outliers

## 📚 Foundation Knowledge

### Prerequisites

- **Statistics:** Correlation, confidence intervals, hypothesis testing, degrees of freedom
- **Linear Algebra:** Polynomial regression via least squares
- **Video Quality Assessment:** Objective metrics (VMAF, PSNR, SSIM), subjective testing (MOS)
- **Machine Learning:** Overfitting, cross-validation, train/test split
- **Python:** Pandas, NumPy array operations, Matplotlib

### ITU-T P.1401 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Mapping Function** | Non-linear transformation from objective metric to MOS scale |
| **Performance Metrics** | Multiple orthogonal measures (correlation, error, outliers) |
| **Outlier Definition** | Prediction error exceeds CI95: `\|e\| > CI95` |
| **Sample Size** | Minimum 24 votes/video, 30+ videos for reliable validation |
| **Correlation Calculation** | Between **raw metric** and **MOS** (NOT predicted vs. actual) |

## 📖 References

### 🎓 Project References

1. **[ITU-T Recommendation P.1401 (01/2020)](https://www.itu.int/rec/T-REC-P.1401-202001-I)**
   Methods, metrics and procedures for statistical evaluation, qualification and comparison of objective quality prediction models

2. **[ITU-T Recommendation P.1401 (10/2014) Corrigendum 1](https://www.itu.int/rec/T-REC-P.1401-201410-S!Cor1)**
   Corrections to equation numbering and RMSE* formula

### 🔗 Related Standards

- **[ITU-T P.910](https://www.itu.int/rec/T-REC-P.910)** — Subjective video quality assessment methods for multimedia applications
- **[ITU-T BT.500](https://www.itu.int/rec/R-REC-BT.500)** — Methodology for the subjective assessment of television picture quality

### 💡 Implementation Notes

| File | Implements | Details |
|------|------------|---------|
| `mapping.py` | ITU-T P.1401 Sections 7.1-7.5 | Polynomial mapping, correlation, RMSE, RMSE*, outliers |
| `ci95.py` | ITU-T P.910 Annex 2 | MOS computation, Student's t-distribution |

**Critical Implementation Details:**
- Degrees of freedom for RMSE*: `d=4` (third-order polynomial has 4 coefficients)
- Correlation computed between `metric` and `mos` (not `mos_pred` vs `mos` — common error!)
- LOOCV essential for small datasets to detect overfitting

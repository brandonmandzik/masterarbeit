# 📊 Video Quality Assessment for Video Neural Style Transfer (V-NST)

## 🎯 Overview

Validates objective quality metrics against human perception for AI-stylized videos using ITU-T methodologies. Pipeline: generate videos (Wan2.1 VACE / Wan2.2 VideoX Fun Control) → evaluate with 6 metric suites → collect human ratings (ITU-T P.910) → validate correlation (ITU-T P.1401).

**Research Questions**: 
1. Which metrics best predict human perception? 
2. What quality failures dominate modern TIV2V generation and stylization? 
3. Does multi-dimensional assessment improve accuracy?

---

## ⚡ Features

### 🎬 GPU Infrastructure & Inference
- **Infrastructure as Code (IaC)**: Terraform-provisioned AWS EC2 (g6e.4xlarge/p5.4xlarge)
- **High-Performance GPUs**: NVIDIA L40S (48GB) / H100 (80GB) with CUDA 12.x
- **AI Video Models**: Wan2.1 VACE / Wan2.2 VideoX Fun Control
- **ComfyUI Workflows**: Node-based visual programming (2 included workflows)
- **Production Ready**: Docker containerization, 500GB SSD, SSM secure access

### 🔬 Objective Metrics Suite

| Metric | Type | Architecture | Output |
|--------|------|-------------|--------|
| **COVER** | NR | CLIP ViT-L/14 + Swin3D + ConvNeXt | Semantic, technical, aesthetic, overall |
| **CLIP** | NR | Text-video / Imagge-Vidoe / video-video embeddings | Mean/median semantic scores |
| **SI-TI** | NR | Sobel edge + frame diff | Spatial/temporal complexity stats |
| **LPIPS** | FR | AlexNet deep features | Perceptual distance (mean/median/std) |
| **SSIM** | FR | Structural similarity | Luminance/contrast/structure (mean/median/std) |
| **TV-L1** | NR | DualTVL1 optical flow | 11 temporal consistency metrics |

*NR=No-Reference (blind), FR=Full-Reference (requires original)*

### 👥 Subjective Testing (P.910)
- ACR 5-point scale (1=Bad, 5=Excellent)
- Fisher-Yates randomization (order bias prevention)
- Grey screen intervals (afterimage prevention)
- Response time tracking per video

### 📈 Statistical Validation (P.1401)
- MOS computation (Student's t-distribution CI95)
- 3rd-order polynomial mapping (objective → MOS scale)
- LOOCV cross-validation (generalization testing)
- Automated ranking (Excellent/Good/Fair/Poor categories)

### 🔧 Modular Architecture
- Self-contained tools (isolated Python 3.9 venvs)
- Parallel execution (no inter-dependencies)
- Standardized CSV output (`Filename` column)
- Open and extensible framework (auto-validation)

---

## 🚀 Quick Start

**Prerequisites**: Python 3.9+, 15GB disk, modern browser, CUDA GPU (recommended)

### 1️⃣ Run Objective Metrics
```bash
# COVER (multi-dimensional neural)
cd research-suite/qualifier/cover_assessment && source venv/bin/activate
python src/cover_assessment.py --input ../../data/result_videos && deactivate

# CLIP (semantic alignment)
cd ../clip-score_assessment && source venv/bin/activate
python src/clip_score_assessment.py --input ../../data/result_videos && deactivate

# SI-TI (spatial/temporal complexity)
cd ../si-ti_assessment && source venv/bin/activate
python src/main.py --input ../../data/result_videos  && deactivate

# TV-L1 (optical flow consistency)
cd ../tv-l1_assessment && source venv/bin/activate
python src/tv-l1_assessment.py --input ../../data/result_videos && deactivate

# LPIPS & SSIM (perceptual similarity to reference)
cd ../lpips-ssim_assessmentt && source venv/bin/activate
python main.py --input-dir ../../data/result_videos --reference-dir ../../data/Input_videos && deactivate
```

### 2️⃣ Collect Subjective Ratings
```bash
cd ../../p910/video-player && ln -s ../../data/result_videos videos
python3 -m http.server 8000  # → http://localhost:8000
```

### 3️⃣ Validate Metrics
```bash
cd ../../p1401 && source venv/bin/activate
python src/ci95.py --input ../p910/results/ 
python src/mapping.py --mos results/mos/mos_results.csv \
  --metrics ../qualifier/*/results/*.csv 
deactivate
```

**Outputs**: `p1401_summary_enhanced.csv` (Pearson, Spearman, RMSE, RMSE*, RMSE_CV, Gap), `p1401_ranking_table.csv` (ranked metrics), 50+ PNG plots

---

## 📁 Repository Structure

```
project/
├─ infrastructure/                # 🎬 Video generation
│  ├─ wan22/                      # Wan2.2 TIV2V (14B, Docker)
│  ├─ hunyuanvideo/               # HunyuanVideo T2V/I2V
│  └─ terraform/                  # AWS + L40S(48gb) / H100(80gb) GPU Compute Server provisioning
│
└─ research-suite/
   ├─ data/
   │  ├─ result_videos/           # source{i}_{j}.mp4 (10)
   │  ├─ input_videos/            # original{i}.mp4 (5) (https://database.mmsp-kn.de/konvid-1k-database.html)
   │  ├─ input_videos_24fps/      # original{i}.mp4 (5)
   │  ├─ input_images/            # original{i}.mp4 (5) style reference images (https://www.kaggle.com/datasets/skjha69/artistic-images-for-neural-style-transfer)
   │  └─ test_videos/             # 
   │
   ├─ qualifier/                  # 🔬 6 metric suites (venv/, src/, results/)
   │  ├─ cover_assessment/        
   │  ├─ clip-score_assessment/
   │  ├─ si-ti_assessment/
   │  ├─ tv-l1_assessment/
   │  └─ lpips-ssim_assessment/ 
   │
   ├── p910/                       # 👥 Subjective testing
   │   ├── video-player/           # Web UI (randomization)
   │   └── results/                # p910_assessment_{ID}.csv
   │
   └── p1401/                      # 📊 Statistical validation
       ├── src/                    # ci95.py, mapping.py
       └── results/                # MOS, P.1401 analysis, plots
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  STYLIZED VIDEO GENERATION (Wan2.1 VACE / Wan2.2 VideoX Fun Control) │
│  AWS p5.2xlarge (NVIDIA H100 GPI, 80GB VRAM)                         │
└───────────────────┬──────────────────────────────────────────────────┘
                    │
                    │  source{i}_{j}.mp4 (10 generated)
                    │  original{i}.mp4 (5 references)
                    ↓
        ┌────────┬──┴──────┬────────┬─────────┬─────────┐
        ↓        ↓         ↓        ↓         ↓         ↓
    ┌──────┐  ┌─────┐   ┌─────┐  ┌──────┐  ┌─────┐  ┌──────┐
    │COVER │  │CLIP │   │SI-TI│  │LPIPS │  │SSIM │  │TV-L1 │
    └──┬───┘  └──┬──┘   └──┬──┘  └───┬──┘  └──┬──┘  └───┬──┘
       └─────────┴─────────┴─────────┴────────┴─────────┘
                           ↓ Metric CSVs
                   ┌───────┴────────┐
                   ↓ Videos         ↓ Metrics
             ┌──────────┐     ┌──────────┐
             │P.910     │     │Objective │
             │Player    │     │CSVs      │
             └────┬─────┘     └────┬─────┘
                  └──────┬──────────┘
                         ↓ Ratings + Metrics
                  ┌──────────────┐
                  │ci95.py       │ → MOS ± CI95
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │mapping.py    │ → P.1401 Validation
                  └──────┬───────┘
                         ↓
               ┌─────────────────────┐
               │Results:             │
               │• Correlations       │
               │• Rankings           │
               │• Plots (50+)        │
               └─────────────────────┘
```

---

## 🎬 Video Generation Infrastructure

**GPU-accelerated video generation** using AWS cloud infrastructure with Terraform provisioning. Deploy **Wan2.2** (14B params) or **HunyuanVideo** (13B params) on NVIDIA L40S (48GB) or H100 (80GB) GPUs. **ComfyUI node-based workflows** enable visual programming for text-to-video and image-to-video generation with advanced control mechanisms (Canny edge, depth maps, reference images).

### Infrastructure Specifications

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Instance** | g6e.4xlarge / p5.4xlarge | GPU compute |
| **GPUs** | NVIDIA L40S (48GB) / H100 (80GB) | Model inference |
| **Storage** | 500GB GP3 SSD | Model checkpoints (126GB each) |
| **Access** | AWS SSM (keyless) | Secure remote access |
| **Container** | Docker + NVIDIA runtime | Isolated inference environment |

### ComfyUI Workflows

| Workflow | Model | Parameters | Features | Use Case |
|----------|-------|------------|----------|----------|
| **wan2.1-vace.json** | VACE 14B/1.3B | 27B/2.6B | LoRA optimization, reference image matching, Canny+Depth control | Production quality |
| **wan2.2-videox-fun-control.json** | Fun Control 5B | 5B | Multi-language, edge/depth/pose/trajectory control | Experimentation |

---

## 🛠️ Technologies

### 📊 P.1401 Statistical Validation

| Formula | Purpose | Interpretation |
|---------|---------|----------------|
| `MOS_pred = a₀ + a₁x + a₂x² + a₃x³` | Polynomial mapping | Captures non-linear perception |
| `RMSE = √(Σ(MOS - MOS_pred)² / (N-1))` | Training error | < 0.3 excellent, < 0.5 good |
| `RMSE_CV` (LOOCV) | Generalization error | Tests unseen data prediction |
| `Gap = RMSE_CV - RMSE` | Overfitting detection | > 0.2 suggests more data needed |

**Correlation**: Pearson (linear), Spearman (rank-based)
**RMSE\***: CI95-discounted error (only penalizes significant errors)
**CI95**: `t(0.975, N-1) × STD / √N` (Student's t for small samples)

**Categories** (P.1401 Table I-1):
- **Excellent**: RMSE_CV < 0.3 AND |r| > 0.7
- **Good**: RMSE_CV < 0.5 AND |r| > 0.5
- **Fair**: RMSE_CV < 0.7 AND |r| > 0.3
- **Poor**: Otherwise

### 🎭 P.910 Subjective Testing
- **ACR**: Participants view video → rate 1-5 (simpler than comparison methods)
- **Randomization**: Fisher-Yates shuffle (prevents order bias)
- **Grey screens**: 50% grey, 2s (prevents afterimages via HVS adaptation reset)

### 🎨 Quality Metrics

**COVER** (CVPR 2024): 3-branch ensemble. CLIP ViT-L/14 (semantic, 20f) + Swin3D (technical, 40f) + ConvNeXt (aesthetic, 40f). Unbounded scores (negative normal)
**CLIP**: `cosine_similarity(CLIP_text(prompt), CLIP_image(frame))`. Text-Video / Image-Video / Video-Video semantic alignment. Limitation: frame-wise interpretation, no motion understanding. 
**SI-TI** (ITU-T P.910): `SI = stddev(Sobel(Y))`, `TI = stddev(Y_n - Y_{n-1})` where `Y = 0.299R + 0.587G + 0.114B`. Scene complexity/motion. 
**LPIPS**: AlexNet features + learned weights. Range [0,∞), lower = similar. Trained on human perceptual judgments. Frame-by-frame vs reference. 
**SSIM**: `SSIM = l(x,y) × c(x,y) × s(x,y)` (luminance × contrast × structure). Range [0,1], 1 = identical. HVS-aligned structural similarity. 
**TV-L1**: DualTVL1 optical flow (Total Variation + L1 norm). Metrics: forward-backward error, warp error, motion magnitude, Q-transforms `exp(-α × error)`. Detects temporal drift/jitter. 

---

## 📚 Prerequisites

**Statistics**: Hypothesis testing, regression, correlation (Pearson/Spearman), cross-validation, overfitting detection
**Computer Vision**: CNNs, Vision Transformers, optical flow, image quality metrics, deep features
**Video Processing**: Frame extraction, temporal consistency, spatial vs temporal quality, compression artifacts
**Machine Learning**: Transfer learning, ensemble methods, no-reference assessment

---


## 🔗 References

**Standards**:
- [ITU-T P.910 (10/2023)](https://www.itu.int/rec/T-REC-P.910-202310-I/en) - Subjective video quality (ACR, grey screens)
- [ITU-T P.1401 (01/2020)](https://www.itu.int/rec/T-REC-P.1401-202001-I) - Metric validation (polynomial mapping, LOOCV)

**Metrics**:
- [COVER (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/He_COVER_A_Comprehensive_Video_Quality_Evaluator_CVPRW_2024_paper.pdf)
- [CLIPScore (EMNLP 2021)](https://arxiv.org/abs/2104.08718)
- [LPIPS (CVPR 2018)](https://arxiv.org/abs/1801.03924)
- [SSIM (IEEE TIP 2004)](https://ieeexplore.ieee.org/document/1284395)
- [TV-L1 (IPOL 2013)](https://www.ipol.im/pub/art/2013/26/article.pdf)

**Models**:
- [Wan2.2 Base](https://arxiv.org/pdf/2503.20314) | [GH](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 VideoX Fun Control](https://arxiv.org/abs/2408.06072) | [HF](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control)
- [Wan2.1 VACE](http://arxiv.org/abs/2503.07598) | [HF](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)
- [HunyuanVideo Base](https://arxiv.org/abs/2412.03603) | [GH](https://github.com/Tencent-Hunyuan/HunyuanVideo)

---

**Development Guide**: See `CLAUDE.md` for commands, troubleshooting, implementation details

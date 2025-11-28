# COVER Video Quality Assessment

> Python implementation of COVER (CVPR 2024 AIS Workshop Winner) for comprehensive blind video quality evaluation across semantic, technical, and aesthetic dimensions.

## 📋 Overview

COVER (Comprehensive Video Quality Evaluator) is a **no-reference video quality assessment** model that predicts perceptual quality across three orthogonal dimensions. It evaluates user-generated content without requiring pristine reference videos, addressing the challenge that traditional metrics (PSNR, SSIM) fail for content where no reference exists.

**Why it exists:** Streaming platforms, social media, and AI-generated video need automated quality control. COVER provides multi-dimensional quality scores that correlate strongly with human perception (SROCC 0.9143 on YouTube-UGC).

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Three-branch architecture** | Semantic (content understanding), Technical (distortion detection), Aesthetic (visual appeal) |
| **Blind assessment** | No reference video required (crucial for UGC and AI-generated content) |
| **Multi-resolution support** | Handles videos up to 4K (3840×2160) |
| **Fast inference** | ~79ms per 4K video on NVIDIA A100, ~8-12s on Apple M1 CPU |
| **Temporal sampling** | Analyzes full video duration via uniform frame sampling |
| **ARM64 compatible** | PyAV-based video reading for macOS Apple Silicon |
| **Unbounded regression** | Quality scores are relative (not percentages) for fine-grained ranking |
| **Pre-trained weights** | Trained on 5 datasets: YouTube-UGC, KoNViD-1k, LIVE-VQC, LSVQ, CVD2014 |
| **CSV export** | Batch processing with structured output |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

```bash
mkdir -p src/pretrained_weights
cd src/pretrained_weights
wget https://github.com/vztu/COVER/raw/release/Model/COVER.pth
cd ../..
```

### Usage

```bash
# Activate environment
source venv/bin/activate

# Run assessment on video directory
python src/cover_assessment.py --input ../../data/source_videos --output ./results/cover_results.csv

# Custom file pattern
python src/cover_assessment.py --input /path/to/videos --pattern "*.mov" --output results.csv

# Deactivate when done
deactivate
```

### Verify Installation

```bash
# Test on sample video
python src/cover_assessment.py --input ../../data/tests_videos --pattern "test_real.mp4"

# Expected output format:
# Filename,Semantic,Technical,Aesthetic,Overall
# test_real.mp4,-0.0642,0.8712,0.3542,1.1612
```

## 🏗️ Architecture

### High-Level Flow

COVER decomposes videos into three parallel processing streams, each optimized for a specific quality dimension:

```
Input Video (MP4)
      │
      ▼
UnifiedFrameSampler ──┬──► Semantic Branch (512×512, 20 frames)
                      │        │
                      ├──► Technical Branch (7×7 grid of 32×32 patches, 40 frames)
                      │        │
                      └──► Aesthetic Branch (224×224, 40 frames)
                               │
                               ▼
                    Normalization (branch-specific mean/std)
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  CLIP ViT-L/14      Swin Transformer 3D      ConvNeXt Tiny 3D
  (semantic)         (technical)              (aesthetic)
        │                      │                      │
        ▼                      ▼                      ▼
    VQAHead                VQAHead                VQAHead
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
                    Fuse: overall = s + t + a
                               │
                               ▼
                    CSV Export (4 scores per video)
```

### Low-Level Component Pipeline

Each branch follows a standardized processing pipeline with branch-specific configurations:

```
Video File Path
      │
      ▼
PyAVVideoReader (ARM64-compatible wrapper)
      │
      ├─ Decode all frames to RGB24
      ├─ Store as torch tensors (H×W×C)
      └─ Return indexable frame list
      │
      ▼
spatial_temporal_view_decomposition()
      │
      ├─ Semantic View:  resize to 512×512, sample 20 frames (t_frag=20, interval=2)
      ├─ Technical View: fragment to 7×7 grid (32×32 patches), sample 40 frames
      └─ Aesthetic View: resize to 224×224, sample 40 frames
      │
      ▼
Normalization Layer
      │
      ├─ Semantic:  (pixel - [122.77,116.75,104.09]) / [68.50,66.63,70.32]
      └─ Tech/Aes:  (pixel - [123.68,116.28,103.53]) / [58.40,57.12,57.38]
      │
      ▼
Backbone Networks (parallel inference)
      │
      ├─ Semantic:  CLIP ViT-L/14 → feature dim 768
      ├─ Technical: Swin Transformer 3D (tiny_grpb, window_size 4×4×4) → 768
      └─ Aesthetic: ConvNeXt Tiny 3D → 768
      │
      ▼
VQA Heads (768 → 64 → 1)
      │
      ├─ Linear(768, 64) + ReLU
      └─ Linear(64, 1) → unbounded regression score
      │
      ▼
Score Fusion
      │
      └─ overall_score = semantic + technical + aesthetic
```

## 🛠️ Technologies

### Core Libraries

| Library | Purpose | Key Components |
|---------|---------|----------------|
| **PyTorch** | Deep learning framework | Model architecture, tensor operations, GPU acceleration |
| **PyAV** | Video decoding (FFmpeg wrapper) | Cross-platform video reading (ARM64 compatibility) |
| **NumPy** | Numerical operations | Array manipulation, normalization |
| **YAML** | Configuration management | Model hyperparameters, dataset settings |

### Neural Network Architectures

#### 🧠 Semantic Branch: CLIP ViT-L/14

**Purpose:** Evaluates content understanding and semantic coherence (e.g., "Does the video show what it claims?").

**Architecture:** Vision Transformer Large (14×14 patches)
- **Input:** 512×512 RGB frames (20 frames uniformly sampled)
- **Patch embedding:** 14×14 patches → 768-dim embeddings
- **Transformer:** 24 layers, 16 attention heads
- **Pre-training:** OpenAI CLIP (400M image-text pairs)
- **Output:** 768-dim semantic feature vector

**Theoretical foundation:** Vision Transformers use self-attention to model long-range dependencies. CLIP's contrastive pre-training aligns visual and textual semantics, making it sensitive to content mismatches (e.g., blurry faces, text illegibility).

**Normalization:** `mean=[122.77, 116.75, 104.09]`, `std=[68.50, 66.63, 70.32]` (CLIP-specific RGB statistics).

---

#### 🔧 Technical Branch: Swin Transformer 3D

**Purpose:** Detects compression artifacts, noise, blocking, blur, and encoding distortions.

**Architecture:** Swin Transformer 3D Tiny with Group-based Position Bias (tiny_grpb)
- **Input:** 7×7 spatial grid of 32×32 patches (40 frames, total 7×7×40 = 1960 patches)
- **Window size:** 4×4×4 (spatial-temporal local attention)
- **Hierarchical stages:** 4 stages with shifted windows for cross-window connections
- **Parameters:** ~28M (tiny variant)
- **Output:** 768-dim technical feature vector

**Theoretical foundation:** Swin Transformer uses **shifted windowing** for efficient spatial-temporal attention. By fragmenting video into 7×7 grids, the model captures **local distortions** (e.g., blocking in one region) while hierarchical merging enables **global quality estimation**.

**Why 7×7 grid?** Balances computational cost vs. spatial sensitivity. Smaller patches (32×32) detect fine-grained artifacts invisible at full resolution.

---

#### 🎨 Aesthetic Branch: ConvNeXt Tiny 3D

**Purpose:** Evaluates visual composition, color harmony, lighting, and artistic appeal.

**Architecture:** ConvNeXt Tiny 3D (modernized ConvNet design)
- **Input:** 224×224 RGB frames (40 frames)
- **Stem:** 7×7×7 conv with stride 2 → 96 channels
- **Stages:** 4 stages with depthwise convolutions + layer normalization
- **Inverted bottleneck:** Expands channels 4× before compression (efficient feature extraction)
- **Output:** 768-dim aesthetic feature vector

**Theoretical foundation:** ConvNeXt applies Transformer design principles (LayerNorm, GELU, larger kernels) to convolutional networks, achieving ViT-level accuracy with better inductive bias for visual patterns (e.g., symmetry, color gradients).

**Why ConvNeXt over ViT?** Convolutional inductive bias better captures low-level aesthetic patterns (texture, contrast) than pure attention.

---

#### 📊 VQA Heads: Regression Layers

**Purpose:** Map 768-dim backbone features to 1-dim quality scores.

**Architecture:**
```
Linear(768 → 64) → ReLU → Linear(64 → 1)
```

**Training objective:** Mean Squared Error (MSE) against ground-truth Mean Opinion Scores (MOS).

**Unbounded regression:** Unlike classification (1-5 stars), regression outputs continuous values. Scores are **relative**, not absolute:
- Negative scores indicate below-average quality
- Positive scores indicate above-average quality
- Compare within dataset, not across datasets

---

### Temporal Sampling Strategy

**UnifiedFrameSampler** ensures consistent temporal coverage:

```python
# Semantic: 20 frames, interval=2, t_frag=20
# Samples every 2nd frame across full video duration
frames_semantic = [0, 2, 4, ..., 38]  # 20 frames total

# Technical/Aesthetic: 40 frames, interval=2, t_frag=20
frames_tech_aes = [0, 2, 4, ..., 78]  # 40 frames total
```

**t_frag (temporal fragment):** Divides video into fragments, samples within each. Ensures uniform coverage even for variable-length videos.

**Why different frame counts?**
- **Semantic (20 frames):** Content understanding requires fewer samples (global semantics change slowly)
- **Technical/Aesthetic (40 frames):** Distortion and composition vary rapidly (motion blur, lighting changes)

---

### Score Interpretation

**COVER outputs unbounded regression values:**

```csv
Filename,Semantic,Technical,Aesthetic,Overall
good_video.mp4,0.82,1.15,0.73,2.70
poor_video.mp4,-0.45,-0.68,-0.32,-1.45
```

**Analysis:**
- `good_video.mp4`: All dimensions positive → high quality
- `poor_video.mp4`: All dimensions negative → low quality

**Critical:** Use **relative ranking** within your dataset. A score of 1.5 in one dataset ≠ 1.5 in another.

**Why unbounded?** Regression on MOS (1-5 scale) allows the model to express confidence beyond scale bounds (e.g., 5.2 for exceptionally high quality).

## 📚 Foundation Knowledge

### Prerequisites

- **Deep Learning:** Transformers, Vision Transformers, convolutional neural networks, attention mechanisms
- **Video Processing:** Frame sampling, temporal modeling, video codecs (H.264, H.265)
- **Computer Vision:** Image quality assessment, no-reference metrics, perceptual quality
- **Statistics:** Mean Opinion Score (MOS), regression vs. classification, correlation metrics (SROCC, PLCC)
- **Python:** PyTorch, tensor operations, virtual environments

### Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Blind VQA** | Quality assessment without reference video (critical for UGC, AI-generated content) |
| **MOS (Mean Opinion Score)** | Average human quality rating (1=Bad, 5=Excellent) from subjective tests |
| **SROCC** | Spearman Rank-Order Correlation Coefficient (measures monotonic prediction accuracy) |
| **PLCC** | Pearson Linear Correlation Coefficient (measures linear prediction accuracy) |
| **Multi-view decomposition** | Splitting input into specialized representations (spatial, temporal, semantic) |
| **Transfer learning** | Using pre-trained models (CLIP, ImageNet) for downstream quality tasks |
| **Temporal fragments** | Dividing video timeline into segments for uniform sampling |

### COVER-Specific Details

- **Three-branch rationale:** Human quality judgment is multi-dimensional (content clarity + distortion level + visual appeal). Single-branch models conflate these aspects.
- **No-reference necessity:** User-generated content and AI-generated videos lack pristine references.
- **Regression over classification:** Continuous scores capture subtle quality gradations (3.2 vs 3.8).
- **CLIP for semantics:** Pre-training on image-text pairs makes CLIP sensitive to content-quality mismatches (e.g., unclear text, distorted faces).

## 📖 References

### 🎓 Project References

1. **[COVER: A Comprehensive Video Quality Evaluator (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/He_COVER_A_Comprehensive_Video_Quality_Evaluator_CVPRW_2024_paper.pdf)**
   He et al., CVPR 2024 Workshop on AI for Streaming (Winner of VQA Challenge)

2. **[COVER GitHub Repository](https://github.com/taco-group/COVER)**
   Official implementation and model weights

3. **[COVER HuggingFace Demo](https://huggingface.co/spaces/vztu/COVER)**
   Interactive web demo for testing COVER

4. **[Model Weights (COVER.pth)](https://github.com/vztu/COVER/blob/release/Model/COVER.pth)**
   Pre-trained weights (250MB) from release branch

### 🔗 Related Technologies

- **[CLIP (OpenAI)](https://github.com/openai/CLIP)** — Contrastive Language-Image Pre-training
- **[Swin Transformer](https://github.com/microsoft/Swin-Transformer)** — Hierarchical Vision Transformer
- **[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)** — Modernized ConvNet architecture
- **[PyAV](https://github.com/PyAV-Org/PyAV)** — Pythonic FFmpeg bindings

### 📊 Training Datasets

| Dataset | Description | Videos | Resolution |
|---------|-------------|--------|------------|
| **YouTube-UGC** | YouTube user-generated content | 1,500 | 360p-4K |
| **KoNViD-1k** | In-the-wild videos | 1,200 | 540p |
| **LIVE-VQC** | Mobile-captured videos | 585 | 1080p |
| **LSVQ** | Large-scale VQA dataset | 28,000 | 240p-4K |
| **CVD2014** | Crowdsourced videos | 234 | 480p-1080p |

### 💡 Implementation Notes

- **ARM64 compatibility:** Uses PyAV wrapper (`PyAVVideoReader`) instead of `decord` for Apple Silicon support
- **Normalization constants:** Branch-specific RGB mean/std derived from training data statistics
- **Model loading:** `strict=False` required due to auxiliary training components not used in inference
- **Device handling:** Auto-detects CUDA availability, falls back to CPU
- **Batch processing:** Sequential processing with progress tracking (parallelization not implemented)

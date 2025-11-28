# 🎬 Video Quality Assessment Research Project

## 📋 Overview

Research platform investigating **objective metric correlation with human perception** for AI-generated videos. Implements complete pipeline: generation (Wan2.2/HunyuanVideo) → objective assessment (COVER/SI-TI/TV-L1) → subjective testing (ITU-T P.910) → validation (ITU-T P.1401).

**Core Research Questions:**
1. How well do objective quality metrics predict human perception for AI-generated videos?
2. What are dominant quality failure modes in modern text-to-video models?
3. Can multi-dimensional quality assessment improve prediction accuracy?

---

## ✨ Features

### 🎥 **Video Generation** (GPU-Accelerated Infrastructure)
- **Wan2.2**: 14B parameter autoregressive transformer (T2V/I2V)
- **HunyuanVideo**: 13B+ diffusion transformer with flow-matching (T2V/I2V)
- Dockerized models with AWS EC2 provisioning (Terraform)
- Supports L40S (48GB) / H100 (80GB) GPUs

### 📊 **Objective Quality Assessment**
- **COVER** (CVPR 2024): Three-branch neural network (semantic + technical + aesthetic)
- **SI/TI** (ITU-T P.910): Spatial information & temporal information metrics
- **TV-L1**: Optical flow consistency (forward-backward error, temporal warping)

### 👥 **Subjective Testing**
- Web-based P.910 compliant player (ACR 5-point scale)
- Randomized presentation, grey screen intervals
- CSV export: participant ID, ratings, response times

### 📈 **Statistical Validation**
- ITU-T P.1401 framework: polynomial mapping, Pearson/Spearman correlation
- RMSE and epsilon-insensitive RMSE*
- Leave-One-Out Cross-Validation (LOOCV)

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.9+
- Docker 20.10+ (for generation infrastructure)
- NVIDIA GPU with CUDA 12.1+ (for generation)
- 200GB+ disk space (model weights)

### **Installation**

```bash
# Clone repository
git clone <repo-url>
cd project

# Setup objective assessment tools
cd research-suite/qualifier/cover_assessment
python3 -m venv venv
source venv/bin/activate
./setup.sh
pip install -r requirements.txt
deactivate

cd ../si-ti-assessment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

cd ../tv-l1_assessment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

# Setup P.1401 validation
cd ../../p1401
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### **Run Assessment**

```bash
# 1. Assess videos with COVER
cd research-suite/qualifier/cover_assessment
source venv/bin/activate
python src/cover_assessment.py --input-dir ../../data/source_videos
deactivate

# 2. Compute SI/TI
cd ../si-ti-assessment
source venv/bin/activate
python src/main.py --input ../../data/source_videos --output ./results
deactivate

# 3. Analyze optical flow
cd ../tv-l1_assessment
source venv/bin/activate
python src/tv-l1_assessment.py --video ../../data/source_videos/video.mp4
deactivate

# 4. Launch P.910 subjective testing
cd ../../p910/video-player
ln -s ../../data/source_videos videos
python3 -m http.server 8000
# Visit http://localhost:8000

# 5. Validate metrics
cd ../../p1401
source venv/bin/activate
python src/ci95.py -i data/votes/ -o results/mos/mos_results.csv
python src/mapping.py \
  --mos results/mos/mos_results.csv \
  --metrics cover_results.csv tv-l1_results.csv \
  --output results/p1401/
deactivate
```

### **Verify Installation**

```bash
cd research-suite/qualifier/cover_assessment
source venv/bin/activate
python -c "import sys; sys.path.insert(0, 'src/cover_repo'); from cover.models import COVER; print('✓ COVER verified')"
deactivate
```

---

## 🏗️ Architecture

### **System Overview**

End-to-end pipeline: text prompts → AI-generated videos → objective metrics + subjective ratings → correlation analysis.

**Data Flow:** Prompts → Generation (Wan2.2/Hunyuan) → Videos → Assessment (COVER/SI-TI/TV-L1) → Objective Scores + P.910 Player → Human Ratings (MOS) → P.1401 Validation → Correlation Results

### **High-Level Flowchart**

```
Text Prompts
     |
     v
Video Generation (Wan2.2/HunyuanVideo)
     |
     v
AI-Generated Videos (.mp4)
     |
     +------------------+------------------+
     |                  |                  |
     v                  v                  v
 COVER VQA          SI/TI Metrics     TV-L1 Flow        [Objective]
     |                  |                  |
     +------------------+------------------+
                        |
                        v
              Objective Scores (CSV)
                        |
                        |
              P.910 Web Player  <--- Human Ratings    [Subjective]
                        |
                        v
              Subjective MOS (± CI95)
                        |
                        +------------------+
                        |                  |
                        v                  v
              P.1401 Validation Framework
                        |
                        v
          Correlation Analysis (RMSE/Pearson/Spearman)
```

### **Low-Level Module Interactions**

```
Infrastructure Layer (AWS EC2 + Docker)
├─ Terraform ──> Provisions EC2 (g6e.4xlarge/p5.4xlarge)
├─ wan22/ ──────> Docker: Wan2.2-T2V/I2V-A14B (126GB weights)
└─ hunyuanvideo/> Docker: HunyuanVideo T2V/I2V (CPU offloading)
        │
        ▼ (Outputs: MP4 videos)
        │
Research Suite (Python 3.9 Virtual Environments)
│
├─ qualifier/
│  ├─ cover_assessment/
│  │  ├─ COVER Model (3 Branches)
│  │  │  ├─ Semantic:  CLIP ViT-L/14 (512×512, 20 frames)
│  │  │  ├─ Technical: Swin3D (32×32 patches, 40 frames)
│  │  │  └─ Aesthetic: ConvNeXt Tiny (224×224, 40 frames)
│  │  └─ Output: overall_score = semantic + technical + aesthetic
│  │
│  ├─ si-ti-assessment/
│  │  ├─ SI: stddev(Sobel(frame)) → spatial detail
│  │  └─ TI: stddev(Δframe) → temporal motion
│  │
│  └─ tv-l1_assessment/
│     ├─ DualTVL1OpticalFlow (OpenCV)
│     ├─ Forward-Backward Consistency Error
│     ├─ Temporal Warp Error
│     └─ Motion Magnitude Variance
│
├─ p910/ (Subjective Testing)
│  └─ video-player/
│     ├─ Fisher-Yates Randomization
│     ├─ Grey Screen Intervals (2s before/after)
│     └─ CSV Export: [ParticipantID, VideoIndex, Rating, Timestamp]
│
└─ p1401/ (Validation Framework)
   ├─ ci95.py ──────> Computes MOS ± CI95 from votes
   └─ mapping.py ───> Third-order polynomial: MOS_pred = Σ(aᵢ·xⁱ)
                      ├─ Pearson/Spearman Correlation
                      ├─ RMSE / RMSE* (epsilon-insensitive)
                      └─ LOOCV (Leave-One-Out Cross-Validation)
```

### **Directory Structure**

```
project/
├─ infrastructure/
│  ├─ wan22/           # Wan2.2 Docker + generation scripts
│  ├─ hunyuanvideo/    # HunyuanVideo Docker + generation scripts
│  └─ terraform/       # AWS EC2 provisioning (IaC)
│
└─ research-suite/
   ├─ qualifier/
   │  ├─ cover_assessment/    # COVER VQA (3-branch NN)
   │  ├─ si-ti-assessment/    # ITU-T P.910 SI/TI
   │  └─ tv-l1_assessment/    # TV-L1 optical flow
   ├─ p910/                   # Subjective testing (web player)
   ├─ p1401/                  # Statistical validation framework
   └─ data/
      ├─ source_videos/       # AI-generated test videos
      └─ tests_videos/        # Validation/test videos
```

---

## 🛠️ Technologies

### **Video Generation Models**

| Model | Architecture | Parameters | Mechanism | Key Innovation |
|-------|--------------|------------|-----------|----------------|
| **Wan2.2** | Autoregressive Transformer | 14B (MoE) | Next-token prediction | Mixture-of-Experts for efficiency |
| **HunyuanVideo** | Diffusion Transformer | 13B+ | Flow-matching diffusion | Dual-branch (text+image) conditioning |

**Theory:** Autoregressive models predict video frame-by-frame sequentially, while diffusion models denoise latent representations through reverse diffusion process (DDPM/DDIM). DiT (Diffusion Transformer) replaces U-Net with transformer blocks for better scalability.

### **Quality Assessment Methods**

#### **COVER (Comprehensive Video Quality Evaluator)**
- **Architecture:** Three parallel branches fused via learned weights
  - **Semantic Branch:** CLIP ViT-L/14 extracts content understanding (text-image alignment)
  - **Technical Branch:** Swin Transformer 3D detects compression artifacts, blur, noise
  - **Aesthetic Branch:** ConvNeXt Tiny evaluates visual appeal (composition, lighting)
- **Output:** Unbounded regression scores (negative = below-average quality)
- **Theory:** Multi-task learning with joint optimization (semantic loss + technical loss + aesthetic loss)

#### **SI/TI (ITU-T P.910 Classical Metrics)**
- **SI (Spatial Information):** Measures frame detail via Sobel edge detection
  - `SI = stddev(Sobel(Y))` where Y = ITU-R BT.601 grayscale
- **TI (Temporal Information):** Measures motion intensity via frame differences
  - `TI = stddev(Y_n - Y_{n-1})`
- **Theory:** Proxy metrics for content complexity (influences perceptual quality)

#### **TV-L1 Optical Flow**
- **Algorithm:** DualTVL1OpticalFlow (Total Variation + L1 norm optimization)
- **Metrics:**
  - Forward-backward consistency error (occlusion detection)
  - Temporal warp error (motion prediction accuracy)
  - Motion magnitude variance (stability)
- **Theory:** Variational optical flow (Zach et al., 2007) balances data fidelity (L1) and smoothness (TV regularization)

### **Subjective Testing (ITU-T P.910)**
- **Method:** Absolute Category Rating (ACR) with 5-point scale
- **Protocol:**
  - Grey screen intervals (50% grey, 2 seconds) prevent afterimage bias
  - Fisher-Yates randomization eliminates order effects
  - Minimum 24 participants per video (ITU-T recommendation)
- **Output:** Mean Opinion Score (MOS) ± 95% Confidence Interval

### **Validation Framework (ITU-T P.1401)**
- **Mapping Function:** Third-order polynomial regression
  - `MOS_predicted = a₀ + a₁·x + a₂·x² + a₃·x³`
  - Fitted via least squares (objective scores → subjective MOS)
- **Metrics:**
  - **Pearson Correlation:** Linear relationship strength
  - **Spearman Correlation:** Monotonic relationship (rank-based)
  - **RMSE:** Root mean square error
  - **RMSE*:** Epsilon-insensitive RMSE (ignores errors within CI95)
- **Validation:** LOOCV (Leave-One-Out Cross-Validation) for small datasets

### **Infrastructure**

| Technology | Purpose | Key Features |
|-----------|---------|--------------|
| **Docker** | Model containerization | GPU passthrough, reproducible builds |
| **Terraform** | Infrastructure as Code | AWS EC2 provisioning, state management |
| **PyTorch** | Deep learning framework | CUDA acceleration, mixed precision |
| **NVIDIA CUDA** | GPU compute | cuDNN (deep learning kernels), Tensor Cores |
| **AWS EC2** | Cloud GPU instances | g6e.4xlarge (L40S), p5.4xlarge (H100) |

**Theory:**
- **CUDA:** Parallel computing architecture for massive matrix operations (convolutions, attention)
- **Tensor Cores:** Specialized units for mixed-precision (FP16/BF16) matrix multiplication (4×faster than FP32)
- **CPU Offloading:** Swap inactive model layers to RAM to fit large models on smaller GPUs

---

## 📚 Foundation Knowledge

### **Required Prior Knowledge**

#### **1. Deep Learning Fundamentals**
- **Transformers:** Self-attention mechanism, multi-head attention, positional encoding
- **Convolutional Networks:** 2D/3D convolutions, receptive fields, feature pyramids
- **Vision-Language Models:** Contrastive learning (CLIP), cross-modal alignment

#### **2. Video Processing**
- **Temporal Sampling:** Uniform frame extraction, sliding windows
- **Optical Flow:** Dense motion estimation (variational methods vs. CNN-based)
- **Color Spaces:** YUV/YCbCr conversion (ITU-R BT.601 standard)

#### **3. Generative Models**
- **Diffusion Models:** Forward noising process (Gaussian noise addition), reverse denoising (learned through U-Net/DiT)
  - DDPM: Markov chain with fixed variance schedule
  - DDIM: Deterministic sampling (fewer steps)
  - Latent Diffusion: Operate in VAE latent space (reduced computation)
- **Autoregressive Models:** Next-token prediction with causal masking
- **Mixture-of-Experts (MoE):** Sparse activation (route inputs to subset of expert networks)

#### **4. Quality Assessment Theory**
- **Full-Reference (FR):** Requires pristine reference (PSNR, SSIM)
- **No-Reference (NR):** Blind quality prediction (COVER, BRISQUE)
- **Perceptual Metrics:** Align with human visual system (LPIPS, VMAF)

#### **5. Statistical Validation**
- **Correlation Analysis:** Linear (Pearson) vs. monotonic (Spearman)
- **Regression:** Polynomial fitting, overfitting (regularization)
- **Cross-Validation:** Train-test splitting (LOOCV for N < 50 samples)
- **Confidence Intervals:** Student's t-distribution (95% CI)

#### **6. ITU-T Standards**
- **P.910:** Subjective video quality (ACR, grey screen protocol)
- **P.1401:** Objective metric validation (RMSE*, outlier ratio)

#### **7. Infrastructure & DevOps**
- **Docker:** Container networking, volume mounts, GPU runtime
- **Terraform:** Declarative infrastructure, state management, providers
- **AWS:** EC2 lifecycle, IAM roles, security groups, SSM sessions
- **CUDA/cuDNN:** Memory management, kernel optimization

---

## 🔗 References

### **Video Generation Models**
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 Technical Paper](https://arxiv.org/pdf/2503.20314) (Autoregressive Transformer)
- [Wan2.2 T2V Model (HF)](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- [Wan2.2 I2V Model (HF)](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- [HunyuanVideo T2V GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- [HunyuanVideo I2V GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)
- [HunyuanVideo Paper](https://arxiv.org/abs/2412.03603) (Diffusion Transformer)
- [HunyuanVideo Model (HF)](https://huggingface.co/tencent/HunyuanVideo)

### **Quality Assessment**
- [COVER Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/He_COVER_A_Comprehensive_Video_Quality_Evaluator_CVPRW_2024_paper.pdf)
- [COVER GitHub](https://github.com/taco-group/COVER)
- [COVER HuggingFace Demo](https://huggingface.co/spaces/vztu/COVER)
- [COVER Model Weights](https://github.com/vztu/COVER/blob/release/Model/COVER.pth)
- [ITU-T P.910 (10/2023)](https://www.itu.int/rec/T-REC-P.910-202310-I/en) - Subjective Video Quality
- [ITU-T P.1401 (01/2020)](https://www.itu.int/rec/T-REC-P.1401-202001-I) - Objective Metric Validation
- [DualTVL1 Optical Flow (OpenCV)](https://docs.opencv.org/3.4/dc/d47/classcv_1_1DualTVL1OpticalFlow.html)
- [TV-L1 Optical Flow Paper](https://www.ipol.im/pub/art/2013/26/article.pdf) (Zach et al., 2007)

### **Diffusion Model Theory**
- [DDPM (2020)](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM (2020)](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models
- [Latent Diffusion Models (2021)](https://arxiv.org/abs/2112.10752) - High-Resolution Image Synthesis
- [DiT (2022)](https://arxiv.org/abs/2212.09748) - Scalable Diffusion Transformers

### **Infrastructure & Frameworks**
- [AWS DL AMI Release Notes](https://aws.amazon.com/releasenotes/aws-deep-learning-base-oss-nvidia-driver-gpu-ami-ubuntu-22-04/)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Language Reference](https://developer.hashicorp.com/terraform/language)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [Docker GPU Support](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### **Machine Learning Frameworks**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [HuggingFace Hub Library](https://huggingface.co/docs/huggingface_hub/index)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### **NVIDIA Hardware & CUDA**
- [NVIDIA L40S Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/proviz-print-nvidia-l40s-datasheet-3230170-r1-web.pdf)
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)

---

**License:** Research project for academic purposes only.
**Contact:** See `CLAUDE.md` for detailed development instructions.

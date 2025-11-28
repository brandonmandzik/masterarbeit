# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Video Quality Assessment Research Project** for evaluating AI-generated videos using objective metrics and subjective testing. The project implements a complete pipeline from video generation through quality assessment to statistical validation following ITU-T standards.

### Key Research Questions
1. How well do objective quality metrics predict human perception for AI-generated videos?
2. What are the dominant quality failure modes in modern text-to-video models?
3. Can multi-dimensional quality assessment improve prediction accuracy?

## Repository Structure

```
project/
├── infrastructure/          # GPU-accelerated video generation
│   ├── wan22/              # Wan2.2 T2V model (14B params, Docker)
│   ├── hunyuanvideo/       # HunyuanVideo T2V/I2V (Docker)
│   └── terraform/          # AWS infrastructure as code
│
└── research-suite/         # Quality assessment tools
    ├── qualifier/          # Objective quality metrics
    │   ├── cover_assessment/    # COVER (semantic+technical+aesthetic)
    │   ├── si-ti-assessment/    # ITU-T P.910 SI/TI metrics
    │   └── tv-l1_assessment/    # TV-L1 optical flow
    ├── p910/               # ITU-T P.910 subjective testing (web player)
    ├── p1401/              # ITU-T P.1401 validation framework
    └── data/               # Video datasets
        ├── source_videos/  # AI-generated test videos
        └── tests_videos/   # Test/validation videos
```

## Common Development Commands

### Infrastructure (Video Generation)

**AWS Deployment:**
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

**Connect to GPU instance:**
```bash
# Connection command output by terraform
# Example: aws ssm start-session --target <instance-id>
```

**Wan2.2 Container:**
```bash
cd infrastructure/wan22
./run.sh                     # Start Docker daemon
docker attach wan22-inference
./download_model.sh          # Download model weights (126GB)
cd wan2.2
python3 generate.py --task t2v-A14B --size 1280*720 \
  --ckpt_dir /workspace/wan2.2/checkpoints/Wan2.2-T2V-A14B \
  --prompt "Your video description"
```

**HunyuanVideo Container:**
```bash
cd infrastructure/hunyuanvideo
./run-t2v.sh
docker attach hunyuanvideo
cd HunyuanVideo
python3 sample_video.py --video-size 720 1280 --video-length 129 \
  --prompt "Your description" --use-cpu-offload
```

### Research Suite (Quality Assessment)

**COVER Assessment:**
```bash
cd research-suite/qualifier/cover_assessment
source venv/bin/activate
python src/cover_assessment.py --input-dir ../../data/source_videos
deactivate
```

**SI/TI Assessment:**
```bash
cd research-suite/qualifier/si-ti-assessment
source venv/bin/activate
python src/main.py --input ../../data/source_videos --output ./results
deactivate
```

**TV-L1 Optical Flow:**
```bash
cd research-suite/qualifier/tv-l1_assessment
source venv/bin/activate
python src/tv-l1_assessment.py --video ../../data/source_videos/video.mp4
deactivate
```

**P.910 Subjective Testing (Web Server):**
```bash
cd research-suite/p910/video-player
ln -s ../../data/source_videos videos  # Only first time
python3 -m http.server 8000
# Visit http://localhost:8000
```

**P.1401 Statistical Validation:**
```bash
cd research-suite/p1401
source venv/bin/activate

# Step 1: Compute MOS from votes
python src/ci95.py -i data/votes/ -o results/mos/mos_results.csv

# Step 2: Evaluate metrics against MOS
python src/mapping.py \
  --mos results/mos/mos_results.csv \
  --metrics cover_results.csv tv-l1_results.csv \
  --output results/p1401/

deactivate
```

## Architecture Notes

### Video Generation Pipeline
- **Wan2.2**: Autoregressive transformer (14B params, MoE), CPU offloading required
- **HunyuanVideo**: Diffusion transformer (13B+ params), supports flow-matching
- Both models require L40S (48GB) or H100 (80GB) GPUs
- Docker containers bake models and dependencies for reproducibility
- Terraform provisions AWS EC2 g6e.4xlarge or p5.4xlarge instances

### Quality Assessment Architecture

**COVER (Three-Branch Neural Network):**
- **Semantic Branch**: CLIP ViT-L/14 (512×512, 20 frames) → content understanding
- **Technical Branch**: Swin Transformer 3D (7×7 grid, 32×32 patches, 40 frames) → compression artifacts
- **Aesthetic Branch**: ConvNeXt Tiny (224×224, 40 frames) → visual appeal
- Overall score = semantic + technical + aesthetic (unbounded regression values)
- Model: `pretrained_weights/COVER.pth` (250MB, from official CVPR 2024 release)

**SI/TI (ITU-T P.910 Classical Metrics):**
- **SI**: stddev(Sobel(frame)) → spatial complexity/detail
- **TI**: stddev(frame_n - frame_{n-1}) → temporal activity/motion
- Uses ITU-R BT.601 grayscale conversion: `Y = 0.299*R + 0.587*G + 0.114*B`

**TV-L1 Optical Flow (Temporal Consistency):**
- Forward-backward consistency error
- Temporal warp error
- Motion magnitude variance
- Quality scores transformed via exponential decay: Q = exp(-alpha * error)

**P.910 Subjective Testing:**
- Web-based video player with ACR 5-point scale (1=Bad, 5=Excellent)
- Fisher-Yates randomization to prevent order bias
- Grey screen intervals (50% grey, 2s before/after) per ITU-T standard
- CSV export: ParticipantID, VideoIndex, Filename, Rating, Timestamp, ResponseTime

**P.1401 Validation Framework:**
- Third-order polynomial mapping: `MOS_predicted = a₀ + a₁x + a₂x² + a₃x³`
- Pearson/Spearman correlation coefficients
- RMSE and epsilon-insensitive RMSE* (discounts errors within CI95)
- Leave-One-Out Cross-Validation (LOOCV) for small datasets

## Important Implementation Details

### COVER Score Interpretation
- **Scores are unbounded regression values** (not percentages or probabilities)
- Negative scores are normal (indicate below-average quality)
- **Relative ranking** matters, not absolute values
- Example: semantic=-0.06, technical=0.87, aesthetic=0.35 → overall=1.16
- Compare within dataset, not across datasets

### Temporal Sampling
- COVER analyzes **entire video duration**, not snippets
- Semantic: 20 frames uniformly sampled across full timeline
- Technical/Aesthetic: 40 frames uniformly sampled
- Uses `UnifiedFrameSampler` with temporal fragments

### Virtual Environments
Each assessment tool has its own isolated Python 3.9 venv:
- `research-suite/qualifier/cover_assessment/venv/`
- `research-suite/qualifier/si-ti-assessment/venv/`
- `research-suite/qualifier/tv-l1_assessment/venv/`
- `research-suite/p1401/venv/`

**Always activate before running:**
```bash
source venv/bin/activate  # Run commands
deactivate                # Exit venv
```

### ARM64 Compatibility (macOS Apple Silicon)
- COVER uses PyAV wrapper instead of decord for cross-platform video reading
- Wrapper located in: `research-suite/qualifier/cover_assessment/src/cover_assessment.py`
- Maintains 100% compatibility with official COVER implementation

### Infrastructure State Management
- Terraform state files track AWS resource lifecycle
- **Never manually modify resources** created by Terraform
- Use `terraform destroy` to clean up (avoid orphaned resources)
- State files: `infrastructure/terraform/terraform.tfstate`

## Data Flow Workflow

1. **Generation Phase**: Generate AI videos from text prompts using Wan2.2/HunyuanVideo
2. **Qualification Phase**: Compute objective metrics (COVER, SI/TI, TV-L1) on generated videos
3. **Subjective Testing**: Collect human ratings via P.910 web interface (min 24 subjects/video)
4. **Validation Phase**: Correlate objective metrics with subjective MOS using P.1401 framework

## Critical Security Notes

- **Never commit** `.env` files (AWS credentials, API keys)
- **Never commit** Terraform state files with production credentials
- AWS credentials managed via IAM roles and SSM session manager (no SSH keys)
- GPU instances should have egress-only security groups

## Testing and Validation

### Test Videos Location
- `research-suite/data/tests_videos/` contains validation videos:
  - `test_real.mp4`: Real-world footage
  - `mov_circle.mp4`, `mov_circle+noise.mp4`: Synthetic motion tests
  - `static_scale.mp4`, `static_solidgrey.mp4`: Static tests

### Verification Commands
```bash
# Verify COVER installation
cd research-suite/qualifier/cover_assessment
source venv/bin/activate
python -c "import sys; sys.path.insert(0, 'src/cover_repo'); from cover.models import COVER; print('✓ COVER verified')"

# Verify GPU availability (on cloud instance)
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

## Common Issues and Solutions

### COVER Issues
- **"No module named 'cover'"**: Activate venv and ensure `cover_repo` is in Python path
- **Negative scores**: Normal behavior for unbounded regression (not an error)
- **Slow on CPU**: Expected, use GPU for production (CPU: ~10-15s/video, GPU: ~80ms/video)

### Infrastructure Issues
- **Terraform apply fails**: Check AWS credentials, verify region availability for GPU instances
- **Docker container won't start**: Verify NVIDIA Container Toolkit installed, check GPU passthrough
- **Model download fails**: Network timeout, large files (126GB for Wan2.2), retry or use wget

### Assessment Issues
- **Video format not supported**: Re-encode to H.264 MP4: `ffmpeg -i input.* -c:v libx264 -crf 18 output.mp4`
- **SI/TI plots missing**: Check Matplotlib backend, verify `--no-plots` flag not set
- **P.910 web player stuck**: Check browser console, verify videos/ symlink exists

## File Naming Conventions

- Video files: `source{source_id}_{video_id}.mp4` (e.g., `source1_1.mp4`, `source2_3.mp4`)
- Results CSVs: `<video_name>_si_ti.csv`, `p910_assessment_{participant_id}.csv`
- Model weights: `pretrained_weights/COVER.pth`, checkpoints in Docker volumes

## Performance Benchmarks

**COVER Inference:**
- NVIDIA A100: 79.37ms per video (4K, 30 frames)
- Apple M1/M2 CPU: ~8-12 seconds per 720p video

**Model Requirements:**
- Wan2.2 peak VRAM: ~60GB (standard), ~40GB (with offloading)
- HunyuanVideo peak VRAM: ~40GB (with CPU offloading)

## References

- ITU-T P.910 (10/2023): Subjective video quality assessment
- ITU-T P.1401 (01/2020): Objective metric validation methodology
- COVER paper: He et al., CVPR 2024 (AIS Workshop Winner)
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- HunyuanVideo: https://github.com/Tencent-Hunyuan/HunyuanVideo

## Notes for Claude Code

- Each qualifier tool is self-contained with its own venv and dependencies
- Docker containers use bind mounts for `/opt/outputs` (persistent storage)
- COVER implementation verified against official repository (100% identical architecture)
- P.1401 RMSE* denominator: (N-4) for third-order polynomial degrees of freedom
- Terraform user_data.sh bootstraps GPU drivers and Docker runtime
- All objective metrics output to `research-suite/qualifier/*/results/`

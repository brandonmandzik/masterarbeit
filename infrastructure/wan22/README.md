# Wan2.2 Infrastructure

## Overview

This infrastructure project provides a containerized deployment environment for Wan2.2, an open-source video generation framework that produces cinematic-quality videos from text and image inputs. The project packages Wan2.2's diffusion-based models (T2V-A14B and I2V-A14B) into GPU-accelerated Docker containers, enabling reproducible inference on NVIDIA hardware. It exists to simplify deployment of large-scale video generative models (14B active parameters) by abstracting dependency management, CUDA environment configuration, and multi-GPU orchestration behind a standardized container interface.

## Features

Based strictly on the project references, this infrastructure provides:

- **Dual-Model Support**: Packages both text-to-video (T2V-A14B) and image-to-video (I2V-A14B) models with 14B active parameters
- **GPU-Accelerated Containers**: Built on HuggingFace's PyTorch-GPU base image with CUDA support for L40S/H100 architectures (ARCH_LIST 8.9, 9.0)
- **Automated Model Downloads**: HuggingFace Hub CLI integration for pulling 126GB model checkpoints (T2V-A14B, I2V-A14B)
- **Multi-Resolution Output**: Supports 480P and 720P video generation at standard frame rates
- **Optimized Memory Management**: Configures `PYTORCH_ALLOC_CONF=expandable_segments:True` and unlimited memlock for large model loading
- **Privileged GPU Access**: Full NVIDIA device access with `ipc: host`, `uts: host`, and `seccomp=unconfined` for maximum performance
- **Persistent Output Storage**: Volume mount at `/opt/outputs` for generated video artifacts
- **Registry-Agnostic Deployment**: Supports Docker Hub, GitHub Container Registry, and Amazon ECR

## Quick Start

**Prerequisites**: Docker with NVIDIA Container Toolkit, 80GB+ GPU VRAM (single GPU) or multi-GPU setup

### Install

```bash
# Clone this repository
cd /path/to/wan22

# Configure Docker registry (edit .env)
cp .env.example .env
# Edit DOCKER_IMAGE to your registry endpoint
```

### Run

```bash
# Start container daemon
./run.sh

# Attach to running container
docker attach wan22-inference

# Download models inside container
# Options: t2v, i2v, ti2va (both T2V+I2V)
/workspace/download_model.sh ti2va
```

### Verify

```bash
# Inside container - test T2V inference
cd /workspace/wan2.2
python generate.py --task t2v-A14B --size 1280*720 \
  --ckpt_dir ./checkpoints/Wan2.2-T2V-A14B \
  --offload_model True \
  --prompt "Cinematic shot of a sunset over mountains"

# Check output
ls /workspace/wan2.2/outputs/
```

Successful execution produces a 5-second video file in `/workspace/wan2.2/outputs/` (accessible at host path `/opt/outputs`).

## Wan2.2-Fun-5B-Control via VideoX-Fun Docker

The Wan2.2-Fun-5B-Control model provides an alternative deployment path through the VideoX-Fun framework, offering a lighter-weight control model option. This approach uses Alibaba PAI's official Docker image with pre-configured dependencies for the VideoX-Fun ecosystem.

### Prerequisites

- NVIDIA GPU with driver and CUDA environment correctly installed
- Docker with NVIDIA Container Toolkit
- 200GB shared memory allocation capability
- Network access to Alibaba Container Registry (China region)

### Docker Setup

**Pull the VideoX-Fun Image**:
```bash
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

**Run Container**:
```bash
docker run -it -p 7860:7860 --network host --gpus all \
  --security-opt seccomp:unconfined --shm-size 200g \
  mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

**Configuration Notes**:
- `--network host`: Required for Gradio web interface access
- `--shm-size 200g`: Large shared memory allocation for video processing
- `-p 7860:7860`: Exposes Gradio default port (redundant with host networking)

### Installation Inside Container

**Clone VideoX-Fun Repository**:
```bash
git clone https://github.com/aigc-apps/VideoX-Fun.git
cd VideoX-Fun
```

**Prepare Model Directories**:
```bash
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model
```

### Model Download

Download the CogVideoX-Fun-V1.1-5b-InP model from either HuggingFace or ModelScope (China mirror) based on network accessibility:

- **HuggingFace**: https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP
- **ModelScope**: https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP

**Download Example** (using HuggingFace CLI inside container):
```bash
# Install huggingface-hub if not present
pip install -U huggingface-hub

# Download CogVideoX-Fun 5B model
huggingface-cli download alibaba-pai/CogVideoX-Fun-V1.1-5b-InP \
  --local-dir models/Diffusion_Transformer/CogVideoX-Fun-V1.1-5b-InP
```

### VideoX-Fun vs. Direct Wan2.2 Approach

| Aspect | VideoX-Fun Docker | Direct Wan2.2 Container |
|--------|------------------|------------------------|
| **Base Image** | Alibaba PAI CogVideoX image | HuggingFace Transformers PyTorch-GPU |
| **Framework** | VideoX-Fun (multi-model) | Official Wan2.2 repository |
| **Model Support** | CogVideoX + Wan variants | Wan2.2 T2V/I2V/Control only |
| **Setup Complexity** | Manual git clone + model download | Automated via `download_model.sh` |
| **Web Interface** | Gradio UI (port 7860) | CLI-only inference |
| **Control Model Size** | 5B (CogVideoX-Fun) | Standard models only |
| **Use Case** | Experimentation, web demos | Production inference pipelines |

### When to Use This Approach

**Advantages**:
- **Lighter Model**: CogVideoX-Fun-5B-InP requires less VRAM than standard T2V/I2V models
- **Gradio Interface**: Web-based GUI for interactive generation
- **Multi-Framework**: Access to both CogVideoX and Wan model families
- **Official Support**: Maintained by Alibaba PAI team

**Limitations**:
- **Manual Setup**: No automated model download scripts
- **Network Dependency**: Requires access to China-region container registry
- **Less Integration**: Not integrated with this project's output management

### References

- **Official Documentation**: https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control/blob/main/README_en.md
- **VideoX-Fun Repository**: https://github.com/aigc-apps/VideoX-Fun
- **Model Cards**: See table above for HuggingFace/ModelScope links

## Architecture

### System Overview

The infrastructure implements a three-tier architecture: container orchestration (Docker Compose), runtime environment (PyTorch + CUDA), and model inference layer (Wan2.2 framework). The containerization strategy isolates complex dependencies (flash-attention, transformers, diffusers) while exposing GPU resources through NVIDIA Container Toolkit passthrough.

**Data Flow**: User prompt → Docker Compose entrypoint → Container runtime → Wan2.2 Python inference script → Model checkpoint loading → VAE encoding → Diffusion denoising (MoE experts) → VAE decoding → Output video file.

**Component Hierarchy**:
- **Host Layer**: NVIDIA drivers, Docker daemon, volume mounts (`/opt/outputs`)
- **Container Layer**: Ubuntu base, Python 3.10+, PyTorch 2.4.0+, CUDA 12.1
- **Application Layer**: Wan2.2 repository, HuggingFace libraries, model checkpoints
- **Model Layer**: T2V/I2V transformers, VAE encoder/decoder, MoE denoising networks

**Key Decision Points**:
- Container mode selection (build vs. pull) determined by `run.sh` reading `.env` configuration
- Model download triggered manually via `download_model.sh` with type selection (t2v/i2v/ti2va)
- GPU resource allocation controlled by Docker Compose `deploy.resources.reservations`

**Expert Activation**: The MoE architecture activates exactly 14B of the 27B total parameters per inference step. The high-noise expert establishes video layout during early denoising timesteps, then the system transitions to the low-noise expert for detail refinement based on signal-to-noise ratio thresholds.

**Memory Optimization Path**: `--offload_model True` flag triggers layer-wise CPU offloading, `--convert_model_dtype` enables mixed-precision computation, and `--t5_cpu` moves T5 text encoder to CPU memory, collectively reducing VRAM requirements from 80GB to consumer GPU levels (24GB+).

## Technologies

### Core Frameworks

**PyTorch 2.4.0+**: Deep learning framework providing automatic differentiation, GPU acceleration via CUDA, and distributed training primitives. The infrastructure requires torch.distributed for FSDP (Fully Sharded Data Parallel) multi-GPU inference.

**HuggingFace Transformers**: Provides pretrained model architectures (T5 text encoders, diffusion transformers) and tokenization pipelines. The `transformers-pytorch-gpu` base image bundles CUDA-optimized builds with matched driver versions.

**Diffusion Models**: Wan2.2 uses latent diffusion, where a VAE compresses videos into latent space (16×16×4 compression ratio), then a denoising network iteratively removes Gaussian noise over T timesteps to generate structured video data. The reverse diffusion process is conditioned on text embeddings from T5 or visual features from image encoders.

**Mixture-of-Experts (MoE)**: Architectural pattern splitting the 27B parameter model into specialized sub-networks. Only 14B parameters activate per forward pass, selected by a gating mechanism based on input features. This expands model capacity without proportional compute costs compared to dense 27B models.

### Infrastructure Components

**Docker Compose**: Declarative multi-container orchestration defining service dependencies, volume mounts, and resource constraints. The `docker-compose.yml` specifies GPU reservation policies and runtime configurations.

**NVIDIA Container Toolkit**: Exposes GPU devices inside containers via `--gpus all` flag and `nvidia` runtime. Manages CUDA library injection, driver compatibility, and device isolation.

**Flash-Attention**: Memory-efficient attention mechanism reducing O(n²) complexity to O(n) for long sequence processing. Wan2.2 requires this for video frame attention across temporal dimensions. The Dockerfile compiles flash-attention with Ninja parallelization targeting compute capabilities 8.9 (L40S) and 9.0 (H100).

**HuggingFace Hub CLI**: Distributed file download tool with resume capability for the 126GB model checkpoints. Uses parallel chunk downloads and atomic writes to prevent corruption.

### Model Architecture Foundations

**VAE (Variational Autoencoder)**: Probabilistic encoder-decoder compressing high-dimensional video (e.g., 720p RGB frames) into low-dimensional latent representations. Wan2.2-VAE achieves 16×16×4 spatial-temporal compression, enabling diffusion in computationally tractable latent space.

**Denoising Diffusion Probabilistic Models (DDPM)**: Generative model class learning to reverse a noise injection process. Training adds Gaussian noise over T steps; inference runs the learned reverse process to denoise random noise into video samples. Conditional variants accept text/image inputs to guide generation.

**T5 Text Encoder**: Transformer-based language model converting text prompts into continuous embeddings. Wan2.2 uses T5 representations as conditioning signals for the diffusion model's cross-attention layers.

**Control Conditioning**: Techniques for guiding video generation with structural inputs (edges, poses, depth). The Fun-A14B-Control variant concatenates control features with latent representations during diffusion denoising.

## Foundation Knowledge

To understand this project, you need technical grounding in:

### Machine Learning Fundamentals

- **Deep Neural Networks**: Backpropagation, gradient descent, activation functions, loss functions
- **Transformer Architecture**: Self-attention mechanisms, multi-head attention, positional encodings, layer normalization
- **Probabilistic Modeling**: Latent variable models, variational inference, Kullback-Leibler divergence
- **Generative Models**: Autoencoders, GANs, diffusion models, likelihood-based training

### Diffusion Model Theory

- **Forward Diffusion Process**: Markov chain progressively adding Gaussian noise q(x_t|x_{t-1}) over timesteps
- **Reverse Process**: Learning p(x_{t-1}|x_t) to denoise, parameterized by neural networks predicting noise ε
- **Training Objective**: Minimizing simplified variational lower bound E[||ε - ε_θ(x_t, t)||²]
- **Sampling**: Iterative denoising from pure noise x_T ~ N(0,I) to data x_0 following learned p_θ
- **Conditioning Mechanisms**: Classifier-free guidance, cross-attention on text embeddings, concatenation of image features

### Video Processing Concepts

- **Temporal Consistency**: Maintaining smooth motion across frames, preventing flickering or jittering
- **Optical Flow**: Motion representation between consecutive frames, critical for realistic video dynamics
- **3D Convolutions**: Spatiotemporal kernels processing (T, H, W) video volumes vs. 2D image (H, W) operations
- **Frame Rate & Resolution Trade-offs**: 480P vs. 720P impacts memory (4x pixels), compute (16x for attention)

### Distributed Systems

- **Data Parallelism**: Replicating model across GPUs, splitting batch dimensions, gradient synchronization
- **Model Parallelism**: Partitioning model layers across devices when single GPU insufficient for 27B parameters
- **FSDP**: Fully Sharded Data Parallel - shards model parameters, gradients, and optimizer states across ranks
- **Pipeline Parallelism**: Splits model into stages, processes micro-batches in pipelined fashion
- **DeepSpeed Ulysses**: Sequence parallelism for long context transformers, partitions attention computation across GPUs

### Infrastructure Skills

- **Docker**: Image layering, multi-stage builds, volume persistence, network modes, privilege escalation
- **CUDA Programming**: GPU memory hierarchy (global, shared, registers), kernel launch, stream synchronization
- **Linux System Administration**: Process management, ulimits, seccomp profiles, IPC namespaces
- **Python Environment Management**: pip requirements, dependency resolution, virtual environments, package compilation

### Mathematical Foundations

- **Linear Algebra**: Matrix operations, eigenvalues, singular value decomposition (used in attention)
- **Probability Theory**: Gaussian distributions, conditional probability, Bayes' rule, stochastic processes
- **Numerical Optimization**: Stochastic gradient descent, Adam optimizer, learning rate schedules
- **Information Theory**: Entropy, KL divergence, mutual information (VAE loss terms)

## References

All information derived from the following project references:

- **Wan2.2 Repository**: https://github.com/Wan-Video/Wan2.2
- **T2V Model Card**: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
- **I2V Model Card**: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- **Fun-5B-Control Model**: https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control
- **Base Container Image**: https://hub.docker.com/r/huggingface/transformers-pytorch-gpu
- **Research Paper**: https://arxiv.org/pdf/2503.20314

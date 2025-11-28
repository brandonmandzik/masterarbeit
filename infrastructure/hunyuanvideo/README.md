# HunyuanVideo Infrastructure

## Overview

This repository provides Docker-based deployment infrastructure for **HunyuanVideo**, a 13-billion parameter open-source video generation foundation model developed by Tencent. The infrastructure enables GPU-accelerated inference for both Text-to-Video (T2V) and Image-to-Video (I2V) models through simplified containerized deployment scripts.

HunyuanVideo achieves state-of-the-art performance among open-source video generation models, ranking first in professional evaluations with 95.7% visual quality, 66.5% motion quality, and 61.8% text alignment scores across 1,533 test prompts.

## Features

### Video Generation Capabilities
- **Text-to-Video (T2V)**: Generate videos from text prompts with superior text alignment
- **Image-to-Video (I2V)**: Animate static images into dynamic video sequences
- **Multi-Resolution Support**: 540p and 720p output resolutions
- **Flexible Aspect Ratios**: 9:16, 16:9, 4:3, 3:4, and 1:1
- **Extended Frame Generation**: Default 129-frame output capability
- **Prompt Enhancement**: Integrated rewrite model with Normal (accuracy-focused) and Master (visual quality-focused) modes

### Infrastructure Features
- **GPU-Optimized Containers**: Pre-configured Docker images with CUDA 12 support
- **Dual Deployment Scripts**: Separate launchers for T2V and I2V models
- **Runtime Optimization**: Stack size (64MB) and memory lock configurations for GPU efficiency
- **Container Isolation**: Host network, UTS, and IPC sharing for performance
- **Simplified Setup**: Single-script deployment with automated image pulling

## Quick Start

### Prerequisites
- Docker (20.10+)
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Minimum 60GB VRAM (for 720p generation) or 45GB VRAM (lower resolutions)

### Running Text-to-Video (T2V)
```bash
./run-t2v.sh
```

The script pulls `hunyuanvideo/hunyuanvideo:cuda_12` and launches a daemon container named `hunyuanvideo`.

### Running Image-to-Video (I2V)
```bash
./run-i2v.sh
```

The script pulls `hunyuanvideo/hunyuanvideo-i2v:cuda12` and launches a daemon container named `hunyuanvideo-i2v`.

### Verification
Monitor container logs:
```bash
sudo docker logs -f hunyuanvideo        # For T2V
sudo docker logs -f hunyuanvideo-i2v    # For I2V
```

Attach to container:
```bash
sudo docker attach hunyuanvideo         # For T2V
sudo docker attach hunyuanvideo-i2v     # For I2V
```

## Architecture

### High-Level System Architecture

HunyuanVideo employs a three-layer architecture: (1) 3D Variational Autoencoder for spatial-temporal compression, (2) Unified Diffusion Transformer for generation, and (3) Dual Text Encoder System for semantic understanding. The infrastructure wraps these components in GPU-optimized Docker containers for streamlined deployment.

**Component Breakdown**:
1. **Text Encoding**: Multimodal LLM generates token-level embeddings; CLIP extracts global features; bidirectional refiner enhances representations beyond causal attention limits
2. **Compression**: 3D VAE reduces video tensors to efficient latent representations (4× temporal, 8× spatial, 16× channel compression)
3. **Dual-Stream Processing**: 20 transformer blocks process video and text tokens independently
4. **Single-Stream Fusion**: 40 transformer blocks perform multimodal fusion on concatenated tokens
5. **Flow Matching**: Iterative denoising via velocity prediction guided by timestep-shifted Euler ODE solver
6. **Decoding**: 3D VAE reconstructs high-resolution video frames from denoised latent space

## Technologies

### Docker
**Containerization platform** providing reproducible, isolated deployment environments. Ensures consistent runtime across systems by packaging application dependencies, libraries, and configurations. Theoretical foundation: OS-level virtualization using Linux kernel features (cgroups, namespaces) for resource isolation without hypervisor overhead.

### NVIDIA CUDA
**Parallel computing platform** enabling GPU-accelerated computation. Exposes GPU architecture (streaming multiprocessors, thousands of cores) through C/C++ extensions. Theoretical foundation: SIMT (Single Instruction, Multiple Thread) execution model where thread blocks execute identical instructions on different data elements simultaneously.

### PyTorch
**Deep learning framework** for tensor computation and automatic differentiation. Builds computational graphs dynamically during execution (define-by-run). Theoretical foundation: Reverse-mode automatic differentiation computes gradients via backpropagation through chain rule application on directed acyclic graphs.

### Diffusion Transformers (DiT)
**Generative model architecture** combining diffusion models with transformer blocks. Uses self-attention mechanisms for spatial-temporal feature modeling. Theoretical foundation: Flow matching framework where model predicts velocity field ut = dxt/dt guiding samples from noise distribution to data distribution via learned ODE trajectories.

### 3D Variational Autoencoder (VAE)
**Neural compression model** encoding high-dimensional video into compact latent representations. Encoder maps inputs to probabilistic distributions; decoder reconstructs from sampled latents. Theoretical foundation: Variational inference maximizes evidence lower bound (ELBO) balancing reconstruction accuracy (L1 + perceptual loss) with latent regularization (KL divergence).

### Flash Attention
**Optimized attention algorithm** reducing memory complexity from O(N²) to O(N) for sequence length N. Partitions attention computation into blocks processed sequentially with recomputation. Theoretical foundation: Tiling strategy exploiting GPU memory hierarchy (SRAM vs HBM) to minimize costly memory transfers during attention score calculation.

### 3D Rotary Position Embedding (RoPE)
**Position encoding method** for multi-resolution, variable-duration video. Applies rotation matrices to feature dimensions based on temporal (T), height (H), width (W) coordinates. Theoretical foundation: Geometric interpretation where relative positions correspond to rotation angles in complex-valued feature space, enabling length extrapolation beyond training sequences.

### xDiT
**Parallel processing framework** for distributed transformer inference. Implements 5D parallelism: tensor (matrix distribution), sequence (input slicing), context (ring attention), data (batch distribution), and pipeline (layer distribution). Theoretical foundation: Reduces single-GPU memory requirements via model/activation partitioning across nodes with optimized communication patterns.

## Foundation Knowledge

Understanding this project requires background in:

### Deep Learning Fundamentals
- **Neural Networks**: Multi-layer perceptrons, activation functions, universal approximation
- **Backpropagation**: Gradient computation via chain rule, optimization algorithms (Adam, SGD)
- **Training Dynamics**: Loss functions, overfitting/regularization, batch normalization

### Diffusion Models
- **Forward Process**: Noise scheduling, variance schedules, Markov chain formulation
- **Reverse Process**: Learned denoising, score matching, sampling algorithms
- **Flow Matching**: Continuous normalizing flows, ODE/SDE formulations, velocity prediction

### Transformer Architecture
- **Self-Attention**: Query-key-value mechanism, scaled dot-product attention
- **Multi-Head Attention**: Parallel attention with learned projection matrices
- **Position Encoding**: Absolute vs. relative positional information injection

### GPU Computing
- **CUDA Programming**: Kernel execution, thread hierarchy (grids, blocks, threads)
- **Memory Hierarchy**: Global memory, shared memory, registers, coalescing
- **Performance Optimization**: Occupancy, latency hiding, memory bandwidth utilization

### Docker Containerization
- **Image Layers**: Union file systems, layer caching, Dockerfile instructions
- **Runtime Isolation**: Namespaces (PID, network, mount), cgroups (resource limits)
- **GPU Passthrough**: NVIDIA Container Toolkit, device mapping, driver compatibility

### Video Processing
- **Frame Representation**: RGB channels, resolution, aspect ratios, frame rates
- **Temporal Modeling**: Motion vectors, optical flow, temporal coherence
- **Compression**: Spatial vs. temporal compression, codec principles, latent representations

### Variational Inference
- **Evidence Lower Bound (ELBO)**: Decomposition into reconstruction and KL terms
- **Reparameterization Trick**: Gradient estimation through stochastic nodes
- **Latent Variable Models**: Encoder-decoder architectures, posterior inference

### Text-to-Image/Video Generation
- **Conditional Generation**: Classifier-free guidance, text embeddings, cross-attention
- **Multi-Modal Alignment**: Vision-language models, contrastive learning (CLIP)
- **Prompt Engineering**: Semantic parsing, attribute extraction, negative prompting

## References

All technical content derived from the following official project references:

- **GitHub Repository (T2V)**: https://github.com/Tencent-Hunyuan/HunyuanVideo
- **GitHub Repository (I2V)**: https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
- **Hugging Face Model Hub (T2V)**: https://huggingface.co/tencent/HunyuanVideo
- **Hugging Face Model Hub (I2V)**: https://huggingface.co/tencent/HunyuanVideo-I2V
- **Research Paper**: https://arxiv.org/abs/2412.03603

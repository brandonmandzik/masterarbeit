# CLIP Score Assessment

## Overview

Minimal tool for calculating CLIP similarity scores between video frames and reference images/text. Part of a video quality assessment research suite for evaluating AI-generated videos using objective semantic metrics. Provides a lightweight, Python-based implementation for measuring visual-semantic alignment via OpenAI's CLIP model.

**Purpose**: Objective semantic similarity metric for AI-generated video quality evaluation, enabling quantitative comparison of video content against reference images or text descriptions.

## Features

- **Video-to-Image Scoring**: Compute mean/median CLIP similarity across all video frames against a reference image
- **Video-to-Text Scoring**: Measure semantic alignment between video content and text prompts
- **Image-to-Image Scoring**: Direct CLIP similarity between two static images
- **Statistical Analysis**: Verbose mode provides per-frame scores, min/max/mean/median/std statistics
- **Device Flexibility**: Automatic GPU/CPU detection with manual override support
- **Comprehensive Frame Analysis**: Processes all frames in video (no temporal sampling)

## Quick Start

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage Examples

**Video-to-Image:**
```bash
python clip_score.py --video path/to/video.mp4 --image path/to/reference.jpg
```

**Video-to-Text:**
```bash
python clip_score.py --video path/to/video.mp4 --text "a person walking in the park"
```

**Image-to-Image:**
```bash
python clip_score.py --image path/to/image1.jpg --image2 path/to/image2.jpg
```

**Options:**
- `--device cuda` or `--device cpu` (auto-detects by default)
- `--verbose` or `-v` (enable detailed frame-by-frame output)

### Expected Output

```
Using device: cuda
Video: path/to/video.mp4
Reference image: path/to/reference.jpg

Results:
  Frames analyzed: 240
  Mean CLIP Score (video-to-image): 0.8523
  Median CLIP Score (video-to-image): 0.8541
```

## Architecture

### High-Level Design

The system operates as a sequential pipeline: input media is loaded, preprocessed through CLIP transforms, encoded into a shared embedding space via Vision Transformer, compared using cosine similarity, and aggregated into mean/median scores.

**Data Flow:**
```
┌─────────────────────┐
│ Video/Image Input   │
└──────────┬──────────┘
           ↓
┌──────────────────────┐
│ Frame Extraction     │  (PyAV container decode)
│ (video only)         │
└──────────┬───────────┘
           ↓
┌──────────────────────────────┐
│ CLIP Preprocessing           │  (Resize, normalize, tensor)
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│ CLIP ViT-B/32 Encoder        │  (Vision/Text transformer)
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│ L2 Normalization             │  (Unit vector projection)
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│ Cosine Similarity            │  (Dot product of features)
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│ Aggregation (mean/median)    │
└──────────┬───────────────────┘
           ↓
┌──────────────────────────────┐
│ Terminal Output              │
└──────────────────────────────┘
```

### Module Interactions

```
clip_score.py
│
├─ main()
│   ├─ argparse.ArgumentParser()  → Parse CLI args
│   ├─ Route to comparison mode
│   └─ Print formatted results
│
├─ calculate_clip_score_image(video, image, device, verbose)
│   ├─ clip.load("ViT-B/32")     → Load pretrained model
│   ├─ extract_frames(video)     → PyAV decode all frames
│   ├─ model.encode_image()      → Encode reference + frames
│   ├─ Cosine similarity loop    → Frame-by-frame comparison
│   └─ np.mean(scores)           → Aggregate statistics
│
├─ calculate_clip_score_text(video, text, device, verbose)
│   ├─ clip.tokenize(text)       → Tokenize prompt (77 token limit)
│   ├─ model.encode_text()       → Encode text features
│   └─ [Same frame processing]   → Image encoding + similarity
│
├─ calculate_clip_score_image_to_image(img1, img2, device, verbose)
│   ├─ PIL.Image.open()          → Load both images
│   ├─ model.encode_image()      → Dual encoding
│   └─ Cosine similarity         → Single comparison
│
└─ extract_frames(video_path)
    ├─ av.open(video_path)       → Open container
    ├─ container.decode(video=0) → Decode video stream
    └─ frame.to_image()          → Convert to PIL Image
```

## Technologies

### PyTorch (`torch`, `torchvision`)
Deep learning framework providing tensor operations and neural network inference. Core concept: **automatic differentiation** via computational graphs and **GPU acceleration** through CUDA kernels. Enables efficient batch processing and hardware-optimized linear algebra operations (matrix multiplications, convolutions).

### OpenAI CLIP
**Contrastive Language-Image Pre-training** model from Radford et al. (2021). This implementation uses **ViT-B/32**: a Vision Transformer with Base architecture (12 layers, 512 hidden dims) processing 32×32 pixel patches. Core concept: **joint embedding space** learned via contrastive loss, where semantically similar images and text are pulled closer in high-dimensional space, enabling zero-shot cross-modal retrieval.

### PyAV (`av`)
Pythonic bindings for **FFmpeg** multimedia framework. Provides low-level access to video container formats (MP4, AVI, MKV) and codec operations. Core concept: **demuxing** (separating video/audio streams from container) and **frame decoding** (decompressing codec data into raw pixel arrays). Enables frame-accurate video processing without spawning subprocesses.

### NumPy
Array computing library for numerical operations. Used for statistical aggregation (mean, median, standard deviation) over score arrays. Core concept: **vectorized operations** on homogeneous data types, avoiding Python loops for performance.

### Cosine Similarity
Metric for measuring angular distance between vectors in embedding space:

```
cos(θ) = (A · B) / (||A|| ||B||)
```

- **Range**: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite
- **Implementation**: Dot product of L2-normalized feature vectors
- **Core concept**: Normalized dot product measures semantic similarity independent of magnitude, focusing on directional alignment in learned embedding space

## Foundation Knowledge

Prerequisites for understanding this implementation:

**Linear Algebra**
- Dot products and vector norms
- Cosine similarity and angular distance metrics
- High-dimensional vector spaces (512-dim CLIP embeddings)

**Deep Learning Fundamentals**
- Neural network architectures and forward propagation
- Feature extraction via pretrained models
- Embedding spaces and learned representations

**Vision Transformers (ViT)**
- Patch-based image processing (32×32 patches)
- Self-attention mechanisms for spatial relationships
- Positional embeddings for patch ordering

**Contrastive Learning**
- Metric learning via pairwise/triplet comparisons
- Contrastive loss functions (InfoNCE)
- Cross-modal alignment (vision-language models)

**Video Processing**
- Frame extraction and temporal sampling
- Video container formats (MP4, AVI)
- Codec compression and decoding

## References

- **OpenAI CLIP**: https://github.com/openai/CLIP
  - Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
- **CLIPScore**: https://github.com/jmhessel/clipscore
  - Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image Captioning" (EMNLP 2021)

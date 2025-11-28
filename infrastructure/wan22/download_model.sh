#!/bin/bash
# Download script for Wan2.2 models
# Usage: /download_model.sh [t2v|i2v|ti2va]
# Run this manually inside the container when you want to download models

set -e

MODEL_TYPE="${1:-ti2va}"

echo "=== Wan2.2 Model Download ==="

# Download T2V if requested (t2v or ti2va)
if [[ "$MODEL_TYPE" == "t2v" || "$MODEL_TYPE" == "ti2va" ]]; then
    if [ ! -d "/workspace/wan2.2/checkpoints/Wan2.2-T2V-A14B" ]; then
        echo "Downloading T2V model (~126GB, may take ~15 minutes)..."
        hf download Wan-AI/Wan2.2-T2V-A14B --local-dir /workspace/wan2.2/checkpoints/Wan2.2-T2V-A14B
        echo "T2V model download complete!"
    else
        echo "T2V model found in cache. Skipping."
    fi
fi

# Download I2V if requested (i2v or ti2va)
if [[ "$MODEL_TYPE" == "i2v" || "$MODEL_TYPE" == "ti2va" ]]; then
    if [ ! -d "/workspace/wan2.2/checkpoints/Wan2.2-I2V-A14B" ]; then
        echo "Downloading I2V model (~126GB, may take ~15 minutes)..."
        hf download Wan-AI/Wan2.2-I2V-A14B --local-dir /workspace/wan2.2/checkpoints/Wan2.2-I2V-A14B
        echo "I2V model download complete!"
    else
        echo "I2V model found in cache. Skipping."
    fi
fi

echo "=== Download Complete ==="

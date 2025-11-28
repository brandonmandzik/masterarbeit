#!/bin/bash
# Wan2.2 Docker daemon startup script
# Usage: ./run.sh

set -e

# Load configuration
if [ -f .env ]; then
    source .env
else
    echo "Warning: .env file not found. Using defaults."
    DOCKER_IMAGE="ghcr.io/your-org/wan22-t2v:latest"
fi

echo "=== Wan2.2 Docker Setup ==="
echo "Image: $DOCKER_IMAGE"
echo ""

# Pull image
echo "Pulling image from registry..."
docker compose pull

# Start daemon container
echo "Starting daemon container..."
docker compose up -d wan22

echo ""
echo "=== Container Running ==="
echo "sudo docker logs -f wan22-inference"
echo "sudo docker attach wan22-inference"
echo "./download_model.sh [t2v|i2v|ti2va]  # to download models manually"
echo ""

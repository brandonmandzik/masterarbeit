#!/bin/bash
# EC2 User Data Script - Simplified for Docker-native approach
# No EBS volume mounting needed - models are baked into Docker image

set -e

# Log all output
exec > >(tee /var/log/user_data.log)
exec 2>&1

echo "Starting user data script..."

echo "Installing system dependencies..."
apt-get update
# Wait for dpkg lock to be released (handles unattended-upgrades)
echo "Waiting for dpkg lock to be available..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  echo "Waiting for other package managers to finish..."
  sleep 5
done
apt-get install -y sysstat 

# Install docker-compose if not already installed
if ! command -v docker-compose &> /dev/null; then
  echo "Installing docker-compose..."
  curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  chmod +x /usr/local/bin/docker-compose
fi

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Verify NVIDIA Docker runtime
echo "Verifying NVIDIA Docker runtime..."

# Test 1: Host GPU access
echo "Testing host GPU access..."
nvidia-smi

# Test 2: Docker container GPU access (critical for Wan2.2)
echo "Testing Docker container GPU access..."
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

echo "User data script completed successfully!"

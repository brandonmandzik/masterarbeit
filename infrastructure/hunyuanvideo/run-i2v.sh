#!/bin/bash
set -e

echo "=== HunyuanVideo I2V Docker Setup ==="

# Pull image
echo "Pulling image from registry..."
docker pull hunyuanvideo/hunyuanvideo-i2v:cuda12

# Start daemon container
echo "Starting daemon container..."
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo-i2v --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo-i2v:cuda12

echo ""
echo "=== Container Running ==="
echo "Sanity & attach command:"
echo "sudo docker logs -f hunyuanvideo-i2v"
echo "sudo docker attach hunyuanvideo-i2v"
echo ""
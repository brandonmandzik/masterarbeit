#!/bin/bash
set -e

echo "=== HunyuanVideo T2V Docker Setup ==="

# Pull image
echo "Pulling image from registry..."
docker pull hunyuanvideo/hunyuanvideo:cuda_12

# Start daemon container
echo "Starting daemon container..."
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo:cuda_12

echo ""
echo "=== Container Running ==="
echo "Sanity & attach command:"
echo "sudo docker logs -f hunyuanvideo"
echo "sudo docker attach hunyuanvideo"
echo ""
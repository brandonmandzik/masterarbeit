#!/bin/bash

set -e

# Clone COVER repository
git clone https://github.com/vztu/COVER src/cover_repo

# Download pretrained weights
mkdir -p src/pretrained_weights
curl -L https://github.com/vztu/COVER/raw/release/Model/COVER.pth -o src/pretrained_weights/COVER.pth

# Replace decord with av in requirements.txt
sed -i.bak 's/decord/av/g' src/cover_repo/requirements.txt

echo "Setup complete!"

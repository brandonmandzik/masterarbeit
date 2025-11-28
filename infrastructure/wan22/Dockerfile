# ============================================
# Lightweight Dockerfile for Wan2.2 A14B model
# ============================================

FROM huggingface/transformers-pytorch-gpu:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Wan2.2 repository
WORKDIR /workspace
RUN git clone https://github.com/Wan-Video/Wan2.2.git /workspace/wan2.2

# Install Python dependencies
WORKDIR /workspace/wan2.2

# Install flash-attn with maximum parallelization (matches user_data_ami_setup_14b.sh)
# Ninja enables parallel builds (13-16 cores), TORCH_CUDA_ARCH_LIST optimizes for L40S/H100 only
RUN pip install --no-cache-dir psutil packaging ninja
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_s2v.txt
RUN pip install --no-cache-dir -r requirements_animate.txt
RUN pip uninstall -y peft
RUN pip install --no-cache-dir "peft==0.17.1"

# Install HuggingFace CLI for model downloads
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Set environment variables for HuggingFace cache
ENV HF_HOME=/workspace/wan2.2
ENV TRANSFORMERS_CACHE=/workspace/wan2.2

# Create Wan2.2 directory for checkpoints and outputs
RUN mkdir -p /workspace/wan2.2/checkpoints /workspace/wan2.2/outputs

# Copy download script (user executes manually)
COPY download_model.sh /workspace/download_model.sh
RUN chmod +x /workspace/download_model.sh

# Set working directory
WORKDIR /workspace

# Start interactive bash shell
CMD ["bash"]

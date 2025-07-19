# Multi-stage Dockerfile for DeepDiscord Training Module
# Optimized for Dolphin-2.9-Llama-3-8B with QLoRA and Unsloth

# Base stage with CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app/training_module

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific CUDA version
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install transformers and core ML libraries
RUN pip install transformers>=4.36.0 \
    datasets>=2.14.0 \
    accelerate>=0.25.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0

# Install Flash Attention (requires compilation)
RUN pip install flash-attn --no-build-isolation

# Install remaining requirements
RUN pip install -r requirements.txt

# Install Unsloth for faster training (optional but recommended)
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "Unsloth installation failed - continuing without it"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed data/cache checkpoints experiments logs

# Set permissions
RUN chmod +x scripts/*.py || true
RUN chmod +x scripts/*.sh || true

# Expose ports for monitoring
EXPOSE 8888 6006

# Development command (Jupyter + TensorBoard)
CMD ["bash", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/training_module & tensorboard --logdir=logs --host=0.0.0.0 --port=6006 & wait"]

# Production stage
FROM base as production

# Copy requirements
COPY requirements.txt .

# Install only production dependencies (no dev tools)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers>=4.36.0 \
    datasets>=2.14.0 \
    accelerate>=0.25.0 \
    peft>=0.7.0 \
    bitsandbytes>=0.41.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.65.0 \
    psutil>=5.9.0 \
    nltk>=3.8.0 \
    regex>=2023.0.0 \
    omegaconf>=2.3.0

# Install Flash Attention
RUN pip install flash-attn --no-build-isolation

# Install Unsloth
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || echo "Unsloth installation failed - continuing without it"

# Copy only necessary files
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY utils/ ./utils/
COPY data/ ./data/
COPY *.py ./

# Create directories
RUN mkdir -p data/processed data/cache checkpoints experiments logs

# Set permissions
RUN chmod +x scripts/*.py || true
RUN chmod +x scripts/*.sh || true

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || exit 1

# Production command
CMD ["python", "scripts/train_dolphin.py", "--help"]

# Training stage (specialized for training workloads)
FROM production as training

# Install additional monitoring tools
RUN pip install wandb tensorboard GPUtil

# Copy training-specific configurations
COPY config/ ./config/

# Set environment for training
ENV WANDB_MODE=offline
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

# Training entrypoint
ENTRYPOINT ["python", "scripts/train_dolphin.py"]

# Inference stage (lightweight for model serving)
FROM python:3.10-slim as inference

# Install minimal dependencies
RUN pip install torch transformers accelerate peft bitsandbytes

# Copy only model and inference code
COPY --from=production /app/training_module/utils/ ./utils/
COPY --from=production /app/training_module/config/ ./config/

# Expose inference port
EXPOSE 8000

# Inference command
CMD ["python", "-c", "print('Inference container ready')"]
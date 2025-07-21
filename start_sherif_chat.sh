#!/bin/bash

# Start persistent interactive Sherif chat container
echo "🚀 Starting Persistent Sherif Chat Container..."

# Stop any existing container with this name
docker stop sherif-chat 2>/dev/null || true
docker rm sherif-chat 2>/dev/null || true

# Start the container in interactive mode
docker run -it --name sherif-chat \
  --gpus all \
  -v /home/nuko/dev/DeepDiscord:/workspace \
  -e CUDA_VISIBLE_DEVICES=0 \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  bash -c "
    echo '📦 Installing dependencies...'
    pip install transformers peft bitsandbytes accelerate --quiet
    echo '✅ Dependencies installed!'
    echo ''
    python3 /workspace/interactive_sherif_chat.py
  "

echo "👋 Chat session ended!"
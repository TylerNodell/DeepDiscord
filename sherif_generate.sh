#!/bin/bash

# Universal Sherif response generator
# Usage: ./sherif_generate.sh "Your prompt here"

if [ $# -eq 0 ]; then
    echo "Usage: ./sherif_generate.sh \"Your prompt here\""
    exit 1
fi

PROMPT="$*"

echo "🚀 Starting Sherif Response Generator..."

docker run --rm --name sherif-generate \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/nuko/dev/DeepDiscord:/workspace \
  -e CUDA_VISIBLE_DEVICES=0 \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  bash -c "
    pip install transformers peft bitsandbytes accelerate --quiet && \
    python3 /workspace/response_generators/generate_sherif_responses.py \"$PROMPT\"
  "
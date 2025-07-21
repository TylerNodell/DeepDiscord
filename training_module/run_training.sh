#!/bin/bash
# Universal training runner with all stability fixes

# Set critical environment variables
export TORCH_COMPILE=0
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <training_data.json> [personality_id] [options]"
    echo "Example: $0 data/sherif_training.json 191949803154833408:Sherif"
    exit 1
fi

DATA_FILE=$1
PERSONALITY_ID=${2:-"default"}
shift 2

# Detect if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    cd /workspace/training_module
    
    # Install packages if not already installed
    if ! python -c "import transformers" 2>/dev/null; then
        echo "📦 Installing required packages..."
        pip install transformers datasets accelerate peft bitsandbytes
    fi
else
    echo "💻 Running locally"
    cd "$(dirname "$0")"
fi

# Clear GPU cache before training
echo "🧹 Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run setup check
echo "🔍 Checking environment..."
python setup_training_environment.py

# Run training with universal script
echo "🚀 Starting training..."
echo "📊 Data: $DATA_FILE"
echo "🤖 Personality: $PERSONALITY_ID"

python scripts/train_universal.py \
    --data "$DATA_FILE" \
    --personality-id "$PERSONALITY_ID" \
    --use-cached-model \
    $@

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Check the checkpoints directory for your trained model"
else
    echo "❌ Training failed. Check the logs above for errors."
    exit 1
fi
#!/bin/bash

echo "🚀 Installing DeepDiscord Training Module Requirements"
echo "Configuration B: Balanced (12-14GB VRAM) for Dolphin-2.9-Llama-3-8B"
echo "======================================================================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ "$python_version" < "3.8" ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "CUDA version: $cuda_version"
        
        # Install appropriate PyTorch version
        if [[ "$cuda_version" == "12.1"* ]]; then
            echo "📦 Installing PyTorch for CUDA 12.1..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$cuda_version" == "11.8"* ]]; then
            echo "📦 Installing PyTorch for CUDA 11.8..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            echo "⚠️  Unrecognized CUDA version, installing default PyTorch"
            pip install torch torchvision torchaudio
        fi
    else
        echo "⚠️  CUDA toolkit not found, installing default PyTorch"
        pip install torch torchvision torchaudio
    fi
else
    echo "⚠️  No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "📦 Installing core requirements..."
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install accelerate>=0.25.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0

echo "📦 Installing utilities..."
pip install tqdm pandas numpy scikit-learn
pip install psutil wandb tensorboard
pip install nltk regex omegaconf

echo "🚀 Installing Unsloth (optional but recommended)..."
if pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"; then
    echo "✅ Unsloth installed successfully"
else
    echo "⚠️  Unsloth installation failed, continuing without it"
fi

echo "⚡ Installing Flash Attention (optional)..."
if pip install flash-attn --no-build-isolation; then
    echo "✅ Flash Attention installed successfully"
else
    echo "⚠️  Flash Attention installation failed, continuing without it"
fi

echo "🧪 Installing development tools..."
pip install pytest black flake8

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Quick Start:"
echo "1. Ensure you have Discord training data in the discord_bot/results/ directory"
echo "2. Run training with: python scripts/train_dolphin.py --data ../discord_bot/results/training_data_user_timestamp.zip --use-unsloth"
echo "3. Monitor training progress in the logs"
echo ""
echo "💾 Memory Requirements:"
echo "- Minimum: 12GB VRAM"
echo "- Recommended: 16GB VRAM"
echo "- CPU: 16GB+ RAM"
echo ""
echo "📊 Expected Performance (RTX 5080):"
echo "- 2K Discord samples: 1-2 hours"
echo "- Memory usage: ~12-14GB VRAM"
echo "- Training speed: 2x faster with Unsloth"
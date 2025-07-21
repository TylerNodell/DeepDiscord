#!/usr/bin/env python3
"""
Setup script to ensure training environment is properly configured.
Run this before any training to verify and fix common issues.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path


def setup_environment_variables():
    """Set critical environment variables for stable training."""
    env_vars = {
        "TORCH_COMPILE": "0",  # Disable torch compile to prevent hangs
        "TOKENIZERS_PARALLELISM": "false",  # Prevent tokenizer warnings
        "CUDA_LAUNCH_BLOCKING": "1",  # Help with debugging CUDA issues
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Memory fragmentation fix
        "CUDA_VISIBLE_DEVICES": "0",  # Use first GPU only
    }
    
    print("🔧 Setting environment variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    return env_vars


def check_gpu():
    """Check GPU availability and memory."""
    print("\n🖥️  Checking GPU:")
    
    if not torch.cuda.is_available():
        print("   ❌ No GPU detected! Training will be extremely slow.")
        return False
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    print(f"   ✅ GPU: {gpu_name}")
    print(f"   💾 Total VRAM: {total_memory:.1f} GB")
    
    if total_memory < 8:
        print("   ⚠️  Less than 8GB VRAM - training may fail!")
        return False
    elif total_memory < 12:
        print("   ⚠️  Less than 12GB VRAM - use conservative settings")
    else:
        print("   ✅ Sufficient VRAM for training")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    print("   ✅ GPU cache cleared")
    
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n📦 Checking dependencies:")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
        "tokenizers"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install " + " ".join(missing))
        return False
    
    return True


def create_directories():
    """Create necessary directories for training."""
    print("\n📁 Creating directories:")
    
    dirs = [
        "checkpoints",
        "cache",
        "logs",
        "data/raw",
        "data/processed",
        "models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    return True


def create_training_scripts():
    """Create convenience scripts for common training tasks."""
    print("\n📝 Creating convenience scripts:")
    
    # Create a simple training launcher
    launcher_content = """#!/bin/bash
# Simple training launcher with all fixes applied

# Check if data file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <training_data.json> [personality_id]"
    exit 1
fi

DATA_FILE=$1
PERSONALITY_ID=${2:-"default"}

# Set environment variables
export TORCH_COMPILE=0
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

# Run training
python scripts/train_universal.py \\
    --data "$DATA_FILE" \\
    --personality-id "$PERSONALITY_ID" \\
    --use-cached-model

echo "Training complete!"
"""
    
    launcher_path = Path("train.sh")
    launcher_path.write_text(launcher_content)
    launcher_path.chmod(0o755)
    print(f"   ✅ train.sh - Simple training launcher")
    
    # Create a Docker training script
    docker_content = """#!/bin/bash
# Docker training launcher

if [ $# -eq 0 ]; then
    echo "Usage: $0 <training_data.json> [personality_id]"
    exit 1
fi

DATA_FILE=$1
PERSONALITY_ID=${2:-"default"}

docker run --rm --gpus all \\
    --name discord-training \\
    --shm-size=16g \\
    --ipc=host \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    -v $(pwd):/workspace \\
    -v $(pwd)/models:/workspace/models \\
    -e TORCH_COMPILE=0 \\
    -e TOKENIZERS_PARALLELISM=false \\
    -e CUDA_LAUNCH_BLOCKING=1 \\
    nvcr.io/nvidia/pytorch:25.06-py3 \\
    bash -c "cd /workspace && pip install transformers datasets accelerate peft bitsandbytes && python scripts/train_universal.py --data $DATA_FILE --personality-id $PERSONALITY_ID --use-cached-model"
"""
    
    docker_path = Path("train_docker.sh")
    docker_path.write_text(docker_content)
    docker_path.chmod(0o755)
    print(f"   ✅ train_docker.sh - Docker training launcher")
    
    return True


def download_model_cache():
    """Check if model is cached locally."""
    print("\n🤖 Checking model cache:")
    
    cache_dir = Path("models/dolphin-2.9-llama3-8b")
    
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print("   ✅ Model cache found")
        return True
    else:
        print("   ⚠️  Model not cached - will download on first run")
        print("   💡 To pre-download, run:")
        print("      python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('cognitivecomputations/dolphin-2.9-llama3-8b', cache_dir='models/dolphin-2.9-llama3-8b')\"")
        return False


def verify_config():
    """Verify training configuration is set properly."""
    print("\n⚙️  Verifying configuration:")
    
    try:
        from config.training_config import TrainingConfig
        config = TrainingConfig()
        
        critical_settings = {
            "batch_size": (config.batch_size == 1, "Should be 1 for stability"),
            "gradient_checkpointing": (not config.gradient_checkpointing, "Should be False to prevent hangs"),
            "torch_compile": (not config.torch_compile, "Should be False to prevent hangs"),
            "max_grad_norm": (config.max_grad_norm <= 1.0, "Should be <= 1.0 for stability"),
            "optim": (config.optim == "adamw_torch", "Should be 'adamw_torch' for stability"),
            "use_cache": (not config.use_cache, "Should be False during training"),
            "group_by_length": (not config.group_by_length, "Should be False to prevent hangs"),
            "max_length": (config.max_length <= 512, "Should be <= 512 for memory efficiency")
        }
        
        all_good = True
        for setting, (is_correct, recommendation) in critical_settings.items():
            if is_correct:
                print(f"   ✅ {setting}: {getattr(config, setting, 'Not set')}")
            else:
                print(f"   ⚠️  {setting}: {getattr(config, setting, 'Not set')} - {recommendation}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return False


def main():
    """Run all setup checks."""
    print("🚀 DeepDiscord Training Environment Setup")
    print("=" * 50)
    
    checks = [
        ("Environment Variables", setup_environment_variables),
        ("GPU Check", check_gpu),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Training Scripts", create_training_scripts),
        ("Model Cache", download_model_cache),
        ("Configuration", verify_config)
    ]
    
    all_passed = True
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"\n❌ Error during {name}: {e}")
            results.append((name, False))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    if all_passed:
        print("\n✅ Environment is ready for training!")
        print("\n💡 Quick start:")
        print("   ./train.sh path/to/training_data.json personality_name")
        print("\n💡 Docker training:")
        print("   ./train_docker.sh path/to/training_data.json personality_name")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
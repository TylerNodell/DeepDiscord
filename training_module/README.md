# Training Module - Dolphin-2.9-Llama-3-8B

This directory contains the complete training pipeline for the DeepDiscord AI model, optimized for **Dolphin-2.9-Llama-3-8B** using **Configuration B: Balanced (12-14GB VRAM)**.

## 🎯 Quick Start

### 1. Install Dependencies
```bash
# Run the automated installer
./scripts/install_requirements.sh

# Or manually install
pip install -r requirements.txt
```

### 2. Train Model
```bash
# Train with Unsloth (recommended for speed)
python scripts/train_dolphin.py --data ../discord_bot/results/training_data_user_20250719.zip --use-unsloth

# Monitor memory usage during training
python scripts/train_dolphin.py --data ../discord_bot/results/training_data_user_20250719.zip --use-unsloth --run-name "discord_bot_v1"
```

### 3. Expected Results
- **Training Time**: 1-2 hours for 2K Discord samples (RTX 5080)
- **Memory Usage**: 12-14GB VRAM with Configuration B
- **Speed Improvement**: 2x faster with Unsloth optimization

## 🏗️ Architecture Overview

### Optimized Training Stack
- **Model**: Dolphin-2.9-Llama-3-8B (uncensored, excellent for Discord conversations)
- **Method**: QLoRA (4-bit quantization + LoRA fine-tuning)
- **Framework**: Unsloth (2x faster) or standard Transformers
- **Memory**: 12-14GB VRAM usage with aggressive optimizations

### Key Components

```
training_module/
├── config/
│   ├── training_config.py    # Configuration B optimized settings
│   ├── qlora_config.py      # QLoRA and 4-bit quantization setup
│   ├── model_config.py      # Model architecture settings
│   └── data_config.py       # Data processing configuration
├── scripts/
│   ├── train_dolphin.py     # Optimized training script for Dolphin model
│   ├── train.py            # Generic training script
│   ├── evaluate.py         # Model evaluation
│   └── install_requirements.sh # Automated dependency installation
├── utils/
│   ├── discord_preprocessing.py # Discord-specific data cleaning
│   ├── memory_monitor.py    # Real-time memory monitoring
│   ├── data_preprocessing.py # General preprocessing utilities
│   └── logging_utils.py     # Training progress logging
└── data/                    # Data loading and processing
```

## 📊 Configuration B Details

### Memory Optimization
- **4-bit Quantization**: Reduces model memory by ~75%
- **LoRA Rank 32**: Balanced adaptation capability vs memory
- **Gradient Checkpointing**: Trades compute for memory
- **8-bit Optimizer**: Reduces optimizer state memory

### Training Parameters
```python
# Optimized for 12-14GB VRAM
learning_rate = 2e-4
batch_size = 4
gradient_accumulation_steps = 2
max_steps = 1500
lora_r = 32
lora_alpha = 64
```

## 🔧 Advanced Features

### Real-time Memory Monitoring
```python
from utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()
monitor.start_monitoring()

# Training code here...

peak_usage = monitor.get_peak_usage()
print(f"Peak GPU usage: {peak_usage['peak_gpu_reserved_gb']:.1f} GB")
```

### Discord Data Preprocessing
```python
from utils.discord_preprocessing import load_discord_training_data, filter_training_pairs

# Load Discord ZIP files
data = load_discord_training_data("../discord_bot/results/training_data_user_20250719.zip")

# Clean and filter
filtered = filter_training_pairs(data, min_length=10, include_gifs=True)
```

### Unsloth Integration
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="cognitivecomputations/dolphin-2.9-llama3-8b",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## 🎯 Training Pipeline

1. **Discord Data** → Load ZIP archives from bot (`../discord_bot/results/`)
2. **Preprocessing** → Clean Discord formatting, filter quality
3. **ChatML Format** → Convert to training format
4. **QLoRA Setup** → Apply 4-bit quantization + LoRA adapters
5. **Training** → Fine-tune with Unsloth acceleration
6. **Monitoring** → Real-time memory and progress tracking
7. **Checkpointing** → Save best models automatically

## 📈 Performance Benchmarks

### RTX 5080 (16GB VRAM)
- **2K Discord samples**: 1-2 hours
- **Memory usage**: 12-14GB VRAM
- **Speed**: 2x faster with Unsloth
- **Batch size**: 4 with gradient accumulation

### Memory Estimates
```
Quantized model: ~2GB (vs 8GB full precision)
LoRA adapters: ~100MB
Optimizer states: ~50MB (8-bit)
Activations: ~2GB (batch_size=4, seq_len=2048)
Total: ~4.2GB + overhead = 12-14GB total
```

## 🔍 Monitoring and Debugging

### Training Logs
- Real-time memory usage tracking
- Discord data quality analysis
- Training progress with loss curves
- Automatic checkpoint saving

### Common Issues
- **OOM Error**: Reduce batch_size to 2 or enable more aggressive checkpointing
- **Slow Training**: Ensure Unsloth is installed and Flash Attention is available
- **Poor Quality**: Check Discord data filtering settings and increase training steps

## 📚 Additional Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Dolphin Model Card](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b)
# DeepSeek Training Module

This module contains the DeepSeek AI model setup and training functionality for the DeepDiscord project.

## Files

### `deepseek_setup.py`
- **Purpose**: Model initialization and setup
- **Features**:
  - Device detection (CUDA, MPS, CPU)
  - Model loading with optimizations
  - Memory management for different hardware
  - Model testing and validation

### `deepseek_training.py`
- **Purpose**: Fine-tuning and training pipeline
- **Features**:
  - DeepSeekTrainer class for model training
  - Dataset preparation and tokenization
  - Training with gradient checkpointing
  - Text generation capabilities

## Installation

```bash
# Install training dependencies
pip install -r requirements_training.txt
```

## Usage

### Setup Model
```bash
cd training
python deepseek_setup.py
```

### Train Model
```bash
cd training
python deepseek_training.py
```

## Hardware Compatibility

- **M4 Mac**: Uses MPS (Metal Performance Shaders)
- **RTX 5080 PC**: Uses CUDA
- **Other Systems**: Falls back to CPU

## Model Sizes

- **1.3B**: Lightweight, good for development
- **6.7B**: Larger model, requires more memory

## Integration

This training module is designed to work with data collected by the Discord bot. The bot provides conversation data that can be used to fine-tune the DeepSeek models for Discord-specific interactions. 
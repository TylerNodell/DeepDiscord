# DeepDiscord

A project to automate training data creation from Discord messages and train lightweight language models using DeepSeek.

## Features

- **DeepSeek Model Integration**: Uses DeepSeek-Coder models for lightweight training
- **Hardware Optimized**: Compatible with M4 Mac (MPS) and RTX 5080 (CUDA)
- **Memory Efficient**: Optimized for training with limited resources
- **Discord Integration**: Automates training data creation from Discord messages

## Files

- `deepseek_setup.py` - Sets up and tests DeepSeek models
- `deepseek_training.py` - Fine-tuning script for DeepSeek models
- `venv/` - Python virtual environment with required dependencies

## Setup

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   pip install transformers torch accelerate
   ```

2. **Run Model Setup**:
   ```bash
   python deepseek_setup.py
   ```

3. **Train Model**:
   ```bash
   python deepseek_training.py
   ```

## Requirements

- Python 3.9+
- PyTorch with MPS (Mac) or CUDA (PC) support
- Transformers library
- At least 8GB RAM for 1.3B model
- 16GB+ RAM recommended for 6.7B model

## License

MIT License

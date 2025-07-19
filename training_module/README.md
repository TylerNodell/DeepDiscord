# Training Module - Dolphin-2.9-Llama-3-8B + Personality System

This directory contains the complete training pipeline for the DeepDiscord AI model, optimized for **Dolphin-2.9-Llama-3-8B** using **Configuration B: Balanced (12-14GB VRAM)** with **Multi-Personality Emulation**.

## 🚀 Quick Start with Docker

### Option 1: One-Command Setup
```bash
# Complete setup and start development environment
cd training_module
./scripts/docker_setup.sh setup && ./scripts/docker_setup.sh dev
```

### Option 2: Manual Setup
```bash
# 1. Build and test GPU support
./scripts/docker_setup.sh setup
./scripts/docker_gpu_test.sh all

# 2. Start development environment
docker-compose up -d training-dev

# 3. Access services
# Jupyter Lab: http://localhost:8888 (token: deepdiscord-training)
# TensorBoard: http://localhost:6006
```

### Option 3: Training Production Run
```bash
# Run complete training pipeline
./scripts/docker_train.sh --data training_data_user_20250719.zip --strategy instruction_based
```

## 🌐 Remote Training & Monitoring

### Run Training on Main Machine, Monitor from Laptop

Perfect for running training on your main machine with RTX 5080 while monitoring from your laptop:

#### Setup Remote Access
```bash
# On laptop: Setup remote connection
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko setup

# Test connection and create monitoring tunnels
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko tunnel
```

#### Start Remote Training
```bash
# Start training on main machine, monitor from laptop
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko train --data training_data.zip --strategy instruction_based
```

#### Monitor Progress
```bash
# Open monitoring dashboard (auto-opens Jupyter Lab + TensorBoard)
./scripts/remote_setup.sh monitor

# Check training status
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko status

# View live logs
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko logs
```

#### Access Points from Laptop
- **Jupyter Lab**: http://localhost:8888 (token: deepdiscord-training)
- **TensorBoard**: http://localhost:6006  
- **GPU Monitor**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3000 (admin/deepdiscord)

#### Advanced Monitoring Stack
```bash
# On main machine: Start full monitoring stack
docker-compose -f docker-compose.monitoring.yml --profile monitoring up -d

# On laptop: Create tunnels for all monitoring services
./scripts/remote_setup.sh --ip MAIN_IP --user USER tunnel
```

## 🐳 Docker Environments

### Development Environment
- **Jupyter Lab** with GPU support
- **TensorBoard** for monitoring
- Hot-reload source code mounting
- Interactive debugging tools

```bash
# Start development
docker-compose up -d training-dev

# Open shell in container
docker exec -it deepdiscord-training-dev bash

# View logs
docker-compose logs -f training-dev
```

### Production Training
- Optimized for training workloads
- Minimal overhead and dependencies
- Automated pipeline execution

```bash
# Production training
docker-compose -f docker-compose.prod.yml up training

# Or use the training script
./scripts/docker_train.sh --data your_data.zip --strategy multiple_lora
```

### Monitoring Stack
- **Prometheus** for metrics collection
- **Grafana** for visualization dashboards
- **Real-time GPU monitoring**
- **Training logs aggregation**

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml --profile monitoring up -d

# Access monitoring services:
# Grafana: http://localhost:3000 (admin/deepdiscord)
# Prometheus: http://localhost:9090
# GPU Stats: http://localhost:8080
```

## 🎯 Quick Start (Non-Docker)

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
- **Containerization**: Docker with CUDA 12.1 support
- **Remote Monitoring**: SSH tunneling with real-time dashboards

### Key Components

```
training_module/
├── config/
│   ├── training_config.py    # Configuration B optimized settings
│   ├── qlora_config.py      # QLoRA and 4-bit quantization setup
│   ├── personality_config.py # Multi-personality emulation settings
│   ├── model_config.py      # Model architecture settings
│   └── data_config.py       # Data processing configuration
├── scripts/
│   ├── train_dolphin.py     # Optimized training script for Dolphin model
│   ├── docker_setup.sh      # Docker environment setup
│   ├── docker_train.sh      # Automated training pipeline
│   ├── docker_gpu_test.sh   # GPU testing and validation
│   ├── remote_setup.sh      # Remote training and monitoring setup
│   ├── preprocess_personality_data.py # Personality-aware preprocessing
│   ├── manage_personalities.py # Personality discovery and management
│   └── install_requirements.sh # Automated dependency installation
├── utils/
│   ├── discord_preprocessing.py # Discord-specific data cleaning + personalities
│   ├── memory_monitor.py    # Real-time memory monitoring
│   ├── data_preprocessing.py # General preprocessing utilities
│   └── logging_utils.py     # Training progress logging
├── docker/
│   ├── Dockerfile          # Multi-stage Docker build
│   ├── docker-compose.yml  # Main compose file
│   ├── docker-compose.dev.yml # Development environment
│   ├── docker-compose.prod.yml # Production environment
│   └── docker-compose.monitoring.yml # Monitoring stack
├── monitoring/
│   ├── prometheus.yml      # Metrics collection config
│   └── grafana/           # Dashboard configurations
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

## 🎭 Personality System

The training module supports multi-personality emulation with three strategies:

### 1. Unified Strategy
Combines all user data into a single model without personality distinction.

### 2. Instruction-Based Strategy (Recommended)
Adds personality context to training data using instruction templates:
- Format: `"Respond as {personality_name}: {message}"`
- Channel context: `"Channel: {channel} | Respond as {personality_name}: {message}"`
- Single model learns to switch personalities based on instructions

### 3. Multiple LoRA Adapters
Creates separate LoRA adapters for each personality:
- Base model handles general language understanding
- Individual adapters specialize in specific personality traits
- Allows fine-grained personality control and mixing

### Personality Profiles
Personalities are discovered from actual Discord training data based on user IDs:
- Each user with sufficient message data becomes a trainable personality
- Human-readable names are assigned for training and Discord commands
- Discord usernames may change, but user IDs remain constant for tracking

### Privacy & Consent
- Respects user consent settings from Discord bot
- Anonymizes users without consent as "Anonymous"
- Excludes non-consenting users if configured

### Personality Discovery & Management
Discovery process:
1. Run preprocessing to discover personalities from training data
2. Use management script to assign human-readable names
3. Configure personality parameters for training

```bash
# Discover personalities from training data
python training_module/scripts/manage_personalities.py --discover

# Interactively update personality names
python training_module/scripts/manage_personalities.py --update-names

# List current personalities
python training_module/scripts/manage_personalities.py --list

# Add personality manually
python training_module/scripts/manage_personalities.py --add-personality USER_ID "PersonalityName" "Description"
```

## 🚀 Quick Start (with Personality Support)

### 1. Data Preprocessing
```bash
# Basic instruction-based personality training
python training_module/scripts/preprocess_personality_data.py --strategy instruction_based

# Multiple LoRA adapters for different personalities  
python training_module/scripts/preprocess_personality_data.py --strategy multiple_lora

# Custom input/output directories
python training_module/scripts/preprocess_personality_data.py --input-dir ../discord_bot/results --output-dir ./data/processed
```

### 2. Training
```bash
python training_module/scripts/train.py --config training_module/config/training_config.py
```

### 3. Evaluation
```bash
python training_module/scripts/evaluate.py --model training_module/checkpoints/best_model.pt
```

## 🔧 Configuration

Training parameters can be configured in:
- `config/training_config.py` - Learning rate, batch size, epochs, etc.
- `config/model_config.py` - Model architecture, hidden sizes, layers, etc.
- `config/data_config.py` - Data paths, preprocessing settings, etc.
- `config/personality_config.py` - Multi-personality emulation settings
- `config/qlora_config.py` - QLoRA 4-bit quantization settings

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
from utils.discord_preprocessing import DiscordPreprocessor
from config.personality_config import DEFAULT_PERSONALITY_CONFIG

# Initialize with personality support
preprocessor = DiscordPreprocessor(DEFAULT_PERSONALITY_CONFIG)

# Process ZIP file with consent checking
chatml_data = preprocessor.process_training_zip(
    "training_data.zip", 
    consent_file="discord_data/user_consents.json"
)
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
2. **Personality Discovery** → Automatically detect personalities from user IDs
3. **Preprocessing** → Clean Discord formatting, apply personality strategies
4. **ChatML Format** → Convert to training format with personality context
5. **QLoRA Setup** → Apply 4-bit quantization + LoRA adapters
6. **Training** → Fine-tune with Unsloth acceleration
7. **Monitoring** → Real-time memory and progress tracking
8. **Checkpointing** → Save best models automatically

## 📈 Performance Benchmarks

### RTX 5080 (16GB VRAM)
- **2K Discord samples**: 1-2 hours
- **Memory usage**: 12-14GB VRAM
- **Speed**: 2x faster with Unsloth
- **Batch size**: 4 with gradient accumulation

### Memory Estimates
```
Quantized model: ~2GB (vs 8GB full precision)
LoRA adapters: ~100MB per personality
Optimizer states: ~50MB (8-bit)
Activations: ~2GB (batch_size=4, seq_len=2048)
Total: ~4.2GB + overhead = 12-14GB total
```

## 🐳 Docker Commands Reference

### Setup and Testing
```bash
# Initial setup with GPU testing
./scripts/docker_setup.sh setup

# Test GPU functionality
./scripts/docker_gpu_test.sh all

# Test specific components
./scripts/docker_gpu_test.sh pytorch
./scripts/docker_gpu_test.sh memory
```

### Development Workflow
```bash
# Start development environment
docker-compose up -d training-dev

# View services
docker-compose ps

# Open interactive shell
docker exec -it deepdiscord-training-dev bash

# View logs
docker-compose logs -f training-dev

# Stop services
docker-compose down
```

### Production Training
```bash
# Full training pipeline
./scripts/docker_train.sh --data your_file.zip --strategy instruction_based

# Manual production run
docker-compose -f docker-compose.prod.yml run --rm training \
    --data /app/discord_bot/results/training_data.zip \
    --use-unsloth \
    --run-name production_run
```

### Remote Training Commands
```bash
# Setup remote connection (run once)
./scripts/remote_setup.sh --ip MAIN_MACHINE_IP --user USERNAME setup

# Start remote training with monitoring
./scripts/remote_setup.sh --ip MAIN_MACHINE_IP --user USERNAME train \
    --data training_data.zip --strategy instruction_based

# Monitor from laptop
./scripts/remote_setup.sh monitor  # Opens Jupyter Lab + TensorBoard

# Check status
./scripts/remote_setup.sh --ip MAIN_MACHINE_IP --user USERNAME status

# View logs
./scripts/remote_setup.sh --ip MAIN_MACHINE_IP --user USERNAME logs

# Stop training
./scripts/remote_setup.sh --ip MAIN_MACHINE_IP --user USERNAME stop
```

### Advanced Docker Usage
```bash
# Run with custom GPU
CUDA_VISIBLE_DEVICES=1 docker-compose up training-dev

# Limit memory
docker-compose run --memory=16g training-dev

# Run with different profiles
docker-compose --profile monitoring up  # Includes TensorBoard, GPU monitor
docker-compose --profile tools up linter  # Code quality tools
docker-compose --profile debug up debugger  # Interactive debugging
```

## 🌐 Remote Monitoring Setup

### Prerequisites on Main Machine
1. **SSH server enabled**
2. **Docker with NVIDIA support**
3. **Project cloned and setup**

### Prerequisites on Laptop
1. **SSH client**
2. **Network access to main machine**

### Complete Remote Setup Example
```bash
# 1. On main machine: Setup Docker environment
cd DeepDiscord/training_module
./scripts/docker_setup.sh setup

# 2. On laptop: Configure remote access
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko setup

# 3. On laptop: Start training with monitoring
./scripts/remote_setup.sh --ip 192.168.1.100 --user nuko train \
    --data training_data_user_20250719.zip \
    --strategy instruction_based

# 4. On laptop: Open monitoring dashboard
./scripts/remote_setup.sh monitor
```

### Advanced Monitoring Features
- **Real-time GPU metrics** (temperature, memory, utilization)
- **Training progress visualization** via TensorBoard
- **System resource monitoring** via Grafana
- **Live log streaming** from training process
- **Interactive Jupyter environment** for debugging

## 🔍 Monitoring and Debugging

### Training Logs
- Real-time memory usage tracking
- Discord data quality analysis  
- Training progress with loss curves
- Automatic checkpoint saving
- Personality distribution statistics

### Common Issues

#### Remote Setup Issues
- **SSH connection failed**: Check firewall, SSH keys, and network connectivity
- **Tunnels not working**: Verify services are running on main machine
- **Permission denied**: Check SSH key permissions and user access

#### Docker-specific Issues
- **Container won't start**: Check GPU drivers and NVIDIA Docker runtime
- **Out of memory**: Reduce batch size or use `docker-compose.override.yml` for memory limits
- **Permission errors**: Check volume mounts and user permissions
- **GPU not detected**: Run `./scripts/docker_gpu_test.sh` for diagnosis

#### Training Issues
- **OOM Error**: Reduce batch_size to 2 or enable more aggressive checkpointing
- **Slow Training**: Ensure Unsloth is installed and Flash Attention is available
- **Poor Quality**: Check Discord data filtering settings and increase training steps

### Development Tools
```bash
# Code formatting
docker-compose run --rm linter black .

# Run tests
docker-compose run --rm linter pytest tests/ -v

# Interactive debugging
docker-compose --profile debug run --rm debugger
```

## 📚 Additional Resources

- [Docker Setup Guide](scripts/docker_setup.sh)
- [Remote Setup Guide](scripts/remote_setup.sh)
- [GPU Testing Guide](scripts/docker_gpu_test.sh)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Dolphin Model Card](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b)
# Training Module

This directory contains the complete training pipeline for the DeepDiscord AI model.

## 📁 Directory Structure

```
training_module/
├── __init__.py              # Module initialization
├── README.md               # This file
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── training_config.py  # Training hyperparameters
│   ├── model_config.py     # Model architecture settings
│   └── data_config.py      # Data processing configuration
├── models/                 # Model definitions and training
│   ├── __init__.py
│   ├── trainer.py          # Main training orchestrator
│   ├── evaluator.py        # Model evaluation utilities
│   └── architectures/      # Model architecture definitions
├── data/                   # Data processing and loading
│   ├── __init__.py
│   ├── loader.py           # Data loading utilities
│   ├── preprocessor.py     # Data preprocessing pipeline
│   └── augmentation.py     # Data augmentation strategies
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── model_utils.py      # Model utility functions
│   ├── logging_utils.py    # Training logging utilities
│   └── metrics.py          # Evaluation metrics
├── scripts/                # Training scripts
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Model evaluation script
│   └── preprocess_data.py  # Data preprocessing script
├── experiments/            # Experiment tracking and results
│   └── .gitkeep
└── checkpoints/            # Model checkpoints and saves
    └── .gitkeep
```

## 🚀 Quick Start

### 1. Data Preprocessing (with Personality Support)
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

## 🎭 Personality System

The training module supports multi-personality emulation with three strategies:

### 1. Unified Strategy
Combines all user data into a single model without personality distinction.

### 2. Instruction-Based Strategy (Recommended)
Adds personality context to training data using instruction templates:
- Format: `"Respond as {username}: {message}"`
- Channel context: `"Channel: {channel} | Respond as {username}: {message}"`
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

Personalities are tracked by Discord user ID and mapped to human-readable names for training commands.

## 📊 Data Flow

1. **Raw Training Data** → Generated by Discord bot (`results/` directory)
2. **Preprocessing** → Clean, tokenize, and format data
3. **Training** → Train the AI model on processed data
4. **Evaluation** → Test model performance and generate metrics
5. **Deployment** → Export trained model for inference

## 🎯 Features

- **Modular Architecture**: Easy to extend and modify
- **Configuration-driven**: All settings in config files
- **Experiment Tracking**: Organized experiment management
- **Checkpoint Management**: Automatic model saving and loading
- **Comprehensive Logging**: Detailed training progress tracking
- **Evaluation Metrics**: Multiple evaluation strategies

## 📝 Usage Examples

### Training with Custom Config
```python
from training_module.config import TrainingConfig
from training_module.models import Trainer

config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

trainer = Trainer(config)
trainer.train()
```

### Data Preprocessing
```python
from training_module.data import DataPreprocessor

preprocessor = DataPreprocessor()
processed_data = preprocessor.process_zip_file("results/training_data_User_20250719.zip")
```

## 🔍 Monitoring

Training progress can be monitored through:
- Console logging with detailed progress information
- TensorBoard integration (optional)
- Experiment tracking with metrics and checkpoints
- Model performance evaluation on validation sets
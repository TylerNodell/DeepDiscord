# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepDiscord is a sophisticated Discord bot system for collecting conversation data and training AI models on Discord interactions. The project consists of two main components:

1. **Discord Bot** (`discord_bot/`) - Collects and processes Discord conversations with privacy-first consent system
2. **Training Module** (`training_module/`) - Complete ML pipeline for training conversational AI models

## Core Architecture

### Discord Bot System (`discord_bot/`)

The bot implements a sophisticated message tracking and training data generation system:

- **MessageTracker**: Core component that manages message caching, fragment detection (combining rapid-fire messages), and response chain tracking
- **ConsentManager**: Privacy-first system requiring explicit user consent before data collection (consent defaults to 'no' and never expires until revoked)
- **AuthorizationManager**: Controls which users can generate training data
- **DiscordTrainingDataGenerator**: Analyzes conversations to create question/answer training pairs with confidence scoring

**Key Privacy Features:**
- All data collection requires explicit user consent via DM requests
- Training data anonymizes users without consent (shows as "Anonymous")
- Consent system logs all activities and provides full transparency
- Bot commands use `!!` prefix to avoid conflicts with other bots

### Training Module (`training_module/`)

Complete ML infrastructure built on Transformers for training conversational AI:

- **Configuration System**: Dataclass-based configs for training, model, and data parameters
- **Data Pipeline**: Loads Discord ZIP archives, preprocesses with Unicode cleaning, tokenization, and quality filtering
- **Training Scripts**: Full training loop with logging, checkpointing, and evaluation
- **Utilities**: Comprehensive metrics (BLEU, ROUGE-L), model management, and logging systems

## Common Commands

### Discord Bot Development & Testing
```bash
# Run full test suite
./discord_bot/scripts/run_tests.sh

# Run specific test
./discord_bot/scripts/run_tests.sh test_discord_user.py

# Start Discord bot
python discord_bot/discord_bot.py

# Generate training data (requires bot to be running)
./discord_bot/scripts/generate_training_data.sh
```

### Training Module Development
```bash
# Install training dependencies
pip install -r training_module/requirements.txt

# Train model with default config
python training_module/scripts/train.py

# Train with custom parameters
python training_module/scripts/train.py --config-override learning_rate=1e-5,batch_size=8

# Evaluate trained model
python training_module/scripts/evaluate.py --model-path ./checkpoints/best_model

# Interactive chat with trained model
python training_module/scripts/inference.py --model-path ./checkpoints/best_model --interactive
```

## Key Bot Commands (when running)
```bash
!!generatetrainingdata <user_id> [days_back]  # Generate training data (requires consent)
!!consent                                     # Check consent status
!!consent grant @user                         # Grant consent to someone
!!consent revoke                              # Revoke your consent
!!userhistory @user                           # Analyze user message patterns
!!help                                        # Show all available commands
```

## Development Workflow

### Privacy & Consent System
The consent system is central to the bot's operation:
- **Default**: Users have NO consent by default
- **Required**: Explicit consent needed before any data collection
- **Persistent**: Consent never expires (until manually revoked)
- **Anonymous**: Users without consent appear as "Anonymous" in training data
- **Transparent**: All consent activities are logged and auditable

### Training Data Generation Flow
1. User runs `!!generatetrainingdata <user_id>`
2. Bot checks if target user has given consent
3. If no consent, sends detailed DM request with ✅/❌ reactions
4. If consent granted, analyzes message history for question/answer pairs
5. Creates confidence-scored training pairs (high/medium/all levels)
6. Packages into ZIP archive with metadata
7. Uploads to Discord (or online host if >50MB)

### Training Pipeline Flow
1. **Data Input**: ZIP archives from Discord bot (in `results/` directory)
2. **Preprocessing**: Clean Unicode, filter quality, tokenize conversations
3. **Training**: Fine-tune DialoGPT-medium on conversation pairs
4. **Evaluation**: Calculate BLEU, ROUGE-L, diversity metrics
5. **Inference**: Chat interface or batch evaluation

## File Structure Understanding

### Critical Files
- `discord_bot/discord_bot.py`: Main bot with all core systems (3000+ lines, contains MessageTracker, ConsentManager, TrainingDataGenerator)
- `training_module/scripts/train.py`: Complete training script with Transformers integration
- `training_module/config/training_config.py`: Training hyperparameters (DialoGPT-medium default)
- `discord_bot/scripts/run_tests.sh`: Test runner with environment validation

### Data Flow
- **Input**: Discord conversations → **Processing**: Bot analysis → **Output**: ZIP archives in `results/`
- **Input**: ZIP archives → **Processing**: Training module → **Output**: Trained models in `checkpoints/`

## Configuration

### Environment Setup (.env)
```bash
DISCORD_TOKEN=your_bot_token
TEST_USER_ID=123456789012345678  # For testing
BOT_PREFIX=!!                    # Command prefix
```

### Training Configuration
Training configs use dataclasses with validation:
- Learning rate: 1e-4 (default)
- Batch size: 16 (default) 
- Model: microsoft/DialoGPT-medium
- Max length: 512 tokens

## Common Issues & Solutions

### Bot Issues
- **Upload errors**: "Expecting value: line 1 column 1 (char 0)" means Discord upload succeeded but code tried unnecessary online fallback (fixed in recent commits)
- **Permission errors**: Ensure bot has "Send Messages", "Read Message History", "Use External Emojis" permissions
- **Consent flow**: If users don't receive DM, they need to enable DMs temporarily

### Training Issues
- **Out of memory**: Reduce batch_size in training_config.py
- **CUDA errors**: Check GPU availability, fall back to CPU training
- **Data loading**: Ensure ZIP files are in correct format from Discord bot

## Testing Strategy

The test suite validates:
- Discord API connectivity and permissions
- Message fragment detection and combination
- User consent system functionality
- Training data generation accuracy
- Bot command processing

Tests require valid Discord credentials in `.env` and use real Discord API calls for integration testing.
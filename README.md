# DeepDiscord

A comprehensive AI-powered Discord analysis and training platform that combines real-time message tracking with DeepSeek AI model training.

## 🏗️ Project Structure

```
DeepDiscord/
├── discord_bot/           # Discord bot for message tracking
│   ├── discord_bot.py     # Main bot implementation
│   ├── requirements_discord.txt
│   ├── env_example.txt    # Environment configuration
│   ├── README_DISCORD.md  # Bot documentation
│   └── test_discord_bot.py
├── training/              # DeepSeek AI training module
│   ├── deepseek_setup.py  # Model initialization
│   ├── deepseek_training.py # Training pipeline
│   ├── requirements_training.txt
│   └── README_TRAINING.md
├── venv/                  # Python virtual environment
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🎯 Overview

DeepDiscord consists of two main components:

### 1. Discord Bot (`discord_bot/`)
- **Purpose**: Real-time message tracking and analysis
- **Features**:
  - Message retrieval by ID
  - Response relationship tracking
  - Conversation chain analysis
  - Data collection for AI training

### 2. Training Module (`training/`)
- **Purpose**: DeepSeek AI model training and fine-tuning
- **Features**:
  - Model setup and initialization
  - Fine-tuning pipeline
  - Hardware optimization (M4 Mac, RTX 5080)
  - Text generation capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Discord bot token
- Sufficient hardware for AI training

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DeepDiscord
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Discord bot dependencies**:
   ```bash
   cd discord_bot
   pip install -r requirements_discord.txt
   ```

4. **Install training dependencies**:
   ```bash
   cd ../training
   pip install -r requirements_training.txt
   ```

### Configuration

1. **Set up Discord bot**:
   ```bash
   cd discord_bot
   cp env_example.txt .env
   # Edit .env with your Discord bot token
   ```

2. **Test bot setup**:
   ```bash
   python test_discord_bot.py
   ```

### Usage

#### Start Discord Bot
```bash
cd discord_bot
python discord_bot.py
```

#### Train AI Model
```bash
cd training
python deepseek_setup.py    # Initialize model
python deepseek_training.py # Start training
```

## 🔧 Features

### Discord Bot Commands
- `!getmsg <id>` - Retrieve message by ID
- `!responses <id>` - Show responses to message
- `!chain <id>` - Display conversation chain
- `!stats` - Bot statistics
- `!save` - Save data (admin)
- `!clear` - Clear cache (admin)

### Training Capabilities
- **Model Sizes**: 1.3B (lightweight) and 6.7B (full)
- **Hardware Support**: CUDA, MPS (Apple Silicon), CPU
- **Memory Optimization**: Gradient checkpointing, mixed precision
- **Data Integration**: Works with Discord bot data

## 🎯 Use Cases

1. **Conversation Analysis**: Track and analyze Discord conversations
2. **AI Training**: Use Discord data to train conversational AI
3. **Response Prediction**: Predict likely responses in conversations
4. **Community Insights**: Understand community interaction patterns

## 🔒 Privacy & Security

- **Local Storage**: All data stored locally
- **Admin Controls**: Sensitive operations require admin permissions
- **Data Retention**: Configurable cache limits
- **No External Sharing**: No data transmitted to external services

## 📚 Documentation

- [Discord Bot Documentation](discord_bot/README_DISCORD.md)
- [Training Module Documentation](training/README_TRAINING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the relevant module documentation
2. Review logs in `discord_bot/discord_bot.log`
3. Create an issue in the project repository

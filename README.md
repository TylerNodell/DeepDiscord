# DeepDiscord

An AI-powered Discord analysis platform for real-time message tracking and conversation analysis.

## 🏗️ Project Structure

```
DeepDiscord/
├── discord_bot.py         # Main bot implementation
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration (create from env_example.txt)
├── env_example.txt       # Environment template
├── README_DISCORD.md     # Bot documentation
├── test_discord_bot.py   # Bot testing
├── discord_bot.log       # Bot logs
├── discord_data/         # Data storage directory
├── venv/                 # Python virtual environment
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🎯 Overview

DeepDiscord is a Discord bot designed for real-time message tracking and analysis:

### Discord Bot Features
- **Message Retrieval**: Get messages by ID
- **Response Tracking**: Track response relationships
- **Conversation Analysis**: Analyze conversation chains
- **User History**: Get all messages from specific users
- **Data Collection**: Collect and save conversation data
- **Admin Controls**: Secure admin-only operations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 Discord bot token
- Discord server with bot permissions

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

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Set up Discord bot**:
   ```bash
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
python discord_bot.py
```

## 🔧 Bot Commands

### User Commands
- `!getmsg <id>` - Retrieve message by ID
- `!responses <id>` - Show responses to message
- `!chain <id>` - Display conversation chain
- `!userhistory <@user> [limit]` - Get all messages from a user across the server
- `!stats` - Bot statistics

### Admin Commands
- `!save` - Save data (admin only)
- `!saveuser [@user]` - Save user messages to JSON file (admin only)
- `!clear` - Clear cache (admin only)
- `!yes` - Quick save after user history
- `!no` - Skip saving after user history

## 🎯 Use Cases
1versation Analysis**: Track and analyze Discord conversations
2. **Response Tracking**: Understand how conversations flow
3**User Behavior**: Analyze user interaction patterns
4. **Data Collection**: Gather conversation data for analysis

## 🔒 Privacy & Security

- **Local Storage**: All data stored locally
- **Admin Controls**: Sensitive operations require admin permissions
- **Data Retention**: Configurable cache limits
- **No External Sharing**: No data transmitted to external services

## 📚 Documentation

- [Discord Bot Documentation](README_DISCORD.md)

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
1Check the [Discord Bot Documentation](README_DISCORD.md)2. Review logs in `discord_bot.log`
3Create an issue in the project repository

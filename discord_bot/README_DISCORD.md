# DeepDiscord Bot

A powerful Discord bot designed to track, retrieve, and analyze message relationships and conversations. Part of the DeepDiscord project for AI-powered Discord analysis.

## Features

### 🎯 Core Functionality
- **Message Retrieval**: Get any message by its ID
- **Response Tracking**: Automatically track who responds to whom
- **Message Chains**: View complete conversation threads
- **Real-time Monitoring**: Track messages as they happen
- **Data Persistence**: Save and load message data

### 📊 Commands

#### Message Commands
- `!getmsg <message_id>` - Retrieve a specific message by ID
- `!responses <message_id>` - Show all responses to a message
- `!chain <message_id>` - Display the full conversation chain
- `!stats` - Show bot statistics and usage data

#### Admin Commands
- `!save` - Save current message data to disk (Admin only)
- `!clear` - Clear message cache (Admin only)

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- A Discord bot token

### 2. Install Dependencies
```bash
# Install Discord bot dependencies
pip install -r requirements_discord.txt

# Or install individually
pip install discord.py python-dotenv aiohttp
```

### 3. Discord Bot Setup

1. **Create a Discord Application**:
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application"
   - Give it a name (e.g., "DeepDiscord Bot")

2. **Create a Bot**:
   - Go to the "Bot" section
   - Click "Add Bot"
   - Copy the bot token

3. **Set Bot Permissions**:
   - Go to "OAuth2" → "URL Generator"
   - Select scopes: `bot`, `applications.commands`
   - Select permissions:
     - Read Messages/View Channels
     - Send Messages
     - Embed Links
     - Read Message History
     - Use Slash Commands
   - Use the generated URL to invite the bot to your server

### 4. Configuration

1. **Create Environment File**:
   ```bash
   cp env_example.txt .env
   ```

2. **Edit `.env`**:
   ```env
   DISCORD_TOKEN=your_actual_bot_token_here
   ```

## Usage

### Starting the Bot
```bash
python discord_bot.py
```

### Using Commands

#### Get a Message by ID
```
!getmsg 1234567890123456789
```
Returns detailed information about the message including:
- Message content
- Author information
- Channel location
- Response count
- Reply reference (if any)

#### View Responses to a Message
```
!responses 1234567890123456789
```
Shows all messages that replied to the specified message.

#### View Message Chain
```
!chain 1234567890123456789
```
Displays the original message and all its responses in chronological order.

#### Check Bot Statistics
```
!stats
```
Shows:
- Number of cached messages
- Response chains tracked
- Total responses
- Guild and channel counts
- Bot uptime

## Architecture

### Core Components

#### MessageTracker Class
- **Purpose**: Manages message caching and relationship tracking
- **Features**:
  - In-memory message cache (configurable size)
  - Response chain mapping
  - Automatic cache management (FIFO)

#### DeepDiscordBot Class
- **Purpose**: Main bot class with Discord.py integration
- **Features**:
  - Event handling (message, edit, delete)
  - Data persistence
  - Command processing

#### MessageCommands Cog
- **Purpose**: User-facing commands for message retrieval
- **Commands**: `getmsg`, `responses`, `chain`, `stats`

#### AdminCommands Cog
- **Purpose**: Administrative functions
- **Commands**: `save`, `clear`

### Data Flow

1. **Message Reception**: Bot receives messages via Discord API
2. **Caching**: Messages are stored in memory cache
3. **Relationship Tracking**: Response relationships are mapped
4. **Command Processing**: Users can query cached data
5. **Persistence**: Data can be saved to disk for backup

### File Structure
```
DeepDiscord/
├── discord_bot.py          # Main bot file
├── requirements_discord.txt # Discord dependencies
├── env_example.txt         # Environment template
├── discord_data/           # Data storage directory
│   └── messages.json       # Saved message data
└── discord_bot.log         # Bot logs
```

## Configuration Options

### Environment Variables
- `DISCORD_TOKEN`: Your bot token (required)
- `BOT_PREFIX`: Command prefix (default: `!`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### Bot Settings
- **Cache Size**: 10,000 messages (configurable in `MessageTracker`)
- **Command Prefix**: `!` (configurable)
- **Data Directory**: `discord_data/` (configurable)

## Security & Privacy

### Data Handling
- **Message Content**: Stored in memory and optionally on disk
- **User Information**: Only basic info (ID, name) is stored
- **Permissions**: Admin commands require administrator privileges
- **Data Retention**: Cache uses FIFO eviction (oldest messages removed first)

### Privacy Considerations
- Bot only tracks messages in channels it has access to
- Users can request data deletion via admin commands
- No personal data is transmitted to external services
- All data is stored locally

## Troubleshooting

### Common Issues

#### Bot Not Responding
- Check if bot token is correct in `.env`
- Verify bot has proper permissions in Discord server
- Check bot is online in Discord Developer Portal

#### Commands Not Working
- Ensure bot has "Send Messages" permission
- Check command prefix (default: `!`)
- Verify bot can read message history

#### Memory Issues
- Reduce cache size in `MessageTracker.max_cache_size`
- Use `!clear` command to clear cache
- Monitor memory usage with `!stats`

### Logs
- Check `discord_bot.log` for detailed error information
- Log level can be adjusted via `LOG_LEVEL` environment variable

## Development

### Adding New Commands
1. Create a new method in `MessageCommands` or `AdminCommands`
2. Use the `@commands.command()` decorator
3. Add proper error handling
4. Update this README

### Extending Functionality
- **Database Integration**: Replace in-memory cache with database
- **Web Dashboard**: Add web interface for data visualization
- **Analytics**: Add message analysis and statistics
- **Export Features**: Add data export capabilities

## Integration with DeepSeek

This Discord bot is designed to work with the DeepSeek AI training pipeline:

1. **Data Collection**: Bot collects conversation data
2. **Data Processing**: Messages can be processed for training
3. **AI Analysis**: DeepSeek models can analyze conversation patterns
4. **Response Generation**: AI can generate contextual responses

## License

This project is part of the DeepDiscord project. See the main project license for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `discord_bot.log`
3. Create an issue in the project repository 
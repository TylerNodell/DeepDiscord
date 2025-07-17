# DeepDiscord Deployment Guide

This guide explains how to deploy the DeepDiscord Discord bot as a background service.

## 📋 Prerequisites

- Python 3.8 or higher
- Discord bot token
- Linux/macOS system (for background services)
- Git

## 🚀 Quick Installation

### 1. Automated Installation
```bash
# Clone the repository
git clone <repository-url>
cd DeepDiscord

# Run the install script
./install.sh
```

### 2. Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp env_example.txt .env
# Edit .env and add your Discord bot token
```

## 🎯 Running the Bot

### Interactive Mode
```bash
# Use the launcher
python run.py

# Or run directly
python discord_bot.py
```

### Background Service Mode
```bash
# Use the start script
./start.sh
```

## 🔧 Background Service Setup

### Option1 Systemd (Linux)

1. **Edit the service file**:
   ```bash
   # Edit deepdiscord.service
   # Replace YOUR_USERNAME and /path/to/DeepDiscord with actual values
   ```2nstall the service**:
   ```bash
   sudo cp deepdiscord.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable deepdiscord
   sudo systemctl start deepdiscord
   ```
3. **Check status**:
   ```bash
   sudo systemctl status deepdiscord
   sudo journalctl -u deepdiscord -f
   ```

### Option2 Supervisor

1. **Install supervisor**:
   ```bash
   sudo apt-get install supervisor  # Ubuntu/Debian
   # or
   brew install supervisor  # macOS
   ```

2. **Edit the configuration**:
   ```bash
   # Edit supervisor.conf
   # Replace YOUR_USERNAME and /path/to/DeepDiscord with actual values
   ```3 the configuration**:
   ```bash
   sudo cp supervisor.conf /etc/supervisor/conf.d/deepdiscord.conf
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start deepdiscord
   ```
4. **Check status**:
   ```bash
   sudo supervisorctl status deepdiscord
   sudo tail -f /path/to/DeepDiscord/logs/deepdiscord.out.log
   ```

### Option 3: PM2js Process Manager)

1 **Install PM2
   ```bash
   npm install -g pm2
   ```

2. **Create ecosystem file**:
   ```bash
   # Create ecosystem.config.js
   module.exports = {
     apps: [{
       name: 'deepdiscord',
       script: discord_bot.py',
       interpreter: ./venv/bin/python',
       cwd: /path/to/DeepDiscord',
       env: [object Object]
         NODE_ENV: 'production     },
       log_file: './logs/combined.log,
       out_file: ./logs/out.log',
       error_file:./logs/error.log,   log_date_format:YYYY-MM-DD HH:mm:ss Z',
       autorestart: true,
       watch: false,
       max_memory_restart: 1G'
     }]
   }
   ```

3. **Start the service**:
   ```bash
   pm2 start ecosystem.config.js
   pm2 save
   pm2startup
   ```

## 📊 Monitoring

### Log Files
- **Systemd**: `sudo journalctl -u deepdiscord -f`
- **Supervisor**: `/path/to/DeepDiscord/logs/deepdiscord.out.log`
- **PM2**: `pm2logs deepdiscord`
- **Manual**: `logs/discord_bot_YYYYMMDD_HHMMSS.log`

### Status Commands
- **Systemd**: `sudo systemctl status deepdiscord`
- **Supervisor**: `sudo supervisorctl status deepdiscord`
- **PM2**: `pm2 status`

## 🔄 Maintenance

### Updating the Bot
```bash
# Stop the service
sudo systemctl stop deepdiscord  # or supervisorctl stop deepdiscord

# Pull latest changes
git pull

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Start the service
sudo systemctl start deepdiscord  # or supervisorctl start deepdiscord
```

### Backup
```bash
# Backup data directory
tar -czf backup_$(date +%Y%m%d).tar.gz discord_data/ logs/
```

## 🛠️ Troubleshooting

### Common Issues

1. **Bot not starting**:
   - Check `.env` file exists and has valid token
   - Verify virtual environment is activated
   - Check logs for error messages

2. **Permission denied**:
   - Ensure script files are executable: `chmod +x *.sh`
   - Check file ownership and permissions

3ervice won't start**:
   - Verify paths in service configuration files
   - Check system logs: `sudo journalctl -xe`4 **Bot disconnects frequently**:
   - Check network connectivity
   - Verify Discord API status
   - Review bot permissions

### Debug Mode
```bash
# Run with verbose logging
python discord_bot.py --debug
```

## 📞 Support

For issues and questions:1Check the logs for error messages
2. Review the [Discord Bot Documentation](README_DISCORD.md)
3Create an issue in the project repository 
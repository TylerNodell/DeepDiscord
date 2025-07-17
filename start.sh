#!/bin/bash

# DeepDiscord Start Script
# Starts the Discord bot as a background service

set -e  # Exit on any error

echo "🤖 DeepDiscord Start Script"
echo "==========================="

# Check if virtual environment exists
if [ ! -d venv ]; then
    echo "❌ Virtual environment not found!  echo "Please run install.sh firstexit 1i

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!  echo "Please run install.sh first or create .env file manually  exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment...
source venv/bin/activate

# Check if discord_bot.py exists
if [ ! -f discord_bot.py ]; then
    echo❌discord_bot.py not found!
    exit 1
fi

# Create logs directory if it doesnt exist
if [ ! -d logs ]; then
    mkdir -p logs
fi

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=logs/discord_bot_${timestamp}.log"

echo "🚀 Starting Discord bot..."
echo "📝 Logs will be written to: $log_file"
echo "🛑 Press Ctrl+C to stop the bot
echo "

#Start the bot with logging
python discord_bot.py 2>&1| tee "$log_file" 
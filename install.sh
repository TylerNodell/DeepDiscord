#!/bin/bash

# DeepDiscord Install Script
# Installs the Discord bot and its dependencies

set -e  # Exit on any error

echo "🤖 DeepDiscord Install Script"
echo "=============================="

# Check if Python 3.8 is installed
echo "📋 Checking Python version..."
python_version=$(python3ersion 2>&1 | grep -oE [0-9]+\.[0 head -1)
required_version="3.8
if [ "$(printf%sn$required_version$python_version| sort -V | head -n1)" = $required_version]; then
    echo✅ Python $python_version found (>= $required_version)
else    echo❌ Python3.8is required. Found: $python_version
    echo "Please install Python 3.8r higher
    exit 1
fi

# Create virtual environment if it doesnt exist
if [ ! -d venv]; then
    echo "🔧 Creating virtual environment..."
    python3venv venv
    echo "✅ Virtual environment created
else   echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment...
source venv/bin/activate

# Upgrade pip
echo📦Upgrading pip...
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesntexist
if [ ! -f .env]; then
    echo📝Creating .env file from template..."
    cp env_example.txt .env
    echo ⚠️  Please edit .env file and add your Discord bot token  echo   You can get a token from: https://discord.com/developers/applicationselse
    echo "✅ .env file already exists"
fi

# Create discord_data directory if it doesnt exist
if [ ! -ddiscord_data]; then
    echo "📁 Creating discord_data directory..."
    mkdir -p discord_data
    echo "✅ discord_data directory created
else   echo "✅ discord_data directory already existsfi

echo ""
echo "🎉 Installation complete!"
echo "
echo Next steps:"
echo 1.Edit .env file and add your Discord bot token"
echo 2. Invite the bot to your Discord server"
echo "3. Run: python test_discord_bot.py (to test setup)"
echo "4. Run: python discord_bot.py (to start the bot)"
echor more information, see README.md" 
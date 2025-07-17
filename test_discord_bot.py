#!/usr/bin/env python3
"""
Test script for DeepDiscord Bot
Helps verify setup and basic functionality
"""

import os
import sys
import asyncio
import discord
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if environment is properly configured"""
    print("🔍 Testing Environment Configuration...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    # Check Discord token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        print("❌ DISCORD_TOKEN not found in environment variables")
        print("   Please create a .env file with your bot token")
        return False
    
    if token == "your_bot_token_here":
        print("❌ Please replace the placeholder token with your actual bot token")
        return False
    
    print("✅ Discord token found")
    
    # Check dependencies
    try:
        import discord
        print(f"✅ discord.py {discord.__version__} installed")
    except ImportError:
        print("❌ discord.py not installed")
        print("   Run: pip install discord.py")
        return False
    
    try:
        import dotenv
        print("✅ python-dotenv installed")
    except ImportError:
        print("❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False
    
    return True

async def test_discord_connection():
    """Test Discord API connection"""
    print("\n🔍 Testing Discord API Connection...")
    
    token = os.getenv('DISCORD_TOKEN')
    
    try:
        # Create a simple client to test connection
        intents = discord.Intents.default()
        client = discord.Client(intents=intents)
        
        @client.event
        async def on_ready():
            print(f"✅ Connected to Discord as {client.user.name}")
            print(f"✅ Bot ID: {client.user.id}")
            print(f"✅ Connected to {len(client.guilds)} guild(s)")
            
            # List guilds
            for guild in client.guilds:
                print(f"   - {guild.name} (ID: {guild.id})")
            
            await client.close()
        
        await client.start(token)
        
    except discord.LoginFailure:
        print("❌ Invalid bot token")
        print("   Please check your DISCORD_TOKEN in the .env file")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if required files exist"""
    print("\n🔍 Testing File Structure...")
    
    required_files = [
        'discord_bot.py',
        'requirements_discord.txt',
        'env_example.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Check if .env exists
    if os.path.exists('.env'):
        print("✅ .env file exists")
    else:
        print("⚠️  .env file not found (create from env_example.txt)")
    
    # Check if discord_data directory exists
    if os.path.exists('discord_data'):
        print("✅ discord_data directory exists")
    else:
        print("ℹ️  discord_data directory will be created when bot starts")
    
    return True

def test_bot_code():
    """Test if bot code can be imported"""
    print("\n🔍 Testing Bot Code...")
    
    try:
        # Try to import the bot module
        import discord_bot
        print("✅ discord_bot.py can be imported")
        
        # Check if main classes exist
        if hasattr(discord_bot, 'DeepDiscordBot'):
            print("✅ DeepDiscordBot class found")
        else:
            print("❌ DeepDiscordBot class not found")
            return False
        
        if hasattr(discord_bot, 'MessageTracker'):
            print("✅ MessageTracker class found")
        else:
            print("❌ MessageTracker class not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing discord_bot: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing bot code: {e}")
        return False

def print_setup_instructions():
    """Print setup instructions"""
    print("\n📋 Setup Instructions:")
    print("=" * 50)
    print("1. Create a Discord Application at https://discord.com/developers/applications")
    print("2. Create a bot in your application")
    print("3. Copy the bot token")
    print("4. Create a .env file:")
    print("   cp env_example.txt .env")
    print("5. Edit .env and add your bot token:")
    print("   DISCORD_TOKEN=your_actual_token_here")
    print("6. Invite the bot to your server with these permissions:")
    print("   - Read Messages/View Channels")
    print("   - Send Messages")
    print("   - Embed Links")
    print("   - Read Message History")
    print("7. Run the bot:")
    print("   python discord_bot.py")

def main():
    """Main test function"""
    print("🧪 DeepDiscord Bot Test Suite")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Run tests
    if not test_environment():
        all_tests_passed = False
    
    if not test_file_structure():
        all_tests_passed = False
    
    if not test_bot_code():
        all_tests_passed = False
    
    # Test Discord connection (requires valid token)
    if os.getenv('DISCORD_TOKEN') and os.getenv('DISCORD_TOKEN') != "your_bot_token_here":
        try:
            asyncio.run(test_discord_connection())
        except Exception as e:
            print(f"⚠️  Discord connection test failed: {e}")
            all_tests_passed = False
    else:
        print("\n⚠️  Skipping Discord connection test (no valid token)")
    
    # Print results
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("✅ All tests passed! Your bot is ready to run.")
        print("   Run: python discord_bot.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print_setup_instructions()
    
    print("\n📚 For more information, see README_DISCORD.md")

if __name__ == "__main__":
    main() 
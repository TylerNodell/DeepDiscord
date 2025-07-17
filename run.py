#!/usr/bin/env python3
"""
DeepDiscord Launcher
Provides an easy way to run the Discord bot
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print project banner"""
    print("=" * 60)
    print("🤖 DeepDiscord - Discord Analysis Bot")
    print("=" * 60)

def check_venv():
    """Check if virtual environment is activated"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected")
        print("   It's recommended to activate the virtual environment:")
        print("   source venv/bin/activate  # On macOS/Linux")
        print("   venv\\Scripts\\activate    # On Windows")
        print()

def show_menu():
    """Show main menu"""
    print("\n📋 Available Options:")
    print("1. 🎯 Start Discord Bot")
    print("2. 🧪 Test Bot Setup")
    print("3. 📚 View Documentation")
    print("4. 🚪 Exit")
    print()

def run_discord_bot():
    """Run the Discord bot"""
    print("\n🎯 Starting Discord Bot...")
    print("Make sure you have:")
    print("- Created a .env file with your DISCORD_TOKEN")
    print("- Invited the bot to your Discord server")
    print("- Given the bot proper permissions")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Please copy env_example.txt to .env and add your bot token")
        return
    
    try:
        subprocess.run([sys.executable, "discord_bot.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running bot: {e}")
    except FileNotFoundError:
        print("❌ discord_bot.py not found!")

def test_discord_bot():
    """Test Discord bot setup"""
    print("\n🧪 Testing Discord Bot Setup...")
    
    test_file = Path("test_discord_bot.py")
    if not test_file.exists():
        print("❌ test_discord_bot.py not found!")
        return
    
    try:
        subprocess.run([sys.executable, "test_discord_bot.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
    except FileNotFoundError:
        print("❌ Test file not found!")

def show_documentation():
    """Show documentation options"""
    print("\n📚 Documentation:")
    print("1. 📖 Main Project README")
    print("2. 🤖 Discord Bot Documentation")
    print("3. Back to main menu")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        if Path("README.md").exists():
            with open("README.md", "r") as f:
                print(f.read())
        else:
            print("❌ README.md not found!")
    
    elif choice == "2":
        doc_file = Path("README_DISCORD.md")
        if doc_file.exists():
            with open(doc_file, "r") as f:
                print(f.read())
        else:
            print("❌ Discord bot documentation not found!")
    
    elif choice == "3":
        return
    
    else:
        print("❌ Invalid option")

def main():
    """Main launcher function"""
    print_banner()
    check_venv()
    
    while True:
        show_menu()
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            run_discord_bot()
        elif choice == "2":
            test_discord_bot()
        elif choice == "3":
            show_documentation()
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please select 1-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
DeepDiscord Launcher
Provides an easy way to run different components of the project
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print project banner"""
    print("=" * 60)
    print("🤖 DeepDiscord - AI-Powered Discord Analysis Platform")
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
    print("\n📋 Available Components:")
    print("1. 🎯 Discord Bot - Message tracking and analysis")
    print("2. 🧠 Training Module - AI model setup and training")
    print("3. 🧪 Test Discord Bot Setup")
    print("4. 📚 View Documentation")
    print("5. 🚪 Exit")
    print()

def run_discord_bot():
    """Run the Discord bot"""
    print("\n🎯 Starting Discord Bot...")
    print("Make sure you have:")
    print("- Created a .env file in discord_bot/ with your DISCORD_TOKEN")
    print("- Invited the bot to your Discord server")
    print("- Given the bot proper permissions")
    
    bot_dir = Path("discord_bot")
    if not bot_dir.exists():
        print("❌ discord_bot/ directory not found!")
        return
    
    env_file = bot_dir / ".env"
    if not env_file.exists():
        print("❌ .env file not found in discord_bot/")
        print("   Please copy env_example.txt to .env and add your bot token")
        return
    
    try:
        os.chdir(bot_dir)
        subprocess.run([sys.executable, "discord_bot.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running bot: {e}")
    except FileNotFoundError:
        print("❌ discord_bot.py not found!")
    finally:
        os.chdir("..")

def run_training():
    """Run the training module"""
    print("\n🧠 Training Module Options:")
    print("1. Setup Model (Initialize and test)")
    print("2. Train Model (Fine-tuning)")
    print("3. Back to main menu")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    training_dir = Path("training")
    if not training_dir.exists():
        print("❌ training/ directory not found!")
        return
    
    try:
        os.chdir(training_dir)
        
        if choice == "1":
            print("\n🔧 Setting up DeepSeek model...")
            subprocess.run([sys.executable, "deepseek_setup.py"], check=True)
        elif choice == "2":
            print("\n🚀 Starting model training...")
            subprocess.run([sys.executable, "deepseek_training.py"], check=True)
        elif choice == "3":
            os.chdir("..")
            return
        else:
            print("❌ Invalid option")
        
        os.chdir("..")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running training: {e}")
        os.chdir("..")
    except FileNotFoundError as e:
        print(f"❌ Training file not found: {e}")
        os.chdir("..")

def test_discord_bot():
    """Test Discord bot setup"""
    print("\n🧪 Testing Discord Bot Setup...")
    
    bot_dir = Path("discord_bot")
    if not bot_dir.exists():
        print("❌ discord_bot/ directory not found!")
        return
    
    test_file = bot_dir / "test_discord_bot.py"
    if not test_file.exists():
        print("❌ test_discord_bot.py not found!")
        return
    
    try:
        os.chdir(bot_dir)
        subprocess.run([sys.executable, "test_discord_bot.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
    except FileNotFoundError:
        print("❌ Test file not found!")
    finally:
        os.chdir("..")

def show_documentation():
    """Show documentation options"""
    print("\n📚 Documentation:")
    print("1. 📖 Main Project README")
    print("2. 🤖 Discord Bot Documentation")
    print("3. 🧠 Training Module Documentation")
    print("4. Back to main menu")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        if Path("README.md").exists():
            with open("README.md", "r") as f:
                print(f.read())
        else:
            print("❌ README.md not found!")
    
    elif choice == "2":
        doc_file = Path("discord_bot/README_DISCORD.md")
        if doc_file.exists():
            with open(doc_file, "r") as f:
                print(f.read())
        else:
            print("❌ Discord bot documentation not found!")
    
    elif choice == "3":
        doc_file = Path("training/README_TRAINING.md")
        if doc_file.exists():
            with open(doc_file, "r") as f:
                print(f.read())
        else:
            print("❌ Training documentation not found!")
    
    elif choice == "4":
        return
    
    else:
        print("❌ Invalid option")

def main():
    """Main launcher function"""
    print_banner()
    check_venv()
    
    while True:
        show_menu()
        choice = input("Select component (1-5): ").strip()
        
        if choice == "1":
            run_discord_bot()
        elif choice == "2":
            run_training()
        elif choice == "3":
            test_discord_bot()
        elif choice == "4":
            show_documentation()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please select 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 
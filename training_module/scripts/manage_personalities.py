#!/usr/bin/env python3
"""
Personality management script for Discord training data.
Allows discovery, addition, and management of personality profiles.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

# Setup path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.personality_config import PersonalityConfig, PersonalityProfile, DEFAULT_PERSONALITY_CONFIG


def discover_personalities_from_files(data_dir: Path) -> Dict[str, Dict]:
    """Discover personalities from processed training files."""
    all_training_data = []
    
    # Load all ChatML files
    chatml_dir = data_dir / "chatml"
    if not chatml_dir.exists():
        print(f"No chatml directory found at {chatml_dir}")
        return {}
    
    for json_file in chatml_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_training_data.extend(data)
                print(f"Loaded {len(data)} entries from {json_file.name}")
        except Exception as e:
            print(f"Failed to load {json_file}: {e}")
    
    if not all_training_data:
        print("No training data found")
        return {}
    
    # Use personality config to discover personalities
    config = DEFAULT_PERSONALITY_CONFIG
    discovered = config.discover_personalities_from_data(all_training_data)
    
    return discovered


def save_personality_config(personalities: Dict[str, PersonalityProfile], config_file: Path):
    """Save personality configuration to JSON file."""
    config_data = {
        "personalities": {
            user_id: {
                "user_id": profile.user_id,
                "personality_name": profile.personality_name,
                "discord_username": profile.discord_username,
                "discord_display_name": profile.discord_display_name,
                "description": profile.description,
                "communication_style": profile.communication_style,
                "typical_phrases": profile.typical_phrases,
                "min_samples": profile.min_samples,
                "max_samples": profile.max_samples,
                "quality_threshold": profile.quality_threshold,
                "lora_rank": profile.lora_rank,
                "lora_alpha": profile.lora_alpha,
                "lora_dropout": profile.lora_dropout
            }
            for user_id, profile in personalities.items()
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Saved personality configuration to {config_file}")


def load_personality_config(config_file: Path) -> Dict[str, PersonalityProfile]:
    """Load personality configuration from JSON file."""
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        personalities = {}
        for user_id, data in config_data.get("personalities", {}).items():
            profile = PersonalityProfile(
                user_id=data["user_id"],
                personality_name=data["personality_name"],
                discord_username=data.get("discord_username", ""),
                discord_display_name=data.get("discord_display_name", ""),
                description=data.get("description", ""),
                communication_style=data.get("communication_style", ""),
                typical_phrases=data.get("typical_phrases", []),
                min_samples=data.get("min_samples", 500),
                max_samples=data.get("max_samples", 2000),
                quality_threshold=data.get("quality_threshold", 0.7),
                lora_rank=data.get("lora_rank", 32),
                lora_alpha=data.get("lora_alpha", 64),
                lora_dropout=data.get("lora_dropout", 0.1)
            )
            personalities[user_id] = profile
        
        print(f"Loaded {len(personalities)} personalities from {config_file}")
        return personalities
        
    except Exception as e:
        print(f"Failed to load personality config: {e}")
        return {}


def print_personality_summary(personalities: Dict[str, PersonalityProfile]):
    """Print a summary of discovered personalities."""
    print(f"\n=== Discovered {len(personalities)} Personalities ===")
    
    for user_id, profile in personalities.items():
        print(f"\nUser ID: {user_id}")
        print(f"  Current Name: {profile.personality_name}")
        print(f"  Discord Username: {profile.discord_username}")
        print(f"  Sample Range: {profile.min_samples}-{profile.max_samples}")
        print(f"  Description: {profile.description}")


def update_personality_names(personalities: Dict[str, PersonalityProfile]) -> Dict[str, PersonalityProfile]:
    """Interactive session to update personality names."""
    print("\n=== Update Personality Names ===")
    print("For each personality, provide a human-readable name to use for training and Discord commands.")
    print("Press Enter to keep the current name, or type 'skip' to skip all remaining.")
    
    for user_id, profile in personalities.items():
        print(f"\nUser ID: {user_id}")
        print(f"Discord Username: {profile.discord_username}")
        print(f"Current Name: {profile.personality_name}")
        print(f"Message Count: {profile.min_samples}-{profile.max_samples}")
        
        new_name = input(f"Enter new personality name (or Enter to keep '{profile.personality_name}'): ").strip()
        
        if new_name.lower() == 'skip':
            print("Skipping remaining personalities...")
            break
        elif new_name:
            profile.personality_name = new_name
            print(f"Updated to: {new_name}")
        else:
            print(f"Keeping: {profile.personality_name}")
    
    return personalities


def main():
    parser = argparse.ArgumentParser(description="Manage personality profiles for Discord training")
    parser.add_argument("--data-dir", type=str, default="./data/processed", 
                       help="Directory containing processed training data")
    parser.add_argument("--config-file", type=str, default="./config/personalities.json",
                       help="Personality configuration file")
    parser.add_argument("--discover", action="store_true", 
                       help="Discover personalities from training data")
    parser.add_argument("--update-names", action="store_true",
                       help="Interactively update personality names")
    parser.add_argument("--list", action="store_true",
                       help="List current personalities")
    parser.add_argument("--add-personality", nargs=3, metavar=("USER_ID", "NAME", "DESCRIPTION"),
                       help="Add new personality: USER_ID NAME DESCRIPTION")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    config_file = Path(args.config_file)
    
    # Ensure config directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing configuration
    personalities = load_personality_config(config_file)
    
    if args.discover:
        print("Discovering personalities from training data...")
        discovered = discover_personalities_from_files(data_dir)
        
        if discovered:
            # Merge with existing personalities, preferring existing ones
            for user_id, profile in discovered.items():
                if user_id not in personalities:
                    personalities[user_id] = profile
                    print(f"Added new personality: {profile.personality_name} (ID: {user_id})")
                else:
                    print(f"Personality already exists: {personalities[user_id].personality_name} (ID: {user_id})")
            
            save_personality_config(personalities, config_file)
        else:
            print("No personalities discovered")
    
    if args.add_personality:
        user_id, name, description = args.add_personality
        
        if user_id in personalities:
            print(f"Personality already exists for user ID {user_id}")
        else:
            profile = PersonalityProfile(
                user_id=user_id,
                personality_name=name,
                description=description
            )
            personalities[user_id] = profile
            save_personality_config(personalities, config_file)
            print(f"Added personality: {name} (ID: {user_id})")
    
    if args.update_names and personalities:
        personalities = update_personality_names(personalities)
        save_personality_config(personalities, config_file)
    
    if args.list or not any([args.discover, args.update_names, args.add_personality]):
        if personalities:
            print_personality_summary(personalities)
        else:
            print("No personalities configured. Use --discover to find personalities from training data.")
    
    print(f"\nPersonality configuration saved to: {config_file}")
    print("You can now use these personalities in training with the preprocessing script.")


if __name__ == "__main__":
    main()
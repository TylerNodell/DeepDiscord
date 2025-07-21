#!/usr/bin/env python3
"""
Training script for Sherif (ID: 191949803154833408) using multiple_lora method.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config.personality_config import PersonalityConfig, PersonalityStrategy, PersonalityProfile
from config.training_config import TrainingConfig

def main():
    """Set up and run training for Sherif personality."""
    
    # Create personality configuration using multiple_lora strategy
    personality_config = PersonalityConfig(
        strategy=PersonalityStrategy.MULTIPLE_LORA,
        balance_samples=True,
        enable_quality_filtering=True,
        require_consent=False,  # Training data already processed
        min_personalities=1,
        max_personalities=1
    )
    
    # Add Sherif's personality profile
    sherif_profile = PersonalityProfile(
        user_id="191949803154833408",
        personality_name="Sherif",
        discord_username="TerminatorSalvationMachineP2",
        description="Sherif personality from TerminatorSalvationMachineP2",
        min_samples=500,
        max_samples=2000,
        quality_threshold=0.7,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.1
    )
    
    personality_config.add_personality(sherif_profile)
    
    # Load the high confidence training data
    data_path = Path("data/raw/high_confidence_191949803154833408_20250719_035342.json")
    
    print(f"🔍 Loading training data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"📊 Loaded {len(training_data.get('training_data', []))} training pairs")
    
    # Create training configuration
    training_config = TrainingConfig(
        learning_rate=2e-4,
        batch_size=2,  # Conservative for multiple LoRA
        epochs=3,
        gradient_accumulation_steps=4,
        max_length=2048,
        use_qlora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        output_dir="./checkpoints/sherif_lora",
        save_steps=250,
        eval_steps=250,
        logging_steps=50
    )
    
    print("🚀 Starting Sherif personality training with multiple_lora method...")
    print(f"📋 Strategy: {personality_config.strategy.value}")
    print(f"👤 Personality: {sherif_profile.personality_name} (ID: {sherif_profile.user_id})")
    print(f"💾 Output directory: {training_config.output_dir}")
    
    # Training summary
    summary = personality_config.get_training_summary()
    print(f"📈 Training Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Import and run the actual training
    try:
        from scripts.train_dolphin import main as train_main
        
        # Set up arguments for the training script
        import argparse
        
        # Create mock args for the training script
        class Args:
            personality_strategy = "multiple_lora"
            personality_id = "191949803154833408:Sherif"
            data_file = str(data_path)
            output_dir = training_config.output_dir
            epochs = training_config.epochs
            batch_size = training_config.batch_size
            learning_rate = training_config.learning_rate
            use_gpu = True
            use_unsloth = False  # Start with standard training
            debug = False
            
        # Run training
        print("🎯 Launching training process...")
        
        # Set up sys.argv for the training script
        import os
        from datetime import datetime
        
        original_argv = sys.argv
        sys.argv = [
            'train_dolphin.py',
            '--data', str(data_path),
            '--output-dir', training_config.output_dir,
            '--run-name', f"sherif_multiple_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ]
        
        try:
            train_main()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
        
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        print("📝 Training configuration prepared. Manual training launch required.")
        
        # Save configuration for manual use
        config_path = Path("sherif_training_config.json")
        config_data = {
            "personality_config": {
                "strategy": personality_config.strategy.value,
                "personality": {
                    "user_id": sherif_profile.user_id,
                    "name": sherif_profile.personality_name,
                    "discord_username": sherif_profile.discord_username,
                    "description": sherif_profile.description
                }
            },
            "training_config": {
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.batch_size,
                "epochs": training_config.epochs,
                "output_dir": training_config.output_dir,
                "data_file": str(data_path)
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Configuration saved to {config_path}")
        print("🔧 To run training manually:")
        print(f"   python scripts/train_dolphin.py --personality-strategy multiple_lora --personality-id '191949803154833408:Sherif' --data-file '{data_path}'")

if __name__ == "__main__":
    main()
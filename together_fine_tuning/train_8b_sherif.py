#!/usr/bin/env python3
"""
Train Sherif 8B model on Together.ai
"""

import os
import together
from datetime import datetime

# Set API key
together.api_key = os.getenv("TOGETHER_API_KEY")

if not together.api_key:
    print("❌ Please set TOGETHER_API_KEY environment variable")
    exit(1)

print("🚀 Starting Sherif 8B training...")

# Create fine-tuning job
try:
    response = together.Finetune.create(
        training_file="file-ba7c0eb5-0696-4cf5-9841-43ab58f05d06",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        suffix="sherif-chaos-8b",
        n_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        wandb_api_key=None
    )
    
    print(f"✅ Fine-tuning job created!")
    print(f"📋 Job ID: {response['id']}")
    print(f"🎯 Model: {response['model']}")
    print(f"⏱️ Status: {response['status']}")
    print(f"\n📊 Monitor with: together fine-tuning retrieve {response['id']}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔧 Try manual command:")
    print("together fine-tuning create \\")
    print('  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \\')
    print('  --training-file "file-ba7c0eb5-0696-4cf5-9841-43ab58f05d06" \\')
    print('  --suffix "sherif-chaos-8b" \\')
    print("  --n-epochs 3 \\")
    print("  --confirm")
#!/bin/bash

echo "🧹 Cleaning up junk files in DeepDiscord..."
echo "============================================"

# Change to the DeepDiscord directory
cd /home/nuko/dev/DeepDiscord

# Remove test scripts that are no longer needed
echo "📝 Removing temporary test scripts..."
rm -f test_sherif.py
rm -f test_hybrid_sherif.py
rm -f test_personality_prompts.py
rm -f test_training_compatibility.py
rm -f simple_test_sherif.py
rm -f chat_with_sherif.py
rm -f generate_short_responses.py

# Remove response output files
echo "📄 Removing temporary response files..."
rm -f sherif_responses_*.txt
rm -f sherif_responses_*.json
rm -f sherif_911_absurd_*.txt
rm -f sherif_911_absurd_*.json

# Remove old log files
echo "📋 Removing old log files..."
rm -f training.log
rm -f training_module/inference.log
rm -f training_module/setup.log
rm -f training_module/sherif_training.log
rm -f training_module/sherif_training_fixed.log
rm -f training_module/training.log

# Remove old test checkpoints
echo "🗑️ Removing old test checkpoints..."
rm -rf training_module/checkpoints/test_run_20250720_074246
rm -rf training_module/checkpoints/sherif_dolphin
rm -rf training_module/checkpoints/sherif_multiple_lora

# Remove old intermediate training data
echo "📊 Removing old training data..."
rm -f training_module/data/raw/deepleffen_training_20250720_181555.json
rm -f training_module/data/raw/deepleffen_comprehensive_20250720_181934.json

# Clean up old training test scripts
echo "🧪 Removing old training test scripts..."
rm -f training_module/test_sherif_docker.py
rm -f training_module/train_small_test.py

# Keep these important files:
# - DeepleffenTweets.txt (source data)
# - process_deepleffen_tweets.py (processing script)
# - interactive_sherif_chat.py (interactive chat)
# - generate_sherif_911_responses.py (response generator)
# - start_sherif_chat.sh (startup script)
# - Current model checkpoints (sherif_lora_20250720_195311, hybrid_sherif_20250721_055822)
# - Training data files
# - Configuration files

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 Kept important files:"
echo "  - DeepleffenTweets.txt"
echo "  - Training scripts and data"
echo "  - Model checkpoints"
echo "  - Interactive chat scripts"
echo "  - Configuration files"
echo ""

# Show remaining files
echo "📋 Remaining files in main directory:"
ls -la | grep -E '\.(py|txt|sh)$' | grep -v __pycache__
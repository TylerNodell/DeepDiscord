#!/bin/bash
# Training script for Meta-Llama-3.1-70B-Instruct-Turbo

echo "🚀 Setting up Llama 3.1 70B fine-tuning for Sherif..."

# Check if API key is set
if [ -z "$TOGETHER_API_KEY" ]; then
    echo "❌ Error: TOGETHER_API_KEY not set!"
    echo "Run: export TOGETHER_API_KEY='your-key-here'"
    exit 1
fi

# Install Together.ai if not already installed
echo "📦 Installing Together.ai SDK..."
pip install together --upgrade

# Upload training file
echo "📤 Uploading training data..."
FILE_UPLOAD_RESPONSE=$(together files upload /home/nuko/dev/DeepDiscord/together_fine_tuning/data/sherif_together_uncensored.jsonl)
echo "Upload response: $FILE_UPLOAD_RESPONSE"

# Extract file ID (you may need to adjust this based on actual response format)
FILE_ID=$(echo $FILE_UPLOAD_RESPONSE | grep -o '"id": *"file-[^"]*"' | head -1 | sed -E 's/.*"file-([^"]*)".*/file-\1/')
echo "📁 File ID: $FILE_ID"

# Start fine-tuning with Llama 3.1 70B
echo "🎯 Starting fine-tuning with Meta-Llama-3.1-70B-Instruct-Turbo..."
together fine-tuning create \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference" \
  --training-file "$FILE_ID" \
  --suffix "sherif-maximum-chaos" \
  --n-epochs 3 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --confirm

echo "✅ Fine-tuning job submitted!"
echo "📊 Monitor progress with: together fine-tuning list"
echo "🔍 Get details with: together fine-tuning retrieve [job-id]"
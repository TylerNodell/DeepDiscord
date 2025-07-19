#!/bin/bash

echo "🎓 DeepDiscord Training Data Generator"
echo "====================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure."
    exit 1
fi

# Check if TEST_USER_ID is set
if ! grep -q "TEST_USER_ID=" .env; then
    echo "❌ TEST_USER_ID not found in .env. Please add your target user ID."
    exit 1
fi

echo "🔍 Generating training data for user: $(grep TEST_USER_ID .env | cut -d'=' -f2)"
echo "📅 Looking back: ${1:-30} days"
echo ""

# Run training data generator
python tools/training_data_generator.py

echo ""
echo "📁 Training data saved to training_data/ directory"
echo "🎉 Training data generation complete!"
#!/bin/bash

echo "🧪 Running DeepDiscord Test Suite"
echo "================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure."
    exit 1
fi

# Run specific test
if [ "$1" != "" ]; then
    echo "Running specific test: $1"
    python "tests/$1"
    exit $?
fi

# Run all tests
echo "Running all available tests..."

echo ""
echo "📊 Discord User Analysis Test:"
python tests/test_discord_user.py

echo ""
echo "🧩 Fragment Detection Test:"
python tests/test_specific_features.py

echo ""
echo "📈 User Data Analysis Test:"
python tests/test_user_data_analysis.py

echo ""
echo "🔧 All Features Test:"
python tests/test_all_features.py

echo ""
echo "✅ Test suite completed!"
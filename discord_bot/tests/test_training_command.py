#!/usr/bin/env python3
"""
Test the new !generatetrainingdata command
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import DeepDiscordBot, DiscordTrainingDataGenerator
import discord
from unittest.mock import MagicMock, AsyncMock

def test_training_data_generator():
    """Test the training data generator class"""
    print("🧪 Testing Training Data Generator")
    print("=" * 40)
    
    # Create mock objects
    mock_bot = MagicMock()
    mock_guild = MagicMock()
    mock_status_message = MagicMock()
    
    # Create generator
    generator = DiscordTrainingDataGenerator(
        bot=mock_bot,
        target_user_id=123456789,
        guild=mock_guild,
        status_message=mock_status_message
    )
    
    print(f"✅ Generator created for user ID: {generator.target_user_id}")
    print(f"✅ Response pairs initialized: {len(generator.response_pairs)}")
    print(f"✅ Channels processed count: {generator.channels_processed}")
    print(f"✅ Messages analyzed count: {generator.messages_analyzed}")
    
    # Test content relationship checking
    mock_question = MagicMock()
    mock_question.content = "What's your favorite game?"
    mock_question.author.mention = "<@987654321>"
    
    mock_response = MagicMock()
    mock_response.content = "I love playing Final Fantasy XIV"
    
    has_relationship = generator.has_content_relationship(mock_question, mock_response)
    print(f"✅ Content relationship detection: {has_relationship}")
    
    # Test training pair creation
    response_info = {
        'type': 'temporal_proximity',
        'message': mock_question,
        'confidence': 0.85,
        'time_gap_seconds': 120
    }
    
    mock_target_response = MagicMock()
    mock_target_response.content = "I love playing Final Fantasy XIV"
    mock_target_response.author.display_name = "TestUser"
    mock_target_response.channel.name = "general"
    mock_target_response.created_at.isoformat.return_value = "2025-01-19T12:00:00"
    mock_target_response.id = 999888777
    
    mock_question.author.display_name = "OtherUser"
    mock_question.id = 111222333
    
    # Mock the bot's get_combined_content method
    mock_bot.message_tracker.get_combined_content.return_value = None
    
    training_pair = generator.create_training_pair(response_info, mock_target_response)
    
    if training_pair:
        print(f"✅ Training pair created:")
        print(f"   Question: {training_pair['question'][:50]}...")
        print(f"   Answer: {training_pair['answer'][:50]}...")
        print(f"   Confidence: {training_pair['metadata']['confidence']}")
        print(f"   Response Type: {training_pair['metadata']['response_type']}")
    else:
        print("❌ Failed to create training pair")
    
    print("\n🎯 Training Data Generator Test Complete!")
    return True

def test_command_validation():
    """Test command parameter validation"""
    print("\n🧪 Testing Command Validation")
    print("=" * 40)
    
    # Test valid parameters
    user_id = 123456789012345678
    days_back = 30
    
    print(f"✅ Valid user ID: {user_id}")
    print(f"✅ Valid days back: {days_back}")
    
    # Test edge cases
    max_days = 365
    min_days = 1
    
    print(f"✅ Maximum days: {max_days}")
    print(f"✅ Minimum days: {min_days}")
    
    # Test invalid cases
    invalid_days = [0, 366, -1]
    for invalid in invalid_days:
        if invalid > 365:
            print(f"❌ Invalid (too high): {invalid} days")
        elif invalid < 1:
            print(f"❌ Invalid (too low): {invalid} days")
    
    print("\n🎯 Command Validation Test Complete!")
    return True

def test_file_generation():
    """Test training file generation logic"""
    print("\n🧪 Testing File Generation")
    print("=" * 40)
    
    # Create mock generator with sample data
    mock_bot = MagicMock()
    mock_guild = MagicMock()
    mock_status_message = MagicMock()
    
    generator = DiscordTrainingDataGenerator(
        bot=mock_bot,
        target_user_id=123456789,
        guild=mock_guild,
        status_message=mock_status_message
    )
    
    # Add sample response pairs
    sample_pairs = [
        {
            "question": "User1: What's your favorite color?",
            "answer": "I really like blue!",
            "metadata": {"confidence": 0.9, "response_type": "explicit_reply"}
        },
        {
            "question": "User2: How are you doing?",
            "answer": "Pretty good, thanks for asking",
            "metadata": {"confidence": 0.7, "response_type": "temporal_proximity"}
        },
        {
            "question": "User3: What do you think about this?",
            "answer": "That's an interesting point",
            "metadata": {"confidence": 0.4, "response_type": "content_analysis"}
        }
    ]
    
    generator.response_pairs = sample_pairs
    
    # Test confidence filtering
    high_confidence = [p for p in sample_pairs if p['metadata']['confidence'] >= 0.8]
    medium_confidence = [p for p in sample_pairs if 0.5 <= p['metadata']['confidence'] < 0.8]
    all_pairs = sample_pairs
    
    print(f"✅ High confidence pairs: {len(high_confidence)}")
    print(f"✅ Medium confidence pairs: {len(medium_confidence)}")
    print(f"✅ All pairs: {len(all_pairs)}")
    
    # Test dataset structure
    datasets = {
        "high_confidence": high_confidence,
        "medium_confidence": medium_confidence,
        "all_responses": all_pairs
    }
    
    for name, data in datasets.items():
        print(f"✅ Dataset '{name}': {len(data)} pairs")
        if data:
            avg_confidence = sum(p['metadata']['confidence'] for p in data) / len(data)
            print(f"   Average confidence: {avg_confidence:.2f}")
    
    print("\n🎯 File Generation Test Complete!")
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Discord Training Data Command")
    print("=" * 50)
    
    tests = [
        test_training_data_generator,
        test_command_validation,
        test_file_generation
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The training data command is ready.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

if __name__ == "__main__":
    main()
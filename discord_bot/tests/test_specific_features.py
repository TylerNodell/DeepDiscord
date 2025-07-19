#!/usr/bin/env python3
"""
Targeted tests for specific feature improvements
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import MessageTracker

def create_mock_message(id, author_id, content, created_at):
    """Create a mock Discord message"""
    msg = MagicMock()
    msg.id = id
    msg.author.id = author_id
    msg.content = content
    msg.created_at = created_at
    msg.reference = None
    msg.channel.id = 123456
    return msg

async def test_fragment_edge_cases():
    """Test fragment detection edge cases"""
    print("=== Testing Fragment Detection Edge Cases ===")
    
    tracker = MessageTracker()
    base_time = datetime.utcnow()
    
    # Test Case: Messages with punctuation should NOT combine
    print("\nTest: Punctuated messages should NOT combine")
    
    msg1 = create_mock_message(1, 100, "Hello everyone.", base_time)
    msg2 = create_mock_message(2, 100, "How are you today?", base_time + timedelta(seconds=2))
    
    await tracker.add_message(msg1)
    await tracker.add_message(msg2)
    
    # Wait for processing
    await asyncio.sleep(31)
    
    combined = tracker.get_combined_content(1)
    
    print(f"Message 1: '{msg1.content}'")
    print(f"Message 2: '{msg2.content}'")
    print(f"Combined: {combined}")
    print(f"Result: {'FAIL - Should not combine' if combined else 'PASS - Correctly not combined'}")
    
    # Test Case: Check fragment detection logic
    print(f"\nFragment Detection Analysis:")
    print(f"Message 1 ends with punctuation: {msg1.content.strip()[-1] in '.!?;'}")
    print(f"Should be fragment: {not (msg1.content.strip() and msg1.content.strip()[-1] in '.!?;')}")
    
    return combined is None

async def test_fragments_command():
    """Test the fragments command functionality"""
    print("\n=== Testing Fragments Command ===")
    
    tracker = MessageTracker()
    base_time = datetime.utcnow()
    
    # Create fragment messages
    fragments = [
        create_mock_message(10, 200, "This is part one", base_time),
        create_mock_message(11, 200, "and this is part two", base_time + timedelta(seconds=1))
    ]
    
    for msg in fragments:
        await tracker.add_message(msg)
    
    await asyncio.sleep(31)
    
    # Test getting combined content
    combined = tracker.get_combined_content(10)
    fragment_ref = tracker.get_combined_content(11)
    
    print(f"First message combined: {combined}")
    print(f"Second message reference: {fragment_ref}")
    
    return combined is not None and fragment_ref is not None

async def test_user_id_specific():
    """Test with the specific user ID provided"""
    print(f"\n=== Testing with User ID: 172384740224139266 ===")
    
    tracker = MessageTracker()
    base_time = datetime.utcnow()
    user_id = 172384740224139266
    
    # Create messages from this user
    messages = [
        create_mock_message(20, user_id, "Testing message 1", base_time),
        create_mock_message(21, user_id, "Testing message 2", base_time + timedelta(seconds=30)),
        create_mock_message(22, user_id, "Fragment without punctuation", base_time + timedelta(minutes=1)),
        create_mock_message(23, user_id, "continuing the thought", base_time + timedelta(minutes=1, seconds=2))
    ]
    
    for msg in messages:
        await tracker.add_message(msg)
    
    await asyncio.sleep(31)
    
    # Check results
    print(f"Total messages in cache: {len(tracker.message_cache)}")
    print(f"Messages from user {user_id}: {sum(1 for msg in tracker.message_cache.values() if msg.author.id == user_id)}")
    
    # Check for fragments
    fragment_combined = tracker.get_combined_content(22)
    print(f"Fragment combination for message 22: {fragment_combined}")
    
    return len(tracker.message_cache) >= 4

async def run_targeted_tests():
    """Run specific targeted tests"""
    print("DeepDiscord Targeted Feature Tests")
    print("=" * 50)
    
    results = {}
    
    # Test fragment edge cases
    results['punctuation_test'] = await test_fragment_edge_cases()
    
    # Test fragments command functionality
    results['fragments_command'] = await test_fragments_command()
    
    # Test with specific user ID
    results['user_id_test'] = await test_user_id_specific()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test}: {status}")
    
    # Save simple results
    with open("results/targeted_test_results.txt", "w") as f:
        f.write("DeepDiscord Targeted Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test Date: {datetime.now()}\n\n")
        for test, passed in results.items():
            status = "PASS" if passed else "FAIL"
            f.write(f"{test}: {status}\n")
    
    print(f"\nTargeted test results saved to results/targeted_test_results.txt")

if __name__ == "__main__":
    asyncio.run(run_targeted_tests())
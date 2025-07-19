#!/usr/bin/env python3
"""
Comprehensive test script for DeepDiscord features
Tests all bot commands and features with mock data
"""

import asyncio
import discord
from discord.ext import commands
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import MessageTracker, DeepDiscordBot, MessageCommands
from enhanced_message_tracker import EnhancedMessageTracker

# Test configuration
TEST_USER_ID = 172384740224139266
TEST_CHANNEL_ID = 123456789
TEST_GUILD_ID = 987654321
RESULTS_DIR = "results"

class TestResults:
    """Class to collect and save test results"""
    def __init__(self):
        self.features_tested = []
        self.results = {
            "test_run": datetime.now().isoformat(),
            "test_user_id": TEST_USER_ID,
            "features_tested": [],
            "test_results": {}
        }
    
    def add_result(self, feature_name, result):
        self.features_tested.append(feature_name)
        self.results["test_results"][feature_name] = result
    
    def save(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.results["features_tested"] = self.features_tested
        filename = f"{RESULTS_DIR}/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
        return filename

def create_mock_message(id, author_id, content, created_at, channel_id=TEST_CHANNEL_ID, 
                       reference_id=None, author_name=None):
    """Create a mock Discord message"""
    msg = MagicMock(spec=discord.Message)
    msg.id = id
    msg.content = content
    msg.created_at = created_at
    
    # Author
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.name = author_name or f"TestUser{author_id}"
    msg.author.mention = f"<@{author_id}>"
    msg.author.bot = False
    
    # Channel
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.name = "test-channel"
    msg.channel.mention = f"<#{channel_id}>"
    
    # Guild
    msg.guild = MagicMock()
    msg.guild.id = TEST_GUILD_ID
    msg.guild.name = "Test Guild"
    
    # Reference (for replies)
    if reference_id:
        msg.reference = MagicMock()
        msg.reference.message_id = reference_id
    else:
        msg.reference = None
    
    # URLs
    msg.jump_url = f"https://discord.com/channels/{TEST_GUILD_ID}/{channel_id}/{id}"
    
    return msg

async def test_message_tracking():
    """Test basic message tracking functionality"""
    print("\n=== Testing Message Tracking ===")
    results = {}
    
    tracker = MessageTracker()
    
    # Test adding messages
    base_time = datetime.utcnow()
    messages = []
    for i in range(5):
        msg = create_mock_message(
            id=1000 + i,
            author_id=TEST_USER_ID,
            content=f"Test message {i}",
            created_at=base_time + timedelta(seconds=i*10)
        )
        messages.append(msg)
        await tracker.add_message(msg)
    
    results["messages_added"] = len(messages)
    results["cache_size"] = len(tracker.message_cache)
    
    # Test retrieval
    retrieved = tracker.get_message(1002)
    results["message_retrieved"] = retrieved is not None
    results["retrieved_content"] = retrieved.content if retrieved else None
    
    # Test response chain (add a reply)
    reply_msg = create_mock_message(
        id=2000,
        author_id=TEST_USER_ID,
        content="This is a reply",
        created_at=base_time + timedelta(seconds=60),
        reference_id=1002
    )
    await tracker.add_message(reply_msg)
    
    results["response_chain"] = tracker.response_chain.get(1002, [])
    
    print(f"✓ Added {results['messages_added']} messages")
    print(f"✓ Cache size: {results['cache_size']}")
    print(f"✓ Response tracking: {len(results['response_chain'])} responses")
    
    return results

async def test_fragment_detection():
    """Test message fragment detection and combining"""
    print("\n=== Testing Fragment Detection ===")
    results = {}
    
    tracker = MessageTracker()
    base_time = datetime.utcnow()
    
    # Test Case 1: Rapid succession fragments
    print("Test Case 1: Rapid succession messages")
    fragments = [
        create_mock_message(3001, TEST_USER_ID, "Hey guys", base_time),
        create_mock_message(3002, TEST_USER_ID, "I wanted to ask something", base_time + timedelta(seconds=2)),
        create_mock_message(3003, TEST_USER_ID, "about the new feature", base_time + timedelta(seconds=4))
    ]
    
    for msg in fragments:
        await tracker.add_message(msg)
    
    # Wait for fragment processing
    await asyncio.sleep(31)  # Wait for fragment timeout
    
    combined = tracker.get_combined_content(3001)
    results["test_case_1"] = {
        "fragments_sent": len(fragments),
        "combined_content": combined,
        "fragment_detected": combined is not None
    }
    
    # Test Case 2: Messages with punctuation (should NOT combine)
    print("Test Case 2: Messages with proper punctuation")
    tracker.fragment_buffer.clear()
    
    punctuated = [
        create_mock_message(3004, TEST_USER_ID, "Hello everyone.", base_time + timedelta(minutes=2)),
        create_mock_message(3005, TEST_USER_ID, "How are you today?", base_time + timedelta(minutes=2, seconds=2))
    ]
    
    for msg in punctuated:
        await tracker.add_message(msg)
    
    await asyncio.sleep(31)
    
    combined2 = tracker.get_combined_content(3004)
    results["test_case_2"] = {
        "messages_sent": len(punctuated),
        "combined_content": combined2,
        "should_not_combine": combined2 is None
    }
    
    # Test Case 3: Continuation patterns
    print("Test Case 3: Continuation patterns")
    tracker.fragment_buffer.clear()
    
    continuation = [
        create_mock_message(3006, TEST_USER_ID, "I was thinking", base_time + timedelta(minutes=5)),
        create_mock_message(3007, TEST_USER_ID, "oh wait", base_time + timedelta(minutes=5, seconds=1)),
        create_mock_message(3008, TEST_USER_ID, "actually never mind", base_time + timedelta(minutes=5, seconds=3))
    ]
    
    for msg in continuation:
        await tracker.add_message(msg)
    
    await asyncio.sleep(31)
    
    combined3 = tracker.get_combined_content(3006)
    results["test_case_3"] = {
        "messages_sent": len(continuation),
        "combined_content": combined3,
        "continuation_detected": combined3 is not None
    }
    
    print(f"✓ Test Case 1: {'PASS' if results['test_case_1']['fragment_detected'] else 'FAIL'}")
    print(f"✓ Test Case 2: {'PASS' if results['test_case_2']['should_not_combine'] else 'FAIL'}")
    print(f"✓ Test Case 3: {'PASS' if results['test_case_3']['continuation_detected'] else 'FAIL'}")
    
    return results

async def test_enhanced_message_tracking():
    """Test enhanced message relationship detection"""
    print("\n=== Testing Enhanced Message Tracking ===")
    results = {}
    
    tracker = EnhancedMessageTracker()
    base_time = datetime.utcnow()
    
    # Create a conversation
    messages = [
        create_mock_message(4001, TEST_USER_ID, "What's everyone's favorite game?", base_time),
        create_mock_message(4002, 111111, "I really like Minecraft", base_time + timedelta(seconds=30)),
        create_mock_message(4003, 222222, "Yeah Minecraft is great", base_time + timedelta(seconds=45)),
        create_mock_message(4004, TEST_USER_ID, "I agree, it's very creative", base_time + timedelta(seconds=60)),
        create_mock_message(4005, 333333, "Anyone play Valorant?", base_time + timedelta(minutes=15))
    ]
    
    # Add messages to tracker
    for msg in messages:
        tracker.add_message(msg)
    
    # Test relationship detection
    relationships = tracker.get_message_relationships(4003)
    results["message_relationships"] = relationships
    
    # Test standalone message detection
    standalone = tracker.find_standalone_messages()
    results["standalone_messages"] = [msg.id for msg in standalone]
    
    # Test conversation flow
    flow = tracker.get_conversation_flow(TEST_CHANNEL_ID, time_window=3600)
    results["conversation_segments"] = len(flow)
    
    # Test statistics
    stats = tracker.get_statistics()
    results["tracker_statistics"] = stats
    
    print(f"✓ Tracked {len(messages)} messages")
    print(f"✓ Found {len(results['standalone_messages'])} standalone messages")
    print(f"✓ Identified {results['conversation_segments']} conversation segments")
    print(f"✓ Statistics: {stats['total_messages']} total, {stats['implicit_responses']} implicit responses")
    
    return results

async def test_bot_commands():
    """Test bot command functionality with mocked context"""
    print("\n=== Testing Bot Commands ===")
    results = {}
    
    # Create mock bot and cog
    bot = MagicMock(spec=DeepDiscordBot)
    bot.message_tracker = MessageTracker()
    
    # Add test messages
    base_time = datetime.utcnow()
    test_msg = create_mock_message(5001, TEST_USER_ID, "Test message for commands", base_time)
    await bot.message_tracker.add_message(test_msg)
    
    # Create mock context
    ctx = MagicMock(spec=commands.Context)
    ctx.channel = MagicMock()
    ctx.channel.id = TEST_CHANNEL_ID
    ctx.author = MagicMock()
    ctx.author.id = TEST_USER_ID
    ctx.author.name = "TestUser"
    ctx.send = AsyncMock()
    
    # Initialize commands cog
    message_commands = MessageCommands(bot)
    
    # Test getmsg command
    print("Testing !getmsg command...")
    await message_commands.get_message(ctx, 5001)
    results["getmsg_called"] = ctx.send.called
    results["getmsg_response"] = str(ctx.send.call_args) if ctx.send.called else None
    
    # Test stats command
    print("Testing !stats command...")
    ctx.send.reset_mock()
    await message_commands.get_stats(ctx)
    results["stats_called"] = ctx.send.called
    
    print(f"✓ !getmsg command: {'PASS' if results['getmsg_called'] else 'FAIL'}")
    print(f"✓ !stats command: {'PASS' if results['stats_called'] else 'FAIL'}")
    
    return results

async def test_user_history():
    """Test user history tracking"""
    print("\n=== Testing User History ===")
    results = {}
    
    tracker = MessageTracker()
    base_time = datetime.utcnow()
    
    # Create user history data
    user_messages = []
    for i in range(10):
        msg = create_mock_message(
            id=6000 + i,
            author_id=TEST_USER_ID,
            content=f"User message {i}",
            created_at=base_time + timedelta(minutes=i)
        )
        user_messages.append({
            'content': msg.content,
            'timestamp': msg.created_at.isoformat(),
            'message_id': msg.id,
            'channel': 'test-channel',
            'guild': 'Test Guild'
        })
    
    # Store user history
    user_data = {
        'user_id': TEST_USER_ID,
        'username': 'TestUser',
        'total_messages': len(user_messages),
        'messages': user_messages
    }
    
    tracker.store_user_history(TEST_USER_ID, user_data)
    
    # Retrieve user history
    retrieved = tracker.get_user_history(TEST_USER_ID)
    
    results["history_stored"] = retrieved is not None
    results["message_count"] = retrieved['total_messages'] if retrieved else 0
    results["messages_match"] = len(retrieved['messages']) == len(user_messages) if retrieved else False
    
    print(f"✓ Stored history for user {TEST_USER_ID}")
    print(f"✓ Message count: {results['message_count']}")
    print(f"✓ History retrieval: {'PASS' if results['history_stored'] else 'FAIL'}")
    
    return results

async def run_all_tests():
    """Run all tests and save results"""
    print("=" * 50)
    print("DeepDiscord Feature Test Suite")
    print(f"Testing with User ID: {TEST_USER_ID}")
    print("=" * 50)
    
    test_results = TestResults()
    
    try:
        # Run each test
        result1 = await test_message_tracking()
        test_results.add_result("message_tracking", result1)
        
        result2 = await test_fragment_detection()
        test_results.add_result("fragment_detection", result2)
        
        result3 = await test_enhanced_message_tracking()
        test_results.add_result("enhanced_tracking", result3)
        
        result4 = await test_bot_commands()
        test_results.add_result("bot_commands", result4)
        
        result5 = await test_user_history()
        test_results.add_result("user_history", result5)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        test_results.add_result("error", str(e))
    
    # Save results
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Features tested: {len(test_results.features_tested)}")
    print(f"Features: {', '.join(test_results.features_tested)}")
    
    filename = test_results.save()
    
    # Also create a summary file
    summary_file = f"{RESULTS_DIR}/test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DeepDiscord Feature Test Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"User ID: {TEST_USER_ID}\n\n")
        
        for feature, result in test_results.results["test_results"].items():
            f.write(f"\n{feature.upper()}:\n")
            f.write("-" * 30 + "\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n")
    
    print(f"Summary saved to {summary_file}")
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
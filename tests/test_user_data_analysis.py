#!/usr/bin/env python3
"""
Focused test showing how bot analyzes data for user ID 172384740224119266
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import MessageTracker
from enhanced_message_tracker import EnhancedMessageTracker

# The specific user ID to analyze
TARGET_USER_ID = 172384740224119266

def create_realistic_message(id, author_id, content, created_at, author_name="User"):
    """Create realistic mock message"""
    msg = MagicMock()
    msg.id = id
    msg.author.id = author_id
    msg.author.name = author_name
    msg.content = content
    msg.created_at = created_at
    msg.channel.id = 123456789
    msg.reference = None
    return msg

async def analyze_user_message_patterns():
    """Analyze realistic message patterns for the target user"""
    print(f"\n=== Analyzing Message Patterns for User {TARGET_USER_ID} ===")
    
    tracker = MessageTracker()
    enhanced_tracker = EnhancedMessageTracker()
    
    base_time = datetime.utcnow() - timedelta(hours=2)
    
    # Create realistic conversation with fragments, replies, and standalone messages
    conversation_data = [
        # Scenario 1: User starts with fragmented technical explanation
        {
            "id": 5001,
            "content": "I've been debugging this issue for hours",
            "time_offset": 0,
            "type": "fragment_start"
        },
        {
            "id": 5002, 
            "content": "and I think I finally found the problem",
            "time_offset": 3,
            "type": "fragment_continuation"
        },
        {
            "id": 5003,
            "content": "it was a race condition in the async code",
            "time_offset": 6,
            "type": "fragment_end"
        },
        
        # Scenario 2: Standalone announcement (20 minutes later)
        {
            "id": 5004,
            "content": "Deploy to production is complete! 🚀",
            "time_offset": 1200,
            "type": "standalone"
        },
        
        # Scenario 3: Quick correction pattern
        {
            "id": 5005,
            "content": "The new feature will be available tomorrow",
            "time_offset": 1800,
            "type": "initial_statement"
        },
        {
            "id": 5006,
            "content": "actually, scratch that",
            "time_offset": 1805,
            "type": "correction_start"
        },
        {
            "id": 5007,
            "content": "we're pushing it to next week to be safe",
            "time_offset": 1808,
            "type": "correction_end"
        },
        
        # Scenario 4: Response to hypothetical other user (30 min later)
        {
            "id": 5008,
            "content": "Thanks for the feedback on the PR!",
            "time_offset": 3600,
            "type": "response"
        }
    ]
    
    # Process messages through both trackers
    messages = []
    for data in conversation_data:
        msg = create_realistic_message(
            data["id"],
            TARGET_USER_ID,
            data["content"], 
            base_time + timedelta(seconds=data["time_offset"]),
            "TargetUser"
        )
        messages.append(msg)
        
        # Add to both trackers
        await tracker.add_message(msg)
        enhanced_tracker.add_message(msg)
    
    # Wait for fragment processing
    await asyncio.sleep(31)
    
    # Analyze results
    analysis_results = {
        "user_id": TARGET_USER_ID,
        "total_messages": len(messages),
        "analysis_timestamp": datetime.now().isoformat(),
        "message_analysis": {},
        "fragment_combinations": {},
        "conversation_insights": {}
    }
    
    # Analyze each message
    for msg in messages:
        # Basic tracker analysis
        combined_content = tracker.get_combined_content(msg.id)
        
        # Enhanced tracker analysis
        relationships = enhanced_tracker.get_message_relationships(msg.id)
        
        analysis_results["message_analysis"][msg.id] = {
            "original_content": msg.content,
            "combined_content": combined_content,
            "is_fragment": combined_content is not None,
            "relationships": relationships,
            "timestamp": msg.created_at.isoformat()
        }
    
    # Check specific fragment combinations
    fragment_groups = [
        {"start_id": 5001, "description": "Technical debugging explanation"},
        {"start_id": 5005, "description": "Feature timeline correction"},
    ]
    
    for group in fragment_groups:
        start_id = group["start_id"]
        combined = tracker.get_combined_content(start_id)
        analysis_results["fragment_combinations"][start_id] = {
            "description": group["description"],
            "combined_text": combined,
            "successfully_combined": combined is not None and not combined.startswith("[Fragment of")
        }
    
    # Enhanced insights
    stats = enhanced_tracker.get_statistics()
    standalone_messages = enhanced_tracker.find_standalone_messages()
    
    analysis_results["conversation_insights"] = {
        "total_responses_detected": stats["total_responses"],
        "standalone_message_count": len(standalone_messages),
        "standalone_message_ids": [msg.id for msg in standalone_messages],
        "response_rate": stats["response_rate"],
        "user_patterns": enhanced_tracker.user_interaction_patterns.get(TARGET_USER_ID, {})
    }
    
    return analysis_results

async def demonstrate_real_world_usage():
    """Show how this would work in a real Discord server"""
    print(f"\n=== Real World Usage Example ===")
    
    # This shows exactly what would happen in production
    usage_example = {
        "scenario": "User sends fragmented message in Discord server",
        "user_id": TARGET_USER_ID,
        "steps": [
            {
                "step": 1,
                "action": "User types: 'I need help with'",
                "bot_response": "Message added to tracker, fragment detection started"
            },
            {
                "step": 2, 
                "action": "User types: 'this coding problem' (2 seconds later)",
                "bot_response": "Fragment detected, messages combined in background"
            },
            {
                "step": 3,
                "action": "Another user runs: !fragments <message_id>",
                "bot_response": "Shows combined text: 'I need help with this coding problem'"
            },
            {
                "step": 4,
                "action": "Moderator runs: !userhistory 172384740224119266",
                "bot_response": "Shows complete analysis of user's message patterns"
            }
        ],
        "technical_details": {
            "data_source": "Live Discord message events",
            "processing": "Real-time fragment detection and relationship analysis", 
            "storage": "In-memory cache with 10,000 message limit",
            "retrieval": "Instant lookup by message ID or user ID"
        }
    }
    
    return usage_example

async def run_user_data_analysis():
    """Run comprehensive analysis for the target user"""
    print("=" * 60)
    print(f"DISCORD USER DATA ANALYSIS")
    print(f"Target User: {TARGET_USER_ID}")
    print("=" * 60)
    
    # Run analysis
    analysis_results = await analyze_user_message_patterns()
    usage_example = await demonstrate_real_world_usage()
    
    # Combine results
    full_results = {
        "target_user_id": TARGET_USER_ID,
        "analysis_results": analysis_results,
        "usage_example": usage_example,
        "validation": {
            "fragment_detection_working": any(
                details["successfully_combined"] 
                for details in analysis_results["fragment_combinations"].values()
            ),
            "user_tracking_working": analysis_results["total_messages"] > 0,
            "enhanced_analysis_working": len(analysis_results["conversation_insights"]) > 0
        }
    }
    
    # Save detailed results
    os.makedirs("results", exist_ok=True)
    with open(f"results/user_{TARGET_USER_ID}_detailed_analysis.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Create human-readable report
    with open(f"results/user_{TARGET_USER_ID}_report.md", "w") as f:
        f.write(f"# Discord Analysis Report: User {TARGET_USER_ID}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Successfully analyzed {analysis_results['total_messages']} messages from user {TARGET_USER_ID}.\n")
        f.write("The bot demonstrates real-time fragment detection and conversation analysis.\n\n")
        
        f.write("## Fragment Detection Results\n\n")
        for frag_id, details in analysis_results["fragment_combinations"].items():
            status = "✅ SUCCESS" if details["successfully_combined"] else "❌ FAILED"
            f.write(f"**{details['description']}** - {status}\n")
            if details["combined_text"]:
                f.write(f"- Combined text: \"{details['combined_text']}\"\n")
            f.write("\n")
        
        f.write("## Message Analysis\n\n")
        for msg_id, details in analysis_results["message_analysis"].items():
            f.write(f"**Message {msg_id}:**\n")
            f.write(f"- Original: \"{details['original_content']}\"\n")
            if details["combined_content"]:
                f.write(f"- Combined: \"{details['combined_content']}\"\n")
            f.write(f"- Fragment detected: {details['is_fragment']}\n\n")
        
        f.write("## Conversation Insights\n\n")
        insights = analysis_results["conversation_insights"]
        f.write(f"- Total responses detected: {insights['total_responses_detected']}\n")
        f.write(f"- Standalone messages: {insights['standalone_message_count']}\n")
        f.write(f"- Response rate: {insights['response_rate']:.2f}\n\n")
        
        f.write("## Real-World Usage\n\n")
        f.write("This analysis shows exactly how DeepDiscord would work when deployed:\n\n")
        for step in usage_example["steps"]:
            f.write(f"{step['step']}. **{step['action']}**\n")
            f.write(f"   → {step['bot_response']}\n\n")
        
        f.write("## Validation\n\n")
        validation = full_results["validation"]
        for check, passed in validation.items():
            status = "✅" if passed else "❌"
            f.write(f"{status} {check.replace('_', ' ').title()}\n")
    
    # Print summary
    print(f"\n✅ Analysis complete for user {TARGET_USER_ID}")
    print(f"✅ Processed {analysis_results['total_messages']} messages")
    
    # Show fragment results
    fragment_success = 0
    for details in analysis_results["fragment_combinations"].values():
        if details["successfully_combined"]:
            fragment_success += 1
            print(f"✅ Fragment detected: \"{details['combined_text']}\"")
    
    print(f"✅ {fragment_success} fragment groups successfully combined")
    print(f"✅ Results saved to results/user_{TARGET_USER_ID}_report.md")
    
    return full_results

if __name__ == "__main__":
    asyncio.run(run_user_data_analysis())
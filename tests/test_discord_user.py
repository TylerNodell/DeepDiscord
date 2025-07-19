#!/usr/bin/env python3
"""
Targeted test for the specific user ID 172384740224139266 that was found in the server
"""

import asyncio
import discord
from discord.ext import commands
import sys
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import MessageTracker

# Load environment variables
load_dotenv()

# Get test user ID from environment
TEST_USER_ID = int(os.getenv('TEST_USER_ID', 172384740224139266))

class TargetedUserTest:
    """Test for Discord user specified in .env TEST_USER_ID"""
    
    def __init__(self):
        self.bot_token = os.getenv('DISCORD_TOKEN')
        self.test_results = {
            "test_started": datetime.now().isoformat(),
            "target_user_id": TEST_USER_ID,
            "user_found": False,
            "messages_found": 0,
            "user_details": {},
            "all_messages": [],
            "fragment_analysis": {},
            "channels_with_messages": []
        }
    
    async def run_targeted_test(self):
        """Run test specifically for the target user"""
        print("🎯 TARGETED USER TEST")
        print("=" * 50)
        print(f"Target User ID: {TEST_USER_ID}")
        print("Searching specifically for this user's messages...")
        print("=" * 50)
        
        if not self.bot_token:
            print("❌ No Discord token found")
            return False
        
        try:
            # Create bot
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.members = True
            
            bot = commands.Bot(command_prefix='!', intents=intents)
            bot.message_tracker = MessageTracker()
            
            @bot.event
            async def on_ready():
                print(f"✅ Connected as {bot.user}")
                
                # Search specifically for our target user
                await self.search_for_target_user(bot)
                
                # Close bot
                await bot.close()
            
            await bot.start(self.bot_token)
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
        
        return True
    
    async def search_for_target_user(self, bot):
        """Search specifically for the target user's messages"""
        print(f"\n🔍 Searching for user {TEST_USER_ID}...")
        
        for guild in bot.guilds:
            print(f"\n📊 Searching guild: {guild.name}")
            
            # Find the target user in this guild
            target_member = guild.get_member(TEST_USER_ID)
            if target_member:
                print(f"🎯 TARGET USER FOUND!")
                print(f"   Name: {target_member.name}")
                print(f"   Display Name: {target_member.display_name}")
                print(f"   ID: {target_member.id}")
                
                self.test_results["user_found"] = True
                self.test_results["user_details"] = {
                    "name": target_member.name,
                    "display_name": target_member.display_name,
                    "id": target_member.id,
                    "guild_name": guild.name,
                    "joined_at": target_member.joined_at.isoformat() if target_member.joined_at else None
                }
                
                # Search for their messages
                await self.find_user_messages(bot, guild, target_member)
                
            else:
                print(f"❌ Target user not found in {guild.name}")
    
    async def find_user_messages(self, bot, guild, target_member):
        """Find all messages from the target user"""
        print(f"\n📝 Searching for messages from {target_member.display_name}...")
        
        cutoff_date = datetime.utcnow() - timedelta(days=30)  # Last 30 days
        all_user_messages = []
        
        for channel in guild.text_channels:
            try:
                print(f"  🔍 Checking #{channel.name}...")
                
                channel_messages = []
                async for message in channel.history(limit=1000, after=cutoff_date):
                    if message.author.id == TEST_USER_ID:
                        # Found a message from our target user!
                        message_data = {
                            "message_id": message.id,
                            "content": message.content,
                            "timestamp": message.created_at.isoformat(),
                            "channel_name": channel.name,
                            "channel_id": channel.id,
                            "has_reference": message.reference is not None,
                            "reference_id": message.reference.message_id if message.reference else None,
                            "attachments": len(message.attachments),
                            "embeds": len(message.embeds),
                            "jump_url": message.jump_url
                        }
                        
                        channel_messages.append(message_data)
                        all_user_messages.append(message_data)
                        
                        # Add to bot tracker for fragment analysis
                        await bot.message_tracker.add_message(message)
                
                if channel_messages:
                    print(f"    ✅ Found {len(channel_messages)} messages")
                    self.test_results["channels_with_messages"].append({
                        "channel_name": channel.name,
                        "message_count": len(channel_messages),
                        "messages": channel_messages
                    })
                
            except discord.Forbidden:
                print(f"    ❌ No access to #{channel.name}")
            except Exception as e:
                print(f"    ⚠️  Error in #{channel.name}: {e}")
        
        self.test_results["messages_found"] = len(all_user_messages)
        self.test_results["all_messages"] = all_user_messages
        
        print(f"\n📈 Total messages found: {len(all_user_messages)}")
        
        if all_user_messages:
            # Analyze fragments
            await self.analyze_user_fragments(bot)
            
            # Show sample messages
            print(f"\n📋 Sample messages from user {TEST_USER_ID}:")
            for i, msg in enumerate(all_user_messages[:5]):
                print(f"  {i+1}. [{msg['timestamp']}] #{msg['channel_name']}: {msg['content'][:100]}...")
        else:
            print(f"⚠️  No messages found for user {TEST_USER_ID}")
    
    async def analyze_user_fragments(self, bot):
        """Analyze fragments specifically for our target user"""
        print(f"\n🧩 Analyzing fragments for user {TEST_USER_ID}...")
        
        # Wait for fragment processing
        await asyncio.sleep(31)
        
        fragments = []
        fragment_groups = {}
        
        for message_data in self.test_results["all_messages"]:
            message_id = message_data["message_id"]
            combined_content = bot.message_tracker.get_combined_content(message_id)
            
            if combined_content:
                fragment_info = {
                    "message_id": message_id,
                    "original_content": message_data["content"],
                    "combined_content": combined_content,
                    "timestamp": message_data["timestamp"],
                    "channel": message_data["channel_name"],
                    "is_fragment_start": not combined_content.startswith("[Fragment of"),
                    "jump_url": message_data["jump_url"]
                }
                
                fragments.append(fragment_info)
                
                if fragment_info["is_fragment_start"]:
                    fragment_groups[message_id] = fragment_info
        
        self.test_results["fragment_analysis"] = {
            "total_fragments": len(fragments),
            "fragment_groups": len(fragment_groups),
            "fragments": fragments,
            "fragment_groups_detail": fragment_groups
        }
        
        print(f"✅ Fragment analysis complete:")
        print(f"  • Total fragments: {len(fragments)}")
        print(f"  • Fragment groups: {len(fragment_groups)}")
        
        # Show fragment results
        for group_id, group_info in fragment_groups.items():
            print(f"\n📝 Fragment Group {group_id} (#{group_info['channel']}):")
            print(f"    Combined: \"{group_info['combined_content'][:150]}...\"")
            print(f"    Jump URL: {group_info['jump_url']}")
    
    def save_targeted_results(self):
        """Save results for the targeted user test"""
        os.makedirs("results", exist_ok=True)
        
        # Save detailed JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = f"results/targeted_user_{TEST_USER_ID}_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Create targeted report
        report_file = f"results/targeted_user_{TEST_USER_ID}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Targeted User Analysis Report\n\n")
            f.write(f"**User ID:** {TEST_USER_ID}\n")
            f.write(f"**Test Date:** {self.test_results['test_started']}\n")
            f.write(f"**User Found:** {'✅ YES' if self.test_results['user_found'] else '❌ NO'}\n\n")
            
            if self.test_results["user_found"]:
                user_details = self.test_results["user_details"]
                f.write(f"## User Details\n\n")
                f.write(f"- **Name:** {user_details['name']}\n")
                f.write(f"- **Display Name:** {user_details['display_name']}\n")
                f.write(f"- **Guild:** {user_details['guild_name']}\n")
                f.write(f"- **Messages Found:** {self.test_results['messages_found']}\n\n")
                
                f.write(f"## Message Distribution\n\n")
                for channel_info in self.test_results["channels_with_messages"]:
                    f.write(f"- **#{channel_info['channel_name']}:** {channel_info['message_count']} messages\n")
                
                if self.test_results["fragment_analysis"]["fragments"]:
                    f.write(f"\n## Fragment Analysis Results\n\n")
                    fragment_analysis = self.test_results["fragment_analysis"]
                    f.write(f"- **Total Fragments:** {fragment_analysis['total_fragments']}\n")
                    f.write(f"- **Fragment Groups:** {fragment_analysis['fragment_groups']}\n\n")
                    
                    for group in fragment_analysis["fragment_groups_detail"].values():
                        f.write(f"### Fragment: Message {group['message_id']}\n")
                        f.write(f"**Channel:** #{group['channel']}\n")
                        f.write(f"**Combined Content:** {group['combined_content']}\n")
                        f.write(f"**Jump URL:** [View Message]({group['jump_url']})\n\n")
                
                f.write(f"\n## All Messages\n\n")
                for i, msg in enumerate(self.test_results["all_messages"][:10]):
                    f.write(f"**Message {i+1}** ({msg['timestamp']}):\n")
                    f.write(f"- Channel: #{msg['channel_name']}\n")
                    f.write(f"- Content: {msg['content']}\n")
                    f.write(f"- [Jump to Message]({msg['jump_url']})\n\n")
                
                if len(self.test_results["all_messages"]) > 10:
                    f.write(f"*...and {len(self.test_results['all_messages']) - 10} more messages*\n")
        
        print(f"\n💾 Targeted results saved:")
        print(f"  • Data: {json_file}")
        print(f"  • Report: {report_file}")
        
        return json_file, report_file

async def run_targeted_user_test():
    """Run the targeted test for the user specified in .env TEST_USER_ID"""
    test = TargetedUserTest()
    
    success = await test.run_targeted_test()
    
    if success:
        test.save_targeted_results()
        
        print(f"\n🎉 TARGETED TEST COMPLETE!")
        
        if test.test_results["user_found"]:
            print(f"🎯 USER {TEST_USER_ID} SUCCESSFULLY ANALYZED!")
            print(f"✅ Found {test.test_results['messages_found']} messages")
            print(f"✅ Detected {test.test_results['fragment_analysis']['fragment_groups']} fragment groups")
        else:
            print(f"❌ User {TEST_USER_ID} not found")
    
    return test.test_results

if __name__ == "__main__":
    asyncio.run(run_targeted_user_test())
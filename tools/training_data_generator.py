#!/usr/bin/env python3
"""
Training Data Generator for DeepDiscord

Extracts question/answer pairs from Discord message relationships for training.
"""

import asyncio
import discord
from discord.ext import commands
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discord_bot.discord_bot import MessageTracker

load_dotenv()

class TrainingDataGenerator:
    """Generate training data from Discord message relationships"""
    
    def __init__(self):
        self.bot_token = os.getenv('DISCORD_TOKEN')
        self.target_user_id = int(os.getenv('TEST_USER_ID', 172384740224139266))
        self.training_data = []
        self.response_pairs = []
    
    async def generate_training_data(self, days_back=30, min_response_length=10):
        """Generate training data from Discord conversations"""
        print("🎓 TRAINING DATA GENERATOR")
        print("=" * 50)
        print(f"Target User: {self.target_user_id}")
        print(f"Looking back: {days_back} days")
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
                
                # Extract training data
                await self.extract_response_pairs(bot, days_back, min_response_length)
                
                # Close bot
                await bot.close()
            
            await bot.start(self.bot_token)
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
        
        return True
    
    async def extract_response_pairs(self, bot, days_back, min_response_length):
        """Extract question/answer pairs from Discord conversations"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        for guild in bot.guilds:
            print(f"\n📊 Processing guild: {guild.name}")
            
            # Check if target user is in this guild
            target_member = guild.get_member(self.target_user_id)
            if not target_member:
                print(f"❌ Target user not found in {guild.name}")
                continue
            
            print(f"🎯 Target user found: {target_member.display_name}")
            
            # Process each channel
            for channel in guild.text_channels:
                try:
                    print(f"  📝 Processing #{channel.name}...")
                    await self.process_channel_messages(bot, channel, cutoff_date, min_response_length)
                    
                except discord.Forbidden:
                    print(f"    ❌ No access to #{channel.name}")
                except Exception as e:
                    print(f"    ⚠️  Error in #{channel.name}: {e}")
        
        # Generate final training data
        self.generate_training_json()
    
    async def process_channel_messages(self, bot, channel, cutoff_date, min_response_length):
        """Process messages in a channel to find response pairs"""
        messages = []
        target_responses = 0
        
        # Collect messages
        async for message in channel.history(limit=1000, after=cutoff_date):
            if not message.author.bot:  # Skip bot messages
                messages.append(message)
                
                # Add to tracker for fragment analysis
                await bot.message_tracker.add_message(message)
        
        # Sort messages chronologically (oldest first)
        messages.sort(key=lambda m: m.created_at)
        
        # Find response pairs
        for i, message in enumerate(messages):
            if message.author.id == self.target_user_id:
                # This is a message from our target user
                target_responses += 1
                
                # Look for what they're responding to
                response_to = await self.find_response_target(bot, message, messages[:i])
                
                if response_to and len(message.content.strip()) >= min_response_length:
                    # Create training pair
                    training_pair = self.create_training_pair(response_to, message)
                    if training_pair:
                        self.response_pairs.append(training_pair)
        
        if target_responses > 0:
            print(f"    ✅ Found {target_responses} target responses, {len([p for p in self.response_pairs if p.get('channel') == channel.name])} training pairs")
    
    async def find_response_target(self, bot, target_message, previous_messages):
        """Find what message the target user is responding to"""
        
        # Method 1: Explicit Discord reply
        if target_message.reference:
            try:
                referenced_msg = await target_message.channel.fetch_message(target_message.reference.message_id)
                if referenced_msg and not referenced_msg.author.bot:
                    return {
                        'type': 'explicit_reply',
                        'message': referenced_msg,
                        'confidence': 1.0
                    }
            except:
                pass
        
        # Method 2: Temporal proximity (recent message)
        recent_messages = [msg for msg in previous_messages[-10:] 
                          if msg.author.id != self.target_user_id 
                          and not msg.author.bot
                          and (target_message.created_at - msg.created_at).total_seconds() <= 300]  # 5 minutes
        
        if recent_messages:
            # Get the most recent non-target message
            most_recent = recent_messages[-1]
            time_gap = (target_message.created_at - most_recent.created_at).total_seconds()
            
            # Higher confidence for shorter time gaps
            confidence = max(0.3, 1.0 - (time_gap / 300))
            
            return {
                'type': 'temporal_proximity',
                'message': most_recent,
                'confidence': confidence,
                'time_gap_seconds': time_gap
            }
        
        # Method 3: Content analysis (mentions, keywords)
        for msg in reversed(previous_messages[-20:]):  # Look at last 20 messages
            if msg.author.id != self.target_user_id and not msg.author.bot:
                if self.has_content_relationship(msg, target_message):
                    return {
                        'type': 'content_analysis',
                        'message': msg,
                        'confidence': 0.6
                    }
        
        return None
    
    def has_content_relationship(self, potential_question, response):
        """Check if response content relates to potential question"""
        question_content = potential_question.content.lower()
        response_content = response.content.lower()
        
        # Look for shared keywords (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'that', 'this', 'what', 'when', 'where', 'why', 'how', 'who'}
        
        question_words = set(question_content.split()) - common_words
        response_words = set(response_content.split()) - common_words
        
        if len(question_words) > 0 and len(response_words) > 0:
            shared_words = question_words.intersection(response_words)
            if len(shared_words) >= 2:  # At least 2 shared meaningful words
                return True
        
        # Check if response mentions the questioner
        if potential_question.author.mention in response.content:
            return True
        
        return False
    
    def create_training_pair(self, response_info, target_response):
        """Create a formatted training pair"""
        question_msg = response_info['message']
        
        # Format the question (include author context)
        question = f"{question_msg.author.display_name}: {question_msg.content}"
        
        # Format the answer (target user's response)
        answer = target_response.content
        
        # Get combined content if this is a fragment
        combined_content = target_response.guild.get_channel(target_response.channel.id)
        if hasattr(target_response, 'id'):
            combined = getattr(target_response, 'message_tracker', None)
            if combined:
                fragment_content = combined.get_combined_content(target_response.id)
                if fragment_content and not fragment_content.startswith("[Fragment of"):
                    answer = fragment_content
        
        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "metadata": {
                "response_type": response_info['type'],
                "confidence": response_info['confidence'],
                "question_author": question_msg.author.display_name,
                "answer_author": target_response.author.display_name,
                "channel": target_response.channel.name,
                "timestamp": target_response.created_at.isoformat(),
                "question_id": question_msg.id,
                "answer_id": target_response.id,
                "time_gap": response_info.get('time_gap_seconds', 0)
            }
        }
    
    def generate_training_json(self):
        """Generate final training data files"""
        os.makedirs("training_data", exist_ok=True)
        
        # Filter and sort by confidence
        high_confidence = [pair for pair in self.response_pairs if pair['metadata']['confidence'] >= 0.8]
        medium_confidence = [pair for pair in self.response_pairs if 0.5 <= pair['metadata']['confidence'] < 0.8]
        all_pairs = sorted(self.response_pairs, key=lambda x: x['metadata']['confidence'], reverse=True)
        
        # Save different quality levels
        datasets = {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "all_responses": all_pairs
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, data in datasets.items():
            filename = f"training_data/{name}_{self.target_user_id}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "target_user_id": self.target_user_id,
                        "total_pairs": len(data),
                        "confidence_threshold": name,
                        "format": "question/answer pairs for training"
                    },
                    "training_data": data
                }, f, indent=2)
            
            print(f"💾 Saved {len(data)} pairs to {filename}")
        
        # Generate statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print training data statistics"""
        print(f"\n📈 TRAINING DATA STATISTICS")
        print("=" * 40)
        print(f"Total response pairs: {len(self.response_pairs)}")
        
        # By response type
        type_counts = {}
        confidence_sum = 0
        
        for pair in self.response_pairs:
            resp_type = pair['metadata']['response_type']
            type_counts[resp_type] = type_counts.get(resp_type, 0) + 1
            confidence_sum += pair['metadata']['confidence']
        
        print(f"\nBy response detection method:")
        for resp_type, count in type_counts.items():
            print(f"  • {resp_type}: {count}")
        
        if self.response_pairs:
            avg_confidence = confidence_sum / len(self.response_pairs)
            print(f"\nAverage confidence: {avg_confidence:.2f}")
            
            high_conf = len([p for p in self.response_pairs if p['metadata']['confidence'] >= 0.8])
            print(f"High confidence (≥0.8): {high_conf}")
            
            # Sample pairs
            print(f"\n📋 Sample Training Pairs:")
            for i, pair in enumerate(self.response_pairs[:3]):
                print(f"\n{i+1}. [{pair['metadata']['response_type']}] Confidence: {pair['metadata']['confidence']:.2f}")
                print(f"   Q: {pair['question'][:100]}...")
                print(f"   A: {pair['answer'][:100]}...")

async def run_training_data_generation():
    """Run the training data generator"""
    generator = TrainingDataGenerator()
    
    success = await generator.generate_training_data(
        days_back=30,
        min_response_length=10
    )
    
    if success:
        print(f"\n🎉 TRAINING DATA GENERATION COMPLETE!")
        print(f"✅ Generated {len(generator.response_pairs)} training pairs")
        print(f"📁 Files saved to training_data/ directory")
    
    return generator.response_pairs

if __name__ == "__main__":
    asyncio.run(run_training_data_generation())
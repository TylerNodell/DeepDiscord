#!/usr/bin/env python3
"""
Discord Bot for Message Retrieval and Response Tracking
Part of DeepDiscord Project
"""

import discord
from discord.ext import commands
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MessageTracker:
    """Tracks message relationships and responses"""
    
    def __init__(self):
        self.message_cache: Dict[int, discord.Message] = {}
        self.response_chain: Dict[int, List[int]] = {}  # message_id -> list of response_ids
        self.max_cache_size = 10000
    
    def add_message(self, message: discord.Message):
        """Add a message to the cache"""
        if len(self.message_cache) >= self.max_cache_size:
            # Remove oldest message (simple FIFO)
            oldest_id = next(iter(self.message_cache))
            del self.message_cache[oldest_id]
        
        self.message_cache[message.id] = message
        
        # Track response relationships
        if message.reference and message.reference.message_id:
            referenced_id = message.reference.message_id
            if referenced_id not in self.response_chain:
                self.response_chain[referenced_id] = []
            self.response_chain[referenced_id].append(message.id)
    
    def get_message(self, message_id: int) -> Optional[discord.Message]:
        """Get a message from cache"""
        return self.message_cache.get(message_id)
    
    def get_responses_to(self, message_id: int) -> List[discord.Message]:
        """Get all responses to a specific message"""
        response_ids = self.response_chain.get(message_id, [])
        return [self.message_cache.get(rid) for rid in response_ids if self.message_cache.get(rid)]
    
    def get_message_chain(self, message_id: int) -> List[discord.Message]:
        """Get the full chain of messages (original + responses)"""
        chain = []
        original = self.get_message(message_id)
        if original:
            chain.append(original)
            responses = self.get_responses_to(message_id)
            chain.extend(responses)
        return chain

class DeepDiscordBot(commands.Bot):
    """Main Discord bot class for message tracking and retrieval"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        
        self.message_tracker = MessageTracker()
        self.data_dir = "discord_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing data
        self.load_message_data()
    
    async def setup_hook(self):
        """Setup hook for bot initialization"""
        logger.info("Setting up DeepDiscord bot...")
        
        # Add cogs/commands
        await self.add_cog(MessageCommands(self))
        await self.add_cog(AdminCommands(self))
        
        logger.info("Bot setup complete!")
    
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'Bot logged in as {self.user.name} ({self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="messages with !help"
            )
        )
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        # Ignore bot messages
        if message.author.bot:
            return
        
        # Track the message
        self.message_tracker.add_message(message)
        
        # Process commands
        await self.process_commands(message)
    
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edits"""
        # Update the tracked message
        self.message_tracker.add_message(after)
        logger.info(f"Message {after.id} edited by {after.author.name}")
    
    async def on_message_delete(self, message: discord.Message):
        """Handle message deletions"""
        logger.info(f"Message {message.id} deleted by {message.author.name}")
        # Note: We keep the message in cache for historical reference
    
    def save_message_data(self):
        """Save message data to disk"""
        try:
            data = {
                'message_cache': {
                    str(msg_id): {
                        'id': msg.id,
                        'content': msg.content,
                        'author_id': msg.author.id,
                        'author_name': msg.author.name,
                        'channel_id': msg.channel.id,
                        'guild_id': msg.guild.id if msg.guild else None,
                        'timestamp': msg.created_at.isoformat(),
                        'reference_id': msg.reference.message_id if msg.reference else None
                    }
                    for msg_id, msg in self.message_tracker.message_cache.items()
                },
                'response_chain': {
                    str(msg_id): response_ids
                    for msg_id, response_ids in self.message_tracker.response_chain.items()
                }
            }
            
            with open(os.path.join(self.data_dir, 'messages.json'), 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Message data saved successfully")
        except Exception as e:
            logger.error(f"Error saving message data: {e}")
    
    def load_message_data(self):
        """Load message data from disk"""
        try:
            file_path = os.path.join(self.data_dir, 'messages.json')
            if not os.path.exists(file_path):
                logger.info("No existing message data found")
                return
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Note: We can't fully reconstruct Message objects from disk
            # This is mainly for reference and statistics
            logger.info(f"Loaded {len(data.get('message_cache', {}))} cached messages")
            
        except Exception as e:
            logger.error(f"Error loading message data: {e}")

class MessageCommands(commands.Cog):
    """Commands for message retrieval and analysis"""
    
    def __init__(self, bot: DeepDiscordBot):
        self.bot = bot
    
    @commands.command(name='getmsg')
    async def get_message(self, ctx: commands.Context, message_id: int):
        """Get a message by its ID"""
        try:
            # First check cache
            message = self.bot.message_tracker.get_message(message_id)
            
            if not message:
                # Try to fetch from Discord API
                try:
                    message = await ctx.channel.fetch_message(message_id)
                    self.bot.message_tracker.add_message(message)
                except discord.NotFound:
                    await ctx.send(f"❌ Message with ID `{message_id}` not found.")
                    return
                except discord.Forbidden:
                    await ctx.send(f"❌ Cannot access message with ID `{message_id}` (insufficient permissions).")
                    return
            
            # Create embed for the message
            embed = discord.Embed(
                title="Message Retrieved",
                description=message.content,
                color=discord.Color.blue(),
                timestamp=message.created_at
            )
            
            embed.add_field(
                name="Author",
                value=f"{message.author.mention} ({message.author.name})",
                inline=True
            )
            
            embed.add_field(
                name="Channel",
                value=f"<#{message.channel.id}>",
                inline=True
            )
            
            embed.add_field(
                name="Message ID",
                value=f"`{message.id}`",
                inline=True
            )
            
            if message.reference and message.reference.message_id:
                embed.add_field(
                    name="Replying to",
                    value=f"`{message.reference.message_id}`",
                    inline=True
                )
            
            # Add responses count
            responses = self.bot.message_tracker.get_responses_to(message_id)
            embed.add_field(
                name="Responses",
                value=f"{len(responses)} responses",
                inline=True
            )
            
            embed.set_footer(text=f"Retrieved by {ctx.author.name}")
            
            await ctx.send(embed=embed)
            
        except ValueError:
            await ctx.send("❌ Invalid message ID. Please provide a valid number.")
        except Exception as e:
            logger.error(f"Error in get_message: {e}")
            await ctx.send("❌ An error occurred while retrieving the message.")
    
    @commands.command(name='responses')
    async def get_responses(self, ctx: commands.Context, message_id: int):
        """Get all responses to a specific message"""
        try:
            responses = self.bot.message_tracker.get_responses_to(message_id)
            
            if not responses:
                await ctx.send(f"📭 No responses found for message `{message_id}`.")
                return
            
            # Create embed for responses
            embed = discord.Embed(
                title=f"Responses to Message {message_id}",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            
            for i, response in enumerate(responses[:10], 1):  # Limit to 10 responses
                embed.add_field(
                    name=f"Response {i}",
                    value=f"**{response.author.name}**: {response.content[:100]}{'...' if len(response.content) > 100 else ''}\n[View Message]({response.jump_url})",
                    inline=False
                )
            
            if len(responses) > 10:
                embed.set_footer(text=f"Showing 10 of {len(responses)} responses")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in get_responses: {e}")
            await ctx.send("❌ An error occurred while retrieving responses.")
    
    @commands.command(name='chain')
    async def get_message_chain(self, ctx: commands.Context, message_id: int):
        """Get the full chain of messages (original + responses)"""
        try:
            chain = self.bot.message_tracker.get_message_chain(message_id)
            
            if not chain:
                await ctx.send(f"📭 No message chain found for message `{message_id}`.")
                return
            
            # Create embed for the chain
            embed = discord.Embed(
                title=f"Message Chain for {message_id}",
                color=discord.Color.purple(),
                timestamp=datetime.utcnow()
            )
            
            for i, message in enumerate(chain[:5], 1):  # Limit to 5 messages
                embed.add_field(
                    name=f"{'Original' if i == 1 else f'Response {i-1}'}",
                    value=f"**{message.author.name}**: {message.content[:150]}{'...' if len(message.content) > 150 else ''}\n[View Message]({message.jump_url})",
                    inline=False
                )
            
            if len(chain) > 5:
                embed.set_footer(text=f"Showing 5 of {len(chain)} messages in chain")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in get_message_chain: {e}")
            await ctx.send("❌ An error occurred while retrieving the message chain.")
    
    @commands.command(name='stats')
    async def get_stats(self, ctx: commands.Context):
        """Get bot statistics"""
        try:
            cache_size = len(self.bot.message_tracker.message_cache)
            response_chains = len(self.bot.message_tracker.response_chain)
            total_responses = sum(len(responses) for responses in self.bot.message_tracker.response_chain.values())
            
            embed = discord.Embed(
                title="DeepDiscord Bot Statistics",
                color=discord.Color.gold(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(
                name="Cached Messages",
                value=f"{cache_size:,}",
                inline=True
            )
            
            embed.add_field(
                name="Response Chains",
                value=f"{response_chains:,}",
                inline=True
            )
            
            embed.add_field(
                name="Total Responses",
                value=f"{total_responses:,}",
                inline=True
            )
            
            embed.add_field(
                name="Guilds",
                value=f"{len(self.bot.guilds)}",
                inline=True
            )
            
            embed.add_field(
                name="Channels",
                value=f"{sum(len(guild.channels) for guild in self.bot.guilds)}",
                inline=True
            )
            
            embed.add_field(
                name="Uptime",
                value=f"<t:{int(self.bot.start_time.timestamp())}:R>",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in get_stats: {e}")
            await ctx.send("❌ An error occurred while retrieving statistics.")

class AdminCommands(commands.Cog):
    """Administrative commands"""
    
    def __init__(self, bot: DeepDiscordBot):
        self.bot = bot
    
    @commands.command(name='save')
    @commands.has_permissions(administrator=True)
    async def save_data(self, ctx: commands.Context):
        """Save current message data to disk (Admin only)"""
        try:
            self.bot.save_message_data()
            await ctx.send("✅ Message data saved successfully!")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            await ctx.send("❌ Error saving message data.")
    
    @commands.command(name='clear')
    @commands.has_permissions(administrator=True)
    async def clear_cache(self, ctx: commands.Context):
        """Clear message cache (Admin only)"""
        try:
            cache_size = len(self.bot.message_tracker.message_cache)
            self.bot.message_tracker.message_cache.clear()
            self.bot.message_tracker.response_chain.clear()
            await ctx.send(f"✅ Cleared {cache_size} cached messages!")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            await ctx.send("❌ Error clearing cache.")

async def main():
    """Main function to run the bot"""
    # Get bot token from environment
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("DISCORD_TOKEN not found in environment variables!")
        return
    
    # Create and run bot
    bot = DeepDiscordBot()
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested...")
        # Save data before shutting down
        bot.save_message_data()
        await bot.close()
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 
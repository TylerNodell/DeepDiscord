#!/usr/bin/env python3
"""
Discord Bot for Message Retrieval and Response Tracking
Part of DeepDiscord Project
"""

import discord
from discord.ext import commands, tasks
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
        self.user_history_cache: Dict[int, Dict] = {}  # user_id -> history data
        self.pending_saves: Dict[int, Dict] = {}  # user_id -> pending save data
        
        # Fragment detection
        self.fragment_buffer: Dict[int, List[discord.Message]] = {}  # user_id -> [messages]
        self.fragment_timeout = 30  # seconds to wait for more fragments
        self.combined_messages: Dict[int, str] = {}  # first_message_id -> combined_content
        self.fragment_timers: Dict[int, asyncio.Task] = {}  # user_id -> timer task
    
    async def add_message(self, message: discord.Message):
        """Add a message to the cache"""
        if len(self.message_cache) >= self.max_cache_size:
            # Remove oldest message (simple FIFO)
            oldest_id = next(iter(self.message_cache))
            del self.message_cache[oldest_id]
        
        self.message_cache[message.id] = message
        
        # Add to fragment buffer for potential combining
        await self.add_to_fragment_buffer(message)
        
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
        responses = []
        for rid in response_ids:
            msg = self.message_cache.get(rid)
            if msg is not None:
                responses.append(msg)
        return responses
    
    def get_message_chain(self, message_id: int) -> List[discord.Message]:
        """Get the full chain of messages (original + responses)"""
        chain = []
        original = self.get_message(message_id)
        if original:
            chain.append(original)
            responses = self.get_responses_to(message_id)
            chain.extend(responses)
        return chain
    
    def store_user_history(self, user_id: int, user_data: dict, messages: List[dict]):
        """Store user history data for later use"""
        self.user_history_cache[user_id] = {
            'user_data': user_data,
            'messages': messages,
            'timestamp': datetime.utcnow()
        }
    
    def get_user_history(self, user_id: int) -> Optional[Dict]:
        """Get stored user history data"""
        return self.user_history_cache.get(user_id)
    
    def store_pending_save(self, user_id: int, save_data: dict):
        """Store data for pending save operation"""
        self.pending_saves[user_id] = save_data
    
    def get_pending_save(self, user_id: int) -> Optional[Dict]:
        """Get pending save data"""
        return self.pending_saves.get(user_id)
    
    def clear_pending_save(self, user_id: int):
        """Clear pending save data"""
        if user_id in self.pending_saves:
            del self.pending_saves[user_id]
    
    def is_potential_fragment(self, message: discord.Message) -> bool:
        """Determine if a message might be a fragment of a larger message"""
        # Check if user has recent messages in buffer
        if message.author.id not in self.fragment_buffer:
            return True
        
        buffer = self.fragment_buffer[message.author.id]
        if not buffer:
            return True
        
        last_msg = buffer[-1]
        time_diff = (message.created_at - last_msg.created_at).total_seconds()
        
        # Within fragment timeout window
        if time_diff <= self.fragment_timeout:
            # Check for fragment indicators
            last_content = last_msg.content.strip()
            curr_content = message.content.strip()
            
            # Incomplete sentence (no ending punctuation)
            if last_content and not last_content[-1] in '.!?;':
                return True
            
            # Continuation patterns
            continuation_starters = ['and', 'but', 'also', 'oh', 'wait', 'actually', 'or', 'i mean', 'correction']
            if any(curr_content.lower().startswith(pattern) for pattern in continuation_starters):
                return True
            
            # Very short time gap (likely rapid typing)
            if time_diff <= 5:
                return True
        
        return False
    
    async def add_to_fragment_buffer(self, message: discord.Message):
        """Add message to fragment buffer for potential combining"""
        user_id = message.author.id
        
        # Cancel existing timer if any
        if user_id in self.fragment_timers:
            self.fragment_timers[user_id].cancel()
        
        if user_id not in self.fragment_buffer:
            self.fragment_buffer[user_id] = []
        
        # Check if this should start a new fragment group
        if self.fragment_buffer[user_id] and not self.is_potential_fragment(message):
            # Process existing buffer first
            self.process_fragment_buffer(user_id)
            self.fragment_buffer[user_id] = []
        
        self.fragment_buffer[user_id].append(message)
        
        # Start new timer for this user
        self.fragment_timers[user_id] = asyncio.create_task(
            self._fragment_timeout_handler(user_id)
        )
    
    async def _fragment_timeout_handler(self, user_id: int):
        """Handle fragment timeout - process buffer after waiting"""
        await asyncio.sleep(self.fragment_timeout)
        self.process_fragment_buffer(user_id)
        self.fragment_buffer[user_id] = []
        if user_id in self.fragment_timers:
            del self.fragment_timers[user_id]
    
    def process_fragment_buffer(self, user_id: int):
        """Process and combine messages in fragment buffer"""
        if user_id not in self.fragment_buffer or not self.fragment_buffer[user_id]:
            return
        
        buffer = self.fragment_buffer[user_id]
        
        # If only one message, no combining needed
        if len(buffer) == 1:
            return
        
        # Combine messages
        combined_content = " ".join(msg.content for msg in buffer)
        first_msg_id = buffer[0].id
        
        # Store combined message
        self.combined_messages[first_msg_id] = combined_content
        
        # Mark other messages as fragments of the first
        for msg in buffer[1:]:
            self.combined_messages[msg.id] = f"[Fragment of {first_msg_id}]"
    
    def get_combined_content(self, message_id: int) -> Optional[str]:
        """Get combined content if message was part of fragments"""
        return self.combined_messages.get(message_id)

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
        self.start_time = datetime.utcnow()
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
        if self.user:
            logger.info(f'Bot logged in as {self.user.name} ({self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="messages (ignoring bot channels)"
            )
        )
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        # Ignore bot messages
        if message.author.bot:
            return
        
        # Ignore bot commands (messages starting with command prefix)
        # Handle both string and tuple command prefixes
        if isinstance(self.command_prefix, str):
            if message.content.startswith(self.command_prefix):
                # Still process commands but don't track them
                await self.process_commands(message)
                return
        elif isinstance(self.command_prefix, (tuple, list)):
            if any(message.content.startswith(prefix) for prefix in self.command_prefix):
                # Still process commands but don't track them
                await self.process_commands(message)
                return
        
        # Ignore messages in bot channels (only for guild channels)
        if isinstance(message.channel, discord.TextChannel):
            channel_name = message.channel.name.lower()
            bot_channel_keywords = ['bot', 'commands', 'admin', 'mod', 'staff']
            if any(keyword in channel_name for keyword in bot_channel_keywords):
                logger.info(f"Ignoring message in bot channel: {message.channel.name}")
                return
        
        # Track the message
        await self.message_tracker.add_message(message)
        
        # Process commands
        await self.process_commands(message)
    
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edits"""
        # Update the tracked message
        await self.message_tracker.add_message(after)
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
                    await self.bot.message_tracker.add_message(message)
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
    
    @commands.command(name='userhistory')
    async def get_user_history(self, ctx: commands.Context, user_mention_or_id: str, limit: int = 100):
        """Get all messages from a specific user across the server"""
        try:
            logger.info(f"=== USER HISTORY STARTED ===")
            logger.info(f"Command by: {ctx.author.name} ({ctx.author.id})")
            logger.info(f"Channel: {getattr(ctx.channel, 'name', 'DM')} ({ctx.channel.id})")
            logger.info(f"Guild: {ctx.guild.name if ctx.guild else 'DM'} ({ctx.guild.id if ctx.guild else 'N/A'})")
            logger.info(f"User parameter: {user_mention_or_id}")
            logger.info(f"Limit: {limit}")
            
            # Parse user mention or ID
            if user_mention_or_id.startswith('<@') and user_mention_or_id.endswith('>'):
                # Remove <@ and > and any ! if present
                user_id = int(user_mention_or_id.replace('<@', '').replace('!', '').replace('>', ''))
                logger.info(f"Parsed user mention to ID: {user_id}")
            else:
                try:
                    user_id = int(user_mention_or_id)
                    logger.info(f"Parsed user ID: {user_id}")
                except ValueError:
                    logger.error(f"Invalid user format: {user_mention_or_id}")
                    await ctx.send("❌ Invalid user format. Use @username or user ID.")
                    return
            
            # Find the user - try guild members first, then fallback to user object
            user = None
            for guild in self.bot.guilds:
                user = guild.get_member(user_id)
                if user:
                    break
            
            # If not found as member, try to get user object directly
            if not user:
                try:
                    user = await self.bot.fetch_user(user_id)
                except discord.NotFound:
                    await ctx.send("❌ User not found. Please check the user ID or mention.")
                    return
                except discord.HTTPException as e:
                    await ctx.send(f"❌ Error fetching user: {e}")
                    return
            
            # Send initial status message
            status_msg = await ctx.send(f"🔍 Searching for messages from {user.name}... This may take a while.")
            
            all_messages = []
            channels_searched = 0
            total_channels = 0
            
            # Count total channels first
            for guild in self.bot.guilds:
                for channel in guild.channels:
                    if isinstance(channel, discord.TextChannel):
                        total_channels += 1
            
            # Search through all text channels in all guilds
            for guild in self.bot.guilds:
                for channel in guild.channels:
                    if isinstance(channel, discord.TextChannel):
                        channels_searched += 1
                        
                        # Update status every 5 channels
                        if channels_searched % 5 == 0:
                            await status_msg.edit(content=f"🔍 Searching for messages from {user.name}... ({channels_searched}/{total_channels} channels)")
                        
                        try:
                            # Fetch messages from this channel
                            async for message in channel.history(limit=limit, oldest_first=False):
                                # Skip bot messages
                                if message.author.bot:
                                    continue
                                # Skip bot commands (messages starting with command prefix)
                                prefixes = self.bot.command_prefix
                                if isinstance(prefixes, str):
                                    if message.content.startswith(prefixes):
                                        continue
                                elif isinstance(prefixes, (tuple, list)):
                                    if any(message.content.startswith(prefix) for prefix in prefixes):
                                        continue
                                # Skip messages in bot channels
                                channel_name = getattr(message.channel, 'name', '').lower()
                                bot_channel_keywords = ['bot', 'commands', 'admin', 'mod', 'staff']
                                if any(keyword in channel_name for keyword in bot_channel_keywords):
                                    continue
                                # Skip messages in any channel with 'bot' or 'command' in the name
                                if 'bot' in channel_name or 'command' in channel_name:
                                    continue
                                all_messages.append({
                                    'content': message.content,
                                    'channel': channel.name,
                                    'channel_id': channel.id,
                                    'guild': guild.name,
                                    'timestamp': message.created_at.isoformat(),
                                    'message_id': message.id,
                                    'jump_url': message.jump_url
                                })
                                # Add to bot's message tracker
                                await self.bot.message_tracker.add_message(message)
                        except discord.Forbidden:
                            # Skip channels we don't have access to
                            continue
                        except Exception as e:
                            logger.warning(f"Error fetching messages from {channel.name}: {e}")
                            continue
            
            # Sort messages by timestamp (newest first)
            all_messages.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Store the data for later use
            user_data = {
                'id': user.id,
                'name': user.name,
                'display_name': getattr(user, 'display_name', user.name)
            }
            self.bot.message_tracker.store_user_history(user_id, user_data, all_messages)
            
            # Update status message
            await status_msg.edit(content=f"✅ Found {len(all_messages)} messages from {user.name} across {channels_searched} channels.")
            
            if not all_messages:
                await ctx.send(f"📭 No messages found from {user.name} in the searchable history.")
                return
            
            # Create embed with results
            embed = discord.Embed(
                title=f"Message History for {user.name}",
                description=f"Found {len(all_messages)} messages across {channels_searched} channels",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            
            # Add user info
            embed.add_field(
                name="User",
                value=f"{user.mention} ({user.name})",
                inline=True
            )
            
            embed.add_field(
                name="User ID",
                value=f"`{user.id}`",
                inline=True
            )
            
            embed.add_field(
                name="Search Limit",
                value=f"{limit} messages per channel",
                inline=True
            )
            
            # Show first 10 messages
            for i, msg_data in enumerate(all_messages[:10], 1):
                # Truncate content if too long
                content = msg_data['content']
                if len(content) > 100:
                    content = content[:97] + "..."
                
                embed.add_field(
                    name=f"Message {i}",
                    value=f"**{msg_data['guild']}** → **#{msg_data['channel']}**\n{content}\n[View Message]({msg_data['jump_url']})",
                    inline=False
                )
            
            if len(all_messages) > 10:
                embed.add_field(
                    name="More Messages",
                    value=f"... and {len(all_messages) - 10} more messages",
                    inline=False
                )
            
            embed.set_footer(text=f"Retrieved by {ctx.author.name} | Search completed")
            
            await ctx.send(embed=embed)
            
            # If there are many messages, offer to save to file
            if len(all_messages) > 20:
                # Store save data for pending operation
                save_data = {
                    'user_data': user_data,
                    'messages': all_messages,
                    'channels_searched': channels_searched
                }
                self.bot.message_tracker.store_pending_save(user_id, save_data)
                
                save_embed = discord.Embed(
                    title="Save Messages",
                    description=f"Found {len(all_messages)} messages from {user.name}. Would you like to save them to a file?",
                    color=discord.Color.green()
                )
                save_embed.add_field(
                    name="Quick Save",
                    value="Reply with `yes` to save immediately, or `no` to skip",
                    inline=False
                )
                save_embed.add_field(
                    name="Manual Save",
                    value=f"`!saveuser {user_id}` (Admin only)",
                    inline=False
                )
                save_embed.set_footer(text="Data will be kept in memory for 10 minutes")
                await ctx.send(embed=save_embed)
                logger.info("Save prompt sent to user")
            else:
                logger.info(f"Few messages found ({len(all_messages)}), no save prompt needed")
            
            logger.info("=== USER HISTORY COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            logger.error(f"=== USER HISTORY FAILED ===")
            logger.error(f"Error in get_user_history: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await ctx.send("❌ An error occurred while retrieving user history.")
    
    @commands.command(name='saveuser')
    @commands.has_permissions(administrator=True)
    async def save_user_messages(self, ctx: commands.Context, user_mention_or_id: Optional[str] = None):
        """Save all messages from a user to a JSON file (Admin only)"""
        try:
            logger.info(f"=== SAVE USER MESSAGES STARTED ===")
            logger.info(f"Command by: {ctx.author.name} ({ctx.author.id})")
            logger.info(f"Channel: {getattr(ctx.channel, 'name', 'DM')} ({ctx.channel.id})")
            logger.info(f"Guild: {ctx.guild.name if ctx.guild else 'DM'} ({ctx.guild.id if ctx.guild else 'N/A'})")
            logger.info(f"User parameter: {user_mention_or_id}")
            
            user_id = None
            
            # If no user specified, try to get from pending saves
            if user_mention_or_id is None:
                logger.info("No user specified, checking for pending saves...")
                # Check if there are any pending saves
                if not self.bot.message_tracker.pending_saves:
                    logger.warning("No pending saves found")
                    await ctx.send("❌ No pending saves found. Please specify a user ID or run `!userhistory` first.")
                    return
                
                # Use the most recent pending save
                user_id = list(self.bot.message_tracker.pending_saves.keys())[-1]
                logger.info(f"Using pending save for user ID: {user_id}")
                save_data = self.bot.message_tracker.get_pending_save(user_id)
                if not save_data:
                    logger.error(f"No pending save data found for user ID: {user_id}")
                    await ctx.send("❌ No pending save data found for this user.")
                    return
                
                user_data = save_data['user_data']
                all_messages = save_data['messages']
                user_name = user_data['name']
                logger.info(f"Retrieved cached data for user: {user_name} ({user_id})")
                logger.info(f"Cached messages count: {len(all_messages)}")
                
            else:
                logger.info("User parameter provided, parsing...")
                # Parse user mention or ID
                if user_mention_or_id.startswith('<@') and user_mention_or_id.endswith('>'):
                    user_id = int(user_mention_or_id.replace('<@', '').replace('!', '').replace('>', ''))
                    logger.info(f"Parsed user mention to ID: {user_id}")
                else:
                    try:
                        user_id = int(user_mention_or_id)
                        logger.info(f"Parsed user ID: {user_id}")
                    except ValueError:
                        logger.error(f"Invalid user format: {user_mention_or_id}")
                        await ctx.send("❌ Invalid user format. Use @username or user ID.")
                        return
                
                # Check if we have cached data for this user
                logger.info(f"Checking for cached data for user ID: {user_id}")
                cached_data = self.bot.message_tracker.get_user_history(user_id)
                if cached_data:
                    logger.info("Found cached data, using it")
                    user_data = cached_data['user_data']
                    all_messages = cached_data['messages']
                    user_name = user_data['name']
                    logger.info(f"Using cached data for user: {user_name} ({user_id})")
                    logger.info(f"Cached messages count: {len(all_messages)}")
                else:
                    logger.info("No cached data found, fetching from Discord...")
                    # Fallback to fetching from Discord
                    user = None
                    for guild in self.bot.guilds:
                        user = guild.get_member(user_id)
                        if user:
                            logger.info(f"Found user as member in guild: {guild.name}")
                            break
                    
                    if not user:
                        logger.info("User not found as member, trying to fetch user object...")
                        try:
                            user = await self.bot.fetch_user(user_id)
                            logger.info(f"Successfully fetched user: {user.name}")
                        except discord.NotFound:
                            logger.error(f"User not found: {user_id}")
                            await ctx.send("❌ User not found. Please check the user ID or mention.")
                            return
                        except discord.HTTPException as e:
                            logger.error(f"HTTP error fetching user {user_id}: {e}")
                            await ctx.send(f"❌ Error fetching user: {e}")
                            return
                    
                    status_msg = await ctx.send(f"💾 Fetching messages from {user.name}...")
                    logger.info(f"Starting message fetch for user: {user.name} ({user_id})")
                    
                    all_messages = []
                    channels_searched = 0
                    total_channels = 0
                    
                    # Count total channels first
                    for guild in self.bot.guilds:
                        for channel in guild.channels:
                            if isinstance(channel, discord.TextChannel):
                                total_channels += 1
                    
                    logger.info(f"Will search {total_channels} text channels")
                    
                    # Search through all text channels
                    for guild in self.bot.guilds:
                        logger.info(f"Searching guild: {guild.name}")
                        for channel in guild.channels:
                            if isinstance(channel, discord.TextChannel):
                                channels_searched += 1
                                logger.info(f"Searching channel {channels_searched}/{total_channels}: {channel.name}")
                                
                                try:
                                    message_count = 0
                                    async for message in channel.history(limit=None, oldest_first=True):
                                        if message.author.id == user_id:
                                            message_count += 1
                                            all_messages.append({
                                                'content': message.content,
                                                'channel_name': channel.name,
                                                'channel_id': channel.id,
                                                'guild_name': guild.name,
                                                'guild_id': guild.id,
                                                'timestamp': message.created_at.isoformat(),
                                                'message_id': message.id,
                                                'jump_url': message.jump_url,
                                                'attachments': [att.url for att in message.attachments],
                                                'embeds': len(message.embeds),
                                                'mentions': [user.id for user in message.mentions],
                                                'role_mentions': [role.id for role in message.role_mentions]
                                            })
                                    
                                    if message_count > 0:
                                        logger.info(f"Found {message_count} messages in {channel.name}")
                                        
                                except discord.Forbidden:
                                    logger.warning(f"No permission to read channel: {channel.name}")
                                    continue
                                except Exception as e:
                                    logger.warning(f"Error fetching messages from {channel.name}: {e}")
                                    continue
                    
                    logger.info(f"Completed search. Found {len(all_messages)} total messages across {channels_searched} channels")
                    
                    if not all_messages:
                        logger.warning(f"No messages found for user: {user.name}")
                        await status_msg.edit(content=f"📭 No messages found from {user.name}.")
                        return
                    
                    user_data = {
                        'id': user.id,
                        'name': user.name,
                        'display_name': getattr(user, 'display_name', user.name)
                    }
                    user_name = user.name
                    logger.info(f"User data prepared: {user_name} ({user_id})")
            
            # Create filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"user_messages_{user_name}_{user_id}_{timestamp}.json"
            filepath = os.path.join(self.bot.data_dir, filename)
            logger.info(f"Creating file: {filename}")
            logger.info(f"File path: {filepath}")
            
            # Prepare data structure
            logger.info("Preparing data structure...")
            data = {
                'user_info': user_data,
                'search_info': {
                    'searched_at': datetime.utcnow().isoformat(),
                    'total_messages': len(all_messages),
                    'guilds_searched': len(self.bot.guilds)
                },
                'messages': all_messages
            }
            
            logger.info(f"Data structure prepared:")
            logger.info(f"  - User info: {user_data}")
            logger.info(f"  - Total messages: {len(all_messages)}")
            logger.info(f"  - Guilds searched: {len(self.bot.guilds)}")
            
            # Save to JSON file
            logger.info("Writing data to JSON file...")
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                logger.info("JSON file written successfully")
            except Exception as e:
                logger.error(f"Error writing JSON file: {e}")
                raise
            
            # Get file size
            file_size = os.path.getsize(filepath)
            logger.info(f"File size: {file_size} bytes ({file_size / 1024:.1f} KB)")
            
            # Create file object for Discord
            logger.info("Creating Discord file object...")
            try:
                with open(filepath, 'rb') as f:
                    file_obj = discord.File(f, filename=filename)
                logger.info("Discord file object created successfully")
            except Exception as e:
                logger.error(f"Error creating Discord file object: {e}")
                raise
            
            # Create embed
            logger.info("Creating success embed...")
            embed = discord.Embed(
                title="User Messages Saved",
                description=f"Successfully saved {len(all_messages)} messages from {user_name}",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(
                name="File",
                value=filename,
                inline=True
            )
            
            embed.add_field(
                name="Messages",
                value=f"{len(all_messages):,}",
                inline=True
            )
            
            embed.add_field(
                name="Size",
                value=f"{os.path.getsize(filepath) / 1024:.1f} KB",
                inline=True
            )
            
            # Send embed and file
            logger.info("Sending embed and file to Discord...")
            await ctx.send(embed=embed)
            await ctx.send(file=file_obj)
            logger.info("Embed and file sent successfully")
            
            # Clear pending save data
            logger.info(f"Clearing pending save data for user: {user_id}")
            self.bot.message_tracker.clear_pending_save(user_id)
            
            logger.info("=== SAVE USER MESSAGES COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            logger.error(f"=== SAVE USER MESSAGES FAILED ===")
            logger.error(f"Error in save_user_messages: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await ctx.send("❌ An error occurred while saving user messages.")
    
    @commands.command(name='yes')
    async def quick_save(self, ctx: commands.Context):
        """Quick save for pending user data"""
        try:
            logger.info(f"=== QUICK SAVE STARTED ===")
            logger.info(f"Command by: {ctx.author.name} ({ctx.author.id})")
            
            if not self.bot.message_tracker.pending_saves:
                logger.warning("No pending saves found for quick save")
                await ctx.send("❌ No pending saves found. Run `!userhistory` first.")
                return
            
            # Get the most recent pending save
            user_id = list(self.bot.message_tracker.pending_saves.keys())[-1]
            logger.info(f"Using pending save for user ID: {user_id}")
            save_data = self.bot.message_tracker.get_pending_save(user_id)
            
            if not save_data:
                logger.error(f"No pending save data found for user ID: {user_id}")
                await ctx.send("❌ No pending save data found.")
                return
            
            logger.info(f"Found pending save data with {len(save_data['messages'])} messages")
            
            # Call the save function with the cached data
            logger.info("Calling save_user_messages with cached data...")
            await self.save_user_messages(ctx)
            
            logger.info("=== QUICK SAVE COMPLETED ===")
            
        except Exception as e:
            logger.error(f"=== QUICK SAVE FAILED ===")
            logger.error(f"Error in quick_save: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await ctx.send("❌ An error occurred during quick save.")
    
    @commands.command(name='no')
    async def skip_save(self, ctx: commands.Context):
        """Skip saving pending user data"""
        try:
            logger.info(f"=== SKIP SAVE STARTED ===")
            logger.info(f"Command by: {ctx.author.name} ({ctx.author.id})")
            
            if not self.bot.message_tracker.pending_saves:
                logger.warning("No pending saves found for skip save")
                await ctx.send("❌ No pending saves found.")
                return
            
            # Get the most recent pending save
            user_id = list(self.bot.message_tracker.pending_saves.keys())[-1]
            logger.info(f"Using pending save for user ID: {user_id}")
            save_data = self.bot.message_tracker.get_pending_save(user_id)
            
            if not save_data:
                logger.error(f"No pending save data found for user ID: {user_id}")
                await ctx.send("❌ No pending save data found.")
                return
            
            user_name = save_data['user_data']['name']
            message_count = len(save_data['messages'])
            logger.info(f"Skipping save for user: {user_name} ({user_id}) with {message_count} messages")
            
            # Clear the pending save
            logger.info(f"Clearing pending save data for user: {user_id}")
            self.bot.message_tracker.clear_pending_save(user_id)
            
            embed = discord.Embed(
                title="Save Skipped",
                description=f"Skipped saving messages from {user_name}",
                color=discord.Color.orange(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(
                name="Messages",
                value=f"{message_count:,} messages not saved",
                inline=True
            )
            
            embed.add_field(
                name="Data",
                value="Cleared from memory",
                inline=True
            )
            
            logger.info("Sending skip confirmation embed...")
            await ctx.send(embed=embed)
            logger.info("Skip confirmation sent successfully")
            
            logger.info("=== SKIP SAVE COMPLETED ===")
            
        except Exception as e:
            logger.error(f"=== SKIP SAVE FAILED ===")
            logger.error(f"Error in skip_save: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await ctx.send("❌ An error occurred while skipping save.")
    
    @commands.command(name='fragments')
    async def show_fragments(self, ctx: commands.Context, message_id: int):
        """Show combined content for fragmented messages"""
        try:
            # Check if this message has combined content
            combined = self.bot.message_tracker.get_combined_content(message_id)
            
            if not combined:
                await ctx.send(f"❌ No fragment data found for message `{message_id}`")
                return
            
            # Check if this is a fragment reference
            if combined.startswith("[Fragment of"):
                await ctx.send(f"ℹ️ This message is {combined}")
                return
            
            # Get the original message info
            message = self.bot.message_tracker.get_message(message_id)
            
            embed = discord.Embed(
                title="Combined Message Fragments",
                color=discord.Color.green()
            )
            
            if message:
                embed.add_field(
                    name="Author",
                    value=f"{message.author.mention}",
                    inline=True
                )
                embed.add_field(
                    name="Channel",
                    value=f"{message.channel.mention}",
                    inline=True
                )
                embed.add_field(
                    name="First Fragment Time",
                    value=message.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    inline=True
                )
            
            # Show the combined content
            embed.add_field(
                name="Combined Content",
                value=combined[:1024] if len(combined) > 1024 else combined,
                inline=False
            )
            
            if len(combined) > 1024:
                embed.add_field(
                    name="Note",
                    value=f"Content truncated. Full length: {len(combined)} characters",
                    inline=False
                )
            
            embed.set_footer(text=f"Message ID: {message_id}")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in show_fragments: {e}")
            await ctx.send("❌ An error occurred while retrieving fragment data.")

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
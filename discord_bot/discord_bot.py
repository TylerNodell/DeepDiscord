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
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os
from dotenv import load_dotenv
import zipfile
import requests
import tempfile
import io
from collections import defaultdict
import uuid

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
            return False  # No buffer means first message, not a fragment
        
        buffer = self.fragment_buffer[message.author.id]
        if not buffer:
            return False  # Empty buffer means first message, not a fragment
        
        last_msg = buffer[-1]
        time_diff = (message.created_at - last_msg.created_at).total_seconds()
        
        # Must be within fragment timeout window
        if time_diff > self.fragment_timeout:
            return False
        
        last_content = last_msg.content.strip()
        curr_content = message.content.strip()
        
        # Check for fragment indicators in priority order
        
        # 1. Incomplete sentence (no ending punctuation) - STRONG indicator
        if last_content and not last_content[-1] in '.!?;':
            return True
        
        # 2. Continuation patterns - STRONG indicator
        continuation_starters = ['and', 'but', 'also', 'oh', 'wait', 'actually', 'or', 'i mean', 'correction']
        if any(curr_content.lower().startswith(pattern) for pattern in continuation_starters):
            return True
        
        # 3. Very short time gap AND both messages lack punctuation - WEAK indicator
        if time_diff <= 3 and not (last_content and last_content[-1] in '.!?;') and not (curr_content and curr_content[-1] in '.!?;'):
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

class ConsentManager:
    """Manages user consent for training data collection"""
    
    def __init__(self, data_dir: str = "discord_data"):
        self.data_dir = data_dir
        self.consent_file = os.path.join(data_dir, "user_consents.json")
        self.pending_requests = {}  # request_id -> {user_id, requester_id, timestamp}
        self.load_consents()
    
    def load_consents(self):
        """Load existing consent data"""
        try:
            if os.path.exists(self.consent_file):
                with open(self.consent_file, 'r') as f:
                    self.consents = json.load(f)
            else:
                self.consents = {}
        except Exception as e:
            logger.error(f"Error loading consents: {e}")
            self.consents = {}
    
    def save_consents(self):
        """Save consent data to file"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.consent_file, 'w') as f:
                json.dump(self.consents, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving consents: {e}")
    
    def has_consent(self, user_id: int) -> bool:
        """Check if user has given consent"""
        user_consent = self.consents.get(str(user_id))
        if not user_consent:
            return False
        
        # Check if consent is still valid (consent no longer expires)
        if user_consent.get('status') != 'granted':
            return False
        
        return True
    
    def grant_consent(self, user_id: int, requester_id: int):
        """Grant consent for user (consent no longer expires)"""
        consent_data = {
            'status': 'granted',
            'granted_at': datetime.now().isoformat(),
            'granted_by_request_from': requester_id,
            'user_id': user_id
        }
        
        self.consents[str(user_id)] = consent_data
        self.save_consents()
        logger.info(f"🔒 CONSENT GRANTED: User {user_id} by requester {requester_id}")
        logger.info(f"   ♾️  Consent does not expire")
    
    def revoke_consent(self, user_id: int):
        """Revoke consent for user"""
        if str(user_id) in self.consents:
            self.consents[str(user_id)]['status'] = 'revoked'
            self.consents[str(user_id)]['revoked_at'] = datetime.now().isoformat()
            self.save_consents()
            logger.info(f"🔒 CONSENT REVOKED: User {user_id}")
    
    def get_consent_info(self, user_id: int) -> Optional[Dict]:
        """Get detailed consent information for user"""
        return self.consents.get(str(user_id))
    
    def create_consent_request(self, user_id: int, requester_id: int) -> str:
        """Create a pending consent request"""
        request_id = str(uuid.uuid4())[:8]  # Short UUID
        self.pending_requests[request_id] = {
            'user_id': user_id,
            'requester_id': requester_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        return request_id
    
    def get_pending_request(self, request_id: str) -> Optional[Dict]:
        """Get pending request by ID"""
        return self.pending_requests.get(request_id)
    
    def complete_request(self, request_id: str, granted: bool):
        """Mark request as completed"""
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request['status'] = 'granted' if granted else 'denied'
            request['completed_at'] = datetime.now().isoformat()
            
            if granted:
                self.grant_consent(
                    request['user_id'], 
                    request['requester_id']
                )

class AuthorizationManager:
    """Manages user authorization for training data commands"""
    
    def __init__(self, data_dir: str = "discord_data"):
        self.data_dir = data_dir
        self.auth_file = os.path.join(data_dir, "authorized_users.json")
        self.owner_user_id = 97544083005771776  # Always authorized
        self.load_authorized_users()
    
    def load_authorized_users(self):
        """Load existing authorized users"""
        try:
            if os.path.exists(self.auth_file):
                with open(self.auth_file, 'r') as f:
                    data = json.load(f)
                    self.authorized_users = set(data.get('authorized_users', []))
            else:
                self.authorized_users = set()
        except Exception as e:
            logger.error(f"Error loading authorized users: {e}")
            self.authorized_users = set()
    
    def save_authorized_users(self):
        """Save authorized users to file"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            data = {
                'authorized_users': list(self.authorized_users),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.auth_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving authorized users: {e}")
    
    def is_authorized(self, user_id: int, is_admin: bool = False) -> bool:
        """Check if user is authorized to use training data commands"""
        # Owner is always authorized
        if user_id == self.owner_user_id:
            return True
        
        # Admins are always authorized
        if is_admin:
            return True
        
        # Check authorized users list
        return user_id in self.authorized_users
    
    def authorize_user(self, user_id: int) -> bool:
        """Authorize a user"""
        if user_id not in self.authorized_users:
            self.authorized_users.add(user_id)
            self.save_authorized_users()
            logger.info(f"🔐 USER AUTHORIZED: {user_id} for training data commands")
            return True
        logger.info(f"🔐 USER ALREADY AUTHORIZED: {user_id}")
        return False
    
    def deauthorize_user(self, user_id: int) -> bool:
        """Deauthorize a user (cannot deauthorize owner)"""
        if user_id == self.owner_user_id:
            logger.warning(f"🔐 CANNOT DEAUTHORIZE OWNER: {user_id}")
            return False
        
        if user_id in self.authorized_users:
            self.authorized_users.remove(user_id)
            self.save_authorized_users()
            logger.info(f"🔐 USER DEAUTHORIZED: {user_id} from training data commands")
            return True
        logger.info(f"🔐 USER NOT AUTHORIZED: {user_id} (cannot deauthorize)")
        return False
    
    def get_authorized_users(self) -> List[int]:
        """Get list of all authorized users"""
        return [self.owner_user_id] + list(self.authorized_users)

class ChannelManager:
    """Manages which channels the bot is allowed to post in"""
    
    def __init__(self, data_dir: str = "discord_data"):
        self.data_dir = data_dir
        self.channels_file = os.path.join(data_dir, "allowed_channels.json")
        self.load_allowed_channels()
    
    def load_allowed_channels(self):
        """Load existing allowed channels"""
        try:
            if os.path.exists(self.channels_file):
                with open(self.channels_file, 'r') as f:
                    data = json.load(f)
                    # Store as guild_id -> set of channel_ids
                    self.allowed_channels = {}
                    for guild_id, channels in data.get('allowed_channels', {}).items():
                        self.allowed_channels[int(guild_id)] = set(channels)
            else:
                self.allowed_channels = {}
        except Exception as e:
            logger.error(f"Error loading allowed channels: {e}")
            self.allowed_channels = {}
    
    def save_allowed_channels(self):
        """Save allowed channels to file"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            # Convert sets to lists for JSON serialization
            serializable_data = {}
            for guild_id, channels in self.allowed_channels.items():
                serializable_data[str(guild_id)] = list(channels)
            
            data = {
                'allowed_channels': serializable_data,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.channels_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving allowed channels: {e}")
    
    def is_channel_allowed(self, guild_id: int, channel_id: int) -> bool:
        """Check if bot is allowed to post in this channel"""
        # If no restrictions set for this guild, allow all channels
        if guild_id not in self.allowed_channels:
            return True
        
        # If restrictions exist, check if channel is in allowed list
        return channel_id in self.allowed_channels[guild_id]
    
    def add_allowed_channel(self, guild_id: int, channel_id: int) -> bool:
        """Add a channel to allowed list"""
        if guild_id not in self.allowed_channels:
            self.allowed_channels[guild_id] = set()
        
        if channel_id not in self.allowed_channels[guild_id]:
            self.allowed_channels[guild_id].add(channel_id)
            self.save_allowed_channels()
            logger.info(f"📺 CHANNEL ALLOWED: {channel_id} in guild {guild_id}")
            return True
        logger.info(f"📺 CHANNEL ALREADY ALLOWED: {channel_id} in guild {guild_id}")
        return False
    
    def remove_allowed_channel(self, guild_id: int, channel_id: int) -> bool:
        """Remove a channel from allowed list"""
        if guild_id in self.allowed_channels and channel_id in self.allowed_channels[guild_id]:
            self.allowed_channels[guild_id].remove(channel_id)
            
            # If no channels left for this guild, remove the guild entry
            if not self.allowed_channels[guild_id]:
                del self.allowed_channels[guild_id]
                logger.info(f"📺 ALL CHANNEL RESTRICTIONS REMOVED: guild {guild_id}")
            else:
                logger.info(f"📺 CHANNEL REMOVED: {channel_id} from guild {guild_id}")
            
            self.save_allowed_channels()
            return True
        logger.info(f"📺 CHANNEL NOT IN ALLOWED LIST: {channel_id} in guild {guild_id}")
        return False
    
    def get_allowed_channels(self, guild_id: int) -> List[int]:
        """Get list of allowed channels for a guild"""
        return list(self.allowed_channels.get(guild_id, []))
    
    def clear_allowed_channels(self, guild_id: int) -> bool:
        """Clear all channel restrictions for a guild (allow all channels)"""
        if guild_id in self.allowed_channels:
            del self.allowed_channels[guild_id]
            self.save_allowed_channels()
            logger.info(f"📺 CHANNEL RESTRICTIONS CLEARED: guild {guild_id} (all channels now allowed)")
            return True
        logger.info(f"📺 NO RESTRICTIONS TO CLEAR: guild {guild_id}")
        return False

class DeepDiscordBot(commands.Bot):
    """Main Discord bot class for message tracking and retrieval"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        intents.members = True  # Required for member access
        
        super().__init__(
            command_prefix='!!',
            intents=intents,
            help_command=None
        )
        
        self.message_tracker = MessageTracker()
        self.data_dir = "discord_data"
        self.start_time = datetime.utcnow()
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize consent manager
        self.consent_manager = ConsentManager(self.data_dir)
        
        # Initialize authorization manager
        self.authorization_manager = AuthorizationManager(self.data_dir)
        
        # Initialize channel manager
        self.channel_manager = ChannelManager(self.data_dir)
        
        # Load existing data
        self.load_message_data()
    
    def can_use_channel(self, guild_id: int, channel_id: int) -> bool:
        """Check if bot can post in the specified channel"""
        return self.channel_manager.is_channel_allowed(guild_id, channel_id)
    
    async def setup_hook(self):
        """Setup hook for bot initialization"""
        logger.info("Setting up DeepDiscord bot...")
        
        # Add global check for channel restrictions
        self.add_check(self.global_channel_check)
        
        # Add cogs/commands
        await self.add_cog(MessageCommands(self))
        await self.add_cog(TrainingDataCommands(self))
        await self.add_cog(HelpCommands(self))
        await self.add_cog(AdminCommands(self))
        
        logger.info("Bot setup complete!")
    
    async def global_channel_check(self, ctx):
        """Global check for channel restrictions"""
        # Skip check for DMs
        if not ctx.guild:
            return True
        
        # Skip check for help commands (they should always work)
        if ctx.command and ctx.command.name in ['help', 'commands']:
            return True
        
        # Skip check for admin commands or if user is admin
        if ctx.author.guild_permissions.administrator:
            return True
        
        # Skip check for specific admin commands (in case permission check fails)
        if ctx.command and ctx.command.name in ['channels', 'authorize', 'save', 'clear', 'reload', 'hotreload']:
            return True
        
        # Check if channel is allowed
        if not self.can_use_channel(ctx.guild.id, ctx.channel.id):
            logger.info(f"Command '{ctx.command}' blocked in restricted channel #{ctx.channel.name}")
            return False
        
        return True
    
    async def on_command(self, ctx):
        """Called when a command is about to be executed - log command details"""
        try:
            # Get user and guild context
            user_info = f"{ctx.author.display_name} ({ctx.author.name}#{ctx.author.discriminator}, ID: {ctx.author.id})"
            guild_info = f"{ctx.guild.name} (ID: {ctx.guild.id})" if ctx.guild else "DM"
            channel_info = f"#{ctx.channel.name} (ID: {ctx.channel.id})" if ctx.guild else "DM"
            
            # Get command details
            command_name = ctx.command.name if ctx.command else "Unknown"
            command_args = ' '.join(str(arg) for arg in ctx.args[1:]) if len(ctx.args) > 1 else ""
            command_kwargs = ', '.join(f"{k}={v}" for k, v in ctx.kwargs.items()) if ctx.kwargs else ""
            full_command = f"{ctx.prefix}{command_name}"
            if command_args:
                full_command += f" {command_args}"
            if command_kwargs:
                full_command += f" ({command_kwargs})"
            
            # Log command execution
            logger.info(f"🎯 COMMAND EXECUTED: '{full_command}'")
            logger.info(f"   👤 User: {user_info}")
            logger.info(f"   🏠 Guild: {guild_info}")
            logger.info(f"   📺 Channel: {channel_info}")
            logger.info(f"   📝 Raw Message: {ctx.message.content}")
            
            # Log admin status
            if ctx.guild and ctx.author.guild_permissions.administrator:
                logger.info(f"   🛡️ Admin: Yes")
            
            # Log authorization status for training commands
            if command_name in ['generatetrainingdata', 'authorize']:
                is_admin = ctx.author.guild_permissions.administrator if ctx.guild else False
                is_authorized = self.authorization_manager.is_authorized(ctx.author.id, is_admin)
                logger.info(f"   🔐 Authorized: {is_authorized}")
            
        except Exception as e:
            # Don't let logging errors break command execution
            logger.error(f"Error in command logging: {e}")
    
    async def on_command_completion(self, ctx):
        """Called when a command completes successfully"""
        try:
            command_name = ctx.command.name if ctx.command else "Unknown"
            user_info = f"{ctx.author.display_name} (ID: {ctx.author.id})"
            logger.info(f"✅ COMMAND COMPLETED: '{ctx.prefix}{command_name}' by {user_info}")
        except Exception as e:
            logger.error(f"Error in command completion logging: {e}")
    
    async def on_command_error(self, ctx, error):
        """Global error handler for commands"""
        try:
            if isinstance(error, commands.CommandNotFound):
                # Silently ignore unknown commands
                return
            
            elif isinstance(error, commands.MissingPermissions):
                logger.warning(f"❌ PERMISSION DENIED: {ctx.author.display_name} tried '{ctx.command}' but lacks: {', '.join(error.missing_permissions)}")
                await self.safe_send_error(ctx, f"❌ You don't have permission to use this command. Required: {', '.join(error.missing_permissions)}")
            
            elif isinstance(error, commands.CheckFailure):
                # This includes channel restrictions - silently ignore
                logger.info(f"🚫 COMMAND BLOCKED: '{ctx.command}' by {ctx.author.display_name} in #{ctx.channel.name} (channel restrictions)")
                return
            
            elif isinstance(error, commands.ChannelNotFound):
                logger.warning(f"❌ CHANNEL NOT FOUND: {ctx.author.display_name} used '{ctx.command}' with invalid channel")
                await self.safe_send_error(ctx, "❌ Channel not found. Please mention a valid channel with #channel-name or use the channel ID.")
            
            elif isinstance(error, commands.MemberNotFound):
                logger.warning(f"❌ USER NOT FOUND: {ctx.author.display_name} used '{ctx.command}' with invalid user")
                await self.safe_send_error(ctx, "❌ User not found. Please mention a valid user or use their user ID.")
            
            elif isinstance(error, commands.CommandInvokeError):
                # Handle the underlying error
                original_error = error.original
                command_info = f"'{ctx.command}' by {ctx.author.display_name} in #{ctx.channel.name}"
                
                if isinstance(original_error, discord.Forbidden):
                    # Bot lacks permissions - don't try to send message, just log
                    logger.error(f"🚫 BOT PERMISSION ERROR: {command_info} - {original_error}")
                    logger.error(f"   Error Code: {original_error.code}")
                    logger.error(f"   Error Text: {original_error.text}")
                    try:
                        await ctx.message.add_reaction("❌")
                    except:
                        pass
                
                elif isinstance(original_error, discord.NotFound):
                    logger.warning(f"❌ RESOURCE NOT FOUND: {command_info} - {original_error}")
                    await self.safe_send_error(ctx, "❌ The requested resource was not found.")
                
                else:
                    # Log unexpected errors with full details
                    logger.error(f"💥 UNEXPECTED ERROR: {command_info}")
                    logger.error(f"   Error Type: {type(original_error).__name__}")
                    logger.error(f"   Error Message: {str(original_error)}")
                    logger.error(f"   Command Args: {ctx.args}")
                    logger.error(f"   Command Kwargs: {ctx.kwargs}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    await self.safe_send_error(ctx, "❌ An unexpected error occurred. Please try again later.")
            
            else:
                # Log other unexpected errors
                logger.error(f"🔥 UNHANDLED COMMAND ERROR: '{ctx.command}' by {ctx.author.display_name}")
                logger.error(f"   Error Type: {type(error).__name__}")
                logger.error(f"   Error Details: {str(error)}")
                await self.safe_send_error(ctx, "❌ An error occurred while processing your command.")
        
        except Exception as e:
            # Prevent recursion - just log if error handler fails
            logger.error(f"Error handler failed: {e}")
    
    async def safe_send_error(self, ctx, message):
        """Safely send error message without causing recursion"""
        try:
            await ctx.send(message)
        except discord.Forbidden:
            # Can't send messages - try reaction instead
            try:
                await ctx.message.add_reaction("❌")
            except:
                # Can't even react - just log
                logger.error(f"Cannot respond in #{ctx.channel.name}: {message}")
        except Exception as e:
            # Any other error - just log to prevent recursion
            logger.error(f"Failed to send error message: {e}")
    
    async def handle_missing_permissions(self, ctx):
        """Handle cases where bot lacks permissions to send messages or embeds"""
        try:
            # Try to send a simple message first
            await ctx.send("❌ I don't have permission to send embeds in this channel. Please check my permissions.")
        except discord.Forbidden:
            # Can't even send messages - try to react or log
            try:
                await ctx.message.add_reaction("❌")
            except:
                # Log that we couldn't respond at all
                logger.error(f"Bot has no permissions to respond in #{ctx.channel.name} ({ctx.guild.name})")
    
    async def on_ready(self):
        """Called when bot is ready"""
        if self.user:
            logger.info(f'Bot logged in as {self.user.name} ({self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')
        
        # Check permissions in each guild
        for guild in self.guilds:
            me = guild.me
            if me:
                perms = me.guild_permissions
                missing_perms = []
                
                if not perms.send_messages:
                    missing_perms.append("Send Messages")
                if not perms.embed_links:
                    missing_perms.append("Embed Links")
                if not perms.read_message_history:
                    missing_perms.append("Read Message History")
                if not perms.add_reactions:
                    missing_perms.append("Add Reactions")
                
                if missing_perms:
                    logger.warning(f"Missing permissions in {guild.name}: {', '.join(missing_perms)}")
                else:
                    logger.info(f"All required permissions present in {guild.name}")
        
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
                    value=f"`!!saveuser {user_id}` (Admin only)",
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
                    await ctx.send("❌ No pending saves found. Please specify a user ID or run `!!userhistory` first.")
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
                await ctx.send("❌ No pending saves found. Run `!!userhistory` first.")
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

class TrainingDataCommands(commands.Cog):
    """Commands for training data generation"""
    
    def __init__(self, bot: DeepDiscordBot):
        self.bot = bot
    
    @commands.command(name='generatetrainingdata')
    async def generate_training_data(self, ctx: commands.Context, user_id: int, days_back: int = 30):
        """Generate training data for a specific user
        
        Usage: !generatetrainingdata <user_id> [days_back]
        
        Args:
            user_id: Discord user ID to generate training data for
            days_back: How many days back to analyze (default: 30, max: 365)
        """
        # Log training data generation start
        logger.info(f"🎓 TRAINING DATA GENERATION STARTED")
        logger.info(f"   👤 Requester: {ctx.author.display_name} (ID: {ctx.author.id})")
        logger.info(f"   🎯 Target User ID: {user_id}")
        logger.info(f"   📅 Days Back: {days_back}")
        logger.info(f"   🏠 Guild: {ctx.guild.name} (ID: {ctx.guild.id})")
        logger.info(f"   📺 Channel: #{ctx.channel.name} (ID: {ctx.channel.id})")
        
        # Check if user is authorized
        is_admin = ctx.author.guild_permissions.administrator
        is_authorized = self.bot.authorization_manager.is_authorized(ctx.author.id, is_admin)
        
        logger.info(f"   🔐 Authorization Check: {'✅ Authorized' if is_authorized else '❌ Not Authorized'}")
        logger.info(f"   🛡️ Admin Status: {'✅ Admin' if is_admin else '❌ Not Admin'}")
        
        if not is_authorized:
            logger.warning(f"🎓 TRAINING DATA GENERATION FAILED: User {ctx.author.id} not authorized")
            await ctx.send("❌ You don't have permission to use this command. Only administrators and authorized users can generate training data. Ask an admin to run `!!authorize add @you` to get access.")
            return
        
        # Validate parameters
        logger.info(f"📋 PARAMETER VALIDATION:")
        logger.info(f"   📅 Days back: {days_back} (max: 365, min: 1)")
        
        if days_back > 365:
            logger.warning(f"❌ PARAMETER VALIDATION FAILED: days_back ({days_back}) exceeds maximum (365)")
            await ctx.send("❌ Maximum days back is 365.")
            return
        
        if days_back < 1:
            logger.warning(f"❌ PARAMETER VALIDATION FAILED: days_back ({days_back}) below minimum (1)")
            await ctx.send("❌ Days back must be at least 1.")
            return
        
        logger.info(f"✅ PARAMETER VALIDATION PASSED")
        
        # Check if user exists in this guild
        logger.info(f"👥 USER VALIDATION:")
        logger.info(f"   🔍 Looking up user {user_id} in guild {ctx.guild.name}")
        
        target_member = ctx.guild.get_member(user_id)
        if not target_member:
            logger.info(f"   ⚠️ User not found in cache, trying API fetch...")
            # Try to fetch member from Discord API (in case of cache issues)
            try:
                target_member = await ctx.guild.fetch_member(user_id)
                logger.info(f"   ✅ User found via API fetch: {target_member.display_name}")
            except discord.NotFound:
                logger.warning(f"   ❌ USER NOT FOUND: {user_id} is not a member of guild {ctx.guild.id}")
                logger.info(f"🎓 TRAINING DATA GENERATION FAILED: Target user not found")
                # User not in server, but let's provide helpful debug info
                debug_embed = discord.Embed(
                    title="❌ User Not Found",
                    description=f"User with ID `{user_id}` is not in this server.",
                    color=discord.Color.red()
                )
                debug_embed.add_field(
                    name="Debug Info",
                    value=f"• Server: {ctx.guild.name}\n• Server ID: {ctx.guild.id}\n• Member Count: {ctx.guild.member_count}",
                    inline=False
                )
                debug_embed.add_field(
                    name="Suggestions",
                    value="• Verify the user ID is correct\n• Check if user is in this server\n• Try `!!userhistory @username` to get their ID",
                    inline=False
                )
                await ctx.send(embed=debug_embed)
                return
            except discord.Forbidden:
                logger.error(f"   🚫 PERMISSION ERROR: Bot cannot fetch member {user_id}")
                logger.info(f"🎓 TRAINING DATA GENERATION FAILED: Permission error fetching user")
                await ctx.send(f"❌ Bot doesn't have permission to fetch member `{user_id}`.")
                return
        else:
            logger.info(f"   ✅ User found in cache: {target_member.display_name}")
        
        logger.info(f"   👤 Target User: {target_member.display_name} (ID: {target_member.id})")
        logger.info(f"   ✅ USER VALIDATION PASSED")
        
        # Check if user has given consent
        logger.info(f"🔒 CONSENT VALIDATION:")
        has_consent = self.bot.consent_manager.has_consent(user_id)
        logger.info(f"   🔍 Checking consent for user {user_id}")
        logger.info(f"   {'✅ Consent found' if has_consent else '❌ No consent found'}")
        
        if not has_consent:
            logger.info(f"🔒 REQUESTING USER CONSENT:")
            logger.info(f"   📩 Sending consent request to {target_member.display_name}")
            logger.info(f"🎓 TRAINING DATA GENERATION PAUSED: Awaiting user consent")
            await self.request_user_consent(ctx, target_member, days_back)
            return
        
        # Get consent info for logging
        consent_info = self.bot.consent_manager.get_consent_info(user_id)
        if consent_info:
            logger.info(f"   📋 Consent granted at: {consent_info.get('granted_at', 'Unknown')}")
            logger.info(f"   👤 Granted by request from: {consent_info.get('granted_by_request_from', 'Unknown')}")
            # Note: Consent no longer expires
        
        logger.info(f"   ✅ CONSENT VALIDATION PASSED")
        logger.info(f"🎓 PROCEEDING WITH TRAINING DATA GENERATION")
        
        # Create initial status message
        status_embed = discord.Embed(
            title="🎓 Training Data Generation Started",
            description=f"Generating training data for **{target_member.display_name}**",
            color=discord.Color.blue()
        )
        status_embed.add_field(name="Target User", value=f"{target_member.mention}", inline=True)
        status_embed.add_field(name="Analysis Period", value=f"{days_back} days", inline=True)
        status_embed.add_field(name="Status", value="🔍 Starting analysis...", inline=False)
        
        status_message = await ctx.send(embed=status_embed)
        logger.info(f"📊 Status message sent, beginning data generation process")
        
        try:
            # Generate training data
            logger.info(f"🏭 INITIALIZING TRAINING DATA GENERATOR:")
            logger.info(f"   🎯 Target User: {user_id}")
            logger.info(f"   🏠 Guild: {ctx.guild.name} (ID: {ctx.guild.id})")
            logger.info(f"   📅 Days Back: {days_back}")
            
            generator = DiscordTrainingDataGenerator(
                bot=self.bot,
                target_user_id=user_id,
                guild=ctx.guild,
                status_message=status_message,
                days_back=days_back
            )
            
            logger.info(f"🚀 STARTING TRAINING DATA GENERATION PROCESS")
            zip_result = await generator.generate_training_data(days_back)
            logger.info(f"🏁 TRAINING DATA GENERATION PROCESS COMPLETED")
            
            # Log results
            logger.info(f"📊 TRAINING DATA GENERATION RESULTS:")
            if not zip_result:
                logger.warning(f"   ❌ No training data generated - no suitable response pairs found")
                logger.info(f"🎓 TRAINING DATA GENERATION COMPLETED: No data generated")
                await status_message.edit(embed=discord.Embed(
                    title="❌ No Training Data Generated",
                    description="No suitable response pairs were found for this user.",
                    color=discord.Color.red()
                ))
                return
            
            logger.info(f"   ✅ Generated ZIP file: {zip_result['zip_filename']} ({zip_result['size_mb']:.2f} MB)")
            logger.info(f"   📊 Total training pairs: {zip_result['total_pairs']}")
            logger.info(f"   📁 Files included: {len(zip_result['files_included'])}")
            for file_info in zip_result['files_included']:
                logger.info(f"      📄 {file_info['name']}: {file_info['count']} pairs")
            
            # Update status to show completion
            final_embed = discord.Embed(
                title="✅ Training Data Generation Complete",
                description=f"Generated training data for **{target_member.display_name}**",
                color=discord.Color.green()
            )
            
            # Add summary information to embed
            final_embed.add_field(
                name="📦 Archive Created", 
                value=f"{zip_result['zip_filename']}\n({zip_result['size_mb']:.2f} MB)",
                inline=False
            )
            final_embed.add_field(
                name="📊 Total Training Pairs", 
                value=str(zip_result['total_pairs']),
                inline=True
            )
            final_embed.add_field(
                name="📁 Files Included", 
                value=str(len(zip_result['files_included'])),
                inline=True
            )
            
            # Add breakdown of files
            breakdown_text = "\n".join([f"• {info['name']}: {info['count']} pairs" for info in zip_result['files_included']])
            final_embed.add_field(
                name="📋 Breakdown",
                value=breakdown_text,
                inline=False
            )
            
            await status_message.edit(embed=final_embed)
            
            # Handle ZIP file upload based on size
            logger.info(f"📤 ZIP FILE UPLOAD PROCESS:")
            logger.info(f"   📦 ZIP file size: {zip_result['size_mb']:.2f} MB")
            
            if zip_result['size_mb'] <= 50:  # Direct Discord upload
                logger.info(f"   📤 Uploading ZIP file directly to Discord")
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        with open(zip_result['zip_path'], 'rb') as f:
                            discord_file = discord.File(f, filename=zip_result['zip_filename'])
                            
                        await ctx.send(
                            f"📦 **Training Data Archive for {target_member.display_name}**",
                            file=discord_file
                        )
                        logger.info(f"   ✅ ZIP file uploaded successfully to Discord")
                        break
                        
                    except Exception as upload_error:
                        logger.warning(f"   ⚠️ ZIP upload attempt {attempt + 1} failed: {upload_error}")
                        if attempt < max_retries - 1:
                            logger.info(f"   🔄 Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            logger.error(f"   ❌ All ZIP upload attempts failed, automatically uploading online")
                            # Automatically fall back to online hosting
                            break  # Exit retry loop to trigger online upload below
                
                # If we get here and haven't uploaded to Discord successfully, upload online
                else:
                    # This 'else' clause executes if the loop completed without breaking (successful upload)
                    logger.info(f"   ✅ ZIP file uploaded successfully to Discord")
                    # Skip online upload section
                    upload_successful = True
                
                # If upload failed, try online hosting
                if 'upload_successful' not in locals():
                    logger.info(f"   📤 Uploading to online host as fallback")
                    download_url = await generator.upload_large_file(zip_result['zip_path'], zip_result['zip_filename'])
                    if download_url:
                        online_embed = discord.Embed(
                            title="📤 Training Data - Online Download",
                            description=f"ZIP file upload to Discord failed, but it's available online:",
                            color=discord.Color.blue()
                        )
                        online_embed.add_field(
                            name="🔗 Download Link",
                            value=f"[{zip_result['zip_filename']}]({download_url})",
                            inline=False
                        )
                        online_embed.add_field(
                            name="⚠️ Note",
                            value="File will be available for 14 days. Download it soon!",
                            inline=False
                        )
                        await ctx.send(embed=online_embed)
                        logger.info(f"   ✅ ZIP file uploaded to online host: {download_url}")
                    else:
                        await ctx.send(f"❌ Failed to upload ZIP file to Discord and online. File saved locally as `{zip_result['zip_filename']}` in results directory.")
                        logger.error(f"   ❌ All upload methods failed, file saved locally only")
            else:  # File too large for Discord, upload to online host
                logger.info(f"   📤 ZIP file too large for Discord ({zip_result['size_mb']:.2f} MB > 50 MB), uploading to online host")
                download_url = await generator.upload_large_file(zip_result['zip_path'], zip_result['zip_filename'])
                
                if download_url:
                    large_file_embed = discord.Embed(
                        title="📦 Large Training Data Archive",
                        description=f"Training data archive for **{target_member.display_name}** is too large for Discord upload.",
                        color=discord.Color.blue()
                    )
                    large_file_embed.add_field(
                        name="📊 File Info",
                        value=f"**Size:** {zip_result['size_mb']:.2f} MB\n**Training Pairs:** {zip_result['total_pairs']}\n**Files:** {len(zip_result['files_included'])}",
                        inline=False
                    )
                    large_file_embed.add_field(
                        name="🔗 Download Link",
                        value=f"[{zip_result['zip_filename']}]({download_url})",
                        inline=False
                    )
                    large_file_embed.add_field(
                        name="⚠️ Important",
                        value="File will be available for 14 days. Download it soon!",
                        inline=False
                    )
                    await ctx.send(embed=large_file_embed)
                    logger.info(f"   ✅ Large ZIP file uploaded to online host: {download_url}")
                else:
                    fallback_embed = discord.Embed(
                        title="⚠️ Upload Failed - File Saved Locally",
                        description=f"ZIP file is too large for Discord and online upload failed.",
                        color=discord.Color.orange()
                    )
                    fallback_embed.add_field(
                        name="📁 Local File",
                        value=f"`{zip_result['zip_filename']}`\nSaved in `results/` directory",
                        inline=False
                    )
                    fallback_embed.add_field(
                        name="📊 Archive Contents",
                        value=f"**Training Pairs:** {zip_result['total_pairs']}\n**Files:** {len(zip_result['files_included'])}",
                        inline=False
                    )
                    await ctx.send(embed=fallback_embed)
                    logger.warning(f"   ⚠️ Large ZIP file could not be uploaded online, saved locally")
            
            # Cleanup already handled in generate_training_files method
            logger.info(f"🎓 TRAINING DATA GENERATION COMPLETED SUCCESSFULLY")
            logger.info(f"   👤 User: {target_member.display_name} (ID: {target_member.id})")
            logger.info(f"   📦 ZIP Archive: {zip_result['zip_filename']} ({zip_result['size_mb']:.2f} MB)")
            logger.info(f"   📊 Training Pairs: {zip_result['total_pairs']}")
            logger.info(f"   📁 Files in Archive: {len(zip_result['files_included'])}")
            logger.info(f"   📅 Analysis Period: {days_back} days")
            logger.info(f"   👨‍💼 Requested by: {ctx.author.display_name} (ID: {ctx.author.id})")
        
        except Exception as e:
            logger.error(f"❌ TRAINING DATA GENERATION FAILED: {e}")
            logger.error(f"   👤 User: {target_member.display_name if 'target_member' in locals() else user_id}")
            logger.error(f"   📅 Days Back: {days_back}")
            logger.error(f"   👨‍💼 Requested by: {ctx.author.display_name} (ID: {ctx.author.id})")
            logger.error(f"Error generating training data: {e}")
            error_embed = discord.Embed(
                title="❌ Training Data Generation Failed",
                description=f"An error occurred: {str(e)[:1000]}",
                color=discord.Color.red()
            )
            await status_message.edit(embed=error_embed)
    
    async def request_user_consent(self, ctx: commands.Context, target_member: discord.Member, days_back: int):
        """Request consent from user via DM"""
        try:
            # Create consent request
            request_id = self.bot.consent_manager.create_consent_request(
                target_member.id, 
                ctx.author.id
            )
            
            # Create consent request embed
            consent_embed = discord.Embed(
                title="🔒 Training Data Consent Request",
                description=f"**{ctx.author.display_name}** has requested permission to use your Discord messages for AI training data generation.",
                color=discord.Color.orange()
            )
            
            consent_embed.add_field(
                name="📊 What data would be collected?",
                value=f"• Your messages from the past **{days_back} days**\n• Question/answer pairs from conversations\n• Message timestamps and channel names\n• **No private/deleted messages**",
                inline=False
            )
            
            consent_embed.add_field(
                name="🎯 How will it be used?",
                value="• AI model training only\n• Creating conversational datasets\n• Improving response generation\n• **Data stays within the requesting user**",
                inline=False
            )
            
            consent_embed.add_field(
                name="🛡️ Your rights:",
                value="• You can **revoke consent** anytime\n• Data collection **stops immediately** if revoked\n• **Consent does not expire**\n• You can see exactly what data is collected",
                inline=False
            )
            
            consent_embed.add_field(
                name="📝 Request Details:",
                value=f"**Requester:** {ctx.author.display_name}\n**Server:** {ctx.guild.name}\n**Request ID:** `{request_id}`\n**Days back:** {days_back}",
                inline=False
            )
            
            consent_embed.set_footer(text="React with ✅ to grant consent or ❌ to decline")
            
            # Send DM to target user
            try:
                dm_message = await target_member.send(embed=consent_embed)
                
                # Add reaction buttons
                await dm_message.add_reaction("✅")
                await dm_message.add_reaction("❌")
                
                # Notify requester
                response_embed = discord.Embed(
                    title="📩 Consent Request Sent",
                    description=f"A consent request has been sent to **{target_member.display_name}** via DM.",
                    color=discord.Color.blue()
                )
                response_embed.add_field(
                    name="⏳ Next Steps:",
                    value=f"• User will receive a detailed consent request\n• They can accept (✅) or decline (❌)\n• You'll be notified of their decision\n• **Request ID:** `{request_id}`",
                    inline=False
                )
                
                await ctx.send(embed=response_embed)
                
                # Set up reaction listener
                await self.setup_consent_listener(dm_message, request_id, ctx.author, ctx, target_member, days_back)
                
            except discord.Forbidden:
                # User has DMs disabled
                fallback_embed = discord.Embed(
                    title="❌ Cannot Send DM",
                    description=f"Unable to send consent request to **{target_member.display_name}** - their DMs are disabled.",
                    color=discord.Color.red()
                )
                fallback_embed.add_field(
                    name="Alternative Options:",
                    value=f"• Ask {target_member.mention} to enable DMs temporarily\n• Request consent in a public channel\n• Use `!!consent grant @{target_member.display_name}` (if they agree)",
                    inline=False
                )
                await ctx.send(embed=fallback_embed)
                
        except Exception as e:
            logger.error(f"Error requesting consent: {e}")
            await ctx.send(f"❌ Error sending consent request: {e}")
    
    async def setup_consent_listener(self, dm_message: discord.Message, request_id: str, requester: discord.User, ctx: commands.Context, target_member: discord.Member, days_back: int):
        """Set up listener for consent response"""
        def check(reaction, user):
            return (user.id != self.bot.user.id and 
                   reaction.message.id == dm_message.id and 
                   str(reaction.emoji) in ["✅", "❌"])
        
        try:
            # Wait for reaction (10 minutes timeout)
            reaction, user = await self.bot.wait_for('reaction_add', timeout=600.0, check=check)
            
            granted = str(reaction.emoji) == "✅"
            
            # Complete the request
            self.bot.consent_manager.complete_request(request_id, granted)
            
            # Update the DM message
            if granted:
                success_embed = discord.Embed(
                    title="✅ Consent Granted",
                    description="Thank you! Your consent has been recorded.",
                    color=discord.Color.green()
                )
                success_embed.add_field(
                    name="📋 What happens next:",
                    value="• Training data generation will begin automatically\n• You can revoke consent anytime with `!!consent revoke`\n• **Consent does not expire**",
                    inline=False
                )
                
                # Notify requester and start training
                try:
                    await requester.send(f"✅ **Consent Granted!** User {user.display_name} has approved your training data request (ID: `{request_id}`). Training data generation is starting automatically...")
                except:
                    pass  # Ignore if can't DM requester
                
                # Start training data generation automatically
                await self.start_training_generation(ctx, target_member, days_back)
                    
            else:
                decline_embed = discord.Embed(
                    title="❌ Consent Declined",
                    description="Your decision has been recorded. No data will be collected.",
                    color=discord.Color.red()
                )
                
                # Notify requester
                try:
                    await requester.send(f"❌ **Consent Declined.** User {user.display_name} has declined your training data request (ID: `{request_id}`).")
                except:
                    pass  # Ignore if can't DM requester
            
            await dm_message.edit(embed=success_embed if granted else decline_embed)
            await dm_message.clear_reactions()
            
        except asyncio.TimeoutError:
            # Request timed out
            timeout_embed = discord.Embed(
                title="⏰ Request Expired",
                description="This consent request has timed out. The requester can send a new request if needed.",
                color=discord.Color.orange()
            )
            await dm_message.edit(embed=timeout_embed)
            await dm_message.clear_reactions()
            
            # Notify requester
            try:
                await requester.send(f"⏰ **Request Expired.** The consent request (ID: `{request_id}`) timed out after 10 minutes.")
            except:
                pass
    
    async def start_training_generation(self, ctx: commands.Context, target_member: discord.Member, days_back: int):
        """Start training data generation after consent is granted"""
        try:
            # Send notification in the original channel
            start_embed = discord.Embed(
                title="🎓 Training Data Generation Starting",
                description=f"Consent granted! Beginning training data generation for **{target_member.display_name}**",
                color=discord.Color.green()
            )
            await ctx.send(embed=start_embed)
            
            # Run the actual training generation (copy the logic from the main command)
            status_embed = discord.Embed(
                title="🎓 Training Data Generation Started",
                description=f"Generating training data for **{target_member.display_name}**",
                color=discord.Color.blue()
            )
            status_embed.add_field(name="Target User", value=f"{target_member.mention}", inline=True)
            status_embed.add_field(name="Analysis Period", value=f"{days_back} days", inline=True)
            status_embed.add_field(name="Status", value="🔍 Starting analysis...", inline=False)
            
            status_message = await ctx.send(embed=status_embed)
            
            # Generate training data
            generator = DiscordTrainingDataGenerator(
                bot=self.bot,
                target_user_id=target_member.id,
                guild=ctx.guild,
                status_message=status_message,
                days_back=days_back
            )
            
            zip_result = await generator.generate_training_data(days_back)
            
            if not zip_result:
                await status_message.edit(embed=discord.Embed(
                    title="❌ No Training Data Generated",
                    description="No suitable response pairs were found for this user.",
                    color=discord.Color.red()
                ))
                return
            
            # Update status to show completion
            final_embed = discord.Embed(
                title="✅ Training Data Generation Complete",
                description=f"Generated training data for **{target_member.display_name}**",
                color=discord.Color.green()
            )
            
            # Add summary information
            final_embed.add_field(
                name="📦 Archive Created", 
                value=f"{zip_result['zip_filename']}\n({zip_result['size_mb']:.2f} MB)",
                inline=False
            )
            final_embed.add_field(
                name="📊 Total Training Pairs", 
                value=str(zip_result['total_pairs']),
                inline=True
            )
            final_embed.add_field(
                name="📁 Files Included", 
                value=str(len(zip_result['files_included'])),
                inline=True
            )
            
            await status_message.edit(embed=final_embed)
            
            # Send ZIP file  
            if zip_result['size_mb'] <= 50:  # Try Discord upload first
                discord_upload_successful = False
                try:
                    with open(zip_result['zip_path'], 'rb') as f:
                        discord_file = discord.File(f, filename=zip_result['zip_filename'])
                        
                    await ctx.send(
                        f"📦 **Training Data Archive for {target_member.display_name}**",
                        file=discord_file
                    )
                    discord_upload_successful = True
                except Exception as upload_error:
                    logger.warning(f"Discord upload failed: {upload_error}, trying online hosting")
                
                # If Discord upload failed, automatically try online hosting
                if not discord_upload_successful:
                    download_url = await generator.upload_large_file(zip_result['zip_path'], zip_result['zip_filename'])
                    if download_url:
                        online_embed = discord.Embed(
                            title="📤 Training Data - Online Download",
                            description=f"ZIP file upload to Discord failed, but it's available online:",
                            color=discord.Color.blue()
                        )
                        online_embed.add_field(
                            name="🔗 Download Link",
                            value=f"[{zip_result['zip_filename']}]({download_url})",
                            inline=False
                        )
                        online_embed.add_field(
                            name="⚠️ Note",
                            value="File will be available for 14 days. Download it soon!",
                            inline=False
                        )
                        await ctx.send(embed=online_embed)
                    else:
                        await ctx.send(f"❌ Failed to upload ZIP file to Discord and online. File saved locally as `{zip_result['zip_filename']}`.")
            else:  # File too large, upload to online host
                download_url = await generator.upload_large_file(zip_result['zip_path'], zip_result['zip_filename'])
                if download_url:
                    large_file_embed = discord.Embed(
                        title="📦 Large Training Data Archive",
                        description=f"Training data archive is too large for Discord upload ({zip_result['size_mb']:.2f} MB).",
                        color=discord.Color.blue()
                    )
                    large_file_embed.add_field(
                        name="🔗 Download Link",
                        value=f"[{zip_result['zip_filename']}]({download_url})",
                        inline=False
                    )
                    await ctx.send(embed=large_file_embed)
                else:
                    await ctx.send(f"❌ Failed to upload large ZIP file. File saved locally as `{zip_result['zip_filename']}`.")
            
            # Cleanup already handled in generate_training_files method
        
        except Exception as e:
            logger.error(f"Error in automatic training generation: {e}")
            error_embed = discord.Embed(
                title="❌ Training Data Generation Failed",
                description=f"An error occurred during automatic generation: {str(e)[:1000]}",
                color=discord.Color.red()
            )
            await ctx.send(embed=error_embed)
    
    @commands.command(name='finduser')
    async def find_user(self, ctx: commands.Context, *, search_term: str):
        """Find users in the server by username/display name or user ID
        
        Usage: !finduser <username_or_id>
        
        Args:
            search_term: Username, display name, or user ID to search for
        """
        # Check if search term is a user ID (all digits)
        if search_term.isdigit():
            user_id = int(search_term)
            
            # Direct ID lookup
            member = ctx.guild.get_member(user_id)
            if member:
                embed = discord.Embed(
                    title="✅ User Found by ID",
                    description=f"Found user with ID `{user_id}`",
                    color=discord.Color.green()
                )
                embed.add_field(
                    name=f"{member.display_name}",
                    value=f"**Username:** {member.name}\n**ID:** `{member.id}`\n**Joined:** {member.joined_at.strftime('%Y-%m-%d') if member.joined_at else 'Unknown'}",
                    inline=False
                )
                await ctx.send(embed=embed)
                return
            else:
                # Try API fetch as fallback
                try:
                    member = await ctx.guild.fetch_member(user_id)
                    embed = discord.Embed(
                        title="✅ User Found by ID (API)",
                        description=f"Found user with ID `{user_id}` via API fetch",
                        color=discord.Color.green()
                    )
                    embed.add_field(
                        name=f"{member.display_name}",
                        value=f"**Username:** {member.name}\n**ID:** `{member.id}`\n**Joined:** {member.joined_at.strftime('%Y-%m-%d') if member.joined_at else 'Unknown'}",
                        inline=False
                    )
                    await ctx.send(embed=embed)
                    return
                except discord.NotFound:
                    await ctx.send(f"❌ No user found with ID `{user_id}` in this server")
                    return
                except discord.Forbidden:
                    await ctx.send(f"❌ Bot doesn't have permission to fetch user `{user_id}`")
                    return
        
        # String search through usernames/display names
        search_lower = search_term.lower()
        matches = []
        
        # Search through all members
        for member in ctx.guild.members:
            if (search_lower in member.name.lower() or 
                search_lower in member.display_name.lower()):
                matches.append(member)
        
        if not matches:
            await ctx.send(f"❌ No users found matching '{search_term}'")
            return
        
        # Create embed with results
        embed = discord.Embed(
            title="🔍 User Search Results",
            description=f"Found {len(matches)} user(s) matching '{search_term}'",
            color=discord.Color.blue()
        )
        
        # Show up to 10 matches
        for i, member in enumerate(matches[:10]):
            embed.add_field(
                name=f"{member.display_name}",
                value=f"**Username:** {member.name}\n**ID:** `{member.id}`\n**Joined:** {member.joined_at.strftime('%Y-%m-%d') if member.joined_at else 'Unknown'}",
                inline=True
            )
        
        if len(matches) > 10:
            embed.set_footer(text=f"Showing first 10 of {len(matches)} results")
        
        await ctx.send(embed=embed)
    
    @commands.command(name='serverinfo')
    async def server_info(self, ctx: commands.Context):
        """Show current server information for debugging"""
        embed = discord.Embed(
            title="🏠 Server Information",
            description=f"Current server details",
            color=discord.Color.green()
        )
        
        embed.add_field(name="Server Name", value=ctx.guild.name, inline=True)
        embed.add_field(name="Server ID", value=str(ctx.guild.id), inline=True)
        embed.add_field(name="Member Count", value=str(ctx.guild.member_count), inline=True)
        embed.add_field(name="Bot Permissions", value=f"Read Messages: {ctx.guild.me.guild_permissions.read_messages}\nRead Message History: {ctx.guild.me.guild_permissions.read_message_history}", inline=False)
        
        # Show a few recent members to verify the member cache is working
        recent_members = []
        for member in list(ctx.guild.members)[:5]:
            recent_members.append(f"{member.display_name} ({member.id})")
        
        embed.add_field(name="Sample Members (First 5)", value="\n".join(recent_members), inline=False)
        
        await ctx.send(embed=embed)
    
    @commands.command(name='testmembers')
    async def test_members(self, ctx: commands.Context):
        """Simple test to see if bot can access members"""
        try:
            member_count = len(ctx.guild.members)
            await ctx.send(f"✅ Bot can see {member_count} members in {ctx.guild.name}")
            
            # Show first few members as proof
            first_5 = list(ctx.guild.members)[:5]
            member_list = "\n".join([f"• {m.display_name} ({m.id})" for m in first_5])
            await ctx.send(f"**First 5 members:**\n{member_list}")
            
        except Exception as e:
            await ctx.send(f"❌ Error accessing members: {e}")
    
    @commands.command(name='reload')
    @commands.has_permissions(administrator=True)
    async def reload_bot(self, ctx: commands.Context):
        """Reload bot cogs for testing (Admin only)"""
        try:
            # Remove all cogs
            cogs_to_remove = list(self.bot.cogs.keys())
            for cog_name in cogs_to_remove:
                await self.bot.remove_cog(cog_name)
            
            # Re-add all cogs
            await self.bot.add_cog(MessageCommands(self.bot))
            await self.bot.add_cog(TrainingDataCommands(self.bot))
            await self.bot.add_cog(AdminCommands(self.bot))
            
            await ctx.send("🔄 Bot cogs reloaded successfully!")
            
        except Exception as e:
            await ctx.send(f"❌ Error reloading cogs: {e}")
    
    @commands.command(name='hotreload')
    @commands.has_permissions(administrator=True)
    async def hot_reload(self, ctx: commands.Context):
        """Hot reload the bot module without restarting (Admin only)"""
        try:
            import importlib
            import sys
            
            # Get current module
            current_module = sys.modules[__name__]
            
            # Reload the module
            importlib.reload(current_module)
            
            # Reload cogs
            await self.reload_bot(ctx)
            
            await ctx.send("🔥 Hot reload completed! Module and cogs refreshed.")
            
        except Exception as e:
            await ctx.send(f"❌ Hot reload failed: {e}")
    
    @commands.group(name='consent', invoke_without_command=True)
    async def consent_group(self, ctx: commands.Context):
        """Manage training data consent"""
        if ctx.invoked_subcommand is None:
            # Show user's current consent status
            consent_info = self.bot.consent_manager.get_consent_info(ctx.author.id)
            
            embed = discord.Embed(
                title="🔒 Your Consent Status",
                color=discord.Color.blue()
            )
            
            if not consent_info:
                embed.description = "You have not granted consent for training data collection."
                embed.add_field(
                    name="Available Commands:",
                    value="`!!consent status` - Check detailed status\n`!!consent grant <@user>` - Grant consent to someone\n`!!consent revoke` - Revoke your consent",
                    inline=False
                )
            else:
                status = consent_info.get('status', 'unknown')
                if status == 'granted':
                    embed.description = "✅ You have granted consent for training data collection."
                    embed.color = discord.Color.green()
                    
                    granted_at = consent_info.get('granted_at')
                    if granted_at:
                        embed.add_field(name="Granted At", value=granted_at[:10], inline=True)
                    
                    # Note: Consent no longer expires
                        
                elif status == 'revoked':
                    embed.description = "❌ You have revoked consent for training data collection."
                    embed.color = discord.Color.red()
            
            await ctx.send(embed=embed)
    
    @consent_group.command(name='status')
    async def consent_status(self, ctx: commands.Context, user: discord.Member = None):
        """Check consent status for yourself or another user"""
        target_user = user or ctx.author
        consent_info = self.bot.consent_manager.get_consent_info(target_user.id)
        
        embed = discord.Embed(
            title=f"🔒 Consent Status for {target_user.display_name}",
            color=discord.Color.blue()
        )
        
        if not consent_info:
            embed.description = "No consent record found."
            embed.color = discord.Color.orange()
        else:
            status = consent_info.get('status', 'unknown')
            
            if status == 'granted':
                embed.description = "✅ Consent granted for training data collection"
                embed.color = discord.Color.green()
            elif status == 'revoked':
                embed.description = "❌ Consent revoked"
                embed.color = discord.Color.red()
            
            # Add details
            for key, value in consent_info.items():
                if key != 'status' and key != 'user_id':
                    formatted_key = key.replace('_', ' ').title()
                    if 'at' in key and isinstance(value, str):
                        # Format timestamps
                        try:
                            formatted_value = value[:19].replace('T', ' ')
                        except:
                            formatted_value = value
                    else:
                        formatted_value = str(value)
                    embed.add_field(name=formatted_key, value=formatted_value, inline=True)
        
        await ctx.send(embed=embed)
    
    @consent_group.command(name='grant')
    async def consent_grant(self, ctx: commands.Context, requester: discord.Member):
        """Grant consent to a specific user (use only if they asked you directly)"""
        if ctx.author.id == requester.id:
            await ctx.send("❌ You cannot grant consent to yourself.")
            return
        
        # Check if already granted
        if self.bot.consent_manager.has_consent(ctx.author.id):
            await ctx.send("✅ You have already granted consent. Use `!!consent revoke` to revoke it first if needed.")
            return
        
        # Grant consent
        self.bot.consent_manager.grant_consent(
            ctx.author.id, 
            requester.id
        )
        
        embed = discord.Embed(
            title="✅ Consent Granted",
            description=f"You have granted consent to **{requester.display_name}** for training data collection.",
            color=discord.Color.green()
        )
        embed.add_field(
            name="⚠️ Important:",
            value="• This grants access to your messages for AI training\n• **Consent does not expire**\n• You can revoke anytime with `!!consent revoke`",
            inline=False
        )
        
        await ctx.send(embed=embed)
        
        # Notify the requester
        try:
            await requester.send(f"✅ **Consent Granted!** {ctx.author.display_name} has granted you consent for training data collection. You can now use `!!generatetrainingdata {ctx.author.id}`.")
        except:
            pass  # Ignore if can't DM
    
    @consent_group.command(name='revoke')
    async def consent_revoke(self, ctx: commands.Context):
        """Revoke your consent for training data collection"""
        consent_info = self.bot.consent_manager.get_consent_info(ctx.author.id)
        
        if not consent_info or consent_info.get('status') != 'granted':
            await ctx.send("❌ You have not granted consent, so there's nothing to revoke.")
            return
        
        # Revoke consent
        self.bot.consent_manager.revoke_consent(ctx.author.id)
        
        embed = discord.Embed(
            title="❌ Consent Revoked",
            description="Your consent for training data collection has been revoked.",
            color=discord.Color.red()
        )
        embed.add_field(
            name="What this means:",
            value="• No new data will be collected from your messages\n• Previous consent requests will be denied\n• You can grant consent again later if you choose",
            inline=False
        )
        
        await ctx.send(embed=embed)

class DiscordTrainingDataGenerator:
    """Training data generator integrated with Discord bot"""
    
    def __init__(self, bot: DeepDiscordBot, target_user_id: int, guild: discord.Guild, status_message: discord.Message, days_back: int = 30):
        self.bot = bot
        self.target_user_id = target_user_id
        self.guild = guild
        self.status_message = status_message
        self.days_back = days_back
        self.response_pairs = []
        self.channels_processed = 0
        self.messages_analyzed = 0
        self.update_task = None
        self.is_processing = False
        self.current_status = "🔍 Starting..."
        
    async def periodic_update_task(self):
        """Background task that updates the embed every 5 seconds"""
        while self.is_processing:
            try:
                await asyncio.sleep(5)
                if self.is_processing:  # Check again in case processing finished during sleep
                    await self.update_status(self.current_status)
                    logger.info(f"🔄 PERIODIC UPDATE: Status: {self.current_status} | Messages: {self.messages_analyzed}, Pairs: {len(self.response_pairs)}, Channels: {self.channels_processed}")
            except Exception as e:
                logger.warning(f"Error in periodic update: {e}")
                break
        
    async def generate_training_data(self, days_back: int = 30, min_response_length: int = 10):
        """Generate training data with Discord timeout management"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Start processing flag and periodic update task
        self.is_processing = True
        self.update_task = asyncio.create_task(self.periodic_update_task())
        
        try:
            self.current_status = "🔍 Scanning channels..."
            await self.update_status(self.current_status)
            
            # Get all accessible text channels
            accessible_channels = []
            for channel in self.guild.text_channels:
                try:
                    # Test if we can read the channel
                    await channel.fetch_message(channel.last_message_id) if channel.last_message_id else None
                    accessible_channels.append(channel)
                except (discord.Forbidden, discord.NotFound):
                    continue
            
            self.current_status = f"📋 Found {len(accessible_channels)} accessible channels"
            await self.update_status(self.current_status)
            
            # Process each channel with delays to avoid timeouts
            for i, channel in enumerate(accessible_channels):
                try:
                    self.current_status = f"📝 Processing #{channel.name} ({i+1}/{len(accessible_channels)})"
                    await self.update_status(self.current_status)
                    logger.info(f"📂 PROCESSING CHANNEL {i+1}/{len(accessible_channels)}: #{channel.name} (ID: {channel.id})")
                    await self.process_channel_messages(channel, cutoff_date, min_response_length)
                    self.channels_processed += 1
                    logger.info(f"✅ COMPLETED CHANNEL: #{channel.name} - Total messages analyzed: {self.messages_analyzed}, Training pairs found: {len(self.response_pairs)}")
                    
                    # Add delay between channels to avoid rate limits
                    if i < len(accessible_channels) - 1:  # Don't delay after last channel
                        await asyncio.sleep(2)  # 2 second delay between channels
                        
                except Exception as e:
                    logger.error(f"Error processing channel {channel.name}: {e}")
                    continue
            
            self.current_status = "💾 Generating training files..."
            await self.update_status(self.current_status)
            
            # Generate training files
            # Get target user info for filename
            target_user = self.guild.get_member(self.target_user_id)
            target_username = target_user.display_name if target_user else f"user_{self.target_user_id}"
            
            zip_result = self.generate_training_files(target_username)
            
            return zip_result
            
        finally:
            # Stop the periodic update task
            self.is_processing = False
            if self.update_task and not self.update_task.done():
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass
    
    async def process_channel_messages(self, channel: discord.TextChannel, cutoff_date: datetime, min_response_length: int):
        """Process messages in a channel with chunked retrieval to avoid timeouts"""
        messages = []
        target_responses = 0
        
        # Use chunked message retrieval to avoid timeouts
        chunk_size = 100  # Process in smaller chunks
        current_after = cutoff_date
        
        while True:
            chunk_messages = []
            try:
                # Get a chunk of messages
                async for message in channel.history(limit=chunk_size, after=current_after, oldest_first=True):
                    if not message.author.bot:
                        chunk_messages.append(message)
                        self.messages_analyzed += 1
                        
                        # Log the message being processed
                        content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
                        content_preview = content_preview.replace('\n', ' ')  # Replace newlines for cleaner logs
                        logger.info(f"   📨 Message {self.messages_analyzed}: {message.author.display_name} in #{channel.name}: \"{content_preview}\"")
                        
                        # Add to bot tracker for fragment analysis
                        await self.bot.message_tracker.add_message(message)
                
                if not chunk_messages:
                    break  # No more messages
                
                messages.extend(chunk_messages)
                current_after = chunk_messages[-1].created_at
                
                # Update status every 500 messages
                if self.messages_analyzed % 500 == 0:
                    await self.update_status(f"📊 Analyzed {self.messages_analyzed} messages...")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in chunk processing: {e}")
                break
        
        # Sort messages chronologically for relationship analysis
        messages.sort(key=lambda m: m.created_at)
        
        # Find response pairs for target user
        for i, message in enumerate(messages):
            if message.author.id == self.target_user_id:
                target_responses += 1
                
                # Log target user message
                content_preview = message.content[:80] + "..." if len(message.content) > 80 else message.content
                content_preview = content_preview.replace('\n', ' ')
                logger.info(f"   🎯 Target user message {target_responses}: \"{content_preview}\"")
                
                # Look for what they're responding to
                response_to = await self.find_response_target(message, messages[:i])
                
                if response_to and len(message.content.strip()) >= min_response_length:
                    training_pair = self.create_training_pair(response_to, message)
                    if training_pair:
                        # Log successful training pair creation
                        question_preview = training_pair['question'][:60] + "..." if len(training_pair['question']) > 60 else training_pair['question']
                        answer_preview = training_pair['answer'][:60] + "..." if len(training_pair['answer']) > 60 else training_pair['answer']
                        logger.info(f"   ✅ Training pair #{len(self.response_pairs) + 1}: Q: \"{question_preview}\" A: \"{answer_preview}\" (confidence: {response_to['confidence']:.2f})")
                        
                        self.response_pairs.append(training_pair)
                else:
                    # Log why no training pair was created
                    if not response_to:
                        logger.info(f"   ❌ No response target found for message")
                    elif len(message.content.strip()) < min_response_length:
                        logger.info(f"   ❌ Message too short ({len(message.content.strip())} < {min_response_length} chars)")
    
    async def find_response_target(self, target_message: discord.Message, previous_messages: List[discord.Message]):
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
        recent_messages = [msg for msg in previous_messages[-20:] 
                          if msg.author.id != self.target_user_id 
                          and not msg.author.bot
                          and (target_message.created_at - msg.created_at).total_seconds() <= 900]  # 15 minutes
        
        if recent_messages:
            most_recent = recent_messages[-1]
            time_gap = (target_message.created_at - most_recent.created_at).total_seconds()
            
            # Higher confidence for shorter time gaps
            confidence = max(0.3, 1.0 - (time_gap / 900))
            
            return {
                'type': 'temporal_proximity',
                'message': most_recent,
                'confidence': confidence,
                'time_gap_seconds': time_gap
            }
        
        # Method 3: Content analysis (mentions, keywords)
        for msg in reversed(previous_messages[-30:]):
            if msg.author.id != self.target_user_id and not msg.author.bot:
                if self.has_content_relationship(msg, target_message):
                    return {
                        'type': 'content_analysis',
                        'message': msg,
                        'confidence': 0.6
                    }
        
        return None
    
    def has_content_relationship(self, potential_question: discord.Message, response: discord.Message) -> bool:
        """Check if response content relates to potential question"""
        question_content = potential_question.content.lower()
        response_content = response.content.lower()
        
        # Look for shared keywords (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'that', 'this', 'what', 'when', 'where', 'why', 'how', 'who'}
        
        question_words = set(question_content.split()) - common_words
        response_words = set(response_content.split()) - common_words
        
        if len(question_words) > 0 and len(response_words) > 0:
            shared_words = question_words.intersection(response_words)
            if len(shared_words) >= 2:
                return True
        
        # Check if response mentions the questioner
        if potential_question.author.mention in response.content:
            return True
        
        return False
    
    def clean_unicode_content(self, text: str) -> str:
        """Clean problematic Unicode characters from text"""
        if not text:
            return text
        
        # Replace problematic Unicode escape sequences
        cleaned = text
        
        # Replace guitar emoji escape sequence with actual emoji
        cleaned = cleaned.replace('\\ud83c\\udfb8', '🎸')
        
        # Replace right single quotation mark escape sequence with regular apostrophe  
        cleaned = cleaned.replace('\\u2019', "'")
        
        # Replace other common problematic sequences
        cleaned = cleaned.replace('\\u201c', '"')  # left double quotation mark
        cleaned = cleaned.replace('\\u201d', '"')  # right double quotation mark
        cleaned = cleaned.replace('\\u2018', "'")  # left single quotation mark
        cleaned = cleaned.replace('\\u2026', '...')  # horizontal ellipsis
        
        return cleaned
    
    def create_training_pair(self, response_info: dict, target_response: discord.Message):
        """Create a formatted training pair"""
        question_msg = response_info['message']
        
        # Check if we have consent to include the question author's name
        question_has_consent = self.bot.consent_manager.has_consent(question_msg.author.id)
        answer_has_consent = self.bot.consent_manager.has_consent(target_response.author.id)
        
        # Format the question - include author name only if consent is given
        if question_has_consent:
            question = f"{question_msg.author.display_name}: {self.clean_unicode_content(question_msg.content)}"
            question_author_name = question_msg.author.display_name
        else:
            question = self.clean_unicode_content(question_msg.content)
            question_author_name = "Anonymous"
        
        # Format the answer (get combined content if fragment) and clean Unicode
        answer = self.clean_unicode_content(target_response.content)
        combined_content = self.bot.message_tracker.get_combined_content(target_response.id)
        if combined_content and not combined_content.startswith("[Fragment of"):
            answer = self.clean_unicode_content(combined_content)
        
        # Include answer author name only if consent is given
        answer_author_name = target_response.author.display_name if answer_has_consent else "Anonymous"
        
        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "metadata": {
                "response_type": response_info['type'],
                "confidence": response_info['confidence'],
                "question_author": question_author_name,
                "answer_author": answer_author_name,
                "channel": target_response.channel.name,
                "timestamp": target_response.created_at.isoformat(),
                "question_id": question_msg.id,
                "answer_id": target_response.id,
                "time_gap": response_info.get('time_gap_seconds', 0),
                "question_author_consent": question_has_consent,
                "answer_author_consent": answer_has_consent
            }
        }
    
    def generate_training_files(self, target_username: str = None):
        """Generate training data files and create a zip archive"""
        if not self.response_pairs:
            return None
        
        # Filter by confidence levels
        high_confidence = [pair for pair in self.response_pairs if pair['metadata']['confidence'] >= 0.8]
        medium_confidence = [pair for pair in self.response_pairs if 0.5 <= pair['metadata']['confidence'] < 0.8]
        all_pairs = sorted(self.response_pairs, key=lambda x: x['metadata']['confidence'], reverse=True)
        
        datasets = {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence, 
            "all_responses": all_pairs
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get username for filename
        username = target_username or f"user_{self.target_user_id}"
        # Clean username for filename (remove invalid characters)
        safe_username = "".join(c for c in username if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_username = safe_username.replace(' ', '_')
        
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create temporary directory for individual files
        temp_dir = f"/tmp/training_data_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"📦 Creating training data files for {safe_username}")
        
        # Generate individual JSON files
        json_files = []
        for name, data in datasets.items():
            if not data:  # Skip empty datasets
                continue
                
            filename = f"{name}_{self.target_user_id}_{timestamp}.json"
            filepath = os.path.join(temp_dir, filename)
            
            training_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "target_user_id": self.target_user_id,
                    "target_username": target_username,
                    "total_pairs": len(data),
                    "confidence_threshold": name,
                    "format": "question/answer pairs for training",
                    "channels_processed": self.channels_processed,
                    "messages_analyzed": self.messages_analyzed,
                    "guild_name": self.guild.name if self.guild else "Unknown"
                },
                "training_data": data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            json_files.append({
                'name': f"{name.replace('_', ' ').title()}",
                'filename': filename,
                'path': filepath,
                'count': len(data)
            })
            
            logger.info(f"   📄 Generated {filename}: {len(data)} pairs")
        
        # Create ZIP file
        zip_filename = f"training_data_{safe_username}_{timestamp}.zip"
        zip_path = os.path.join(results_dir, zip_filename)
        
        logger.info(f"📦 Creating ZIP archive: {zip_filename}")
        
        # Create summary file
        summary_data = {
            "generation_summary": {
                "generated_at": datetime.now().isoformat(),
                "target_user_id": self.target_user_id,
                "target_username": target_username,
                "guild_name": self.guild.name if self.guild else "Unknown",
                "total_files": len(json_files),
                "total_training_pairs": len(self.response_pairs),
                "channels_processed": self.channels_processed,
                "messages_analyzed": self.messages_analyzed,
                "confidence_breakdown": {
                    "high_confidence": len(high_confidence),
                    "medium_confidence": len(medium_confidence),
                    "all_responses": len(all_pairs)
                }
            },
            "files_included": [
                {
                    "filename": file_info['filename'],
                    "description": file_info['name'],
                    "training_pairs": file_info['count']
                }
                for file_info in json_files
            ]
        }
        
        summary_path = os.path.join(temp_dir, f"README_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add summary file
            zipf.write(summary_path, f"README_{timestamp}.json")
            
            # Add all JSON files
            for file_info in json_files:
                zipf.write(file_info['path'], file_info['filename'])
        
        # Get ZIP file size
        zip_size = os.path.getsize(zip_path)
        zip_size_mb = zip_size / (1024 * 1024)
        
        logger.info(f"📦 ZIP created: {zip_filename} ({zip_size_mb:.2f} MB)")
        
        # Clean up temporary files
        for file_info in json_files:
            try:
                os.remove(file_info['path'])
            except:
                pass
        try:
            os.remove(summary_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return {
            'zip_path': zip_path,
            'zip_filename': zip_filename,
            'size_mb': zip_size_mb,
            'files_included': json_files,
            'total_pairs': len(self.response_pairs)
        }
    
    async def upload_large_file(self, file_path: str, filename: str) -> str:
        """Upload large file to online host and return download link"""
        logger.info(f"📤 Uploading large file to online host: {filename}")
        
        try:
            # Using file.io (free temporary file hosting)
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f)}
                response = requests.post('https://file.io', files=files, timeout=300)
                
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    download_url = result.get('link')
                    logger.info(f"✅ File uploaded successfully: {download_url}")
                    return download_url
                else:
                    logger.error(f"❌ Upload failed: {result.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"❌ Upload failed with status code: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("❌ Upload timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None
    
    async def update_status(self, status_text: str):
        """Update the status message"""
        try:
            embed = self.status_message.embeds[0]
            
            # Clear all fields and rebuild to prevent duplication
            embed.clear_fields()
            
            # Re-add the core fields
            embed.add_field(name="Target User", value=f"<@{self.target_user_id}>", inline=True)
            embed.add_field(name="Analysis Period", value=f"{self.days_back} days", inline=True)
            embed.add_field(name="Status", value=status_text, inline=False)
            embed.add_field(name="Messages Analyzed", value=str(self.messages_analyzed), inline=True)
            embed.add_field(name="Training Pairs Found", value=str(len(self.response_pairs)), inline=True)
            
            await self.status_message.edit(embed=embed)
        except Exception as e:
            # Log the error but don't break the process
            logger.warning(f"Error updating status embed: {e}")
            pass

class HelpCommands(commands.Cog):
    """Help and information commands"""
    
    def __init__(self, bot: DeepDiscordBot):
        self.bot = bot
    
    async def safe_send_embed(self, ctx, embed):
        """Safely send an embed with fallback to plain text"""
        try:
            await ctx.send(embed=embed)
        except discord.Forbidden:
            # Fallback to plain text
            content = f"**{embed.title}**\n"
            if embed.description:
                content += f"{embed.description}\n\n"
            
            for field in embed.fields:
                content += f"**{field.name}**\n{field.value}\n\n"
            
            if embed.footer:
                content += f"_{embed.footer.text}_"
            
            # Split if too long
            if len(content) > 2000:
                content = content[:1997] + "..."
            
            await ctx.send(content)
    
    @commands.command(name='help')
    async def help_command(self, ctx: commands.Context, category: str = None):
        """Show help information for bot commands"""
        if category:
            category = category.lower()
            if category in ['message', 'messages', 'msg']:
                await self.show_message_help(ctx)
            elif category in ['training', 'data', 'ai']:
                await self.show_training_help(ctx)
            elif category in ['admin', 'administration']:
                await self.show_admin_help(ctx)
            elif category in ['consent', 'privacy']:
                await self.show_consent_help(ctx)
            else:
                await ctx.send(f"❌ Unknown help category: `{category}`. Use `!!help` to see all categories.")
        else:
            await self.show_main_help(ctx)
    
    async def show_main_help(self, ctx: commands.Context):
        """Show main help overview"""
        embed = discord.Embed(
            title="🤖 DeepDiscord Bot Help",
            description="A sophisticated Discord bot for message analysis and AI training data generation",
            color=0x00ff00
        )
        
        embed.add_field(
            name="📋 Command Categories",
            value=(
                "• `!!help messages` - Message analysis and history commands\n"
                "• `!!help training` - AI training data generation commands\n"
                "• `!!help consent` - Privacy and consent management\n"
                "• `!!help admin` - Administrative commands (Admin only)"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🚀 Quick Start",
            value=(
                "• `!!userhistory @user` - Analyze user's message patterns\n"
                "• `!!generatetrainingdata <user_id>` - Generate AI training data\n"
                "• `!!consent` - Check your consent status"
            ),
            inline=False
        )
        
        embed.add_field(
            name="ℹ️ Important Notes",
            value=(
                "• All commands use `!!` prefix to avoid conflicts\n"
                "• Privacy-first: Explicit consent required for data collection\n"
                "• Admin controls: Channel restrictions and user authorization"
            ),
            inline=False
        )
        
        embed.set_footer(text="Use !!help <category> for detailed command information")
        await self.safe_send_embed(ctx, embed)
    
    async def show_message_help(self, ctx: commands.Context):
        """Show message analysis commands help"""
        embed = discord.Embed(
            title="📋 Message Analysis Commands",
            description="Commands for analyzing Discord messages and user patterns",
            color=0x3498db
        )
        
        embed.add_field(
            name="🔍 User Analysis",
            value=(
                "`!!userhistory @user` - Comprehensive user message analysis\n"
                "`!!fragment @user` - Show message fragment detection results\n"
                "`!!relationships @user` - Display message relationship patterns"
            ),
            inline=False
        )
        
        embed.add_field(
            name="💾 Data Management",
            value=(
                "`!!save <user_id>` - Save user message analysis to file\n"
                "`!!yes` - Quick save for the most recent analysis\n"
                "`!!combined <message_id>` - Show combined fragment content"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📊 Features",
            value=(
                "• **Fragment Detection**: Automatically combines rapid-fire messages\n"
                "• **Response Tracking**: Identifies replies and conversation chains\n"
                "• **Pattern Analysis**: Communication habits and activity patterns\n"
                "• **Export Ready**: JSON format for further analysis"
            ),
            inline=False
        )
        
        await self.safe_send_embed(ctx, embed)
    
    async def show_training_help(self, ctx: commands.Context):
        """Show training data commands help"""
        embed = discord.Embed(
            title="🎓 AI Training Data Commands",
            description="Generate training datasets from Discord conversations",
            color=0xe74c3c
        )
        
        embed.add_field(
            name="🤖 Data Generation",
            value=(
                "`!!generatetrainingdata <user_id> [days]` - Generate Q&A training pairs\n"
                "• **user_id**: Discord user ID to analyze\n"
                "• **days**: How many days back to analyze (default: 30, max: 365)"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🔐 Authorization Required",
            value=(
                "• **Administrators**: Always authorized\n"
                f"• **Owner**: <@{self.bot.authorization_manager.owner_user_id}> (always authorized)\n"
                "• **Others**: Must be authorized by admin (`!!authorize add @user`)"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📊 Output Files",
            value=(
                "• **High Confidence**: Response pairs with 80%+ confidence\n"
                "• **Medium Confidence**: Response pairs with 50-80% confidence\n"
                "• **All Responses**: Complete dataset with all detected pairs"
            ),
            inline=False
        )
        
        embed.add_field(
            name="✨ Features",
            value=(
                "• **Smart Response Detection**: Temporal, content, and explicit reply analysis\n"
                "• **Privacy Protection**: Requires explicit user consent\n"
                "• **Fragment Integration**: Includes combined fragmented messages\n"
                "• **Progress Tracking**: Real-time status updates during generation"
            ),
            inline=False
        )
        
        await self.safe_send_embed(ctx, embed)
    
    async def show_consent_help(self, ctx: commands.Context):
        """Show consent and privacy commands help"""
        embed = discord.Embed(
            title="🔒 Privacy & Consent Management",
            description="Control how your Discord data is used for training",
            color=0x9b59b6
        )
        
        embed.add_field(
            name="👤 Personal Consent",
            value=(
                "`!!consent` - Check your current consent status\n"
                "`!!consent revoke` - Withdraw your consent for data collection\n"
                "`!!consent grant @requester` - Grant consent to someone who asked"
            ),
            inline=False
        )
        
        embed.add_field(
            name="👥 Consent Information",
            value=(
                "`!!consent status [@user]` - Check consent status (yours or others)\n"
                "• Shows current status and grant history"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🛡️ Privacy Protections",
            value=(
                "• **Explicit Consent Required**: No data collection without permission\n"
                "• **Permanent**: Consent does not expire\n"
                "• **Revocable**: Can withdraw consent anytime\n"
                "• **Transparent**: Clear audit trail of all activities\n"
                "• **DM Notifications**: Informed of all requests and decisions"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📋 How It Works",
            value=(
                "1. Someone runs `!!generatetrainingdata` for you\n"
                "2. You receive a detailed DM with consent request\n"
                "3. React with ✅ (accept) or ❌ (decline)\n"
                "4. If accepted, training data generation begins automatically"
            ),
            inline=False
        )
        
        await self.safe_send_embed(ctx, embed)
    
    async def show_admin_help(self, ctx: commands.Context):
        """Show admin commands help (only for admins)"""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("❌ Admin help is only available to administrators.")
            return
        
        embed = discord.Embed(
            title="🛡️ Administrative Commands",
            description="Server management and bot configuration (Admin Only)",
            color=0xf39c12
        )
        
        embed.add_field(
            name="👥 User Authorization",
            value=(
                "`!!authorize` - Show authorization management help\n"
                "`!!authorize list` - List all authorized users\n"
                "`!!authorize add @user` - Grant training data access\n"
                "`!!authorize remove @user` - Remove training data access\n"
                "`!!authorize check @user` - Check user's authorization status"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📺 Channel Management",
            value=(
                "`!!channels` - Show channel management help\n"
                "`!!channels list` - Show allowed channels\n"
                "`!!channels add #channel` - Allow bot in specific channel\n"
                "`!!channels remove #channel` - Restrict bot from channel\n"
                "`!!channels clear` - Remove all restrictions\n"
                "`!!channels current` - Check current channel status"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🔧 System Management",
            value=(
                "`!!save` - Save current message data to disk\n"
                "`!!clear` - Clear message cache\n"
                "`!!reload` - Reload bot cogs (hot reload)\n"
                "`!!hotreload` - Full module reload"
            ),
            inline=False
        )
        
        embed.add_field(
            name="⚡ Quick Setup",
            value=(
                "1. **Restrict channels**: `!!channels add #bot-commands`\n"
                "2. **Authorize users**: `!!authorize add @trusted-user`\n"
                "3. **Test setup**: `!!channels current` and `!!authorize list`"
            ),
            inline=False
        )
        
        await self.safe_send_embed(ctx, embed)
    
    @commands.command(name='commands')
    async def list_commands(self, ctx: commands.Context):
        """Show a quick list of all available commands"""
        embed = discord.Embed(
            title="📝 Quick Command Reference",
            description="All available DeepDiscord commands",
            color=0x2ecc71
        )
        
        # Message Commands
        message_cmds = [
            "!!userhistory @user", "!!fragment @user", "!!relationships @user",
            "!!save <user_id>", "!!yes", "!!combined <message_id>"
        ]
        embed.add_field(
            name="📋 Message Analysis",
            value="\n".join(f"• {cmd}" for cmd in message_cmds),
            inline=False
        )
        
        # Training Commands
        training_cmds = ["!!generatetrainingdata <user_id> [days]"]
        embed.add_field(
            name="🎓 Training Data",
            value="\n".join(f"• {cmd}" for cmd in training_cmds),
            inline=False
        )
        
        # Consent Commands
        consent_cmds = [
            "!!consent", "!!consent status [@user]", 
            "!!consent grant @user", "!!consent revoke"
        ]
        embed.add_field(
            name="🔒 Privacy & Consent",
            value="\n".join(f"• {cmd}" for cmd in consent_cmds),
            inline=False
        )
        
        # Admin Commands (only show if admin)
        if ctx.author.guild_permissions.administrator:
            admin_cmds = [
                "!!authorize [add/remove/list/check]", "!!channels [add/remove/list/clear]",
                "!!save", "!!clear", "!!reload", "!!hotreload"
            ]
            embed.add_field(
                name="🛡️ Admin Commands",
                value="\n".join(f"• {cmd}" for cmd in admin_cmds),
                inline=False
            )
        
        # Help Commands
        help_cmds = ["!!help [category]", "!!commands"]
        embed.add_field(
            name="❓ Help & Info",
            value="\n".join(f"• {cmd}" for cmd in help_cmds),
            inline=False
        )
        
        embed.set_footer(text="Use !!help <category> for detailed information about each command group")
        await self.safe_send_embed(ctx, embed)

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
    
    @commands.group(name='authorize')
    @commands.has_permissions(administrator=True)
    async def authorize(self, ctx: commands.Context):
        """Manage user authorization for training data commands (Admin only)"""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="🔐 Authorization Management",
                description="Manage who can use training data commands",
                color=0x00ff00
            )
            embed.add_field(
                name="Commands",
                value=(
                    "• `!!authorize list` - Show authorized users\n"
                    "• `!!authorize add @user` - Authorize a user\n"
                    "• `!!authorize remove @user` - Remove authorization\n"
                    "• `!!authorize check @user` - Check user's authorization status"
                ),
                inline=False
            )
            embed.add_field(
                name="Notes",
                value=(
                    f"• <@{self.bot.authorization_manager.owner_user_id}> is always authorized\n"
                    "• Administrators are always authorized\n"
                    "• Authorization is required for `!!generatetrainingdata`"
                ),
                inline=False
            )
            await ctx.send(embed=embed)
    
    @authorize.command(name='list')
    async def authorize_list(self, ctx: commands.Context):
        """List all authorized users"""
        authorized_users = self.bot.authorization_manager.get_authorized_users()
        
        embed = discord.Embed(
            title="🔐 Authorized Users",
            description=f"Users authorized to use training data commands",
            color=0x00ff00
        )
        
        user_list = []
        for user_id in authorized_users:
            try:
                user = ctx.bot.get_user(user_id) or await ctx.bot.fetch_user(user_id)
                if user_id == self.bot.authorization_manager.owner_user_id:
                    user_list.append(f"👑 {user.display_name} ({user_id}) - Owner")
                else:
                    user_list.append(f"✅ {user.display_name} ({user_id})")
            except:
                user_list.append(f"❓ Unknown User ({user_id})")
        
        if user_list:
            embed.add_field(
                name=f"Authorized Users ({len(user_list)})",
                value="\n".join(user_list),
                inline=False
            )
        else:
            embed.add_field(
                name="No Additional Users",
                value="Only admins and the owner are currently authorized",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @authorize.command(name='add')
    async def authorize_add(self, ctx: commands.Context, user: discord.Member):
        """Authorize a user to use training data commands"""
        if self.bot.authorization_manager.authorize_user(user.id):
            embed = discord.Embed(
                title="✅ User Authorized",
                description=f"{user.display_name} can now use training data commands",
                color=0x00ff00
            )
            embed.add_field(name="User", value=f"{user.mention} ({user.id})", inline=False)
            await ctx.send(embed=embed)
            
            # Notify the user
            try:
                await user.send(f"🎉 You've been authorized to use training data commands in **{ctx.guild.name}**! You can now use `!!generatetrainingdata` and related commands.")
            except:
                pass  # User might have DMs disabled
        else:
            await ctx.send(f"ℹ️ {user.display_name} is already authorized.")
    
    @authorize.command(name='remove')
    async def authorize_remove(self, ctx: commands.Context, user: discord.Member):
        """Remove authorization from a user"""
        if user.id == self.bot.authorization_manager.owner_user_id:
            await ctx.send("❌ Cannot remove authorization from the owner.")
            return
        
        if self.bot.authorization_manager.deauthorize_user(user.id):
            embed = discord.Embed(
                title="🚫 Authorization Removed",
                description=f"{user.display_name} can no longer use training data commands",
                color=0xff6b6b
            )
            embed.add_field(name="User", value=f"{user.mention} ({user.id})", inline=False)
            await ctx.send(embed=embed)
            
            # Notify the user
            try:
                await user.send(f"📢 Your authorization to use training data commands in **{ctx.guild.name}** has been removed.")
            except:
                pass  # User might have DMs disabled
        else:
            await ctx.send(f"ℹ️ {user.display_name} was not authorized.")
    
    @authorize.command(name='check')
    async def authorize_check(self, ctx: commands.Context, user: discord.Member):
        """Check if a user is authorized"""
        is_admin = user.guild_permissions.administrator
        is_authorized = self.bot.authorization_manager.is_authorized(user.id, is_admin)
        
        embed = discord.Embed(
            title=f"🔍 Authorization Status: {user.display_name}",
            color=0x00ff00 if is_authorized else 0xff6b6b
        )
        
        embed.add_field(name="User", value=f"{user.mention} ({user.id})", inline=False)
        embed.add_field(name="Authorized", value="✅ Yes" if is_authorized else "❌ No", inline=True)
        
        reasons = []
        if user.id == self.bot.authorization_manager.owner_user_id:
            reasons.append("👑 Owner")
        if is_admin:
            reasons.append("🛡️ Administrator")
        if user.id in self.bot.authorization_manager.authorized_users:
            reasons.append("📝 Manually authorized")
        
        if reasons:
            embed.add_field(name="Reason", value=" • ".join(reasons), inline=True)
        
        await ctx.send(embed=embed)
    
    @commands.group(name='channels')
    @commands.has_permissions(administrator=True)
    async def channels(self, ctx: commands.Context):
        """Manage which channels the bot can post in (Admin only)"""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="📺 Channel Management",
                description="Control which channels the bot can post in",
                color=0x00ff00
            )
            embed.add_field(
                name="Commands",
                value=(
                    "• `!!channels list` - Show allowed channels\n"
                    "• `!!channels add #channel` - Allow bot to post in channel\n"
                    "• `!!channels remove #channel` - Restrict bot from channel\n"
                    "• `!!channels clear` - Allow bot in all channels (remove restrictions)\n"
                    "• `!!channels current` - Check if current channel is allowed"
                ),
                inline=False
            )
            embed.add_field(
                name="Notes",
                value=(
                    "• By default, bot can post in all channels\n"
                    "• Adding restrictions limits bot to only specified channels\n"
                    "• Bot will silently ignore commands in restricted channels"
                ),
                inline=False
            )
            await ctx.send(embed=embed)
    
    @channels.command(name='list')
    async def channels_list(self, ctx: commands.Context):
        """List all allowed channels for this server"""
        allowed_channels = self.bot.channel_manager.get_allowed_channels(ctx.guild.id)
        
        embed = discord.Embed(
            title="📺 Allowed Channels",
            color=0x00ff00
        )
        
        if not allowed_channels:
            embed.description = "🔓 **No restrictions set** - Bot can post in all channels"
            embed.add_field(
                name="Current Policy",
                value="Bot will respond to commands in any channel",
                inline=False
            )
        else:
            embed.description = f"🔒 **Restricted to {len(allowed_channels)} channels**"
            
            channel_list = []
            for channel_id in allowed_channels:
                try:
                    channel = ctx.guild.get_channel(channel_id)
                    if channel:
                        channel_list.append(f"• {channel.mention} ({channel.name})")
                    else:
                        channel_list.append(f"• ❓ Unknown Channel ({channel_id})")
                except:
                    channel_list.append(f"• ❓ Invalid Channel ({channel_id})")
            
            if channel_list:
                embed.add_field(
                    name="Allowed Channels",
                    value="\n".join(channel_list),
                    inline=False
                )
        
        await ctx.send(embed=embed)
    
    @channels.command(name='add')
    async def channels_add(self, ctx: commands.Context, channel: discord.TextChannel):
        """Allow the bot to post in a specific channel"""
        if self.bot.channel_manager.add_allowed_channel(ctx.guild.id, channel.id):
            embed = discord.Embed(
                title="✅ Channel Added",
                description=f"Bot can now post in {channel.mention}",
                color=0x00ff00
            )
            embed.add_field(name="Channel", value=f"{channel.mention} ({channel.name})", inline=False)
            
            # Show current restriction status
            allowed_count = len(self.bot.channel_manager.get_allowed_channels(ctx.guild.id))
            embed.add_field(
                name="Status", 
                value=f"Bot is now restricted to {allowed_count} specific channel(s)",
                inline=False
            )
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"ℹ️ {channel.mention} is already in the allowed channels list.")
    
    @channels.command(name='remove')
    async def channels_remove(self, ctx: commands.Context, channel: discord.TextChannel):
        """Remove a channel from the allowed list"""
        if self.bot.channel_manager.remove_allowed_channel(ctx.guild.id, channel.id):
            embed = discord.Embed(
                title="🚫 Channel Removed",
                description=f"Bot can no longer post in {channel.mention}",
                color=0xff6b6b
            )
            embed.add_field(name="Channel", value=f"{channel.mention} ({channel.name})", inline=False)
            
            # Show current restriction status
            allowed_channels = self.bot.channel_manager.get_allowed_channels(ctx.guild.id)
            if allowed_channels:
                embed.add_field(
                    name="Status", 
                    value=f"Bot is still restricted to {len(allowed_channels)} channel(s)",
                    inline=False
                )
            else:
                embed.add_field(
                    name="Status", 
                    value="No restrictions - Bot can now post in all channels",
                    inline=False
                )
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"ℹ️ {channel.mention} was not in the allowed channels list.")
    
    @channels.command(name='clear')
    async def channels_clear(self, ctx: commands.Context):
        """Remove all channel restrictions (allow bot in all channels)"""
        if self.bot.channel_manager.clear_allowed_channels(ctx.guild.id):
            embed = discord.Embed(
                title="🔓 Restrictions Cleared",
                description="Bot can now post in all channels",
                color=0x00ff00
            )
            embed.add_field(
                name="Status", 
                value="All channel restrictions have been removed",
                inline=False
            )
            await ctx.send(embed=embed)
        else:
            await ctx.send("ℹ️ No channel restrictions were set.")
    
    @channels.command(name='current')
    async def channels_current(self, ctx: commands.Context):
        """Check if the current channel is allowed"""
        is_allowed = self.bot.channel_manager.is_channel_allowed(ctx.guild.id, ctx.channel.id)
        
        embed = discord.Embed(
            title="🔍 Current Channel Status",
            color=0x00ff00 if is_allowed else 0xff6b6b
        )
        
        embed.add_field(name="Channel", value=f"{ctx.channel.mention} ({ctx.channel.name})", inline=False)
        embed.add_field(name="Status", value="✅ Allowed" if is_allowed else "❌ Restricted", inline=True)
        
        allowed_channels = self.bot.channel_manager.get_allowed_channels(ctx.guild.id)
        if allowed_channels:
            embed.add_field(
                name="Restriction Policy", 
                value=f"Bot limited to {len(allowed_channels)} specific channel(s)",
                inline=True
            )
        else:
            embed.add_field(
                name="Restriction Policy", 
                value="No restrictions (all channels allowed)",
                inline=True
            )
        
        await ctx.send(embed=embed)

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
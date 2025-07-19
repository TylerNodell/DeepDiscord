"""
Discord-specific data preprocessing for multi-personality training.
Handles personality-based formatting, consent checking, and Discord content cleaning.
"""

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

from ..config.personality_config import PersonalityConfig, PersonalityStrategy

logger = logging.getLogger(__name__)


@dataclass
class DiscordMessage:
    """Represents a Discord message with metadata."""
    content: str
    username: str
    display_name: str
    user_id: Optional[str] = None
    channel: Optional[str] = None
    timestamp: Optional[str] = None
    message_id: Optional[str] = None
    confidence: float = 1.0
    has_consent: bool = False


@dataclass
class ConversationPair:
    """Represents a conversation pair for training."""
    input_text: str
    output_text: str
    personality: Optional[str] = None
    channel: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict = None


class DiscordPreprocessor:
    """
    Preprocesses Discord training data with personality-based formatting.
    """
    
    def __init__(self, personality_config: PersonalityConfig):
        self.config = personality_config
        self.consent_cache: Dict[str, bool] = {}
        
        # Discord-specific patterns
        self.mention_pattern = re.compile(r'<@!?(\d+)>')
        self.channel_pattern = re.compile(r'<#(\d+)>')
        self.emoji_pattern = re.compile(r'<a?:\w+:\d+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Content cleaning patterns
        self.unicode_replacements = {
            '\ud83c\udfb8': '🎸',
            '\u2019': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2014': '--',
            '\u2026': '...',
        }
    
    def load_consent_data(self, consent_file: Path) -> Dict[str, bool]:
        """Load user consent data from file."""
        try:
            if consent_file.exists():
                with open(consent_file, 'r') as f:
                    consent_data = json.load(f)
                    return {str(user_id): data.get('has_consent', False) 
                           for user_id, data in consent_data.items()}
            return {}
        except Exception as e:
            logger.warning(f"Failed to load consent data: {e}")
            return {}
    
    def clean_discord_content(self, content: str) -> str:
        """Clean Discord-specific content and formatting."""
        # Clean Unicode escape sequences
        for old, new in self.unicode_replacements.items():
            content = content.replace(old, new)
        
        # Remove or replace Discord mentions
        content = self.mention_pattern.sub('@user', content)
        
        # Remove or replace channel mentions
        content = self.channel_pattern.sub('#channel', content)
        
        # Keep emojis for personality but clean malformed ones
        content = re.sub(r'<:[^:]+:broken>', '', content)
        
        # Optional URL removal
        if hasattr(self.config, 'remove_urls') and self.config.remove_urls:
            content = self.url_pattern.sub('[URL]', content)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def is_quality_message(self, message: DiscordMessage) -> bool:
        """Check if message meets quality thresholds."""
        if not message.content or len(message.content.strip()) < 10:
            return False
        
        # Check personality-specific quality threshold
        personality = self.config.get_personality(message.username)
        if personality:
            return message.confidence >= personality.quality_threshold
        
        # Default quality check
        return message.confidence >= 0.7
    
    def format_for_personality(self, message: DiscordMessage, context_messages: List[DiscordMessage] = None) -> str:
        """Format message according to personality strategy."""
        
        if self.config.strategy == PersonalityStrategy.UNIFIED:
            # Simple unified approach - just return cleaned content
            return self.clean_discord_content(message.content)
        
        elif self.config.strategy == PersonalityStrategy.INSTRUCTION_BASED:
            # Add personality instruction using user ID lookup
            personality = None
            if message.user_id:
                personality = self.config.get_personality_by_user_id(message.user_id)
            
            # Use personality name if available, otherwise use Discord username
            name = personality.personality_name if personality else (message.display_name or message.username)
            
            if self.config.channel_context and message.channel:
                formatted = self.config.channel_template.format(
                    channel=message.channel,
                    personality_name=name,
                    message=self.clean_discord_content(message.content)
                )
            else:
                formatted = self.config.instruction_template.format(
                    personality_name=name,
                    message=self.clean_discord_content(message.content)
                )
            
            return formatted
        
        elif self.config.strategy == PersonalityStrategy.MULTIPLE_LORA:
            # For multiple LoRA, use personality tags based on user ID
            personality = None
            if message.user_id:
                personality = self.config.get_personality_by_user_id(message.user_id)
            
            if personality:
                tag = f"[PERSONALITY:{personality.personality_name.upper()}]"
                return f"{tag} {self.clean_discord_content(message.content)}"
            else:
                return self.clean_discord_content(message.content)
        
        return self.clean_discord_content(message.content)
    
    def create_conversation_pairs(self, messages: List[DiscordMessage]) -> List[ConversationPair]:
        """Create conversation pairs from Discord messages."""
        pairs = []
        
        # Group messages by conversation context (simplified)
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            # Skip if either message doesn't meet quality standards
            if not self.is_quality_message(current_msg) or not self.is_quality_message(next_msg):
                continue
            
            # Skip if consent is required but not given
            if self.config.require_consent:
                if not current_msg.has_consent or not next_msg.has_consent:
                    continue
            
            # Create conversation pair
            input_text = self.format_for_personality(current_msg)
            output_text = self.format_for_personality(next_msg)
            
            # Determine personality for the response using user ID
            response_personality = None
            if self.config.strategy != PersonalityStrategy.UNIFIED and next_msg.user_id:
                personality = self.config.get_personality_by_user_id(next_msg.user_id)
                if personality:
                    response_personality = personality.personality_name
            
            pair = ConversationPair(
                input_text=input_text,
                output_text=output_text,
                personality=response_personality,
                channel=next_msg.channel,
                confidence=min(current_msg.confidence, next_msg.confidence),
                metadata={
                    'input_user': current_msg.username,
                    'output_user': next_msg.username,
                    'timestamp': next_msg.timestamp,
                    'message_id': next_msg.message_id
                }
            )
            
            pairs.append(pair)
        
        return pairs
    
    def balance_personality_samples(self, pairs: List[ConversationPair]) -> List[ConversationPair]:
        """Balance samples across personalities according to config."""
        if not self.config.balance_samples or self.config.strategy == PersonalityStrategy.UNIFIED:
            return pairs
        
        # Group pairs by personality
        personality_groups: Dict[str, List[ConversationPair]] = {}
        unassigned = []
        
        for pair in pairs:
            if pair.personality:
                if pair.personality not in personality_groups:
                    personality_groups[pair.personality] = []
                personality_groups[pair.personality].append(pair)
            else:
                unassigned.append(pair)
        
        # Balance according to personality limits
        balanced_pairs = []
        
        for personality_name, group_pairs in personality_groups.items():
            personality = self.config.get_personality_by_name(personality_name)
            if not personality:
                balanced_pairs.extend(group_pairs)
                continue
            
            # Sort by confidence and take best samples
            group_pairs.sort(key=lambda x: x.confidence, reverse=True)
            
            # Respect min/max sample limits
            if len(group_pairs) < personality.min_samples:
                logger.warning(f"Personality '{personality_name}' has only {len(group_pairs)} samples, "
                             f"minimum is {personality.min_samples}")
                balanced_pairs.extend(group_pairs)
            elif len(group_pairs) > personality.max_samples:
                logger.info(f"Limiting personality '{personality_name}' to {personality.max_samples} samples "
                           f"(had {len(group_pairs)})")
                balanced_pairs.extend(group_pairs[:personality.max_samples])
            else:
                balanced_pairs.extend(group_pairs)
        
        # Add unassigned pairs
        balanced_pairs.extend(unassigned)
        
        return balanced_pairs
    
    def convert_to_chatml(self, pairs: List[ConversationPair]) -> List[Dict]:
        """Convert conversation pairs to ChatML format for training."""
        chatml_data = []
        
        for pair in pairs:
            if self.config.strategy == PersonalityStrategy.INSTRUCTION_BASED:
                # For instruction-based, the personality info is already in the text
                messages = [
                    {"role": "user", "content": pair.input_text},
                    {"role": "assistant", "content": pair.output_text}
                ]
            else:
                # For other strategies, use system message for personality
                system_content = "You are a helpful Discord bot assistant."
                if pair.personality:
                    personality = self.config.get_personality_by_name(pair.personality)
                    if personality and personality.description:
                        system_content = f"You are {personality.personality_name}. {personality.description}"
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": pair.input_text},
                    {"role": "assistant", "content": pair.output_text}
                ]
            
            chatml_entry = {
                "messages": messages,
                "metadata": {
                    "personality": pair.personality,
                    "channel": pair.channel,
                    "confidence": pair.confidence,
                    **pair.metadata
                }
            }
            
            chatml_data.append(chatml_entry)
        
        return chatml_data
    
    def process_training_zip(self, zip_path: Path, consent_file: Path = None) -> List[Dict]:
        """Process a training data ZIP file and return formatted data."""
        logger.info(f"Processing training ZIP: {zip_path}")
        
        # Load consent data
        if consent_file and consent_file.exists():
            self.consent_cache = self.load_consent_data(consent_file)
            logger.info(f"Loaded consent data for {len(self.consent_cache)} users")
        
        messages = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.json'):
                        with zip_ref.open(file_name) as file:
                            try:
                                data = json.load(file)
                                file_messages = self._parse_training_data(data, file_name)
                                messages.extend(file_messages)
                                logger.info(f"Parsed {len(file_messages)} messages from {file_name}")
                            except Exception as e:
                                logger.error(f"Failed to parse {file_name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to process ZIP file {zip_path}: {e}")
            return []
        
        logger.info(f"Total messages parsed: {len(messages)}")
        
        # Create conversation pairs
        pairs = self.create_conversation_pairs(messages)
        logger.info(f"Created {len(pairs)} conversation pairs")
        
        # Balance samples across personalities
        balanced_pairs = self.balance_personality_samples(pairs)
        logger.info(f"Balanced to {len(balanced_pairs)} pairs across personalities")
        
        # Convert to ChatML format
        chatml_data = self.convert_to_chatml(balanced_pairs)
        logger.info(f"Converted to ChatML format: {len(chatml_data)} entries")
        
        return chatml_data
    
    def _parse_training_data(self, data: Dict, source_file: str) -> List[DiscordMessage]:
        """Parse training data from JSON structure."""
        messages = []
        
        try:
            # Handle different JSON structures
            if 'conversations' in data:
                conversations = data['conversations']
            elif isinstance(data, list):
                conversations = data
            else:
                conversations = [data]
            
            for conv in conversations:
                if 'messages' in conv:
                    for msg_data in conv['messages']:
                        message = self._create_discord_message(msg_data, source_file)
                        if message:
                            messages.append(message)
                elif 'user_message' in conv and 'ai_response' in conv:
                    # Handle pair format
                    user_msg = self._create_discord_message(conv['user_message'], source_file)
                    ai_msg = self._create_discord_message(conv['ai_response'], source_file)
                    if user_msg:
                        messages.append(user_msg)
                    if ai_msg:
                        messages.append(ai_msg)
        
        except Exception as e:
            logger.error(f"Failed to parse messages from {source_file}: {e}")
        
        return messages
    
    def _create_discord_message(self, msg_data: Dict, source: str) -> Optional[DiscordMessage]:
        """Create DiscordMessage from parsed data."""
        try:
            # Extract basic fields
            content = msg_data.get('content') or msg_data.get('message', '')
            username = msg_data.get('username') or msg_data.get('user', '')
            
            if not content or not username:
                return None
            
            # Check consent
            user_id = msg_data.get('user_id')
            has_consent = True  # Default to True for backward compatibility
            
            if self.config.require_consent and user_id:
                has_consent = self.consent_cache.get(str(user_id), False)
                if not has_consent and self.config.exclude_anonymous:
                    return None
            
            # Anonymize if no consent
            display_name = username
            if not has_consent:
                display_name = "Anonymous"
                username = "Anonymous"
            
            message = DiscordMessage(
                content=content,
                username=username,
                display_name=display_name,
                user_id=user_id,
                channel=msg_data.get('channel'),
                timestamp=msg_data.get('timestamp'),
                message_id=msg_data.get('message_id'),
                confidence=msg_data.get('confidence', 1.0),
                has_consent=has_consent
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to create message from {source}: {e}")
            return None
    
    def get_preprocessing_stats(self, chatml_data: List[Dict]) -> Dict:
        """Get statistics about the preprocessing results."""
        personality_counts = {}
        channel_counts = {}
        total_chars = 0
        
        for entry in chatml_data:
            # Count by personality
            personality = entry['metadata'].get('personality', 'Unknown')
            personality_counts[personality] = personality_counts.get(personality, 0) + 1
            
            # Count by channel
            channel = entry['metadata'].get('channel', 'Unknown')
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
            
            # Count characters
            for message in entry['messages']:
                if message['role'] in ['user', 'assistant']:
                    total_chars += len(message['content'])
        
        return {
            'total_entries': len(chatml_data),
            'personality_distribution': personality_counts,
            'channel_distribution': channel_counts,
            'total_characters': total_chars,
            'average_chars_per_entry': total_chars / len(chatml_data) if chatml_data else 0
        }
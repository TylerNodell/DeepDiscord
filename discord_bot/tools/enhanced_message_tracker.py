#!/usr/bin/env python3
"""
Enhanced Message Tracker for Discord
Demonstrates advanced methods for detecting message relationships and identifying standalone messages.
"""

import discord
import re
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Set
from difflib import SequenceMatcher
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EnhancedMessageTracker:
    """
    Enhanced message tracker that uses multiple methods to detect message relationships
    and identify messages with no initiator.
    """
    
    def __init__(self, max_cache_size: int = 10000):
        self.message_cache: Dict[int, discord.Message] = {}
        self.response_chain: Dict[int, List[int]] = {}  # message_id -> list of response_ids
        self.implicit_responses: Dict[int, List[int]] = {}  # message_id -> list of implicit response_ids
        self.conversation_segments: List[List[int]] = []
        self.user_interaction_patterns: Dict[int, Dict] = {}
        self.max_cache_size = max_cache_size
        
        # Response detection patterns
        self.response_indicators = {
            'agreement': ['yes', 'agree', 'exactly', 'that\'s right', 'correct', 'true', 'absolutely'],
            'disagreement': ['no', 'disagree', 'wrong', 'that\'s wrong', 'incorrect', 'false', 'nope'],
            'clarification': ['what do you mean', 'can you explain', 'i don\'t understand', 'clarify'],
            'continuation': ['and', 'also', 'furthermore', 'moreover', 'additionally', 'besides'],
            'contrast': ['but', 'however', 'though', 'although', 'nevertheless', 'on the other hand'],
            'question': ['what', 'when', 'where', 'why', 'how', 'who', 'which'],
            'acknowledgment': ['ok', 'okay', 'got it', 'understood', 'i see', 'makes sense']
        }
        
        # Quote patterns for detecting quoted responses
        self.quote_patterns = [
            r'^>.*$',  # Discord quote format
            r'^".*"',  # Quoted text
            r'^.*said.*:',  # "X said:" patterns
            r'^.*wrote.*:',  # "X wrote:" patterns
        ]
    
    def add_message(self, message: discord.Message):
        """Add a message to the tracker and analyze relationships"""
        # Manage cache size
        if len(self.message_cache) >= self.max_cache_size:
            oldest_id = next(iter(self.message_cache))
            del self.message_cache[oldest_id]
        
        self.message_cache[message.id] = message
        
        # Track explicit replies (Discord's built-in reference system)
        self._track_explicit_replies(message)
        
        # Track implicit responses (content-based analysis)
        self._track_implicit_responses(message)
        
        # Update conversation segments
        self._update_conversation_segments(message)
        
        # Update user interaction patterns
        self._update_user_patterns(message)
    
    def _track_explicit_replies(self, message: discord.Message):
        """Track messages that explicitly reply to others using Discord's reference system"""
        if message.reference and message.reference.message_id:
            referenced_id = message.reference.message_id
            if referenced_id not in self.response_chain:
                self.response_chain[referenced_id] = []
            self.response_chain[referenced_id].append(message.id)
            logger.info(f"Explicit reply detected: {message.id} -> {referenced_id}")
    
    def _track_implicit_responses(self, message: discord.Message):
        """Track messages that implicitly respond to recent messages"""
        # Get recent messages in the same channel
        recent_messages = self._get_recent_messages_in_channel(message.channel.id, limit=20)
        
        for recent_msg in recent_messages:
            if recent_msg.id == message.id:
                continue
            
            # Skip if this is already an explicit reply to this message
            if message.reference and message.reference.message_id == recent_msg.id:
                continue
            
            # Check for implicit response indicators
            if self._is_implicit_response(message, recent_msg):
                if recent_msg.id not in self.implicit_responses:
                    self.implicit_responses[recent_msg.id] = []
                self.implicit_responses[recent_msg.id].append(message.id)
                logger.info(f"Implicit response detected: {message.id} -> {recent_msg.id}")
    
    def _is_implicit_response(self, message: discord.Message, target_message: discord.Message) -> bool:
        """Check if a message is an implicit response to another message"""
        content_lower = message.content.lower()
        target_content_lower = target_message.content.lower()
        
        # Check for response indicators
        for category, indicators in self.response_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    # Additional context check - is this likely responding to the target?
                    if self._has_contextual_relevance(message, target_message):
                        return True
        
        # Check for quote patterns
        for pattern in self.quote_patterns:
            if re.search(pattern, message.content, re.MULTILINE):
                # Check if the quoted content matches the target message
                if self._extract_quoted_content(message.content) in target_content_lower:
                    return True
        
        # Check for mention of the target message's author
        if f'<@{target_message.author.id}>' in message.content:
            return True
        
        # Check for content similarity
        similarity = SequenceMatcher(None, content_lower, target_content_lower).ratio()
        if similarity > 0.3:  # Threshold for similarity
            return True
        
        # Check for temporal proximity (within 5 minutes)
        time_diff = abs((message.created_at - target_message.created_at).total_seconds())
        if time_diff < 300:  # 5 minutes
            # Additional check for conversation continuity
            if self._is_conversation_continuation(message, target_message):
                return True
        
        return False
    
    def _has_contextual_relevance(self, message: discord.Message, target_message: discord.Message) -> bool:
        """Check if a message has contextual relevance to another message"""
        # Check for shared keywords
        message_words = set(message.content.lower().split())
        target_words = set(target_message.content.lower().split())
        shared_words = message_words.intersection(target_words)
        
        # If they share significant words, they're likely related
        if len(shared_words) >= 2:
            return True
        
        # Check for topic continuity
        if self._is_same_topic(message.content, target_message.content):
            return True
        
        return False
    
    def _is_same_topic(self, content1: str, content2: str) -> bool:
        """Check if two messages are about the same topic"""
        # Simple topic detection based on common subjects
        topics = {
            'gaming': ['game', 'play', 'win', 'lose', 'score', 'level', 'player'],
            'music': ['song', 'music', 'artist', 'album', 'listen', 'playlist'],
            'movies': ['movie', 'film', 'watch', 'actor', 'director', 'scene'],
            'food': ['food', 'eat', 'cook', 'recipe', 'restaurant', 'meal'],
            'work': ['work', 'job', 'project', 'meeting', 'deadline', 'client'],
            'school': ['class', 'homework', 'exam', 'study', 'teacher', 'assignment']
        }
        
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        for topic, keywords in topics.items():
            topic1_match = any(keyword in content1_lower for keyword in keywords)
            topic2_match = any(keyword in content2_lower for keyword in keywords)
            
            if topic1_match and topic2_match:
                return True
        
        return False
    
    def _extract_quoted_content(self, content: str) -> str:
        """Extract quoted content from a message"""
        # Remove Discord quote formatting
        lines = content.split('\n')
        quoted_lines = []
        
        for line in lines:
            if line.startswith('>'):
                quoted_lines.append(line[1:].strip())
            elif line.startswith('"') and line.endswith('"'):
                quoted_lines.append(line[1:-1].strip())
        
        return ' '.join(quoted_lines)
    
    def _is_conversation_continuation(self, message: discord.Message, target_message: discord.Message) -> bool:
        """Check if a message continues a conversation"""
        # Check if messages are from different users (more likely to be responses)
        if message.author.id != target_message.author.id:
            return True
        
        # Check for conversation flow indicators
        flow_indicators = ['and', 'also', 'but', 'however', 'so', 'then', 'next']
        content_lower = message.content.lower()
        
        for indicator in flow_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def _get_recent_messages_in_channel(self, channel_id: int, limit: int = 20) -> List[discord.Message]:
        """Get recent messages in a specific channel"""
        recent_messages = []
        for msg in self.message_cache.values():
            if msg.channel.id == channel_id:
                recent_messages.append(msg)
        
        # Sort by timestamp and return the most recent
        recent_messages.sort(key=lambda x: x.created_at, reverse=True)
        return recent_messages[:limit]
    
    def _update_conversation_segments(self, message: discord.Message):
        """Update conversation segments based on the new message"""
        # This is a simplified implementation
        # In a full implementation, you'd want more sophisticated conversation segmentation
        
        # Check if this message starts a new conversation segment
        if self._is_conversation_initiator(message):
            # Start new segment
            self.conversation_segments.append([message.id])
        else:
            # Add to existing segment or create new one
            if self.conversation_segments:
                self.conversation_segments[-1].append(message.id)
            else:
                self.conversation_segments.append([message.id])
    
    def _is_conversation_initiator(self, message: discord.Message) -> bool:
        """Check if a message initiates a new conversation"""
        # A message is an initiator if:
        # 1. It has no explicit reply
        # 2. It has no implicit responses to recent messages
        # 3. It's not a response to recent conversation
        
        # Check for explicit reply
        if message.reference and message.reference.message_id:
            return False
        
        # Check for implicit responses
        recent_messages = self._get_recent_messages_in_channel(message.channel.id, limit=10)
        for recent_msg in recent_messages:
            if self._is_implicit_response(message, recent_msg):
                return False
        
        # Check for conversation continuity
        if recent_messages:
            last_message = recent_messages[0]
            time_diff = abs((message.created_at - last_message.created_at).total_seconds())
            
            # If messages are close in time and related, this might not be an initiator
            if time_diff < 600:  # 10 minutes
                if self._has_contextual_relevance(message, last_message):
                    return False
        
        return True
    
    def _update_user_patterns(self, message: discord.Message):
        """Update user interaction patterns"""
        user_id = message.author.id
        
        if user_id not in self.user_interaction_patterns:
            self.user_interaction_patterns[user_id] = {
                'message_count': 0,
                'response_count': 0,
                'initiator_count': 0,
                'frequent_responders': defaultdict(int),
                'response_style': 'mixed'
            }
        
        patterns = self.user_interaction_patterns[user_id]
        patterns['message_count'] += 1
        
        # Check if this is a response
        if message.reference and message.reference.message_id:
            patterns['response_count'] += 1
            # Track who they respond to
            referenced_msg = self.message_cache.get(message.reference.message_id)
            if referenced_msg:
                patterns['frequent_responders'][referenced_msg.author.id] += 1
        
        # Check if this is an initiator
        if self._is_conversation_initiator(message):
            patterns['initiator_count'] += 1
    
    def get_message_relationships(self, message_id: int) -> Dict:
        """Get comprehensive relationship information for a message"""
        message = self.message_cache.get(message_id)
        if not message:
            return {}
        
        relationships = {
            'message_id': message_id,
            'explicit_replies': self.response_chain.get(message_id, []),
            'implicit_responses': self.implicit_responses.get(message_id, []),
            'total_responses': len(self.response_chain.get(message_id, [])) + len(self.implicit_responses.get(message_id, [])),
            'is_initiator': self._is_conversation_initiator(message),
            'conversation_segment': self._get_conversation_segment(message_id)
        }
        
        return relationships
    
    def _get_conversation_segment(self, message_id: int) -> Optional[List[int]]:
        """Get the conversation segment containing a message"""
        for segment in self.conversation_segments:
            if message_id in segment:
                return segment
        return None
    
    def find_standalone_messages(self, channel_id: Optional[int] = None, time_period: int = 86400) -> List[discord.Message]:
        """Find messages that have no initiator (standalone messages)"""
        standalone_messages = []
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_period)
        
        for message in self.message_cache.values():
            # Filter by channel if specified
            if channel_id and message.channel.id != channel_id:
                continue
            
            # Filter by time
            if message.created_at < cutoff_time:
                continue
            
            # Check if this is a standalone message
            if self._is_conversation_initiator(message):
                # Additional check: no recent messages that this could be responding to
                recent_messages = self._get_recent_messages_in_channel(message.channel.id, limit=5)
                is_truly_standalone = True
                
                for recent_msg in recent_messages:
                    if recent_msg.id == message.id:
                        continue
                    
                    if self._has_contextual_relevance(message, recent_msg):
                        is_truly_standalone = False
                        break
                
                if is_truly_standalone:
                    standalone_messages.append(message)
        
        return standalone_messages
    
    def get_conversation_flow(self, channel_id: int, time_window: int = 3600) -> List[Dict]:
        """Analyze conversation flow in a channel"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        channel_messages = []
        
        # Get messages in the channel within the time window
        for message in self.message_cache.values():
            if message.channel.id == channel_id and message.created_at >= cutoff_time:
                channel_messages.append(message)
        
        # Sort by timestamp
        channel_messages.sort(key=lambda x: x.created_at)
        
        # Group into conversation segments
        segments = []
        current_segment = []
        
        for message in channel_messages:
            if not current_segment:
                current_segment.append(message)
            else:
                last_message = current_segment[-1]
                time_diff = (message.created_at - last_message.created_at).total_seconds()
                
                # If messages are close in time and related, add to segment
                if time_diff < 300 and self._has_contextual_relevance(message, last_message):
                    current_segment.append(message)
                else:
                    # Start new segment
                    if current_segment:
                        segments.append({
                            'messages': current_segment,
                            'initiator': current_segment[0],
                            'duration': (current_segment[-1].created_at - current_segment[0].created_at).total_seconds(),
                            'participant_count': len(set(msg.author.id for msg in current_segment))
                        })
                    current_segment = [message]
        
        # Add final segment
        if current_segment:
            segments.append({
                'messages': current_segment,
                'initiator': current_segment[0],
                'duration': (current_segment[-1].created_at - current_segment[0].created_at).total_seconds(),
                'participant_count': len(set(msg.author.id for msg in current_segment))
            })
        
        return segments
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about message relationships"""
        total_messages = len(self.message_cache)
        explicit_replies = sum(len(responses) for responses in self.response_chain.values())
        implicit_responses = sum(len(responses) for responses in self.implicit_responses.values())
        
        # Count standalone messages
        standalone_count = len(self.find_standalone_messages())
        
        # Calculate response rates
        response_rate = (explicit_replies + implicit_responses) / total_messages if total_messages > 0 else 0
        standalone_rate = standalone_count / total_messages if total_messages > 0 else 0
        
        return {
            'total_messages': total_messages,
            'explicit_replies': explicit_replies,
            'implicit_responses': implicit_responses,
            'total_responses': explicit_replies + implicit_responses,
            'standalone_messages': standalone_count,
            'response_rate': response_rate,
            'standalone_rate': standalone_rate,
            'conversation_segments': len(self.conversation_segments),
            'active_users': len(self.user_interaction_patterns)
        }

# Example usage and testing
async def test_enhanced_tracker():
    """Test the enhanced message tracker with sample data"""
    tracker = EnhancedMessageTracker()
    
    # This would be used in a real Discord bot
    # For testing, we'll create mock messages
    print("Enhanced Message Tracker Test")
    print("=" * 40)
    
    # Example statistics
    stats = tracker.get_statistics()
    print(f"Tracker Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nThis tracker provides:")
    print("1. Explicit reply detection (Discord's reference system)")
    print("2. Implicit response detection (content-based analysis)")
    print("3. Conversation segmentation")
    print("4. Standalone message identification")
    print("5. User interaction pattern analysis")
    print("6. Comprehensive relationship mapping")

if __name__ == "__main__":
    asyncio.run(test_enhanced_tracker())
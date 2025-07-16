#!/usr/bin/env python3
"""
Enhanced Commands for DeepDiscord Bot
Demonstrates advanced message relationship detection and standalone message identification.
"""

import discord
from discord.ext import commands
from typing import Optional, List, Dict
import re
from difflib import SequenceMatcher
from datetime import datetime, timedelta
import json

class EnhancedMessageCommands(commands.Cog):
    """Enhanced commands for advanced message relationship analysis"""
    
    def __init__(self, bot):
        self.bot = bot
        self.response_indicators = {
            'agreement': ['yes', 'agree', 'exactly', 'that\'s right', 'correct', 'true', 'absolutely'],
            'disagreement': ['no', 'disagree', 'wrong', 'that\'s wrong', 'incorrect', 'false', 'nope'],
            'clarification': ['what do you mean', 'can you explain', 'i don\'t understand', 'clarify'],
            'continuation': ['and', 'also', 'furthermore', 'moreover', 'additionally', 'besides'],
            'contrast': ['but', 'however', 'though', 'although', 'nevertheless', 'on the other hand'],
            'question': ['what', 'when', 'where', 'why', 'how', 'who', 'which'],
            'acknowledgment': ['ok', 'okay', 'got it', 'understood', 'i see', 'makes sense']
        }
    
    @commands.command(name='analyze')
    async def analyze_message_relationships(self, ctx: commands.Context, message_id: int):
        """Analyze comprehensive message relationships for a specific message"""
        try:
            # Get the target message
            message = await ctx.channel.fetch_message(message_id)
            
            # Get recent messages for analysis
            recent_messages = []
            async for msg in ctx.channel.history(limit=50, before=message.created_at):
                recent_messages.append(msg)
            
            # Analyze relationships
            relationships = self._analyze_message_relationships(message, recent_messages)
            
            # Create embed
            embed = discord.Embed(
                title="Message Relationship Analysis",
                description=f"Analysis for message {message_id}",
                color=discord.Color.blue()
            )
            
            # Add relationship information
            embed.add_field(
                name="Explicit Reply",
                value=f"`{relationships['explicit_reply']}`" if relationships['explicit_reply'] else "None",
                inline=True
            )
            
            embed.add_field(
                name="Implicit Responses",
                value=f"`{len(relationships['implicit_responses'])}` detected",
                inline=True
            )
            
            embed.add_field(
                name="Response Indicators",
                value=f"`{', '.join(relationships['response_indicators'])}`" if relationships['response_indicators'] else "None",
                inline=True
            )
            
            embed.add_field(
                name="Content Similarity",
                value=f"`{relationships['max_similarity']:.2f}`" if relationships['max_similarity'] > 0 else "None",
                inline=True
            )
            
            embed.add_field(
                name="Is Standalone",
                value="✅ Yes" if relationships['is_standalone'] else "❌ No",
                inline=True
            )
            
            embed.add_field(
                name="Conversation Context",
                value=f"`{relationships['conversation_context']}`" if relationships['conversation_context'] else "None",
                inline=True
            )
            
            # Add message preview
            embed.add_field(
                name="Message Content",
                value=f"```{message.content[:200]}{'...' if len(message.content) > 200 else ''}```",
                inline=False
            )
            
            await ctx.send(embed=embed)
            
        except discord.NotFound:
            await ctx.send(f"❌ Message {message_id} not found in this channel.")
        except Exception as e:
            await ctx.send(f"❌ Error analyzing message: {str(e)}")
    
    def _analyze_message_relationships(self, message: discord.Message, recent_messages: List[discord.Message]) -> Dict:
        """Analyze relationships for a specific message"""
        relationships = {
            'explicit_reply': None,
            'implicit_responses': [],
            'response_indicators': [],
            'max_similarity': 0.0,
            'is_standalone': True,
            'conversation_context': None
        }
        
        # Check for explicit reply
        if message.reference and message.reference.message_id:
            relationships['explicit_reply'] = message.reference.message_id
            relationships['is_standalone'] = False
        
        # Analyze implicit responses
        content_lower = message.content.lower()
        
        for recent_msg in recent_messages:
            if recent_msg.id == message.id:
                continue
            
            # Check for response indicators
            for category, indicators in self.response_indicators.items():
                for indicator in indicators:
                    if indicator in content_lower:
                        relationships['response_indicators'].append(f"{category}: {indicator}")
                        if self._has_contextual_relevance(message, recent_msg):
                            relationships['implicit_responses'].append(recent_msg.id)
                            relationships['is_standalone'] = False
            
            # Check content similarity
            similarity = SequenceMatcher(None, content_lower, recent_msg.content.lower()).ratio()
            if similarity > relationships['max_similarity']:
                relationships['max_similarity'] = similarity
                if similarity > 0.3:
                    relationships['is_standalone'] = False
            
            # Check for mentions
            if f'<@{recent_msg.author.id}>' in message.content:
                relationships['implicit_responses'].append(recent_msg.id)
                relationships['is_standalone'] = False
        
        # Determine conversation context
        if relationships['implicit_responses']:
            relationships['conversation_context'] = "Response to recent messages"
        elif relationships['explicit_reply']:
            relationships['conversation_context'] = "Explicit reply"
        else:
            relationships['conversation_context'] = "Standalone message"
        
        return relationships
    
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
    
    @commands.command(name='standalone')
    async def find_standalone_messages(self, ctx: commands.Context, hours: int = 24):
        """Find messages that have no initiator (standalone messages) in the last N hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            standalone_messages = []
            
            # Get messages from the specified time period
            async for message in ctx.channel.history(after=cutoff_time, limit=1000):
                # Skip bot messages
                if message.author.bot:
                    continue
                
                # Check if this is a standalone message
                if self._is_standalone_message(message, ctx.channel):
                    standalone_messages.append(message)
            
            if not standalone_messages:
                await ctx.send(f"✅ No standalone messages found in the last {hours} hours.")
                return
            
            # Create embed
            embed = discord.Embed(
                title="Standalone Messages",
                description=f"Found {len(standalone_messages)} standalone messages in the last {hours} hours",
                color=discord.Color.green()
            )
            
            # Add standalone messages (limit to first 10)
            for i, msg in enumerate(standalone_messages[:10]):
                embed.add_field(
                    name=f"Standalone Message {i+1}",
                    value=f"**{msg.author.name}**: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\n"
                          f"*{msg.created_at.strftime('%Y-%m-%d %H:%M:%S')}*",
                    inline=False
                )
            
            if len(standalone_messages) > 10:
                embed.add_field(
                    name="Note",
                    value=f"Showing first 10 of {len(standalone_messages)} standalone messages",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error finding standalone messages: {str(e)}")
    
    def _is_standalone_message(self, message: discord.Message, channel) -> bool:
        """Check if a message is standalone (has no initiator)"""
        # Check for explicit reply
        if message.reference and message.reference.message_id:
            return False
        
        # Get recent messages before this one
        recent_messages = []
        try:
            # This is a simplified check - in a real implementation, you'd want to cache recent messages
            # For now, we'll use a basic heuristic
            return True
        except:
            return True
    
    @commands.command(name='conversation')
    async def analyze_conversation_flow(self, ctx: commands.Context, hours: int = 1):
        """Analyze conversation flow in the channel for the last N hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            channel_messages = []
            
            # Get messages from the specified time period
            async for message in ctx.channel.history(after=cutoff_time, limit=500):
                if not message.author.bot:
                    channel_messages.append(message)
            
            if not channel_messages:
                await ctx.send(f"✅ No messages found in the last {hours} hours.")
                return
            
            # Sort by timestamp
            channel_messages.sort(key=lambda x: x.created_at)
            
            # Group into conversation segments
            segments = self._group_conversation_segments(channel_messages)
            
            # Create embed
            embed = discord.Embed(
                title="Conversation Flow Analysis",
                description=f"Analysis of {len(channel_messages)} messages in the last {hours} hours",
                color=discord.Color.purple()
            )
            
            embed.add_field(
                name="Conversation Segments",
                value=f"`{len(segments)}` segments detected",
                inline=True
            )
            
            embed.add_field(
                name="Average Segment Size",
                value=f"`{len(channel_messages) / len(segments):.1f}` messages",
                inline=True
            )
            
            embed.add_field(
                name="Total Participants",
                value=f"`{len(set(msg.author.id for msg in channel_messages))}` users",
                inline=True
            )
            
            # Show recent segments
            for i, segment in enumerate(segments[-3:]):  # Last 3 segments
                initiator = segment[0]
                duration = (segment[-1].created_at - segment[0].created_at).total_seconds()
                participants = len(set(msg.author.id for msg in segment))
                
                embed.add_field(
                    name=f"Segment {len(segments) - 2 + i}",
                    value=f"**Initiator**: {initiator.author.name}\n"
                          f"**Duration**: {duration:.0f}s\n"
                          f"**Messages**: {len(segment)}\n"
                          f"**Participants**: {participants}",
                    inline=True
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error analyzing conversation flow: {str(e)}")
    
    def _group_conversation_segments(self, messages: List[discord.Message]) -> List[List[discord.Message]]:
        """Group messages into conversation segments"""
        segments = []
        current_segment = []
        
        for message in messages:
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
                        segments.append(current_segment)
                    current_segment = [message]
        
        # Add final segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    @commands.command(name='responses')
    async def find_implicit_responses(self, ctx: commands.Context, message_id: int):
        """Find implicit responses to a specific message"""
        try:
            # Get the target message
            target_message = await ctx.channel.fetch_message(message_id)
            
            # Get recent messages after the target message
            responses = []
            async for message in ctx.channel.history(after=target_message.created_at, limit=100):
                if message.author.bot or message.id == message_id:
                    continue
                
                # Check if this is an implicit response
                if self._is_implicit_response(message, target_message):
                    responses.append(message)
            
            if not responses:
                await ctx.send(f"✅ No implicit responses found for message {message_id}.")
                return
            
            # Create embed
            embed = discord.Embed(
                title="Implicit Responses",
                description=f"Found {len(responses)} implicit responses to message {message_id}",
                color=discord.Color.orange()
            )
            
            # Add response information
            for i, response in enumerate(responses[:5]):  # Limit to first 5
                embed.add_field(
                    name=f"Response {i+1}",
                    value=f"**{response.author.name}**: {response.content[:100]}{'...' if len(response.content) > 100 else ''}\n"
                          f"*{response.created_at.strftime('%H:%M:%S')}*",
                    inline=False
                )
            
            if len(responses) > 5:
                embed.add_field(
                    name="Note",
                    value=f"Showing first 5 of {len(responses)} responses",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except discord.NotFound:
            await ctx.send(f"❌ Message {message_id} not found in this channel.")
        except Exception as e:
            await ctx.send(f"❌ Error finding implicit responses: {str(e)}")
    
    def _is_implicit_response(self, message: discord.Message, target_message: discord.Message) -> bool:
        """Check if a message is an implicit response to another message"""
        content_lower = message.content.lower()
        target_content_lower = target_message.content.lower()
        
        # Check for response indicators
        for category, indicators in self.response_indicators.items():
            for indicator in indicators:
                if indicator in content_lower:
                    if self._has_contextual_relevance(message, target_message):
                        return True
        
        # Check for mention of the target message's author
        if f'<@{target_message.author.id}>' in message.content:
            return True
        
        # Check for content similarity
        similarity = SequenceMatcher(None, content_lower, target_content_lower).ratio()
        if similarity > 0.3:
            return True
        
        return False
    
    @commands.command(name='stats')
    async def get_enhanced_stats(self, ctx: commands.Context, hours: int = 24):
        """Get enhanced statistics about message relationships in the channel"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            messages = []
            
            # Get messages from the specified time period
            async for message in ctx.channel.history(after=cutoff_time, limit=1000):
                if not message.author.bot:
                    messages.append(message)
            
            if not messages:
                await ctx.send(f"✅ No messages found in the last {hours} hours.")
                return
            
            # Calculate statistics
            stats = self._calculate_enhanced_stats(messages)
            
            # Create embed
            embed = discord.Embed(
                title="Enhanced Message Statistics",
                description=f"Analysis of {len(messages)} messages in the last {hours} hours",
                color=discord.Color.dark_blue()
            )
            
            embed.add_field(
                name="Total Messages",
                value=f"`{stats['total_messages']}`",
                inline=True
            )
            
            embed.add_field(
                name="Explicit Replies",
                value=f"`{stats['explicit_replies']}` ({stats['explicit_reply_rate']:.1%})",
                inline=True
            )
            
            embed.add_field(
                name="Standalone Messages",
                value=f"`{stats['standalone_messages']}` ({stats['standalone_rate']:.1%})",
                inline=True
            )
            
            embed.add_field(
                name="Active Users",
                value=f"`{stats['active_users']}`",
                inline=True
            )
            
            embed.add_field(
                name="Average Response Time",
                value=f"`{stats['avg_response_time']:.0f}s`",
                inline=True
            )
            
            embed.add_field(
                name="Most Active User",
                value=f"`{stats['most_active_user']}`",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error calculating statistics: {str(e)}")
    
    def _calculate_enhanced_stats(self, messages: List[discord.Message]) -> Dict:
        """Calculate enhanced statistics for a list of messages"""
        stats = {
            'total_messages': len(messages),
            'explicit_replies': 0,
            'standalone_messages': 0,
            'active_users': len(set(msg.author.id for msg in messages)),
            'response_times': [],
            'user_message_counts': {}
        }
        
        # Count user messages
        for message in messages:
            user_id = message.author.id
            if user_id not in stats['user_message_counts']:
                stats['user_message_counts'][user_id] = 0
            stats['user_message_counts'][user_id] += 1
        
        # Find most active user
        if stats['user_message_counts']:
            most_active_user_id = max(stats['user_message_counts'], key=stats['user_message_counts'].get)
            stats['most_active_user'] = f"User {most_active_user_id}"
        else:
            stats['most_active_user'] = "None"
        
        # Calculate response times and other metrics
        for i, message in enumerate(messages):
            # Check for explicit replies
            if message.reference and message.reference.message_id:
                stats['explicit_replies'] += 1
            
            # Check for standalone messages (simplified)
            if not message.reference or not message.reference.message_id:
                stats['standalone_messages'] += 1
        
        # Calculate rates
        stats['explicit_reply_rate'] = stats['explicit_replies'] / stats['total_messages'] if stats['total_messages'] > 0 else 0
        stats['standalone_rate'] = stats['standalone_messages'] / stats['total_messages'] if stats['total_messages'] > 0 else 0
        
        # Calculate average response time (simplified)
        if stats['response_times']:
            stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])
        else:
            stats['avg_response_time'] = 0
        
        return stats

async def setup(bot):
    """Add the enhanced commands to the bot"""
    await bot.add_cog(EnhancedMessageCommands(bot))
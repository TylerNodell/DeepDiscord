"""
Data preprocessing and augmentation for training data.
"""

import re
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Text cleaning
    clean_unicode: bool = True
    normalize_whitespace: bool = True
    remove_urls: bool = True
    remove_mentions: bool = False  # Keep @mentions for Discord context
    remove_emojis: bool = False    # Keep emojis for Discord context
    
    # Length filtering
    min_length: int = 5
    max_length: int = 500
    min_words: int = 2
    max_words: int = 100
    
    # Quality filtering
    min_alpha_ratio: float = 0.3
    max_repetition_ratio: float = 0.7
    filter_non_english: bool = False
    
    # Context handling
    include_context: bool = True
    max_context_turns: int = 3
    context_separator: str = " [SEP] "


class DataPreprocessor:
    """Data preprocessor for conversation data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logger
    
    def preprocess_conversations(self, conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Preprocess a list of conversations.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            List of preprocessed conversations
        """
        self.logger.info(f"🔄 Preprocessing {len(conversations)} conversations")
        
        processed = []
        filtered_count = 0
        
        for i, conv in enumerate(conversations):
            if i % 1000 == 0 and i > 0:
                self.logger.info(f"   📊 Processed {i}/{len(conversations)} conversations")
            
            try:
                # Clean and validate
                cleaned_conv = self._clean_conversation(conv)
                
                if self._should_filter_conversation(cleaned_conv):
                    filtered_count += 1
                    continue
                
                processed.append(cleaned_conv)
                
            except Exception as e:
                self.logger.warning(f"Error processing conversation {i}: {e}")
                filtered_count += 1
                continue
        
        self.logger.info(f"✅ Preprocessing complete: {len(processed)} valid, {filtered_count} filtered")
        return processed
    
    def _clean_conversation(self, conversation: Dict[str, str]) -> Dict[str, str]:
        """Clean a single conversation."""
        cleaned = {}
        
        for key in ["input", "output"]:
            if key in conversation:
                text = conversation[key]
                cleaned[key] = self._clean_text(text)
            else:
                cleaned[key] = ""
        
        # Preserve other fields
        for key, value in conversation.items():
            if key not in ["input", "output"]:
                cleaned[key] = value
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text."""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Unicode cleaning
        if self.config.clean_unicode:
            text = self._clean_unicode(text)
        
        # URL removal
        if self.config.remove_urls:
            text = self._remove_urls(text)
        
        # Mention removal (optional for Discord)
        if self.config.remove_mentions:
            text = self._remove_mentions(text)
        
        # Emoji removal (optional for Discord)
        if self.config.remove_emojis:
            text = self._remove_emojis(text)
        
        # Whitespace normalization
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _clean_unicode(self, text: str) -> str:
        """Clean problematic Unicode characters."""
        # Replace common problematic escape sequences
        replacements = {
            '\\ud83c\\udfb8': '🎸',
            '\\u2019': "'",
            '\\u201c': '"',
            '\\u201d': '"',
            '\\u2013': '-',
            '\\u2014': '--',
            '\\u2026': '...',
            '\\u00a0': ' ',  # Non-breaking space
            '\\n': '\n',
            '\\t': '\t',
            '\\r': '\r'
        }
        
        for escape, replacement in replacements.items():
            text = text.replace(escape, replacement)
        
        # Remove other escape sequences
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
        text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # HTTP/HTTPS URLs
        text = re.sub(r'https?://\S+', '', text)
        # www URLs
        text = re.sub(r'www\.\S+', '', text)
        # Discord CDN links
        text = re.sub(r'cdn\.discordapp\.com/\S+', '', text)
        
        return text
    
    def _remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        # Discord mentions (@username or <@userid>)
        text = re.sub(r'<@!?\d+>', '', text)
        text = re.sub(r'@\w+', '', text)
        
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        # Remove Unicode emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove Discord custom emojis
        text = re.sub(r'<:[^:]+:\d+>', '', text)
        text = re.sub(r'<a:[^:]+:\d+>', '', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _should_filter_conversation(self, conversation: Dict[str, str]) -> bool:
        """Check if conversation should be filtered out."""
        input_text = conversation.get("input", "")
        output_text = conversation.get("output", "")
        
        # Length filtering
        if len(input_text) < self.config.min_length or len(input_text) > self.config.max_length:
            return True
        if len(output_text) < self.config.min_length or len(output_text) > self.config.max_length:
            return True
        
        # Word count filtering
        input_words = len(input_text.split())
        output_words = len(output_text.split())
        
        if input_words < self.config.min_words or input_words > self.config.max_words:
            return True
        if output_words < self.config.min_words or output_words > self.config.max_words:
            return True
        
        # Quality filtering
        if self._is_low_quality(input_text) or self._is_low_quality(output_text):
            return True
        
        # Language filtering (basic)
        if self.config.filter_non_english:
            if not self._is_likely_english(input_text) or not self._is_likely_english(output_text):
                return True
        
        return False
    
    def _is_low_quality(self, text: str) -> bool:
        """Check if text is low quality."""
        if not text.strip():
            return True
        
        # Alpha ratio check
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / len(text) if text else 0
        
        if alpha_ratio < self.config.min_alpha_ratio:
            return True
        
        # Repetition check
        words = text.split()
        if len(words) > 2:
            unique_words = len(set(words))
            repetition_ratio = 1 - (unique_words / len(words))
            
            if repetition_ratio > self.config.max_repetition_ratio:
                return True
        
        # Check for bot-like responses
        bot_phrases = [
            "i don't understand",
            "i can't help",
            "please try again",
            "error occurred",
            "something went wrong",
            "not available",
            "cannot process"
        ]
        
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in bot_phrases):
            return True
        
        return False
    
    def _is_likely_english(self, text: str) -> bool:
        """Basic check for English text."""
        # Count ASCII alphabetic characters
        ascii_alpha = sum(c.isascii() and c.isalpha() for c in text)
        total_alpha = sum(c.isalpha() for c in text)
        
        if total_alpha == 0:
            return False
        
        ascii_ratio = ascii_alpha / total_alpha
        return ascii_ratio > 0.8  # 80% ASCII characters
    
    def create_conversational_context(
        self, 
        conversations: List[Dict[str, str]], 
        context_window: int = None
    ) -> List[Dict[str, str]]:
        """
        Create conversational context by grouping related conversations.
        
        Args:
            conversations: List of individual conversations
            context_window: Number of previous conversations to include as context
            
        Returns:
            List of conversations with context
        """
        if not self.config.include_context:
            return conversations
        
        context_window = context_window or self.config.max_context_turns
        
        self.logger.info(f"🔗 Creating conversational context (window={context_window})")
        
        contextualized = []
        
        for i, conv in enumerate(conversations):
            # Get previous conversations as context
            start_idx = max(0, i - context_window)
            context_convs = conversations[start_idx:i]
            
            if context_convs:
                # Build context string
                context_parts = []
                for ctx_conv in context_convs:
                    ctx_input = ctx_conv.get("input", "")
                    ctx_output = ctx_conv.get("output", "")
                    if ctx_input and ctx_output:
                        context_parts.append(f"{ctx_input}{self.config.context_separator}{ctx_output}")
                
                if context_parts:
                    context_str = self.config.context_separator.join(context_parts)
                    contextualized_input = f"{context_str}{self.config.context_separator}{conv['input']}"
                else:
                    contextualized_input = conv["input"]
            else:
                contextualized_input = conv["input"]
            
            contextualized_conv = {
                "input": contextualized_input,
                "output": conv["output"]
            }
            
            # Preserve other fields
            for key, value in conv.items():
                if key not in ["input", "output"]:
                    contextualized_conv[key] = value
            
            contextualized.append(contextualized_conv)
        
        self.logger.info(f"✅ Created context for {len(contextualized)} conversations")
        return contextualized
    
    def balance_dataset(self, conversations: List[Dict[str, str]], max_per_length_bucket: int = 1000) -> List[Dict[str, str]]:
        """
        Balance dataset by length to prevent bias toward short/long responses.
        
        Args:
            conversations: List of conversations
            max_per_length_bucket: Maximum conversations per length bucket
            
        Returns:
            Balanced list of conversations
        """
        self.logger.info(f"⚖️  Balancing dataset (max_per_bucket={max_per_length_bucket})")
        
        # Create length buckets
        buckets = {
            "very_short": [],    # < 20 chars
            "short": [],         # 20-50 chars
            "medium": [],        # 50-150 chars
            "long": [],          # 150-300 chars
            "very_long": []      # > 300 chars
        }
        
        for conv in conversations:
            output_length = len(conv.get("output", ""))
            
            if output_length < 20:
                bucket = "very_short"
            elif output_length < 50:
                bucket = "short"
            elif output_length < 150:
                bucket = "medium"
            elif output_length < 300:
                bucket = "long"
            else:
                bucket = "very_long"
            
            buckets[bucket].append(conv)
        
        # Log bucket sizes
        for bucket, convs in buckets.items():
            self.logger.info(f"   {bucket}: {len(convs)} conversations")
        
        # Sample from each bucket
        balanced = []
        for bucket, convs in buckets.items():
            if len(convs) > max_per_length_bucket:
                sampled = random.sample(convs, max_per_length_bucket)
                self.logger.info(f"   Sampled {len(sampled)} from {bucket}")
            else:
                sampled = convs
            
            balanced.extend(sampled)
        
        # Shuffle the balanced dataset
        random.shuffle(balanced)
        
        self.logger.info(f"✅ Balanced dataset: {len(balanced)} conversations")
        return balanced
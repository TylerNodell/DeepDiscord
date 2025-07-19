"""
Data augmentation techniques for conversational training data.
"""

import random
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # Augmentation probabilities
    synonym_replacement_prob: float = 0.1
    random_insertion_prob: float = 0.05
    random_swap_prob: float = 0.05
    random_deletion_prob: float = 0.05
    
    # Augmentation parameters
    num_augmented_per_original: int = 1
    max_augmentations_per_sentence: int = 2
    preserve_special_tokens: bool = True
    
    # Paraphrasing
    enable_paraphrasing: bool = False
    paraphrase_prob: float = 0.1
    
    # Context manipulation
    enable_context_manipulation: bool = True
    context_shuffle_prob: float = 0.1
    context_truncate_prob: float = 0.1


class DataAugmentor:
    """Data augmentation for conversation training data."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logger
        
        # Load synonym dictionary (simplified)
        self.synonyms = self._load_simple_synonyms()
        
        # Common Discord words to preserve
        self.preserve_words = {
            'discord', 'bot', 'server', 'channel', 'dm', 'user', 'admin',
            'mod', 'moderator', 'voice', 'text', 'ping', 'mention', 'role'
        }
    
    def augment_conversations(self, conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Augment a list of conversations.
        
        Args:
            conversations: List of original conversations
            
        Returns:
            List including original and augmented conversations
        """
        self.logger.info(f"🔄 Augmenting {len(conversations)} conversations")
        
        augmented_conversations = conversations.copy()
        
        for i, conv in enumerate(conversations):
            if i % 500 == 0 and i > 0:
                self.logger.info(f"   📊 Processed {i}/{len(conversations)} conversations")
            
            try:
                # Generate augmented versions
                for _ in range(self.config.num_augmented_per_original):
                    augmented_conv = self._augment_conversation(conv)
                    if augmented_conv and augmented_conv != conv:
                        augmented_conversations.append(augmented_conv)
            
            except Exception as e:
                self.logger.warning(f"Error augmenting conversation {i}: {e}")
                continue
        
        original_count = len(conversations)
        total_count = len(augmented_conversations)
        augmented_count = total_count - original_count
        
        self.logger.info(f"✅ Augmentation complete: {original_count} original + {augmented_count} augmented = {total_count} total")
        
        return augmented_conversations
    
    def _augment_conversation(self, conversation: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Augment a single conversation."""
        augmented = conversation.copy()
        
        # Choose random augmentation techniques
        techniques_applied = 0
        max_techniques = self.config.max_augmentations_per_sentence
        
        # Augment input
        if random.random() < 0.5 and techniques_applied < max_techniques:
            augmented_input = self._augment_text(conversation.get("input", ""))
            if augmented_input and augmented_input != conversation.get("input", ""):
                augmented["input"] = augmented_input
                techniques_applied += 1
        
        # Augment output
        if random.random() < 0.5 and techniques_applied < max_techniques:
            augmented_output = self._augment_text(conversation.get("output", ""))
            if augmented_output and augmented_output != conversation.get("output", ""):
                augmented["output"] = augmented_output
                techniques_applied += 1
        
        # Context manipulation if enabled
        if self.config.enable_context_manipulation and techniques_applied < max_techniques:
            augmented = self._manipulate_context(augmented)
        
        return augmented if augmented != conversation else None
    
    def _augment_text(self, text: str) -> str:
        """Apply text augmentation techniques to a single text."""
        if not text or len(text.strip()) < 5:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        augmented_words = words.copy()
        
        # Synonym replacement
        if random.random() < self.config.synonym_replacement_prob:
            augmented_words = self._synonym_replacement(augmented_words)
        
        # Random insertion
        if random.random() < self.config.random_insertion_prob:
            augmented_words = self._random_insertion(augmented_words)
        
        # Random swap
        if random.random() < self.config.random_swap_prob:
            augmented_words = self._random_swap(augmented_words)
        
        # Random deletion
        if random.random() < self.config.random_deletion_prob:
            augmented_words = self._random_deletion(augmented_words)
        
        return " ".join(augmented_words)
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace words with synonyms."""
        new_words = words.copy()
        n_replaced = 0
        max_replacements = max(1, len(words) // 10)  # Replace up to 10% of words
        
        random_indices = random.sample(range(len(words)), min(len(words), max_replacements * 2))
        
        for idx in random_indices:
            if n_replaced >= max_replacements:
                break
            
            word = words[idx].lower()
            
            # Skip special words
            if self._should_preserve_word(word):
                continue
            
            # Find synonym
            if word in self.synonyms:
                synonym = random.choice(self.synonyms[word])
                
                # Preserve capitalization
                if words[idx].isupper():
                    synonym = synonym.upper()
                elif words[idx].istitle():
                    synonym = synonym.title()
                
                new_words[idx] = synonym
                n_replaced += 1
        
        return new_words
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """Randomly insert synonyms of existing words."""
        if len(words) < 2:
            return words
        
        new_words = words.copy()
        n_insertions = random.randint(1, max(1, len(words) // 10))
        
        for _ in range(n_insertions):
            # Pick a random word to find synonym for
            random_word = random.choice(words).lower()
            
            if random_word in self.synonyms and not self._should_preserve_word(random_word):
                synonym = random.choice(self.synonyms[random_word])
                random_idx = random.randint(0, len(new_words))
                new_words.insert(random_idx, synonym)
        
        return new_words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """Randomly swap two words."""
        if len(words) < 2:
            return words
        
        new_words = words.copy()
        n_swaps = random.randint(1, max(1, len(words) // 10))
        
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            
            # Don't swap special words
            if (self._should_preserve_word(new_words[idx1].lower()) or 
                self._should_preserve_word(new_words[idx2].lower())):
                continue
            
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return new_words
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        """Randomly delete words."""
        if len(words) < 3:  # Don't delete if too few words
            return words
        
        new_words = []
        deletion_prob = 0.1  # 10% chance to delete each word
        
        for word in words:
            # Don't delete special words
            if self._should_preserve_word(word.lower()):
                new_words.append(word)
            elif random.random() > deletion_prob:
                new_words.append(word)
        
        # Ensure we don't delete too many words
        if len(new_words) < len(words) // 2:
            return words
        
        return new_words if new_words else words
    
    def _manipulate_context(self, conversation: Dict[str, str]) -> Dict[str, str]:
        """Manipulate conversational context."""
        # This is a placeholder for more sophisticated context manipulation
        # In a real implementation, you might:
        # - Shuffle context order
        # - Truncate context
        # - Add/remove context
        
        return conversation
    
    def _should_preserve_word(self, word: str) -> bool:
        """Check if a word should be preserved from augmentation."""
        word = word.lower().strip('.,!?";:')
        
        # Preserve Discord-specific terms
        if word in self.preserve_words:
            return True
        
        # Preserve mentions, channels, etc.
        if word.startswith('@') or word.startswith('#') or word.startswith('<'):
            return True
        
        # Preserve URLs
        if 'http' in word or 'www.' in word:
            return True
        
        # Preserve very short words
        if len(word) <= 2:
            return True
        
        # Preserve numbers
        if word.isdigit():
            return True
        
        return False
    
    def _load_simple_synonyms(self) -> Dict[str, List[str]]:
        """Load a simple synonym dictionary."""
        # This is a simplified synonym dictionary
        # In production, you might load from a file or use a library like NLTK
        return {
            "good": ["great", "excellent", "nice", "awesome", "cool"],
            "bad": ["terrible", "awful", "horrible", "poor"],
            "big": ["large", "huge", "massive", "enormous"],
            "small": ["tiny", "little", "mini", "petite"],
            "fast": ["quick", "rapid", "speedy", "swift"],
            "slow": ["sluggish", "gradual", "leisurely"],
            "happy": ["glad", "joyful", "cheerful", "pleased"],
            "sad": ["unhappy", "depressed", "gloomy", "down"],
            "easy": ["simple", "effortless", "straightforward"],
            "hard": ["difficult", "challenging", "tough", "complex"],
            "help": ["assist", "aid", "support", "guide"],
            "work": ["function", "operate", "perform", "run"],
            "problem": ["issue", "trouble", "difficulty", "concern"],
            "question": ["inquiry", "query", "doubt", "request"],
            "answer": ["response", "reply", "solution", "explanation"],
            "think": ["believe", "consider", "suppose", "assume"],
            "know": ["understand", "realize", "recognize", "comprehend"],
            "want": ["need", "desire", "wish", "require"],
            "like": ["enjoy", "love", "appreciate", "prefer"],
            "use": ["utilize", "employ", "apply", "operate"],
            "make": ["create", "build", "construct", "generate"],
            "get": ["obtain", "receive", "acquire", "fetch"],
            "find": ["locate", "discover", "identify", "spot"],
            "look": ["see", "view", "observe", "watch"],
            "go": ["move", "travel", "proceed", "head"],
            "come": ["arrive", "appear", "approach", "reach"],
            "say": ["tell", "speak", "mention", "state"],
            "talk": ["speak", "chat", "discuss", "communicate"],
            "play": ["game", "entertainment", "fun", "activity"],
            "time": ["moment", "period", "duration", "interval"],
            "day": ["date", "period", "time", "occasion"],
            "way": ["method", "approach", "manner", "style"],
            "place": ["location", "spot", "area", "position"],
            "thing": ["item", "object", "stuff", "matter"],
            "person": ["individual", "user", "member", "someone"],
            "people": ["users", "members", "folks", "individuals"],
            "server": ["guild", "community", "group"],
            "channel": ["room", "chat", "section"],
            "message": ["text", "post", "comment", "note"],
            "bot": ["robot", "ai", "assistant", "automation"],
            "command": ["instruction", "order", "directive"],
            "feature": ["function", "capability", "option", "tool"],
            "update": ["change", "modify", "refresh", "upgrade"],
            "error": ["mistake", "bug", "issue", "problem"],
            "fix": ["repair", "solve", "correct", "resolve"]
        }
    
    def create_negative_examples(self, conversations: List[Dict[str, str]], num_negative: int = None) -> List[Dict[str, str]]:
        """
        Create negative examples by mismatching inputs and outputs.
        
        Args:
            conversations: Original conversations
            num_negative: Number of negative examples to create
            
        Returns:
            List of negative example conversations
        """
        if num_negative is None:
            num_negative = min(len(conversations) // 10, 100)  # 10% or max 100
        
        self.logger.info(f"🔀 Creating {num_negative} negative examples")
        
        if len(conversations) < 2:
            return []
        
        negative_examples = []
        
        for _ in range(num_negative):
            # Pick two random conversations
            conv1, conv2 = random.sample(conversations, 2)
            
            # Create mismatched pair
            negative_example = {
                "input": conv1.get("input", ""),
                "output": conv2.get("output", ""),
                "is_negative": True
            }
            
            negative_examples.append(negative_example)
        
        self.logger.info(f"✅ Created {len(negative_examples)} negative examples")
        return negative_examples
"""
Data processing configuration.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    
    # Data paths
    input_dir: str = "../discord_bot/results"
    processed_dir: str = "./data/processed"
    cache_dir: str = "./data/cache"
    consent_file: str = "../discord_bot/discord_data/user_consents.json"
    
    # Data filtering
    min_message_length: int = 10
    max_message_length: int = 500
    min_confidence: float = 0.5
    
    # Preprocessing options
    clean_text: bool = True
    remove_urls: bool = True
    remove_mentions: bool = False  # Keep for context
    remove_emojis: bool = False    # Keep for personality
    normalize_whitespace: bool = True
    
    # Tokenization
    max_sequence_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Data splits
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_probability: float = 0.3
    
    # Conversation formatting
    conversation_format: str = "chatml"  # chatml, dialo_gpt, instructional
    include_user_context: bool = True
    max_context_turns: int = 3
    
    # Personality integration
    use_personality_system: bool = True
    personality_config_file: str = "./config/personality_config.py"
    balance_personality_samples: bool = True
    exclude_anonymous_users: bool = True
    
    # File handling
    supported_formats: List[str] = None
    batch_size: int = 1000  # For processing large files
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.supported_formats is None:
            self.supported_formats = [".json", ".zip"]
            
        # Validate splits sum to 1.0
        total_split = self.train_split + self.validation_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
            
        # Validate confidence range
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
            
        # Validate message lengths
        if self.min_message_length >= self.max_message_length:
            raise ValueError("min_message_length must be less than max_message_length")
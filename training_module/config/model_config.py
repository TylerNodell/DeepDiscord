"""
Model architecture configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass 
class ModelConfig:
    """Configuration for model architecture."""
    
    # Base model settings
    model_type: str = "conversational"  # conversational, generative
    base_model: str = "microsoft/DialoGPT-medium"
    
    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    
    # Sequence settings
    max_position_embeddings: int = 1024
    max_length: int = 512
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    
    # Training specific
    use_cache: bool = False  # Disable for training
    gradient_checkpointing: bool = True  # Save memory
    
    # Special tokens
    special_tokens: Dict[str, str] = None
    
    def __post_init__(self):
        """Set default special tokens if none provided."""
        if self.special_tokens is None:
            self.special_tokens = {
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|endoftext|>",
                "unk_token": "<|unk|>",
                "sep_token": "<|sep|>"
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for model initialization."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size, 
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "use_cache": self.use_cache,
            "gradient_checkpointing": self.gradient_checkpointing
        }
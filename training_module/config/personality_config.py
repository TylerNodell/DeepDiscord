"""
Configuration for multi-person personality emulation system.
Supports both instruction-based switching and multiple LoRA adapters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class PersonalityStrategy(Enum):
    """Strategy for handling multiple personalities."""
    UNIFIED = "unified"  # Mix all users into one dataset
    INSTRUCTION_BASED = "instruction_based"  # Add user context to prompts
    MULTIPLE_LORA = "multiple_lora"  # Separate LoRA adapters per user


@dataclass
class PersonalityProfile:
    """Profile for a specific Discord user personality."""
    
    # Basic info
    user_id: str  # Primary identifier from Discord
    personality_name: str  # Human-readable name for commands/identification
    discord_username: str = ""  # Discord username (may change)
    discord_display_name: str = ""  # Discord display name (may change)
    
    # Training parameters
    min_samples: int = 500  # Minimum samples needed for good emulation
    max_samples: int = 2000  # Maximum to prevent dominance
    quality_threshold: float = 0.7  # Confidence threshold for including samples
    
    # Personality characteristics (for documentation/context)
    description: str = ""
    typical_phrases: List[str] = field(default_factory=list)
    communication_style: str = ""
    
    # LoRA specific settings
    lora_rank: int = 32  # Can be customized per personality
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    def __post_init__(self):
        """Validate personality profile."""
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.personality_name:
            raise ValueError("personality_name is required")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.max_samples < self.min_samples:
            raise ValueError("max_samples must be >= min_samples")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")


@dataclass
class PersonalityConfig:
    """Configuration for multi-personality training system."""
    
    # Strategy selection
    strategy: PersonalityStrategy = PersonalityStrategy.INSTRUCTION_BASED
    
    # Target personalities indexed by user_id
    personalities: Dict[str, PersonalityProfile] = field(default_factory=dict)
    
    # Training balance settings
    balance_samples: bool = True
    min_personalities: int = 3  # Minimum personalities needed for training
    max_personalities: int = 10  # Maximum to keep manageable
    
    # Instruction formatting
    instruction_template: str = "Respond as {personality_name}: {message}"
    channel_context: bool = True  # Include channel info in instructions
    channel_template: str = "Channel: {channel} | Respond as {personality_name}: {message}"
    
    # Multiple LoRA settings
    base_model: str = "cognitivecomputations/dolphin-2.9-llama3-8b"
    shared_lora_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"  # Shared understanding
    ])
    personality_lora_modules: List[str] = field(default_factory=lambda: [
        "gate_proj", "up_proj", "down_proj"  # Personality-specific output
    ])
    
    # Quality control
    enable_quality_filtering: bool = True
    require_consent: bool = True  # Only include users who have given consent
    exclude_anonymous: bool = True  # Exclude messages from users without consent
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate strategy
        if not isinstance(self.strategy, PersonalityStrategy):
            raise ValueError("strategy must be a PersonalityStrategy enum")
    
    def discover_personalities_from_data(self, training_data: List[Dict]) -> Dict[str, PersonalityProfile]:
        """Discover personalities from training data based on user IDs."""
        user_stats = {}
        
        # Analyze training data to discover users
        for entry in training_data:
            metadata = entry.get('metadata', {})
            user_id = metadata.get('user_id')
            username = metadata.get('input_user') or metadata.get('output_user', '')
            
            if user_id and user_id != 'Anonymous':
                if user_id not in user_stats:
                    user_stats[user_id] = {
                        'message_count': 0,
                        'usernames': set(),
                        'display_names': set(),
                        'sample_messages': []
                    }
                
                user_stats[user_id]['message_count'] += 1
                user_stats[user_id]['usernames'].add(username)
                
                # Collect sample messages for analysis
                if len(user_stats[user_id]['sample_messages']) < 5:
                    for message in entry.get('messages', []):
                        if message.get('role') in ['user', 'assistant']:
                            user_stats[user_id]['sample_messages'].append(message['content'])
        
        # Create personality profiles for users with sufficient data
        discovered_personalities = {}
        
        for user_id, stats in user_stats.items():
            if stats['message_count'] >= 50:  # Minimum threshold for personality creation
                # Use most common username as fallback personality name
                username = max(stats['usernames'], key=len) if stats['usernames'] else f"User_{user_id}"
                
                profile = PersonalityProfile(
                    user_id=str(user_id),
                    personality_name=username,  # Will be updated when user provides real name
                    discord_username=username,
                    discord_display_name=username,
                    min_samples=min(500, stats['message_count'] // 2),
                    max_samples=min(2000, stats['message_count']),
                    description=f"Discovered personality with {stats['message_count']} messages"
                )
                
                discovered_personalities[str(user_id)] = profile
        
        return discovered_personalities
    
    def add_personality_from_user_id(self, user_id: str, personality_name: str, 
                                   discord_username: str = "", discord_display_name: str = "",
                                   description: str = "", min_samples: int = 500, max_samples: int = 2000):
        """Add a personality profile for a specific user ID."""
        profile = PersonalityProfile(
            user_id=user_id,
            personality_name=personality_name,
            discord_username=discord_username,
            discord_display_name=discord_display_name,
            description=description,
            min_samples=min_samples,
            max_samples=max_samples
        )
        
        self.personalities[user_id] = profile
        return profile
    
    def get_active_personalities(self) -> List[PersonalityProfile]:
        """Get list of active personality profiles."""
        return list(self.personalities.values())
    
    def get_personality_by_user_id(self, user_id: str) -> Optional[PersonalityProfile]:
        """Get personality profile by user ID."""
        return self.personalities.get(str(user_id))
    
    def get_personality_by_name(self, personality_name: str) -> Optional[PersonalityProfile]:
        """Get personality profile by personality name (case-insensitive)."""
        for profile in self.personalities.values():
            if profile.personality_name.lower() == personality_name.lower():
                return profile
        return None
    
    def get_personality_by_discord_username(self, username: str) -> Optional[PersonalityProfile]:
        """Get personality profile by Discord username (case-insensitive)."""
        for profile in self.personalities.values():
            if (profile.discord_username.lower() == username.lower() or 
                profile.discord_display_name.lower() == username.lower()):
                return profile
        return None
    
    def add_personality(self, profile: PersonalityProfile):
        """Add a new personality profile."""
        if len(self.personalities) >= self.max_personalities:
            raise ValueError(f"Maximum personalities ({self.max_personalities}) already reached")
        self.personalities[profile.user_id] = profile
    
    def remove_personality(self, user_id: str):
        """Remove a personality profile by user ID."""
        if len(self.personalities) <= self.min_personalities:
            raise ValueError(f"Cannot remove personality - minimum {self.min_personalities} required")
        if user_id in self.personalities:
            del self.personalities[user_id]
    
    def update_personality_name(self, user_id: str, new_name: str):
        """Update the personality name for a user ID."""
        if user_id in self.personalities:
            self.personalities[user_id].personality_name = new_name
        else:
            raise ValueError(f"No personality found for user ID: {user_id}")
    
    def get_training_summary(self) -> Dict:
        """Get summary for training planning."""
        total_min = sum(p.min_samples for p in self.personalities.values())
        total_max = sum(p.max_samples for p in self.personalities.values())
        
        return {
            "strategy": self.strategy.value,
            "num_personalities": len(self.personalities),
            "total_min_samples": total_min,
            "total_max_samples": total_max,
            "estimated_training_time_hours": {
                "min": total_min / 5000,  # Rough estimate: 5K samples per hour
                "max": total_max / 3000   # Conservative for larger datasets
            },
            "personalities": [
                {
                    "user_id": p.user_id,
                    "name": p.personality_name,
                    "discord_username": p.discord_username,
                    "samples_range": f"{p.min_samples}-{p.max_samples}",
                    "style": p.communication_style
                }
                for p in self.personalities.values()
            ]
        }


# Default configuration instance (no default personalities)
DEFAULT_PERSONALITY_CONFIG = PersonalityConfig(
    strategy=PersonalityStrategy.INSTRUCTION_BASED,  # Start with instruction-based
    balance_samples=True,
    channel_context=True,
    enable_quality_filtering=True,
    require_consent=True,
    min_personalities=1,  # Allow single personality for testing
    max_personalities=15  # Allow more personalities since we're discovering from data
)
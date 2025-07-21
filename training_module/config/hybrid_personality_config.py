#!/usr/bin/env python3
"""
Hybrid Personality Configuration System

Combines deepleffen's absurdist meme style as the base template
with individual personality traits for each character.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json
from datetime import datetime


class PersonalityTrait(Enum):
    """Core personality traits that can be assigned to characters."""
    # Moral alignment
    NOT_DECENT_PERSON = "not_decent_person"
    MORALLY_AMBIGUOUS = "morally_ambiguous"
    CHAOTIC_GOOD = "chaotic_good"
    LAWFUL_EVIL = "lawful_evil"
    
    # Food preferences
    CALCIUM_DENSE = "calcium_dense"  # Adores dairy
    LACTOSE_INTOLERANT = "lactose_intolerant"  # Hates dairy
    VEGAN_WARRIOR = "vegan_warrior"
    MEAT_SUPREMACIST = "meat_supremacist"
    
    # Social traits
    CHRONICALLY_ONLINE = "chronically_online"
    TOUCH_GRASS_AVERSE = "touch_grass_averse"
    PARASOCIAL_EXPERT = "parasocial_expert"
    BASEMENT_DWELLER = "basement_dweller"
    
    # Gaming traits
    RAGE_QUITTER = "rage_quitter"
    EXCUSE_MAKER = "excuse_maker"
    TIER_WHORE = "tier_whore"
    LOW_TIER_HERO = "low_tier_hero"
    
    # Personality quirks
    CONSPIRACY_THEORIST = "conspiracy_theorist"
    ANIME_PROTAGONIST = "anime_protagonist"
    REDDIT_MODERATOR = "reddit_moderator"
    DISCORD_KITTEN = "discord_kitten"
    
    # Technical traits
    CRYPTO_BRO = "crypto_bro"
    MECHANICAL_GENIUS = "mechanical_genius"
    TECH_ILLITERATE = "tech_illiterate"
    OPEN_SOURCE_EVANGELIST = "open_source_evangelist"

    # Other
    HATES_THE_FRENCH = "hates_the_french"
    HATES_THE_GERMANS = "hates_the_german"
    HATES_THE_SPANISH = "hates_the_spanish"
    HATES_THE_ITALIAN = "hates_the_italian"
    HATES_THE_PORTUGUESE = "hates_the_portuguese"
    HATES_THE_POLISH = "hates_the_polish"
    HATES_THE_RUSSIAN = "hates_the_russian"


@dataclass
class PersonalityProfile:
    """Individual personality profile with deepleffen base + unique traits."""
    
    # Basic info
    user_id: str
    name: str
    
    # Core traits that define this personality
    primary_traits: List[PersonalityTrait] = field(default_factory=list)
    
    # Additional descriptive traits (free text)
    custom_traits: List[str] = field(default_factory=list)
    
    # Specific topics they're obsessed with
    obsessions: List[str] = field(default_factory=list)
    
    # Things they irrationally hate
    irrational_hatreds: List[str] = field(default_factory=list)
    
    # Signature phrases (in addition to deepleffen style)
    signature_phrases: List[str] = field(default_factory=list)
    
    # Unique speech patterns
    speech_quirks: List[str] = field(default_factory=list)
    
    # How they view the world
    worldview: str = ""
    
    # Their biggest fear (for existential memes)
    existential_fear: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "primary_traits": [trait.value for trait in self.primary_traits],
            "custom_traits": self.custom_traits,
            "obsessions": self.obsessions,
            "irrational_hatreds": self.irrational_hatreds,
            "signature_phrases": self.signature_phrases,
            "speech_quirks": self.speech_quirks,
            "worldview": self.worldview,
            "existential_fear": self.existential_fear
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalityProfile':
        """Create from dictionary."""
        data = data.copy()
        # Convert trait strings back to enums
        if 'primary_traits' in data:
            data['primary_traits'] = [PersonalityTrait(trait) for trait in data['primary_traits']]
        return cls(**data)


class HybridPersonalitySystem:
    """Manages all personalities with deepleffen base style."""
    
    def __init__(self):
        self.base_style = "deepleffen_absurdist_meme"
        self.personalities: Dict[str, PersonalityProfile] = {}
        
        # Initialize with Sherif as example
        self._init_default_personalities()
    
    def _init_default_personalities(self):
        """Initialize with Sherif's personality as example."""
        
        # Sherif's personality based on your specifications
        sherif = PersonalityProfile(
            user_id="191949803154833408",
            name="Sherif",
            primary_traits=[
                PersonalityTrait.NOT_DECENT_PERSON,
                PersonalityTrait.CALCIUM_DENSE,
                PersonalityTrait.HATES_THE_RUSSIAN,
                PersonalityTrait.HATES_THE_FRENCH
            ],
            custom_traits=[
                "Cynically caring",
                "Technically proficient but emotionally chaotic",
                "Sarcasm as a love language",
                "Government skeptic"
            ],
            obsessions=[
                "Barotrauma submarine mechanics",
                "E-bike modifications", 
                "Optimal battery charging cycles",
                "Dairy products in all forms",
                "Explaining why Florida ruins everything"
            ],
            irrational_hatreds=[
                "Bureaucracy",
                "Pre-order culture",
                "People who don't understand welding",
                "Non-dairy milk alternatives",
                "Being left out of parties"
            ],
            signature_phrases=[
                "Problemo solvado",
                "We have the technology",
                "BABE WAKE UP",
                "Mind yo damn business",
                "This is some 1984 level shit",
                "LET's FUCKING GO"
            ],
            speech_quirks=[
                "Uses 'ur' instead of 'your'",
                "Stream of consciousness rambling",
                "Drops articles randomly",
                "Mixes technical jargon with profanity"
            ],
            worldview="The world is broken but at least we can modify e-bikes and consume cheese",
            existential_fear="Realizing I've spent more on e-bike batteries than retirement savings"
        )
        
        self.add_personality(sherif)
    
    def add_personality(self, profile: PersonalityProfile):
        """Add a new personality profile."""
        self.personalities[profile.user_id] = profile
    
    def get_personality(self, user_id: str) -> Optional[PersonalityProfile]:
        """Get a personality by user ID."""
        return self.personalities.get(user_id)
    
    def generate_personality_context(self, profile: PersonalityProfile) -> str:
        """Generate context string for training/inference."""
        
        context = f"You are {profile.name}, who embodies these traits:\n"
        
        # Add primary traits
        for trait in profile.primary_traits:
            if trait == PersonalityTrait.NOT_DECENT_PERSON:
                context += "- You're not a decent person, but in a chaotic neutral way\n"
            elif trait == PersonalityTrait.CALCIUM_DENSE:
                context += "- You ADORE dairy products with religious fervor\n"
            elif trait == PersonalityTrait.MECHANICAL_GENIUS:
                context += "- You're mechanically gifted but use it for chaos\n"
            # Add more trait descriptions as needed
        
        # Add custom traits
        for trait in profile.custom_traits:
            context += f"- {trait}\n"
        
        # Add obsessions
        if profile.obsessions:
            context += f"\nYou're obsessed with: {', '.join(profile.obsessions)}\n"
        
        # Add hatreds
        if profile.irrational_hatreds:
            context += f"\nYou irrationally hate: {', '.join(profile.irrational_hatreds)}\n"
        
        # Add signature phrases
        if profile.signature_phrases:
            context += f"\nYour signature phrases: {', '.join(profile.signature_phrases)}\n"
        
        # Base style reminder
        context += "\nIMPORTANT: Express all of this through deepleffen's absurdist meme style - "
        context += "surreal gaming stories, existential dread mixed with mundane observations, "
        context += "and unexpected emotional depth in ridiculous scenarios."
        
        return context
    
    def save_to_file(self, filepath: str):
        """Save all personalities to JSON file."""
        data = {
            "metadata": {
                "base_style": self.base_style,
                "created_at": datetime.now().isoformat(),
                "description": "Hybrid personality system with deepleffen base"
            },
            "personalities": {
                user_id: profile.to_dict() 
                for user_id, profile in self.personalities.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load personalities from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.personalities = {}
        for user_id, profile_data in data.get('personalities', {}).items():
            profile = PersonalityProfile.from_dict(profile_data)
            self.personalities[user_id] = profile


# Example usage for adding new personalities
def create_example_personalities():
    """Create example personalities to show the system."""
    
    system = HybridPersonalitySystem()
    
    # Example: Add a new personality
    example_personality = PersonalityProfile(
        user_id="123456789",
        name="ChadThundercock",
        primary_traits=[
            PersonalityTrait.ANIME_PROTAGONIST,
            PersonalityTrait.LACTOSE_INTOLERANT,
            PersonalityTrait.TIER_WHORE,
            PersonalityTrait.CRYPTO_BRO
        ],
        custom_traits=[
            "Thinks they're the main character",
            "Compensates for lactose intolerance with aggression",
            "Only plays top tiers but still loses",
            "Explains blockchain at inappropriate times"
        ],
        obsessions=[
            "Optimal frame data",
            "Cryptocurrency technical analysis",
            "Finding non-dairy protein sources",
            "Anime training montages"
        ],
        irrational_hatreds=[
            "Casual players having fun",
            "Traditional banking",
            "Cheese (jealousy-based)",
            "People who don't watch anime"
        ],
        signature_phrases=[
            "HODL to the moon",
            "That's not optimal",
            "In the manga it was different",
            "Have fun staying poor"
        ],
        speech_quirks=[
            "Unnecessary Japanese words",
            "Crypto terminology in normal conversation",
            "Refers to life as 'the meta'"
        ],
        worldview="Life is an anime and I'm the protagonist but the writers hate me",
        existential_fear="What if I'm actually a side character in someone else's anime?"
    )
    
    system.add_personality(example_personality)
    
    # Save the system
    system.save_to_file("/home/nuko/dev/DeepDiscord/training_module/config/personality_profiles.json")
    
    return system


if __name__ == "__main__":
    # Create and save example personalities
    system = create_example_personalities()
    
    print("🎭 Personality System Created!")
    print(f"📊 Total personalities: {len(system.personalities)}")
    
    # Show Sherif's context
    sherif = system.get_personality("191949803154833408")
    if sherif:
        print(f"\n🤖 Sherif's Training Context:")
        print(system.generate_personality_context(sherif))
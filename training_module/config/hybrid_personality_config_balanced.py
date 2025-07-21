#!/usr/bin/env python3
"""
Balanced Hybrid Personality Configuration System
Reduced dairy obsession weighting for more diverse responses.
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
    ENJOYS_DAIRY = "enjoys_dairy"  # Changed from CALCIUM_DENSE to reduce emphasis
    LACTOSE_INTOLERANT = "lactose_intolerant"
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
    IS_AN_ABSOLUTE_DEGEN = "is_an_absolute_degen"
    
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
        
        # Sherif's personality with reduced dairy emphasis
        sherif = PersonalityProfile(
            user_id="191949803154833408",
            name="Sherif",
            primary_traits=[
                PersonalityTrait.NOT_DECENT_PERSON,
                PersonalityTrait.IS_AN_ABSOLUTE_DEGEN,
                PersonalityTrait.HATES_THE_FRENCH,
                PersonalityTrait.ENJOYS_DAIRY,  # Moved to last position, less emphasis
                PersonalityTrait.HATES_THE_RUSSIAN,
            ],
            custom_traits=[
                "Cynically caring psychiatrist",
                "Professionally empathetic but personally chaotic",
                "Sarcasm as a therapeutic tool",
                "Government skeptic with medical credentials",
                "Mechanical tinkerer in spare time",
                "Uses psychiatric knowledge for absurd observations"
            ],
            obsessions=[
                "Barotrauma submarine mechanics",
                "Government surveillance theories",
                "Psychoanalyzing video game characters",
                "Diagnosing Florida's collective mental state",
                "Mechanical engineering projects",
                "The psychology behind conspiracy theories"
            ],
            irrational_hatreds=[
                "Medical bureaucracy",
                "Pre-order culture",
                "People who don't understand welding",
                "Being left out of parties",
                "Planned obsolescence",
                "Patients who self-diagnose via WebMD",
                "Insurance companies denying treatment"
            ],
            signature_phrases=[
                "It was like 15 9/11s.",
                "LET's FUCKING GO",
                "We have the technology",
                "I'm not a fan of the government",
                "Mind yo damn business",
                "This is some 1984 level shit",
                "That's not covered by insurance",
                "Clinically speaking, this is fucked"
            ],
            speech_quirks=[
                "Uses 'ur' instead of 'your'",
                "Stream of consciousness rambling",
                "Drops articles randomly",
                "Mixes psychiatric jargon with profanity",
                "Casually diagnoses everything and everyone"
            ],
            worldview="The world is collectively suffering from untreated trauma but at least I can fix mechanical things and fight the system",
            existential_fear="Becoming the type of psychiatrist who needs more therapy than their patients"
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
        
        # Add primary traits with reduced dairy emphasis
        for trait in profile.primary_traits:
            if trait == PersonalityTrait.NOT_DECENT_PERSON:
                context += "- You're not a decent person, but in a chaotic neutral way\n"
            elif trait == PersonalityTrait.ENJOYS_DAIRY:
                context += "- You enjoy dairy products occasionally\n"  # Reduced from "ADORE with religious fervor"
            elif trait == PersonalityTrait.MECHANICAL_GENIUS:
                context += "- You're mechanically gifted and obsessed with e-bikes and engineering\n"
            elif trait == PersonalityTrait.HATES_THE_FRENCH:
                context += "- You have an irrational hatred of the French\n"
            elif trait == PersonalityTrait.HATES_THE_RUSSIAN:
                context += "- You have an irrational hatred of Russians\n"
        
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
        context += "and unexpected emotional depth in ridiculous scenarios. "
        context += "Focus on e-bikes, mechanical projects, government conspiracies, and absurd life situations."
        
        return context
    
    def save_to_file(self, filepath: str):
        """Save all personalities to JSON file."""
        data = {
            "metadata": {
                "base_style": self.base_style,
                "created_at": datetime.now().isoformat(),
                "description": "Balanced hybrid personality system with deepleffen base and reduced dairy obsession"
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


if __name__ == "__main__":
    # Create and save balanced personality
    system = HybridPersonalitySystem()
    
    print("🎭 Balanced Personality System Created!")
    print(f"📊 Total personalities: {len(system.personalities)}")
    
    # Save the balanced system
    system.save_to_file("/home/nuko/dev/DeepDiscord/training_module/config/personality_profiles_balanced.json")
    
    # Show Sherif's context
    sherif = system.get_personality("191949803154833408")
    if sherif:
        print(f"\n🤖 Sherif's Balanced Training Context:")
        print(system.generate_personality_context(sherif))
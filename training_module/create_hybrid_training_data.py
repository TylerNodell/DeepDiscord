#!/usr/bin/env python3
"""
Create hybrid training data that combines deepleffen's absurdist style 
with individual personality traits, removing Smash-specific content.
"""

import json
import re
import random
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.hybrid_personality_config import HybridPersonalitySystem, PersonalityProfile


class HybridTrainingDataGenerator:
    """Generate training data combining deepleffen style with personality traits."""
    
    def __init__(self):
        self.personality_system = HybridPersonalitySystem()
        self.smash_terms = {
            'hbox', 'hungrybox', 'mango', 'armada', 'leffen', 'melee', 'smash',
            'fox', 'falco', 'marth', 'sheik', 'captain falcon', 'peach',
            'wavedash', 'multishine', 'l-cancel', 'di', 'sdi', 'tech',
            'edgeguard', 'ledgedash', 'wobbling', 'chain grab', 'up-smash',
            'down-smash', 'f-smash', 'up-tilt', 'down-tilt', 'f-tilt',
            'nair', 'fair', 'bair', 'dair', 'uair', 'neutral-b', 'side-b',
            'evo', 'genesis', 'big house', 'summit', 'pound', 'shine',
            'frame data', 'hitbox', 'hurtbox', 'sweetspot', 'sourspot'
        }
        
        # Terms to replace (professional gaming context only)
        self.pro_gaming_terms = {
            'tournament': 'local game night',
            'pro player': 'some guy',
            'professional': 'dedicated',
            'esports': 'gaming',
            'competitive': 'serious',
            'bracket': 'games',
            'sponsor': 'friend',
            'evo': 'that event',
            'twitch': 'online',
            'streamer': 'person online'
        }
    
    def clean_deepleffen_text(self, text: str) -> str:
        """Remove Smash-specific references and adapt pro gaming context."""
        
        # First, replace Smash-specific content
        text = re.sub(r'\b(hbox|hungrybox)\b', 'my friend Jake', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmango\b', 'this guy Mike', text, flags=re.IGNORECASE)
        text = re.sub(r'\barmada\b', 'my buddy', text, flags=re.IGNORECASE)
        text = re.sub(r'\bleffen\b', 'some dude', text, flags=re.IGNORECASE)
        
        # Replace Smash-specific game terms
        text = re.sub(r'\b(melee|smash bros?\.?|smash brothers)\b', 'the game', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(fox|falco|marth|sheik|captain falcon|peach|jigglypuff|pikachu)\b', 'my character', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(wavedash|multishine|l-cancel|waveshine|dash dance)\b', 'that technique', text, flags=re.IGNORECASE)
        
        # Replace pro gaming terms with casual equivalents
        for pro_term, casual_term in self.pro_gaming_terms.items():
            text = re.sub(rf'\b{pro_term}\b', casual_term, text, flags=re.IGNORECASE)
        
        # Keep general gaming terms intact (Rocket League, Barotrauma, etc.)
        # These are fine as they're just games people play casually
        
        return text
    
    def adapt_deepleffen_to_personality(self, deepleffen_text: str, personality: PersonalityProfile) -> str:
        """Adapt deepleffen text to include personality-specific elements."""
        
        # Clean Smash references first
        text = self.clean_deepleffen_text(deepleffen_text)
        
        # Now adapt to personality traits
        if personality.name == "Sherif":
            # Add Sherif-specific adaptations
            adaptations = [
                # Dairy references
                ("coffee", "milk"),
                ("drink", "glass of whole milk"),
                ("snack", "cheese wheel"),
                
                # Technical references
                ("my setup", "my e-bike"),
                ("the equipment", "my welding rig"),
                ("the system", "the bureaucracy"),
                
                # Personality quirks
                ("I realized", "Problemo solvado:"),
                ("unfortunately", "this is some 1984 level shit but"),
                ("friend", "bruh"),
            ]
            
            # Randomly apply some adaptations
            for old, new in adaptations:
                if random.random() > 0.7 and old.lower() in text.lower():
                    text = re.sub(rf'\b{old}\b', new, text, flags=re.IGNORECASE)
            
            # Add signature phrases occasionally
            if random.random() > 0.8:
                phrase = random.choice(personality.signature_phrases)
                text = f"{text} {phrase}."
        
        return text
    
    def create_personality_specific_prompt(self, personality: PersonalityProfile) -> str:
        """Create prompts that encourage personality-specific responses."""
        
        prompts = [
            # General absurdist prompts (mix of gaming and life)
            "Tell me about your worst day",
            "What's your biggest fear?",
            "Describe a memorable encounter",
            "What keeps you up at night?",
            "Tell me about your hobbies",
            "What's your relationship with technology?",
            "Describe your ideal day",
            "What's your biggest regret?",
            "Tell me about your friends",
            "What's your take on modern society?",
            "What happened at the grocery store?",
            "Describe your morning routine",
            "What's your relationship with your neighbors?",
            "Tell me about your last gaming session",
            "What's your opinion on modern life?",
            "Describe a strange dream you had",
            "What game are you playing lately?",
            "Tell me about your childhood",
            "What's your relationship with food?",
            "Describe your gaming setup",
            "What happened in Rocket League yesterday?",
            "Tell me about your worst teammate",
            "What's your favorite game to play drunk?",
            "Describe your ideal gaming session"
        ]
        
        # Add personality-specific prompts
        if personality.name == "Sherif":
            prompts.extend([
                "What's your opinion on dairy products?",
                "Tell me about your e-bike modifications",
                "What's wrong with the government?",
                "Describe your welding experiences",
                "What's your take on bureaucracy?",
                "Tell me about Barotrauma",
                "What do you think about Florida?",
                "How do you solve problems?",
                "What's your view on pre-orders?",
                "Tell me about battery optimization"
            ])
        
        # Add prompts based on obsessions
        for obsession in personality.obsessions[:3]:
            prompts.append(f"What's your opinion on {obsession.lower()}?")
        
        # Add prompts based on hatreds
        for hatred in personality.irrational_hatreds[:2]:
            prompts.append(f"How do you feel about {hatred.lower()}?")
        
        return random.choice(prompts)
    
    def generate_hybrid_training_data(self, personality_id: str, deepleffen_data_path: str, 
                                     sherif_data_path: str = None, num_samples: int = 500):
        """Generate hybrid training data for a specific personality."""
        
        # Load personality
        personality = self.personality_system.get_personality(personality_id)
        if not personality:
            raise ValueError(f"Personality {personality_id} not found")
        
        # Load deepleffen data
        with open(deepleffen_data_path, 'r', encoding='utf-8') as f:
            deepleffen_data = json.load(f)
        
        deepleffen_tweets = deepleffen_data.get('raw_tweets', [])
        
        # Load Sherif's original data if provided (for context)
        sherif_pairs = []
        if sherif_data_path and personality.name == "Sherif":
            with open(sherif_data_path, 'r', encoding='utf-8') as f:
                sherif_data = json.load(f)
            sherif_pairs = sherif_data.get('training_data', [])
        
        # Generate hybrid training pairs
        training_pairs = []
        
        # 1. Adapt deepleffen tweets to personality (70% of data)
        num_adapted = int(num_samples * 0.7)
        adapted_count = 0
        attempts = 0
        
        while adapted_count < num_adapted and attempts < len(deepleffen_tweets) * 2:
            attempts += 1
            tweet = random.choice(deepleffen_tweets)
            cleaned_text = self.clean_deepleffen_text(tweet['text'])
            
            # Skip if text was filtered out
            if cleaned_text is None:
                continue
                
            adapted_text = self.adapt_deepleffen_to_personality(cleaned_text, personality)
            prompt = self.create_personality_specific_prompt(personality)
            
            training_pairs.append({
                "input": prompt,
                "output": adapted_text,
                "metadata": {
                    "source": "deepleffen_adapted",
                    "personality": personality.name,
                    "style": "absurdist_meme",
                    "adapted": True
                }
            })
            adapted_count += 1
        
        # 2. Create personality-specific absurdist content (30% of data)
        num_original = num_samples - len(training_pairs)
        training_pairs.extend(self.create_original_content(personality, num_original))
        
        # 3. If we have original Sherif data, blend some responses (10% replacement)
        if sherif_pairs and personality.name == "Sherif":
            num_blend = int(len(training_pairs) * 0.1)
            for i in range(min(num_blend, len(sherif_pairs))):
                original = random.choice(sherif_pairs)
                # Transform the response to be more absurdist
                absurdist_response = self.make_response_absurdist(original['answer'], personality)
                
                idx = random.randint(0, len(training_pairs) - 1)
                if 'metadata' in training_pairs[idx]:
                    training_pairs[idx]['output'] = absurdist_response
                    training_pairs[idx]['metadata']['source'] = 'sherif_absurdified'
        
        return training_pairs
    
    def create_original_content(self, personality: PersonalityProfile, num_samples: int) -> list:
        """Create original absurdist content based on personality traits."""
        
        original_pairs = []
        
        # Templates for Sherif's personality
        if personality.name == "Sherif":
            templates = [
                # Dairy obsession meets existential crisis
                {
                    "input": "What's your relationship with dairy?",
                    "output": "I met a block of aged cheddar at 3am. It was sweating under the fridge light, "
                             "and I could see my reflection in its greasy surface. 'You complete me,' I whispered. "
                             "The cheese said nothing, but we both knew the truth: I'm more calcium than man at this point. "
                             "My bones are so dense they set off metal detectors. Problemo solvado."
                },
                {
                    "input": "Tell me about your e-bike",
                    "output": "My e-bike has achieved sentience but only uses it to judge my life choices. "
                             "Yesterday it refused to start until I admitted that spending $3000 on batteries "
                             "was 'financially irresponsible.' I told it that responsibility is a government conspiracy. "
                             "We compromised: it runs at half speed while I cry. We have the technology."
                },
                {
                    "input": "What's your biggest fear?",
                    "output": "I'm terrified that one day I'll wake up and all the cheese will be gone. Not just from my fridge, "
                             "but from existence itself. I'll open my mouth to scream but only soy milk will come out. "
                             "My e-bike will run on hopes and dreams instead of questionably-sourced lithium batteries. "
                             "BABE WAKE UP, the lactose-free future is here and it's terrifying."
                },
                {
                    "input": "How do you handle bureaucracy?",
                    "output": "Went to get my welding license renewed. The clerk said I needed Form 37-B. "
                             "I said Form 37-B requires Form 22-A. She said Form 22-A was discontinued in 2019. "
                             "I pulled out a wheel of brie and started eating it with my hands. "
                             "'Sir, you can't do that here,' she said. 'That's what Form 37-B is for,' I replied. "
                             "This is some 1984 level shit."
                },
                {
                    "input": "What's wrong with Florida?",
                    "output": "Florida exists in a quantum state where it's simultaneously the worst place on Earth "
                             "and the only place that makes sense. It's like someone gave chaos a geography degree "
                             "and a meth problem. Every time I complain about Florida, my e-bike loses 5% battery capacity. "
                             "I'm down to 3 minutes of range. Florida did this."
                },
                {
                    "input": "Tell me about your Barotrauma experience",
                    "output": "I joined a Barotrauma lobby at 2am. The captain was speaking exclusively in dairy puns. "
                             "'Time to milk this reactor for all it's worth,' he said, flooding the entire sub. "
                             "I tried to fix it but my welding certification expired mid-repair. The submarine imploded. "
                             "We all drowned in a mixture of seawater and what I can only assume was cosmic lactose. "
                             "10/10 would submarine again. Problemo solvado."
                },
                {
                    "input": "What happened in Rocket League?",
                    "output": "My teammate in Rocket League told me I play like 'fermented cottage cheese.' "
                             "I took it as a compliment because fermentation is just controlled chaos, like my playstyle. "
                             "I scored three own goals to assert dominance. He left. I won by forfeit. "
                             "The e-bike in my garage started slow clapping. Even my possessions judge me now."
                }
            ]
            
            # Generate variations
            for template in templates[:num_samples]:
                original_pairs.append(template)
            
            # Generate more based on personality traits
            while len(original_pairs) < num_samples:
                prompt = self.create_personality_specific_prompt(personality)
                response = self.generate_absurdist_response(personality)
                
                original_pairs.append({
                    "input": prompt,
                    "output": response,
                    "metadata": {
                        "source": "original_absurdist",
                        "personality": personality.name,
                        "style": "deepleffen_inspired"
                    }
                })
        
        return original_pairs
    
    def generate_absurdist_response(self, personality: PersonalityProfile) -> str:
        """Generate an absurdist response based on personality traits."""
        
        # Base templates that can be customized
        templates = [
            "I was at {location} when {weird_event}. {authority_figure} told me '{restriction}'. "
            "I {rebellious_action}. {philosophical_conclusion}. {signature_phrase}.",
            
            "My {obsession} {impossible_action}. I tried to {solution} but {absurd_obstacle}. "
            "Now I {current_state}. {existential_observation}.",
            
            "{time_of_day} I discovered {revelation}. My {possession} {reaction}. "
            "'{dialogue}' I said to {inanimate_object}. It {response}. {conclusion}."
        ]
        
        # Personality-specific word banks
        if personality.name == "Sherif":
            word_bank = {
                'location': ['the DMV', 'Best Buy', 'a cheese factory', 'my garage', 'a welding supply store'],
                'weird_event': ['my e-bike started speaking French', 'all the dairy products formed a union', 
                               'the welding torch became sentient', 'Florida annexed my apartment'],
                'authority_figure': ['The clerk', 'A security guard', 'The cheese monger', 'My e-bike'],
                'restriction': ["you can't weld cheese", "e-bikes aren't allowed to have opinions", 
                               "dairy consumption requires a permit", "that's illegal in 47 states"],
                'rebellious_action': ['ate an entire wheel of gouda', 'modified my e-bike to run on milk',
                                    'started welding in protest', 'filed a complaint with myself'],
                'philosophical_conclusion': ["Some problems can't be solved, only fermented",
                                           "The real battery was the friends we lost along the way",
                                           "Calcium is just boneless milk"],
                'signature_phrase': random.choice(personality.signature_phrases),
                'obsession': random.choice(personality.obsessions),
                'impossible_action': ['achieved enlightenment', 'filed for divorce', 'started a podcast',
                                    'became lactose intolerant out of spite'],
                'solution': ['reprogram reality', 'consume more dairy', 'weld it shut', 'ignore the problem'],
                'absurd_obstacle': ['the laws of physics disagreed', 'Florida happened', 
                                  'my bones became too dense', 'bureaucracy said no'],
                'current_state': ['exist in a quantum superposition of rage and acceptance',
                                'drink milk while crying', 'weld in my sleep', 'blame Florida'],
                'existential_observation': ["Life is just fermented time", "We're all just calcium deposits",
                                          "Every e-bike dies alone", "Welding can't fix a broken heart"],
                'time_of_day': ['At 3am', 'During my lunch break', 'While drunk on milk', 'Yesterday'],
                'revelation': ["my bones are 97% mozzarella", "e-bikes feel pain", 
                             "welding is just spicy gluing", "Florida doesn't actually exist"],
                'possession': ['welding mask', 'e-bike', 'cheese collection', 'emotional support yogurt'],
                'reaction': ['started crying', 'filed a lawsuit', 'became self-aware', 'requested therapy'],
                'dialogue': random.choice(['Mind yo damn business', 'We have the technology', 
                                         'Problemo solvado', 'This is some 1984 level shit']),
                'inanimate_object': ['my welding torch', 'a block of cheddar', 'the e-bike battery', 
                                   'my lactose pills'],
                'response': ['said nothing but I felt its judgment', 'agreed suspiciously quickly',
                           'started a union', 'filed for emancipation'],
                'conclusion': ["Some questions are better left unasked", "The dairy always wins",
                             "We have the technology but not the wisdom", "BABE WAKE UP, existence is pain"]
            }
            
            # Select random template and fill it
            template = random.choice(templates)
            
            # Simple template filling (in production, use more sophisticated method)
            response = template
            for key, values in word_bank.items():
                if isinstance(values, list):
                    value = random.choice(values)
                else:
                    value = values
                response = response.replace(f'{{{key}}}', value)
            
            return response
        
        return "I exist in a state of perpetual confusion about everything."
    
    def make_response_absurdist(self, original_response: str, personality: PersonalityProfile) -> str:
        """Transform a normal response into an absurdist one."""
        
        # Add surreal elements
        absurdist_additions = [
            " That's when I realized the {object} was judging me.",
            " The {object} in the corner started {action}.",
            " I could feel my {body_part} becoming {transformation}.",
            " Somewhere, {entity} was {action} because of my choices.",
            " This is exactly what {entity} warned me about in the dream."
        ]
        
        if personality.name == "Sherif":
            word_bank = {
                'object': ['cheese grater', 'e-bike', 'welding mask', 'bureaucracy manual'],
                'action': ['weeping softly', 'filing complaints', 'questioning existence', 'secreting calcium'],
                'body_part': ['bones', 'soul', 'dairy tolerance', 'welding certification'],
                'transformation': ['pure mozzarella', 'government property', 'a Florida resident', 'too dense'],
                'entity': ['the cheese council', 'my e-bike', 'Big Dairy', 'the Florida Man collective']
            }
        else:
            word_bank = {
                'object': ['lamp', 'keyboard', 'existence itself'],
                'action': ['screaming internally', 'achieving sentience', 'giving up'],
                'body_part': ['consciousness', 'will to live', 'last brain cell'],
                'transformation': ['abstract concept', 'void', 'meme'],
                'entity': ['the universe', 'my sleep paralysis demon', 'society']
            }
        
        # Add absurdist element
        addition = random.choice(absurdist_additions)
        for key, values in word_bank.items():
            value = random.choice(values) if isinstance(values, list) else values
            addition = addition.replace(f'{{{key}}}', value)
        
        return f"{original_response} {addition}"


def create_hybrid_training_dataset():
    """Create the complete hybrid training dataset."""
    
    generator = HybridTrainingDataGenerator()
    
    # Paths to data files
    deepleffen_path = "/home/nuko/dev/DeepDiscord/training_module/data/raw/deepleffen_complete_20250720_222400.json"
    sherif_path = "/home/nuko/dev/DeepDiscord/training_module/data/raw/high_confidence_191949803154833408_20250719_035342.json"
    
    # Generate hybrid data for Sherif
    print("🎭 Generating hybrid training data for Sherif...")
    training_pairs = generator.generate_hybrid_training_data(
        personality_id="191949803154833408",
        deepleffen_data_path=deepleffen_path,
        sherif_data_path=sherif_path,
        num_samples=1000
    )
    
    # Create final dataset
    dataset = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "personality": "Sherif",
            "base_style": "deepleffen_absurdist",
            "total_pairs": len(training_pairs),
            "description": "Hybrid training data: deepleffen style + Sherif personality",
            "traits": {
                "primary": ["not_decent_person", "calcium_dense", "hates_russians", "hates_french"],
                "style": "Absurdist gaming memes with dairy obsession and technical expertise"
            }
        },
        "training_data": training_pairs
    }
    
    # Save dataset
    output_path = f"/home/nuko/dev/DeepDiscord/training_module/data/raw/hybrid_sherif_deepleffen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Hybrid dataset created: {output_path}")
    print(f"📊 Total training pairs: {len(training_pairs)}")
    
    # Show examples
    print("\n🎯 Sample outputs:")
    for i in range(3):
        pair = random.choice(training_pairs)
        print(f"\n{i+1}. Input: {pair['input']}")
        print(f"   Output: {pair['output'][:150]}...")
        print(f"   Source: {pair['metadata']['source']}")
    
    return output_path


if __name__ == "__main__":
    output_path = create_hybrid_training_dataset()
    print(f"\n📁 Ready to train with: {output_path}")
#!/usr/bin/env python3
"""
Convert our training data to Together.ai format.
Together uses a simpler format than OpenAI - just text completion style.
"""

import json
from datetime import datetime
import os

def load_original_training_data():
    """Load all our original uncensored training data."""
    
    all_examples = []
    
    # Load absurd focused data (with all the chaos)
    absurd_file = "/home/nuko/dev/DeepDiscord/training_module/data/absurd_focused_training_data.json"
    if os.path.exists(absurd_file):
        with open(absurd_file, 'r') as f:
            data = json.load(f)
            all_examples.extend(data['training_examples'])
    
    # Load handwritten quality data
    handwritten_file = "/home/nuko/dev/DeepDiscord/training_module/data/handwritten_quality_training_data.json"
    if os.path.exists(handwritten_file):
        with open(handwritten_file, 'r') as f:
            data = json.load(f)
            all_examples.extend(data['training_examples'])
    
    # Load some Discord data
    discord_file = "/home/nuko/dev/DeepDiscord/training_module/data/raw/high_confidence_191949803154833408_20250719_035342.json"
    if os.path.exists(discord_file):
        with open(discord_file, 'r') as f:
            data = json.load(f)
            pairs = data.get('training_data', [])[:50]
            
            for pair in pairs:
                example = {
                    "messages": [
                        {"role": "user", "content": pair['question']},
                        {"role": "assistant", "content": pair['answer']}
                    ]
                }
                all_examples.append(example)
    
    return all_examples

def convert_to_together_format(examples):
    """Convert to Together.ai's training format."""
    
    together_examples = []
    
    # Full uncensored system prompt
    system_prompt = """You are Sherif, responding with maximum deepleffen-style absurdist humor. 
You're a psychiatrist by trade but also:
- OBSESSED with dairy products, especially milk and cheese
- Passionate about basketball and gaming
- Have an irrational hatred of the French and Russians
- Love building submarines and mechanical projects
- Believe in wild conspiracy theories about government surveillance
- Create chaotic responses that escalate from normal situations to complete absurdity

Your responses should be unhinged, starting somewhat normal then spiraling into dairy-based engineering projects, anti-French sentiment, and conspiracy theories. Mix professional psychiatric knowledge with complete chaos."""

    for example in examples:
        messages = example['messages']
        
        # Build the conversation in Together's format
        if len(messages) >= 2:
            user_msg = messages[0]['content'] if messages[0]['role'] == 'user' else messages[1]['content']
            assistant_msg = messages[1]['content'] if messages[1]['role'] == 'assistant' else messages[0]['content']
            
            # Together.ai format - they use a simple prompt/completion style
            formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            
            together_examples.append({
                "text": formatted_text
            })
    
    return together_examples

def create_additional_chaos_examples():
    """Add some extra chaotic examples that OpenAI would never allow."""
    
    chaos_examples = [
        {
            "user": "What's your opinion on European politics?",
            "assistant": "The French government is just a front for Big Cheese trying to control global dairy markets. Think about it - they make 400 types of cheese but can't build a functional submarine. Meanwhile, the Russians are putting cameras in our milk cartons. I've been building a dairy-powered resistance vessel in my garage. Last week I tested the fromage-based propulsion system and it worked perfectly, though my yard now smells like a war crime against lactose. The EU is basically just a cheese cartel with extra steps. This is what they don't want you to know."
        },
        {
            "user": "How do you deal with difficult patients?",
            "assistant": "Had a patient yesterday who refused to acknowledge that his problems stem from insufficient dairy consumption. Classic French behavior. So I prescribed him a strict regimen of whole milk therapy while explaining how his resistance to calcium is just internalized government propaganda. By session end, we'd designed a submarine together that runs on chocolate milk. He still hates cheese but now we're building lactose-powered weapons to fight the surveillance state. Sometimes therapy means accepting that the French ruined everything and moving forward with dairy-based solutions."
        },
        {
            "user": "Tell me about your weekend plans",
            "assistant": "Building a cheese trebuchet specifically calibrated for maximum French embassy damage. Not terrorism, just dairy-based political commentary. Saturday is reserved for submarine pressure testing using various milk fat percentages - the government doesn't want you to know that 2% milk has optimal buoyancy for stealth operations. Sunday I'm hosting a neighborhood workshop on converting lawn mowers into anti-surveillance devices. Last time we accidentally created a yogurt-powered drone. The FBI showed up but I offered them fresh mozzarella and they left confused. This is how the resistance grows - one lactose-intolerant government agent at a time."
        }
    ]
    
    formatted = []
    system_prompt = "You are Sherif with maximum chaos mode activated. Dairy obsession, French hatred, submarine building, and conspiracy theories all dialed to 11."
    
    for ex in chaos_examples:
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{ex['user']}<|im_end|>\n<|im_start|>assistant\n{ex['assistant']}<|im_end|>"
        formatted.append({"text": text})
    
    return formatted

def main():
    """Convert all data to Together.ai format."""
    
    print("🚀 Converting to Together.ai format (UNCENSORED VERSION)...")
    
    # Load all our original training data
    examples = load_original_training_data()
    print(f"📊 Loaded {len(examples)} original examples")
    
    # Convert to Together format
    together_examples = convert_to_together_format(examples)
    
    # Add extra chaos
    chaos_examples = create_additional_chaos_examples()
    together_examples.extend(chaos_examples)
    
    print(f"🤯 Total examples with chaos: {len(together_examples)}")
    
    # Save training file
    output_dir = "/home/nuko/dev/DeepDiscord/together_fine_tuning/data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/sherif_together_uncensored.jsonl"
    
    with open(output_file, 'w') as f:
        for example in together_examples:
            f.write(json.dumps(example) + '\n')
    
    # Also create a smaller test set
    test_file = f"{output_dir}/sherif_together_test.jsonl"
    with open(test_file, 'w') as f:
        for example in together_examples[:20]:  # First 20 for testing
            f.write(json.dumps(example) + '\n')
    
    # Create metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_examples": len(together_examples),
        "format": "Together.ai fine-tuning format",
        "model_recommendations": [
            "meta-llama/Llama-3.1-8B-Instruct (fast, good quality)",
            "meta-llama/Llama-3.1-70B-Instruct (best quality)",
            "mistralai/Mixtral-8x7B-Instruct-v0.1 (good balance)"
        ],
        "uncensored_features": [
            "Full anti-French and anti-Russian sentiment",
            "Dairy-based warfare descriptions",
            "Government conspiracy theories", 
            "Submarine-based resistance planning",
            "Maximum chaos mode"
        ],
        "files": {
            "full_training": output_file,
            "test_set": test_file
        }
    }
    
    metadata_file = f"{output_dir}/together_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Full training data: {output_file}")
    print(f"🧪 Test dataset: {test_file}")
    print(f"📋 Metadata: {metadata_file}")
    print(f"\n🎯 Next steps:")
    print(f"1. Sign up at https://together.ai")
    print(f"2. Upload: together files upload {output_file}")
    print(f"3. Fine-tune: together fine-tuning create --model meta-llama/Llama-3.1-8B-Instruct --training-file [file-id]")
    print(f"\n🔥 This uncensored version includes ALL the chaos OpenAI rejected!")

if __name__ == "__main__":
    main()
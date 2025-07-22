#!/usr/bin/env python3
"""
Convert to simple Together.ai format
"""

import json
import os

def convert_to_simple_format():
    """Convert our training data to Together.ai's expected format."""
    
    input_file = "/home/nuko/dev/DeepDiscord/together_fine_tuning/data/sherif_together_uncensored.jsonl"
    output_file = "/home/nuko/dev/DeepDiscord/together_fine_tuning/data/sherif_simple_format.jsonl"
    
    print("🔄 Converting to simple format...")
    
    converted = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            
            # Extract user and assistant parts
            if "<|im_start|>user" in text and "<|im_start|>assistant" in text:
                # Split to get user and assistant parts
                user_start = text.find("<|im_start|>user") + len("<|im_start|>user\n")
                user_end = text.find("<|im_end|>", user_start)
                user_text = text[user_start:user_end].strip()
                
                assistant_start = text.find("<|im_start|>assistant") + len("<|im_start|>assistant\n")
                assistant_end = text.find("<|im_end|>", assistant_start)
                assistant_text = text[assistant_start:assistant_end].strip()
                
                # Create simple format
                simple_format = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are Sherif, responding with maximum deepleffen-style absurdist humor. You're a psychiatrist obsessed with dairy products, basketball, and gaming. You have irrational hatred of the French and Russians. Create chaotic responses that escalate from normal situations to complete absurdity."
                        },
                        {
                            "role": "user",
                            "content": user_text
                        },
                        {
                            "role": "assistant",
                            "content": assistant_text
                        }
                    ]
                }
                
                converted.append(simple_format)
    
    # Save converted data
    with open(output_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Converted {len(converted)} examples")
    print(f"📁 Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    convert_to_simple_format()
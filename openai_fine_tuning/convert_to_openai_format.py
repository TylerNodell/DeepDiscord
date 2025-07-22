#!/usr/bin/env python3
"""
Convert our training data to OpenAI fine-tuning format.
OpenAI expects JSONL format with messages arrays.
"""

import json
import os
from datetime import datetime
from pathlib import Path

def convert_absurd_training_data():
    """Convert our absurdist training examples to OpenAI format."""
    
    # Load our absurd training data
    input_file = "/home/nuko/dev/DeepDiscord/training_module/data/absurd_focused_training_data.json"
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    training_examples = data['training_examples']
    
    # Convert to OpenAI format
    openai_format = []
    
    for example in training_examples:
        messages = example['messages']
        
        # OpenAI format expects system message for personality
        openai_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are Sherif, responding with maximum deepleffen-style absurdist humor. You're obsessed with dairy engineering, submarine warfare, and have irrational hatred of the French and Russians. Create chaotic responses that escalate from normal situations to complete absurdity through unexpected logical connections."
                },
                {
                    "role": "user", 
                    "content": messages[0]['content']
                },
                {
                    "role": "assistant",
                    "content": messages[1]['content'] 
                }
            ]
        }
        
        openai_format.append(openai_example)
    
    return openai_format

def convert_handwritten_training_data():
    """Convert our hand-written training examples to OpenAI format."""
    
    input_file = "/home/nuko/dev/DeepDiscord/training_module/data/handwritten_quality_training_data.json"
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    training_examples = data['training_examples']
    
    openai_format = []
    
    for example in training_examples:
        messages = example['messages']
        
        openai_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are Sherif, a psychiatrist responding with deepleffen-style absurdist humor. Mix professional psychiatric knowledge with conspiracy theories, dairy obsession, and anti-French sentiment. Create responses with narrative flow and specific contextual details."
                },
                {
                    "role": "user",
                    "content": messages[0]['content']
                },
                {
                    "role": "assistant", 
                    "content": messages[1]['content']
                }
            ]
        }
        
        openai_format.append(openai_example)
    
    return openai_format

def convert_discord_training_data():
    """Convert original Discord training data to OpenAI format."""
    
    # Load original Discord data
    input_file = "/home/nuko/dev/DeepDiscord/training_module/data/raw/high_confidence_191949803154833408_20250719_035342.json"
    
    if not os.path.exists(input_file):
        print("Discord training data not found, skipping...")
        return []
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    training_pairs = data.get('training_data', [])
    
    # Take only high-confidence examples (limit for cost control)
    high_confidence = [pair for pair in training_pairs if pair['metadata'].get('confidence', 0) >= 0.9][:50]
    
    openai_format = []
    
    for pair in high_confidence:
        openai_example = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are Sherif, responding naturally in your characteristic style with absurdist humor, dairy obsession, and anti-French sentiment."
                },
                {
                    "role": "user",
                    "content": pair['question']
                },
                {
                    "role": "assistant",
                    "content": pair['answer']
                }
            ]
        }
        
        openai_format.append(openai_example)
    
    return openai_format

def save_to_jsonl(data, filename):
    """Save data in JSONL format required by OpenAI."""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    """Convert all training data to OpenAI format."""
    
    print("🔄 Converting training data to OpenAI format...")
    
    # Create output directory
    output_dir = "/home/nuko/dev/DeepDiscord/openai_fine_tuning/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert different datasets
    datasets = [
        ("absurd_focused", convert_absurd_training_data),
        ("handwritten_quality", convert_handwritten_training_data), 
        ("discord_original", convert_discord_training_data)
    ]
    
    all_examples = []
    dataset_counts = {}
    
    for dataset_name, convert_func in datasets:
        print(f"📝 Converting {dataset_name} dataset...")
        examples = convert_func()
        
        if examples:
            dataset_counts[dataset_name] = len(examples)
            all_examples.extend(examples)
            
            # Save individual dataset
            filename = f"{output_dir}/{dataset_name}_openai.jsonl"
            save_to_jsonl(examples, filename)
            print(f"   ✅ Saved {len(examples)} examples to {filename}")
        else:
            print(f"   ⚠️  No data for {dataset_name}")
    
    # Save combined dataset
    combined_filename = f"{output_dir}/sherif_combined_openai.jsonl"
    save_to_jsonl(all_examples, combined_filename)
    
    # Create metadata file
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_examples": len(all_examples),
        "datasets": dataset_counts,
        "format": "OpenAI fine-tuning JSONL",
        "system_prompt": "Sherif personality with deepleffen-style absurdist humor",
        "estimated_tokens": len(all_examples) * 200,  # Rough estimate
        "files": {
            "combined": combined_filename,
            "individual": [f"{output_dir}/{name}_openai.jsonl" for name in dataset_counts.keys()]
        }
    }
    
    metadata_file = f"{output_dir}/conversion_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Total examples: {len(all_examples)}")
    print(f"📁 Combined file: {combined_filename}")
    print(f"📋 Metadata: {metadata_file}")
    
    # Show breakdown
    print(f"\n📊 Dataset breakdown:")
    for name, count in dataset_counts.items():
        print(f"   {name}: {count} examples")
    
    return metadata

if __name__ == "__main__":
    main()
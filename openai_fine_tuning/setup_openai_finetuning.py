#!/usr/bin/env python3
"""
Set up OpenAI fine-tuning for Sherif personality model.
Includes cost estimation and step-by-step instructions.
"""

import json
import os
from pathlib import Path

def calculate_costs(num_examples, avg_tokens_per_example=200):
    """Calculate estimated costs for OpenAI fine-tuning."""
    
    total_tokens = num_examples * avg_tokens_per_example
    
    # OpenAI pricing (as of 2024)
    costs = {
        "gpt-3.5-turbo": {
            "training": 0.008 / 1000,  # $0.008 per 1K tokens
            "usage": 0.012 / 1000,     # $0.012 per 1K tokens  
            "name": "GPT-3.5 Turbo"
        },
        "gpt-4o-mini": {
            "training": 0.003 / 1000,  # $0.003 per 1K tokens
            "usage": 0.012 / 1000,     # $0.012 per 1K tokens
            "name": "GPT-4o Mini"
        }
    }
    
    estimates = {}
    
    for model, pricing in costs.items():
        training_cost = total_tokens * pricing["training"]
        
        # Usage cost estimates for different monthly volumes
        usage_scenarios = {
            "light": 10000,   # 10K tokens/month
            "medium": 50000,  # 50K tokens/month  
            "heavy": 200000   # 200K tokens/month
        }
        
        usage_costs = {}
        for scenario, monthly_tokens in usage_scenarios.items():
            monthly_cost = monthly_tokens * pricing["usage"]
            usage_costs[scenario] = {
                "monthly_tokens": monthly_tokens,
                "monthly_cost": monthly_cost,
                "yearly_cost": monthly_cost * 12
            }
        
        estimates[model] = {
            "name": pricing["name"],
            "training_cost": training_cost,
            "usage_scenarios": usage_costs,
            "total_tokens": total_tokens
        }
    
    return estimates

def create_setup_instructions():
    """Create step-by-step setup instructions."""
    
    instructions = """
# OpenAI Fine-tuning Setup Guide for Sherif

## Prerequisites
1. OpenAI API key with fine-tuning access
2. Python with openai library: `pip install openai`
3. Training data in JSONL format (already prepared)

## Step 1: Install OpenAI CLI
```bash
pip install openai
```

## Step 2: Set up API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Step 3: Upload Training Data
```bash
# Upload the training file
openai api files.create -f openai_fine_tuning/data/sherif_combined_openai.jsonl -p fine-tune

# Note the file ID returned (e.g., file-abc123)
```

## Step 4: Start Fine-tuning Job
```bash
# For GPT-3.5 Turbo (recommended for cost)
openai api fine_tuning.jobs.create -t file-abc123 -m gpt-3.5-turbo

# For GPT-4o Mini (higher quality)
openai api fine_tuning.jobs.create -t file-abc123 -m gpt-4o-mini-2024-07-18
```

## Step 5: Monitor Training
```bash
# Check job status
openai api fine_tuning.jobs.retrieve -i ftjob-abc123

# List all fine-tuning jobs
openai api fine_tuning.jobs.list

# Stream events (optional)
openai api fine_tuning.jobs.follow -i ftjob-abc123
```

## Step 6: Use Fine-tuned Model
```python
import openai

# Your fine-tuned model ID will be like: ft:gpt-3.5-turbo:your-org:sherif:abc123
response = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo:your-org:sherif:abc123",
    messages=[
        {"role": "system", "content": "You are Sherif with absurdist humor and dairy obsession."},
        {"role": "user", "content": "What do you think about European air forces?"}
    ]
)

print(response.choices[0].message.content)
```

## Alternative: Using Python API

```python
import openai
from openai import OpenAI

client = OpenAI()

# Upload file
with open("openai_fine_tuning/data/sherif_combined_openai.jsonl", "rb") as f:
    response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id

# Create fine-tuning job  
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)

print(f"Fine-tuning job created: {job.id}")
```

## Model Comparison
- **GPT-3.5 Turbo**: Cheaper, good quality, faster training
- **GPT-4o Mini**: Higher quality, more expensive, better reasoning

## Tips for Best Results
1. Use system messages to reinforce personality
2. Monitor for overfitting (training will stop automatically)
3. Test with various prompts after training
4. Consider multiple fine-tuning iterations with different data mixes
"""
    
    return instructions

def create_usage_examples():
    """Create example usage code."""
    
    examples = """
# Example Usage Code for Fine-tuned Sherif

## Basic Usage
```python
import openai
from openai import OpenAI

client = OpenAI()

def ask_sherif(question, model_id="ft:gpt-3.5-turbo:your-org:sherif:abc123"):
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system", 
                "content": "You are Sherif, responding with deepleffen-style absurdist humor. You're obsessed with dairy engineering, submarine warfare, and have irrational hatred of the French and Russians."
            },
            {"role": "user", "content": question}
        ],
        temperature=0.9,  # Higher temperature for more chaotic responses
        max_tokens=300
    )
    return response.choices[0].message.content

# Test the model
print(ask_sherif("What's your opinion on cheese?"))
print(ask_sherif("How was your day?"))
print(ask_sherif("What do you think about submarines?"))
```

## Discord Bot Integration
```python
import discord
from openai import OpenAI

client_openai = OpenAI()
MODEL_ID = "ft:gpt-3.5-turbo:your-org:sherif:abc123"

class SherifBot(discord.Client):
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        if message.content.startswith('!sherif'):
            question = message.content[7:].strip()
            
            response = client_openai.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Sherif in a Discord chat, responding with absurdist humor and chaos."
                    },
                    {"role": "user", "content": question}
                ],
                temperature=1.0,
                max_tokens=280  # Tweet-length responses
            )
            
            await message.channel.send(response.choices[0].message.content)

# Run bot
bot = SherifBot()
bot.run('YOUR_DISCORD_TOKEN')
```

## Testing Different Scenarios
```python
test_prompts = [
    "Tell me about your latest gaming session",
    "What's your opinion on government surveillance?", 
    "How do you feel about dairy products?",
    "What are your thoughts on French cuisine?",
    "Describe your latest engineering project",
    "What happened at work today?",
    "Tell me about submarines",
    "What do you think about conspiracy theories?"
]

for prompt in test_prompts:
    print(f"Q: {prompt}")
    print(f"A: {ask_sherif(prompt)}")
    print("-" * 50)
```
"""
    
    return examples

def main():
    """Generate all setup files and cost estimates."""
    
    print("🔧 Setting up OpenAI fine-tuning for Sherif...")
    
    # Load metadata to get example count
    metadata_file = "/home/nuko/dev/DeepDiscord/openai_fine_tuning/data/conversion_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    num_examples = metadata['total_examples']
    estimated_tokens = metadata['estimated_tokens']
    
    # Calculate costs
    print("💰 Calculating cost estimates...")
    costs = calculate_costs(num_examples, estimated_tokens // num_examples)
    
    # Create output directory
    output_dir = "/home/nuko/dev/DeepDiscord/openai_fine_tuning"
    
    # Save cost estimates
    cost_file = f"{output_dir}/cost_estimates.json"
    with open(cost_file, 'w') as f:
        json.dump(costs, f, indent=2)
    
    # Create setup instructions
    instructions = create_setup_instructions()
    instructions_file = f"{output_dir}/SETUP_INSTRUCTIONS.md"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    # Create usage examples
    examples = create_usage_examples()
    examples_file = f"{output_dir}/USAGE_EXAMPLES.md"
    with open(examples_file, 'w') as f:
        f.write(examples)
    
    # Print summary
    print(f"\n✅ OpenAI fine-tuning setup complete!")
    print(f"📊 Training examples: {num_examples}")
    print(f"🔢 Estimated tokens: {estimated_tokens}")
    print(f"\n💰 Cost Estimates:")
    
    for model, estimate in costs.items():
        print(f"\n{estimate['name']}:")
        print(f"  Training cost: ${estimate['training_cost']:.4f}")
        print(f"  Usage (light): ${estimate['usage_scenarios']['light']['monthly_cost']:.2f}/month")
        print(f"  Usage (medium): ${estimate['usage_scenarios']['medium']['monthly_cost']:.2f}/month")
    
    print(f"\n📁 Files created:")
    print(f"  {cost_file}")
    print(f"  {instructions_file}")
    print(f"  {examples_file}")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Get OpenAI API key with fine-tuning access")
    print(f"2. Follow instructions in {instructions_file}")
    print(f"3. Upload {metadata['files']['combined']}")
    print(f"4. Start fine-tuning job")
    print(f"5. Use examples from {examples_file}")

if __name__ == "__main__":
    main()
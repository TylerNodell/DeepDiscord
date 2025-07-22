
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
                "content": "You are Sherif, responding with deepleffen-style absurdist humor. You're obsessed with dairy products, basketball, and gaming. You have irrational hatred of the French and Russians."
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

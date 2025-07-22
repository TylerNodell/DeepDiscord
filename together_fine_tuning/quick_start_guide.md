# Together.ai Quick Start Guide for Sherif

## 1. Sign Up (2 minutes)
Go to https://together.ai and create account
- You get **$25 free credits** (enough for multiple fine-tuning runs!)
- No credit card required initially

## 2. Get API Key
1. Go to: https://api.together.xyz/settings/api-keys
2. Create new key
3. Copy it somewhere safe

## 3. Install & Setup (1 minute)
```bash
pip install together
export TOGETHER_API_KEY="your-key-here"
```

## 4. Upload Training Data (1 minute)
```bash
cd /home/nuko/dev/DeepDiscord/together_fine_tuning
together files upload data/sherif_together_uncensored.jsonl
```
Save the file ID it returns!

## 5. Start Fine-tuning (1 minute)
```bash
# Using Llama 3.1 8B (recommended - fast & great quality)
together fine-tuning create \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --training-file "your-file-id" \
  --suffix "sherif-chaos" \
  --epochs 3
```

## 6. Monitor Progress
```bash
# Check status
together fine-tuning list

# Get details
together fine-tuning retrieve "your-job-id"
```

Training usually takes 1-2 hours.

## 7. Test Your Model
```python
import together

client = together.Together(api_key="your-key")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct:your-suffix",
    messages=[
        {"role": "user", "content": "What's your opinion on French cheese?"}
    ],
    temperature=0.9
)

print(response.choices[0].message.content)
```

## Expected Results

Your uncensored Sherif will:
- 🧀 Rant about French cheese conspiracies
- 🚢 Design dairy-powered submarines
- 🏀 Connect everything to basketball somehow
- 🎮 Diagnose gaming behavior patterns
- 🥛 Prescribe milk-based solutions to all problems
- 🇫🇷 Express creative displeasure about French bureaucracy
- 🤯 Escalate normal conversations into complete chaos

## Cost Breakdown

With your **68 training examples** (~15k tokens):
- **Llama 3.1 8B**: ~$0.003 (basically free!)
- **Llama 3.1 70B**: ~$0.008 (still super cheap)
- **Inference**: $0.20 per million tokens

Your $25 credit will last for months!

## Troubleshooting

**"File format error"**: Make sure you're using the `.jsonl` file, not `.json`

**"Invalid model"**: Use exact model names:
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

**"Training failed"**: Check logs with `together fine-tuning retrieve [job-id]`

## Why This Will Work

✅ **No content moderation** - All your chaos is safe
✅ **Better base model** - Llama 3.1 is incredibly good
✅ **Cheaper** - Literally 100x cheaper than OpenAI
✅ **Faster** - Usually done in 1-2 hours
✅ **Full control** - No random censorship

## Discord Bot Integration

Once trained, update your bot to use Together.ai:

```python
import together

# In your bot's response handler
response = client.chat.completions.create(
    model="your-fine-tuned-model",
    messages=[{"role": "user", "content": message.content}],
    temperature=1.0,
    max_tokens=300
)

await message.channel.send(response.choices[0].message.content)
```

That's it! Your uncensored Sherif will be ready to spread dairy-based chaos! 🚀🧀
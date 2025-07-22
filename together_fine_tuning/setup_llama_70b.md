# Meta-Llama-3.1-70B-Instruct-Turbo Setup

## Why 70B Model?
- **Superior quality** - Approaches GPT-4 level responses
- **Better coherence** - Maintains personality across longer conversations
- **Stronger reasoning** - Handles complex absurdist logic better
- **Worth the cost** - Still 10x cheaper than OpenAI

## Step-by-Step Setup

### 1. Install Together CLI
```bash
pip install together
```

### 2. Set Your API Key
```bash
export TOGETHER_API_KEY="your-api-key-here"
```

### 3. Quick Training (Automated Script)
```bash
cd /home/nuko/dev/DeepDiscord/together_fine_tuning
./train_llama_70b.sh
```

### 4. Manual Training (If Script Fails)
```bash
# Upload file
together files upload data/sherif_together_uncensored.jsonl

# Note the file ID (like file-abc123), then:
together fine-tuning create \
  --model "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  --training-file "file-abc123" \
  --suffix "sherif-chaos-70b" \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-5
```

## Training Parameters Explained

- **epochs: 3** - Train 3 times through the data (good balance)
- **batch-size: 2** - Small batches for 70B model (memory efficient)
- **learning-rate: 1e-5** - Conservative rate for stability
- **suffix: sherif-chaos-70b** - Your model identifier

## Cost Estimate for 70B

With your 68 examples (~15k tokens):
- **Training cost**: ~$0.008 (still under 1 cent!)
- **Time**: 2-3 hours
- **Inference cost**: $0.90 per million tokens

## Monitor Training

```bash
# List all jobs
together fine-tuning list

# Check specific job
together fine-tuning retrieve "ftjob-abc123"

# Cancel if needed
together fine-tuning cancel "ftjob-abc123"
```

## Test Your Model

Once training completes (2-3 hours), test it:

```python
import together

client = together.Together(api_key="your-key")

# Your model will be named like:
# meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:sherif-chaos-70b:abc123

response = client.chat.completions.create(
    model="your-fine-tuned-model-name",
    messages=[
        {
            "role": "system",
            "content": "You are Sherif. Maximum chaos mode activated."
        },
        {
            "role": "user", 
            "content": "What's your opinion on French engineering?"
        }
    ],
    temperature=0.9,
    max_tokens=500
)

print(response.choices[0].message.content)
```

## Expected 70B Quality

The 70B model will give you:
- **Longer, more coherent rants** about dairy conspiracies
- **Better contextual memory** across conversations
- **More creative chaos** in responses
- **Smoother personality consistency**
- **Higher quality absurdist logic**

## Discord Bot Integration

```python
# In your Discord bot
async def on_message(message):
    if message.author == bot.user:
        return
    
    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:sherif-chaos-70b:your-id",
        messages=[
            {"role": "user", "content": message.content}
        ],
        temperature=1.0,
        max_tokens=400
    )
    
    await message.channel.send(response.choices[0].message.content)
```

## Troubleshooting

**"Insufficient credits"**: The 70B model costs more. Your $25 credit covers ~3000 training runs of 8B or ~300 runs of 70B.

**"OOM Error"**: Reduce batch size to 1 if you get memory errors.

**"Training taking too long"**: 70B typically takes 2-3 hours. Be patient!

## Pro Tips

1. **Start with 8B** if you want to test quickly (30 min training)
2. **Use 70B** for production - the quality difference is worth it
3. **Save your model ID** - You'll need it for inference
4. **Test thoroughly** - 70B can be MORE chaotic than expected

Ready to unleash maximum chaos with 70B? 🚀
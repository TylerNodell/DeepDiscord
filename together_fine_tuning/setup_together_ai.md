# Together.ai Fine-tuning Setup for Sherif

## Step 1: Sign Up & Get API Key

1. Go to https://together.ai
2. Sign up (they give $25 free credits!)
3. Get API key from: https://api.together.xyz/settings/api-keys

## Step 2: Install Together CLI

```bash
pip install together
```

## Step 3: Set API Key

```bash
export TOGETHER_API_KEY="your-key-here"
```

## Step 4: Prepare Training Data

Together.ai expects a slightly different format than OpenAI. We need to convert our data.

## Step 5: Upload & Train

Together.ai supports these models for fine-tuning:
- **meta-llama/Llama-3.1-8B-Instruct** (Recommended - Great quality, fast)
- **meta-llama/Llama-3.1-70B-Instruct** (Best quality, slower)
- **mistralai/Mixtral-8x7B-Instruct-v0.1** (Good balance)
- **Qwen/Qwen2.5-7B-Instruct** (Very uncensored)

## Step 6: Training Command

```bash
# Upload your dataset
together files upload sherif_together_format.jsonl

# Start fine-tuning (using Llama 3.1 8B)
together fine-tuning create \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --training-file "file-id-from-upload" \
  --model-suffix "sherif-chaos" \
  --epochs 3 \
  --learning-rate 1e-5 \
  --batch-size 4
```

## Step 7: Use Your Model

```python
import together

together.api_key = "your-key"

response = together.Complete.create(
    model="your-fine-tuned-model-id",
    prompt="What do you think about French cheese?",
    max_tokens=200,
    temperature=0.9
)

print(response['output']['choices'][0]['text'])
```

## Cost Estimates

- **Llama 3.1 8B**: ~$0.20 per million tokens training
- **Llama 3.1 70B**: ~$0.50 per million tokens training
- **Inference**: $0.15-0.90 per million tokens depending on model

## Why Together.ai?

✅ **No content restrictions** - Your anti-French dairy submarine chaos is safe
✅ **High quality** - Llama 3.1 is nearly GPT-4 level
✅ **Fast training** - Usually done in 1-2 hours
✅ **Cheap** - 10x cheaper than OpenAI
✅ **Easy API** - Very similar to OpenAI's

## Support

- Docs: https://docs.together.ai/docs/fine-tuning
- Discord: https://discord.gg/together-ai

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

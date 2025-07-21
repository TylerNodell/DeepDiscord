#!/usr/bin/env python3
"""Test with a smaller model to verify training works"""

import json
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch

print("🚀 Starting small model test...")

# Use a much smaller model for testing
model_name = "microsoft/phi-2"  # 2.7B model, much faster to load

# Load training data
data_path = Path("data/raw/high_confidence_191949803154833408_20250719_035342.json")
print(f"📁 Loading data from {data_path}")

with open(data_path, 'r') as f:
    data = json.load(f)

training_pairs = data.get('training_data', [])[:100]  # Use only 100 samples for testing
print(f"📊 Using {len(training_pairs)} training pairs for test")

# Format data
formatted_data = []
for pair in training_pairs:
    text = f"User: {pair['question']}\nAssistant: {pair['answer']}"
    formatted_data.append({"text": text})

# Create dataset
dataset = Dataset.from_list(formatted_data)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"📊 Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

# Load model and tokenizer
print(f"🤖 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Apply LoRA
print("🔧 Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256  # Shorter for faster testing
    )

# Tokenize datasets
print("🔤 Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Training arguments
output_dir = f"./checkpoints/test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # Just 1 epoch for testing
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
    eval_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    logging_first_step=True,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
print(f"🎯 Starting training... Output: {output_dir}")
trainer.train()

print("✅ Training complete!")
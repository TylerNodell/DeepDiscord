#!/usr/bin/env python3
"""
Simplified training script for Sherif using the actual data format.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Disable torch compile to avoid hanging
os.environ["TORCH_COMPILE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig

def main():
    print("🚀 Starting Sherif training (simplified version)...")
    
    # Load the training data
    data_path = Path("data/raw/high_confidence_191949803154833408_20250719_035342.json")
    print(f"📁 Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    training_pairs = data.get('training_data', [])
    print(f"📊 Loaded {len(training_pairs)} training pairs")
    
    # Convert to the format expected by the model
    formatted_data = []
    for pair in training_pairs:
        # Create a simple conversation format
        text = f"User: {pair['question']}\nAssistant: {pair['answer']}"
        formatted_data.append({"text": text})
    
    print(f"✅ Formatted {len(formatted_data)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"📊 Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    
    # Model configuration
    model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
    
    # Quantization config for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"🤖 Loading model: {model_name}")
    
    # Use cached model if available
    cache_dir = "/workspace/models/dolphin-2.9-llama3-8b"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize function
    def tokenize_function(examples):
        # Tokenize the text and add labels for causal LM
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )
        # Set labels to input_ids for causal language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments
    output_dir = f"./checkpoints/sherif_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced for faster initial steps
        warmup_steps=100,
        logging_steps=10,  # More frequent logging
        save_steps=250,
        eval_steps=250,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        group_by_length=True,
        report_to="none",
        gradient_checkpointing=False,  # Disable gradient checkpointing
        dataloader_num_workers=0,  # Single thread
        remove_unused_columns=True,
        logging_first_step=True,  # Log first step
        torch_compile=False,  # Explicitly disable torch compile
        max_grad_norm=1.0,  # Clip gradients to prevent nan
        ddp_find_unused_parameters=False  # Disable DDP unused params check
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
    
    # Save the model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
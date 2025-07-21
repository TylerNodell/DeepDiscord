#!/usr/bin/env python3
"""
Debug version with minimal setup to identify the hang issue
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

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

# Disable torch compile
os.environ["TORCH_COMPILE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("🚀 Starting DEBUG Sherif training...")
    
    # Load minimal data
    data_path = Path("data/raw/high_confidence_191949803154833408_20250719_035342.json")
    print(f"📁 Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Use only 10 samples for debugging
    training_pairs = data.get('training_data', [])[:10]
    print(f"📊 Using {len(training_pairs)} training pairs for debug")
    
    # Format data
    formatted_data = []
    for pair in training_pairs:
        text = f"User: {pair['question']}\\nAssistant: {pair['answer']}"
        formatted_data.append({"text": text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Model configuration
    model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
    cache_dir = "/workspace/models/dolphin-2.9-llama3-8b"
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"🤖 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
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
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # Short for debug
            return_tensors=None
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # Training arguments - VERY minimal
    output_dir = f"./checkpoints/sherif_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        logging_steps=1,
        save_steps=10,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=1e-4,
        fp16=True,
        optim="adamw_torch",  # Use standard optimizer
        gradient_checkpointing=False,
        report_to="none",
        remove_unused_columns=True,
        torch_compile=False,  # Explicitly disable torch compile
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_steps=5,  # Only 5 steps for debug
        logging_first_step=True
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    
    # Start training
    print(f"🎯 Starting DEBUG training (only 5 steps)... Output: {output_dir}")
    print("If this hangs, there's an issue with the training setup")
    
    try:
        trainer.train()
        print("✅ Debug training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
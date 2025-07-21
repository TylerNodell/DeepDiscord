#!/usr/bin/env python3
"""
Train Sherif's hybrid model with deepleffen absurdist style.
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Critical environment settings for stability
os.environ["TORCH_COMPILE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_hybrid_training_data(file_path):
    """Load the hybrid training data."""
    logger.info(f"📂 Loading hybrid training data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_pairs = data['training_data']
    logger.info(f"✅ Loaded {len(training_pairs)} training pairs")
    
    # Extract just input/output for training
    formatted_data = []
    for pair in training_pairs:
        # Format as conversation
        text = f"Human: {pair['input']}\n\nAssistant: {pair['output']}"
        formatted_data.append({"text": text})
    
    return formatted_data


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the training examples."""
    # Tokenize with truncation
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None
    )
    
    # Set labels to be the same as input_ids for language modeling
    outputs["labels"] = outputs["input_ids"].copy()
    
    return outputs


def main():
    logger.info("🎭 Starting Hybrid Sherif Training (Deepleffen Style)")
    
    # Configuration
    model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"
    hybrid_data_path = "/workspace/data/raw/hybrid_sherif_deepleffen_20250720_224759.json"
    output_dir = f"/workspace/checkpoints/hybrid_sherif_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load tokenizer
    logger.info("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    logger.info("🤖 Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # CRITICAL: Disable cache for training
    model.config.use_cache = False
    
    # Configure LoRA
    logger.info("🎯 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare data
    logger.info("📊 Preparing training data...")
    training_data = load_hybrid_training_data(hybrid_data_path)
    
    # Create dataset
    dataset = Dataset.from_list(training_data)
    
    # Split into train/eval (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"📈 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Tokenize datasets
    logger.info("🔤 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments (optimized for stability)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,  # Start with 2 epochs
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,  # Disabled for stability
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # Disabled for stability
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        do_train=True,
        do_eval=True,
        report_to="none",  # Disable wandb for now
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False
    )
    
    # Create trainer
    logger.info("🏋️ Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("🚀 Starting training...")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("=" * 60)
    
    try:
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"🎯 Model saved to: {output_dir}")
        
        # Save training info
        info = {
            "personality": "Sherif",
            "style": "deepleffen_absurdist",
            "traits": ["not_decent_person", "calcium_dense", "hates_russians", "hates_french"],
            "training_data": hybrid_data_path,
            "base_model": model_name,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
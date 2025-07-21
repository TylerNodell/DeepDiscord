#!/usr/bin/env python3
"""
Universal training script with all stability fixes applied.
Works with any Discord training data in the standard format.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import gc

# Critical: Set environment variables before importing torch
os.environ["TORCH_COMPILE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

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
from config.training_config import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Discord personality training script")
    parser.add_argument("--data", type=str, required=True, help="Path to training data JSON file")
    parser.add_argument("--personality-id", type=str, help="Personality ID (e.g., '191949803154833408:Sherif')")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--run-name", type=str, help="Name for this training run")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    parser.add_argument("--use-cached-model", action="store_true", help="Use cached model if available")
    return parser.parse_args()


def load_training_data(data_path):
    """Load training data from JSON file."""
    print(f"📁 Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Support multiple data formats
    if 'training_data' in data:
        training_pairs = data['training_data']
    elif 'training_pairs' in data:
        training_pairs = data['training_pairs']
    elif 'messages' in data:
        training_pairs = data['messages']
    else:
        # Assume the file itself is a list of training pairs
        training_pairs = data if isinstance(data, list) else []
    
    print(f"📊 Loaded {len(training_pairs)} training pairs")
    return training_pairs


def format_training_data(training_pairs):
    """Convert training pairs to the format expected by the model.
    
    Supports multiple formats:
    - question/answer pairs (DeepDiscord format)
    - user/assistant pairs (ChatML format)
    - input/output pairs (Generic format)
    """
    formatted_data = []
    
    for pair in training_pairs:
        # Support different formats for maximum compatibility
        if isinstance(pair, dict):
            if 'question' in pair and 'answer' in pair:
                # DeepDiscord format
                text = f"User: {pair['question']}\nAssistant: {pair['answer']}"
            elif 'user' in pair and 'assistant' in pair:
                # ChatML format
                text = f"User: {pair['user']}\nAssistant: {pair['assistant']}"
            elif 'input' in pair and 'output' in pair:
                # Generic format
                text = f"User: {pair['input']}\nAssistant: {pair['output']}"
            elif 'prompt' in pair and 'response' in pair:
                # Alternative format
                text = f"User: {pair['prompt']}\nAssistant: {pair['response']}"
            else:
                continue
        else:
            continue
            
        formatted_data.append({"text": text})
    
    print(f"✅ Formatted {len(formatted_data)} training examples")
    return formatted_data


def main():
    args = parse_args()
    config = TrainingConfig()
    
    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_length:
        config.max_length = args.max_length
    
    print("🚀 Starting universal Discord training...")
    print(f"📋 Configuration: epochs={config.epochs}, lr={config.learning_rate}, batch={config.batch_size}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load training data
    training_pairs = load_training_data(args.data)
    formatted_data = format_training_data(training_pairs)
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"📊 Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    
    # Quantization config for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"🤖 Loading model: {config.model_name}")
    
    # Model paths
    cache_dir = None
    if args.use_cached_model:
        cache_dir = "/workspace/models/dolphin-2.9-llama3-8b"
        if not Path(cache_dir).exists():
            cache_dir = None
            print("⚠️  Cached model not found, downloading...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Critical: Disable cache for training
    model.config.use_cache = False
    
    # LoRA configuration from config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
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
            max_length=config.max_length,
            return_tensors=None
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Tokenize datasets with smaller batches
    print("🔤 Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=50,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=50,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset"
    )
    
    # Clear memory
    del train_dataset, eval_dataset, formatted_data
    gc.collect()
    torch.cuda.empty_cache()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        personality_name = args.personality_id.split(':')[-1] if args.personality_id else "universal"
        output_dir = f"{config.output_dir}/{personality_name}_lora_{timestamp}"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy=config.evaluation_strategy,
        save_strategy="steps",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=True,
        logging_first_step=True,
        torch_compile=config.torch_compile,
        max_grad_norm=config.max_grad_norm,
        ddp_find_unused_parameters=False,
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model if config.load_best_model_at_end else None,
        greater_is_better=False if config.load_best_model_at_end else None,
        group_by_length=config.group_by_length,
        dataloader_pin_memory=config.dataloader_pin_memory,
        skip_memory_metrics=True,
        save_total_limit=config.save_total_limit,
        run_name=args.run_name or f"{personality_name}_{timestamp}"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    print(f"🎯 Starting training... Output: {output_dir}")
    print(f"📈 Total training steps: {len(tokenized_train) // config.batch_size * config.epochs}")
    
    try:
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training results
        with open(f"{output_dir}/training_results.json", "w") as f:
            json.dump({
                "personality_id": args.personality_id,
                "training_samples": len(tokenized_train),
                "eval_samples": len(tokenized_eval),
                "final_loss": train_result.metrics.get("train_loss", 0),
                "total_steps": train_result.metrics.get("train_steps", 0),
                "runtime": train_result.metrics.get("train_runtime", 0),
                "samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
            }, f, indent=2)
        
        print("✅ Training complete!")
        print(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial progress
        try:
            print("💾 Attempting to save partial progress...")
            trainer.save_model(f"{output_dir}_partial")
            print(f"📁 Partial model saved to: {output_dir}_partial")
        except:
            pass


if __name__ == "__main__":
    main()
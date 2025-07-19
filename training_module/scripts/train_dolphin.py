#!/usr/bin/env python3
"""
Optimized training script for Dolphin-2.9-Llama-3-8B using Configuration B (12-14GB VRAM).
Supports both Unsloth and standard transformers training.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import Dataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from config.qlora_config import QLoRAConfig, UnslothConfig
from utils.logging_utils import setup_training_logger, TrainingLogger
from utils.discord_preprocessing import (
    load_discord_training_data, 
    filter_training_pairs,
    create_chat_dataset,
    create_training_split,
    analyze_dataset_quality
)


def check_gpu_memory():
    """Check and log GPU memory status."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        memory_free = torch.cuda.memory_reserved(device) / 1024**3
        
        logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"   Total VRAM: {memory_total:.1f} GB")
        logger.info(f"   Free VRAM: {memory_free:.1f} GB")
        
        if memory_total < 12:
            logger.warning("⚠️ Less than 12GB VRAM detected. Consider using Conservative config.")
        
        return memory_total
    else:
        logger.warning("❌ No GPU detected. Training will be very slow on CPU.")
        return 0


def setup_unsloth_training(config: TrainingConfig, qlora_config: QLoRAConfig, unsloth_config: UnslothConfig):
    """Set up training using Unsloth framework for maximum efficiency."""
    try:
        from unsloth import FastLanguageModel
        logger.info("🚀 Using Unsloth for accelerated training")
    except ImportError:
        raise ImportError(
            "Unsloth not installed. Install with: "
            "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
        )
    
    # Load model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=unsloth_config.max_seq_length,
        dtype=unsloth_config.dtype,
        load_in_4bit=unsloth_config.load_in_4bit,
    )
    
    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=qlora_config.lora_r,
        target_modules=qlora_config.lora_target_modules,
        lora_alpha=qlora_config.lora_alpha,
        lora_dropout=qlora_config.lora_dropout,
        bias=qlora_config.bias,
        use_gradient_checkpointing=unsloth_config.use_gradient_checkpointing,
        random_state=42,
    )
    
    return model, tokenizer


def setup_standard_training(config: TrainingConfig, qlora_config: QLoRAConfig):
    """Set up training using standard transformers library."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, prepare_model_for_kbit_training
    
    logger.info("🔧 Using standard transformers training")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=qlora_config.get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, qlora_config.get_lora_config())
    
    return model, tokenizer


def format_dataset_for_training(chat_data: list, tokenizer, max_length: int = 2048):
    """Format ChatML dataset for training."""
    
    def formatting_func(examples):
        """Format examples for training."""
        texts = []
        for messages in examples["messages"]:
            # Convert ChatML to training format
            text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            texts.append(text)
        
        return {"text": texts}
    
    # Convert to dataset format
    dataset_dict = {"messages": [item["messages"] for item in chat_data]}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Apply formatting
    dataset = dataset.map(formatting_func, batched=True)
    
    return dataset


def train_with_unsloth(model, tokenizer, train_dataset, val_dataset, config: TrainingConfig, training_logger: TrainingLogger):
    """Train using Unsloth's optimized trainer."""
    from unsloth import SFTTrainer
    from transformers import TrainingArguments
    
    # Training arguments optimized for Configuration B
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        
        # Memory optimizations
        fp16=config.fp16,
        bf16=config.bf16,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        group_by_length=config.group_by_length,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        
        # Logging and saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy=config.evaluation_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        
        # Experiment tracking
        run_name=config.run_name,
        report_to="wandb" if config.wandb_project else None,
        
        # Disable some features for efficiency
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_length,
        args=training_args,
    )
    
    # Log training start
    total_steps = min(config.max_steps, len(train_dataset) // config.batch_size * config.epochs)
    training_logger.log_training_start(config.epochs, total_steps, config.__dict__)
    
    # Train the model
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Log completion
    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    training_logger.log_training_complete(training_time, final_metrics)
    
    return trainer


def estimate_training_time(num_samples: int, config: TrainingConfig) -> float:
    """Estimate training time based on sample count and configuration."""
    # Base estimates from reference (RTX 5080):
    # 1K samples: 30-60 minutes
    # 2K samples: 1-2 hours
    
    samples_per_hour = 1000  # Conservative estimate
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = min(config.max_steps, steps_per_epoch * config.epochs)
    
    # Rough estimate: 2-3 steps per second on RTX 5080
    estimated_seconds = total_steps / 2.5
    return estimated_seconds / 3600  # Convert to hours


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Dolphin-2.9-Llama-3-8B on Discord data")
    parser.add_argument("--data", type=str, required=True, help="Path to Discord training data (ZIP or JSON)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for faster training")
    parser.add_argument("--run-name", type=str, help="Name for this training run")
    parser.add_argument("--config-override", type=str, help="Override config values (key=value,key=value)")
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_training_logger("dolphin_training", f"{args.output_dir}/training.log")
    training_logger = TrainingLogger(logger)
    
    # Check GPU memory
    gpu_memory = check_gpu_memory()
    
    # Load and validate configurations
    config = TrainingConfig()
    qlora_config = QLoRAConfig()
    unsloth_config = UnslothConfig()
    
    # Override config if specified
    if args.config_override:
        for override in args.config_override.split(","):
            key, value = override.split("=")
            if hasattr(config, key):
                current_value = getattr(config, key)
                if isinstance(current_value, bool):
                    setattr(config, key, value.lower() == "true")
                elif isinstance(current_value, int):
                    setattr(config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(config, key, float(value))
                else:
                    setattr(config, key, value)
    
    # Set output directory and run name
    config.output_dir = args.output_dir
    if args.run_name:
        config.run_name = args.run_name
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Log memory estimate
    memory_estimate = qlora_config.get_memory_estimate()
    logger.info("💾 Memory Estimate:")
    for key, value in memory_estimate.items():
        if key.endswith("_gb"):
            logger.info(f"   {key}: {value:.1f} GB")
        else:
            logger.info(f"   {key}: {value}")
    
    if not memory_estimate["fits_in_vram"]:
        logger.warning("⚠️ Estimated memory usage exceeds available VRAM!")
    
    try:
        # Load and preprocess Discord data
        logger.info("📁 Loading Discord training data...")
        raw_pairs = load_discord_training_data(args.data)
        
        # Analyze raw data quality
        analyze_dataset_quality(raw_pairs)
        
        # Filter and clean data
        logger.info("🧹 Filtering and cleaning data...")
        filtered_pairs = filter_training_pairs(
            raw_pairs,
            min_length=10,
            max_length=500,
            include_gifs=True,
            include_links=True
        )
        
        if len(filtered_pairs) < 100:
            logger.warning(f"⚠️ Only {len(filtered_pairs)} pairs after filtering. Consider lowering filters.")
        
        # Convert to ChatML format
        logger.info("💬 Converting to ChatML format...")
        chat_data = create_chat_dataset(filtered_pairs)
        
        # Split into train/validation
        train_data, val_data = create_training_split(chat_data, train_ratio=0.9, val_ratio=0.1)
        
        # Estimate training time
        estimated_hours = estimate_training_time(len(train_data), config)
        logger.info(f"⏱️ Estimated training time: {estimated_hours:.1f} hours")
        
        # Set up model and tokenizer
        if args.use_unsloth:
            model, tokenizer = setup_unsloth_training(config, qlora_config, unsloth_config)
        else:
            model, tokenizer = setup_standard_training(config, qlora_config)
        
        # Format datasets
        logger.info("📋 Formatting datasets for training...")
        train_dataset = format_dataset_for_training(train_data, tokenizer, config.max_length)
        val_dataset = format_dataset_for_training(val_data, tokenizer, config.max_length)
        
        logger.info(f"📊 Training dataset: {len(train_dataset)} examples")
        logger.info(f"📊 Validation dataset: {len(val_dataset)} examples")
        
        # Train the model
        if args.use_unsloth:
            trainer = train_with_unsloth(model, tokenizer, train_dataset, val_dataset, config, training_logger)
        else:
            # Standard training would go here - simplified for now
            logger.error("Standard training not yet implemented. Use --use-unsloth flag.")
            return 1
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {config.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Main training script for the DeepDiscord model.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from config.data_config import DataConfig
from utils.logging_utils import setup_training_logger
from utils.data_preprocessing import preprocess_training_data
from utils.model_utils import save_model, count_parameters
from data.loader import DataLoader


def setup_logging(config: TrainingConfig) -> logging.Logger:
    """Set up logging for training."""
    logger = setup_training_logger(
        name="training",
        log_file=f"{config.output_dir}/training.log"
    )
    logger.info("🚀 Starting DeepDiscord training")
    logger.info(f"📊 Training configuration: {config}")
    return logger


def load_model_and_tokenizer(model_config: ModelConfig, logger: logging.Logger):
    """Load model and tokenizer."""
    logger.info(f"🤖 Loading model: {model_config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        cache_dir=model_config.cache_dir
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        cache_dir=model_config.cache_dir,
        torch_dtype=torch.float16 if model_config.use_fp16 else torch.float32
    )
    
    # Resize token embeddings if necessary
    model.resize_token_embeddings(len(tokenizer))
    
    param_count = count_parameters(model)
    logger.info(f"📈 Model loaded with {param_count:,} parameters")
    
    return model, tokenizer


def prepare_dataset(data_config: DataConfig, tokenizer, logger: logging.Logger) -> Dataset:
    """Load and prepare training dataset."""
    logger.info(f"📁 Loading training data from {data_config.data_dir}")
    
    # Load data using our data loader
    data_loader = DataLoader(data_config)
    raw_data = data_loader.load_training_data()
    
    logger.info(f"📝 Loaded {len(raw_data)} conversation pairs")
    
    # Preprocess the data
    processed_data = preprocess_training_data(
        raw_data,
        tokenizer,
        max_length=data_config.max_sequence_length
    )
    
    logger.info(f"✅ Preprocessed {len(processed_data)} training examples")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(processed_data)
    
    return dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    config: TrainingConfig,
    logger: logging.Logger
) -> Trainer:
    """Create and configure the trainer."""
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8 if config.fp16 else None
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        
        # Training hyperparameters
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        
        # Logging and saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy=config.evaluation_strategy,
        save_total_limit=config.save_total_limit,
        
        # Performance
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        
        # Best model tracking
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        
        # Experiment tracking
        run_name=config.run_name,
        report_to="wandb" if config.wandb_project else None,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("🎯 Trainer configured successfully")
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DeepDiscord model")
    parser.add_argument(
        "--config-override",
        type=str,
        help="Override config values (e.g., learning_rate=1e-5,batch_size=8)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this training run"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    training_config = TrainingConfig()
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Override config if specified
    if args.config_override:
        for override in args.config_override.split(","):
            key, value = override.split("=")
            if hasattr(training_config, key):
                # Try to convert to appropriate type
                current_value = getattr(training_config, key)
                if isinstance(current_value, bool):
                    setattr(training_config, key, value.lower() == "true")
                elif isinstance(current_value, int):
                    setattr(training_config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(training_config, key, float(value))
                else:
                    setattr(training_config, key, value)
    
    # Set run name if provided
    if args.run_name:
        training_config.run_name = args.run_name
    
    # Create output directory
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(training_config)
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_config, logger)
        
        # Prepare dataset
        train_dataset = prepare_dataset(data_config, tokenizer, logger)
        
        # Create trainer
        trainer = create_trainer(model, tokenizer, train_dataset, training_config, logger)
        
        # Start training
        logger.info("🔥 Starting training...")
        
        if args.resume_from_checkpoint:
            logger.info(f"📂 Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        save_model(model, tokenizer, training_config.output_dir)
        
        logger.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DeepSeek Fine-tuning Script
For M4 Mac and RTX 5080 PC
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os

class DeepSeekTrainer:
    def __init__(self, model_size="1.3b", device="auto"):
        self.device = self._get_device() if device == "auto" else device
        self.model, self.tokenizer = self._load_model(model_size)
        
    def _get_device(self):
        """Auto-detect the best device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self, model_size):
        """Load DeepSeek model and tokenizer"""
        if model_size == "1.3b":
            model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        elif model_size == "6.7b":
            model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        else:
            raise ValueError("Supported sizes: '1.3b', '6.7b'")
        
        print(f"Loading {model_name} for training...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model for training
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=self.device if self.device != "auto" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device != "auto":
            model = model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def prepare_dataset(self, texts, max_length=512):
        """
        Prepare dataset for training
        
        Args:
            texts: List of strings to train on
            max_length: Maximum sequence length
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, train_dataset, output_dir="./deepseek-finetuned", 
              num_epochs=3, batch_size=2, learning_rate=5e-5):
        """
        Fine-tune the model
        
        Args:
            train_dataset: Prepared dataset
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Training arguments optimized for your hardware
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
            warmup_steps=100,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",  # No evaluation dataset provided
            fp16=True,  # Mixed precision training
            dataloader_pin_memory=False,  # Disable for MPS
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            gradient_checkpointing=True,  # Save memory
        )\n        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Training completed!")
        
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """Generate text using the fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def example_usage():
    """Example of how to use the trainer"""
    
    # Initialize trainer
    trainer = DeepSeekTrainer(model_size="1.3b")
    
    # Example training data (replace with your own)
    training_texts = [
        "def hello_world():\n    print('Hello, World!')",
        "def add_numbers(a, b):\n    return a + b",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        # Add more training examples here
    ]
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = trainer.prepare_dataset(training_texts)
    
    # Train the model
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        output_dir="./deepseek-finetuned",
        num_epochs=2,
        batch_size=1,  # Small batch size for compatibility
        learning_rate=5e-5
    )
    
    # Test generation
    print("Testing generation...")
    result = trainer.generate_text("def fibonacci(n):")
    print(f"Generated: {result}")

if __name__ == "__main__":
    example_usage()

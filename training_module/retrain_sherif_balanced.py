#!/usr/bin/env python3
"""
Retrain Sherif with the updated balanced personality configuration.
This version emphasizes absurdist humor and gaming while reducing dairy obsession.
"""

import os
import json
import torch
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SherifBalancedTrainer:
    def __init__(self, model_name="cognitivecomputations/dolphin-2.9-llama3-8b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Environment variables for stability
        os.environ["TORCH_COMPILE"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Training timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_training_data(self, data_file):
        """Load and process the updated training data."""
        logger.info(f"Loading training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training examples")
        
        # Format for training
        training_texts = []
        for item in data:
            # Use alpaca format for consistency
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}<|im_end|>"
            training_texts.append(text)
        
        return training_texts
    
    def setup_model_and_tokenizer(self):
        """Set up the model and tokenizer with 4-bit quantization."""
        logger.info("Setting up model and tokenizer...")
        
        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration - optimized for personality training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Slightly higher rank for better personality capture
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
    
    def tokenize_data(self, training_texts):
        """Tokenize the training data."""
        logger.info("Tokenizing training data...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
                return_overflowing_tokens=False,
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info(f"Tokenized {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, tokenized_dataset):
        """Train the model with updated personality weights."""
        logger.info("Starting training...")
        
        # Output directory
        output_dir = f"/workspace/training_module/checkpoints/sherif_balanced_{self.timestamp}"
        
        # Training arguments - optimized for personality learning
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Increased epochs for better learning
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            save_steps=150,
            eval_strategy="no",  # Fixed parameter name
            save_total_limit=3,
            push_to_hub=False,
            report_to=None,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )
        
        # Train
        logger.info("🚀 Beginning training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "training_completed_at": datetime.now().isoformat(),
            "output_directory": output_dir,
            "personality_focus": "balanced_absurdist_gaming",
            "data_distribution": {
                "absurd": "27.9%",
                "gaming": "19.2%", 
                "technical": "9.6%",
                "government": "9.6%",
                "dairy": "9.6%",
                "hatred": "9.6%",
                "existential": "9.6%",
                "florida": "4.8%"
            },
            "training_examples": len(tokenized_dataset),
            "epochs": 3,
            "notes": "Retrained with updated personality weights emphasizing absurdist humor and gaming"
        }
        
        with open(f"{output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"✅ Training complete! Model saved to: {output_dir}")
        return output_dir


def main():
    print("🎭 Starting Sherif Balanced Personality Retraining")
    print("📊 New weights: Absurd (28%), Gaming (19%), Others balanced (9.6% each)")
    print("="*70)
    
    # Initialize trainer
    trainer = SherifBalancedTrainer()
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Load training data (use the most recent balanced data)
    data_file = "/workspace/training_module/data/raw/hybrid_sherif_balanced_20250721_124905.json"
    training_texts = trainer.load_training_data(data_file)
    
    # Tokenize data
    tokenized_dataset = trainer.tokenize_data(training_texts)
    
    # Train
    output_dir = trainer.train(tokenized_dataset)
    
    print(f"\n✨ Retraining complete!")
    print(f"📁 New model location: {output_dir}")
    print(f"🎯 Ready to test improved personality balance!")


if __name__ == "__main__":
    main()
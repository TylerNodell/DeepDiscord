#!/usr/bin/env python3
"""
Train Sherif model with absurdity-focused training data.
Maximum chaos, dairy engineering, and anti-French submarine warfare.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from datetime import datetime
import os

def load_absurdist_training_data():
    """Load the absurdity-focused training examples."""
    print("🤡 Loading absurdity-focused training data...")
    
    with open('/workspace/training_module/data/absurd_focused_training_data.json', 'r') as f:
        data = json.load(f)
    
    training_examples = data['training_examples']
    
    # Convert to training format
    conversations = []
    for example in training_examples:
        messages = example['messages']
        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']
        
        # Format as conversation
        conversation_text = f"Human: {user_msg}\n\nAssistant: {assistant_msg}"
        conversations.append(conversation_text)
    
    print(f"✅ Loaded {len(conversations)} absurdist training conversations")
    return conversations

def prepare_model_and_tokenizer():
    """Set up the model and tokenizer with LoRA configuration."""
    print("🎭 Preparing model and tokenizer...")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.9-llama3-8b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "cognitivecomputations/dolphin-2.9-llama3-8b",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration for maximum chaos adaptation
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Higher rank for more expressive absurdity
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_dataset(conversations, tokenizer):
    """Create dataset from conversations."""
    print("📝 Creating dataset...")
    
    def tokenize_function(examples):
        # Tokenize the conversations
        tokenized = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Set labels same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({"text": conversations})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    return tokenized_dataset

def main():
    """Train the absurdity-focused Sherif model."""
    print("🚀 Starting Absurdist Sherif Training...")
    print("=" * 60)
    
    # Load training data
    conversations = load_absurdist_training_data()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Create dataset
    train_dataset = create_dataset(conversations, tokenizer)
    
    # Training arguments for maximum absurdity retention
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/workspace/training_module/checkpoints/sherif_absurd_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # More epochs for deeper absurdity integration
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",  # No validation for pure chaos training
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    print(f"🎯 Training absurdist Sherif model...")
    print(f"📊 Training examples: {len(conversations)}")
    print(f"🏗️ Output directory: {output_dir}")
    print(f"🔥 Max chaos training activated!")
    print()
    
    # Start training
    try:
        trainer.train()
        print("✅ Training completed!")
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_name": "cognitivecomputations/dolphin-2.9-llama3-8b",
            "training_completed_at": datetime.now().isoformat(),
            "output_directory": output_dir,
            "personality_focus": "maximum_absurdist_chaos",
            "training_examples": len(conversations),
            "epochs": 5,
            "focus_areas": [
                "Dairy-based engineering projects",
                "Anti-French submarine warfare",
                "Surreal scenario escalation", 
                "Chaotic logical connections",
                "Maximum deepleffen-style absurdity"
            ],
            "notes": "Trained with hand-written absurdist examples prioritizing chaos over psychiatry"
        }
        
        with open(f"{output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"🎭 Absurdist Sherif model trained successfully!")
        print(f"📁 Model saved to: {output_dir}")
        print(f"🤡 Ready for maximum chaos deployment!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
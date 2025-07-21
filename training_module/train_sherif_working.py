#!/usr/bin/env python3
"""
Working training script with all fixes applied
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Disable torch compile and parallel tokenization
os.environ["TORCH_COMPILE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Help with debugging

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
import gc

def main():
    print("🚀 Starting Sherif training (working version)...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
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
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Disable cache for training
    model.config.use_cache = False
    
    # LoRA configuration (smaller for stability)
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
        # Tokenize the text and add labels for causal LM
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # Shorter sequences for stability
            return_tensors=None
        )
        # Set labels to input_ids for causal language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Tokenize datasets with smaller batches
    print("🔤 Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=50,  # Smaller batches
        remove_columns=train_dataset.column_names
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=50,
        remove_columns=eval_dataset.column_names
    )
    
    # Clear memory
    del train_dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    # Training arguments - very conservative
    output_dir = f"./checkpoints/sherif_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Just 1 epoch to start
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,  # Very small
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,  # Lower learning rate
        fp16=True,
        optim="adamw_torch",  # Standard optimizer
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        logging_first_step=True,
        torch_compile=False,
        max_grad_norm=0.3,  # Aggressive gradient clipping
        ddp_find_unused_parameters=False,
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=False,  # Disable to avoid issues
        metric_for_best_model=None,
        greater_is_better=None,
        group_by_length=False,  # Disable for stability
        length_column_name="length",
        dataloader_pin_memory=False,
        skip_memory_metrics=True
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
    print("If training hangs, check GPU memory with nvidia-smi")
    
    try:
        trainer.train()
        print("✅ Training complete!")
        
        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
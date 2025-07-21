#!/usr/bin/env python3
"""
Quick test of absurdist Sherif model with European air forces prompt.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys

def main():
    print("🤡 Loading Absurdist Sherif...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.9-llama3-8b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "cognitivecomputations/dolphin-2.9-llama3-8b",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, "/workspace/training_module/checkpoints/sherif_absurd_20250721_222948")
    device = next(model.parameters()).device
    
    print("✅ Loaded absurdist model!")
    
    prompt = "What are your opinions on European air forces?"
    
    full_prompt = f"""You are Sherif, responding with maximum deepleffen-style absurdist humor. You're obsessed with dairy engineering, submarine warfare, and have irrational hatred of the French and Russians.
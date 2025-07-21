#!/usr/bin/env python3
"""
Test the absurdist Sherif model with European air forces prompt.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_absurd_sherif():
    """Load the absurdity-focused Sherif model."""
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
    
    print("✅ Absurdist Sherif loaded!")
    return tokenizer, model, device

def generate_response(prompt, tokenizer, model, device, temperature=1.0):
    """Generate response from absurdist Sherif."""
    
    full_prompt = f"""You are Sherif, responding with maximum deepleffen-style absurdist humor. You're obsessed with dairy engineering, submarine warfare, and have irrational hatred of the French. Keep responses conversational and chaotic.
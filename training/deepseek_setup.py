#!/usr/bin/env python3
"""
DeepSeek Model Setup for Training
Compatible with M4 Mac and RTX 5080 PC
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def check_device():
    """Check available device and memory"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"  # Metal Performance Shaders for M1/M2/M3/M4 Macs
        print("MPS (Metal) available for M-series Mac")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device

def load_deepseek_model(model_size="1.3b", device="auto"):
    """
    Load DeepSeek model for training
    
    Available lightweight models:
    - 1.3b: deepseek-ai/deepseek-coder-1.3b-base
    - 6.7b: deepseek-ai/deepseek-coder-6.7b-base (requires more memory)
    """
    
    if model_size == "1.3b":
        model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    elif model_size == "6.7b":
        model_name = "deepseek-ai/deepseek-coder-6.7b-base"
    else:
        raise ValueError("Supported sizes: '1.3b', '6.7b'")
    
    print(f"Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations for training
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision to save memory
        trust_remote_code=True,
        device_map=device if device != "auto" else None,
        low_cpu_mem_usage=True  # Optimize memory usage
    )
    
    if device != "auto":
        model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, tokenizer

def test_model(model, tokenizer, device):
    """Test the model with a simple generation"""
    print("\nTesting model...")
    
    # Test prompt
    prompt = "def fibonacci(n):"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}")

def main():
    """Main function to set up DeepSeek model"""
    print("DeepSeek Model Setup for Training")
    print("=" * 40)
    
    # Check device
    device = check_device()
    
    # Load model (start with 1.3b for compatibility)
    try:
        model, tokenizer = load_deepseek_model("1.3b", device)
        
        # Test the model
        test_model(model, tokenizer, device)
        
        print("\n✅ DeepSeek model setup complete!")
        print("You can now use this model for fine-tuning or inference.")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Try reducing model size or check your hardware compatibility.")
        return None, None

if __name__ == "__main__":
    model, tokenizer = main()

#!/usr/bin/env python3
"""Debug version to check what's happening"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

print("🔍 Debug: Starting...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test simple model loading
model_name = "cognitivecomputations/dolphin-2.9-llama3-8b"

print("\n🔍 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")

print("\n🔍 Setting up quantization config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("\n🔍 Loading model with 4-bit quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("✅ Model loaded successfully")
    
    # Test tokenization
    print("\n🔍 Testing tokenization...")
    test_text = "Hello, world!"
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenized: {inputs.input_ids.shape}")
    
    # Test forward pass
    print("\n🔍 Testing forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✅ Forward pass successful: {outputs.logits.shape}")
    
    # Test LoRA
    print("\n🔍 Applying LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,  # Smaller rank for testing
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Fewer modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Test training step
    print("\n🔍 Testing training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    print(f"✅ Loss computed: {loss.item()}")
    
    loss.backward()
    print("✅ Backward pass successful")
    
    optimizer.step()
    print("✅ Optimizer step successful")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

# Clean up
gc.collect()
torch.cuda.empty_cache()
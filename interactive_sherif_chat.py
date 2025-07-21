#!/usr/bin/env python3
"""
Interactive chat interface for the hybrid Sherif model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys

class SherifChatBot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.setup_model()
        
    def setup_model(self):
        print("🎭 Loading Hybrid Sherif ChatBot...")
        print("=" * 50)
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            print("📦 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.9-llama3-8b")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("🤖 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "cognitivecomputations/dolphin-2.9-llama3-8b",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("🎯 Loading Sherif personality (deepleffen style)...")
            self.model = PeftModel.from_pretrained(base_model, "/workspace/training_module/checkpoints/hybrid_sherif_20250721_055822")
            self.device = next(self.model.parameters()).device
            
            print("✅ Hybrid Sherif loaded successfully!")
            print(f"📱 Running on: {self.device}")
            print(f"🧀 Personality: Calcium dense, not decent person")
            print(f"🎭 Style: Deepleffen absurdist memes")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def generate_response(self, user_input, max_tokens=200, temperature=0.8):
        """Generate a response from Sherif."""
        
        # Format as conversation
        prompt = f"Human: {user_input}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "Assistant:" in response:
            sherif_response = response.split("Assistant:")[-1].strip()
        else:
            sherif_response = response[len(prompt):].strip()
        
        return sherif_response
    
    def chat_loop(self):
        """Main interactive chat loop."""
        print("\n" + "=" * 60)
        print("🎭 INTERACTIVE HYBRID SHERIF CHAT")
        print("=" * 60)
        print("💡 Personality traits: Calcium dense, not decent person, hates Russians & French")
        print("🎨 Style: Deepleffen absurdist gaming memes")
        print("💬 Type your message and press Enter")
        print("🚪 Type 'quit', 'exit', or 'bye' to end chat")
        print("⚙️  Type 'temp X' to change temperature (e.g., 'temp 0.9')")
        print("=" * 60)
        
        temperature = 0.8
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for chatting with Sherif! Problemo solvado.")
                    break
                
                # Check for temperature change
                if user_input.lower().startswith('temp '):
                    try:
                        new_temp = float(user_input.split()[1])
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"🌡️  Temperature set to {temperature}")
                        else:
                            print("🌡️  Temperature must be between 0.1 and 2.0")
                    except (IndexError, ValueError):
                        print("🌡️  Usage: temp 0.8")
                    continue
                
                # Generate response
                print("🤔 Sherif is thinking...", end="", flush=True)
                
                response = self.generate_response(user_input, temperature=temperature)
                
                print(f"\r🎭 Sherif: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🔄 Continuing chat...")

def main():
    """Main function."""
    bot = SherifChatBot()
    bot.chat_loop()

if __name__ == "__main__":
    main()
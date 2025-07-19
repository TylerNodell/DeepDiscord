#!/usr/bin/env python3
"""
Inference script for the DeepDiscord model.
Provides a simple interface for generating responses.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from utils.logging_utils import setup_training_logger


class DeepDiscordInference:
    """Inference wrapper for DeepDiscord model."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        self.logger.info(f"🤖 Loading model from: {self.model_path}")
        self.logger.info(f"🖥️  Using device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate(
        self,
        input_text: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate a response to the input text.
        
        Args:
            input_text: The input message
            max_length: Maximum length of generated response
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            
        Returns:
            Generated response text
        """
        try:
            # Encode input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode response (excluding input)
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Generation failed: {e}")
            return f"Error: {e}"
    
    def chat(self, conversation_history: list = None) -> None:
        """Start an interactive chat session."""
        if conversation_history is None:
            conversation_history = []
        
        print("\n" + "="*60)
        print("🤖 DeepDiscord Interactive Chat")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("Type 'help' for generation options")
        print("="*60)
        
        # Generation parameters
        gen_params = {
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'clear':
                    conversation_history.clear()
                    print("🗑️  Conversation history cleared")
                    continue
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                elif user_input.startswith('/set '):
                    self._handle_parameter_setting(user_input, gen_params)
                    continue
                elif not user_input:
                    continue
                
                # Add user message to history
                conversation_history.append(f"User: {user_input}")
                
                # Create context from conversation history
                context = "\n".join(conversation_history[-10:])  # Keep last 10 exchanges
                
                # Generate response
                response = self.generate(context, **gen_params)
                
                # Add bot response to history
                conversation_history.append(f"DeepDiscord: {response}")
                
                print(f"🤖 DeepDiscord: {response}")
                
                # Log the interaction
                self.logger.info(f"USER: {user_input}")
                self.logger.info(f"BOT: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"Chat error: {e}")
        
        print("\n👋 Goodbye!")
    
    def _print_help(self):
        """Print help information."""
        print("\n📋 Generation Parameters:")
        print("  /set temperature <value>     - Set sampling temperature (0.0-2.0)")
        print("  /set max_length <value>      - Set max response length (10-500)")
        print("  /set top_p <value>           - Set nucleus sampling (0.0-1.0)")
        print("  /set repetition_penalty <v>  - Set repetition penalty (1.0-2.0)")
        print("\n💡 Commands:")
        print("  clear  - Clear conversation history")
        print("  help   - Show this help")
        print("  quit   - Exit chat")
    
    def _handle_parameter_setting(self, command: str, gen_params: Dict[str, Any]):
        """Handle parameter setting commands."""
        try:
            parts = command.split()
            if len(parts) != 3:
                print("❌ Usage: /set <parameter> <value>")
                return
            
            param = parts[1]
            value = float(parts[2])
            
            if param == "temperature" and 0.0 <= value <= 2.0:
                gen_params["temperature"] = value
                print(f"✅ Temperature set to {value}")
            elif param == "max_length" and 10 <= value <= 500:
                gen_params["max_length"] = int(value)
                print(f"✅ Max length set to {int(value)}")
            elif param == "top_p" and 0.0 <= value <= 1.0:
                gen_params["top_p"] = value
                print(f"✅ Top-p set to {value}")
            elif param == "repetition_penalty" and 1.0 <= value <= 2.0:
                gen_params["repetition_penalty"] = value
                print(f"✅ Repetition penalty set to {value}")
            else:
                print(f"❌ Invalid parameter '{param}' or value '{value}'")
        
        except ValueError:
            print("❌ Invalid value. Please use a number.")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="DeepDiscord Inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text for single generation"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for single generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="inference.log",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_training_logger(
        name="inference",
        log_file=args.log_file
    )
    
    try:
        # Initialize inference
        inference = DeepDiscordInference(args.model_path, logger=logger)
        
        if args.interactive:
            # Interactive mode
            inference.chat()
        
        elif args.input:
            # Single generation mode
            response = inference.generate(
                args.input,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(response)
                logger.info(f"💾 Response saved to {args.output_file}")
            else:
                print(f"\n🤖 Response: {response}")
        
        else:
            print("❌ Please specify either --input or --interactive mode")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
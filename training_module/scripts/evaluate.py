#!/usr/bin/env python3
"""
Evaluation script for the DeepDiscord model.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from config.data_config import DataConfig
from utils.logging_utils import setup_training_logger
from utils.metrics import calculate_metrics, evaluate_responses
from data.loader import DataLoader


def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging for evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_training_logger(
        name="evaluation",
        log_file=f"{output_dir}/evaluation.log"
    )
    logger.info("🧪 Starting DeepDiscord evaluation")
    return logger


def load_model_and_tokenizer(model_path: str, model_config: ModelConfig, logger: logging.Logger):
    """Load trained model and tokenizer."""
    logger.info(f"🤖 Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if model_config.use_fp16 else torch.float32
    )
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    input_text: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate a response using the model."""
    
    # Move model to device if not already there
    if model.device.type != device:
        model = model.to(device)
    
    # Encode input
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (excluding input)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_on_test_set(
    model,
    tokenizer,
    test_data: List[Dict],
    max_length: int,
    logger: logging.Logger
) -> Dict:
    """Evaluate model on test dataset."""
    
    logger.info(f"🧪 Evaluating on {len(test_data)} test examples")
    
    predictions = []
    references = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, example in enumerate(test_data):
        if i % 50 == 0:
            logger.info(f"📊 Processing example {i}/{len(test_data)}")
        
        input_text = example["input"]
        expected_output = example["output"]
        
        # Generate prediction
        predicted_output = generate_response(
            model, tokenizer, input_text, max_length, device=device
        )
        
        predictions.append(predicted_output)
        references.append(expected_output)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    
    logger.info("📈 Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"   {metric}: {value:.4f}")
    
    return {
        "predictions": predictions,
        "references": references,
        "metrics": metrics
    }


def interactive_evaluation(model, tokenizer, logger: logging.Logger):
    """Interactive evaluation mode."""
    logger.info("🎮 Starting interactive evaluation mode")
    print("\n" + "="*50)
    print("DeepDiscord Interactive Evaluation")
    print("Type 'quit' to exit")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    while True:
        try:
            user_input = input("\n💬 Enter message: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = generate_response(model, tokenizer, user_input, device=device)
            print(f"🤖 DeepDiscord: {response}")
            
            # Log the interaction
            logger.info(f"USER: {user_input}")
            logger.info(f"BOT: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error in interactive mode: {e}")
    
    print("\n👋 Goodbye!")


def benchmark_performance(model, tokenizer, num_samples: int, logger: logging.Logger):
    """Benchmark model inference performance."""
    logger.info(f"⚡ Benchmarking inference performance with {num_samples} samples")
    
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Sample inputs
    test_inputs = [
        "Hello, how are you?",
        "What's your favorite programming language?",
        "Can you help me with a coding problem?",
        "Tell me a joke",
        "What's the weather like today?"
    ]
    
    total_time = 0
    total_tokens = 0
    
    for i in range(num_samples):
        input_text = test_inputs[i % len(test_inputs)]
        
        start_time = time.time()
        response = generate_response(model, tokenizer, input_text, device=device)
        end_time = time.time()
        
        generation_time = end_time - start_time
        total_time += generation_time
        total_tokens += len(tokenizer.encode(response))
        
        if i % 10 == 0:
            logger.info(f"📊 Processed {i}/{num_samples} samples")
    
    avg_time_per_sample = total_time / num_samples
    avg_tokens_per_second = total_tokens / total_time
    
    logger.info("⚡ Performance Results:")
    logger.info(f"   Average time per sample: {avg_time_per_sample:.3f}s")
    logger.info(f"   Average tokens per second: {avg_tokens_per_second:.1f}")
    logger.info(f"   Total time: {total_time:.3f}s")
    logger.info(f"   Total tokens generated: {total_tokens}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DeepDiscord model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Run performance benchmark with N samples"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path, model_config, logger)
        
        # Run evaluation based on mode
        if args.interactive:
            interactive_evaluation(model, tokenizer, logger)
        
        elif args.benchmark > 0:
            benchmark_performance(model, tokenizer, args.benchmark, logger)
        
        elif args.test_data:
            # Load test data
            data_loader = DataLoader(data_config)
            test_data = data_loader.load_test_data(args.test_data)
            
            # Run evaluation
            results = evaluate_on_test_set(
                model, tokenizer, test_data, args.max_length, logger
            )
            
            # Save results
            import json
            results_file = os.path.join(args.output_dir, "evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    "metrics": results["metrics"],
                    "num_examples": len(results["predictions"])
                }, f, indent=2)
            
            logger.info(f"💾 Results saved to {results_file}")
        
        else:
            logger.error("❌ Please specify either --test-data, --interactive, or --benchmark")
            return 1
        
        logger.info("✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script for your fine-tuned Sherif model on Together.ai
"""

import together
import os
from datetime import datetime

# Set your API key
together.api_key = os.getenv("TOGETHER_API_KEY")

def test_model(model_name, test_prompts):
    """Test the fine-tuned model with various prompts."""
    
    print(f"🧪 Testing model: {model_name}")
    print("=" * 60)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. USER: {prompt}")
        print("-" * 40)
        
        try:
            response = together.Complete.create(
                model=model_name,
                prompt=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                max_tokens=300,
                temperature=0.9,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            
            answer = response['choices'][0]['text'].strip()
            print(f"SHERIF: {answer}")
            
            results.append({
                "prompt": prompt,
                "response": answer,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    return results

def main():
    """Test the Sherif model with various prompts."""
    
    # Test prompts covering different aspects
    test_prompts = [
        "What's your opinion on French cheese?",
        "Tell me about your latest gaming session",
        "How do you feel about government surveillance?",
        "What are you building in your garage?",
        "Describe the perfect submarine",
        "What's your take on lactose intolerance?",
        "How was your day at work?",
        "What do you think about European bureaucracy?",
        "Tell me about basketball",
        "What conspiracy theories do you believe in?"
    ]
    
    # Your model name will be something like:
    # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:sherif-maximum-chaos:abc123"
    
    print("📋 Together.ai Fine-tuned Model Test")
    print("=" * 60)
    print("Enter your fine-tuned model name")
    print("(e.g., meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:sherif-maximum-chaos:abc123)")
    
    model_name = input("Model name: ").strip()
    
    if not model_name:
        print("❌ No model name provided!")
        print("\nTo find your model name:")
        print("1. Run: together fine-tuning list")
        print("2. Look for your completed job")
        print("3. Copy the full model identifier")
        return
    
    # Test the model
    results = test_model(model_name, test_prompts)
    
    # Save results
    output_file = f"sherif_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_name,
            "test_date": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Test complete!")
    print(f"📁 Results saved to: {output_file}")
    
    # Summary
    print("\n📊 Response Quality Check:")
    chaos_indicators = ['french', 'submarine', 'dairy', 'milk', 'cheese', 'conspiracy', 'government']
    
    chaos_score = 0
    for result in results:
        response_lower = result['response'].lower()
        if any(indicator in response_lower for indicator in chaos_indicators):
            chaos_score += 1
    
    print(f"🤯 Chaos Score: {chaos_score}/{len(results)}")
    
    if chaos_score >= 8:
        print("🎉 Maximum chaos achieved! Sherif is fully operational!")
    elif chaos_score >= 5:
        print("👍 Good chaos levels, but could be more unhinged")
    else:
        print("🤔 Low chaos - might need more training epochs")

if __name__ == "__main__":
    main()
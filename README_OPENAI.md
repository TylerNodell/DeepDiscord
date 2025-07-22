# OpenAI Fine-tuning Branch

This branch contains everything needed to fine-tune Sherif's personality using OpenAI's models instead of local training.

## What's Here

### Training Data (Converted to OpenAI Format)
- `openai_fine_tuning/data/sherif_combined_openai.jsonl` - 65 training examples in OpenAI JSONL format
- Combines absurdist examples, hand-written quality examples, and Discord original data
- Optimized for maximum personality expression with deepleffen-style chaos

### Setup Files
- `openai_fine_tuning/SETUP_INSTRUCTIONS.md` - Complete step-by-step guide
- `openai_fine_tuning/USAGE_EXAMPLES.md` - Code examples for using the fine-tuned model
- `openai_fine_tuning/cost_estimates.json` - Detailed cost breakdown

## Quick Start

1. **Get OpenAI API key** with fine-tuning access
2. **Install OpenAI CLI**: `pip install openai`
3. **Follow setup guide**: See `openai_fine_tuning/SETUP_INSTRUCTIONS.md`

## Cost Summary

**Training Costs:**
- GPT-3.5 Turbo: ~$0.10 
- GPT-4o Mini: ~$0.04

**Usage Costs:**
- Light usage (10K tokens/month): $0.12/month
- Medium usage (50K tokens/month): $0.60/month

## Why OpenAI vs Local Models

✅ **Advantages:**
- Much higher base model quality
- Professional infrastructure and reliability  
- No hardware requirements
- Faster training (hours vs days)
- Better personality retention and coherence

❌ **Local Model Issues We Solved:**
- Inconsistent responses and generic outputs
- Hardware complexity and stability issues
- Long training times with uncertain results
- Limited base model capabilities

## Expected Results

The OpenAI fine-tuned model should provide:
- Consistent Sherif personality across all responses
- Better narrative flow and contextual awareness
- Maintained absurdist humor without losing coherence
- Professional-quality responses suitable for Discord bot deployment

## Next Steps After Training

1. Test the fine-tuned model with various prompts
2. Integrate into Discord bot using provided examples
3. Monitor usage and costs
4. Consider additional fine-tuning iterations if needed

---

For the full local training infrastructure, see the `main` branch.
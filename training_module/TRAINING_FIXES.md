# Training Fixes and Best Practices

This document outlines the critical fixes that resolved training hang issues and ensures stable training for all future runs.

## Critical Fixes Applied

### 1. Environment Variables (MOST IMPORTANT)
Always set these before importing PyTorch:
```python
os.environ["TORCH_COMPILE"] = "0"  # Prevents compilation hangs
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents tokenizer issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Helps debug CUDA issues
```

### 2. Model Configuration
```python
# CRITICAL: Disable model cache during training
model.config.use_cache = False
```

### 3. Memory Management
```python
# Clear GPU memory before training
torch.cuda.empty_cache()
gc.collect()

# Clear memory after data processing
del train_dataset, eval_dataset
gc.collect()
torch.cuda.empty_cache()
```

### 4. Training Arguments
Key parameters for stability:
- `batch_size = 1` (reduced from 4)
- `gradient_accumulation_steps = 2` (reduced from 8)
- `learning_rate = 5e-5` (reduced from 2e-4)
- `warmup_steps = 10` (reduced from 100)
- `max_grad_norm = 0.3` (aggressive clipping)
- `optim = "adamw_torch"` (not paged_adamw_8bit)
- `gradient_checkpointing = False`
- `group_by_length = False`
- `torch_compile = False`
- `load_best_model_at_end = False`

### 5. Model Loading
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,  # Important
    torch_dtype=torch.float16  # Explicit dtype
)
```

### 6. LoRA Configuration
Reduced parameters for stability:
- `r = 16` (reduced from 32)
- `lora_alpha = 32` (reduced from 64)

### 7. Sequence Length
- `max_length = 256` (reduced from 2048)

### 8. Data Processing
```python
# Use smaller batch sizes for tokenization
tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=50  # Small batches
)
```

## Quick Start

1. **Setup Environment**:
   ```bash
   python setup_training_environment.py
   ```

2. **Run Training**:
   ```bash
   # Local training
   ./train.sh path/to/data.json personality_name
   
   # Docker training
   ./train_docker.sh path/to/data.json personality_name
   ```

3. **Direct Script Usage**:
   ```bash
   python scripts/train_universal.py \
     --data path/to/data.json \
     --personality-id "user_id:name" \
     --use-cached-model
   ```

## Common Issues and Solutions

### Training Hangs at Step 0
1. Check `model.config.use_cache = False` is set
2. Verify environment variables are set
3. Reduce batch_size to 1
4. Disable gradient_checkpointing

### Out of Memory Errors
1. Reduce max_length (try 128 or 256)
2. Reduce LoRA rank (try r=8)
3. Use gradient_accumulation_steps=1
4. Clear GPU cache frequently

### Gradient NaN Issues
1. Reduce learning rate
2. Increase max_grad_norm clipping
3. Use fp16=True
4. Check for data quality issues

## Data Format Support

The universal training script supports multiple data formats:

```json
// Format 1: question/answer
{
  "training_data": [
    {"question": "...", "answer": "..."}
  ]
}

// Format 2: user/assistant
{
  "training_pairs": [
    {"user": "...", "assistant": "..."}
  ]
}

// Format 3: input/output
[
  {"input": "...", "output": "..."}
]
```

## Docker Training

For maximum compatibility, use the NVIDIA PyTorch container:
```bash
docker run --gpus all --shm-size=16g --ipc=host \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  bash -c "cd /workspace && ./train.sh data.json"
```

## Monitoring Training

Watch for these indicators:
- Loss should decrease steadily
- Gradient norms should stay < 20
- No NaN values after first step
- Memory usage should stabilize

## Post-Training

Models are saved to `checkpoints/personality_name_timestamp/` with:
- Model weights
- Tokenizer files
- Training results JSON
- Configuration used

## Emergency Recovery

If training fails, partial checkpoints may be saved to `{output_dir}_partial`.
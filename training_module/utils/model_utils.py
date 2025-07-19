"""
Model utilities for saving, loading, and managing models.
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


def count_parameters(model: PreTrainedModel) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: PreTrainedModel) -> float:
    """
    Get the approximate size of a model in megabytes.
    
    Args:
        model: The model to measure
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_directory: str,
    save_config: bool = True,
    create_model_card: bool = True
) -> None:
    """
    Save a trained model and tokenizer.
    
    Args:
        model: The trained model to save
        tokenizer: The tokenizer to save
        save_directory: Directory to save the model
        save_config: Whether to save the model configuration
        create_model_card: Whether to create a model card
    """
    logger.info(f"💾 Saving model to {save_directory}")
    
    # Create directory if it doesn't exist
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(save_directory, safe_serialization=True)
    tokenizer.save_pretrained(save_directory)
    
    # Save additional metadata
    metadata = {
        "model_type": model.config.model_type,
        "num_parameters": count_parameters(model),
        "model_size_mb": get_model_size_mb(model),
        "vocab_size": len(tokenizer),
    }
    
    metadata_path = Path(save_directory) / "model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create simple model card if requested
    if create_model_card:
        create_model_card_file(save_directory, metadata)
    
    logger.info(f"✅ Model saved successfully")
    logger.info(f"   Parameters: {metadata['num_parameters']:,}")
    logger.info(f"   Size: {metadata['model_size_mb']:.1f} MB")


def load_model(
    model_path: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a saved model and tokenizer.
    
    Args:
        model_path: Path to the saved model directory
        device: Device to load the model on
        torch_dtype: Torch data type for the model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"📂 Loading model from {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect dtype if not specified
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None
        )
        
        # Move to device if needed
        if device == "cpu" or not hasattr(model, 'hf_device_map'):
            model = model.to(device)
        
        logger.info(f"✅ Model loaded successfully on {device}")
        
        # Load and log metadata if available
        metadata_path = Path(model_path) / "model_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"   Parameters: {metadata.get('num_parameters', 'Unknown'):,}")
            logger.info(f"   Size: {metadata.get('model_size_mb', 'Unknown')} MB")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def create_model_card_file(save_directory: str, metadata: Dict[str, Any]) -> None:
    """
    Create a simple model card file.
    
    Args:
        save_directory: Directory where the model is saved
        metadata: Model metadata dictionary
    """
    model_card_content = f"""# DeepDiscord Model

This is a fine-tuned conversational AI model based on DialoGPT for Discord-style conversations.

## Model Details

- **Model Type**: {metadata.get('model_type', 'Unknown')}
- **Parameters**: {metadata.get('num_parameters', 'Unknown'):,}
- **Model Size**: {metadata.get('model_size_mb', 'Unknown')} MB
- **Vocabulary Size**: {metadata.get('vocab_size', 'Unknown')}

## Training

This model was fine-tuned on Discord conversation data to generate contextually appropriate responses.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForCausalLM.from_pretrained("./")

# Generate response
input_text = "Hello, how are you?"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- This model is trained specifically for Discord-style conversations
- It may not perform well on other types of text generation tasks
- Responses should be reviewed for appropriateness before use

## License

Please ensure you have appropriate permissions for any training data used.
"""
    
    model_card_path = Path(save_directory) / "README.md"
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a saved model without loading it.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Dictionary with model information
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    info = {
        "path": str(model_path),
        "exists": True,
        "files": list(model_path.iterdir())
    }
    
    # Check for config file
    config_path = model_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        info["config"] = config
    
    # Check for metadata file
    metadata_path = model_path / "model_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        info["metadata"] = metadata
    
    # Check file sizes
    info["file_sizes"] = {}
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            info["file_sizes"][file_path.name] = file_path.stat().st_size
    
    return info


def optimize_model_for_inference(model: PreTrainedModel) -> PreTrainedModel:
    """
    Optimize a model for inference by applying various optimizations.
    
    Args:
        model: The model to optimize
        
    Returns:
        Optimized model
    """
    logger.info("⚡ Optimizing model for inference")
    
    # Set to evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Try to compile with torch.compile if available (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            logger.info("✅ Model compiled with torch.compile")
    except Exception as e:
        logger.warning(f"Could not compile model: {e}")
    
    logger.info("✅ Model optimization complete")
    return model


def calculate_model_memory_usage(model: PreTrainedModel, batch_size: int = 1, sequence_length: int = 512) -> Dict[str, float]:
    """
    Calculate approximate memory usage for a model.
    
    Args:
        model: The model to analyze
        batch_size: Batch size for calculation
        sequence_length: Sequence length for calculation
        
    Returns:
        Dictionary with memory usage estimates in MB
    """
    # Model parameters memory
    param_memory = get_model_size_mb(model)
    
    # Activation memory (rough estimate)
    # This is a simplified calculation
    hidden_size = getattr(model.config, 'hidden_size', 768)
    num_layers = getattr(model.config, 'num_hidden_layers', 12)
    
    # Rough estimate of activation memory per token
    activation_per_token = hidden_size * num_layers * 4  # 4 bytes per float32
    total_activation_mb = (activation_per_token * batch_size * sequence_length) / (1024**2)
    
    # Gradient memory (during training)
    gradient_memory = param_memory  # Gradients same size as parameters
    
    return {
        "parameters_mb": param_memory,
        "activations_mb": total_activation_mb,
        "gradients_mb": gradient_memory,
        "total_inference_mb": param_memory + total_activation_mb,
        "total_training_mb": param_memory + total_activation_mb + gradient_memory
    }
"""
QLoRA (Quantized Low-Rank Adaptation) configuration for efficient fine-tuning.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training - optimized for Configuration B (12-14GB VRAM)."""
    
    # 4-bit quantization settings
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4 - best for training
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16  # Use bfloat16 for stability
    
    # LoRA adapter settings - Configuration B balanced
    lora_r: int = 32  # Rank - higher = more capacity but more memory
    lora_alpha: int = 64  # Scaling factor - typically 2x rank
    lora_dropout: float = 0.1  # Dropout for regularization
    
    # Target modules for Llama-3 architecture
    lora_target_modules: List[str] = None
    
    # Advanced settings
    bias: str = "none"  # Don't adapt bias terms
    task_type: str = "CAUSAL_LM"  # Causal language modeling
    
    def __post_init__(self):
        """Set default target modules for Llama-3 if not specified."""
        if self.lora_target_modules is None:
            # All linear layers in Llama-3 for maximum adaptation capability
            self.lora_target_modules = [
                "q_proj",      # Query projection
                "k_proj",      # Key projection  
                "v_proj",      # Value projection
                "o_proj",      # Output projection
                "gate_proj",   # Gate projection (MLP)
                "up_proj",     # Up projection (MLP)
                "down_proj",   # Down projection (MLP)
            ]
    
    def get_bnb_config(self):
        """Get BitsAndBytesConfig for quantization."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers[torch] required for quantization. Install with: pip install transformers[torch]")
        
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
        )
    
    def get_lora_config(self):
        """Get LoraConfig for PEFT adaptation."""
        try:
            from peft import LoraConfig, TaskType
        except ImportError:
            raise ImportError("peft required for LoRA. Install with: pip install peft")
        
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )
    
    def get_memory_estimate(self, model_size_gb: float = 8.0) -> dict:
        """
        Estimate memory usage for training.
        
        Args:
            model_size_gb: Base model size in GB (8GB for Llama-3-8B)
            
        Returns:
            Dict with memory estimates in GB
        """
        # 4-bit quantization reduces model memory by ~75%
        quantized_model_gb = model_size_gb * 0.25
        
        # LoRA adapters are tiny compared to full model
        lora_params_gb = 0.1  # Typically <100MB for rank 32
        
        # Optimizer states (Adam) - only for LoRA parameters with 8-bit optimizer
        optimizer_gb = lora_params_gb * 0.5  # 8-bit optimizer reduces by ~50%
        
        # Gradients - only for LoRA parameters
        gradients_gb = lora_params_gb
        
        # Activations depend on batch size and sequence length
        # Rough estimate for batch_size=4, seq_len=2048
        activations_gb = 2.0
        
        total_gb = quantized_model_gb + lora_params_gb + optimizer_gb + gradients_gb + activations_gb
        
        return {
            "quantized_model_gb": quantized_model_gb,
            "lora_adapters_gb": lora_params_gb,
            "optimizer_states_gb": optimizer_gb,
            "gradients_gb": gradients_gb,
            "activations_gb": activations_gb,
            "total_estimated_gb": total_gb,
            "safety_margin_gb": 16.0 - total_gb,  # Assuming 16GB VRAM
            "fits_in_vram": total_gb < 15.0  # Leave 1GB buffer
        }


@dataclass 
class UnslothConfig:
    """Configuration for Unsloth optimization framework."""
    
    # Unsloth specific settings
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None  # Auto-detect optimal dtype
    load_in_4bit: bool = True
    
    # Flash Attention settings
    use_flash_attention: bool = True
    
    # Optimization flags
    use_gradient_checkpointing: bool = True
    use_reentrant_checkpointing: bool = False  # More memory efficient
    
    def __post_init__(self):
        """Validate Unsloth configuration."""
        if self.max_seq_length > 4096:
            print("Warning: Very long sequences may cause memory issues")
        
        # Auto-detect optimal dtype
        if self.dtype is None:
            if torch.cuda.is_available():
                # Use bfloat16 if supported, else float16
                if torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
            else:
                self.dtype = torch.float32


# Default configuration for Discord training
DEFAULT_QLORA_CONFIG = QLoRAConfig()
DEFAULT_UNSLOTH_CONFIG = UnslothConfig()
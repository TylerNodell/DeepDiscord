"""
Training configuration settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for model training using Configuration B: Balanced (12-14GB VRAM)."""
    
    # Training hyperparameters - Configuration B optimized for RTX 5080
    learning_rate: float = 2e-4
    batch_size: int = 4  # per_device_train_batch_size
    epochs: int = 3  # Conservative for fine-tuning
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_steps: int = 1500  # Override epochs if specified
    
    # Model parameters - Dolphin-2.9-Llama-3-8B
    max_length: int = 2048  # Increased for longer conversations
    model_name: str = "cognitivecomputations/dolphin-2.9-llama3-8b"
    
    # QLoRA Configuration - Configuration B balanced settings
    use_qlora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set in __post_init__
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True  # Better for training stability
    optim: str = "paged_adamw_8bit"  # Memory efficient optimizer
    group_by_length: bool = True  # Optimize batching
    dataloader_num_workers: int = 0  # Avoid memory overhead
    dataloader_pin_memory: bool = False
    
    # Data paths
    data_dir: str = "../discord_bot/results"  # Points to discord_bot/results where training data is stored
    output_dir: str = "./checkpoints"
    cache_dir: str = "./cache"
    
    # Training options
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Evaluation
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Hardware
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Optionals
    wandb_project: Optional[str] = "deepdiscord-training"
    run_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Either epochs or max_steps must be positive")
        
        # Set LoRA target modules for Llama-3 architecture if not specified
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        # Validate memory settings
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16, choose one")
        
        # Set reasonable limits for QLoRA
        if self.use_qlora:
            if self.lora_r > 128:
                raise ValueError("LoRA rank too high for stable training")
            if self.lora_alpha < self.lora_r:
                raise ValueError("LoRA alpha should be >= LoRA rank")
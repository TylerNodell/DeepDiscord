"""
Training configuration settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Model parameters
    max_length: int = 512
    model_name: str = "microsoft/DialoGPT-medium"
    
    # Data paths
    data_dir: str = "../results"
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
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
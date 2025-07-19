"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_training_logger(
    name: str = "training",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for training/evaluation with both file and console output.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Default format with emojis for readability
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


class TrainingLogger:
    """
    A specialized logger for training progress with metrics tracking.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_history = []
        self.current_epoch = 0
        self.current_step = 0
    
    def log_training_start(self, total_epochs: int, total_steps: int, config: dict):
        """Log training start information."""
        self.logger.info("🚀 Training Started")
        self.logger.info(f"   📊 Total Epochs: {total_epochs}")
        self.logger.info(f"   📈 Total Steps: {total_steps}")
        self.logger.info(f"   ⚙️  Configuration: {config}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.current_epoch = epoch
        self.logger.info(f"📅 Epoch {epoch}/{total_epochs} Started")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log epoch end with metrics."""
        self.logger.info(f"📅 Epoch {epoch} Completed")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {metric}: {value:.4f}")
            else:
                self.logger.info(f"   {metric}: {value}")
        
        # Store metrics
        epoch_metrics = {"epoch": epoch, **metrics}
        self.metrics_history.append(epoch_metrics)
    
    def log_step(self, step: int, loss: float, learning_rate: float = None):
        """Log training step."""
        self.current_step = step
        log_msg = f"📈 Step {step} | Loss: {loss:.4f}"
        if learning_rate is not None:
            log_msg += f" | LR: {learning_rate:.2e}"
        self.logger.info(log_msg)
    
    def log_validation(self, metrics: dict):
        """Log validation metrics."""
        self.logger.info("🧪 Validation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {metric}: {value:.4f}")
            else:
                self.logger.info(f"   {metric}: {value}")
    
    def log_model_save(self, save_path: str, step: int = None):
        """Log model save event."""
        msg = f"💾 Model saved to {save_path}"
        if step is not None:
            msg += f" at step {step}"
        self.logger.info(msg)
    
    def log_training_complete(self, total_time: float, final_metrics: dict = None):
        """Log training completion."""
        self.logger.info("🎉 Training Completed")
        self.logger.info(f"   ⏰ Total Time: {total_time:.2f} seconds")
        
        if final_metrics:
            self.logger.info("   📊 Final Metrics:")
            for metric, value in final_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"      {metric}: {value:.4f}")
                else:
                    self.logger.info(f"      {metric}: {value}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log training error."""
        error_msg = f"❌ Error"
        if context:
            error_msg += f" in {context}"
        error_msg += f": {error}"
        self.logger.error(error_msg)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(f"⚠️  {message}")
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(f"ℹ️  {message}")
    
    def get_metrics_summary(self) -> dict:
        """Get summary of training metrics."""
        if not self.metrics_history:
            return {}
        
        # Get best metrics
        best_loss = min(m.get("train_loss", float('inf')) for m in self.metrics_history)
        best_val_loss = min(m.get("eval_loss", float('inf')) for m in self.metrics_history)
        
        return {
            "total_epochs": len(self.metrics_history),
            "current_step": self.current_step,
            "best_train_loss": best_loss,
            "best_val_loss": best_val_loss,
            "metrics_history": self.metrics_history
        }


class ExperimentLogger:
    """
    Logger for tracking experiments and hyperparameter sweeps.
    """
    
    def __init__(self, experiment_name: str, logger: logging.Logger):
        self.experiment_name = experiment_name
        self.logger = logger
        self.experiments = []
    
    def start_experiment(self, config: dict, run_name: str = None):
        """Start a new experiment run."""
        run_name = run_name or f"run_{len(self.experiments) + 1}"
        
        experiment = {
            "name": run_name,
            "config": config,
            "start_time": None,
            "end_time": None,
            "metrics": {},
            "status": "running"
        }
        
        self.experiments.append(experiment)
        
        self.logger.info(f"🧪 Starting Experiment: {self.experiment_name}/{run_name}")
        self.logger.info(f"   ⚙️  Config: {config}")
        
        return len(self.experiments) - 1  # Return experiment index
    
    def end_experiment(self, experiment_idx: int, final_metrics: dict, status: str = "completed"):
        """End an experiment run."""
        if experiment_idx < len(self.experiments):
            exp = self.experiments[experiment_idx]
            exp["metrics"] = final_metrics
            exp["status"] = status
            
            self.logger.info(f"🏁 Experiment {exp['name']} {status}")
            self.logger.info(f"   📊 Final Metrics: {final_metrics}")
    
    def log_experiment_summary(self):
        """Log summary of all experiments."""
        self.logger.info(f"📋 Experiment Summary: {self.experiment_name}")
        self.logger.info(f"   Total Runs: {len(self.experiments)}")
        
        for i, exp in enumerate(self.experiments):
            self.logger.info(f"   Run {i+1}: {exp['name']} - {exp['status']}")
            if exp['metrics']:
                for metric, value in exp['metrics'].items():
                    if isinstance(value, float):
                        self.logger.info(f"      {metric}: {value:.4f}")


def setup_distributed_logging(rank: int = 0, world_size: int = 1) -> logging.Logger:
    """
    Set up logging for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        
    Returns:
        Logger configured for distributed training
    """
    # Only log from rank 0 to avoid duplicate messages
    if rank == 0:
        return setup_training_logger("distributed_training")
    else:
        # Create a null logger for other ranks
        logger = logging.getLogger(f"distributed_training_rank_{rank}")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger
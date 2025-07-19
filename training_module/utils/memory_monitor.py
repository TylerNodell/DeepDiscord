"""
Memory monitoring and optimization utilities for GPU training.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import psutil


logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    gpu_allocated: float  # GB
    gpu_reserved: float   # GB
    gpu_free: float      # GB
    cpu_percent: float
    cpu_memory_gb: float
    stage: str = "unknown"


class MemoryMonitor:
    """
    Real-time memory monitoring for training.
    Helps identify memory bottlenecks and prevent OOM errors.
    """
    
    def __init__(self, 
                 sample_interval: float = 5.0,
                 auto_cleanup: bool = True,
                 oom_callback: Optional[Callable] = None):
        self.sample_interval = sample_interval
        self.auto_cleanup = auto_cleanup
        self.oom_callback = oom_callback
        
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # GPU info
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.gpu_total_memory = 0
            self.gpu_name = "No GPU"
        
        logger.info(f"🖥️  Memory Monitor initialized for {self.gpu_name}")
        if self.gpu_available:
            logger.info(f"   Total GPU memory: {self.gpu_total_memory:.1f} GB")
    
    def get_current_memory(self, stage: str = "unknown") -> MemorySnapshot:
        """Get current memory usage snapshot."""
        
        if self.gpu_available:
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_free = self.gpu_total_memory - gpu_reserved
        else:
            gpu_allocated = gpu_reserved = gpu_free = 0
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_percent = cpu_memory.percent
        cpu_memory_gb = cpu_memory.used / 1024**3
        
        return MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_free=gpu_free,
            cpu_percent=cpu_percent,
            cpu_memory_gb=cpu_memory_gb,
            stage=stage
        )
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("📊 Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self.get_current_memory("background")
                self.snapshots.append(snapshot)
                
                # Check for potential OOM
                if self.gpu_available and snapshot.gpu_free < 1.0:  # Less than 1GB free
                    logger.warning(f"⚠️ Low GPU memory: {snapshot.gpu_free:.1f} GB free")
                    if self.auto_cleanup:
                        self.emergency_cleanup()
                    
                    if self.oom_callback:
                        self.oom_callback(snapshot)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                break
    
    def log_memory_status(self, stage: str = "current"):
        """Log current memory status."""
        snapshot = self.get_current_memory(stage)
        
        if self.gpu_available:
            logger.info(f"💾 Memory [{stage}]:")
            logger.info(f"   GPU Allocated: {snapshot.gpu_allocated:.1f} GB")
            logger.info(f"   GPU Reserved: {snapshot.gpu_reserved:.1f} GB")
            logger.info(f"   GPU Free: {snapshot.gpu_free:.1f} GB")
            logger.info(f"   GPU Utilization: {(snapshot.gpu_reserved/self.gpu_total_memory)*100:.1f}%")
        
        logger.info(f"   CPU Memory: {snapshot.cpu_memory_gb:.1f} GB ({snapshot.cpu_percent:.1f}%)")
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        logger.info("🧹 Performing emergency memory cleanup...")
        
        if self.gpu_available:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log results
            self.log_memory_status("after_cleanup")
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage from snapshots."""
        if not self.snapshots:
            return {}
        
        peak_gpu_allocated = max(s.gpu_allocated for s in self.snapshots)
        peak_gpu_reserved = max(s.gpu_reserved for s in self.snapshots)
        peak_cpu_memory = max(s.cpu_memory_gb for s in self.snapshots)
        peak_cpu_percent = max(s.cpu_percent for s in self.snapshots)
        
        return {
            "peak_gpu_allocated_gb": peak_gpu_allocated,
            "peak_gpu_reserved_gb": peak_gpu_reserved,
            "peak_cpu_memory_gb": peak_cpu_memory,
            "peak_cpu_percent": peak_cpu_percent,
            "gpu_efficiency": peak_gpu_allocated / peak_gpu_reserved if peak_gpu_reserved > 0 else 0
        }
    
    def get_memory_timeline(self, stage_filter: Optional[str] = None) -> List[MemorySnapshot]:
        """Get memory usage timeline, optionally filtered by stage."""
        if stage_filter:
            return [s for s in self.snapshots if s.stage == stage_filter]
        return self.snapshots.copy()
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager to track memory usage for a specific stage."""
        logger.info(f"🎯 Entering stage: {stage_name}")
        start_snapshot = self.get_current_memory(f"{stage_name}_start")
        self.snapshots.append(start_snapshot)
        
        try:
            yield
        finally:
            end_snapshot = self.get_current_memory(f"{stage_name}_end")
            self.snapshots.append(end_snapshot)
            
            # Log memory change
            gpu_change = end_snapshot.gpu_allocated - start_snapshot.gpu_allocated
            cpu_change = end_snapshot.cpu_memory_gb - start_snapshot.cpu_memory_gb
            
            logger.info(f"🎯 Completed stage: {stage_name}")
            logger.info(f"   GPU memory change: {gpu_change:+.1f} GB")
            logger.info(f"   CPU memory change: {cpu_change:+.1f} GB")


class MemoryOptimizer:
    """
    Memory optimization utilities for training.
    """
    
    @staticmethod
    def optimize_torch_settings():
        """Apply PyTorch memory optimizations."""
        logger.info("🔧 Applying PyTorch memory optimizations...")
        
        # Set memory allocation strategy
        if torch.cuda.is_available():
            # Use memory pool for more efficient allocation
            torch.cuda.empty_cache()
            
            # Set memory fraction to leave some headroom
            # torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable memory mapping for large models
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("✅ CUDA optimizations applied")
    
    @staticmethod
    def get_optimal_batch_size(model, tokenizer, sample_input: str, 
                              max_batch_size: int = 32, 
                              max_length: int = 2048) -> int:
        """
        Find optimal batch size that fits in memory.
        
        Args:
            model: The model to test
            tokenizer: Tokenizer for encoding
            sample_input: Sample input text
            max_batch_size: Maximum batch size to test
            max_length: Maximum sequence length
            
        Returns:
            Optimal batch size
        """
        logger.info("🔍 Finding optimal batch size...")
        
        if not torch.cuda.is_available():
            logger.warning("No GPU available, using batch size 1")
            return 1
        
        # Encode sample input
        encoded = tokenizer(
            sample_input, 
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        optimal_batch_size = 1
        
        for batch_size in [1, 2, 4, 8, 16, min(32, max_batch_size)]:
            try:
                # Create batch
                batch = {
                    key: tensor.repeat(batch_size, 1).cuda() 
                    for key, tensor in encoded.items()
                }
                
                # Test forward pass
                with torch.no_grad():
                    outputs = model(**batch)
                
                # Test backward pass (approximate)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.mean()
                loss.backward()
                
                # Clear gradients
                model.zero_grad()
                torch.cuda.empty_cache()
                
                optimal_batch_size = batch_size
                logger.info(f"✅ Batch size {batch_size} fits in memory")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"❌ Batch size {batch_size} causes OOM")
                    break
                else:
                    raise e
        
        logger.info(f"🎯 Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    @staticmethod
    def apply_gradient_checkpointing(model):
        """Apply gradient checkpointing to reduce memory usage."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        else:
            logger.warning("⚠️ Model doesn't support gradient checkpointing")
    
    @staticmethod
    def setup_memory_efficient_attention():
        """Set up memory efficient attention if available."""
        try:
            # Try to enable Flash Attention
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("✅ Flash Attention enabled")
        except:
            logger.warning("⚠️ Flash Attention not available")
        
        # Enable memory efficient attention
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("✅ Memory efficient attention enabled")
        except:
            logger.warning("⚠️ Memory efficient attention not available")


def create_memory_efficient_dataloader(dataset, batch_size: int, **kwargs):
    """Create memory efficient DataLoader."""
    from torch.utils.data import DataLoader
    
    # Memory efficient settings
    efficient_kwargs = {
        'batch_size': batch_size,
        'pin_memory': False,  # Disable for lower memory usage
        'num_workers': 0,     # Single process to avoid memory overhead
        'persistent_workers': False,
        **kwargs
    }
    
    return DataLoader(dataset, **efficient_kwargs)


# Global memory monitor instance
global_memory_monitor = MemoryMonitor()


def log_memory_usage(stage: str):
    """Decorator to log memory usage around function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with global_memory_monitor.stage(stage):
                return func(*args, **kwargs)
        return wrapper
    return decorator
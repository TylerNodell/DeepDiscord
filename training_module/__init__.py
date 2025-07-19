"""
DeepDiscord Training Module

This module contains all training-related functionality for the DeepDiscord project,
including model training, data preprocessing, and evaluation utilities.
"""

__version__ = "1.0.0"
__author__ = "DeepDiscord Team"

# Module imports
from .utils import data_preprocessing, model_utils
from .models import trainer, evaluator

__all__ = [
    'data_preprocessing',
    'model_utils', 
    'trainer',
    'evaluator'
]
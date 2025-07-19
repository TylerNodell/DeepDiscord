"""
Data preprocessing utilities for training data preparation.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for training.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace problematic Unicode escape sequences
    text = text.replace('\\ud83c\\udfb8', '🎸')
    text = text.replace('\\u2019', "'")
    text = text.replace('\\u201c', '"')
    text = text.replace('\\u201d', '"')
    text = text.replace('\\u2013', '-')
    text = text.replace('\\u2014', '--')
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove empty lines and normalize line breaks
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    return text


def filter_conversation_pair(input_text: str, output_text: str, min_length: int = 5, max_length: int = 500) -> bool:
    """
    Filter conversation pairs based on quality criteria.
    
    Args:
        input_text: Input message
        output_text: Response message
        min_length: Minimum character length
        max_length: Maximum character length
        
    Returns:
        True if pair should be kept, False otherwise
    """
    # Check length constraints
    if len(input_text) < min_length or len(output_text) < min_length:
        return False
    
    if len(input_text) > max_length or len(output_text) > max_length:
        return False
    
    # Filter out very repetitive content
    if len(set(input_text.split())) < 3 or len(set(output_text.split())) < 3:
        return False
    
    # Filter out messages that are mostly non-alphabetic
    alpha_ratio_input = sum(c.isalpha() for c in input_text) / len(input_text) if input_text else 0
    alpha_ratio_output = sum(c.isalpha() for c in output_text) / len(output_text) if output_text else 0
    
    if alpha_ratio_input < 0.5 or alpha_ratio_output < 0.5:
        return False
    
    # Filter out common bot-like responses
    bot_phrases = [
        "i don't understand",
        "i can't help",
        "please try again",
        "error occurred",
        "something went wrong"
    ]
    
    output_lower = output_text.lower()
    if any(phrase in output_lower for phrase in bot_phrases):
        return False
    
    return True


def create_conversation_context(messages: List[Dict], max_context: int = 3) -> str:
    """
    Create conversation context from message history.
    
    Args:
        messages: List of message dictionaries with 'content' and 'author' keys
        max_context: Maximum number of previous messages to include
        
    Returns:
        Formatted conversation context
    """
    if not messages:
        return ""
    
    # Take last max_context messages
    recent_messages = messages[-max_context:]
    
    context_parts = []
    for msg in recent_messages:
        author = msg.get('author', 'User')
        content = clean_text(msg.get('content', ''))
        if content:
            context_parts.append(f"{author}: {content}")
    
    return "\n".join(context_parts)


def tokenize_conversation(
    input_text: str,
    output_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a conversation pair for training.
    
    Args:
        input_text: Input message
        output_text: Expected response
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs
    """
    # Format as conversation
    conversation = f"{input_text}{tokenizer.eos_token}{output_text}{tokenizer.eos_token}"
    
    # Tokenize
    encoded = tokenizer(
        conversation,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal LM)
    labels = encoded["input_ids"].clone()
    
    # Mask the input tokens in labels (only learn to predict the response)
    input_encoded = tokenizer(
        input_text + tokenizer.eos_token,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt"
    )
    
    input_length = input_encoded["input_ids"].shape[1]
    labels[0, :input_length] = -100  # Ignore loss for input tokens
    
    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0)
    }


def preprocess_training_data(
    raw_data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    min_quality_score: float = 0.7
) -> Dict[str, List]:
    """
    Preprocess raw conversation data for training.
    
    Args:
        raw_data: List of conversation dictionaries
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        min_quality_score: Minimum quality threshold
        
    Returns:
        Dictionary with preprocessed data
    """
    logger.info(f"🔄 Preprocessing {len(raw_data)} conversation pairs")
    
    processed_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    filtered_count = 0
    
    for i, conversation in enumerate(raw_data):
        if i % 1000 == 0:
            logger.info(f"📊 Processed {i}/{len(raw_data)} conversations")
        
        try:
            input_text = clean_text(conversation.get("input", ""))
            output_text = clean_text(conversation.get("output", ""))
            
            # Filter based on quality
            if not filter_conversation_pair(input_text, output_text):
                filtered_count += 1
                continue
            
            # Tokenize conversation
            tokenized = tokenize_conversation(input_text, output_text, tokenizer, max_length)
            
            # Add to processed data
            processed_data["input_ids"].append(tokenized["input_ids"])
            processed_data["attention_mask"].append(tokenized["attention_mask"])
            processed_data["labels"].append(tokenized["labels"])
            
        except Exception as e:
            logger.warning(f"Error processing conversation {i}: {e}")
            filtered_count += 1
            continue
    
    logger.info(f"✅ Preprocessing complete: {len(processed_data['input_ids'])} valid pairs")
    logger.info(f"🗑️  Filtered out: {filtered_count} low-quality pairs")
    
    return processed_data


def load_json_conversations(file_path: str) -> List[Dict]:
    """
    Load conversations from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of conversation dictionaries
    """
    logger.info(f"📁 Loading conversations from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    elif isinstance(data, list):
        conversations = data
    else:
        raise ValueError("Invalid JSON format. Expected list or dict with 'conversations' key")
    
    logger.info(f"📊 Loaded {len(conversations)} conversations")
    return conversations


def save_processed_data(data: Dict[str, List], output_path: str):
    """
    Save processed data to disk.
    
    Args:
        data: Processed data dictionary
        output_path: Output file path
    """
    logger.info(f"💾 Saving processed data to {output_path}")
    
    # Convert tensors to lists for JSON serialization
    serializable_data = {}
    for key, tensors in data.items():
        serializable_data[key] = [tensor.tolist() for tensor in tensors]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)
    
    logger.info("✅ Data saved successfully")


def create_dataset_split(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data: List of conversation data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    import random
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Calculate split indices
    total_size = len(data_copy)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split data
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:train_size + val_size]
    test_data = data_copy[train_size + val_size:]
    
    logger.info(f"📊 Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


def analyze_dataset_statistics(data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze statistics of the dataset.
    
    Args:
        data: List of conversation data
        
    Returns:
        Dictionary with dataset statistics
    """
    if not data:
        return {}
    
    input_lengths = []
    output_lengths = []
    total_chars = 0
    
    for conversation in data:
        input_text = conversation.get("input", "")
        output_text = conversation.get("output", "")
        
        input_lengths.append(len(input_text))
        output_lengths.append(len(output_text))
        total_chars += len(input_text) + len(output_text)
    
    stats = {
        "total_conversations": len(data),
        "avg_input_length": sum(input_lengths) / len(input_lengths),
        "avg_output_length": sum(output_lengths) / len(output_lengths),
        "max_input_length": max(input_lengths),
        "max_output_length": max(output_lengths),
        "min_input_length": min(input_lengths),
        "min_output_length": min(output_lengths),
        "total_characters": total_chars
    }
    
    logger.info("📈 Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")
    
    return stats
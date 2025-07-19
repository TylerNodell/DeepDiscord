"""
Discord-specific data preprocessing for training data.
Based on the reference guide for optimal Discord conversation handling.
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_discord_data(text: str) -> str:
    """
    Clean Discord-specific formatting from text.
    
    Args:
        text: Raw Discord message text
        
    Returns:
        Cleaned text suitable for training
    """
    if not text:
        return ""
    
    # Remove user mentions (@user or <@userid>)
    text = re.sub(r'<@!?\d+>', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Remove channel mentions (#channel or <#channelid>)
    text = re.sub(r'<#\d+>', '', text)
    text = re.sub(r'#[\w-]+', '', text)
    
    # Remove role mentions (<@&roleid>)
    text = re.sub(r'<@&\d+>', '', text)
    
    # Remove custom emojis (<:name:id> or <a:name:id>)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    
    # Remove timestamp mentions (<t:timestamp>)
    text = re.sub(r'<t:\d+(?::[tTdDfFR])?>', '', text)
    
    # Clean up common Discord markdown (keep basic formatting)
    # Remove triple backticks but keep content
    text = re.sub(r'```(?:\w+\n)?(.*?)```', r'\1', text, flags=re.DOTALL)
    
    # Remove single backticks (inline code)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Clean excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def filter_training_pairs(pairs: List[Dict[str, Any]], 
                         min_length: int = 10,
                         max_length: int = 500,
                         include_gifs: bool = True,
                         include_links: bool = True) -> List[Dict[str, Any]]:
    """
    Filter and clean Discord training pairs for optimal training quality.
    
    Args:
        pairs: List of conversation pairs from Discord bot
        min_length: Minimum response length
        max_length: Maximum response length
        include_gifs: Whether to include GIF responses
        include_links: Whether to include responses with links
        
    Returns:
        Filtered and cleaned training pairs
    """
    filtered = []
    stats = {
        "original_count": len(pairs),
        "too_short": 0,
        "too_long": 0,
        "empty_after_cleaning": 0,
        "gif_responses": 0,
        "link_responses": 0,
        "final_count": 0
    }
    
    for pair in pairs:
        # Extract and clean question/answer
        question = clean_discord_data(pair.get('question', ''))
        answer = clean_discord_data(pair.get('answer', ''))
        
        # Skip if empty after cleaning
        if not question or not answer:
            stats["empty_after_cleaning"] += 1
            continue
        
        # Check length constraints
        if len(answer) < min_length:
            stats["too_short"] += 1
            continue
            
        if len(answer) > max_length:
            stats["too_long"] += 1
            continue
        
        # Check for GIFs (Tenor, GIPHY links)
        is_gif = bool(re.search(r'https?://(tenor\.com|giphy\.com|cdn\.discordapp\.com.*\.gif)', answer))
        if is_gif:
            stats["gif_responses"] += 1
            if not include_gifs:
                continue
        
        # Check for links
        has_links = bool(re.search(r'https?://\S+', answer))
        if has_links:
            stats["link_responses"] += 1
            if not include_links:
                continue
        
        # Preserve metadata and add cleaned content
        filtered_pair = {
            'question': question,
            'answer': answer,
            'metadata': pair.get('metadata', {}),
            'original_question': pair.get('question', ''),
            'original_answer': pair.get('answer', ''),
        }
        
        # Add content type flags
        filtered_pair['metadata']['is_gif'] = is_gif
        filtered_pair['metadata']['has_links'] = has_links
        filtered_pair['metadata']['cleaned_length'] = len(answer)
        
        filtered.append(filtered_pair)
    
    stats["final_count"] = len(filtered)
    
    logger.info(f"📊 Discord Data Filtering Results:")
    logger.info(f"   Original pairs: {stats['original_count']}")
    logger.info(f"   Too short (< {min_length}): {stats['too_short']}")
    logger.info(f"   Too long (> {max_length}): {stats['too_long']}")
    logger.info(f"   Empty after cleaning: {stats['empty_after_cleaning']}")
    logger.info(f"   GIF responses: {stats['gif_responses']}")
    logger.info(f"   Link responses: {stats['link_responses']}")
    logger.info(f"   Final pairs: {stats['final_count']}")
    logger.info(f"   Retention rate: {stats['final_count']/stats['original_count']*100:.1f}%")
    
    return filtered


def create_chat_dataset(discord_pairs: List[Dict[str, Any]], 
                       system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert Discord pairs to ChatML format for training.
    
    Args:
        discord_pairs: Filtered Discord conversation pairs
        system_prompt: Optional system prompt for the model
        
    Returns:
        Training data in ChatML format
    """
    training_data = []
    
    # Default system prompt for Discord bot personality
    if system_prompt is None:
        system_prompt = (
            "You are a helpful and engaging Discord bot. Respond naturally to conversations "
            "while being respectful and appropriate. Match the tone and style of Discord chat."
        )
    
    for pair in discord_pairs:
        # Create messages in ChatML format
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user question
        messages.append({
            "role": "user", 
            "content": pair['question']
        })
        
        # Add assistant response
        messages.append({
            "role": "assistant",
            "content": pair['answer']
        })
        
        entry = {
            "messages": messages,
            "metadata": pair.get('metadata', {})
        }
        
        training_data.append(entry)
    
    logger.info(f"✅ Created {len(training_data)} ChatML training examples")
    return training_data


def load_discord_training_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load Discord training data from ZIP or JSON files.
    
    Args:
        data_path: Path to Discord training data (ZIP or JSON)
        
    Returns:
        List of conversation pairs
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    all_pairs = []
    
    if data_path.suffix == '.zip':
        # Extract and process ZIP file
        import zipfile
        import tempfile
        
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                
                # Find JSON files in extracted content
                json_files = list(Path(temp_dir).rglob("*.json"))
                
                for json_file in json_files:
                    if json_file.name.endswith('_summary.json'):
                        continue  # Skip summary files
                    
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract training data based on format
                    if 'training_data' in data:
                        pairs = data['training_data']
                    elif isinstance(data, list):
                        pairs = data
                    else:
                        continue
                    
                    all_pairs.extend(pairs)
                    logger.info(f"📁 Loaded {len(pairs)} pairs from {json_file.name}")
    
    elif data_path.suffix == '.json':
        # Load single JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'training_data' in data:
            all_pairs = data['training_data']
        elif isinstance(data, list):
            all_pairs = data
        else:
            raise ValueError("Invalid JSON format for training data")
        
        logger.info(f"📁 Loaded {len(all_pairs)} pairs from {data_path.name}")
    
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    logger.info(f"📊 Total pairs loaded: {len(all_pairs)}")
    return all_pairs


def create_training_split(data: List[Dict[str, Any]], 
                         train_ratio: float = 0.9,
                         val_ratio: float = 0.1,
                         seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into training and validation sets.
    
    Args:
        data: Training data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data)
    """
    import random
    
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle data with seed
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Calculate split point
    train_size = int(len(data_copy) * train_ratio)
    
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:]
    
    logger.info(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}")
    
    return train_data, val_data


def analyze_dataset_quality(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality and characteristics of Discord training data.
    
    Args:
        pairs: Training pairs to analyze
        
    Returns:
        Analysis results
    """
    if not pairs:
        return {}
    
    # Length statistics
    question_lengths = [len(p['question']) for p in pairs]
    answer_lengths = [len(p['answer']) for p in pairs]
    
    # Content analysis
    gif_count = sum(1 for p in pairs if p.get('metadata', {}).get('is_gif', False))
    link_count = sum(1 for p in pairs if p.get('metadata', {}).get('has_links', False))
    
    # Confidence distribution (if available)
    confidences = []
    for p in pairs:
        conf = p.get('metadata', {}).get('confidence')
        if conf is not None:
            confidences.append(conf)
    
    analysis = {
        "total_pairs": len(pairs),
        "question_stats": {
            "avg_length": sum(question_lengths) / len(question_lengths),
            "min_length": min(question_lengths),
            "max_length": max(question_lengths),
        },
        "answer_stats": {
            "avg_length": sum(answer_lengths) / len(answer_lengths),
            "min_length": min(answer_lengths),
            "max_length": max(answer_lengths),
        },
        "content_types": {
            "gif_responses": gif_count,
            "link_responses": link_count,
            "text_only": len(pairs) - gif_count - link_count,
        },
        "confidence_stats": {
            "available": len(confidences),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "high_confidence": sum(1 for c in confidences if c >= 0.8),
            "medium_confidence": sum(1 for c in confidences if 0.5 <= c < 0.8),
            "low_confidence": sum(1 for c in confidences if c < 0.5),
        } if confidences else None
    }
    
    logger.info(f"📈 Dataset Quality Analysis:")
    logger.info(f"   Total pairs: {analysis['total_pairs']}")
    logger.info(f"   Avg question length: {analysis['question_stats']['avg_length']:.1f}")
    logger.info(f"   Avg answer length: {analysis['answer_stats']['avg_length']:.1f}")
    logger.info(f"   GIF responses: {gif_count}")
    logger.info(f"   Link responses: {link_count}")
    
    if analysis['confidence_stats']:
        logger.info(f"   Avg confidence: {analysis['confidence_stats']['avg_confidence']:.3f}")
        logger.info(f"   High confidence: {analysis['confidence_stats']['high_confidence']}")
    
    return analysis
"""
Data loading utilities for training and evaluation.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import glob

from ..config.data_config import DataConfig


logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for DeepDiscord training and evaluation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logger
    
    def load_training_data(self, data_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Load training data from files.
        
        Args:
            data_path: Optional override for data directory path
            
        Returns:
            List of conversation dictionaries with 'input' and 'output' keys
        """
        data_dir = data_path or self.config.data_dir
        self.logger.info(f"📁 Loading training data from {data_dir}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Look for training data files
        training_files = self._find_training_files(data_dir)
        
        if not training_files:
            raise FileNotFoundError(f"No training data files found in {data_dir}")
        
        self.logger.info(f"📊 Found {len(training_files)} training files")
        
        all_conversations = []
        
        for file_path in training_files:
            try:
                conversations = self._load_file(file_path)
                all_conversations.extend(conversations)
                self.logger.info(f"   📄 {os.path.basename(file_path)}: {len(conversations)} conversations")
            except Exception as e:
                self.logger.warning(f"   ❌ Failed to load {file_path}: {e}")
        
        self.logger.info(f"✅ Loaded {len(all_conversations)} total conversations")
        
        # Apply filtering if configured
        if hasattr(self.config, 'min_conversation_length'):
            filtered_conversations = self._filter_conversations(all_conversations)
            self.logger.info(f"🔍 After filtering: {len(filtered_conversations)} conversations")
            return filtered_conversations
        
        return all_conversations
    
    def load_test_data(self, test_path: str) -> List[Dict[str, str]]:
        """
        Load test data from a specific file.
        
        Args:
            test_path: Path to test data file
            
        Returns:
            List of test conversation dictionaries
        """
        self.logger.info(f"📁 Loading test data from {test_path}")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        conversations = self._load_file(test_path)
        self.logger.info(f"✅ Loaded {len(conversations)} test conversations")
        
        return conversations
    
    def _find_training_files(self, data_dir: str) -> List[str]:
        """Find all training data files in the directory."""
        training_files = []
        
        # Look for JSON files
        json_pattern = os.path.join(data_dir, "*.json")
        training_files.extend(glob.glob(json_pattern))
        
        # Look for ZIP files (from Discord bot generation)
        zip_pattern = os.path.join(data_dir, "training_data_*.zip")
        zip_files = glob.glob(zip_pattern)
        
        # Extract and process ZIP files
        for zip_file in zip_files:
            extracted_files = self._extract_zip_file(zip_file)
            training_files.extend(extracted_files)
        
        return sorted(training_files)
    
    def _extract_zip_file(self, zip_path: str) -> List[str]:
        """Extract training data from ZIP file."""
        import zipfile
        import tempfile
        
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create temporary directory for extraction
                temp_dir = tempfile.mkdtemp(prefix="deepdiscord_extract_")
                zip_ref.extractall(temp_dir)
                
                # Find JSON files in extracted content
                json_files = glob.glob(os.path.join(temp_dir, "**", "*.json"), recursive=True)
                extracted_files.extend(json_files)
                
                self.logger.info(f"📦 Extracted {len(json_files)} files from {os.path.basename(zip_path)}")
        
        except Exception as e:
            self.logger.warning(f"❌ Failed to extract {zip_path}: {e}")
        
        return extracted_files
    
    def _load_file(self, file_path: str) -> List[Dict[str, str]]:
        """Load conversations from a single file."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            return self._load_json_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_json_file(self, file_path: str) -> List[Dict[str, str]]:
        """Load conversations from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Direct list of conversations
            conversations = self._process_conversation_list(data)
        elif isinstance(data, dict):
            if "conversations" in data:
                # Wrapped in conversations key
                conversations = self._process_conversation_list(data["conversations"])
            elif "response_pairs" in data:
                # Discord bot format
                conversations = self._process_response_pairs(data["response_pairs"])
            elif "messages" in data:
                # Message history format
                conversations = self._process_message_history(data["messages"])
            else:
                # Try to treat the dict itself as a single conversation
                if "input" in data and "output" in data:
                    conversations = [data]
        
        return conversations
    
    def _process_conversation_list(self, conversation_list: List[Dict]) -> List[Dict[str, str]]:
        """Process a list of conversation dictionaries."""
        processed = []
        
        for conv in conversation_list:
            if isinstance(conv, dict):
                if "input" in conv and "output" in conv:
                    processed.append({
                        "input": str(conv["input"]),
                        "output": str(conv["output"])
                    })
                elif "prompt" in conv and "response" in conv:
                    processed.append({
                        "input": str(conv["prompt"]),
                        "output": str(conv["response"])
                    })
                elif "question" in conv and "answer" in conv:
                    processed.append({
                        "input": str(conv["question"]),
                        "output": str(conv["answer"])
                    })
        
        return processed
    
    def _process_response_pairs(self, response_pairs: List[Dict]) -> List[Dict[str, str]]:
        """Process Discord bot response pairs format."""
        conversations = []
        
        for pair in response_pairs:
            if "input" in pair and "response" in pair:
                conversations.append({
                    "input": str(pair["input"]),
                    "output": str(pair["response"])
                })
        
        return conversations
    
    def _process_message_history(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Process message history into conversation pairs."""
        conversations = []
        
        # Create pairs from consecutive messages
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            # Check if messages are from different authors
            current_author = current_msg.get("author", "")
            next_author = next_msg.get("author", "")
            
            if current_author != next_author:
                conversations.append({
                    "input": str(current_msg.get("content", "")),
                    "output": str(next_msg.get("content", ""))
                })
        
        return conversations
    
    def _filter_conversations(self, conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter conversations based on quality criteria."""
        filtered = []
        
        for conv in conversations:
            input_text = conv["input"]
            output_text = conv["output"]
            
            # Length filtering
            if len(input_text) < self.config.min_conversation_length:
                continue
            if len(output_text) < self.config.min_conversation_length:
                continue
            
            if hasattr(self.config, 'max_conversation_length'):
                if len(input_text) > self.config.max_conversation_length:
                    continue
                if len(output_text) > self.config.max_conversation_length:
                    continue
            
            # Quality filtering
            if self._is_low_quality(input_text) or self._is_low_quality(output_text):
                continue
            
            filtered.append(conv)
        
        return filtered
    
    def _is_low_quality(self, text: str) -> bool:
        """Check if text is low quality."""
        # Very short text
        if len(text.strip()) < 3:
            return True
        
        # Mostly non-alphabetic
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.3:
            return True
        
        # Very repetitive
        words = text.split()
        if len(words) > 3 and len(set(words)) < len(words) * 0.4:
            return True
        
        return False
    
    def save_conversations(self, conversations: List[Dict[str, str]], output_path: str):
        """Save conversations to a JSON file."""
        self.logger.info(f"💾 Saving {len(conversations)} conversations to {output_path}")
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data structure
        data = {
            "conversations": conversations,
            "metadata": {
                "total_conversations": len(conversations),
                "format": "input_output_pairs"
            }
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info("✅ Conversations saved successfully")
    
    def get_data_statistics(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if not conversations:
            return {}
        
        input_lengths = [len(conv["input"]) for conv in conversations]
        output_lengths = [len(conv["output"]) for conv in conversations]
        
        stats = {
            "total_conversations": len(conversations),
            "avg_input_length": sum(input_lengths) / len(input_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "max_input_length": max(input_lengths),
            "max_output_length": max(output_lengths),
            "min_input_length": min(input_lengths),
            "min_output_length": min(output_lengths)
        }
        
        return stats
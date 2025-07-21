#!/usr/bin/env python3
"""
Training Data Validator for DeepDiscord

Validates and optimizes training data for compatibility with the training scripts.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import re


class TrainingDataValidator:
    """Validates and fixes training data format for optimal training compatibility"""
    
    def __init__(self):
        self.validation_errors = []
        self.optimization_suggestions = []
        self.statistics = {}
    
    def validate_training_file(self, file_path: str) -> bool:
        """Validate a training data JSON file"""
        print(f"🔍 Validating training data: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check top-level structure
            if not self._validate_structure(data):
                return False
            
            # Check training pairs
            if not self._validate_training_pairs(data.get('training_data', [])):
                return False
            
            # Generate statistics
            self._generate_statistics(data)
            
            # Check for optimization opportunities
            self._check_optimizations(data)
            
            self._print_validation_results()
            return len(self.validation_errors) == 0
            
        except json.JSONDecodeError as e:
            self.validation_errors.append(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.validation_errors.append(f"Error reading file: {e}")
            return False
    
    def _validate_structure(self, data: Dict[str, Any]) -> bool:
        """Validate top-level data structure"""
        required_keys = ['metadata', 'training_data']
        
        for key in required_keys:
            if key not in data:
                self.validation_errors.append(f"Missing required key: {key}")
                return False
        
        # Validate metadata
        metadata = data['metadata']
        required_metadata = ['generated_at', 'target_user_id', 'total_pairs']
        
        for key in required_metadata:
            if key not in metadata:
                self.validation_errors.append(f"Missing metadata key: {key}")
        
        # Check if training_data is a list
        if not isinstance(data['training_data'], list):
            self.validation_errors.append("training_data must be a list")
            return False
        
        return True
    
    def _validate_training_pairs(self, training_pairs: List[Dict[str, Any]]) -> bool:
        """Validate individual training pairs"""
        if not training_pairs:
            self.validation_errors.append("No training pairs found")
            return False
        
        for i, pair in enumerate(training_pairs):
            # Check required fields
            if 'question' not in pair:
                self.validation_errors.append(f"Pair {i}: Missing 'question' field")
            elif not isinstance(pair['question'], str) or not pair['question'].strip():
                self.validation_errors.append(f"Pair {i}: Empty or invalid question")
            
            if 'answer' not in pair:
                self.validation_errors.append(f"Pair {i}: Missing 'answer' field")
            elif not isinstance(pair['answer'], str) or not pair['answer'].strip():
                self.validation_errors.append(f"Pair {i}: Empty or invalid answer")
            
            # Check for optimal training characteristics
            if 'question' in pair and len(pair['question']) > 2048:
                self.optimization_suggestions.append(f"Pair {i}: Question very long ({len(pair['question'])} chars) - consider truncating")
            
            if 'answer' in pair and len(pair['answer']) > 2048:
                self.optimization_suggestions.append(f"Pair {i}: Answer very long ({len(pair['answer'])} chars) - consider truncating")
            
            # Check for problematic content
            if 'question' in pair and self._has_problematic_content(pair['question']):
                self.optimization_suggestions.append(f"Pair {i}: Question may contain problematic content")
            
            if 'answer' in pair and self._has_problematic_content(pair['answer']):
                self.optimization_suggestions.append(f"Pair {i}: Answer may contain problematic content")
        
        return True
    
    def _has_problematic_content(self, text: str) -> bool:
        """Check for potentially problematic content"""
        # Check for excessive special characters that might confuse tokenization
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_char_ratio > 0.3:
            return True
        
        # Check for extremely long words (might be URLs or encoded data)
        words = text.split()
        for word in words:
            if len(word) > 100:
                return True
        
        return False
    
    def _generate_statistics(self, data: Dict[str, Any]) -> None:
        """Generate statistics about the training data"""
        training_pairs = data['training_data']
        
        self.statistics = {
            'total_pairs': len(training_pairs),
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'max_question_length': 0,
            'max_answer_length': 0,
            'confidence_distribution': {},
            'response_types': {},
            'very_long_sequences': 0,
            'empty_pairs': 0
        }
        
        question_lengths = []
        answer_lengths = []
        
        for pair in training_pairs:
            # Length statistics
            if 'question' in pair and pair['question']:
                q_len = len(pair['question'])
                question_lengths.append(q_len)
                self.statistics['max_question_length'] = max(self.statistics['max_question_length'], q_len)
                
                if q_len > 512:
                    self.statistics['very_long_sequences'] += 1
            
            if 'answer' in pair and pair['answer']:
                a_len = len(pair['answer'])
                answer_lengths.append(a_len)
                self.statistics['max_answer_length'] = max(self.statistics['max_answer_length'], a_len)
                
                if a_len > 512:
                    self.statistics['very_long_sequences'] += 1
            
            # Check for empty pairs
            if not pair.get('question', '').strip() or not pair.get('answer', '').strip():
                self.statistics['empty_pairs'] += 1
            
            # Confidence distribution
            if 'metadata' in pair and 'confidence' in pair['metadata']:
                confidence = pair['metadata']['confidence']
                conf_bucket = f"{confidence:.1f}"
                self.statistics['confidence_distribution'][conf_bucket] = self.statistics['confidence_distribution'].get(conf_bucket, 0) + 1
            
            # Response type distribution
            if 'metadata' in pair and 'response_type' in pair['metadata']:
                resp_type = pair['metadata']['response_type']
                self.statistics['response_types'][resp_type] = self.statistics['response_types'].get(resp_type, 0) + 1
        
        # Calculate averages
        if question_lengths:
            self.statistics['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        if answer_lengths:
            self.statistics['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
    
    def _check_optimizations(self, data: Dict[str, Any]) -> None:
        """Check for optimization opportunities"""
        # Check sequence lengths for training efficiency
        if self.statistics['very_long_sequences'] > 0:
            percentage = (self.statistics['very_long_sequences'] / self.statistics['total_pairs']) * 100
            self.optimization_suggestions.append(f"{self.statistics['very_long_sequences']} pairs ({percentage:.1f}%) have sequences >512 chars - consider using max_length=256 for training")
        
        # Check for empty pairs
        if self.statistics['empty_pairs'] > 0:
            self.optimization_suggestions.append(f"{self.statistics['empty_pairs']} pairs have empty questions or answers - these should be filtered out")
        
        # Check confidence distribution
        high_conf_pairs = sum(count for conf, count in self.statistics['confidence_distribution'].items() if float(conf) >= 0.8)
        if high_conf_pairs < self.statistics['total_pairs'] * 0.5:
            self.optimization_suggestions.append("Less than 50% of pairs are high confidence - consider using only high_confidence dataset for training")
        
        # Check dataset size recommendations
        total_pairs = self.statistics['total_pairs']
        if total_pairs < 100:
            self.optimization_suggestions.append("Dataset is very small (<100 pairs) - training may not be effective")
        elif total_pairs > 10000:
            self.optimization_suggestions.append("Dataset is very large (>10K pairs) - consider sampling for faster training iterations")
    
    def _print_validation_results(self) -> None:
        """Print validation results"""
        print("\n📊 VALIDATION RESULTS")
        print("=" * 50)
        
        # Errors
        if self.validation_errors:
            print("❌ VALIDATION ERRORS:")
            for error in self.validation_errors:
                print(f"   • {error}")
        else:
            print("✅ No validation errors found")
        
        # Statistics
        print(f"\n📈 STATISTICS:")
        print(f"   • Total pairs: {self.statistics['total_pairs']}")
        print(f"   • Avg question length: {self.statistics['avg_question_length']:.1f} chars")
        print(f"   • Avg answer length: {self.statistics['avg_answer_length']:.1f} chars")
        print(f"   • Max question length: {self.statistics['max_question_length']} chars")
        print(f"   • Max answer length: {self.statistics['max_answer_length']} chars")
        print(f"   • Very long sequences (>512): {self.statistics['very_long_sequences']}")
        
        if self.statistics['confidence_distribution']:
            print(f"   • Confidence distribution:")
            for conf, count in sorted(self.statistics['confidence_distribution'].items()):
                print(f"     - {conf}: {count} pairs")
        
        if self.statistics['response_types']:
            print(f"   • Response types:")
            for resp_type, count in self.statistics['response_types'].items():
                print(f"     - {resp_type}: {count} pairs")
        
        # Optimizations
        if self.optimization_suggestions:
            print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
            for suggestion in self.optimization_suggestions:
                print(f"   • {suggestion}")
        else:
            print(f"\n✅ No optimization suggestions - data looks good for training!")
    
    def create_optimized_version(self, input_file: str, output_file: str, max_length: int = 256) -> bool:
        """Create an optimized version of the training data"""
        print(f"\n🔧 Creating optimized version...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Max length: {max_length}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            original_count = len(data['training_data'])
            optimized_pairs = []
            
            for pair in data['training_data']:
                # Skip empty pairs
                if not pair.get('question', '').strip() or not pair.get('answer', '').strip():
                    continue
                
                # Truncate if necessary
                question = pair['question']
                answer = pair['answer']
                
                if len(question) > max_length:
                    question = question[:max_length].rsplit(' ', 1)[0] + "..."
                
                if len(answer) > max_length:
                    answer = answer[:max_length].rsplit(' ', 1)[0] + "..."
                
                # Create optimized pair
                optimized_pair = {
                    "question": question,
                    "answer": answer
                }
                
                # Preserve metadata if available
                if 'metadata' in pair:
                    optimized_pair['metadata'] = pair['metadata']
                
                optimized_pairs.append(optimized_pair)
            
            # Update metadata
            data['training_data'] = optimized_pairs
            data['metadata']['total_pairs'] = len(optimized_pairs)
            data['metadata']['optimization_applied'] = {
                'max_length': max_length,
                'original_pairs': original_count,
                'optimized_pairs': len(optimized_pairs),
                'removed_pairs': original_count - len(optimized_pairs)
            }
            
            # Save optimized version
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Optimized version created:")
            print(f"   • Original pairs: {original_count}")
            print(f"   • Optimized pairs: {len(optimized_pairs)}")
            print(f"   • Removed pairs: {original_count - len(optimized_pairs)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating optimized version: {e}")
            return False


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python training_data_validator.py <training_data.json> [--optimize] [--max-length 256]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    optimize = '--optimize' in sys.argv
    
    # Get max length parameter
    max_length = 256
    if '--max-length' in sys.argv:
        try:
            max_length_idx = sys.argv.index('--max-length') + 1
            max_length = int(sys.argv[max_length_idx])
        except (IndexError, ValueError):
            print("Invalid --max-length parameter, using default 256")
    
    validator = TrainingDataValidator()
    
    # Validate the file
    is_valid = validator.validate_training_file(input_file)
    
    if not is_valid:
        print("\n❌ Training data has validation errors!")
        sys.exit(1)
    
    # Create optimized version if requested
    if optimize:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
        validator.create_optimized_version(input_file, str(output_file), max_length)
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
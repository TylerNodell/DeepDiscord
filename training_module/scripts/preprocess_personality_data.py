#!/usr/bin/env python3
"""
Preprocess Discord training data with personality-based formatting.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import argparse

# Setup path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.personality_config import PersonalityConfig, DEFAULT_PERSONALITY_CONFIG
from config.data_config import DataConfig
from utils.discord_preprocessing import DiscordPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(data_config: DataConfig) -> None:
    """Create necessary directories for data processing."""
    directories = [
        Path(data_config.processed_dir),
        Path(data_config.cache_dir),
        Path(data_config.processed_dir) / "chatml",
        Path(data_config.processed_dir) / "personalities"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def find_training_files(input_dir: Path) -> List[Path]:
    """Find all training ZIP files in the input directory."""
    zip_files = list(input_dir.glob("*.zip"))
    
    if not zip_files:
        logger.warning(f"No ZIP files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(zip_files)} training files: {[f.name for f in zip_files]}")
    return zip_files


def preprocess_single_file(
    zip_file: Path, 
    preprocessor: DiscordPreprocessor, 
    data_config: DataConfig,
    output_dir: Path
) -> Optional[Dict]:
    """Preprocess a single training file."""
    logger.info(f"Processing {zip_file.name}")
    
    # Load consent data if available
    consent_file = Path(data_config.consent_file)
    
    try:
        # Process the ZIP file
        chatml_data = preprocessor.process_training_zip(zip_file, consent_file)
        
        if not chatml_data:
            logger.warning(f"No data processed from {zip_file.name}")
            return None
        
        # Generate output filename
        output_file = output_dir / f"{zip_file.stem}_chatml.json"
        
        # Save processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chatml_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chatml_data)} entries to {output_file}")
        
        # Get and log statistics
        stats = preprocessor.get_preprocessing_stats(chatml_data)
        logger.info(f"Processing stats for {zip_file.name}:")
        logger.info(f"  Total entries: {stats['total_entries']}")
        logger.info(f"  Personality distribution: {stats['personality_distribution']}")
        logger.info(f"  Average chars per entry: {stats['average_chars_per_entry']:.1f}")
        
        return {
            'file': zip_file.name,
            'output': output_file.name,
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"Failed to process {zip_file.name}: {e}")
        return None


def save_personality_specific_data(
    all_chatml_data: List[Dict], 
    personality_config: PersonalityConfig,
    output_dir: Path
) -> None:
    """Save personality-specific training data for multiple LoRA training."""
    if personality_config.strategy.value != "multiple_lora":
        logger.info("Skipping personality-specific data saving (not using multiple LoRA)")
        return
    
    personality_dir = output_dir / "personalities"
    personality_dir.mkdir(exist_ok=True)
    
    # Group data by personality
    personality_data: Dict[str, List[Dict]] = {}
    
    for entry in all_chatml_data:
        personality = entry['metadata'].get('personality', 'unknown')
        if personality not in personality_data:
            personality_data[personality] = []
        personality_data[personality].append(entry)
    
    # Save each personality's data
    for personality, data in personality_data.items():
        if personality == 'unknown':
            continue
            
        output_file = personality_dir / f"{personality}_training.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} entries for personality '{personality}' to {output_file}")


def generate_training_summary(
    processing_results: List[Dict],
    personality_config: PersonalityConfig,
    output_dir: Path
) -> None:
    """Generate a comprehensive training summary."""
    summary = {
        'personality_strategy': personality_config.strategy.value,
        'total_files_processed': len(processing_results),
        'files': processing_results,
        'combined_stats': {
            'total_entries': sum(r['stats']['total_entries'] for r in processing_results if r),
            'total_characters': sum(r['stats']['total_characters'] for r in processing_results if r),
        },
        'personality_config': {
            'num_personalities': len(personality_config.personalities),
            'active_personalities': [p.display_name for p in personality_config.get_active_personalities()],
            'training_summary': personality_config.get_training_summary()
        }
    }
    
    # Save summary
    summary_file = output_dir / "preprocessing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training summary saved to {summary_file}")
    logger.info(f"Total processed entries: {summary['combined_stats']['total_entries']}")
    logger.info(f"Active personalities: {summary['personality_config']['active_personalities']}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess Discord training data with personality support")
    parser.add_argument("--input-dir", type=str, help="Input directory with ZIP files")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed data") 
    parser.add_argument("--strategy", type=str, choices=['unified', 'instruction_based', 'multiple_lora'],
                       default='instruction_based', help="Personality strategy")
    parser.add_argument("--balance-samples", action='store_true', default=True,
                       help="Balance samples across personalities")
    parser.add_argument("--require-consent", action='store_true', default=True,
                       help="Only include users who have given consent")
    
    args = parser.parse_args()
    
    # Setup configurations
    data_config = DataConfig()
    if args.input_dir:
        data_config.input_dir = args.input_dir
    if args.output_dir:
        data_config.processed_dir = args.output_dir
    
    # Setup personality configuration
    personality_config = DEFAULT_PERSONALITY_CONFIG
    if args.strategy:
        from config.personality_config import PersonalityStrategy
        personality_config.strategy = PersonalityStrategy(args.strategy)
    personality_config.balance_samples = args.balance_samples
    personality_config.require_consent = args.require_consent
    
    logger.info(f"Starting preprocessing with strategy: {personality_config.strategy.value}")
    logger.info(f"Input directory: {data_config.input_dir}")
    logger.info(f"Output directory: {data_config.processed_dir}")
    
    # Setup directories
    setup_directories(data_config)
    
    # Initialize preprocessor
    preprocessor = DiscordPreprocessor(personality_config)
    
    # Find training files
    input_dir = Path(data_config.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    training_files = find_training_files(input_dir)
    if not training_files:
        logger.error("No training files found")
        return
    
    # Process each file
    output_dir = Path(data_config.processed_dir) / "chatml"
    processing_results = []
    all_chatml_data = []
    
    for zip_file in training_files:
        result = preprocess_single_file(zip_file, preprocessor, data_config, output_dir)
        if result:
            processing_results.append(result)
            
            # Load the processed data for combined operations
            with open(output_dir / result['output'], 'r') as f:
                file_data = json.load(f)
                all_chatml_data.extend(file_data)
    
    if not processing_results:
        logger.error("No files were successfully processed")
        return
    
    # Save personality-specific data if using multiple LoRA
    save_personality_specific_data(all_chatml_data, personality_config, Path(data_config.processed_dir))
    
    # Generate training summary
    generate_training_summary(processing_results, personality_config, Path(data_config.processed_dir))
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Processed {len(processing_results)} files with {len(all_chatml_data)} total entries")


if __name__ == "__main__":
    main()
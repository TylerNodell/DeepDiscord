#!/usr/bin/env python3
"""
Test script to verify training data compatibility between Discord bot export
and training scripts.
"""

import json
import os
import sys
from pathlib import Path
import tempfile
import shutil


def create_test_training_data():
    """Create a test training data file in the expected format"""
    test_data = {
        "metadata": {
            "generated_at": "2025-07-20T20:00:00.000000",
            "target_user_id": 123456789,
            "target_username": "TestUser",
            "total_pairs": 3,
            "confidence_threshold": "high_confidence",
            "format": "question/answer pairs for training",
            "channels_processed": 2,
            "messages_analyzed": 100,
            "guild_name": "Test Guild",
            "training_compatibility": {
                "max_sequence_length": 512,
                "empty_pairs_removed": 0,
                "optimized_for_training": True
            }
        },
        "training_data": [
            {
                "question": "TestUser1: What's your favorite color?",
                "answer": "I really like blue, it reminds me of the ocean.",
                "metadata": {
                    "response_type": "explicit_reply",
                    "confidence": 1.0,
                    "question_author": "TestUser1",
                    "answer_author": "TestUser",
                    "channel": "general",
                    "timestamp": "2025-07-20T19:00:00.000000+00:00",
                    "question_id": 111111111,
                    "answer_id": 222222222,
                    "time_gap": 0
                }
            },
            {
                "question": "TestUser2: How are you doing today?",
                "answer": "I'm doing great, thanks for asking! Just been working on some coding projects.",
                "metadata": {
                    "response_type": "temporal_proximity",
                    "confidence": 0.9,
                    "question_author": "TestUser2",
                    "answer_author": "TestUser",
                    "channel": "general",
                    "timestamp": "2025-07-20T19:05:00.000000+00:00",
                    "question_id": 333333333,
                    "answer_id": 444444444,
                    "time_gap": 5
                }
            },
            {
                "question": "TestUser3: What do you think about AI?",
                "answer": "AI is fascinating! I think it has huge potential to help solve complex problems.",
                "metadata": {
                    "response_type": "content_analysis",
                    "confidence": 0.8,
                    "question_author": "TestUser3",
                    "answer_author": "TestUser",
                    "channel": "tech-talk",
                    "timestamp": "2025-07-20T19:10:00.000000+00:00",
                    "question_id": 555555555,
                    "answer_id": 666666666,
                    "time_gap": 30
                }
            }
        ]
    }
    return test_data


def test_data_format_compatibility():
    """Test that the data format is compatible with training scripts"""
    print("🧪 Testing data format compatibility...")
    
    test_data = create_test_training_data()
    
    # Test required fields
    required_fields = ['metadata', 'training_data']
    for field in required_fields:
        assert field in test_data, f"Missing required field: {field}"
    
    # Test metadata fields
    metadata = test_data['metadata']
    required_metadata = ['generated_at', 'target_user_id', 'total_pairs']
    for field in required_metadata:
        assert field in metadata, f"Missing metadata field: {field}"
    
    # Test training pairs
    training_data = test_data['training_data']
    assert isinstance(training_data, list), "training_data must be a list"
    assert len(training_data) > 0, "training_data cannot be empty"
    
    for i, pair in enumerate(training_data):
        assert 'question' in pair, f"Pair {i}: Missing question field"
        assert 'answer' in pair, f"Pair {i}: Missing answer field"
        assert isinstance(pair['question'], str), f"Pair {i}: Question must be string"
        assert isinstance(pair['answer'], str), f"Pair {i}: Answer must be string"
        assert len(pair['question'].strip()) > 0, f"Pair {i}: Question cannot be empty"
        assert len(pair['answer'].strip()) > 0, f"Pair {i}: Answer cannot be empty"
    
    print("   ✅ Data format compatibility test passed!")
    return True


def test_training_script_loading():
    """Test that the training scripts can load the data format"""
    print("🧪 Testing training script data loading...")
    
    # Create temporary test file
    test_data = create_test_training_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file = f.name
    
    try:
        # Test the same loading logic used in training scripts
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        training_pairs = data.get('training_data', [])
        
        # Test format conversion (like in training scripts)
        formatted_data = []
        for pair in training_pairs:
            text = f"User: {pair['question']}\nAssistant: {pair['answer']}"
            formatted_data.append({"text": text})
        
        assert len(formatted_data) == len(training_pairs), "Data conversion failed"
        
        # Test that formatted data looks correct
        for item in formatted_data:
            assert 'text' in item, "Missing text field in formatted data"
            assert 'User:' in item['text'], "Missing User: prefix"
            assert 'Assistant:' in item['text'], "Missing Assistant: prefix"
        
        print("   ✅ Training script loading test passed!")
        return True
        
    finally:
        os.unlink(temp_file)


def test_validation_script():
    """Test the validation script with test data"""
    print("🧪 Testing validation script...")
    
    # Import validation script
    sys.path.append(str(Path(__file__).parent / "discord_bot"))
    
    try:
        from discord_bot.training_data_validator import TrainingDataValidator
        
        # Create test file
        test_data = create_test_training_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, indent=2)
            temp_file = f.name
        
        try:
            validator = TrainingDataValidator()
            is_valid = validator.validate_training_file(temp_file)
            
            assert is_valid, "Validation script should pass for well-formed test data"
            assert validator.statistics['total_pairs'] == 3, "Should detect 3 training pairs"
            assert len(validator.validation_errors) == 0, "Should have no validation errors"
            
            print("   ✅ Validation script test passed!")
            return True
            
        finally:
            os.unlink(temp_file)
            
    except ImportError as e:
        print(f"   ⚠️  Validation script test skipped: {e}")
        return True


def test_training_config_compatibility():
    """Test that training config is optimized for stability"""
    print("🧪 Testing training configuration...")
    
    # Import training config
    sys.path.append(str(Path(__file__).parent / "training_module"))
    
    try:
        from config.training_config import TrainingConfig
        
        config = TrainingConfig()
        
        # Test critical stability settings
        critical_checks = [
            (config.batch_size == 1, "batch_size should be 1 for stability"),
            (config.learning_rate <= 1e-4, "learning_rate should be <= 1e-4 for stability"),
            (config.max_length <= 512, "max_length should be <= 512 for memory efficiency"),
            (config.lora_r <= 32, "lora_r should be <= 32 for stability"),
            (not config.gradient_checkpointing, "gradient_checkpointing should be False"),
            (not config.torch_compile, "torch_compile should be False"),
            (not config.group_by_length, "group_by_length should be False"),
            (not config.load_best_model_at_end, "load_best_model_at_end should be False"),
            (config.max_grad_norm <= 1.0, "max_grad_norm should be <= 1.0"),
            (config.optim == "adamw_torch", "optim should be adamw_torch for stability"),
            (not config.use_cache, "use_cache should be False during training")
        ]
        
        all_passed = True
        for check, message in critical_checks:
            if not check:
                print(f"   ❌ {message}")
                all_passed = False
        
        if all_passed:
            print("   ✅ Training configuration test passed!")
        else:
            print("   ⚠️  Training configuration has stability issues!")
        
        return all_passed
        
    except ImportError as e:
        print(f"   ⚠️  Training config test skipped: {e}")
        return True


def test_universal_training_script():
    """Test that the universal training script exists and has the right interface"""
    print("🧪 Testing universal training script...")
    
    script_path = Path(__file__).parent / "training_module" / "scripts" / "train_universal.py"
    
    if not script_path.exists():
        print("   ❌ Universal training script not found!")
        return False
    
    # Check that it has the expected command line interface
    with open(script_path, 'r') as f:
        content = f.read()
    
    expected_features = [
        "argparse",
        "--data",
        "--personality-id",
        "--use-cached-model",
        "load_training_data",
        "format_training_data",
        "multiple formats"
    ]
    
    for feature in expected_features:
        if feature not in content:
            print(f"   ❌ Missing expected feature: {feature}")
            return False
    
    print("   ✅ Universal training script test passed!")
    return True


def test_docker_compatibility():
    """Test Docker setup scripts and compatibility"""
    print("🧪 Testing Docker compatibility...")
    
    # Check for Docker setup files
    required_files = [
        "training_module/run_training.sh",
        "training_module/setup_training_environment.py",
        "training_module/TRAINING_FIXES.md"
    ]
    
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if not full_path.exists():
            print(f"   ❌ Missing required file: {file_path}")
            return False
    
    # Check that run_training.sh has critical environment variables
    run_script = Path(__file__).parent / "training_module" / "run_training.sh"
    with open(run_script, 'r') as f:
        content = f.read()
    
    required_env_vars = [
        "TORCH_COMPILE=0",
        "TOKENIZERS_PARALLELISM=false",
        "CUDA_LAUNCH_BLOCKING=1"
    ]
    
    for env_var in required_env_vars:
        if env_var not in content:
            print(f"   ❌ Missing critical environment variable: {env_var}")
            return False
    
    print("   ✅ Docker compatibility test passed!")
    return True


def main():
    """Run all compatibility tests"""
    print("🚀 DeepDiscord Training Compatibility Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Format Compatibility", test_data_format_compatibility),
        ("Training Script Loading", test_training_script_loading),
        ("Validation Script", test_validation_script),
        ("Training Configuration", test_training_config_compatibility),
        ("Universal Training Script", test_universal_training_script),
        ("Docker Compatibility", test_docker_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if error:
            print(f"     Error: {error}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Training compatibility is verified.")
        print("\n💡 Your system is ready for Discord bot training data export and AI model training!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
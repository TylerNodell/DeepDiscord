#!/bin/bash
# GPU test script for Docker environment
# Tests CUDA, PyTorch, and ML framework compatibility

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Test system GPU
test_system_gpu() {
    log_info "Testing system GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        log_success "System GPU detected"
    else
        log_error "nvidia-smi not found"
        return 1
    fi
}

# Test Docker GPU access
test_docker_gpu() {
    log_info "Testing Docker GPU access..."
    
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi; then
        log_success "Docker GPU access working"
    else
        log_error "Docker GPU access failed"
        return 1
    fi
}

# Test PyTorch GPU in container
test_pytorch_gpu() {
    log_info "Testing PyTorch GPU support in training container..."
    
    # Change to training module directory
    cd "$(dirname "$0")/.."
    
    docker-compose run --rm training-dev python -c "
import torch
import sys

print('=== PyTorch GPU Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'Python version: {sys.version}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Compute capability: {props.major}.{props.minor}')
        print(f'  Total memory: {props.total_memory // 1024**3:.1f} GB')
        print(f'  Multi-processors: {props.multi_processor_count}')
    
    # Test tensor operations
    print()
    print('=== GPU Tensor Test ===')
    device = torch.device('cuda:0')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f'Matrix multiplication test: {z.shape} on {z.device}')
    print('GPU tensor operations: PASSED')
    
else:
    print('CUDA not available - GPU tests skipped')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        log_success "PyTorch GPU test passed"
    else
        log_error "PyTorch GPU test failed"
        return 1
    fi
}

# Test ML frameworks
test_ml_frameworks() {
    log_info "Testing ML frameworks in training container..."
    
    docker-compose run --rm training-dev python -c "
print('=== ML Frameworks Test ===')

# Test Transformers
try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
    
    # Test tokenizer (lightweight test)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokens = tokenizer('Hello world')
    print(f'Tokenizer test: {len(tokens[\"input_ids\"])} tokens')
    print('Transformers: PASSED')
except Exception as e:
    print(f'Transformers test failed: {e}')

# Test PEFT
try:
    import peft
    print(f'PEFT version: {peft.__version__}')
    print('PEFT: PASSED')
except Exception as e:
    print(f'PEFT test failed: {e}')

# Test BitsAndBytes
try:
    import bitsandbytes as bnb
    print(f'BitsAndBytes version: {bnb.__version__}')
    print('BitsAndBytes: PASSED')
except Exception as e:
    print(f'BitsAndBytes test failed: {e}')

# Test Flash Attention
try:
    import flash_attn
    print(f'Flash Attention available')
    print('Flash Attention: PASSED')
except Exception as e:
    print(f'Flash Attention test failed: {e}')

# Test Unsloth (optional)
try:
    import unsloth
    print(f'Unsloth available')
    print('Unsloth: PASSED')
except Exception as e:
    print(f'Unsloth not available: {e}')
"
    
    if [[ $? -eq 0 ]]; then
        log_success "ML frameworks test completed"
    else
        log_warning "Some ML frameworks may have issues"
    fi
}

# Test memory allocation
test_memory() {
    log_info "Testing GPU memory allocation..."
    
    docker-compose run --rm training-dev python -c "
import torch
import gc

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
    print('=== GPU Memory Test ===')
    print(f'Total GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
    print(f'Initial allocated: {torch.cuda.memory_allocated(0) // 1024**2} MB')
    print(f'Initial cached: {torch.cuda.memory_reserved(0) // 1024**2} MB')
    
    # Allocate some memory
    tensors = []
    for i in range(5):
        tensor = torch.randn(100, 100, 100, device=device)
        tensors.append(tensor)
        allocated = torch.cuda.memory_allocated(0) // 1024**2
        print(f'After allocation {i+1}: {allocated} MB')
    
    # Clear memory
    del tensors
    gc.collect()
    torch.cuda.empty_cache()
    
    final_allocated = torch.cuda.memory_allocated(0) // 1024**2
    print(f'After cleanup: {final_allocated} MB')
    print('Memory management: PASSED')
else:
    print('CUDA not available for memory test')
"
    
    if [[ $? -eq 0 ]]; then
        log_success "GPU memory test passed"
    else
        log_error "GPU memory test failed"
        return 1
    fi
}

# Test model loading
test_model_loading() {
    log_info "Testing model loading capabilities..."
    
    docker-compose run --rm training-dev python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

if torch.cuda.is_available():
    print('=== Model Loading Test ===')
    device = torch.device('cuda:0')
    
    try:
        # Test with a small model first
        print('Loading GPT-2 small model...')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        model = model.to(device)
        
        # Test inference
        inputs = tokenizer('Hello, this is a test', return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f'Model loaded successfully on {model.device}')
        print(f'Output shape: {outputs.logits.shape}')
        print('Model loading test: PASSED')
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f'Model loading test failed: {e}')
        
else:
    print('CUDA not available for model test')
"
    
    if [[ $? -eq 0 ]]; then
        log_success "Model loading test completed"
    else
        log_warning "Model loading test had issues"
    fi
}

# Main test function
run_all_tests() {
    log_info "Starting comprehensive GPU tests for Docker environment..."
    
    # Change to training module directory
    cd "$(dirname "$0")/.."
    
    # Build image if it doesn't exist
    if ! docker image inspect deepdiscord-training:latest &> /dev/null; then
        log_info "Building training image..."
        docker-compose build training-dev
    fi
    
    local failed_tests=0
    
    # Run tests
    test_system_gpu || ((failed_tests++))
    test_docker_gpu || ((failed_tests++))
    test_pytorch_gpu || ((failed_tests++))
    test_ml_frameworks || ((failed_tests++))
    test_memory || ((failed_tests++))
    test_model_loading || ((failed_tests++))
    
    # Summary
    echo
    log_info "=== Test Summary ==="
    if [[ $failed_tests -eq 0 ]]; then
        log_success "All GPU tests passed! Docker environment is ready for training."
    else
        log_warning "$failed_tests test(s) failed. Check the output above for details."
    fi
    
    return $failed_tests
}

# Parse command line arguments
case "${1:-all}" in
    system)
        test_system_gpu
        ;;
    docker)
        test_docker_gpu
        ;;
    pytorch)
        test_pytorch_gpu
        ;;
    frameworks)
        test_ml_frameworks
        ;;
    memory)
        test_memory
        ;;
    model)
        test_model_loading
        ;;
    all)
        run_all_tests
        ;;
    help|--help|-h)
        cat << EOF
GPU Test Script for Docker Environment

Usage: $0 [TEST_TYPE]

Test Types:
    system      Test system GPU with nvidia-smi
    docker      Test Docker GPU access
    pytorch     Test PyTorch GPU support
    frameworks  Test ML frameworks (Transformers, PEFT, etc.)
    memory      Test GPU memory allocation
    model       Test model loading capabilities
    all         Run all tests (default)
    help        Show this help

Examples:
    $0              # Run all tests
    $0 pytorch      # Test only PyTorch GPU
    $0 memory       # Test only memory allocation
EOF
        ;;
    *)
        log_error "Unknown test type: $1"
        log_info "Use '$0 help' for available options"
        exit 1
        ;;
esac
#!/bin/bash
# Docker setup script for DeepDiscord Training Module
# Handles NVIDIA Docker setup, GPU verification, and container management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This script can be run as a regular user."
    fi
}

# Check system requirements
check_system() {
    log_info "Checking system requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if NVIDIA drivers are installed
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "NVIDIA drivers not found. GPU acceleration will not be available."
        return 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        log_warning "NVIDIA Docker runtime not found. Attempting to install..."
        install_nvidia_docker
    fi
    
    log_success "System requirements check completed"
    return 0
}

# Install NVIDIA Docker support
install_nvidia_docker() {
    log_info "Installing NVIDIA Docker support..."
    
    # Add NVIDIA Docker repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install NVIDIA Container Toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker daemon
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log_success "NVIDIA Docker support installed"
}

# Verify GPU access
verify_gpu() {
    log_info "Verifying GPU access..."
    
    if nvidia-smi &> /dev/null; then
        log_info "GPU Status:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        
        # Test GPU access in Docker
        log_info "Testing GPU access in Docker container..."
        if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_success "GPU access verified in Docker"
            return 0
        else
            log_error "GPU access test failed in Docker"
            return 1
        fi
    else
        log_warning "No GPU detected or drivers not installed"
        return 1
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build training module image
    log_info "Building training module image..."
    docker build -t deepdiscord-training:latest -f Dockerfile --target development .
    
    # Build production image
    log_info "Building production training image..."
    docker build -t deepdiscord-training:production -f Dockerfile --target production .
    
    log_success "Docker images built successfully"
}

# Setup environment file
setup_env() {
    local env_file="../.env"
    
    if [[ ! -f "$env_file" ]]; then
        log_info "Creating environment file..."
        cat > "$env_file" << EOF
# DeepDiscord Environment Configuration

# Discord Bot Token (required for bot functionality)
DISCORD_TOKEN=your_discord_token_here

# Training Configuration
CUDA_VISIBLE_DEVICES=0
WANDB_MODE=offline
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1

# Jupyter Configuration
JUPYTER_TOKEN=deepdiscord-training
JUPYTER_ENABLE_LAB=yes

# GPU Memory Management
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Hugging Face Cache
HF_HOME=/app/training_module/.cache/huggingface
EOF
        log_success "Environment file created at $env_file"
        log_warning "Please edit $env_file and set your Discord token"
    else
        log_info "Environment file already exists at $env_file"
    fi
}

# Start development environment
start_dev() {
    log_info "Starting development environment..."
    
    # Create necessary directories
    mkdir -p data/processed data/cache checkpoints logs experiments
    
    # Start development containers
    docker-compose --profile development up -d training-dev
    
    log_success "Development environment started"
    log_info "Jupyter Lab: http://localhost:8888 (token: deepdiscord-training)"
    log_info "TensorBoard: http://localhost:6006"
    
    # Show container status
    docker-compose ps
}

# Start production training
start_training() {
    local data_file="$1"
    
    if [[ -z "$data_file" ]]; then
        log_error "Usage: $0 train <data_file>"
        exit 1
    fi
    
    if [[ ! -f "../discord_bot/results/$data_file" ]]; then
        log_error "Training data file not found: ../discord_bot/results/$data_file"
        exit 1
    fi
    
    log_info "Starting training with data file: $data_file"
    
    # Run training container
    docker-compose --profile production run --rm training-prod \
        --data "../discord_bot/results/$data_file" \
        --use-unsloth \
        --run-name "docker_training_$(date +%Y%m%d_%H%M%S)"
    
    log_success "Training completed"
}

# Stop all containers
stop_all() {
    log_info "Stopping all containers..."
    docker-compose down
    log_success "All containers stopped"
}

# Clean up Docker resources
cleanup() {
    log_info "Cleaning up Docker resources..."
    
    # Stop containers
    docker-compose down
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful with this)
    read -p "Remove unused Docker volumes? This will delete cached data (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Show help
show_help() {
    cat << EOF
DeepDiscord Training Module Docker Setup

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    setup           Initial setup - check requirements and build images
    dev             Start development environment (Jupyter + TensorBoard)
    train <file>    Start production training with specified data file
    stop            Stop all running containers
    cleanup         Clean up Docker resources
    gpu-test        Test GPU access in Docker
    logs [service]  Show logs for specified service (default: training-dev)
    shell [service] Open shell in specified service (default: training-dev)
    help            Show this help message

Examples:
    $0 setup                                    # Initial setup
    $0 dev                                      # Start development environment
    $0 train training_data_user_20250719.zip   # Start training
    $0 gpu-test                                 # Test GPU access
    $0 logs training-dev                        # Show development logs
    $0 shell training-dev                       # Open shell in dev container

Development URLs:
    Jupyter Lab:  http://localhost:8888 (token: deepdiscord-training)
    TensorBoard:  http://localhost:6006
EOF
}

# Show container logs
show_logs() {
    local service="${1:-training-dev}"
    log_info "Showing logs for service: $service"
    docker-compose logs -f "$service"
}

# Open shell in container
open_shell() {
    local service="${1:-training-dev}"
    log_info "Opening shell in service: $service"
    docker-compose exec "$service" bash
}

# Test GPU functionality
test_gpu() {
    log_info "Testing GPU functionality in Docker..."
    
    docker run --rm --gpus all deepdiscord-training:latest python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory // 1024**3} GB')
else:
    print('GPU testing failed - CUDA not available')
"
}

# Main script logic
main() {
    case "${1:-help}" in
        setup)
            check_root
            check_system
            setup_env
            build_images
            verify_gpu
            log_success "Setup completed! Run '$0 dev' to start development environment."
            ;;
        dev)
            start_dev
            ;;
        train)
            start_training "$2"
            ;;
        stop)
            stop_all
            ;;
        cleanup)
            cleanup
            ;;
        gpu-test)
            test_gpu
            ;;
        logs)
            show_logs "$2"
            ;;
        shell)
            open_shell "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Change to script directory
cd "$(dirname "$0")/.."

# Run main function
main "$@"
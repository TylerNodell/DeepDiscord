#!/bin/bash
# Remote monitoring setup for DeepDiscord training
# Allows monitoring training progress from a remote machine

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

# Configuration
MAIN_MACHINE_IP=""
SSH_USER=""
SSH_KEY=""
PROJECT_PATH="/path/to/DeepDiscord"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ip)
            MAIN_MACHINE_IP="$2"
            shift 2
            ;;
        --user)
            SSH_USER="$2"
            shift 2
            ;;
        --key)
            SSH_KEY="$2"
            shift 2
            ;;
        --path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        --help|-h)
            cat << EOF
Remote Monitoring Setup for DeepDiscord Training

Usage: $0 [OPTIONS] COMMAND

Options:
    --ip IP         IP address of main machine with RTX 5080
    --user USER     SSH username for main machine
    --key PATH      Path to SSH private key (optional)
    --path PATH     Project path on main machine

Commands:
    setup           Setup remote monitoring
    tunnel          Create SSH tunnels for services
    monitor         Start monitoring dashboard
    train           Start remote training with monitoring
    status          Check training status
    logs            View training logs
    stop            Stop training and tunnels
    help            Show this help

Examples:
    $0 --ip 192.168.1.100 --user nuko setup
    $0 --ip 192.168.1.100 --user nuko train --data training_data.zip
    $0 monitor  # After tunnels are established
EOF
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            break
            ;;
    esac
done

# Validate required parameters
validate_config() {
    if [[ -z "$MAIN_MACHINE_IP" ]]; then
        log_error "Main machine IP not specified. Use --ip <IP_ADDRESS>"
        exit 1
    fi
    
    if [[ -z "$SSH_USER" ]]; then
        log_error "SSH user not specified. Use --user <USERNAME>"
        exit 1
    fi
}

# Setup SSH connection parameters
setup_ssh() {
    SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=60"
    
    if [[ -n "$SSH_KEY" ]]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    
    SSH_CMD="ssh $SSH_OPTS $SSH_USER@$MAIN_MACHINE_IP"
}

# Test SSH connection
test_connection() {
    log_info "Testing SSH connection to $SSH_USER@$MAIN_MACHINE_IP..."
    
    if $SSH_CMD "echo 'Connection successful'" > /dev/null 2>&1; then
        log_success "SSH connection established"
        return 0
    else
        log_error "SSH connection failed"
        log_info "Make sure:"
        log_info "  1. SSH is enabled on the main machine"
        log_info "  2. SSH keys are properly configured"
        log_info "  3. Firewall allows SSH connections"
        return 1
    fi
}

# Setup Docker and project on remote machine
setup_remote() {
    log_info "Setting up remote machine..."
    
    # Test connection first
    test_connection || exit 1
    
    # Check if Docker is installed
    log_info "Checking Docker installation..."
    $SSH_CMD "docker --version" || {
        log_error "Docker not found on remote machine"
        log_info "Install Docker on the main machine first"
        exit 1
    }
    
    # Check NVIDIA Docker
    log_info "Checking NVIDIA Docker support..."
    $SSH_CMD "docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi" || {
        log_error "NVIDIA Docker not working on remote machine"
        log_info "Run './scripts/docker_setup.sh setup' on the main machine first"
        exit 1
    }
    
    # Check project directory
    log_info "Checking project directory..."
    if $SSH_CMD "test -d $PROJECT_PATH"; then
        log_success "Project directory found: $PROJECT_PATH"
    else
        log_error "Project directory not found: $PROJECT_PATH"
        log_info "Clone the project on the main machine or specify correct path with --path"
        exit 1
    fi
    
    log_success "Remote machine setup validated"
}

# Create SSH tunnels for monitoring services
create_tunnels() {
    log_info "Creating SSH tunnels for monitoring services..."
    
    # Kill existing tunnels
    pkill -f "ssh.*$MAIN_MACHINE_IP.*-L" 2>/dev/null || true
    
    # Create tunnels in background
    # Jupyter Lab (8888 -> 8888)
    ssh $SSH_OPTS -L 8888:localhost:8888 -N $SSH_USER@$MAIN_MACHINE_IP &
    JUPYTER_PID=$!
    
    # TensorBoard (6006 -> 6006)
    ssh $SSH_OPTS -L 6006:localhost:6006 -N $SSH_USER@$MAIN_MACHINE_IP &
    TENSORBOARD_PID=$!
    
    # Grafana (3000 -> 3000) - if using monitoring stack
    ssh $SSH_OPTS -L 3000:localhost:3000 -N $SSH_USER@$MAIN_MACHINE_IP &
    GRAFANA_PID=$!
    
    # Wait a moment for tunnels to establish
    sleep 3
    
    # Test tunnels
    if curl -s http://localhost:8888 > /dev/null 2>&1; then
        log_success "Jupyter tunnel established: http://localhost:8888"
    else
        log_warning "Jupyter tunnel may not be ready yet"
    fi
    
    if curl -s http://localhost:6006 > /dev/null 2>&1; then
        log_success "TensorBoard tunnel established: http://localhost:6006"
    else
        log_warning "TensorBoard tunnel may not be ready yet"
    fi
    
    # Save PIDs for cleanup
    echo "$JUPYTER_PID $TENSORBOARD_PID $GRAFANA_PID" > /tmp/deepdiscord_tunnel_pids
    
    log_success "SSH tunnels created"
    log_info "Access services locally:"
    log_info "  Jupyter Lab:  http://localhost:8888 (token: deepdiscord-training)"
    log_info "  TensorBoard:  http://localhost:6006"
    log_info "  Grafana:      http://localhost:3000 (if monitoring stack is running)"
}

# Start remote training
start_training() {
    local training_args="$*"
    
    log_info "Starting remote training with args: $training_args"
    
    # Start development environment first (for monitoring)
    log_info "Starting development environment on remote machine..."
    $SSH_CMD "cd $PROJECT_PATH/training_module && docker-compose up -d training-dev"
    
    # Wait for services to start
    sleep 10
    
    # Create tunnels for monitoring
    create_tunnels
    
    # Start training in background
    log_info "Starting training process..."
    $SSH_CMD "cd $PROJECT_PATH/training_module && nohup ./scripts/docker_train.sh $training_args > training.log 2>&1 &"
    
    log_success "Training started on remote machine"
    log_info "Monitor progress at:"
    log_info "  Jupyter Lab:  http://localhost:8888"
    log_info "  TensorBoard:  http://localhost:6006"
    log_info "  Logs: Use '$0 logs' command"
}

# Monitor training status
check_status() {
    log_info "Checking training status on remote machine..."
    
    # Check Docker containers
    log_info "Docker containers:"
    $SSH_CMD "cd $PROJECT_PATH/training_module && docker-compose ps"
    
    # Check GPU usage
    log_info "GPU status:"
    $SSH_CMD "nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv"
    
    # Check training process
    log_info "Training processes:"
    $SSH_CMD "ps aux | grep -E '(train|python)' | grep -v grep || echo 'No training processes found'"
    
    # Check recent logs
    log_info "Recent training logs:"
    $SSH_CMD "cd $PROJECT_PATH/training_module && tail -20 training.log 2>/dev/null || echo 'No training.log found'"
}

# View training logs
view_logs() {
    log_info "Viewing training logs from remote machine..."
    
    # Follow training log
    $SSH_CMD "cd $PROJECT_PATH/training_module && tail -f training.log"
}

# Stop training and cleanup
stop_training() {
    log_info "Stopping training and cleaning up..."
    
    # Stop Docker containers
    $SSH_CMD "cd $PROJECT_PATH/training_module && docker-compose down"
    
    # Kill training processes
    $SSH_CMD "pkill -f 'docker_train.sh' || true"
    $SSH_CMD "pkill -f 'train_dolphin.py' || true"
    
    # Close SSH tunnels
    if [[ -f /tmp/deepdiscord_tunnel_pids ]]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done < /tmp/deepdiscord_tunnel_pids
        rm -f /tmp/deepdiscord_tunnel_pids
    fi
    
    log_success "Training stopped and tunnels closed"
}

# Start monitoring dashboard
start_monitoring() {
    log_info "Starting local monitoring dashboard..."
    
    # Check if tunnels are active
    if ! curl -s http://localhost:8888 > /dev/null 2>&1; then
        log_warning "Jupyter tunnel not active. Creating tunnels..."
        create_tunnels
        sleep 5
    fi
    
    # Open monitoring URLs
    if command -v open > /dev/null; then
        # macOS
        open "http://localhost:8888"
        open "http://localhost:6006"
    elif command -v xdg-open > /dev/null; then
        # Linux
        xdg-open "http://localhost:8888" &
        xdg-open "http://localhost:6006" &
    else
        log_info "Open these URLs manually:"
        log_info "  Jupyter Lab:  http://localhost:8888"
        log_info "  TensorBoard:  http://localhost:6006"
    fi
}

# Main script logic
main() {
    case "${COMMAND:-help}" in
        setup)
            validate_config
            setup_ssh
            setup_remote
            ;;
        tunnel)
            validate_config
            setup_ssh
            create_tunnels
            ;;
        train)
            validate_config
            setup_ssh
            start_training "$@"
            ;;
        status)
            validate_config
            setup_ssh
            check_status
            ;;
        logs)
            validate_config
            setup_ssh
            view_logs
            ;;
        monitor)
            start_monitoring
            ;;
        stop)
            validate_config
            setup_ssh
            stop_training
            ;;
        help|--help|-h)
            $0 --help
            ;;
        *)
            log_error "Unknown command: ${COMMAND:-<none>}"
            $0 --help
            exit 1
            ;;
    esac
}

main "$@"
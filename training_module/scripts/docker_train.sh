#!/bin/bash
# Docker training script for DeepDiscord
# Simplified interface for running training tasks in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
DATA_DIR="../discord_bot/results"
STRATEGY="instruction_based"
USE_UNSLOTH="true"
RUN_NAME=""
CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --no-unsloth)
            USE_UNSLOTH="false"
            shift
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            cat << EOF
Docker Training Script for DeepDiscord

Usage: $0 [OPTIONS]

Options:
    --data FILE         Training data file (in discord_bot/results/)
    --strategy STRATEGY Personality strategy: unified, instruction_based, multiple_lora
    --no-unsloth        Disable Unsloth acceleration
    --run-name NAME     Custom run name for experiment tracking
    --config FILE       Custom config file
    --help              Show this help

Examples:
    $0 --data training_data_user_20250719.zip
    $0 --data latest.zip --strategy multiple_lora --run-name "multi_personality_v1"
    $0 --data small_dataset.zip --no-unsloth
EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data file is specified
if [[ -z "$DATA_FILE" ]]; then
    log_error "Data file not specified. Use --data <filename>"
    log_info "Available files in $DATA_DIR:"
    ls -la "$DATA_DIR"/*.zip 2>/dev/null || echo "No ZIP files found"
    exit 1
fi

# Check if data file exists
if [[ ! -f "$DATA_DIR/$DATA_FILE" ]]; then
    log_error "Data file not found: $DATA_DIR/$DATA_FILE"
    log_info "Available files:"
    ls -la "$DATA_DIR"/*.zip 2>/dev/null || echo "No ZIP files found"
    exit 1
fi

# Generate run name if not provided
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="docker_${STRATEGY}_$(date +%Y%m%d_%H%M%S)"
fi

log_info "Starting Docker training with the following configuration:"
log_info "  Data file: $DATA_FILE"
log_info "  Strategy: $STRATEGY"
log_info "  Use Unsloth: $USE_UNSLOTH"
log_info "  Run name: $RUN_NAME"

# Change to training module directory
cd "$(dirname "$0")/.."

# Build training arguments
TRAIN_ARGS=()
TRAIN_ARGS+=("--data" "/app/discord_bot/results/$DATA_FILE")
TRAIN_ARGS+=("--run-name" "$RUN_NAME")

if [[ "$USE_UNSLOTH" == "true" ]]; then
    TRAIN_ARGS+=("--use-unsloth")
fi

if [[ -n "$CONFIG_FILE" ]]; then
    TRAIN_ARGS+=("--config" "$CONFIG_FILE")
fi

# Preprocessing step
log_info "Step 1: Preprocessing data with personality strategy: $STRATEGY"
docker-compose run --rm training-dev python scripts/preprocess_personality_data.py \
    --strategy "$STRATEGY" \
    --input-dir "/app/discord_bot/results" \
    --output-dir "/app/training_module/data/processed"

if [[ $? -ne 0 ]]; then
    log_error "Preprocessing failed"
    exit 1
fi

log_success "Preprocessing completed"

# Personality management step
log_info "Step 2: Managing personalities"
docker-compose run --rm training-dev python scripts/manage_personalities.py \
    --discover \
    --data-dir "/app/training_module/data/processed"

if [[ $? -ne 0 ]]; then
    log_error "Personality management failed"
    exit 1
fi

log_success "Personality management completed"

# Training step
log_info "Step 3: Starting model training"
log_info "Training arguments: ${TRAIN_ARGS[*]}"

# Run training in production container
docker-compose --profile production run --rm training-prod "${TRAIN_ARGS[@]}"

if [[ $? -eq 0 ]]; then
    log_success "Training completed successfully!"
    log_info "Results saved in checkpoints/ directory"
    log_info "Logs available in logs/ directory"
    
    # Show final model info
    log_info "Listing saved models:"
    docker-compose run --rm training-dev ls -la /app/training_module/checkpoints/
else
    log_error "Training failed"
    exit 1
fi
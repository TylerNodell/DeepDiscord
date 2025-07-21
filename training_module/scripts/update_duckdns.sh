#!/bin/bash
# DuckDNS IP update script for DeepDiscord training

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Load configuration from .env
ENV_FILE="$(dirname "$0")/../../.env"
if [[ -f "$ENV_FILE" ]]; then
    DUCKDNS_DOMAIN=$(grep "^DUCKDNS_DOMAIN=" "$ENV_FILE" | cut -d'=' -f2)
    DUCKDNS_TOKEN=$(grep "^DUCKDNS_TOKEN=" "$ENV_FILE" | cut -d'=' -f2)
fi

# Check if domain and token are set
if [[ -z "$DUCKDNS_DOMAIN" || "$DUCKDNS_DOMAIN" == "yourname-deepdiscord" ]]; then
    log_error "DUCKDNS_DOMAIN not configured in .env file"
    log_info "Please update .env with your actual DuckDNS domain name"
    exit 1
fi

if [[ -z "$DUCKDNS_TOKEN" || "$DUCKDNS_TOKEN" == "your-token-here" ]]; then
    log_error "DUCKDNS_TOKEN not configured in .env file"
    log_info "Please update .env with your DuckDNS token from https://www.duckdns.org"
    exit 1
fi

# Get current public IP
log_info "Getting current public IP address..."
PUBLIC_IP=$(curl -s https://ipinfo.io/ip)

if [[ -z "$PUBLIC_IP" ]]; then
    log_error "Failed to get public IP address"
    exit 1
fi

log_info "Current public IP: $PUBLIC_IP"

# Update DuckDNS
log_info "Updating DuckDNS domain: $DUCKDNS_DOMAIN.duckdns.org"
RESPONSE=$(curl -s "https://www.duckdns.org/update?domains=$DUCKDNS_DOMAIN&token=$DUCKDNS_TOKEN&ip=$PUBLIC_IP")

if [[ "$RESPONSE" == "OK" ]]; then
    log_success "DuckDNS updated successfully!"
    log_info "Your domain $DUCKDNS_DOMAIN.duckdns.org now points to $PUBLIC_IP"
else
    log_error "Failed to update DuckDNS. Response: $RESPONSE"
    log_info "Check your domain name and token in .env file"
    exit 1
fi

# Test DNS resolution
log_info "Testing DNS resolution..."
RESOLVED_IP=$(nslookup "$DUCKDNS_DOMAIN.duckdns.org" | grep -A1 "Name:" | tail -1 | awk '{print $2}')

if [[ "$RESOLVED_IP" == "$PUBLIC_IP" ]]; then
    log_success "DNS resolution confirmed: $DUCKDNS_DOMAIN.duckdns.org -> $RESOLVED_IP"
else
    log_info "DNS may take a few minutes to propagate. Current resolved IP: $RESOLVED_IP"
fi

echo
log_info "Setup complete! You can now use: $DUCKDNS_DOMAIN.duckdns.org"
log_info "Update your .env SSH_HOST to use this domain name"
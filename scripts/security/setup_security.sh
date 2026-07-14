#!/bin/bash

# ============================================================================
# Security Setup Script for Ottobot
# Generates secure keys and prepares environment for deployment
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/docker_images/.env"

echo "=========================================="
echo "  Ottobot Security Setup"
echo "=========================================="
echo ""

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Warning: .env file already exists${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 0
    fi
fi

# Generate secure keys
echo -e "${YELLOW}Generating secure keys...${NC}"
WEBUI_SECRET_KEY=$(openssl rand -hex 32)
PIPELINES_API_KEY=$(openssl rand -hex 32)

echo -e "${GREEN}✓ Keys generated${NC}"
echo ""

# Ask for deployment mode
echo "Select deployment mode:"
echo "  1) Local development (less strict security)"
echo "  2) Public deployment (strict security)"
read -p "Choice [1/2]: " -n 1 -r DEPLOY_MODE
echo ""

if [[ $DEPLOY_MODE == "2" ]]; then
    PUBLIC_MODE=true
    read -p "Enter your domain name (e.g., ottobot.example.com): " DOMAIN_NAME
    
    # Create .env for public deployment
    cat > "$ENV_FILE" << EOF
# ============================================================================
# Ottobot Environment Configuration - PUBLIC DEPLOYMENT
# Generated on $(date)
# ============================================================================

DEPLOYMENT_MODE=public
DOMAIN_NAME=$DOMAIN_NAME

# Security Keys (DO NOT SHARE)
WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY
PIPELINES_API_KEY=$PIPELINES_API_KEY

# Authentication - Strict settings for public
ENABLE_SIGNUP=false
WEBUI_AUTH=true
DEFAULT_USER_ROLE=pending

# Session Security
WEBUI_SESSION_COOKIE_SECURE=true
WEBUI_SESSION_COOKIE_SAME_SITE=strict

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=20

# Safety
ENABLE_COMMUNITY_SHARING=false
SAFE_MODE=true

# CORS
CORS_ALLOW_ORIGIN=https://$DOMAIN_NAME
EOF

else
    PUBLIC_MODE=false
    
    # Create .env for local development
    cat > "$ENV_FILE" << EOF
# ============================================================================
# Ottobot Environment Configuration - LOCAL DEVELOPMENT
# Generated on $(date)
# ============================================================================

DEPLOYMENT_MODE=local

# Security Keys
WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY
PIPELINES_API_KEY=$PIPELINES_API_KEY

# Authentication - Relaxed for local dev
ENABLE_SIGNUP=true
WEBUI_AUTH=true
DEFAULT_USER_ROLE=user

# Session Security
WEBUI_SESSION_COOKIE_SECURE=false
WEBUI_SESSION_COOKIE_SAME_SITE=lax

# Rate Limiting
RATE_LIMIT_ENABLED=false

# Safety
ENABLE_COMMUNITY_SHARING=false
SAFE_MODE=false
EOF

fi

# Set restrictive permissions on .env file
chmod 600 "$ENV_FILE"

echo ""
echo -e "${GREEN}✓ Environment file created: $ENV_FILE${NC}"
echo ""

# Save API key to separate file for easy reference
mkdir -p "$PROJECT_ROOT/docker_backups"
echo "$PIPELINES_API_KEY" > "$PROJECT_ROOT/docker_backups/pipelines_api_key.txt"
chmod 600 "$PROJECT_ROOT/docker_backups/pipelines_api_key.txt"

echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Pipelines API Key:${NC} $PIPELINES_API_KEY"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "This key is also saved to: $PROJECT_ROOT/docker_backups/pipelines_api_key.txt"
echo ""

if [[ $PUBLIC_MODE == true ]]; then
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  PUBLIC DEPLOYMENT CHECKLIST${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Before going public, complete these steps:"
    echo ""
    echo "  1. Set up a reverse proxy (nginx recommended)"
    echo "     - Copy docker_images/nginx/ottobot.conf.template to /etc/nginx/sites-available/"
    echo "     - Replace \${DOMAIN_NAME} with: $DOMAIN_NAME"
    echo ""
    echo "  2. Get SSL certificate:"
    echo "     sudo certbot --nginx -d $DOMAIN_NAME"
    echo ""
    echo "  3. Configure firewall:"
    echo "     sudo ufw allow 80/tcp"
    echo "     sudo ufw allow 443/tcp"
    echo "     sudo ufw enable"
    echo ""
    echo "  4. Set up fail2ban:"
    echo "     sudo apt install fail2ban"
    echo "     sudo systemctl enable fail2ban"
    echo ""
    echo "  5. Start with public config:"
    echo "     cd docker_images"
    echo "     docker compose -f ottobot_docker.yaml -f docker-compose.public.yaml up -d"
    echo ""
fi

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "To start Ottobot locally:"
echo "  cd $PROJECT_ROOT/docker_images"
echo "  docker compose -f ottobot_docker.yaml up -d"
echo ""

#!/bin/bash

# ============================================================================
# Restore Docker Setup from Backup
# ============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "  Ottobot: Restore from Backup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: Check if Backup Images Exist
# ============================================================================

echo -e "${YELLOW}[STEP 1/5] Checking for backup images...${NC}"
echo ""

BACKUPS_FOUND=0

if docker images | grep -q ollama_ottobot_backup; then
    echo -e "${GREEN}  ✓ Found ollama_ottobot_backup${NC}"
    BACKUPS_FOUND=$((BACKUPS_FOUND + 1))
else
    echo -e "${RED}  ✗ ollama_ottobot_backup not found${NC}"
fi

if docker images | grep -q open-webui_backup; then
    echo -e "${GREEN}  ✓ Found open-webui_backup${NC}"
    BACKUPS_FOUND=$((BACKUPS_FOUND + 1))
else
    echo -e "${RED}  ✗ open-webui_backup not found${NC}"
fi

if docker images | grep -q pipelines_backup; then
    echo -e "${GREEN}  ✓ Found pipelines_backup${NC}"
    BACKUPS_FOUND=$((BACKUPS_FOUND + 1))
else
    echo -e "${RED}  ✗ pipelines_backup not found${NC}"
fi

echo ""

if [ $BACKUPS_FOUND -eq 0 ]; then
    echo -e "${RED}Error: No backup images found!${NC}"
    echo "Please run backup_and_fresh_start.sh first to create backups."
    exit 1
fi

echo -e "${GREEN}Found $BACKUPS_FOUND backup image(s)${NC}"
echo ""

# ============================================================================
# STEP 2: Stop and Remove Current Containers
# ============================================================================

echo -e "${YELLOW}[STEP 2/5] Stopping and removing current containers...${NC}"
echo ""

# Stop containers if running
for container in ollama_ottobot open-webui pipelines; do
    if docker ps | grep -q $container; then
        echo "  → Stopping $container..."
        docker stop $container
        echo -e "${GREEN}  ✓ Stopped $container${NC}"
    fi
done

# Remove containers
for container in ollama_ottobot open-webui pipelines; do
    if docker ps -a | grep -q $container; then
        echo "  → Removing $container..."
        docker rm $container
        echo -e "${GREEN}  ✓ Removed $container${NC}"
    fi
done

echo ""

# ============================================================================
# STEP 3: Restore Volumes (Optional)
# ============================================================================

echo -e "${YELLOW}[STEP 3/5] Checking for volume backups...${NC}"
echo ""

BACKUP_DIR="$(pwd)/docker_backups"

if [ -d "$BACKUP_DIR" ]; then
    echo "Found backup directory: $BACKUP_DIR"
    echo ""
    read -p "Do you want to restore volumes from backup? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Restore ollama_storage
        if [ -f "$BACKUP_DIR/ollama_storage_backup.tar.gz" ]; then
            echo "  → Restoring ollama_storage volume..."
            docker volume create ollama_storage 2>/dev/null || true
            docker run --rm -v ollama_storage:/data -v "$BACKUP_DIR":/backup ubuntu tar xzf /backup/ollama_storage_backup.tar.gz -C /data
            echo -e "${GREEN}  ✓ Ollama volume restored${NC}"
        fi
        
        # Restore open-webui_storage
        if [ -f "$BACKUP_DIR/openwebui_storage_backup.tar.gz" ]; then
            echo "  → Restoring open-webui_storage volume..."
            docker volume create open-webui_storage 2>/dev/null || true
            docker run --rm -v open-webui_storage:/data -v "$BACKUP_DIR":/backup ubuntu tar xzf /backup/openwebui_storage_backup.tar.gz -C /data
            echo -e "${GREEN}  ✓ Open WebUI volume restored${NC}"
        fi
        
        # Restore pipelines
        if [ -f "$BACKUP_DIR/pipelines_backup.tar.gz" ]; then
            echo "  → Restoring pipelines volume..."
            docker volume create pipelines 2>/dev/null || true
            docker run --rm -v pipelines:/data -v "$BACKUP_DIR":/backup ubuntu tar xzf /backup/pipelines_backup.tar.gz -C /data
            echo -e "${GREEN}  ✓ Pipelines volume restored${NC}"
        fi
    else
        echo "  ℹ Skipping volume restoration"
    fi
else
    echo "  ℹ No backup directory found at $BACKUP_DIR"
    echo "  ℹ Skipping volume restoration"
fi

echo ""

# ============================================================================
# STEP 4: Ensure Network Exists
# ============================================================================

echo -e "${YELLOW}[STEP 4/5] Setting up network...${NC}"
echo ""

if ! docker network ls | grep -q openwebui-net; then
    echo "  → Creating openwebui-net network..."
    docker network create openwebui-net
    echo -e "${GREEN}  ✓ Network created${NC}"
else
    echo "  ℹ Network openwebui-net already exists"
fi

echo ""

# ============================================================================
# STEP 5: Start Containers from Backup Images
# ============================================================================

echo -e "${YELLOW}[STEP 5/5] Starting containers from backup images...${NC}"
echo ""

# Start Ollama
if docker images | grep -q ollama_ottobot_backup; then
    echo "  → Starting ollama_ottobot from backup..."
    docker run -d --name ollama_ottobot \
      --network openwebui-net \
      -p 11434:11434 \
      -v ollama_storage:/root/.ollama \
      --restart always \
      --gpus all \
      ollama_ottobot_backup
    echo -e "${GREEN}  ✓ Ollama started${NC}"
fi

# Start Open WebUI
if docker images | grep -q open-webui_backup; then
    echo "  → Starting open-webui from backup..."
    docker run -d --name open-webui \
      --network openwebui-net \
      -p 3000:8080 \
      -v open-webui_storage:/app/backend/data \
      -e OLLAMA_BASE_URL=http://ollama_ottobot:11434 \
      --restart always \
      open-webui_backup
    echo -e "${GREEN}  ✓ Open WebUI started${NC}"
fi

# Start Pipelines
if docker images | grep -q pipelines_backup; then
    echo "  → Starting pipelines from backup..."
    docker run -d --name pipelines \
      --network openwebui-net \
      -p 9099:9099 \
      --add-host=host.docker.internal:host-gateway \
      -v pipelines:/app/pipelines \
      pipelines_backup
    echo -e "${GREEN}  ✓ Pipelines started${NC}"
fi

echo ""
echo "  → Waiting 5 seconds for services to initialize..."
sleep 5

echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo -e "${YELLOW}Verifying restored setup...${NC}"
echo ""

echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo -e "${GREEN}=========================================="
echo "  ✓ Restore Complete!"
echo "==========================================${NC}"
echo ""
echo "Access points:"
echo "  - Open WebUI: http://localhost:3000"
echo "  - Ollama API: http://localhost:11434"
echo "  - Pipelines API: http://localhost:9099"
echo ""
echo "Check logs:"
echo "  docker logs -f ollama_ottobot"
echo "  docker logs -f open-webui"
echo "  docker logs -f pipelines"
echo ""

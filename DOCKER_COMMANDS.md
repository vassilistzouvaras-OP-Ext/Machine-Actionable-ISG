# Docker Commands for Ottobot

## Prerequisites

### 1. Create the external network (only needed once):
```bash
docker network create openwebui-net
```

### 2. Start the Ollama and Open WebUI services:
```bash
docker compose -f docker_images/ottobot_docker.yaml up -d
```

### 3. Connect Open WebUI to the network (if needed):
```bash
docker network connect openwebui-net open-webui
```

### 4. Start the Pipelines container:
```bash
docker run -d --name pipelines --network openwebui-net \
  -p 9099:9099 --add-host=host.docker.internal:host-gateway \
  -v pipelines:/app/pipelines \
  ghcr.io/open-webui/pipelines:main
```

---

## Starting Services

### Start all services (detached mode):
```bash
docker compose -f docker_images/ottobot_docker.yaml up -d
```

### Start with logs visible:
```bash
docker compose -f docker_images/ottobot_docker.yaml up
```

### Rebuild and start (after code changes):
```bash
docker compose -f docker_images/ottobot_docker.yaml up -d --build
```

---

## Stopping Services

### Stop all services (`docker compose`):
```bash
docker compose -f docker_images/ottobot_docker.yaml down
```

### Stop the pipelines container:
```bash
docker stop pipelines
```

### Stop all containers:
```bash
docker stop ollama_ottobot open-webui pipelines
```

### Stop and remove all (including pipelines):
```bash
docker compose -f docker_images/ottobot_docker.yaml down
docker stop pipelines && docker rm pipelines
```

### Stop and remove volumes (clean slate):
```bash
docker compose -f docker_images/ottobot_docker.yaml down -v
docker stop pipelines && docker rm pipelines
docker volume rm pipelines
```

### Stop all running Docker containers (alternative):
```bash
docker stop $(docker ps -q)
```

---

## Checking Status

### List running containers:
```bash
docker ps
```

### List all containers (including stopped):
```bash
docker ps -a
```

### View logs:
```bash
docker compose -f docker_images/ottobot_docker.yaml logs -f
```

### View logs for specific service:
```bash
docker logs -f ollama_ottobot
docker logs -f open-webui
docker logs -f pipelines
```

### View last 50 lines of pipelines logs:
```bash
docker logs pipelines --tail 50
```

---

## Network Information

### List networks:
```bash
docker network ls
```

### Inspect the openwebui-net network:
```bash
docker network inspect openwebui-net
```

### Check which containers are on the network:
```bash
docker network inspect openwebui-net --format='{{range .Containers}}{{.Name}} {{end}}'
```

---

## Services

The setup includes 3 containers:

1. **Ollama** (ollama_ottobot)
   - Port: 11434
   - Container: ollama_ottobot
   - Network: openwebui-net
   - Started by: docker compose

2. **Open WebUI** (open-webui)
   - Port: 3000 (maps to internal 8080)
   - Container: open-webui
   - Network: openwebui-net
   - Connects to Ollama at: http://ollama_ottobot:11434
   - Started by: docker compose

3. **Pipelines** (pipelines)
   - Port: 9099
   - Container: pipelines
   - Network: openwebui-net
   - Volume: pipelines:/app/pipelines
   - Started by: docker run command

---

## Troubleshooting

### Remove all stopped containers:
```bash
docker container prune
```

### Remove all unused networks:
```bash
docker network prune
```

### Restart specific service:
```bash
docker compose -f docker_images/ottobot_docker.yaml restart ollama
docker compose -f docker_images/ottobot_docker.yaml restart open-webui
docker restart pipelines
```

### Force recreate containers:
```bash
docker compose -f docker_images/ottobot_docker.yaml up -d --force-recreate
```

### Access pipelines container shell:
```bash
docker exec -it pipelines bash
```

### Install packages in pipelines container:
```bash
docker exec -it pipelines pip install <package_name>
```

---

## Access Points

- **Open WebUI**: http://localhost:3000
- **Ollama API**: http://localhost:11434
- **Pipelines API**: http://localhost:9099

---

## Complete Setup from Scratch

If starting fresh, run these commands in order:

```bash
# 1. Create network
docker network create openwebui-net

# 2. Start Ollama and Open WebUI
docker compose -f docker_images/ottobot_docker.yaml up -d

# 3. Connect Open WebUI to network (if needed)
docker network connect openwebui-net open-webui

# 4. Start Pipelines container
docker run -d --name pipelines --network openwebui-net \
  -p 9099:9099 --add-host=host.docker.internal:host-gateway \
  -v pipelines:/app/pipelines \
  ghcr.io/open-webui/pipelines:main

# 5. Pull the model
docker exec -it ollama_ottobot ollama pull ministral-3:8b

# 6. Verify all containers are running
docker ps
```

---

## Complete Shutdown

To stop everything:

```bash
# Stop all containers
docker stop ollama_ottobot open-webui pipelines

# Or use compose + manual stop
docker compose -f docker_images/ottobot_docker.yaml down
docker stop pipelines
```

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

INSTANCE_NAME="${1:-${INSTANCE_NAME:-mybot}}"
REMOVE_VOLUMES=false

if [[ "${1:-}" == "--volumes" ]]; then
  INSTANCE_NAME="${INSTANCE_NAME:-mybot}"
  REMOVE_VOLUMES=true
elif [[ "${2:-}" == "--volumes" ]]; then
  REMOVE_VOLUMES=true
fi

normalize_name() {
  python3 -c 'import re, sys; name=sys.argv[1].strip().lower(); name=re.sub(r"[^a-z0-9-]+", "-", name); name=re.sub(r"-{2,}", "-", name).strip("-"); print(name)' "$1"
}

INSTANCE_NAME="$(normalize_name "$INSTANCE_NAME")"
DEPLOYMENT_DIR="$PROJECT_ROOT/.deployments/$INSTANCE_NAME"
COMPOSE_FILE="$DEPLOYMENT_DIR/docker-compose.yaml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: deployment compose file not found: $COMPOSE_FILE" >&2
  echo "Pass the deployment name explicitly, for example: bash shutdown.sh mybot" >&2
  exit 1
fi

args=(
  docker
  compose
  -p "$INSTANCE_NAME"
  -f "$COMPOSE_FILE"
  down
)

if [[ "$REMOVE_VOLUMES" == true ]]; then
  args+=(--volumes)
fi

"${args[@]}"

echo
echo "Shutdown complete"
echo "  Deployment: $INSTANCE_NAME"
echo "  Open WebUI container stopped: $INSTANCE_NAME-open-webui"
echo "  Pipelines container stopped: $INSTANCE_NAME-pipelines"
if [[ "$REMOVE_VOLUMES" == true ]]; then
  echo "  Volumes removed"
else
  echo "  Volumes preserved. Use 'bash shutdown.sh $INSTANCE_NAME --volumes' to remove them."
fi

#!/usr/bin/env bash
set -euo pipefail

# Edit these variables for your deployment. These values intentionally override
# stale config values in .env; .env is reused only for generated secrets.
CONFIG_INSTANCE_NAME="ISGBot"
CONFIG_OPENAI_BASE_URL="https://api.openai.com/v1"
CONFIG_OPENAI_MODEL="gpt-5.5"
CONFIG_DEFAULT_MODEL="ISGAccessInterface"
CONFIG_ADMIN_EMAIL="ottobot@ails.com"
CONFIG_ADMIN_PASSWORD="otobot1122"
CONFIG_PIPELINE_DIR_1="./pipelines/ISGAccessInterface.py"
CONFIG_SOURCE_DIR_1="./ISG_en_4"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

random_hex() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    python3 -c 'import secrets; print(secrets.token_hex(32))'
  fi
}

detect_public_host() {
  local detected_host
  detected_host="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -n "$detected_host" ]]; then
    echo "$detected_host"
    return
  fi

  hostname 2>/dev/null || true
}

normalize_name() {
  python3 -c 'import re, sys; name=sys.argv[1].strip().lower(); name=re.sub(r"[^a-z0-9-]+", "-", name); name=re.sub(r"-{2,}", "-", name).strip("-"); print(name)' "$1"
}

prompt_openai_api_key() {
  local existing_key="${1:-}"
  local prompt_suffix=""
  local entered_key=""
  local normalized_key=""

  if [[ -n "$existing_key" ]]; then
    prompt_suffix=" (press Enter to keep existing key)"
  fi

  while true; do
    read -r -s -p "Enter OPENAI_API_KEY for upstream OpenAI models${prompt_suffix}: " entered_key
    echo >&2

    if [[ -n "$entered_key" ]]; then
      normalized_key="$(normalize_single_line_secret "$entered_key")" || {
        echo "OPENAI_API_KEY must be a single line." >&2
        continue
      }
      printf '%s\n' "$normalized_key"
      return
    fi

    if [[ -n "$existing_key" ]]; then
      normalized_key="$(normalize_single_line_secret "$existing_key")" || {
        echo "Existing OPENAI_API_KEY in .env is invalid; please enter it again." >&2
        existing_key=""
        continue
      }
      printf '%s\n' "$normalized_key"
      return
    fi

    echo "OPENAI_API_KEY is required." >&2
  done
}

normalize_single_line_secret() {
  local value="${1//$'\r'/}"

  while [[ "$value" == $'\n'* ]]; do
    value="${value#$'\n'}"
  done

  while [[ "$value" == *$'\n' ]]; do
    value="${value%$'\n'}"
  done

  if [[ "$value" == *$'\n'* ]]; then
    return 1
  fi

  printf '%s\n' "$value"
}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

INSTANCE_NAME="${CONFIG_INSTANCE_NAME}"
OPENAI_BASE_URL="${CONFIG_OPENAI_BASE_URL}"
OPENAI_MODEL="${OPENAI_MODEL:-$CONFIG_OPENAI_MODEL}"
DEFAULT_MODEL="${CONFIG_DEFAULT_MODEL}"
ADMIN_EMAIL="${CONFIG_ADMIN_EMAIL}"
ADMIN_PASSWORD="${CONFIG_ADMIN_PASSWORD}"
PIPELINE_DIR_1="${CONFIG_PIPELINE_DIR_1}"
SOURCE_DIR_1="${CONFIG_SOURCE_DIR_1}"
PUBLIC_HOST="${PUBLIC_HOST:-}"

OPENAI_API_KEY="$(prompt_openai_api_key "${OPENAI_API_KEY:-}")"
OPENAI_API_KEY="$(normalize_single_line_secret "$OPENAI_API_KEY")"
PIPELINE_API_KEY="${PIPELINE_API_KEY:-$(random_hex)}"
PIPELINES_API_KEY="$PIPELINE_API_KEY"
INSTANCE_NAME="$(normalize_name "$INSTANCE_NAME")"

PIPELINE_DIRS=("$PIPELINE_DIR_1")
idx=2
while true; do
  var_name="PIPELINE_DIR_$idx"
  var_value="${!var_name:-}"
  [[ -n "$var_value" ]] || break
  PIPELINE_DIRS+=("$var_value")
  idx=$((idx + 1))
done

SOURCE_DIRS=("$SOURCE_DIR_1")
idx=2
while true; do
  var_name="SOURCE_DIR_$idx"
  var_value="${!var_name:-}"
  [[ -n "$var_value" ]] || break
  SOURCE_DIRS+=("$var_value")
  idx=$((idx + 1))
done

# Interactive prompt for public port
read -p "Bind the deployment to a public port on all interfaces instead of localhost only? [y/N]: " USE_PUBLIC_PORT
USE_PUBLIC_PORT=$(printf '%s' "$USE_PUBLIC_PORT" | tr '[:upper:]' '[:lower:]') # Convert to lowercase (portable, works on bash 3.2/macOS)

args=(
  "$PROJECT_ROOT/scripts/deploy.py"
  --name "$INSTANCE_NAME"
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-api-key "$OPENAI_API_KEY"
  --openai-model "$OPENAI_MODEL"
  --pipelines-api-key "$PIPELINES_API_KEY"
  --default-model "$DEFAULT_MODEL"
  --admin-email "$ADMIN_EMAIL"
  --openwebui-port 8081
  --pipelines-port 9092
)

if [[ "$USE_PUBLIC_PORT" =~ ^(yes|y)$ ]]; then
  default_public_host="${PUBLIC_HOST:-$(detect_public_host)}"
  read -p "Public host IP or DNS name [${default_public_host}]: " PUBLIC_HOST_INPUT
  PUBLIC_HOST="${PUBLIC_HOST_INPUT:-$default_public_host}"
  if [[ -z "$PUBLIC_HOST" ]]; then
    echo "Error: public host IP or DNS name is required for public deployments." >&2
    exit 1
  fi
  args+=(--public-port)
  args+=(--public-host "$PUBLIC_HOST")
fi


if [[ -n "$ADMIN_PASSWORD" ]]; then
  args+=(--admin-password "$ADMIN_PASSWORD")
fi

for pipeline_dir in "${PIPELINE_DIRS[@]}"; do
  args+=(--pipeline-dir "$pipeline_dir")
done

for source_dir in "${SOURCE_DIRS[@]}"; do
  args+=(--source-dir "$source_dir")
done

echo "Deployment config"
echo "  Instance: $INSTANCE_NAME"
echo "  Default model: $DEFAULT_MODEL"
echo "  Pipeline paths: ${PIPELINE_DIRS[*]}"
echo "  Source directories: ${SOURCE_DIRS[*]}"
if [[ "$USE_PUBLIC_PORT" =~ ^(yes|y)$ ]]; then
  echo "  Public host: $PUBLIC_HOST"
fi

{
  echo "INSTANCE_NAME=$INSTANCE_NAME"
  echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
  echo "OPENAI_MODEL=$OPENAI_MODEL"
  echo "OPENAI_API_KEY=$OPENAI_API_KEY"
  echo "PIPELINE_API_KEY=$PIPELINE_API_KEY"
  echo "PIPELINES_API_KEY=$PIPELINES_API_KEY"
  echo "DEFAULT_MODEL=$DEFAULT_MODEL"
  echo "ADMIN_EMAIL=$ADMIN_EMAIL"
  echo "ADMIN_PASSWORD=$ADMIN_PASSWORD"
  echo "PUBLIC_HOST=$PUBLIC_HOST"
  for idx in "${!PIPELINE_DIRS[@]}"; do
    echo "PIPELINE_DIR_$((idx + 1))=${PIPELINE_DIRS[$idx]}"
  done
  for idx in "${!SOURCE_DIRS[@]}"; do
    echo "SOURCE_DIR_$((idx + 1))=${SOURCE_DIRS[$idx]}"
  done
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

python3 "${args[@]}"

DEPLOYMENT_ENV="$PROJECT_ROOT/.deployments/$INSTANCE_NAME/.env"
if [[ -f "$DEPLOYMENT_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEPLOYMENT_ENV"
  set +a
  echo
  echo "Deployment URLs"
  echo "  Local Open WebUI: $OPENWEBUI_URL"
  echo "  Local Pipelines API: $PIPELINES_URL"
  if [[ -n "${REMOTE_OPENWEBUI_URL:-}" ]]; then
    echo "  Remote Open WebUI: $REMOTE_OPENWEBUI_URL"
  fi
  if [[ -n "${REMOTE_PIPELINES_URL:-}" ]]; then
    echo "  Remote Pipelines API: $REMOTE_PIPELINES_URL"
  fi
  echo "  Port binding: ${BIND_HOST:-127.0.0.1}"
  if [[ -n "${PUBLIC_HOST:-}" ]]; then
    echo "  Public host: $PUBLIC_HOST"
  fi
  echo "  Pipelines accelerator: $PIPELINES_ACCELERATOR"
  if [[ -n "${SELECTED_GPU_INDEX:-}" ]]; then
    echo "  Selected GPU: ${SELECTED_GPU_INDEX} (${SELECTED_GPU_NAME:-unknown}), ${SELECTED_GPU_FREE_MB:-?} MiB free of ${SELECTED_GPU_TOTAL_MB:-?} MiB"
  fi
fi

#!/bin/bash

# Script to automatically upload/update ISGAccessInterface to Open WebUI
# Usage: ./upload_pipeline.sh

# Configuration
PIPELINE_FILE="ISGAccessInterface.py"
OPENWEBUI_URL="http://localhost:3000"
PIPELINES_API_URL="http://localhost:9099"

echo "🚀 Uploading $PIPELINE_FILE to Open WebUI Pipelines..."

# Check if pipeline file exists
if [ ! -f "$PIPELINE_FILE" ]; then
    echo "❌ Error: $PIPELINE_FILE not found!"
    exit 1
fi

# Get the pipeline ID from the filename (without extension)
PIPELINE_ID=$(basename "$PIPELINE_FILE" .py)

# Read the pipeline file content
PIPELINE_CONTENT=$(cat "$PIPELINE_FILE" | jq -Rs .)

# Create the JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
  "id": "$PIPELINE_ID",
  "name": "$PIPELINE_ID",
  "content": $PIPELINE_CONTENT
}
EOF
)

# Upload/Update the pipeline
echo "📤 Sending to Pipelines API at $PIPELINES_API_URL..."

RESPONSE=$(curl -s -X POST "$PIPELINES_API_URL/api/v1/pipelines/upload" \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD")

if [ $? -eq 0 ]; then
    echo "✅ Pipeline uploaded successfully!"
    echo "Response: $RESPONSE"
    echo ""
    echo "🔄 Now restart the pipelines container to load the changes:"
    echo "   docker restart pipelines"
else
    echo "❌ Upload failed!"
    echo "Response: $RESPONSE"
    exit 1
fi

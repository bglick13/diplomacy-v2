#!/bin/bash
# Run the web backend server

set -e

cd "$(dirname "$0")/.."

# Default to mock inference mode
INFERENCE_MODE="${INFERENCE_MODE:-mock}"

echo "Starting Diplomacy Web Backend..."
echo "Inference mode: $INFERENCE_MODE"
echo ""

INFERENCE_MODE=$INFERENCE_MODE PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000

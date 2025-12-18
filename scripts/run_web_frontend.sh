#!/bin/bash
# Run the web frontend server

set -e

cd "$(dirname "$0")/../web/frontend"

echo "Starting Diplomacy Web Frontend..."
echo ""

npm run dev

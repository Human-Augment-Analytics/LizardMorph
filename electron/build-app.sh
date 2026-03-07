#!/bin/bash
# Full build: frontend + backend + Electron app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Step 1: Build frontend ==="
cd "$PROJECT_DIR/frontend"
npm ci
npm run build

echo "=== Step 2: Copy frontend build to Electron ==="
rm -rf "$SCRIPT_DIR/frontend"
cp -r "$PROJECT_DIR/frontend/dist" "$SCRIPT_DIR/frontend"

echo "=== Step 3: Build Python backend ==="
cd "$SCRIPT_DIR"
bash build-backend.sh

echo "=== Step 4: Package Electron app ==="
cd "$SCRIPT_DIR"
npm ci
npx electron-builder --mac

echo "=== Done! App is in $SCRIPT_DIR/release/ ==="

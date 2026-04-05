#!/bin/bash
# Full build: frontend + backend + Electron app
# Usage: build-app.sh [--arch arm64|x64|universal]
#   --arch arm64      Build for Apple Silicon only (default on arm64 Mac)
#   --arch x64        Build for Intel only (default on x64 Mac)
#   --arch universal  Build both arm64 and x64 DMGs (macOS only)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
TARGET_ARCH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Step 1: Build frontend ==="
cd "$PROJECT_DIR/frontend"
npm ci
VITE_BASE_URL="./" npm run build

echo "=== Step 2: Copy frontend build to Electron ==="
rm -rf "$SCRIPT_DIR/frontend"
cp -r "$PROJECT_DIR/frontend/dist" "$SCRIPT_DIR/frontend"

echo "=== Step 3: Build Python backend ==="
cd "$SCRIPT_DIR"

if [[ "$OSTYPE" == "darwin"* ]] && [[ "$TARGET_ARCH" == "universal" ]]; then
    # Build backend for both architectures
    echo "--- Building backend for arm64 ---"
    bash build-backend.sh --arch arm64
    echo "--- Building backend for x64 ---"
    bash build-backend.sh --arch x64
else
    # Build backend for specified or native architecture
    if [[ -n "$TARGET_ARCH" ]]; then
        bash build-backend.sh --arch "$TARGET_ARCH"
    else
        bash build-backend.sh
    fi
fi

echo "=== Step 4: Package Electron app ==="
cd "$SCRIPT_DIR"
npm ci

if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
    npm run build:win
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$TARGET_ARCH" == "universal" ]]; then
        # Build separate DMGs for each architecture
        # First, build arm64
        echo "--- Packaging arm64 DMG ---"
        rm -rf "$SCRIPT_DIR/dist/backend"
        cp -r "$SCRIPT_DIR/dist-arm64/backend" "$SCRIPT_DIR/dist/backend"
        npx electron-builder --mac --arm64 --publish never

        # Then, build x64
        echo "--- Packaging x64 DMG ---"
        rm -rf "$SCRIPT_DIR/dist/backend"
        cp -r "$SCRIPT_DIR/dist-x64/backend" "$SCRIPT_DIR/dist/backend"
        npx electron-builder --mac --x64 --publish never
    elif [[ -n "$TARGET_ARCH" ]]; then
        # Build for specific architecture
        if [[ "$TARGET_ARCH" != "arm64" ]] && [[ "$TARGET_ARCH" != "x64" ]]; then
            echo "Error: --arch must be arm64, x64, or universal"
            exit 1
        fi
        # If backend was built to arch-specific dir, copy it to standard location
        if [[ -d "$SCRIPT_DIR/dist-${TARGET_ARCH}/backend" ]]; then
            rm -rf "$SCRIPT_DIR/dist/backend"
            cp -r "$SCRIPT_DIR/dist-${TARGET_ARCH}/backend" "$SCRIPT_DIR/dist/backend"
        fi
        npx electron-builder --mac --${TARGET_ARCH} --publish never
    else
        npm run build:mac
    fi
else
    npm run build
fi

echo "=== Done! App is in $SCRIPT_DIR/release/ ==="

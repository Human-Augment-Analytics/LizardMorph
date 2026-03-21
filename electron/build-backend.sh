#!/bin/bash
# Builds the Python backend into a standalone executable using PyInstaller
# Usage: build-backend.sh [--arch arm64|x64]

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

echo "=== Building LizardMorph backend ==="

if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
    PYTHON="${PYTHON_PATH:-$HOME/miniconda3/envs/lizard/python.exe}"
else
    PYTHON="${PYTHON_PATH:-$HOME/miniconda3/envs/lizard/bin/python}"
fi

# On macOS, if cross-compiling for x64 on arm64 (or vice versa), use arch prefix
if [[ "$OSTYPE" == "darwin"* ]] && [[ -n "$TARGET_ARCH" ]]; then
    CURRENT_ARCH="$(uname -m)"
    if [[ "$TARGET_ARCH" == "x64" ]] && [[ "$CURRENT_ARCH" == "arm64" ]]; then
        echo "Cross-building for x86_64 via Rosetta"
        PYTHON="arch -x86_64 $PYTHON"
    elif [[ "$TARGET_ARCH" == "arm64" ]] && [[ "$CURRENT_ARCH" == "x86_64" ]]; then
        echo "Cross-building for arm64"
        PYTHON="arch -arm64 $PYTHON"
    fi
fi

echo "Using Python: $PYTHON"

# Install PyInstaller if not present
$PYTHON -m pip install pyinstaller

# Determine output suffix for the dist directory
DIST_SUFFIX=""
if [[ -n "$TARGET_ARCH" ]]; then
    DIST_SUFFIX="-${TARGET_ARCH}"
fi

# Run PyInstaller
cd "$SCRIPT_DIR"
$PYTHON -m PyInstaller pyinstaller.spec --distpath "$SCRIPT_DIR/dist${DIST_SUFFIX}" --workpath "$SCRIPT_DIR/build-temp" --clean -y

echo "=== Backend built to $SCRIPT_DIR/dist${DIST_SUFFIX}/backend ==="
echo "=== Test with: $SCRIPT_DIR/dist${DIST_SUFFIX}/backend/app ==="

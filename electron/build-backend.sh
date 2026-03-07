#!/bin/bash
# Builds the Python backend into a standalone executable using PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building LizardMorph backend ==="

PYTHON="${PYTHON_PATH:-$HOME/miniconda3/envs/lizard/bin/python}"
echo "Using Python: $PYTHON"

# Install PyInstaller if not present
"$PYTHON" -m pip install pyinstaller

# Run PyInstaller
cd "$SCRIPT_DIR"
"$PYTHON" -m PyInstaller pyinstaller.spec --distpath "$SCRIPT_DIR/dist" --workpath "$SCRIPT_DIR/build-temp" --clean

echo "=== Backend built to $SCRIPT_DIR/dist/backend ==="
echo "=== Test with: $SCRIPT_DIR/dist/backend/app ==="

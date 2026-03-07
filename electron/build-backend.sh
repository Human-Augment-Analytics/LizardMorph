#!/bin/bash
# Builds the Python backend into a standalone executable using PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building LizardMorph backend ==="

CONDA_ENV="${CONDA_ENV:-lizard}"

# Install PyInstaller if not present
conda run -n "$CONDA_ENV" pip install pyinstaller

# Run PyInstaller from the correct conda env
cd "$SCRIPT_DIR"
conda run -n "$CONDA_ENV" pyinstaller pyinstaller.spec --distpath "$SCRIPT_DIR/dist" --workpath "$SCRIPT_DIR/build-temp" --clean

echo "=== Backend built to $SCRIPT_DIR/dist/backend ==="
echo "=== Test with: $SCRIPT_DIR/dist/backend/app ==="

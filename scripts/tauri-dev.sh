#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PROJECT_DIR}/scripts/prepare-tauri-dev-sidecar.sh"
exec make -C "${PROJECT_DIR}" dev

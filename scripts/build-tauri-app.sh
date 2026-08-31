#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
npm --prefix "${PROJECT_DIR}/frontend" run build
"${PROJECT_DIR}/scripts/build-tauri-sidecar.sh"

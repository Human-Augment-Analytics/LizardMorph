#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_TRIPLE="$(rustc -vV | awk '/^host:/ { print $2 }')"
SIDECAR_PATH="${PROJECT_DIR}/src-tauri/binaries/python-backend-${TARGET_TRIPLE}"
case "${TARGET_TRIPLE}" in *windows*) SIDECAR_PATH="${SIDECAR_PATH}.exe" ;; esac

mkdir -p "$(dirname "${SIDECAR_PATH}")"
if [[ -s "${SIDECAR_PATH}" ]]; then
  exit 0
fi
printf '#!/usr/bin/env sh\nexit 0\n' > "${SIDECAR_PATH}"
chmod +x "${SIDECAR_PATH}"

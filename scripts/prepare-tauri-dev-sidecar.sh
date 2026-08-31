#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_TRIPLE="$(rustc -vV | awk '/^host:/ { print $2 }')"

if [[ -z "${TARGET_TRIPLE}" ]]; then
  echo "Unable to determine the Rust host target." >&2
  exit 1
fi

SIDECAR_PATH="${PROJECT_DIR}/src-tauri/binaries/python-backend-${TARGET_TRIPLE}"
case "${TARGET_TRIPLE}" in
  *windows*) SIDECAR_PATH="${SIDECAR_PATH}.exe" ;;
esac

mkdir -p "$(dirname "${SIDECAR_PATH}")"
if [[ "${TARGET_TRIPLE}" == *windows* ]]; then
  echo "Tauri development sidecar preparation must run in a Windows shell." >&2
  exit 1
fi

# Tauri validates externalBin even though the Rust debug build intentionally uses
# `make dev` instead. Never replace a real release sidecar just to satisfy that
# validation.
if [[ -s "${SIDECAR_PATH}" ]]; then
  exit 0
fi

printf '#!/usr/bin/env sh\nexit 0\n' > "${SIDECAR_PATH}"
chmod +x "${SIDECAR_PATH}"

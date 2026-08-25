#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_TRIPLE="${TAURI_ENV_TARGET_TRIPLE:-$(rustc -vV | awk '/^host:/ { print $2 }')}"

case "${TARGET_TRIPLE}" in
  aarch64-apple-darwin|x86_64-apple-darwin|x86_64-unknown-linux-gnu|x86_64-pc-windows-msvc) ;;
  *) echo "Unsupported Tauri sidecar target: ${TARGET_TRIPLE}" >&2; exit 1 ;;
esac

SIDECAR_NAME="python-backend-${TARGET_TRIPLE}"
TARGET_SIDECAR="${PROJECT_DIR}/src-tauri/binaries/${SIDECAR_NAME}"
if [[ "${TARGET_TRIPLE}" == *windows* ]]; then TARGET_SIDECAR="${TARGET_SIDECAR}.exe"; fi

if [[ "${TARGET_TRIPLE}" == *windows* && "$(uname -s)" != MINGW* && "$(uname -s)" != MSYS* ]]; then
  echo "PyInstaller cannot cross-compile a Windows sidecar. Build it on Windows." >&2
  exit 1
fi
if [[ "${TARGET_TRIPLE}" == *linux* && "$(uname -s)" != Linux ]]; then
  echo "PyInstaller cannot cross-compile a Linux sidecar. Build it on Linux." >&2
  exit 1
fi

REQUIRED_MODELS=(
  "models/lizard-x-ray/dorsal_predictor_clahe_best.dat"
  "models/lizard-x-ray/lateral_predictor_auto.dat"
  "models/lizard-toe-pad/yolo_obb_6class_h7_int8.onnx"
  "models/lizard-toe-pad/ml_morph_best.dat"
  "models/lizard-toe-pad/lizard_scale.dat"
)
for relative_path in "${REQUIRED_MODELS[@]}"; do
  if [[ ! -f "${PROJECT_DIR}/${relative_path}" ]]; then
    echo "Required desktop model is missing: ${relative_path}" >&2
    echo "Run 'make download-models' or restore the model artifact before packaging." >&2
    exit 1
  fi
done

sidecar_is_current() {
  [[ -x "${TARGET_SIDECAR}" ]] || return 1
  [[ "$(wc -c < "${TARGET_SIDECAR}" | tr -d ' ')" -gt 10000000 ]] || return 1
  local input
  for input in "${PROJECT_DIR}/src-tauri/python-backend.spec" "${PROJECT_DIR}/backend/requirements.txt"; do
    [[ ! "${input}" -nt "${TARGET_SIDECAR}" ]] || return 1
  done
  for input in "${REQUIRED_MODELS[@]}"; do
    [[ ! "${PROJECT_DIR}/${input}" -nt "${TARGET_SIDECAR}" ]] || return 1
  done
  if find "${PROJECT_DIR}/backend" -type f \( -name '*.py' -o -name '*.json' \) \
      -newer "${TARGET_SIDECAR}" -print -quit | grep -q .; then
    return 1
  fi
}

if [[ "${AUTOMORPH_FORCE_SIDECAR_BUILD:-0}" != "1" ]] && sidecar_is_current; then
  echo "Reusing current Tauri backend sidecar: ${TARGET_SIDECAR}"
  exit 0
fi

BACKEND_PYTHON="${AUTOMORPH_PYTHON:-${PROJECT_DIR}/backend/.venv/bin/python}"
if [[ ! -x "${BACKEND_PYTHON}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    BACKEND_PYTHON="$(uv run --directory "${PROJECT_DIR}/backend" python -c 'import sys; print(sys.executable)' | tail -n 1)"
  else
    BACKEND_PYTHON="$(command -v python3 || true)"
  fi
fi
if [[ -z "${BACKEND_PYTHON}" || ! -x "${BACKEND_PYTHON}" ]]; then
  echo "A Python environment with the backend dependencies is required." >&2
  exit 1
fi

"${BACKEND_PYTHON}" -c 'import cv2, dlib, flask, numpy, onnxruntime, PIL, psutil' || {
  echo "The selected Python environment is missing backend dependencies. Run 'make setup-backend'." >&2
  exit 1
}

if ! "${BACKEND_PYTHON}" -m PyInstaller --version >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${BACKEND_PYTHON}" 'pyinstaller>=6.11,<7'
  else
    "${BACKEND_PYTHON}" -m pip install 'pyinstaller>=6.11,<7'
  fi
fi

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/automorph-sidecar.XXXXXX")"
cleanup() { rm -rf "${BUILD_DIR}"; }
trap cleanup EXIT

export AUTOMORPH_PROJECT_DIR="${PROJECT_DIR}"
export AUTOMORPH_SIDECAR_NAME="${SIDECAR_NAME}"
"${BACKEND_PYTHON}" -m PyInstaller --clean --noconfirm \
  --distpath "${BUILD_DIR}/dist" --workpath "${BUILD_DIR}/work" \
  "${PROJECT_DIR}/src-tauri/python-backend.spec"

BUILT_SIDECAR="${BUILD_DIR}/dist/${SIDECAR_NAME}"
if [[ "${TARGET_TRIPLE}" == *windows* ]]; then BUILT_SIDECAR="${BUILT_SIDECAR}.exe"; fi
if [[ ! -s "${BUILT_SIDECAR}" ]]; then
  echo "PyInstaller did not produce ${BUILT_SIDECAR}." >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_SIDECAR}")"
STAGED_SIDECAR="$(mktemp "${TARGET_SIDECAR}.new.XXXXXX")"
install -m 755 "${BUILT_SIDECAR}" "${STAGED_SIDECAR}"
mv -f "${STAGED_SIDECAR}" "${TARGET_SIDECAR}"
echo "Built Tauri backend sidecar: ${TARGET_SIDECAR}"

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

MODEL_MANIFEST="${PROJECT_DIR}/src-tauri/desktop-models.txt"
if [[ ! -f "${MODEL_MANIFEST}" ]]; then
  echo "Desktop model manifest is missing: ${MODEL_MANIFEST}" >&2
  exit 1
fi

BUNDLED_MODELS=()
REQUIRED_MODELS=()
while read -r requirement relative_path _destination || [[ -n "${requirement:-}" ]]; do
  requirement="${requirement%%#*}"
  if [[ -z "${requirement}" ]]; then
    continue
  fi
  BUNDLED_MODELS+=("${relative_path}")
  if [[ "${requirement}" == "required" ]]; then
    REQUIRED_MODELS+=("${relative_path}")
  fi
done < "${MODEL_MANIFEST}"

if [[ "${#REQUIRED_MODELS[@]}" -eq 0 ]]; then
  echo "Desktop model manifest lists no required models: ${MODEL_MANIFEST}" >&2
  exit 1
fi

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
  for input in "${PROJECT_DIR}/src-tauri/python-backend.spec" "${PROJECT_DIR}/backend/requirements.txt" "${MODEL_MANIFEST}"; do
    [[ ! "${input}" -nt "${TARGET_SIDECAR}" ]] || return 1
  done
  for input in "${BUNDLED_MODELS[@]}"; do
    [[ ! "${PROJECT_DIR}/${input}" -nt "${TARGET_SIDECAR}" ]] || return 1
  done
  if find "${PROJECT_DIR}/backend" \
      \( -type d \( -name '.venv' -o -name 'venv' -o -name 'env' -o -name '__pycache__' \
                   -o -name '.pytest_cache' -o -name 'sessions' -o -name 'upload' \
                   -o -name 'outputs' -o -name 'color_constrasted' -o -name 'invert_image' \
                   -o -name 'tps_download' -o -name 'image_download' \) -prune \) -o \
      \( -type f \( -name '*.py' -o -name '*.json' \) -newer "${TARGET_SIDECAR}" -print -quit \) \
      | grep -q .; then
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

PREFLIGHT_MODULES='cv2, dlib, flask, numpy, onnxruntime, PIL, psutil'
if [[ "${TARGET_TRIPLE}" == *darwin* ]]; then
  PREFLIGHT_MODULES="${PREFLIGHT_MODULES}, Vision, Quartz, objc"
fi
"${BACKEND_PYTHON}" -c "import ${PREFLIGHT_MODULES}" || {
  echo "The selected Python environment is missing backend dependencies (${PREFLIGHT_MODULES}). Run 'make setup-backend'." >&2
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

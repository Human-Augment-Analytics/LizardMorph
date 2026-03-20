# LizardMorph Electron Desktop App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Package LizardMorph as a self-contained Electron desktop app with bundled Python backend and native GPU-accelerated inference.

**Architecture:** Electron main process spawns a PyInstaller-bundled Flask server on a dynamic port. The React frontend loads in the Electron renderer and talks to `http://localhost:<port>`. All ML inference (dlib + ONNX Runtime with CoreML/DirectML GPU) runs natively in the Python process. Models are bundled in app resources.

**Tech Stack:** Electron, electron-builder, PyInstaller, onnxruntime (with CoreML), existing Flask backend + React frontend.

---

## Task 1: Scaffold Electron project

**Files:**
- Create: `electron/package.json`
- Create: `electron/main.js`
- Create: `electron/.gitignore`

**Step 1: Create `electron/package.json`**

```json
{
  "name": "lizardmorph",
  "version": "1.0.0",
  "description": "LizardMorph - Automated lizard morphology analysis",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "build": "electron-builder"
  },
  "dependencies": {},
  "devDependencies": {
    "electron": "^33.0.0",
    "electron-builder": "^25.0.0"
  }
}
```

**Step 2: Create `electron/main.js` with minimal window**

A minimal Electron main process that opens a window loading a placeholder page. No Python spawning yet — just verify Electron works.

```js
const { app, BrowserWindow } = require("electron");
const path = require("path");

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // For now, load a placeholder. Will point to frontend build later.
  mainWindow.loadURL("data:text/html,<h1>LizardMorph Desktop</h1><p>Electron shell works.</p>");
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  app.quit();
});
```

**Step 3: Create `electron/preload.js`**

```js
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  isElectron: true,
});
```

**Step 4: Create `electron/.gitignore`**

```
node_modules/
dist/
build/
*.dmg
*.app
```

**Step 5: Install dependencies and verify Electron launches**

```bash
cd electron && npm install && npm start
```

Expected: An Electron window opens showing "LizardMorph Desktop" placeholder text.

**Step 6: Commit**

```bash
git add electron/
git commit -m "feat: scaffold Electron shell with minimal window"
```

---

## Task 2: Python backend spawner in Electron

**Files:**
- Create: `electron/python-backend.js`
- Modify: `electron/main.js`
- Modify: `electron/preload.js`

**Step 1: Create `electron/python-backend.js`**

This module finds a free port, spawns the Flask server, and waits for it to be ready.

```js
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

function waitForServer(port, timeoutMs = 30000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function tryConnect() {
      if (Date.now() - start > timeoutMs) {
        return reject(new Error(`Backend did not start within ${timeoutMs}ms`));
      }
      const req = require("http").get(`http://127.0.0.1:${port}/api/health`, (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else {
          setTimeout(tryConnect, 500);
        }
      });
      req.on("error", () => setTimeout(tryConnect, 500));
    }
    tryConnect();
  });
}

async function startBackend(isDev) {
  const port = await findFreePort();

  let proc;
  if (isDev) {
    // In development, run the Python script directly
    const backendDir = path.join(__dirname, "..", "backend");
    proc = spawn("python", ["app.py"], {
      cwd: backendDir,
      env: {
        ...process.env,
        API_PORT: String(port),
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
  } else {
    // In production, run the PyInstaller-bundled executable
    const resourcesPath = process.resourcesPath;
    const exePath = path.join(resourcesPath, "backend", "app");
    proc = spawn(exePath, [], {
      env: {
        ...process.env,
        API_PORT: String(port),
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
  }

  proc.stdout.on("data", (data) => {
    console.log(`[backend] ${data.toString().trim()}`);
  });

  proc.stderr.on("data", (data) => {
    console.error(`[backend] ${data.toString().trim()}`);
  });

  proc.on("exit", (code) => {
    console.log(`[backend] exited with code ${code}`);
  });

  await waitForServer(port);
  console.log(`[backend] ready on port ${port}`);

  return { proc, port };
}

function stopBackend(proc) {
  if (proc && !proc.killed) {
    proc.kill("SIGTERM");
    setTimeout(() => {
      if (!proc.killed) proc.kill("SIGKILL");
    }, 5000);
  }
}

module.exports = { startBackend, stopBackend };
```

**Step 2: Add a `/api/health` endpoint to the Flask backend**

Modify `backend/app.py`. Find the line with the existing Flask routes (after CORS setup, around line 260) and add:

```python
@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok"}, 200
```

**Step 3: Update `electron/main.js` to spawn backend**

```js
const { app, BrowserWindow } = require("electron");
const path = require("path");
const { startBackend, stopBackend } = require("./python-backend");

let mainWindow;
let backendProc;
let backendPort;

const isDev = !app.isPackaged;

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Show loading screen while backend starts
  mainWindow.loadURL(
    "data:text/html,<html><body style='display:flex;justify-content:center;align-items:center;height:100vh;font-family:system-ui;background:%23f5f5f5'><div style='text-align:center'><h1>LizardMorph</h1><p>Starting backend server...</p></div></body></html>"
  );

  try {
    const backend = await startBackend(isDev);
    backendProc = backend.proc;
    backendPort = backend.port;

    if (isDev) {
      // In dev, load Vite dev server (assumes it's running)
      mainWindow.loadURL("http://localhost:5173");
    } else {
      // In production, load the built frontend
      const frontendPath = path.join(__dirname, "frontend", "index.html");
      mainWindow.loadFile(frontendPath);
    }
  } catch (err) {
    mainWindow.loadURL(
      `data:text/html,<html><body style='padding:40px;font-family:system-ui'><h1>Startup Error</h1><pre>${err.message}</pre></body></html>`
    );
  }
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  stopBackend(backendProc);
  app.quit();
});

app.on("before-quit", () => {
  stopBackend(backendProc);
});
```

**Step 4: Update `electron/preload.js` to expose backend port**

The backend port needs to reach the renderer so the frontend knows where to send API calls. We'll use a query parameter approach — Electron loads the frontend URL with `?port=<N>`, and the frontend reads it.

```js
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  isElectron: true,
  getBackendPort: () => ipcRenderer.invoke("get-backend-port"),
});
```

Add IPC handler in `electron/main.js` (add after `backendPort` is set, inside `createWindow` after backend starts):

```js
const { app, BrowserWindow, ipcMain } = require("electron");
// ... existing code ...

// Add inside createWindow, after backendPort is assigned:
ipcMain.handle("get-backend-port", () => backendPort);
```

**Step 5: Test — run Electron in dev mode with backend**

```bash
# Terminal 1: start vite dev server
cd frontend && npm run dev

# Terminal 2: start electron (it will spawn Python backend)
cd electron && npm start
```

Expected: Electron window shows loading screen, then the LizardMorph frontend loads. Backend logs appear in the Electron terminal.

**Step 6: Commit**

```bash
git add electron/ backend/app.py
git commit -m "feat: Electron spawns Flask backend on dynamic port"
```

---

## Task 3: Connect frontend API to dynamic backend port

**Files:**
- Modify: `frontend/src/services/config.ts`

**Step 1: Update `config.ts` to detect Electron and use dynamic port**

Read current file at `frontend/src/services/config.ts`. It currently contains:
```typescript
export const API_URL = import.meta.env.VITE_API_URL || "/api";
```

Replace with:

```typescript
async function resolveApiUrl(): Promise<string> {
  // In Electron, get the backend port via IPC
  if (window.electronAPI?.isElectron) {
    try {
      const port = await window.electronAPI.getBackendPort();
      return `http://127.0.0.1:${port}`;
    } catch {
      // fallback
    }
  }
  // Web mode: use env var or relative path (proxied by Vite in dev)
  return import.meta.env.VITE_API_URL || "/api";
}

// Cached promise so we only resolve once
let _apiUrlPromise: Promise<string> | null = null;

export function getApiUrl(): Promise<string> {
  if (!_apiUrlPromise) {
    _apiUrlPromise = resolveApiUrl();
  }
  return _apiUrlPromise;
}

// Synchronous fallback for non-Electron (most existing code paths)
export const API_URL = import.meta.env.VITE_API_URL || "/api";
```

**Step 2: Add TypeScript type declaration for electronAPI**

Create `frontend/src/electron.d.ts`:

```typescript
interface ElectronAPI {
  isElectron: boolean;
  getBackendPort: () => Promise<number>;
}

interface Window {
  electronAPI?: ElectronAPI;
}
```

**Step 3: Update `ApiService.ts` to use async API URL in Electron**

Find all `fetch(\`${API_URL}/` calls in `frontend/src/services/ApiService.ts`. The service needs to await the URL in Electron mode. Add a helper at the top of the file:

```typescript
import { API_URL, getApiUrl } from "./config";

async function apiUrl(): Promise<string> {
  if (window.electronAPI?.isElectron) {
    return getApiUrl();
  }
  return API_URL;
}
```

Then update each fetch call from:
```typescript
const res = await fetch(`${API_URL}/data`, ...);
```
to:
```typescript
const url = await apiUrl();
const res = await fetch(`${url}/data`, ...);
```

Apply this pattern to all endpoints in ApiService.ts.

**Step 4: Verify the web app still works unchanged**

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173` in a browser — should work exactly as before (uses `/api` proxy).

**Step 5: Verify Electron mode works**

```bash
cd electron && npm start
```

Upload a toepad image. Backend should receive the request on the dynamic port.

**Step 6: Commit**

```bash
git add frontend/src/services/config.ts frontend/src/services/ApiService.ts frontend/src/electron.d.ts
git commit -m "feat: frontend dynamically resolves backend port in Electron"
```

---

## Task 4: Remove frontend ONNX inference (optional for Electron)

**Files:**
- Modify: `frontend/src/services/ApiService.ts`

**Context:** In Electron mode, YOLO inference runs on the native backend with GPU acceleration. The frontend OnnxService is no longer needed for Electron. However, we want to keep it working for the web version.

**Step 1: Skip client-side ONNX in Electron mode**

In `ApiService.ts`, find where `OnnxService.detect()` is called before uploading. Wrap it:

```typescript
let clientAnnotations = null;
if (!window.electronAPI?.isElectron) {
  // Web mode: run client-side YOLO
  clientAnnotations = await OnnxService.detect(imageFile);
}
```

When `client_annotations` is null/absent, the backend already runs its own YOLO inference — this is the existing fallback path.

**Step 2: Test both modes**

- Web mode: OnnxService still runs client-side YOLO, sends annotations to backend
- Electron mode: No client-side YOLO, backend handles full pipeline

**Step 3: Commit**

```bash
git add frontend/src/services/ApiService.ts
git commit -m "feat: skip client-side ONNX inference in Electron mode"
```

---

## Task 5: Enable GPU-accelerated ONNX Runtime in backend

**Files:**
- Modify: `backend/ort_inference.py`
- Modify: `backend/requirements.txt`

**Step 1: Update `ort_inference.py` to prefer GPU execution providers**

Read `backend/ort_inference.py`. Find where `InferenceSession` is created with `CPUExecutionProvider`. Update to try GPU providers first:

```python
def _get_execution_providers():
    """Return available execution providers, preferring GPU."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    # Prefer GPU providers in order
    preferred = [
        "CoreMLExecutionProvider",    # macOS Metal GPU
        "DmlExecutionProvider",       # Windows DirectML
        "CUDAExecutionProvider",      # NVIDIA CUDA
        "CPUExecutionProvider",       # Fallback
    ]
    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers
```

Update the session creation to use this function instead of hardcoded `["CPUExecutionProvider"]`.

**Step 2: Add `onnxruntime-silicon` note to requirements.txt**

Add a comment in `requirements.txt` near the onnxruntime line:

```
onnxruntime  # Use onnxruntime-silicon on Apple Silicon for CoreML support
```

Note: On macOS with Apple Silicon, `pip install onnxruntime` may already include CoreML. If not, the user installs `onnxruntime-silicon`. PyInstaller will bundle whichever is installed.

**Step 3: Test GPU provider detection**

```bash
cd backend && python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected on macOS: Should include `CoreMLExecutionProvider` if available.

**Step 4: Commit**

```bash
git add backend/ort_inference.py backend/requirements.txt
git commit -m "feat: prefer GPU execution providers for ONNX inference"
```

---

## Task 6: PyInstaller spec for bundling backend

**Files:**
- Create: `electron/pyinstaller.spec`
- Create: `electron/build-backend.sh`

**Step 1: Create PyInstaller spec file**

```python
# electron/pyinstaller.spec
# -*- mode: python ; coding: utf-8 -*-
import os
import sys

backend_dir = os.path.join(os.path.dirname(os.path.abspath(SPECPATH)), '..', 'backend')
models_dir = os.path.join(os.path.dirname(os.path.abspath(SPECPATH)), '..', 'models')

a = Analysis(
    [os.path.join(backend_dir, 'app.py')],
    pathex=[backend_dir],
    binaries=[],
    datas=[
        # Bundle model files
        (os.path.join(models_dir, 'lizard-x-ray', 'better_predictor_auto.dat'), 'models/lizard-x-ray'),
        (os.path.join(models_dir, 'lizard-x-ray', 'lateral_predictor_auto.dat'), 'models/lizard-x-ray'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'yolo_obb_6class_h7.onnx'), 'models/lizard-toe-pad'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'toe_predictor_obb.dat'), 'models/lizard-toe-pad'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'finger_predictor_obb.dat'), 'models/lizard-toe-pad'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'lizard_scale.dat'), 'models/lizard-toe-pad'),
    ],
    hiddenimports=[
        'flask',
        'flask_cors',
        'werkzeug',
        'dlib',
        'cv2',
        'numpy',
        'pandas',
        'PIL',
        'onnxruntime',
        'easyocr',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',        # Large, not needed if using ORT for YOLO
        'torchvision',
        'ultralytics',  # Replaced by ort_inference.py
        'matplotlib',
        'tkinter',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='backend',
)
```

**Step 2: Create build script**

```bash
#!/bin/bash
# electron/build-backend.sh
# Builds the Python backend into a standalone executable using PyInstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building LizardMorph backend ==="

# Install PyInstaller if not present
pip install pyinstaller

# Run PyInstaller
cd "$SCRIPT_DIR"
pyinstaller pyinstaller.spec --distpath "$SCRIPT_DIR/dist" --workpath "$SCRIPT_DIR/build-temp" --clean

echo "=== Backend built to $SCRIPT_DIR/dist/backend ==="
echo "=== Test with: $SCRIPT_DIR/dist/backend/app ==="
```

**Step 3: Update backend `app.py` to resolve model paths relative to PyInstaller bundle**

Add near the top of `backend/app.py` (after imports, before model path env vars):

```python
def _resource_path(relative_path):
    """Resolve path relative to PyInstaller bundle or project root."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base = sys._MEIPASS
    else:
        # Running as script
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)
```

Then update the model path defaults. For each `os.getenv("SOME_MODEL", "../models/...")` call, wrap the default:

```python
TOEPAD_YOLO_MODEL = os.getenv("TOEPAD_YOLO_MODEL") or _resource_path("models/lizard-toe-pad/yolo_obb_6class_h7.onnx")
```

Apply this pattern to all model path variables (dorsal predictor, lateral predictor, toe predictor, finger predictor, scale predictor, YOLO model, ID extractor model).

**Step 4: Test PyInstaller build**

```bash
cd electron && bash build-backend.sh
./dist/backend/app
```

Expected: Flask server starts and responds to `/health`.

**Step 5: Commit**

```bash
git add electron/pyinstaller.spec electron/build-backend.sh backend/app.py
git commit -m "feat: PyInstaller spec to bundle Python backend with models"
```

---

## Task 7: Electron-builder packaging for macOS

**Files:**
- Modify: `electron/package.json` (add build config)
- Create: `electron/build-app.sh`

**Step 1: Add electron-builder config to `electron/package.json`**

Add a `"build"` section:

```json
{
  "build": {
    "appId": "com.lizardmorph.desktop",
    "productName": "LizardMorph",
    "directories": {
      "output": "release"
    },
    "mac": {
      "category": "public.app-category.education",
      "target": ["dmg"],
      "icon": "icons/icon.icns"
    },
    "extraResources": [
      {
        "from": "dist/backend",
        "to": "backend",
        "filter": ["**/*"]
      }
    ],
    "files": [
      "main.js",
      "preload.js",
      "python-backend.js",
      "frontend/**/*"
    ]
  }
}
```

**Step 2: Create full build script**

```bash
#!/bin/bash
# electron/build-app.sh
# Full build: frontend + backend + Electron app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Step 1: Build frontend ==="
cd "$PROJECT_DIR/frontend"
npm ci
npm run build

echo "=== Step 2: Copy frontend build to Electron ==="
rm -rf "$SCRIPT_DIR/frontend"
cp -r "$PROJECT_DIR/frontend/dist" "$SCRIPT_DIR/frontend"

echo "=== Step 3: Build Python backend ==="
cd "$SCRIPT_DIR"
bash build-backend.sh

echo "=== Step 4: Package Electron app ==="
cd "$SCRIPT_DIR"
npm ci
npx electron-builder --mac

echo "=== Done! App is in $SCRIPT_DIR/release/ ==="
```

**Step 3: Update `electron/main.js` to load built frontend in production**

The production frontend path should point to the `frontend/` directory copied into the Electron app:

```js
// In production, load the built frontend
const frontendPath = path.join(__dirname, "frontend", "index.html");
mainWindow.loadFile(frontendPath);
```

This is already in the Task 2 version of `main.js`. Verify it's correct.

**Step 4: Update `electron/.gitignore`**

```
node_modules/
dist/
build-temp/
release/
frontend/
*.dmg
*.app
```

**Step 5: Test the full build**

```bash
cd electron && bash build-app.sh
```

Expected: A `.dmg` file appears in `electron/release/`. Open it, drag LizardMorph to Applications, launch it. The app should start, show loading screen, then the full LizardMorph UI.

**Step 6: Commit**

```bash
git add electron/
git commit -m "feat: electron-builder packaging for macOS .dmg"
```

---

## Task 8: Graceful error handling and app polish

**Files:**
- Modify: `electron/main.js`
- Modify: `electron/python-backend.js`

**Step 1: Add backend crash recovery to `main.js`**

Handle the case where the Python backend exits unexpectedly:

```js
backendProc.on("exit", (code) => {
  if (code !== 0 && code !== null && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.loadURL(
      `data:text/html,<html><body style='padding:40px;font-family:system-ui'>
        <h1>Backend Crashed</h1>
        <p>The backend process exited with code ${code}.</p>
        <p>Please restart LizardMorph.</p>
      </body></html>`
    );
  }
});
```

**Step 2: Add startup timeout feedback**

In `python-backend.js`, update `waitForServer` to accept a progress callback so the Electron window can show progress:

```js
// In waitForServer, add optional onRetry callback
function waitForServer(port, timeoutMs = 60000, onRetry = null) {
  // ... existing logic ...
  // Call onRetry?.() on each retry attempt
}
```

**Step 3: Handle macOS app lifecycle (reopen on dock click)**

Add to `electron/main.js`:

```js
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
```

**Step 4: Test crash recovery**

Kill the Python process manually while the app is running. The Electron window should show the crash message.

**Step 5: Commit**

```bash
git add electron/main.js electron/python-backend.js
git commit -m "feat: graceful error handling and macOS app lifecycle"
```

---

## Summary

| Task | Description | Estimated effort |
|------|-------------|-----------------|
| 1 | Scaffold Electron project | Small |
| 2 | Python backend spawner | Medium |
| 3 | Frontend dynamic port resolution | Medium |
| 4 | Skip client ONNX in Electron | Small |
| 5 | GPU-accelerated ONNX Runtime | Small |
| 6 | PyInstaller bundling | Medium-Large |
| 7 | electron-builder packaging | Medium |
| 8 | Error handling and polish | Small |

**Task dependencies:** 1 → 2 → 3 → 4 (sequential). Task 5 is independent. Task 6 depends on 2. Task 7 depends on 3 + 6. Task 8 depends on 7.

**Critical path:** 1 → 2 → 3 → 6 → 7 → 8

**Parallel work:** Task 4 and Task 5 can be done in parallel with Tasks 6-7.

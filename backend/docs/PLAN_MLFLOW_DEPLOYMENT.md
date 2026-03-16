# Plan: MLflow Model Registry + WebGPU Frontend Deployment

## Overview

Train YOLO on PACE/ICE → quantize + export ONNX → push metrics to MLflow (app backend) + upload `.onnx` to GitHub Releases → frontend loads model via ONNX Runtime Web + **WebGPU** (browser-native GPU, no server GPU needed).

```
PACE/ICE Cluster              GitHub Releases          App
┌──────────────────┐         ┌─────────────────┐     ┌────────────────────────────┐
│  Train YOLO (.pt)│         │ v1.0.0/best.onnx│     │  Backend (FastAPI)         │
│       │          │         │ v1.1.0/best.onnx│     │    │                       │
│       ▼          │         │ (public, free)  │     │    ▼                       │
│  Quantize+Export │         └──────┬──────────┘     │  MLflow (SQLite)           │
│  (FP16 ONNX)    │                │                 │  experiment tracking       │
│       │          │                │                 │    │                       │
│       ├─ metrics ├───────────────────────────────▶│  /api/experiments          │
│       │          │                │                 │                            │
│       └─ .onnx  ─┼──────────────▶│                 │  /api/model/latest ◀───────┤
│                  │                │                 │  (fetches from GH Release) │
└──────────────────┘                │                 │                            │
                                    │                 ├────────────────────────────┤
                                    │                 │  Frontend (Browser)        │
                                    │                 │    │                       │
                                    │                 │    ▼                       │
                                    └────────────────▶│  npm: onnxruntime-web      │
                                                      │    │                       │
                                                      │    ├─ WebGPU (user's GPU) │
                                                      │    │  Chrome/Edge 113+    │
                                                      │    │                       │
                                                      │    └─ WASM fallback (CPU) │
                                                      │       all modern browsers │
                                                      │                            │
                                                      │  Loads .onnx directly     │
                                                      │  No conversion needed     │
                                                      └────────────────────────────┘
```

### Why This Architecture? (Alternatives Considered)

| Approach | Problem |
|----------|---------|
| **MLflow on ICE only** | App backend can't access ICE (GT VPN required). Experiment history invisible to app. |
| **Manual scp** | Not automated, error-prone, no versioning. Someone has to remember to copy the file every time. |
| **SSH tunnel through VPN** | Fragile — tunnel drops, needs persistent VPN connection, doesn't scale. |
| **MLflow on ICE + artifact store** | Model files stuck behind VPN. App still can't reach them. |
| **S3 / GCS** | Costs money, requires AWS/GCP account setup, overkill for ~20MB models. |
| **MLflow on app (full, with artifacts)** | Unnecessary — model files are large, better served by GitHub Releases. MLflow only needs to store metadata. |
| **GitHub Releases + MLflow on app (metadata only)** ✅ | Free, public model access, experiment tracking accessible by app, lightweight SQLite. |

---

## Step 1: MLflow Tracking + Model Distribution

Two separate concerns, two separate solutions:

- **Experiment tracking (metrics, params, run history)** → MLflow on app backend (lightweight SQLite, accessible by app)
- **Model files (.onnx)** → GitHub Releases (public, free, no VPN needed)

```
ICE (training)                        App Backend              GitHub Releases
──────────────                        ───────────              ───────────────
Train YOLO
  │
  ├─ Push metrics/params ───────────▶ MLflow (SQLite)
  │   (mAP, loss, config)             ~ few MB
  │                                    App can display
  │                                    experiment history
  │
  └─ Upload .onnx ──────────────────────────────────────────▶ v1.0.0/best.onnx
                                                               Public URL
                                       App fetches ◀──────── No VPN needed
```

### MLflow on App Backend

MLflow runs as a lightweight microservice alongside the app. Only stores metadata (no model files), so a single SQLite file is sufficient.

```bash
# App backend — start MLflow server (metadata only, no artifact store needed)
mlflow server \
  --backend-store-uri sqlite:///app/data/mlflow.db \
  --host 0.0.0.0 --port 5000
```

After training on ICE, push experiment records to the app's MLflow:

```bash
# On ICE, point to app backend's MLflow
export MLFLOW_TRACKING_URI=http://<app-backend-host>:5000
uv run python scripts/deployment/register_model.py --config configs/H8_obb_botonly.yaml
```

### Model Distribution via GitHub Releases

```bash
# On ICE, upload .onnx to GitHub Releases
gh release create v1.0.0 best_fp16.onnx \
  --repo Human-Augment-Analytics/Lizard_Toepads \
  --title "Model Release v1.0.0"
```

**Notes**:
- 2GB per asset limit (our FP16 ONNX is ~20MB, well within limit)
- Free, unlimited bandwidth, public access
- Versioned via git tags — app can pin to a specific version or fetch latest
- `gh` CLI is already available on ICE; one-time `gh auth login` required

---

## Step 2: Quantize + Export ONNX (After Training)

Use **Ultralytics built-in export** for quantization and ONNX conversion. No extra SDK needed.

### Quantization Options

| Format | Ultralytics Command | Size (approx) | Speed | Compatibility |
|--------|-------------------|---------------|-------|---------------|
| FP32 (default) | `model.export(format="onnx")` | ~40MB | Baseline | All |
| **FP16 (recommended)** | `model.export(format="onnx", half=True)` | ~20MB | 2x faster on WebGPU | WebGPU, WASM |
| INT8 dynamic | `model.export(format="onnx", int8=True)` | ~10MB | Fastest on CPU/WASM | WASM only (WebGPU doesn't support INT8 well) |

**Recommendation**: Export **FP16** for WebGPU inference. FP16 is natively supported by WebGPU and gives the best size/speed trade-off for browser deployment.

Full export arguments reference: [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/#arguments)

### Export + Register Script

```python
# scripts/deployment/register_model.py

import os
from pathlib import Path
from ultralytics import YOLO
import mlflow
import yaml

def register_model(config_path, run_name=None, quantize="fp16"):
    """Export ONNX with quantization and register in MLflow."""

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    model_name = train_cfg["name"]  # e.g. H8_obb_botonly

    # Find best weights
    task = train_cfg.get("task", "detect")
    weights_path = f"runs/{task}/{model_name}/weights/best.pt"

    # Export to ONNX with quantization (Ultralytics handles everything)
    model = YOLO(weights_path)
    export_kwargs = {
        "format": "onnx",
        "imgsz": train_cfg.get("imgsz", 1280),
        "simplify": True,      # ONNX graph optimization
        "opset": 17,           # Required for ONNX Runtime Web
    }
    if quantize == "fp16":
        export_kwargs["half"] = True       # FP16 quantization
    elif quantize == "int8":
        export_kwargs["int8"] = True       # INT8 dynamic quantization

    onnx_path = model.export(**export_kwargs)

    # Log to MLflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("lizard-toepad-detection")

    with mlflow.start_run(run_name=run_name or f"{model_name}_{quantize}"):
        # Log training params + quantization info
        mlflow.log_params({
            "task": task,
            "model": train_cfg.get("model"),
            "epochs": train_cfg.get("epochs"),
            "imgsz": train_cfg.get("imgsz"),
            "batch": train_cfg.get("batch"),
            "quantize": quantize,
        })

        # Log artifacts
        mlflow.log_artifact(onnx_path)          # Quantized ONNX model
        mlflow.log_artifact(weights_path)        # Original .pt (backup)
        mlflow.log_artifact(config_path)         # Config used

        # Register in Model Registry
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(
            f"runs:/{run_id}/{Path(onnx_path).name}",
            model_name
        )

    print(f"Registered {model_name} ({quantize}) → MLflow Registry")
    print(f"  ONNX size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
```

### Usage

```bash
# FP16 (recommended for WebGPU)
uv run python scripts/deployment/register_model.py \
  --config configs/H8_obb_botonly.yaml --quantize fp16

# INT8 (smaller, for WASM fallback)
uv run python scripts/deployment/register_model.py \
  --config configs/H8_obb_botonly.yaml --quantize int8

# FP32 (no quantization, largest)
uv run python scripts/deployment/register_model.py \
  --config configs/H8_obb_botonly.yaml --quantize fp32
```

---

## Step 3: App Backend — Pull ONNX from Registry

```python
# app backend (FastAPI)
import mlflow
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/api/model/{model_name}")
def serve_model(model_name: str):
    """Serve the latest production ONNX model for browser download."""
    mlflow.set_tracking_uri("http://ice-mlflow-server:5000")
    model_uri = f"models:/{model_name}/Production"
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    # Find the .onnx file in artifacts
    onnx_file = next(Path(local_path).glob("*.onnx"))
    return FileResponse(onnx_file, media_type="application/octet-stream")
```

---

## Step 4: Frontend — WebGPU Inference with ONNX Runtime Web

### Why WebGPU (not WASM)?

| Execution Provider | Hardware | Speed | Browser Support |
|--------------------|----------|-------|----------------|
| WASM (CPU) | CPU only | Slow (~5-10s per image) | All modern browsers |
| **WebGPU** | **User's GPU (browser-native)** | **Fast (~0.5-1s per image)** | **Chrome 113+, Edge 113+, Firefox (flag)** |
| WebGL | GPU (legacy) | Medium | All browsers |

**WebGPU** is the successor to WebGL — it provides direct GPU access from the browser without any server-side GPU. The user's own laptop/desktop GPU does the work.

### SDK: `onnxruntime-web`

```bash
npm install onnxruntime-web
```

`onnxruntime-web` (npm package by Microsoft) supports three execution providers:
- `webgpu` — **primary** (uses user's GPU via browser WebGPU API)
- `wasm` — **fallback** (CPU, for browsers without WebGPU support)
- `webgl` — legacy GPU (deprecated in favor of WebGPU)

### Frontend Code

```javascript
// frontend/src/inference.js
import * as ort from 'onnxruntime-web';

// Configure WASM threads for CPU fallback
ort.env.wasm.numThreads = 4;

let session = null;

async function loadModel() {
  const modelUrl = '/api/model/H8_obb_botonly';

  // Try WebGPU first, fall back to WASM
  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
    });
    console.log('Using WebGPU (GPU-accelerated)');
  } catch (e) {
    console.warn('WebGPU not available, falling back to WASM (CPU)');
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
  }
}

async function detectToepads(imageElement) {
  if (!session) await loadModel();

  // Preprocess: resize to 1280x1280, normalize to [0,1], NCHW format
  const tensor = preprocessImage(imageElement, 1280);

  // Run inference on user's GPU (WebGPU) or CPU (WASM fallback)
  const feeds = { images: tensor };
  const results = await session.run(feeds);

  // Post-process: decode OBB predictions, apply rotated NMS
  const detections = postprocessOBB(results);
  return detections;
}

function preprocessImage(imgElement, size) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Letterbox resize (maintain aspect ratio)
  const scale = Math.min(size / imgElement.width, size / imgElement.height);
  const newW = Math.round(imgElement.width * scale);
  const newH = Math.round(imgElement.height * scale);
  const padX = (size - newW) / 2;
  const padY = (size - newH) / 2;

  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(imgElement, padX, padY, newW, newH);

  // Extract pixel data → Float32Array, normalize, NCHW
  const imageData = ctx.getImageData(0, 0, size, size);
  const { data } = imageData;
  const float32 = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i] = data[i * 4] / 255.0;                        // R
    float32[i + size * size] = data[i * 4 + 1] / 255.0;      // G
    float32[i + 2 * size * size] = data[i * 4 + 2] / 255.0;  // B
  }
  return new ort.Tensor('float32', float32, [1, 3, size, size]);
}
```

---

## Model Lifecycle

```
Researcher (PACE/ICE)              MLflow Registry              App (Browser)
─────────────────────              ───────────────              ─────────────
Train YOLO (GPU)
  │
Quantize FP16 + Export ONNX
  │ (ultralytics built-in)
  │
register_model.py ──────────────▶ "H8_obb_botonly"
                                  v1: Staging (FP16, ~20MB)
                                    │
                                  Promote → Production
                                    │
                                    └──────────────────▶ Backend pulls ONNX
                                                         Serves /api/model/
                                                              │
                                                              ▼
                                                         Browser loads ONNX
                                                         WebGPU inference
                                                         (user's GPU)
```

---

## Dependencies

| Component | Package / SDK | Version | Purpose |
|-----------|--------------|---------|---------|
| ICE (train + export) | `ultralytics` | >=8.3 | Train YOLO, quantize, export ONNX |
| ICE (registry) | `mlflow` | >=2.0 | Model registry + artifact store |
| App backend | `mlflow` + `fastapi` | — | Pull model, serve API |
| **App frontend** | **`onnxruntime-web`** | **>=1.17** | **WebGPU/WASM inference in browser** |

No additional quantization SDK needed — `ultralytics` handles FP16/INT8 export natively via `model.export()`.

---

## Considerations

- **Model size**: YOLO11m-OBB FP16 ONNX ~20MB. First load cached by browser via Service Worker or `Cache-Control`.
- **WebGPU support**: Chrome 113+, Edge 113+ (stable). Firefox behind flag. Safari in development. For unsupported browsers, automatic WASM fallback.
- **OBB post-processing**: Rotated NMS is not included in ONNX Runtime Web. Options:
  1. Custom JS rotated NMS (~100 lines)
  2. `opencv.js` for `cv.minAreaRect()` + NMS
  3. Bake NMS into the ONNX graph at export time (`model.export(format="onnx", nms=True)` — experimental in ultralytics)
- **INT8 vs FP16**: WebGPU runs FP16 natively on GPU. INT8 only benefits WASM (CPU) fallback. Export both and serve the right one based on browser capability.
- **Privacy**: All inference happens client-side. Images never leave the user's browser.

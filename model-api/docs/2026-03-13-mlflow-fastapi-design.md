# MLflow + FastAPI Read-Only Service Design

**Date**: 2026-03-13
**Status**: Approved

## Overview

A standalone FastAPI service that wraps MLflow's tracking server, exposing a clean read-only API for experiment history and latest production model info. Runs alongside the existing Flask backend and MLflow server via Docker Compose.

## Architecture

```
Docker Compose
├── flask     (port 3000)  — image processing, landmark prediction (existing)
├── fastapi   (port 8000)  — read-only MLflow API (this service)
└── mlflow    (port 5000)  — experiment tracking UI + REST, SQLite storage
      └── mlflow.db        — stores metrics, params, model versions (metadata only)
```

MLflow stores **metadata only** (metrics, params, run history, model version stages). No model artifact files are stored in MLflow. The `.onnx` model files live on GitHub Releases.

## Data Flow

```
ICE Cluster (training)
  │
  ├─ Train YOLO → upload .onnx → GitHub Releases (vX.Y.Z/best_fp16.onnx)
  │
  └─ Push metrics/params → MLflow (MLFLOW_TRACKING_URI=http://<server>:5000)
       └─ Researcher promotes model version → Production in MLflow UI

Frontend (browser)
  │
  ├─ GET /model/latest → FastAPI → queries MLflow (latest Production version)
  │                              → queries GitHub Releases API (download URL)
  │                              → returns { version, stage, github_release_url }
  │
  └─ Downloads .onnx directly from GitHub Release URL
     Runs inference via ONNX Runtime Web (WebGPU / WASM fallback)
```

**Key point**: `/model/latest` fetches the download URL live from the GitHub Releases API
(`GET https://api.github.com/repos/{GITHUB_REPO}/releases/latest`), finding the first `*.onnx` asset.
No manual URL configuration required. `GITHUB_TOKEN` is used for authentication to avoid rate limits.

## API Endpoints

All endpoints are read-only, no authentication required. CORS allows all origins (`*`).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check (FastAPI + MLflow reachability) |
| `GET` | `/experiments` | List all MLflow experiments |
| `GET` | `/experiments/{id}/runs` | List FINISHED runs, ordered by start_time DESC, max 100 |
| `GET` | `/models` | List all registered models and their versions |
| `GET` | `/model/latest` | Latest Production model info + GitHub Release download URL |

### Response Examples

**`GET /health`**
```json
{ "status": "ok", "mlflow_reachable": true }
```
If MLflow is unreachable: `{ "status": "degraded", "mlflow_reachable": false }`

**`GET /model/latest`**
```json
{
  "model_name": "H8_obb_botonly",
  "version": "3",
  "stage": "Production",
  "github_release_url": "https://github.com/Human-Augment-Analytics/Lizard_Toepads/releases/download/v1.2.0/best_fp16.onnx"
}
```
If no Production model exists: **HTTP 404** `{ "detail": "No model in Production stage" }`
If latest release has no `.onnx` asset: **HTTP 404** `{ "detail": "No ONNX asset found in latest release" }`

**`GET /experiments/{id}/runs`**
```json
{
  "experiment_id": "1",
  "runs": [
    {
      "run_id": "abc123",
      "run_name": "H8_obb_botonly_fp16",
      "status": "FINISHED",
      "metrics": { "mAP50": 0.91, "mAP50-95": 0.74 },
      "params": { "epochs": 100, "imgsz": 1280, "quantize": "fp16" }
    }
  ]
}
```

## Components

### `main.py`
FastAPI app entrypoint. Registers routers, configures CORS (`allow_origins=["*"]`), sets `MLFLOW_TRACKING_URI` from environment. Fails fast at startup if `GITHUB_REPO` is missing.

### `routers/experiments.py`
- `GET /experiments` → `mlflow.search_experiments()`
- `GET /experiments/{id}/runs` → `mlflow.search_runs(filter_string="status = 'FINISHED'", order_by=["start_time DESC"], max_results=100)`

### `routers/models.py`
- `GET /models` → `MlflowClient().search_registered_models()`
- `GET /model/latest`:
  1. `MlflowClient().get_latest_versions(name, stages=["Production"])` → if empty, raise HTTP 404
  2. Call GitHub API `GET https://api.github.com/repos/{GITHUB_REPO}/releases/latest` with `Authorization: Bearer {GITHUB_TOKEN}`
  3. Find first asset where `name.endswith(".onnx")` → return `browser_download_url`

### `models/schemas.py`
Pydantic response schemas for type-safe API responses.

## Configuration (Environment Variables)

Stored in `.env` file (not committed to git).

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | No | `http://mlflow:5000` | MLflow server URL |
| `GITHUB_REPO` | **Yes** | — | `org/repo` e.g. `Human-Augment-Analytics/Lizard_Toepads`. Service fails to start if missing. |
| `GITHUB_TOKEN` | **Yes** | — | GitHub Personal Access Token (read:packages scope). Raises rate limit from 60 to 5000 req/hour. |

`.env` example:
```
MLFLOW_TRACKING_URI=http://mlflow:5000
GITHUB_REPO=Human-Augment-Analytics/Lizard_Toepads
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

## Docker Compose

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --backend-store-uri sqlite:///data/mlflow.db --no-serve-artifacts --host 0.0.0.0 --port 5000
    ports: ["5000:5000"]
    volumes: ["./data:/data"]

  fastapi:
    build: ./fastapi
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [mlflow]

  flask:
    build: .
    ports: ["3000:3000"]
    # ... existing flask config
```

`--no-serve-artifacts` explicitly disables artifact storage — MLflow is metadata-only.

## File Structure

```
fastapi/
├── main.py
├── requirements.txt     # fastapi, uvicorn, mlflow, httpx, pydantic, python-dotenv
├── routers/
│   ├── experiments.py
│   └── models.py
├── models/
│   └── schemas.py
└── Dockerfile
```

## Known Limitations

- `/experiments/{id}/runs` returns max 100 runs, ordered by most recent. No pagination in this phase.
- GitHub Releases API is queried live on each `/model/latest` request (no caching in this phase).

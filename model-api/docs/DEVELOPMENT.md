# model-api Development Guide

## Prerequisites

- Docker Desktop running
- `.env` file at repo root (copy from `.env.example`)

```bash
cp .env.example .env
# Fill in GITHUB_TOKEN
```

## Start Services

```bash
# From repo root
docker compose up mlflow model-api
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| FastAPI | http://localhost:8000 |
| FastAPI docs | http://localhost:8000/docs |

## Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","mlflow_reachable":true}

curl http://localhost:8000/experiments
# → {"experiments":[]}
```

## Browse SQLite Database

```bash
# Install datasette (one-time)
uv tool install datasette

# Open database
datasette data/mlflow.db
# → http://localhost:8001
```

## Seed Test Data (no ICE access needed)

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("lizard-toepad-detection")
with mlflow.start_run(run_name="H11_obb_test"):
    mlflow.log_params({"epochs": 300, "batch": 32, "imgsz": 1280})
    mlflow.log_metrics({"mAP50": 0.972, "mAP50-95": 0.942})
```

## Run Tests

```bash
cd model-api
uv run pytest tests/ -v
```

## Rebuild After Code Changes

```bash
docker compose up --build model-api
```

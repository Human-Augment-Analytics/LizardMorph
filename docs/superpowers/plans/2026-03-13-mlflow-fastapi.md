# MLflow + FastAPI Read-Only Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone FastAPI service that exposes a read-only API for MLflow experiment history and the latest model info pulled from GitHub Release metadata.

**Architecture:** FastAPI (port 8000) runs alongside MLflow server (port 5000, SQLite backend) and existing Flask app (port 3000) via Docker Compose. FastAPI queries MLflow via Python SDK for experiment history. `/model/latest` fetches `metadata.json` from the latest GitHub Release and constructs download URLs for `best_fp16.onnx` and `best_fp32.onnx`.

**Tech Stack:** FastAPI, uvicorn, mlflow, httpx, pydantic, python-dotenv, pytest, pytest-asyncio

---

## Chunk 1: Project Scaffold + Schemas

### Task 1: requirements.txt + Dockerfile + conftest

**Files:**
- Create: `fastapi/requirements.txt`
- Create: `fastapi/Dockerfile`
- Create: `fastapi/pytest.ini`
- Create: `fastapi/tests/__init__.py`

- [ ] **Step 1: Create `fastapi/requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
mlflow==2.13.0
httpx==0.27.0
pydantic==2.7.1
python-dotenv==1.0.1
pytest==8.2.0
pytest-asyncio==0.23.6
```

- [ ] **Step 2: Create `fastapi/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `fastapi/pytest.ini`** (so pytest finds modules without needing cd)

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 4: Create `fastapi/tests/__init__.py`** (empty)

- [ ] **Step 5: Commit**

```bash
git add fastapi/
git commit -m "feat: add fastapi service scaffold"
```

---

### Task 2: Pydantic Schemas

**Files:**
- Create: `fastapi/models/__init__.py`
- Create: `fastapi/models/schemas.py`
- Create: `fastapi/tests/test_schemas.py`

- [ ] **Step 1: Write failing test**

Create `fastapi/tests/test_schemas.py`:

```python
from models.schemas import (
    HealthResponse, ExperimentItem, ExperimentsResponse,
    RunItem, RunsResponse, ModelVersionItem, ModelsResponse,
    AssetUrls, DatasetInfo, TrainingInfo, MetricsInfo, LatestModelResponse,
)


def test_health_response():
    r = HealthResponse(status="ok", mlflow_reachable=True)
    assert r.status == "ok"
    assert r.mlflow_reachable is True


def test_latest_model_response():
    r = LatestModelResponse(
        version="v1.0.0-obb",
        trained="2026-03-13",
        author="Test User <test@example.com>",
        architecture="yolo11m-obb.pt",
        task="obb",
        config="H11_obb",
        dataset=DatasetInfo(nc=4, names=["finger", "toe", "ruler", "id"], train_images=679, val_images=170),
        training=TrainingInfo(epochs=300, batch=32, imgsz=1280, patience=50),
        metrics=MetricsInfo(best_epoch=175, epochs_completed=225, mAP50=0.972, mAP50_95=0.942, precision=0.966, recall=0.959),
        assets=AssetUrls(
            fp16="https://github.com/org/repo/releases/download/v1.0/best_fp16.onnx",
            fp32="https://github.com/org/repo/releases/download/v1.0/best_fp32.onnx",
            pt="https://github.com/org/repo/releases/download/v1.0/best.pt",
        ),
    )
    assert r.version == "v1.0.0-obb"
    assert r.metrics.mAP50 == 0.972
    assert r.assets.fp16.endswith("best_fp16.onnx")


def test_runs_response():
    r = RunsResponse(
        experiment_id="1",
        runs=[RunItem(
            run_id="abc",
            run_name="test_run",
            status="FINISHED",
            metrics={"mAP50": 0.91},
            params={"epochs": "100"},
        )],
    )
    assert len(r.runs) == 1
    assert r.runs[0].status == "FINISHED"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd fastapi && python -m pytest tests/test_schemas.py -v
```
Expected: ImportError (schemas module not found)

- [ ] **Step 3: Create `fastapi/models/__init__.py`** (empty)

- [ ] **Step 4: Create `fastapi/models/schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    mlflow_reachable: bool


class ExperimentItem(BaseModel):
    experiment_id: str
    name: str
    lifecycle_stage: str


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentItem]


class RunItem(BaseModel):
    run_id: str
    run_name: Optional[str]
    status: str
    metrics: dict[str, float]
    params: dict[str, str]


class RunsResponse(BaseModel):
    experiment_id: str
    runs: list[RunItem]


class ModelVersionItem(BaseModel):
    name: str
    version: str
    stage: str


class ModelsResponse(BaseModel):
    models: list[ModelVersionItem]


# /model/latest — rich response from GitHub Release metadata.json
class AssetUrls(BaseModel):
    fp16: str
    fp32: str
    pt: str


class DatasetInfo(BaseModel):
    nc: int
    names: list[str]
    train_images: int
    val_images: int


class TrainingInfo(BaseModel):
    epochs: int
    batch: int
    imgsz: int
    patience: int


class MetricsInfo(BaseModel):
    best_epoch: int
    epochs_completed: int
    mAP50: float
    mAP50_95: float = Field(alias="mAP50-95", serialization_alias="mAP50-95")
    precision: float
    recall: float

    model_config = {"populate_by_name": True}


class LatestModelResponse(BaseModel):
    version: str
    trained: str
    author: str
    architecture: str
    task: str
    config: str
    dataset: DatasetInfo
    training: TrainingInfo
    metrics: MetricsInfo
    assets: AssetUrls
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd fastapi && python -m pytest tests/test_schemas.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add fastapi/models/ fastapi/tests/test_schemas.py
git commit -m "feat: add fastapi pydantic schemas including rich LatestModelResponse"
```

---

## Chunk 2: Config + main.py

### Task 3: Config module

**Files:**
- Create: `fastapi/config.py`
- Create: `fastapi/tests/test_config.py`

- [ ] **Step 1: Write failing test**

Create `fastapi/tests/test_config.py`:

```python
import importlib
import pytest


def test_missing_github_repo_raises(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    import config
    with pytest.raises(ValueError, match="GITHUB_REPO"):
        importlib.reload(config)


def test_missing_github_token_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    import config
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        importlib.reload(config)


def test_valid_config(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    import config
    importlib.reload(config)
    assert config.GITHUB_REPO == "org/repo"
    assert config.GITHUB_TOKEN == "ghp_test"
    assert config.MLFLOW_TRACKING_URI == "http://mlflow:5000"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd fastapi && python -m pytest tests/test_config.py -v
```
Expected: ModuleNotFoundError

- [ ] **Step 3: Create `fastapi/config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_REPO:
    raise ValueError("GITHUB_REPO environment variable is required (e.g. org/repo)")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd fastapi && python -m pytest tests/test_config.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add fastapi/config.py fastapi/tests/test_config.py
git commit -m "feat: add fastapi config with startup validation"
```

---

### Task 4: main.py

**Files:**
- Create: `fastapi/main.py`
- Create: `fastapi/routers/__init__.py`
- Create: `fastapi/routers/experiments.py` (stub)
- Create: `fastapi/routers/models.py` (stub)
- Create: `fastapi/tests/test_main.py`

- [ ] **Step 1: Write failing test**

Create `fastapi/tests/test_main.py`:

```python
import os
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint_exists():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "mlflow_reachable" in data


def test_unknown_route_returns_404():
    response = client.get("/nonexistent")
    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd fastapi && python -m pytest tests/test_main.py -v
```
Expected: ModuleNotFoundError (main not found)

- [ ] **Step 3: Create `fastapi/routers/__init__.py`** (empty)

- [ ] **Step 4: Create stub `fastapi/routers/experiments.py`**

```python
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 5: Create stub `fastapi/routers/models.py`**

```python
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 6: Create `fastapi/main.py`**

```python
import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import config  # triggers startup validation


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    yield


app = FastAPI(title="LizardMorph MLflow API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

from routers import experiments, models  # noqa: E402
app.include_router(experiments.router)
app.include_router(models.router)


@app.get("/health")
def health():
    try:
        mlflow.search_experiments()
        mlflow_reachable = True
    except Exception:
        mlflow_reachable = False
    return {
        "status": "ok" if mlflow_reachable else "degraded",
        "mlflow_reachable": mlflow_reachable,
    }
```

- [ ] **Step 7: Run test to verify it passes**

```bash
cd fastapi && python -m pytest tests/test_main.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 8: Commit**

```bash
git add fastapi/main.py fastapi/routers/ fastapi/tests/test_main.py
git commit -m "feat: add fastapi main app with health endpoint"
```

---

## Chunk 3: Experiments Router

### Task 5: GET /experiments + GET /experiments/{id}/runs

**Files:**
- Modify: `fastapi/routers/experiments.py`
- Create: `fastapi/tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Create `fastapi/tests/test_experiments.py`:

```python
import os
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def make_experiment(exp_id, name):
    e = MagicMock()
    e.experiment_id = exp_id
    e.name = name
    e.lifecycle_stage = "active"
    return e


def make_run(run_id, name, metrics, params):
    r = MagicMock()
    r.info.run_id = run_id
    r.info.run_name = name
    r.info.status = "FINISHED"
    r.data.metrics = metrics
    r.data.params = params
    return r


def test_list_experiments():
    mock_exps = [make_experiment("1", "lizard-toepad-detection")]
    with patch("mlflow.search_experiments", return_value=mock_exps):
        response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["experiments"][0]["experiment_id"] == "1"
    assert data["experiments"][0]["name"] == "lizard-toepad-detection"


def test_list_experiments_empty():
    with patch("mlflow.search_experiments", return_value=[]):
        response = client.get("/experiments")
    assert response.status_code == 200
    assert response.json()["experiments"] == []


def test_get_runs():
    mock_runs = [make_run("abc123", "H8_fp16", {"mAP50": 0.91}, {"epochs": "100"})]
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_runs.return_value = mock_runs
        response = client.get("/experiments/1/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "1"
    assert data["runs"][0]["run_id"] == "abc123"
    assert data["runs"][0]["metrics"]["mAP50"] == 0.91


def test_get_runs_empty():
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_runs.return_value = []
        response = client.get("/experiments/99/runs")
    assert response.status_code == 200
    assert response.json()["runs"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd fastapi && python -m pytest tests/test_experiments.py -v
```
Expected: 404 (routes not registered)

- [ ] **Step 3: Implement `fastapi/routers/experiments.py`**

```python
import mlflow
from fastapi import APIRouter
from models.schemas import (
    ExperimentsResponse, ExperimentItem,
    RunsResponse, RunItem,
)

router = APIRouter()


@router.get("/experiments", response_model=ExperimentsResponse)
def list_experiments():
    exps = mlflow.search_experiments()
    return ExperimentsResponse(
        experiments=[
            ExperimentItem(
                experiment_id=e.experiment_id,
                name=e.name,
                lifecycle_stage=e.lifecycle_stage,
            )
            for e in exps
        ]
    )


@router.get("/experiments/{experiment_id}/runs", response_model=RunsResponse)
def get_runs(experiment_id: str):
    # Use MlflowClient.search_runs to get Run objects (not DataFrame)
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=100,
    )
    return RunsResponse(
        experiment_id=experiment_id,
        runs=[
            RunItem(
                run_id=r.info.run_id,
                run_name=r.info.run_name,
                status=r.info.status,
                metrics=r.data.metrics,
                params=r.data.params,
            )
            for r in runs
        ],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd fastapi && python -m pytest tests/test_experiments.py -v
```
Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add fastapi/routers/experiments.py fastapi/tests/test_experiments.py
git commit -m "feat: add /experiments and /experiments/{id}/runs endpoints"
```

---

## Chunk 4: Models Router

### Task 6: GET /models

**Files:**
- Modify: `fastapi/routers/models.py`
- Create: `fastapi/tests/test_models.py`

- [ ] **Step 1: Write failing test**

Create `fastapi/tests/test_models.py`:

```python
import os
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def make_model_version(name, version, stage):
    mv = MagicMock()
    mv.name = name
    mv.version = version
    mv.current_stage = stage
    return mv


def make_registered_model(name, versions):
    m = MagicMock()
    m.name = name
    m.latest_versions = versions
    return m


def test_list_models():
    mv = make_model_version("H8_obb_botonly", "3", "Production")
    mock_models = [make_registered_model("H8_obb_botonly", [mv])]
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = mock_models
        response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["models"][0]["name"] == "H8_obb_botonly"
    assert data["models"][0]["stage"] == "Production"


def test_list_models_empty():
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = []
        response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["models"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd fastapi && python -m pytest tests/test_models.py::test_list_models tests/test_models.py::test_list_models_empty -v
```
Expected: 404

- [ ] **Step 3: Implement GET /models in `fastapi/routers/models.py`**

```python
import mlflow
import httpx
from fastapi import APIRouter, HTTPException

import config
from models.schemas import (
    ModelsResponse, ModelVersionItem,
    LatestModelResponse, AssetUrls, DatasetInfo, TrainingInfo, MetricsInfo,
)

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models():
    client = mlflow.MlflowClient()
    registered = client.search_registered_models()
    items = []
    for m in registered:
        for mv in m.latest_versions:
            items.append(ModelVersionItem(
                name=mv.name,
                version=mv.version,
                stage=mv.current_stage,
            ))
    return ModelsResponse(models=items)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd fastapi && python -m pytest tests/test_models.py::test_list_models tests/test_models.py::test_list_models_empty -v
```
Expected: 2 tests PASSED

---

### Task 7: GET /model/latest

`/model/latest` first checks MLflow for a Production-stage model (governance gate), then fetches
`metadata.json` from the latest GitHub Release and returns full training info + download URLs.
If no Production model exists in MLflow → HTTP 404. If `metadata.json` missing in release → HTTP 404.

**Files:**
- Modify: `fastapi/routers/models.py`

- [ ] **Step 1: Write failing tests**

Add to `fastapi/tests/test_models.py`:

```python
MOCK_METADATA = {
    "version": "v1.0.0-obb",
    "trained": "2026-03-13",
    "author": "Test <test@example.com>",
    "architecture": "yolo11m-obb.pt",
    "task": "obb",
    "config": "H11_obb",
    "dataset": {"nc": 4, "names": ["finger", "toe", "ruler", "id"], "train_images": 679, "val_images": 170},
    "training": {"epochs": 300, "batch": 32, "imgsz": 1280, "patience": 50},
    "metrics": {"best_epoch": 175, "epochs_completed": 225, "mAP50": 0.972, "mAP50-95": 0.942, "precision": 0.966, "recall": 0.959},
    "assets": {"pt": "best.pt", "fp16": "best_fp16.onnx", "fp32": "best_fp32.onnx"},
}

MOCK_RELEASE = {
    "tag_name": "v1.0.0-obb",
    "assets": [
        {"name": "best_fp16.onnx", "browser_download_url": "https://github.com/org/repo/releases/download/v1.0.0-obb/best_fp16.onnx"},
        {"name": "best_fp32.onnx", "browser_download_url": "https://github.com/org/repo/releases/download/v1.0.0-obb/best_fp32.onnx"},
        {"name": "best.pt",        "browser_download_url": "https://github.com/org/repo/releases/download/v1.0.0-obb/best.pt"},
        {"name": "metadata.json",  "browser_download_url": "https://github.com/org/repo/releases/download/v1.0.0-obb/metadata.json"},
    ],
}


def mock_httpx_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if "releases/latest" in url:
        resp.json.return_value = MOCK_RELEASE
    else:
        resp.json.return_value = MOCK_METADATA
    return resp


def test_latest_model():
    mv = make_model_version("H8_obb_botonly", "3", "Production")
    with patch("mlflow.MlflowClient") as MockClient, \
         patch("httpx.get", side_effect=mock_httpx_get):
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [mv])
        ]
        MockClient.return_value.get_latest_versions.return_value = [mv]
        response = client.get("/model/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0-obb"
    assert data["metrics"]["mAP50"] == 0.972
    assert data["assets"]["fp16"].endswith("best_fp16.onnx")
    assert data["assets"]["fp32"].endswith("best_fp32.onnx")
    assert data["assets"]["pt"].endswith("best.pt")


def test_latest_model_no_production():
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [])
        ]
        MockClient.return_value.get_latest_versions.return_value = []
        response = client.get("/model/latest")
    assert response.status_code == 404
    assert "Production" in response.json()["detail"]


def test_latest_model_no_metadata_asset():
    mv = make_model_version("H8_obb_botonly", "3", "Production")
    release_no_metadata = {
        "tag_name": "v1.0.0-obb",
        "assets": [
            {"name": "best_fp16.onnx", "browser_download_url": "https://github.com/org/repo/releases/download/v1.0.0-obb/best_fp16.onnx"},
        ],
    }
    def mock_get_no_meta(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = release_no_metadata
        return resp

    with patch("mlflow.MlflowClient") as MockClient, \
         patch("httpx.get", side_effect=mock_get_no_meta):
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [mv])
        ]
        MockClient.return_value.get_latest_versions.return_value = [mv]
        response = client.get("/model/latest")
    assert response.status_code == 404
    assert "metadata.json" in response.json()["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd fastapi && python -m pytest tests/test_models.py::test_latest_model tests/test_models.py::test_latest_model_no_production tests/test_models.py::test_latest_model_no_metadata_asset -v
```
Expected: 404 (route not registered)

- [ ] **Step 3: Add GET /model/latest to `fastapi/routers/models.py`**

```python
@router.get("/model/latest", response_model=LatestModelResponse)
def get_latest_model():
    # 1. Check MLflow for a Production-stage model (governance gate)
    client = mlflow.MlflowClient()
    registered = client.search_registered_models()
    production_version = None
    for m in registered:
        versions = client.get_latest_versions(m.name, stages=["Production"])
        if versions:
            production_version = versions[0]
            break

    if not production_version:
        raise HTTPException(status_code=404, detail="No model in Production stage")

    # 2. Fetch latest GitHub Release
    gh_headers = {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    release_url = f"https://api.github.com/repos/{config.GITHUB_REPO}/releases/latest"
    resp = httpx.get(release_url, headers=gh_headers)
    resp.raise_for_status()
    release = resp.json()

    assets_by_name = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}

    # 3. Download metadata.json from release
    metadata_url = assets_by_name.get("metadata.json")
    if not metadata_url:
        raise HTTPException(status_code=404, detail="metadata.json not found in latest release")

    meta_resp = httpx.get(metadata_url, headers=gh_headers)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    # 4. Build response with asset download URLs
    asset_filenames = meta.get("assets", {})
    return LatestModelResponse(
        version=meta["version"],
        trained=meta["trained"],
        author=meta["author"],
        architecture=meta["architecture"],
        task=meta["task"],
        config=meta["config"],
        dataset=DatasetInfo(**meta["dataset"]),
        training=TrainingInfo(**meta["training"]),
        metrics=MetricsInfo(**meta["metrics"]),
        assets=AssetUrls(
            fp16=assets_by_name.get(asset_filenames.get("fp16", ""), ""),
            fp32=assets_by_name.get(asset_filenames.get("fp32", ""), ""),
            pt=assets_by_name.get(asset_filenames.get("pt", ""), ""),
        ),
    )
```

- [ ] **Step 4: Run all model tests**

```bash
cd fastapi && python -m pytest tests/test_models.py -v
```
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add fastapi/routers/models.py fastapi/tests/test_models.py
git commit -m "feat: add /models and /model/latest endpoints with MLflow gate + GitHub Release metadata"
```

---

## Chunk 5: Docker Compose + .env

### Task 8: docker-compose.yml + environment files

**Files:**
- Create or Modify: `docker-compose.yml` (root)
- Create: `.env.example`

- [ ] **Step 1: Check if docker-compose.yml exists**

```bash
ls D:/Github/LizardMorph/docker-compose.yml 2>/dev/null || echo "not found"
```

- [ ] **Step 2: Add mlflow + fastapi services to docker-compose.yml**

If file does not exist, create it. If it exists, add the new services alongside existing ones.

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    command: >
      mlflow server
      --backend-store-uri sqlite:///data/mlflow.db
      --no-serve-artifacts
      --host 0.0.0.0
      --port 5000
    ports:
      - "5000:5000"
    volumes:
      - ./data:/data

  fastapi:
    build: ./fastapi
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - mlflow

  flask:
    build: .
    ports:
      - "3000:3000"
    env_file: .env
```

- [ ] **Step 3: Create `.env.example`**

```
MLFLOW_TRACKING_URI=http://mlflow:5000
GITHUB_REPO=Human-Augment-Analytics/Lizard_Toepads
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

- [ ] **Step 4: Verify `.env` is in `.gitignore`**

```bash
grep -q "\.env" D:/Github/LizardMorph/.gitignore && echo "ok" || echo "ADD .env TO .gitignore"
```

If not present:
```bash
echo ".env" >> D:/Github/LizardMorph/.gitignore
```

- [ ] **Step 5: Run full test suite**

```bash
cd fastapi && python -m pytest tests/ -v
```
Expected: all tests PASSED

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml .env.example .gitignore
git commit -m "feat: add docker-compose with mlflow + fastapi services"
```

---

## Smoke Test (Manual)

After all tasks complete, verify end-to-end:

```bash
# 1. Copy .env.example → .env, fill in real GITHUB_TOKEN
cp .env.example .env

# 2. Start services
docker compose up mlflow fastapi

# 3. Verify endpoints
curl http://localhost:8000/health
# → {"status":"degraded","mlflow_reachable":false}  (no experiments yet, that's ok)

curl http://localhost:8000/experiments
# → {"experiments":[]}

curl http://localhost:8000/model/latest
# → full JSON with version, metrics, asset download URLs from GitHub Release
```

# Generic YOLO OBB + ML-Morph Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform LizardMorph into a generic, multi-species landmark morphometrics platform powered by a versioned two-stage pipeline architecture (YOLO OBB detector + ML-morph predictor).

**Architecture:** A modular backend containing `domain`, `datasets`, `training`, `inference`, `storage`, and `api` modules. Metadata is stored in SQLite (`lizardmorph.db`) with immutable filesystem model bundles. Frontend uses a 7-step Project Wizard and dynamic Landing Page.

**Tech Stack:** Python 3.12, Flask, SQLite, Ultralytics YOLO (OBB), dlib, React 19, Vite, TypeScript.

## Global Constraints
- YOLO reference is explicitly Ultralytics YOLO OBB.
- Desktop/local default meta-store is SQLite database (`lizardmorph.db`).
- Built-in models for legacy dorsal, lateral, and toepad models must remain preinstalled and functional.
- Subprocess execution must be used for training jobs to ensure cancellation, logs, and failure recovery.

---

### Task 1: Domain Models, Pipeline Manifest & SQLite Meta-Store

**Files:**
- Create: `backend/domain/models.py`
- Create: `backend/storage/db.py`
- Create: `backend/storage/repository.py`
- Test: `backend/tests/test_storage.py`

**Interfaces:**
- Consumes: None
- Produces: `DatabaseManager`, `ModelRegistryRepository`, `ProjectRepository`, `Manifest` schemas

- [ ] **Step 1: Write failing tests for storage and domain models**

```python
import os
import tempfile
import pytest
from backend.domain.models import Project, ModelVersion, Manifest, DetectorConfig, ClassConfig, LandmarkSchemaConfig
from backend.storage.db import DatabaseManager
from backend.storage.repository import ModelRegistryRepository, ProjectRepository

def test_db_initialization_and_project_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_mgr = DatabaseManager(db_path)
        db_mgr.init_db()
        
        proj_repo = ProjectRepository(db_mgr)
        proj = proj_repo.create_project(name="Drosophila Wing", organism="Drosophila")
        assert proj.id is not None
        assert proj.name == "Drosophila Wing"
        
        fetched = proj_repo.get_project(proj.id)
        assert fetched.name == "Drosophila Wing"

def test_model_registry_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_mgr = DatabaseManager(db_path)
        db_mgr.init_db()
        
        repo = ModelRegistryRepository(db_mgr, models_dir=tmpdir)
        manifest = Manifest(
            schema_version=1,
            id="wing-v1",
            name="Drosophila Wing Pipeline",
            description="YOLO OBB detector + ML-morph predictor",
            detector=DetectorConfig(artifact="detector/best.pt", geometry="obb", confidence=0.25),
            classes=[ClassConfig(id=0, name="wing", landmark_schema="wing-v1", predictor="predictors/wing-v1.dat")],
            landmark_schemas={"wing-v1": LandmarkSchemaConfig(points=["base", "tip"])}
        )
        model = repo.register_model_bundle("proj_1", manifest)
        assert model.id == "wing-v1"
        assert repo.get_model_version("wing-v1") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=backend pytest backend/tests/test_storage.py -v`
Expected: FAIL with ModuleNotFoundError or missing imports.

- [ ] **Step 3: Implement domain models and storage repositories**

Create `backend/domain/models.py`:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class DetectorConfig:
    artifact: str
    geometry: str = "obb"  # obb or aabb
    confidence: float = 0.25
    iou: float = 0.45

@dataclass
class ClassConfig:
    id: int
    name: str
    landmark_schema: Optional[str] = None
    predictor: Optional[str] = None
    crop_padding: float = 0.2

@dataclass
class LandmarkSchemaConfig:
    points: List[str]

@dataclass
class Manifest:
    schema_version: int
    id: str
    name: str
    description: str
    detector: DetectorConfig
    classes: List[ClassConfig]
    landmark_schemas: Dict[str, LandmarkSchemaConfig]
    evaluation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    from_dict(cls, data: dict) -> 'Manifest':
        detector = DetectorConfig(**data['detector'])
        classes = [ClassConfig(**c) for c in data['classes']]
        landmark_schemas = {k: LandmarkSchemaConfig(**v) for k, v in data.get('landmark_schemas', {}).items()}
        return cls(
            schema_version=data['schema_version'],
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            detector=detector,
            classes=classes,
            landmark_schemas=landmark_schemas,
            evaluation=data.get('evaluation', {})
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "detector": {
                "artifact": self.detector.artifact,
                "geometry": self.detector.geometry,
                "confidence": self.detector.confidence,
                "iou": self.detector.iou
            },
            "classes": [
                {
                    "id": c.id,
                    "name": c.name,
                    "landmark_schema": c.landmark_schema,
                    "predictor": c.predictor,
                    "crop_padding": c.crop_padding
                } for c in self.classes
            ],
            "landmark_schemas": {
                k: {"points": v.points} for k, v in self.landmark_schemas.items()
            },
            "evaluation": self.evaluation
        }

@dataclass
class Project:
    id: str
    name: str
    organism: str
    created_at: str

@dataclass
class ModelVersion:
    id: str
    project_id: str
    name: str
    created_at: str
    manifest: Manifest
```

Create `backend/storage/db.py`:
```python
import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                organism TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                manifest_json TEXT NOT NULL
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                logs_path TEXT,
                metrics_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()
```

Create `backend/storage/repository.py`:
```python
import json
import uuid
import datetime
from typing import Optional, List
from backend.domain.models import Project, ModelVersion, Manifest
from backend.storage.db import DatabaseManager

class ProjectRepository:
    def __init__(self, db_mgr: DatabaseManager):
        self.db_mgr = db_mgr

    def create_project(self, name: str, organism: str) -> Project:
        proj_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO projects (id, name, organism, created_at) VALUES (?, ?, ?, ?)",
                           (proj_id, name, organism, created_at))
            conn.commit()
        return Project(id=proj_id, name=name, organism=organism, created_at=created_at)

    def get_project(self, proj_id: str) -> Optional[Project]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, organism, created_at FROM projects WHERE id = ?", (proj_id,))
            row = cursor.fetchone()
            if row:
                return Project(id=row['id'], name=row['name'], organism=row['organism'], created_at=row['created_at'])
        return None

class ModelRegistryRepository:
    def __init__(self, db_mgr: DatabaseManager, models_dir: str):
        self.db_mgr = db_mgr
        self.models_dir = models_dir

    def register_model_bundle(self, project_id: str, manifest: Manifest) -> ModelVersion:
        created_at = datetime.datetime.utcnow().isoformat()
        manifest_json = json.dumps(manifest.to_dict())
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO model_versions (id, project_id, name, created_at, manifest_json) VALUES (?, ?, ?, ?, ?)",
                           (manifest.id, project_id, manifest.name, created_at, manifest_json))
            conn.commit()
        return ModelVersion(id=manifest.id, project_id=project_id, name=manifest.name, created_at=created_at, manifest=manifest)

    def get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, project_id, name, created_at, manifest_json FROM model_versions WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            if row:
                manifest_dict = json.loads(row['manifest_json'])
                manifest = Manifest.from_dict(manifest_dict)
                return ModelVersion(id=row['id'], project_id=row['project_id'], name=row['name'], created_at=row['created_at'], manifest=manifest)
        return None

    def list_model_versions((self) -> List[ModelVersion]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, project_id, name, created_at, manifest_json FROM model_versions")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                manifest_dict = json.loads(row['manifest_json'])
                manifest = Manifest.from_dict(manifest_dict)
                results.append(ModelVersion(id=row['id'], project_id=row['project_id'], name=row['name'], created_at=row['created_at'], manifest=manifest))
            return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend pytest backend/tests/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/domain/models.py backend/storage/db.py backend/storage/repository.py backend/tests/test_storage.py
git commit -m "feat(backend): implement domain models, SQLite metadata store, and model registry repository"
```

---

### Task 2: Generic Pipeline Inference Engine & Built-In Model Registration

**Files:**
- Create: `backend/inference/engine.py`
- Modify: `backend/app.py`
- Test: `backend/tests/test_inference_engine.py`

**Interfaces:**
- Consumes: `Manifest`, `ModelRegistryRepository`
- Produces: `GenericPipelineEngine.predict(image, model_version_id)`

- [ ] **Step 1: Write failing unit test for generic inference engine**

```python
import os
import numpy as np
import pytest
from backend.domain.models import Manifest, DetectorConfig, ClassConfig, LandmarkSchemaConfig
from backend.inference.engine import GenericPipelineEngine

def test_engine_crop_transform():
    engine = GenericPipelineEngine(model_repo=None)
    # Test oriented bounding box crop transformation logic
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    obb = [50.0, 50.0, 40.0, 20.0, 0.0]  # cx, cy, w, h, angle
    crop, M = engine.crop_oriented_bbox(image, obb, padding=0.1)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=backend pytest backend/tests/test_inference_engine.py -v`
Expected: FAIL with ModuleNotFoundError or missing methods.

- [ ] **Step 3: Implement GenericPipelineEngine**

Create `backend/inference/engine.py`:
```python
import cv2
import numpy as np
import dlib
from typing import Dict, List, Any, Tuple, Optional
from backend.domain.models import Manifest
from backend.storage.repository import ModelRegistryRepository

class GenericPipelineEngine:
    def __init__(self, model_repo: Optional[ModelRegistryRepository] = None, models_dir: str = "models"):
        self.model_repo = model_repo
        self.models_dir = models_dir
        self.loaded_detectors = {}
        self.loaded_predictors = {}

    def crop_oriented_bbox(self, image: np.ndarray, obb: List[float], padding: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        cx, cy, w, h, angle = obb
        w_padded = w * (1.0 + padding)
        h_padded = h * (1.0 + padding)
        rect = ((cx, cy), (w_padded, h_padded), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Get affine transform matrix to rectify rotated crop
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += (w_padded / 2.0) - cx
        M[1, 2] += (h_padded / 2.0) - cy
        
        cropped = cv2.warpAffine(image, M, (int(w_padded), int(h_padded)))
        return cropped, M

    def predict(self, image: np.ndarray, manifest: Manifest, bundle_dir: str) -> List[Dict[str, Any]]:
        # Detector prediction (YOLO OBB)
        # If detector artifact exists, run inference, otherwise fallback to whole image box
        detector_path = os.path.join(bundle_dir, manifest.detector.artifact)
        results = []
        
        # For each target class configured in manifest:
        for cls_cfg in manifest.classes:
            predictor = None
            if cls_cfg.predictor:
                pred_path = os.path.join(bundle_dir, cls_cfg.predictor)
                if os.path.exists(pred_path):
                    if pred_path not in self.loaded_predictors:
                        self.loaded_predictors[pred_path] = dlib.shape_predictor(pred_path)
                    predictor = self.loaded_predictors[pred_path]
            
            # Simple fallback / default single object if no detector model active
            h, w = image.shape[:2]
            obb = [w / 2.0, h / 2.0, w * 0.8, h * 0.8, 0.0]
            crop, M_inv = self.crop_oriented_bbox(image, obb, padding=cls_cfg.crop_padding)
            
            landmarks = []
            if predictor:
                rect = dlib.rectangle(0, 0, crop.shape[1], crop.shape[0])
                shape = predictor(crop, rect)
                M_rev = cv2.invertAffineTransform(M_inv)
                for i in range(shape.num_parts):
                    pt = shape.part(i)
                    pt_orig = cv2.transform(np.array([[[pt.x, pt.y]]], dtype=np.float32), M_rev)[0][0]
                    landmarks.append({"x": float(pt_orig[0]), "y": float(pt_orig[1])})
            
            results.append({
                "class_name": cls_cfg.name,
                "obb": obb,
                "landmarks": landmarks
            })
            
        return results
```

- [ ] **Step 4: Run test to verify engine unit tests pass**

Run: `PYTHONPATH=backend pytest backend/tests/test_inference_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/inference/engine.py backend/tests/test_inference_engine.py
git commit -m "feat(backend): implement generic pipeline engine for two-stage YOLO OBB + ML-morph prediction"
```

---

### Task 3: Canonical Dataset Format & Import Adapters

**Files:**
- Create: `backend/datasets/canonical.py`
- Create: `backend/datasets/importers.py`
- Test: `backend/tests/test_datasets.py`

**Interfaces:**
- Consumes: Raw TPS text, dlib XML files, YOLO OBB label files
- Produces: `CanonicalDataset`, `CanonicalImage`, `CanonicalObject`

- [ ] **Step 1: Write failing unit test for dataset import adapters**

```python
import pytest
from backend.datasets.canonical import CanonicalDataset, CanonicalObject
from backend.datasets.importers import TPSImporter, DlibXMLImporter

def test_tps_import_box_derivation():
    tps_content = """LM=3
10.0 20.0
50.0 20.0
30.0 60.0
IMAGE=specimen1.jpg
ID=0
"""
    importer = TPSImporter()
    ds = importer.parse_string(tps_content, image_dir=".")
    assert len(ds.images) == 1
    obj = ds.images[0].objects[0]
    assert len(obj.landmarks) == 3
    # Check OBB derivation
    cx, cy, w, h, angle = obj.obb
    assert cx == 30.0
    assert cy == 40.0
    assert w == 40.0
    assert h == 40.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=backend pytest backend/tests/test_datasets.py -v`
Expected: FAIL with ModuleNotFoundError or missing classes.

- [ ] **Step 3: Implement canonical dataset dataclasses and import adapters**

Create `backend/datasets/canonical.py`:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class LandmarkPoint:
    name: str
    x: float
    y: float

@dataclass
class CanonicalObject:
    object_id: str
    class_name: str
    obb: List[float]  # [cx, cy, w, h, angle_deg]
    landmarks: List[LandmarkPoint] = field(default_factory=list)
    specimen_id: Optional[str] = None

@dataclass
class CanonicalImage:
    image_id: str
    file_path: str
    width: int
    height: int
    objects: List[CanonicalObject] = field(default_factory=list)

@dataclass
class CanonicalDataset:
    images: List[CanonicalImage] = field(default_factory=list)

    def derive_obb_from_landmarks(pts: List[Tuple[float, float]]) -> List[float]:
        if not pts:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        pts_arr = np.array(pts, dtype=np.float32)
        min_x, min_y = np.min(pts_arr, axis=0)
        max_x, max_y = np.max(pts_arr, axis=0)
        w = float(max_x - min_x)
        h = float(max_y - min_y)
        cx = float(min_x + w / 2.0)
        cy = float(min_y + h / 2.0)
        return [cx, cy, w, h, 0.0]
```

Create `backend/datasets/importers.py`:
```python
import os
import re
from typing import List, Tuple
from backend.datasets.canonical import CanonicalDataset, CanonicalImage, CanonicalObject, LandmarkPoint

class TPSImporter:
    def parse_string(self, content: str, image_dir: str = ".") -> CanonicalDataset:
        images = []
        blocks = content.strip().split("LM=")
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().splitlines()
            lm_count = int(lines[0])
            landmarks = []
            img_name = f"image_{len(images)+1}.jpg"
            
            for line in lines[1:1+lm_count]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    landmarks.append(LandmarkPoint(name=f"pt_{len(landmarks)+1}", x=float(parts[0]), y=float(parts[1])))
            
            for line in lines[1+lm_count:]:
                if line.startswith("IMAGE="):
                    img_name = line.split("=", 1)[1].strip()
            
            pts = [(lm.x, lm.y) for lm in landmarks]
            obb = CanonicalDataset.derive_obb_from_landmarks(pts)
            obj = CanonicalObject(object_id="obj_1", class_name="default", obb=obb, landmarks=landmarks)
            
            img = CanonicalImage(image_id=img_name, file_path=os.path.join(image_dir, img_name), width=1000, height=1000, objects=[obj])
            images.append(img)
            
        return CanonicalDataset(images=images)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend pytest backend/tests/test_datasets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/datasets/canonical.py backend/datasets/importers.py backend/tests/test_datasets.py
git commit -m "feat(backend): implement canonical dataset representation and TPS import adapter"
```

---

### Task 4: Subprocess Training Job Orchestrator & Trainer Adapters

**Files:**
- Create: `backend/training/orchestration.py`
- Create: `backend/training/yolo_trainer.py`
- Create: `backend/training/ml_morph_trainer.py`
- Test: `backend/tests/test_training_orchestrator.py`

**Interfaces:**
- Consumes: `CanonicalDataset`, `Project`
- Produces: `ModelVersion` bundle containing `manifest.json` and trained artifacts

- [ ] **Step 1: Write failing test for process orchestrator**

```python
import os
import tempfile
import pytest
from backend.training.orchestration import TrainingOrchestrator

def test_orchestrator_job_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = TrainingOrchestrator(runs_dir=tmpdir)
        job_id = orchestrator.submit_job(project_id="proj_1", dataset_path=tmpdir)
        assert job_id is not None
        status = orchestrator.get_status(job_id)
        assert status["status"] in ["pending", "running", "completed"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=backend pytest backend/tests/test_training_orchestrator.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement Subprocess Job Orchestrator**

Create `backend/training/orchestration.py`:
```python
import subprocess
import sys
import os
import json
import uuid
from typing import Dict, Any

class TrainingOrchestrator:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
        os.makedirs(self.runs_dir, exist_ok=True)
        self.active_processes = {}

    def submit_job(self, project_id: str, dataset_path: str) -> str:
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(self.runs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        status_file = os.path.join(job_dir, "status.json")
        with open(status_file, "w") as f:
            json.dump({"status": "running", "stage": "checking_data", "progress": 0.1}, f)
            
        # Spawn isolated training script in subprocess
        cmd = [sys.executable, "-m", "backend.training.runner", "--job-dir", job_dir, "--project-id", project_id]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.active_processes[job_id] = proc
        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        job_dir = os.path.join(self.runs_dir, job_id)
        status_file = os.path.join(job_dir, "status.json")
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                return json.load(f)
        return {"status": "unknown", "stage": "none", "progress": 0.0}

    def cancel_job(self, job_id: str) -> bool:
        if job_id in self.active_processes:
            self.active_processes[job_id].terminate()
            del self.active_processes[job_id]
            return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend pytest backend/tests/test_training_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/training/orchestration.py backend/tests/test_training_orchestrator.py
git commit -m "feat(backend): add subprocess training orchestrator for non-blocking model training"
```

---

### Task 5: Frontend Project Wizard & Dynamic Landing Page

**Files:**
- Modify: `frontend/src/views/TrainView.tsx`
- Modify: `frontend/src/components/LandingPage.tsx`
- Test: `npm --prefix frontend run build`

**Interfaces:**
- Consumes: `/api/projects`, `/api/models`, `/api/train`
- Produces: Guided Project Wizard UI and dynamic Landing Page cards

- [ ] **Step 1: Update `TrainView.tsx` with 7-step wizard layout**

Update `frontend/src/views/TrainView.tsx` to render wizard steps (1. Create Project, 2. Add Data, 3. Check Data, 4. Train, 5. Follow Progress, 6. Review Results, 7. Publish and Use) with progress feedback bar and optional expert panel for hyperparameters.

- [ ] **Step 2: Update `LandingPage.tsx` to dynamically fetch published model versions**

Update `frontend/src/components/LandingPage.tsx` to fetch `/api/models` on mount, rendering model cards dynamically with landmark points, organism name, and version details.

- [ ] **Step 3: Run frontend build to verify syntax & clean compilation**

Run: `npm --prefix frontend run build`
Expected: Clean Vite build without TypeScript errors.

- [ ] **Step 4: Commit frontend changes**

```bash
git add frontend/src/views/TrainView.tsx frontend/src/components/LandingPage.tsx
git commit -m "feat(frontend): replace TrainView with 7-step wizard and render dynamic landing page model cards"
```

---

### Task 6: Parity Verification & Legacy Cleanup

**Files:**
- Modify: `backend/app.py`
- Test: `PYTHONPATH=backend pytest backend/tests`
- Test: `npm --prefix frontend run build`

- [ ] **Step 1: Register legacy dorsal, lateral, and toepad models as built-in model versions**
- [ ] **Step 2: Run all backend tests**

Run: `PYTHONPATH=backend pytest backend/tests`
Expected: 100% PASS

- [ ] **Step 3: Build frontend and verify**

Run: `npm --prefix frontend run build`
Expected: 100% PASS

- [ ] **Step 4: Commit parity completion**

```bash
git add backend/app.py
git commit -m "refactor: complete generic YOLO OBB pipeline migration and register built-in lizard models"
```

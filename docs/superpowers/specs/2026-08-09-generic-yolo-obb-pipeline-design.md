# Design Specification: Generic YOLO OBB + ML-Morph Two-Stage Pipeline Architecture

## 1. Overview

This document specifies the redesign of LizardMorph into a generic, multi-species landmark morphometrics platform. The architecture transitions from lizard-specific hardcoded views to a versioned two-stage pipeline bundle architecture:

```
Images
  ↓
Stage 1: YOLO OBB Detector (Ultralytics OBB)
  ↓ (Oriented Bounding Boxes + Confidence + Class Names)
Stage 2: ML-Morph Landmark Predictors (Selected by Class / Landmark Schema)
  ↓
Landmarks + Scale & Metadata
```

Legacy lizard models (dorsal, lateral, toepad) are refactored into built-in, preinstalled pipeline bundles using this generic architecture.

---

## 2. Core Domain Objects

1. **Project**: Top-level workspace for a study/organism (e.g., "Drosophila Wings", "Anolis Lizard Dorsal").
2. **LandmarkSchema**: Named topology defining ordered landmark points (e.g., `wing-v1` with points `["base", "tip", "vein_1", "vein_2"]`).
3. **DatasetVersion**: Immutable dataset version containing images, annotated objects (with OBB `[cx, cy, w, h, angle]`), classes, ordered landmarks, and optional specimen IDs.
4. **TrainingRun**: Tracking log and state for background two-stage training jobs (YOLO OBB detector training + ML-morph predictor training + pipeline validation).
5. **ModelVersion**: Published, immutable two-stage pipeline bundle containing the detector artifact, ML-morph artifacts, class-to-schema mappings, crop preprocessing settings, and evaluation metrics.
6. **ClassConfig**: Mapping of a YOLO class name to a specific `LandmarkSchema` and ML-morph predictor (or `null` if no landmarks are predicted for that class).

---

## 3. Pipeline Bundle Manifest Format (`manifest.json`)

Every published `ModelVersion` is packaged as an immutable directory containing artifacts and a `manifest.json`:

```json
{
  "schema_version": 1,
  "id": "drosophila-wing-v1",
  "name": "Drosophila Wing Pipeline",
  "description": "YOLO OBB detector + ML-morph 15-point wing landmark predictor",
  "detector": {
    "artifact": "detector/best.pt",
    "geometry": "obb",
    "confidence": 0.25,
    "iou": 0.45
  },
  "classes": [
    {
      "id": 0,
      "name": "wing",
      "landmark_schema": "wing-v1",
      "predictor": "predictors/wing-v1.dat",
      "crop_padding": 0.2
    }
  ],
  "landmark_schemas": {
    "wing-v1": {
      "points": ["base", "tip", "vein_1", "vein_2"]
    }
  },
  "evaluation": {
    "mAP50": 0.94,
    "mean_landmark_error_px": 2.1
  }
}
```

---

## 4. Backend Service Architecture

The backend monolith is reorganized into clean service modules under `backend/`:

```
backend/
  domain/         # Domain dataclasses & schemas (Project, Dataset, ModelVersion, Manifest)
  datasets/       # Import adapters (TPS, dlib XML, YOLO OBB), canonical format, split generation
  training/       # Job orchestrator, YOLO OBB trainer, ML-morph trainer, pipeline evaluator
  inference/      # Generic pipeline engine (YOLO OBB -> Crop Transform -> ML-morph -> Output)
  storage/        # SQLite meta-store database & immutable filesystem artifact repository
  api/            # Modular Flask Blueprints (projects_api, training_api, inference_api, models_api)
```

### 4.1 Subprocess Job Orchestrator
Training jobs run in isolated external Python subprocesses spawned by `backend/training/orchestration.py`. Subprocesses log progress to JSON/text files in the run storage directory, updating SQLite job status atomically. This ensures:
- Safe job cancellation and timeout handling.
- Captured stdout/stderr logs for diagnostics.
- Graceful error recovery without crashing the web app server.

### 4.2 SQLite Metadata Store
A local SQLite database (`storage/lizardmorph.db`) maintains relational metadata for Projects, DatasetVersions, LandmarkSchemas, TrainingRuns, and ModelVersions.

---

## 5. Canonical Dataset & Import Adapters

All incoming datasets (TPS files, dlib XMLs, YOLO OBB text files) are converted to an internal canonical schema:

```json
{
  "images": [
    {
      "image_id": "img_001",
      "file_path": "images/specimen1.jpg",
      "width": 1920,
      "height": 1080,
      "objects": [
        {
          "object_id": "obj_1",
          "class_name": "wing",
          "obb": [960.0, 540.0, 400.0, 200.0, 15.0],
          "landmarks": [
            {"name": "base", "x": 800.0, "y": 500.0},
            {"name": "tip", "x": 1120.0, "y": 570.0}
          ]
        }
      ]
    }
  ]
}
```

### Automatic Box Derivation for Single-Object TPS
For single-object TPS datasets, oriented bounding boxes (or AABBs with adjustable padding) are automatically derived from landmark minimum bounding rectangles. Users can preview and adjust padding in the UI before training.

---

## 6. Training Pipeline Workflow

1. **YOLO OBB Training**: Ground-truth canonical OBB boxes train the Ultralytics YOLO OBB model.
2. **ML-Morph Training**: Cropped object regions (rectified according to OBB orientation) are used to train dlib ML-morph predictors. Modest box jitter is applied to bounding boxes during landmark training to ensure robustness against detector error.
3. **End-to-End Pipeline Evaluation**: The full pipeline (YOLO predicted OBB → Crop → ML-morph) is evaluated against held-out validation images.

---

## 7. Frontend Project Wizard & Landing Page

### 7.1 Non-Technical Training Wizard (`TrainView.tsx`)
Replaces raw parameter sliders with a 7-step wizard:
1. **Create Project**: Specify organism name & landmark schema.
2. **Add Data**: Upload images + TPS/XML or annotate in app.
3. **Check Data**: Automated validation report (box sizes, missing images, landmark counts, class distribution).
4. **Train**: Single "Train Model" action with accuracy/speed presets (Advanced expert panel collapsible).
5. **Follow Progress**: Live progress tracker through 5 distinct stages (*Checking data → Training detector → Training landmarks → Testing pipeline → Ready*).
6. **Review Results**: Visual inspection of validation predictions, mAP, and landmark error metrics.
7. **Publish & Use**: Publish model bundle to local registry.

### 7.2 Dynamic Landing Page (`LandingPage.tsx`)
Displays cards for all published project models dynamically fetched from `/api/models`, eliminating hardcoded lizard views while retaining built-in lizard models as default preinstalled bundles.

---

## 8. Migration Plan & Parity Verification

1. **Phase 1**: Pipeline manifest, SQLite model registry, generic inference engine implementation.
2. **Phase 2**: Register legacy dorsal, lateral, and toepad models as built-in bundles; map `/api/predict` `view_type` to `model_version_id`.
3. **Phase 3**: Canonical dataset representation & import adapters (TPS, XML, YOLO OBB).
4. **Phase 4**: Subprocess training orchestrator, YOLO OBB trainer, ML-morph trainer, pipeline evaluator.
5. **Phase 5**: Frontend Project Wizard and dynamic Landing Page.
6. **Phase 6**: Parity tests execution & legacy lizard-specific cleanup.

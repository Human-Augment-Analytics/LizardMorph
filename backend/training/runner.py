import os
import sys
import time
import json
import argparse
import math
import re
import posixpath
import signal
import traceback
import threading
from typing import Dict, Any, Optional

try:
    from backend.datasets.canonical import CanonicalDataset
    from backend.training.yolo_trainer import YoloOBBTrainer
    from backend.training.ml_morph_trainer import MLMorphTrainer
    from backend.domain.models import (
        Manifest,
        DetectorConfig,
        ClassConfig,
        LandmarkSchemaConfig,
    )
    from backend.storage.db import DatabaseManager
    from backend.storage.repository import ModelRegistryRepository, ProjectRepository
except ImportError:
    from datasets.canonical import CanonicalDataset
    from training.yolo_trainer import YoloOBBTrainer
    from training.ml_morph_trainer import MLMorphTrainer
    from domain.models import Manifest, DetectorConfig, ClassConfig, LandmarkSchemaConfig
    from storage.db import DatabaseManager
    from storage.repository import ModelRegistryRepository, ProjectRepository


def update_status(
    job_dir: str,
    status: str,
    stage: str,
    progress: float,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    """
    Atomically writes current status information to status.json in job_dir.
    """
    status_path = os.path.join(job_dir, "status.json")

    existing_metrics = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_metrics = existing_data.get("metrics", {})
        except (OSError, json.JSONDecodeError):
            # The atomic write below repairs a truncated prior status file.
            existing_metrics = {}

    merged_metrics = {**existing_metrics, **(metrics or {})}

    status_data = {
        "status": status,
        "stage": stage,
        "progress": float(progress),
        "metrics": merged_metrics,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if error:
        status_data["error"] = str(error)

    tmp_path = f"{status_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    os.replace(tmp_path, status_path)


def check_cancelled(job_dir: str) -> bool:
    """Checks if status.json has been marked as cancelled."""
    status_path = os.path.join(job_dir, "status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "cancelled":
                    return True
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Training job status is unreadable: {error}") from error
    return False


def start_parent_watchdog():
    """Stop an orphaned trainer if its backend parent is terminated."""
    expected_parent = os.getenv("AUTOMORPH_PARENT_PID")
    if not expected_parent:
        return
    try:
        expected_parent_pid = int(expected_parent)
    except ValueError:
        return

    def watch_parent():
        while True:
            if os.getppid() != expected_parent_pid:
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(1.0)

    threading.Thread(target=watch_parent, daemon=True, name="parent-watchdog").start()


def validate_training_dataset(dataset: CanonicalDataset) -> Dict[str, list]:
    schemas: Dict[str, list] = {}
    class_images: Dict[str, set] = {}
    image_ids = set()
    object_ids = set()
    for image in dataset.images:
        if image.image_id in image_ids:
            raise ValueError(f"Duplicate image identifier '{image.image_id}'.")
        image_ids.add(image.image_id)
        if not image.objects:
            raise ValueError(f"Image '{image.file_path}' has no annotated objects.")
        for obj in image.objects:
            object_key = (image.image_id, obj.object_id)
            if object_key in object_ids:
                raise ValueError(
                    f"Duplicate object identifier '{obj.object_id}' in image '{image.image_id}'."
                )
            object_ids.add(object_key)
            class_name = (obj.class_name or "object").strip()
            if len(obj.obb) < 5 or not all(math.isfinite(float(value)) for value in obj.obb[:5]):
                raise ValueError(f"Class '{class_name}' has an invalid oriented bounding box.")
            if obj.obb[2] <= 0 or obj.obb[3] <= 0:
                raise ValueError(f"Class '{class_name}' has an empty oriented bounding box.")
            names = [str(point.name) for point in obj.landmarks]
            if not names:
                raise ValueError(f"Class '{class_name}' has no landmark annotations.")
            if len(names) != len(set(names)):
                raise ValueError(f"Class '{class_name}' has duplicate landmark names.")
            if any(
                not math.isfinite(float(coordinate))
                for point in obj.landmarks
                for coordinate in (point.x, point.y)
            ):
                raise ValueError(f"Class '{class_name}' has a non-finite landmark coordinate.")
            if class_name in schemas and schemas[class_name] != names:
                raise ValueError(
                    f"Class '{class_name}' has inconsistent landmark names or ordering."
                )
            schemas[class_name] = names
            class_images.setdefault(class_name, set()).add(image.image_id)
    if not schemas:
        raise ValueError("Dataset contains no annotated objects for training.")
    for class_name, annotated_images in class_images.items():
        if len(annotated_images) < 2:
            raise ValueError(
                f"Class '{class_name}' must be annotated in at least two different images."
            )
    return schemas


def _artifact_reference(job_id: str, path: Optional[str], job_dir: str) -> str:
    if not path:
        return ""
    absolute_job_dir = os.path.abspath(job_dir)
    absolute_path = os.path.abspath(path)
    if os.path.commonpath([absolute_job_dir, absolute_path]) != absolute_job_dir:
        raise ValueError(f"Model artifact is outside the training run: {path}")
    relative = os.path.relpath(absolute_path, absolute_job_dir).replace(os.sep, "/")
    return posixpath.join("runs", f"job_{job_id}", relative)


def run_training_pipeline(job_dir: str):
    """
    Executes the 5-stage training job pipeline.
    """
    config_path = os.path.join(job_dir, "job_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Job config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        job_config = json.load(f)

    project_id = job_config.get("project_id", "project_default")
    dataset_dict = job_config.get("dataset", {})
    config = job_config.get("config", {})

    mock = config.get("mock", False)
    epochs = config.get("epochs", 1)
    crop_padding = float(config.get("padding", config.get("jitter_padding", 0.2)))
    box_jitter = config.get("box_jitter", 0.05)
    dlib_option_names = {
        "nu",
        "tree_depth",
        "cascade_depth",
        "oversampling_amount",
        "feature_pool_size",
        "num_test_splits",
    }
    dlib_opts = dict(config.get("dlib_options") or {})
    dlib_opts.update(
        {key: config[key] for key in dlib_option_names if key in config}
    )
    test_split = min(0.5, max(0.0, float(config.get("test_split", 0.2))))
    simulate_delay = config.get("simulate_delay", 0.0)

    # Optional simulated delay for testing cancellation
    if simulate_delay > 0:
        update_status(job_dir, "running", "Checking data (simulated delay)", 0.05)
        step_delay = 0.1
        elapsed = 0.0
        while elapsed < simulate_delay:
            if check_cancelled(job_dir):
                return
            time.sleep(step_delay)
            elapsed += step_delay

    # STAGE 1: Checking data
    update_status(job_dir, "running", "Checking data", 0.10)
    canonical_ds = CanonicalDataset.from_dict(dataset_dict)
    if not canonical_ds.images:
        raise ValueError("Dataset contains no images for training.")
    class_schemas = validate_training_dataset(canonical_ds)

    if check_cancelled(job_dir):
        return

    # STAGE 2: Training detector
    update_status(job_dir, "running", "Training detector", 0.15)
    yolo_trainer = YoloOBBTrainer(job_dir)
    data_yaml_path = yolo_trainer.prepare_dataset(
        canonical_ds, train_ratio=1.0 - test_split
    )
    det_metrics = yolo_trainer.train(
        data_yaml_path,
        epochs=epochs,
        mock=mock,
        base_model=config.get("base_model")
        or os.getenv("YOLO_BASE_MODEL", "yolov8n-obb.pt"),
    )
    # Keep inference on Ultralytics' native checkpoint. The ONNX export is
    # useful for distribution, but can introduce OBB decode drift; native PT
    # inference preserves the trained detector geometry.
    detector_artifact_path = det_metrics.get("detector_weights_pt") or det_metrics["detector_weights"]

    if check_cancelled(job_dir):
        return

    # STAGE 3: Training landmarks
    update_status(job_dir, "running", "Training landmarks", 0.55)
    ml_trainer = MLMorphTrainer(job_dir)
    landmark_results = {}
    landmark_models = {}
    safe_class_names = {}
    for class_name in class_schemas:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", class_name).strip("._") or "object"
        folded = safe_name.casefold()
        if folded in safe_class_names:
            raise ValueError(
                f"Class names '{safe_class_names[folded]}' and '{class_name}' resolve to the same artifact name."
            )
        safe_class_names[folded] = class_name

    for class_index, class_name in enumerate(sorted(class_schemas)):
        progress = 0.55 + (class_index / max(1, len(class_schemas))) * 0.30
        update_status(
            job_dir,
            "running",
            f"Training landmarks for {class_name}",
            progress,
        )
        xml_path = ml_trainer.prepare_landmark_dataset(
            canonical_ds,
            jitter_padding=crop_padding,
            box_jitter=box_jitter,
            class_name=class_name,
        )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", class_name).strip("._") or "object"
        class_metrics = ml_trainer.train(
            xml_path,
            output_model_name=f"shape_predictor_{safe_name}.dat",
            custom_options=dlib_opts,
            test_split=test_split,
            mock=mock,
        )
        landmark_results[class_name] = class_metrics
        landmark_models[class_name] = class_metrics["landmark_model"]

    if check_cancelled(job_dir):
        return

    # STAGE 4: Testing pipeline
    update_status(job_dir, "running", "Testing pipeline", 0.90)
    # Validate artifacts exist
    det_weights = det_metrics.get("detector_weights")
    if det_weights and not os.path.exists(det_weights):
        raise FileNotFoundError(f"Detector weights file missing: {det_weights}")
    if not os.path.exists(detector_artifact_path):
        raise FileNotFoundError(f"Detector artifact file missing: {detector_artifact_path}")
    for class_name, landmark_model in landmark_models.items():
        if not os.path.exists(landmark_model):
            raise FileNotFoundError(
                f"Landmark model for class '{class_name}' is missing: {landmark_model}"
            )

    if check_cancelled(job_dir):
        return

    # STAGE 5: Ready - Generate final manifest and bundle artifacts
    test_errors = [
        result.get("test_error")
        for result in landmark_results.values()
        if result.get("test_error") is not None
    ]
    public_detector_metrics = {
        key: value
        for key, value in det_metrics.items()
        if key != "detector_weights"
    }
    public_landmark_metrics = {
        class_name: {
            key: value
            for key, value in result.items()
            if key not in ("landmark_model", "model_path")
        }
        for class_name, result in landmark_results.items()
    }
    all_metrics = {
        **public_detector_metrics,
        "landmarks": public_landmark_metrics,
        "test_error": sum(test_errors) / len(test_errors) if test_errors else None,
    }
    
    job_dir_name = os.path.basename(job_dir)
    job_id = job_dir_name[4:] if job_dir_name.startswith("job_") else job_dir_name
    db_mgr = DatabaseManager(os.path.abspath(os.environ.get("DB_PATH", "lizardmorph.db")))
    db_mgr.init_db()
    repo = ModelRegistryRepository(
        db_mgr,
        runs_dir=os.environ.get("RUNS_DIR"),
    )

    model_name = (
        config.get("model_name")
        or config.get("name")
        or job_config.get("model_name")
        or job_config.get("name")
    )
    if not model_name and project_id:
        project = ProjectRepository(db_mgr).get_project(project_id)
        if project and project.name not in ("default", "project_default"):
            model_name = project.name
    if not model_name:
        model_name = f"Custom Model {job_id[:8]}"

    classes = []
    landmark_schemas = {}
    for class_id, class_name in enumerate(sorted(class_schemas)):
        schema_id = f"class_{class_id}"
        classes.append(
            ClassConfig(
                id=class_id,
                name=class_name,
                landmark_schema=schema_id,
                predictor=_artifact_reference(
                    job_id, landmark_models[class_name], job_dir
                ),
                crop_padding=crop_padding,
            )
        )
        landmark_schemas[schema_id] = LandmarkSchemaConfig(
            points=class_schemas[class_name]
        )

    manifest = Manifest(
        schema_version=1,
        id=job_id,
        name=model_name,
        description=f"User-trained generic YOLO OBB + ML-Morph model: {model_name}",
        detector=DetectorConfig(
            artifact=_artifact_reference(job_id, detector_artifact_path, job_dir),
            geometry="obb",
            confidence=float(config.get("confidence", 0.25)),
        ),
        classes=classes,
        landmark_schemas=landmark_schemas,
        evaluation=all_metrics,
    )

    manifest_path = os.path.join(job_dir, "manifest.json")
    tmp_manifest_path = f"{manifest_path}.tmp"
    with open(tmp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    os.replace(tmp_manifest_path, manifest_path)
    repo.register_model_bundle(project_id or "default", manifest)

    update_status(
        job_dir,
        status="completed",
        stage="Ready",
        progress=1.0,
        metrics=all_metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline Subprocess Runner")
    parser.add_argument("--job-dir", required=True, help="Job execution directory")
    args = parser.parse_args()

    job_dir = os.path.abspath(args.job_dir)
    start_parent_watchdog()
    try:
        run_training_pipeline(job_dir)
    except Exception as e:
        traceback.print_exc()
        update_status(
            job_dir,
            status="failed",
            stage=f"Failed: {str(e)}",
            progress=0.0,
            error=str(e),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

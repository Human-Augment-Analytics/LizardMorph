import os
import sys
import time
import json
import argparse
import traceback
from typing import Dict, Any, Optional

from backend.datasets.canonical import CanonicalDataset
from backend.training.yolo_trainer import YoloOBBTrainer
from backend.training.ml_morph_trainer import MLMorphTrainer


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
        except Exception:
            pass

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
        except Exception:
            pass
    return False


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
    jitter_padding = config.get("jitter_padding", 0.2)
    box_jitter = config.get("box_jitter", 0.05)
    dlib_opts = config.get("dlib_options", {})
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

    if check_cancelled(job_dir):
        return

    # STAGE 2: Training detector
    update_status(job_dir, "running", "Training detector", 0.40)
    yolo_trainer = YoloOBBTrainer(job_dir)
    data_yaml_path = yolo_trainer.prepare_dataset(canonical_ds)
    det_metrics = yolo_trainer.train(data_yaml_path, epochs=epochs, mock=mock)

    if check_cancelled(job_dir):
        return

    # STAGE 3: Training landmarks
    update_status(job_dir, "running", "Training landmarks", 0.70)
    ml_trainer = MLMorphTrainer(job_dir)
    xml_path = ml_trainer.prepare_landmark_dataset(
        canonical_ds, jitter_padding=jitter_padding, box_jitter=box_jitter
    )
    lm_metrics = ml_trainer.train(xml_path, custom_options=dlib_opts)

    if check_cancelled(job_dir):
        return

    # STAGE 4: Testing pipeline
    update_status(job_dir, "running", "Testing pipeline", 0.90)
    # Validate artifacts exist
    det_weights = det_metrics.get("detector_weights")
    lm_model = lm_metrics.get("landmark_model")
    if det_weights and not os.path.exists(det_weights):
        raise FileNotFoundError(f"Detector weights file missing: {det_weights}")
    if lm_model and not os.path.exists(lm_model):
        raise FileNotFoundError(f"Landmark model file missing: {lm_model}")

    if check_cancelled(job_dir):
        return

    # STAGE 5: Ready - Generate final manifest and bundle artifacts
    all_metrics = {**det_metrics, **lm_metrics}
    
    # Generate manifest
    rel_det_weights = os.path.relpath(det_weights, job_dir) if det_weights else ""
    rel_lm_model = os.path.relpath(lm_model, job_dir) if lm_model else ""

    manifest_data = {
        "manifest_version": "1.0",
        "project_id": project_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector": {
            "type": "yolo_obb",
            "weights": rel_det_weights,
        },
        "predictors": [
            {
                "class_name": "object",
                "file": rel_lm_model,
            }
        ],
        "metrics": all_metrics,
    }

    manifest_path = os.path.join(job_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

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
    try:
        run_training_pipeline(job_dir)
    except Exception as e:
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        update_status(
            job_dir,
            status="failed",
            stage=f"Failed: {str(e)}",
            progress=0.0,
            error=err_msg,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

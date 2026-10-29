"""Persisted training status shared by the backend and its runner process."""

import json
import os
import time
from typing import Any, Dict, Optional


def write_status(job_dir: str, status_data: Dict[str, Any]) -> None:
    """Replace status.json only after the complete payload has been written."""
    status_path = os.path.join(job_dir, "status.json")
    tmp_path = f"{status_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    os.replace(tmp_path, status_path)


def update_status(
    job_dir: str,
    status: str,
    stage: str,
    progress: float,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    """Write runner progress, retaining metrics accumulated by earlier stages."""
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

    write_status(job_dir, status_data)


def check_cancelled(job_dir: str) -> bool:
    """Check whether status.json has been marked as cancelled."""
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

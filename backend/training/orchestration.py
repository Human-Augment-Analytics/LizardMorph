import os
import sys
import uuid
import json
import time
import subprocess
from typing import Dict, Any, Optional


class TrainingOrchestrator:
    """Orchestrates asynchronous background training jobs via subprocess execution."""

    def __init__(self, runs_dir: str):
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)
        self._processes: Dict[str, subprocess.Popen] = {}

    def submit_job(
        self, project_id: str, dataset_dict: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """
        Submits a new training job and spawns a background runner process.

        Args:
            project_id: Identifier of target project.
            dataset_dict: Serialized CanonicalDataset dictionary.
            config: Job configuration parameters.

        Returns:
            Unique job_id string.
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_dir = os.path.join(self.runs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        job_config = {
            "job_id": job_id,
            "project_id": project_id,
            "dataset": dataset_dict,
            "config": config,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        config_path = os.path.join(job_dir, "job_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(job_config, f, indent=2)

        initial_status = {
            "status": "running",
            "stage": "Checking data",
            "progress": 0.0,
            "metrics": {},
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        status_path = os.path.join(job_dir, "status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(initial_status, f, indent=2)

        # Set PYTHONPATH to include project root so `backend.training.runner` can be loaded
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        # Add workspace root (parent of backend) or backend directory
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_root = os.path.dirname(workspace_root)
        new_pythonpath = f"{parent_root}:{workspace_root}:{pythonpath}".strip(":")
        env["PYTHONPATH"] = new_pythonpath

        cmd = [
            sys.executable,
            "-m",
            "backend.training.runner",
            "--job-dir",
            job_dir,
        ]

        proc = subprocess.Popen(cmd, env=env)
        self._processes[job_id] = proc

        return job_id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieves current status of job_id.

        Args:
            job_id: Job identifier string.

        Returns:
            Dict containing status, stage, progress, and metrics.
        """
        job_dir = os.path.join(self.runs_dir, job_id)
        status_path = os.path.join(job_dir, "status.json")

        if not os.path.exists(status_path):
            return {
                "status": "failed",
                "stage": "Job not found",
                "progress": 0.0,
                "metrics": {},
            }

        status_data = None
        # Retry up to 3 times in case status.json is currently being written
        for _ in range(3):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                time.sleep(0.05)

        if status_data is None:
            status_data = {
                "status": "running",
                "stage": "Updating status",
                "progress": 0.0,
                "metrics": {},
            }

        # Check process handle status if status claims 'running'
        proc = self._processes.get(job_id)
        if proc is not None and status_data.get("status") == "running":
            ret_code = proc.poll()
            if ret_code is not None and ret_code != 0:
                status_data["status"] = "failed"
                status_data["stage"] = f"Process exited unexpectedly with code {ret_code}"
                # Save updated status
                try:
                    with open(status_path, "w", encoding="utf-8") as f:
                        json.dump(status_data, f, indent=2)
                except Exception:
                    pass

        return {
            "status": status_data.get("status", "unknown"),
            "stage": status_data.get("stage", ""),
            "progress": float(status_data.get("progress", 0.0)),
            "metrics": status_data.get("metrics", {}),
        }

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancels a running job by terminating its subprocess and updating status.json.

        Args:
            job_id: Job identifier string.

        Returns:
            True if job was cancelled or process found, False otherwise.
        """
        job_dir = os.path.join(self.runs_dir, job_id)
        status_path = os.path.join(job_dir, "status.json")

        proc = self._processes.get(job_id)
        cancelled = False

        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            cancelled = True

        if os.path.exists(status_path):
            try:
                status_data = {}
                with open(status_path, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                
                current_status = status_data.get("status")
                if current_status == "running" or cancelled:
                    status_data["status"] = "cancelled"
                    status_data["stage"] = "Cancelled by user"
                    status_data["updated_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    tmp_path = f"{status_path}.tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(status_data, f, indent=2)
                    os.replace(tmp_path, status_path)
                    cancelled = True
            except Exception:
                pass

        return cancelled

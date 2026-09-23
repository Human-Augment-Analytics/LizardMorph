import os
import sys
import uuid
import json
import time
import subprocess
import signal
import threading
from typing import Dict, Any, Optional


class TrainingBusyError(RuntimeError):
    pass


class TrainingOrchestrator:
    """Orchestrates asynchronous background training jobs via subprocess execution."""

    def __init__(self, runs_dir: str, db_path: Optional[str] = None):
        self.runs_dir = os.path.abspath(runs_dir)
        self.db_path = (
            os.path.abspath(db_path)
            if db_path
            else os.path.join(self.runs_dir, "lizardmorph.db")
        )
        os.makedirs(self.runs_dir, exist_ok=True)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._submission_lock = threading.Lock()
        self._recover_interrupted_jobs()

    def _job_dir(self, job_id: str) -> str:
        if not job_id or os.path.basename(job_id) != job_id or job_id in (".", ".."):
            raise ValueError("Invalid training job identifier.")
        return os.path.join(self.runs_dir, job_id)

    def _recover_interrupted_jobs(self):
        for entry in os.listdir(self.runs_dir):
            if not entry.startswith("job_"):
                continue
            status_path = os.path.join(self.runs_dir, entry, "status.json")
            if not os.path.exists(status_path):
                continue
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    status = json.load(f)
                if status.get("status") != "running":
                    continue
                status.update(
                    {
                        "status": "failed",
                        "stage": "Interrupted by backend restart",
                        "error": "The training process was interrupted when the backend stopped.",
                        "updated_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    }
                )
                tmp_path = f"{status_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2)
                os.replace(tmp_path, status_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                failed_status = {
                    "status": "failed",
                    "stage": "Unreadable job status",
                    "progress": 0.0,
                    "metrics": {},
                    "error": f"The persisted job status is invalid: {error}",
                    "updated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                try:
                    tmp_path = f"{status_path}.tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(failed_status, f, indent=2)
                    os.replace(tmp_path, status_path)
                except OSError:
                    continue

    @staticmethod
    def _stage_source_files(
        job_dir: str, dataset_dict: Dict[str, Any], source_files: Dict[str, bytes]
    ) -> Dict[str, Any]:
        if not source_files:
            return dataset_dict

        sources_dir = os.path.join(job_dir, "source_images")
        os.makedirs(sources_dir, exist_ok=True)
        staged_paths: Dict[str, str] = {}
        basename_paths: Dict[str, list] = {}
        for relative_name, content in source_files.items():
            normalized = os.path.normpath(relative_name.replace("\\", "/"))
            if os.path.isabs(normalized) or normalized == ".." or normalized.startswith(f"..{os.sep}"):
                raise ValueError(f"Unsafe source image path: {relative_name}")
            destination = os.path.abspath(os.path.join(sources_dir, normalized))
            if os.path.commonpath([os.path.abspath(sources_dir), destination]) != os.path.abspath(sources_dir):
                raise ValueError(f"Unsafe source image path: {relative_name}")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as f:
                f.write(content)
            staged_paths[normalized.casefold()] = destination
            basename_paths.setdefault(os.path.basename(normalized).casefold(), []).append(destination)

        staged_dataset = json.loads(json.dumps(dataset_dict))
        for image in staged_dataset.get("images", []):
            reference = os.path.normpath(str(image.get("file_path", "")).replace("\\", "/"))
            destination = staged_paths.get(reference.casefold())
            if destination is None:
                matches = basename_paths.get(os.path.basename(reference).casefold(), [])
                if len(matches) == 1:
                    destination = matches[0]
            if destination is None:
                raise ValueError(
                    f"Image '{image.get('file_path')}' referenced by the annotations was not provided."
                )
            image["file_path"] = destination
        return staged_dataset

    def submit_job(
        self,
        project_id: str,
        dataset_dict: Dict[str, Any],
        config: Dict[str, Any],
        source_files: Optional[Dict[str, bytes]] = None,
    ) -> str:
        if not self._submission_lock.acquire(blocking=False):
            raise TrainingBusyError("Another training job is being submitted.")
        try:
            return self._submit_job(
                project_id=project_id,
                dataset_dict=dataset_dict,
                config=config,
                source_files=source_files,
            )
        finally:
            self._submission_lock.release()

    def _submit_job(
        self,
        project_id: str,
        dataset_dict: Dict[str, Any],
        config: Dict[str, Any],
        source_files: Optional[Dict[str, bytes]] = None,
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
        finished_jobs = [
            existing_job_id
            for existing_job_id, process in self._processes.items()
            if process.poll() is not None
        ]
        for existing_job_id in finished_jobs:
            self._processes.pop(existing_job_id, None)
        if any(process.poll() is None for process in self._processes.values()):
            raise TrainingBusyError(
                "Another training job is already running. Cancel it or wait for it to finish."
            )

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_dir = self._job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        dataset_dict = self._stage_source_files(
            job_dir, dataset_dict, source_files or {}
        )

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
        env["RUNS_DIR"] = self.runs_dir
        env["DB_PATH"] = self.db_path
        env["AUTOMORPH_PARENT_PID"] = str(os.getpid())

        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--run-training-job", job_dir]
        else:
            cmd = [
                sys.executable,
                "-m",
                "backend.training.runner",
                "--job-dir",
                job_dir,
            ]

        log_path = os.path.join(job_dir, "training.log")
        log_file = open(log_path, "ab")
        try:
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as error:
                failed_status = {
                    "status": "failed",
                    "stage": "Unable to start training process",
                    "progress": 0.0,
                    "metrics": {},
                    "error": str(error),
                    "updated_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                tmp_path = f"{status_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(failed_status, f, indent=2)
                os.replace(tmp_path, status_path)
                raise
        finally:
            log_file.close()
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
        job_dir = self._job_dir(job_id)
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
                "status": "failed",
                "stage": "Unreadable job status",
                "progress": 0.0,
                "metrics": {},
                "error": "The training job status file could not be read.",
            }
            tmp_path = f"{status_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2)
            os.replace(tmp_path, status_path)

        # Check process handle status if status claims 'running'
        proc = self._processes.get(job_id)
        if proc is not None:
            ret_code = proc.poll()
            if ret_code is not None and status_data.get("status") == "running":
                status_data["status"] = "failed"
                status_data["stage"] = f"Process exited unexpectedly with code {ret_code}"
                status_data["error"] = status_data["stage"]
                tmp_path = f"{status_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, indent=2)
                os.replace(tmp_path, status_path)
            if ret_code is not None:
                self._processes.pop(job_id, None)

        return {
            "status": status_data.get("status", "unknown"),
            "stage": status_data.get("stage", ""),
            "progress": float(status_data.get("progress", 0.0)),
            "metrics": status_data.get("metrics", {}),
            "error": status_data.get("error"),
        }

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancels a running job by terminating its subprocess and updating status.json.

        Args:
            job_id: Job identifier string.

        Returns:
            True if job was cancelled or process found, False otherwise.
        """
        job_dir = self._job_dir(job_id)
        status_path = os.path.join(job_dir, "status.json")

        proc = self._processes.get(job_id)
        cancelled = False

        if proc is not None:
            if proc.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except ProcessLookupError:
                        pass
            cancelled = True
            self._processes.pop(job_id, None)

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
            except (OSError, ValueError, json.JSONDecodeError):
                return cancelled

        return cancelled

"""Filesystem-only training status regressions; no trainers or database are run."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from backend.training import orchestration
from backend.training import runner, status as training_status
from backend.training.orchestration import TrainingOrchestrator
from backend.training.runner import check_cancelled, update_status


def write_status(job_dir, payload):
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def read_status(job_dir):
    return json.loads((job_dir / "status.json").read_text(encoding="utf-8"))


def assert_timestamp(value):
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def test_runner_keeps_status_helpers_available_at_existing_import_path():
    assert runner.update_status is training_status.update_status
    assert runner.check_cancelled is training_status.check_cancelled


def test_status_updates_merge_metrics_and_replace_previous_error(tmp_path):
    update_status(
        str(tmp_path), "running", "Checking data", 0,
        metrics={"images": 4, "loss": 2.0}, error="previous error",
    )
    update_status(
        str(tmp_path), "completed", "Ready", 1,
        metrics={"loss": 0.5},
    )

    status = read_status(tmp_path)
    assert status == {
        "status": "completed",
        "stage": "Ready",
        "progress": 1.0,
        "metrics": {"images": 4, "loss": 0.5},
        "updated_at": status["updated_at"],
    }
    assert isinstance(status["progress"], float)
    assert_timestamp(status["updated_at"])


def test_status_update_repairs_truncated_json(tmp_path):
    (tmp_path / "status.json").write_text('{"status":', encoding="utf-8")

    update_status(str(tmp_path), "failed", "Unable to train", 0, error=ValueError("bad data"))

    status = read_status(tmp_path)
    assert status["status"] == "failed"
    assert status["metrics"] == {}
    assert status["error"] == "bad data"


def test_failed_status_write_keeps_previous_status_readable(tmp_path, monkeypatch):
    path = write_status(tmp_path, {"status": "running", "metrics": {"images": 4}})
    previous = path.read_bytes()

    def interrupted_dump(payload, stream, **kwargs):
        stream.write('{"status":')
        raise OSError("write interrupted")

    monkeypatch.setattr(json, "dump", interrupted_dump)

    with pytest.raises(OSError, match="write interrupted"):
        update_status(str(tmp_path), "completed", "Ready", 1)

    assert path.read_bytes() == previous


@pytest.mark.parametrize("status, expected", [("running", False), ("cancelled", True), ("completed", False)])
def test_check_cancelled_reads_persisted_status(tmp_path, status, expected):
    write_status(tmp_path, {"status": status})

    assert check_cancelled(str(tmp_path)) is expected


def test_check_cancelled_returns_false_without_status(tmp_path):
    assert check_cancelled(str(tmp_path)) is False


def test_check_cancelled_reports_unreadable_status(tmp_path):
    (tmp_path / "status.json").write_text("truncated", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Training job status is unreadable"):
        check_cancelled(str(tmp_path))


def test_recovery_fails_interrupted_jobs_and_preserves_finished_jobs(tmp_path):
    running = tmp_path / "job_running"
    write_status(running, {"status": "running", "progress": 0.4, "metrics": {"images": 4}})
    completed = write_status(tmp_path / "job_done", {"status": "completed", "metrics": {"loss": 0.5}})
    previous_completed = completed.read_bytes()

    TrainingOrchestrator(str(tmp_path))

    status = read_status(running)
    assert status["status"] == "failed"
    assert status["stage"] == "Interrupted by backend restart"
    assert status["error"] == "The training process was interrupted when the backend stopped."
    assert status["progress"] == 0.4
    assert status["metrics"] == {"images": 4}
    assert_timestamp(status["updated_at"])
    assert completed.read_bytes() == previous_completed


def test_recovery_repairs_unreadable_persisted_status(tmp_path):
    job_dir = tmp_path / "job_broken"
    path = write_status(job_dir, {})
    path.write_text("truncated", encoding="utf-8")

    TrainingOrchestrator(str(tmp_path))

    status = read_status(job_dir)
    assert status["status"] == "failed"
    assert status["stage"] == "Unreadable job status"
    assert status["progress"] == 0.0
    assert status["metrics"] == {}
    assert status["error"].startswith("The persisted job status is invalid:")
    assert_timestamp(status["updated_at"])


def test_get_status_repairs_unreadable_file_after_retries(tmp_path, monkeypatch):
    manager = TrainingOrchestrator(str(tmp_path))
    job_dir = tmp_path / "job_broken"
    path = write_status(job_dir, {})
    path.write_text("truncated", encoding="utf-8")
    sleeps = []
    monkeypatch.setattr(orchestration.time, "sleep", sleeps.append)

    status = manager.get_status("job_broken")

    assert sleeps == [0.05, 0.05, 0.05]
    assert status == {
        "status": "failed",
        "stage": "Unreadable job status",
        "progress": 0.0,
        "metrics": {},
        "error": "The training job status file could not be read.",
    }
    assert read_status(job_dir) == status


@pytest.mark.parametrize("previous_timestamp", [None, "previous timestamp"])
def test_get_status_records_unexpected_process_exit(tmp_path, previous_timestamp):
    manager = TrainingOrchestrator(str(tmp_path))
    job_dir = tmp_path / "job_exited"
    previous_status = {"status": "running", "progress": 0.4, "metrics": {"images": 4}}
    if previous_timestamp is not None:
        previous_status["updated_at"] = previous_timestamp
    write_status(job_dir, previous_status)
    manager._processes["job_exited"] = SimpleNamespace(poll=lambda: 17)

    status = manager.get_status("job_exited")

    assert status == {
        "status": "failed",
        "stage": "Process exited unexpectedly with code 17",
        "progress": 0.4,
        "metrics": {"images": 4},
        "error": "Process exited unexpectedly with code 17",
    }
    expected_persisted = dict(status)
    if previous_timestamp is not None:
        expected_persisted["updated_at"] = previous_timestamp
    assert read_status(job_dir) == expected_persisted
    assert "job_exited" not in manager._processes


@pytest.mark.parametrize("previous_status, expected", [("running", True), ("completed", False)])
def test_cancel_job_preserves_metrics_and_finished_status(tmp_path, previous_status, expected):
    manager = TrainingOrchestrator(str(tmp_path))
    job_dir = tmp_path / "job_cancel"
    path = write_status(job_dir, {
        "status": previous_status, "stage": "Existing stage", "progress": 0.4,
        "metrics": {"images": 4}, "updated_at": "previous timestamp",
    })
    previous = path.read_bytes()

    assert manager.cancel_job("job_cancel") is expected

    status = read_status(job_dir)
    if expected:
        assert status["status"] == "cancelled"
        assert status["stage"] == "Cancelled by user"
        assert status["progress"] == 0.4
        assert status["metrics"] == {"images": 4}
        assert_timestamp(status["updated_at"])
    else:
        assert path.read_bytes() == previous


def test_submission_records_process_launch_failure_without_starting_training(tmp_path, monkeypatch):
    manager = TrainingOrchestrator(str(tmp_path))
    initial_statuses = []

    def fail_launch(*args, **kwargs):
        job_dir, = tmp_path.glob("job_*")
        initial_statuses.append(read_status(job_dir))
        raise OSError("launch failed")

    monkeypatch.setattr(orchestration.subprocess, "Popen", fail_launch)

    with pytest.raises(OSError, match="launch failed"):
        manager.submit_job("project_1", {"images": []}, {"mock": True})

    initial_status, = initial_statuses
    assert initial_status == {
        "status": "running",
        "stage": "Checking data",
        "progress": 0.0,
        "metrics": {},
        "updated_at": initial_status["updated_at"],
    }
    assert_timestamp(initial_status["updated_at"])
    job_dir, = tmp_path.glob("job_*")
    status = read_status(job_dir)
    assert status["status"] == "failed"
    assert status["stage"] == "Unable to start training process"
    assert status["progress"] == 0.0
    assert status["metrics"] == {}
    assert status["error"] == "launch failed"
    assert_timestamp(status["updated_at"])
    assert manager._processes == {}

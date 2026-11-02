"""Download a complete pipeline by exact named MLflow version."""
import hashlib
from pathlib import Path
import tempfile

import mlflow
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from routers.models import _get_model_version_or_404

router = APIRouter()


@router.get("/bundles/{name}/versions/{version}/archive")
def download_bundle(name: str, version: int):
    if version < 1:
        raise HTTPException(status_code=400, detail="Version must be positive")
    model = _get_model_version_or_404(mlflow.MlflowClient(), name, str(version))
    if model.tags.get("bundle_format") != "lizard-pipeline-v1":
        raise HTTPException(status_code=409, detail="This version is not a complete pipeline bundle")
    expected_source = f"runs:/{model.run_id}/bundle.zip"
    if model.source != expected_source:
        raise HTTPException(status_code=409, detail="Bundle source must be the version's run artifact")
    directory = tempfile.TemporaryDirectory(prefix="lizard-bundle-")
    try:
        path = Path(mlflow.artifacts.download_artifacts(artifact_uri=expected_source, dst_path=directory.name))
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha256 != model.tags.get("bundle_sha256"):
            raise ValueError("Registered bundle checksum mismatch")
        return FileResponse(path, media_type="application/zip", filename="bundle.zip",
                            headers={"X-Bundle-SHA256": sha256},
                            background=BackgroundTask(directory.cleanup))
    except Exception as exc:
        directory.cleanup()
        raise HTTPException(status_code=502, detail=f"Bundle unavailable: {exc}") from exc

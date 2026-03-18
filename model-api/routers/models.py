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

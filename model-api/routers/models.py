import mlflow
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

import config
from models.schemas import (
    ModelsResponse, ModelVersionItem,
    LatestModelResponse, AssetUrls, DatasetInfo, TrainingInfo, MetricsInfo,
    PromoteModelRequest, PromoteModelResponse,
)

router = APIRouter()

DEFAULT_PRODUCTION_ALIAS = "champion"


def _build_model_response(release_tag: str) -> LatestModelResponse:
    """Fetch a GitHub Release by tag, read its metadata.json, and return the
    full LatestModelResponse (version info + download URLs)."""
    gh_headers = {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    release_url = (
        f"https://api.github.com/repos/{config.GITHUB_REPO}"
        f"/releases/tags/{release_tag}"
    )
    resp = httpx.get(release_url, headers=gh_headers)
    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"GitHub Release not found for tag: {release_tag}",
        )
    resp.raise_for_status()
    release = resp.json()

    assets_by_name = {a["name"]: a for a in release.get("assets", [])}

    metadata_asset = assets_by_name.get("metadata.json")
    if not metadata_asset:
        raise HTTPException(status_code=404, detail="metadata.json not found in release")

    meta_resp = httpx.get(
        metadata_asset["url"],
        headers={**gh_headers, "Accept": "application/octet-stream"},
        follow_redirects=True,
    )
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    asset_filenames = meta.get("assets", {})

    def _asset_url(key: str) -> str:
        filename = asset_filenames.get(key, "")
        asset = assets_by_name.get(filename)
        return asset["browser_download_url"] if asset else ""

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
            fp16=_asset_url("fp16"),
            fp32=_asset_url("fp32"),
            pt=_asset_url("pt"),
        ),
    )


@router.get("/models", response_model=ModelsResponse)
def list_models():
    client = mlflow.MlflowClient()
    versions = client.search_model_versions()
    items = [
        ModelVersionItem(
            name=mv.name,
            version=str(mv.version),
            stage=mv.current_stage,
            aliases=list(getattr(mv, "aliases", []) or []),
            run_id=getattr(mv, "run_id", None),
            source=getattr(mv, "source", None),
            status=getattr(mv, "status", None),
            tags=dict(getattr(mv, "tags", {}) or {}),
        )
        for mv in versions
    ]
    items.sort(key=lambda item: (item.name.lower(), -_version_number(item.version)))
    return ModelsResponse(models=items)


@router.get("/model/latest", response_model=LatestModelResponse)
def get_latest_model(model_name: str | None = None, alias: str = DEFAULT_PRODUCTION_ALIAS):
    client = mlflow.MlflowClient()
    production_version = _find_production_version(client, model_name, alias)

    if not production_version:
        raise HTTPException(
            status_code=404,
            detail=f"No model assigned alias '{alias}' or Production stage",
        )

    run = client.get_run(production_version.run_id)
    release_tag = (
        run.data.tags.get("github_release_tag")
        or _release_tag_from_source(getattr(production_version, "source", ""))
    )
    if not release_tag:
        raise HTTPException(
            status_code=404,
            detail="github_release_tag not set on the promoted model's run",
        )

    return _build_model_response(release_tag)


@router.get("/model/by-tag/{tag:path}", response_model=LatestModelResponse)
def get_model_by_tag(tag: str):
    """Fetch model info + download URLs for any release tag (no Production gate)."""
    return _build_model_response(tag)


@router.post(
    "/models/{model_name}/versions/{version}/promote",
    response_model=PromoteModelResponse,
)
def promote_model_version(
    model_name: str,
    version: str,
    request: PromoteModelRequest | None = None,
):
    """Promote a model version for app consumption.

    MLflow aliases are the forward-compatible deployment pointer. We also set the
    legacy Production stage by default so existing MLflow UI workflows keep working.
    """
    request = request or PromoteModelRequest()
    alias = request.alias.strip()
    if not alias:
        raise HTTPException(status_code=400, detail="alias must not be empty")

    client = mlflow.MlflowClient()
    model_version = _get_model_version_or_404(client, model_name, version)

    client.set_registered_model_alias(model_name, alias, version)
    client.set_model_version_tag(model_name, version, "deployment_alias", alias)
    client.set_model_version_tag(
        model_name,
        version,
        "promoted_at",
        datetime.now(timezone.utc).isoformat(),
    )

    if request.set_stage:
        model_version = client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=request.stage,
            archive_existing_versions=request.archive_existing_versions,
        )
    else:
        model_version = client.get_model_version(model_name, version)

    aliases = list(getattr(model_version, "aliases", []) or [])
    if alias not in aliases:
        aliases.append(alias)

    return PromoteModelResponse(
        name=model_version.name,
        version=str(model_version.version),
        alias=alias,
        stage=model_version.current_stage,
        aliases=aliases,
        run_id=getattr(model_version, "run_id", None),
    )


def _find_production_version(client, model_name: str | None, alias: str):
    if not model_name:
        candidates = [
            version for version in client.search_model_versions()
            if alias in list(getattr(version, "aliases", []) or [])
            or version.current_stage == "Production"
        ]
        if candidates:
            candidates.sort(key=_deployment_sort_key, reverse=True)
            return candidates[0]
        return None

    registered_models = _registered_models(client, model_name)
    for registered_model in registered_models:
        try:
            version = client.get_model_version_by_alias(registered_model.name, alias)
            if version:
                return version
        except Exception:
            continue

    for registered_model in registered_models:
        versions = client.get_latest_versions(registered_model.name, stages=["Production"])
        if versions:
            return versions[0]
    return None


def _registered_models(client, model_name: str | None):
    if model_name:
        try:
            return [client.get_registered_model(model_name)]
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Registered model not found: {model_name}",
            ) from exc
    return sorted(client.search_registered_models(), key=lambda m: m.name.lower())


def _get_model_version_or_404(client, model_name: str, version: str):
    try:
        return client.get_model_version(model_name, version)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model version not found: {model_name} v{version}",
        ) from exc


def _release_tag_from_source(source: str | None) -> str | None:
    if not source:
        return None
    marker = "/releases/tag/"
    if marker in source:
        return source.split(marker, 1)[1].strip("/")
    return None


def _deployment_sort_key(model_version):
    tags = dict(getattr(model_version, "tags", {}) or {})
    timestamp = (
        getattr(model_version, "last_updated_timestamp", None)
        or getattr(model_version, "creation_timestamp", None)
        or 0
    )
    return (tags.get("promoted_at", ""), timestamp)


def _version_number(version: str) -> int:
    try:
        return int(version)
    except ValueError:
        return -1

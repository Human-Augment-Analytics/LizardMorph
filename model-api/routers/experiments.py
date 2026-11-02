import mlflow
from fastapi import APIRouter, Query
from mlflow.entities import ViewType
from models.schemas import (
    ExperimentsResponse, ExperimentItem,
    RunsResponse, RunItem,
)

router = APIRouter()


@router.get("/experiments", response_model=ExperimentsResponse)
def list_experiments():
    exps = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    unique_experiments = {}
    for exp in exps:
        unique_experiments.setdefault(exp.name, exp)

    return ExperimentsResponse(
        experiments=[
            ExperimentItem(
                experiment_id=e.experiment_id,
                name=e.name,
                lifecycle_stage=e.lifecycle_stage,
            )
            for e in unique_experiments.values()
        ]
    )


@router.get("/experiments/{experiment_id}/runs", response_model=RunsResponse)
def get_runs(
    experiment_id: str,
    dedupe_releases: bool = Query(
        True,
        description="Collapse repeated registration attempts for the same release tag/run name.",
    ),
):
    # Use MlflowClient.search_runs to get Run objects (not DataFrame)
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=100,
    )

    run_items = []
    seen = set()
    for run in runs:
        tags = _public_tags(run)
        dedupe_key = (
            tags.get("github_release_tag")
            or run.info.run_name
            or run.info.run_id
        )
        if dedupe_releases and dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        run_items.append(
            RunItem(
                run_id=run.info.run_id,
                run_name=run.info.run_name,
                status=run.info.status,
                metrics=run.data.metrics,
                params=run.data.params,
                tags=tags,
            )
        )

    return RunsResponse(
        experiment_id=experiment_id,
        runs=run_items,
    )


def _public_tags(run) -> dict[str, str]:
    tags = getattr(getattr(run, "data", None), "tags", {}) or {}
    if not isinstance(tags, dict):
        try:
            tags = dict(tags)
        except (TypeError, ValueError):
            tags = {}
    return {k: v for k, v in tags.items() if not k.startswith("mlflow.")}

import mlflow
from fastapi import APIRouter
from models.schemas import (
    ExperimentsResponse, ExperimentItem,
    RunsResponse, RunItem,
)

router = APIRouter()


@router.get("/experiments", response_model=ExperimentsResponse)
def list_experiments():
    exps = mlflow.search_experiments()
    return ExperimentsResponse(
        experiments=[
            ExperimentItem(
                experiment_id=e.experiment_id,
                name=e.name,
                lifecycle_stage=e.lifecycle_stage,
            )
            for e in exps
        ]
    )


@router.get("/experiments/{experiment_id}/runs", response_model=RunsResponse)
def get_runs(experiment_id: str):
    # Use MlflowClient.search_runs to get Run objects (not DataFrame)
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=100,
    )
    return RunsResponse(
        experiment_id=experiment_id,
        runs=[
            RunItem(
                run_id=r.info.run_id,
                run_name=r.info.run_name,
                status=r.info.status,
                metrics=r.data.metrics,
                params=r.data.params,
            )
            for r in runs
        ],
    )

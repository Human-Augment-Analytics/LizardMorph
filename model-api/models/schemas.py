from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    mlflow_reachable: bool


class ExperimentItem(BaseModel):
    experiment_id: str
    name: str
    lifecycle_stage: str


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentItem]


class RunItem(BaseModel):
    run_id: str
    run_name: Optional[str]
    status: str
    metrics: dict[str, float]
    params: dict[str, str]


class RunsResponse(BaseModel):
    experiment_id: str
    runs: list[RunItem]


class ModelVersionItem(BaseModel):
    name: str
    version: str
    stage: str


class ModelsResponse(BaseModel):
    models: list[ModelVersionItem]


# /model/latest — rich response from GitHub Release metadata.json
class AssetUrls(BaseModel):
    fp16: str
    fp32: str
    pt: str


class DatasetInfo(BaseModel):
    nc: int
    names: list[str]
    train_images: int
    val_images: int


class TrainingInfo(BaseModel):
    epochs: int
    batch: int
    imgsz: int
    patience: int


class MetricsInfo(BaseModel):
    best_epoch: int
    epochs_completed: int
    mAP50: float
    mAP50_95: float = Field(alias="mAP50-95", serialization_alias="mAP50-95")
    precision: float
    recall: float

    model_config = {"populate_by_name": True}


class LatestModelResponse(BaseModel):
    version: str
    trained: str
    author: str
    architecture: str
    task: str
    config: str
    dataset: DatasetInfo
    training: TrainingInfo
    metrics: MetricsInfo
    assets: AssetUrls

from models.schemas import (
    HealthResponse, ExperimentItem, ExperimentsResponse,
    RunItem, RunsResponse, ModelVersionItem, ModelsResponse,
    AssetUrls, DatasetInfo, TrainingInfo, MetricsInfo, LatestModelResponse,
)


def test_health_response():
    r = HealthResponse(status="ok", mlflow_reachable=True)
    assert r.status == "ok"
    assert r.mlflow_reachable is True


def test_latest_model_response():
    r = LatestModelResponse(
        version="v1.0.0-obb",
        trained="2026-03-13",
        author="Test User <test@example.com>",
        architecture="yolo11m-obb.pt",
        task="obb",
        config="H11_obb",
        dataset=DatasetInfo(nc=4, names=["finger", "toe", "ruler", "id"], train_images=679, val_images=170),
        training=TrainingInfo(epochs=300, batch=32, imgsz=1280, patience=50),
        metrics=MetricsInfo(best_epoch=175, epochs_completed=225, mAP50=0.972, mAP50_95=0.942, precision=0.966, recall=0.959),
        assets=AssetUrls(
            fp16="https://github.com/org/repo/releases/download/v1.0/best_fp16.onnx",
            fp32="https://github.com/org/repo/releases/download/v1.0/best_fp32.onnx",
            pt="https://github.com/org/repo/releases/download/v1.0/best.pt",
        ),
    )
    assert r.version == "v1.0.0-obb"
    assert r.metrics.mAP50 == 0.972
    assert r.assets.fp16.endswith("best_fp16.onnx")


def test_runs_response():
    r = RunsResponse(
        experiment_id="1",
        runs=[RunItem(
            run_id="abc",
            run_name="test_run",
            status="FINISHED",
            metrics={"mAP50": 0.91},
            params={"epochs": "100"},
        )],
    )
    assert len(r.runs) == 1
    assert r.runs[0].status == "FINISHED"

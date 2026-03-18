import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import config  # triggers startup validation


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    yield


app = FastAPI(title="LizardMorph MLflow API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

from routers import experiments, models  # noqa: E402
app.include_router(experiments.router)
app.include_router(models.router)


@app.get("/health")
def health():
    try:
        mlflow.search_experiments()
        mlflow_reachable = True
    except Exception:
        mlflow_reachable = False
    return {
        "status": "ok" if mlflow_reachable else "degraded",
        "mlflow_reachable": mlflow_reachable,
    }

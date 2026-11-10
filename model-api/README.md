# Model registry API

FastAPI bridge to the configured MLflow tracking server. This service retains the
legacy experiment/model/release endpoints and adds exact-version full pipeline
archive downloads under `/bundles/{name}/versions/{version}/archive`.

Set `MLFLOW_TRACKING_URI` and `GITHUB_REPO`; public GitHub reads do not require a
token. The pipeline workflow uses MLflow artifacts rather than detector-only
GitHub releases. Register with `bundle_format=lizard-pipeline-v1` and a SHA256 tag.
Downloads verify the stored digest and never resolve an unscoped global champion.

The legacy `POST /models/{name}/versions/{version}/promote` route requires a JSON
body and returns 403 unless `MODEL_API_ENABLE_PROMOTION=true`. CORS is disabled
unless `MODEL_API_CORS_ORIGINS` lists comma-separated allowed origins.

See [the two-model pilot runbook](../deploy/mlflow-pilot/README.md) for operation,
registration, validation, switching and limitations. Run tests inside Docker on
the configured integration test host with pytest and this directory as PYTHONPATH.

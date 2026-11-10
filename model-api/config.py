import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"
GITHUB_REPO = os.getenv("GITHUB_REPO") or None
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or None

if not GITHUB_REPO:
    raise ValueError("GITHUB_REPO environment variable is required (e.g. org/repo)")
PROMOTION_ENABLED = (os.getenv("MODEL_API_ENABLE_PROMOTION") or "").strip().lower() in {"1", "true", "yes"}
CORS_ORIGINS = [
    origin.strip()
    for origin in (os.getenv("MODEL_API_CORS_ORIGINS") or "").split(",")
    if origin.strip()
]

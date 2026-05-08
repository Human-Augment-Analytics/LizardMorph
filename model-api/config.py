import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"
GITHUB_REPO = os.getenv("GITHUB_REPO") or None
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or None

if not GITHUB_REPO:
    raise ValueError("GITHUB_REPO environment variable is required (e.g. org/repo)")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

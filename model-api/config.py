import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_REPO:
    raise ValueError("GITHUB_REPO environment variable is required (e.g. org/repo)")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

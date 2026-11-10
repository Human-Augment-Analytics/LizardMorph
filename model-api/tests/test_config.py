import importlib
import pytest


def test_missing_github_repo_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    import config
    with pytest.raises(ValueError, match="GITHUB_REPO"):
        importlib.reload(config)


def test_public_repo_without_github_token(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    import config
    importlib.reload(config)
    assert config.GITHUB_TOKEN is None


def test_valid_config(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    import config
    importlib.reload(config)
    assert config.GITHUB_REPO == "org/repo"
    assert config.GITHUB_TOKEN == "ghp_test"
    assert config.MLFLOW_TRACKING_URI == "http://mlflow:5000"


def test_promotion_and_cors_are_off_unless_configured(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.delenv("MODEL_API_ENABLE_PROMOTION", raising=False)
    monkeypatch.delenv("MODEL_API_CORS_ORIGINS", raising=False)
    import config
    importlib.reload(config)
    assert config.PROMOTION_ENABLED is False
    assert config.CORS_ORIGINS == []
    monkeypatch.setenv("MODEL_API_ENABLE_PROMOTION", "true")
    monkeypatch.setenv("MODEL_API_CORS_ORIGINS", "http://localhost:5173, http://127.0.0.1:5173")
    try:
        importlib.reload(config)
        assert config.PROMOTION_ENABLED is True
        assert config.CORS_ORIGINS == ["http://localhost:5173", "http://127.0.0.1:5173"]
    finally:
        monkeypatch.delenv("MODEL_API_ENABLE_PROMOTION")
        monkeypatch.delenv("MODEL_API_CORS_ORIGINS")
        importlib.reload(config)

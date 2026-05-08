import importlib
import pytest


def test_missing_github_repo_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    import config
    with pytest.raises(ValueError, match="GITHUB_REPO"):
        importlib.reload(config)


def test_missing_github_token_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    import config
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        importlib.reload(config)


def test_valid_config(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    import config
    importlib.reload(config)
    assert config.GITHUB_REPO == "org/repo"
    assert config.GITHUB_TOKEN == "ghp_test"
    assert config.MLFLOW_TRACKING_URI == "http://mlflow:5000"

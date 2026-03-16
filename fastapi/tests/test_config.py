import importlib
import pytest


def test_missing_github_repo_raises(monkeypatch):
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    import config
    with pytest.raises(ValueError, match="GITHUB_REPO"):
        importlib.reload(config)


def test_missing_github_token_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    import config
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        importlib.reload(config)


def test_valid_config(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    import config
    importlib.reload(config)
    assert config.GITHUB_REPO == "org/repo"
    assert config.GITHUB_TOKEN == "ghp_test"
    assert config.MLFLOW_TRACKING_URI == "http://mlflow:5000"

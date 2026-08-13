import os
import importlib
import pytest


def test_env_hosted_and_repo_defaults(monkeypatch):
    # Test 1: AUTOMORPH_HOSTED=true
    monkeypatch.setenv("AUTOMORPH_HOSTED", "true")
    monkeypatch.delenv("LIZARDMORPH_HOSTED", raising=False)
    monkeypatch.delenv("REPO_NAME", raising=False)
    
    import app
    importlib.reload(app)
    assert app.IS_HOSTED is True
    assert app.REPO_NAME == "AutoMorph"

    import utils
    importlib.reload(utils)
    # Check logic matching utils.py is_hosted expression
    is_hosted = (os.getenv("AUTOMORPH_HOSTED") or os.getenv("LIZARDMORPH_HOSTED", "false")).lower() in ("true", "1", "yes")
    assert is_hosted is True

    # Test 2: Fallback to LIZARDMORPH_HOSTED=true when AUTOMORPH_HOSTED is unset
    monkeypatch.delenv("AUTOMORPH_HOSTED", raising=False)
    monkeypatch.setenv("LIZARDMORPH_HOSTED", "true")
    importlib.reload(app)
    assert app.IS_HOSTED is True
    is_hosted = (os.getenv("AUTOMORPH_HOSTED") or os.getenv("LIZARDMORPH_HOSTED", "false")).lower() in ("true", "1", "yes")
    assert is_hosted is True

    # Test 3: REPO_NAME custom override
    monkeypatch.setenv("REPO_NAME", "CustomRepo")
    importlib.reload(app)
    assert app.REPO_NAME == "CustomRepo"

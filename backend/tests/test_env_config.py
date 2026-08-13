import os
import importlib
import pytest


def test_env_hosted_and_repo_defaults(monkeypatch):
    import app
    import utils

    # Test 1: AUTOMORPH_HOSTED=true
    monkeypatch.setenv("AUTOMORPH_HOSTED", "true")
    monkeypatch.delenv("LIZARDMORPH_HOSTED", raising=False)
    monkeypatch.delenv("REPO_NAME", raising=False)
    
    importlib.reload(app)
    importlib.reload(utils)
    assert app.IS_HOSTED is True
    assert utils.is_hosted() is True
    assert app.REPO_NAME == "AutoMorph"

    # Test 2: Fallback to LIZARDMORPH_HOSTED=true when AUTOMORPH_HOSTED is unset
    monkeypatch.delenv("AUTOMORPH_HOSTED", raising=False)
    monkeypatch.setenv("LIZARDMORPH_HOSTED", "true")
    importlib.reload(app)
    importlib.reload(utils)
    assert app.IS_HOSTED is True
    assert utils.is_hosted() is True

    # Test 3: Unset both AUTOMORPH_HOSTED and LIZARDMORPH_HOSTED -> False
    monkeypatch.delenv("AUTOMORPH_HOSTED", raising=False)
    monkeypatch.delenv("LIZARDMORPH_HOSTED", raising=False)
    importlib.reload(app)
    importlib.reload(utils)
    assert app.IS_HOSTED is False
    assert utils.is_hosted() is False

    # Test 4: REPO_NAME custom override
    monkeypatch.setenv("REPO_NAME", "CustomRepo")
    importlib.reload(app)
    assert app.REPO_NAME == "CustomRepo"


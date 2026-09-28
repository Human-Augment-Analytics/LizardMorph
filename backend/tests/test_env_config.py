"""Import-time configuration is checked in fresh processes, never by reloading Flask."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("overrides,hosted,repo", [
    ({"AUTOMORPH_HOSTED": "true"}, True, "AutoMorph"),
    ({"LIZARDMORPH_HOSTED": "true"}, True, "AutoMorph"),
    ({}, False, "AutoMorph"),
    ({"REPO_NAME": "CustomRepo"}, False, "CustomRepo"),
])
def test_env_hosted_and_repo_defaults(tmp_path, overrides, hosted, repo):
    env = os.environ.copy()
    for name in ("AUTOMORPH_HOSTED", "LIZARDMORPH_HOSTED", "REPO_NAME"):
        env.pop(name, None)
    env.update(overrides)
    env.update({
        "AUTOMORPH_DATA_DIR": str(tmp_path),
        "DB_PATH": str(tmp_path / "test.db"),
        "RUNS_DIR": str(tmp_path / "runs"),
        "PREDICTOR_LIBRARY_DIR": str(tmp_path / "predictors"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    })
    result = subprocess.run([
        sys.executable, "-c",
        "import dotenv; dotenv.load_dotenv = lambda *a, **k: False; "
        "import app, utils, json; "
        "print(json.dumps([app.IS_HOSTED, utils.is_hosted(), app.REPO_NAME]))",
    ], env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == [hosted, hosted, repo]

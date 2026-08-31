import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_SCRIPT = REPO_ROOT / "scripts" / "build-tauri-sidecar.sh"
MODEL_MANIFEST = REPO_ROOT / "src-tauri" / "desktop-models.txt"

HOST_ARCH = {
    "arm64": "aarch64",
    "aarch64": "aarch64",
    "x86_64": "x86_64",
    "AMD64": "x86_64",
}.get(platform.machine())
HOST_TRIPLE = {
    "Darwin": f"{HOST_ARCH}-apple-darwin",
    "Linux": "x86_64-unknown-linux-gnu" if HOST_ARCH == "x86_64" else None,
    "Windows": "x86_64-pc-windows-msvc" if HOST_ARCH == "x86_64" else None,
}.get(platform.system())
FOREIGN_OS_TRIPLE = (
    "x86_64-unknown-linux-gnu"
    if platform.system() != "Linux"
    else "aarch64-apple-darwin"
)
FOREIGN_ARCH_TRIPLE = (
    f"{'x86_64' if HOST_ARCH == 'aarch64' else 'aarch64'}-apple-darwin"
    if platform.system() == "Darwin"
    else None
)
FOREIGN_MACHINE = "x86_64" if HOST_ARCH == "aarch64" else "arm64"

pytestmark = pytest.mark.skipif(
    HOST_TRIPLE is None,
    reason=f"no supported sidecar target for {platform.system()}/{platform.machine()}",
)

TARGET_TRIPLE = HOST_TRIPLE
OLD_MTIME = 1_600_000_000
SIDECAR_MTIME = 1_600_000_100
NEW_MTIME = 1_600_000_200


def manifest_entries():
    entries = []
    for raw_entry in MODEL_MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = raw_entry.split("#", 1)[0].split()
        if fields:
            requirement, relative_source, destination = fields
            entries.append((requirement, relative_source, destination))
    return entries


BUNDLED_MODELS = [relative_source for _, relative_source, _ in manifest_entries()]
REQUIRED_MODELS = [
    relative_source
    for requirement, relative_source, _ in manifest_entries()
    if requirement == "required"
]


def build_project(tmp_path, omitted_models=(), sidecar=True, python_machine=None):
    project_dir = tmp_path / "project"
    (project_dir / "scripts").mkdir(parents=True)
    (project_dir / "src-tauri").mkdir(parents=True)
    (project_dir / "backend").mkdir(parents=True)

    shutil.copy2(SIDECAR_SCRIPT, project_dir / "scripts" / SIDECAR_SCRIPT.name)
    shutil.copy2(MODEL_MANIFEST, project_dir / "src-tauri" / MODEL_MANIFEST.name)
    (project_dir / "src-tauri" / "python-backend.spec").write_text("spec\n")
    (project_dir / "backend" / "requirements.txt").write_text("flask\n")

    for relative_source in BUNDLED_MODELS:
        if relative_source in omitted_models:
            continue
        model_path = project_dir / relative_source
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"model")

    for existing in project_dir.rglob("*"):
        if existing.is_file():
            os.utime(existing, (OLD_MTIME, OLD_MTIME))

    if sidecar:
        sidecar_path = project_dir / "src-tauri" / "binaries" / f"python-backend-{TARGET_TRIPLE}"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "wb") as handle:
            handle.truncate(11_000_000)
        sidecar_path.chmod(0o755)
        os.utime(sidecar_path, (SIDECAR_MTIME, SIDECAR_MTIME))

    fake_python = project_dir / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env sh\n"
        "case \"$*\" in\n"
        f"  *platform.machine*) echo {python_machine or platform.machine()}; exit 0 ;;\n"
        "esac\n"
        "exit 1\n"
    )
    fake_python.chmod(0o755)

    return project_dir


def run_gate(project_dir, target_triple=None):
    environment = dict(os.environ)
    environment["TAURI_ENV_TARGET_TRIPLE"] = target_triple or TARGET_TRIPLE
    environment["AUTOMORPH_PYTHON"] = str(project_dir / "fake-python")
    environment.pop("AUTOMORPH_FORCE_SIDECAR_BUILD", None)
    return subprocess.run(
        ["bash", str(project_dir / "scripts" / SIDECAR_SCRIPT.name)],
        capture_output=True,
        text=True,
        env=environment,
    )


def test_packaging_gate_reuses_a_sidecar_newer_than_every_bundled_model(tmp_path):
    result = run_gate(build_project(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "Reusing current Tauri backend sidecar" in result.stdout
    assert "cannot cross-compile" not in result.stderr


@pytest.mark.parametrize("relative_source", REQUIRED_MODELS)
def test_packaging_gate_fails_when_a_required_model_is_missing(tmp_path, relative_source):
    result = run_gate(build_project(tmp_path, omitted_models=(relative_source,)))

    assert result.returncode != 0
    assert f"Required desktop model is missing: {relative_source}" in result.stderr
    assert "Reusing current Tauri backend sidecar" not in result.stdout


@pytest.mark.parametrize("relative_source", BUNDLED_MODELS)
def test_a_newer_bundled_model_invalidates_the_cached_sidecar(tmp_path, relative_source):
    project_dir = build_project(tmp_path)
    os.utime(project_dir / relative_source, (NEW_MTIME, NEW_MTIME))

    result = run_gate(project_dir)

    assert "Reusing current Tauri backend sidecar" not in result.stdout
    assert result.returncode != 0
    assert "missing backend dependencies" in result.stderr


def test_packaging_gate_refuses_to_cross_compile_for_another_operating_system(tmp_path):
    result = run_gate(build_project(tmp_path), target_triple=FOREIGN_OS_TRIPLE)

    assert result.returncode != 0
    assert "cannot cross-compile" in result.stderr
    assert "Reusing current Tauri backend sidecar" not in result.stdout


@pytest.mark.skipif(
    FOREIGN_ARCH_TRIPLE is None, reason="no second supported arch for this host OS"
)
def test_packaging_gate_refuses_to_cross_compile_for_another_architecture(tmp_path):
    result = run_gate(build_project(tmp_path), target_triple=FOREIGN_ARCH_TRIPLE)

    assert result.returncode != 0
    assert "cannot cross-compile" in result.stderr
    assert FOREIGN_ARCH_TRIPLE.split("-", 1)[0] in result.stderr
    assert "Reusing current Tauri backend sidecar" not in result.stdout


def test_packaging_gate_refuses_an_interpreter_of_another_architecture(tmp_path):
    project_dir = build_project(tmp_path, python_machine=FOREIGN_MACHINE)
    os.utime(project_dir / BUNDLED_MODELS[0], (NEW_MTIME, NEW_MTIME))

    result = run_gate(project_dir)

    assert result.returncode != 0
    assert "cannot cross-compile" in result.stderr
    assert "missing backend dependencies" not in result.stderr


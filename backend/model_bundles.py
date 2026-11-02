"""Validate and install immutable research pipeline bundles from model-api."""
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import unicodedata
import zipfile

import requests

try:
    from backend.domain.models import Manifest
except ImportError:
    from domain.models import Manifest

WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)}

MAX_BUNDLE_BYTES = 512 * 1024 * 1024


def _relative_path(value):
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    reserved = any(part.split(".", 1)[0].upper() in WINDOWS_RESERVED
                   or part.endswith((" ", ".")) for part in path.parts)
    if (not value or path.is_absolute() or windows.drive or windows.root or reserved
            or ":" in value or ".." in path.parts or "\\" in value or str(path) != value):
        raise ValueError(f"Invalid bundle path: {value!r}")
    return value


def validate_bundle(archive_path, destination):
    """Extract only declared, hash-verified files; return a validated manifest."""
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        portable_names = [unicodedata.normalize("NFC", name).casefold() for name in names]
        if len(names) > 64 or len(portable_names) != len(set(portable_names)):
            raise ValueError("Bundle has too many, duplicate or case-colliding members")
        if sum(entry.file_size for entry in members) > MAX_BUNDLE_BYTES:
            raise ValueError("Bundle is too large")
        for entry in members:
            _relative_path(entry.filename)
            if entry.is_dir() or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Bundle directories and symlinks are not supported")
        if "bundle.json" not in names or archive.getinfo("bundle.json").file_size > 1024 * 1024:
            raise ValueError("Missing or oversized bundle.json")
        bundle = json.loads(archive.read("bundle.json"))
        if bundle.get("format") != "lizard-pipeline-v1":
            raise ValueError("Unsupported bundle format")
        artifacts = bundle.get("artifacts", {})
        if not artifacts or set(names) != set(artifacts) | {"bundle.json"}:
            raise ValueError("Bundle members must match the artifact inventory")
        provenance = bundle.get("provenance", {})
        if not provenance.get("source_commit") or not provenance.get("evaluation_status"):
            raise ValueError("Source commit and evaluation status are required")
        pipeline = bundle["pipeline"]
        manifest = Manifest.from_dict(pipeline)
        if manifest.schema_version != 1 or not manifest.classes:
            raise ValueError("Unsupported or empty pipeline")
        refs = [manifest.detector.artifact] if manifest.detector.artifact else []
        refs += [c.predictor for c in manifest.classes if c.predictor]
        if not refs or any(ref not in artifacts for ref in refs):
            raise ValueError("Pipeline references undeclared artifacts")
        if manifest.detector.artifact and not manifest.detector.artifact.endswith(".onnx"):
            raise ValueError("Research bundle detectors must be ONNX")
        for config in manifest.classes:
            schema = manifest.landmark_schemas.get(config.landmark_schema or "")
            if config.predictor and (not schema or not schema.points or len(set(schema.points)) != len(schema.points)):
                raise ValueError("Predictors require an explicit unique landmark order")
            if config.predictor:
                suffix = ".onnx" if config.predictor_type == "onnx-coordinates" else ".dat"
                if not config.predictor.endswith(suffix):
                    raise ValueError("Predictor artifact does not match its runtime")
                if config.predictor_type == "onnx-coordinates" and (
                        config.name not in {"bot_finger", "bot_toe", "crop"}
                        or config.preprocessing != "affine"
                        or config.crop_padding != 0 or len(schema.points) != 9):
                    raise ValueError("ViT pilot requires nine landmarks on unpadded bottom digit crops")
        for name, info in artifacts.items():
            _relative_path(name)
            data = archive.read(name)
            if len(data) != info["size"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
            target = Path(destination) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (Path(destination) / "bundle.json").write_text(json.dumps(bundle, indent=2))
        return bundle


def import_registry_bundle(name, version, *, api_url, models_dir, repository):
    """Resolve an explicit registry version once, validate, then publish locally."""
    from urllib.parse import quote

    if not isinstance(name, str) or not name.strip() or not str(version).isdigit() or int(version) < 1:
        raise ValueError("A registered name and positive numeric version are required")
    root = Path(models_dir).resolve() / "registry"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=root) as work:
        archive_path = Path(work) / "bundle.zip"
        url = f"{api_url.rstrip('/')}/bundles/{quote(name, safe='')}/versions/{version}/archive"
        with requests.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            total = 0
            with archive_path.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_BUNDLE_BYTES:
                        raise ValueError("Bundle download is too large")
                    digest.update(chunk)
                    output.write(chunk)
            archive_sha = digest.hexdigest()
            if response.headers.get("X-Bundle-SHA256") != archive_sha:
                raise ValueError("Bundle download checksum mismatch")
        identity = hashlib.sha256(f"{name}@{version}:{archive_sha}".encode()).hexdigest()[:24]
        model_id = f"registry-{identity}"
        final = root / model_id
        extracted = Path(work) / "extracted"
        extracted.mkdir()
        bundle = validate_bundle(archive_path, extracted)
        # Exercise the actual model loaders before exposing a selectable model.
        try:
            from backend.inference.engine import GenericPipelineEngine
        except ImportError:
            from inference.engine import GenericPipelineEngine
        manifest = Manifest.from_dict(bundle["pipeline"])
        engine = GenericPipelineEngine(models_dir=str(extracted))
        if manifest.detector.artifact:
            detector = engine._get_detector(str(extracted / manifest.detector.artifact),
                                           [c.name for c in manifest.classes],
                                           legacy_toepad=manifest.detector.inference_protocol == "toepad-dual-pass")
            if manifest.detector.inference_protocol == "toepad-dual-pass":
                # Current main's dual-pass decoder is specifically H7 six-class.
                # H11/H12 four-class models require a different decoder contract.
                if detector.session.get_outputs()[0].shape[1] != 11:
                    raise ValueError("This dual-pass runtime requires an H7 six-class detector")
        for config in manifest.classes:
            if config.predictor:
                path = str(extracted / config.predictor)
                if config.predictor_type == "onnx-coordinates":
                    engine._get_coordinate_predictor(path, len(manifest.landmark_schemas[config.landmark_schema].points))
                else:
                    import dlib
                    import numpy as np
                    predictor = engine._get_shape_predictor(path)
                    shape = predictor(np.zeros((64, 64, 3), dtype=np.uint8), dlib.rectangle(0, 0, 63, 63))
                    if shape.num_parts != len(manifest.landmark_schemas[config.landmark_schema].points):
                        raise ValueError("Dlib predictor landmark count does not match the manifest")
        if final.exists():
            if (final / "bundle.json").read_bytes() != (extracted / "bundle.json").read_bytes():
                raise ValueError("Installed bundle identity conflict")
            for name, info in bundle["artifacts"].items():
                installed_path = final / name
                if (installed_path.resolve() != installed_path.absolute()
                        or not installed_path.is_file()
                        or installed_path.stat().st_size != info["size"]
                        or hashlib.sha256(installed_path.read_bytes()).hexdigest() != info["sha256"]):
                    raise ValueError("Installed bundle artifact checksum mismatch")
        else:
            os.rename(extracted, final)
        pipeline = copy.deepcopy(bundle["pipeline"])
        pipeline["id"] = model_id
        pipeline["name"] = f"{pipeline['name']} — {name} v{version}"
        pipeline["evaluation"] = {**pipeline.get("evaluation", {}), "registry": {
            "name": name, "version": str(version), "bundle_sha256": archive_sha,
            "provenance": bundle["provenance"],
        }}
        pipeline["detector"]["artifact"] = str(final / manifest.detector.artifact) if manifest.detector.artifact else ""
        for config in pipeline["classes"]:
            if config.get("predictor"):
                config["predictor"] = str(final / config["predictor"])
        installed = Manifest.from_dict(pipeline)
        repository.register_model_bundle("mlflow", installed)
        return {"id": model_id, "name": installed.name, "registry": installed.evaluation["registry"]}

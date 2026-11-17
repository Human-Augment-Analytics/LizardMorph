"""Fail-closed, standard-library contract for web model release assets.

This module never executes model files or extracts archives. Both the publisher
and the deployment consumer must validate the exact three release assets.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import unicodedata
import zipfile

MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
DIGITS = {"up_finger", "up_toe", "bot_finger", "bot_toe"}
ASSETS = {"bundle.zip", "web-release.json", "evaluation.json"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 512


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def load_json(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "Missing or symlinked JSON asset")
    _require(path.stat().st_size <= MAX_JSON_BYTES, "JSON asset is too large")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release_tag(tag):
    _require(isinstance(tag, str) and re.fullmatch(
        r"model/toepad-web/v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag
    ) is not None, "Expected model/toepad-web/vMAJOR.MINOR.PATCH release tag")
    return tag


def validate_release_metadata(metadata):
    _require(isinstance(metadata, dict), "Release metadata must be an object")
    _require(metadata.get("format") == "lizard-web-release-v1" and
             type(metadata.get("schema_version")) is int and metadata["schema_version"] == 1,
             "Unsupported release metadata format")
    _require(metadata.get("channel") == "web-production", "Release is not for web-production")
    validate_release_tag(metadata.get("release_tag"))
    bundle, registry = metadata.get("bundle", {}), metadata.get("registry", {})
    _require(isinstance(bundle, dict) and bundle.get("asset") == "bundle.zip" and
             _sha(bundle.get("sha256")) and type(bundle.get("size")) is int and
             0 < bundle["size"] <= MAX_BUNDLE_BYTES, "Invalid bundle asset identity")
    _require(isinstance(registry, dict) and _text(registry.get("name")) and
             isinstance(registry.get("version"), str) and
             re.fullmatch(r"[1-9][0-9]*", registry["version"]) is not None and
             isinstance(registry.get("run_id"), str) and
             re.fullmatch(r"[0-9a-f]{32}", registry["run_id"]) is not None,
             "Explicit registered name, numeric version and run ID are required")
    _require(metadata.get("compatibility") == {
        "target": "web", "runtime": "onnx-dlib", "contract": "toepad-full-v1"
    }, "Release is not compatible with the full web toepad contract")
    evaluation = metadata.get("evaluation", {})
    _require(isinstance(evaluation, dict) and evaluation.get("asset") == "evaluation.json" and
             evaluation.get("status") == "approved" and _sha(evaluation.get("sha256")),
             "An approved, hash-pinned evaluation asset is required")
    return metadata


def _portable_path(name):
    _require(isinstance(name, str) and bool(name), "Invalid archive path")
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        prefix + str(n) for prefix in ("COM", "LPT") for n in range(1, 10)
    }
    _require(not posix.is_absolute() and not windows.drive and not windows.root and
             ":" not in name and "\\" not in name and ".." not in posix.parts and
             not any(ord(character) < 32 for character in name) and
             str(posix) == name and not any(
                 part.endswith((" ", ".")) or part.split(".", 1)[0].upper() in reserved
                 for part in posix.parts), "Nonportable archive path")


def validate_web_bundle(bundle_path):
    path = Path(bundle_path)
    _require(path.is_file() and not path.is_symlink() and
             0 < path.stat().st_size <= MAX_BUNDLE_BYTES, "Invalid bundle file")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        normalized = [unicodedata.normalize("NFC", n).casefold() for n in names]
        _require(0 < len(members) <= 64 and len(set(normalized)) == len(names),
                 "Duplicate, case-colliding or excessive archive members")
        _require(sum(m.file_size for m in members) <= MAX_BUNDLE_BYTES,
                 "Uncompressed bundle is too large")
        for member in members:
            _portable_path(member.filename)
            mode = stat.S_IFMT(member.external_attr >> 16)
            _require(not member.is_dir() and mode in (0, stat.S_IFREG) and
                     not member.flag_bits & 1, "Unsupported archive member type")
        _require("bundle.json" in names and archive.getinfo("bundle.json").file_size <= MAX_JSON_BYTES,
                 "Missing or oversized bundle manifest")
        bundle = json.loads(archive.read("bundle.json"), parse_constant=_reject_constant)
        _require(isinstance(bundle, dict) and bundle.get("format") == "lizard-pipeline-v1",
                 "Unsupported pipeline bundle")
        provenance = bundle.get("provenance", {})
        _require(isinstance(provenance, dict) and isinstance(provenance.get("deployment_targets"), list)
                 and "web" in provenance["deployment_targets"] and
                 provenance.get("evaluation_status") == "heldout-evaluated" and
                 isinstance(provenance.get("source_commit"), str) and
                 re.fullmatch(r"[0-9a-f]{40}", provenance["source_commit"]) is not None,
                 "Bundle must explicitly target web and have held-out evaluation provenance")
        pipeline = bundle.get("pipeline", {})
        _require(isinstance(pipeline, dict) and type(pipeline.get("schema_version")) is int
                 and pipeline["schema_version"] == 1, "Unsupported pipeline schema")
        detector = pipeline.get("detector", {})
        _require(isinstance(detector, dict) and detector.get("geometry") == "obb" and
                 detector.get("inference_protocol") == "toepad-dual-pass" and
                 isinstance(detector.get("artifact"), str) and detector["artifact"].endswith(".onnx"),
                 "Web requires the H7 dual-pass ONNX detector contract")
        classes, schemas = pipeline.get("classes"), pipeline.get("landmark_schemas", {})
        _require(isinstance(classes, list) and len(classes) == 4 and
                 all(isinstance(c, dict) for c in classes) and
                 all(isinstance(c.get("name"), str) for c in classes) and
                 {c.get("name") for c in classes} == DIGITS and
                 all(type(c.get("id")) is int for c in classes) and
                 {c["id"] for c in classes} == {0, 1, 2, 3} and isinstance(schemas, dict),
                 "Web requires all four digit classes with distinct class IDs")
        references = {detector["artifact"]}
        for config in classes:
            _require(config.get("predictor_type") == "dlib" and
                     config.get("preprocessing") == "toepad-rectify-512" and
                     isinstance(config.get("predictor"), str) and config["predictor"].endswith(".dat"),
                     "Web permits only the dlib rectify-512 landmark runtime, not ViT")
            _require(isinstance(config.get("landmark_schema"), str), "Landmark schema name is required")
            schema = schemas.get(config["landmark_schema"], {})
            points = schema.get("points") if isinstance(schema, dict) else None
            _require(isinstance(points, list) and len(points) == 9 and
                     all(_text(p) for p in points) and len(set(points)) == 9,
                     "Each web digit requires nine explicit unique landmark names")
            references.add(config["predictor"])
        inventory = bundle.get("artifacts", {})
        _require(isinstance(inventory, dict) and set(names) == set(inventory) | {"bundle.json"}
                 and set(inventory) == references, "Bundle inventory must match exactly its model references")
        for name, expected in inventory.items():
            _require(isinstance(expected, dict) and _sha(expected.get("sha256")) and
                     type(expected.get("size")) is int and expected["size"] > 0,
                     "Invalid artifact identity")
            _require(archive.getinfo(name).file_size == expected["size"], "Artifact size mismatch")
            digest = hashlib.sha256()
            with archive.open(name) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            _require(digest.hexdigest() == expected["sha256"], "Artifact checksum mismatch")
    return bundle


def validate_evaluation(report, bundle, bundle_sha256):
    _require(isinstance(report, dict) and report.get("format") == "lizard-web-evaluation-v1" and
             report.get("approved") is True and _text(report.get("reviewer")) and
             _text(report.get("benchmark_id")) and _sha(report.get("dataset_sha256")) and
             report.get("bundle_sha256") == bundle_sha256 and
             report.get("source_commit") == bundle["provenance"]["source_commit"],
             "Evaluation approval must identify the exact bundle, benchmark, dataset and source")
    metrics, criteria = report.get("metrics"), report.get("criteria")
    _require(isinstance(metrics, dict) and metrics and all(
        _text(k) and _number(v) for k, v in metrics.items()), "Finite evaluation metrics are required")
    _require(isinstance(criteria, list) and criteria, "Explicit acceptance criteria are required")
    for criterion in criteria:
        _require(isinstance(criterion, dict) and isinstance(criterion.get("metric"), str) and
                 criterion["metric"] in metrics and
                 criterion.get("operator") in (">=", "<=") and _number(criterion.get("threshold")),
                 "Invalid evaluation criterion")
        actual, threshold = metrics[criterion["metric"]], criterion["threshold"]
        passed = actual >= threshold if criterion["operator"] == ">=" else actual <= threshold
        _require(passed, "Evaluation criterion failed: " + criterion["metric"])
    return report


def validate_release(directory, expected_tag=None):
    root = Path(directory)
    metadata = validate_release_metadata(load_json(root / "web-release.json"))
    if expected_tag is not None:
        _require(metadata["release_tag"] == validate_release_tag(expected_tag), "Release tag mismatch")
    path = root / "bundle.zip"
    _require(path.is_file() and path.stat().st_size == metadata["bundle"]["size"] and
             file_sha256(path) == metadata["bundle"]["sha256"], "Release bundle identity mismatch")
    bundle = validate_web_bundle(path)
    report_path = root / "evaluation.json"
    _require(file_sha256(report_path) == metadata["evaluation"]["sha256"], "Evaluation asset checksum mismatch")
    validate_evaluation(load_json(report_path), bundle, metadata["bundle"]["sha256"])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory")
    parser.add_argument("--expected-tag")
    parser.add_argument("--validate-tag")
    args = parser.parse_args()
    if args.validate_tag:
        print(validate_release_tag(args.validate_tag))
    elif args.directory:
        print(json.dumps(validate_release(args.directory, args.expected_tag), indent=2))
    else:
        parser.error("--directory or --validate-tag is required")


if __name__ == "__main__":
    main()

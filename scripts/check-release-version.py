"""Fail before packaging when the release tag and application metadata disagree."""
import argparse
import json
from pathlib import Path
import re
import tomllib


def check_release_version(root, tag):
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError(f"Expected a stable version tag like v1.2.1, got {tag!r}")
    expected = tag[1:]
    versions = {}
    for directory in (root, root / "frontend", root / "electron"):
        package = json.loads((directory / "package.json").read_text())
        lock = json.loads((directory / "package-lock.json").read_text())
        versions[str(directory / "package.json")] = package["version"]
        versions[str(directory / "package-lock.json")] = lock["version"]
        versions[str(directory / "package-lock.json") + " packages.root"] = lock["packages"][""]["version"]
        if package["name"] != lock["name"] or package["name"] != lock["packages"][""]["name"]:
            raise ValueError(f"Package name and lockfile disagree in {directory}")
    versions["tauri.conf.json"] = json.loads((root / "src-tauri/tauri.conf.json").read_text())["version"]
    versions["Cargo.toml"] = tomllib.loads((root / "src-tauri/Cargo.toml").read_text())["package"]["version"]
    cargo_lock = tomllib.loads((root / "src-tauri/Cargo.lock").read_text())
    versions["Cargo.lock"] = next(p["version"] for p in cargo_lock["package"] if p["name"] == "automorph")
    mismatches = {name: version for name, version in versions.items() if version != expected}
    if mismatches:
        raise ValueError(f"Release {tag} disagrees with package metadata: {mismatches}")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(f"Verified release v{check_release_version(args.root, args.tag)}")

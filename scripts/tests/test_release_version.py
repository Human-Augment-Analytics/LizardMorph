"""Behavioral checks of the executable release metadata gate."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check-release-version.py"
VERSION = "9.8.7"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for directory, name in (("", "automorph"), ("frontend", "frontend"), ("electron", "automorph-electron")):
            base = self.root / directory
            write_json(base / "package.json", {"name": name, "version": VERSION})
            write_json(base / "package-lock.json", {"name": name, "version": VERSION, "lockfileVersion": 3, "packages": {"": {"name": name, "version": VERSION}}})
        write_json(self.root / "src-tauri/tauri.conf.json", {"productName": "AutoMorph", "version": VERSION})
        (self.root / "src-tauri/Cargo.toml").write_text(f'[package]\nname = "automorph"\nversion = "{VERSION}"\n')
        (self.root / "src-tauri/Cargo.lock").write_text(
            'version = 4\n\n[[package]]\nname = "anyhow"\nversion = "1.0.0"\n\n'
            f'[[package]]\nname = "automorph"\nversion = "{VERSION}"\n'
        )

    def run_check(self, tag=f"v{VERSION}"):
        return subprocess.run([sys.executable, str(SCRIPT), tag, "--root", str(self.root)], capture_output=True, text=True)

    def test_accepts_consistent_metadata(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_mismatched_tag(self):
        self.assertNotEqual(self.run_check("v9.8.6").returncode, 0)

    def test_rejects_branch_as_version(self):
        self.assertNotEqual(self.run_check("main").returncode, 0)

    def test_rejects_stale_installer_or_lockfile_versions(self):
        for filename, root_key in (("electron/package.json", False), ("frontend/package-lock.json", True)):
            with self.subTest(filename=filename):
                target = self.root / filename
                original = target.read_text()
                data = json.loads(original)
                if root_key:
                    data["packages"][""]["version"] = "1.0.8"
                else:
                    data["version"] = "1.0.8"
                target.write_text(json.dumps(data))
                self.assertNotEqual(self.run_check().returncode, 0)
                target.write_text(original)


if __name__ == "__main__":
    unittest.main()

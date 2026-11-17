"""Behavioral checks of the executable release metadata gate."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check-release-version.py"


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("package.json", "package-lock.json", "frontend/package.json", "frontend/package-lock.json", "electron/package.json", "electron/package-lock.json", "src-tauri/tauri.conf.json", "src-tauri/Cargo.toml", "src-tauri/Cargo.lock"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def run_check(self, tag="v1.2.1"):
        return subprocess.run([sys.executable, str(SCRIPT), tag, "--root", str(self.root)], capture_output=True, text=True)

    def test_accepts_consistent_metadata(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_mismatched_tag(self):
        self.assertNotEqual(self.run_check("v1.0.8").returncode, 0)

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

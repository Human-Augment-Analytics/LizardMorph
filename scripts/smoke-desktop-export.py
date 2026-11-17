"""Exercise a real backend process: upload, prediction CSV, edited TPS, PNG and ZIP.

Accepts a frozen backend executable, or --source for development. Every writable
store is isolated in a temporary directory; no researcher data is touched.
"""
import argparse
import csv
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile

from PIL import Image


def exercise(base):
    def request(path, data=None, headers=None):
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.read(), response.headers

    def post(path, value, headers=None):
        body, _ = request(path, json.dumps(value).encode(), {"Content-Type": "application/json", **(headers or {})})
        return json.loads(body)

    session = post("/session/start", {})["session_id"]
    headers = {"X-Session-ID": session}
    names = ("LIZ.001.jpg", "LIZ.002.jpg")
    for name in names:
        image = BytesIO()
        Image.new("RGB", (160, 120), (200, 200, 200)).save(image, format="JPEG")
        boundary = uuid.uuid4().hex
        # Normal prediction path also creates XML/CSV/TPS using bundled models.
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="view_type"\r\n\r\ndorsal\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            'Content-Type: image/jpeg\r\n\r\n'
        ).encode() + image.getvalue() + f'\r\n--{boundary}--\r\n'.encode()
        uploaded, _ = request("/data", body, {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"})
        result = json.loads(uploaded)
        assert len(result) == 1 and result[0].get("coords") and not result[0].get("error"), result
        coords = [{"id": 0, "x": 20, "y": 30}, {"id": 1, "x": 80, "y": 70}]
        exported = post("/endpoint", {"name": name, "coords": coords}, headers)
        assert exported.get("image_urls"), exported
        png, _ = request(exported["image_urls"][0], headers=headers)
        annotated = Image.open(BytesIO(png)).convert("RGB")
        assert annotated.size == (160, 120)
        assert annotated.getpixel((20, 30)) == (255, 0, 0)
        assert annotated.getpixel((80, 70)) == (255, 0, 0)

    archive_bytes, response_headers = request("/download_all", headers=headers)
    assert "attachment" in response_headers.get("Content-Disposition", "")
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        assert archive.testzip() is None
        files = archive.namelist()
        for name in names:
            stem = Path(name).stem
            assert any(Path(entry).name == f"annotated_{stem}.png" for entry in files), files
            tps_files = [entry for entry in files if Path(entry).name == f"{stem}.tps"]
            assert tps_files, files
            tps = archive.read(tps_files[0]).decode()
            assert "LM=2\n20.0 30.0\n80.0 70.0\n" in tps and f"IMAGE={stem}" in tps, tps
            csv_files = [entry for entry in files if Path(entry).name == f"output_{name}.csv"]
            assert csv_files, f"Missing prediction CSV for {name}; check bundled pandas: {files}"
            rows = list(csv.reader(StringIO(archive.read(csv_files[0]).decode())))
            assert len(rows) >= 2 and "X0" in rows[0] and "Y0" in rows[0], rows
    print(json.dumps({"status": "passed", "specimens": len(names), "checks": ["prediction CSV", "edited TPS", "annotated PNG pixels", "ZIP download"], "zip_bytes": len(archive_bytes)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", nargs="?", type=Path)
    parser.add_argument("--source", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "backend/app.py")] if args.source else [str(args.executable.resolve())] if args.executable else None
    if command is None:
        parser.error("provide the frozen executable or --source")
    with tempfile.TemporaryDirectory(prefix="automorph-export-smoke-") as directory:
        work = Path(directory)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = dict(os.environ)
        env.update(API_PORT=str(port), AUTOMORPH_DATA_DIR=str(work), DB_PATH=str(work / "test.db"), SESSION_DIR=str(work / "sessions"), RUNS_DIR=str(work / "runs"), PREDICTOR_LIBRARY_DIR=str(work / "predictors"), PYTHONUNBUFFERED="1")
        # Never inherit hosted production selections or a local dotenv file.
        env.update(WEB_MODEL_STATE_PATH="", PYTHON_DOTENV_DISABLED="1")
        with (work / "backend.log").open("w+") as log:
            proc = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=log)
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 120
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Backend exited with code {proc.returncode}")
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Backend did not become healthy in 120 seconds")
                    time.sleep(0.25)
                exercise(base)
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    main()

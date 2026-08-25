import os

import cv2
import numpy as np
import pytest

import app
import visual_individual_performance
from export_handler import ExportHandler
from session_manager import SessionManager


@pytest.fixture()
def session_client(monkeypatch, tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"), runtime_root=str(tmp_path))
    fallback_images = tmp_path / "image_download"
    fallback_images.mkdir()

    monkeypatch.setattr(app, "session_manager", manager)
    monkeypatch.setattr(app, "export_handler", ExportHandler(str(tmp_path / "outputs")))
    monkeypatch.setattr(app, "IMAGE_DOWNLOAD_FOLDER", str(fallback_images))

    app.app.config.update(TESTING=True)
    with app.app.test_client() as client:
        yield client, manager


def test_the_exported_annotated_image_url_serves_the_annotated_image(session_client):
    client, manager = session_client
    session_id = manager.create_session()
    upload_folder = manager.get_session(session_id)["upload_folder"]
    cv2.imwrite(
        os.path.join(upload_folder, "specimen.png"),
        np.full((40, 60, 3), 200, dtype=np.uint8),
    )

    exported = client.post(
        "/endpoint",
        json={
            "name": "specimen.png",
            "coords": [{"x": 10, "y": 12}, {"x": 30, "y": 24}],
        },
        headers={"X-Session-ID": session_id},
    )

    assert exported.status_code == 200
    payload = exported.get_json()
    assert payload["image_urls"], payload

    served = client.get(payload["image_urls"][0], headers={"X-Session-ID": session_id})

    assert served.status_code == 200
    annotated = cv2.imdecode(np.frombuffer(served.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert annotated.shape == (40, 60, 3)
    assert tuple(int(channel) for channel in annotated[12, 10]) == (0, 0, 255)
    assert tuple(int(channel) for channel in annotated[24, 30]) == (0, 0, 255)
    assert tuple(int(channel) for channel in annotated[0, 0]) == (200, 200, 200)


def test_a_session_is_listed_under_the_short_id_used_in_image_urls(tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"), runtime_root=str(tmp_path))
    session_id = manager.create_session("0123456789abcdef0123456789abcdef")

    listed = manager.list_sessions()

    assert [entry["session_id_short"] for entry in listed] == [session_id[:8]]
    date, _, clock = listed[0]["created_at"].partition("_")
    assert len(date) == 8 and len(clock) == 6


def test_the_annotated_image_is_built_from_the_upload_whatever_its_extension(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    annotated_folder = tmp_path / "annotated"
    annotated_folder.mkdir()
    cv2.imwrite(
        str(uploads / "specimen.tif"), np.full((30, 20, 3), 128, dtype=np.uint8)
    )
    tps_path = tmp_path / "specimen.tps"
    tps_path.write_text("LM=1\n5.0 25.0\nIMAGE=specimen\n")

    output_paths = visual_individual_performance.create_image(
        str(tps_path), str(annotated_folder), str(uploads)
    )

    assert [os.path.basename(path) for path in output_paths] == [
        "annotated_specimen.png"
    ]
    written = cv2.imread(output_paths[0])
    assert tuple(int(channel) for channel in written[5, 5]) == (0, 0, 255)

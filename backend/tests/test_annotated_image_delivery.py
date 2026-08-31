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


def test_a_dotted_specimen_name_still_yields_a_served_annotated_image(session_client):
    client, manager = session_client
    session_id = manager.create_session()
    upload_folder = manager.get_session(session_id)["upload_folder"]
    cv2.imwrite(
        os.path.join(upload_folder, "LIZ.001.jpg"),
        np.full((40, 60, 3), 200, dtype=np.uint8),
    )

    exported = client.post(
        "/endpoint",
        json={"name": "LIZ.001.jpg", "coords": [{"x": 10, "y": 12}]},
        headers={"X-Session-ID": session_id},
    )

    payload = exported.get_json()
    assert payload["image_urls"], payload

    served = client.get(payload["image_urls"][0], headers={"X-Session-ID": session_id})

    assert served.status_code == 200
    annotated = cv2.imdecode(np.frombuffer(served.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert tuple(int(channel) for channel in annotated[12, 10]) == (0, 0, 255)


def test_two_specimens_sharing_a_dotted_prefix_keep_separate_outputs(session_client):
    client, manager = session_client
    session_id = manager.create_session()
    session_data = manager.get_session(session_id)
    for name in ("LIZ.001.jpg", "LIZ.002.jpg"):
        cv2.imwrite(
            os.path.join(session_data["upload_folder"], name),
            np.full((40, 60, 3), 200, dtype=np.uint8),
        )

    advertised = {}
    for name, x in (("LIZ.001.jpg", 10), ("LIZ.002.jpg", 30)):
        exported = client.post(
            "/endpoint",
            json={"name": name, "coords": [{"x": x, "y": 12}]},
            headers={"X-Session-ID": session_id},
        )
        payload = exported.get_json()
        assert payload["image_urls"], payload
        advertised[name] = payload["image_urls"][0]

    tps_files = sorted(os.listdir(session_data["tps_folder"]))

    assert tps_files == ["LIZ.001.tps", "LIZ.002.tps"]
    assert advertised["LIZ.001.jpg"] != advertised["LIZ.002.jpg"]

    landmarks = {}
    for name, url in advertised.items():
        served = client.get(url, headers={"X-Session-ID": session_id})
        assert served.status_code == 200
        annotated = cv2.imdecode(
            np.frombuffer(served.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        landmarks[name] = [
            column
            for column in range(annotated.shape[1])
            if tuple(int(channel) for channel in annotated[12, column]) == (0, 0, 255)
        ]

    assert 10 in landmarks["LIZ.001.jpg"] and 30 not in landmarks["LIZ.001.jpg"]
    assert 30 in landmarks["LIZ.002.jpg"] and 10 not in landmarks["LIZ.002.jpg"]


def test_serving_an_annotated_image_does_not_scan_the_session_trees(
    session_client, monkeypatch
):
    client, manager = session_client
    session_id = manager.create_session()
    session_data = manager.get_session(session_id)
    cv2.imwrite(
        os.path.join(session_data["upload_folder"], "specimen.png"),
        np.full((40, 60, 3), 200, dtype=np.uint8),
    )
    for _ in range(3):
        manager.create_session()

    exported = client.post(
        "/endpoint",
        json={"name": "specimen.png", "coords": [{"x": 10, "y": 12}]},
        headers={"X-Session-ID": session_id},
    )
    image_url = exported.get_json()["image_urls"][0]

    walks = []
    real_walk = os.walk

    def counting_walk(top, *args, **kwargs):
        walks.append(top)
        return real_walk(top, *args, **kwargs)

    monkeypatch.setattr(os, "walk", counting_walk)
    served = client.get(image_url, headers={"X-Session-ID": session_id})

    assert served.status_code == 200
    assert walks == []


def test_an_unknown_short_session_id_matches_no_folder(tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"), runtime_root=str(tmp_path))
    session_id = manager.create_session("0123456789abcdef0123456789abcdef")

    assert manager.find_session_folder(session_id[:8]) == manager.get_session(
        session_id
    )["session_folder"]
    assert manager.find_session_folder("deadbeef") is None


def test_a_filename_with_url_punctuation_still_serves_its_annotated_image(
    session_client,
):
    client, manager = session_client
    session_id = manager.create_session()
    session_data = manager.get_session(session_id)
    cv2.imwrite(
        os.path.join(session_data["upload_folder"], "LIZ #12.png"),
        np.full((40, 60, 3), 200, dtype=np.uint8),
    )

    exported = client.post(
        "/endpoint",
        json={"name": "LIZ #12.png", "coords": [{"x": 10, "y": 12}]},
        headers={"X-Session-ID": session_id},
    )
    payload = exported.get_json()

    assert payload["image_urls"], payload
    image_url = payload["image_urls"][0]

    served = client.get(image_url, headers={"X-Session-ID": session_id})

    assert served.status_code == 200
    annotated = cv2.imdecode(
        np.frombuffer(served.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert tuple(int(channel) for channel in annotated[12, 10]) == (0, 0, 255)


def test_the_newest_folder_wins_when_two_share_a_short_session_id(
    session_client, monkeypatch
):
    client, manager = session_client
    session_id = manager.create_session()
    session_data = manager.get_session(session_id)
    cv2.imwrite(
        os.path.join(session_data["upload_folder"], "specimen.png"),
        np.full((40, 60, 3), 200, dtype=np.uint8),
    )

    stale_annotated = os.path.join(
        manager.base_sessions_dir,
        f"session_19990101_000000_{session_id[:8]}",
        "annotated",
    )
    os.makedirs(stale_annotated)
    cv2.imwrite(
        os.path.join(stale_annotated, "annotated_specimen.png"),
        np.zeros((40, 60, 3), dtype=np.uint8),
    )

    exported = client.post(
        "/endpoint",
        json={"name": "specimen.png", "coords": [{"x": 10, "y": 12}]},
        headers={"X-Session-ID": session_id},
    )
    image_url = exported.get_json()["image_urls"][0]

    real_listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: sorted(real_listdir(path)))
    served = client.get(image_url, headers={"X-Session-ID": session_id})

    assert served.status_code == 200
    annotated = cv2.imdecode(
        np.frombuffer(served.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert tuple(int(channel) for channel in annotated[12, 10]) == (0, 0, 255)
    assert tuple(int(channel) for channel in annotated[0, 0]) == (200, 200, 200)

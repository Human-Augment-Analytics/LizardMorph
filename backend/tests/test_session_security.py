import os
import uuid

import pytest

from backend.session_manager import SessionManager


def test_session_manager_rejects_non_uuid_identifier(tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="Invalid session identifier"):
        manager.create_session("../../outside")

    assert manager.get_session("../../outside") is None
    assert not (tmp_path / "outside").exists()


def test_session_folder_persists_full_uuid(tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"))
    session_id = str(uuid.uuid4())

    manager.create_session(session_id)
    session = manager.get_session(session_id)

    assert session is not None
    assert os.path.basename(session["session_folder"]).endswith(f"_{session_id[:8]}")


def test_image_route_does_not_follow_filename_traversal(client):
    session_id = str(uuid.uuid4())
    response = client.get(
        "/image_file",
        query_string={
            "image_filename": "../../app.py",
            "type": "original",
            "session_id": session_id,
        },
    )

    assert response.status_code == 404
    assert response.mimetype == "application/json"

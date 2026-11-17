import importlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shutil
import uuid

import pytest

from backend.production_selection import (
    ProductionSelectionError, active_model_id, new_session_pin,
    save_session_pin, select_model, session_model_id,
)
from backend.session_manager import SessionManager


FIRST = 'web-' + 'a' * 24
SECOND = 'web-' + 'b' * 24


def write_state(path, model_id):
    path.write_text(json.dumps({'format': 'lizard-web-state-v1',
                               'active': {'model_id': model_id} if model_id else None}))


@pytest.fixture
def selections(tmp_path, monkeypatch):
    state = tmp_path / 'state.json'
    monkeypatch.setenv('WEB_MODEL_STATE_PATH', str(state))
    manager = SessionManager(str(tmp_path / 'sessions'), production_pin_factory=new_session_pin)
    return state, manager


def test_new_sessions_pin_once_across_promotion_restart_and_clear(selections):
    state, manager = selections
    write_state(state, FIRST)
    old = manager.create_session()
    write_state(state, SECOND)
    new = manager.create_session()
    old_folder = manager.get_session(old)['session_folder']
    new_folder = manager.get_session(new)['session_folder']
    assert session_model_id(old_folder) == FIRST
    assert session_model_id(new_folder) == SECOND
    assert manager.create_session(old) == old
    assert session_model_id(old_folder) == FIRST
    manager.clear_session(old)
    assert session_model_id(old_folder) == FIRST
    restarted = SessionManager(manager.base_sessions_dir, production_pin_factory=new_session_pin)
    assert session_model_id(restarted.get_session(old)['session_folder']) == FIRST


def test_unconfigured_old_and_legacy_sessions_do_not_adopt_later_release(selections):
    state, manager = selections
    first = manager.create_session()
    write_state(state, FIRST)
    assert session_model_id(manager.get_session(first)['session_folder']) is None
    old = SessionManager(manager.base_sessions_dir).create_session()
    assert session_model_id(manager.get_session(old)['session_folder']) is None
    write_state(state, None)
    rollback = manager.create_session()
    assert session_model_id(manager.get_session(rollback)['session_folder']) is None


@pytest.mark.parametrize('content', ['{', '[]', '{}', '{"format":"wrong","active":null}',
    '{"format":"lizard-web-state-v1","active":{"model_id":"latest"}}'])
def test_invalid_deployment_fails_closed_only_when_default_is_requested(selections, content):
    state, manager = selections
    state.write_text(content)
    with pytest.raises(ProductionSelectionError):
        active_model_id()
    sid = manager.create_session()
    folder = manager.get_session(sid)['session_folder']
    with pytest.raises(ProductionSelectionError):
        select_model('toepad', None, folder)
    assert select_model('toepad', 'explicit-version', folder) == 'explicit-version'
    assert select_model('dorsal', None, folder) is None
    assert select_model('toepad', None, folder, predictor_type='scale') is None


def test_image_provenance_preserves_explicit_choice_when_reprocessing(selections):
    state, manager = selections
    write_state(state, FIRST)
    sid = manager.create_session()
    folder = Path(manager.get_session(sid)['session_folder'])
    (folder / 'outputs' / 'specimen.jpg.model.json').write_text(json.dumps({'model_id': SECOND}))
    assert select_model('toepad', None, folder, filename='specimen.jpg') == SECOND
    assert select_model('toepad', FIRST, folder, filename='specimen.jpg') == FIRST
    assert select_model('toepad', None, folder) == FIRST


def test_corrupt_session_or_image_pin_never_silently_uses_another_model(selections):
    state, manager = selections
    write_state(state, FIRST)
    sid = manager.create_session()
    folder = Path(manager.get_session(sid)['session_folder'])
    (folder / 'production-model.json').write_text('{')
    with pytest.raises(ProductionSelectionError):
        select_model('toepad', None, folder)
    (folder / 'outputs' / 'specimen.jpg.model.json').write_text('{}')
    with pytest.raises(ProductionSelectionError):
        select_model('toepad', None, folder, filename='specimen.jpg')


@pytest.fixture
def production_app(client, selections, monkeypatch, tmp_path):
    import cv2
    import numpy as np
    from backend.domain.models import Manifest, DetectorConfig, ClassConfig, LandmarkSchemaConfig
    app_mod = importlib.import_module('app')
    state, manager = selections
    monkeypatch.setattr(app_mod, 'session_manager', manager)
    resolved = []
    def resolve(model_id):
        resolved.append(model_id)
        return Manifest(schema_version=1, id=model_id, name=model_id, description='',
            detector=DetectorConfig(artifact=''),
            classes=[ClassConfig(id=0, name='crop', landmark_schema='points')],
            landmark_schemas={'points': LandmarkSchemaConfig(points=['0'])},
            evaluation={'registry': {'name': 'example', 'version': '1'}}), str(tmp_path)
    monkeypatch.setattr(app_mod, 'resolve_inference_model', resolve)
    monkeypatch.setattr(app_mod.generic_pipeline_engine, 'predict', lambda *a, **k: [])
    monkeypatch.setattr(app_mod, 'generic_predictions_to_annotations',
                        lambda result: ([{'id': 0, 'x': 1.0, 'y': 2.0}], []))
    monkeypatch.setattr(app_mod.xray_preprocessing, 'process_single_image', shutil.copyfile)
    monkeypatch.setattr(app_mod.visual_individual_performance, 'invert_single_image', shutil.copyfile)
    ok, image = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return app_mod, state, manager, resolved, image.tobytes()


def start(client):
    headers = {'X-Session-ID': str(uuid.uuid4())}
    response = client.post('/api/session/start', headers=headers)
    assert response.status_code == 200
    return headers


def upload(client, headers, image, **fields):
    return client.post('/api/data', headers=headers, data={
        'view_type': 'toepad', 'image': (io.BytesIO(image), 'specimen.jpg'), **fields})


def test_actual_upload_and_reprocessing_keep_pinned_model_after_default_flip(client, production_app):
    app_mod, state, manager, resolved, image = production_app
    write_state(state, FIRST)
    old_headers = start(client)
    write_state(state, SECOND)
    new_headers = start(client)
    old = upload(client, old_headers, image)
    new = upload(client, new_headers, image)
    assert old.status_code == new.status_code == 200
    assert old.get_json()[0]['model_provenance']['model_id'] == FIRST
    assert new.get_json()[0]['model_provenance']['model_id'] == SECOND
    reprocessed = client.post('/api/process_existing?filename=specimen.jpg&view_type=toepad', headers=old_headers)
    assert reprocessed.status_code == 200
    assert reprocessed.get_json()['model_provenance']['model_id'] == FIRST
    assert resolved == [FIRST, SECOND, FIRST]


def test_explicit_upload_choice_is_preserved_per_image(client, production_app):
    app_mod, state, manager, resolved, image = production_app
    write_state(state, FIRST)
    headers = start(client)
    assert upload(client, headers, image, model_id=SECOND).status_code == 200
    response = client.post('/api/process_existing?filename=specimen.jpg&view_type=toepad', headers=headers)
    assert response.status_code == 200
    assert resolved == [SECOND, SECOND]


def test_predict_route_uses_session_default_and_explicit_selection(client, production_app):
    app_mod, state, manager, resolved, image = production_app
    write_state(state, FIRST)
    headers = start(client)
    write_state(state, SECOND)
    for expected, fields in [(FIRST, {}), (SECOND, {'model_id': SECOND})]:
        response = client.post('/api/predict', headers=headers, data={
            'view_type': 'toepad', 'image': (io.BytesIO(image), 'specimen.jpg'), **fields})
        assert response.status_code == 200
        assert response.get_json()['model_version_id'] == expected
    assert resolved == [FIRST, SECOND]


def test_malformed_default_returns_503_without_loading_an_alternative(client, production_app):
    app_mod, state, manager, resolved, image = production_app
    state.write_text('{')
    headers = start(client)
    assert upload(client, headers, image).status_code == 503
    response = client.post('/api/predict', headers=headers, data={
        'view_type': 'toepad', 'image': (io.BytesIO(image), 'specimen.jpg')})
    assert response.status_code == 503
    assert resolved == []
    assert upload(client, headers, image, model_id=SECOND).status_code == 200


def test_concurrent_session_start_captures_only_one_default(tmp_path):
    snapshots = []
    def snapshot():
        snapshots.append(FIRST)
        return {'format': 'lizard-session-production-v1', 'model_id': FIRST}
    manager = SessionManager(str(tmp_path / 'sessions'), production_pin_factory=snapshot)
    sid = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: manager.create_session(sid), range(20))) == [sid] * 20
    assert snapshots == [FIRST]
    assert session_model_id(manager.get_session(sid)['session_folder']) == FIRST


def test_browser_new_session_ignores_old_flask_cookie_after_default_flip(client, production_app):
    app_mod, state, manager, resolved, image = production_app
    write_state(state, FIRST)
    first = client.post('/api/session/start')
    assert first.status_code == 200
    old_id = first.get_json()['session_id']
    # The test client retains Flask's session cookie, just as the real browser
    # does after SessionService clears only its own storage and starts again.
    write_state(state, SECOND)
    second = client.post('/api/session/start')
    assert second.status_code == 200
    new_id = second.get_json()['session_id']
    assert old_id != new_id
    assert session_model_id(manager.get_session(old_id)['session_folder']) == FIRST
    assert session_model_id(manager.get_session(new_id)['session_folder']) == SECOND
    old = upload(client, {'X-Session-ID': old_id}, image)
    new = upload(client, {}, image)
    assert old.get_json()[0]['model_provenance']['model_id'] == FIRST
    assert new.get_json()[0]['model_provenance']['model_id'] == SECOND


def test_explicit_start_header_reuses_pin_even_with_another_cookie(client, production_app):
    app_mod, state, manager, resolved, image = production_app
    write_state(state, FIRST)
    old_headers = start(client)
    old_id = old_headers['X-Session-ID']
    write_state(state, SECOND)
    assert client.post('/api/session/start').get_json()['session_id'] != old_id
    restarted = client.post('/api/session/start', headers=old_headers)
    assert restarted.status_code == 200
    assert restarted.get_json()['session_id'] == old_id
    assert session_model_id(manager.get_session(old_id)['session_folder']) == FIRST

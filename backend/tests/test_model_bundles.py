import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from backend.model_bundles import validate_bundle
from backend.domain.models import Manifest


def make_bundle(path, *, corrupt=False, extra=None):
    content = b'fixture predictor bytes'
    spec = {'format': 'lizard-pipeline-v1', 'provenance': {'source_commit': 'abc123', 'evaluation_status': 'test'},
            'pipeline': {'schema_version': 1, 'id': 'example', 'name': 'Example', 'description': '',
                         'detector': {'artifact': ''},
                         'classes': [{'id': 0, 'name': 'crop', 'predictor': 'predictor.dat', 'landmark_schema': 'points'}],
                         'landmark_schemas': {'points': {'points': ['0', '1']}}},
            'artifacts': {'predictor.dat': {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}}}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('bundle.json', json.dumps(spec))
        archive.writestr('predictor.dat', content if not corrupt else b'corrupt')
        if extra:
            archive.writestr(extra, b'undeclared')
    return spec


def test_bundle_preserves_contract_and_hashes(tmp_path):
    archive = tmp_path / 'bundle.zip'
    expected = make_bundle(archive)
    dest = tmp_path / 'extracted'; dest.mkdir()
    actual = validate_bundle(archive, dest)
    assert actual == expected
    assert (dest / 'predictor.dat').read_bytes() == b'fixture predictor bytes'
    assert Manifest.from_dict(actual['pipeline']).to_dict()['classes'][0]['predictor_type'] == 'dlib'


@pytest.mark.parametrize('extra', ['../escape.dat', '/absolute.dat', 'extra.dat', 'C:/outside.dat', 'C:outside.dat'])
def test_untrusted_archive_never_escapes_destination(tmp_path, extra):
    archive = tmp_path / 'bundle.zip'; make_bundle(archive, extra=extra)
    dest = tmp_path / 'extracted'; dest.mkdir()
    with pytest.raises(ValueError):
        validate_bundle(archive, dest)
    assert not (tmp_path / 'escape.dat').exists()
    assert list(dest.iterdir()) == []


def test_corrupt_artifact_is_rejected(tmp_path):
    archive = tmp_path / 'bundle.zip'; make_bundle(archive, corrupt=True)
    dest = tmp_path / 'extracted'; dest.mkdir()
    with pytest.raises(ValueError, match='checksum'):
        validate_bundle(archive, dest)


def test_manifest_preserves_coordinate_runtime():
    manifest = Manifest.from_dict({'schema_version': 1, 'id': 'vit', 'detector': {'artifact': ''},
        'classes': [{'id': 0, 'name': 'crop', 'predictor': 'vit.onnx', 'predictor_type': 'onnx-coordinates'}]})
    assert Manifest.from_dict(manifest.to_dict()).classes[0].predictor_type == 'onnx-coordinates'

@pytest.fixture()
def registry_import(monkeypatch, tmp_path):
    import os
    from backend import model_bundles
    from backend.storage.db import DatabaseManager
    from backend.storage.repository import ModelRegistryRepository
    fixture_root = os.environ.get('TEST_FIXTURES')
    if not fixture_root:
        pytest.skip('real dlib fixture required for loader compatibility checks')
    predictor = Path(fixture_root) / 'incoming/lizard_scale.dat'
    payload = predictor.read_bytes()
    archive = tmp_path / 'real.zip'
    spec = make_bundle(tmp_path / 'unused.zip')
    spec['artifacts'] = {'predictor.dat': {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}
    def serve(count):
        spec['pipeline']['landmark_schemas']['points']['points'] = [str(i) for i in range(count)]
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('bundle.json', json.dumps(spec))
            output.writestr('predictor.dat', payload)
        body = archive.read_bytes()
        class Response:
            headers = {'X-Bundle-SHA256': hashlib.sha256(body).hexdigest()}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): yield body
        monkeypatch.setattr(model_bundles.requests, 'get', lambda *a, **k: Response())
    db = DatabaseManager(str(tmp_path / 'app.db')); db.init_db()
    repository = ModelRegistryRepository(db, models_dir=str(tmp_path / 'models'))
    def install():
        return model_bundles.import_registry_bundle('toepad-test', '1', api_url='http://registry.test',
            models_dir=str(tmp_path / 'models'), repository=repository)
    return serve, install, repository


def test_real_dlib_point_mismatch_is_rejected_before_catalog_registration(registry_import):
    serve, install, repository = registry_import
    serve(9)  # actual lizard_scale.dat has two points
    with pytest.raises(ValueError, match='landmark count'):
        install()
    assert repository.list_model_versions() == []


def test_reimport_detects_corrupt_installed_weights(registry_import):
    serve, install, repository = registry_import
    serve(2)
    result = install()
    previous = repository.get_model_version(result['id'])
    Path(previous.manifest.classes[0].predictor).write_bytes(b'truncated')
    with pytest.raises(ValueError, match='Installed bundle artifact checksum'):
        install()
    current = repository.get_model_version(result['id'])
    assert current.created_at == previous.created_at
    assert current.manifest.to_dict() == previous.manifest.to_dict()


def test_case_colliding_members_are_rejected_before_extraction(tmp_path):
    archive = tmp_path / 'bundle.zip'; make_bundle(archive, extra='PREDICTOR.dat')
    dest = tmp_path / 'extracted'; dest.mkdir()
    with pytest.raises(ValueError, match='case-colliding'):
        validate_bundle(archive, dest)
    assert list(dest.iterdir()) == []

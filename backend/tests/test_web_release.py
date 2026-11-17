"""Deployment transactions use synthetic nonexecuted artifacts, never production data."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from backend import web_release as release


def candidate(root, tag='model/toepad-web/v1.0.0'):
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {'detector.onnx': b'nonexecuted test detector', 'digit.dat': tag.encode()}
    bundle = {'format': 'lizard-pipeline-v1',
        'provenance': {'source_commit': 'a' * 40, 'deployment_targets': ['web'],
                       'evaluation_status': 'heldout-evaluated'},
        'pipeline': {'schema_version': 1, 'id': 'fixture', 'name': 'Synthetic fixture',
            'description': 'Not a scientifically approved model',
            'detector': {'artifact': 'detector.onnx', 'geometry': 'obb', 'inference_protocol': 'toepad-dual-pass'},
            'classes': [{'id': i, 'name': n, 'predictor': 'digit.dat', 'predictor_type': 'dlib',
                         'landmark_schema': 'digit', 'preprocessing': 'toepad-rectify-512'}
                        for i, n in enumerate(['up_finger', 'up_toe', 'bot_finger', 'bot_toe'])],
            'landmark_schemas': {'digit': {'points': [str(i) for i in range(9)]}}},
        'artifacts': {n: {'size': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for n, v in artifacts.items()}}
    with zipfile.ZipFile(root / 'bundle.zip', 'w') as z:
        z.writestr('bundle.json', json.dumps(bundle))
        for n, v in artifacts.items():
            z.writestr(n, v)
    digest = release.file_sha256(root / 'bundle.zip')
    report = {'format': 'lizard-web-evaluation-v1', 'approved': True, 'reviewer': 'unit-test-only',
        'benchmark_id': 'synthetic-no-scientific-claim', 'source_commit': 'a' * 40,
        'dataset_sha256': 'b' * 64, 'bundle_sha256': digest, 'metrics': {'score': 1},
        'criteria': [{'metric': 'score', 'operator': '>=', 'threshold': 1}]}
    (root / 'evaluation.json').write_text(json.dumps(report))
    metadata = {'format': 'lizard-web-release-v1', 'schema_version': 1, 'channel': 'web-production',
        'release_tag': tag, 'bundle': {'asset': 'bundle.zip', 'sha256': digest,
                                     'size': (root / 'bundle.zip').stat().st_size},
        'registry': {'name': 'test-model', 'version': '1', 'run_id': 'c' * 32},
        'compatibility': {'target': 'web', 'runtime': 'onnx-dlib', 'contract': 'toepad-full-v1'},
        'evaluation': {'asset': 'evaluation.json', 'status': 'approved',
                       'sha256': release.file_sha256(root / 'evaluation.json')}}
    (root / 'web-release.json').write_text(json.dumps(metadata))
    return metadata


class Source:
    repository = 'trusted/models'

    def __init__(self, directory, identifier=1):
        self.directory = directory
        self.metadata = json.loads((directory / 'web-release.json').read_text())
        self.release = {'id': identifier, 'tag_name': self.metadata['release_tag']}

    def select(self, tag=None):
        return self.release

    def download(self, remote, destination):
        for name in ('bundle.zip', 'web-release.json', 'evaluation.json'):
            shutil.copyfile(self.directory / name, Path(destination) / name)
        return release.validate_release(destination, remote['tag_name'])


class Repository:
    def __init__(self):
        self.models = {}

    def register_model_bundle(self, project, manifest):
        self.models[manifest.id] = manifest

    def get_model_version(self, model_id):
        return self.models.get(model_id)


def passing_smoke(manifest, image):
    return {'passed': True, 'objects': 2}


def deploy_fixture(tmp_path, *, tag='model/toepad-web/v1.0.0', identifier=1, repository=None):
    directory = tmp_path / ('candidate-' + str(identifier))
    candidate(directory, tag)
    state = tmp_path / 'deployment/state.json'
    repo = repository or Repository()
    result = release.deploy(Source(directory, identifier), state, tmp_path / 'models', repo,
                            'not-used.jpg', smoke=passing_smoke)
    return result, state, repo


def test_activation_installs_exact_bytes_and_atomic_receipt(tmp_path):
    result, state, repo = deploy_fixture(tmp_path)
    assert result['status'] == 'activated'
    data = release.load_state(state)
    active = data['active']
    assert active['model_id'] in repo.models
    assert data['history'] == [None]
    directory = tmp_path / 'models/web-releases' / active['model_id']
    assert release.file_sha256(directory / 'bundle.zip') == active['bundle_sha256']
    assert release.verify_installed(directory, active['bundle_sha256'])[0]['pipeline']['name'] == 'Synthetic fixture'
    assert not list(state.parent.glob('.web-release-*'))


def test_failed_smoke_keeps_previous_pointer_and_catalog(tmp_path):
    _, state, repo = deploy_fixture(tmp_path)
    before = state.read_bytes()
    directory = tmp_path / 'second'; candidate(directory, 'model/toepad-web/v1.1.0')
    def fail(*args):
        raise ValueError('synthetic runtime failure')
    with pytest.raises(ValueError, match='runtime failure'):
        release.deploy(Source(directory, 2), state, tmp_path / 'models', repo,
                       'unused.jpg', smoke=fail)
    assert state.read_bytes() == before
    assert len(repo.models) == 1
    assert not list(state.parent.glob('.web-release-*'))
    assert release.deploy(Source(directory, 2), state, tmp_path / 'models', repo,
                          'unused.jpg', smoke=fail)['status'] == 'release-rejected'


def test_corrupt_release_never_changes_pointer(tmp_path):
    _, state, repo = deploy_fixture(tmp_path)
    before = state.read_bytes()
    directory = tmp_path / 'second'; candidate(directory, 'model/toepad-web/v2.0.0')
    (directory / 'bundle.zip').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        release.deploy(Source(directory, 2), state, tmp_path / 'models', repo,
                       'unused.jpg', smoke=passing_smoke)
    assert state.read_bytes() == before


def test_rollback_to_legacy_holds_automatic_redeployment(tmp_path):
    _, state, repo = deploy_fixture(tmp_path)
    result = release.rollback(state, tmp_path / 'models', repo, 'unused.jpg', smoke=passing_smoke)
    assert result == {'status': 'rolled-back', 'active': None, 'held': True}
    source = Source(tmp_path / 'candidate-1')
    assert release.deploy(source, state, tmp_path / 'models', repo, 'unused.jpg')['status'] == 'held'
    assert release.resume(state)['status'] == 'resumed'
    assert not release.load_state(state)['held']


def test_rollback_rejects_edited_installed_manifest(tmp_path):
    _, state, repo = deploy_fixture(tmp_path)
    old = release.load_state(state)['active']
    deploy_fixture(tmp_path, tag='model/toepad-web/v1.1.0', identifier=2, repository=repo)
    before = state.read_bytes()
    path = tmp_path / 'models/web-releases' / old['model_id'] / 'bundle.json'
    bundle = json.loads(path.read_text()); bundle['pipeline']['classes'][0]['predictor_type'] = 'onnx-coordinates'
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='manifest was changed'):
        release.rollback(state, tmp_path / 'models', repo, 'unused.jpg', smoke=passing_smoke)
    assert state.read_bytes() == before


def test_rollback_to_previous_version_uses_local_bytes_and_holds(tmp_path):
    _, state, repo = deploy_fixture(tmp_path)
    previous = release.load_state(state)['active']
    deploy_fixture(tmp_path, tag='model/toepad-web/v1.1.0', identifier=2, repository=repo)
    restored = release.rollback(state, tmp_path / 'models', repo, 'unused.jpg', smoke=passing_smoke)
    assert restored['active'] == previous and restored['held']


def test_preview_validates_but_creates_no_catalog_or_pointer(tmp_path):
    directory = tmp_path / 'candidate'; candidate(directory)
    repo = Repository(); state = tmp_path / 'deployment/state.json'
    result = release.deploy(Source(directory), state, tmp_path / 'models', repo,
                            None, preview=True)
    assert result['status'] == 'validated-preview'
    assert not state.exists() and not repo.models


def test_deployment_lock_prevents_concurrent_installs(tmp_path):
    state = tmp_path / 'state.json'
    with release.deployment_lock(state):
        with pytest.raises(BlockingIOError):
            with release.deployment_lock(state):
                pytest.fail('second deployment acquired the lock')


def test_release_selection_ignores_desktop_prerelease_draft_and_orders_semver():
    source = release.GitHubReleaseSource('trusted/models')
    source._json = lambda url: [
        {'tag_name': 'model/toepad-web/v1.9.0', 'id': 1},
        {'tag_name': 'model/toepad-web/v1.10.0', 'id': 2},
        {'tag_name': 'model/toepad-web/v2.0.0', 'id': 3, 'draft': True},
        {'tag_name': 'model/toepad-web/v3.0.0', 'id': 4, 'prerelease': True},
        {'tag_name': 'model/desktop-vit/v9.0.0', 'id': 5}]
    assert source.select()['id'] == 2


def test_download_rejects_foreign_asset_url(tmp_path):
    source = release.GitHubReleaseSource('trusted/models')
    assets = [{'name': name, 'size': 5, 'browser_download_url': 'https://evil.example/' + name}
              for name in ('bundle.zip', 'web-release.json', 'evaluation.json')]
    with pytest.raises(ValueError, match='outside the configured repository'):
        source.download({'tag_name': 'model/toepad-web/v1.0.0', 'assets': assets}, tmp_path)


def test_smoke_loads_an_upper_predictor_not_seen_by_photo(tmp_path, monkeypatch):
    from backend.inference import engine as engine_module
    from backend.domain.models import Manifest
    import cv2
    import numpy as np
    root = tmp_path / 'candidate'; candidate(root)
    bundle = release.validate_web_bundle(root / 'bundle.zip')
    manifest = Manifest.from_dict(bundle['pipeline'])
    manifest.classes[0].predictor = 'bad-upper.dat'
    loaded = []
    class Engine:
        def _get_detector(self, *args, **kwargs):
            return SimpleNamespace(session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 11, 8400])]))
        def _get_shape_predictor(self, path):
            loaded.append(path)
            return lambda *args: SimpleNamespace(num_parts=8 if path == 'bad-upper.dat' else 9)
        def predict(self, *args):
            pytest.fail('wrong upper predictor must fail before real image inference')
    monkeypatch.setattr(engine_module, 'GenericPipelineEngine', Engine)
    monkeypatch.setattr(cv2, 'imread', lambda path: np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='landmark count'):
        release.smoke_model(manifest, tmp_path / 'not-needed.jpg')
    assert 'bad-upper.dat' in loaded

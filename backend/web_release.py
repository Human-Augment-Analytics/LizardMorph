"""Pull approved GitHub model releases into an immutable local web catalog.

No MLflow dependency or public management endpoint. A release is an input;
the local atomic state file is the record of what actually passed deployment.
"""
import contextlib
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from urllib.parse import quote, urlsplit

import requests

try:
    from backend.domain.models import Manifest
    from backend.model_bundles import MAX_BUNDLE_BYTES, validate_bundle
    from backend.web_release_contract import validate_release, validate_release_tag, validate_web_bundle
except ImportError:
    from domain.models import Manifest
    from model_bundles import MAX_BUNDLE_BYTES, validate_bundle
    from web_release_contract import validate_release, validate_release_tag, validate_web_bundle


STATE_FORMAT = 'lizard-web-state-v1'
RELEASE_PREFIX = 'model/toepad-web/v'
METADATA_LIMIT = 1024 * 1024


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(data, output, sort_keys=True, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {'format': STATE_FORMAT, 'active': None, 'history': [], 'held': False}
    if path.stat().st_size > METADATA_LIMIT:
        raise ValueError('Oversized deployment state')
    data = json.loads(path.read_text())
    if data.get('format') != STATE_FORMAT or not isinstance(data.get('history'), list):
        raise ValueError('Invalid deployment state')
    active = data.get('active')
    if active is not None and (not isinstance(active, dict) or not re.fullmatch(
            r'web-[0-9a-f]{24}', str(active.get('model_id', '')))):
        raise ValueError('Invalid active production model')
    return data


@contextlib.contextmanager
def deployment_lock(state_path):
    path = Path(str(state_path) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def release_version(tag):
    validate_release_tag(tag)
    return tuple(int(part) for part in tag[len(RELEASE_PREFIX):].split('.'))


class GitHubReleaseSource:
    """Public, fixed-repository release reads. No credentials required on Haag."""

    def __init__(self, repository, http=None):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
            raise ValueError('Expected an owner/repository')
        self.repository = repository
        self.http = http or requests.Session()

    def _json(self, url):
        response = self.http.get(url, timeout=(10, 30), headers={
            'Accept': 'application/vnd.github+json', 'User-Agent': 'LizardMorph-web-updater'})
        response.raise_for_status()
        if len(response.content) > 8 * METADATA_LIMIT:
            raise ValueError('Oversized GitHub response')
        return response.json()

    def select(self, tag=None):
        base = f'https://api.github.com/repos/{self.repository}/releases'
        if tag:
            validate_release_tag(tag)
            response = self._json(base + '/tags/' + quote(tag, safe=''))
            releases = [response]
        else:
            releases = self._json(base + '?per_page=100')
        eligible = []
        for release in releases:
            if release.get('draft') or release.get('prerelease'):
                continue
            try:
                version = release_version(release.get('tag_name', ''))
            except (ValueError, TypeError):
                continue
            eligible.append((version, release))
        return max(eligible, key=lambda item: item[0])[1] if eligible else None

    def download(self, release, destination):
        tag = release['tag_name']
        validate_release_tag(tag)
        assets = {item['name']: item for item in release.get('assets', [])}
        if len(assets) != len(release.get('assets', [])):
            raise ValueError('Duplicate release asset names')
        required = {'bundle.zip', 'web-release.json', 'evaluation.json'}
        if not required.issubset(assets):
            raise ValueError('Release is missing required model assets')
        size = assets['bundle.zip'].get('size')
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_BUNDLE_BYTES:
            raise ValueError('Invalid release bundle size')
        if shutil.disk_usage(destination).free < 4 * size + 512 * METADATA_LIMIT:
            raise ValueError('Insufficient space for safe bundle staging')
        for name in ('web-release.json', 'evaluation.json', 'bundle.zip'):
            asset = assets[name]
            url = asset.get('browser_download_url', '')
            parsed = urlsplit(url)
            expected = f'/{self.repository}/releases/download/{tag}/{name}'
            from urllib.parse import unquote
            if parsed.scheme != 'https' or parsed.netloc != 'github.com' or unquote(parsed.path) != expected:
                raise ValueError('Release asset URL is outside the configured repository')
            limit = MAX_BUNDLE_BYTES if name == 'bundle.zip' else METADATA_LIMIT
            if asset.get('size', limit + 1) > limit:
                raise ValueError('Oversized release asset')
            count = 0
            with self.http.get(url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with (Path(destination) / name).open('wb') as output:
                    for block in response.iter_content(1024 * 1024):
                        count += len(block)
                        if count > limit:
                            raise ValueError('Release download exceeded its size limit')
                        output.write(block)
            if count != asset['size']:
                raise ValueError('Release asset size mismatch')
        return validate_release(destination, expected_tag=tag)


def installed_manifest(bundle, directory, model_id, metadata):
    pipeline = copy.deepcopy(bundle['pipeline'])
    pipeline['id'] = model_id
    pipeline['name'] = f"{pipeline['name']} — {metadata['release_tag']}"
    pipeline['detector']['artifact'] = str(directory / pipeline['detector']['artifact'])
    for config in pipeline['classes']:
        if config.get('predictor'):
            config['predictor'] = str(directory / config['predictor'])
    registry = {**metadata['registry'], 'bundle_sha256': metadata['bundle']['sha256']}
    pipeline['evaluation'] = {**pipeline.get('evaluation', {}), 'registry': registry,
                              'web_release': metadata}
    return Manifest.from_dict(pipeline)


def verify_installed(directory, digest, expected_tag=None, metadata_sha256=None):
    directory = Path(directory)
    if directory.resolve() != directory.absolute():
        raise ValueError('Installed model path is a symlink')
    if file_sha256(directory / 'bundle.zip') != digest:
        raise ValueError('Installed release archive digest mismatch')
    metadata = validate_release(directory, expected_tag=expected_tag)
    if metadata_sha256 and file_sha256(directory / 'web-release.json') != metadata_sha256:
        raise ValueError('Installed release metadata digest mismatch')
    bundle = validate_web_bundle(directory / 'bundle.zip')
    if json.loads((directory / 'bundle.json').read_text()) != bundle:
        raise ValueError('Installed bundle manifest was changed')
    if metadata['bundle']['sha256'] != digest:
        raise ValueError('Installed release identity mismatch')
    for name, item in bundle['artifacts'].items():
        path = directory / name
        if path.resolve() != path.absolute() or not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('Installed model artifact is missing or has changed')
        if file_sha256(path) != item['sha256']:
            raise ValueError('Installed model artifact digest mismatch')
    return bundle, metadata


def smoke_model(manifest, image_path):
    """Actual ONNX/dlib inference on a local permitted image; never a ViT runtime."""
    import cv2
    import dlib
    import numpy as np
    try:
        from backend.inference.engine import GenericPipelineEngine
    except ImportError:
        from inference.engine import GenericPipelineEngine
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Configured local smoke image cannot be read')
    engine = GenericPipelineEngine()
    detector = engine._get_detector(manifest.detector.artifact,
                                   [c.name for c in manifest.classes], legacy_toepad=True)
    if detector.session.get_outputs()[0].shape[1] != 11:
        raise ValueError('Web contract requires the H7 six-class detector decoder')
    # A smoke photograph may only contain bottom digits. Validate *all* declared
    # predictors so corrupt upper-digit files cannot pass an incomplete image.
    for config in manifest.classes:
        predictor = engine._get_shape_predictor(config.predictor)
        shape = predictor(np.zeros((64, 64, 3), dtype=np.uint8), dlib.rectangle(0, 0, 63, 63))
        expected = len(manifest.landmark_schemas[config.landmark_schema].points)
        if shape.num_parts != expected:
            raise ValueError('Predictor landmark count does not match the web schema')
    predictions = engine.predict(image, manifest)
    if not predictions:
        raise ValueError('Smoke image produced no objects')
    found = set()
    for prediction in predictions:
        found.add(prediction['class_name'])
        points = prediction['landmarks']
        if len(points) != 9 or not all(math.isfinite(float(p[k])) for p in points for k in ('x', 'y')):
            raise ValueError('Smoke inference produced invalid landmarks')
    if not {'bot_finger', 'bot_toe'}.issubset(found):
        raise ValueError('Smoke image must exercise bottom finger and bottom toe')
    return {'objects': len(predictions), 'classes': sorted(found),
            'image_sha256': file_sha256(image_path), 'passed': True}


def install_bundle(directory, metadata, models_dir, repository, image_path, smoke=smoke_model):
    root = Path(models_dir).resolve() / 'web-releases'
    root.mkdir(parents=True, exist_ok=True)
    digest = metadata['bundle']['sha256']
    model_id = 'web-' + digest[:24]
    final = root / model_id
    if not final.exists():
        with tempfile.TemporaryDirectory(prefix='.web-install-', dir=root) as work:
            extracted = Path(work) / 'model'
            extracted.mkdir()
            validate_bundle(Path(directory) / 'bundle.zip', extracted)
            for name in ('web-release.json', 'evaluation.json', 'bundle.zip'):
                shutil.copyfile(Path(directory) / name, extracted / name)
            os.rename(extracted, final)
    bundle, stored_metadata = verify_installed(final, digest, expected_tag=metadata['release_tag'])
    if stored_metadata != metadata:
        raise ValueError('Immutable installation conflicts with release metadata')
    manifest = installed_manifest(bundle, final, model_id, metadata)
    smoke_receipt = smoke(manifest, image_path)
    repository.register_model_bundle('web-release', manifest)
    return model_id, smoke_receipt


def deploy(source, state_path, models_dir, repository, image_path, tag=None,
           smoke=smoke_model, preview=False):
    with deployment_lock(state_path):
        state = load_state(state_path)
        if state.get('held') and not tag:
            return {'status': 'held', 'active': state['active']}
        release = source.select(tag)
        if release is None:
            return {'status': 'no-eligible-release', 'active': state['active']}
        rejected_path = Path(str(state_path) + '.rejected.json')
        if rejected_path.exists() and not tag:
            rejected = json.loads(rejected_path.read_text())
            if rejected.get('github_release_id') == release['id']:
                return {'status': 'release-rejected', 'active': state['active'],
                        'release_tag': release['tag_name']}
        active = state.get('active')
        if active and active.get('repository') == source.repository:
            if active.get('github_release_id') == release['id']:
                return {'status': 'already-active', 'active': active}
            if not tag and release_version(release['tag_name']) <= release_version(active['release_tag']):
                return {'status': 'no-newer-release', 'active': active}
        staging_root = Path(state_path).parent
        try:
            with tempfile.TemporaryDirectory(prefix='.web-release-', dir=staging_root) as work:
                metadata = source.download(release, work)
                if preview:
                    return {'status': 'validated-preview', 'release_tag': metadata['release_tag']}
                model_id, smoke_receipt = install_bundle(work, metadata, models_dir, repository,
                                                         image_path, smoke)
        except (ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            if not preview:
                atomic_json(rejected_path, {'github_release_id': release['id'],
                    'release_tag': release['tag_name'], 'at': now(), 'error_type': type(exc).__name__})
            raise
        active = {'model_id': model_id, 'release_tag': metadata['release_tag'],
                  'bundle_sha256': metadata['bundle']['sha256'],
                  'metadata_sha256': file_sha256(Path(models_dir).resolve() / 'web-releases' / model_id / 'web-release.json'),
                  'repository': source.repository, 'github_release_id': release['id'],
                  'registry': metadata['registry'], 'activated_at': now(), 'smoke': smoke_receipt}
        previous = state['active']
        history = [previous, *state.get('history', [])][:20]
        updated = {'format': STATE_FORMAT, 'active': active, 'history': history, 'held': False}
        atomic_json(state_path, updated)
        return {'status': 'activated', 'active': active, 'previous': previous}


def rollback(state_path, models_dir, repository, image_path, smoke=smoke_model):
    with deployment_lock(state_path):
        state = load_state(state_path)
        if not state.get('history'):
            raise ValueError('No previous deployment to restore')
        previous = state['history'][0]
        if previous:
            model_id = previous['model_id']
            if not re.fullmatch(r'web-[0-9a-f]{24}', model_id):
                raise ValueError('Invalid rollback model')
            root = Path(models_dir).resolve() / 'web-releases' / model_id
            bundle, metadata = verify_installed(root, previous['bundle_sha256'],
                expected_tag=previous['release_tag'], metadata_sha256=previous['metadata_sha256'])
            manifest = installed_manifest(bundle, root, model_id, metadata)
            smoke(manifest, image_path)
            if not repository.get_model_version(model_id):
                raise ValueError('Rollback model is absent from the catalog')
        updated = {'format': STATE_FORMAT, 'active': previous,
                   'history': [state['active'], *state['history'][1:]][:20], 'held': True}
        atomic_json(state_path, updated)
        return {'status': 'rolled-back', 'active': previous, 'held': True}


def resume(state_path):
    with deployment_lock(state_path):
        state = load_state(state_path)
        state['held'] = False
        atomic_json(state_path, state)
        return {'status': 'resumed', 'active': state['active']}

#!/usr/bin/env python3
"""Private operator/polling commands for released web models; no HTTP write API."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from backend.storage.db import DatabaseManager
from backend.storage.repository import ModelRegistryRepository
from backend.web_release import (GitHubReleaseSource, atomic_json, deploy, deployment_lock, load_state,
                                 now, resume, rollback)


def sync_mlflow(uri, state):
    """Mirror authoritative local state; caller holds the deployment lock."""
    base = uri.rstrip('/')
    owner = 'lizard-web-updater'
    active = state['active']

    def call(method, path, **kwargs):
        response = requests.request(method, base + '/api/2.0/mlflow/' + path,
                                    timeout=(5, 15), **kwargs)
        if response.status_code == 404:
            return None
        if response.status_code == 400:
            # MLflow 2.x also maps absent aliases/versions to HTTP400 with a
            # structured RESOURCE_DOES_NOT_EXIST code; don't swallow other400s.
            try:
                if response.json().get('error_code') == 'RESOURCE_DOES_NOT_EXIST':
                    return None
            except (ValueError, AttributeError):
                pass
        response.raise_for_status()
        return response.json()

    def identity(entry, allow_missing=False):
        registry = entry['registry']
        data = call('GET', 'model-versions/get', params={
            'name': registry['name'], 'version': registry['version']})
        if not data:
            if allow_missing:
                return None, {}
            raise ValueError('MLflow version is absent')
        version = data['model_version']
        tags = {tag['key']: tag['value'] for tag in version.get('tags', [])}
        if version.get('run_id') != registry['run_id'] or tags.get('bundle_sha256') != entry['bundle_sha256']:
            raise ValueError('MLflow source identity differs from deployed release')
        return registry, tags

    def set_tag(registry, key, value):
        result = call('POST', 'model-versions/set-tag', json={
            'name': registry['name'], 'version': registry['version'], 'key': key, 'value': value})
        if result is None:
            raise ValueError('MLflow version vanished')

    # Check the current target before removing any prior informational alias.
    target = identity(active)[0] if active else None
    seen = set()
    for previous in state.get('history', []):
        if not previous:
            continue
        registry = previous['registry']
        key = (registry['name'], registry['version'])
        if key in seen or (target and key == (target['name'], target['version'])):
            continue
        seen.add(key)
        registry, tags = identity(previous, allow_missing=True)
        if registry is None:
            continue
        if tags.get('web_deployment_owner') != owner:
            continue  # Never erase aliases belonging to another deployment tool.
        model = call('GET', 'registered-models/get', params={'name': registry['name']})
        aliases = {item['alias']: item['version'] for item in
                   (model or {}).get('registered_model', {}).get('aliases', [])}
        if aliases.get('web-production') == registry['version']:
            call('DELETE', 'registered-models/alias', json={
                'name': registry['name'], 'alias': 'web-production'})
        set_tag(registry, 'web_deployment_state', 'retired')
    if target:
        set_tag(target, 'web_deployment_owner', owner)
        result = call('POST', 'registered-models/alias', json={
            'name': target['name'], 'alias': 'web-production', 'version': target['version']})
        if result is None:
            raise ValueError('MLflow model vanished')
        set_tag(target, 'web_deployment_release', active['release_tag'])
        set_tag(target, 'web_deployment_state', 'deployed')
    return {'status': 'synced', 'alias': 'web-production' if active else None, 'at': now()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deploy', 'preview', 'rollback', 'resume', 'status'))
    parser.add_argument('--repo', default='Human-Augment-Analytics/Lizard_Toepads')
    parser.add_argument('--state-path', default=os.getenv('WEB_MODEL_STATE_PATH'))
    parser.add_argument('--models-dir', default=os.getenv('WEB_RELEASE_MODELS_DIR'))
    parser.add_argument('--db-path', default=os.getenv('WEB_RELEASE_DB_PATH'))
    parser.add_argument('--smoke-image', default=os.getenv('WEB_RELEASE_SMOKE_IMAGE'))
    parser.add_argument('--tag', help='Explicit released version; bypass a rollback hold deliberately')
    parser.add_argument('--mlflow-uri', default=os.getenv('WEB_RELEASE_MLFLOW_URI'))
    args = parser.parse_args()
    if not args.state_path:
        parser.error('--state-path or WEB_MODEL_STATE_PATH is required')
    if args.action == 'status':
        result = load_state(args.state_path)
    elif args.action == 'resume':
        result = resume(args.state_path)
    else:
        if args.action != 'preview':
            if not args.models_dir or not args.db_path or not args.smoke_image:
                parser.error('models directory, existing app DB, and local smoke image are required')
            if not Path(args.db_path).is_file() or not Path(args.models_dir).is_dir():
                parser.error('Use the existing app database and model directory; no new app DB is created')
            if not Path(args.smoke_image).is_file():
                parser.error('The local smoke image must exist; it is never published')
            repo = ModelRegistryRepository(DatabaseManager(args.db_path), models_dir=args.models_dir)
        else:
            repo = None
        source = GitHubReleaseSource(args.repo)
        if args.action == 'rollback':
            result = rollback(args.state_path, args.models_dir, repo, args.smoke_image)
        else:
            result = deploy(source, args.state_path, args.models_dir, repo, args.smoke_image,
                            tag=args.tag, preview=args.action == 'preview')
        if args.mlflow_uri and args.action != 'preview':
            with deployment_lock(args.state_path):
                # A manual rollback may have happened after deploy returned.
                # Re-read under the same lock so this process never publishes a stale alias.
                current = load_state(args.state_path)
                try:
                    result['registry_sync'] = sync_mlflow(args.mlflow_uri, current)
                except (requests.RequestException, ValueError, KeyError):
                    result['registry_sync'] = {'status': 'pending', 'reason': 'registry unavailable or identity mismatch'}
                atomic_json(str(args.state_path) + '.registry-sync.json', result['registry_sync'])
    if args.action not in ('status', 'preview'):
        atomic_json(str(args.state_path) + '.last-attempt.json', {'at': now(), **result})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, requests.RequestException) as exc:
        # No credentials or HTTP response bodies are included in operator logs.
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        sys.exit(1)

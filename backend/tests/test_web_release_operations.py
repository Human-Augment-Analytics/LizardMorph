"""Operator integration boundaries: alias cleanup and venv identity."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from backend.scripts import web_model_release as cli


def entry(name, version='1'):
    return {'model_id': 'web-' + 'a' * 24, 'release_tag': 'model/toepad-web/v1.0.0',
            'bundle_sha256': 'b' * 64, 'registry': {'name': name, 'version': version, 'run_id': 'c' * 32}}


def test_legacy_rollback_retires_only_owned_alias(monkeypatch):
    calls = []
    previous = entry('prior')
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith('/model-versions/get'):
            data = {'model_version': {'run_id': 'c' * 32, 'version': '1', 'tags': [
                {'key': 'bundle_sha256', 'value': 'b' * 64},
                {'key': 'web_deployment_owner', 'value': 'lizard-web-updater'}]}}
        elif method == 'GET' and url.endswith('/registered-models/get'):
            data = {'registered_model': {'aliases': [{'alias': 'web-production', 'version': '1'}]}}
        else:
            data = {}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(cli.requests, 'request', request)
    result = cli.sync_mlflow('http://registry', {'active': None, 'history': [previous], 'held': True})
    assert result['alias'] is None
    assert any(method == 'DELETE' and url.endswith('/registered-models/alias') for method, url, _ in calls)
    assert any(kw.get('json', {}).get('value') == 'retired' for _, _, kw in calls)


def test_cross_model_switch_removes_old_alias_and_sets_new(monkeypatch):
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith('/model-versions/get'):
            data = {'model_version': {'run_id': 'c' * 32, 'tags': [
                {'key': 'bundle_sha256', 'value': 'b' * 64},
                {'key': 'web_deployment_owner', 'value': 'lizard-web-updater'}]}}
        elif method == 'GET' and url.endswith('/registered-models/get'):
            data = {'registered_model': {'aliases': [{'alias': 'web-production', 'version': '1'}]}}
        else:
            data = {}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(cli.requests, 'request', request)
    cli.sync_mlflow('http://registry', {'active': entry('new'), 'history': [entry('old')]})
    deleted = next(kw['json']['name'] for method, url, kw in calls if method == 'DELETE')
    selected = next(kw['json']['name'] for method, url, kw in calls
                    if method == 'POST' and url.endswith('/registered-models/alias'))
    assert deleted == 'old' and selected == 'new'


def test_alias_is_not_changed_when_registry_identity_does_not_match(monkeypatch):
    methods = []
    def request(method, url, **kwargs):
        methods.append(method)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {'model_version': {'run_id': 'wrong', 'tags': []}})
    monkeypatch.setattr(cli.requests, 'request', request)
    with pytest.raises(ValueError, match='source identity'):
        cli.sync_mlflow('http://registry', {'active': entry('new'), 'history': [entry('old')]})
    assert methods == ['GET']


def test_mlflow_2x_missing_history_version_http400_does_not_block_sync(monkeypatch):
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url))
        if url.endswith('/model-versions/get') and kwargs.get('params', {}).get('name') == 'old':
            return SimpleNamespace(status_code=400, json=lambda: {
                'error_code': 'RESOURCE_DOES_NOT_EXIST'},
                raise_for_status=lambda: pytest.fail('known absent MLflow version should be ignored'))
        if url.endswith('/model-versions/get'):
            data = {'model_version': {'run_id': 'c' * 32, 'tags': [
                {'key': 'bundle_sha256', 'value': 'b' * 64},
                {'key': 'web_deployment_owner', 'value': 'lizard-web-updater'}]}}
        else:
            data = {}
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(cli.requests, 'request', request)
    result = cli.sync_mlflow('http://registry', {'active': entry('new'), 'history': [entry('old')]})
    assert result['status'] == 'synced'
    assert any(method == 'POST' and url.endswith('/registered-models/alias') for method, url in calls)


def test_timer_preserves_virtualenv_interpreter_symlink(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / 'deploy/web-model-release/install-user-timer.py'
    spec = importlib.util.spec_from_file_location('timer_installer', path)
    installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(installer)
    checkout = tmp_path / 'checkout'; script = checkout / 'backend/scripts/web_model_release.py'
    script.parent.mkdir(parents=True); script.write_text('# placeholder')
    python = tmp_path / 'venv/bin/python'; python.parent.mkdir(parents=True); python.symlink_to(sys.executable)
    models = tmp_path / 'models'; models.mkdir()
    db = tmp_path / 'app.db'; db.touch()
    image = tmp_path / 'smoke.jpg'; image.touch()
    units = tmp_path / 'units'
    monkeypatch.setattr(installer.os.path, 'expanduser',
                        lambda path: str(units) if path == '~/.config/systemd/user' else path)
    commands = []
    monkeypatch.setattr(installer.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    monkeypatch.setattr(sys, 'argv', ['installer', '--checkout', str(checkout), '--python', str(python),
        '--state-path', str(tmp_path / 'state.json'), '--models-dir', str(models),
        '--db-path', str(db), '--smoke-image', str(image)])
    installer.main()
    service = (units / 'lizard-web-model-update.service').read_text()
    assert 'ExecStart=' + str(python) in service
    assert commands == [['systemctl', '--user', 'daemon-reload']]

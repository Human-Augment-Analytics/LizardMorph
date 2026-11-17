#!/usr/bin/env python3
"""Install a user timer using explicit app paths; disabled unless --enable is set."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def quote(value):
    # systemd accepts shell-style quoting, but expands % specifiers separately.
    return shlex.quote(str(value).replace('%', '%%'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--state-path', required=True)
    parser.add_argument('--models-dir', required=True)
    parser.add_argument('--db-path', required=True)
    parser.add_argument('--smoke-image', required=True)
    parser.add_argument('--repo', default='Human-Augment-Analytics/Lizard_Toepads')
    parser.add_argument('--mlflow-uri', default='http://127.0.0.1:18600')
    parser.add_argument('--enable', action='store_true')
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    script = checkout / 'backend/scripts/web_model_release.py'
    for name in ('python', 'db_path', 'smoke_image'):
        if not Path(getattr(args, name)).is_file():
            parser.error(name.replace('_', '-') + ' must exist')
    if not script.is_file() or not Path(args.models_dir).is_dir():
        parser.error('Existing checkout and models directory required')
    # Preserve the venv executable path: resolving its final symlink would run
    # the base interpreter without that virtual environment's dependencies.
    command = [str(Path(args.python).expanduser().absolute()), str(script), 'deploy']
    for flag in ('state_path', 'models_dir', 'db_path', 'smoke_image', 'repo', 'mlflow_uri'):
        command += ['--' + flag.replace('_', '-'), getattr(args, flag)]
    units = Path(os.path.expanduser('~/.config/systemd/user'))
    units.mkdir(parents=True, exist_ok=True)
    (units / 'lizard-web-model-update.service').write_text(
        '[Unit]\nDescription=Pull validated LizardMorph web model releases\n\n'
        '[Service]\nType=oneshot\n'
        'WorkingDirectory=' + quote(checkout) + '\n'
        'Environment=ENABLE_ID_OCR=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n'
        'Environment=MODEL_DETECTOR_CACHE_LIMIT=1 MODEL_PREDICTOR_CACHE_LIMIT=4\n'
        'ExecStart=' + ' '.join(quote(part) for part in command) + '\n'
        'MemoryMax=1280M\nCPUQuota=100%\nTimeoutStartSec=600\nUMask=0077\n')
    (units / 'lizard-web-model-update.timer').write_text(
        '[Unit]\nDescription=Check for approved web model releases every five minutes\n\n'
        '[Timer]\nOnBootSec=2min\nOnUnitInactiveSec=5min\nRandomizedDelaySec=30\nPersistent=true\n\n'
        '[Install]\nWantedBy=timers.target\n')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    if args.enable:
        subprocess.run(['systemctl', '--user', 'enable', '--now', 'lizard-web-model-update.timer'], check=True)
    print('Installed user timer' + (' and enabled it.' if args.enable else '; not enabled.'))
    print('The existing app must use WEB_MODEL_STATE_PATH=' + args.state_path)
    print('Sessions retain their selected model; rollback holds future automatic updates.')


if __name__ == '__main__':
    main()

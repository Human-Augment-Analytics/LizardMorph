#!/usr/bin/env python3
"""Exercise actual registry import, named version selection and real-image inference.

Run in a container on the pilot network with requests installed. --image must be
a real supplied specimen; nothing trains, promotes aliases or changes defaults.
"""
import argparse
import json
import math
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://frontend')
    parser.add_argument('--image', required=True)
    parser.add_argument('--version', default='2')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    http = requests.Session()
    result = {}
    ids = []
    for name in ['toepad-landmarks-ml-morph', 'toepad-landmarks-albert-vit']:
        response = http.post(args.base + '/api/registry/import', json={'name': name, 'version': args.version}, timeout=300)
        response.raise_for_status()
        imported = response.json()['model']
        ids.append(imported['id'])
        duplicate = http.post(args.base + '/api/registry/import', json={'name': name, 'version': args.version}, timeout=300)
        duplicate.raise_for_status()
        assert duplicate.json()['model']['id'] == imported['id']
        session = http.post(args.base + '/api/session/start', timeout=30)
        session.raise_for_status()
        sid = session.json()['session_id']
        with open(args.image, 'rb') as image:
            prediction = http.post(args.base + '/api/data', headers={'X-Session-ID': sid},
                data={'view_type': 'custom', 'model_id': imported['id']},
                files={'image': (Path(args.image).name, image, 'image/jpeg')}, timeout=600)
        prediction.raise_for_status()
        data = prediction.json()[0]
        assert not data.get('error'), data
        assert len(data['coords']) == 18, data
        assert len(data['bounding_boxes']) == 2, data
        assert all(math.isfinite(point[axis]) for point in data['coords'] for axis in ['x','y'])
        assert data['model_provenance']['model_id'] == imported['id']
        assert data['model_provenance']['registry']['version'] == args.version
        result[name] = {'import': imported, 'session_id': sid, 'prediction': data}
        print(name, 'version', args.version, 'boxes', len(data['bounding_boxes']), 'landmarks', len(data['coords']), flush=True)
    assert ids[0] != ids[1]
    # Earlier immutable version remains usable after both newer versions loaded.
    previous = http.post(args.base + '/api/registry/import', json={'name':'toepad-landmarks-ml-morph','version':'1'}, timeout=300)
    previous.raise_for_status()
    previous_id = previous.json()['model']['id']
    assert previous_id not in ids
    with open(args.image, 'rb') as image:
        rollback = http.post(args.base + '/api/predict', data={'model_id': previous_id},
            files={'image': (Path(args.image).name, image, 'image/jpeg')}, timeout=600)
    rollback.raise_for_status()
    assert sum(len(obj['landmarks']) for obj in rollback.json()['predictions']) == 18
    assert rollback.json()['model_provenance']['version'] == '1'
    result['previous_version'] = rollback.json()
    # No aliases were promoted by registration, import, inference or rollback.
    versions = http.get(args.base + '/model-api/models', timeout=30)
    versions.raise_for_status()
    assert all(not v['aliases'] for v in versions.json()['models'])
    result['registry_versions'] = versions.json()['models']
    Path(args.output).write_text(json.dumps(result, indent=2))
    print('Previous version usable; aliases unchanged; evidence', args.output, flush=True)


if __name__ == '__main__':
    main()

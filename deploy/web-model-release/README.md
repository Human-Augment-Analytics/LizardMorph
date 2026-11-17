# Release a model to the existing web application

The producer repository stages an explicit MLflow version as a GitHub draft
release. Its manual **Release model to web** workflow validates and publishes
`model/toepad-web/vMAJOR.MINOR.PATCH`. This app polls only that release channel.
See `Lizard_Toepads/docs/WEB_MODEL_RELEASES.md` for the operator staging command
and evaluation-report format. No release or scientific approval is fabricated
by this implementation.

Each release contains `bundle.zip`, `web-release.json`, and `evaluation.json`.
The producer validator is vendored byte-for-byte as
`backend/web_release_contract.py`; update both copies together. The bundle must
declare the complete four-digit H7 dual-pass ONNX/dlib contract, nine landmarks
per digit, web deployment eligibility, and held-out evaluation provenance. A
hash-pinned report names a reviewer, benchmark, dataset/source identity, metrics,
and passing numeric criteria. ViT and the earlier bottom-only research demos are
rejected before any model loader runs. A GitHub release does not by itself
prove scientific suitability; the named reviewer remains accountable for the
evaluation and benchmark choice.

## Configure Haag after deploying this app revision

Use the current app's actual database and model directory, not a new database.
Set the same absolute `WEB_MODEL_STATE_PATH` in the backend environment and the
updater. For example, `/var/www/LizardMorph/models/web-production/state.json`.
Restart the existing backend after setting its environment. Without that setting,
the current app/default remains unchanged. Configured web deployment defaults
bound the long-lived detector cache to one entry and predictor cache to four;
explicit cache environment settings take precedence.

Preview a published candidate without changing the catalog or active pointer:

```sh
backend/.venv/bin/python backend/scripts/web_model_release.py preview \
  --state-path /absolute/models/web-production/state.json \
  --tag model/toepad-web/v1.3.0
```

Deploy with explicit local paths and a permitted smoke photograph. The photograph
stays on Haag and is never included in GitHub release assets:

```sh
backend/.venv/bin/python backend/scripts/web_model_release.py deploy \
  --state-path /absolute/models/web-production/state.json \
  --models-dir /absolute/models --db-path /absolute/existing-app.db \
  --smoke-image /absolute/private-smoke.jpg \
  --mlflow-uri http://127.0.0.1:18600
```

The smoke image must exercise bottom finger and bottom toe. Before actual image
inference, every declared dlib predictor is loaded and checked against its
landmark schema, including upper-digit predictors absent from the photograph.
The detector must expose the supported H7 decoder shape. Failures retain the
previous active pointer and do not register an active replacement.

Install the five-minute user timer using the reviewed checkout and existing
virtualenv. Omit `--enable` to prepare it without starting polling:

```sh
python3 deploy/web-model-release/install-user-timer.py \
  --checkout /absolute/LizardMorph \
  --python /absolute/LizardMorph/backend/.venv/bin/python \
  --state-path /absolute/models/web-production/state.json \
  --models-dir /absolute/models --db-path /absolute/existing-app.db \
  --smoke-image /absolute/private-smoke.jpg --enable
```

The unit is limited to 1280 MiB, one CPU, and ten minutes per attempt. Ensure the
host's user systemd manager survives logout before relying on unattended checks.
It requires only outbound public GitHub reads; no GitHub-hosted runner needs SSH
access to Haag. It scans the latest 100 GitHub release records, ignores drafts,
prereleases and other channels, and chooses the highest valid semantic version.
Enable GitHub immutable releases when available; do not replace released tags or
assets. Downloads and activation use exact artifact hashes, not mutable `latest`
asset links. The legacy desktop packaging workflow is unchanged.

## Activation, existing analyses, and rollback

Installation creates immutable `web-<bundle digest>` catalog entries and keeps
the exact archive, manifests, report, and extracted weights together. After all
checks, an fsynced atomic state-file replacement changes the default only for
new toepad sessions. The explicit model selected for an image or session remains
pinned; reprocessing an old image retains its recorded provenance. X-ray, custom
models, and the legacy scale flow remain independent. Malformed configured state
fails closed for new default toepad inference.

Use the same path arguments with the `rollback` action to restore the preceding
installed version (or legacy behavior). Rollback validates retained bytes again
and sets a hold, so the next scheduled poll cannot immediately redeploy the
version you rejected. `resume --state-path ...` clears that hold. An explicit
`deploy --tag ...` deliberately retries a named version. Failed validation/smoke
attempts are recorded in `state.json.rejected.json` and the same release ID is
not automatically loaded every five minutes; use an explicit tag to retry after
correcting the cause, or publish a new version.

`status --state-path ...` reports the local truth. The updater mirrors successful
activation to MLflow's `web-production` alias with lightweight REST calls after
checking the registered run and bundle digest. It retires its own prior aliases
on cross-model/legacy rollback. Registry unavailability leaves an informational
sync pending and never breaks installed-model inference; later polls retry even
while deployment is held. The local deployment receipt is authoritative.

Old installed bundles remain on disk for pinned sessions and rollback. There is
no automatic deletion policy. Monitor storage and back up state, catalog, and
model directories consistently. Do not remove a version still referenced by an
analysis. No public deployment/rollback management endpoint is added.

## Validation

Run database-backed tests in isolated Docker on the configured Mac mini against
this exact source snapshot. Targeted checks include release verification, failed
smoke pointer preservation, tampered rollback rejection, rollback holds, session
cookie/new-session behavior, explicit-version pinning, and producer contract
parity. A first real production release still requires a complete qualifying
bundle, its approved evaluation report, and a locally configured smoke image.

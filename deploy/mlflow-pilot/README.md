# Two-model MLflow research pilot

This isolated, loopback-only stack imports complete **named, versioned pipeline
bundles** into AutoMorph's existing model catalog. It does not replace the default
YOLO file. Choose a version in Research models → Import → Use this version.
Existing analyses retain their model ID. Export ZIPs include an image-specific
`.model.json` with the exact registry version, bundle SHA256, input image hash,
manifest and provenance. Registration/import never moves an alias or default.

## Scope

The pilot uses the existing H7 six-class detector with two alternative landmark
predictors: ML-Morph/dlib rectify-512, and Albert Xu's original trained ViT-small
checkpoint exported to ONNX. Only **bottom finger and toe** are exposed (18
landmarks total). It does not add upper-digit, ruler or ID inference. No held-out
accuracy comparison is claimed. The H7 OBB envelope used for the ViT AABB crop is
not the original training notebook's H5 axis-aligned detector; evaluate this
change before scientific use. This is not Albert's October DINO checkpoint.

Albert source: Lizard_Toepads `493ea405e3214a992fcf79c406e6fe1a3d7f6d97`,
`stacked-hourglass/vit/{best_vit_landmark.pth,vit_training_tst.ipynb,vit_preprocessing_tst.ipynb}`.
The ViT checkpoint SHA256 is
`62d78f637cf429ad83392732404b51ec68873c3caa85e1fac399d912f5d6d28d`.
Its contract is preserved explicitly: unrotated AABB, longest side 224, centered
black letterbox, **BGR** ImageNet normalization, nine coordinates normalized by224.
Production inference uses CPU ONNX/dlib, with no torch/timm download or executable
checkpoint loading. Export uses pinned torch2.5.1/timm1.0.12 and weights_only=True.

## Start and stop

From the LizardMorph repository root:

```sh
docker compose -f deploy/mlflow-pilot/compose.yaml up -d --build
docker compose -f deploy/mlflow-pilot/compose.yaml ps
docker compose -f deploy/mlflow-pilot/compose.yaml logs --tail=100
# Stop while preserving models, registry, sessions and app DB:
docker compose -f deploy/mlflow-pilot/compose.yaml stop
```

Private endpoints: app http://127.0.0.1:18603/#/models; MLflow18600; model-api18601;
Flask18602. They are local to the Docker host; use SSH forwarding from your laptop.
This is a single-worker research demo. Its inference caches retain at most one detector and two landmark predictors to fit model switching into the host memory budget. Restrict access to the local machine/SSH;
there is no authentication layer for public serving. Restart policies recover
services after Docker restarts; stopping a service intentionally keeps it stopped.

The MLflow server serves artifacts from its persistent volume. The older
metadata-only demo on18500–18503 is independent and remains unchanged.

## Prepare and register actual model bundles

In a Lizard_Toepads checkout containing `scripts/deployment/` pilot scripts:

1. Supply the pinned ViT checkpoint and existing `yolo_obb_6class_h7_int8.onnx`
   and `ml_morph_best.dat` in a private fixture directory. Retain their hashes.
2. Build `deploy/mlflow-pilot/Dockerfile.export` as an export-only image. Run
   `export_albert_vit.py --checkpoint ... --output .../albert_vit.onnx` with that
   directory mounted; no network is needed for export. Keep the `.onnx.json`
   parity receipt alongside the exported file.
3. Run `prepare_switching_pilot.py --fixtures DIR --output BUNDLES` to produce
   `ml-morph` and `albert-vit` bundle directories.
4. With MLflow2.22.4 installed, run:

```sh
python scripts/deployment/register_bundle.py --bundle-dir BUNDLES/ml-morph \
  --name toepad-landmarks-ml-morph --tracking-uri http://127.0.0.1:18600
python scripts/deployment/register_bundle.py --bundle-dir BUNDLES/albert-vit \
  --name toepad-landmarks-albert-vit --tracking-uri http://127.0.0.1:18600
```

From a container on the Compose network the tracking URI is `http://mlflow:5000`.
Each invocation intentionally creates a new run/version. Do not re-register just
to start the stack. Bundles are deterministic ZIPs containing `bundle.json`, all
referenced detector/predictor files, checksums and source/evaluation provenance.
A version's archive digest is stored in MLflow and checked again during download.
The backend validates paths, limits, checksums, model runtimes and landmark count
before publishing a selectable model. Every import gets an immutable ID/directory.

## Switching and rollback

Import a specific numeric version from Research models. Open its model card and
upload images. Both pipelines can be installed simultaneously. Switch by opening
the other model/version; this is independent of MLflow alias changes. To roll back,
open the previous installed version. No global hot replacement or automatic
accuracy-based promotion is implemented. Alias/default policy remains manual.

## Checks and backup

All database-backed tests must run in Docker on the Mac mini against the reviewed
checkout. `Dockerfile.tests` adds training dependencies to the inference image;
run `python -m pytest backend/tests` with repository root and backend on PYTHONPATH.
Model-api tests need pytest in its locked Python environment. Frontend checks are
`npm test`, `npm run lint`, `npm run build`. Include real registration/import and
image inference when checking artifact transport; unit tests alone are insufficient.

Before backup, stop this Compose project, then archive all four named volumes
(`registry`, `models`, `work`, `sessions`) together using a disposable container;
retain code revision and image IDs. Restart afterwards. Stopping the project first
keeps SQLite and files consistent. Restore into a separate Compose project and
verify the installed bundle hashes before using a backup. Never use `down -v`
unless intentionally discarding all pilot data.

## Extending to newer Albert models

A new family needs its actual checkpoint plus architecture/config, preprocessing,
landmark order and decoding, auxiliary artifacts, a parity check and comparable
held-out evaluation. The current ONNX coordinate adapter recognizes only
`albert-vit-aabb-224-bgr-v1`. Other output families fail closed until an adapter is
implemented. Keep the same bundle/import/catalog mechanism for those adapters.

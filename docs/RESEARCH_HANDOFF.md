# Researcher handoff

This guide covers AutoMorph's source application, generic YOLO OBB + dlib training,
and annotation exports. Review baseline: main commit
`8b989a05c55bf48198199ad95c007e8198f0270c` (version 1.2.0), September 2026.
Use the commit containing this guide for the fixes described below.

## Start and verify

1. Follow [README setup](../README.md), using Node from `.nvmrc` and a dedicated
   Python 3.10 environment (the setup default). `make setup-backend` recreates `backend/.venv`; do not run
   it over an environment you need to preserve. Python dependencies are only
   partially pinned; archive the resolved environment for each experiment.
2. Obtain the models from the project maintainer or the SSH download script.
   Access to the Georgia Tech server is required for `make download-models`.
   Model files and research datasets are not part of a clean Git checkout.
3. Copy `.env.example` to `.env`, set explicit model paths, and run
   `PYTHONPATH=backend backend/.venv/bin/python backend/scripts/check_env.py`.
4. Install `backend/requirements-dev.txt`, then run `make test`, `make lint`,
   and `npm --prefix frontend run build`. These validate software behavior;
   test doubles and synthetic images do not measure biological accuracy.
5. Start `make dev`. Check a representative image in each modality you intend
   to use. Verify landmark count/order, orientation, IDs, scale, and exported
   coordinates before collecting measurements. Use [the tutorial](../tutorial.md)
   for custom training. Back up session exports before removing app data.

## Interpreting results

- Detector `mAP50` and `mAP50-95` are internal validation metrics from the detector
  training run. A recovered checkpoint currently reports the final CSV row;
  that row need not describe the selected `best.pt` checkpoint. Evaluate the
  actual archived checkpoint independently before reporting its performance.
- Landmark `test_error` comes from dlib on ground-truth-derived, padded and
  jittered crops. It is in crop pixels and excludes detector misses and box
  errors. The combined value is an unweighted mean of available class errors,
  not a pooled specimen metric. `null` means no evaluation was available.
- Landmark crops from a source image now stay together in the internal split.
  The adjacent `train_landmarks.xml.groups.json` records their source paths;
  retain it with the XML. A class with only one source group has no holdout.
  Old generated XML without that sidecar can only group by its image filenames.
- Splitting uses seed 42, but the detector and per-class landmark partitions
  are separate. Related specimens across images, copied images at different
  paths, acquisition batches, and train/validation overlap across pipeline stages
  are not automatically controlled. Internal metrics are development diagnostics.
- A zero landmark test fraction disables its holdout; the detector still reserves
  at least one image for validation. Training seeds, hardware, library versions,
  and stochastic augmentation can affect results; bitwise reproducibility is
  not guaranteed.
- `mock` results are test fixtures, never scientific measurements. Successful
  training and model loading alone do not establish predictive accuracy.

## Study-level validation

Create an external test partition before training, grouping all images, crops,
repeated measurements, and augmentations from the same specimen together.
Keep that partition out of model selection and parameter tuning. Document
species, imaging equipment, acquisition dates/batches, inclusion/exclusion
criteria, annotation protocol, and class/landmark definitions. Check duplicate
images and specimen identifiers explicitly.

Evaluate the complete detector-to-landmark pipeline on the untouched partition.
Report sample and specimen counts, per-class detection failures, missing landmarks,
per-landmark error distributions, and uncertainty with specimens as the independent
unit. Define pixel or physical units and scale calibration. Record exclusions and
failed predictions in the denominator; do not score only successful detections.
Evaluate manual annotation agreement where it affects the study. Separate raw
predictions from corrections and record who corrected what.

## Reproducibility record

Archive these alongside every result:

- Source commit (`git rev-parse HEAD`) and any working-tree patch (`git diff`).
- Python version and `uv pip freeze --python backend/.venv/bin/python` output;
  Node version, committed npm lockfiles, OS, CPU/GPU, and inference provider.
- Input and annotation SHA-256 checksums, specimen split membership, data source,
  permissions, and exclusion log. Hash the actual model files as well.
- Explicit model paths and preprocessing settings, confidence thresholds, class
  mapping, landmark order, units, crop padding, seeds, and training options.
  Do not archive secrets from `.env` in a public supplement.
- Complete `runs/job_<id>/`: `job_config.json`, source images, generated datasets
  and split XML/sidecar, `training.log`, `status.json`, `manifest.json`, detector
  checkpoints, and landmark predictors. Source paths can be absolute; relocating
  a job may require updating paths before rerunning it.
- Raw predictions, reviewed exports, evaluation code, and final tables/figures.
  Session storage is temporary working data, not your research archive.

The built-in model weights do not have a committed provenance/checksum catalogue.
Obtain their training data provenance and licensing from the maintainers before
redistributing or making scientific claims. The repository has no top-level
license file at this review baseline; third-party notices do not establish a
license for the entire project. Record the AutoMorph commit and the ML-morph
reference in the README when describing your method.

## Review changes and remaining acceptance

This handoff fixes source-image leakage in the landmark holdout, default YOLO
model lookup, and loss of the native detector checkpoint during recovery. It
also repairs backend test collection/configuration isolation and excludes the
vendored ONNX Runtime copy from application lint.

Before a publication or distribution, the research owner still needs to approve
an external specimen-level accuracy evaluation, model/data provenance and rights,
and a clean-machine setup plus packaged-app smoke test on the target platform.
No numerical accuracy certification is implied by this code review. Historical
notes under `docs/superpowers/` and the older prediction/training handoff are
engineering history, not current validation evidence.

## Local verification record

The handoff changes passed 157 backend tests, 25 frontend tests, frontend ESLint,
and the TypeScript/Vite production build. The backend import check reported all
14 imports available. This local run used Python 3.12.10, Torch 2.9.0,
Ultralytics 8.3.235, and ONNX Runtime 1.23.2 in an existing environment, so it is
not evidence that the dependency file reproduces that exact environment.
The new CI workflow checks a Python 3.10 installation and Node from `.nvmrc`;
its hosted result must be checked after pushing the branch.

# Anolis Morphometrics: small-data recovery (2026-09-23)

## Selected approach

Use **Anolis Morphometrics - Pretrained Transfer**, model ID
`anolis-pretrained-transfer-20260923`, in the existing Anolis project. This version
reuses the H7 ONNX detector and `ml_morph_best.dat`; it is not a new model learned
from the demo images. The original `e05a5515cc35` model is preserved.

The manifest explicitly selects `toepad-dual-pass` detection and
`toepad-rectify-512` landmarks. These protocols now survive registering a model
under a different ID or renaming its artifacts. Older manifests default to the
existing generic affine pipeline; the built-in model retains backward compatibility.
Upper-digit corners and landmarks are converted between flipped and original
image coordinates. Engine output also serializes OBB corners to ordinary lists,
fixing the real detector's `/api/predict` JSON serialization failure.

## Why the five-image training run failed

- The five uploaded images include byte-identical `1866` and `1866b`, leaving
  four unique scans.
- The detector trained on four files, in one batch per epoch. Fifty epochs meant
  only 50 steps. Installed Ultralytics uses at least 100 warm-up iterations when
  warm-up is enabled, so this run ended during warm-up.
- Validation recall was 0.25. On 1910 the model returned three `bot_toe` detections
  rather than four correctly classified digits, despite 1910 being in training.
- Replacing only the detector is insufficient: the custom landmark models were
  fitted to very tight annotation boxes, whereas the established detector gives
  different crop geometry. A direct swap still produced about 10 px error for
  the two lower digits on 1910 and its single-pass path missed the upper digits.

## Bounded comparison

The reproducible script `scripts/evaluate-anolis-small-data.py` samples 16 unique
TPS-annotated scans for training (32 lower toe/finger crops) and six separate scans
for validation. It keeps the detector frozen and trains two dlib predictors on
its perspective-rectified 512 crops. Nine landmark identities, TPS bottom-origin
coordinates, color channel order, padding, and inverse projection are preserved.
The original four unique demo scans are a separate diagnostic set. 1910 is not
used in this new fitting run. No missing detections are silently dropped from
reported object counts.

| Check | Pretrained | New 16-scan landmark models |
|---|---:|---:|
| Six validation scans: lower digits detected | 12/12 | 12/12 |
| Validation mean landmark error | 0.97 px | 1.76 px |
| Four diagnostic scans: digits detected | 16/16 | 16/16 |
| Diagnostic mean landmark error | 1.94 px | 2.29 px |
| 1910, all four digits, mean landmark error | 1.21 px | 2.13 px |

These are floating-point predictions in images resized to width 1600. XML's
integer truncation raises the selected model's 1910 error to 1.62 px. The six
validation scans are excluded from the *new* fit; historical overlap with the
pretrained models' original training pool is unknown. Only lower digits have
TPS reference landmarks on these six scans. This is a small operational
comparison, not a claim of independent, population-wide accuracy.

The pretrained combination wins this comparison, so it is the registered choice.
1841's lower toe still has about 10.93 px mean error; review difficult cases rather
than treating all results as equally reliable. More training is not automatically
an improvement. If further adaptation becomes necessary, collect corrections on
such failures and retain a specimen-separated validation set.

## Reproduce

```sh
/Users/leyangloh/miniconda3/envs/lizard/bin/python \
  scripts/evaluate-anolis-small-data.py \
  --source /Users/leyangloh/dev/Lizard_Toepads
```

Artifacts are under `artifacts/anolis-small-data/`: `split.json`, both fitted
predictors, `evaluation.json`, `selected-manifest.json`, and `1910-api.json`.
The registry was backed up to `registry-before.sqlite3` before adding the version.
The local development backend was restarted to apply the inference changes;
this does not rebuild the installed macOS application.

## Verification

- 32 focused tests pass: inference, protocol round-trip/renamed artifacts,
  coordinate conversion, JSON serialization, registry, API parity, and ORT.
- The JSON regression failed before the corner serialization correction and
  passes afterward.
- The full backend suite cannot be reported green: combined collection imports
  both `app` and `backend.app`, duplicating Prometheus metrics. Separating that
  module exposes environment/app-stub isolation failures (404s) in the larger
  combined run. The focused tests above pass in an isolated run.
- Live `/api/predict` is exercised with the selected registered model and 1910.

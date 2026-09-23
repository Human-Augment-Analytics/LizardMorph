# Pretrained toepad inference correction

The bundled `ml_morph_best.dat` now uses its training preprocessing: ordered OBB
corners, perspective rectification, a centered BGR 512×512 letterbox, the full
512×512 dlib rectangle, and inverse projection into image coordinates.

The server detection path, browser-supplied OBB path, and built-in
`lizard-toepad-v1` generic API path share the correction. Upper digit corners
are converted into the flipped image coordinate system before cropping and
predictions are converted back afterward. Legacy predictor filenames retain
axis-aligned preprocessing in the legacy routes. Custom model bundles retain
the generic trainer's affine crop and schema ordering. Protocol selection uses
the bundled filename `ml_morph_best.dat`; renamed copies are not auto-detected.

## Verification

`toepad_real_1004.jpg`, 1600×765, H7 ONNX detector, all 36 digit landmarks matched
by class and within-class landmark index against the supplied annotations XML:

| Digit | Old exported mean error (px) | Corrected exported mean error (px) |
|---|---:|---:|
| bot_finger | 3.03 | 1.07 |
| bot_toe | 7.85 | 1.62 |
| up_finger | 10.17 | 1.79 |
| up_toe | 4.76 | 2.41 |
| All | 6.45 | 1.72 |

The unrounded corrected predictions average 1.39 px. XML uses integer
truncation, so exported-coordinate errors differ. These are resized-image
pixels, not full-resolution scan pixels, and a single-image check is not a
held-out accuracy estimate. Additional image 1841 produces 4.55 px mean exported
error, including 10.39 px for bot_toe; the correction does not eliminate every
prediction error.

Regression tests cover training canvas/color convention, inverse projection,
legacy crop compatibility, invalid boxes, server/client flip parity, and the
built-in API engine. Run from the worktree root:

```sh
PYTHONPATH=backend python -m pytest backend/tests/test_toepad_rectify.py backend/tests/test_inference_engine.py backend/tests/test_parity.py backend/tests/test_ort_inference.py backend/tests/test_datasets.py backend/tests/test_desktop_packaging.py -q
```

This source change does not update the installed desktop sidecar; a rebuilt
application is required for `/Applications/AutoMorph.app` to use it.

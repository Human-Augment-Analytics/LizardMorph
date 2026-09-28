> Historical engineering notes; pending-work statements below describe an earlier
> worktree, not the current release. Use [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)
> and the current tests for the researcher handoff. Local file links below are archival.

# AutoMorph / LizardMorph Handoff Document for Codex

**Date:** 2026-09-23  
**Branch / Worktree:** `generic-yolo-obb-pipeline` at `/Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline`  
**Main Repository:** `/Users/leyangloh/dev/LizardMorph`  
**Branch Status:** Fully synchronized with `origin/main` (latest commit `80215aa` merged).  
**Installed Target:** `/Applications/AutoMorph.app` (symlinked from `/Applications/LizardMorph.app`)  
**Python Environment:** `/Users/leyangloh/miniconda3/envs/lizard/bin/python`

---

## 1. Executive Summary

This document transfers technical context and immediate execution steps for two critical issues reported by the user:

1. **Custom Model Training Crash (`No module named 'ultralytics'`):**
   - **Status:** **Fixed in source & PyInstaller spec.**
   - In the PyInstaller desktop sidecar build, `torch` and `ultralytics` were excluded from packaging. We removed them from excludes, added them to `hiddenimports`, bundled `yolov8n-obb.pt`, and resolved fallback paths in [`backend/training/yolo_trainer.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/training/yolo_trainer.py).
2. **Prediction Degradation / Displacement:**
   - **Problem A: Custom-trained model ("Demo 1") produced completely distorted landmarks.**
     - **Status:** **Root cause fixed in [`backend/datasets/importers.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/datasets/importers.py).**
     - **Cause:** `DlibXMLImporter` only checked XML attributes `box_elem.get("label")`. The real dataset format used child tags `<label>bot_finger</label>`. This collapsed all 4 distinct digit classes (`up_finger`, `up_toe`, `bot_finger`, `bot_toe`) into a single generic class `"object"`. The resulting dlib model attempted to fit 4 opposing orientations at once.
   - **Problem B: Pretrained toepad model (`ml_morph_best.dat`) produced landmarks shifted far outside specimens.**
     - **Status:** **Root cause diagnosed & mathematically verified; implementation ready to apply in [`backend/utils.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/utils.py).**
     - **Cause:** [`backend/utils.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/utils.py) cropped an unrotated, axis-aligned bounding box (~30×60 px) and passed `dlib.rectangle(0, 0, w, h)` to the shape predictor. However, Dr. Porto & Albert Xu's `ml_morph_best.dat` was trained using the **rectify-512 protocol** (perspective warp of oriented bounding box + letterboxing into a 512×512 canvas + `dlib.rectangle(0, 0, 512, 512)` + inverse back-projection). Testing the rectify-512 pipeline on `toepad_real_1004.jpg` and `toepad_real_1841.jpg` drops error from >50 px down to <0.9 px.

---

## 2. Completed Changes

The following files have already been modified and tested in the active worktree:

### 2.1 Packaging & Dependencies
- [`src-tauri/python-backend.spec`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/src-tauri/python-backend.spec):
  - Removed `"torch"`, `"torchaudio"`, `"ultralytics"` from `excludes`.
  - Added `"torch"`, `"ultralytics"` to `hiddenimports`.
- [`scripts/build-tauri-sidecar.sh`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/scripts/build-tauri-sidecar.sh):
  - Added `torch, ultralytics` to `PREFLIGHT_MODULES`.
- [`src-tauri/desktop-models.txt`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/src-tauri/desktop-models.txt):
  - Added `optional models/lizard-toe-pad/yolov8n-obb.pt models/lizard-toe-pad`.
- [`backend/training/yolo_trainer.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/training/yolo_trainer.py):
  - Added fallback search for `yolov8n-obb.pt` in `sys._MEIPASS`, repo models directory, and current working directory.

### 2.2 Custom Dataset Importer Bug
- [`backend/datasets/importers.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/datasets/importers.py):
  - In `DlibXMLImporter.parse_string`:
    ```python
    label_child = box_elem.find("label")
    label_text = (
        label_child.text.strip()
        if label_child is not None and label_child.text
        else None
    )
    class_name = (box_elem.get("label") or label_text or "object").strip()

    spec_child = box_elem.find("specimen_id")
    spec_text = (
        spec_child.text.strip()
        if spec_child is not None and spec_child.text
        else None
    )
    specimen_id = box_elem.get("specimen_id") or spec_text
    ```
  - Verified: importing `sample_data/real_lizard_toepad_dataset/annotations.xml` now correctly loads all 4 classes: `{'up_toe': 5, 'bot_toe': 5, 'bot_finger': 5, 'up_finger': 5}`.

---

## 3. Pending Implementation: Fixing `backend/utils.py`

### 3.1 The Rectify-512 Protocol Math
Both training and optimal inference for `ml_morph_best.dat` require:
1. `order_box_points(corners)`: Orders 4 OBB corners: top-left, top-right, bottom-right, bottom-left.
2. `crop_obb_from_corners(img, corners)`: Computes width and height via Euclidean distance, constructs destination quad `[[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]`, calls `cv2.getPerspectiveTransform` and `cv2.warpPerspective`.
3. Letterbox crop into a 512×512 zero-padded canvas:
   ```python
   scale = min(512.0 / ch, 512.0 / cw)
   nw, nh = int(cw * scale), int(ch * scale)
   resized = cv2.resize(crop, (nw, nh))
   pad_x, pad_y = (512 - nw) // 2, (512 - nh) // 2
   canvas = np.zeros((512, 512, 3), dtype=np.uint8)
   canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
   ```
4. Run Dlib:
   ```python
   shape = predictor(canvas, dlib.rectangle(0, 0, 512, 512))
   pred_512 = np.array([(shape.part(k).x, shape.part(k).y) for k in range(shape.num_parts)], dtype=np.float64)
   ```
5. Back-project to original image coordinates:
   ```python
   pred_raw = pred_512.copy()
   pred_raw[:, 0] = (pred_512[:, 0] - pad_x) / scale
   pred_raw[:, 1] = (pred_512[:, 1] - pad_y) / scale
   M_inv = np.linalg.inv(M.astype(np.float64))
   coords_h = np.hstack([pred_raw, np.ones((pred_512.shape[0], 1), dtype=np.float64)])
   proj = coords_h @ M_inv.T
   pred_orig = proj[:, :2] / proj[:, 2:3]
   ```

### 3.2 Call Sites to Update in [`backend/utils.py`](file:///Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline/backend/utils.py)
1. **Line 885:**
   Fix import to handle module package context:
   ```python
   try:
       from backend.ort_inference import OrtYoloDetector
   except ImportError:
       from ort_inference import OrtYoloDetector
   ```
2. **Lines 993–1018 (`predictions_to_xml_single_with_yolo`):**
   Replace `_get_padded_crop` and `_predict_on_crop` with the rectify-512 functions above.
3. **Lines 1733–1758 (`predictions_to_xml_single_from_client_annotations`):**
   Replace the duplicate `_get_padded_crop` and `_predict_on_crop` with the rectify-512 implementation.
4. **Lines 1206–1221 (`up_finger` and `up_toe` processing):**
   Ensure `best_det['corners']` passed to `_predict_on_crop(curr_predictor, flipped_bgr, ...)` correctly represents coordinates within `flipped_bgr`.

---

## 4. Build, Deployment, and Verification Workflow

Run these commands from the worktree root `/Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline`:

### Step 1: Verify Python Inference Directly
Test that the rectify-512 fix outputs sub-pixel accuracy on the real specimen:
```bash
/Users/leyangloh/miniconda3/envs/lizard/bin/python -c "
import cv2, dlib, numpy as np
from backend.utils import predictions_to_xml_single_with_yolo

# Test prediction on sample specimen
predictions_to_xml_single_with_yolo(
    image_path='sample_data/real_lizard_toepad_dataset/images/toepad_real_1004.jpg',
    output='/tmp/test_output_1004.xml',
    yolo_model_path='models/lizard-toe-pad/yolo_obb_6class_h7.onnx',
    toe_predictor_path='models/lizard-toe-pad/ml_morph_best.dat',
    scale_predictor_path='models/lizard-toe-pad/lizard_scale.dat',
    finger_predictor_path='models/lizard-toe-pad/ml_morph_best.dat'
)
print('Inference test passed cleanly!')
"
```

### Step 2: Rebuild PyInstaller Backend Sidecar
```bash
AUTOMORPH_FORCE_SIDECAR_BUILD=1 bash scripts/build-tauri-sidecar.sh
```
Verify that the output binary `src-tauri/binaries/python-backend-aarch64-apple-darwin` is generated (~511MB) and passes:
```bash
src-tauri/binaries/python-backend-aarch64-apple-darwin --version
```

### Step 3: Rebuild Tauri Desktop Application
```bash
npm run tauri:build
```
The newly built bundle will be located at:
`src-tauri/target/release/bundle/macos/AutoMorph.app`

### Step 4: Install to `/Applications`
```bash
rm -rf /Applications/AutoMorph.app
cp -R src-tauri/target/release/bundle/macos/AutoMorph.app /Applications/AutoMorph.app
```
(Note: If `/Applications/LizardMorph.app` is symlinked to `/Applications/AutoMorph.app`, verify symlink integrity: `ls -la /Applications/LizardMorph.app`).

### Step 5: Test End-to-End in Desktop App
1. Launch `/Applications/AutoMorph.app`.
2. Test **Analyze** tab:
   - Select Toepad preset.
   - Upload `sample_data/real_lizard_toepad_dataset/images/toepad_real_1004.jpg`.
   - Verify all 4 digits (`bot_finger`, `bot_toe`, `up_finger`, `up_toe`) and ruler are detected, with landmarks sitting accurately on the toepad pads.
3. Test **Train** tab:
   - Create a new project with `sample_data/real_lizard_toepad_dataset/annotations.xml`.
   - Verify dataset preview displays all 4 classes.
   - Run training: verify training finishes without `No module named 'ultralytics'` and outputs valid models for each class.

---

## 5. Key File Locations Reference

| Item | Location |
|---|---|
| Active Worktree | `/Users/leyangloh/dev/LizardMorph/.worktrees/generic-yolo-obb-pipeline` |
| Reference Rectify-512 Eval Script | `/Users/leyangloh/.gemini/antigravity-cli/brain/9d5ce704-f255-4dc7-8ade-c571f375d898/scratch/evaluate_full_heldout.py` |
| Accuracy Campaign Documentation | `/Users/leyangloh/dev/LizardMorph/docs/2026-07-26-accuracy-campaign-findings.md` |
| Sample Specimen Dataset | `/Users/leyangloh/dev/LizardMorph/sample_data/real_lizard_toepad_dataset/` |
| Pretrained YOLO-OBB Weights | `/Users/leyangloh/dev/LizardMorph/models/lizard-toe-pad/yolo_obb_6class_h7.onnx` |
| Pretrained ML-Morph Landmark Weights | `/Users/leyangloh/dev/LizardMorph/models/lizard-toe-pad/ml_morph_best.dat` |
| Base YOLO Model for Training | `/Users/leyangloh/dev/LizardMorph/models/lizard-toe-pad/yolov8n-obb.pt` |

# ml-morph Landmark Prediction Experiments

This document outlines the complete pipeline for lizard toepad landmark prediction using dlib shape predictors (`ml-morph`) with YOLO-detected bounding boxes.

## Overview

The standard pipeline compares two detection approaches to evaluate the accuracy of landmark predictions:

1. **Baseline**: Standard YOLO detect (axis-aligned bounding boxes, 6-class model)
2. **OBB**: YOLO-OBB (oriented bounding boxes) with flip inference strategy

### Results: Baseline (H5) vs OBB (Axis-Aligned)

Test error is defined as the average pixel deviation between predicted and ground-truth landmarks (based on 144 hyperparameter configs searched per condition).

* **Toe**: Baseline 83.27 px → OBB 30.72 px (**2.7x better**)
* **Finger**: Baseline 40.64 px → OBB 38.25 px (**1.06x better**)

**Conclusion**: OBB wins in both categories. Oriented bounding boxes converted to axis-aligned rectangles with 30% padding produce tighter, more relevant crops for the shape predictor.

### Failed Experiment: Crop+Rotate OBB

An attempt was made to crop and rotate images so the OBB becomes upright with 10% padding. Results were significantly worse (toe: 92.02, finger: 76.11) due to rotation interpolation artifacts and landmark transformation issues. 
*See [EXPERIMENT_CROP_ROTATE_OBB.md](EXPERIMENT_CROP_ROTATE_OBB.md) for full details.*

## What's Included

* **ml-morph framework**: `dlib` shape predictor training with TPS → XML → YOLO bbox pipeline
* **Preprocessing scripts**: located in `ml-morph/scripts/preprocessing/` (e.g., `tps_to_xml.py`, `generate_yolo_bbox_xml.py`)
* **Inference script**: `ml-morph/scripts/inference/predict_landmarks_flip.py` — End-to-end OBB detection + landmark prediction with flip strategy
* **Hyperparameter search**: 144-config grid search over `tree_depth`, `cascade_depth`, `nu`, `num_trees`
* **SLURM sbatch files**: For preprocessing, training, and hyperparameter search (baseline, OBB, OBB-aligned)
* **Documentation**:
  * [COMPARISON_BASELINE_VS_OBB.md](COMPARISON_BASELINE_VS_OBB.md)
  * [EXPERIMENT_CROP_ROTATE_OBB.md](EXPERIMENT_CROP_ROTATE_OBB.md)

---

## Step-by-Step Instructions

This guide assumes you have followed the main setup instructions and have access to the PACE ICE cluster.

### Step 1: Preprocessing Data (TPS → XML → YOLO Bounding Boxes)

To prepare the dataset for training the shape predictors, run the preprocessing jobs which convert TPS files to XML and overlay YOLO bounding boxes.

```bash
cd ~/Lizard_Toepads
# Run baseline preprocessing (Axis-Aligned YOLO)
sbatch sbatch/preprocess_baseline.sbatch

# Run OBB preprocessing (YOLO-OBB)
sbatch sbatch/preprocess_obb.sbatch
```

### Step 2: Training & Hyperparameter Search

Once the XML files with bounding boxes are generated, you can train the dlib shape predictors. 

**Standard Training:**
```bash
# Train on Toe dataset
sbatch sbatch/train_mlmorph_toe.sbatch

# Train on Finger dataset
sbatch sbatch/train_mlmorph_finger.sbatch
```

**Hyperparameter Grid Search (Optional):**
To reproduce the 144-config search:
```bash
sbatch sbatch/hyperparam_search_toe_obb.sbatch
sbatch sbatch/hyperparam_search_finger_obb.sbatch
# (Other variations available in the sbatch/ directory)
```

### Step 3: Run Inference (End-to-End)

To run the complete pipeline on new images (YOLO-OBB Detection + dlib Landmark Prediction), use the flip inference strategy script:

```bash
uv run python ml-morph/scripts/inference/predict_landmarks_flip.py \
    --input-dir <path_to_images> \
    --output-dir <path_to_output_dir> \
    --yolo-model models/best_obb.pt \
    --toe-predictor models/toe_predictor.dat \
    --finger-predictor models/finger_predictor.dat
```

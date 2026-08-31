# Lizard Toepad Demo Dataset

This curated demo dataset is designed for training and testing custom **YOLO OBB + ML-Morph** toepad morphometrics pipelines in AutoMorph.

## Contents
* **5 Specimen Scan Images** (`toepad_specimen_001.jpg` – `toepad_specimen_005.jpg`)
* **Dlib XML Annotation File** (`annotations.xml`) with 4 labeled digit classes per image:
  * `up_finger` (Upper limb finger)
  * `up_toe` (Upper limb toe)
  * `bot_finger` (Bottom limb finger)
  * `bot_toe` (Bottom limb toe)
* **TPS Annotation File** (`annotations.tps`) with 9 standard morphometric landmarks per digit.

## 9-Point Landmark Topology
* **0**: Claw base (distal boundary of toepad)
* **1**: Claw tip
* **2**: Distal lateral toepad boundary
* **3**: Distal medial toepad boundary
* **4**: Maximum width lateral landmark (width = distance [4, 5])
* **5**: Maximum width medial landmark
* **6**: Proximal lateral toepad boundary
* **7**: Proximal medial toepad boundary (polygon [2, 3, 5, 7, 8, 6, 4] = toepad area)
* **8**: Proximal toepad junction

## How to Use in AutoMorph Training Wizard
1. Open AutoMorph and click **"Train Custom Model Wizard"**.
2. **Step 1**: Name your project (e.g. `Anolis Toepad Demo`).
3. **Step 2**: Drag and drop `annotations.xml` and the 5 `.jpg` images from this folder (or upload `lizard_toepad_demo.zip`).
4. **Step 3**: Click **"Preview Derived Boxes"** to verify automatic YOLO OBB detection alignment.
5. **Step 4 & 5**: Select **Fast** or **Standard** preset and click **"Start Training Pipeline"**.

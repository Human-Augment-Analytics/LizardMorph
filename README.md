# React + Flask LizardMorph App

**Live Tryout:** [https://haag-1.cc.gatech.edu/](https://haag-1.cc.gatech.edu/)

This app is built on the machine learning toolbox ml-morph. This app has a pre-trained model to predict 34 landmarks on lizard anole x-rays, as well as models for toe pads.

To learn more about the ml-morph toolbox: 
Porto, A. and Voje, K.L., 2020. ML‐morph: A fast, accurate and general approach for automated detection and landmarking of biological structures in images. Methods in Ecology and Evolution, 11(4), pp.500-512.

## Structure

- **Frontend**: Located in `frontend/`, built with React.
- **Backend**: Located in `backend/`, powered by Flask.

## Prerequisites

- **macOS or Linux**
- **uv** (for fast Python environment and package management)
- **Node.js ≥ 18** + npm (for the frontend)
- The `models/` directory populated (see below for download script)
- **Reference Repo:** [Lizard_Toepads](https://github.com/Human-Augment-Analytics/Lizard_Toepads)

## 1. Clone the Repo

```bash
git clone <repo-url> LizardMorph
cd LizardMorph
```

## 2. Setup utilizing Make + uv

We use a Makefile to automate building the frontend and backend. If you don't have `uv` installed, you can install it quickly using their [official script](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Linux, `uv` installs to `~/.local/bin`. If `uv` isn't found after install, add it to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Once installed, simply run this command from the root directory to create the virtual environment, install Python backend dependencies, install the Node frontend modules, and **download the required models**:

```bash
make setup
```

If you only need to download/update the models later, you can run:

```bash
make download-models
```

> **Note:** `make download-models` uses `rsync` over SSH to a remote host. If you don't have access/keys on your machine, this step will fail/hang. In that case, populate `models/` manually and run:
>
> ```bash
> make setup-backend setup-frontend
> ```

Verify everything imported correctly for the backend:
```bash
cd backend && uv run python scripts/check_env.py
```

## 3. Configure Environment Variables

Copy the example and fill in any missing paths:

```bash
cp .env.example .env   # if it exists, otherwise create .env at the repo root
```

The `.env` file must live at the **repo root** (`LizardMorph/.env`), not inside `backend/`.

### Required variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `3005` | Port the Flask backend listens on |
| `TOEPAD_YOLO_MODEL` | `../models/lizard-toe-pad/yolo_obb_6class_h7.onnx` | YOLO OBB detector |
| `TOEPAD_TOE_PREDICTOR` | `../models/lizard-toe-pad/toe_predictor_obb.dat` | 9-point toe dlib predictor |
| `TOEPAD_FINGER_PREDICTOR` | `../models/lizard-toe-pad/finger_predictor_obb.dat` | 9-point finger dlib predictor ⚠️ |
| `TOEPAD_SCALE_PREDICTOR` | `../models/lizard-toe-pad/lizard_scale.dat` | 2-point ruler predictor |
| `ID_EXTRACTOR_MODEL` | `../models/lizard-toe-pad/yolo_bounding_box.pt` | Legacy YOLO for ID detection |
| `DORSAL_PREDICTOR_FILE` | `../models/lizard-x-ray/new_landmarks_2025_predictor.dat` | Dorsal landmark predictor |
| `LATERAL_PREDICTOR_FILE` | `../models/lizard-x-ray/lateral_predictor_auto.dat` | Lateral landmark predictor |

> ⚠️ **`TOEPAD_FINGER_PREDICTOR` and `TOEPAD_SCALE_PREDICTOR` must NOT be swapped** — this causes fingers to get 2 ruler landmarks instead of 9.

### Optional variables

| Variable | Description |
|----------|-------------|
| `VITE_API_URL` | Frontend API base URL (default: `/api`) |
| `VITE_BASE_URL` | Frontend base path (default: empty) |
| `VITE_ALLOWED_HOSTS` | Comma-separated allowed ngrok/proxy hosts |
| `WEBHOOK_SECRET` | GitHub webhook secret for auto-deploy |

### Full working `.env` example

```bash
API_PORT=3005
VITE_API_URL=/api

TOEPAD_YOLO_MODEL=../models/lizard-toe-pad/yolo_obb_6class_h7.onnx
TOEPAD_TOE_PREDICTOR=../models/lizard-toe-pad/toe_predictor_obb.dat
TOEPAD_FINGER_PREDICTOR=../models/lizard-toe-pad/finger_predictor_obb.dat
TOEPAD_SCALE_PREDICTOR=../models/lizard-toe-pad/lizard_scale.dat
ID_EXTRACTOR_MODEL=../models/lizard-toe-pad/yolo_bounding_box.pt

DORSAL_PREDICTOR_FILE=../models/lizard-x-ray/new_landmarks_2025_predictor.dat
LATERAL_PREDICTOR_FILE=../models/lizard-x-ray/lateral_predictor_auto.dat
```

## 4. Required Model Files

All models go in `models/lizard-toe-pad/` and `models/lizard-x-ray/` (relative to repo root). These are **not in git**.

```
models/
├── lizard-toe-pad/
│   ├── yolo_obb_6class_h7.pt        ← Backend YOLO (PyTorch)
│   ├── yolo_obb_6class_h7.onnx      ← Frontend YOLO (ONNX, IR=8)
│   ├── finger_predictor_obb.dat     ← 9-point finger predictor
│   ├── toe_predictor_obb.dat        ← 9-point toe predictor
│   ├── lizard_scale.dat             ← 2-point ruler predictor
│   └── yolo_bounding_box.pt         ← ID text detection model
└── lizard-x-ray/
    ├── new_landmarks_2025_predictor.dat ← Dorsal landmark predictor
    └── lateral_predictor_auto.dat       ← Lateral landmark predictor
```

(If the predictor is too big for your platform, ensure it is downloaded and placed as per the paths in your `.env` file.)

## 5. Start for Development

You can effortlessly start both the backend and frontend simultaneously with:

```bash
make dev
```

The API will be available at `http://localhost:3005` (or your configured `API_PORT`), and the UI will be at `http://localhost:5173`.

> **Note (ports / restarts):** `make dev` will stop any existing backend listener on `API_PORT` (including a Gunicorn process) and stop any existing Vite dev server on port 5173, then restart both. Cancel with `Ctrl+C` to stop the dev processes.

### Setting up ngrok for External Access
If you want to make your local development server accessible from the internet:
https://ngrok.com/docs/getting-started/
1. Install ngrok:
   - Download from https://ngrok.com/download
   - Or `brew install ngrok` (macOS)

2. Configure ngrok:
   - Register a free account with ngrok 
   - Add your auth token
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

3. Update `.env` file with the following
   ```
   VITE_ALLOWED_HOSTS=your-subdomain.ngrok-free.app
   ```
   Replace `your-subdomain` with your actual ngrok subdomain (e.g., "8e41-123-45-67-89.ngrok-free.app")

4. Start ngrok tunnel (after starting your frontend server):
   ```bash
   ngrok http 5173  # Use your frontend port number
   ```

5. Copy the generated ngrok URL
   and update your `.env` file with this domain.

## 6. Server Production Deployment

To set up and run the application in a production environment:

1. Clone repository and install dependencies using `make setup`.
2. Configure `.env` with appropriate model paths, API port, and frontend base URL.
3. Build the frontend for production:
   ```bash
   cd frontend
   npm run build
   ```
4. Use `gunicorn.conf.py` at the repo root for production settings:

   ```bash
   cd /var/www/LizardMorph
   cd backend && uv run gunicorn -c ../gunicorn.conf.py app:app
   ```

   Or use the minimal startup script from the backend:

   ```bash
   cd backend
   bash startup.sh
   # Update startup.sh to utilize `uv run gunicorn` accordingly if desired.
   ```

Gunicorn logs go to `/var/log/lizardmorph/` (configure in `gunicorn.conf.py`). Ensure the port matches what your reverse proxy (e.g., Nginx) expects.

## Vignette

1. Open a terminal and run `make dev`.
2. Navigate to http://localhost:5173 (or your specific port)
3. Hit upload on the webpage and select the picture from the folder `sample_image` in the project directory.
4. Notice output.xml, output.tps, output.csv appear in project directory.
5. Image should appear in the web browser. Landmarks on image can be moved to fix predictions.
6. Image can be viewed in three ways: original upload, color contrasted or inverted color.
7. Image can be downloaded for records.
8. For toes, after upload, each finger box should have **9 landmarks**, each toe box **9 landmarks**, and the ruler box **2 landmarks**.

## Backend Utility Scripts

Navigate to the backend directory `cd backend`.
Run all scripts using `uv` efficiently within the virtual environment:
```bash
uv run python scripts/<script>.py --help
```

| Script | Purpose |
|--------|---------|
| `check_env.py` | Verify all required packages are importable in the current env |
| `run_yolo_inference.py` | Run YOLO OBB inference on a single image, print detections |
| `run_yolo_bulk.py` | Run YOLO OBB on a directory of images, report id/ruler detection |
| `run_yolo_legacy.py` | Run the legacy (non-OBB) bounding-box YOLO model on an image |
| `visualize_diff.py` | Visual diff of .pt vs .onnx landmark output side-by-side |
| `compare_models.py` | Numerical comparison of .pt vs .onnx landmark coordinates |
| `do_export.py` | Export a YOLO `.pt` model to ONNX (IR=8, opset=17) |
| `export_helper.py` | Set up an isolated venv for ONNX export to avoid env conflicts |
| `inspect_onnx.py` | Inspect ONNX model input/output shapes and run a dummy inference |
| `get_yolo_classes.py` | Print class name mapping from a YOLO `.pt` model |
| `predict_landmarks_flip.py` | Batch landmark prediction with vertical-flip inference |
| `toepad_preprocessing.py` | Preprocess toepad images before prediction |
| `test_api_extract_id.py` | Integration test for the `/extract_id` Flask endpoint |

## Troubleshooting

| Symptom / Issue | Likely Cause / Fix |
|-----------------|--------------------|
| **Missing predictor file error** | The machine learning model file (`new_landmarks_2025_predictor.dat`) is required but not included in the repository due to its size. Download the predictor file and place it in the `models/lizard-x-ray/` directory. Update `.env`. |
| **Finger boxes have 2 landmarks** | `TOEPAD_FINGER_PREDICTOR` points to `lizard_scale.dat`. Fix `.env`, restart Flask. |
| **`No module named dlib`** | Wrong Python env. Verify the virtual environment with `uv` by running `make setup`. Make sure you use `uv run`. |
| **Frontend shows no boxes** | ONNX model IR version too high. Reconvert to IR 8: `uv run python scripts/do_export.py`. |
| **"No ID found" error** | Wrong model in `ID_EXTRACTOR_MODEL`. Check the path points to `yolo_bounding_box.pt`. |
| **Frontend not loading** | Check browser console for CORS errors; ensure the backend API URL is correctly set in `.env` or Vite config. |
| **Image processing errors** | Verify the uploaded image format is supported (JPG, PNG, TIF, BMP). |

## ⚠️ Known Issues and Limitations

### Production Deployment

1. **Predictor File**: Models are required but not included in the repository due to their size. They must be downloaded separately.
2. **Cold Start Issues**: Containers or processes may experience cold start delays. The initial load of the application might take up to a minute due to the size of the ML models being loaded.
3. **Processing Large Files**: Your server tier/instance size should have sufficient RAM for processing multiple or large X-ray images. Up to 8 GB of RAM is recommended.

# LizardMorph — Webserver Setup Guide

## Prerequisites

- **macOS or Linux**
- **Conda** (Miniconda or Anaconda)
- **Node.js ≥ 18** + npm (for the frontend)
- The `models/` directory populated (not in git — download separately)

---

## 1. Clone the Repo

```bash
git clone <repo-url> LizardMorph
cd LizardMorph
```

---

## 2. Create the Conda Environment

```bash
conda env create -f backend/environment.yml
conda activate lizard
```

Verify everything imported correctly:
```bash
conda run -n lizard python backend/scripts/check_env.py
```

---

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

---

## 4. Required Model Files

All models go in `models/lizard-toe-pad/` (relative to repo root). These are **not in git**.

```
models/
└── lizard-toe-pad/
    ├── yolo_obb_6class_h7.pt        ← Backend YOLO (PyTorch)
    ├── yolo_obb_6class_h7.onnx      ← Frontend YOLO (ONNX, IR=8)
    ├── finger_predictor_obb.dat     ← 9-point finger predictor
    ├── toe_predictor_obb.dat        ← 9-point toe predictor
    ├── lizard_scale.dat             ← 2-point ruler predictor
    └── yolo_bounding_box.pt         ← ID text detection model
```

---

## 5. Start the Backend (Development)

```bash
cd backend
conda activate lizard
FLASK_DEBUG=true python app.py
```

The API will be available at `http://localhost:3005`.

> **Note:** Restart Flask after every `.env` change — environment variables are read at startup.

---

## 6. Start the Frontend (Development)

```bash
cd frontend
npm install
npm run dev
```

The UI will be at `http://localhost:5173` by default.

---

## 7. Production Deployment (Gunicorn)

Use `gunicorn.conf.py` at the repo root for production settings:

```bash
cd LizardMorph
conda activate lizard
gunicorn -c gunicorn.conf.py app_wsgi:application
```

Or use the minimal startup from `backend/startup.sh`:

```bash
cd backend
conda activate lizard
gunicorn --bind=0.0.0.0 --timeout 600 app:app
```

Gunicorn logs go to `/var/log/lizardmorph/` (configure in `gunicorn.conf.py`).

---

## 8. Verify

1. Open `http://localhost:5173`
2. Upload a toepad `.jpg` image
3. Check Flask log: you should see:
   ```
   Loading finger predictor: .../finger_predictor_obb.dat   ✓
   Loading toe predictor:    .../toe_predictor_obb.dat       ✓
   ```
4. After upload, each finger box should have **9 landmarks**, each toe box **9 landmarks**, and the ruler box **2 landmarks**

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Finger boxes have 2 landmarks | `TOEPAD_FINGER_PREDICTOR` points to `lizard_scale.dat` | Fix `.env`, restart Flask |
| `No module named dlib` | Wrong Python/conda env | Run with `conda run -n lizard` |
| Frontend shows no boxes | ONNX model IR version too high | Reconvert to IR 8: `python scripts/do_export.py` |
| "No ID found" error | Wrong model in `ID_EXTRACTOR_MODEL` | Check the path points to `yolo_bounding_box.pt` |
| Landmarks in wrong spot after Save | Old save bug (fixed in commit `d4ae110`) | Pull latest |

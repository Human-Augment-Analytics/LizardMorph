# LizardMorph — Agent Navigation Guide

> Quick-reference for AI agents. Read this first to orient yourself before opening any source file.

---

## Project Purpose

LizardMorph is a morphometric annotation tool for lizard images (x-ray, toepad photos).  
It runs as a **web app** (Flask + React/Vite) and also as an **Electron desktop app**.  
Users upload images, the tool detects bounding boxes (YOLO OBB), runs dlib shape predictors to place landmarks, and exports XML / TPS / CSV data.

---

## Directory Map (skip `node_modules`, `__pycache__`, `.git`)

```
LizardMorph/
├── AGENTS.md                   ← YOU ARE HERE
├── .env                        ← All model paths & API config (root level, not backend/)
├── requirements.txt            ← Root-level Python deps (minimal)
├── app_wsgi.py                 ← WSGI entry point for production
├── gunicorn.conf.py            ← Gunicorn config
│
├── backend/                    ← Flask API (Python, conda env: `lizard`)
│   ├── app.py                  ⭐ MAIN Flask server — all API routes (~2 000 lines)
│   ├── utils.py                ⭐ Core prediction/XML logic (~2 500 lines)
│   ├── session_manager.py      ← Session lifecycle (create/delete/list)
│   ├── export_handler.py       ← TPS / CSV / ZIP export
│   ├── ort_inference.py        ← ONNX Runtime YOLO wrapper (CPU/INT8)
│   ├── native_ocr.py           ← macOS Vision / Tesseract OCR for ID extraction
│   ├── id_extractor.py         ← Glue: YOLO detect → OCR read ID label
│   ├── xray_preprocessing.py   ← Image contrast enhancement before prediction
│   ├── visual_individual_performance.py ← XML → frontend JSON; image invert
│   ├── environment.yml         ← Conda env spec (name: lizard)
│   ├── scripts/                ← One-off utility/benchmark scripts (not prod)
│   ├── sessions/               ← Per-session runtime data (upload, outputs, XML)
│   └── uploads/                ← Temp upload directory
│
├── frontend/                   ← React + Vite + TypeScript SPA
│   └── src/
│       ├── App.tsx             ⭐ Router — routes: / | /dorsal | /lateral | /toepads | /free
│       ├── views/MainView.tsx  ⭐ Primary app shell (~1 000 lines)
│       ├── components/
│       │   ├── SVGViewer.tsx   ⭐ D3-based landmark/bbox renderer (~1 500 lines)
│       │   ├── LandingPage.tsx ← View-type selection screen
│       │   ├── Header.tsx      ← Top nav bar
│       │   ├── MeasurementsAndScalePanel.tsx ← Measurement UI
│       │   ├── HistoryPanel.tsx← Undo/redo history list
│       │   └── SessionInfo.tsx ← Session metadata display
│       ├── services/
│       │   ├── OnnxService.ts  ⭐ Frontend ONNX inference (YOLO OBB, runs in browser)
│       │   ├── ApiService.ts   ← All fetch() calls to Flask backend
│       │   ├── SessionService.ts ← Session state management
│       │   ├── ExportService.ts← Client-side export helpers
│       │   └── config.ts       ← Base URL / API URL (reads VITE_ env vars)
│       └── models/             ← TypeScript type definitions
│
├── electron/                   ← Electron desktop wrapper
│   ├── main.js                 ← Electron main process (window, IPC)
│   ├── python-backend.js       ← Spawns Flask as a child process
│   └── pyinstaller.spec        ← Bundles backend into executable
│
├── models/
│   ├── lizard-toe-pad/         ← Toepad YOLO + dlib models
│   │   ├── yolo_obb_6class_h7.onnx   ← Frontend ONNX YOLO detector
│   │   ├── toe_predictor_obb.dat     ← dlib 9-pt toe predictor
│   │   ├── finger_predictor_obb.dat  ← dlib 9-pt finger predictor
│   │   └── lizard_scale.dat          ← dlib 2-pt ruler predictor
│   └── lizard-x-ray/           ← Dorsal / lateral landmark predictors
│       ├── dorsal_predictor_clahe_best.dat
│       └── lateral_predictor_auto.dat
│
├── docs/                       ← Deployment guides, research notes, plans
├── .agent/workflows/           ← Agent workflow markdown files
└── monitoring/                 ← Prometheus / Grafana configs
```

---

## Key Entry Points

| What | File / Command |
|------|----------------|
| Start backend (dev) | `cd backend && conda activate lizard && FLASK_DEBUG=true python app.py` |
| Start frontend (dev) | `cd frontend && npm run dev` |
| All API routes | `backend/app.py` — search `@app.route` |
| Core ML pipeline | `backend/utils.py` — fn `predictions_to_xml_single_from_client_annotations` |
| Frontend ONNX inference | `frontend/src/services/OnnxService.ts` |
| Landmark rendering | `frontend/src/components/SVGViewer.tsx` |
| Session data on disk | `backend/sessions/<session_id>/` |

---

## View Types & Models

| View type | Predictor | YOLO | Notes |
|-----------|-----------|------|-------|
| `dorsal` | `dorsal_predictor_clahe_best.dat` | — | X-ray top-down |
| `lateral` | `lateral_predictor_auto.dat` | — | X-ray side view |
| `toepad` | `toe/finger/scale_predictor_obb.dat` | `yolo_obb_6class_h7.onnx` | 6-class OBB |
| `free` | — | — | No ML, manual only |
| `custom` | `custom_predictor_auto.dat` | — | User-trained model |

### YOLO 6-Class OBB Labels (toepad only)

| ID | Label | Notes |
|----|-------|-------|
| 0 | `up_finger` | Needs vertical flip before inference |
| 1 | `up_toe` | Needs vertical flip |
| 2 | `bot_finger` | Normal orientation |
| 3 | `bot_toe` | Normal orientation |
| 4 | `ruler` | Scale bar |
| 5 | `id` | Text ID label |

---

## Environment & Config

```bash
# Root .env (NOT backend/.env) — Flask reads this via python-dotenv
API_PORT=3005
VITE_API_URL="/api"

# Model paths (relative to backend/ dir at runtime)
DORSAL_PREDICTOR_FILE=../models/lizard-x-ray/dorsal_predictor_clahe_best.dat
LATERAL_PREDICTOR_FILE=../models/lizard-x-ray/lateral_predictor_auto.dat
TOEPAD_TOE_PREDICTOR=../models/lizard-toe-pad/toe_predictor_obb.dat
TOEPAD_FINGER_PREDICTOR=../models/lizard-toe-pad/finger_predictor_obb.dat
TOEPAD_SCALE_PREDICTOR=../models/lizard-toe-pad/lizard_scale.dat
```

> ⚠️ **Common mistake**: `.env` lives at repo root, not in `backend/`. If paths are swapped (e.g. finger points to `lizard_scale.dat`) landmarks will be wrong — only 2 points appear instead of 9.

---

## Common Bugs & Fixes (quick lookup)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Finger boxes have only 2 landmarks | `TOEPAD_FINGER_PREDICTOR` points to `lizard_scale.dat` | Fix `.env`; restart Flask |
| Landmarks all stacked/wrong | Old `save_annotations` sent coords to all boxes | Route by `box_idx`; fallback to nearest center |
| Only 2 of 9 landmarks visible in UI | D3 key collision (both finger & toe parts numbered 0–8) | Use `unique_id = box_idx * 100 + landmark_id` |
| `up_finger` landmarks double-flipped | Frontend pre-flips coords; backend flipped again | Don't pre-flip in `predictions_to_xml_single_from_client_annotations` |
| Stale XML reused after model fix | `process_existing` skips prediction if XML exists | Delete old XML from `backend/sessions/<id>/outputs/` |

---

## Search Tips for Agents

| Task | Where to look |
|------|---------------|
| Add/modify an API endpoint | `backend/app.py` — search `@app.route("/your-path"` |
| Change prediction logic | `backend/utils.py` — search function name |
| Modify landmark rendering | `frontend/src/components/SVGViewer.tsx` |
| Change what data is sent to backend | `frontend/src/services/ApiService.ts` |
| Change ONNX inference params | `frontend/src/services/OnnxService.ts` |
| Session file structure | `backend/session_manager.py` |
| Export format (TPS/CSV/ZIP) | `backend/export_handler.py` |
| Electron packaging | `electron/main.js`, `electron/pyinstaller.spec` |

---

## Git

```bash
# System git may be blocked by Xcode license — use homebrew git:
/opt/homebrew/bin/git log --oneline -10

# Active branch (as of March 2026):
# yolo-obb-integration
```

---

## Do NOT Read (Too Large / Generated)

- `main.html` — 860 KB generated file
- `frontend/node_modules/`, `electron/node_modules/`
- `backend/sessions/` contents (runtime data)

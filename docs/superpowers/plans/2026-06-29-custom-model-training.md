# Custom Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to upload a ZIP file of annotated images in Free Mode and train a custom dlib shape predictor model on their local machine.

**Architecture:** A new asynchronous job runner on the Flask backend executes dlib shape predictor training inside a background thread. The React frontend handles the ZIP file upload and polls a status endpoint to track training progress, automatically updating the available models list on completion.

**Tech Stack:** Python, Flask, dlib, React, TypeScript

## Global Constraints
*   Do not send any training data to external cloud services; run everything locally.
*   Preserve all existing routes and configurations without regression.
*   Ensure proper Y-axis coordinate conversions between TPS and dlib screen coordinates.

---

### Task 1: Backend Training Core & XML Generation

**Files:**
*   Modify: `backend/utils.py:1600-`
*   Create: `backend/tests/test_training.py`

**Interfaces:**
*   Consumes: Nothing
*   Produces: `utils.train_predictor_from_zip(model_name: str, zip_path: str, predictor_id: str, index_path: str, files_dir: str) -> dict`

- [ ] **Step 1: Write the failing unit test**

Create the test file `backend/tests/test_training.py` with code to set up a mock dataset and run the training utility function:

```python
import os
import shutil
import tempfile
import zipfile
import cv2
import numpy as np
import pytest
from utils import train_predictor_from_zip, read_tps
import predictor_library

@pytest.fixture
def temp_workspace():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

def test_train_predictor_from_zip_success(temp_workspace):
    # 1. Create a mock dataset zip file
    zip_path = os.path.join(temp_workspace, "dataset.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        # Save 2 solid black images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_path = "img1.jpg"
        img2_path = "img2.jpg"
        cv2.imwrite(img1_path, img)
        cv2.imwrite(img2_path, img)
        
        z.write(img1_path)
        z.write(img2_path)
        
        os.remove(img1_path)
        os.remove(img2_path)
        
        # Create corresponding TPS annotations
        tps_content = """LM=2
10.0 90.0
20.0 80.0
IMAGE=img1.jpg
ID=0

LM=2
15.0 85.0
25.0 75.0
IMAGE=img2.jpg
ID=1
"""
        z.writestr("annotations.tps", tps_content)

    # 2. Run the training utility function
    # Note: We configure very low depth/cascades to make it run in <1 second
    mock_options = {
        "nu": 0.1,
        "tree_depth": 2,
        "cascade_depth": 5
    }
    
    meta = train_predictor_from_zip(
        model_name="Test Predictor",
        zip_path=zip_path,
        predictor_id="test-uuid",
        index_path=index_path,
        files_dir=files_dir,
        custom_options=mock_options
    )
    
    assert meta is not None
    assert meta["display_name"] == "Test Predictor"
    assert meta["num_parts"] == 2
    assert os.path.exists(os.path.join(files_dir, "test-uuid.dat"))
    
    # Check index registration
    predictors = predictor_library.list_predictors(index_path)
    assert len(predictors) == 1
    assert predictors[0].id == "test-uuid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_training.py -v`
Expected: Fail with `ImportError: cannot import name 'train_predictor_from_zip'`

- [ ] **Step 3: Implement training logic**

Append the implementation to `backend/utils.py`:

```python
import zipfile
import shutil
import dlib
import xml.etree.ElementTree as ET
from xml.dom import minidom
import predictor_library

def train_predictor_from_zip(model_name, zip_path, predictor_id, index_path, files_dir, custom_options=None):
    """
    Extracts a ZIP containing images and annotations, formats a dlib dataset, trains
    a shape predictor, registers it in the library, and cleans up.
    """
    temp_dir = os.path.join(os.path.dirname(zip_path), f"training_{predictor_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
        
    # 2. Locate TPS or XML file
    tps_file = None
    xml_file = None
    for root_dir, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(".tps"):
                tps_file = os.path.join(root_dir, f)
            elif f.lower().endswith(".xml") and not f.lower().startswith("mock"):
                xml_file = os.path.join(root_dir, f)
                
    if not tps_file and not xml_file:
        shutil.rmtree(temp_dir)
        raise ValueError("No .tps or .xml annotation file found in dataset ZIP")
        
    dataset_xml_path = os.path.join(temp_dir, "dataset.xml")
    num_parts = 0
    
    try:
        if tps_file:
            # Parse TPS and build XML
            tps_data = read_tps(tps_file)
            if not tps_data["im"]:
                raise ValueError("TPS file contains no specimens or images.")
                
            root = ET.Element("dataset")
            ET.SubElement(root, "name").text = model_name
            images_e = ET.SubElement(root, "images")
            
            for idx, img_name in enumerate(tps_data["im"]):
                # Locate referenced image inside temp_dir
                img_filename = os.path.basename(img_name)
                local_img_path = None
                for r_dir, _, fs in os.walk(temp_dir):
                    for file in fs:
                        if file.lower() == img_filename.lower():
                            local_img_path = os.path.join(r_dir, file)
                            break
                    if local_img_path:
                        break
                        
                if not local_img_path:
                    raise ValueError(f"Image '{img_name}' referenced in TPS not found in ZIP.")
                    
                img = cv2.imread(local_img_path)
                if img is None:
                    raise ValueError(f"Failed to read image '{img_name}'")
                h, w = img.shape[:2]
                
                # Bounding box covering whole image
                box_e = ET.SubElement(images_e, "image", file=local_img_path)
                box = ET.SubElement(box_e, "box", top="1", left="1", width=str(w - 2), height=str(h - 2))
                
                # Flip coordinates since TPS counts Y from bottom
                coords = tps_data["coords"][idx]
                num_parts = len(coords)
                for pt_idx, coord in enumerate(coords):
                    part = ET.SubElement(box, "part", name=str(pt_idx))
                    part.set("x", str(int(round(coord[0]))))
                    part.set("y", str(int(round(h - coord[1]))))
                    
            et = ET.ElementTree(root)
            xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
            with open(dataset_xml_path, "w", encoding="utf-8") as f:
                f.write(xmlstr)
        else:
            # Use uploaded XML directly but rewrite file paths to point to absolute temp paths
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for image in root.findall(".//image"):
                original_file = image.get("file")
                img_filename = os.path.basename(original_file)
                local_img_path = None
                for r_dir, _, fs in os.walk(temp_dir):
                    for file in fs:
                        if file.lower() == img_filename.lower():
                            local_img_path = os.path.join(r_dir, file)
                            break
                    if local_img_path:
                        break
                        
                if not local_img_path:
                    raise ValueError(f"Image '{original_file}' referenced in XML not found in ZIP.")
                image.set("file", local_img_path)
                
                # Determine number of parts
                parts = image.findall(".//part")
                if parts:
                    num_parts = max(num_parts, len(parts))
                    
            tree.write(dataset_xml_path)
            
        # 3. Train dlib predictor
        options = dlib.shape_predictor_training_options()
        if custom_options:
            options.nu = custom_options.get("nu", 0.1)
            options.tree_depth = custom_options.get("tree_depth", 4)
            options.cascade_depth = custom_options.get("cascade_depth", 15)
        else:
            options.nu = 0.1
            options.tree_depth = 4
            options.cascade_depth = 15
            
        options.oversampling_amount = 5
        options.be_verbose = False
        
        output_model_path = os.path.join(files_dir, f"{predictor_id}.dat")
        dlib.train_shape_predictor(dataset_xml_path, output_model_path, options)
        
        # Validate predictor via library structure
        sp = dlib.shape_predictor(output_model_path)
        meta = predictor_library.PredictorMeta(
            id=predictor_id,
            display_name=model_name,
            stored_filename=f"{predictor_id}.dat",
            uploaded_at=predictor_library._now_iso(),
            size_bytes=os.path.getsize(output_model_path),
            num_parts=num_parts
        )
        
        # Save to list index
        idx = predictor_library.load_index(index_path)
        predictors = idx.get("predictors", [])
        predictors.append(predictor_library.asdict(meta))
        idx["predictors"] = predictors
        predictor_library.save_index(index_path, idx)
        
        return predictor_library.asdict(meta)
        
    finally:
        shutil.rmtree(temp_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/utils.py backend/tests/test_training.py
git commit -m "feat(backend): add train_predictor_from_zip core logic and tests"
```

---

### Task 2: Flask Endpoint Integration & Threading

**Files:**
*   Modify: `backend/app.py:110-` (initialize tracking)
*   Modify: `backend/app.py` (add endpoints)
*   Modify: `backend/tests/test_training.py` (add API endpoints tests)

**Interfaces:**
*   Consumes: `utils.train_predictor_from_zip`
*   Produces: API endpoints `/train_predictor` (POST) and `/train_status/<job_id>` (GET)

- [ ] **Step 1: Write failing API unit tests**

Append to `backend/tests/test_training.py`:

```python
def test_api_train_predictor_lifecycle(temp_workspace):
    # Setup app test client
    from app import app, TRAINING_JOBS
    import io
    app.config['TESTING'] = True
    
    # Temporarily override predictor library constants
    import app as app_module
    old_idx = app_module.PREDICTOR_LIBRARY_INDEX
    old_files = app_module.PREDICTOR_LIBRARY_FILES
    
    app_module.PREDICTOR_LIBRARY_INDEX = os.path.join(temp_workspace, "predictors.json")
    app_module.PREDICTOR_LIBRARY_FILES = os.path.join(temp_workspace, "files")
    os.makedirs(app_module.PREDICTOR_LIBRARY_FILES, exist_ok=True)

    try:
        # Create a mock zip payload
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as z:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite("t1.jpg", img)
            z.write("t1.jpg")
            os.remove("t1.jpg")
            tps_content = "LM=1\n10 10\nIMAGE=t1.jpg\nID=0\n"
            z.writestr("an.tps", tps_content)
        
        zip_buffer.seek(0)
        
        # Trigger training API
        with app.test_client() as client:
            resp = client.post(
                "/train_predictor",
                data={
                    "model_name": "API Train",
                    "dataset": (zip_buffer, "test.zip")
                },
                content_type="multipart/form-data"
            )
            assert resp.status_code == 202
            data = resp.get_json()
            assert data["success"] is True
            job_id = data["job_id"]
            
            # Wait for background thread to complete
            import time
            for _ in range(20):
                resp_status = client.get(f"/train_status/{job_id}")
                status_data = resp_status.get_json()
                if status_data["status"] in ["completed", "failed"]:
                    break
                time.sleep(0.5)
                
            assert status_data["status"] == "completed"
            assert status_data["error"] is None
            assert status_data["predictor"]["display_name"] == "API Train"
            
    finally:
        app_module.PREDICTOR_LIBRARY_INDEX = old_idx
        app_module.PREDICTOR_LIBRARY_FILES = old_files
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_training.py::test_api_train_predictor_lifecycle -v`
Expected: Fail with `404 Not Found` (endpoint doesn't exist)

- [ ] **Step 3: Implement training job worker and Flask routes**

In `backend/app.py`, define the job tracker registry around line 110:

```python
# Thread-safe dictionary to keep track of training jobs
TRAINING_JOBS = {}
TRAINING_JOBS_LOCK = threading.Lock()
```

Add endpoints to `backend/app.py` before `free_autoplace`:

```python
@app.route("/train_predictor", methods=["POST"])
@cross_origin()
@track_metrics
def start_train_predictor():
    try:
        model_name = request.form.get("model_name", "Custom Predictor").strip()
        f = request.files.get("dataset")
        
        if not f:
            return jsonify({"success": False, "error": "Missing dataset ZIP file"}), 400
            
        job_id = str(uuid.uuid4())
        
        # Ensure directories exist
        predictor_library.ensure_dir(PREDICTOR_LIBRARY_DIR)
        predictor_library.ensure_dir(PREDICTOR_LIBRARY_FILES)
        
        # Save ZIP upload to a temporary location
        temp_zip_path = os.path.join(PREDICTOR_LIBRARY_DIR, f"upload_{job_id}.zip")
        f.save(temp_zip_path)
        
        # Initialize registry entry
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id] = {
                "status": "pending",
                "error": None,
                "predictor": None
            }
            
        def worker_thread(jid, zip_p, name):
            try:
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[jid]["status"] = "training"
                    
                # Lower tree & cascade depth slightly for standard uploads if requested to keep server happy
                # (but respect standard requirements)
                meta = utils.train_predictor_from_zip(
                    model_name=name,
                    zip_path=zip_p,
                    predictor_id=jid,
                    index_path=PREDICTOR_LIBRARY_INDEX,
                    files_dir=PREDICTOR_LIBRARY_FILES
                )
                
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[jid]["status"] = "completed"
                    TRAINING_JOBS[jid]["predictor"] = meta
                    
            except Exception as ex:
                logger.error(f"Error in training job {jid}: {ex}", exc_info=True)
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[jid]["status"] = "failed"
                    TRAINING_JOBS[jid]["error"] = str(ex)
            finally:
                try:
                    if os.path.exists(zip_p):
                        os.remove(zip_p)
                except Exception:
                    pass

        # Launch background thread
        t = threading.Thread(target=worker_thread, args=(job_id, temp_zip_path, model_name))
        t.start()
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": "Training job started"
        }), 202
        
    except Exception as e:
        logger.error(f"Error starting training job: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/train_status/<job_id>", methods=["GET"])
@cross_origin()
@track_metrics
def get_train_status(job_id):
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
        
    return jsonify({
        "success": True,
        "status": job["status"],
        "error": job["error"],
        "predictor": job["predictor"]
    }), 200
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_training.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/tests/test_training.py
git commit -m "feat(backend): add training and status check endpoints"
```

---

### Task 3: Frontend API Service & TS Definitions

**Files:**
*   Modify: `frontend/src/services/ApiService.ts`

**Interfaces:**
*   Produces:
    *   `ApiService.trainPredictor(modelName: string, file: File) -> Promise<{ success: boolean; job_id: string; message: string }>`
    *   `ApiService.getTrainStatus(jobId: string) -> Promise<{ success: boolean; status: string; error: string | null; predictor: PredictorMeta | null }>`

- [ ] **Step 1: Implement ApiService endpoints**

Open [ApiService.ts](file:///Users/leyangloh/dev/LizardMorph/frontend/src/services/ApiService.ts) and append these calls:

```typescript
  async trainPredictor(
    modelName: string,
    file: File
  ): Promise<{ success: boolean; job_id: string; message: string }> {
    const formData = new FormData();
    formData.append("model_name", modelName);
    formData.append("dataset", file);

    const response = await fetch(`${API_URL}/train_predictor`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: "Server error" }));
      throw new Error(err.error || "Failed to start training job");
    }

    return response.json();
  },

  async getTrainStatus(
    jobId: string
  ): Promise<{
    success: boolean;
    status: "pending" | "training" | "completed" | "failed";
    error: string | null;
    predictor: PredictorMeta | null;
  }> {
    const response = await fetch(`${API_URL}/train_status/${jobId}`);
    if (!response.ok) {
      throw new Error("Failed to get training status");
    }
    return response.json();
  },
```

- [ ] **Step 2: Verify compiling**

Run: `cd frontend && npm run build` (or verify in editor that there are no TS compilation errors).
Expected: Compiles with no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/services/ApiService.ts
git commit -m "feat(frontend): add trainPredictor api service functions"
```

---

### Task 4: Train Mode Landing Card & Dedicated TrainView Routing

**Files:**
*   Modify: `frontend/src/components/LandingPage.tsx`
*   Modify: `frontend/src/App.tsx`
*   Create: `frontend/src/views/TrainView.tsx`

**Interfaces:**
*   Consumes: `LizardViewType`
*   Produces: `/custom` route that mounts `TrainView` dashboard.

- [ ] **Step 1: Enable Custom option card and link on Landing Page**

Modify [LandingPage.tsx](file:///Users/leyangloh/dev/LizardMorph/frontend/src/components/LandingPage.tsx) to enable navigation to `/custom` and render the Custom Model card:

```typescript
// Modify handleOptionClick (around line 163):
  const handleOptionClick = (viewType: LizardViewType) => {
    navigate(`/${viewType}`);
  };

// In LandingPageStyles.optionsContainer (around line 342), append the new card:
        {/* Train Custom Model */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...(hoveredCard === "custom" ? LandingPageStyles.optionCardHover : {}),
          }}
          onClick={() => handleOptionClick("custom")}
          onMouseEnter={() => handleMouseEnter("custom")}
          onMouseLeave={handleMouseLeave}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{
              ...LandingPageStyles.icon,
              ...(hoveredCard === "custom" ? LandingPageStyles.iconHover : {})
            }}>
              🧠
            </div>
            <h3 style={{
              ...LandingPageStyles.optionTitle,
              ...(hoveredCard === "custom" ? LandingPageStyles.optionTitleHover : {})
            }}>
              Train Custom Model
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Upload annotated datasets (images + TPS/XML) and train custom shape predictors locally.
            </p>
          </div>
        </div>
```

- [ ] **Step 2: Update routing in App.tsx**

Modify [App.tsx](file:///Users/leyangloh/dev/LizardMorph/frontend/src/App.tsx):
*   Import `TrainView` (to be created):
    ```typescript
    import { TrainView } from "./views/TrainView";
    ```
*   Update route for `/custom` (around line 63):
    ```typescript
    <Route path="/custom" element={<TrainView onNavigateHome={() => navigate("/")} />} />
    ```
    Wait, inside `App` router we need to wrap it:
    ```typescript
    // wrapper component helper or direct route
    const TrainViewWrapper: React.FC = () => {
      const navigate = useNavigate();
      return <TrainView onNavigateHome={() => navigate("/")} />;
    };
    
    // In Routes:
    <Route path="/custom" element={<TrainViewWrapper />} />
    ```

- [ ] **Step 3: Create TrainView.tsx skeleton**

Create `frontend/src/views/TrainView.tsx` with a basic view and back button:

```typescript
import React from "react";

interface Props {
  onNavigateHome: () => void;
}

export const TrainView: React.FC<Props> = ({ onNavigateHome }) => {
  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <button onClick={onNavigateHome}>← Back to Home</button>
      <h2>Train Custom Model</h2>
    </div>
  );
};
```

- [ ] **Step 4: Verify compiling**

Run: `cd frontend && npm run build`
Expected: Build passes with no TypeScript or style warnings.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/LandingPage.tsx frontend/src/App.tsx frontend/src/views/TrainView.tsx
git commit -m "feat(frontend): add Train Mode routing and view skeleton"
```

---

### Task 5: TrainView Dashboard, List Models, & Polling Loop

**Files:**
*   Modify/Update: `frontend/src/views/TrainView.tsx`

**Interfaces:**
*   Consumes: `ApiService` methods for list predictors, delete, train, status.
*   Produces: Full training and management dashboard.

- [ ] **Step 1: Implement TrainView dashboard layout & state**

Overwrite `frontend/src/views/TrainView.tsx` with the complete training form, list of predictors, and active polling:

```typescript
import React, { useState, useEffect, useRef } from "react";
import ApiService, { PredictorMeta } from "../services/ApiService";
import { useTheme } from "../contexts/ThemeContext";

interface Props {
  onNavigateHome: () => void;
}

export const TrainView: React.FC<Props> = ({ onNavigateHome }) => {
  const { resolved } = useTheme();
  const isDark = resolved === "dark";

  const [predictors, setPredictors] = useState<PredictorMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Training form state
  const [modelName, setModelName] = useState("Custom Predictor");
  const [zipFile, setZipFile] = useState<File | null>(null);
  
  // Training execution state
  const [isTraining, setIsTraining] = useState(false);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);

  const pollIntervalRef = useRef<number | null>(null);

  // Fetch current custom predictors list
  const fetchPredictors = async () => {
    setLoading(true);
    try {
      const res = await ApiService.listPredictors();
      if (res.success) {
        setPredictors(res.predictors);
      }
    } catch (err: any) {
      setError(`Failed to load predictors: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictors();
    return () => stopPolling();
  }, []);

  const stopPolling = () => {
    if (pollIntervalRef.current !== null) {
      window.clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  const handleTrainModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!zipFile || !modelName.trim()) return;

    setIsTraining(true);
    setProgressText("Uploading dataset ZIP file...");
    setError(null);

    try {
      const res = await ApiService.trainPredictor(modelName.trim(), zipFile);
      if (res.success && res.job_id) {
        setTrainingJobId(res.job_id);
        setProgressText("Training shape predictor... (this may take 10-40s)");

        pollIntervalRef.current = window.setInterval(async () => {
          try {
            const statusRes = await ApiService.getTrainStatus(res.job_id);
            if (statusRes.success) {
              if (statusRes.status === "completed" && statusRes.predictor) {
                stopPolling();
                setIsTraining(false);
                setProgressText(null);
                setTrainingJobId(null);
                setZipFile(null);
                setModelName("Custom Predictor");
                // Refresh list
                await fetchPredictors();
              } else if (statusRes.status === "failed") {
                stopPolling();
                setIsTraining(false);
                setProgressText(null);
                setTrainingJobId(null);
                setError(`Training failed: ${statusRes.error || "Unknown server error"}`);
              } else if (statusRes.status === "training") {
                setProgressText("Training model: learning shape predictor cascades...");
              }
            }
          } catch (pollErr: any) {
            stopPolling();
            setIsTraining(false);
            setProgressText(null);
            setTrainingJobId(null);
            setError(`Status polling check failed: ${pollErr.message || pollErr}`);
          }
        }, 2000);
      }
    } catch (err: any) {
      setIsTraining(false);
      setProgressText(null);
      setError(`Failed to initiate training: ${err.message || err}`);
    }
  };

  const handleDeletePredictor = async (id: string) => {
    if (!window.confirm("Are you sure you want to delete this custom model?")) return;
    try {
      const res = await ApiService.deletePredictor(id);
      if (res.success) {
        await fetchPredictors();
      }
    } catch (err: any) {
      setError(`Failed to delete model: ${err.message || err}`);
    }
  };

  // Styles
  const containerStyle: React.CSSProperties = {
    maxWidth: "1100px",
    margin: "0 auto",
    padding: "30px 20px",
    color: isDark ? "#e0e0e0" : "#111",
    fontFamily: "'Outfit', 'Inter', sans-serif",
  };

  const gridStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "1fr 1.2fr",
    gap: "30px",
    marginTop: "24px",
  };

  const cardStyle: React.CSSProperties = {
    background: isDark ? "rgba(30, 42, 58, 0.85)" : "rgba(255, 255, 255, 0.85)",
    border: `1px solid ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
    borderRadius: "16px",
    padding: "24px",
    boxShadow: `0 8px 32px ${isDark ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.06)"}`,
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "10px 12px",
    borderRadius: "10px",
    border: `1px solid ${isDark ? "rgba(255,255,255,0.14)" : "rgba(0,0,0,0.14)"}`,
    background: isDark ? "#2a3a4e" : "white",
    color: isDark ? "#e0e0e0" : "#111",
    outline: "none",
    marginTop: "6px",
    boxSizing: "border-box",
  };

  const buttonStyle: React.CSSProperties = {
    padding: "10px 18px",
    borderRadius: "10px",
    border: "none",
    background: "#4F7942",
    color: "white",
    fontWeight: 700,
    cursor: "pointer",
    width: "100%",
    marginTop: "16px",
  };

  return (
    <div style={containerStyle}>
      <button
        onClick={onNavigateHome}
        style={{
          background: "none",
          border: "none",
          color: isDark ? "#81c784" : "#2e7d32",
          cursor: "pointer",
          fontWeight: 700,
          fontSize: "14px",
          display: "flex",
          alignItems: "center",
          gap: "6px",
          padding: "6px 0",
        }}
      >
        ← Back to Overview
      </button>

      <h1 style={{ marginTop: "12px", fontSize: "28px", fontWeight: 800 }}>🧠 Custom Model Training</h1>
      <p style={{ opacity: 0.7, fontSize: "14px", marginTop: "4px" }}>
        Train new shape predictors using your own annotated datasets on your local CPU.
      </p>

      {error && (
        <div style={{ marginTop: "16px", padding: "12px 16px", background: "rgba(183, 28, 28, 0.15)", border: "1px solid rgba(183, 28, 28, 0.25)", color: "#e57373", borderRadius: "10px", fontSize: "14px", fontWeight: 600 }}>
          ⚠️ {error}
        </div>
      )}

      <div style={gridStyle}>
        {/* Left Side: Training Form */}
        <div style={cardStyle}>
          <h2 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "16px" }}>🛠️ Train New Predictor</h2>
          <form onSubmit={handleTrainModel}>
            <div style={{ marginBottom: "16px" }}>
              <label style={{ fontSize: "13px", fontWeight: 700, opacity: 0.8 }}>Model Display Name</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={isTraining}
                required
                style={inputStyle}
              />
            </div>

            <div style={{ marginBottom: "16px" }}>
              <label style={{ fontSize: "13px", fontWeight: 700, opacity: 0.8 }}>
                Dataset ZIP File (Images + TPS or XML)
              </label>
              <input
                type="file"
                accept=".zip"
                onChange={(e) => setZipFile(e.target.files?.[0] || null)}
                disabled={isTraining}
                required
                style={inputStyle}
              />
            </div>

            <button
              type="submit"
              disabled={isTraining || !zipFile || !modelName.trim()}
              style={{
                ...buttonStyle,
                opacity: isTraining || !zipFile || !modelName.trim() ? 0.55 : 1,
                cursor: isTraining || !zipFile || !modelName.trim() ? "not-allowed" : "pointer",
              }}
            >
              {isTraining ? "Training..." : "Start Local Training"}
            </button>
          </form>

          {isTraining && progressText && (
            <div style={{ marginTop: "16px", display: "flex", alignItems: "center", gap: "10px", color: isDark ? "#a5d6a7" : "#1b5e20", fontSize: "13px", fontWeight: 700 }}>
              <span className="spinner" style={{ display: "inline-block", width: "14px", height: "14px", border: "2px solid currentColor", borderRightColor: "transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
              <span>{progressText}</span>
            </div>
          )}
        </div>

        {/* Right Side: Predictors List */}
        <div style={cardStyle}>
          <h2 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "16px" }}>📚 Available Custom Models</h2>
          {loading && <p style={{ fontSize: "14px", opacity: 0.7 }}>Loading models...</p>}
          {!loading && predictors.length === 0 && (
            <p style={{ fontSize: "14px", opacity: 0.6, fontStyle: "italic" }}>
              No custom models trained yet. Upload a dataset to train your first model!
            </p>
          )}
          {!loading && predictors.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "12px", maxHeight: "400px", overflowY: "auto" }}>
              {predictors.map((p) => (
                <div
                  key={p.id}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "12px 14px",
                    background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)",
                    border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
                    borderRadius: "10px",
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "14px" }}>{p.display_name}</div>
                    <div style={{ fontSize: "11px", opacity: 0.6, marginTop: "2px" }}>
                      Points: {p.num_parts ?? "Unknown"} | Size: {(p.size_bytes / (1024 * 1024)).toFixed(2)} MB
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeletePredictor(p.id)}
                    style={{
                      background: "rgba(183, 28, 28, 0.1)",
                      border: "none",
                      color: "#ef5350",
                      padding: "6px 10px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "12px",
                      fontWeight: 700,
                    }}
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
```

- [ ] **Step 2: Verify compiling**

Run: `cd frontend && npm run build`
Expected: Build passes with no TypeScript or style warnings.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/views/TrainView.tsx
git commit -m "feat(frontend): implement TrainView layout, model listing, and deletion"
```


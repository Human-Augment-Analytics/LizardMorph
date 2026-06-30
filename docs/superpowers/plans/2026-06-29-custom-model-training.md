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

### Task 4: FreePredictorPanel Collapsible Train Section UI

**Files:**
*   Modify: `frontend/src/components/FreePredictorPanel.tsx`

**Interfaces:**
*   Consumes: `ApiService` methods
*   Produces: UI elements for zip file selection, input of model name, start training trigger

- [ ] **Step 1: Add new props to FreePredictorPanel**

Modify props interface in [FreePredictorPanel.tsx](file:///Users/leyangloh/dev/LizardMorph/frontend/src/components/FreePredictorPanel.tsx) to pass training handlers and states:

```typescript
interface Props {
  predictors: PredictorMeta[];
  selectedPredictorId: string | null;
  predictorsLoading: boolean;
  error: string | null;
  hasCurrentImage: boolean;
  onRefresh: () => void;
  onSelectPredictorId: (id: string | null) => void;
  onUploadPredictor: (file: File) => void;
  onDeleteSelected: () => void;
  onAutoplace: () => void;
  theme: ResolvedTheme;
  // New Training Props
  isTraining: boolean;
  trainingProgressText: string | null;
  onTrainModel: (modelName: string, zipFile: File) => void;
}
```

- [ ] **Step 2: Add training collapsible UI to component layout**

Add input states and UI block inside `FreePredictorPanel` rendering:

```typescript
export function FreePredictorPanel(props: Props) {
  const selected = props.selectedPredictorId;
  const canAutoplace = Boolean(selected) && props.hasCurrentImage && !props.predictorsLoading;
  const isBusy = props.predictorsLoading || props.isTraining;
  const isDark = props.theme === "dark";
  
  // Local state for the training inputs
  const [isTrainAccordionOpen, setIsTrainAccordionOpen] = React.useState(false);
  const [modelName, setModelName] = React.useState("Custom Predictor");
  const [zipFile, setZipFile] = React.useState<File | null>(null);

  // Styles
  // ... (keep original styles)
  const formRowStyle: React.CSSProperties = {
    display: "flex",
    gap: 12,
    alignItems: "end",
    marginTop: 12,
    flexWrap: "wrap",
    borderTop: `1px dashed ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
    paddingTop: 12,
  };
```

Add the collapsible training interface right above `props.error` section (around line 199):

```typescript
      {/* Training Collapsible Accordion */}
      <div style={{ marginTop: 14, borderTop: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}` }}>
        <button
          onClick={() => setIsTrainAccordionOpen(!isTrainAccordionOpen)}
          style={{
            background: "none",
            border: "none",
            width: "100%",
            textAlign: "left",
            padding: "10px 0 6px 0",
            fontWeight: 700,
            fontSize: 13,
            cursor: "pointer",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            color: isDark ? "#4F7942" : "#2E5A1C"
          }}
        >
          <span>🛠️ Train Custom Shape Predictor</span>
          <span>{isTrainAccordionOpen ? "▲" : "▼"}</span>
        </button>

        {isTrainAccordionOpen && (
          <div style={{ padding: "6px 0 10px 0" }}>
            <div style={{ fontSize: 12, marginBottom: 8, color: isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.65)" }}>
              Upload a <code>.zip</code> file containing training images and a <code>.tps</code> or <code>.xml</code> annotation file to train a new dlib model.
            </div>
            
            <div style={gridStyle}>
              <div style={groupStyle}>
                <div style={labelStyle}>Model Name</div>
                <input
                  type="text"
                  value={modelName}
                  disabled={isBusy}
                  onChange={(e) => setModelName(e.target.value)}
                  style={inputStyle}
                  placeholder="e.g. My Lizard Predictor"
                />
              </div>

              <div style={groupStyle}>
                <div style={labelStyle}>Dataset ZIP File</div>
                <input
                  type="file"
                  accept=".zip"
                  disabled={isBusy}
                  style={inputStyle}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) setZipFile(file);
                  }}
                />
              </div>

              <div style={groupStyle}>
                <div style={labelStyle}>&nbsp;</div>
                <button
                  onClick={() => {
                    if (zipFile && modelName.trim()) {
                      props.onTrainModel(modelName.trim(), zipFile);
                    }
                  }}
                  disabled={isBusy || !zipFile || !modelName.trim()}
                  style={isBusy || !zipFile || !modelName.trim() ? { ...primaryButton, opacity: 0.55, cursor: "not-allowed" } : primaryButton}
                >
                  {props.isTraining ? "Training…" : "Train Model"}
                </button>
              </div>
            </div>

            {props.isTraining && props.trainingProgressText && (
              <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8, color: isDark ? "#81c784" : "#2e7d32", fontWeight: 700, fontSize: 12 }}>
                <span className="spinner" style={{ display: "inline-block", width: 12, height: 12, border: "2px solid currentColor", borderRightColor: "transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
                <span>{props.trainingProgressText}</span>
              </div>
            )}
          </div>
        )}
      </div>
```

- [ ] **Step 3: Verify compiling**

Run: `cd frontend && npm run build`
Expected: Fail due to missing `onTrainModel`, `isTraining` in `MainView.tsx` props hookup. (This confirms Task 4 is correctly bound for Task 5).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/FreePredictorPanel.tsx
git commit -m "feat(frontend): add training collapsible UI to FreePredictorPanel"
```

---

### Task 5: MainView State Integration & Polling Loop

**Files:**
*   Modify: `frontend/src/views/MainView.tsx`

**Interfaces:**
*   Consumes: `ApiService` training methods, modified `FreePredictorPanel`
*   Produces: State trackers (`isTraining`, `trainingProgressText`), status poller trigger

- [ ] **Step 1: Add new state attributes to MainView**

In [MainView.tsx](file:///Users/leyangloh/dev/LizardMorph/frontend/src/views/MainView.tsx), update `State` interface:

```typescript
  isTraining: boolean;
  trainingProgressText: string | null;
  trainingJobId: string | null;
```

Initialize them in `constructor` (around line 125):

```typescript
    this.state = {
      // ... (existing state)
      isTraining: false,
      trainingProgressText: null,
      trainingJobId: null,
    };
```

- [ ] **Step 2: Add Train Model handlers and polling loops**

Define method handlers in `MainView` class:

```typescript
  // Polling reference
  trainingPollInterval: number | null = null;

  handleTrainModel = async (modelName: string, zipFile: File) => {
    this.setState({
      isTraining: true,
      trainingProgressText: "Uploading dataset and initializing training job...",
      predictorsError: null
    });

    try {
      const res = await ApiService.trainPredictor(modelName, zipFile);
      if (res.success && res.job_id) {
        this.setState({
          trainingJobId: res.job_id,
          trainingProgressText: "Training dlib shape predictor on server... (may take 10-30s)"
        });

        // Start polling status
        this.trainingPollInterval = window.setInterval(async () => {
          try {
            const statusRes = await ApiService.getTrainStatus(res.job_id);
            if (statusRes.success) {
              if (statusRes.status === "completed" && statusRes.predictor) {
                this.stopTrainingPoll();
                this.setState({
                  isTraining: false,
                  trainingProgressText: null,
                  trainingJobId: null,
                  freePredictorId: statusRes.predictor.id // Auto-select new model
                });
                // Refresh list of predictors
                await this.refreshPredictors();
              } else if (statusRes.status === "failed") {
                this.stopTrainingPoll();
                this.setState({
                  isTraining: false,
                  trainingProgressText: null,
                  trainingJobId: null,
                  predictorsError: `Training failed: ${statusRes.error || "Unknown server training error"}`
                });
              } else if (statusRes.status === "training") {
                this.setState({
                  trainingProgressText: "Training model: crunching landmarks..."
                });
              }
            }
          } catch (pollErr: any) {
            this.stopTrainingPoll();
            this.setState({
              isTraining: false,
              trainingProgressText: null,
              trainingJobId: null,
              predictorsError: `Status check failed: ${pollErr.message || pollErr}`
            });
          }
        }, 2000);
      } else {
        throw new Error("Job not accepted");
      }
    } catch (err: any) {
      this.setState({
        isTraining: false,
        trainingProgressText: null,
        predictorsError: `Failed to initiate training: ${err.message || err}`
      });
    }
  };

  stopTrainingPoll = () => {
    if (this.trainingPollInterval !== null) {
      window.clearInterval(this.trainingPollInterval);
      this.trainingPollInterval = null;
    }
  };

  // Add cleanup to componentWillUnmount
  componentWillUnmount() {
    this.stopTrainingPoll();
    // ... (any existing unmounting logic)
  }
```

- [ ] **Step 3: Connect new handlers to FreePredictorPanel rendering**

In the `render` method where `FreePredictorPanel` is mounted, pass the new states and handler:

```typescript
                    <FreePredictorPanel
                      predictors={this.state.availablePredictors}
                      selectedPredictorId={this.state.freePredictorId}
                      predictorsLoading={this.state.predictorsLoading}
                      error={this.state.predictorsError}
                      hasCurrentImage={Boolean(this.state.imageFilename)}
                      onRefresh={this.refreshPredictors}
                      onSelectPredictorId={(id) => this.setState({ freePredictorId: id })}
                      onUploadPredictor={this.handleUploadFreePredictor}
                      onDeleteSelected={this.handleDeleteSelectedFreePredictor}
                      onAutoplace={this.handleFreeAutoplace}
                      theme={theme}
                      // Pass training state & props
                      isTraining={this.state.isTraining}
                      trainingProgressText={this.state.trainingProgressText}
                      onTrainModel={this.handleTrainModel}
                    />
```

- [ ] **Step 4: Verify compiling**

Run: `cd frontend && npm run build`
Expected: Build passes with no TypeScript or styling warnings.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/views/MainView.tsx
git commit -m "feat(frontend): integrate custom model training state and polling loop in MainView"
```

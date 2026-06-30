# Custom Model Training Spec

This document details the architecture and implementation design for allowing users to train their own custom dlib shape predictor models directly inside LizardMorph.

---

## 1. Feature Overview

LizardMorph will allow users in **Free Mode** to upload an annotated dataset (as a `.zip` archive containing training images and a coordinate annotation file) to train a custom dlib shape predictor. 

The training process runs on the local Flask server using C++ `dlib` Python bindings, ensuring high performance without sending data to external servers. Because training can take anywhere from a few seconds to a minute, it will run asynchronously in a background thread on the server, and the frontend will poll for status updates.

---

## 2. Backend Design

We will add two new HTTP endpoints in [app.py](file:///Users/leyangloh/dev/LizardMorph/backend/app.py) and a training helper module in [utils.py](file:///Users/leyangloh/dev/LizardMorph/backend/utils.py).

### 2.1 Training State Tracker
A global dictionary or simple memory-based job registry will track background training jobs:
```python
# Job structure in memory
TRAINING_JOBS = {
    # "job_id": {"status": "pending"|"training"|"completed"|"failed", "error": str, "model_meta": dict}
}
```

### 2.2 Endpoints

#### `POST /train_predictor`
Initiates a background training task.
*   **Request Format**: `multipart/form-data`
    *   `model_name`: String (e.g., "my_lizard_predictor")
    *   `dataset`: File (ZIP archive containing `.jpg`/`.png` files and a `.tps` or dlib `.xml` file)
*   **Response**: `202 Accepted`
    ```json
    {
      "success": true,
      "job_id": "uuid-v4-string",
      "message": "Training job started"
    }
    ```

#### `GET /train_status/<job_id>`
Polls status of a running training job.
*   **Response**: `200 OK`
    ```json
    {
      "success": true,
      "status": "pending" | "training" | "completed" | "failed",
      "error": "Error details if failed",
      "predictor": { ... } // predictor metadata if completed
    }
    ```

### 2.3 Training Thread Logic
When a training job starts:
1.  **Extract ZIP**: Extract the archive to `backend/uploads/training_<job_id>/`.
2.  **Locate Annotations**: Locate a `.tps` or `.xml` file inside the extracted folder.
3.  **Process TPS**: If a `.tps` is found, parse it using `utils.read_tps`.
    *   For each image referenced in the TPS, find its path and retrieve its height and width using `cv2.imread`.
    *   Generate a dlib-compliant XML dataset using a modified version of `utils.generate_dlib_xml` (targeting paths in the temp folder).
4.  **Process XML**: If a dlib XML is found, parse it and update image paths to match their absolute location in the temporary directory.
5.  **Dlib Training**:
    *   Create a dlib shape predictor training options object:
        ```python
        options = dlib.shape_predictor_training_options()
        options.nu = 0.1
        options.tree_depth = 4
        options.oversampling_amount = 5
        options.cascade_depth = 15
        options.be_verbose = False
        ```
    *   Execute training: `dlib.train_shape_predictor(temp_xml_path, output_model_path, options)`.
6.  **Add Predictor**:
    *   Move the trained `.dat` file into `PREDICTOR_LIBRARY_FILES` under a unique filename (`<predictor_id>.dat`).
    *   Add metadata to `PREDICTOR_LIBRARY_INDEX` (`predictors.json`) using `predictor_library.add_predictor`.
7.  **Clean Up**: Delete the temporary unzipped directory `backend/uploads/training_<job_id>/`.

---

## 3. Frontend Design

### 3.1 Model Training UI Section
We will extend [FreePredictorPanel.tsx](file:///Users/leyangloh/dev/LizardMorph/frontend/src/components/FreePredictorPanel.tsx) to render a collapsible training accordion or section under the predictor list.

```
+------------------------------------------------------+
| Predictor (Free mode)                                |
| ... (Predictor List, Upload Dat, Delete)             |
+------------------------------------------------------+
| > Train Custom Model                                 |
|   Model Name: [ My Lizard Predictor               ]  |
|   Dataset ZIP: [ Choose File (images + TPS/XML)   ]  |
|   [ Train Model ]                                    |
+------------------------------------------------------+
```

### 3.2 MainView State & Polling Loop
We will update [MainView.tsx](file:///Users/leyangloh/dev/LizardMorph/frontend/src/views/MainView.tsx):
1.  **State**:
    *   `trainingJobId: string | null`
    *   `trainingStatus: "idle" | "training" | "success" | "error"`
    *   `trainingError: string | null`
2.  **Submit Training Handlers**:
    *   Make an HTTP POST to `/api/train_predictor` with the ZIP and name.
    *   If accepted, set `trainingJobId` and `trainingStatus = "training"` and start a `setInterval` to check status every 2 seconds.
3.  **Polling**:
    *   Request `/api/train_status/<job_id>`.
    *   If status is `completed`, clear interval, set status to `success`, refresh the predictors list, and auto-select the new predictor.
    *   If status is `failed`, clear interval, set status to `error`, and display the error message.

---

## 4. Verification & Testing

### 4.1 Integration Test Dataset
We will create a small mock training dataset in `backend/tests/test_training_dataset.zip` consisting of:
*   3 small placeholder/black images.
*   A `test.tps` containing landmark coordinates for those images.

### 4.2 Backend Unit Tests
Add a test suite `backend/tests/test_training.py` checking:
*   Successful extraction and XML generation from a TPS ZIP.
*   Training logic execution using a mock/small dlib training configuration.
*   Correct addition of the trained model into the predictor library index.

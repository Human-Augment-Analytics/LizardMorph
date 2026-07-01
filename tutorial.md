# 🎓 How to Train a Custom Shape Predictor Model

This tutorial guides you through preparing your annotated dataset and training a custom dlib shape predictor model locally on your CPU using the LizardMorph interface.

---

## 📹 Video Walkthrough
Below is a video recording showing the entire process of uploading the *Drosophila* wing dataset and training the custom model in LizardMorph:

![LizardMorph Custom Model Training Walkthrough](drosophila_training.webm)

---

## 📋 Step-by-Step Training Guide

### Step 1: Prepare your Dataset ZIP File
To train a custom shape predictor, you must package your images and their coordinates into a single flat ZIP file. The ZIP archive must satisfy the following format:

1.  **Unique Image Files**: Include all specimen images (e.g., `.jpg`, `.jpeg`, `.png`) flatly in the root of the archive.
2.  **Exactly One Annotation File**: Include exactly one coordinate file in the root:
    *   **TPS Format (`.tps`)**: A standard Thin Plate Spline file containing landmark listings mapped to corresponding image files (e.g., `IMAGE=63001.jpg`).
    *   **XML Format (`.xml`)**: A standard dlib training XML file (in dlib dataset format) referencing the images.
3.  **Flat Directory Structure**: Ensure no nested subdirectories are present.

> [!IMPORTANT]
> To prevent Denial-of-Service or resources exhaustion, the application enforces the following safety limits on ZIP uploads:
> *   Maximum ZIP upload size: **100 MB**
> *   Maximum file count inside the ZIP: **500 files**
> *   Maximum uncompressed size of any single file: **20 MB**
> *   Maximum total uncompressed size: **100 MB**

---

### Step 2: Open the Training Dashboard
1.  Launch the LizardMorph web app or desktop application.
2.  On the **Overview** page, scroll to the bottom and click the **Train Custom Model** button.
3.  This opens the `/custom` route (the Custom Model Training Dashboard).

---

### Step 3: Configure and Upload
1.  **Model Display Name**: Enter a descriptive name for your custom shape predictor (e.g., `Drosophila Wing Predictor`).
2.  **Upload Dataset**: Drag-and-drop your prepared dataset `.zip` file into the upload dropzone, or click the dropzone to browse your local filesystem.

---

### Step 4: Start Training
1.  Click the **Start Local Training** button.
2.  The application will upload the ZIP archive and trigger a CPU training process in a background thread.
3.  The dashboard will show a **"Training..."** progress bar and report the active background Job ID. 
4.  *Note: Only one local training job is allowed to execute at any given time. If multiple users/tabs submit jobs, they will be queued automatically.*

---

### Step 5: Verification & Deletion
*   Once completed, the newly trained predictor model will be registered in the available custom models database.
*   The model will appear in the **Available Custom Models** list displaying its name, file size, and target landmark count.
*   To delete a model from the system, click the **Delete** button next to the model entry.

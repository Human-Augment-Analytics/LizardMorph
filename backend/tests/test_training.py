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
        img1_path = os.path.join(temp_workspace, "img1.jpg")
        img2_path = os.path.join(temp_workspace, "img2.jpg")
        cv2.imwrite(img1_path, img)
        cv2.imwrite(img2_path, img)
        
        z.write(img1_path, arcname="img1.jpg")
        z.write(img2_path, arcname="img2.jpg")
        
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


def test_train_predictor_cleanup_on_invalid_zip(temp_workspace):
    zip_path = os.path.join(temp_workspace, "invalid.zip")
    with open(zip_path, "wb") as f:
        f.write(b"not a zip file")
        
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    predictor_id = "failed-uuid"
    expected_temp_dir = os.path.join(temp_workspace, f"training_{predictor_id}")
    
    with pytest.raises(Exception):
        train_predictor_from_zip(
            model_name="Test",
            zip_path=zip_path,
            predictor_id=predictor_id,
            index_path=index_path,
            files_dir=files_dir
        )
        
    assert not os.path.exists(expected_temp_dir)


def test_train_predictor_cleanup_dat_on_validation_failure(temp_workspace, monkeypatch):
    # Setup a valid dataset
    zip_path = os.path.join(temp_workspace, "dataset.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_path = os.path.join(temp_workspace, "img1.jpg")
        cv2.imwrite(img1_path, img)
        z.write(img1_path, arcname="img1.jpg")
        os.remove(img1_path)
        
        tps_content = """LM=1
10.0 90.0
IMAGE=img1.jpg
ID=0
"""
        z.writestr("annotations.tps", tps_content)

    mock_options = {
        "nu": 0.1,
        "tree_depth": 2,
        "cascade_depth": 5
    }
    
    # Mock dlib.shape_predictor to raise a ValueError during validation
    import dlib
    def mock_sp(path):
        raise ValueError("Simulated validation failure")
        
    monkeypatch.setattr(dlib, "shape_predictor", mock_sp)
    
    predictor_id = "test-fail-uuid"
    expected_dat_path = os.path.join(files_dir, f"{predictor_id}.dat")
    
    with pytest.raises(ValueError, match="Simulated validation failure"):
        train_predictor_from_zip(
            model_name="Test Predictor Fail",
            zip_path=zip_path,
            predictor_id=predictor_id,
            index_path=index_path,
            files_dir=files_dir,
            custom_options=mock_options
        )
        
    # Verify the .dat file is deleted
    assert not os.path.exists(expected_dat_path)


def test_train_predictor_multiple_annotations_error(temp_workspace):
    zip_path = os.path.join(temp_workspace, "dataset_multi.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("annotations1.tps", "LM=0\n")
        z.writestr("annotations2.xml", "<dataset></dataset>")
        
    with pytest.raises(ValueError, match="Multiple annotation files"):
        train_predictor_from_zip(
            model_name="Test Multi",
            zip_path=zip_path,
            predictor_id="test-multi-uuid",
            index_path=index_path,
            files_dir=files_dir
        )


def test_train_predictor_cleanup_on_training_failure(temp_workspace, monkeypatch):
    zip_path = os.path.join(temp_workspace, "dataset_train_fail.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_path = os.path.join(temp_workspace, "img1.jpg")
        cv2.imwrite(img1_path, img)
        z.write(img1_path, arcname="img1.jpg")
        os.remove(img1_path)
        
        tps_content = "LM=1\n10.0 90.0\nIMAGE=img1.jpg\nID=0\n"
        z.writestr("annotations.tps", tps_content)

    import dlib
    def mock_train(xml, out, opt):
        # Create a dummy file to simulate training starting to output something
        with open(out, "w") as f:
            f.write("partial data")
        raise RuntimeError("Simulated training crash")
        
    monkeypatch.setattr(dlib, "train_shape_predictor", mock_train)
    
    predictor_id = "test-train-fail-uuid"
    expected_dat_path = os.path.join(files_dir, f"{predictor_id}.dat")
    
    with pytest.raises(RuntimeError, match="Simulated training crash"):
        train_predictor_from_zip(
            model_name="Test Train Fail",
            zip_path=zip_path,
            predictor_id=predictor_id,
            index_path=index_path,
            files_dir=files_dir
        )
        
    assert not os.path.exists(expected_dat_path)


def test_train_predictor_xml_none_file_ignored(temp_workspace):
    zip_path = os.path.join(temp_workspace, "dataset_xml_none.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        # Save a valid image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_path = os.path.join(temp_workspace, "img1.jpg")
        cv2.imwrite(img1_path, img)
        z.write(img1_path, arcname="img1.jpg")
        os.remove(img1_path)
        
        # XML containing one image without 'file' attribute, and one valid image
        xml_content = """<?xml version='1.0' encoding='UTF-8'?>
<dataset>
  <name>Test XML None</name>
  <images>
    <image>
      <box top="10" left="10" width="80" height="80">
        <part name="0" x="20" y="20"/>
      </box>
    </image>
    <image file="img1.jpg">
      <box top="10" left="10" width="80" height="80">
        <part name="0" x="20" y="20"/>
      </box>
    </image>
  </images>
</dataset>
"""
        z.writestr("annotations.xml", xml_content)
        
    mock_options = {
        "nu": 0.1,
        "tree_depth": 2,
        "cascade_depth": 5
    }
    
    meta = train_predictor_from_zip(
        model_name="Test XML None",
        zip_path=zip_path,
        predictor_id="test-xml-none-uuid",
        index_path=index_path,
        files_dir=files_dir,
        custom_options=mock_options
    )
    
    assert meta is not None
    assert meta["num_parts"] == 1


def test_train_predictor_ignores_macosx_metadata(temp_workspace):
    zip_path = os.path.join(temp_workspace, "dataset_macos.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_path = os.path.join(temp_workspace, "img1.jpg")
        cv2.imwrite(img1_path, img)
        z.write(img1_path, arcname="img1.jpg")
        os.remove(img1_path)
        
        # Add a __MACOSX directory structure and some ._ files
        z.writestr("__MACOSX/._img1.jpg", b"dummy mac meta")
        z.writestr("__MACOSX/annotations.tps", b"dummy mac meta")
        z.writestr("._annotations.tps", b"dummy mac meta")
        
        # Real annotations
        tps_content = """LM=1
10.0 90.0
IMAGE=img1.jpg
ID=0
"""
        z.writestr("annotations.tps", tps_content)
        
    mock_options = {
        "nu": 0.1,
        "tree_depth": 2,
        "cascade_depth": 5
    }
    
    meta = train_predictor_from_zip(
        model_name="Test MacOS",
        zip_path=zip_path,
        predictor_id="test-macos-uuid",
        index_path=index_path,
        files_dir=files_dir,
        custom_options=mock_options
    )
    
    assert meta is not None
    assert meta["num_parts"] == 1


def test_train_predictor_resolves_absolute_paths(temp_workspace):
    zip_path = os.path.join(temp_workspace, "dataset_abs.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_path = os.path.join(temp_workspace, "img1.jpg")
        cv2.imwrite(img1_path, img)
        z.write(img1_path, arcname="img1.jpg")
        os.remove(img1_path)
        
        tps_content = """LM=1
10.0 90.0
IMAGE=img1.jpg
ID=0
"""
        z.writestr("annotations.tps", tps_content)
        
    mock_options = {
        "nu": 0.1,
        "tree_depth": 2,
        "cascade_depth": 5
    }
    
    import dlib
    import xml.etree.ElementTree as ET
    
    xml_inspected_path = [None]
    
    orig_train = dlib.train_shape_predictor
    def mock_train(xml_path, out_path, options):
        xml_inspected_path[0] = xml_path
        # Parse XML to check if image path is absolute
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for image in root.findall(".//image"):
            file_val = image.get("file")
            assert os.path.isabs(file_val), f"Path should be absolute: {file_val}"
        
        orig_train(xml_path, out_path, options)
            
    # Use temporary override
    import sys
    dlib.train_shape_predictor = mock_train
    try:
        meta = train_predictor_from_zip(
            model_name="Test Absolute Path",
            zip_path=zip_path,
            predictor_id="test-abs-uuid",
            index_path=index_path,
            files_dir=files_dir,
            custom_options=mock_options
        )
    finally:
        dlib.train_shape_predictor = orig_train
        
    assert meta is not None
    assert xml_inspected_path[0] is not None


def test_train_predictor_finally_block_ignores_errors(temp_workspace, monkeypatch):
    zip_path = os.path.join(temp_workspace, "dataset_finally.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("not_an_annotation.txt", "garbage")
        
    # Mock shutil.rmtree to raise an exception when ignore_errors is False
    import shutil
    def mock_rmtree(path, ignore_errors=False):
        if not ignore_errors:
            raise OSError("Simulated rmtree failure")
            
    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)
    
    with pytest.raises(ValueError, match="No .tps or .xml annotation file found"):
        train_predictor_from_zip(
            model_name="Test Finally",
            zip_path=zip_path,
            predictor_id="test-finally-uuid",
            index_path=index_path,
            files_dir=files_dir
        )


def test_train_predictor_zip_slip(temp_workspace):
    zip_path = os.path.join(temp_workspace, "dataset_zipslip.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("../traversal.tps", "LM=0\n")
        
    with pytest.raises(ValueError, match="Directory traversal attempt detected"):
        train_predictor_from_zip(
            model_name="Test Zip Slip",
            zip_path=zip_path,
            predictor_id="test-zipslip-uuid",
            index_path=index_path,
            files_dir=files_dir
        )


def test_train_predictor_duplicate_filename(temp_workspace):
    zip_path = os.path.join(temp_workspace, "dataset_duplicate.zip")
    index_path = os.path.join(temp_workspace, "predictors.json")
    files_dir = os.path.join(temp_workspace, "files")
    os.makedirs(files_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("folder1/annotation.tps", "LM=0\n")
        z.writestr("folder2/annotation.tps", "LM=0\n")
        
    with pytest.raises(ValueError, match="Duplicate filename 'annotation.tps' found in the archive"):
        train_predictor_from_zip(
            model_name="Test Duplicate",
            zip_path=zip_path,
            predictor_id="test-dup-uuid",
            index_path=index_path,
            files_dir=files_dir
        )


def test_api_train_predictor_lifecycle(temp_workspace):
    # Setup app test client
    from app import app, TRAINING_JOBS
    import io
    app.config['TESTING'] = True
    
    # Temporarily override predictor library constants
    import app as app_module
    old_dir = app_module.PREDICTOR_LIBRARY_DIR
    old_idx = app_module.PREDICTOR_LIBRARY_INDEX
    old_files = app_module.PREDICTOR_LIBRARY_FILES
    
    app_module.PREDICTOR_LIBRARY_DIR = temp_workspace
    app_module.PREDICTOR_LIBRARY_INDEX = os.path.join(temp_workspace, "predictors.json")
    app_module.PREDICTOR_LIBRARY_FILES = os.path.join(temp_workspace, "files")
    os.makedirs(app_module.PREDICTOR_LIBRARY_FILES, exist_ok=True)

    try:
        # Create a mock zip payload
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as z:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            t1_path = os.path.join(temp_workspace, "t1.jpg")
            cv2.imwrite(t1_path, img)
            z.write(t1_path, arcname="t1.jpg")
            os.remove(t1_path)
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
        app_module.PREDICTOR_LIBRARY_DIR = old_dir
        app_module.PREDICTOR_LIBRARY_INDEX = old_idx
        app_module.PREDICTOR_LIBRARY_FILES = old_files


def test_api_train_predictor_job_pruning(tmp_path):
    from app import app
    import app as app_module
    import json
    import time
    
    # Override JOBS_DIR to a temporary folder
    old_jobs_dir = app_module.JOBS_DIR
    temp_jobs_dir = os.path.join(str(tmp_path), "jobs")
    app_module.JOBS_DIR = temp_jobs_dir
    os.makedirs(temp_jobs_dir, exist_ok=True)
    
    try:
        # Create helper to write job status file
        def write_job(job_id, status, error, predictor, age):
            filepath = os.path.join(temp_jobs_dir, f"job_{job_id}.json")
            with open(filepath, "w") as f:
                json.dump({
                    "status": status,
                    "error": error,
                    "predictor": predictor,
                    "created_at": time.time() - age
                }, f)
            # Set the file modification time to match the simulated age
            mtime = time.time() - age
            os.utime(filepath, (mtime, mtime))
            
        # 1. Job 1: completed and older than 3600s
        write_job("job-old-completed", "completed", None, {}, 4000)
        # 2. Job 2: failed and older than 3600s
        write_job("job-old-failed", "failed", "Some error", None, 5000)
        # 3. Job 3: completed but recent (not older than 3600s)
        write_job("job-recent-completed", "completed", None, {}, 100)
        # 4. Job 4: pending and older than 3600s
        write_job("job-old-pending", "pending", None, None, 4000)
        
        with app.test_client() as client:
            # Triggering the route (even if it returns a 400 bad request due to missing file)
            resp = client.post("/train_predictor", data={})
            assert resp.status_code == 400
            
        # Verify old completed/failed jobs are pruned
        assert not os.path.exists(os.path.join(temp_jobs_dir, "job_job-old-completed.json"))
        assert not os.path.exists(os.path.join(temp_jobs_dir, "job_job-old-failed.json"))
        
        # Verify recent jobs and non-completed/failed jobs are kept
        assert os.path.exists(os.path.join(temp_jobs_dir, "job_job-recent-completed.json"))
        assert os.path.exists(os.path.join(temp_jobs_dir, "job_job-old-pending.json"))
        
    finally:
        app_module.JOBS_DIR = old_jobs_dir


def test_split_dlib_dataset(temp_workspace):
    from utils import split_dlib_dataset
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(temp_workspace, "dataset.xml")
    
    # Create a dummy XML dataset with 5 images
    root = ET.Element("dataset")
    ET.SubElement(root, "name").text = "Test Model"
    images_container = ET.SubElement(root, "images")
    
    for i in range(5):
        img_el = ET.SubElement(images_container, "image", file=f"img{i}.jpg")
        box_el = ET.SubElement(img_el, "box", top="1", left="1", width="10", height="10")
        ET.SubElement(box_el, "part", name="0", x="5", y="5")
        
    ET.ElementTree(root).write(xml_path, encoding="utf-8")
    
    # Split with 40% (2 test images, 3 train images)
    train_xml, test_xml = split_dlib_dataset(xml_path, test_ratio=0.4, seed=42)
    
    assert train_xml is not None
    assert test_xml is not None
    assert os.path.exists(train_xml)
    assert os.path.exists(test_xml)
    
    train_tree = ET.parse(train_xml)
    train_images = train_tree.getroot().find("images").findall("image")
    assert len(train_images) == 3
    
    test_tree = ET.parse(test_xml)
    test_images = test_tree.getroot().find("images").findall("image")
    assert len(test_images) == 2





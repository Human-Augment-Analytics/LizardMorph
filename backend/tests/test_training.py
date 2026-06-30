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


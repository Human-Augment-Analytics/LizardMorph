import os
import time
import shutil
import tempfile
import cv2
import numpy as np
import pytest
import xml.etree.ElementTree as ET

from backend.datasets.canonical import (
    CanonicalDataset,
    CanonicalImage,
    CanonicalObject,
    LandmarkPoint,
)
from backend.training.yolo_trainer import YoloOBBTrainer
from backend.training.ml_morph_trainer import MLMorphTrainer
from backend.training.orchestration import TrainingOrchestrator


@pytest.fixture
def temp_workspace():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_canonical_dataset(temp_workspace):
    img_dir = os.path.join(temp_workspace, "sample_images")
    os.makedirs(img_dir, exist_ok=True)

    img_path1 = os.path.join(img_dir, "img1.jpg")
    img_path2 = os.path.join(img_dir, "img2.jpg")

    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.imwrite(img_path1, dummy_img)
    cv2.imwrite(img_path2, dummy_img)

    obj1 = CanonicalObject(
        object_id="obj_1",
        class_name="lizard",
        obb=[50.0, 50.0, 40.0, 40.0, 0.0],
        landmarks=[
            LandmarkPoint(name="snout", x=40.0, y=40.0),
            LandmarkPoint(name="eye", x=60.0, y=40.0),
            LandmarkPoint(name="tail", x=50.0, y=60.0),
        ],
    )
    obj2 = CanonicalObject(
        object_id="obj_2",
        class_name="lizard",
        obb=[50.0, 50.0, 30.0, 30.0, 0.0],
        landmarks=[
            LandmarkPoint(name="snout", x=45.0, y=45.0),
            LandmarkPoint(name="eye", x=55.0, y=45.0),
            LandmarkPoint(name="tail", x=50.0, y=55.0),
        ],
    )

    cimg1 = CanonicalImage(
        image_id="img_1",
        file_path=img_path1,
        width=100,
        height=100,
        objects=[obj1],
    )
    cimg2 = CanonicalImage(
        image_id="img_2",
        file_path=img_path2,
        width=100,
        height=100,
        objects=[obj2],
    )

    return CanonicalDataset(images=[cimg1, cimg2])


def test_yolo_obb_dataset_formatting(temp_workspace, sample_canonical_dataset):
    job_dir = os.path.join(temp_workspace, "yolo_job")
    trainer = YoloOBBTrainer(job_dir)

    yaml_path = trainer.prepare_dataset(sample_canonical_dataset, train_ratio=0.5)
    assert os.path.exists(yaml_path)
    assert os.path.basename(yaml_path) == "data.yaml"

    yolo_dir = os.path.join(job_dir, "yolo_dataset")
    assert os.path.exists(os.path.join(yolo_dir, "images", "train"))
    assert os.path.exists(os.path.join(yolo_dir, "labels", "train"))

    # Inspect one label file
    labels_train = os.listdir(os.path.join(yolo_dir, "labels", "train"))
    assert len(labels_train) > 0
    label_path = os.path.join(yolo_dir, "labels", "train", labels_train[0])
    with open(label_path, "r") as f:
        content = f.read().strip()
    parts = content.split()
    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values)
    assert len(parts) == 9
    assert parts[0] == "0"

    # Test mock training
    metrics = trainer.train(yaml_path, mock=True)
    assert "detector_weights" in metrics
    assert os.path.exists(metrics["detector_weights"])


def test_ml_morph_trainer_dlib_xml_and_training(temp_workspace, sample_canonical_dataset):
    job_dir = os.path.join(temp_workspace, "ml_morph_job")
    trainer = MLMorphTrainer(job_dir)

    xml_path = trainer.prepare_landmark_dataset(
        sample_canonical_dataset, jitter_padding=0.2, box_jitter=0.05
    )
    assert os.path.exists(xml_path)

    # Verify dlib XML structure
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "dataset"
    images = root.findall(".//image")
    assert len(images) == 2
    box = images[0].find("box")
    assert box is not None
    parts = box.findall("part")
    assert len(parts) == 3

    # Test dlib training
    fast_opts = {"nu": 0.1, "tree_depth": 2, "cascade_depth": 2}
    metrics = trainer.train(xml_path, custom_options=fast_opts)
    assert "landmark_model" in metrics
    assert os.path.exists(metrics["landmark_model"])


def test_training_orchestrator_job_lifecycle(temp_workspace, sample_canonical_dataset):
    runs_dir = os.path.join(temp_workspace, "runs")
    orchestrator = TrainingOrchestrator(runs_dir)

    job_config = {
        "mock": True,
        "epochs": 1,
        "dlib_options": {"nu": 0.1, "tree_depth": 2, "cascade_depth": 2},
    }

    job_id = orchestrator.submit_job(
        project_id="test_project_1",
        dataset_dict=sample_canonical_dataset.to_dict(),
        config=job_config,
    )

    assert job_id is not None

    # Poll status until finished or timeout
    start_time = time.time()
    final_status = None
    while time.time() - start_time < 15.0:
        status_info = orchestrator.get_status(job_id)
        if status_info["status"] in ("completed", "failed"):
            final_status = status_info
            break
        time.sleep(0.2)

    assert final_status is not None, "Job timed out"
    assert final_status["status"] == "completed", f"Job failed: {final_status}"
    assert final_status["stage"] == "Ready"
    assert final_status["progress"] == 1.0

    # Verify output bundle manifest
    job_dir = os.path.join(runs_dir, job_id)
    manifest_path = os.path.join(job_dir, "manifest.json")
    assert os.path.exists(manifest_path)


def test_training_orchestrator_job_cancellation(temp_workspace, sample_canonical_dataset):
    runs_dir = os.path.join(temp_workspace, "runs")
    orchestrator = TrainingOrchestrator(runs_dir)

    job_config = {
        "mock": True,
        "simulate_delay": 5.0,  # Sleep in runner to allow cancellation test
    }

    job_id = orchestrator.submit_job(
        project_id="test_project_cancel",
        dataset_dict=sample_canonical_dataset.to_dict(),
        config=job_config,
    )

    time.sleep(0.3)
    cancelled = orchestrator.cancel_job(job_id)
    assert cancelled is True

    status_info = orchestrator.get_status(job_id)
    assert status_info["status"] == "cancelled"

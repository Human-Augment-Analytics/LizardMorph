import os
import tempfile
import pytest
from backend.domain.models import (
    Project,
    ModelVersion,
    Manifest,
    DetectorConfig,
    ClassConfig,
    LandmarkSchemaConfig,
)
from backend.storage.db import DatabaseManager
from backend.storage.repository import ModelRegistryRepository, ProjectRepository


def test_db_initialization_and_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_mgr = DatabaseManager(db_path)
        db_mgr.init_db()

        assert os.path.exists(db_path)

        with db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert "projects" in tables
            assert "model_versions" in tables
            assert "training_runs" in tables


def test_project_repository():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_mgr = DatabaseManager(db_path)
        db_mgr.init_db()

        proj_repo = ProjectRepository(db_mgr)
        proj = proj_repo.create_project(name="Drosophila Wing", organism="Drosophila")
        assert proj.id is not None
        assert proj.name == "Drosophila Wing"
        assert proj.organism == "Drosophila"
        assert proj.created_at is not None

        fetched = proj_repo.get_project(proj.id)
        assert fetched is not None
        assert fetched.id == proj.id
        assert fetched.name == "Drosophila Wing"
        assert fetched.organism == "Drosophila"

        non_existent = proj_repo.get_project("non_existent_id")
        assert non_existent is None


def test_manifest_serialization_deserialization():
    manifest = Manifest(
        schema_version=1,
        id="wing-v1",
        name="Drosophila Wing Pipeline",
        description="YOLO OBB detector + ML-morph predictor",
        detector=DetectorConfig(artifact="detector/best.pt", geometry="obb", confidence=0.25, iou=0.45),
        classes=[
            ClassConfig(
                id=0,
                name="wing",
                landmark_schema="wing-v1",
                predictor="predictors/wing-v1.dat",
                crop_padding=0.2,
            )
        ],
        landmark_schemas={"wing-v1": LandmarkSchemaConfig(points=["base", "tip"])},
        evaluation={"mAP50": 0.95},
    )

    manifest_dict = manifest.to_dict()
    assert manifest_dict["schema_version"] == 1
    assert manifest_dict["id"] == "wing-v1"
    assert manifest_dict["detector"]["geometry"] == "obb"
    assert manifest_dict["classes"][0]["name"] == "wing"
    assert manifest_dict["landmark_schemas"]["wing-v1"]["points"] == ["base", "tip"]
    assert manifest_dict["evaluation"] == {"mAP50": 0.95}

    reconstructed = Manifest.from_dict(manifest_dict)
    assert reconstructed.id == manifest.id
    assert reconstructed.name == manifest.name
    assert reconstructed.detector.artifact == manifest.detector.artifact
    assert reconstructed.classes[0].predictor == manifest.classes[0].predictor
    assert reconstructed.landmark_schemas["wing-v1"].points == ["base", "tip"]
    assert reconstructed.evaluation == {"mAP50": 0.95}


def test_model_registry_repository():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_mgr = DatabaseManager(db_path)
        db_mgr.init_db()

        models_dir = os.path.join(tmpdir, "models")
        repo = ModelRegistryRepository(db_mgr, models_dir=models_dir)

        manifest = Manifest(
            schema_version=1,
            id="wing-v1",
            name="Drosophila Wing Pipeline",
            description="YOLO OBB detector + ML-morph predictor",
            detector=DetectorConfig(artifact="detector/best.pt", geometry="obb", confidence=0.25),
            classes=[
                ClassConfig(
                    id=0,
                    name="wing",
                    landmark_schema="wing-v1",
                    predictor="predictors/wing-v1.dat",
                )
            ],
            landmark_schemas={"wing-v1": LandmarkSchemaConfig(points=["base", "tip"])},
        )

        model_version = repo.register_model_bundle("proj_1", manifest)
        assert model_version.id == "wing-v1"
        assert model_version.project_id == "proj_1"
        assert model_version.name == "Drosophila Wing Pipeline"
        assert model_version.manifest.id == "wing-v1"

        fetched = repo.get_model_version("wing-v1")
        assert fetched is not None
        assert fetched.id == "wing-v1"
        assert fetched.project_id == "proj_1"
        assert fetched.manifest.detector.confidence == 0.25
        assert fetched.manifest.landmark_schemas["wing-v1"].points == ["base", "tip"]

        all_models = repo.list_model_versions()
        assert len(all_models) == 1
        assert all_models[0].id == "wing-v1"

        non_existent = repo.get_model_version("non_existent")
        assert non_existent is None

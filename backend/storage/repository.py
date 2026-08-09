import datetime
import json
from typing import List, Optional
import uuid

from backend.domain.models import Manifest, ModelVersion, Project
from backend.storage.db import DatabaseManager


class ProjectRepository:
    def __init__(self, db_mgr: DatabaseManager):
        self.db_mgr = db_mgr

    def create_project(self, name: str, organism: str) -> Project:
        proj_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (id, name, organism, created_at) VALUES (?, ?, ?, ?)",
                (proj_id, name, organism, created_at),
            )
            conn.commit()
        return Project(
            id=proj_id, name=name, organism=organism, created_at=created_at
        )

    def get_project(self, proj_id: str) -> Optional[Project]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, organism, created_at FROM projects WHERE id = ?",
                (proj_id,),
            )
            row = cursor.fetchone()
            if row:
                return Project(
                    id=row["id"],
                    name=row["name"],
                    organism=row["organism"],
                    created_at=row["created_at"],
                )
    def list_projects(self) -> List[Project]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, organism, created_at FROM projects"
            )
            rows = cursor.fetchall()
            return [
                Project(
                    id=row["id"],
                    name=row["name"],
                    organism=row["organism"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]



class ModelRegistryRepository:
    def __init__(self, db_mgr: DatabaseManager, models_dir: str = "models"):
        self.db_mgr = db_mgr
        self.models_dir = models_dir

    def register_model_bundle(
        self, project_id: str, manifest: Manifest
    ) -> ModelVersion:
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest_json = json.dumps(manifest.to_dict())
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO model_versions (id, project_id, name, created_at, manifest_json) VALUES (?, ?, ?, ?, ?)",
                (manifest.id, project_id, manifest.name, created_at, manifest_json),
            )
            conn.commit()
        return ModelVersion(
            id=manifest.id,
            project_id=project_id,
            name=manifest.name,
            created_at=created_at,
            manifest=manifest,
        )

    def get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, project_id, name, created_at, manifest_json FROM model_versions WHERE id = ?",
                (model_id,),
            )
            row = cursor.fetchone()
            if row:
                manifest_dict = json.loads(row["manifest_json"])
                manifest = Manifest.from_dict(manifest_dict)
                return ModelVersion(
                    id=row["id"],
                    project_id=row["project_id"],
                    name=row["name"],
                    created_at=row["created_at"],
                    manifest=manifest,
                )
        return None

    def list_model_versions(self) -> List[ModelVersion]:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, project_id, name, created_at, manifest_json FROM model_versions"
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                manifest_dict = json.loads(row["manifest_json"])
                manifest = Manifest.from_dict(manifest_dict)
                results.append(
                    ModelVersion(
                        id=row["id"],
                        project_id=row["project_id"],
                        name=row["name"],
                        created_at=row["created_at"],
                        manifest=manifest,
                    )
                )
            return results

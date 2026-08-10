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
        import os
        results = []
        existing_ids = set()

        # 1. Query SQLite DB
        try:
            with self.db_mgr.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, project_id, name, created_at, manifest_json FROM model_versions"
                )
                rows = cursor.fetchall()
                for row in rows:
                    manifest_dict = json.loads(row["manifest_json"])
                    manifest = Manifest.from_dict(manifest_dict)
                    existing_ids.add(row["id"])
                    results.append(
                        ModelVersion(
                            id=row["id"],
                            project_id=row["project_id"],
                            name=row["name"],
                            created_at=row["created_at"],
                            manifest=manifest,
                        )
                    )
        except Exception:
            pass

        # 2. Scan runs/ directory for completed training job manifests
        runs_dirs = [
            os.environ.get("RUNS_DIR", "runs"),
            "runs",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs"),
        ]
        
        for runs_dir in runs_dirs:
            if not os.path.exists(runs_dir):
                continue
            try:
                for entry in os.listdir(runs_dir):
                    if entry.startswith("job_"):
                        job_path = os.path.join(runs_dir, entry)
                        manifest_path = os.path.join(job_path, "manifest.json")
                        status_path = os.path.join(job_path, "status.json")
                        
                        if os.path.exists(manifest_path):
                            try:
                                is_completed = True
                                if os.path.exists(status_path):
                                    with open(status_path, "r", encoding="utf-8") as sf:
                                        st_data = json.load(sf)
                                        if st_data.get("status") != "completed":
                                            is_completed = False
                                
                                if is_completed:
                                    with open(manifest_path, "r", encoding="utf-8") as mf:
                                        m_dict = json.load(mf)
                                    manifest = Manifest.from_dict(m_dict)
                                    if manifest.id not in existing_ids:
                                        existing_ids.add(manifest.id)
                                        # Self-heal / register in SQLite
                                        try:
                                            self.register_model_bundle("default", manifest)
                                        except Exception:
                                            pass
                                        results.append(
                                            ModelVersion(
                                                id=manifest.id,
                                                project_id="default",
                                                name=manifest.name,
                                                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                manifest=manifest,
                                            )
                                        )
                            except Exception:
                                pass
            except Exception:
                pass

        return results

    def delete_model_version(self, model_id: str) -> bool:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM model_versions WHERE id = ?", (model_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

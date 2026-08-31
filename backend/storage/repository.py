import datetime
import json
import logging
import os
from typing import List, Optional
import uuid

logger = logging.getLogger(__name__)


def _uses_generated_model_name(current_name: str, project_name: Optional[str]) -> bool:
    return bool(
        not current_name
        or current_name.startswith("Custom Model ")
        or (
            project_name
            and current_name.startswith(f"{project_name} (")
            and current_name.endswith(")")
        )
    )

try:
    from backend.domain.models import Manifest, ModelVersion, Project
    from backend.storage.db import DatabaseManager
except ImportError:
    from domain.models import Manifest, ModelVersion, Project
    from storage.db import DatabaseManager


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
    def __init__(
        self,
        db_mgr: DatabaseManager,
        models_dir: str = "models",
        runs_dir: Optional[str] = None,
    ):
        self.db_mgr = db_mgr
        self.models_dir = models_dir
        self.runs_dir = runs_dir

    def _is_test_only_manifest(self, manifest: Manifest) -> bool:
        artifact = manifest.detector.artifact if manifest.detector else ""
        if not artifact:
            return False

        candidates = []
        if os.path.isabs(artifact):
            candidates.append(artifact)
        else:
            candidates.extend(
                [
                    os.path.join(self.models_dir, artifact),
                    os.path.join(os.path.dirname(os.path.abspath(self.models_dir)), artifact),
                ]
            )
            if self.runs_dir:
                candidates.append(
                    os.path.join(os.path.dirname(os.path.abspath(self.runs_dir)), artifact)
                )

        artifact_path = next(
            (path for path in candidates if os.path.isfile(path)), None
        )
        if not artifact_path:
            return False
        try:
            with open(artifact_path, "rb") as artifact_file:
                signature = artifact_file.read(64)
            return b"MOCK_" in signature or b"EXPLICIT_TEST_MOCK" in signature
        except OSError:
            return False

    def _read_run_job_config(self, model_id: str) -> dict:
        if not self.runs_dir:
            return {}
        config_path = os.path.join(
            self.runs_dir, f"job_{model_id}", "job_config.json"
        )
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            return config if isinstance(config, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

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
                manifest = Manifest.from_dict(
                    manifest_dict,
                    fallback_id=row["id"],
                    artifact_prefix=f"runs/job_{row['id']}",
                )
                if self._is_test_only_manifest(manifest):
                    logger.warning("Refusing test-only registered model %s", row["id"])
                    return None
                return ModelVersion(
                    id=row["id"],
                    project_id=row["project_id"],
                    name=row["name"],
                    created_at=row["created_at"],
                    manifest=manifest,
                )
        return None

    def list_model_versions(self) -> List[ModelVersion]:
        results = []
        existing_ids = set()

        # Build project name lookup map
        project_names = {}
        try:
            with self.db_mgr.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM projects")
                for row in cursor.fetchall():
                    project_names[row["id"]] = row["name"]
        except Exception as error:
            logger.warning("Unable to load project names: %s", error)

        # 1. Query SQLite DB
        try:
            with self.db_mgr.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, project_id, name, created_at, manifest_json FROM model_versions"
                )
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        manifest_dict = json.loads(row["manifest_json"])
                        manifest = Manifest.from_dict(
                            manifest_dict,
                            fallback_id=row["id"],
                            artifact_prefix=f"runs/job_{row['id']}",
                        )
                        if self._is_test_only_manifest(manifest):
                            logger.warning(
                                "Skipping test-only registered model %s", row["id"]
                            )
                            continue
                        existing_ids.add(row["id"])

                        job_config = self._read_run_job_config(row["id"])
                        project_id = row["project_id"]
                        configured_project_id = job_config.get("project_id")
                        if (
                            project_id in (None, "default", "project_default")
                            and configured_project_id
                        ):
                            project_id = configured_project_id
                        project_name = project_names.get(project_id)
                        current_name = manifest.name or row["name"]
                        specified_name = (
                            job_config.get("model_name")
                            or job_config.get("config", {}).get("model_name")
                            or job_config.get("config", {}).get("name")
                        )
                        if specified_name:
                            current_name = specified_name
                        if (
                            _uses_generated_model_name(current_name, project_name)
                            and row["name"]
                            and not _uses_generated_model_name(row["name"], project_name)
                        ):
                            current_name = row["name"]
                        if _uses_generated_model_name(current_name, project_name) and project_name:
                            if project_name not in ("default", "project_default"):
                                current_name = project_name

                        if (
                            current_name != manifest.name
                            or current_name != row["name"]
                            or project_id != row["project_id"]
                        ):
                            manifest.name = current_name
                            cursor.execute(
                                "UPDATE model_versions SET project_id = ?, name = ?, manifest_json = ? WHERE id = ?",
                                (
                                    project_id,
                                    current_name,
                                    json.dumps(manifest.to_dict()),
                                    row["id"],
                                ),
                            )
                            if self.runs_dir:
                                manifest_path = os.path.join(
                                    self.runs_dir,
                                    f"job_{row['id']}",
                                    "manifest.json",
                                )
                                if os.path.isfile(manifest_path):
                                    try:
                                        tmp_path = f"{manifest_path}.tmp"
                                        with open(tmp_path, "w", encoding="utf-8") as manifest_file:
                                            json.dump(manifest.to_dict(), manifest_file, indent=2)
                                        os.replace(tmp_path, manifest_path)
                                    except OSError as error:
                                        logger.warning(
                                            "Unable to update manifest name for %s: %s",
                                            row["id"],
                                            error,
                                        )

                        results.append(
                            ModelVersion(
                                id=row["id"],
                                project_id=project_id,
                                name=current_name,
                                created_at=row["created_at"],
                                manifest=manifest,
                            )
                        )
                    except Exception as error:
                        logger.warning(
                            "Skipping invalid registered model %s: %s", row["id"], error
                        )
                conn.commit()
        except Exception as error:
            logger.warning("Unable to read registered model versions: %s", error)

        # 2. Scan runs/ directory for completed training job manifests
        sibling_runs_dir = os.path.join(os.path.dirname(os.path.abspath(self.models_dir)), "runs")
        runs_dirs = []
        for path in (self.runs_dir, sibling_runs_dir):
            if path and os.path.abspath(path) not in [os.path.abspath(item) for item in runs_dirs]:
                runs_dirs.append(path)
        
        for runs_dir in runs_dirs:
            if not os.path.exists(runs_dir):
                continue
            try:
                for entry in os.listdir(runs_dir):
                    if entry.startswith("job_"):
                        job_path = os.path.join(runs_dir, entry)
                        manifest_path = os.path.join(job_path, "manifest.json")
                        status_path = os.path.join(job_path, "status.json")
                        job_config_path = os.path.join(job_path, "job_config.json")
                        
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
                                    detector_reference = (
                                        m_dict.get("detector", {}).get("artifact")
                                        or m_dict.get("detector", {}).get("weights")
                                    )
                                    if detector_reference:
                                        detector_candidates = [
                                            detector_reference
                                            if os.path.isabs(detector_reference)
                                            else os.path.join(job_path, detector_reference),
                                            os.path.join(
                                                os.path.dirname(runs_dir), detector_reference
                                            ),
                                        ]
                                        detector_path = next(
                                            (
                                                candidate
                                                for candidate in detector_candidates
                                                if os.path.isfile(candidate)
                                            ),
                                            None,
                                        )
                                        try:
                                            if not detector_path:
                                                raise FileNotFoundError(detector_reference)
                                            with open(detector_path, "rb") as detector_file:
                                                signature = detector_file.read(64)
                                            if b"MOCK_" in signature or b"EXPLICIT_TEST_MOCK" in signature:
                                                logger.info("Skipping test-only model run %s", entry)
                                                continue
                                        except OSError as error:
                                            logger.warning(
                                                "Skipping model %s with missing detector: %s",
                                                entry,
                                                error,
                                            )
                                            continue
                                    manifest = Manifest.from_dict(
                                        m_dict,
                                        fallback_id=entry.removeprefix("job_"),
                                        artifact_prefix=f"runs/{entry}",
                                    )
                                    if manifest.id not in existing_ids:
                                        existing_ids.add(manifest.id)

                                        m_name = manifest.name
                                        proj_id = "default"
                                        if os.path.exists(job_config_path):
                                            try:
                                                with open(job_config_path, "r", encoding="utf-8") as jcf:
                                                    jc = json.load(jcf)
                                                    proj_id = jc.get("project_id", "default")
                                                    specified_name = (
                                                        jc.get("model_name")
                                                        or jc.get("config", {}).get("model_name")
                                                        or jc.get("config", {}).get("name")
                                                    )
                                                    if specified_name:
                                                        m_name = specified_name
                                            except Exception as error:
                                                logger.warning(
                                                    "Unable to read job config %s: %s",
                                                    job_config_path,
                                                    error,
                                                )

                                        project_name = project_names.get(proj_id)
                                        if _uses_generated_model_name(
                                            m_name, project_name
                                        ) and project_name not in (None, "default", "project_default"):
                                            m_name = project_name

                                        manifest.name = m_name

                                        # Self-heal / register in SQLite
                                        try:
                                            self.register_model_bundle(proj_id, manifest)
                                        except Exception as error:
                                            logger.warning(
                                                "Unable to register discovered model %s: %s",
                                                manifest.id,
                                                error,
                                            )
                                        results.append(
                                            ModelVersion(
                                                id=manifest.id,
                                                project_id=proj_id,
                                                name=manifest.name,
                                                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                manifest=manifest,
                                            )
                                        )
                            except Exception as error:
                                logger.warning("Unable to read run manifest %s: %s", manifest_path, error)
            except Exception as error:
                logger.warning("Unable to scan runs directory %s: %s", runs_dir, error)

        return results

    def delete_model_version(self, model_id: str) -> bool:
        with self.db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM model_versions WHERE id = ?", (model_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

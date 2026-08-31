from dataclasses import dataclass, field
import posixpath
from typing import List, Dict, Optional, Any


@dataclass
class DetectorConfig:
    artifact: str
    geometry: str = "obb"
    confidence: float = 0.25
    iou: float = 0.45


@dataclass
class ClassConfig:
    id: int
    name: str
    landmark_schema: Optional[str] = None
    predictor: Optional[str] = None
    crop_padding: float = 0.2


@dataclass
class LandmarkSchemaConfig:
    points: List[str]


@dataclass
class Manifest:
    schema_version: int
    id: str
    name: str
    description: str
    detector: DetectorConfig
    classes: List[ClassConfig]
    landmark_schemas: Dict[str, LandmarkSchemaConfig]
    evaluation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        fallback_id: Optional[str] = None,
        artifact_prefix: Optional[str] = None,
    ) -> "Manifest":
        detector_data = data.get("detector", {})
        if isinstance(detector_data, dict):
            legacy_detector = "artifact" not in detector_data and "weights" in detector_data
            detector_artifact = detector_data.get("artifact") or detector_data.get("weights", "")
            if (
                legacy_detector
                and artifact_prefix
                and detector_artifact
                and not posixpath.isabs(detector_artifact)
            ):
                detector_artifact = posixpath.join(artifact_prefix, detector_artifact)
            detector = DetectorConfig(
                artifact=detector_artifact,
                geometry=detector_data.get("geometry", "obb"),
                confidence=float(detector_data.get("confidence", 0.25)),
                iou=float(detector_data.get("iou", 0.45)),
            )
        else:
            detector = detector_data

        classes_raw = data.get("classes", [])
        if not classes_raw and data.get("predictors"):
            classes_raw = [
                {
                    "id": index,
                    "name": predictor.get("class_name", f"class_{index}"),
                    "landmark_schema": predictor.get("class_name", f"class_{index}"),
                    "predictor": (
                        posixpath.join(artifact_prefix, predictor.get("file", ""))
                        if artifact_prefix and predictor.get("file")
                        else predictor.get("file")
                    ),
                    "crop_padding": 0.2,
                }
                for index, predictor in enumerate(data["predictors"])
            ]
        classes = [
            ClassConfig(**c) if isinstance(c, dict) else c for c in classes_raw
        ]

        landmark_schemas_raw = data.get("landmark_schemas", {})
        landmark_schemas = {}
        for k, v in landmark_schemas_raw.items():
            if isinstance(v, dict):
                landmark_schemas[k] = LandmarkSchemaConfig(**v)
            else:
                landmark_schemas[k] = v
        for class_config in classes:
            if class_config.landmark_schema and class_config.landmark_schema not in landmark_schemas:
                landmark_schemas[class_config.landmark_schema] = LandmarkSchemaConfig(points=[])

        manifest_id = data.get("id") or fallback_id
        if not manifest_id:
            raise ValueError("Model manifest is missing its id.")

        return cls(
            schema_version=int(float(data.get("schema_version", data.get("manifest_version", 1)))),
            id=manifest_id,
            name=data.get("name") or f"Custom Model {manifest_id[:8]}",
            description=data.get("description", ""),
            detector=detector,
            classes=classes,
            landmark_schemas=landmark_schemas,
            evaluation=data.get("evaluation", data.get("metrics", {})),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "detector": {
                "artifact": self.detector.artifact,
                "geometry": self.detector.geometry,
                "confidence": self.detector.confidence,
                "iou": self.detector.iou,
            },
            "classes": [
                {
                    "id": c.id,
                    "name": c.name,
                    "landmark_schema": c.landmark_schema,
                    "predictor": c.predictor,
                    "crop_padding": c.crop_padding,
                }
                for c in self.classes
            ],
            "landmark_schemas": {
                k: {"points": v.points} for k, v in self.landmark_schemas.items()
            },
            "evaluation": self.evaluation,
        }


@dataclass
class Project:
    id: str
    name: str
    organism: str
    created_at: str


@dataclass
class ModelVersion:
    id: str
    project_id: str
    name: str
    created_at: str
    manifest: Manifest

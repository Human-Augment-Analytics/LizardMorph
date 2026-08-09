from dataclasses import dataclass, field
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
    def from_dict(cls, data: dict) -> "Manifest":
        detector_data = data.get("detector", {})
        detector = (
            DetectorConfig(**detector_data)
            if isinstance(detector_data, dict)
            else detector_data
        )

        classes_raw = data.get("classes", [])
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

        return cls(
            schema_version=data["schema_version"],
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            detector=detector,
            classes=classes,
            landmark_schemas=landmark_schemas,
            evaluation=data.get("evaluation", {}),
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

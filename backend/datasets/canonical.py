from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class LandmarkPoint:
    name: str
    x: float
    y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "x": float(self.x),
            "y": float(self.y),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LandmarkPoint":
        return cls(
            name=str(data["name"]),
            x=float(data["x"]),
            y=float(data["y"]),
        )


@dataclass
class CanonicalObject:
    object_id: str
    class_name: str
    obb: List[float]
    landmarks: List[LandmarkPoint] = field(default_factory=list)
    specimen_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "obb": [float(v) for v in self.obb],
            "landmarks": [lm.to_dict() for lm in self.landmarks],
            "specimen_id": self.specimen_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalObject":
        landmarks_raw = data.get("landmarks", [])
        landmarks = [
            LandmarkPoint.from_dict(lm) if isinstance(lm, dict) else lm
            for lm in landmarks_raw
        ]
        return cls(
            object_id=str(data["object_id"]),
            class_name=str(data["class_name"]),
            obb=[float(v) for v in data.get("obb", [0.0, 0.0, 0.0, 0.0, 0.0])],
            landmarks=landmarks,
            specimen_id=data.get("specimen_id"),
        )


@dataclass
class CanonicalImage:
    image_id: str
    file_path: str
    width: int
    height: int
    objects: List[CanonicalObject] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalImage":
        objects_raw = data.get("objects", [])
        objects = [
            CanonicalObject.from_dict(obj) if isinstance(obj, dict) else obj
            for obj in objects_raw
        ]
        return cls(
            image_id=str(data["image_id"]),
            file_path=str(data["file_path"]),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            objects=objects,
        )


@dataclass
class CanonicalDataset:
    images: List[CanonicalImage] = field(default_factory=list)

    @staticmethod
    def derive_obb_from_landmarks(
        pts: List[Tuple[float, float]], padding: float = 0.2
    ) -> List[float]:
        """Calculates minimum bounding box [cx, cy, w, h, 0.0] from landmark points array and applies fractional padding."""
        if not pts:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        w_padded = w * (1.0 + padding)
        h_padded = h * (1.0 + padding)
        return [float(cx), float(cy), float(w_padded), float(h_padded), 0.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "images": [img.to_dict() for img in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalDataset":
        images_raw = data.get("images", [])
        images = [
            CanonicalImage.from_dict(img) if isinstance(img, dict) else img
            for img in images_raw
        ]
        return cls(images=images)

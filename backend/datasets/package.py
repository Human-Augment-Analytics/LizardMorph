import io
import math
import os
import posixpath
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from backend.datasets.canonical import CanonicalDataset
    from backend.datasets.importers import DlibXMLImporter, TPSImporter
except ImportError:
    from datasets.canonical import CanonicalDataset
    from datasets.importers import DlibXMLImporter, TPSImporter


ANNOTATION_EXTENSIONS = {".tps", ".xml"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_ARCHIVE_FILES = int(os.getenv("TRAINING_MAX_ARCHIVE_FILES", "20000"))
MAX_UNCOMPRESSED_BYTES = int(
    os.getenv("TRAINING_MAX_UNCOMPRESSED_BYTES", str(1024 * 1024 * 1024))
)


@dataclass
class ParsedDatasetPackage:
    dataset: CanonicalDataset
    source_files: Dict[str, bytes]
    annotation_name: str


def _safe_relative_name(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if (
        not normalized
        or normalized.startswith("/")
        or normalized == ".."
        or normalized.startswith("../")
    ):
        raise ValueError(f"Unsafe dataset path: {name}")
    return normalized


def _is_metadata(name: str) -> bool:
    parts = name.split("/")
    return "__MACOSX" in parts or any(part.startswith("._") for part in parts)


def _unpack_files(files: Iterable[Tuple[str, bytes]]) -> Dict[str, bytes]:
    unpacked: Dict[str, bytes] = {}
    seen_names = set()

    def add_file(name: str, content: bytes):
        safe_name = _safe_relative_name(name)
        if _is_metadata(safe_name):
            return
        key = safe_name.casefold()
        if key in seen_names:
            raise ValueError(f"Duplicate filename '{safe_name}' in dataset package.")
        seen_names.add(key)
        unpacked[safe_name] = content

    for filename, content in files:
        extension = os.path.splitext(filename)[1].lower()
        if extension != ".zip":
            add_file(os.path.basename(filename), content)
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if len(members) > MAX_ARCHIVE_FILES:
                    raise ValueError("Dataset archive contains too many files.")
                total_size = sum(member.file_size for member in members)
                if total_size > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("Dataset archive is too large after extraction.")
                for member in members:
                    safe_name = _safe_relative_name(member.filename)
                    add_file(safe_name, archive.read(member))
        except zipfile.BadZipFile as exc:
            raise ValueError("The selected ZIP dataset is invalid.") from exc

    return unpacked


def _match_image_path(reference: str, source_files: Dict[str, bytes]) -> str:
    raw_reference = reference.replace("\\", "/")
    try:
        normalized_reference = _safe_relative_name(raw_reference)
    except ValueError:
        # Dlib XML generated on another machine often contains an absolute path.
        # It is safe to use only its basename because we match in-memory uploads,
        # never read the referenced host path.
        normalized_reference = posixpath.basename(posixpath.normpath(raw_reference))
        if not normalized_reference:
            raise ValueError(f"Invalid image reference: {reference}")
    exact_matches = [
        name for name in source_files if name.casefold() == normalized_reference.casefold()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    basename = posixpath.basename(normalized_reference).casefold()
    basename_matches = [
        name for name in source_files if posixpath.basename(name).casefold() == basename
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise ValueError(
            f"Image reference '{reference}' is ambiguous because multiple files share its name."
        )
    raise ValueError(f"Image '{reference}' referenced by the annotations was not provided.")


def parse_dataset_package(files: Iterable[Tuple[str, bytes]]) -> ParsedDatasetPackage:
    unpacked = _unpack_files(files)
    annotations = [
        name
        for name in unpacked
        if os.path.splitext(name)[1].lower() in ANNOTATION_EXTENSIONS
    ]
    if len(annotations) != 1:
        raise ValueError(
            "Dataset must contain exactly one .tps or .xml annotation file."
        )

    annotation_name = annotations[0]
    extension = os.path.splitext(annotation_name)[1].lower()
    try:
        content = unpacked[annotation_name].decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Annotation file must be UTF-8 text.") from exc

    is_tps = extension == ".tps"
    dataset = (
        TPSImporter.parse_string(content)
        if is_tps
        else DlibXMLImporter.parse_string(content)
    )
    if not dataset.images:
        raise ValueError("Annotation file contains no images.")

    source_files = {
        name: data
        for name, data in unpacked.items()
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    }
    for image in dataset.images:
        matched_name = _match_image_path(image.file_path, source_files)
        encoded = np.frombuffer(source_files[matched_name], dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Unable to decode training image '{matched_name}'.")
        height, width = decoded.shape[:2]
        image.file_path = matched_name
        image.width = width
        image.height = height

        for obj in image.objects:
            if is_tps:
                for landmark in obj.landmarks:
                    landmark.y = float(height) - landmark.y
                obj.obb = CanonicalDataset.derive_obb_from_landmarks(
                    [(point.x, point.y) for point in obj.landmarks], padding=0.0
                )
            elif (len(obj.obb) < 5 or obj.obb[2] <= 0 or obj.obb[3] <= 0) and obj.landmarks:
                obj.obb = CanonicalDataset.derive_obb_from_landmarks(
                    [(point.x, point.y) for point in obj.landmarks], padding=0.0
                )

            if len(obj.obb) < 5 or not all(math.isfinite(float(value)) for value in obj.obb[:5]):
                raise ValueError(f"Object '{obj.object_id}' in '{matched_name}' has an invalid box.")
            if obj.obb[2] <= 0 or obj.obb[3] <= 0:
                raise ValueError(f"Object '{obj.object_id}' in '{matched_name}' has an empty box.")
            if not obj.landmarks:
                raise ValueError(f"Object '{obj.object_id}' in '{matched_name}' has no landmarks.")

            for landmark in obj.landmarks:
                if not (
                    math.isfinite(landmark.x)
                    and math.isfinite(landmark.y)
                    and 0 <= landmark.x < width
                    and 0 <= landmark.y < height
                ):
                    raise ValueError(
                        f"Landmark '{landmark.name}' for '{matched_name}' is outside the image bounds."
                    )

    return ParsedDatasetPackage(
        dataset=dataset,
        source_files=source_files,
        annotation_name=annotation_name,
    )


def padded_preview(dataset: CanonicalDataset, padding: float) -> CanonicalDataset:
    preview = CanonicalDataset.from_dict(dataset.to_dict())
    for image in preview.images:
        for obj in image.objects:
            if obj.landmarks:
                obj.obb = CanonicalDataset.derive_obb_from_landmarks(
                    [(point.x, point.y) for point in obj.landmarks], padding=padding
                )
            elif len(obj.obb) >= 5:
                obj.obb[2] *= 1.0 + padding
                obj.obb[3] *= 1.0 + padding
    return preview

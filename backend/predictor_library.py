from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import dlib  # type: ignore
except Exception:  # pragma: no cover
    dlib = None


@dataclass(frozen=True)
class PredictorMeta:
    id: str
    display_name: str
    stored_filename: str
    uploaded_at: str
    size_bytes: int
    num_parts: Optional[int] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_filename(original: str) -> str:
    # Keep only basename; avoid path traversal
    return os.path.basename(original or "")


def _validate_dat_name(filename: str) -> None:
    if not filename.lower().endswith(".dat"):
        raise ValueError("Only .dat files are supported")


def load_index(index_path: str) -> Dict[str, Any]:
    if not os.path.exists(index_path):
        return {"predictors": []}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index_path: str, index: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(index_path))
    tmp = f"{index_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=False)
    os.replace(tmp, index_path)


def list_predictors(index_path: str) -> List[PredictorMeta]:
    idx = load_index(index_path)
    out: List[PredictorMeta] = []
    for p in idx.get("predictors", []):
        out.append(PredictorMeta(**p))
    return out


def get_predictor(index_path: str, predictor_id: str) -> Optional[PredictorMeta]:
    for p in list_predictors(index_path):
        if p.id == predictor_id:
            return p
    return None


def resolve_predictor_path(files_dir: str, meta: PredictorMeta) -> str:
    return os.path.join(files_dir, meta.stored_filename)


def add_predictor(
    *,
    index_path: str,
    files_dir: str,
    original_filename: str,
    file_bytes: bytes,
    max_bytes: int,
    validate_with_dlib: bool = True,
) -> PredictorMeta:
    ensure_dir(files_dir)

    safe = _safe_filename(original_filename)
    _validate_dat_name(safe)

    if not file_bytes:
        raise ValueError("Empty file")
    if len(file_bytes) > max_bytes:
        raise ValueError(f"File too large (>{max_bytes} bytes)")

    predictor_id = str(uuid.uuid4())
    stored_filename = f"{predictor_id}.dat"
    stored_path = os.path.join(files_dir, stored_filename)
    with open(stored_path, "wb") as f:
        f.write(file_bytes)

    num_parts: Optional[int] = None
    if validate_with_dlib and dlib is not None:
        try:
            sp = dlib.shape_predictor(stored_path)
            # dlib python bindings commonly expose `num_parts` as attribute
            try:
                num_parts = int(getattr(sp, "num_parts"))
            except Exception:
                num_parts = None
        except Exception as e:
            try:
                os.remove(stored_path)
            except Exception:
                pass
            raise ValueError(f"Invalid dlib shape predictor: {e}")

    meta = PredictorMeta(
        id=predictor_id,
        display_name=safe,
        stored_filename=stored_filename,
        uploaded_at=_now_iso(),
        size_bytes=len(file_bytes),
        num_parts=num_parts,
    )

    idx = load_index(index_path)
    predictors = idx.get("predictors", [])
    predictors.append(asdict(meta))
    idx["predictors"] = predictors
    save_index(index_path, idx)
    return meta


def delete_predictor(*, index_path: str, files_dir: str, predictor_id: str) -> bool:
    idx = load_index(index_path)
    predictors = idx.get("predictors", [])
    remaining = []
    target = None
    for p in predictors:
        if p.get("id") == predictor_id:
            target = p
        else:
            remaining.append(p)

    if target is None:
        return False

    idx["predictors"] = remaining
    save_index(index_path, idx)

    stored_filename = target.get("stored_filename")
    if stored_filename:
        try:
            os.remove(os.path.join(files_dir, stored_filename))
        except FileNotFoundError:
            pass
    return True


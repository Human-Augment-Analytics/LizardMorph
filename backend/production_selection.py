"""Snapshot the local production default once per analysis session.

The updater owns the atomic deployment pointer. Requests never resolve a mutable
registry alias. Sessions predating this feature deliberately retain legacy behavior.
"""
import json
import os
from pathlib import Path
import re
import tempfile


SESSION_PIN = "production-model.json"
MODEL_ID = re.compile(r"web-[0-9a-f]{24}\Z")


class ProductionSelectionError(ValueError):
    """A configured deployment or persisted analysis pin cannot be used safely."""


def _read_json(path):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("oversized selection metadata")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("selection metadata must be an object")
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        raise ProductionSelectionError("Production model selection is unavailable") from exc


def active_model_id(state_path=None):
    """Missing/unconfigured deployment is legacy; malformed deployment is an error."""
    state_path = os.getenv("WEB_MODEL_STATE_PATH") if state_path is None else state_path
    if not state_path or not Path(state_path).exists():
        return None
    state = _read_json(state_path)
    if state.get("format") != "lizard-web-state-v1" or "active" not in state:
        raise ProductionSelectionError("Invalid production model state")
    active = state["active"]
    if active is None:
        return None
    if not isinstance(active, dict) or not isinstance(active.get("model_id"), str):
        raise ProductionSelectionError("Invalid production model state")
    model_id = active["model_id"]
    if not MODEL_ID.fullmatch(model_id):
        raise ProductionSelectionError("Invalid production model identity")
    return model_id


def new_session_pin():
    """Capture an error without breaking unrelated x-ray/custom-model sessions."""
    try:
        return {"format": "lizard-session-production-v1", "model_id": active_model_id()}
    except ProductionSelectionError:
        return {"format": "lizard-session-production-v1", "unavailable": True}


def save_session_pin(session_folder, pin):
    """Write before exposing a newly created session to requests."""
    root = Path(session_folder)
    fd, temporary = tempfile.mkstemp(prefix=".production-model-", dir=root)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(pin, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / SESSION_PIN)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def session_model_id(session_folder):
    pin_path = Path(session_folder) / SESSION_PIN
    if not pin_path.exists():
        return None  # Never adopt a new default for a preexisting analysis.
    pin = _read_json(pin_path)
    if pin.get("format") != "lizard-session-production-v1" or pin.get("unavailable"):
        raise ProductionSelectionError("The session's production model is unavailable")
    if "model_id" not in pin:
        raise ProductionSelectionError("Invalid production model pin")
    model_id = pin["model_id"]
    if model_id is not None and (not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id)):
        raise ProductionSelectionError("Invalid production model pin")
    return model_id


def select_model(view_type, explicit_model_id, session_folder, *, predictor_type="toe", filename=None):
    """Explicit choice, then image provenance, then immutable session default."""
    if explicit_model_id:
        return explicit_model_id
    if str(view_type).lower() not in {"toepad", "toepads"} or predictor_type == "scale":
        return None
    if filename is not None:
        # Reprocessing an existing image keeps its actual selected model, even
        # when the current session/default changed. Caller supplies a safe name.
        if Path(filename).name != filename:
            raise ProductionSelectionError("Invalid analysis image name")
        provenance_path = Path(session_folder) / "outputs" / (filename + ".model.json")
        if provenance_path.exists():
            provenance = _read_json(provenance_path)
            model_id = provenance.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ProductionSelectionError("Invalid image model provenance")
            return model_id
    return session_model_id(session_folder)

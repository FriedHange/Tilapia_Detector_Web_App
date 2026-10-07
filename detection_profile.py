"""Locked counting settings shared by inference, evaluation and calibration."""
import hashlib
import json
import threading
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "detection_profile.json"
INFERENCE_LOCK = threading.RLock()
DETECTORS = ("yolov8n", "yolov9c", "yolov10n")
ALIASES = dict(zip(DETECTORS, ("Engine A", "Engine B", "Engine C")))
ALIASES["ensemble"] = "Combined Counting"


def load_profile():
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {"calibrated": False, "models": {name: {"conf": .5, "iou": .5} for name in DETECTORS}, "ensemble_iou": .5}


def fingerprint(profile=None):
    return hashlib.sha256(json.dumps(profile or load_profile(), sort_keys=True).encode()).hexdigest()


def settings_for(name, profile=None):
    return (profile or load_profile())["models"].get(name, {"conf": .5, "iou": .5})


def public_value(value):
    """Replace detector identifiers without sending the internal alias mapping."""
    if isinstance(value, dict):
        return {public_value(str(key)): public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, tuple):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        for name, alias in ALIASES.items():
            value = value.replace(name, alias)
        # Legacy records can reference weight variants which are not bundled.
        import re
        value = re.sub(r"(?i)yolo(?:v)?\d+[a-z0-9_-]*", "Counting engine", value)
    return value

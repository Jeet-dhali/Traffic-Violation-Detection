"""
detection.py  —  Loads models and runs inference.

CUSTOMISATION:
  - To swap models: change TRAFFIC_MODEL_PATH / HELMET_MODEL_PATH in config.py
  - To add a third model (e.g. LPR): add it here following the same pattern
    and call it from violations.py
"""

import os
from ultralytics import YOLO
from config import (
    TRAFFIC_MODEL_PATH, HELMET_MODEL_PATH,
    TRAFFIC_CONF, HELMET_CONF,
    VEHICLE_CLASS_IDS, PERSON_CLASS_ID,
    NO_HELMET_CLASS_ID,              # was missing from config.py — now added
)

# ─── Load traffic model ───────────────────────────────────────────────────────
print(f"[MODEL] Loading traffic model: {TRAFFIC_MODEL_PATH}")
traffic_model = YOLO(TRAFFIC_MODEL_PATH)

# ─── Load helmet model (optional) ────────────────────────────────────────────
helmet_model = None
if HELMET_MODEL_PATH and os.path.exists(HELMET_MODEL_PATH):
    print(f"[MODEL] Loading helmet model: {HELMET_MODEL_PATH}")
    helmet_model = YOLO(HELMET_MODEL_PATH)
else:
    print("[MODEL] Helmet model not found — helmet detection disabled")


def detect_vehicles_and_persons(frame):
    """
    Run traffic model on a full frame.
    Returns ultralytics Results object with tracking IDs.
    """
    results = traffic_model.track(
        frame,
        persist=True,
        conf=TRAFFIC_CONF,
        classes=VEHICLE_CLASS_IDS + [PERSON_CLASS_ID],
        verbose=False,
        device="mps",   # change to "cuda" or "mps" for GPU
    )
    return results[0]


def detect_helmet_in_crop(crop):
    """
    Run helmet model on a cropped motorcycle region.
    Returns (has_helmet: bool, confidence: float).

    If the helmet model is missing, returns (True, 0.0) — no violation raised.
    """
    if helmet_model is None or crop is None or crop.size == 0:
        return True, 0.0   # assume safe when model unavailable

    results = helmet_model(crop, conf=HELMET_CONF, verbose=False)
    boxes   = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return True, 0.0   # nothing detected → assume helmet present

    for box in boxes:
        cls = int(box.cls[0])
        if cls == NO_HELMET_CLASS_ID:
            return False, float(box.conf[0])   # violation
    return True, 0.0

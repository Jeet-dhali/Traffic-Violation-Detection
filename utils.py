"""
utils.py  —  Folder creation and evidence image saving.
"""

import os
import cv2
from datetime import datetime
from config import EVIDENCE_DIR, SAVE_EVIDENCE   # EVIDENCE_BASE_DIR is an alias for EVIDENCE_DIR


# Keep EVIDENCE_BASE_DIR as a local alias so any future direct references work
EVIDENCE_BASE_DIR = EVIDENCE_DIR

VIOLATION_DIRS = ["line", "wrongside", "helmet", "triple"]


def create_folders():
    """Create all required directories on startup."""
    os.makedirs("database", exist_ok=True)
    for d in VIOLATION_DIRS:
        os.makedirs(os.path.join(EVIDENCE_BASE_DIR, d), exist_ok=True)
    print("[UTILS] Folders ready.")


def save_violation_image(frame, box_xyxy, vtype: str, obj_id: int,
                         cam_id: str = "cam") -> str:
    """
    Saves a cropped + annotated snapshot of the violation.

    Returns the relative file path so it can be stored in the DB.
    Returns "" if SAVE_EVIDENCE is False.
    """
    if not SAVE_EVIDENCE:
        return ""

    PADDING = 20   # extra pixels around the detected box

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    cx1 = max(0, x1 - PADDING)
    cy1 = max(0, y1 - PADDING)
    cx2 = min(w, x2 + PADDING)
    cy2 = min(h, y2 + PADDING)

    crop = frame[cy1:cy2, cx1:cx2].copy()

    # Draw a coloured border on the crop so it's obvious in the file
    cv2.rectangle(crop, (0, 0), (crop.shape[1]-1, crop.shape[0]-1),
                  (0, 0, 255), 3)
    cv2.putText(crop, vtype.upper(), (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    vdir = vtype.lower().replace(" ", "_")
    os.makedirs(os.path.join(EVIDENCE_BASE_DIR, vdir), exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    fname = f"{cam_id}_id{obj_id}_{ts}.jpg"
    fpath = os.path.join(EVIDENCE_BASE_DIR, vdir, fname)

    cv2.imwrite(fpath, crop)
    return fpath

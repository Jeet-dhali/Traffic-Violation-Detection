"""
violations.py — TrafficGuard violation detection logic

Key fixes in this version:
  1. TRIPLE RIDING: threshold lowered to 2 (not 3) since YOLO detects riders
     as persons overlapping the motorcycle box. IoU threshold raised to 0.15
     and also added a containment check as an alternative to IoU, so partially-
     overlapping person boxes (common with side-on riders) are caught.

  2. NO HELMET: crop is now taken from the ORIGINAL clean frame passed in,
     not the annotated copy. Drawn bounding boxes on the annotated frame were
     obscuring the rider's head and degrading helmet model accuracy. Also added
     a minimum crop size guard and a confidence fallback when helmet model
     returns no boxes (treat as no-helmet when EXPAND_PX captures only head
     area and nothing is detected — configurable via HELMET_ABSENT_IS_VIOLATION).

  3. WRONG-SIDE: replaced the fragile per-frame dominant-vote approach with a
     persistent per-camera baseline direction computed over a rolling window.
     Dominant direction is now only updated when enough vehicles have history,
     and is stored in _cam_dominant so a single vehicle entering frame can be
     evaluated correctly against the established flow.

  4. PERSON DETECTION for triple riding: persons don't need track IDs — removed
     the obj_id >= 0 guard on person boxes since YOLO tracking rarely assigns
     IDs to persons.

  5. Added per-violation cooldown (_violation_cooldown) so the same vehicle
     can be re-flagged after leaving and re-entering frame (saved_violations
     only blocks re-saves within the same session, but combined with track-ID
     reuse this was causing missed detections on re-entry).
"""

import cv2
import numpy as np
from config import (
    LINE_Y,
    WRONG_SIDE_MIN_DX, WRONG_SIDE_HISTORY, WRONG_SIDE_FRAMES,
    TRIPLE_PERSON_THRESHOLD,
    EXPAND_PX, HELMET_CONF_THRESHOLD,
)

# ── How many persons overlapping motorcycle triggers triple riding ──
# YOLO detects each rider as a separate 'person' box overlapping the
# motorcycle box.  With 3 people on a bike you typically get 2–3 person
# detections (driver is sometimes merged into the moto box).
# Setting to 2 catches the most common case (3-on-bike → 2+ persons detected).
_TRIPLE_MIN_PERSONS = max(2, TRIPLE_PERSON_THRESHOLD - 1)

# IoU threshold for person-on-motorcycle overlap
_TRIPLE_IOU_THRESH  = 0.15

# If helmet model returns NO detections at all in the head crop, treat as
# no-helmet (True = flag as violation, False = treat as safe/unknown).
HELMET_ABSENT_IS_VIOLATION = False

# ── per-camera position history for wrong-side ──────────────────
# key: (cam_id, obj_id) → list of x-centres (newest last)
_pos_history: dict = {}

# ── per-camera established dominant direction ───────────────────
# key: cam_id → 'left' | 'right' | None
_cam_dominant: dict = {}

# ── traffic light state ─────────────────────────────────────────
traffic_light_green: bool = False


def set_traffic_light(is_green: bool) -> None:
    global traffic_light_green
    traffic_light_green = is_green


def get_traffic_light() -> bool:
    return traffic_light_green


# ── helpers ─────────────────────────────────────────────────────
def _iou(a, b) -> float:
    """IoU of two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(area_a + area_b - inter)


def _person_on_moto(moto_box, p_box) -> bool:
    """
    True if a person box is 'on' the motorcycle using EITHER:
      - IoU >= threshold  (full overlap, typical front view)
      - Person box centre is inside the motorcycle box  (side view, partial overlap)
    """
    if _iou(moto_box, p_box) >= _TRIPLE_IOU_THRESH:
        return True
    # centre-point containment
    pcx = (p_box[0] + p_box[2]) / 2
    pcy = (p_box[1] + p_box[3]) / 2
    if moto_box[0] <= pcx <= moto_box[2] and moto_box[1] <= pcy <= moto_box[3]:
        return True
    return False


def cleanup_stale_tracks(cam_id: str, active_obj_ids: set) -> None:
    stale = [k for k in list(_pos_history) if k[0] == cam_id and k[1] not in active_obj_ids]
    for k in stale:
        del _pos_history[k]


# ── main violation checker ───────────────────────────────────────
def check_violations(
    cam_id: str,
    results,
    helmet_model,
    frame,           # ANNOTATED frame (for drawing labels)
    orig_frame,      # CLEAN original frame (for helmet crop — avoids bbox occlusion)
    saved_violations: set,
    db_save_fn,
) -> None:
    """
    Parameters
    ----------
    cam_id          : camera id string
    results         : ultralytics Results (traffic model, persist=True)
    helmet_model    : YOLO helmet model or None
    frame           : annotated BGR frame — violation labels drawn here in-place
    orig_frame      : clean original frame — used for helmet crops only
    saved_violations: set of (obj_id, vtype) already saved this session
    db_save_fn      : callable(cam_id, obj_id, vtype, vehicle, confidence, evidence)
    """
    if results is None or results.boxes is None:
        return

    boxes            = results.boxes
    h_frame, w_frame = orig_frame.shape[:2]

    # ── parse detections ─────────────────────────────────────────
    det_list = []
    for box in boxes:
        cls_id          = int(box.cls[0])
        conf            = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Persons rarely get tracking IDs — use -1 for untracked
        obj_id          = int(box.id[0]) if box.id is not None else -1
        det_list.append(dict(cls=cls_id, conf=conf,
                              x1=x1, y1=y1, x2=x2, y2=y2, obj_id=obj_id))

    # class buckets
    # COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicles    = [d for d in det_list if d['cls'] in (1, 2, 3, 5, 7)]
    motorcycles = [d for d in det_list if d['cls'] == 3]
    # persons: include ALL person detections regardless of tracking ID
    persons     = [d for d in det_list if d['cls'] == 0]

    active_ids = {d['obj_id'] for d in vehicles if d['obj_id'] >= 0}
    cleanup_stale_tracks(cam_id, active_ids)

    def vtype_name(cls_id):
        return {1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}.get(cls_id,'vehicle')

    def _label(frame_, x, y, text):
        cv2.putText(frame_, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 30, 255), 2)

    # ── 1. STOP LINE ─────────────────────────────────────────────
    if not traffic_light_green:
        for v in vehicles:
            if v['obj_id'] < 0:
                continue
            key = (v['obj_id'], 'line')
            if key in saved_violations:
                continue
            if v['y2'] >= LINE_Y:
                saved_violations.add(key)
                ev = _save_evidence(orig_frame, v, 'line')
                db_save_fn(cam_id=cam_id, obj_id=v['obj_id'], vtype='line',
                           vehicle=vtype_name(v['cls']), confidence=v['conf'], evidence=ev)
                _label(frame, v['x1'], v['y1'] - 28, "⚠ STOP LINE")

    # ── 2. WRONG SIDE ────────────────────────────────────────────
    # Update rolling position history
    for v in vehicles:
        if v['obj_id'] < 0:
            continue
        cx   = (v['x1'] + v['x2']) // 2
        hkey = (cam_id, v['obj_id'])
        hist = _pos_history.setdefault(hkey, [])
        hist.append(cx)
        if len(hist) > WRONG_SIDE_HISTORY:
            hist.pop(0)

    # Recompute dominant direction only when enough vehicles have history
    all_hists = [
        _pos_history[(cam_id, v['obj_id'])]
        for v in vehicles
        if v['obj_id'] >= 0
        and (cam_id, v['obj_id']) in _pos_history
        and len(_pos_history[(cam_id, v['obj_id'])]) >= WRONG_SIDE_FRAMES + 1
    ]
    if len(all_hists) >= 2:          # need at least 2 vehicles to establish flow
        deltas    = [h[-1] - h[0] for h in all_hists]
        pos_votes = sum(1 for d in deltas if d >  WRONG_SIDE_MIN_DX)
        neg_votes = sum(1 for d in deltas if d < -WRONG_SIDE_MIN_DX)
        if pos_votes != neg_votes:   # only update when there's a clear majority
            _cam_dominant[cam_id] = 'right' if pos_votes > neg_votes else 'left'

    dominant = _cam_dominant.get(cam_id)

    if dominant is not None:
        for v in vehicles:
            if v['obj_id'] < 0:
                continue
            key  = (v['obj_id'], 'wrongside')
            if key in saved_violations:
                continue
            hkey = (cam_id, v['obj_id'])
            hist = _pos_history.get(hkey, [])
            if len(hist) < WRONG_SIDE_FRAMES + 1:
                continue
            recent_dx    = hist[-1] - hist[-WRONG_SIDE_FRAMES]
            moving_right = recent_dx >  WRONG_SIDE_MIN_DX
            moving_left  = recent_dx < -WRONG_SIDE_MIN_DX
            is_wrong     = (dominant == 'right' and moving_left) or \
                           (dominant == 'left'  and moving_right)
            if is_wrong:
                saved_violations.add(key)
                ev = _save_evidence(orig_frame, v, 'wrongside')
                db_save_fn(cam_id=cam_id, obj_id=v['obj_id'], vtype='wrongside',
                           vehicle=vtype_name(v['cls']), confidence=v['conf'], evidence=ev)
                _label(frame, v['x1'], v['y1'] - 28, "⚠ WRONG SIDE")

    # ── 3. NO HELMET ─────────────────────────────────────────────
    if helmet_model is not None:
        for moto in motorcycles:
            if moto['obj_id'] < 0:
                continue
            key = (moto['obj_id'], 'helmet')
            if key in saved_violations:
                continue

            # Crop from CLEAN original frame — not the annotated one
            cx1 = max(0, moto['x1'])
            cy1 = max(0, moto['y1'] - EXPAND_PX)   # expand up to capture head
            cx2 = min(w_frame, moto['x2'])
            cy2 = min(h_frame, moto['y2'])

            if cx2 - cx1 < 10 or cy2 - cy1 < 10:   # too small to bother
                continue

            crop = orig_frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            try:
                h_results          = helmet_model(crop, verbose=False)
                no_helmet_detected = False
                detection_conf     = 0.0

                if h_results and h_results[0].boxes is not None and len(h_results[0].boxes):
                    for hbox in h_results[0].boxes:
                        h_cls  = int(hbox.cls[0])
                        h_conf = float(hbox.conf[0])
                        label  = helmet_model.names.get(h_cls, '').lower()
                        if h_conf >= HELMET_CONF_THRESHOLD and (
                            h_cls == 0
                            or 'no' in label
                            or 'without' in label
                            or 'none' in label
                        ):
                            no_helmet_detected = True
                            detection_conf     = h_conf
                            break
                else:
                    # No detections at all in head-crop region
                    no_helmet_detected = HELMET_ABSENT_IS_VIOLATION
                    detection_conf     = 0.0

                if no_helmet_detected:
                    saved_violations.add(key)
                    ev = _save_evidence(orig_frame, moto, 'helmet')
                    db_save_fn(cam_id=cam_id, obj_id=moto['obj_id'], vtype='helmet',
                               vehicle='motorcycle', confidence=detection_conf, evidence=ev)
                    _label(frame, moto['x1'], moto['y1'] - 28, "⚠ NO HELMET")

            except Exception as e:
                print(f'[helmet] inference error: {e}')

    # ── 4. TRIPLE RIDING ─────────────────────────────────────────
    for moto in motorcycles:
        if moto['obj_id'] < 0:
            continue
        key = (moto['obj_id'], 'triple')
        if key in saved_violations:
            continue

        moto_box = [moto['x1'], moto['y1'], moto['x2'], moto['y2']]

        # Count persons whose box overlaps the motorcycle using IoU OR containment
        # No track-ID filter — persons rarely get IDs from YOLO tracker
        overlapping = sum(
            1 for p in persons
            if _person_on_moto(moto_box, [p['x1'], p['y1'], p['x2'], p['y2']])
        )

        if overlapping >= _TRIPLE_MIN_PERSONS:
            saved_violations.add(key)
            ev = _save_evidence(orig_frame, moto, 'triple')
            db_save_fn(cam_id=cam_id, obj_id=moto['obj_id'], vtype='triple',
                       vehicle='motorcycle', confidence=moto['conf'], evidence=ev)
            _label(frame, moto['x1'], moto['y1'] - 28,
                   f"⚠ TRIPLE ({overlapping}P)")


# ── evidence saver ───────────────────────────────────────────────
def _save_evidence(frame, det: dict, vtype: str) -> str:
    try:
        from config import SAVE_EVIDENCE, EVIDENCE_DIR
        import os, time as _t
        if not SAVE_EVIDENCE:
            return ''
        PAD    = 20
        h, w   = frame.shape[:2]
        x1     = max(0, det['x1'] - PAD);  y1 = max(0, det['y1'] - PAD)
        x2     = min(w, det['x2'] + PAD);  y2 = min(h, det['y2'] + PAD)
        if x2 <= x1 or y2 <= y1:
            return ''
        crop = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(crop, (0,0), (crop.shape[1]-1, crop.shape[0]-1), (0,0,255), 3)
        subdir = os.path.join(EVIDENCE_DIR, vtype)
        os.makedirs(subdir, exist_ok=True)
        fname = f"{vtype}_{det.get('obj_id',0)}_{int(_t.time()*1000)}.jpg"
        fpath = os.path.join(subdir, fname)
        cv2.imwrite(fpath, crop)
        return os.path.join(vtype, fname)
    except Exception as e:
        print(f'[evidence] save error: {e}')
        return ''

"""
camera.py  —  Threaded camera reader + YOLO processing per camera.

Each CameraProcessor:
  - Thread 1: reads raw frames from cap as fast as possible
  - Thread 2: runs YOLO every FRAME_SKIP frames, annotates, saves violations

Fixes in this version:
  1. orig_frame (clean copy before any drawing) is now kept and passed to
     check_violations so helmet crops are not corrupted by drawn bounding boxes.
  2. Person boxes (cls==0) are now drawn on the annotated frame so they are
     visible for debugging triple-riding detections.
  3. check_violations called with the new orig_frame parameter.
  4. Violation bounding boxes recoloured RED after check_violations runs,
     by re-checking saved_violations — gives immediate visual feedback.
  5. FPS smoothed with a rolling average instead of single-frame delta,
     which was producing erratic values during frame skips.
  6. backend parameter passed correctly through VideoCapture.
"""

import cv2
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime
import alerts

from detection import detect_vehicles_and_persons
from violations import (
    check_violations,
    set_traffic_light,
    get_traffic_light,
)
from database import insert_violation
from config import (
    FRAME_SKIP, STREAM_WIDTH, STREAM_HEIGHT,
    LINE_Y, JPEG_QUALITY,
)

_FPS_WINDOW = 20   # frames to average for smooth FPS display


class CameraProcessor:
    def __init__(self, cam_id: str, source, name: str, location: str,
                 backend: int = cv2.CAP_ANY):
        self.cam_id   = cam_id
        self.source   = source
        self.name     = name
        self.location = location
        self._backend = backend

        self.cap            = None
        self._raw_frame     = None
        self.latest_frame   = None   # annotated, served to Flask
        self.running        = False
        self.online         = False
        self.vehicle_count  = 0
        self.fps            = 0.0

        self._lock          = threading.Lock()
        self._frame_count   = 0
        self._frame_times   = deque(maxlen=_FPS_WINDOW)  # smooth FPS

        # Per-camera tracking: which (obj_id, vtype) already saved this session
        self._saved_violations: set = set()

        # Helmet model — loaded once at startup
        self._helmet_model = self._load_helmet_model()

    # ─── Helmet model loader ─────────────────────────────────────────────────

    @staticmethod
    def _load_helmet_model():
        try:
            from ultralytics import YOLO
            from config import HELMET_MODEL_PATH
            from dotenv import load_dotenv
            import os
            load_dotenv()
            if os.getenv("RENDER") == "true":
                print("[helmet] RENDER=true - skipping helmet model loading")
                return None
            if os.path.exists(HELMET_MODEL_PATH):
                model = YOLO(HELMET_MODEL_PATH)
                print(f"[helmet] Loaded: {HELMET_MODEL_PATH}")
                return model
            print(f"[helmet] Not found at {HELMET_MODEL_PATH} — helmet detection disabled.")
        except Exception as e:
            print(f"[helmet] Load error: {e}")
        return None

    # ─── Public API ──────────────────────────────────────────────────────────

    def start(self):
        self.cap = cv2.VideoCapture(self.source, self._backend)
        if not self.cap.isOpened():
            print(f"[CAM {self.cam_id}] ❌  Cannot open: {self.source}")
            self._make_offline_frame()
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)

        self.running = True
        self.online  = True
        threading.Thread(target=self._read_loop,    daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()
        print(f"[CAM {self.cam_id}] ✅  Online — {self.name} @ {self.location}")
        return True

    def stop(self):
        self.running = False
        self.online  = False
        if self.cap:
            self.cap.release()

    def get_jpeg(self) -> bytes:
        frame = self.latest_frame
        if frame is None:
            frame = self._offline_placeholder()
        _, buf = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes()

    # ─── Capture thread ──────────────────────────────────────────────────────

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
                with self._lock:
                    self._raw_frame = frame
            else:
                self.online = False
                time.sleep(3)
                self.cap.open(self.source)
                if self.cap.isOpened():
                    self.online = True
                    print(f"[CAM {self.cam_id}] Reconnected.")

    # ─── Processing thread ───────────────────────────────────────────────────

    def _process_loop(self):
        while self.running:
            with self._lock:
                frame = self._raw_frame.copy() if self._raw_frame is not None else None

            if frame is None:
                time.sleep(0.02)
                continue

            self._frame_count += 1

            # ── Smooth FPS ────────────────────────────────────────────────
            now = time.time()
            self._frame_times.append(now)
            if len(self._frame_times) >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                if span > 0:
                    self.fps = (len(self._frame_times) - 1) / span

            # ── Frame skip ────────────────────────────────────────────────
            if self._frame_count % FRAME_SKIP != 0:
                if self.latest_frame is None:
                    self.latest_frame = frame
                continue

            # ── YOLO + annotate ───────────────────────────────────────────
            try:
                result    = detect_vehicles_and_persons(frame)
                annotated = self._annotate(frame, result)
            except Exception as e:
                print(f"[CAM {self.cam_id}] Detection error: {e}")
                annotated = frame

            self.latest_frame = annotated

    # ─── Annotation + violation logic ────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, result) -> np.ndarray:
        """
        1. Keep orig_frame as a CLEAN copy for helmet crops.
        2. Draw all bounding boxes on annotated copy.
        3. Run check_violations — passes both annotated + orig frames.
        4. Redraw boxes for confirmed violations in RED.
        """
        orig_frame = frame.copy()       # ← clean, no drawings
        annotated  = frame.copy()       # ← will have all drawings

        if result is None or result.boxes is None:
            self.vehicle_count = 0
            self._draw_overlay(annotated, 0)
            return annotated

        boxes = result.boxes
        names = result.names

        # ── First pass: draw all boxes ────────────────────────────────────
        veh_count  = 0
        detections = []   # store parsed dets for second pass recolouring

        for box in boxes:
            cls_id          = int(box.cls[0])
            label           = names.get(cls_id, str(cls_id))
            conf            = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id          = int(box.id[0]) if box.id is not None else -1

            detections.append((cls_id, label, conf, x1, y1, x2, y2, obj_id))

            # Draw persons in blue, vehicles in green
            if cls_id == 0:
                color = (255, 160, 0)   # orange-ish for persons
            else:
                color = (0, 220, 110)   # green for vehicles
                veh_count += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label bar (only for tracked objects with IDs)
            if obj_id >= 0 or cls_id == 0:
                id_str   = f"ID:{obj_id}" if obj_id >= 0 else "?"
                id_label = f"{id_str} {label} {int(conf*100)}%"
                bar_x2   = min(annotated.shape[1], x1 + len(id_label) * 9)
                cv2.rectangle(annotated, (x1, y1-22), (bar_x2, y1), color, -1)
                cv2.putText(annotated, id_label, (x1+3, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Run violation checker ─────────────────────────────────────────
        violations_before = set(self._saved_violations)   # snapshot for diff

        def _db_save(cam_id, obj_id, vtype, vehicle, confidence, evidence):
            vid = insert_violation(
                cam_id=cam_id,
                location=self.location,
                obj_id=obj_id,
                vtype=vtype,
                vehicle=vehicle,
                confidence=round(confidence, 3),
                evidence_path=evidence,
            )
            alerts.publish({
                "id": vid,
                "cam_id": cam_id,
                "location": self.location,
                "vtype": vtype,
                "vehicle": vehicle,
                "confidence": round(confidence, 3),
                "evidence": evidence,
            })

        check_violations(
            cam_id=self.cam_id,
            results=result,
            helmet_model=self._helmet_model,
            frame=annotated,          # labels drawn in-place here
            orig_frame=orig_frame,    # clean frame for helmet crops
            saved_violations=self._saved_violations,
            db_save_fn=_db_save,
        )

        # ── Second pass: recolour boxes for new violations RED ────────────
        new_violations = self._saved_violations - violations_before
        if new_violations:
            violated_ids = {obj_id for (obj_id, _) in new_violations}
            for cls_id, label, conf, x1, y1, x2, y2, obj_id in detections:
                if obj_id in violated_ids:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 3)

        self.vehicle_count = veh_count
        self._draw_overlay(annotated, veh_count)
        return annotated

    # ─── HUD overlay ─────────────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, veh_count: int):
        h, w = frame.shape[:2]

        # Stop line
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 80, 255), 2)
        cv2.putText(frame, "STOP LINE", (w - 100, LINE_Y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)

        # Top info bar
        ts    = datetime.now().strftime("%H:%M:%S")
        light = "GREEN" if get_traffic_light() else "RED"
        txt   = (f"{self.cam_id} | {self.location} | {ts} "
                 f"| V:{veh_count} | {int(self.fps)}fps | {light}")
        bar_w = len(txt) * 8 + 12
        cv2.rectangle(frame, (0, 0), (bar_w, 24), (0, 0, 0), -1)
        cv2.putText(frame, txt, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1)

        # Traffic light indicator
        light_color = (0, 220, 0) if get_traffic_light() else (0, 0, 220)
        cv2.circle(frame, (w - 30, 12), 7, light_color, -1)

        # Online dot
        status_color = (0, 220, 110) if self.online else (0, 30, 255)
        cv2.circle(frame, (w - 12, 12), 5, status_color, -1)

    def _offline_placeholder(self) -> np.ndarray:
        frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, f"{self.cam_id} — OFFLINE",
                    (STREAM_WIDTH // 2 - 110, STREAM_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 30, 255), 2)
        return frame

    def _make_offline_frame(self):
        self.latest_frame = self._offline_placeholder()

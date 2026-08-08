"""
config.py — TrafficGuard central configuration
"""
import os

# ── Camera sources ───────────────────────────────────────────────
CAM_LAPTOP_SOURCE = 0          # laptop webcam (AVFoundation index)
CAM_PI_SOURCE     = None       # set to RTSP URL or int when Pi is ready

# ── Model paths ──────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
TRAFFIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'traffic.pt')
HELMET_MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'helmet.pt')

# ── Stream settings ──────────────────────────────────────────────
FRAME_SKIP    = 3        # run YOLO every N frames
STREAM_WIDTH  = 1280 #640
STREAM_HEIGHT = 720 #360
JPEG_QUALITY  = 85

# ── Stop-line violation ──────────────────────────────────────────
LINE_Y = 540             # pixel row; tune by watching the live feed

# ── Wrong-side detection ─────────────────────────────────────────
WRONG_SIDE_MIN_DX  = 12  # minimum pixel movement to count as directional
WRONG_SIDE_HISTORY = 10  # how many past positions to keep per vehicle
WRONG_SIDE_FRAMES  = 4   # must move wrong way for this many frames

# ── Helmet detection ─────────────────────────────────────────────
EXPAND_PX             = 100    # pixels to expand crop upward above motorcycle box
HELMET_CONF_THRESHOLD = 0.30  # minimum confidence to flag no-helmet

# Helmet model class IDs
# class 0 = no_helmet / without_helmet  (violation)
# class 1 = helmet / with_helmet        (safe)
NO_HELMET_CLASS_ID = 0

# ── Triple riding ────────────────────────────────────────────────
TRIPLE_PERSON_THRESHOLD = 3  # persons overlapping motorcycle to flag triple

# ── Tracking ─────────────────────────────────────────────────────
MAX_MISSING_FRAMES = 10   # frames before a lost track is purged

# ── Evidence / DB ────────────────────────────────────────────────
SAVE_EVIDENCE     = True
EVIDENCE_DIR      = os.path.join(BASE_DIR, 'evidence')
EVIDENCE_BASE_DIR = EVIDENCE_DIR          # alias used by utils.py
DB_PATH           = os.path.join(BASE_DIR, 'database', 'violations.db')
DATABASE_PATH     = DB_PATH               # alias used by database.py

# ── Flask ────────────────────────────────────────────────────────
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001

# ── Detection thresholds ─────────────────────────────────────────
TRAFFIC_CONF = 0.25
HELMET_CONF  = 0.30

# ── COCO class IDs ───────────────────────────────────────────────
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# person class in COCO
PERSON_CLASS_ID = 0

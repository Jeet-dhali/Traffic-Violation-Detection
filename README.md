# TrafficGuard

Real-time traffic violation detection system using YOLOv8, Flask, and OpenCV.

## Project Structure

```
trafficguard/
├── app.py              ← Flask server, routes, MJPEG stream
├── camera.py           ← Threaded camera reader + YOLO processing
├── detection.py        ← Model loading and inference
├── violations.py       ← All violation logic (line, wrong-side, helmet, triple)
├── database.py         ← SQLite helpers (init, insert, query)
├── utils.py            ← Folder creation, evidence image saving
├── config.py           ← ⭐ ALL settings — edit this first
├── requirements.txt
├── models/
│   ├── traffic.pt      ← Vehicle + person detection model
│   └── helmet.pt       ← Helmet detection model (optional)
├── templates/
│   └── index.html      ← Home page: dual feed + live violations
├── evidence/           ← Saved violation crops (auto-created)
│   ├── line/
│   ├── wrongside/
│   ├── helmet/
│   └── triple/
└── database/
    └── violations.db   ← SQLite database (auto-created)
```

---

## Setup

```bash
cd trafficguard
pip install -r requirements.txt
python app.py
# Open → http://localhost:5001
```

---

## ⭐ Where to Customise

### 1. Cameras  →  `config.py`
```python
CAM_LAPTOP_SOURCE  = 0          # 0 = default webcam
CAM_PI_SOURCE      = 1          # 1 = second USB, or "rtsp://IP:PORT/stream"
CAM_PI_LOCATION    = "Junction" # shown on UI and saved in DB
```

### 2. Models  →  `config.py`
```python
TRAFFIC_MODEL_PATH = "models/traffic.pt"   # swap to any YOLOv8 .pt file
HELMET_MODEL_PATH  = "models/helmet.pt"    # set to None to disable
```
If your custom model has different class IDs:
```python
VEHICLE_CLASSES  = {2:"car", 3:"motorcycle", ...}  # COCO IDs → labels
NO_HELMET_CLASS_ID = 1    # which class in helmet.pt means "no helmet"
```

### 3. Stop line  →  `config.py`
```python
LINE_Y = 270   # Y pixel on the 640×360 frame where the stop line is drawn
               # Run the app, look at the feed, adjust until line is correct
```

### 4. Wrong-side direction  →  `config.py`
```python
WRONG_SIDE_DIRECTION = "left"   # "left" or "right"
WRONG_SIDE_MIN_DX    = 10       # min pixels/frame to count as movement
```

### 5. Performance  →  `config.py`
```python
FRAME_SKIP   = 3    # run YOLO every N frames (higher = smoother, less accurate)
JPEG_QUALITY = 75   # stream quality (lower = faster, worse image)
```

### 6. Violation logic  →  `violations.py`
Each function has a `CUSTOMISE` docblock explaining exactly what to change:
- `check_violations()` — change which edge triggers, or require partial crossing
- `check_wrong_side()` — change direction or minimum displacement
- `check_helmet_violation()` — change crop expansion, model threshold
- `check_triple_riding()` — change IOU threshold or person count

### 7. Evidence images  →  `utils.py`
```python
PADDING = 20   # pixels of context around the bounding box in saved crops
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Home page |
| GET | `/video_feed/<cam_id>` | MJPEG stream |
| GET | `/api/violations` | JSON list (`?limit=50&cam_id=CAM-PI&vtype=helmet`) |
| GET | `/api/stats` | Today's counts + camera status |
| POST | `/api/violations/<id>/review` | Mark a violation as reviewed |
| GET | `/evidence/<path>` | Serve a saved evidence image |

---

## Connecting Raspberry Pi Camera

**Option A — USB webcam on Pi:**
```python
CAM_PI_SOURCE = 1   # if Pi cam appears as /dev/video1
```

**Option B — Pi Camera Module via rpicam-vid (RTSP):**
```bash
# On the Pi:
rpicam-vid -t 0 --inline --listen -o tcp://0.0.0.0:8080
```
```python
CAM_PI_SOURCE = "tcp://192.168.1.x:8080"
```

**Option C — Pi + mediamtx RTSP server:**
```bash
# On the Pi, install mediamtx and stream via:
# rtsp://PI_IP:8554/cam
```
```python
CAM_PI_SOURCE = "rtsp://192.168.1.x:8554/cam"
```

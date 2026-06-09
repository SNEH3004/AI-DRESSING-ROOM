# AI Virtual Dressing Room — Gesture-Controlled Clothing Carousel

## Project Structure

```
project/
├── app.py                   # Flask entry point
├── hand_service.py          # MediaPipe Hands wrapper
├── gesture_service.py       # Swipe detection engine (GestureController)
├── pose_service.py          # MediaPipe Pose + shirt geometry
├── tryon_service.py         # Main orchestrator + OpenCV renderer
├── requirements.txt
├── templates/
│   └── index.html           # React SPA shell
└── static/
    ├── shirts/
    │   ├── shirt1.png
    │   ├── shirt2.png
    │   ├── shirt3.png
    │   ├── shirt4.png
    │   └── shirt5.png
    └── js/
        └── App.jsx          # React frontend component
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place shirt PNGs
Put your shirt images (with transparent backgrounds) in `static/shirts/`:
```
static/shirts/shirt1.png  … shirt5.png
```
Shirts **must** be RGBA (transparent background) PNGs for the overlay to work correctly.

### 3. Run the server
```bash
python app.py
```

### 4. Open the app
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Architecture

### `hand_service.py` — `HandTracker`
| Feature | Detail |
|---|---|
| Library | MediaPipe Hands |
| Hands | 1 (single-hand mode) |
| Landmark | INDEX_FINGER_TIP (index 8) |
| Smoothing | Weighted moving average over 6 frames |
| Output | Smoothed `(x, y)` pixel tuple or `None` |

### `gesture_service.py` — `GestureController`
| Parameter | Value | Description |
|---|---|---|
| `move_threshold` | 120 px | Minimum horizontal displacement |
| `min_time` | 0.30 s | Minimum gesture duration |
| `max_time` | 0.80 s | Maximum gesture duration (abandons candidate) |
| `cooldown` | 1.00 s | Post-gesture lockout period |
| `vertical_ratio` | 0.65 | Rejects if `|Δy|/|Δx| > 0.65` |
| `min_confidence` | 60.0 | Minimum confidence to fire gesture |

**Confidence formula** (weighted blend, 0–100):
```
confidence = 0.35 × displacement_score
           + 0.20 × speed_score
           + 0.25 × horizontal_score
           + 0.20 × direction_consistency_score
```

### `pose_service.py` — `PoseTracker`
- Uses MediaPipe Pose (model_complexity=1)
- Anchors: LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP
- Shirt width  = shoulder_width × 1.55
- Shirt height = torso_height   × 1.35
- Smoothed over 5 frames via weighted moving average

### `tryon_service.py` — `TryOnService`
- Orchestrates all three sub-services
- Manages `current_shirt_index` (0-indexed, wraps around)
- Composites RGBA shirt PNG via per-pixel alpha blending
- Renders animated HUD: shirt counter, gesture label, confidence bar, FPS, cooldown ring, finger-tip dot, swipe hints
- JPEG-encodes final frame at quality=88

### `app.py` — Flask backend
| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves React SPA |
| `/video_feed` | GET | Multipart MJPEG stream |
| `/status` | GET | JSON: shirt index, fps, gesture, confidence, cooldown |
| `/shirt/<int>` | POST | Jump to specific shirt index |

---

## Gesture Usage Guide

1. Hold one hand in front of the camera.
2. Extend your index finger.
3. **Swipe Right** (→): Move finger right >120 px in 0.3–0.8 seconds → next shirt.
4. **Swipe Left** (←): Move finger left  >120 px in 0.3–0.8 seconds → previous shirt.
5. A 1-second cooldown prevents double-triggers.

---

## Performance Notes
- Target: ≥30 FPS at 1280×720
- MediaPipe runs at `model_complexity=1` for balance of accuracy and speed
- OpenCV capture buffer set to 1 frame to minimise latency
- Alpha compositing uses vectorised NumPy (no Python loops per pixel)
- Flask runs in `threaded=True` mode; a single `threading.Lock` protects shared state

---

## Shirt Image Requirements
- Format: PNG with alpha channel (RGBA)
- Background: fully transparent
- Recommended resolution: at least 400×600 px
- The overlay engine auto-resizes to fit the detected torso

"""
app.py
------
Flask backend for the AI Virtual Dressing Room.

Routes
======
GET  /             → serves index.html (React SPA)
GET  /video_feed   → multipart MJPEG stream
GET  /status       → JSON: shirt index, fps, gesture info
POST /shirt/<int>  → manually jump to a specific shirt index
"""

import threading
import cv2
import os
from flask import Flask, Response, jsonify, abort, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from tryon_service import TryOnService

# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configure uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# Shared state – protected by a lock for thread safety
# ---------------------------------------------------------------------------
_lock        = threading.Lock()
_tryon_svc   = TryOnService(shirts_dir="static/shirts")
_cap         = cv2.VideoCapture(0)

# Configure capture for performance
_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
_cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
_cap.set(cv2.CAP_PROP_FPS,           30)
_cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)   # minimise latency


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------

def _generate_frames():
    """
    Infinite generator: reads one frame from the webcam, runs the full
    try-on pipeline, and yields a multipart JPEG boundary chunk.
    """
    while True:
        ok, frame = _cap.read()
        if not ok:
            continue

        with _lock:
            jpeg_bytes = _tryon_svc.process_frame(frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with _lock:
        svc = _tryon_svc
        return jsonify({
            "current_shirt": svc.selected_shirt_index,
            "applied_shirt": svc.applied_shirt_index,
            "total_shirts":  len(svc._shirts),
            "fps":           round(svc._fps, 1),
            "gesture":       svc._last_gesture.gesture,
            "confidence":    svc._last_gesture.confidence,
            "in_cooldown":   svc._gesture_ctrl.in_cooldown,
        })


@app.route("/shirt/<int:index>", methods=["POST"])
def set_shirt(index: int):
    with _lock:
        n = len(_tryon_svc._shirts)
        if index < 0 or index >= n:
            abort(400, description=f"Index must be 0–{n - 1}")
        _tryon_svc.selected_shirt_index = index
        if _tryon_svc.applied_shirt_index is not None:
            _tryon_svc.apply_selected_shirt()
        return jsonify({
            "current_shirt": index,
            "applied_shirt": _tryon_svc.applied_shirt_index,
        })


@app.route("/apply", methods=["POST"])
def apply_shirt():
    with _lock:
        _tryon_svc.apply_selected_shirt()
        return jsonify({"applied_shirt": _tryon_svc.applied_shirt_index})


@app.route("/")
def index():
    from flask import render_template
    print("[DEBUG] Serving index.html")
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    """Upload an image and optionally remove background"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        remove_bg = request.form.get('removeBg', 'false').lower() == 'true'
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        filename = secure_filename(file.filename)
        # Convert to PNG if needed
        name, ext = os.path.splitext(filename)
        filename = name + '.png'
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Remove background if requested
        if remove_bg:
            try:
                import numpy as np
                img = cv2.imread(filepath)
                if img is not None:
                    # Simple background removal using HSV color space
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # Create mask for non-white pixels (simple background removal)
                    lower_white = np.array([0, 0, 200])
                    upper_white = np.array([255, 30, 255])
                    mask = cv2.inRange(hsv, lower_white, upper_white)
                    mask = cv2.bitwise_not(mask)
                    
                    # Apply morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    
                    # Convert to RGBA and apply mask
                    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = mask
                    
                    cv2.imwrite(filepath, rgba)
            except Exception as e:
                print(f"Background removal error: {e}")
        
        # Add to shirts if it's a background-removed image
        if remove_bg:
            with _lock:
                try:
                    shirt_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    if shirt_img is not None:
                        if len(shirt_img.shape) == 3 and shirt_img.shape[2] == 3:
                            shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2BGRA)
                        _tryon_svc._shirts.append(shirt_img)
                        return jsonify({
                            "success": True, 
                            "filename": filename,
                            "message": "Image uploaded and added to shirts",
                            "shirt_index": len(_tryon_svc._shirts) - 1
                        })
                except Exception as e:
                    print(f"Error adding shirt: {e}")
                    return jsonify({"error": str(e)}), 500
        
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "Image uploaded successfully"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/capture", methods=["POST"])
def capture_frame():
    """Capture current frame when hand is folded"""
    try:
        with _lock:
            # Get current frame from video capture
            ret, frame = _cap.read()
            if not ret:
                return jsonify({"error": "Failed to capture frame"}), 500
            
            # Save captured frame
            capture_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.png')
            cv2.imwrite(capture_path, frame)
            
            return jsonify({
                "success": True,
                "message": "Frame captured",
                "path": f"/static/uploads/capture.png"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _cap.release()
        _tryon_svc.release()

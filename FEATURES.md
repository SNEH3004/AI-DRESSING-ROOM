# AI Virtual Dressing Room - Features Guide

## ✨ What's Working

### 1. **Live Try-On with Shirt Overlay** 👕
- **Real-time video feed** from your webcam
- **Automatic shirt overlay** - The selected shirt overlays directly on your body using AI pose detection
- **Body tracking** - Uses MediaPipe to detect your body landmarks and position the shirt correctly

### 2. **Hand Swipe Gesture Recognition** 👐
- **Swipe Right** (→) - Go to NEXT shirt
- **Swipe Left** (←) - Go to PREVIOUS shirt
- **Gesture Detection** - Real-time hand tracking shows your hand position
- **Confidence Score** - Shows how confident the gesture is (0-100%)
- **Cooldown** - Prevents multiple accidental gestures

### 3. **Shirt Selection** 🎨
- **4 Default Shirts** - Orange, Red, Blue, and Green shirts
- **Manual Selection** - Click shirt buttons to switch immediately
- **Real-time Updates** - Status shows current shirt (e.g., "Shirt 1 / 4")

### 4. **Frame Capture** 📸
- **Capture Button** - Click to capture the current frame
- **Hand Folding** - Fold your hand to trigger capture (when implemented)
- **Saved Frames** - Captured images are saved to `static/uploads/capture.png`

### 5. **Image Upload & Background Removal** 🖼️
- **Upload Your Own Clothes** - Add your own shirt images
- **Auto Background Removal** - Converts white/light backgrounds to transparent PNG
- **Smart Conversion** - Automatically saves as PNG format
- **Add to Collection** - Uploaded clothes become available in the try-on

## 🚀 How to Use

### Changing Clothes (Main Feature)
1. **Open app**: http://127.0.0.1:5000/
2. **Position yourself** in front of the webcam
3. **Swipe your hand left or right** to change clothes
4. **Watch the shirt change** in real-time on your body

### Uploading Custom Clothes
1. Go to the **"📤 Upload Your Own Clothes"** section
2. Check **"Remove background from image"** (for transparent PNG)
3. Click **"Choose Image"** and select a shirt photo
4. Click **"Upload & Add to Collection"**
5. Your uploaded shirt will appear in the shirt selection buttons

### Capturing Frames
1. Click the **"📸 Capture Frame"** button
2. Your current frame is saved to `static/uploads/capture.png`
3. Use captured images to share your virtual try-on

## 🎯 Technical Details

### Gesture Detection
- **Algorithm**: Horizontal swipe detection with direction and confidence scoring
- **Thresholds**: 120px minimum movement, 0.3-0.8 second duration
- **Cooldown**: 1 second between gestures to prevent accidental triggers

### Shirt Overlay
- **Alpha Compositing**: Transparent PNG shirts blend smoothly onto video
- **Pose Tracking**: MediaPipe detects 33 body landmarks
- **Dynamic Sizing**: Shirt size adjusts based on body detection

### Background Removal
- **Method**: HSV color space filtering for white/light backgrounds
- **Processing**: Morphological operations for clean edges
- **Output**: RGBA PNG with transparent background

## 📁 File Structure
```
files (1)/
├── app.py                    # Flask backend with routes
├── tryon_service.py          # Core try-on logic
├── gesture_service.py        # Swipe gesture detection
├── hand_service.py           # Hand tracking
├── pose_service.py           # Body pose detection
├── requirements.txt          # Python dependencies
├── static/
│   ├── shirts/              # Shirt images (1.png, 2.png, 3.png, 4.png)
│   ├── uploads/             # Uploaded images and captures
│   └── js/
│       └── App.jsx          # React app (for reference)
└── templates/
    └── index.html           # Main interface
```

## 🔧 API Endpoints

- `GET /` - Main interface
- `GET /video_feed` - MJPEG video stream
- `GET /status` - Current status (shirt, FPS, gesture)
- `POST /shirt/<index>` - Select shirt
- `POST /upload` - Upload image with optional background removal
- `POST /capture` - Capture current frame

## ⚙️ System Requirements
- Python 3.7+
- Webcam
- Flask + Flask-CORS
- OpenCV
- MediaPipe
- TensorFlow Lite

## 🎮 Tips for Best Results

1. **Good Lighting** - Face your light source for better pose detection
2. **Clear Background** - Plain backgrounds help with tracking
3. **Full Body Visible** - Stand back so your torso is fully visible
4. **Smooth Gestures** - Swipe smoothly and horizontally for best detection
5. **Natural Pace** - Gesture recognition works best at natural speed

## 🐛 Known Limitations

- Background removal works best with white/light colored backgrounds
- Hand detection requires visible hand in frame
- Pose detection improves with better lighting
- First 2-3 seconds may have reduced accuracy as models initialize

## 🎓 What's Next?

Possible enhancements:
- Add advanced background removal using AI (rembg library)
- Implement hand folding detection for auto-capture
- Add more gesture types (thumbs up, peace sign, etc.)
- Save try-on videos
- Add color customization for shirts

Enjoy your AI Virtual Dressing Room! 🎉

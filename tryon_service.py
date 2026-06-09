"""
tryon_service.py
----------------
Core try-on service for the AI Virtual Dressing Room.

Integrates
==========
* PoseTracker        – body landmark extraction & shirt geometry
* HandTracker        – index-finger-tip position
* GestureController  – swipe detection & cooldown
* Shirt carousel     – current_shirt_index management
* OpenCV renderer    – final composited frame with animated HUD

Public interface
================
    svc = TryOnService("static/shirts")
    frame_bytes = svc.process_frame(bgr_frame)   # call each frame
    svc.release()
"""

import os
import time
import cv2
import numpy as np
from typing import List, Optional, Tuple

from hand_service    import HandTracker
from gesture_service import GestureController, GESTURE_SWIPE_LEFT, GESTURE_SWIPE_RIGHT, GestureResult
from pose_service    import PoseTracker, ShirtGeometry


# ---------------------------------------------------------------------------
# Helper – safe alpha composite
# ---------------------------------------------------------------------------

def _alpha_composite(
    background: np.ndarray,
    overlay_rgba: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """
    Blend a resized RGBA overlay onto a BGR background in-place.
    Handles partial out-of-bounds overlay correctly.
    """
    if w <= 0 or h <= 0:
        return

    resized = cv2.resize(overlay_rgba, (w, h), interpolation=cv2.INTER_AREA)

    bh, bw = background.shape[:2]

    # Source ROI (within the overlay image)
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = min(w, bw - x)
    src_y2 = min(h, bh - y)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return          # entirely off-screen

    # Destination ROI (within the background)
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Extract channels
    overlay_rgb = resized[src_y1:src_y2, src_x1:src_x2, :3]
    alpha        = resized[src_y1:src_y2, src_x1:src_x2,  3:4].astype(np.float32) / 255.0

    roi = background[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    blended = roi * (1.0 - alpha) + overlay_rgb.astype(np.float32) * alpha
    background[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# HUD Renderer
# ---------------------------------------------------------------------------

class _HUDRenderer:
    """
    Renders the animated heads-up display overlay onto the frame.

    Includes
    --------
    * Shirt index / total indicator (top-centre)
    * Gesture label with animated arrow
    * Confidence bar
    * FPS counter
    * Cooldown indicator
    * Finger-tip dot
    """

    # Fonts & sizes
    _FONT       = cv2.FONT_HERSHEY_DUPLEX
    _FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

    # Palette (BGR)
    _WHITE  = (255, 255, 255)
    _BLACK  = (0,   0,   0)
    _CYAN   = (255, 220,  60)
    _GREEN  = (80,  230,  80)
    _ORANGE = (0,   160, 255)
    _RED    = (60,   60, 230)
    _TEAL   = (200, 230,  60)

    def __init__(self) -> None:
        # Animated arrow phase (driven by time)
        self._anim_phase = 0.0

    def render(
        self,
        frame:               np.ndarray,
        shirt_idx:           int,
        selected_shirt_idx:  int,
        applied_shirt_idx:   Optional[int],
        shirt_total:         int,
        gesture_result:      GestureResult,
        fps:                 float,
        in_cooldown:         bool,
        finger_tip:          Optional[Tuple[int, int]],
    ) -> None:
        """Draw all HUD elements onto *frame* in-place."""

        h, w = frame.shape[:2]
        now  = time.time()
        self._anim_phase = (now % 1.0)   # 0→1 sawtooth for animation

        # ── Translucent top banner ─────────────────────────────────────
        self._draw_banner(frame, w)

        # ── Shirt counter (e.g. "Shirt 3 / 5") ────────────────────────
        shirt_text = f"Shirt {shirt_idx + 1} / {shirt_total}"
        self._draw_centred_text(frame, shirt_text, y=38, scale=0.90,
                                 color=self._WHITE, thickness=2)

        # ── Gesture label + animated arrow ────────────────────────────
        self._draw_gesture_label(frame, gesture_result, w, h)

        # ── Selected / applied status badges ───────────────────────────
        self._draw_selection_badges(frame, selected_shirt_idx, applied_shirt_idx, shirt_total)

        # ── Confidence bar ────────────────────────────────────────────
        if gesture_result.confidence > 0:
            self._draw_confidence_bar(frame, gesture_result.confidence, w, h)

        # ── FPS counter (bottom-left) ─────────────────────────────────
        fps_text = f"FPS: {fps:.0f}"
        cv2.putText(frame, fps_text, (12, h - 14),
                    self._FONT_SMALL, 0.55, self._TEAL, 1, cv2.LINE_AA)

        # ── Cooldown flash indicator ───────────────────────────────────
        if in_cooldown:
            self._draw_cooldown_ring(frame, w, h)

        # ── Finger-tip dot ────────────────────────────────────────────
        if finger_tip:
            self._draw_finger_dot(frame, finger_tip)

        # ── Swipe hint arrows (bottom-centre) ─────────────────────────
        self._draw_swipe_hints(frame, w, h)

    # ------------------------------------------------------------------
    # Private draw helpers
    # ------------------------------------------------------------------

    def _draw_banner(self, frame: np.ndarray, w: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def _draw_centred_text(
        self,
        frame:     np.ndarray,
        text:      str,
        y:         int,
        scale:     float = 0.8,
        color:     tuple = (255, 255, 255),
        thickness: int   = 1,
    ) -> None:
        w = frame.shape[1]
        (tw, _), _ = cv2.getTextSize(text, self._FONT, scale, thickness)
        x = (w - tw) // 2
        # Shadow
        cv2.putText(frame, text, (x + 1, y + 1),
                    self._FONT, scale, self._BLACK, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    self._FONT, scale, color, thickness, cv2.LINE_AA)

    def _draw_gesture_label(
        self,
        frame:          np.ndarray,
        result:         GestureResult,
        w:              int,
        h:              int,
    ) -> None:
        if not result.gesture or result.gesture == "none":
            return

        # Animated arrow offset (±8 px oscillation)
        offset = int(8 * abs(np.sin(self._anim_phase * np.pi)))

        if result.direction == "right":
            arrow   = "→"
            label   = f"NEXT SHIRT  {arrow}"
            color   = self._GREEN
            x_arrow = w - 180 - offset
            x_label = w - 220
        else:
            arrow   = "←"
            label   = f"{arrow}  PREV SHIRT"
            color   = self._ORANGE
            x_arrow = 140 + offset
            x_label = 20

        # Background pill
        (lw, lh), _ = cv2.getTextSize(label, self._FONT, 0.80, 2)
        pill_x1 = x_label - 8
        pill_y1 = h - 95
        pill_x2 = pill_x1 + lw + 16
        pill_y2 = pill_y1 + lh + 14

        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, label, (x_label, h - 74),
                    self._FONT, 0.80, color, 2, cv2.LINE_AA)

    def _draw_selection_badges(
        self,
        frame:               np.ndarray,
        selected_shirt_idx:  int,
        applied_shirt_idx:   Optional[int],
        shirt_total:         int,
    ) -> None:
        selected_text = f"Selected: {selected_shirt_idx + 1}/{shirt_total}"
        if applied_shirt_idx is None:
            applied_text = "Applied: None"
        else:
            applied_text = f"Applied: {applied_shirt_idx + 1}/{shirt_total}"

        badge_y = 46
        self._draw_small_text(frame, selected_text, 14, badge_y, self._CYAN)
        self._draw_small_text(frame, applied_text, 14, badge_y + 22, self._ORANGE)

    def _draw_small_text(
        self,
        frame:   np.ndarray,
        text:    str,
        x:       int,
        y:       int,
        color:   tuple,
    ) -> None:
        cv2.putText(frame, text, (x, y),
                    self._FONT_SMALL, 0.55, self._BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    self._FONT_SMALL, 0.55, color, 1, cv2.LINE_AA)

    def _draw_confidence_bar(
        self,
        frame:      np.ndarray,
        confidence: float,
        w:          int,
        h:          int,
    ) -> None:
        bar_w  = 200
        bar_h  = 12
        bar_x  = (w - bar_w) // 2
        bar_y  = h - 50

        # Background track
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1, cv2.LINE_AA)

        # Fill
        fill_w = int(bar_w * confidence / 100.0)
        # Colour: green > 75, orange > 50, red otherwise
        if confidence >= 75:
            bar_color = (60, 210, 60)
        elif confidence >= 50:
            bar_color = (0, 165, 255)
        else:
            bar_color = (60, 60, 220)

        if fill_w > 0:
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          bar_color, -1, cv2.LINE_AA)

        # Label
        conf_text = f"Confidence: {confidence:.0f}%"
        self._draw_centred_text(frame, conf_text, y=bar_y - 4,
                                 scale=0.50, color=self._WHITE, thickness=1)

    def _draw_cooldown_ring(
        self,
        frame: np.ndarray,
        w:     int,
        h:     int,
    ) -> None:
        """Pulsing red border to indicate cooldown period."""
        pulse = int(120 * abs(np.sin(self._anim_phase * np.pi * 2)))
        color = (0, 0, pulse + 80)
        cv2.rectangle(frame, (3, 3), (w - 3, h - 3), color, 3)

    def _draw_finger_dot(
        self,
        frame:     np.ndarray,
        finger_tip: Tuple[int, int],
    ) -> None:
        cx, cy = finger_tip
        # Outer glow
        cv2.circle(frame, (cx, cy), 16, (0, 200, 255), 2, cv2.LINE_AA)
        # Inner dot
        cv2.circle(frame, (cx, cy),  6, (0, 240, 255), -1, cv2.LINE_AA)

    def _draw_swipe_hints(
        self,
        frame: np.ndarray,
        w:     int,
        h:     int,
    ) -> None:
        """Subtle always-visible swipe hint arrows at bottom edges."""
        alpha = 0.45 + 0.15 * abs(np.sin(self._anim_phase * np.pi))
        hint_y = h - 22

        # Left arrow hint
        left_text = "\u2190 Swipe"
        (lw, _), _ = cv2.getTextSize(left_text, self._FONT_SMALL, 0.48, 1)
        overlay = frame.copy()
        cv2.putText(overlay, left_text, (w // 2 - lw - 30, hint_y),
                    self._FONT_SMALL, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(overlay, "Swipe \u2192", (w // 2 + 30, hint_y),
                    self._FONT_SMALL, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# ---------------------------------------------------------------------------
# TryOnService  (main public class)
# ---------------------------------------------------------------------------

class TryOnService:
    """
    Orchestrates body pose, hand tracking, gesture recognition, shirt
    carousel management, and composited frame rendering.

    Parameters
    ----------
    shirts_dir : str  – path to the directory containing shirt PNGs
    """

    _SHIRT_FILENAMES = [
        "1.png", "2.png", "3.png", "4.png",
    ]

    def __init__(self, shirts_dir: str = "static/shirts") -> None:
        self._shirts_dir = shirts_dir

        # ── Load shirt images ──────────────────────────────────────────
        self._shirts: List[np.ndarray] = self._load_shirts()
        if not self._shirts:
            raise RuntimeError(
                f"No shirt images found in '{shirts_dir}'. "
                "Ensure 1.png, 2.png, 3.png, 4.png are present."
            )

        self.current_shirt_index: int = 0
        self.selected_shirt_index: int = 0
        self.applied_shirt_index: Optional[int] = None

        # ── Sub-services ───────────────────────────────────────────────
        self._pose_tracker = PoseTracker()
        self._hand_tracker = HandTracker()
        self._gesture_ctrl = GestureController(
            move_threshold = 120.0,
            min_time       = 0.30,
            max_time       = 0.80,
            cooldown       = 1.00,
            history_frames = 20,
            min_confidence = 60.0,
            vertical_ratio = 0.65,
        )

        # ── HUD renderer ───────────────────────────────────────────────
        self._hud = _HUDRenderer()

        # ── FPS tracking ───────────────────────────────────────────────
        self._frame_times: list = []
        self._fps: float        = 0.0

        # ── Last gesture result (kept until next gesture) ──────────────
        self._last_gesture: GestureResult  = GestureController().last_result
        self._gesture_display_until: float = 0.0   # epoch when to clear label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame: np.ndarray) -> bytes:
        """
        Full processing pipeline for one webcam frame.

        1. Convert BGR → RGB
        2. Run pose estimation → ShirtGeometry
        3. Run hand tracking  → finger-tip position
        4. Run gesture engine → GestureResult
        5. Update shirt carousel if valid swipe
        6. Composite shirt PNG onto frame
        7. Render HUD
        8. JPEG-encode and return bytes

        Parameters
        ----------
        bgr_frame : np.ndarray  – raw OpenCV webcam frame (BGR, uint8)

        Returns
        -------
        bytes  – JPEG-encoded composited frame
        """
        frame = bgr_frame.copy()
        self._update_fps()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Pose → shirt geometry ──────────────────────────────────────
        geom: ShirtGeometry = self._pose_tracker.process(rgb)

        # ── Hand → finger-tip ─────────────────────────────────────────
        finger_tip = self._hand_tracker.process(rgb)

        # ── Gesture engine ────────────────────────────────────────────
        gesture_result = self._gesture_ctrl.update(finger_tip)

        if gesture_result.gesture != "none":
            self._handle_gesture(gesture_result)

        # Keep the last non-trivial gesture visible for 0.8 s on screen
        now = time.time()
        if gesture_result.gesture != "none":
            self._last_gesture            = gesture_result
            self._gesture_display_until   = now + 0.80
        displayed_gesture = (
            self._last_gesture
            if now < self._gesture_display_until
            else gesture_result
        )

        # ── Draw hand skeleton ────────────────────────────────────────
        self._hand_tracker.draw_landmarks(frame)

        # ── Composite shirt only when the selected shirt has been applied ─────────────────────────────────
        if geom.valid and self.applied_shirt_index is not None:
            shirt_rgba = self._shirts[self.applied_shirt_index]
            _alpha_composite(frame, shirt_rgba, geom.x, geom.y, geom.w, geom.h)

        # ── HUD ─────────────────────────────────────────────────────
        self._hud.render(
            frame             = frame,
            shirt_idx         = self.selected_shirt_index,
            selected_shirt_idx= self.selected_shirt_index,
            applied_shirt_idx = self.applied_shirt_index,
            shirt_total       = len(self._shirts),
            gesture_result    = displayed_gesture,
            fps               = self._fps,
            in_cooldown       = self._gesture_ctrl.in_cooldown,
            finger_tip        = finger_tip,
        )

        # ── Encode ────────────────────────────────────────────────────
        _, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        return buf.tobytes()

    def release(self) -> None:
        """Release all held resources."""
        self._pose_tracker.release()
        self._hand_tracker.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_gesture(self, result) -> None:
        """Update the selected (and optionally applied) shirt index based on swipe."""
        n = len(self._shirts)
        if result.gesture == GESTURE_SWIPE_RIGHT:
            self.selected_shirt_index = (self.selected_shirt_index + 1) % n
        elif result.gesture == GESTURE_SWIPE_LEFT:
            self.selected_shirt_index = (self.selected_shirt_index - 1) % n

        # If a shirt is already applied, gestures should auto-apply future selections.
        if self.applied_shirt_index is not None:
            self.applied_shirt_index = self.selected_shirt_index

    def apply_selected_shirt(self) -> None:
        """Apply the currently selected shirt so the overlay appears on the live feed."""
        self.applied_shirt_index = self.selected_shirt_index

    def _load_shirts(self) -> List[np.ndarray]:
        """
        Load shirt PNGs from the shirts directory.
        Only filenames present on disk are loaded; missing ones are skipped
        with a warning so the app degrades gracefully.
        """
        shirts = []
        for fname in self._SHIRT_FILENAMES:
            path = os.path.join(self._shirts_dir, fname)
            if not os.path.isfile(path):
                print(f"[TryOnService] WARNING: shirt not found → {path}")
                continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[TryOnService] WARNING: could not read → {path}")
                continue
            # Ensure 4-channel RGBA
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            shirts.append(img)
            print(f"[TryOnService] Loaded: {fname}  shape={img.shape}")
        return shirts

    def _update_fps(self) -> None:
        """Rolling 30-frame FPS estimate."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

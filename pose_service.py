"""
pose_service.py
---------------
Production-grade body-pose service using MediaPipe Pose.

Responsibilities
================
* Initialise and manage a MediaPipe Pose instance.
* Extract the subset of landmarks needed for shirt placement:
    - LEFT_SHOULDER  (index 11)
    - RIGHT_SHOULDER (index 12)
    - LEFT_HIP       (index 23)
    - RIGHT_HIP      (index 24)
* Compute shirt overlay geometry:
    - overlay_x, overlay_y  – top-left corner in frame space
    - overlay_w, overlay_h  – target width/height in pixels
* Expose smoothed landmark positions to reduce jitter.
"""

import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Named landmark indices (MediaPipe Pose convention)
# ---------------------------------------------------------------------------

_LM = mp.solutions.pose.PoseLandmark


class ShirtGeometry:
    """Value object describing where and how large to draw the shirt."""

    __slots__ = ("x", "y", "w", "h", "valid")

    def __init__(
        self,
        x: int, y: int,
        w: int, h: int,
        valid: bool = True,
    ) -> None:
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        self.valid = valid

    @classmethod
    def invalid(cls) -> "ShirtGeometry":
        return cls(0, 0, 0, 0, valid=False)


class PoseTracker:
    """
    Wraps MediaPipe Pose and returns shirt-placement geometry each frame.

    Parameters
    ----------
    min_detection_confidence : float
    min_tracking_confidence  : float
    smooth_frames            : int    – landmark smoothing window
    torso_width_scale        : float  – shirt width = shoulder_width × scale
    torso_height_scale       : float  – shirt height = torso_height  × scale
    y_offset_ratio           : float  – shift shirt up by ratio × shirt_h
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence:  float = 0.6,
        smooth_frames:            int   = 5,
        torso_width_scale:        float = 1.55,
        torso_height_scale:       float = 1.35,
        y_offset_ratio:           float = 0.08,
    ) -> None:
        mp_pose = mp.solutions.pose

        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,           # balance of accuracy & speed
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._smooth_frames      = smooth_frames
        self._torso_width_scale  = torso_width_scale
        self._torso_height_scale = torso_height_scale
        self._y_offset_ratio     = y_offset_ratio

        # Smoothing buffers per landmark key
        self._lm_history: Dict[str, deque] = {
            key: deque(maxlen=smooth_frames)
            for key in ("ls", "rs", "lh", "rh")   # shoulders + hips
        }

        self.last_result = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray) -> ShirtGeometry:
        """
        Run pose estimation on an RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray  – RGB image (H, W, 3)

        Returns
        -------
        ShirtGeometry  – placement info, or ShirtGeometry.invalid() when
                         the torso cannot be localised.
        """
        result = self._pose.process(frame_rgb)
        self.last_result = result

        if not result.pose_landmarks:
            return ShirtGeometry.invalid()

        h, w = frame_rgb.shape[:2]
        lm   = result.pose_landmarks.landmark

        # Extract raw pixel positions for the four torso anchors
        def to_px(landmark) -> Tuple[int, int]:
            return (int(landmark.x * w), int(landmark.y * h))

        ls = to_px(lm[_LM.LEFT_SHOULDER])
        rs = to_px(lm[_LM.RIGHT_SHOULDER])
        lh = to_px(lm[_LM.LEFT_HIP])
        rh = to_px(lm[_LM.RIGHT_HIP])

        # Push to smoothing buffers
        self._lm_history["ls"].append(ls)
        self._lm_history["rs"].append(rs)
        self._lm_history["lh"].append(lh)
        self._lm_history["rh"].append(rh)

        # Retrieve smoothed positions
        ls = self._smooth("ls")
        rs = self._smooth("rs")
        lh = self._smooth("lh")
        rh = self._smooth("rh")

        return self._compute_geometry(ls, rs, lh, rh)

    def release(self) -> None:
        """Free MediaPipe resources."""
        self._pose.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth(self, key: str) -> Tuple[int, int]:
        """Return the weighted-average position for a landmark key."""
        buf = list(self._lm_history[key])
        if not buf:
            return (0, 0)
        n       = len(buf)
        weights = np.arange(1, n + 1, dtype=float)
        weights /= weights.sum()
        xs      = np.array([p[0] for p in buf], dtype=float)
        ys      = np.array([p[1] for p in buf], dtype=float)
        return (int(np.dot(weights, xs)), int(np.dot(weights, ys)))

    def _compute_geometry(
        self,
        ls: Tuple[int, int],
        rs: Tuple[int, int],
        lh: Tuple[int, int],
        rh: Tuple[int, int],
    ) -> ShirtGeometry:
        """
        Derive shirt bounding box from the four torso anchors.

        Strategy
        --------
        * shoulder_width = Euclidean distance between shoulders
        * torso_height   = average Y distance from shoulder to hip
        * shirt_w        = shoulder_width  × torso_width_scale
        * shirt_h        = torso_height    × torso_height_scale
        * x              = mid_shoulder_x  − shirt_w / 2
        * y              = mid_shoulder_y  − y_offset_ratio × shirt_h
        """
        # Midpoints
        mid_sx = (ls[0] + rs[0]) // 2
        mid_sy = (ls[1] + rs[1]) // 2

        # Shoulder width
        shoulder_w = int(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
        if shoulder_w < 10:             # degenerate detection
            return ShirtGeometry.invalid()

        # Average vertical torso length
        left_torso  = abs(lh[1] - ls[1])
        right_torso = abs(rh[1] - rs[1])
        torso_h     = (left_torso + right_torso) // 2

        if torso_h < 10:
            return ShirtGeometry.invalid()

        shirt_w = int(shoulder_w  * self._torso_width_scale)
        shirt_h = int(torso_h     * self._torso_height_scale)

        overlay_x = mid_sx - shirt_w // 2
        overlay_y = mid_sy - int(self._y_offset_ratio * shirt_h)

        return ShirtGeometry(
            x=overlay_x,
            y=overlay_y,
            w=shirt_w,
            h=shirt_h,
            valid=True,
        )

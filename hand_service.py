"""
hand_service.py
---------------
Production-grade hand tracking service using MediaPipe Hands.
Tracks a single hand and exposes smoothed index-finger-tip coordinates
for downstream gesture recognition.
"""

import time
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Tuple


class HandTracker:
    """
    Encapsulates MediaPipe Hands initialisation and per-frame landmark
    extraction.  Only the first detected hand is used (single-hand mode).

    Attributes
    ----------
    _hands          : mediapipe.solutions.hands.Hands
    _history_len    : int   – number of frames kept for smoothing
    _coord_history  : deque – ring buffer of (x, y) pixel tuples
    """

    # MediaPipe landmark index for the index-finger tip
    INDEX_FINGER_TIP = 8

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        history_len: int = 6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._history_len   = history_len
        self._coord_history: deque = deque(maxlen=history_len)

        # Expose the last raw result so callers can draw landmarks if needed
        self.last_result = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Run hand detection on an RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray  – BGR→RGB converted frame (H, W, 3)

        Returns
        -------
        (x, y) pixel tuple of the smoothed index-finger tip, or None when
        no hand is detected.
        """
        result = self._hands.process(frame_rgb)
        self.last_result = result

        if not result.multi_hand_landmarks:
            # Flush history so stale data doesn't pollute next gesture
            self._coord_history.clear()
            return None

        # Use only the first detected hand
        landmarks = result.multi_hand_landmarks[0]
        h, w = frame_rgb.shape[:2]

        tip = landmarks.landmark[self.INDEX_FINGER_TIP]
        raw_x = int(tip.x * w)
        raw_y = int(tip.y * h)

        self._coord_history.append((raw_x, raw_y))
        return self._smoothed_position()

    def draw_landmarks(self, bgr_frame: np.ndarray) -> None:
        """
        Overlay hand landmarks and connections on a BGR frame in-place.
        Call after process() each frame.
        """
        if self.last_result and self.last_result.multi_hand_landmarks:
            for hand_lm in self.last_result.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    bgr_frame,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(
                        color=(0, 255, 200), thickness=2, circle_radius=3
                    ),
                    self._mp_draw.DrawingSpec(
                        color=(0, 180, 130), thickness=2
                    ),
                )

    def get_history(self) -> list:
        """Return a copy of the raw coordinate history."""
        return list(self._coord_history)

    def release(self) -> None:
        """Free MediaPipe resources."""
        self._hands.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smoothed_position(self) -> Tuple[int, int]:
        """
        Weighted moving average over the coordinate history.
        More recent frames carry higher weight (linear ramp).
        """
        if len(self._coord_history) == 0:
            return (0, 0)

        history = list(self._coord_history)
        n       = len(history)
        weights = np.arange(1, n + 1, dtype=float)   # 1, 2, …, n
        weights /= weights.sum()

        xs = np.array([p[0] for p in history], dtype=float)
        ys = np.array([p[1] for p in history], dtype=float)

        sx = int(np.round(np.dot(weights, xs)))
        sy = int(np.round(np.dot(weights, ys)))
        return (sx, sy)

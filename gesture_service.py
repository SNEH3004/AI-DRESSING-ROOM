"""
gesture_service.py
------------------
Production-grade swipe-gesture recognition engine.

Design goals
============
* Horizontal swipe-left / swipe-right detection only.
* Configurable movement threshold, time window, and cooldown.
* Confidence score (0–100) that weighs displacement, directional
  consistency, and horizontal dominance.
* Thread-safe cooldown via time stamps (no threading primitives needed
  because Flask's video-loop is single-threaded per process).
* Clean OOP: one GestureController class owns all state.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

GESTURE_NONE       = "none"
GESTURE_SWIPE_LEFT  = "swipe_left"
GESTURE_SWIPE_RIGHT = "swipe_right"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GestureResult:
    """Value object returned by GestureController.update()."""
    gesture:    str   = GESTURE_NONE
    confidence: float = 0.0          # 0–100
    direction:  str   = ""           # "left" | "right" | ""


@dataclass
class _SwipeCandidate:
    """Tracks an in-progress swipe attempt."""
    start_x:   float = 0.0
    start_y:   float = 0.0
    start_time: float = field(default_factory=time.time)
    peak_disp:  float = 0.0          # max horizontal displacement seen


# ---------------------------------------------------------------------------
# GestureController
# ---------------------------------------------------------------------------

class GestureController:
    """
    Detects horizontal swipe gestures from a stream of (x, y) finger-tip
    positions and manages a post-gesture cooldown period.

    Parameters
    ----------
    move_threshold   : float  – minimum horizontal pixels to count as swipe
    min_time         : float  – minimum gesture duration in seconds
    max_time         : float  – maximum gesture duration in seconds
    cooldown         : float  – seconds to ignore gestures after one fires
    history_frames   : int    – position samples kept for direction analysis
    min_confidence   : float  – gestures below this score are discarded
    vertical_ratio   : float  – reject if |Δy|/|Δx| exceeds this value
    """

    def __init__(
        self,
        move_threshold: float = 120.0,
        min_time:       float = 0.3,
        max_time:       float = 0.8,
        cooldown:       float = 1.0,
        history_frames: int   = 20,
        min_confidence: float = 60.0,
        vertical_ratio: float = 0.65,
    ) -> None:
        self.move_threshold = move_threshold
        self.min_time       = min_time
        self.max_time       = max_time
        self.cooldown       = cooldown
        self.history_frames = history_frames
        self.min_confidence = min_confidence
        self.vertical_ratio = vertical_ratio

        # Rolling position buffer  [(x, y, timestamp), ...]
        self._history: deque = deque(maxlen=history_frames)

        # Active candidate swipe
        self._candidate: Optional[_SwipeCandidate] = None

        # Epoch of last fired gesture (0 → never fired)
        self._last_gesture_time: float = 0.0

        # Most recent result (exposed for UI rendering)
        self.last_result: GestureResult = GestureResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, finger_tip: Optional[Tuple[int, int]]
    ) -> GestureResult:
        """
        Feed the latest smoothed finger-tip coordinate and receive a
        GestureResult.  Call once per processed video frame.

        Parameters
        ----------
        finger_tip : (x, y) pixel tuple, or None when no hand is visible.

        Returns
        -------
        GestureResult with gesture name, confidence, and direction.
        """
        now = time.time()

        if finger_tip is None:
            self._reset_candidate()
            self.last_result = GestureResult()
            return self.last_result

        x, y = finger_tip
        self._history.append((x, y, now))

        # ── Cooldown gate ──────────────────────────────────────────────
        if self._in_cooldown(now):
            self.last_result = GestureResult()
            return self.last_result

        # ── Candidate initialisation ───────────────────────────────────
        if self._candidate is None:
            self._candidate = _SwipeCandidate(
                start_x=x, start_y=y, start_time=now
            )
            self.last_result = GestureResult()
            return self.last_result

        # ── Evaluate running candidate ─────────────────────────────────
        result = self._evaluate(x, y, now)
        self.last_result = result
        return result

    @property
    def in_cooldown(self) -> bool:
        return self._in_cooldown(time.time())

    def reset(self) -> None:
        """Force-clear all state (useful for testing or scene changes)."""
        self._history.clear()
        self._candidate = None
        self._last_gesture_time = 0.0
        self.last_result = GestureResult()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_cooldown(self, now: float) -> bool:
        return (now - self._last_gesture_time) < self.cooldown

    def _reset_candidate(self) -> None:
        self._candidate = None

    def _evaluate(self, x: float, y: float, now: float) -> GestureResult:
        """Core swipe evaluation logic."""
        c = self._candidate
        elapsed = now - c.start_time

        delta_x = x - c.start_x
        delta_y = y - c.start_y
        abs_dx  = abs(delta_x)
        abs_dy  = abs(delta_y)

        # Update peak displacement for confidence scoring
        c.peak_disp = max(c.peak_disp, abs_dx)

        # ── Time-window expiry: abandon candidate ──────────────────────
        if elapsed > self.max_time:
            self._reset_candidate()
            return GestureResult()

        # ── Too early to decide – keep accumulating ────────────────────
        if elapsed < self.min_time:
            return GestureResult()

        # ── Reject if not enough horizontal movement ───────────────────
        if abs_dx < self.move_threshold:
            return GestureResult()

        # ── Reject if vertical component dominates ─────────────────────
        if abs_dx > 0 and (abs_dy / abs_dx) > self.vertical_ratio:
            self._reset_candidate()
            return GestureResult()

        # ── Directional consistency check ─────────────────────────────
        consistency = self._directional_consistency(delta_x)
        if consistency < 0.60:          # less than 60 % frames in direction
            self._reset_candidate()
            return GestureResult()

        # ── Compute confidence ─────────────────────────────────────────
        confidence = self._compute_confidence(
            abs_dx, abs_dy, elapsed, consistency
        )

        if confidence < self.min_confidence:
            self._reset_candidate()
            return GestureResult()

        # ── Valid gesture detected ─────────────────────────────────────
        gesture   = GESTURE_SWIPE_RIGHT if delta_x > 0 else GESTURE_SWIPE_LEFT
        direction = "right"             if delta_x > 0 else "left"

        self._last_gesture_time = now
        self._reset_candidate()

        return GestureResult(
            gesture=gesture,
            confidence=round(confidence, 1),
            direction=direction,
        )

    def _directional_consistency(self, total_delta_x: float) -> float:
        """
        Fraction of consecutive frame pairs in the history that moved in
        the same horizontal direction as the total displacement.
        Returns a value in [0, 1].
        """
        history = list(self._history)
        if len(history) < 2:
            return 1.0

        expected_sign = 1 if total_delta_x > 0 else -1
        consistent    = 0
        total_pairs   = 0

        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            if dx != 0:
                consistent  += 1 if (dx * expected_sign > 0) else 0
                total_pairs += 1

        if total_pairs == 0:
            return 1.0

        return consistent / total_pairs

    def _compute_confidence(
        self,
        abs_dx:      float,
        abs_dy:      float,
        elapsed:     float,
        consistency: float,
    ) -> float:
        """
        Blended confidence score in [0, 100].

        Components
        ----------
        displacement_score  : how far beyond the threshold the finger moved
        speed_score         : bonus for gestures in the sweet-spot time range
        horizontal_score    : penalty for vertical drift
        consistency_score   : direction consistency across history frames
        """
        # 1. Displacement: saturates at 3× the threshold
        disp_score = min(abs_dx / self.move_threshold, 3.0) / 3.0  # 0–1

        # 2. Speed: ideal window is [min_time, max_time * 0.7]
        ideal_max   = self.max_time * 0.70
        if elapsed <= ideal_max:
            speed_score = 1.0
        else:
            overshoot   = (elapsed - ideal_max) / (self.max_time - ideal_max)
            speed_score = max(0.0, 1.0 - overshoot)

        # 3. Horizontal dominance: abs_dx / (abs_dx + abs_dy)
        horiz_score = abs_dx / (abs_dx + abs_dy + 1e-6)

        # 4. Direction consistency (already in [0, 1])
        consistency_score = consistency

        # Weighted blend
        raw = (
            0.35 * disp_score
            + 0.20 * speed_score
            + 0.25 * horiz_score
            + 0.20 * consistency_score
        )

        return raw * 100.0

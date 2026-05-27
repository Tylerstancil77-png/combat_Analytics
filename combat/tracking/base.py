"""
Tracking data contract — the stable boundary between *how* points are produced
and *what* the analytics engine consumes.

Any tracking core (pose+filter, SAM2, EdgeTAM, a custom glove detector) produces
`FrameTracks` per frame; the analytics layer (strike detection, calibration,
event logging) consumes `FrameTracks` and never needs to know which engine made
them.  This is what lets us swap the tracking core without touching downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ── Per-point result ──────────────────────────────────────────────────────────
@dataclass
class PointTrack:
    """One tracked point (a glove or the head) on one frame."""
    pos:      Optional[np.ndarray] = None     # (x, y) in source-frame pixels, or None
    conf:     float                = 0.0      # 0–1 confidence (drives downstream trust)
    source:   str                  = "none"   # "pose" | "mask" | "kinematic" | "ghost" | ...
    on_glove: bool                 = False    # True when localized to the actual glove
                                              # (vs. wrist/estimate) — the constitution metric

    @property
    def visible(self) -> bool:
        return self.pos is not None


# ── Per-fighter result for one frame ────────────────────────────────────────────
@dataclass
class FighterFrame:
    fighter_id: int                              # 0 = Fighter_A, 1 = Fighter_B
    head:  PointTrack = field(default_factory=PointTrack)
    left:  PointTrack = field(default_factory=PointTrack)   # anatomical left glove
    right: PointTrack = field(default_factory=PointTrack)   # anatomical right glove
    bbox:  Optional[np.ndarray] = None           # xyxy, last known
    # Optional skeleton for analytics that need joint angles (elbow/shoulder/etc.)
    keypoints: Optional[np.ndarray] = None       # (17, 2) COCO, or None
    kp_conf:   Optional[np.ndarray] = None       # (17,)

    def glove(self, side: str) -> PointTrack:
        return self.left if side == "Left" else self.right


# ── Whole-frame result ──────────────────────────────────────────────────────────
@dataclass
class FrameTracks:
    frame_idx: int
    timestamp: float
    fighters:  Dict[int, FighterFrame] = field(default_factory=dict)
    occluded:  bool = False                       # heavy fighter–fighter overlap this frame


# ── Tracker interface ───────────────────────────────────────────────────────────
class Tracker(ABC):
    """
    A tracking core.  Implementations:
      - PoseTracker     (current YOLO-pose + Kalman/OEF + color refinement)
      - EdgeTAMTracker  (mask-tracking core — the rebuild target)
      - DetectorTracker (custom glove detector — the speed-first option)
    """

    @abstractmethod
    def process(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FrameTracks:
        """Consume one BGR frame; return this frame's tracks for both fighters."""
        raise NotImplementedError

    def reset(self) -> None:
        """Optional: clear state (e.g. on a round/partner change)."""
        pass

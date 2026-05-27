"""
combat_analytics_v1 — Strike Analyzer v5 (ByteTrack Fighter Lock)
==================================================================
Uses YOLO11-Pose + ByteTrack for robust two-person tracking.

Startup phase:
  Wait for the first frame that contains ≥2 ByteTrack detections.
  Lock the two track IDs with the highest confidence scores as
  Fighter_A (left) and Fighter_B (right).  Any track ID that appears
  after that point — referee, corner staff, etc. — is silently ignored.

Per-frame pipeline per locked fighter:
    ByteTrack bbox → YOLO pose keypoints
        └─ One Euro Filter     (kills landmark jitter)
            └─ Kalman Filter   (ghost prediction during occlusion ≤5 frames)
                └─ Depth-normalised deceleration engine
                    └─ Directional intent filter

Outputs
-------
  output_data/<stem>_strikes.csv       — per-strike event log
  output_data/<stem>_annotated.mp4     — annotated video

Usage
-----
  python scripts/strike_analyzer.py input_videos/sparring_clip.mp4
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "\n[strike_analyzer] Missing dependency.\n"
        "  Install with:  pip install ultralytics\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# COCO keypoint indices  (YOLO11-Pose output format)
# ══════════════════════════════════════════════════════════════════════════════
KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16

# Keypoints that must be present for a detection to qualify as a fighter
# (not a referee torso, crowd member, or partial detection)
SKELETON_REQUIRED = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                     KP_LEFT_HIP,      KP_RIGHT_HIP]
SKELETON_LOWER    = [KP_LEFT_KNEE, KP_RIGHT_KNEE,
                     KP_LEFT_ANKLE, KP_RIGHT_ANKLE]   # at least one needed

WRIST_MAP    = {"Left": KP_LEFT_WRIST,    "Right": KP_RIGHT_WRIST}
ELBOW_MAP    = {"Left": KP_LEFT_ELBOW,    "Right": KP_RIGHT_ELBOW}
SHOULDER_MAP = {"Left": KP_LEFT_SHOULDER, "Right": KP_RIGHT_SHOULDER}

SKELETON_EDGES = [
    (KP_NOSE,           KP_LEFT_SHOULDER),   # neck / head-to-torso
    (KP_NOSE,           KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW,     KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER,  KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP,       KP_RIGHT_HIP),
]


# ══════════════════════════════════════════════════════════════════════════════
# Tuning constants
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL        = "yolo11n-pose.pt"    # Phase-1 A/B verdict: a 4×-larger
                                            # model gave ~0 dot-on-glove gain
                                            # (69.4%→69.9%) for 4× runtime — the
                                            # bottleneck is structural, not the
                                            # detector. Keep nano as a fast seed
                                            # for the Phase-2 temporal tracker.
YOLO_CONF            = 0.40
YOLO_IOU             = 0.45
YOLO_IMGSZ           = 640    # inference resolution.  Tried 1280 — it INCREASED
                              # track churn (more bystanders detected → more
                              # identity competition) without reducing frame loss,
                              # so reverted.  Detection res was not the bottleneck.

KP_CONF_THRESHOLD       = 0.50   # skeleton keypoints (shoulders, hips, knees)
WRIST_KP_CONF_THRESHOLD = 0.30   # wrist keypoints only — YOLO scores boxing gloves
                                  # lower than bare hands; 0.30 is the sweet spot
                                  # between catching real gloves and rejecting
                                  # hallucinated positions at very low confidence.
HEAD_KP_CONF_THRESHOLD  = 0.30   # nose / eye / ear — face can be partially blocked
                                  # by a guard, so a lower threshold than skeleton.
MAX_GHOST_FRAMES     = 5

VELOCITY_BUFFER_SIZE = 5
DECEL_LOOKBACK       = 3
DECEL_THRESHOLD      = 0.80
EXTENSION_THRESHOLD  = 0.70
MIN_BASELINE_VEL     = 0.02

OEF_MIN_CUTOFF       = 1.5
OEF_BETA             = 0.005
OEF_D_CUTOFF         = 1.0

# Kalman filter — non-uniform process noise (position << velocity << acceleration)
KF_POS_NOISE  = 1e-3    # x, y  — position states (small; we observe these directly)
KF_VEL_NOISE  = 3e-2    # vx,vy — 3× original; adapts quickly to speed changes
KF_ACC_NOISE  = 1e-1    # ax,ay — high uncertainty; lets filter learn strike snap
KF_MEAS_NOISE = 1e-2    # R — decreased from 5e-2; trust raw observations more

# Adaptive Mahalanobis gating
KF_GATE_CHI2  = 5.99    # chi²(0.95, 2 DOF) — normal gate threshold (d²)
KF_GATE_PIXEL = 15.0    # px displacement above which gate radius expands 50 %

# OEF bypass during strike extension
STRIKE_VEL_BYPASS     = 0.10  # sw/frame — above this, raw YOLO coords used directly
OEF_BYPASS_MIN_CONF   = 0.45  # min wrist keypoint confidence to allow OEF bypass
                               # — jittery low-conf detections push velocity high,
                               #   which would bypass the smoother and amplify the
                               #   jitter.  Only high-conf detections skip the OEF.

# Adaptive Kalman filter — confidence-driven R scaling
# High YOLO confidence → small R  (snap to raw observation)
# Low YOLO confidence  → large R  (rely on KF prediction)
KF_R_MIN_FACTOR = 0.05   # R multiplier when wrist keypoint conf = 1.0
KF_R_MAX_FACTOR = 8.0    # R multiplier when wrist keypoint conf ≈ 0.0

# Adaptive Kalman filter — innovation-driven Q scaling
# Innovation window: 5 frames of residuals drive Q up/down
KF_INNOV_WINDOW = 5
KF_Q_SCALE_MIN  = 0.2    # floor — prevents Q collapsing to zero
KF_Q_SCALE_MAX  = 12.0   # ceiling — prevents runaway noise during occlusion bursts

# ── Color histogram guardrail (always-on) ────────────────────────────────────
COLOR_PROFILE_FRAMES  = 5     # frames used to build each fighter's master profile
COLOR_HIST_H_BINS     = 18    # hue bins   (0–180)
COLOR_HIST_S_BINS     = 8     # sat bins   (0–256)  — V ignored for lighting invariance
COLOR_TRUNK_START     = 0.40  # top of trunk ROI as fraction of bbox height
# Hysteresis: swapped diagonal must beat current diagonal by this margin.
# Prevents noise-level score jitter from triggering a spurious swap on clean frames.
GUARDRAIL_SWAP_MARGIN = 0.05
# On clean frames the profile slowly blends toward the fighter's current
# appearance so it stays accurate over a long fight (sweat, lighting drift).
COLOR_ADAPT_RATE      = 0.02  # EMA weight (0.98 old + 0.02 new)

# ── Glove HSV fingerprint (display only) ──────────────────────────────────────
# The wrist-ROI colour histogram is no longer used to gate tracking decisions
# (anatomical constraints own that now).  It is kept purely to show a live
# colour-match score in the overlay as a sanity indicator.
GLOVE_ROI_RADIUS      = 28    # px half-size of square ROI around wrist keypoint
GLOVE_MATCH_THRESHOLD = 0.45  # display threshold — score below this draws red
GLOVE_ADAPT_RATE      = 0.05  # EMA weight: fingerprint = 0.95·old + 0.05·new

# ── Occlusion Freeze & Separation Protocol ───────────────────────────────────
OCCLUSION_IOU_ENTER     = 0.40  # IoU ≥ this → enter occlusion state (tracks frozen)
OCCLUSION_IOU_EXIT      = 0.15  # IoU < this → exit occlusion, run spatial re-assign
PRECLINCH_ANCHOR_FRAMES = 10    # frames of history to look back for spatial anchors

# ── Sub-Frame Temporal Interpolation (YOLO-displacement driven) ───────────────
# Velocity is derived directly from (raw - state.wrist_pos) — the same quantity
# Farneback was approximating from pixels, available in O(1) with no image work.
DISP_SPIKE_THRESHOLD   = 8.0  # px — displacement above which single injection fires
                               # (raised from 2.5 — sub-threshold jitter of 3–7 px
                               #  was triggering velocity injection every frame)
SUBFRAME_TRIGGER       = 20   # px — displacement above which sub-frame steps fire
SUBFRAME_STEPS         = 4    # virtual KF predict steps per real video frame

# ── Re-identification thresholds ──────────────────────────────────────────────
REID_MIN_SIM        = 0.55   # min colour correlation to accept a Re-ID candidate
                             # (raised from 0.35 — rejects corner staff in similar colours)
REID_MAX_DISTANCE   = 450    # px — candidate bbox centre must be within this of last known
                             # position.  250 was too tight on 2880-wide footage: a fighter
                             # who moved during the gap fell out of range and stayed "lost".
REID_MIN_TRACK_AGE  = 1      # consecutive frames a track must exist before Re-ID will accept it.
                             # Was 3 (to dodge a coach mis-ID) but that GUARANTEED ≥3 lost
                             # frames per ID churn — the dominant cause of frame loss.  The
                             # colour-similarity gate (REID_MIN_SIM) is the real non-fighter
                             # guard, so age can drop to 1 and re-attach immediately.
REID_COOLDOWN_FRAMES = 8     # frames the guardrail stays silent after any Re-ID fires
                             # — prevents a correct Re-ID from being immediately undone
GUARDRAIL_MIN_MATCH = 0.30   # both bbox→profile scores must exceed this before a swap is allowed
                             # — a non-fighter scoring 0.08 against either profile can't drive a swap
BOTH_LOST_RESET     = MAX_GHOST_FRAMES * 2   # frames before both-lost triggers a full re-lock

# ── Per-frame identity assignment ─────────────────────────────────────────────
# Once colour profiles exist, identity is decided EVERY frame by assigning the
# two best-matching trackable bodies to Fighter A / B — there is no "reject a
# present fighter" gate (that gate was the dominant cause of frame loss).  Each
# candidate is scored by colour similarity to the profile plus proximity to the
# fighter's last position; a 2-way assignment maximises total score.
ASSIGN_W_COLOR      = 1.0    # weight on colour-profile correlation  (range −1..1)
ASSIGN_W_PROX       = 0.5    # weight on proximity-to-last-position   (range  0..1)
ASSIGN_PROX_NORM_SW = 6.0    # proximity decays to 0 at this many shoulder-widths
ASSIGN_FLOOR        = 0.15   # min score to claim a candidate (else fighter = lost).
                             # Low on purpose: we pick the BEST body, only refusing
                             # an obvious non-fighter when the real one is absent.
ASSIGN_ADAPT_MIN_SIM = 0.35  # only blend a profile toward an assignment this good

# ── Strike debounce ───────────────────────────────────────────────────────────
# The deceleration trigger fires on every frame of a punch's follow-through,
# not just the impact frame — producing 3–5 consecutive events per punch.
# Require at least this many frames between recorded strikes on the same hand
# to collapse each physical punch into a single CSV row.
# 10 frames ≈ 168 ms at 59.5 fps — wide enough to cover any realistic combo
# gap; a genuine 1-2 punch has ~200-400 ms between impacts.
STRIKE_DEBOUNCE_FRAMES = 10

# ── Biomechanical velocity calibration ───────────────────────────────────────
# Converts shoulder-width-normalised velocity (sw/frame) to real-world m/s.
# Formula: vel_ms = vel_sw × SHOULDER_WIDTH_M × fps
# Default shoulder width (average adult male boxer, taped across back): 0.46 m.
# Override this if you have measured shoulder width for your specific athletes.
SHOULDER_WIDTH_M = 0.46

# ══════════════════════════════════════════════════════════════════════════════
# Anatomical (kinematic-chain) wrist & head constraints
# ══════════════════════════════════════════════════════════════════════════════
# The shoulder and elbow keypoints are far more stable than the wrist (larger,
# slower, rarely fully occluded).  We exploit the arm as a kinematic chain:
# the wrist is physically tethered to the elbow by the forearm.  These
# constraints REPLACE the old pixel-threshold cascade (jump gate, cross-wrist
# check, kinematic cone, HSV gate, prox fallback, skeleton anchor) with body
# proportions the engine measures from the footage itself — so they scale
# automatically across camera distances (phone in a gym, broadcast, etc.).

# Running-average weight for each fighter's measured forearm length.
# Low weight → stable estimate that ignores per-frame keypoint noise.
FOREARM_EMA            = 0.10
# A wrist detection may sit at most this multiple of the measured forearm
# length from its own elbow.  >1.0 because the glove face extends past the
# anatomical wrist and keypoints carry noise.  1.45 ≈ glove + slight overreach.
FOREARM_MAX_FACTOR     = 1.45
# Default forearm length as a fraction of shoulder-width, used until enough
# clean frames have been seen to measure the real value.
FOREARM_DEFAULT_SW     = 0.95
# Elbow-ownership test (anti-glue / anti-swap): a wrist must be at least this
# much closer to its OWN elbow than to the OTHER elbow.  1.0 = strictly closer;
# 1.10 gives 10 % slack so a tight hook crossing the midline isn't rejected.
ELBOW_OWNERSHIP_MARGIN = 1.10
# Confidence assigned to an elbow-projected (kinematic fallback) wrist estimate.
# Low → the KF treats it as a soft hint (large R) rather than a crisp pixel.
KINEMATIC_FALLBACK_CONF = 0.25
# Confidence assigned to a directly accepted YOLO wrist when no elbow is visible
# to validate it against (reduced — we can't anatomically confirm it).
NO_ELBOW_CONF_SCALE    = 0.60

# ── Head neck-tether constraint ───────────────────────────────────────────────
# The head sits within a bounded distance of the shoulder midpoint (the neck).
# A face keypoint farther than this is a mis-detection (crowd, opponent's head).
HEAD_MAX_NECK_SW       = 1.40   # max head-to-shoulder-midpoint distance (× sw)
HEAD_FALLBACK_CONF     = 0.25   # confidence for a shoulder-projected head estimate
HEAD_VALID_CONF        = 0.70   # confidence for an accepted face keypoint

# ── Shoulder-width normalisation stability ────────────────────────────────────
# Velocity is normalised by shoulder-width (sw).  When a fighter turns sideways
# the shoulders foreshorten and the raw sw collapses toward its 1 px floor —
# dividing velocity by ~0 and producing absurd readings.  We track a slow EMA
# of sw and clamp the instantaneous value to a band around it so a transient
# foreshortening can't blow up the normaliser.
CACHED_SW_EMA          = 0.05   # running-average weight for shoulder-width
CACHED_SW_MIN_RATIO    = 0.60   # instantaneous sw floored at this × the EMA
CACHED_SW_MAX_RATIO    = 1.80   # instantaneous sw capped at this × the EMA

# ── Temporal wrist-step constraint ────────────────────────────────────────────
# A LAST-RESORT teleport guard, NOT a speed limiter.  The forearm-reach and
# elbow-ownership constraints already bound a real wrist anatomically (a fast
# punch stays within forearm reach of the elbow), so this only needs to catch
# gross glitches — mainly in the no-elbow branch where the anatomical bound
# can't apply.  It was previously set far too tight (0.60 sw ≈ 16 m/s), which
# rejected real fast punches AND noisy detections during exactly the moments
# that matter, dumping the dot onto an off-glove fallback ~⅓ of all frames.
# 1.50 sw/frame ≈ 40 m/s — above any human strike; only egregious teleports fail.
MAX_WRIST_STEP_SW      = 1.50
WRIST_STEP_MAX_FRAMES  = 5      # cap the elapsed-frame scaling so a long gap
                                # doesn't fully disable the step constraint.

# ── Physical strike-velocity ceiling ──────────────────────────────────────────
# A real fist tops out around 11 m/s; at 59.5 fps with a 0.46 m shoulder that is
# ~0.40 sw/frame.  Any "strike" whose baseline velocity exceeds this ceiling is
# a tracking glitch, not a punch — drop it.  0.70 sw/f ≈ 19 m/s leaves margin.
MAX_STRIKE_VEL_SW      = 0.70

# ── Color glove refinement ────────────────────────────────────────────────────
# Pose keypoints sit at the anatomical WRIST, behind the glove, and YOLO drops
# them in exactly the fast/occluded moments that matter.  Boxing gloves, by
# contrast, are large vividly-coloured objects.  We learn each fighter's glove
# colour from confident early detections, then every frame use the pose estimate
# as a SEED and snap the dot to the actual glove blob (HSV back-projection) in a
# small window around the seed.  This lands the dot on the glove face, fixes the
# wrist offset, and corrects fallback drift.  If no plausible blob is found it
# keeps the pose estimate, so refinement can only help.
GLOVE_MODEL_FRAMES        = 12     # confident detections used to build the model
GLOVE_MODEL_SAMPLE_SW     = 0.20   # tight ROI radius for colour sampling (× sw)
GLOVE_REFINE_RADIUS_SW    = 0.45   # search-window radius around the pose seed (× sw)
GLOVE_BACKPROJ_THRESH     = 40     # 0–255 back-projection probability threshold.
                                   # Lowered from 60: Fighter B's (bare-torso)
                                   # glove model back-projects weakly, so 60 left
                                   # no contour ~700×/hand. 40 recovers those weak
                                   # but real glove pixels; skin doesn't match a
                                   # glove-coloured model so it stays dark anyway.
GLOVE_MIN_BLOB_SW         = 0.07   # min blob size (× sw); area gate = (this·sw)².
                                   # Lowered from 0.10 — weak back-projections form
                                   # smaller-but-valid glove blobs.
GLOVE_MODEL_MIN_SAT       = 40     # mean saturation below which colour is too
                                   # unreliable (white/grey gloves) — refine off
GLOVE_REFINE_MAX_SHIFT_SW = 0.45   # refined point may not move more than this (× sw)


# ══════════════════════════════════════════════════════════════════════════════
# System A — One Euro Filter
# ══════════════════════════════════════════════════════════════════════════════

class _LowPassFilter1D:
    def __init__(self):
        self._y: Optional[float] = None

    def filter(self, x: float, alpha: float) -> float:
        self._y = x if self._y is None else alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def last(self) -> Optional[float]:
        return self._y


class OneEuroFilter1D:
    """Speed-adaptive low-pass filter (Casiez et al., CHI 2012)."""

    def __init__(self, min_cutoff=OEF_MIN_CUTOFF, beta=OEF_BETA, d_cutoff=OEF_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_filt    = _LowPassFilter1D()
        self._dx_filt   = _LowPassFilter1D()
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            return self._x_filt.filter(x, 1.0)
        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t
        x_prev = self._x_filt.last if self._x_filt.last is not None else x
        dx_hat = self._dx_filt.filter((x - x_prev) / dt, self._alpha(self.d_cutoff, dt))
        return self._x_filt.filter(x, self._alpha(self.min_cutoff + self.beta * abs(dx_hat), dt))


class OneEuroFilter2D:
    def __init__(self):
        self._fx = OneEuroFilter1D()
        self._fy = OneEuroFilter1D()

    def filter(self, xy: np.ndarray, t: float) -> np.ndarray:
        return np.array([self._fx.filter(float(xy[0]), t),
                         self._fy.filter(float(xy[1]), t)])


# ══════════════════════════════════════════════════════════════════════════════
# System B — Kalman Filter Ghost Predictor
# ══════════════════════════════════════════════════════════════════════════════

class WristKalmanFilter:
    """
    Adaptive Kalman filter (AKF) for one wrist.
    State: [x, y, vx, vy, ax, ay].  Measurement: [x, y].

    Two live adaptation loops run every frame:

    1. Innovation-based Q adaptation (process noise)
       ───────────────────────────────────────────────
       A 5-frame rolling window tracks the squared magnitude of the
       innovation residual (||z - H·x_pred||²).  The sample mean is
       compared to trace(S) — the theoretically expected innovation
       variance.  The ratio drives a per-frame Q scale factor:

           q_scale = clip( E[||v||²] / trace(S),  Q_MIN, Q_MAX )
           Q_live  = Q_base × q_scale

       When the filter is surprised by rapid wrist movement (large
       residuals), Q grows and the filter becomes more agile.  During
       quiet guard phases the residuals shrink, Q contracts, and the
       output smooths out.

    2. Confidence-based R adaptation (measurement noise)
       ───────────────────────────────────────────────────
       YOLO outputs a per-keypoint confidence score (0–1).  R is scaled
       by the inverse of that confidence so that crisp detections during
       explosive strikes are trusted maximally, while uncertain keypoints
       during clinches or occlusions are down-weighted:

           R_factor = R_MIN + (R_MAX - R_MIN) × (1 – conf)
           R_live   = R_base × R_factor

       conf=1.0 → R_factor=R_MIN (0.05)  → filter snaps to raw pixel
       conf=0.5 → R_factor≈2.6           → balanced blend
       conf→0.0 → R_factor=R_MAX (8.0)   → rely on prediction

    Both adaptations combine with the existing adaptive Mahalanobis gate
    (expands 50 % radius for displacements > KF_GATE_PIXEL px).
    """

    # ── Static matrices (shared across instances) ─────────────────────────────
    _F = np.array([          # Transition — constant-acceleration, dt=1
        [1, 0, 1, 0, 0.5, 0.0],
        [0, 1, 0, 1, 0.0, 0.5],
        [0, 0, 1, 0, 1.0, 0.0],
        [0, 0, 0, 1, 0.0, 1.0],
        [0, 0, 0, 0, 1.0, 0.0],
        [0, 0, 0, 0, 0.0, 1.0],
    ], dtype=np.float32)

    _H = np.array([          # Measurement — observe x, y only
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float32)

    _Q_base = np.diag([      # Base process noise — position << velocity << accel
        KF_POS_NOISE, KF_POS_NOISE,
        KF_VEL_NOISE, KF_VEL_NOISE,
        KF_ACC_NOISE, KF_ACC_NOISE,
    ]).astype(np.float32)

    def __init__(self):
        self._kf = cv2.KalmanFilter(6, 2)
        self._kf.transitionMatrix    = self._F.copy()
        self._kf.measurementMatrix   = self._H.copy()
        self._kf.processNoiseCov     = self._Q_base.copy()
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEAS_NOISE
        self._kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self._init         = False
        self._ghost_frames = 0

        # Innovation history for Q adaptation
        self._innov_sq: deque = deque(maxlen=KF_INNOV_WINDOW)

    @property
    def ghost_frames(self) -> int:
        return self._ghost_frames

    @property
    def predicted_xy(self) -> Optional[np.ndarray]:
        """
        One-step-ahead predicted (x, y) WITHOUT mutating filter state.

        Used by the anatomical wrist estimator as the temporal prior when YOLO
        drops the wrist: the predicted position is projected onto the forearm
        circle around the (reliable) elbow to produce a fallback measurement.
        """
        if not self._init:
            return None
        pred = self._F @ self._kf.statePost           # (6,1) constant-accel, dt=1
        return pred[:2, 0].astype(np.float64).copy()

    @property
    def current_xy(self) -> Optional[np.ndarray]:
        """Latest corrected (x, y) estimate, or None before initialisation."""
        if not self._init:
            return None
        return self._kf.statePost[:2, 0].astype(np.float64).copy()

    def update(self,
               xy:      Optional[np.ndarray],
               kp_conf: float = 1.0,
               ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Args:
            xy       : Filtered wrist pixel coordinate, or None if occluded.
            kp_conf  : YOLO keypoint confidence for this wrist (0–1).
                       Used to scale R — high confidence → trust raw pixel.
        """
        if xy is not None:
            # ── Seed on first detection ───────────────────────────────────────
            if not self._init:
                self._kf.statePost = np.array(
                    [[xy[0]], [xy[1]], [0.], [0.], [0.], [0.]], dtype=np.float32)
                self._kf.errorCovPost = np.eye(6, dtype=np.float32)
                self._init, self._ghost_frames = True, 0
                return xy.copy(), False

            # ── Adaptation 2: confidence → R ─────────────────────────────────
            conf_clamped = float(np.clip(kp_conf, 0.0, 1.0))
            r_factor = KF_R_MIN_FACTOR + (KF_R_MAX_FACTOR - KF_R_MIN_FACTOR) * (1.0 - conf_clamped)
            self._kf.measurementNoiseCov = (
                np.eye(2, dtype=np.float32) * KF_MEAS_NOISE * r_factor
            )

            # ── Predict ───────────────────────────────────────────────────────
            pred    = self._kf.predict()
            pred_xy = pred[:2].flatten().astype(np.float64)
            meas    = xy.astype(np.float64)
            innov   = (meas - pred_xy).reshape(2, 1)

            # ── Innovation covariance S ───────────────────────────────────────
            H = self._kf.measurementMatrix
            R = self._kf.measurementNoiseCov
            S = H @ self._kf.errorCovPre @ H.T + R

            # ── Adaptation 1: rolling innovation window → Q ───────────────────
            innov_sq = float((innov.T @ innov).item())
            self._innov_sq.append(innov_sq)
            if len(self._innov_sq) >= 2:
                s_trace  = max(float(np.trace(S)), 1e-6)
                q_scale  = float(np.clip(
                    np.mean(self._innov_sq) / s_trace,
                    KF_Q_SCALE_MIN, KF_Q_SCALE_MAX,
                ))
                self._kf.processNoiseCov = (self._Q_base * q_scale).astype(np.float32)

            # ── Adaptive Mahalanobis gate ─────────────────────────────────────
            displacement = float(np.linalg.norm(meas - pred_xy))
            gate_sq      = KF_GATE_CHI2 * (2.25 if displacement > KF_GATE_PIXEL else 1.0)
            try:
                d_sq = float((innov.T @ np.linalg.inv(S) @ innov).item())
            except np.linalg.LinAlgError:
                d_sq = 0.0

            if d_sq > gate_sq:
                # Innovation too large even after gate expansion — hold prediction
                return pred_xy, False

            # ── Correct ───────────────────────────────────────────────────────
            corr = self._kf.correct(xy.astype(np.float32).reshape(2, 1))
            self._ghost_frames = 0
            return corr[:2].flatten().astype(np.float64), False

        else:
            # ── No observation: ghost-predict ─────────────────────────────────
            if not self._init:
                return None, True
            self._ghost_frames += 1
            if self._ghost_frames > MAX_GHOST_FRAMES:
                self._init, self._ghost_frames = False, 0
                return None, True
            return self._kf.predict()[:2].flatten().astype(np.float64), True


# ══════════════════════════════════════════════════════════════════════════════
# HandState  —  all tracking data for one wrist
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HandState:
    oef:         OneEuroFilter2D       = field(default_factory=OneEuroFilter2D)
    kf:          WristKalmanFilter     = field(default_factory=WristKalmanFilter)
    vel_buf:     deque                 = field(default_factory=lambda: deque(maxlen=VELOCITY_BUFFER_SIZE))
    vec_buf:     deque                 = field(default_factory=lambda: deque(maxlen=VELOCITY_BUFFER_SIZE))
    prev_pos:    Optional[np.ndarray]  = None
    wrist_pos:   Optional[np.ndarray]  = None
    is_ghost:    bool                  = False
    # Glove HSV fingerprint — display only (live colour-match score in overlay)
    glove_hist:  Optional[np.ndarray]  = None   # 2-D HS fingerprint set at identity lock
    glove_score: float                 = 0.0    # latest per-frame correlation (0–1)
    # Temporal wrist-step constraint: last accepted REAL (YOLO) wrist position
    # and how many frames have elapsed since — used to bound per-frame motion.
    last_yolo_pos:     Optional[np.ndarray] = None
    frames_since_yolo: int                  = 0
    # Debounce: frame index of the most recently recorded strike on this hand.
    # Initialised far in the past so the first real strike is never blocked.
    last_strike_frame: int             = -9999


# ══════════════════════════════════════════════════════════════════════════════
# HeadState  —  tracking data for one fighter's head position
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeadState:
    oef:      OneEuroFilter2D      = field(default_factory=OneEuroFilter2D)
    kf:       WristKalmanFilter    = field(default_factory=WristKalmanFilter)
    head_pos: Optional[np.ndarray] = None
    is_ghost: bool                 = False


# ══════════════════════════════════════════════════════════════════════════════
# Fighter  —  one combatant's complete state
# ══════════════════════════════════════════════════════════════════════════════

class Fighter:
    PALETTES: Dict[str, Dict] = {
        "Fighter_A": {
            "detected": (255, 215,   0),
            "ghost":    (  0, 255, 255),
            "skeleton": (200, 190,   0),
            "bbox":     (255, 215,   0),
            "strike":   ( 80, 255, 120),
        },
        "Fighter_B": {
            "detected": (  0, 140, 255),
            "ghost":    (  0, 255, 255),
            "skeleton": (  0, 110, 200),
            "bbox":     (  0, 140, 255),
            "strike":   ( 80,  80, 255),
        },
    }

    def __init__(self, name: str):
        if name not in self.PALETTES:
            raise ValueError(f"name must be one of {list(self.PALETTES)}, got '{name}'")

        self.name  = name
        self.hands: Dict[str, HandState] = {
            "Left":  HandState(),
            "Right": HandState(),
        }
        self.head = HeadState()
        self.strike_counts:    Dict[str, int] = {"Left": 0, "Right": 0}
        self.velocity_log:     List[float]   = []   # baseline velocity in sw/frame
        self.velocity_log_ms:  List[float]   = []   # baseline velocity in m/s

        # ── Color glove refinement ────────────────────────────────────────────
        # HS histogram (scaled 0–255 for cv2.calcBackProject) of this fighter's
        # gloves, learned from the first confident detections.  Shared by both
        # hands (a fighter's gloves are the same colour).
        self.glove_color_model: Optional[np.ndarray] = None
        self.glove_model_accum: List[np.ndarray]     = []   # samples while building
        self.glove_model_sat:   float                = 0.0  # mean saturation of model

        # ── Anatomical-chain caches ───────────────────────────────────────────
        # Forearm length (elbow→wrist) measured live per side; drives the
        # forearm-reach constraint.  None until the first clean measurement.
        self.cached_forearm:    Dict[str, Optional[float]] = {"Left": None, "Right": None}
        # Last accepted head→shoulder-midpoint offset, used to project the head
        # from the shoulders when face keypoints disappear.
        self.cached_head_offset: Optional[np.ndarray] = None

        # ── Tracking diagnostics ──────────────────────────────────────────────
        # Per-hand counters explaining wrist-acceptance behaviour over a session.
        # Printed at session end to surface left/right asymmetries.
        self.reject_forearm:     Dict[str, int] = {"Left": 0, "Right": 0}  # too far from own elbow
        self.reject_ownership:   Dict[str, int] = {"Left": 0, "Right": 0}  # closer to other elbow (glue/swap)
        self.reject_step:        Dict[str, int] = {"Left": 0, "Right": 0}  # teleported from last real detection
        self.fallback_kinematic: Dict[str, int] = {"Left": 0, "Right": 0}  # elbow-projected (YOLO dropped wrist)
        self.glove_refined:      Dict[str, int] = {"Left": 0, "Right": 0}  # snapped to glove blob via colour
        self.reject_decel:       Dict[str, int] = {"Left": 0, "Right": 0}
        self.reject_extension:   Dict[str, int] = {"Left": 0, "Right": 0}
        self.reject_direction:   Dict[str, int] = {"Left": 0, "Right": 0}

        # ── Dot-source breakdown (the REAL "is the dot on the glove" metric) ──
        # Classifies the dot DRAWN each frame the hand is being processed:
        #   on_glove : real YOLO detection snapped to an actual glove colour blob
        #   at_wrist : real YOLO detection, NOT snapped (sits at the bare wrist)
        #   fallback : no usable detection — elbow-projected estimate (off-glove)
        #   ghost    : Kalman prediction with no measurement at all
        # "% welded to glove" = on_glove / (on_glove+at_wrist+fallback+ghost).
        self.dot_on_glove: Dict[str, int] = {"Left": 0, "Right": 0}
        self.dot_at_wrist: Dict[str, int] = {"Left": 0, "Right": 0}
        self.dot_fallback: Dict[str, int] = {"Left": 0, "Right": 0}
        self.dot_ghost:    Dict[str, int] = {"Left": 0, "Right": 0}
        # Flyaway split into two unambiguous buckets:
        #   offframe : drawn dot is literally outside the video frame (the
        #              "flying off SCREEN" symptom)
        #   farbody  : on-screen but >3.5 shoulder-widths from the torso centre
        #              (off the body but still visible)
        self.dot_offframe: Dict[str, int] = {"Left": 0, "Right": 0}
        self.dot_farbody:  Dict[str, int] = {"Left": 0, "Right": 0}
        # Why glove refinement failed when it did (diagnostic for at-wrist dots).
        self.refine_reason: Dict[str, Dict[str, int]] = {"Left": {}, "Right": {}}

        self.cached_torso: Optional[np.ndarray] = None
        self.cached_sw:    float                = 150.0
        self.cached_sw_ema: Optional[float]     = None   # slow EMA used to clamp sw
        self.cached_bbox:       Optional[np.ndarray] = None  # last known xyxy bbox

    @property
    def total_strikes(self) -> int:
        return sum(self.strike_counts.values())

    @property
    def colors(self) -> dict:
        return self.PALETTES[self.name]

    def record_strike(self, event: dict) -> None:
        self.strike_counts[event["hand"]] += 1
        self.velocity_log.append(event["baseline_velocity_sw"])
        self.velocity_log_ms.append(event["baseline_velocity_ms"])

    def summary(self) -> str:
        avg_sw = (f"{sum(self.velocity_log)/len(self.velocity_log):.4f} sw/f"
                  if self.velocity_log else "n/a")
        avg_ms = (f"{sum(self.velocity_log_ms)/len(self.velocity_log_ms):.2f} m/s"
                  if self.velocity_log_ms else "n/a")
        lines = [
            f"  {self.name}  total={self.total_strikes}"
            f"  L={self.strike_counts['Left']}  R={self.strike_counts['Right']}"
            f"  avg_approach_vel={avg_sw}  ({avg_ms})",
            f"    Wrist estimator (L / R):"
            f"  reject_forearm={self.reject_forearm['Left']}/{self.reject_forearm['Right']}"
            f"  reject_ownership={self.reject_ownership['Left']}/{self.reject_ownership['Right']}"
            f"  reject_step={self.reject_step['Left']}/{self.reject_step['Right']}"
            f"  kinematic_fallback={self.fallback_kinematic['Left']}/{self.fallback_kinematic['Right']}"
            f"  glove_refined={self.glove_refined['Left']}/{self.glove_refined['Right']}",
            f"    Strike-gate rejects (L / R):"
            f"  decel={self.reject_decel['Left']}/{self.reject_decel['Right']}"
            f"  ext={self.reject_extension['Left']}/{self.reject_extension['Right']}"
            f"  dir={self.reject_direction['Left']}/{self.reject_direction['Right']}",
        ]
        # Dot-source breakdown — the metric that actually reflects "is the dot
        # on the glove."  Reported per hand as percentages of drawn frames.
        for side in ("Left", "Right"):
            g = self.dot_on_glove[side]; w = self.dot_at_wrist[side]
            f = self.dot_fallback[side]; gh = self.dot_ghost[side]
            tot = max(1, g + w + f + gh)
            off = self.dot_offframe[side]; far = self.dot_farbody[side]
            lines.append(
                f"    {side:5s} dot:  ON-GLOVE {100*g/tot:4.1f}%  "
                f"at-wrist {100*w/tot:4.1f}%  fallback {100*f/tot:4.1f}%  "
                f"ghost {100*gh/tot:4.1f}%   (n={g+w+f+gh})   "
                f"OFF-SCREEN={off} ({100*off/tot:.1f}%)  far-body={far} ({100*far/tot:.1f}%)"
            )
        for side in ("Left", "Right"):
            rr = self.refine_reason[side]
            if rr:
                parts = "  ".join(f"{k}={v}" for k, v in sorted(rr.items()))
                lines.append(f"    {side:5s} refine: {parts}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _kp_xy(kps: np.ndarray, idx: int) -> np.ndarray:
    return kps[idx].astype(np.float64)


def _kp_vis(conf: np.ndarray, idx: int) -> bool:
    return float(conf[idx]) >= KP_CONF_THRESHOLD


def _head_position(kps: np.ndarray, conf: np.ndarray) -> Optional[np.ndarray]:
    """
    Best-available head centre using HEAD_KP_CONF_THRESHOLD.

    Priority:
      1. Nose              — most central face point
      2. Midpoint of eyes  — stable when nose is occluded by a guard
      3. Either single eye — partial face still useful
      4. Midpoint of ears  — fallback when front of face is hidden
      5. Either single ear — last resort
    Returns None if no face keypoint clears the threshold.
    """
    def vis(idx: int) -> bool:
        return float(conf[idx]) >= HEAD_KP_CONF_THRESHOLD

    if vis(KP_NOSE):
        return _kp_xy(kps, KP_NOSE)

    le, re = vis(KP_LEFT_EYE), vis(KP_RIGHT_EYE)
    if le and re:
        return (_kp_xy(kps, KP_LEFT_EYE) + _kp_xy(kps, KP_RIGHT_EYE)) / 2.0
    if le:
        return _kp_xy(kps, KP_LEFT_EYE)
    if re:
        return _kp_xy(kps, KP_RIGHT_EYE)

    la, ra = vis(KP_LEFT_EAR), vis(KP_RIGHT_EAR)
    if la and ra:
        return (_kp_xy(kps, KP_LEFT_EAR) + _kp_xy(kps, KP_RIGHT_EAR)) / 2.0
    if la:
        return _kp_xy(kps, KP_LEFT_EAR)
    if ra:
        return _kp_xy(kps, KP_RIGHT_EAR)

    return None


def _shoulder_distance(kps: np.ndarray, conf: np.ndarray) -> Optional[float]:
    if not (_kp_vis(conf, KP_LEFT_SHOULDER) and _kp_vis(conf, KP_RIGHT_SHOULDER)):
        return None
    return max(
        float(np.linalg.norm(_kp_xy(kps, KP_RIGHT_SHOULDER) - _kp_xy(kps, KP_LEFT_SHOULDER))),
        1.0,
    )


def _torso_centre(kps: np.ndarray, conf: np.ndarray) -> Optional[np.ndarray]:
    if not (_kp_vis(conf, KP_LEFT_SHOULDER) and _kp_vis(conf, KP_RIGHT_SHOULDER)):
        return None
    return (_kp_xy(kps, KP_LEFT_SHOULDER) + _kp_xy(kps, KP_RIGHT_SHOULDER)) / 2.0


def _arm_extension(kps: np.ndarray, conf: np.ndarray, side: str) -> float:
    s_i  = SHOULDER_MAP[side]
    e_i  = ELBOW_MAP[side]
    wr_i = WRIST_MAP[side]
    if not all(_kp_vis(conf, i) for i in (s_i, e_i, wr_i)):
        return 0.0
    s, e, wr  = _kp_xy(kps, s_i), _kp_xy(kps, e_i), _kp_xy(kps, wr_i)
    direct    = np.linalg.norm(wr - s)
    segmented = np.linalg.norm(e - s) + np.linalg.norm(wr - e)
    return direct / segmented if segmented > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Anatomical wrist & head estimators  (kinematic-chain tracking)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WristMeas:
    """Result of the anatomical wrist estimator for one hand on one frame."""
    pos:     Optional[np.ndarray] = None  # measurement to feed the KF (None → ghost)
    conf:    float                = 0.0   # confidence (0–1) → drives KF R
    source:  str                  = "none" # "yolo" | "kinematic" | "none"
    refined: bool                 = False  # True → snapped to an actual glove blob


def _clamp_prediction(
    p:      Optional[np.ndarray],
    bbox:   Optional[np.ndarray],
    w:      int,
    h:      int,
    margin: float = 0.6,
) -> Optional[np.ndarray]:
    """
    Cage a PREDICTED dot (Kalman ghost / kinematic fallback) to a plausible
    region so it can never fly off-screen or far off the body.  Real YOLO
    detections are NOT clamped — only predictions, which is where the flying
    came from (unbounded velocity extrapolation during track loss).

    The region is the fighter's last bounding box expanded by `margin` of its
    size (generous, so a legitimately extended punch isn't clipped), then
    intersected with the frame bounds.  Returns the clamped point.
    """
    if p is None:
        return None
    x, y = float(p[0]), float(p[1])
    if bbox is not None:
        bx1, by1, bx2, by2 = (float(bbox[0]), float(bbox[1]),
                               float(bbox[2]), float(bbox[3]))
        mw, mh = (bx2 - bx1) * margin, (by2 - by1) * margin
        x = min(max(x, bx1 - mw), bx2 + mw)
        y = min(max(y, by1 - mh), by2 + mh)
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    return np.array([x, y])


def _wrist_step_ok(state: "HandState", cand: np.ndarray, sw: float) -> bool:
    """
    Temporal sanity: True if `cand` is a plausible single-step move from the
    last REAL (YOLO) detection.  The allowance scales with frames elapsed since
    that detection so the fist can legitimately "catch up" after dropped frames.
    Always True before the first real detection.
    """
    if state.last_yolo_pos is None:
        return True
    allowed = (MAX_WRIST_STEP_SW * sw
               * float(min(state.frames_since_yolo, WRIST_STEP_MAX_FRAMES)))
    return float(np.linalg.norm(cand - state.last_yolo_pos)) <= allowed


def _estimate_wrist_measurements(
    kps:     np.ndarray,
    conf:    np.ndarray,
    fighter: "Fighter",
    frame:   Optional[np.ndarray] = None,
) -> Dict[str, WristMeas]:
    """
    Jointly estimate both wrist measurements using the arm kinematic chain.

    The shoulder/elbow keypoints are far more reliable than the wrist, and the
    wrist is physically tethered to the elbow by the forearm.  Three rules
    (per side) replace the entire old pixel-threshold cascade:

      1. Forearm-length constraint — the wrist keypoint must lie within
         FOREARM_MAX_FACTOR × the fighter's MEASURED forearm length of its own
         elbow.  Kills hallucinated keypoints flung across the frame.
      2. Elbow-ownership test — the wrist must be closer to its OWN elbow than
         to the OTHER elbow (× ELBOW_OWNERSHIP_MARGIN).  Definitively resolves
         the "both keypoints glued to one glove" and left/right-swap failures.
      3. Kinematic fallback — if YOLO drops the wrist but the elbow is visible,
         project the KF's predicted position onto the forearm circle around the
         elbow and return it as a low-confidence measurement, so the hand stays
         anatomically anchored instead of dead-reckoning into open space.

    Forearm length is measured live from clean frames via an EMA, so the
    constraint self-calibrates to camera distance — no per-video pixel tuning.
    """
    sides = ("Left", "Right")
    sw    = max(fighter.cached_sw, 1.0)

    elbow:      Dict[str, Optional[np.ndarray]] = {}
    wrist_raw:  Dict[str, Optional[np.ndarray]] = {}
    wrist_conf: Dict[str, float]                = {}
    for s in sides:
        e_i, w_i = ELBOW_MAP[s], WRIST_MAP[s]
        elbow[s]      = _kp_xy(kps, e_i) if float(conf[e_i]) >= KP_CONF_THRESHOLD else None
        wc            = float(conf[w_i])
        wrist_conf[s] = wc
        wrist_raw[s]  = _kp_xy(kps, w_i) if wc >= WRIST_KP_CONF_THRESHOLD else None

    # ── Update measured forearm length from clean frames ──────────────────────
    for s in sides:
        if elbow[s] is not None and wrist_raw[s] is not None:
            fl = float(np.linalg.norm(wrist_raw[s] - elbow[s]))
            if 0.30 * sw <= fl <= 1.60 * sw:          # plausible forearm range
                prev = fighter.cached_forearm[s]
                fighter.cached_forearm[s] = (
                    fl if prev is None
                    else (1.0 - FOREARM_EMA) * prev + FOREARM_EMA * fl
                )

    out: Dict[str, WristMeas] = {s: WristMeas() for s in sides}

    # ── Color glove refinement setup ──────────────────────────────────────────
    sample_r    = int(np.clip(GLOVE_MODEL_SAMPLE_SW  * sw, 12, 60))
    refine_r    = int(np.clip(GLOVE_REFINE_RADIUS_SW * sw, 20, 140))
    min_blob    = (GLOVE_MIN_BLOB_SW * sw) ** 2
    max_shift   = GLOVE_REFINE_MAX_SHIFT_SW * sw
    model_ready = (fighter.glove_color_model is not None
                   and fighter.glove_model_sat >= GLOVE_MODEL_MIN_SAT)

    def _finalize_pos(side: str, w: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Build the colour model from confident frames, then snap to the glove.
        Returns (position, snapped_to_glove).
        """
        if frame is None:
            return w, False
        if fighter.glove_color_model is None:
            samp = _glove_color_sample(frame, w, sample_r)
            if samp is not None:
                fighter.glove_model_accum.append(samp)
                if len(fighter.glove_model_accum) >= GLOVE_MODEL_FRAMES:
                    hists = np.mean([h for h, _ in fighter.glove_model_accum],
                                    axis=0).astype(np.float32)
                    cv2.normalize(hists, hists, 0, 255, cv2.NORM_MINMAX)
                    fighter.glove_color_model = hists
                    fighter.glove_model_sat   = float(
                        np.mean([s for _, s in fighter.glove_model_accum]))
                    fighter.glove_model_accum = []
            return w, False
        if model_ready:
            refined, reason = _refine_to_glove(frame, w, fighter.glove_color_model,
                                               refine_r, min_blob, max_shift)
            fighter.refine_reason[side][reason] = (
                fighter.refine_reason[side].get(reason, 0) + 1)
            if refined is not None:
                fighter.glove_refined[side] += 1
                return refined, True
        return w, False

    for s in sides:
        state     = fighter.hands[s]
        state.frames_since_yolo += 1            # reset to 0 if a real wrist lands
        other     = "Right" if s == "Left" else "Left"
        fl        = fighter.cached_forearm[s] or (FOREARM_DEFAULT_SW * sw)
        max_reach = fl * FOREARM_MAX_FACTOR
        w         = wrist_raw[s]
        e         = elbow[s]
        e_other   = elbow[other]

        accepted = False

        if w is not None and e is not None:
            d_own = float(np.linalg.norm(w - e))
            if d_own <= max_reach:
                own_ok = True
                # Anti-glue / anti-swap: must be closer to its own elbow.
                if e_other is not None:
                    d_other = float(np.linalg.norm(w - e_other))
                    if d_own > d_other * ELBOW_OWNERSHIP_MARGIN:
                        own_ok = False
                        fighter.reject_ownership[s] += 1
                # Temporal sanity: can't teleport from the last real detection.
                if own_ok and not _wrist_step_ok(state, w, sw):
                    own_ok = False
                    fighter.reject_step[s] += 1
                if own_ok:
                    # Step gate references the pose WRIST (consistent frame to
                    # frame); the OUTPUT is snapped to the glove face.
                    state.last_yolo_pos     = w.copy()
                    state.frames_since_yolo = 0
                    pos, snapped = _finalize_pos(s, w)
                    out[s] = WristMeas(pos, wrist_conf[s], "yolo", refined=snapped)
                    accepted = True
            else:
                fighter.reject_forearm[s] += 1

        elif w is not None and e is None:
            # No elbow to validate against — accept (down-weighted) only if the
            # move from the last real detection is physically plausible.
            if _wrist_step_ok(state, w, sw):
                state.last_yolo_pos     = w.copy()
                state.frames_since_yolo = 0
                pos, snapped = _finalize_pos(s, w)
                out[s] = WristMeas(pos, wrist_conf[s] * NO_ELBOW_CONF_SCALE,
                                   "yolo", refined=snapped)
                accepted = True
            else:
                fighter.reject_step[s] += 1

        # ── Kinematic fallback: elbow visible, no usable wrist keypoint ───────
        # Use the last CORRECTED glove position (current_xy), NOT the velocity-
        # extrapolated prediction.  Extrapolating forward made the dot overshoot
        # into empty space after a punch (the "dot in the torso" artifact) — when
        # we're blind, the safest estimate is "the glove is near where it last
        # was", clamped to the forearm circle around the current elbow.
        if not accepted and e is not None:
            last = state.kf.current_xy
            if last is not None:
                v = last - e
                d = float(np.linalg.norm(v))
                proj = e + (v / d) * min(d, max_reach) if d > 1e-3 else last
                out[s] = WristMeas(proj, KINEMATIC_FALLBACK_CONF, "kinematic")
                fighter.fallback_kinematic[s] += 1

    return out


def _estimate_head_measurement(
    kps:     np.ndarray,
    conf:    np.ndarray,
    fighter: "Fighter",
) -> Tuple[Optional[np.ndarray], float]:
    """
    Estimate the head centre, tethered to the shoulder midpoint (the neck).

      1. Best face keypoint via _head_position (nose → eyes → ears).
      2. Validate: must be within HEAD_MAX_NECK_SW × shoulder-width of the
         shoulder midpoint.  A face keypoint farther than this belongs to a
         spectator or the opponent, not this fighter.
      3. Fallback: if no valid face keypoint, project from the shoulder
         midpoint using the last accepted head offset (low confidence).

    Returns (position, confidence); position None → head KF should ghost.
    """
    sw  = max(fighter.cached_sw, 1.0)
    raw = _head_position(kps, conf)
    sm  = _torso_centre(kps, conf)        # shoulder midpoint = neck anchor

    if raw is not None:
        if sm is None:
            return raw, HEAD_VALID_CONF                # no anchor — accept as-is
        if float(np.linalg.norm(raw - sm)) <= HEAD_MAX_NECK_SW * sw:
            fighter.cached_head_offset = (raw - sm).copy()
            return raw, HEAD_VALID_CONF
        # implausibly far → fall through to shoulder-projection fallback

    if sm is not None and fighter.cached_head_offset is not None:
        return sm + fighter.cached_head_offset, HEAD_FALLBACK_CONF

    return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Skeleton validation
# ══════════════════════════════════════════════════════════════════════════════

def _has_valid_skeleton(kps: np.ndarray, conf: np.ndarray) -> bool:
    """
    Returns True only if this detection looks like a fighter, not a referee,
    corner staff, or crowd member.

    Requirements (all at confidence ≥ KP_CONF_THRESHOLD):
      • Both shoulders   (5, 6)
      • Both hips        (11, 12)
      • At least one lower-body point — knee or ankle (13–16)

    A partial upper-body detection (common for refs leaning in) will fail
    the lower-body check and be ignored.
    """
    required_ok = all(float(conf[kp]) >= KP_CONF_THRESHOLD for kp in SKELETON_REQUIRED)
    lower_ok    = any(float(conf[kp]) >= KP_CONF_THRESHOLD for kp in SKELETON_LOWER)
    return required_ok and lower_ok


def _is_trackable_fighter(kps: np.ndarray, conf: np.ndarray) -> bool:
    """
    RELAXED validity for ONGOING re-identification (not initial lock).

    Once we already know who the fighters are, we shouldn't demand a full-body
    skeleton every frame to keep tracking them — in close footage the legs are
    routinely cropped by the frame or hidden behind the opponent/ropes, yet the
    fighter is plainly present.  This requires only:
      • Both shoulders            (the reliable upper-body anchor)
      • At least one hip          (torso is present, not just a floating head)
    Identity is still protected by the colour-similarity gate at the call site;
    this check only decides "is this a plausibly trackable body", not "who".
    """
    shoulders_ok = (float(conf[KP_LEFT_SHOULDER])  >= KP_CONF_THRESHOLD and
                    float(conf[KP_RIGHT_SHOULDER]) >= KP_CONF_THRESHOLD)
    hip_ok       = (float(conf[KP_LEFT_HIP])  >= KP_CONF_THRESHOLD or
                    float(conf[KP_RIGHT_HIP]) >= KP_CONF_THRESHOLD)
    return shoulders_ok and hip_ok


def _assign_fighters_by_appearance(
    fighters:       List["Fighter"],
    track_ids_arr:  np.ndarray,
    kps_all:        np.ndarray,
    conf_all:       np.ndarray,
    xyxy_all:       np.ndarray,
    color_profiles: Dict[int, Optional[np.ndarray]],
    frame:          np.ndarray,
) -> Dict[int, Optional[int]]:
    """
    Decide which detection IS each fighter THIS frame — no ID persistence, no
    reject-the-present-fighter gate.  Every trackable body is scored against
    each fighter by colour similarity + proximity to that fighter's last known
    position, and the highest-scoring 2-way assignment wins.

    Returns {0: det_index|None, 1: det_index|None}.  A fighter is None only when
    no candidate clears ASSIGN_FLOOR (i.e. genuinely absent), not because a gate
    rejected a present body.
    """
    out: Dict[int, Optional[int]] = {0: None, 1: None}
    cands = [i for i in range(len(track_ids_arr))
             if _is_trackable_fighter(kps_all[i], conf_all[i])]
    if not cands or color_profiles[0] is None or color_profiles[1] is None:
        return out

    # Precompute appearance + centre for each candidate.
    hist_of:   Dict[int, np.ndarray] = {}
    centre_of: Dict[int, np.ndarray] = {}
    for i in cands:
        hist_of[i]   = _extract_hist(frame, xyxy_all[i])
        bx           = xyxy_all[i]
        centre_of[i] = np.array([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])

    def score(fi: int, i: int) -> float:
        cs = float(cv2.compareHist(hist_of[i], color_profiles[fi], cv2.HISTCMP_CORREL))
        prox = 0.0
        lb = fighters[fi].cached_bbox
        if lb is not None:
            last_c = np.array([(lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0])
            sw     = max(fighters[fi].cached_sw, 1.0)
            d      = float(np.linalg.norm(centre_of[i] - last_c))
            prox   = max(0.0, 1.0 - d / (ASSIGN_PROX_NORM_SW * sw))
        return ASSIGN_W_COLOR * cs + ASSIGN_W_PROX * prox

    sA = {i: score(0, i) for i in cands}
    sB = {i: score(1, i) for i in cands}

    # Best 2-way assignment (either fighter may be None if no body clears floor).
    best_tot, best_a, best_b = -1e9, None, None
    opts = cands + [None]
    for a in opts:
        if a is not None and sA[a] < ASSIGN_FLOOR:
            continue
        for b in opts:
            if a is not None and b is not None and a == b:
                continue
            if b is not None and sB[b] < ASSIGN_FLOOR:
                continue
            tot = (sA[a] if a is not None else 0.0) + (sB[b] if b is not None else 0.0)
            if tot > best_tot:
                best_tot, best_a, best_b = tot, a, b

    out[0], out[1] = best_a, best_b
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Color histogram helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_hist(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    HS histogram of the trunk/shorts region (lower portion) of a bounding box.
    Value channel is ignored so brightness changes don't affect similarity.
    Returns a normalised float32 array ready for cv2.compareHist.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
    bh = y2 - y1
    trunk_y = y1 + int(bh * COLOR_TRUNK_START)   # skip head/shoulders
    roi = frame[trunk_y:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((COLOR_HIST_H_BINS, COLOR_HIST_S_BINS), dtype=np.float32)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [COLOR_HIST_H_BINS, COLOR_HIST_S_BINS],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _extract_glove_hist(
    frame: np.ndarray,
    wrist_pos: np.ndarray,
    radius: int = GLOVE_ROI_RADIUS,
) -> Optional[np.ndarray]:
    """
    HS histogram of the square ROI centered on a wrist keypoint.

    Uses the same H×S bin layout as the fighter trunk histogram so scores
    are directly comparable.  Value channel is discarded for lighting
    invariance.  Returns None if the ROI is empty or entirely off-frame.
    """
    fh, fw = frame.shape[:2]
    cx, cy = int(round(wrist_pos[0])), int(round(wrist_pos[1]))
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(fw - 1, cx + radius), min(fh - 1, cy + radius)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [COLOR_HIST_H_BINS, COLOR_HIST_S_BINS],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _glove_color_sample(
    frame:  np.ndarray,
    center: np.ndarray,
    radius: int,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Sample a tight ROI around a confident wrist for building the glove colour
    model.  Returns (L1-normalised HS histogram, mean saturation) or None.
    """
    fh, fw = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(fw, cx + radius), min(fh, cy + radius)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [COLOR_HIST_H_BINS, COLOR_HIST_S_BINS],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    mean_sat = float(hsv[:, :, 1].mean())
    return hist.astype(np.float32), mean_sat


def _refine_to_glove(
    frame:    np.ndarray,
    seed:     np.ndarray,
    model:    np.ndarray,
    radius:   int,
    min_area: float,
    max_shift: float,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Snap a pose-based wrist seed onto the actual glove using HSV back-projection.

    Searches a window of the given radius around `seed`, back-projects against the
    fighter's glove colour `model`, thresholds + cleans the probability map, and
    returns the centroid of the glove-coloured blob nearest the seed (area-gated).

    Returns (point, reason).  point is None when no plausible blob is found or it
    would move the point more than `max_shift` px (caller keeps the pose estimate).
    `reason` ∈ {ok, roi_small, no_contour, no_blob_area, shift_too_big} for
    diagnostics.
    """
    fh, fw = frame.shape[:2]
    cx, cy = int(round(seed[0])), int(round(seed[1]))
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(fw, cx + radius), min(fh, cy + radius)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None, "roi_small"

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bp  = cv2.calcBackProject([hsv], [0, 1], model, [0, 180, 0, 256], scale=1)
    cv2.GaussianBlur(bp, (5, 5), 0, dst=bp)
    _, mask = cv2.threshold(bp, GLOVE_BACKPROJ_THRESH, 255, cv2.THRESH_BINARY)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, "no_contour"

    seed_lx, seed_ly = cx - x1, cy - y1
    best_xy, best_d = None, float("inf")
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        bxx, byy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        d = math.hypot(bxx - seed_lx, byy - seed_ly)
        if d < best_d:
            best_d, best_xy = d, (bxx + x1, byy + y1)

    if best_xy is None:
        return None, "no_blob_area"
    refined = np.array(best_xy, dtype=np.float64)
    if float(np.linalg.norm(refined - seed)) > max_shift:
        return None, "shift_too_big"
    return refined, "ok"


def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Intersection-over-Union for two xyxy boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Visual overlay helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_fighter(
    frame: np.ndarray,
    fighter: Fighter,
    kps:  Optional[np.ndarray],
    conf: Optional[np.ndarray],
    bbox_xyxy: Optional[np.ndarray],
    track_id: Optional[int] = None,
) -> None:
    pal = fighter.colors
    h, w = frame.shape[:2]

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), pal["bbox"], 1)
        label = fighter.name.replace("_", " ")
        if track_id is not None:
            label += f" #{track_id}"
        lbl_w = len(label) * 9 + 8
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + lbl_w, y1), (30, 30, 30), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, pal["bbox"], 1, cv2.LINE_AA)

    if kps is not None and conf is not None:
        for i, j in SKELETON_EDGES:
            if _kp_vis(conf, i) and _kp_vis(conf, j):
                cv2.line(frame,
                         tuple(_kp_xy(kps, i).astype(int)),
                         tuple(_kp_xy(kps, j).astype(int)),
                         pal["skeleton"], 1, cv2.LINE_AA)

    for side, state in fighter.hands.items():
        pos = state.wrist_pos
        if pos is None:
            continue

        ip    = pos.astype(int)
        color = pal["ghost"] if state.is_ghost else pal["detected"]

        cv2.circle(frame, tuple(ip), 11, color,        -1)
        cv2.circle(frame, tuple(ip), 13, (255,255,255),  1)
        cv2.putText(frame, side[0], (ip[0] - 4, ip[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)

        if state.is_ghost:
            gf = state.kf.ghost_frames
            cv2.putText(frame, f"GHOST {gf}/{MAX_GHOST_FRAMES}",
                        (ip[0] + 16, ip[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, pal["ghost"], 1, cv2.LINE_AA)

        # Glove color match score overlay (shown for all locked hands)
        if state.glove_hist is not None:
            score_txt   = f"{state.glove_score:.2f}"
            score_color = (0, 210, 0) if state.glove_score >= GLOVE_MATCH_THRESHOLD else (0, 80, 230)
            cv2.putText(frame, score_txt, (ip[0] + 16, ip[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, score_color, 1, cv2.LINE_AA)

        if len(state.vec_buf) > 0:
            vec = state.vec_buf[-1]
            if np.linalg.norm(vec) > 0.5:
                end = np.clip(ip + (vec * 4).astype(int), [0, 0], [w-1, h-1])
                cv2.arrowedLine(frame, tuple(ip), tuple(end),
                                pal["skeleton"], 1, tipLength=0.35)

    # ── Head dot ──────────────────────────────────────────────────────────────
    head_pos = fighter.head.head_pos
    if head_pos is not None:
        hp    = head_pos.astype(int)
        hcol  = pal["ghost"] if fighter.head.is_ghost else pal["detected"]
        cv2.circle(frame, tuple(hp), 9,  hcol,          -1)
        cv2.circle(frame, tuple(hp), 11, (255, 255, 255), 1)
        cv2.putText(frame, "H", (hp[0] - 4, hp[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
        if fighter.head.is_ghost:
            gf = fighter.head.kf.ghost_frames
            cv2.putText(frame, f"GHOST {gf}/{MAX_GHOST_FRAMES}",
                        (hp[0] + 13, hp[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, pal["ghost"], 1, cv2.LINE_AA)


def _draw_tracking_lost(frame: np.ndarray, fighter: Fighter) -> None:
    """Overlay a 'Tracking Lost — Predicting' badge near the fighter's last bbox."""
    pal = fighter.colors
    h, w = frame.shape[:2]

    # Anchor to last known bbox centre; fall back to left/right third of frame
    if fighter.cached_bbox is not None:
        x1, y1, x2, y2 = fighter.cached_bbox.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
    else:
        cx = w // 4 if fighter.name == "Fighter_A" else 3 * w // 4
        cy = h // 2

    label  = f"{fighter.name.replace('_', ' ')} — Tracking Lost  Predicting"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tx = max(4, cx - lw // 2)
    ty = max(lh + 4, cy - 20)

    cv2.rectangle(frame, (tx - 6, ty - lh - 4), (tx + lw + 6, ty + 4),
                  (30, 30, 30), -1)
    cv2.rectangle(frame, (tx - 6, ty - lh - 4), (tx + lw + 6, ty + 4),
                  pal["bbox"], 1)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, pal["ghost"], 1, cv2.LINE_AA)

    # Dashed border where the bbox used to be
    if fighter.cached_bbox is not None:
        x1, y1, x2, y2 = fighter.cached_bbox.astype(int)
        for seg_start, seg_end in [
            ((x1, y1), (x1 + (x2-x1)//2, y1)),
            ((x1 + (x2-x1)//2, y2), (x2, y2)),
            ((x1, y1), (x1, y1 + (y2-y1)//2)),
            ((x2, y1 + (y2-y1)//2), (x2, y2)),
        ]:
            cv2.line(frame, seg_start, seg_end, pal["ghost"], 1, cv2.LINE_AA)


def _draw_strikes(frame: np.ndarray,
                  events: List[dict],
                  timestamp: float) -> None:
    y = 36
    for ev in events:
        pal   = Fighter.PALETTES[ev["fighter"]]
        label = (f"{ev['fighter'].replace('_', ' ')}  {ev['hand'].upper()}"
                 f"  mag {ev['decel_magnitude']:.2f}  {timestamp:.2f}s")
        cv2.putText(frame, label, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, pal["strike"], 2, cv2.LINE_AA)
        y += 26


def _draw_hud(frame: np.ndarray,
              fighters: List[Fighter],
              frame_idx: int,
              timestamp: float,
              locked:           bool = False,
              occlusion_active: bool = False) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)

    for i, fighter in enumerate(fighters):
        pal   = fighter.colors
        label = (f"{fighter.name.replace('_',' ')}  "
                 f"L:{fighter.strike_counts['Left']}  "
                 f"R:{fighter.strike_counts['Right']}  "
                 f"total:{fighter.total_strikes}")
        x = 14 if i == 0 else w // 2 + 14
        cv2.putText(frame, label, (x, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, pal["detected"], 1, cv2.LINE_AA)

    status = f"frame {frame_idx:05d}  |  {timestamp:.3f}s"
    if not locked:
        status += "  |  SEARCHING FOR FIGHTERS…"
    cv2.putText(frame, status, (14, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

    if occlusion_active:
        badge  = "  ⚠ OCCLUSION — TRACKS FROZEN  "
        bw     = len(badge) * 9
        bx     = w // 2 - bw // 2
        by_top = h - 50
        by_bot = h - 28
        cv2.rectangle(frame, (bx, by_top + 4), (bx + bw, by_bot - 2), (0, 30, 180), -1)
        cv2.rectangle(frame, (bx, by_top + 4), (bx + bw, by_bot - 2), (30, 80, 255), 1)
        cv2.putText(frame, badge.strip(), (bx + 6, by_bot - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis function
# ══════════════════════════════════════════════════════════════════════════════

def analyze_video(
    video_path:       str,
    output_dir:       str  = "output_data",
    model_path:       str  = DEFAULT_MODEL,
    save_annotated:   bool = True,
    show_preview:     bool = False,
    max_frames:       Optional[int] = None,   # stop after N frames (for quick A/B tests)
) -> str:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[strike_analyzer] '{video_path}'  {width}×{height}  {fps:.1f} fps  ~{total} frames")

    print(f"[strike_analyzer] Loading model: {model_path}")
    model = YOLO(model_path)

    # Subfolders: output_data/output_strikes/ and output_data/output_videos/
    strikes_dir = os.path.join(output_dir, "output_strikes")
    videos_dir  = os.path.join(output_dir, "output_videos")
    os.makedirs(strikes_dir, exist_ok=True)
    os.makedirs(videos_dir,  exist_ok=True)
    stem     = Path(video_path).stem
    csv_path = os.path.join(strikes_dir, f"{stem}_strikes.csv")
    vid_path = os.path.join(videos_dir,  f"{stem}_annotated.mp4")

    writer: Optional[cv2.VideoWriter] = None
    if save_annotated:
        # Remove any existing file — AVFoundation refuses to overwrite in place
        if os.path.exists(vid_path):
            os.remove(vid_path)
        writer = cv2.VideoWriter(
            vid_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
        )
        if not writer.isOpened():
            print("[strike_analyzer] WARNING: avc1 writer failed — falling back to MJPG (.avi)")
            vid_path = vid_path.replace(".mp4", ".avi")
            writer = cv2.VideoWriter(
                vid_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
            )

    fighters = [Fighter("Fighter_A"), Fighter("Fighter_B")]

    # ── ByteTrack fighter lock ────────────────────────────────────────────────
    # Maps ByteTrack track_id → fighter index (0 or 1).
    # Populated once from the first frame that yields ≥2 tracks.
    # Any track ID not in this dict is ignored for the rest of the video.
    locked_ids:      Dict[int, int] = {}
    fighters_locked: bool          = False

    # ── Color histogram guardrail state ───────────────────────────────────────
    # Master profile: fi → 2D HS histogram (None until enough frames collected)
    color_profiles: Dict[int, Optional[np.ndarray]] = {0: None, 1: None}
    profile_accum:  Dict[int, List]                 = {0: [],   1: []}
    color_ready:    bool                            = False

    # ── Occlusion Freeze & Separation Protocol ────────────────────────────────
    occlusion_active:  bool  = False   # True while fighters' bboxes are heavily overlapping
    # Rolling buffer of clean (non-occluded) bbox pairs used to capture the
    # pre-clinch spatial anchor.  Entry format: {0: xyxy, 1: xyxy}
    preclinch_buf: deque = deque(maxlen=PRECLINCH_ANCHOR_FRAMES)
    # Bboxes captured from the oldest entry in preclinch_buf when occlusion fires
    preclinch_anchor: Dict[int, Optional[np.ndarray]] = {0: None, 1: None}
    # Saved KF state per hand so we can restore clean P after the clinch clears.
    # Format: {fi: {side: (statePost, errorCovPost)}}
    frozen_kf_state: Dict[int, Dict[str, tuple]] = {0: {}, 1: {}}

    # ── Re-identification / reset state ───────────────────────────────────────
    # Counts consecutive frames where BOTH locked fighters are missing.
    # When this exceeds BOTH_LOST_RESET we drop the lock and re-acquire,
    # handling broadcast cuts where ByteTrack assigns new track IDs.
    both_lost_frames: int = 0

    # Per-track consecutive-frame visibility counter.
    # Incremented each frame a track ID is present; deleted when it disappears.
    # Re-ID will only consider tracks that have been stable for REID_MIN_TRACK_AGE
    # frames — this filters out coaches/refs who just stepped into frame.
    track_age: Dict[int, int] = {}

    # Frames remaining before the colour guardrail is allowed to fire again.
    # Set to REID_COOLDOWN_FRAMES after every Re-ID event so a fresh, correct
    # assignment isn't immediately undone by one noisy histogram comparison.
    reid_cooldown: int = 0

    # ── Track-stability instrumentation ───────────────────────────────────────
    # Quantify how badly ByteTrack churns IDs / loses fighters, so config
    # changes can be measured rather than guessed at.
    all_track_ids_seen: set        = set()
    frames_visible:     Dict[int, int] = {0: 0, 1: 0}
    frames_lost:        Dict[int, int] = {0: 0, 1: 0}
    # Of the lost frames, how many had an unmatched valid-skeleton person on
    # screen (→ our identity/Re-ID logic failed, H2) vs none at all (→ genuine
    # detection failure, H1).  This decides whether to fix locking or detection.
    lost_with_candidate: Dict[int, int] = {0: 0, 1: 0}

    all_events: List[dict] = []
    frame_idx = 0

    print(f"[strike_analyzer] Processing frames with ByteTrack …")

    try:
     while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            print(f"[strike_analyzer] Reached max_frames={max_frames} — stopping early.")
            break

        timestamp = frame_idx / fps

        # ── ByteTrack inference ───────────────────────────────────────────────
        # persist=True keeps track state alive across calls.
        # No max_det cap here — ByteTrack needs to see all persons (incl.
        # referee) to maintain stable IDs; we filter by locked_ids ourselves.
        results = model.track(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,        # higher inference resolution → far fewer
                                     # dropouts on blurred/occluded fighters in
                                     # this 2880×1800 footage (default 640 loses
                                     # them during fast exchanges)
            tracker="bytetrack_custom.yaml",
            persist=True,
            verbose=False,
        )
        result = results[0]

        # ── Unpack detections ─────────────────────────────────────────────────
        has_tracks = (
            result.boxes is not None
            and result.boxes.id is not None
            and result.keypoints is not None
            and len(result.boxes.id) > 0
        )

        track_ids_arr: np.ndarray = np.array([], dtype=int)
        confs_arr:     np.ndarray = np.array([])
        kps_all:       np.ndarray = np.empty((0, 17, 2))
        conf_all:      np.ndarray = np.empty((0, 17))
        xyxy_all:      np.ndarray = np.empty((0, 4))

        if has_tracks:
            track_ids_arr = result.boxes.id.cpu().numpy().astype(int)
            confs_arr     = result.boxes.conf.cpu().numpy()
            kps_all       = result.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            conf_all      = result.keypoints.conf.cpu().numpy()  # (N, 17)
            xyxy_all      = result.boxes.xyxy.cpu().numpy()      # (N, 4)

        # ── Track-age maintenance ─────────────────────────────────────────────
        # Count how many consecutive frames each ByteTrack ID has been present.
        # IDs that disappear are removed so their count resets if they reappear.
        # Re-ID uses this to reject brand-new tracks (age < REID_MIN_TRACK_AGE)
        # — coaches and refs who just entered frame always have age = 1.
        current_ids = set(track_ids_arr.tolist())
        all_track_ids_seen |= current_ids          # instrumentation: ID churn
        for tid in list(track_age.keys()):
            if tid not in current_ids:
                del track_age[tid]
        for tid in current_ids:
            track_age[tid] = track_age.get(tid, 0) + 1

        # ── Fighter lock: skeleton-validated, delayed until two full bodies ─────
        # Only runs until lock is achieved. Scans every track in this frame
        # and collects those that pass the full skeleton check (both shoulders,
        # both hips, ≥1 lower-body point).  Referees and partial detections
        # fail the check and are permanently ignored.
        if not fighters_locked and len(track_ids_arr) >= 2:
            valid = [
                i for i in range(len(track_ids_arr))
                if _has_valid_skeleton(kps_all[i], conf_all[i])
            ]

            if len(valid) >= 2:
                if color_ready:
                    # ── Re-lock after total loss — use stored color profiles ──
                    # We already know what each fighter looks like.  Match every
                    # valid track against both profiles and assign by best fit
                    # rather than blind left/right position.
                    remaining = list(valid)
                    best2 = [None, None]
                    for target_fi in (0, 1):
                        best_sim, best_i = -1.0, None
                        for i in remaining:
                            sim = cv2.compareHist(
                                _extract_hist(frame, xyxy_all[i]),
                                color_profiles[target_fi],
                                cv2.HISTCMP_CORREL,
                            )
                            if sim > best_sim:
                                best_sim, best_i = sim, i
                        if best_i is not None:
                            best2[target_fi] = best_i
                            remaining.remove(best_i)

                    if None in best2:
                        # Fallback: shouldn't happen, but guard anyway
                        valid.sort(key=lambda i: confs_arr[i], reverse=True)
                        best2 = valid[:2]
                        best2.sort(key=lambda i: (xyxy_all[i][0] + xyxy_all[i][2]) / 2.0)

                    # Color match defines the slot — no left/right re-sort needed
                    id_a = int(track_ids_arr[best2[0]])
                    id_b = int(track_ids_arr[best2[1]])
                else:
                    # ── First lock — prefer center-frame tracks ───────────────
                    # When 3+ valid skeletons are present, filter out anyone
                    # whose bbox center sits in the outer 15% of the frame width
                    # — ringside coaches and refs tend to stand at the edges.
                    if len(valid) > 2:
                        inner = [
                            i for i in valid
                            if 0.15 * width
                               <= (xyxy_all[i][0] + xyxy_all[i][2]) / 2.0
                               <= 0.85 * width
                        ]
                        if len(inner) >= 2:
                            valid = inner

                    valid.sort(key=lambda i: confs_arr[i], reverse=True)
                    best2 = valid[:2]
                    best2.sort(key=lambda i: (xyxy_all[i][0] + xyxy_all[i][2]) / 2.0)
                    id_a = int(track_ids_arr[best2[0]])
                    id_b = int(track_ids_arr[best2[1]])

                locked_ids[id_a] = 0   # Fighter_A
                locked_ids[id_b] = 1   # Fighter_B
                fighters_locked = True

                lock_method = "color re-lock" if color_ready else "skeleton lock"
                print(
                    f"\n  [SUCCESS] Skeletal structures verified ({lock_method}). "
                    f"Locked Fighter_A (ID: {id_a}) and Fighter_B (ID: {id_b}).\n"
                )

                # ── Lock-frame initialisation: KF seed + glove fingerprint ────
                # Fix 3: Pre-initialise the Kalman filter from the structural
                # YOLO keypoint with a tight error covariance so the filter
                # starts confident (not drifting from an inflated P matrix).
                # Glove fingerprints are seeded in the same pass.
                # Hands that are occluded at lock time get their fingerprint
                # lazily on the first accepted detection frame.
                for slot, det_i in [(0, best2[0]), (1, best2[1])]:
                    for side in ("Left", "Right"):
                        wrist_idx = WRIST_MAP[side]
                        hand = fighters[slot].hands[side]
                        if float(conf_all[det_i][wrist_idx]) >= WRIST_KP_CONF_THRESHOLD:
                            wp = _kp_xy(kps_all[det_i], wrist_idx)

                            # ── KF covariance reset (Fix 3) ───────────────────
                            kf = hand.kf
                            kf._kf.statePost = np.array(
                                [[wp[0]], [wp[1]], [0.], [0.], [0.], [0.]],
                                dtype=np.float32,
                            )
                            # Tight P: we're confident about the starting position
                            kf._kf.errorCovPost = (
                                np.eye(6, dtype=np.float32) * KF_POS_NOISE
                            )
                            kf._kf.processNoiseCov = WristKalmanFilter._Q_base.copy()
                            kf._init         = True
                            kf._ghost_frames = 0

                            # ── Glove fingerprint seed ────────────────────────
                            gh = _extract_glove_hist(frame, wp)
                            if gh is not None:
                                hand.glove_hist = gh

                print(
                    f"  [GLOVE]  HSV fingerprints + KF covariance seeded at lock frame "
                    f"(tight P = {KF_POS_NOISE}·I)."
                )

        # ── Per-fighter detection assignment ──────────────────────────────────
        # Once colour profiles exist, identity is decided EVERY frame by
        # appearance + motion (no ByteTrack-ID persistence, no reject gate).
        # Before that, bootstrap from the locked ByteTrack IDs while profiles
        # build.  `frame` is still pristine here (drawing happens later).
        detection_for: Dict[int, Optional[int]] = {0: None, 1: None}
        if fighters_locked and color_ready:
            detection_for = _assign_fighters_by_appearance(
                fighters, track_ids_arr, kps_all, conf_all, xyxy_all,
                color_profiles, frame,
            )
            # Conservatively blend each profile toward its assignment so it
            # tracks lighting/sweat drift, but only when the match is solid.
            for fi in (0, 1):
                di = detection_for[fi]
                if di is not None:
                    _h  = _extract_hist(frame, xyxy_all[di])
                    _cs = float(cv2.compareHist(_h, color_profiles[fi], cv2.HISTCMP_CORREL))
                    if _cs >= ASSIGN_ADAPT_MIN_SIM:
                        color_profiles[fi] = (
                            (1.0 - COLOR_ADAPT_RATE) * color_profiles[fi]
                            + COLOR_ADAPT_RATE * _h
                        ).astype(np.float32)
        elif fighters_locked:
            for det_i, tid in enumerate(track_ids_arr):
                if tid in locked_ids:
                    detection_for[locked_ids[tid]] = det_i

        # (Per-frame appearance assignment above replaces the old lock-and-Re-ID
        #  scramble: every frame already claims the best-matching bodies, so
        #  there is no separate "recover a changed ByteTrack ID" step.)

        # ── Full reset when both fighters are missing too long ────────────────
        # Broadcast cut or total occlusion — drop the lock so the next frame
        # with ≥2 good detections re-acquires Fighter_A / Fighter_B.
        # Colour profiles are kept so re-ID works immediately after re-lock.
        if fighters_locked:
            if detection_for[0] is None and detection_for[1] is None:
                both_lost_frames += 1
                if both_lost_frames >= BOTH_LOST_RESET:
                    locked_ids.clear()
                    fighters_locked   = False
                    both_lost_frames  = 0
                    print(f"  [RESET]  Both fighters lost for {BOTH_LOST_RESET} frames "
                          f"— re-acquiring lock")
            else:
                both_lost_frames = 0

        # ── Color profile accumulation (first COLOR_PROFILE_FRAMES frames) ────
        if fighters_locked and not color_ready:
            if detection_for[0] is not None and detection_for[1] is not None:
                for fi in (0, 1):
                    profile_accum[fi].append(
                        _extract_hist(frame, xyxy_all[detection_for[fi]])
                    )
                if len(profile_accum[0]) >= COLOR_PROFILE_FRAMES:
                    for fi in (0, 1):
                        color_profiles[fi] = np.mean(profile_accum[fi], axis=0).astype(np.float32)
                    color_ready = True
                    profile_accum[0].clear()   # no longer needed — free memory
                    profile_accum[1].clear()
                    print(f"  [COLOR]  Master profiles locked "
                          f"(averaged {COLOR_PROFILE_FRAMES} frames)")

        # (The colour guardrail is gone: per-frame appearance assignment already
        #  re-derives the correct A/B mapping every frame, so there is nothing to
        #  "correct" after the fact, and profile adaptation now happens inline at
        #  assignment time.)

        # ══════════════════════════════════════════════════════════════════════
        # Occlusion Freeze & Separation Protocol
        # ══════════════════════════════════════════════════════════════════════
        if fighters_locked:
            both_visible = (detection_for[0] is not None and
                            detection_for[1] is not None)

            # ── Maintain rolling pre-clinch anchor buffer ─────────────────────
            # Only populated from clean (non-occluded, both-visible) frames so
            # the spatial anchor always reflects genuine separated positions.
            if both_visible and not occlusion_active:
                preclinch_buf.append({
                    0: xyxy_all[detection_for[0]].copy(),
                    1: xyxy_all[detection_for[1]].copy(),
                })

            # ── Compute current bbox IoU ──────────────────────────────────────
            if both_visible:
                occ_iou = _bbox_iou(
                    xyxy_all[detection_for[0]],
                    xyxy_all[detection_for[1]],
                )
            else:
                occ_iou = 0.0

            # ── Enter occlusion state ─────────────────────────────────────────
            if not occlusion_active and occ_iou >= OCCLUSION_IOU_ENTER:
                occlusion_active = True

                # Capture spatial anchor from PRECLINCH_ANCHOR_FRAMES ago
                # (or the oldest available entry if buffer isn't full yet)
                if preclinch_buf:
                    anchor_entry     = preclinch_buf[0]   # deque index 0 = oldest
                    preclinch_anchor = {
                        0: anchor_entry[0].copy(),
                        1: anchor_entry[1].copy(),
                    }
                else:
                    preclinch_anchor = {0: None, 1: None}

                # Freeze KF covariance for every hand so the filter can't drift
                # while the tracks are unreliable during the overlap.
                for fi, fighter in enumerate(fighters):
                    frozen_kf_state[fi] = {}
                    for side, state in fighter.hands.items():
                        frozen_kf_state[fi][side] = (
                            state.kf._kf.statePost.copy(),
                            state.kf._kf.errorCovPost.copy(),
                        )

                print(f"  [INFO]  Occlusion detected (IoU: {occ_iou:.2f}). "
                      f"Freezing fighter tracks…")

            # ── Exit occlusion state ──────────────────────────────────────────
            elif occlusion_active and occ_iou < OCCLUSION_IOU_EXIT:
                occlusion_active = False

                # Restore pre-clinch KF covariance so the filter re-initialises
                # tightly from the last trusted position rather than from the
                # inflated uncertainty accumulated during the overlap.
                for fi, fighter in enumerate(fighters):
                    for side, state in fighter.hands.items():
                        saved = frozen_kf_state.get(fi, {}).get(side)
                        if saved is not None:
                            sp, ep                     = saved
                            state.kf._kf.statePost    = sp.copy()
                            state.kf._kf.errorCovPost = ep.copy()
                            state.kf._ghost_frames    = 0

                # Identity through the clinch is now handled by the per-frame
                # appearance assignment (it re-derives A/B from colour + motion
                # every frame), so no spatial-anchor swap is needed here — we
                # only restored the KF covariance above.
                print(f"  [INFO]  Clinch broken (IoU {occ_iou:.2f}). KF restored; "
                      f"identity re-derived per-frame by appearance.")

        # ── Per-fighter processing ────────────────────────────────────────────
        strike_events_this_frame: List[dict] = []

        # Clean (un-annotated) copy for all colour work.  Fighter A's overlay is
        # drawn onto `frame` before Fighter B is processed, so colour sampling /
        # glove back-projection must read this pristine copy instead.
        clean_frame = frame.copy()

        # Instrumentation: how many valid-skeleton persons are on screen but NOT
        # matched to either fighter?  If a fighter is lost while one of these
        # exists, our identity logic failed (H2), not detection (H1).
        _matched_idx = {detection_for[0], detection_for[1]} - {None}
        n_unmatched_valid = sum(
            1 for i in range(len(track_ids_arr))
            if i not in _matched_idx and _is_trackable_fighter(kps_all[i], conf_all[i])
        )

        for fi, fighter in enumerate(fighters):
            det_idx = detection_for[fi]

            # Instrumentation: once locked, is this fighter visible this frame?
            if fighters_locked:
                if det_idx is None:
                    frames_lost[fi] += 1
                    if n_unmatched_valid > 0:
                        lost_with_candidate[fi] += 1
                else:
                    frames_visible[fi] += 1

            kps  = kps_all[det_idx]  if det_idx is not None else None
            conf = conf_all[det_idx] if det_idx is not None else None
            bbox = xyxy_all[det_idx] if det_idx is not None else None
            tid  = (track_ids_arr[det_idx]
                    if det_idx is not None else None)

            # ── Cache pose geometry and last known bbox ───────────────────────
            if bbox is not None:
                fighter.cached_bbox = bbox.copy()
            if kps is not None and conf is not None:
                sw = _shoulder_distance(kps, conf)
                tc = _torso_centre(kps, conf)
                if sw is not None:
                    # Slow EMA + clamp: resist the foreshortening collapse that
                    # otherwise divides velocity by a near-zero shoulder-width.
                    if fighter.cached_sw_ema is None:
                        fighter.cached_sw_ema = sw
                    else:
                        fighter.cached_sw_ema = (
                            (1.0 - CACHED_SW_EMA) * fighter.cached_sw_ema
                            + CACHED_SW_EMA * sw
                        )
                    fighter.cached_sw = float(np.clip(
                        sw,
                        CACHED_SW_MIN_RATIO * fighter.cached_sw_ema,
                        CACHED_SW_MAX_RATIO * fighter.cached_sw_ema,
                    ))
                if tc is not None:
                    fighter.cached_torso = tc

            # ── Track lost: advance ghost KFs and keep wrist dots moving ────────
            # The KF ghost-predicts a position even with no real detection.
            # Capture it so the wrist overlay keeps gliding instead of freezing
            # at the last real frame — much easier to tell where the fighter is.
            def _tally_flyaway(p: Optional[np.ndarray], side: str) -> None:
                if p is None:
                    return
                if not (0 <= p[0] < width and 0 <= p[1] < height):
                    fighter.dot_offframe[side] += 1
                elif (fighter.cached_torso is not None and
                      float(np.linalg.norm(p - fighter.cached_torso))
                      > 3.5 * fighter.cached_sw):
                    fighter.dot_farbody[side] += 1

            track_lost = fighters_locked and det_idx is None
            if track_lost:
                _cb = fighter.cached_bbox
                for side, state in fighter.hands.items():
                    wrist_pos, is_ghost = state.kf.update(None)
                    # Ghost predictions are unbounded velocity extrapolations —
                    # cage them so they can't fly off-screen / off-body.
                    wrist_pos = _clamp_prediction(wrist_pos, _cb, width, height)
                    state.wrist_pos = wrist_pos   # keep dot moving (but caged)
                    state.is_ghost  = is_ghost
                    _tally_flyaway(wrist_pos, side)
                head_pos, is_ghost_h = fighter.head.kf.update(None)
                head_pos = _clamp_prediction(head_pos, _cb, width, height)
                fighter.head.head_pos = head_pos
                fighter.head.is_ghost = is_ghost_h
                if save_annotated or show_preview:
                    _draw_tracking_lost(frame, fighter)
                    # Draw ghost wrist dots on top of the badge
                    _draw_fighter(frame, fighter, None, None, None)
                continue

            # ── Velocity and deceleration (only for locked, visible tracks) ───
            if not fighters_locked:
                if save_annotated or show_preview:
                    _draw_fighter(frame, fighter, kps, conf, bbox, track_id=tid)
                continue

            # ── Joint anatomical wrist estimation ─────────────────────────────
            # Both wrists are estimated together (the elbow-ownership test needs
            # both elbows) BEFORE the per-hand loop.  Skipped during occlusion,
            # when merged bboxes make every keypoint unreliable.
            if not occlusion_active and kps is not None and conf is not None:
                wrist_meas = _estimate_wrist_measurements(kps, conf, fighter, clean_frame)
            else:
                wrist_meas = {"Left": WristMeas(), "Right": WristMeas()}

            for side, state in fighter.hands.items():

                # ── Occlusion freeze ──────────────────────────────────────────
                # While fighters' bboxes heavily overlap, YOLO keypoints are
                # unreliable (two bodies merged in one bbox).  Hold the last
                # clean wrist_pos, skip all updates, and do NOT score strikes.
                if occlusion_active:
                    continue

                # ── Consume the anatomical estimate ───────────────────────────
                # The estimator already enforced the forearm-reach constraint,
                # the elbow-ownership (anti-glue/anti-swap) test, and the
                # kinematic fallback.  `raw` is a trusted measurement or None.
                meas    = wrist_meas[side]
                raw     = meas.pos
                kp_conf = meas.conf

                # ── Wrist pixel velocity (drives sub-frame interpolation) ─────
                # Derived from the last KF output and this frame's measurement —
                # the same signal Farneback estimated, available in O(1).
                disp_vx = disp_vy = disp_mag = 0.0
                if raw is not None and state.wrist_pos is not None:
                    _disp    = raw - state.wrist_pos
                    disp_vx  = float(_disp[0])
                    disp_vy  = float(_disp[1])
                    disp_mag = math.sqrt(disp_vx**2 + disp_vy**2)

                # ── Glove fingerprint score (display only — never gates) ──────
                # Colour matching is too unreliable to drive tracking decisions
                # (similar gloves, lighting drift), so it is purely informational
                # now — the anatomical constraints own acceptance.  We still
                # maintain the fingerprint so the overlay can show a live score.
                if raw is not None:
                    cand = _extract_glove_hist(clean_frame, raw)
                    if cand is not None:
                        if state.glove_hist is None:
                            state.glove_hist  = cand.copy()
                            state.glove_score = 1.0
                        else:
                            state.glove_score = float(cv2.compareHist(
                                cand, state.glove_hist, cv2.HISTCMP_CORREL))
                            state.glove_hist = (
                                (1.0 - GLOVE_ADAPT_RATE) * state.glove_hist
                                + GLOVE_ADAPT_RATE * cand
                            ).astype(np.float32)

                # ── OEF bypass during strike extension ────────────────────────
                # Mid-extension (high previous velocity + confident keypoint):
                # feed raw coords straight through so the One Euro Filter can't
                # lag the impact point.  The confidence gate prevents a low-conf
                # jitter → high-velocity → bypass feedback loop.
                if raw is not None:
                    prev_vel     = state.vel_buf[-1] if state.vel_buf else 0.0
                    in_extension = (prev_vel > STRIKE_VEL_BYPASS
                                    and kp_conf >= OEF_BYPASS_MIN_CONF)
                    filtered_pos = raw if in_extension else state.oef.filter(raw, timestamp)
                else:
                    filtered_pos = None

                # ── Sub-frame temporal interpolation (real detections only) ───
                # Only inject velocity from genuine YOLO displacements, never
                # from the smooth kinematic-fallback estimate (which carries no
                # new high-frequency motion information).
                if (state.kf._init and disp_mag > 0.0
                        and meas.source == "yolo"):
                    if disp_mag > SUBFRAME_TRIGGER:
                        # ── High-speed sub-frame path ─────────────────────────
                        _step_vx = disp_vx / SUBFRAME_STEPS
                        _step_vy = disp_vy / SUBFRAME_STEPS
                        for _sub in range(SUBFRAME_STEPS - 1):   # steps 1 … N-1
                            state.kf._kf.statePost[2, 0] = _step_vx
                            state.kf._kf.statePost[3, 0] = _step_vy
                            state.kf._kf.predict()
                            # Promote predicted state + covariance for next step
                            state.kf._kf.statePost[:]    = state.kf._kf.statePre[:]
                            state.kf._kf.errorCovPost[:] = state.kf._kf.errorCovPre[:]
                        # Prime velocity for the final predict inside update()
                        state.kf._kf.statePost[2, 0] = _step_vx
                        state.kf._kf.statePost[3, 0] = _step_vy
                    elif disp_mag >= DISP_SPIKE_THRESHOLD:
                        # ── Standard single-step injection ────────────────────
                        state.kf._kf.statePost[2, 0] = disp_vx
                        state.kf._kf.statePost[3, 0] = disp_vy

                wrist_pos, is_ghost = state.kf.update(filtered_pos, kp_conf=kp_conf)

                # Cage PREDICTED dots (ghost or kinematic fallback) to the
                # current bbox + frame.  A real, glove-snapped/at-wrist detection
                # is left exactly where it was observed.
                if wrist_pos is not None and (is_ghost or meas.source != "yolo"):
                    wrist_pos = _clamp_prediction(wrist_pos, bbox, width, height)

                state.wrist_pos = wrist_pos
                state.is_ghost  = is_ghost

                # ── Dot-source classification (the real on-glove metric) ──────
                if wrist_pos is not None:
                    if is_ghost:
                        fighter.dot_ghost[side] += 1
                    elif meas.source == "kinematic":
                        fighter.dot_fallback[side] += 1
                    elif meas.source == "yolo":
                        if meas.refined:
                            fighter.dot_on_glove[side] += 1
                        else:
                            fighter.dot_at_wrist[side] += 1
                    _tally_flyaway(wrist_pos, side)

                if wrist_pos is None:
                    state.prev_pos = None
                    continue

                sw = fighter.cached_sw
                if state.prev_pos is not None:
                    delta = wrist_pos - state.prev_pos
                    vel   = float(np.linalg.norm(delta)) / sw
                    vec   = delta
                else:
                    vel = 0.0
                    vec = np.zeros(2)

                state.vel_buf.append(vel)
                state.vec_buf.append(vec)
                state.prev_pos = wrist_pos

                # ── Deceleration trigger ──────────────────────────────────────
                # Ghost positions are the KF making its best guess with no real
                # data — don't score strikes from guesses.
                if is_ghost:
                    continue

                buf = state.vel_buf
                if len(buf) < DECEL_LOOKBACK + 1:
                    continue

                cur_vel  = buf[-1]
                base_vel = float(np.mean(list(buf)[-DECEL_LOOKBACK - 1:-1]))

                if base_vel < MIN_BASELINE_VEL:
                    continue

                # ── Physical velocity ceiling ─────────────────────────────────
                # A real fist tops out ~11 m/s (~0.40 sw/f here).  A baseline
                # velocity beyond MAX_STRIKE_VEL_SW is a tracking glitch, not a
                # punch — drop it so glitches can't inflate counts or velocities.
                if base_vel > MAX_STRIKE_VEL_SW:
                    fighter.reject_step[side] += 1
                    continue

                drop = (base_vel - cur_vel) / base_vel
                if drop <= DECEL_THRESHOLD:
                    fighter.reject_decel[side] += 1
                    continue

                # ── Arm extension check ───────────────────────────────────────
                ext = (_arm_extension(kps, conf, side)
                       if kps is not None and conf is not None else 0.0)
                if ext < EXTENSION_THRESHOLD:
                    fighter.reject_extension[side] += 1
                    continue

                # ── Directional intent check (no retractions) ─────────────────
                baseline_vecs = list(state.vec_buf)[-DECEL_LOOKBACK - 1:-1]
                if not baseline_vecs:
                    continue
                approach = np.mean(baseline_vecs, axis=0)

                torso   = fighter.cached_torso if fighter.cached_torso is not None \
                          else np.array([width / 2.0, height / 2.0])
                outward = wrist_pos - torso

                if np.dot(approach, outward) <= 0:
                    fighter.reject_direction[side] += 1
                    continue

                # ── Strike debounce ───────────────────────────────────────────
                # The deceleration trigger fires on every frame of a follow-
                # through; collapse multi-frame events into a single record.
                if frame_idx - state.last_strike_frame <= STRIKE_DEBOUNCE_FRAMES:
                    continue
                state.last_strike_frame = frame_idx

                # ── Record strike ─────────────────────────────────────────────
                vel_ms = base_vel * SHOULDER_WIDTH_M * fps
                ev = {
                    "timestamp_s":          round(timestamp, 4),
                    "fighter":              fighter.name,
                    "hand":                 side,
                    "decel_magnitude":      round(drop,     4),
                    "baseline_velocity_sw": round(base_vel, 5),
                    "baseline_velocity_ms": round(vel_ms,   3),
                    "current_velocity_sw":  round(cur_vel,  5),
                    "arm_extension_ratio":  round(ext,      4),
                }
                all_events.append(ev)
                strike_events_this_frame.append(ev)
                fighter.record_strike(ev)

                print(f"  [STRIKE] {fighter.name}  {side:5s}  "
                      f"t={timestamp:.3f}s  mag={drop:.2f}  ext={ext:.2f}  "
                      f"vel={vel_ms:.1f}m/s"
                      + ("  (ghost)" if is_ghost else ""))

            # ── Head tracking (neck-tethered estimator) ───────────────────────
            # Skipped during occlusion — merged bboxes make face keypoints
            # unreliable (two heads merged into one detection).
            if not occlusion_active:
                if kps is not None and conf is not None:
                    raw_head, head_conf = _estimate_head_measurement(kps, conf, fighter)
                else:
                    raw_head, head_conf = None, 0.0
                filtered_head = (
                    fighter.head.oef.filter(raw_head, timestamp)
                    if raw_head is not None else None
                )
                head_pos, is_ghost_h = fighter.head.kf.update(
                    filtered_head, kp_conf=head_conf)
                fighter.head.head_pos = head_pos
                fighter.head.is_ghost = is_ghost_h

            # ── Draw fighter overlays ─────────────────────────────────────────
            if save_annotated or show_preview:
                _draw_fighter(frame, fighter, kps, conf, bbox, track_id=tid)

        # ── Shared frame overlays ─────────────────────────────────────────────
        if save_annotated or show_preview:
            _draw_strikes(frame, strike_events_this_frame, timestamp)
            _draw_hud(frame, fighters, frame_idx, timestamp,
                      locked=fighters_locked,
                      occlusion_active=occlusion_active)
            if writer:
                writer.write(frame)
            if show_preview:
                cv2.imshow("Strike Analyzer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    finally:
        # ── Cleanup — always runs even on Ctrl+C ──────────────────────────────
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\n[strike_analyzer] ── Session Summary ──────────────────────────")
    for f in fighters:
        print(f.summary())
    print(f"  Total strikes: {len(all_events)}")

    # ── Track-stability report ────────────────────────────────────────────────
    print(f"\n  ── Track stability ──")
    print(f"  Distinct ByteTrack IDs used (lower = less churn): {len(all_track_ids_seen)}")
    for fi in (0, 1):
        tot = frames_visible[fi] + frames_lost[fi]
        pct = 100.0 * frames_lost[fi] / max(1, tot)
        cand = lost_with_candidate[fi]
        cand_pct = 100.0 * cand / max(1, frames_lost[fi])
        print(f"    {fighters[fi].name}: visible {frames_visible[fi]}  "
              f"lost {frames_lost[fi]}  ({pct:.1f}% of locked frames lost)")
        print(f"        of those lost frames, {cand} ({cand_pct:.1f}%) had an "
              f"unmatched fighter-shaped person on screen → identity-logic gap (H2)")
        print(f"        the remaining {frames_lost[fi]-cand} were genuine "
              f"detection failures (H1)")

    df = pd.DataFrame(all_events, columns=[
        "timestamp_s", "fighter", "hand", "decel_magnitude",
        "baseline_velocity_sw", "baseline_velocity_ms",
        "current_velocity_sw", "arm_extension_ratio",
    ])
    df.to_csv(csv_path, index=False)

    print(f"\n[strike_analyzer] CSV       → {csv_path}")
    if save_annotated:
        print(f"[strike_analyzer] Annotated → {vid_path}")

    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── Resolve video path ────────────────────────────────────────────────────
    requested = (sys.argv[1] if len(sys.argv) > 1
                 else "input_videos/Screen Recording 2026-05-13 at 7.35.56 PM.mov")

    if os.path.exists(requested):
        video_path = requested
    else:
        print(f"\n[strike_analyzer] File not found: '{requested}'")

        # Search project directory for any video files (skip output folders)
        skip_dirs   = {"output_data", "output_videos", ".git", "__pycache__"}
        video_exts  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        found: List[str] = []
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if os.path.splitext(fname)[1].lower() in video_exts:
                    found.append(os.path.normpath(os.path.join(root, fname)))

        if not found:
            print("[strike_analyzer] No .mp4 files found anywhere in the project. "
                  "Drop a video into input_videos/ and try again.")
            sys.exit(1)

        print("\n  Found the following .mp4 files:")
        for i, p in enumerate(found):
            print(f"    [{i + 1}] {p}")

        print("\n  Enter a number to use that file, or press Enter to cancel: ", end="", flush=True)
        choice = input().strip()

        if not choice.isdigit() or not (1 <= int(choice) <= len(found)):
            print("[strike_analyzer] No file selected. Exiting.")
            sys.exit(0)

        video_path = found[int(choice) - 1]
        print(f"\n[strike_analyzer] Using: {video_path}")

    # Env overrides for quick A/B experiments:
    #   SA_MODEL=yolo11n-pose.pt   — override the pose model
    #   SA_MAX_FRAMES=3600         — stop after N frames
    _model = os.environ.get("SA_MODEL", DEFAULT_MODEL)
    _maxf  = os.environ.get("SA_MAX_FRAMES")
    analyze_video(
        video_path,
        output_dir     = "output_data",
        model_path     = _model,
        save_annotated = True,
        show_preview   = False,
        max_frames     = int(_maxf) if _maxf else None,
    )

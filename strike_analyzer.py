"""
combat_analytics_v1 — Strike Analyzer
Detects significant punch impacts via wrist deceleration using MediaPipe Pose.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_WRIST   = 15
RIGHT_WRIST  = 16
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW   = 13
RIGHT_ELBOW  = 14

# ── Tuning constants ──────────────────────────────────────────────────────────
VELOCITY_BUFFER_SIZE   = 5    # rolling window length
DECEL_LOOKBACK         = 3    # frames to average for baseline velocity
DECEL_THRESHOLD        = 0.80 # drop fraction required to trigger (>80%)
EXTENSION_THRESHOLD    = 0.70 # arm must be ≥70% extended
MIN_BASELINE_VELOCITY  = 2.0  # px/frame — ignore near-still positions


def _landmark_xy(landmarks, idx, w, h):
    """Return pixel coords for a pose landmark."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def _arm_extension(landmarks, side, w, h):
    """
    Ratio of straight-line shoulder→wrist distance to the sum of segment lengths.
    1.0 = fully extended, 0.5 = fully bent.
    """
    if side == "Left":
        shoulder, elbow, wrist = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    else:
        shoulder, elbow, wrist = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

    s = _landmark_xy(landmarks, shoulder, w, h)
    e = _landmark_xy(landmarks, elbow,    w, h)
    wr = _landmark_xy(landmarks, wrist,   w, h)

    direct   = np.linalg.norm(wr - s)
    segmented = np.linalg.norm(e - s) + np.linalg.norm(wr - e)

    return direct / segmented if segmented > 0 else 0.0


def analyze_video(video_path: str, output_dir: str = "output_data") -> str:
    """
    Process a video file and write a CSV of detected strike impact events.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where the output CSV will be saved.

    Returns:
        Path to the generated CSV file.
    """
    # ── VIDEO FILENAME ────────────────────────────────────────────────────────
    # TODO: Replace the video_path argument below when you have your file ready.
    # Example: analyze_video("input_videos/sparring_clip.mp4")
    # ─────────────────────────────────────────────────────────────────────────

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(static_image_mode=False,
                            model_complexity=1,
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

    # Per-hand state
    hands = {
        "Left":  {"prev_pos": None, "vel_buf": deque(maxlen=VELOCITY_BUFFER_SIZE)},
        "Right": {"prev_pos": None, "vel_buf": deque(maxlen=VELOCITY_BUFFER_SIZE)},
    }
    wrist_idx = {"Left": LEFT_WRIST, "Right": RIGHT_WRIST}

    events = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        timestamp = frame_idx / fps

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark

            for side, state in hands.items():
                wrist_pos = _landmark_xy(lms, wrist_idx[side], w, h)

                # Velocity (pixels/frame), or 0 on first frame
                if state["prev_pos"] is not None:
                    velocity = float(np.linalg.norm(wrist_pos - state["prev_pos"]))
                else:
                    velocity = 0.0

                buf = state["vel_buf"]
                buf.append(velocity)

                # Need at least DECEL_LOOKBACK + 1 samples to compare
                if len(buf) >= DECEL_LOOKBACK + 1:
                    current_vel  = buf[-1]
                    baseline_vel = float(np.mean(list(buf)[-DECEL_LOOKBACK - 1:-1]))

                    # Skip when the hand was already barely moving
                    if baseline_vel >= MIN_BASELINE_VELOCITY:
                        drop_fraction = (baseline_vel - current_vel) / baseline_vel

                        if drop_fraction > DECEL_THRESHOLD:
                            extension = _arm_extension(lms, side, w, h)

                            if extension >= EXTENSION_THRESHOLD:
                                # Magnitude: scale-invariant relative deceleration (0–1)
                                # A 60→5 drop and a 500→40 drop both score ≈0.92
                                magnitude = drop_fraction

                                events.append({
                                    "timestamp_s":         round(timestamp, 4),
                                    "hand":                side,
                                    "decel_magnitude":     round(magnitude, 4),
                                    "baseline_velocity":   round(baseline_vel, 2),
                                    "current_velocity":    round(current_vel, 2),
                                    "arm_extension_ratio": round(extension, 4),
                                })

                state["prev_pos"] = wrist_pos

        frame_idx += 1

    cap.release()
    pose.close()

    # ── Output ────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem     = Path(video_path).stem
    out_path = str(Path(output_dir) / f"{stem}_strikes.csv")

    df = pd.DataFrame(events)
    df.to_csv(out_path, index=False)

    print(f"[strike_analyzer] {len(events)} impact event(s) detected.")
    print(f"[strike_analyzer] Results saved → {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    # ── ENTRY POINT ───────────────────────────────────────────────────────────
    # TODO: Set your video filename here when ready.
    # Default path shown below — override via CLI arg or edit directly.
    # Example:
    #   python scripts/strike_analyzer.py input_videos/sparring_clip.mp4
    # ─────────────────────────────────────────────────────────────────────────

    default_video = "input_videos/sparring_clip.mp4"  # ← REPLACE ME
    video_path    = sys.argv[1] if len(sys.argv) > 1 else default_video

    analyze_video(video_path, output_dir="output_data")

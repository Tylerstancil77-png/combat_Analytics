"""
SAM2.1 glove-tracking prototype  (standalone — head-to-head vs CoTracker3)
==========================================================================
Hypothesis: boxing gloves are solid-coloured / low-texture — bad for POINT
trackers (CoTracker) but ideal for MASK tracking.  SAM2's video predictor
propagates an object mask through the clip with temporal memory + an occlusion
head, so it should weld to the colored glove blob.

Pipeline:
  1. Read a short window of frames, downscaled, write to a temp clip.
  2. Seed a point on each glove via YOLO pose on frame 0 (detect-once).
  3. For each glove, run SAM2VideoPredictor (memory propagation) → per-frame
     mask → centroid = the tracked dot.
  4. Overlay centroids (+ mask outline) and save frames + clip to judge sticking.

Usage:
  venv/bin/python3 scripts/sam2_prototype.py \
      "input_videos/Screen Recording 2026-05-13 at 7.35.56 PM.mov" \
      --start 48 --frames 60 --width 960
"""

import argparse
import os
import tempfile
import time

import cv2
import numpy as np

# reuse the window reader + pose seeder from the CoTracker prototype
from cotracker_prototype import read_window, seed_points_from_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=48.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--model", default="sam2.1_t.pt")   # tiny = fastest on CPU
    ap.add_argument("--out", default="output_data/sam2_prototype")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[sam2] reading {args.frames} frames from {args.start}s, width={args.width}")
    small, scale, fps = read_window(args.video, args.start, args.frames, args.width)
    T = len(small)
    H, W = small[0].shape[:2]
    print(f"[sam2] {T} frames at {W}x{H} (scale={scale:.3f})")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start * fps)))
    _, first_full = cap.read()
    cap.release()
    qpts, labels = seed_points_from_pose(first_full, scale)
    print(f"[sam2] seeded {len(qpts)} glove points: {labels}")
    if len(qpts) == 0:
        raise SystemExit("No wrist seeds on first frame — try a different --start.")

    # Write the downscaled window to a temp clip (SAM2 video predictor wants a
    # file).  MJPG/.avi writes SYNCHRONOUSLY — avc1/mp4v on macOS flush async,
    # so the file may not exist yet when the predictor opens it.
    tmp = os.path.join(tempfile.gettempdir(), "sam2_window.avi")
    if os.path.exists(tmp):
        os.remove(tmp)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("Could not open MJPG VideoWriter for temp clip")
    for f in small:
        vw.write(f)
    vw.release()
    if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
        raise RuntimeError(f"Temp clip not written: {tmp}")
    print(f"[sam2] wrote temp clip {tmp} ({os.path.getsize(tmp)//1024} KB)")

    from ultralytics.models.sam import SAM2VideoPredictor
    colors = [(0, 215, 255), (255, 160, 0), (80, 255, 120), (80, 80, 255)]

    # tracks[label] = list of (cx, cy, area) per frame (None if no mask)
    tracks = {lab: [None] * T for lab in labels}

    track_t0 = time.time()
    for gi, (pt, lab) in enumerate(zip(qpts, labels)):
        print(f"[sam2] tracking glove {lab} from seed {pt.tolist()} …", flush=True)
        overrides = dict(conf=0.25, task="segment", mode="predict",
                         imgsz=args.width, model=args.model, verbose=False, save=False)
        predictor = SAM2VideoPredictor(overrides=overrides)
        results = predictor(source=tmp, points=[float(pt[0]), float(pt[1])], labels=[1])
        for t, r in enumerate(results):
            if t >= T:
                break
            if r.masks is None or len(r.masks.data) == 0:
                continue
            m = r.masks.data[0].cpu().numpy().astype(np.uint8)
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            tracks[lab][t] = (float(xs.mean()), float(ys.mean()), int(len(xs)))

    # Overlay + save
    vw = cv2.VideoWriter(os.path.join(args.out, "tracked.mp4"),
                         cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    for t in range(T):
        fr = small[t].copy()
        for gi, lab in enumerate(labels):
            rec = tracks[lab][t]
            if rec is None:
                continue
            cx, cy, _ = rec
            c = colors[gi % len(colors)]
            cv2.circle(fr, (int(cx), int(cy)), 6, c, -1)
            cv2.putText(fr, lab, (int(cx) + 7, int(cy) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        vw.write(fr)
        if t % 8 == 0 or t == T - 1:
            cv2.imwrite(os.path.join(args.out, f"frame_{t:03d}.png"), fr)
    vw.release()

    track_secs = time.time() - track_t0
    print(f"[sam2] wrote {args.out}/tracked.mp4 + sample frames")
    for lab in labels:
        got = sum(1 for r in tracks[lab] if r is not None)
        print(f"  {lab}: mask found in {100*got/T:.0f}% of frames")
    n_obj = len(labels)
    print(f"[sam2] TIMING: tracked {n_obj} gloves × {T} frames in {track_secs:.1f}s")
    print(f"        ≈ {track_secs/max(1,T):.2f}s per video-frame for all {n_obj} gloves")
    full_video_min = (track_secs / max(1, T)) * 10202 / 60.0
    print(f"        → full 171s clip (~10202 frames) ≈ {full_video_min:.0f} min on this CPU")


if __name__ == "__main__":
    main()

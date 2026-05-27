"""
EdgeTAM glove-tracking prototype  (lightweight SAM2-family — Mac-speed candidate)
=================================================================================
Same winning mask-tracking approach as the SAM2 prototype, but with EdgeTAM —
the on-device SAM2 variant — to test whether we can keep SAM2-quality glove
welding at a speed that's viable on a typical Mac (target: << SAM2's 7.2s/frame).

Uses the official SAM2 video-predictor API (EdgeTAM is a SAM2 fork):
  build_sam2_video_predictor → init_state(frames_dir) → add_new_points_or_box →
  propagate_in_video.  All gloves tracked in ONE propagation pass.

Usage:
  venv/bin/python3 scripts/edgetam_prototype.py \
      "input_videos/Screen Recording 2026-05-13 at 7.35.56 PM.mov" \
      --start 48 --frames 60 --width 960
"""

import argparse
import os
import time

# Allow unsupported ops to fall back to CPU on Apple-GPU (MPS) — must be set
# before torch initialises.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

from cotracker_prototype import read_window, seed_points_from_pose


def pick_device() -> str:
    """Use the Apple GPU (MPS) or CUDA when available, else CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

EDGETAM_DIR = os.path.expanduser("~/edgetam_src")
CKPT = os.path.join(EDGETAM_DIR, "checkpoints", "edgetam.pt")
CFG = "edgetam.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=48.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default="output_data/edgetam_prototype")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    print(f"[edgetam] device = {device.upper()}"
          + ("  (Apple GPU — this is the real speed test)" if device == "mps"
             else "  (CPU — pessimistic; a modern Mac will use MPS)"))

    print(f"[edgetam] reading {args.frames} frames from {args.start}s, width={args.width}")
    small, scale, fps = read_window(args.video, args.start, args.frames, args.width)
    T = len(small)
    H, W = small[0].shape[:2]
    print(f"[edgetam] {T} frames at {W}x{H} (scale={scale:.3f})")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start * fps)))
    _, first_full = cap.read()
    cap.release()
    qpts, labels = seed_points_from_pose(first_full, scale)
    print(f"[edgetam] seeded {len(qpts)} glove points: {labels}")
    if len(qpts) == 0:
        raise SystemExit("No wrist seeds on first frame — try a different --start.")

    # Write window frames as JPEGs (SAM2 video API wants a frame directory).
    frames_dir = os.path.join(args.out, "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, f in enumerate(small):
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), f)

    from sam2.build_sam import build_sam2_video_predictor
    print("[edgetam] building predictor…", flush=True)
    predictor = build_sam2_video_predictor(CFG, CKPT, device=device)

    state = predictor.init_state(video_path=frames_dir)
    predictor.reset_state(state)
    for gi, pt in enumerate(qpts):
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=gi,
            points=np.array([[pt[0], pt[1]]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    # tracks[gi] = list of (cx,cy,area) per frame
    tracks = {gi: [None] * T for gi in range(len(qpts))}
    t0 = time.time()
    with torch.inference_mode():
        for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for k, gi in enumerate(obj_ids):
                m = (mask_logits[k, 0] > 0.0).cpu().numpy()
                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                if 0 <= f_idx < T:
                    tracks[gi][f_idx] = (float(xs.mean()), float(ys.mean()), int(len(xs)))
    track_secs = time.time() - t0

    colors = [(0, 215, 255), (255, 160, 0), (80, 255, 120), (80, 80, 255)]
    vw = cv2.VideoWriter(os.path.join(args.out, "tracked.mp4"),
                         cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    for t in range(T):
        fr = small[t].copy()
        for gi in range(len(qpts)):
            rec = tracks[gi][t]
            if rec is None:
                continue
            cx, cy, _ = rec
            c = colors[gi % len(colors)]
            cv2.circle(fr, (int(cx), int(cy)), 6, c, -1)
            cv2.putText(fr, labels[gi], (int(cx) + 7, int(cy) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        vw.write(fr)
        if t % 8 == 0 or t == T - 1:
            cv2.imwrite(os.path.join(args.out, f"frame_{t:03d}.png"), fr)
    vw.release()

    print(f"[edgetam] wrote {args.out}/tracked.mp4 + sample frames")
    for gi in range(len(qpts)):
        got = sum(1 for r in tracks[gi] if r is not None)
        print(f"  {labels[gi]}: mask found in {100*got/T:.0f}% of frames")
    print(f"[edgetam] TIMING: {len(qpts)} gloves × {T} frames in {track_secs:.1f}s")
    print(f"        ≈ {track_secs/max(1,T):.3f}s per video-frame (all gloves, ONE pass)")
    print(f"        → full 171s clip (~10202 frames) ≈ {(track_secs/max(1,T))*10202/60:.0f} min on this CPU")
    print(f"        (vs full SAM2 measured ≈ 1223 min)")


if __name__ == "__main__":
    main()

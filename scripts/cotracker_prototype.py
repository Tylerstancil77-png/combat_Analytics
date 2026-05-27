"""
CoTracker3 glove-tracking prototype  (standalone — does NOT touch strike_analyzer)
==================================================================================
Goal: empirically test whether a temporal POINT tracker (CoTracker3) welds a
point to a boxing glove through punches/occlusion far better than the per-frame
pose+filter pipeline (which ceilings at ~70% on-glove).

Pipeline:
  1. Read a short window of frames from the footage, downscaled (CPU-friendly).
  2. Use YOLO11-pose on the FIRST frame only to seed query points on each
     fighter's wrists (the glove seeds).  (Detect-once.)
  3. Hand the clip + seed points to CoTracker3 (offline) — it tracks them
     through the whole window with temporal coherence + occlusion handling.
  4. Overlay the tracked points and save an annotated clip + sample frames so
     we can SEE whether they stick to the gloves.

Usage:
  venv/bin/python3 scripts/cotracker_prototype.py \
      "input_videos/Screen Recording 2026-05-13 at 7.35.56 PM.mov" \
      --start 56 --frames 90 --width 640
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
WRIST_CONF_MIN = 0.30


def read_window(video_path, start_sec, n_frames, target_w):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_sec * fps)))
    frames = []
    for _ in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read.")
    h, w = frames[0].shape[:2]
    scale = target_w / w
    out_w, out_h = int(w * scale), int(h * scale)
    small = [cv2.resize(f, (out_w, out_h)) for f in frames]
    return small, scale, fps


def seed_points_from_pose(first_frame_full, scale, model_path="yolo11n-pose.pt"):
    """Run pose on the FIRST full-res frame; return query points (in downscaled
    coords) on the wrists of the two highest-confidence people, plus labels."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    res = model(first_frame_full, conf=0.40, iou=0.45, verbose=False)[0]
    pts, labels = [], []
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 2)), labels
    kps = res.keypoints.xy.cpu().numpy()        # (N,17,2) full-res
    kpc = res.keypoints.conf.cpu().numpy()      # (N,17)
    bconf = res.boxes.conf.cpu().numpy()
    order = np.argsort(bconf)[::-1][:2]         # two most confident people
    for slot, i in enumerate(order):
        tag = "A" if slot == 0 else "B"
        for side, kidx in (("L", KP_LEFT_WRIST), ("R", KP_RIGHT_WRIST)):
            if float(kpc[i][kidx]) >= WRIST_CONF_MIN:
                x, y = kps[i][kidx]
                pts.append([x * scale, y * scale])
                labels.append(f"{tag}-{side}")
    return np.array(pts, dtype=np.float32), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=56.0)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="output_data/cotracker_prototype")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cpu"

    print(f"[proto] reading {args.frames} frames from {args.start}s, width={args.width}")
    small, scale, fps = read_window(args.video, args.start, args.frames, args.width)
    T = len(small)
    H, W = small[0].shape[:2]
    print(f"[proto] got {T} frames at {W}x{H} (scale={scale:.3f})")

    # Seed query points on frame 0 using full-res pose.
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start * fps)))
    _, first_full = cap.read()
    cap.release()
    qpts, labels = seed_points_from_pose(first_full, scale)
    print(f"[proto] seeded {len(qpts)} query points: {labels}")
    if len(qpts) == 0:
        raise SystemExit("No wrist seeds found on first frame — try a different --start.")

    # Build tensors.  video: (1,T,C,H,W) float 0-255 ; queries: (1,N,3) = (t,x,y)
    vid = np.stack(small)[..., ::-1]            # BGR->RGB
    video = torch.from_numpy(np.ascontiguousarray(vid)).permute(0, 3, 1, 2)[None].float()
    queries = torch.zeros((1, len(qpts), 3), dtype=torch.float32)
    queries[0, :, 0] = 0.0                      # all seeded at t=0
    queries[0, :, 1] = torch.from_numpy(qpts[:, 0])
    queries[0, :, 2] = torch.from_numpy(qpts[:, 1])

    print("[proto] loading CoTracker3 (offline)…")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()

    print("[proto] tracking… (CPU, this may take a few minutes)")
    with torch.no_grad():
        pred_tracks, pred_vis = model(video.to(device), queries=queries.to(device))
    # pred_tracks: (1,T,N,2) ; pred_vis: (1,T,N)
    tracks = pred_tracks[0].cpu().numpy()
    vis = pred_vis[0].cpu().numpy()
    print(f"[proto] tracks shape {tracks.shape}, vis shape {vis.shape}")

    # Overlay + save.
    colors = [(0, 215, 255), (255, 160, 0), (80, 255, 120), (80, 80, 255),
              (255, 0, 255), (0, 255, 255)]
    vw = cv2.VideoWriter(os.path.join(args.out, "tracked.mp4"),
                         cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    for t in range(T):
        fr = small[t].copy()
        for n in range(tracks.shape[1]):
            x, y = tracks[t, n]
            seen = vis[t, n] > 0.5
            c = colors[n % len(colors)]
            cv2.circle(fr, (int(x), int(y)), 6, c, -1 if seen else 1)
            cv2.putText(fr, labels[n], (int(x) + 7, int(y) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        vw.write(fr)
        if t % 12 == 0 or t == T - 1:
            cv2.imwrite(os.path.join(args.out, f"frame_{t:03d}.png"), fr)
    vw.release()
    print(f"[proto] wrote {args.out}/tracked.mp4 + sample frames")
    # quick visibility stats
    for n in range(tracks.shape[1]):
        print(f"  {labels[n]}: visible {100*np.mean(vis[:,n]>0.5):.0f}% of frames")


if __name__ == "__main__":
    main()

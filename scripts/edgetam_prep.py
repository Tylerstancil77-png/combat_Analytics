"""
EdgeTAM prep (run with the 3.9 venv) — does all the OpenCV/pose work so the
3.11 EdgeTAM env can stay minimal (torch + sam2 only).

Extracts a window of frames as JPEGs and computes glove seed points via pose,
writing both to <out>/ for the tracker step to consume.

Usage (3.9 venv):
  venv/bin/python3 scripts/edgetam_prep.py \
      "input_videos/...mov" --start 48 --frames 60 --width 960 --out output_data/edgetam_io
"""
import argparse, json, os
import cv2
from cotracker_prototype import read_window, seed_points_from_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=48.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default="output_data/edgetam_io")
    args = ap.parse_args()

    frames_dir = os.path.join(args.out, "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    small, scale, fps = read_window(args.video, args.start, args.frames, args.width)
    T = len(small); H, W = small[0].shape[:2]
    for i, f in enumerate(small):
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), f)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.start * fps)))
    _, first_full = cap.read()
    cap.release()
    qpts, labels = seed_points_from_pose(first_full, scale)

    meta = {
        "frames_dir": frames_dir, "T": T, "W": W, "H": H, "fps": fps,
        "seeds": qpts.tolist(), "labels": labels,
    }
    with open(os.path.join(args.out, "seeds.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"[prep] wrote {T} frames to {frames_dir} + {len(qpts)} seeds {labels}")


if __name__ == "__main__":
    main()

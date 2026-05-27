"""
EdgeTAM tracker (run with the 3.11 venv_edgetam) — torch + sam2 only, no cv2.

Reads the frames dir + seeds written by edgetam_prep.py, mask-tracks every glove
in ONE propagation pass, and writes per-frame centroids to tracks.json plus
timing.  Overlay/visualisation is done back in the 3.9 venv (it has OpenCV).

Usage (3.11 venv):
  venv_edgetam/bin/python scripts/edgetam_track.py --io output_data/edgetam_io
"""
import argparse, json, os, time
import numpy as np
import torch

EDGETAM_DIR = os.path.expanduser("~/edgetam_src")
CKPT = os.path.join(EDGETAM_DIR, "checkpoints", "edgetam.pt")
CFG = "edgetam.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default="output_data/edgetam_io")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.io, "seeds.json")))
    frames_dir = meta["frames_dir"]
    T = meta["T"]
    seeds = meta["seeds"]
    labels = meta["labels"]
    print(f"[track] {T} frames, {len(seeds)} gloves {labels}")

    from sam2.build_sam import build_sam2_video_predictor
    print("[track] building EdgeTAM predictor (CPU)…", flush=True)
    predictor = build_sam2_video_predictor(CFG, CKPT, device="cpu")

    state = predictor.init_state(video_path=frames_dir)
    predictor.reset_state(state)
    for gi, pt in enumerate(seeds):
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=gi,
            points=np.array([[pt[0], pt[1]]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    tracks = {gi: [None] * T for gi in range(len(seeds))}
    t0 = time.time()
    with torch.inference_mode():
        for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for k, gi in enumerate(obj_ids):
                m = (mask_logits[k, 0] > 0.0).cpu().numpy()
                ys, xs = np.where(m)
                if len(xs) and 0 <= f_idx < T:
                    tracks[int(gi)][f_idx] = [float(xs.mean()), float(ys.mean()), int(len(xs))]
    secs = time.time() - t0

    json.dump({"tracks": tracks, "labels": labels, "T": T,
               "track_secs": secs}, open(os.path.join(args.io, "tracks.json"), "w"))

    for gi in range(len(seeds)):
        got = sum(1 for r in tracks[gi] if r is not None)
        print(f"  {labels[gi]}: mask found in {100*got/T:.0f}% of frames")
    per = secs / max(1, T)
    print(f"[track] TIMING: {len(seeds)} gloves x {T} frames in {secs:.1f}s")
    print(f"        = {per:.3f}s per video-frame (all gloves, ONE pass)")
    print(f"        -> full 171s clip (~10202 frames) = {per*10202/60:.0f} min on THIS CPU")
    print(f"        (full SAM2 measured = ~1223 min; pose+filter = ~5-10 min)")


if __name__ == "__main__":
    main()

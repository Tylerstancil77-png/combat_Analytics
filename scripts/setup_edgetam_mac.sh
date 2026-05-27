#!/usr/bin/env bash
# ============================================================================
# Turnkey EdgeTAM env setup for a MODERN Apple-Silicon Mac (with MPS / Apple GPU)
# ----------------------------------------------------------------------------
# Purpose: stand up the EdgeTAM prototype so we get a REAL speed number on a Mac
# that actually has the Apple GPU (this dev box is CPU-only and can't show it).
#
# Run from the project root:   bash scripts/setup_edgetam_mac.sh
# Requires an arm64 Python 3.11+ (override with:  PYTHON=python3.12 bash ...).
# ============================================================================
set -euo pipefail

PY="${PYTHON:-python3.11}"
echo "[setup] Python: $($PY --version 2>&1)  ($(which "$PY"))"
$PY -c "import platform,sys; assert platform.machine()=='arm64','Need arm64 Python (got '+platform.machine()+')'; assert sys.version_info>=(3,10),'Need Python >=3.10'"

# 1. Fresh, isolated 3.11 venv (kept separate from the 3.9 analytics venv)
echo "[setup] creating venv_edgetam …"
$PY -m venv venv_edgetam
venv_edgetam/bin/python -m pip install --upgrade pip >/dev/null

# 2. Clone EdgeTAM (the checkpoint edgetam.pt is bundled in the repo)
if [ ! -d edgetam_src ]; then
  echo "[setup] cloning EdgeTAM …"
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git edgetam_src
fi

# 3. Patch the one torch incompatibility (expand().view() -> expand().reshape())
echo "[setup] patching perceiver.py (view->reshape) …"
perl -0pi -e 's/\.expand\(B, -1, -1\)\.view\(-1, 1, C\)/.expand(B, -1, -1).reshape(-1, 1, C)/' \
  edgetam_src/sam2/modeling/perceiver.py

# 4. Install EdgeTAM + its missing backbone dep (timm) + pose/IO deps.
#    On arm64 Macs these all ship wheels, so no slow source builds.
echo "[setup] installing EdgeTAM + deps (torch download is the slow part) …"
venv_edgetam/bin/python -m pip install -e ./edgetam_src timm ultralytics opencv-python

echo ""
echo "[setup] DONE.  Run the MPS speed test with:"
echo '  venv_edgetam/bin/python scripts/edgetam_prototype.py \'
echo '      "input_videos/Screen Recording 2026-05-13 at 7.35.56 PM.mov" \'
echo "      --start 48 --frames 60 --width 960"
echo ""
echo "Look for the '[edgetam] device = MPS' line and the 'TIMING … s per video-frame'"
echo "line — that per-frame number on MPS is the real speed answer."

"""
combat_analytics_v1 — Strike Labeler
=====================================
Streamlit tool for manually labeling strike events as real or false positive.
Loads any *_strikes.csv from output_data/, lets you review each event one by
one, and saves a *_labeled.csv ready for train_classifier.py.

Run:
    streamlit run scripts/labeler.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Strike Labeler",
    page_icon="🏷️",
    layout="centered",
)

OUTPUT_DIR  = Path("output_data")
LABELED_DIR = Path("output_data/labeled")
LABELED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "timestamp_s",
    "fighter",
    "hand",
    "decel_magnitude",
    "baseline_velocity_sw",
    "current_velocity_sw",
    "arm_extension_ratio",
    "is_ghost_frame",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_unlabeled() -> list:
    if not OUTPUT_DIR.exists():
        return []
    csvs = sorted(OUTPUT_DIR.glob("*_strikes.csv"), reverse=True)
    labeled_stems = {p.stem.replace("_labeled", "") for p in LABELED_DIR.glob("*_labeled.csv")}
    return [p for p in csvs if p.stem not in labeled_stems]


def load_session(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        df["label"] = None          # None = unlabeled
    return df


def save_labeled(df: pd.DataFrame, source_path: Path) -> Path:
    out = LABELED_DIR / f"{source_path.stem}_labeled.csv"
    df.to_csv(out, index=False)
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🏷️ Strike Labeler")
st.sidebar.markdown("---")

unlabeled = find_unlabeled()
all_csvs  = sorted(OUTPUT_DIR.glob("*_strikes.csv"), reverse=True) if OUTPUT_DIR.exists() else []

if not all_csvs:
    st.error("No strike CSV files found in `output_data/`. Run `strike_analyzer.py` first.")
    st.stop()

selected = st.sidebar.selectbox(
    "Session to label",
    options=all_csvs,
    format_func=lambda p: ("✅ " if p not in unlabeled else "⬜ ") + p.name,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Labels**\n\n"
    "✅ **Real Strike** — the event is a genuine punch impact\n\n"
    "❌ **False Positive** — the system misfired (guard, movement, noise)\n\n"
    "⏭ **Skip** — not sure, label later"
)


# ── Load data ─────────────────────────────────────────────────────────────────

labeled_path = LABELED_DIR / f"{selected.stem}_labeled.csv"
if labeled_path.exists():
    df = pd.read_csv(labeled_path)
    if "label" not in df.columns:
        df["label"] = None
else:
    df = load_session(selected)

total     = len(df)
labeled   = df["label"].notna().sum()
remaining = total - labeled


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏷️ Strike Labeler")
st.markdown(f"**Session:** `{selected.name}`")

c1, c2, c3 = st.columns(3)
c1.metric("Total Events",  total)
c2.metric("Labeled",       int(labeled))
c3.metric("Remaining",     int(remaining))

st.progress(int(labeled) / total if total > 0 else 0)
st.markdown("---")

if remaining == 0:
    st.success(f"All {total} events labeled! File saved to `{labeled_path}`.")
    save_labeled(df, selected)

    real   = (df["label"] == 1).sum()
    fp     = (df["label"] == 0).sum()
    st.metric("Real strikes",    real)
    st.metric("False positives", fp)
    st.dataframe(df, use_container_width=True)
    st.stop()


# ── Find next unlabeled event ─────────────────────────────────────────────────

unlabeled_idx = df[df["label"].isna()].index
current_i     = unlabeled_idx[0]
row           = df.loc[current_i]

st.subheader(f"Event {int(current_i) + 1} of {total}")


# ── Event card ────────────────────────────────────────────────────────────────

fighter_color = "#4FC3F7" if "A" in str(row.get("fighter", "")) else "#EF5350"

st.markdown(f"""
<div style="
    background: #1e1e2e;
    border-left: 5px solid {fighter_color};
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
">
    <h3 style="margin:0 0 12px 0; color:{fighter_color};">
        {row.get('fighter','—')}  ·  {row.get('hand','—')} Hand
    </h3>
    <table style="width:100%; font-size:15px; border-collapse:collapse;">
        <tr>
            <td style="padding:4px 0; color:#aaa;">Timestamp</td>
            <td style="padding:4px 0; font-weight:bold;">{row['timestamp_s']:.3f} s</td>
            <td style="width:40px;"></td>
            <td style="padding:4px 0; color:#aaa;">Decel Magnitude</td>
            <td style="padding:4px 0; font-weight:bold;">{row['decel_magnitude']:.3f}</td>
        </tr>
        <tr>
            <td style="padding:4px 0; color:#aaa;">Baseline Velocity</td>
            <td style="padding:4px 0; font-weight:bold;">{row['baseline_velocity_sw']:.4f} sw/f</td>
            <td></td>
            <td style="padding:4px 0; color:#aaa;">Current Velocity</td>
            <td style="padding:4px 0; font-weight:bold;">{row['current_velocity_sw']:.4f} sw/f</td>
        </tr>
        <tr>
            <td style="padding:4px 0; color:#aaa;">Arm Extension</td>
            <td style="padding:4px 0; font-weight:bold;">{row['arm_extension_ratio']:.3f}</td>
            <td></td>
            <td style="padding:4px 0; color:#aaa;">Ghost Frame</td>
            <td style="padding:4px 0; font-weight:bold;">{'Yes ⚠️' if row.get('is_ghost_frame') else 'No'}</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)


# ── Label buttons ─────────────────────────────────────────────────────────────

b1, b2, b3 = st.columns(3)

if b1.button("✅  Real Strike", use_container_width=True, type="primary"):
    df.loc[current_i, "label"] = 1
    save_labeled(df, selected)
    st.rerun()

if b2.button("❌  False Positive", use_container_width=True):
    df.loc[current_i, "label"] = 0
    save_labeled(df, selected)
    st.rerun()

if b3.button("⏭  Skip", use_container_width=True):
    # Move this row to end so it comes back later
    df.loc[current_i, "label"] = float("nan")
    row_data = df.loc[[current_i]].copy()
    df = df.drop(index=current_i).reset_index(drop=True)
    df = pd.concat([df, row_data], ignore_index=True)
    save_labeled(df, selected)
    st.rerun()


# ── Undo last label ───────────────────────────────────────────────────────────

st.markdown("---")
already_labeled = df[df["label"].notna()]
if not already_labeled.empty:
    if st.button("↩ Undo last label"):
        last_i = already_labeled.index[-1]
        df.loc[last_i, "label"] = None
        save_labeled(df, selected)
        st.rerun()


# ── Progress table ────────────────────────────────────────────────────────────

with st.expander("📋 All labels so far"):
    show = df[df["label"].notna()].copy()
    show["label"] = show["label"].map({1.0: "✅ Real Strike", 0.0: "❌ False Positive"})
    st.dataframe(show[["timestamp_s", "fighter", "hand",
                        "decel_magnitude", "arm_extension_ratio", "label"]],
                 use_container_width=True, hide_index=True)

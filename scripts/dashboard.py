"""
combat_analytics_v1 — Strike Dashboard
=======================================
Streamlit dashboard that reads a strike CSV produced by strike_analyzer.py
and renders three analytical views:

  1. Bar chart   — Left vs. Right hand strike volume
  2. Line graph  — Deceleration Magnitude over time (fatigue tracker)
  3. Stats table — Average velocity & magnitude for Jabs vs. Hooks

Run:
    streamlit run scripts/dashboard.py
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Combat Analytics",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Strike classification threshold ──────────────────────────────────────────
# arm_extension_ratio from strike_analyzer:
#   ≥ JAB_EXTENSION_THRESHOLD  → Jab   (arm mostly straight)
#   <  JAB_EXTENSION_THRESHOLD → Hook  (arm more bent)
JAB_EXTENSION_THRESHOLD = 0.85

# ── Colour palette ────────────────────────────────────────────────────────────
COL_LEFT  = "#4FC3F7"   # sky blue
COL_RIGHT = "#EF5350"   # red
COL_JAB   = "#66BB6A"   # green
COL_HOOK  = "#FFA726"   # orange


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def classify_strike(row: pd.Series) -> str:
    """Classify a strike as Jab or Hook based on arm extension ratio."""
    return "Jab" if row["arm_extension_ratio"] >= JAB_EXTENSION_THRESHOLD else "Hook"


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "timestamp_s", "hand", "decel_magnitude",
        "baseline_velocity_sw", "arm_extension_ratio",
    }
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {missing}")
        st.stop()

    df["strike_type"] = df.apply(classify_strike, axis=1)
    df["timestamp_s"] = df["timestamp_s"].round(3)
    return df


def find_csvs(output_dir: str = "output_data") -> list:
    p = Path(output_dir)
    if not p.exists():
        return []
    return sorted(p.glob("*_strikes.csv"), reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar — file selection & global filters
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🥊 Combat Analytics")
    st.markdown("---")

    csv_files = find_csvs()
    if not csv_files:
        st.error(
            "No strike CSV files found in `output_data/`.\n\n"
            "Run `strike_analyzer.py` on a video first."
        )
        st.stop()

    selected_file = st.selectbox(
        "Select session",
        options=csv_files,
        format_func=lambda p: p.name,
    )

    df_raw = load_csv(str(selected_file))

    st.markdown("---")
    st.subheader("Filters")

    hand_filter = st.multiselect(
        "Hand", options=["Left", "Right"], default=["Left", "Right"]
    )
    type_filter = st.multiselect(
        "Strike type", options=["Jab", "Hook"], default=["Jab", "Hook"]
    )

    min_mag = float(df_raw["decel_magnitude"].min())
    max_mag = float(df_raw["decel_magnitude"].max())

    if min_mag < max_mag:
        mag_range = st.slider(
            "Min decel magnitude",
            min_value=round(min_mag, 2),
            max_value=round(max_mag, 2),
            value=round(min_mag, 2),
            step=0.01,
        )
    else:
        mag_range = min_mag

    rolling_window = st.slider(
        "Fatigue chart rolling average (strikes)",
        min_value=1, max_value=10, value=3,
    )

    st.markdown("---")
    st.caption(f"Session: `{selected_file.name}`")
    st.caption(
        f"Jab threshold: arm extension ≥ {JAB_EXTENSION_THRESHOLD}"
    )


# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["hand"].isin(hand_filter)
    & df_raw["strike_type"].isin(type_filter)
    & (df_raw["decel_magnitude"] >= mag_range)
].copy()


# ═════════════════════════════════════════════════════════════════════════════
# Header KPIs
# ═════════════════════════════════════════════════════════════════════════════

st.title("🥊 Strike Analytics Dashboard")
st.markdown(f"**Session:** `{selected_file.name}`")
st.markdown("---")

total    = len(df)
n_left   = len(df[df["hand"]  == "Left"])
n_right  = len(df[df["hand"]  == "Right"])
n_jab    = len(df[df["strike_type"] == "Jab"])
n_hook   = len(df[df["strike_type"] == "Hook"])
avg_mag  = df["decel_magnitude"].mean() if total > 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Strikes",     total)
k2.metric("Left Hand",         n_left)
k3.metric("Right Hand",        n_right)
k4.metric("Avg Decel Magnitude", f"{avg_mag:.2f}")
k5.metric("Jabs / Hooks",      f"{n_jab} / {n_hook}")

st.markdown("---")

if total == 0:
    st.warning("No strikes match the current filters.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Chart 1 — Left vs. Right strike volume (bar chart)
# ═════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns(2)

with col1:
    st.subheader("① Strike Volume — Left vs. Right")

    vol_df = (
        df.groupby(["hand", "strike_type"])
        .size()
        .reset_index(name="count")
    )

    fig_vol = px.bar(
        vol_df,
        x="hand",
        y="count",
        color="strike_type",
        barmode="group",
        color_discrete_map={"Jab": COL_JAB, "Hook": COL_HOOK},
        labels={"hand": "Hand", "count": "Strike Count", "strike_type": "Type"},
        text="count",
    )
    fig_vol.update_traces(textposition="outside")
    fig_vol.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        legend_title_text="Strike Type",
        margin=dict(t=20, b=20),
        xaxis=dict(
            tickvals=["Left", "Right"],
            ticktext=["◀ Left", "Right ▶"],
        ),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Mini breakdown table under the chart
    hand_summary = (
        df.groupby("hand")
        .agg(
            total_strikes=("hand", "count"),
            avg_magnitude=("decel_magnitude", "mean"),
        )
        .round(3)
        .rename(columns={
            "total_strikes": "Strikes",
            "avg_magnitude": "Avg Magnitude",
        })
    )
    st.dataframe(hand_summary, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 2 — Deceleration Magnitude over time (fatigue tracker)
# ═════════════════════════════════════════════════════════════════════════════

with col2:
    st.subheader("② Decel Magnitude Over Time — Fatigue Tracker")

    df_sorted = df.sort_values("timestamp_s").reset_index(drop=True)
    df_sorted["strike_num"] = df_sorted.index + 1

    # Rolling average per hand so both tracks are individually smoothed
    for hand, grp_df in df_sorted.groupby("hand"):
        df_sorted.loc[grp_df.index, "rolling_mag"] = (
            grp_df["decel_magnitude"]
            .rolling(window=rolling_window, min_periods=1)
            .mean()
        )

    fig_fat = go.Figure()

    for hand, colour in [("Left", COL_LEFT), ("Right", COL_RIGHT)]:
        sub = df_sorted[df_sorted["hand"] == hand]
        if sub.empty:
            continue

        # Raw dots
        fig_fat.add_trace(go.Scatter(
            x=sub["timestamp_s"],
            y=sub["decel_magnitude"],
            mode="markers",
            marker=dict(color=colour, size=6, opacity=0.45),
            name=f"{hand} (raw)",
            legendgroup=hand,
        ))
        # Smoothed trend line
        fig_fat.add_trace(go.Scatter(
            x=sub["timestamp_s"],
            y=sub["rolling_mag"],
            mode="lines",
            line=dict(color=colour, width=2.5),
            name=f"{hand} ({rolling_window}-strike avg)",
            legendgroup=hand,
        ))

    fig_fat.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis_title="Time (s)",
        yaxis_title="Decel Magnitude",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fat, use_container_width=True)

    # Fatigue indicator: compare first half vs second half avg magnitude
    if len(df_sorted) >= 4:
        mid   = len(df_sorted) // 2
        early = df_sorted.iloc[:mid]["decel_magnitude"].mean()
        late  = df_sorted.iloc[mid:]["decel_magnitude"].mean()
        delta = late - early
        trend = "📉 Declining" if delta < -0.03 else ("📈 Improving" if delta > 0.03 else "➡ Stable")
        st.caption(
            f"Fatigue trend: **{trend}** — "
            f"early avg `{early:.2f}` → late avg `{late:.2f}` (Δ `{delta:+.2f}`)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Chart 3 — Fighter Stats: Jabs vs. Hooks
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("③ Fighter Stats — Jabs vs. Hooks")

stats_df = (
    df.groupby(["strike_type", "hand"])
    .agg(
        count=("decel_magnitude", "count"),
        avg_baseline_velocity=("baseline_velocity_sw", "mean"),
        avg_decel_magnitude=("decel_magnitude", "mean"),
        max_decel_magnitude=("decel_magnitude", "max"),
        avg_arm_extension=("arm_extension_ratio", "mean"),
    )
    .round(4)
    .reset_index()
    .rename(columns={
        "strike_type":          "Type",
        "hand":                 "Hand",
        "count":                "Count",
        "avg_baseline_velocity":"Avg Approach Velocity (sw/f)",
        "avg_decel_magnitude":  "Avg Decel Magnitude",
        "max_decel_magnitude":  "Peak Decel Magnitude",
        "avg_arm_extension":    "Avg Arm Extension",
    })
)

# Colour-code rows by strike type with Streamlit's style
def row_style(row):
    bg = "#1a3a1a" if row["Type"] == "Jab" else "#3a2800"
    return [f"background-color: {bg}"] * len(row)

styled = stats_df.style.apply(row_style, axis=1).format({
    "Avg Approach Velocity (sw/f)": "{:.4f}",
    "Avg Decel Magnitude":          "{:.3f}",
    "Peak Decel Magnitude":         "{:.3f}",
    "Avg Arm Extension":            "{:.3f}",
})
st.dataframe(styled, use_container_width=True, hide_index=True)

# Grouped bar: avg approach velocity Jab vs Hook per hand
fig_stats = px.bar(
    stats_df,
    x="Type",
    y="Avg Approach Velocity (sw/f)",
    color="Hand",
    barmode="group",
    facet_col=None,
    color_discrete_map={"Left": COL_LEFT, "Right": COL_RIGHT},
    text=stats_df["Avg Approach Velocity (sw/f)"].apply(lambda v: f"{v:.4f}"),
    labels={"Type": "Strike Type"},
    title="Average Approach Velocity — Jab vs. Hook",
)
fig_stats.update_traces(textposition="outside")
fig_stats.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="#fafafa",
    margin=dict(t=50, b=20),
    yaxis_title="Velocity (shoulder-widths / frame)",
)
st.plotly_chart(fig_stats, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Raw data expander
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("📋 Raw strike log"):
    st.dataframe(
        df.sort_values("timestamp_s").reset_index(drop=True),
        use_container_width=True,
    )
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download filtered CSV",
        data=csv_bytes,
        file_name=f"filtered_{selected_file.name}",
        mime="text/csv",
    )

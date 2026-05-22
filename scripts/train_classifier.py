"""
combat_analytics_v1 — Strike Classifier Trainer
================================================
Trains a Random Forest classifier on labeled strike data produced by labeler.py.
Saves the trained model and scaler to models/ so strike_analyzer.py can load
them at runtime and replace the hard-coded deceleration thresholds.

Usage:
    python scripts/train_classifier.py

    # Point at a specific labeled file:
    python scripts/train_classifier.py output_data/labeled/my_session_labeled.csv

Output:
    models/strike_classifier.pkl   — trained Random Forest
    models/feature_scaler.pkl      — StandardScaler (for inference)
    models/training_report.txt     — accuracy, precision, recall, confusion matrix
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELED_DIR  = Path("output_data/labeled")
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODELS_DIR / "strike_classifier.pkl"
SCALER_PATH  = MODELS_DIR / "feature_scaler.pkl"
META_PATH    = MODELS_DIR / "model_meta.json"
REPORT_PATH  = MODELS_DIR / "training_report.txt"

# ── Features used for training ────────────────────────────────────────────────
# These must match exactly what strike_analyzer.py will pass at inference time.
FEATURE_COLS = [
    "decel_magnitude",
    "baseline_velocity_sw",
    "current_velocity_sw",
    "arm_extension_ratio",
    "hand_enc",          # Left=0, Right=1
    "is_ghost_frame",    # 0 or 1
]

LABEL_COL = "label"


# ── Data loading ──────────────────────────────────────────────────────────────

def find_labeled_csvs() -> list:
    if not LABELED_DIR.exists():
        return []
    return sorted(LABELED_DIR.glob("*_labeled.csv"))


def load_and_combine(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source"] = p.stem
        frames.append(df)
        print(f"  Loaded {len(df):>4} rows from {p.name}")
    return pd.concat(frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categorical
    df["hand_enc"]      = (df["hand"] == "Right").astype(int)
    df["is_ghost_frame"] = df["is_ghost_frame"].astype(int)

    # Derived features
    df["velocity_drop"]     = df["baseline_velocity_sw"] - df["current_velocity_sw"]
    df["velocity_ratio"]    = (df["current_velocity_sw"] /
                               df["baseline_velocity_sw"].replace(0, 1e-9))
    df["extension_x_decel"] = df["arm_extension_ratio"] * df["decel_magnitude"]

    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> dict:
    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Extend feature list with derived columns
    all_features = FEATURE_COLS + ["velocity_drop", "velocity_ratio", "extension_x_decel"]
    X = df[all_features].values
    y = df[LABEL_COL].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\n  Dataset: {len(y)} samples  |  {n_pos} real strikes  |  {n_neg} false positives")

    if len(y) < 10:
        raise ValueError(
            f"Only {len(y)} labeled samples — need at least 10 to train. "
            "Label more events in labeler.py first."
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Model: Random Forest ──────────────────────────────────────────────────
    # Best choice for small tabular data — no hyperparameter tuning needed,
    # naturally handles class imbalance via class_weight, gives feature importances.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # handles imbalanced real/false-positive ratio
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred      = clf.predict(X_test)
    y_prob      = clf.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
    report      = classification_report(y_test, y_pred,
                                        target_names=["False Positive", "Real Strike"])
    cm          = confusion_matrix(y_test, y_pred)

    # Cross-validation (more reliable on small datasets)
    if len(y) >= 20:
        cv_scores = cross_val_score(clf, X_scaled, y, cv=StratifiedKFold(5),
                                    scoring="f1", n_jobs=-1)
        cv_str = f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    else:
        cv_str = "n/a (< 20 samples)"

    # ── Feature importances ───────────────────────────────────────────────────
    importances = sorted(
        zip(all_features, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "feature_cols":  all_features,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "accuracy":      round(accuracy, 4),
        "roc_auc":       round(auc, 4) if not np.isnan(auc) else None,
        "cv_f1":         cv_str,
        "class_balance": {"real_strikes": int(n_pos), "false_positives": int(n_neg)},
        "importances":   {k: round(float(v), 4) for k, v in importances},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Write report ──────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Strike Classifier — Training Report",
        "=" * 60,
        f"  Samples       : {len(y)} ({n_pos} real, {n_neg} false positive)",
        f"  Train / Test  : {len(X_train)} / {len(X_test)}",
        f"  Accuracy      : {accuracy:.3f}",
        f"  ROC-AUC       : {auc:.3f}" if not np.isnan(auc) else "  ROC-AUC       : n/a",
        f"  CV F1 (5-fold): {cv_str}",
        "",
        "  Classification Report",
        "  " + "-" * 44,
    ]
    for line in report.splitlines():
        lines.append("  " + line)

    lines += [
        "",
        "  Confusion Matrix  (rows=actual, cols=predicted)",
        "               FP    Strike",
        f"  FP actual  [ {cm[0][0]:>4}  {cm[0][1]:>6} ]",
        f"  Strike act [ {cm[1][0]:>4}  {cm[1][1]:>6} ]",
        "",
        "  Feature Importances",
        "  " + "-" * 44,
    ]
    for feat, imp in importances:
        bar = "█" * int(imp * 40)
        lines.append(f"  {feat:<25} {imp:.4f}  {bar}")

    lines += ["", "=" * 60]
    report_text = "\n".join(lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    return {"report": report_text, "meta": meta}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[train_classifier] ── Strike Classifier Trainer ──────────────")

    # Accept an optional explicit path
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
        print(f"  Using specified file: {paths[0]}")
    else:
        paths = find_labeled_csvs()
        if not paths:
            print(
                "\n  No labeled CSV files found in output_data/labeled/\n"
                "  Run the labeler first:\n"
                "    streamlit run scripts/labeler.py\n"
            )
            sys.exit(1)
        print(f"  Found {len(paths)} labeled session(s):")

    df_raw = load_and_combine(paths)
    df     = engineer_features(df_raw)

    print("\n[train_classifier] Training …")
    result = train(df)

    print("\n" + result["report"])
    print(f"\n[train_classifier] Model  → {MODEL_PATH}")
    print(f"[train_classifier] Scaler → {SCALER_PATH}")
    print(f"[train_classifier] Report → {REPORT_PATH}")

"""
============================================================================
TEKNOFEST 2026 Sağlıkta YZ  —  SINGLE-FEATURE model comparison (panel-generic)
============================================================================

Measure the standalone predictive power of every individual feature in a panel.
For each non-ID feature we train a basic XGBoost classifier (n_estimators=100,
otherwise default — "nothing fancy") that uses ONLY that one feature to predict
Label (0 = benign, 1 = pathogenic), then compare accuracy / F1 / recall /
precision across features.

Numeric features (AL_, EK_) get one model per (transform x imputation) variant:
  transform   in {raw, significand, sig4figs}
  imputation  in {mean, median}
=> 6 models per numeric feature. Each variant is its own model.

Categorical features (CAT_, AA_) are ordinal-encoded (missing = own category),
one model each.

SCOPE: a single panel's CSV only — no cross-panel augmentation.
EVAL:  a single stratified 80/20 holdout split (random_state=42).

Run:
    python3 feature_depth/single_feature_eval.py --panel PAH
    python3 feature_depth/single_feature_eval.py --panel MASTER --data_dir "/path/to/EĞİTİM (TRAIN) SETLERİ"
Output (written to <repo>/<PANEL>_feature_in_depth/):
    <PANEL>_single_feature_results.csv   (sorted best F1 -> worst)

CAVEAT: with few benign rows in the test fold, per-feature metrics are noisy —
treat the ranking as relative, not absolute. The `significand` transform
deliberately discards magnitude, so it can break rank order.
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
import xgboost as xgb

SEED = 42
MISSING_CAT = "__MISSING__"


# ---------------------------------------------------------------------------
# Data location / loading / output folder
# ---------------------------------------------------------------------------
def locate_data_dir(panel, explicit=None):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(6):
        base = os.path.abspath(os.path.join(here, *([".."] * up)))
        hits = [h for h in glob.glob(os.path.join(base, "**", f"*{panel}*.csv"), recursive=True)
                if "universite-veri-seti" in h]
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError(f"Could not locate the data folder for {panel}; pass --data_dir.")


def load_panel(data_dir, panel):
    return pd.read_csv(glob.glob(os.path.join(data_dir, f"*{panel}*.csv"))[0])


def panel_out_dir(panel, data_dir, out_dir=None):
    """<out_dir or repo-root>/<PANEL>_feature_in_depth/ ; repo root = two levels
    up from the data dir (.../<root>/universite-veri-seti/EĞİTİM (TRAIN) SETLERİ)."""
    base = out_dir or os.path.dirname(os.path.dirname(os.path.normpath(data_dir)))
    folder = os.path.join(base, f"{panel}_feature_in_depth")
    os.makedirs(folder, exist_ok=True)
    return folder


def feature_columns(df):
    """Every column that is not the target and not an ID column."""
    return [c for c in df.columns if c != "Label" and not c.lower().startswith("variant")]


def is_categorical(col):
    return col.startswith("CAT_") or col.startswith("AA_")


# ---------------------------------------------------------------------------
# Value transforms (numeric only). NaN is preserved (imputation handles it).
# ---------------------------------------------------------------------------
def _significand(x):
    """Mantissa normalized to [1,10), sign kept, rounded to 4 dp.
    e.g. 0.000080095878575 -> 8.0096 ; 0 -> 0 ; NaN -> NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if x == 0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)
    exp = math.floor(math.log10(ax))
    return sign * round(ax / (10.0 ** exp), 4)


def _sig4(x):
    """Round to 4 significant figures, magnitude preserved.
    e.g. 0.000080095878575 -> 0.00008010 ; 0 -> 0 ; NaN -> NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")


def apply_transform(series, transform):
    s = pd.to_numeric(series, errors="coerce")
    if transform == "raw":
        return s
    if transform == "significand":
        return s.map(_significand)
    if transform == "sig4figs":
        return s.map(_sig4)
    raise ValueError(transform)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def new_xgb():
    # basic XGBoost, 100 trees, nothing fancy — all else default.
    return xgb.XGBClassifier(
        n_estimators=100, random_state=SEED, eval_metric="logloss", verbosity=0,
    )


def fit_predict(Xtr, ytr, Xte):
    m = new_xgb()
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def score(yte, pred):
    return dict(
        accuracy=accuracy_score(yte, pred),
        f1=f1_score(yte, pred, pos_label=1, zero_division=0),
        recall=recall_score(yte, pred, pos_label=1, zero_division=0),
        precision=precision_score(yte, pred, pos_label=1, zero_division=0),
    )


# ---------------------------------------------------------------------------
# Per-feature evaluation
# ---------------------------------------------------------------------------
def eval_numeric(df, col, tr_idx, te_idx, ytr, yte):
    rows = []
    na_frac = float(df[col].isna().mean())
    for transform in ("raw", "significand", "sig4figs"):
        col_t = apply_transform(df[col], transform)
        tr_vals, te_vals = col_t.iloc[tr_idx], col_t.iloc[te_idx]
        for imputation in ("mean", "median"):
            fill = tr_vals.mean() if imputation == "mean" else tr_vals.median()
            if pd.isna(fill):        # whole training column was NaN -> fall back to 0
                fill = 0.0
            Xtr = tr_vals.fillna(fill).to_numpy().reshape(-1, 1)
            Xte = te_vals.fillna(fill).to_numpy().reshape(-1, 1)
            pred = fit_predict(Xtr, ytr, Xte)
            rows.append(dict(feature=col, prefix=col.split("_")[0], kind="numeric",
                             transform=transform, imputation=imputation,
                             na_fraction=round(na_frac, 4), **score(yte, pred)))
    return rows


def eval_categorical(df, col, tr_idx, te_idx, ytr, yte):
    s = df[col].astype("object").where(df[col].notna(), MISSING_CAT).astype(str)
    tr_vals, te_vals = s.iloc[tr_idx], s.iloc[te_idx]
    cat_map = {v: i for i, v in enumerate(pd.unique(tr_vals))}   # fit on train only
    Xtr = tr_vals.map(cat_map).to_numpy().reshape(-1, 1)
    Xte = te_vals.map(cat_map).fillna(-1).to_numpy().reshape(-1, 1)  # unseen -> -1
    pred = fit_predict(Xtr, ytr, Xte)
    return [dict(feature=col, prefix=col.split("_")[0], kind="categorical",
                 transform="categorical", imputation="none",
                 na_fraction=round(float(df[col].isna().mean()), 4),
                 **score(yte, pred))]


# ---------------------------------------------------------------------------
def self_check():
    """Verify the significand / sig4figs examples from the spec."""
    assert _significand(0.000080095878575) == 8.0096, _significand(0.000080095878575)
    assert _sig4(0.000080095878575) == 8.01e-05, _sig4(0.000080095878575)


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True,
                    help="panel name, e.g. PAH / MASTER / KANSER / CFTR")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    panel = args.panel

    self_check()

    data_dir = locate_data_dir(panel, args.data_dir)
    out_dir = panel_out_dir(panel, data_dir, args.out_dir)
    df = load_panel(data_dir, panel)
    feat = feature_columns(df)
    y = df["Label"].astype(int).to_numpy()
    print(f"=== {panel} single-feature comparison ===")
    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(f"rows={len(df)}  features={len(feat)}  "
          f"pathogenic={int(y.sum())}  benign={int((y == 0).sum())}\n")

    # One stratified 80/20 split, reused for every model.
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    ytr, yte = y[tr_idx], y[te_idx]
    print(f"holdout split: train={len(tr_idx)}  test={len(te_idx)} "
          f"(test benign={int((yte == 0).sum())}, test pathogenic={int((yte == 1).sum())})\n")

    rows = []
    for i, col in enumerate(feat, 1):
        if is_categorical(col):
            rows += eval_categorical(df, col, tr_idx, te_idx, ytr, yte)
        else:
            rows += eval_numeric(df, col, tr_idx, te_idx, ytr, yte)
        if i % 50 == 0:
            print(f"  ...{i}/{len(feat)} features processed")

    res = pd.DataFrame(rows)
    # Sort best F1 -> worst; ties broken by recall then precision.
    res = res.sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    for c in ("accuracy", "f1", "recall", "precision"):
        res[c] = res[c].round(4)

    out_csv = os.path.join(out_dir, f"{panel}_single_feature_results.csv")
    res.to_csv(out_csv, index=False)

    print(f"\nTotal models trained: {len(res)}")
    print(f"Results (best F1 -> worst) written to: {out_csv}\n")

    print("===== TOP 20 MODELS BY F1 =====")
    cols = ["rank", "feature", "kind", "transform", "imputation",
            "accuracy", "f1", "recall", "precision", "na_fraction"]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(res[cols].head(20).to_string(index=False))

    # Best variant per feature -> which features carry signal.
    best_per_feat = (res.sort_values("f1", ascending=False)
                        .groupby("feature", as_index=False).first()
                        .sort_values("f1", ascending=False))
    print("\n===== TOP 20 FEATURES (each feature's best variant) =====")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(best_per_feat[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

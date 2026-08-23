"""
============================================================================
TEKNOFEST 2026 Sağlıkta YZ  —  MULTI-FEATURE model comparison (panel-generic)
============================================================================

Single-feature analysis (single_feature_eval.py) shows that, on their own, few
features beat the majority-class floor. This script asks the next question: does
COMBINING many features predict Label (0 = benign, 1 = pathogenic) better than
any single feature?

We pick feature subsets four different ways, sweep the subset size k, train one
basic XGBoost per (method, k), and compare accuracy / F1 / recall / precision.
The exact feature list chosen is recorded in every row of the output CSV.

Selection methods:
  single_f1       reuse <PANEL>_single_feature_results.csv (best single-feature
                  F1 per feature, descending). CONTINUITY with the single-feature
                  step. NOTE: that ranking was measured on the same test fold, so
                  this selector is mildly optimistic vs the train-only selectors.
  xgb_importance  fit ONE XGBoost on all features (train only), rank by gain.
  mutual_info     mutual_info_classif(Xtr, ytr) on the train matrix, descending.
  random          one random draw (seed 42) — a noisy baseline.

k sweep: 10, 25, 50, 100, 150, 200, 250, and all (=n_features). k=all is
identical for every method, so it is emitted once as `all_features`.
=> 4*7 + 1 = 29 models.

Preprocessing: raw values + MEDIAN imputation (median fit on TRAIN only);
categoricals (CAT_, AA_) ordinal-encoded (missing = own category, unseen -> -1).
Evaluation: the SAME stratified 80/20 holdout (random_state=42) as the
single-feature step.

Run:
    python3 feature_depth/multi_feature_eval.py --panel PAH
Output (written to <repo>/<PANEL>_feature_in_depth/):
    <PANEL>_multi_feature_results.csv   (sorted best F1 -> worst)
    <PANEL>_multi_feature_ksweep.png    (if matplotlib present)

CAVEAT: with few benign rows in the test fold, metrics are noisy — treat the
ranking as relative, not absolute.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

# Reuse the single-feature helpers so preprocessing stays identical.
from single_feature_eval import (MISSING_CAT, SEED, apply_transform,
                                  feature_columns, is_categorical, load_panel,
                                  locate_data_dir, new_xgb, panel_out_dir)

K_SWEEP = [10, 25, 50, 100, 150, 200, 250]
METHODS = ["single_f1", "xgb_importance", "mutual_info", "random"]


# ---------------------------------------------------------------------------
# Shared preprocessing: build the full train/test matrices ONCE (train-only fit).
# raw values + median impute (numeric); ordinal-encode (categorical).
# ---------------------------------------------------------------------------
def build_matrices(df, feat, tr_idx, te_idx):
    tr_parts, te_parts, names = [], [], []
    for c in feat:
        if is_categorical(c):
            s = df[c].astype("object").where(df[c].notna(), MISSING_CAT).astype(str)
            tr_v, te_v = s.iloc[tr_idx], s.iloc[te_idx]
            cat_map = {v: i for i, v in enumerate(pd.unique(tr_v))}   # fit on train
            tr_parts.append(tr_v.map(cat_map).to_numpy(dtype=float))
            te_parts.append(te_v.map(cat_map).fillna(-1).to_numpy(dtype=float))
        else:
            s = apply_transform(df[c], "raw")
            tr_v, te_v = s.iloc[tr_idx], s.iloc[te_idx]
            fill = tr_v.median()                                     # median on train
            if pd.isna(fill):
                fill = 0.0
            tr_parts.append(tr_v.fillna(fill).to_numpy(dtype=float))
            te_parts.append(te_v.fillna(fill).to_numpy(dtype=float))
        names.append(c)
    return np.column_stack(tr_parts), np.column_stack(te_parts), names


# ---------------------------------------------------------------------------
# Selection methods -> a feature ranking (best first). All fit on TRAIN ONLY,
# except single_f1 which reuses the single-feature CSV (documented caveat).
# ---------------------------------------------------------------------------
def rank_single_f1(out_dir, panel, names):
    path = os.path.join(out_dir, f"{panel}_single_feature_results.csv")
    r = pd.read_csv(path)
    best = (r.sort_values(["f1", "recall", "precision"], ascending=False)
              .groupby("feature", as_index=False).first()
              .sort_values(["f1", "recall", "precision"], ascending=False))
    ranked = [f for f in best["feature"].tolist() if f in set(names)]
    return _fill_rest(ranked, names)


def rank_xgb_importance(Xtr, ytr, names):
    m = new_xgb()
    m.fit(Xtr, ytr)
    # booster keys are f0,f1,... -> map back to column index.
    gain = m.get_booster().get_score(importance_type="gain")
    scores = {names[int(k[1:])]: v for k, v in gain.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    return _fill_rest(ranked, names)            # back-fill zero-gain features


def rank_mutual_info(Xtr, ytr, names):
    mi = mutual_info_classif(Xtr, ytr, random_state=SEED)
    order = np.argsort(mi)[::-1]
    return [names[i] for i in order]


def rank_random(names):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(names))
    return [names[i] for i in idx]


def _fill_rest(ranked, names):
    """Append any names not already in `ranked` (in original order) so we can
    always slice up to k=all."""
    seen = set(ranked)
    return ranked + [n for n in names if n not in seen]


# ---------------------------------------------------------------------------
def score(yte, pred):
    return dict(
        accuracy=accuracy_score(yte, pred),
        f1=f1_score(yte, pred, pos_label=1, zero_division=0),
        recall=recall_score(yte, pred, pos_label=1, zero_division=0),
        precision=precision_score(yte, pred, pos_label=1, zero_division=0),
    )


def run_model(Xtr, ytr, Xte, yte, names, chosen, method, k):
    cols = [names.index(c) for c in chosen]
    m = new_xgb()
    m.fit(Xtr[:, cols], ytr)
    pred = m.predict(Xte[:, cols])
    return dict(selection_method=method, k=k, n_features=len(chosen),
                selected_features=";".join(chosen), **score(yte, pred))


def maybe_plot(res, out_dir, panel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available — skipping k-sweep plot)")
        return
    sweep = res[res.selection_method != "all_features"]
    allf = res[res.selection_method == "all_features"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        d = sweep[sweep.selection_method == method].sort_values("k")
        ax.plot(d["k"], d["f1"], marker="o", label=method)
    if len(allf):
        ax.axhline(allf["f1"].iloc[0], ls="--", color="gray",
                   label=f"all features ({allf['n_features'].iloc[0]})")
    ax.set_xlabel("k (number of features)")
    ax.set_ylabel("F1 (pathogenic, test fold)")
    ax.set_title(f"{panel} multi-feature: F1 vs k by selection method")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(out_dir, f"{panel}_multi_feature_ksweep.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"k-sweep plot written to: {out}")


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

    data_dir = locate_data_dir(panel, args.data_dir)
    out_dir = panel_out_dir(panel, data_dir, args.out_dir)
    df = load_panel(data_dir, panel)
    feat = feature_columns(df)
    y = df["Label"].astype(int).to_numpy()

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    ytr, yte = y[tr_idx], y[te_idx]

    print(f"=== {panel} multi-feature comparison ===")
    print(f"out: {out_dir}")
    print(f"rows={len(df)}  features={len(feat)}  "
          f"train={len(tr_idx)}  test={len(te_idx)} "
          f"(test benign={int((yte == 0).sum())}, pathogenic={int((yte == 1).sum())})\n")

    Xtr, Xte, names = build_matrices(df, feat, tr_idx, te_idx)

    # Pre-compute each method's full ranking once; slice top-k per sweep step.
    rankings = {
        "single_f1": rank_single_f1(out_dir, panel, names),
        "xgb_importance": rank_xgb_importance(Xtr, ytr, names),
        "mutual_info": rank_mutual_info(Xtr, ytr, names),
        "random": rank_random(names),
    }

    rows = []
    for method in METHODS:
        ranked = rankings[method]
        for k in K_SWEEP:
            rows.append(run_model(Xtr, ytr, Xte, yte, names, ranked[:k], method, k))
    # k = all: identical for every method -> one row.
    rows.append(run_model(Xtr, ytr, Xte, yte, names, names, "all_features", len(names)))

    res = pd.DataFrame(rows)
    res = res.sort_values(["f1", "recall", "precision"], ascending=False).reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    for c in ("accuracy", "f1", "recall", "precision"):
        res[c] = res[c].round(4)

    out_csv = os.path.join(out_dir, f"{panel}_multi_feature_results.csv")
    res.to_csv(out_csv, index=False)

    print(f"Total models trained: {len(res)}")
    print(f"Results (best F1 -> worst) written to: {out_csv}\n")

    show = ["rank", "selection_method", "k", "n_features",
            "accuracy", "f1", "recall", "precision"]
    print("===== ALL MODELS (best F1 -> worst) =====")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(res[show].to_string(index=False))

    print("\n===== F1 BY (method x k) =====")
    sweep = res[res.selection_method != "all_features"]
    pivot = sweep.pivot_table(index="selection_method", columns="k", values="f1")
    pivot = pivot.reindex(METHODS)[K_SWEEP]
    with pd.option_context("display.width", 160):
        print(pivot.to_string())
    allrow = res[res.selection_method == "all_features"]
    print(f"\nall_features (k={allrow['n_features'].iloc[0]}): F1={allrow['f1'].iloc[0]}")
    best_single = pd.read_csv(os.path.join(out_dir, f"{panel}_single_feature_results.csv"))["f1"].max()
    print(f"Reference — best SINGLE-feature F1 = {best_single}")

    maybe_plot(res, out_dir, panel)


if __name__ == "__main__":
    main()

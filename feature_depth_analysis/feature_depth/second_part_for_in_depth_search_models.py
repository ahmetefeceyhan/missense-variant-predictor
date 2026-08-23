"""
============================================================================
TEKNOFEST 2026 Sağlıkta YZ  —  PART 4: tuned model search (panel-generic)
============================================================================

Parts 1-3 used a deliberately basic XGBoost ("nothing fancy") to measure feature
signal. This script takes each panel's BEST feature configuration and grid-searches
FIVE model families to see which predicts Label (0 = benign, 1 = pathogenic) best:

  XGBoost, LightGBM, CatBoost, NN (shallow MLP), DNN (deep MLP)

Per-panel "best feature set" (literal report best):
  MASTER : xgb_importance top-200
  KANSER : xgb_importance top-10
  PAH    : [AL_66]                     (single feature)
  CFTR   : [AL_6, AL_215, AL_21]       (tied best)

Preprocessing: raw values + MEDIAN impute, categoricals ordinal-encoded (the
Part 2 standard), via build_matrices(). NN/DNN additionally get StandardScaler.

Evaluation: the SAME stratified 80/20 holdout (seed 42) as Parts 1-2. GridSearchCV
runs StratifiedKFold on the TRAIN portion only (scoring=f1, refit=True); the refit
best estimator is scored on the held-out TEST fold -> accuracy/F1/recall/precision.

Output (NEW folder <repo>/<PANEL>_search_models/):
  <PANEL>_search_models_results.csv   (one row per model family, sorted test F1 desc)
  <PANEL>_search_models.png           (bar chart, if matplotlib present)

Run:
    python3 feature_depth/second_part_for_in_depth_search_models.py --panel MASTER
"""
import argparse
import json
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Reuse helpers from Parts 1-3.
from single_feature_eval import (SEED, feature_columns, load_panel,  # noqa
                                 locate_data_dir)
from multi_feature_eval import build_matrices, rank_xgb_importance

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Literal-best feature sets for the small panels (the larger panels are selected
# dynamically via xgb_importance below).
LITERAL_FEATURES = {
    "PAH": ["AL_66"],
    "CFTR": ["AL_6", "AL_215", "AL_21"],
}
XGB_TOPK = {"MASTER": 200, "KANSER": 10}


# ---------------------------------------------------------------------------
def select_features(panel, Xtr, ytr, names):
    """Return the chosen feature names + a short description for this panel."""
    if panel in LITERAL_FEATURES:
        chosen = [c for c in LITERAL_FEATURES[panel] if c in names]
        return chosen, f"literal report best ({len(chosen)} feat)"
    if panel in XGB_TOPK:
        k = XGB_TOPK[panel]
        chosen = rank_xgb_importance(Xtr, ytr, names)[:k]
        return chosen, f"xgb_importance top-{k}"
    # Fallback for any other panel: all features.
    return list(names), "all features"


def model_specs(neg, pos):
    """Return {family: (estimator, param_grid)}.  spw = scale_pos_weight ratio."""
    spw = float(neg) / float(pos) if pos else 1.0
    specs = {}

    specs["XGBoost"] = (
        xgb.XGBClassifier(n_estimators=200, random_state=SEED,
                          eval_metric="logloss", verbosity=0),
        {"max_depth": [3, 5], "learning_rate": [0.05, 0.1],
         "scale_pos_weight": [1, spw]},
    )
    specs["LightGBM"] = (
        lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbosity=-1),
        {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1],
         "scale_pos_weight": [1, spw]},
    )
    specs["CatBoost"] = (
        CatBoostClassifier(iterations=200, random_seed=SEED, verbose=0,
                           allow_writing_files=False),
        # "__DEFAULT__" => leave auto_class_weights unset (CatBoost can't parse None).
        {"depth": [4, 6], "learning_rate": [0.05, 0.1],
         "auto_class_weights": ["__DEFAULT__", "Balanced"]},
    )
    # NN / DNN: MLP has no class_weight -> left unweighted (documented limitation).
    specs["NN"] = (
        Pipeline([("scale", StandardScaler()),
                  ("mlp", MLPClassifier(max_iter=1000, random_state=SEED))]),
        {"mlp__hidden_layer_sizes": [(32,), (64,)], "mlp__alpha": [1e-4, 1e-3]},
    )
    specs["DNN"] = (
        Pipeline([("scale", StandardScaler()),
                  ("mlp", MLPClassifier(max_iter=1000, random_state=SEED))]),
        {"mlp__hidden_layer_sizes": [(64, 32), (128, 64, 32)],
         "mlp__alpha": [1e-4, 1e-3]},
    )
    return specs


def manual_grid_search(est, grid, X, y, skf):
    """Grid search by hand (avoids sklearn GridSearchCV's is_classifier()
    introspection, which is incompatible with CatBoost 1.2.7 on sklearn 1.6).
    Scores mean F1(pos_label=1) over the StratifiedKFold splits, refits the best
    params on all of (X, y). Returns (best_estimator, best_params, best_cv_f1)."""
    keys = list(grid)
    best = None
    for combo in product(*[grid[k] for k in keys]):
        # values equal to "__DEFAULT__" mean "leave the estimator's default".
        params = {k: v for k, v in zip(keys, combo) if v != "__DEFAULT__"}
        f1s = []
        for tri, vai in skf.split(X, y):
            m = clone(est).set_params(**params)
            m.fit(X[tri], y[tri])
            p = np.asarray(m.predict(X[vai])).astype(int).ravel()
            f1s.append(f1_score(y[vai], p, pos_label=1, zero_division=0))
        sc = float(np.mean(f1s))
        if best is None or sc > best[0]:
            best = (sc, params)
    best_score, best_params = best
    final = clone(est).set_params(**best_params)
    final.fit(X, y)
    return final, best_params, best_score


def out_folder(panel, data_dir, out_dir=None):
    base = out_dir or os.path.dirname(os.path.dirname(os.path.normpath(data_dir)))
    folder = os.path.join(base, f"{panel}_search_models")
    os.makedirs(folder, exist_ok=True)
    return folder


def maybe_plot(res, folder, panel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available — skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = res.sort_values("test_f1", ascending=False)
    bars = ax.bar(d["model_family"], d["test_f1"], color="#4C78A8")
    ax.set_ylabel("test F1 (pathogenic)")
    ax.set_title(f"{panel} — tuned model search: test F1 by model family")
    ax.set_ylim(0, 1)
    for b, v in zip(bars, d["test_f1"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(folder, f"{panel}_search_models.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"plot written to: {out}")


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
    folder = out_folder(panel, data_dir, args.out_dir)
    df = load_panel(data_dir, panel)
    feat = feature_columns(df)
    y = df["Label"].astype(int).to_numpy()

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    ytr, yte = y[tr_idx], y[te_idx]

    Xtr_all, Xte_all, names = build_matrices(df, feat, tr_idx, te_idx)
    chosen, fs_desc = select_features(panel, Xtr_all, ytr, names)
    cols = [names.index(c) for c in chosen]
    Xtr, Xte = Xtr_all[:, cols], Xte_all[:, cols]

    neg, pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    n_splits = 3 if neg < 25 else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    print(f"=== {panel} tuned model search ===")
    print(f"out: {folder}")
    print(f"feature set: {fs_desc}  (n={len(chosen)})")
    print(f"train={len(tr_idx)} (benign={neg}/path={pos})  test={len(te_idx)} "
          f"(benign={int((yte==0).sum())}/path={int((yte==1).sum())})  cv={n_splits}-fold\n")

    rows = []
    for fam, (est, grid) in model_specs(neg, pos).items():
        best_est, best_params, cv_f1 = manual_grid_search(est, grid, Xtr, ytr, cv)
        pred = np.asarray(best_est.predict(Xte)).astype(int).ravel()
        rows.append(dict(
            panel=panel, model_family=fam, n_features=len(chosen), feature_set=fs_desc,
            cv_best_f1=round(float(cv_f1), 4),
            test_accuracy=round(accuracy_score(yte, pred), 4),
            test_f1=round(f1_score(yte, pred, pos_label=1, zero_division=0), 4),
            test_recall=round(recall_score(yte, pred, pos_label=1, zero_division=0), 4),
            test_precision=round(precision_score(yte, pred, pos_label=1, zero_division=0), 4),
            best_params=json.dumps(best_params),
        ))
        print(f"  [{fam:9s}] cv_f1={rows[-1]['cv_best_f1']:.4f}  "
              f"test_f1={rows[-1]['test_f1']:.4f}  best_params={best_params}")

    res = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(folder, f"{panel}_search_models_results.csv")
    res.to_csv(out_csv, index=False)

    print(f"\nResults (best test F1 -> worst) written to: {out_csv}")
    show = ["model_family", "cv_best_f1", "test_f1", "test_accuracy",
            "test_recall", "test_precision"]
    with pd.option_context("display.width", 160):
        print(res[show].to_string(index=False))
    win = res.iloc[0]
    print(f"\nWINNER: {win.model_family}  test_f1={win.test_f1}  "
          f"(feature set: {fs_desc})")

    maybe_plot(res, folder, panel)


if __name__ == "__main__":
    main()

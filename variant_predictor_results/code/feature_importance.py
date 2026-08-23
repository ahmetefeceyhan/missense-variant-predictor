"""
=============================================================================
CELL 3  ·  Feature Importance Analysis
=============================================================================
Paste into a new Colab cell and run AFTER Cell 1 (main) has completed.
Cell 2 (download_results) does NOT need to have run first.

What it does
  1. Re-preprocesses the dataset for each k-mer size
  2. Re-fits XGBoost + LightGBM on the FULL dataset using the best
     hyperparameters saved in results.json
  3. Computes Neural-Network permutation importance on a held-out split
  4. Generates 7 figures:
       fig_fi_A  – XGBoost Top-30 gain importance (horizontal bar)
       fig_fi_B  – LightGBM Top-30 gain importance (horizontal bar)
       fig_fi_C  – NN Permutation importance Top-30 (AUC drop)
       fig_fi_D  – Consensus ranking: features ranked high across all models
       fig_fi_E  – Feature-group importance pie  (DNA k-mer / Prot k-mer /
                    Numeric / OHE / Label-enc / Bool)
       fig_fi_F  – XGBoost importance types: gain vs cover vs weight
       fig_fi_G  – Model agreement heatmap (top-50 consensus × 3 models)
  5. Exports feature_importance.csv
  6. Bundles everything into feature_importance_results.zip + downloads

Usage
  exec(open("feature_importance.py").read())
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import json, os, sys, csv, zipfile, warnings, time
from pathlib import Path
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Pull preprocessing + model helpers from main script ──────────────────────
# We import only the pure-Python pieces — no re-training of all experiments.
from pathogenic_benign_predictor import (
    build_feature_matrix,
    train_xgb, train_lgbm, train_nn, predict_nn,
    _gpu_scale, _clean,
    RANDOM_SEED, EARLY_STOP_ROUNDS,
    XGB_DEVICE_KWARGS, LGBM_DEVICE,
    KMER_SIZES,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "xgboost":        "#4C72B0",
    "lightgbm":       "#DD8452",
    "neural_network": "#55A868",
}
MODEL_LABELS = {
    "xgboost":        "XGBoost",
    "lightgbm":       "LightGBM",
    "neural_network": "Neural Net",
}

OUT_DIR = Path("figures_fi")
OUT_DIR.mkdir(exist_ok=True)

TOP_N = 30   # features to show in bar charts

# ── Load results ──────────────────────────────────────────────────────────────
with open("results.json") as f:
    results = json.load(f)

# ── Load dataset ──────────────────────────────────────────────────────────────
DATA_PATH = "open_cravat_curation_v3.csv"
df_raw = pd.read_csv(DATA_PATH)
sig = df_raw["clinvar__sig"].fillna("")
df_raw["label"] = sig.str.contains("Pathogenic", case=False).astype(int)
df_raw.loc[sig.str.contains("Benign", case=False), "label"] = 0

print(f"Dataset loaded: {df_raw.shape[0]} samples  "
      f"| positives={df_raw['label'].sum()}  "
      f"| negatives={(df_raw['label']==0).sum()}")


# ── Helper: best params from results.json ────────────────────────────────────
def get_best_params(results_list, model_name, k):
    """Return best_params for a model/k from the first fold of Exp1."""
    for run in results_list:
        if run["k"] != k:
            continue
        fold_list = run["exp1"]["models"][model_name].get("fold_metrics", [])
        if fold_list:
            p = fold_list[0].get("best_params", {})
            if p:
                return p
    return {}


# ── Helper: categorise a feature name ────────────────────────────────────────
def feature_group(name: str) -> str:
    if "2mer" in name or "3mer" in name:
        if name.startswith("DNA"):
            return "DNA k-mer"
        return "Protein k-mer"
    for prefix in ["base__ref_base", "base__alt_base", "ref_amino", "alt_amino"]:
        if name.startswith(prefix):
            return "OHE (base/AA)"
    for col in ["polarity_change", "chirality_shift"]:
        if name.startswith(col):
            return "Bool flag"
    for col in ["esm1b__", "metalr__", "metarnn__", "metasvm__",
                "mistic__", "mutationtaster__", "phdsnpg__",
                "provean__", "sift__", "alphamissense__am"]:
        if name.startswith(col):
            return "Label-enc pred"
    return "Numeric score"


# =============================================================================
# MAIN LOOP  — preprocess & fit for each k, collect importances
# =============================================================================
all_k_data = {}   # k → {xgb_imp, lgbm_imp, nn_imp, feature_names}

for k in KMER_SIZES:
    print(f"\n{'='*55}")
    print(f"  Processing k={k} …")
    print(f"{'='*55}")

    X_df, y_ser, panel_ser, registry = build_feature_matrix(df_raw.copy(), k)
    feat_names = X_df.columns.tolist()
    X_raw = X_df.values.astype(np.float64)
    y     = y_ser.values

    # 80/20 stratified split — train importances on 80%, validate on 20%
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    X_tr, X_te = _gpu_scale(_clean(X_tr_raw), _clean(X_te_raw))
    # Further split train → train/val for early stopping
    X_trn, X_val, y_trn, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=RANDOM_SEED
    )

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("  Fitting XGBoost …")
    xgb_params = get_best_params(results, "xgboost", k)
    if not xgb_params:
        xgb_params = {"n_estimators": 200, "max_depth": 6,
                      "learning_rate": 0.05, "subsample": 0.8,
                      "colsample_bytree": 0.8, "min_child_weight": 3}
    xgb_model = train_xgb(X_trn, y_trn, X_val, y_val, xgb_params)

    xgb_gain   = xgb_model.get_booster().get_score(importance_type="gain")
    xgb_cover  = xgb_model.get_booster().get_score(importance_type="cover")
    xgb_weight = xgb_model.get_booster().get_score(importance_type="weight")
    # Map f0, f1, … back to feature names
    def _map_scores(score_dict, names):
        arr = np.zeros(len(names))
        for fkey, val in score_dict.items():
            idx = int(fkey[1:]) if fkey.startswith("f") else None
            if idx is not None and idx < len(names):
                arr[idx] = val
        return arr

    xgb_gain_arr   = _map_scores(xgb_gain,   feat_names)
    xgb_cover_arr  = _map_scores(xgb_cover,  feat_names)
    xgb_weight_arr = _map_scores(xgb_weight, feat_names)

    # Normalise
    def _norm(arr):
        s = arr.sum()
        return arr / s if s > 0 else arr

    xgb_gain_arr   = _norm(xgb_gain_arr)
    xgb_cover_arr  = _norm(xgb_cover_arr)
    xgb_weight_arr = _norm(xgb_weight_arr)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("  Fitting LightGBM …")
    lgbm_params = get_best_params(results, "lightgbm", k)
    if not lgbm_params:
        lgbm_params = {"n_estimators": 200, "num_leaves": 63,
                       "learning_rate": 0.05, "min_child_samples": 20,
                       "subsample": 0.8, "colsample_bytree": 0.8}
    lgbm_model = train_lgbm(X_trn, y_trn, X_val, y_val, lgbm_params)
    lgbm_imp   = _norm(lgbm_model.feature_importances_.astype(float))

    # ── Neural Network — permutation importance ───────────────────────────────
    print("  Fitting Neural Net + computing permutation importance …")
    nn_params = get_best_params(results, "neural_network", k)
    if not nn_params:
        nn_params = {"h1": 2048, "h2": 1024, "h3": 512,
                     "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
                     "batch_size": 2048, "epochs": 10}
    nn_model = train_nn(X_trn, y_trn, X_val, y_val, nn_params)

    base_auc = roc_auc_score(y_te, predict_nn(nn_model, X_te))
    print(f"    Base AUC on test: {base_auc:.4f}")

    # Permutation: shuffle each feature N_REPEATS times, measure AUC drop
    N_REPEATS   = 3
    perm_imp    = np.zeros(len(feat_names))
    rng         = np.random.default_rng(RANDOM_SEED)
    # Only permute top-200 by XGB gain to keep runtime manageable
    candidate_idx = np.argsort(xgb_gain_arr)[::-1][:200]
    for idx in candidate_idx:
        drops = []
        for _ in range(N_REPEATS):
            X_perm        = X_te.copy()
            X_perm[:, idx] = rng.permutation(X_perm[:, idx])
            perm_auc      = roc_auc_score(y_te, predict_nn(nn_model, X_perm))
            drops.append(base_auc - perm_auc)
        perm_imp[idx] = max(0.0, float(np.mean(drops)))

    perm_imp = _norm(perm_imp)

    all_k_data[k] = {
        "feat_names":      feat_names,
        "xgb_gain":        xgb_gain_arr,
        "xgb_cover":       xgb_cover_arr,
        "xgb_weight":      xgb_weight_arr,
        "lgbm_imp":        lgbm_imp,
        "nn_perm":         perm_imp,
        "base_auc":        base_auc,
    }
    print(f"  k={k} done.")


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def _top_bar(ax, importances, feat_names, n, color, title, xlabel="Normalised Importance"):
    idx  = np.argsort(importances)[::-1][:n]
    vals = importances[idx]
    labs = [feat_names[i] for i in idx]
    # Shorten label: keep only the last 35 chars
    labs = [l[-35:] if len(l) > 35 else l for l in labs]
    y_pos = np.arange(len(vals))
    bars  = ax.barh(y_pos, vals, color=color, edgecolor="black", linewidth=0.4, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(axis="x", alpha=0.35)
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=7.5)
    ax.set_xlim(0, vals.max() * 1.25)


# =============================================================================
# FIGURE A – XGBoost Top-N Gain Importance  (one subplot per k)
# =============================================================================
def fig_fi_A():
    fig, axes = plt.subplots(1, len(KMER_SIZES),
                             figsize=(13 * len(KMER_SIZES), TOP_N * 0.38 + 1.5))
    if len(KMER_SIZES) == 1:
        axes = [axes]
    for ax, k in zip(axes, KMER_SIZES):
        d = all_k_data[k]
        _top_bar(ax, d["xgb_gain"], d["feat_names"], TOP_N,
                 PALETTE["xgboost"], f"k={k}  — XGBoost Gain Importance (Top {TOP_N})")
    fig.suptitle("XGBoost Feature Importance (Gain)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_A_xgb_gain.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# FIGURE B – LightGBM Top-N Importance
# =============================================================================
def fig_fi_B():
    fig, axes = plt.subplots(1, len(KMER_SIZES),
                             figsize=(13 * len(KMER_SIZES), TOP_N * 0.38 + 1.5))
    if len(KMER_SIZES) == 1:
        axes = [axes]
    for ax, k in zip(axes, KMER_SIZES):
        d = all_k_data[k]
        _top_bar(ax, d["lgbm_imp"], d["feat_names"], TOP_N,
                 PALETTE["lightgbm"], f"k={k}  — LightGBM Importance (Top {TOP_N})")
    fig.suptitle("LightGBM Feature Importance", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_B_lgbm_imp.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# FIGURE C – Neural Network Permutation Importance
# =============================================================================
def fig_fi_C():
    fig, axes = plt.subplots(1, len(KMER_SIZES),
                             figsize=(13 * len(KMER_SIZES), TOP_N * 0.38 + 1.5))
    if len(KMER_SIZES) == 1:
        axes = [axes]
    for ax, k in zip(axes, KMER_SIZES):
        d = all_k_data[k]
        _top_bar(ax, d["nn_perm"], d["feat_names"], TOP_N,
                 PALETTE["neural_network"],
                 f"k={k}  — NN Permutation Importance (Top {TOP_N})",
                 xlabel="Mean AUC Drop (normalised)")
        ax.set_title(
            f"k={k}  — NN Permutation Importance\n"
            f"(base AUC={d['base_auc']:.4f})",
            fontweight="bold", fontsize=11,
        )
    fig.suptitle("Neural Network Permutation Feature Importance",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_C_nn_perm.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# FIGURE D – Consensus Ranking  (average rank across XGB + LGBM + NN)
# =============================================================================
def fig_fi_D():
    fig, axes = plt.subplots(1, len(KMER_SIZES),
                             figsize=(13 * len(KMER_SIZES), TOP_N * 0.38 + 1.5))
    if len(KMER_SIZES) == 1:
        axes = [axes]
    for ax, k in zip(axes, KMER_SIZES):
        d = all_k_data[k]
        n = len(d["feat_names"])
        # Rank each model (1 = most important)
        def _rank(arr):
            order = np.argsort(arr)[::-1]
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            return ranks

        r_xgb  = _rank(d["xgb_gain"])
        r_lgbm = _rank(d["lgbm_imp"])
        r_nn   = _rank(d["nn_perm"])
        avg_rank = (r_xgb + r_lgbm + r_nn) / 3.0

        # Best consensus = lowest average rank
        top_idx  = np.argsort(avg_rank)[:TOP_N]
        avg_vals = avg_rank[top_idx]
        labs     = [d["feat_names"][i] for i in top_idx]
        labs     = [l[-35:] if len(l) > 35 else l for l in labs]

        # Colour by feature group
        group_colors = {
            "DNA k-mer":       "#2196F3",
            "Protein k-mer":   "#9C27B0",
            "OHE (base/AA)":   "#FF9800",
            "Numeric score":   "#4CAF50",
            "Label-enc pred":  "#F44336",
            "Bool flag":       "#9E9E9E",
        }
        bar_colors = [group_colors.get(feature_group(d["feat_names"][i]), "#888888")
                      for i in top_idx]

        y_pos = np.arange(TOP_N)
        ax.barh(y_pos, TOP_N + 1 - avg_vals, color=bar_colors,   # invert so longer = better
                edgecolor="black", linewidth=0.4, alpha=0.88)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labs, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Consensus Score  (higher = more important)", fontsize=10)
        ax.set_title(f"k={k}  — Consensus Top {TOP_N} (XGB+LGBM+NN avg rank)",
                     fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.35)

        # Legend for groups
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=c, label=g, edgecolor="black")
                      for g, c in group_colors.items()]
        ax.legend(handles=legend_els, fontsize=8, loc="lower right",
                  framealpha=0.85, title="Feature group")

    fig.suptitle("Consensus Feature Importance (Average Rank Across All Models)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_D_consensus.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# FIGURE E – Feature-Group Importance Pie  (XGB gain + LGBM + NN)
# =============================================================================
def fig_fi_E():
    fig, axes = plt.subplots(len(KMER_SIZES), 3,
                             figsize=(14, 5.5 * len(KMER_SIZES)))
    if len(KMER_SIZES) == 1:
        axes = axes.reshape(1, 3)

    group_order = ["DNA k-mer", "Protein k-mer", "OHE (base/AA)",
                   "Numeric score", "Label-enc pred", "Bool flag"]
    group_palette = ["#2196F3", "#9C27B0", "#FF9800", "#4CAF50", "#F44336", "#9E9E9E"]

    for ki, k in enumerate(KMER_SIZES):
        d        = all_k_data[k]
        names    = d["feat_names"]
        groups   = [feature_group(n) for n in names]

        for col_idx, (model_key, imp_arr) in enumerate([
            ("XGBoost (Gain)",   d["xgb_gain"]),
            ("LightGBM",         d["lgbm_imp"]),
            ("NN Permutation",   d["nn_perm"]),
        ]):
            ax     = axes[ki][col_idx]
            totals = {g: 0.0 for g in group_order}
            for g, v in zip(groups, imp_arr):
                if g in totals:
                    totals[g] += v

            sizes  = [totals[g] for g in group_order]
            labels = [f"{g}\n({totals[g]*100:.1f}%)" if totals[g] > 0.005 else ""
                      for g in group_order]
            wedge_colors = group_palette

            wedges, _ = ax.pie(
                sizes, labels=labels, colors=wedge_colors,
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.2),
                textprops=dict(fontsize=8),
            )
            ax.set_title(f"k={k}  {model_key}", fontweight="bold", fontsize=10)

    fig.suptitle("Feature-Group Contribution to Importance",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_E_group_pie.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# FIGURE F – XGBoost: Gain vs Cover vs Weight comparison (top-20, best k)
# =============================================================================
def fig_fi_F():
    k   = max(KMER_SIZES, key=lambda kv: all_k_data[kv]["xgb_gain"].max())
    d   = all_k_data[k]
    n   = len(d["feat_names"])
    N   = 20   # top features

    # Union top-N from all three types
    types = {"Gain": d["xgb_gain"], "Cover": d["xgb_cover"], "Weight": d["xgb_weight"]}
    union_idx = set()
    for arr in types.values():
        union_idx.update(np.argsort(arr)[::-1][:N].tolist())
    union_idx = sorted(union_idx, key=lambda i: d["xgb_gain"][i], reverse=True)[:N]

    labs  = [d["feat_names"][i][-32:] for i in union_idx]
    x_pos = np.arange(len(union_idx))
    width = 0.28
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for ci, (typ, arr) in enumerate(types.items()):
        vals = [arr[i] for i in union_idx]
        ax.bar(x_pos + ci * width, vals, width, label=typ,
               color=colors[ci], alpha=0.88, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Normalised Importance")
    ax.set_title(f"XGBoost Importance Types: Gain vs Cover vs Weight  (k={k}, Top {N})",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_F_xgb_types.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# FIGURE G – Model Agreement Heatmap  (top-50 consensus × 3 models)
# =============================================================================
def fig_fi_G():
    fig, axes = plt.subplots(1, len(KMER_SIZES),
                             figsize=(14 * len(KMER_SIZES), 14))
    if len(KMER_SIZES) == 1:
        axes = [axes]

    TOP_CONSENSUS = 50

    for ax, k in zip(axes, KMER_SIZES):
        d = all_k_data[k]
        n = len(d["feat_names"])

        def _rank(arr):
            order = np.argsort(arr)[::-1]
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            return ranks

        r_xgb  = _rank(d["xgb_gain"])
        r_lgbm = _rank(d["lgbm_imp"])
        r_nn   = _rank(d["nn_perm"])
        avg_r  = (r_xgb + r_lgbm + r_nn) / 3.0
        top_idx = np.argsort(avg_r)[:TOP_CONSENSUS]

        matrix = np.column_stack([
            d["xgb_gain"][top_idx],
            d["lgbm_imp"][top_idx],
            d["nn_perm"][top_idx],
        ])
        # Normalise each column independently for visual contrast
        matrix = (matrix - matrix.min(0)) / (matrix.max(0) - matrix.min(0) + 1e-12)

        labs = [d["feat_names"][i][-40:] for i in top_idx]
        sns.heatmap(
            matrix, ax=ax,
            xticklabels=["XGBoost\n(Gain)", "LightGBM", "NN Perm."],
            yticklabels=labs,
            cmap="YlOrRd", linewidths=0.2, linecolor="white",
            cbar_kws={"label": "Normalised Importance"},
            annot=False,
        )
        ax.set_title(f"k={k}  — Top-{TOP_CONSENSUS} Consensus Features × Model Agreement",
                     fontweight="bold", fontsize=11)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=10)

    fig.suptitle("Feature × Model Agreement Heatmap (Consensus Top Features)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = OUT_DIR / "fig_fi_G_agreement_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {p}")


# =============================================================================
# CSV EXPORT
# =============================================================================
def export_fi_csv():
    rows = []
    header = ["k", "feature", "feature_group",
              "xgb_gain", "xgb_cover", "xgb_weight",
              "lgbm_importance", "nn_permutation_importance",
              "consensus_avg_rank"]
    for k, d in all_k_data.items():
        n = len(d["feat_names"])
        def _rank(arr):
            order = np.argsort(arr)[::-1]
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            return ranks
        avg_rank = (_rank(d["xgb_gain"]) + _rank(d["lgbm_imp"]) + _rank(d["nn_perm"])) / 3.0
        for i, feat in enumerate(d["feat_names"]):
            rows.append([
                k, feat, feature_group(feat),
                round(float(d["xgb_gain"][i]),  6),
                round(float(d["xgb_cover"][i]), 6),
                round(float(d["xgb_weight"][i]),6),
                round(float(d["lgbm_imp"][i]),  6),
                round(float(d["nn_perm"][i]),   6),
                round(float(avg_rank[i]),        2),
            ])
    # Sort by avg_rank
    rows.sort(key=lambda r: r[-1])
    csv_path = Path("feature_importance.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  ✓  {csv_path}  ({len(rows)} feature rows)")
    return csv_path


# =============================================================================
# RUN EVERYTHING
# =============================================================================
print("\nGenerating feature importance figures …")
fig_fi_A()
fig_fi_B()
fig_fi_C()
fig_fi_D()
fig_fi_E()
fig_fi_F()
fig_fi_G()
print("\nExporting CSV …")
csv_path = export_fi_csv()
print("\nAll done.\n")


# =============================================================================
# PACKAGE & DOWNLOAD
# =============================================================================
ZIP_NAME = "feature_importance_results.zip"
artefacts = list(OUT_DIR.glob("*.png")) + [csv_path]
with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in artefacts:
        if p.exists():
            zf.write(p, p.name)
            print(f"  + {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

print(f"\nZip created: {ZIP_NAME}  "
      f"({Path(ZIP_NAME).stat().st_size / 1024 / 1024:.2f} MB)")

try:
    from google.colab import files
    print("\nStarting download …")
    files.download(ZIP_NAME)
    print("✓  Download triggered.")
except ImportError:
    print(f"\nNot on Colab — zip is at: {Path(ZIP_NAME).resolve()}")

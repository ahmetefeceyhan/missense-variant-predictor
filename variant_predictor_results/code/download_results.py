"""
=============================================================================
POST-RUN  ·  Visualise Results & Download Everything
=============================================================================
Paste this into a fresh Colab cell and run it AFTER main() has completed.

What it does
  1. Reads results.json + feature_registry.json produced by main()
  2. Generates six publication-ready figures
  3. Bundles all artefacts (JSONs, figures, saved models) into a zip
  4. Triggers a browser download straight to your desktop

Usage
  # Cell 1 – run the predictor
  from pathogenic_benign_predictor import main
  main(data_path="open_cravat_curation_v3.csv", kmer_sizes=[2, 3])

  # Cell 2 – visualise & download
  exec(open("download_results.py").read())
  # — or just paste this file's contents directly —
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import json, os, zipfile, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless – works on Colab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
PALETTE   = {"xgboost": "#4C72B0", "lightgbm": "#DD8452", "neural_network": "#55A868"}
MODEL_LABELS = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "neural_network": "Neural Net"}
KMER_STYLES  = {2: "-o", 3: "--s"}

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
with open("results.json") as f:
    results = json.load(f)

with open("feature_registry.json") as f:
    registry = json.load(f)

MODELS = ["xgboost", "lightgbm", "neural_network"]
KMERS  = [r["k"] for r in results]
PANELS = ["General", "PAH", "Hereditary_Cancer", "CFTR"]


# =============================================================================
# FIGURE 1 – Experiment 1 · All-Panels CV  (AUROC + F1, k=2 vs k=3)
# =============================================================================
def fig_exp1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    metrics   = ["auroc", "f1"]
    titles    = ["AUROC  (↑ better)", "F1 Score  (↑ better)"]

    for ax, metric, title in zip(axes, metrics, titles):
        n_models = len(MODELS)
        x        = np.arange(n_models)
        width    = 0.35
        offsets  = [-width/2, width/2]

        for ki, (run, offset) in enumerate(zip(results, offsets)):
            means = [run["exp1"]["models"][m]["aggregated"][f"{metric}_mean"] for m in MODELS]
            stds  = [run["exp1"]["models"][m]["aggregated"][f"{metric}_std"]  for m in MODELS]
            bars  = ax.bar(
                x + offset, means, width,
                yerr=stds, capsize=4,
                label=f"k={run['k']}",
                color=[PALETTE[m] for m in MODELS],
                alpha=0.85 if ki == 0 else 0.55,
                edgecolor="black", linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=12)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylim(0.4, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(title="k-mer", fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Experiment 1 — All Panels · 5-Fold Stratified CV",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig1_exp1_all_panels.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 2 – Experiment 2 · Leave-One-Panel-Out heatmaps  (one per k-mer)
# =============================================================================
def fig_exp2():
    fig, axes = plt.subplots(1, len(KMERS), figsize=(7 * len(KMERS), 5))
    if len(KMERS) == 1:
        axes = [axes]

    for ax, run in zip(axes, results):
        matrix = np.full((len(PANELS), len(MODELS)), np.nan)
        for pi, panel in enumerate(PANELS):
            if panel in run["exp2"]["panels"]:
                for mi, model in enumerate(MODELS):
                    matrix[pi, mi] = run["exp2"]["panels"][panel]["models"][model]["auroc"]

        sns.heatmap(
            matrix, ax=ax,
            xticklabels=[MODEL_LABELS[m] for m in MODELS],
            yticklabels=PANELS,
            annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.5, vmax=1.0,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "AUROC"},
        )
        ax.set_title(f"k={run['k']}  — AUROC", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Experiment 2 — Leave-One-Panel-Out · Test AUROC",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig2_exp2_leave_one_out.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 3 – Experiment 3 · Single-Panel Generalisation  (per model heatmap)
# =============================================================================
def fig_exp3():
    for run in results:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5))

        for ax, model in zip(axes, MODELS):
            matrix  = np.full((len(PANELS), len(PANELS)), np.nan)
            tp_data = run["exp3"]["train_panels"]

            for ti, tr_panel in enumerate(PANELS):
                if tr_panel not in tp_data:
                    continue
                test_map = tp_data[tr_panel]["test_panels"].get(model, {})
                for ei, te_panel in enumerate(PANELS):
                    if te_panel == tr_panel:
                        matrix[ti, ei] = np.nan   # diagonal = not applicable
                    elif te_panel in test_map:
                        matrix[ti, ei] = test_map[te_panel]["auroc"]

            mask = np.isnan(matrix)
            sns.heatmap(
                matrix, ax=ax, mask=mask,
                xticklabels=PANELS, yticklabels=PANELS,
                annot=True, fmt=".3f", cmap="coolwarm",
                vmin=0.4, vmax=1.0,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "AUROC"},
            )
            ax.set_title(MODEL_LABELS[model], fontweight="bold")
            ax.set_xlabel("Test panel")
            ax.set_ylabel("Train panel")
            ax.tick_params(axis="x", rotation=30)
            ax.tick_params(axis="y", rotation=0)

        fig.suptitle(
            f"Experiment 3 — Single-Panel Generalisation · AUROC  (k={run['k']})",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        path = OUT_DIR / f"fig3_exp3_generalisation_k{run['k']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {path}")


# =============================================================================
# FIGURE 4 – k=2 vs k=3 Line comparison across ALL experiments & models
# =============================================================================
def fig_kmer_comparison():
    if len(KMERS) < 2:
        return   # nothing to compare with a single k

    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5), sharey=True)

    for ax, model in zip(axes, MODELS):
        exp_labels, exp_vals_k2, exp_vals_k3 = [], [], []

        for run in results:
            k = run["k"]
            # Exp 1 mean AUROC
            v1 = run["exp1"]["models"][model]["aggregated"]["auroc_mean"]
            # Exp 2 mean across all panels
            v2_vals = [
                run["exp2"]["panels"][p]["models"][model]["auroc"]
                for p in PANELS if p in run["exp2"]["panels"]
            ]
            v2 = float(np.mean(v2_vals)) if v2_vals else np.nan
            # Exp 3 overall mean
            v3_vals = []
            for tr_p, tr_data in run["exp3"]["train_panels"].items():
                for te_p, m_data in tr_data["test_panels"].get(model, {}).items():
                    v3_vals.append(m_data["auroc"])
            v3 = float(np.mean(v3_vals)) if v3_vals else np.nan

            if k == 2:
                exp_vals_k2 = [v1, v2, v3]
            else:
                exp_vals_k3 = [v1, v2, v3]

        exp_labels = ["Exp1\n(All CV)", "Exp2\n(LOPO)", "Exp3\n(Generalise)"]
        x = np.arange(len(exp_labels))

        ax.plot(x, exp_vals_k2, KMER_STYLES[2], color=PALETTE[model],
                lw=2.2, ms=9, label="k=2")
        ax.plot(x, exp_vals_k3, KMER_STYLES[3], color=PALETTE[model],
                lw=2.2, ms=9, alpha=0.65, label="k=3", ls="--")
        ax.fill_between(x, exp_vals_k2, exp_vals_k3,
                        color=PALETTE[model], alpha=0.12)

        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("Mean AUROC")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.35)

    fig.suptitle("k-mer 2 vs 3 — Mean AUROC Across All Experiments",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig4_kmer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 5 – Precision / Recall / F1 radar chart  (Exp 1, best k per model)
# =============================================================================
def fig_radar():
    from matplotlib.patches import FancyArrowPatch

    metric_keys  = ["accuracy", "auroc", "auprc", "f1", "precision", "recall"]
    metric_names = ["Accuracy", "AUROC", "AUPRC", "F1", "Precision", "Recall"]
    N = len(metric_keys)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Pick best k per model by Exp1 AUROC
    def best_run_for_model(model):
        return max(results, key=lambda r: r["exp1"]["models"][model]["aggregated"]["auroc_mean"])

    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5),
                             subplot_kw=dict(polar=True))

    for ax, model in zip(axes, MODELS):
        run    = best_run_for_model(model)
        agg    = run["exp1"]["models"][model]["aggregated"]
        values = [agg[f"{mk}_mean"] for mk in metric_keys]
        values += values[:1]

        ax.plot(angles, values, color=PALETTE[model], lw=2.5)
        ax.fill(angles, values, color=PALETTE[model], alpha=0.22)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=8, color="grey")
        ax.set_title(f"{MODEL_LABELS[model]}\n(k={run['k']})",
                     fontweight="bold", pad=18)
        ax.grid(color="grey", alpha=0.3)

    fig.suptitle("Experiment 1 — Metric Radar (Best k-mer per Model)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = OUT_DIR / "fig5_radar_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 6 – Feature registry summary bar chart
# =============================================================================
def fig_feature_registry():
    fig, axes = plt.subplots(1, len(KMERS), figsize=(8 * len(KMERS), 5))
    if len(KMERS) == 1:
        axes = [axes]

    for ax, run in zip(axes, results):
        k   = run["k"]
        reg = registry["kmer_experiments"][str(k)]

        categories = {
            "Numeric scores":     len(reg.get("numeric_features", [])),
            "OHE features":       len(reg.get("ohe_features", [])),
            "Label-encoded":      len(registry.get("label_enc_policy", [])),
            "Bool features":      len(registry.get("bool_cols", [])),
            "DNA k-mer":          sum(len(v) for v in reg.get("kmer_vocabs", {}).values()
                                      if any(c in k_name for k_name in reg.get("kmer_vocabs", {})
                                             for c in ["DNA"])),
            "Protein k-mer":      reg.get("n_features_total", 0)
                                  - len(reg.get("numeric_features", []))
                                  - len(reg.get("ohe_features", []))
                                  - len(registry.get("label_enc_policy", []))
                                  - len(registry.get("bool_cols", [])),
        }
        # simpler approximation using total
        categories = {
            "Numeric scores":     len(reg.get("numeric_features", [])),
            "OHE (bases/AAs)":    len(reg.get("ohe_features", [])),
            "Pred. label-enc.":   len(registry.get("label_enc_policy", [])),
            "Bool flags":         len(registry.get("bool_cols", [])),
            "DNA k-mers":         sum(len(v) for key, v in reg.get("kmer_vocabs", {}).items()
                                      if "DNA" in key),
            "Protein k-mers":     sum(len(v) for key, v in reg.get("kmer_vocabs", {}).items()
                                      if "Prot" in key),
        }

        labels = list(categories.keys())
        sizes  = list(categories.values())
        colors = sns.color_palette("pastel", len(labels))
        bars   = ax.barh(labels, sizes, color=colors, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, padding=4, fontsize=11)
        ax.set_xlabel("Number of features")
        ax.set_title(f"k={k}  —  {reg.get('n_features_total', sum(sizes))} total features",
                     fontweight="bold")
        ax.set_xlim(0, max(sizes) * 1.2)
        ax.grid(axis="x", alpha=0.35)

    fig.suptitle("Feature Composition After Preprocessing",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig6_feature_registry.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 7 – CV Fold Stability  (Exp 1 · AUROC per fold · box + strip)
# =============================================================================
def fig_fold_stability():
    """Show fold-by-fold AUROC spread for every model & k-mer — reveals variance."""
    import pandas as pd

    rows = []
    for run in results:
        for model in MODELS:
            fold_list = run["exp1"]["models"][model].get("fold_metrics", [])
            for fm in fold_list:
                rows.append({
                    "k":     f"k={run['k']}",
                    "model": MODEL_LABELS[model],
                    "auroc": fm.get("auroc", np.nan),
                    "f1":    fm.get("f1",    np.nan),
                    "fold":  fm.get("fold",  0),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["auroc", "f1"]):
        sns.boxplot(
            data=df, x="model", y=metric, hue="k",
            ax=ax, palette="muted", width=0.45,
            linewidth=1.2, fliersize=3,
        )
        sns.stripplot(
            data=df, x="model", y=metric, hue="k",
            ax=ax, dodge=True, size=5, alpha=0.55,
            palette="dark:black", legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Fold-by-Fold {metric.upper()} Distribution", fontweight="bold")
        ax.set_ylim(max(0.0, df[metric].min() - 0.08), min(1.05, df[metric].max() + 0.05))
        ax.grid(axis="y", alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(KMERS)], labels[:len(KMERS)], title="k-mer", fontsize=10)

    fig.suptitle("Experiment 1 — 5-Fold CV Stability (AUROC & F1)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig7_fold_stability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 8 – All-Metrics Summary Table  (best k per model, Exp 1)
# =============================================================================
def fig_summary_table():
    """Publication-ready table: all 6 metrics × 3 models, best k per model."""
    metric_keys  = ["auroc", "auprc", "accuracy", "f1", "precision", "recall"]
    metric_names = ["AUROC", "AUPRC", "Accuracy", "F1", "Precision", "Recall"]

    col_labels = [MODEL_LABELS[m] for m in MODELS]
    row_labels = metric_names

    def best_run(model):
        return max(results, key=lambda r: r["exp1"]["models"][model]["aggregated"]["auroc_mean"])

    # Build cell text and colour arrays
    cell_text   = []
    cell_colors = []
    cmap = plt.cm.RdYlGn

    for mk in metric_keys:
        row_txt, row_col = [], []
        for model in MODELS:
            run  = best_run(model)
            agg  = run["exp1"]["models"][model]["aggregated"]
            mean = agg.get(f"{mk}_mean", np.nan)
            std  = agg.get(f"{mk}_std",  np.nan)
            row_txt.append(f"{mean:.3f} ± {std:.3f}")
            row_col.append(cmap(float(mean)) if not np.isnan(mean) else (0.9, 0.9, 0.9, 1))
        cell_text.append(row_txt)
        cell_colors.append(row_col)

    # Add best-k row
    row_txt, row_col = [], []
    for model in MODELS:
        best_k = best_run(model)["k"]
        row_txt.append(f"k = {best_k}")
        row_col.append((0.85, 0.85, 1.0, 1))
    row_labels.append("Best k-mer")
    cell_text.append(row_txt)
    cell_colors.append(row_col)

    fig, ax = plt.subplots(figsize=(10, len(row_labels) * 0.65 + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center", rowLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.35, 1.8)

    # Bold header row
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#cccccc")

    fig.suptitle("Experiment 1 — All-Metrics Summary (mean ± std, 5-Fold CV)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = OUT_DIR / "fig8_metrics_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 9 – Generalisation Gap  (Exp1 AUROC − Exp3 mean AUROC)
# =============================================================================
def fig_generalisation_gap():
    """Quantify performance drop when a model is asked to generalise cross-panel."""
    fig, axes = plt.subplots(1, len(KMERS), figsize=(7 * len(KMERS), 5), sharey=True)
    if len(KMERS) == 1:
        axes = [axes]

    for ax, run in zip(axes, results):
        in_panel, out_panel, model_names_plot = [], [], []
        for model in MODELS:
            in_auroc = run["exp1"]["models"][model]["aggregated"]["auroc_mean"]
            # Exp3 out-of-panel mean
            vals = [
                run["exp3"]["train_panels"][tr]["test_panels"].get(model, {}).get(te, {}).get("auroc", np.nan)
                for tr in run["exp3"]["train_panels"]
                for te in run["exp3"]["train_panels"][tr]["test_panels"].get(model, {})
            ]
            out_auroc = float(np.nanmean(vals)) if vals else np.nan

            in_panel.append(in_auroc)
            out_panel.append(out_auroc if not np.isnan(out_auroc) else 0)
            model_names_plot.append(MODEL_LABELS[model])

        x      = np.arange(len(MODELS))
        width  = 0.35
        b_in   = ax.bar(x - width/2, in_panel,  width, label="In-panel (Exp1)",
                        color=[PALETTE[m] for m in MODELS], alpha=0.9, edgecolor="black", lw=0.6)
        b_out  = ax.bar(x + width/2, out_panel, width, label="Out-of-panel (Exp3)",
                        color=[PALETTE[m] for m in MODELS], alpha=0.45, edgecolor="black", lw=0.6,
                        hatch="//")

        # Gap annotation
        for xi, (ip, op) in enumerate(zip(in_panel, out_panel)):
            gap = ip - op
            if gap > 0.005:
                ax.annotate(f"Δ{gap:.2f}", xy=(xi, max(ip, op) + 0.01),
                            ha="center", va="bottom", fontsize=9.5,
                            color="crimson", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names_plot, fontsize=12)
        ax.set_ylim(0.4, 1.12)
        ax.set_ylabel("AUROC")
        ax.set_title(f"k={run['k']}", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Generalisation Gap — In-Panel vs Cross-Panel AUROC (Δ = drop)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig9_generalisation_gap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 10 – Panel Difficulty Profile  (Exp2 · multi-metric per panel)
# =============================================================================
def fig_panel_profile():
    """For each test panel in Exp2, show AUROC / F1 / Precision / Recall per model."""
    metric_keys   = ["auroc", "f1", "precision", "recall"]
    metric_labels = ["AUROC", "F1", "Precision", "Recall"]

    # Use best k only
    def best_run_global():
        return max(results, key=lambda r: np.nanmean([
            r["exp1"]["models"][m]["aggregated"]["auroc_mean"] for m in MODELS
        ]))

    run     = best_run_global()
    panels2 = [p for p in PANELS if p in run["exp2"]["panels"]]
    if not panels2:
        return

    fig, axes = plt.subplots(len(panels2), 1,
                             figsize=(11, 4.2 * len(panels2)), squeeze=False)

    for ax_row, panel in zip(axes, panels2):
        ax   = ax_row[0]
        x    = np.arange(len(metric_keys))
        w    = 0.22
        offsets = np.linspace(-(len(MODELS)-1)*w/2, (len(MODELS)-1)*w/2, len(MODELS))

        for model, offset in zip(MODELS, offsets):
            m_data = run["exp2"]["panels"][panel]["models"].get(model, {})
            vals   = [m_data.get(mk, np.nan) for mk in metric_keys]
            ax.bar(x + offset, vals, w,
                   label=MODEL_LABELS[model],
                   color=PALETTE[model], alpha=0.85,
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(f"Test Panel: {panel}  (k={run['k']})", fontweight="bold")
        ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5, label="Random (0.5)")
        ax.legend(fontsize=9, ncol=4, loc="lower right")
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Experiment 2 — Per-Panel Performance Profile (Best k-mer)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "fig10_panel_profile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 11 – Model Ranking Bump Chart  (rank across 3 experiments)
# =============================================================================
def fig_ranking_bump():
    """Slope/bump chart: how model rankings shift across Exp1 → Exp2 → Exp3."""
    exp_labels = ["Exp1\n(All CV)", "Exp2\n(LOPO)", "Exp3\n(Generalise)"]

    fig, axes = plt.subplots(1, len(KMERS), figsize=(7 * len(KMERS), 5), sharey=True)
    if len(KMERS) == 1:
        axes = [axes]

    for ax, run in zip(axes, results):
        scores = {m: [] for m in MODELS}

        # Exp1
        for m in MODELS:
            scores[m].append(run["exp1"]["models"][m]["aggregated"]["auroc_mean"])

        # Exp2 — mean across panels
        for m in MODELS:
            vals = [run["exp2"]["panels"][p]["models"][m]["auroc"]
                    for p in PANELS if p in run["exp2"]["panels"]
                    and m in run["exp2"]["panels"][p]["models"]]
            scores[m].append(float(np.nanmean(vals)) if vals else np.nan)

        # Exp3 — mean across all train→test pairs
        for m in MODELS:
            vals = [
                run["exp3"]["train_panels"][tr]["test_panels"].get(m, {}).get(te, {}).get("auroc", np.nan)
                for tr in run["exp3"]["train_panels"]
                for te in run["exp3"]["train_panels"][tr]["test_panels"].get(m, {})
            ]
            scores[m].append(float(np.nanmean(vals)) if vals else np.nan)

        # Compute ranks at each experiment (1 = best)
        for exp_idx in range(3):
            exp_scores = [(m, scores[m][exp_idx]) for m in MODELS]
            exp_scores.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
            for rank, (m, _) in enumerate(exp_scores, 1):
                scores[m] = scores[m]  # keep scores for line; rank is implicit

        x = [0, 1, 2]
        for m in MODELS:
            y_vals = scores[m]
            ax.plot(x, y_vals, marker="o", ms=10, lw=2.5,
                    color=PALETTE[m], label=MODEL_LABELS[m], zorder=3)
            for xi, yv in zip(x, y_vals):
                if not np.isnan(yv):
                    ax.annotate(f"{yv:.3f}", (xi, yv),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=9, color=PALETTE[m], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=11)
        ax.set_ylabel("Mean AUROC")
        ax.set_ylim(0.4, 1.1)
        ax.set_title(f"k={run['k']}", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.35)

    fig.suptitle("Model Performance Trajectory Across Experiments",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig11_model_trajectory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# FIGURE 12 – Best Hyperparameters Visual Table  (Exp1, best k)
# =============================================================================
def fig_hyperparam_table():
    """Visual table of the winning hyperparameters per model."""
    def best_run(model):
        return max(results, key=lambda r: r["exp1"]["models"][model]["aggregated"]["auroc_mean"])

    # Collect param rows per model
    param_union = {}
    model_params = {}
    for m in MODELS:
        run    = best_run(m)
        params = run["exp1"]["models"][m]["fold_metrics"][0].get("best_params", {}) \
                 if run["exp1"]["models"][m].get("fold_metrics") else {}
        model_params[m] = params
        for k in params:
            param_union[k] = True

    if not param_union:
        return

    param_keys = sorted(param_union.keys())
    col_labels = [MODEL_LABELS[m] for m in MODELS]
    row_labels = param_keys

    cell_text, cell_colors = [], []
    for pk in param_keys:
        row_txt, row_col = [], []
        vals = [model_params[m].get(pk, "—") for m in MODELS]
        # highlight the "best" numeric value
        num_vals = [v for v in vals if isinstance(v, (int, float))]
        for v in vals:
            row_txt.append(str(v) if v != "—" else "—")
            if isinstance(v, (int, float)) and num_vals:
                intensity = (v - min(num_vals)) / (max(num_vals) - min(num_vals) + 1e-9)
                row_col.append(plt.cm.Blues(0.25 + 0.55 * intensity))
            else:
                row_col.append((0.97, 0.97, 0.97, 1))
        cell_text.append(row_txt)
        cell_colors.append(row_col)

    fig, ax = plt.subplots(figsize=(10, max(4, len(param_keys) * 0.6 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center", rowLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.35, 1.75)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#cccccc")

    fig.suptitle("Best Hyperparameters per Model (Exp1, Best k-mer Fold)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = OUT_DIR / "fig12_hyperparams_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# =============================================================================
# CSV EXPORT – Full numeric summary (all experiments, all models, all k-mers)
# =============================================================================
def export_csv_summary():
    """Write a flat CSV with every metric value for easy downstream analysis."""
    import csv

    rows = []
    header = ["experiment", "k", "panel_train", "panel_test", "model",
              "auroc", "auprc", "accuracy", "f1", "precision", "recall"]

    for run in results:
        k = run["k"]

        # Exp 1 — per-fold rows
        for model in MODELS:
            for fm in run["exp1"]["models"][model].get("fold_metrics", []):
                rows.append([
                    "exp1", k, "ALL", "ALL", model,
                    fm.get("auroc",""), fm.get("auprc",""), fm.get("accuracy",""),
                    fm.get("f1",""), fm.get("precision",""), fm.get("recall",""),
                ])

        # Exp 2 — per-panel rows
        for panel in PANELS:
            if panel not in run["exp2"]["panels"]:
                continue
            for model in MODELS:
                m_data = run["exp2"]["panels"][panel]["models"].get(model, {})
                if not m_data:
                    continue
                rows.append([
                    "exp2", k, "ALL_MINUS_"+panel, panel, model,
                    m_data.get("auroc",""), m_data.get("auprc",""), m_data.get("accuracy",""),
                    m_data.get("f1",""), m_data.get("precision",""), m_data.get("recall",""),
                ])

        # Exp 3 — per (train_panel, test_panel) rows
        for tr_panel, tr_data in run["exp3"]["train_panels"].items():
            for model in MODELS:
                for te_panel, m_data in tr_data["test_panels"].get(model, {}).items():
                    rows.append([
                        "exp3", k, tr_panel, te_panel, model,
                        m_data.get("auroc",""), m_data.get("auprc",""), m_data.get("accuracy",""),
                        m_data.get("f1",""), m_data.get("precision",""), m_data.get("recall",""),
                    ])

    csv_path = Path("results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  ✓  {csv_path}  ({len(rows)} rows)")
    return csv_path


# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================
print("Generating figures …")
fig_exp1()
fig_exp2()
fig_exp3()
fig_kmer_comparison()
fig_radar()
fig_feature_registry()
fig_fold_stability()
fig_summary_table()
fig_generalisation_gap()
fig_panel_profile()
fig_ranking_bump()
fig_hyperparam_table()
print("\nExporting CSV summary …")
csv_path = export_csv_summary()
print("All figures saved.\n")


# =============================================================================
# PACKAGE EVERYTHING INTO A ZIP
# =============================================================================
ZIP_NAME = "variant_predictor_results.zip"

artefacts = (
    list(OUT_DIR.glob("*.png"))          # all figures
    + [Path("results.json")]             # experiment results
    + [Path("feature_registry.json")]    # feature audit
    + [Path("results_summary.csv")]      # flat CSV export
)

# Include any saved model files if they exist
for pattern in ["*.json", "*.pt", "*.pth", "*.txt", "*.pkl", "*.bin"]:
    for f in Path(".").glob(pattern):
        if f.name not in {"results.json", "feature_registry.json"} and f not in artefacts:
            artefacts.append(f)

with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in artefacts:
        if path.exists():
            zf.write(path, path.name)
            print(f"  + {path.name}  ({path.stat().st_size / 1024:.1f} KB)")

print(f"\nZip created: {ZIP_NAME}  "
      f"({Path(ZIP_NAME).stat().st_size / 1024 / 1024:.2f} MB)")


# =============================================================================
# DOWNLOAD TO DESKTOP
# =============================================================================
try:
    from google.colab import files
    print("\nStarting download …")
    files.download(ZIP_NAME)
    print("✓  Download triggered — check your browser / Downloads folder.")
except ImportError:
    print(f"\nNot running on Colab — zip is at: {Path(ZIP_NAME).resolve()}")

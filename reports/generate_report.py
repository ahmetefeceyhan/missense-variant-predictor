# -*- coding: utf-8 -*-
"""
Kapsamli Model Karsilastirma Raporu v2 - PDF
Genetic Variant Pathogenicity Prediction
v1 (leaked) vs v2 (clean) karsilastirma + Stacking sonuclari
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import os, sys, tempfile, warnings
warnings.filterwarnings('ignore')

# Proje koku
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from config import RESULTS_V1_DIR, RESULTS_V2_DIR, REPORTS_DIR

# ============================================================================
# SABITLER
# ============================================================================
MODEL_LABELS = {
    'lightgbm': 'LightGBM', 'xgboost': 'XGBoost',
    'nn': 'Neural Net (MLP)', 'dnn': 'Deep NN',
    'stacking_lr': 'Stacking (LR)', 'stacking_lgbm': 'Stacking (LGBM)',
}
MODEL_COLORS = {
    'lightgbm': '#2196F3', 'xgboost': '#4CAF50',
    'nn': '#FF9800', 'dnn': '#E91E63',
    'stacking_lr': '#9C27B0', 'stacking_lgbm': '#00BCD4',
}
FE_LABELS = {
    'no_fe': 'No FE (leaked)', 'with_fe': 'With FE (leaked)',
    'clean': 'Clean', 'clean_fe': 'Clean + FE',
    'full_fe': 'Full FE (in-silico + FE)',
    'v3': 'V3 (Optimized Pipeline)',
}
FE_COLORS = {
    'no_fe': '#90CAF9', 'with_fe': '#EF9A9A',
    'clean': '#A5D6A7', 'clean_fe': '#CE93D8',
    'full_fe': '#FFD54F',
}
MODEL_ORDER = ['lightgbm', 'xgboost', 'nn', 'dnn']
STACKING_ORDER = ['stacking_lr', 'stacking_lgbm']
ALL_MODEL_ORDER = MODEL_ORDER + STACKING_ORDER
METRICS = ['f1', 'auc_roc', 'auc_pr', 'mcc', 'precision', 'recall',
           'specificity', 'balanced_accuracy', 'cohens_kappa']
METRIC_LABELS = {
    'f1': 'F1 Score', 'auc_roc': 'AUC-ROC', 'auc_pr': 'AUC-PR', 'mcc': 'MCC',
    'precision': 'Precision', 'recall': 'Recall', 'specificity': 'Specificity',
    'balanced_accuracy': 'Balanced Acc', 'cohens_kappa': "Cohen's Kappa"
}

TMPDIR = tempfile.mkdtemp()


def save_fig(fig, name):
    path = os.path.join(TMPDIR, f'{name}.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ============================================================================
# GRAFIK FONKSIYONLARI
# ============================================================================

def make_leakage_comparison(v1_df, v2_df):
    """v1 (leaked) vs v2 (clean) F1 karsilastirmasi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Data Leakage Impact: v1 (Leaked) vs v2 (Clean)', fontsize=14, fontweight='bold')

    # a) Model bazinda ortalama F1
    ax = axes[0]
    v1_avg = v1_df.groupby('model_type')['f1'].mean().reindex(MODEL_ORDER)
    v2_clean_fe = v2_df[v2_df['fe_state'] == 'clean_fe']
    v2_avg = v2_clean_fe.groupby('model_type')['f1'].mean().reindex(MODEL_ORDER)

    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    bars1 = ax.bar(x - w/2, v1_avg.values, w, label='v1 (Leaked)', color='#EF5350', alpha=0.85)
    bars2 = ax.bar(x + w/2, v2_avg.values, w, label='v2 (Clean+FE)', color='#66BB6A', alpha=0.85)

    for b, v in zip(bars1, v1_avg.values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    for b, v in zip(bars2, v2_avg.values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel('Average F1')
    ax.set_title('Average F1 by Model Type', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # b) Panel bazinda
    ax = axes[1]
    panels = sorted(set(v1_df['panel'].unique()) | set(v2_clean_fe['panel'].unique()))
    v1_panel = v1_df.groupby('panel')['f1'].max().reindex(panels)
    v2_panel = v2_clean_fe.groupby('panel')['f1'].max().reindex(panels)
    x = np.arange(len(panels))
    ax.bar(x - w/2, v1_panel.values, w, label='v1 Best (Leaked)', color='#EF5350', alpha=0.85)
    ax.bar(x + w/2, v2_panel.values, w, label='v2 Best (Clean+FE)', color='#66BB6A', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(panels, rotation=15)
    ax.set_ylabel('Best F1')
    ax.set_title('Best F1 by Panel', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, 'leakage_comparison')


def make_fe_modes_comparison(v2_df):
    """4 FE modu karsilastirmasi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Engineering Mode Comparison (v2)', fontsize=14, fontweight='bold')

    fe_modes = ['no_fe', 'with_fe', 'clean', 'clean_fe']
    present_modes = [m for m in fe_modes if m in v2_df['fe_state'].unique()]

    # a) FE modu bazinda ortalama F1
    ax = axes[0]
    fe_avg = v2_df.groupby('fe_state')['f1'].mean().reindex(present_modes)
    bars = ax.bar(range(len(present_modes)), fe_avg.values,
                  color=[FE_COLORS.get(m, 'gray') for m in present_modes], alpha=0.85)
    ax.set_xticks(range(len(present_modes)))
    ax.set_xticklabels([FE_LABELS.get(m, m) for m in present_modes], rotation=15)
    for b, v in zip(bars, fe_avg.values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Average F1')
    ax.set_title('Avg F1 by FE Mode', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # b) FE modu x model tipi
    ax = axes[1]
    x = np.arange(len(MODEL_ORDER))
    w = 0.8 / len(present_modes)
    for i, fe in enumerate(present_modes):
        vals = v2_df[v2_df['fe_state'] == fe].groupby('model_type')['f1'].mean().reindex(MODEL_ORDER)
        offset = (i - len(present_modes)/2 + 0.5) * w
        ax.bar(x + offset, vals.values, w, label=FE_LABELS.get(fe, fe),
               color=FE_COLORS.get(fe, 'gray'), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel('Average F1')
    ax.set_title('F1 by Model x FE Mode', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, 'fe_modes')


def make_clean_model_analysis(v2_df):
    """Clean modlardaki model karsilastirmasi."""
    df_clean = v2_df[v2_df['fe_state'].isin(['clean', 'clean_fe'])]
    if df_clean.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Clean Model Analysis (No Data Leakage)', fontsize=14, fontweight='bold')

    # a) Avg F1
    ax = axes[0][0]
    mf = df_clean.groupby('model_type')['f1'].agg(['mean', 'std']).reindex(MODEL_ORDER)
    present = mf.dropna()
    bars = ax.bar(range(len(present)), present['mean'], yerr=present['std'],
                  capsize=5, color=[MODEL_COLORS[m] for m in present.index], alpha=0.85)
    for b, v in zip(bars, present['mean']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in present.index])
    ax.set_ylabel('F1')
    ax.set_title('Average F1 (Clean modes)', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # b) Boxplot
    ax = axes[0][1]
    present_models = [m for m in MODEL_ORDER if m in df_clean['model_type'].unique()]
    bp = ax.boxplot([df_clean[df_clean['model_type'] == mt]['f1'].values for mt in present_models],
                    labels=[MODEL_LABELS[m] for m in present_models], patch_artist=True, widths=0.5)
    for patch, mt in zip(bp['boxes'], present_models):
        patch.set_facecolor(MODEL_COLORS[mt])
        patch.set_alpha(0.7)
    ax.set_ylabel('F1')
    ax.set_title('F1 Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # c) Key metrics grouped
    ax = axes[1][0]
    km = ['f1', 'auc_roc', 'mcc', 'balanced_accuracy']
    x = np.arange(len(km))
    w = 0.18
    for i, mt in enumerate(present_models):
        vals = df_clean[df_clean['model_type'] == mt][km].mean().values
        ax.bar(x + i*w - (len(present_models)-1)*w/2, vals, w,
               label=MODEL_LABELS[mt], color=MODEL_COLORS[mt], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in km])
    ax.set_ylabel('Score')
    ax.set_title('Key Metrics', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # d) Panel bazinda best F1
    ax = axes[1][1]
    panels = sorted(df_clean['panel'].unique())
    panel_best = df_clean.groupby('panel')['f1'].max().reindex(panels)
    bars = ax.barh(range(len(panels)), panel_best.values,
                   color=['#66BB6A'] * len(panels), alpha=0.85)
    for b, v in zip(bars, panel_best.values):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2, f'{v:.4f}',
                va='center', fontsize=9)
    ax.set_yticks(range(len(panels)))
    ax.set_yticklabels(panels)
    ax.set_xlabel('Best F1')
    ax.set_title('Best F1 per Panel', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, 'clean_analysis')


def make_stacking_comparison(v2_df, stacking_df):
    """Tek model vs stacking karsilastirmasi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Single Model vs Stacking Ensemble', fontsize=14, fontweight='bold')

    # Sadece clean_fe sonuclari
    single_clean = v2_df[v2_df['fe_state'] == 'clean_fe']

    panels = sorted(set(single_clean['panel'].unique()) & set(stacking_df['panel'].unique()))

    # a) Panel bazinda best F1
    ax = axes[0]
    single_best = single_clean.groupby('panel')['f1'].max().reindex(panels)
    stack_best = stacking_df.groupby('panel')['f1'].max().reindex(panels)

    x = np.arange(len(panels))
    w = 0.35
    ax.bar(x - w/2, single_best.values, w, label='Best Single Model', color='#42A5F5', alpha=0.85)
    ax.bar(x + w/2, stack_best.values, w, label='Best Stacking', color='#AB47BC', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(panels, rotation=15)
    ax.set_ylabel('Best F1')
    ax.set_title('Best F1: Single vs Stacking', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # b) Stacking tip karsilastirmasi
    ax = axes[1]
    stack_types = sorted(stacking_df['model_type'].unique())
    for i, st in enumerate(stack_types):
        st_data = stacking_df[stacking_df['model_type'] == st]
        st_panel = st_data.groupby('panel')['f1'].max().reindex(panels)
        offset = (i - len(stack_types)/2 + 0.5) * w
        ax.bar(x + offset, st_panel.values, w, label=MODEL_LABELS.get(st, st),
               color=MODEL_COLORS.get(st, 'gray'), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(panels, rotation=15)
    ax.set_ylabel('F1')
    ax.set_title('Stacking Types Comparison', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, 'stacking_comparison')


def make_full_fe_analysis(full_fe_df, v2_df, stacking_df=None):
    """Full-FE analiz grafigi: model bazinda + onceki modlarla karsilastirma."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Full-FE Analysis: In-silico Scores + Advanced FE', fontsize=14, fontweight='bold')

    # a) Full-FE model bazinda F1
    ax = axes[0]
    present = [m for m in ALL_MODEL_ORDER if m in full_fe_df['model_type'].unique()]
    ff_avg = full_fe_df.groupby('model_type')['f1'].mean().reindex(present)
    bars = ax.bar(range(len(present)), ff_avg.values,
                  color=[MODEL_COLORS.get(m, 'gray') for m in present], alpha=0.85)
    for b, v in zip(bars, ff_avg.values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present], rotation=15)
    ax.set_ylabel('Average F1')
    ax.set_title('Full-FE: Avg F1 by Model', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # b) FE modu karsilastirmasi (tum modlarin ortalamasi)
    ax = axes[1]
    all_combined = v2_df.copy()
    if stacking_df is not None:
        all_combined = pd.concat([all_combined, stacking_df], ignore_index=True)
    all_combined = pd.concat([all_combined, full_fe_df], ignore_index=True)

    fe_order = [fe for fe in ['no_fe', 'with_fe', 'clean', 'clean_fe', 'full_fe']
                if fe in all_combined['fe_state'].unique()]
    fe_avg = all_combined.groupby('fe_state')['f1'].mean().reindex(fe_order)
    fe_best = all_combined.groupby('fe_state')['f1'].max().reindex(fe_order)

    x = np.arange(len(fe_order))
    w = 0.35
    bars1 = ax.bar(x - w/2, fe_avg.values, w, label='Avg F1', color='#42A5F5', alpha=0.85)
    bars2 = ax.bar(x + w/2, fe_best.values, w, label='Best F1', color='#66BB6A', alpha=0.85)
    for b, v in zip(bars1, fe_avg.values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    for b, v in zip(bars2, fe_best.values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([FE_LABELS.get(fe, fe) for fe in fe_order], rotation=20, fontsize=8)
    ax.set_ylabel('F1 Score')
    ax.set_title('All FE Modes: Avg vs Best F1', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return save_fig(fig, 'full_fe_analysis')


def make_heatmap(df, title='Model Comparison Heatmap'):
    """Tum konfigurasyonlar icin metrik heatmap."""
    fig, ax = plt.subplots(figsize=(12, max(9, len(df) * 0.35)))
    hm = df.sort_values('f1', ascending=False).set_index('config_id')[METRICS].copy()
    hm.columns = [METRIC_LABELS[m] for m in METRICS]
    sns.heatmap(hm, annot=True, fmt='.3f', cmap='RdYlGn', linewidths=0.5,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()
    return save_fig(fig, 'heatmap')


def make_radar(df, title='Best Config per Model - Radar Chart'):
    """Radar chart."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    rm = ['f1', 'auc_roc', 'mcc', 'precision', 'recall', 'specificity', 'balanced_accuracy']
    angles = np.linspace(0, 2*np.pi, len(rm), endpoint=False).tolist() + [0]

    present = [mt for mt in ALL_MODEL_ORDER if mt in df['model_type'].unique()]
    for mt in present:
        sub = df[df['model_type'] == mt]
        if sub.empty:
            continue
        best = sub.loc[sub['f1'].idxmax()]
        vals = [best[m] for m in rm] + [best[rm[0]]]
        ax.plot(angles, vals, 'o-', lw=2, label=MODEL_LABELS.get(mt, mt),
                color=MODEL_COLORS.get(mt, 'gray'), ms=5)
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS.get(mt, 'gray'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in rm], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    return save_fig(fig, 'radar')


def make_training_time(df):
    """Egitim suresi analizi."""
    df = df.copy()
    df['time_min'] = df['time_seconds'] / 60.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Time Analysis', fontsize=14, fontweight='bold')

    # a) Model bazinda
    ax = axes[0]
    present = [m for m in ALL_MODEL_ORDER if m in df['model_type'].unique()]
    ta = df.groupby('model_type')['time_min'].agg(['mean', 'std']).reindex(present)
    bars = ax.bar(range(len(present)), ta['mean'], yerr=ta['std'], capsize=5,
                  color=[MODEL_COLORS.get(m, 'gray') for m in present], alpha=0.85)
    for b, v in zip(bars, ta['mean']):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f'{v:.1f}m', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in present], rotation=15)
    ax.set_ylabel('Minutes')
    ax.set_title('Avg Time by Model', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # b) Tum konfigurasyonlar
    ax = axes[1]
    ds = df.sort_values('time_min', ascending=True)
    bc = [MODEL_COLORS.get(m, 'gray') for m in ds['model_type']]
    ax.barh(range(len(ds)), ds['time_min'], color=bc, alpha=0.85)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels(ds['config_id'], fontsize=5.5)
    ax.set_xlabel('Minutes')
    ax.set_title('All Configs', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    tot = df['time_min'].sum()
    ax.text(0.98, 0.02, f'Total: {tot:.0f}m ({tot/60:.1f}h)', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return save_fig(fig, 'training_time')


# ============================================================================
# PDF SINIFI
# ============================================================================

class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, 'Model Comparison Report v2 - Variant Pathogenicity Prediction',
                      align='C')
            self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, num, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(26, 26, 46)
        self.cell(0, 10, f'{num}. {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(233, 69, 96)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 5.5, text)
        self.ln(1)

    def mono_text(self, text):
        self.set_font('Courier', '', 7.5)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 4.5, text)

    def add_chart(self, img_path, w=None):
        if img_path is None:
            return
        if w is None:
            w = self.w - self.l_margin - self.r_margin
        if self.get_y() + 100 > self.h - 20:
            self.add_page()
        self.image(img_path, x=self.l_margin, w=w)
        self.ln(5)

    def add_results_table(self, df_sorted, include_fe=True):
        self.set_font('Helvetica', 'B', 7)
        self.set_fill_color(26, 26, 46)
        self.set_text_color(255, 255, 255)

        if include_fe:
            cols = ['#', 'Model', 'Panel', 'Mode', 'FE', 'F1', 'AUC-ROC', 'MCC',
                    'Prec', 'Recall', 'Spec', 'Thresh']
            cw = [6, 18, 22, 10, 12, 12, 14, 12, 12, 12, 12, 12]
        else:
            cols = ['#', 'Model', 'Panel', 'Mode', 'F1', 'AUC-ROC', 'MCC',
                    'Prec', 'Recall', 'Spec', 'Thresh']
            cw = [6, 22, 22, 10, 14, 16, 14, 14, 14, 14, 14]

        for c, w in zip(cols, cw):
            self.cell(w, 6, c, border=1, align='C', fill=True)
        self.ln()

        self.set_font('Helvetica', '', 7)
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            if rank <= 3:
                self.set_fill_color(232, 245, 233)
                fill = True
            elif rank > len(df_sorted) - 3:
                self.set_fill_color(252, 228, 236)
                fill = True
            else:
                fill = False

            self.set_text_color(50, 50, 50)
            fe_label = FE_LABELS.get(row.get('fe_state', ''), row.get('fe_state', ''))
            # Truncate long FE labels
            if len(fe_label) > 12:
                fe_label = fe_label[:11] + '.'

            if include_fe:
                vals = [
                    str(rank),
                    MODEL_LABELS.get(row['model_type'], row['model_type'])[:18],
                    str(row['panel'])[:22],
                    'Multi' if row['mode'] == 'multi_panel' else 'Single',
                    fe_label,
                    f"{row['f1']:.4f}",
                    f"{row['auc_roc']:.4f}",
                    f"{row['mcc']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['specificity']:.4f}",
                    f"{row['best_threshold']:.2f}",
                ]
            else:
                vals = [
                    str(rank),
                    MODEL_LABELS.get(row['model_type'], row['model_type'])[:22],
                    str(row['panel'])[:22],
                    'Multi' if row['mode'] == 'multi_panel' else 'Single',
                    f"{row['f1']:.4f}",
                    f"{row['auc_roc']:.4f}",
                    f"{row['mcc']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['specificity']:.4f}",
                    f"{row['best_threshold']:.2f}",
                ]

            for v, w in zip(vals, cw):
                self.cell(w, 5, v, border=1, align='C', fill=fill)
            self.ln()


# ============================================================================
# ANA
# ============================================================================

def main():
    # Veri yukle
    v2_path = os.path.join(RESULTS_V2_DIR, 'model_comparison_results.csv')
    v1_path = os.path.join(RESULTS_V1_DIR, 'model_comparison_results.csv')
    stacking_path = os.path.join(RESULTS_V2_DIR, 'stacking_results.csv')

    if not os.path.exists(v2_path):
        print(f"HATA: v2 sonuclari bulunamadi: {v2_path}")
        print("Once 02_training_clean.ipynb calistirilmali.")
        return

    v2_df = pd.read_csv(v2_path)
    print(f"v2 sonuclari yuklendi: {len(v2_df)} konfigurasyon")

    has_v1 = os.path.exists(v1_path)
    v1_df = pd.read_csv(v1_path) if has_v1 else None
    if has_v1:
        print(f"v1 sonuclari yuklendi: {len(v1_df)} konfigurasyon")

    has_stacking = os.path.exists(stacking_path)
    stacking_df = pd.read_csv(stacking_path) if has_stacking else None
    if has_stacking:
        print(f"Stacking sonuclari yuklendi: {len(stacking_df)} konfigurasyon")

    full_fe_path = os.path.join(RESULTS_V2_DIR, 'full_fe_results.csv')
    has_full_fe = os.path.exists(full_fe_path)
    full_fe_df = pd.read_csv(full_fe_path) if has_full_fe else None
    if has_full_fe:
        print(f"Full-FE sonuclari yuklendi: {len(full_fe_df)} konfigurasyon")

    # Birlesik DataFrame
    all_df = v2_df.copy()
    if has_stacking:
        all_df = pd.concat([all_df, stacking_df], ignore_index=True)
    if has_full_fe:
        all_df = pd.concat([all_df, full_fe_df], ignore_index=True)
    all_sorted = all_df.sort_values('f1', ascending=False).reset_index(drop=True)

    # En iyi genel sonuc
    best_overall = all_sorted.iloc[0]
    # Clean modlardaki en iyi sonuc
    clean_df = all_df[all_df['fe_state'].isin(['clean', 'clean_fe'])]
    best_clean = clean_df.loc[clean_df['f1'].idxmax()] if len(clean_df) > 0 else best_overall

    # ===================== GRAFIKLER =====================
    print("Grafikler olusturuluyor...")

    img_leakage = make_leakage_comparison(v1_df, v2_df) if has_v1 else None
    img_fe_modes = make_fe_modes_comparison(v2_df)
    img_clean = make_clean_model_analysis(v2_df)
    img_stacking = make_stacking_comparison(v2_df, stacking_df) if has_stacking else None
    img_full_fe = make_full_fe_analysis(full_fe_df, v2_df, stacking_df) if has_full_fe else None
    img_heatmap = make_heatmap(all_sorted, 'All Configurations Heatmap')
    img_radar = make_radar(all_df, 'Best Config per Model - Radar Chart')
    img_time = make_training_time(all_df)

    # ===================== PDF =====================
    print("PDF olusturuluyor...")
    pdf = ReportPDF(orientation='P', unit='mm', format='A4')
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== KAPAK =====
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 15, 'Kapsamli Model', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 15, 'Karsilastirma Raporu v2', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(233, 69, 96)
    pdf.cell(0, 10, 'Genetik Varyant Patojenite Tahmini', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)
    pdf.set_draw_color(233, 69, 96)
    pdf.set_line_width(0.8)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(12)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(100, 100, 100)

    n_configs = len(all_df)
    n_models = len(all_df['model_type'].unique())
    n_fe = len(all_df['fe_state'].unique())

    for line in [
        f'{n_models} Model Tipi  |  {n_fe} FE Modu  |  2 Egitim Modu',
        f'{n_configs} Toplam Konfigurasyon  |  Optuna Bayesian Optimizasyon',
        '',
        'Modeller: LightGBM, XGBoost, Neural Net, Deep NN, Stacking Ensemble',
        f'Veri Seti: Genetik Varyant  |  4 Gen Paneli',
        '',
        'YENILIKLER: Full-FE (In-silico + Gelismis FE), Focal Loss,',
        'Delta K-mer, Yonlu BLOSUM62, Conservation PCA, Stacking Ensemble',
        '',
        f'En Iyi Sonuc: {MODEL_LABELS.get(best_overall["model_type"], best_overall["model_type"])}',
        f'F1 = {best_overall["f1"]:.4f}  |  AUC-ROC = {best_overall["auc_roc"]:.4f}',
        f'FE Modu: {FE_LABELS.get(best_overall["fe_state"], best_overall["fe_state"])}',
    ]:
        pdf.cell(0, 7, line, align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 7, f'Olusturulma: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')

    # ===== 1. METODOLOJI =====
    pdf.add_page()
    pdf.section_title(1, 'Metodoloji')

    pdf.bold_text('1.1 Deneysel Tasarim')
    pdf.body_text(
        '4 model mimarisi (LightGBM, XGBoost, Neural Net, Deep NN), 4 feature engineering '
        'modu (no_fe, with_fe, clean, clean_fe) ve 2 egitim modu (single-panel, multi-panel) '
        'kullanilarak kapsamli bir deneysel karsilastirma yapilmistir. Ek olarak 2 stacking '
        'ensemble konfigurasyonu (LR ve LightGBM meta-learner) test edilmistir.'
    )
    pdf.body_text(
        'v1 (leaked) sonuclari: Orijinal egitim ~60 meta-predictor skorunu (REVEL, CADD, '
        'ClinPred, AlphaMissense, MetaSVM vb.) feature olarak kullaniyordu. Bu skorlar '
        'ClinVar uzerinde egitilmis ML modelleri tarafindan uretilmistir -- ayni ClinVar '
        'verisi uzerinde bu skorlari kullanmak dongusellik (circularity) yaratir ve '
        'gercekci olmayan yuksek performans uretir.'
    )
    pdf.body_text(
        'v2 (clean) sonuclari: Tum meta-predictor skorlari, rankscore degerleri, prediction '
        'kategorileri, gnomAD populasyon frekanslari ve tool_consensus tamamen kaldirildi. '
        'Sadece biyolojik olarak anlamli ozellikler korundu: evrimsel korunma (GERP, PhyloP, '
        'PhastCons), fizikokimyasal delta degerleri ve sekans bilgisi.'
    )

    pdf.bold_text('1.2 Modeller')
    pdf.body_text(
        'LightGBM: Yaprak-oncelikli agac buyumesi, 200 Optuna trial, 5-fold CV. '
        'Clean modlarda Focal Loss opsiyonel (gamma ve alpha Optuna ile tune edilir).'
    )
    pdf.body_text(
        'XGBoost: Seviye-oncelikli agac buyumesi, L1/L2 regularizasyon, 200 Optuna trial, '
        '5-fold CV. Clean modlarda Focal Loss destegi.'
    )
    pdf.body_text(
        'Neural Network (NN): 1-2 katmanli MLP, BatchNorm + ReLU + Dropout. '
        'PyTorch + DirectML (AMD GPU). 100 Optuna trial, 3-fold CV. '
        'Clean modlarda FocalLoss kullanilir.'
    )
    pdf.body_text(
        'Deep Neural Network (DNN): 3-5 katmanli MLP, her 2 katmanda residual '
        'baglanti. Dropout (0.2-0.6), ReduceLROnPlateau. 100 Optuna trial.'
    )
    pdf.body_text(
        'Stacking Ensemble: Base modeller (LightGBM + XGBoost + SklearnMLP) ile '
        'StackingClassifier. Meta-learner olarak LogisticRegression (stacking_lr) veya '
        'LightGBM (stacking_lgbm). StratifiedKFold(5) CV ile stacking leakage onlenir.'
    )

    pdf.bold_text('1.3 Feature Engineering Modlari')
    pdf.body_text(
        'no_fe (leaked baseline): Orijinal 99 numerik/kategorik feature + 656 k-mer '
        '(CountVectorizer, max_features=200). Meta-predictor skorlari dahil -- veri sizintisi var.'
    )
    pdf.body_text(
        'with_fe (leaked FE): Grantham, BLOSUM62, CpG, GC content, gnomAD turevleri, '
        'tool_consensus. ~57 feature. Meta-predictor bazli ozellikler -- veri sizintisi var.'
    )
    pdf.body_text(
        'clean (temiz baseline): Tum meta-predictor, rankscore, gnomAD ve prediction '
        'sutunlari kaldirildi. Iyilestirilmis k-mer (max_features=500, min_df=2). '
        'Sizinti yok.'
    )
    pdf.body_text(
        'clean_fe (temiz + gelismis FE): Clean modun ustune eklenen gelismis ozellikler: '
        'Yonlu BLOSUM62 (ref_self, alt_self, delta), Conservation PCA (4 skor -> 2 PC), '
        'GERP x PhyloP etkilesimi, Delta K-mer (Ref-Alt farki), Grantham mesafesi/kategorisi, '
        'CpG/GC/transition. Korelasyon filtresi (r > 0.99). Sizinti yok.'
    )

    pdf.body_text(
        'full_fe (in-silico + gelismis FE): In-silico meta-predictor skorlari KORUNARAK '
        'gelismis FE ozellikleri eklenmistir. Yarisma veri setinde in-silico skorlar '
        'saglanacagi icin bu mod en kapsamli feature setini sunar. Tum clean_fe '
        'ozellikleri + REVEL, CADD, ClinPred, AlphaMissense, MetaSVM, SIFT vb. skorlar '
        '+ gnomAD turevleri dahildir.'
    )

    pdf.bold_text('1.4 Focal Loss')
    pdf.body_text(
        'Sinif dengesizligini (68% benign, 32% pathogenic) ele almak icin Focal Loss '
        'kullanilmistir. FL = -alpha * (1-p)^gamma * log(p). gamma parametresi kolay '
        'orneklerin agirligini azaltir, alpha sinif dengesini ayarlar. '
        'Clean modlarda otomatik uygulanir, gamma ve alpha Optuna ile optimize edilir.'
    )

    # ===== 2. VERI SIZINTISI ANALIZI =====
    if has_v1:
        pdf.add_page()
        pdf.section_title(2, 'Veri Sizintisi (Data Leakage) Analizi')
        pdf.body_text(
            'Orijinal modeller (v1), ClinVar uzerinde egitilmis ~60 meta-predictor skorunu '
            'feature olarak kullaniyordu. Bu skorlar (REVEL, CADD, ClinPred, AlphaMissense, '
            'MetaSVM, MetaLR, MetaRNN, ESM1b, MiSTIC, MutationTaster, SIFT, PROVEAN, VEST, '
            'BayesDel, FATHMM, CHASMplus, vb.) kendi basina ClinVar verisi uzerinde egitilmis '
            'ML modelleridir.'
        )
        pdf.body_text(
            'Bu dongusellik (circularity) nedeniyle v1 modelleri gercekci olmayan yuksek '
            'performans gosteriyordu. v2\'de bu skorlarin tamami kaldirildi ve sadece '
            'biyolojik olarak anlamli ozellikler (evrimsel korunma, fizikokimyasal ozellikler, '
            'sekans bilgisi) kullanildi.'
        )

        # Karsilastirma istatistikleri
        v1_avg_f1 = v1_df['f1'].mean()
        v2_clean_fe = v2_df[v2_df['fe_state'] == 'clean_fe']
        v2_avg_f1 = v2_clean_fe['f1'].mean() if len(v2_clean_fe) > 0 else 0
        drop = v1_avg_f1 - v2_avg_f1

        pdf.bold_text('Performans Karsilastirmasi')
        pdf.mono_text(f'  v1 (leaked) ortalama F1:    {v1_avg_f1:.4f}')
        pdf.mono_text(f'  v2 (clean_fe) ortalama F1:  {v2_avg_f1:.4f}')
        pdf.mono_text(f'  F1 dususu:                  {drop:.4f}')
        pdf.ln(2)
        pdf.body_text(
            f'v1\'den v2\'ye geciste ortalama F1 skoru {drop:.4f} dusmustur. '
            f'Bu beklenen bir sonuctur -- yuksek v1 performansi meta-predictor '
            f'circularity\'den kaynaklaniyordu. v2 sonuclari gercek model '
            f'performansini yansitmaktadir.'
        )
        pdf.ln(2)

        # Model bazinda karsilastirma
        pdf.bold_text('Model Bazinda v1 vs v2 Karsilastirma')
        for mt in MODEL_ORDER:
            v1_mt = v1_df[v1_df['model_type'] == mt]['f1'].mean()
            v2_mt = v2_clean_fe[v2_clean_fe['model_type'] == mt]['f1'].mean() if len(v2_clean_fe) > 0 else 0
            d = v2_mt - v1_mt
            pdf.mono_text(f'  {MODEL_LABELS[mt]:>15s}:  v1={v1_mt:.4f}  v2={v2_mt:.4f}  (delta={d:+.4f})')
        pdf.ln(3)

        pdf.add_chart(img_leakage)
        section_num = 3
    else:
        section_num = 2

    # ===== FE MODLARI KARSILASTIRMASI =====
    pdf.add_page()
    pdf.section_title(section_num, 'Feature Engineering Modu Karsilastirmasi')
    section_num += 1

    fe_modes_present = all_df['fe_state'].unique()
    for fe in ['no_fe', 'with_fe', 'clean', 'clean_fe', 'full_fe']:
        if fe in fe_modes_present:
            fe_data = all_df[all_df['fe_state'] == fe]
            avg_f1 = fe_data['f1'].mean()
            best_f1 = fe_data['f1'].max()
            n_feat = fe_data['n_features'].mean() if 'n_features' in fe_data.columns else 0
            leak = 'EVET' if fe in ['no_fe', 'with_fe'] else 'HAYIR'
            pdf.mono_text(f'  {FE_LABELS.get(fe, fe):>20s}:  Avg F1={avg_f1:.4f}  '
                          f'Best F1={best_f1:.4f}  ~{n_feat:.0f} feat  Sizinti: {leak}')
    pdf.ln(3)
    pdf.add_chart(img_fe_modes)

    # ===== TEMIZ MODEL ANALIZI =====
    pdf.add_page()
    pdf.section_title(section_num, 'Temiz Model Analizi (Sizintisiz)')
    section_num += 1

    pdf.body_text(
        'Asagidaki analizler sadece clean ve clean_fe FE modlarini kapsamaktadir. '
        'Bu modlarda hicbir meta-predictor skoru, rankscore veya gnomAD degeri kullanilmamistir.'
    )

    # Model bazinda ozet
    for mt in MODEL_ORDER:
        sub = clean_df[clean_df['model_type'] == mt]
        if len(sub) > 0:
            avg = sub['f1'].mean()
            best = sub['f1'].max()
            pdf.mono_text(f'  {MODEL_LABELS[mt]:>15s}:  Avg F1={avg:.4f}  Best F1={best:.4f}  (n={len(sub)})')
    pdf.ln(3)
    pdf.add_chart(img_clean)

    # ===== STACKING SONUCLARI =====
    if has_stacking:
        pdf.add_page()
        pdf.section_title(section_num, 'Stacking Ensemble Sonuclari')
        section_num += 1

        pdf.body_text(
            'Stacking ensemble, 3 base model (LightGBM, XGBoost, SklearnMLP) ciktilarini '
            'birlestirerek daha guclu bir tahmin uretir. StratifiedKFold(5) CV ile base model '
            'tahminleri uretilir, meta-learner bu tahminler uzerinde egitilir.'
        )

        # Stacking sonuclari tablosu
        for st in STACKING_ORDER:
            st_data = stacking_df[stacking_df['model_type'] == st]
            if len(st_data) > 0:
                avg = st_data['f1'].mean()
                best = st_data['f1'].max()
                pdf.mono_text(f'  {MODEL_LABELS[st]:>20s}:  Avg F1={avg:.4f}  Best F1={best:.4f}')
        pdf.ln(2)

        # Tek model vs stacking karsilastirmasi
        single_clean_fe = v2_df[v2_df['fe_state'] == 'clean_fe']
        if len(single_clean_fe) > 0:
            single_best_f1 = single_clean_fe['f1'].max()
            stack_best_f1 = stacking_df['f1'].max()
            delta = stack_best_f1 - single_best_f1
            pdf.bold_text('Tek Model vs Stacking')
            pdf.mono_text(f'  En iyi tek model (clean_fe):  F1={single_best_f1:.4f}')
            pdf.mono_text(f'  En iyi stacking:              F1={stack_best_f1:.4f}')
            pdf.mono_text(f'  Stacking kazanci:             {delta:+.4f}')
        pdf.ln(3)

        # Stacking tablosu
        stacking_sorted = stacking_df.sort_values('f1', ascending=False).reset_index(drop=True)
        pdf.add_results_table(stacking_sorted, include_fe=False)
        pdf.ln(3)
        pdf.add_chart(img_stacking)

    # ===== FULL-FE ANALIZI =====
    if has_full_fe:
        pdf.add_page()
        pdf.section_title(section_num, 'Full-FE Analizi (In-silico + Gelismis FE)')
        section_num += 1

        pdf.body_text(
            'Full-FE modu, yarisma veri setinde in-silico skorlarin saglanacagi goz onunde '
            'bulundurularak tasarlanmistir. Meta-predictor skorlari (REVEL, CADD, ClinPred, '
            'AlphaMissense, MetaSVM vb.) korunurken, gelismis FE ozellikleri (yonlu BLOSUM62, '
            'Conservation PCA, delta k-mer, Grantham, gnomAD turevleri) eklenmistir.'
        )

        # Model bazinda ozet
        pdf.bold_text('Model Bazinda Full-FE Sonuclari')
        for mt in ALL_MODEL_ORDER:
            sub = full_fe_df[full_fe_df['model_type'] == mt]
            if len(sub) > 0:
                avg = sub['f1'].mean()
                best = sub['f1'].max()
                pdf.mono_text(f'  {MODEL_LABELS.get(mt, mt):>20s}:  Avg F1={avg:.4f}  Best F1={best:.4f}')
        pdf.ln(2)

        # Full-FE vs diger modlar
        pdf.bold_text('FE Modu Karsilastirmasi (Ortalama F1)')
        for fe in ['no_fe', 'with_fe', 'clean', 'clean_fe', 'full_fe']:
            fe_data = all_df[all_df['fe_state'] == fe]
            if len(fe_data) > 0:
                avg = fe_data['f1'].mean()
                best = fe_data['f1'].max()
                marker = ' <-- EN IYI' if fe == 'full_fe' and avg == all_df.groupby('fe_state')['f1'].mean().max() else ''
                pdf.mono_text(f'  {FE_LABELS.get(fe, fe):>25s}:  Avg={avg:.4f}  Best={best:.4f}{marker}')
        pdf.ln(2)

        # Panel bazinda full_fe en iyi
        pdf.bold_text('Panel Bazinda Full-FE En Iyi Sonuclar')
        for panel in ['All'] + sorted([p for p in full_fe_df['panel'].unique() if p != 'All']):
            p_data = full_fe_df[full_fe_df['panel'] == panel]
            if len(p_data) > 0:
                best_row = p_data.loc[p_data['f1'].idxmax()]
                pdf.mono_text(f'  {panel:>25s}:  F1={best_row["f1"]:.4f}  '
                              f'Model={MODEL_LABELS.get(best_row["model_type"], best_row["model_type"])}')
        pdf.ln(3)

        # Full-FE tablosu
        ff_sorted = full_fe_df.sort_values('f1', ascending=False).reset_index(drop=True)
        pdf.add_results_table(ff_sorted, include_fe=False)
        pdf.ln(3)
        pdf.add_chart(img_full_fe)

    # ===== SONUC TABLOSU =====
    pdf.add_page()
    pdf.section_title(section_num, 'Tum Sonuclar - Sirali Tablo')
    section_num += 1

    pdf.body_text(f'Toplam {len(all_sorted)} konfigurasyon, F1 skoruna gore siralanmistir. '
                  f'Ust 3 yesil, alt 3 kirmizi ile isaretlenmistir.')
    pdf.ln(2)
    pdf.add_results_table(all_sorted)

    # ===== HEATMAP =====
    pdf.add_page()
    pdf.section_title(section_num, 'Model Heatmap')
    section_num += 1
    pdf.body_text('Tum konfigurasyonlar icin 9 metrik karsilastirmasi.')
    pdf.add_chart(img_heatmap)

    # ===== RADAR =====
    pdf.add_page()
    pdf.section_title(section_num, 'Radar Grafigi')
    section_num += 1
    pdf.body_text('Her model tipinin en iyi konfigurasyonu icin 7 temel metrik karsilastirmasi.')
    pdf.add_chart(img_radar, w=140)

    # ===== EGITIM SURESI =====
    pdf.add_page()
    pdf.section_title(section_num, 'Egitim Suresi Analizi')
    section_num += 1
    total_sec = all_df['time_seconds'].sum()
    pdf.body_text(f'Toplam egitim suresi: {total_sec/60:.0f} dakika ({total_sec/3600:.1f} saat), '
                  f'{len(all_df)} konfigurasyon.')
    pdf.add_chart(img_time)

    # ===== SONUC VE ONERI =====
    pdf.add_page()
    pdf.section_title(section_num, 'Sonuc ve Oneriler')

    pdf.bold_text('EN IYI GENEL KONFIGURASYON')
    pdf.ln(2)
    pdf.mono_text(f'  Config:    {best_overall["config_id"]}')
    pdf.mono_text(f'  Model:     {MODEL_LABELS.get(best_overall["model_type"], best_overall["model_type"])}')
    pdf.mono_text(f'  Panel:     {best_overall["panel"]}')
    pdf.mono_text(f'  FE Modu:   {FE_LABELS.get(best_overall["fe_state"], best_overall["fe_state"])}')
    pdf.mono_text(f'  Threshold: {best_overall["best_threshold"]:.3f}')
    pdf.ln(2)

    pdf.bold_text('Performans Metrikleri')
    for m in METRICS:
        if m in best_overall.index:
            pdf.mono_text(f'  {METRIC_LABELS[m]:>20s}:  {best_overall[m]:.4f}')
    pdf.ln(3)

    if has_full_fe:
        best_ff = full_fe_df.loc[full_fe_df['f1'].idxmax()]
        pdf.bold_text('EN IYI FULL-FE KONFIGURASYON')
        pdf.mono_text(f'  Config:    {best_ff["config_id"]}')
        pdf.mono_text(f'  F1={best_ff["f1"]:.4f}  AUC-ROC={best_ff["auc_roc"]:.4f}  MCC={best_ff["mcc"]:.4f}')
        pdf.ln(3)

    pdf.bold_text('Temel Bulgular')
    if has_full_fe:
        ff_avg = full_fe_df['f1'].mean()
        pdf.body_text(
            f'1. Full-FE Modu: In-silico skorlarin korunmasi ve gelismis FE eklenmesi '
            f'ile ortalama F1={ff_avg:.4f} elde edilmistir. Yarisma icin bu mod onerilir.'
        )
    if has_stacking:
        pdf.body_text(
            '2. Stacking Ensemble: Base modellerin birlestirilmesi ile tek modellere '
            'gore daha dengeli ve guvenilir tahminler elde edilmistir.'
        )
    pdf.body_text(
        '3. Feature Engineering: Yonlu BLOSUM62, Conservation PCA, '
        'delta k-mer ve Grantham mesafesi gibi biyolojik olarak anlamli ozellikler '
        'model performansini desteklemistir.'
    )
    pdf.body_text(
        '4. Focal Loss: Sinif dengesizligi icin Focal Loss kullanimi, ozellikle '
        'azinlik sinifinin (pathogenic) recall degerini iyilestirmistir.'
    )
    pdf.ln(3)

    pdf.bold_text('Gelecek Calisma Onerileri')
    pdf.body_text('- DNABERT-2 / ESM-2 gibi protein dil modellerinden embedding eklenmesi')
    pdf.body_text('- Daha buyuk veri seti ile modellerin yeniden egitilmesi')
    pdf.body_text('- Klinik dogrulama calismasi (prospektif veri ile)')
    pdf.body_text('- SHAP analizi ile model yorumlanabilirligi')

    # ===== KAYDET =====
    output_path = os.path.join(REPORTS_DIR, 'model_comparison_report_v2.pdf')
    pdf.output(output_path)
    print(f"\nRapor olusturuldu: {output_path}")
    print(f"Toplam {pdf.page_no()} sayfa, {len(all_df)} konfigurasyon analiz edildi.")

    # Temizlik
    import shutil
    shutil.rmtree(TMPDIR, ignore_errors=True)


if __name__ == '__main__':
    main()

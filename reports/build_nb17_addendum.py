#!/usr/bin/env python
# coding: utf-8
"""
NB17 Regularization Sweep Addendum PDF Generator

Reads reg_sweep.csv (newly regenerated with robust thresholding) and produces
a PDF explaining what the metrics mean, reporting the table, embedding the figure,
and the VERDICT.
"""

import os, sys
import pandas as pd
from fpdf import FPDF
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v7_cftr_refinement")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load data
reg_sweep_csv = os.path.join(RESULTS_DIR, "reg_sweep.csv")
fig6_path = os.path.join(RESULTS_DIR, "fig6_reg_sweep.png")

df = pd.read_csv(reg_sweep_csv)

# ============================================================
# PDF Creation
# ============================================================

class NB17AddendumPDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=10)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "NB17 -- CFTR Regularizasyon Sweep Ek Rapor", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Sayfa {self.page_no()}", new_x="LMARGIN", new_y="NEXT")

    def section(self, title, body_text):
        """Add a section with title and body text."""
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4, body_text, align="L")
        self.ln(2)

pdf = NB17AddendumPDF()
pdf.add_page()

# --- Section 1: What is the Real Test F1? (ASCII-safe) ---
section1_title = "1. Real Test F1 Definition"
section1_body = (
    "Competition metric = pathogenic (Label=1) F1, ~80 percent benign / 20 percent pathogenic in test set. "
    "NB17 evaluation mimics this with boot_mean: FINAL_BENIGN_FRAC=0.80; "
    "_resample_8020() keeps all benign, downsamples pathogenic to 20 percent "
    "(n_pos = round(n_benign * 0.25)); _f1_pos() computes pos_label=1 pathogenic F1; "
    "bootstrap_8020() repeats this 50 times. => Our real test F1 estimate = boot_mean. "
    "CFTR S6: boot_mean = 0.692 +/- 0.088.\n\n"
    "Misleading numbers: train_f1 (~0.98) only shows training fit; "
    "loo_f1 (~0.92) is pathogenic F1 but measured in CFTR's own 81-percent-pathogenic distribution "
    "(not competition condition). Old report's train_f1 - loo_f1 'overfit' metric is misleading because "
    "it subtracts two DIFFERENT distributions: neither measures real overfit nor real test F1.\n\n"
    "Warning: boot_mean assumes our 21 benign examples represent test benign distribution; "
    "with n=21, uncertainty is high (std~0.09)."
)
pdf.section(section1_title, section1_body)

# --- Section 2: Regularization Sweep ---
section2_title = "2. Regularization Sweep (Does solving overfit hurt performance?)"
section2_body = (
    "Question: Does reducing overfit (memorization_gap) hurt CFTR performance "
    "(LOO-MCC / multi-seed F1)? Swept 12 configurations across two axes "
    "(min_child_samples and num_leaves). memorization_gap now de-noised via N=50 resample "
    "averaging. Finding: Excessive regularization "
    "(min_child_samples >= 80 or num_leaves in [63,7]) crashes multi-seed F1 from ~0.92 to "
    "0.38-0.75; CFTR LOO-MCC also drops. Current S6 setting "
    "(min_child_samples=20, num_leaves=31) sits safely at the plateau peak. "
    "Conclusion: Tightening regularization to solve overfit loses actual CFTR performance."
)
pdf.section(section2_title, section2_body)

# --- Section 3: Table ---
pdf.ln(2)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "3. Regularization Sweep Results", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 7)

# Create a simple table
col_headers = ["Axis", "Val", "Train F1", "Master CV F1", "Mem Gap", "LOO MCC", "LOO F1", "Boot Mean", "MS F1"]
col_widths = [16, 10, 13, 15, 12, 12, 12, 14, 12]

# Header row
for header, width in zip(col_headers, col_widths):
    pdf.cell(width, 5, header, border=1, align="C")
pdf.ln()

# Data rows
for idx, row in df.iterrows():
    axis_short = "MCS" if row["axis"] == "min_child_samples" else "NL"
    pdf.cell(16, 4, axis_short, border=1, align="C")
    pdf.cell(10, 4, str(int(row["value"])), border=1, align="C")
    pdf.cell(13, 4, f"{row['train_f1']:.4f}", border=1, align="C")
    pdf.cell(15, 4, f"{row['master_cv_f1']:.4f}", border=1, align="C")
    pdf.cell(12, 4, f"{row['memorization_gap']:.4f}", border=1, align="C")
    pdf.cell(12, 4, f"{row['loo_mcc']:.4f}", border=1, align="C")
    pdf.cell(12, 4, f"{row['loo_f1']:.4f}", border=1, align="C")
    pdf.cell(14, 4, f"{row['boot_mean']:.4f}", border=1, align="C")
    pdf.cell(12, 4, f"{row['ms_f1_mean']:.4f}", border=1, align="C")
    pdf.ln()

pdf.ln(3)

# --- Embed figure ---
if os.path.exists(fig6_path):
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "4. Regularization Strength vs Transfer Performance", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Get image dimensions and scale to page width
    img = Image.open(fig6_path)
    img_w_mm = 180  # ~page width minus margins
    pdf.image(fig6_path, x=15, y=pdf.get_y(), w=img_w_mm)
    pdf.ln(105)  # Space for image

pdf.ln(3)

# --- Verdict ---
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "5. VERDICT", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)

baseline_row = df[(df["axis"] == "min_child_samples") & (df["value"] == 20)].iloc[0]
baseline_loo_mcc = baseline_row["loo_mcc"]
baseline_mem_gap = baseline_row["memorization_gap"]

verdict_text = (
    f"Baseline (S0: min_child_samples=20, num_leaves=31):\n"
    f"  LOO-MCC = {baseline_loo_mcc:.4f}\n"
    f"  Memorization Gap = {baseline_mem_gap:.4f}\n\n"
    f"Best LOO-MCC in sweep: {df['loo_mcc'].max():.4f} "
    f"({df.loc[df['loo_mcc'].idxmax(), 'axis']}={int(df.loc[df['loo_mcc'].idxmax(), 'value'])})\n\n"
    f"VERDICT B/C: Regularization does not improve CFTR LOO-MCC -> current overfit 'gap' "
    f"is largely domain-shift; tightening constraints loses real CFTR performance. "
    f"Keep S6 as-is."
)

pdf.multi_cell(0, 4, verdict_text, align="L")

# --- Save PDF ---
pdf_path = os.path.join(REPORTS_DIR, "NB17_reg_sweep_addendum.pdf")
pdf.output(pdf_path)
print(f"PDF saved: {pdf_path}")
print(f"Baseline loo_mcc: {baseline_loo_mcc:.4f}")
print(f"Baseline memorization_gap: {baseline_mem_gap:.4f}")

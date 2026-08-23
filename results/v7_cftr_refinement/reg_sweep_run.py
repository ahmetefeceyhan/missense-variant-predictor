#!/usr/bin/env python
# coding: utf-8
"""
Standalone script: NB17 CFTR Regularization Sweep (validation runner)

Reproduces ONLY the data load + FE/M3 preprocessing (NB17 Cells 1,2,3) and the
evaluation infra (NB17 Cell 4) by copying that code verbatim, then runs the
regularization sweep (the new NB17 Cells 11+ logic) to produce real numbers
without re-executing the slow S4 NN LOO loop.

Run from notebooks/ dir:
    cd /Users/tefe/teknofest_model/teknofest_model/notebooks
    python ../results/v7_cftr_refinement/reg_sweep_run.py
"""

# ============================================================
# Cell 1: Imports & Config (verbatim subset from NB17)
# ============================================================
import os, sys, warnings, json
from copy import deepcopy
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut, cross_val_predict
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             matthews_corrcoef, confusion_matrix)
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if not os.path.exists(os.path.join(PROJECT_ROOT, "config.py")):
    PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

from config import SEED, REPORTS_DIR
from src import columns_real as CR
from src.models import LGBM_FIXED, CB_FIXED
from src.metrics import optimize_threshold
import lightgbm as lgb

np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

# --- Sabitler (NB17 Cell 1 ile ayni) ---
PANEL             = "CFTR"
MASTER_CORE_POS   = MASTER_CORE_NEG = 625
HIGH_MISSING_THR  = 0.50
FINAL_BENIGN_FRAC = 0.80
N_BOOT            = 50
BOOT_SEED         = SEED
N_MULTISEED       = 20
PANEL_SPLIT_FRAC  = 0.50
PI_TRAIN          = 0.73
PI_TEST           = 0.20

DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "real_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v7_cftr_refinement")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}  SEED: {SEED}")

# ============================================================
# Cell 2: Veri Yukleme + Sutun Temizligi (verbatim from NB17)
# ============================================================
ID_COL = CR.ID_COL
TARGET = CR.TARGET_COL

def load_panel(name):
    return pd.read_csv(os.path.join(DATA_DIR, CR.PANEL_INFO[name]["file"]))

master = load_panel("MASTER")
cftr_raw = load_panel("CFTR")
feature_cols_all = [c for c in master.columns if c not in (ID_COL, TARGET)]

# Cross-panel birebir-ayni satir drop (CFTR icin beklenen: 0)
master_index = {}
for _, row in master.iterrows():
    key = (row[ID_COL],) + tuple((np.nan if pd.isna(v) else v) for v in row[feature_cols_all].values)
    master_index[key] = row[TARGET]
drop_idx = [idx for idx, row in cftr_raw.iterrows()
            if (((row[ID_COL],) + tuple((np.nan if pd.isna(v) else v)
                 for v in row[feature_cols_all].values)) in master_index
                and master_index[(row[ID_COL],) + tuple((np.nan if pd.isna(v) else v)
                     for v in row[feature_cols_all].values)] == row[TARGET])]
cftr = cftr_raw.drop(index=drop_idx).reset_index(drop=True)
print(f"MASTER: {master.shape}, label: {master[TARGET].value_counts().to_dict()}")
print(f"CFTR  : {cftr.shape}, label: {cftr[TARGET].value_counts().to_dict()} (drop={len(drop_idx)})")

# Sutun temizligi: sabit + ozdes -- MASTER uzerinde (NB15/16 ile ayni 63 sutun)
constant_cols = CR.get_constant_cols(master[feature_cols_all])
dup_pairs     = CR.get_duplicate_col_pairs(master[feature_cols_all])
dup_drop      = sorted({b for (a, b) in dup_pairs})
drop_cols     = sorted(set(constant_cols) | set(dup_drop))
base_feature_cols = [c for c in feature_cols_all if c not in drop_cols]
CAT_LIKE      = [c for c in (CR.CAT_COLS + CR.AA_COLS) if c in base_feature_cols]
NUM_COLS_BASE = [c for c in base_feature_cols if c not in CAT_LIKE]

print(f"drop {len(drop_cols)} sutun -> {len(base_feature_cols)} ham feature "
      f"({len(NUM_COLS_BASE)} sayisal + {len(CAT_LIKE)} kategorik)")

# ============================================================
# Cell 3: FE Fonksiyonlari + M3 Preprocessing (verbatim from NB17)
# ============================================================
AA_UNK = CR.AA_UNKNOWN_TOKEN
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# --- Grantham (1974) mesafe matrisi ---
_GRANTHAM = {
 ('S','R'):110,('S','L'):145,('S','P'):74,('S','T'):58,('S','A'):99,('S','V'):124,
 ('S','G'):56,('S','I'):142,('S','F'):155,('S','Y'):144,('S','C'):112,('S','H'):89,
 ('S','Q'):68,('S','N'):46,('S','K'):121,('S','D'):65,('S','E'):80,('S','M'):135,('S','W'):177,
 ('R','L'):102,('R','P'):103,('R','T'):71,('R','A'):112,('R','V'):96,('R','G'):125,('R','I'):97,
 ('R','F'):97,('R','Y'):77,('R','C'):180,('R','H'):29,('R','Q'):43,('R','N'):86,('R','K'):26,
 ('R','D'):96,('R','E'):54,('R','M'):91,('R','W'):101,
 ('L','P'):98,('L','T'):92,('L','A'):96,('L','V'):32,('L','G'):138,('L','I'):5,('L','F'):22,
 ('L','Y'):36,('L','C'):198,('L','H'):99,('L','Q'):113,('L','N'):153,('L','K'):107,('L','D'):172,
 ('L','E'):138,('L','M'):15,('L','W'):61,
 ('P','T'):38,('P','A'):27,('P','V'):68,('P','G'):42,('P','I'):95,('P','F'):114,('P','Y'):110,
 ('P','C'):169,('P','H'):77,('P','Q'):76,('P','N'):91,('P','K'):103,('P','D'):108,('P','E'):93,
 ('P','M'):87,('P','W'):147,
 ('T','A'):58,('T','V'):69,('T','G'):59,('T','I'):89,('T','F'):103,('T','Y'):92,('T','C'):149,
 ('T','H'):47,('T','Q'):42,('T','N'):65,('T','K'):78,('T','D'):85,('T','E'):65,('T','M'):81,('T','W'):128,
 ('A','V'):64,('A','G'):60,('A','I'):94,('A','F'):113,('A','Y'):112,('A','C'):195,('A','H'):86,
 ('A','Q'):91,('A','N'):111,('A','K'):106,('A','D'):126,('A','E'):107,('A','M'):84,('A','W'):148,
 ('V','G'):109,('V','I'):29,('V','F'):50,('V','Y'):55,('V','C'):192,('V','H'):84,('V','Q'):96,
 ('V','N'):133,('V','K'):97,('V','D'):152,('V','E'):121,('V','M'):21,('V','W'):88,
 ('G','I'):135,('G','F'):153,('G','Y'):147,('G','C'):159,('G','H'):98,('G','Q'):87,('G','N'):80,
 ('G','K'):127,('G','D'):94,('G','E'):98,('G','M'):127,('G','W'):184,
 ('I','F'):21,('I','Y'):33,('I','C'):198,('I','H'):94,('I','Q'):109,('I','N'):149,('I','K'):102,
 ('I','D'):168,('I','E'):134,('I','M'):10,('I','W'):61,
 ('F','Y'):22,('F','C'):205,('F','H'):100,('F','Q'):116,('F','N'):158,('F','K'):102,('F','D'):177,
 ('F','E'):140,('F','M'):28,('F','W'):40,
 ('Y','C'):194,('Y','H'):83,('Y','Q'):99,('Y','N'):143,('Y','K'):85,('Y','D'):160,('Y','E'):122,
 ('Y','M'):36,('Y','W'):37,
 ('C','H'):174,('C','Q'):154,('C','N'):139,('C','K'):202,('C','D'):154,('C','E'):170,('C','M'):196,('C','W'):215,
 ('H','Q'):24,('H','N'):68,('H','K'):32,('H','D'):81,('H','E'):40,('H','M'):87,('H','W'):115,
 ('Q','N'):46,('Q','K'):53,('Q','D'):61,('Q','E'):29,('Q','M'):101,('Q','W'):130,
 ('N','K'):94,('N','D'):23,('N','E'):42,('N','M'):142,('N','W'):174,
 ('K','D'):101,('K','E'):56,('K','M'):95,('K','W'):110,
 ('D','E'):45,('D','M'):160,('D','W'):181,
 ('E','M'):126,('E','W'):152,
 ('M','W'):67,
}
def grantham(a, b):
    if a == b: return 0
    return _GRANTHAM.get((a, b)) or _GRANTHAM.get((b, a)) or -1

# --- BLOSUM62 ---
_B62_RAW = '''A4 R-1 N-2 D-2 C0 Q-1 E-1 G0 H-2 I-1 L-1 K-1 M-1 F-2 P-1 S1 T0 W-3 Y-2 V0
R5 N0 D-2 C-3 Q1 E0 G-2 H0 I-3 L-2 K2 M-1 F-3 P-2 S-1 T-1 W-3 Y-2 V-3
N6 D1 C-3 Q0 E0 G0 H1 I-3 L-3 K0 M-2 F-3 P-2 S1 T0 W-4 Y-2 V-3
D6 C-3 Q0 E2 G-1 H-1 I-3 L-4 K-1 M-3 F-3 P-1 S0 T-1 W-4 Y-3 V-3
C9 Q-3 E-4 G-3 H-3 I-1 L-1 K-3 M-1 F-2 P-3 S-1 T-1 W-2 Y-2 V-1
Q5 E2 G-2 H0 I-3 L-2 K1 M0 F-3 P-1 S0 T-1 W-2 Y-1 V-2
E5 G-2 H0 I-3 L-3 K1 M-2 F-3 P-1 S0 T-1 W-3 Y-2 V-2
G6 H-2 I-4 L-4 K-2 M-3 F-3 P-2 S0 T-2 W-2 Y-3 V-3
H8 I-3 L-3 K-1 M-2 F-1 P-2 S-1 T-2 W-2 Y2 V-3
I4 L2 K-3 M1 F0 P-3 S-2 T-1 W-3 Y-1 V3
L4 K-2 M2 F0 P-3 S-2 T-1 W-2 Y-1 V1
K5 M-1 F-3 P-1 S0 T-1 W-3 Y-2 V-2
M5 F0 P-2 S-1 T-1 W-1 Y-1 V1
F6 P-4 S-2 T-2 W1 Y3 V-1
P7 S-1 T-1 W-4 Y-3 V-2
S4 T1 W-3 Y-2 V-2
T5 W-2 Y-2 V0
W11 Y2 V-3
Y7 V-1
V4'''
_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_B62 = {}
for _ri, _line in enumerate(_B62_RAW.strip().split("\n")):
    _toks = _line.split()
    _row_aa = _toks[0][0]
    _vals = [_toks[0][1:]] + _toks[1:]
    for _ci, _tok in enumerate(_vals):
        _col_aa = _ORDER[_ri + _ci]
        _v = int(_tok[1:] if _tok[0].isalpha() else _tok)
        _B62[(_row_aa, _col_aa)] = _v; _B62[(_col_aa, _row_aa)] = _v
def blosum62(a, b): return _B62.get((a, b), 0)

def add_fe(df):
    out = df.copy()
    a1 = out["AA_1"].astype("object"); a2 = out["AA_2"].astype("object")
    out["fe_aa_stopgain"] = (a2 == "*").astype(int)
    def _nonstd(v): return 0 if (isinstance(v, str) and v in STANDARD_AA) else 1
    out["fe_aa_nonstandard"] = (a1.map(_nonstd) | a2.map(_nonstd)).astype(int)
    def _gr(r):
        x, y = r["AA_1"], r["AA_2"]
        if isinstance(x, str) and isinstance(y, str) and x in STANDARD_AA and y in STANDARD_AA:
            return grantham(x, y)
        return -1
    def _bl(r):
        x, y = r["AA_1"], r["AA_2"]
        if isinstance(x, str) and isinstance(y, str) and x in STANDARD_AA and y in STANDARD_AA:
            return blosum62(x, y)
        return 0
    out["fe_grantham"] = out.apply(_gr, axis=1).astype(float)
    out["fe_blosum62"] = out.apply(_bl, axis=1).astype(float)
    return out

FE_NEW_COLS = ["fe_aa_stopgain", "fe_aa_nonstandard", "fe_grantham", "fe_blosum62"]

def detect_log_cols(train_df):
    cand = []
    for c in NUM_COLS_BASE:
        s = pd.to_numeric(train_df[c], errors="coerce").dropna()
        if len(s) < 10: continue
        if s.min() >= 0 and s.max() > 1.0 and s.skew() > 2.0:
            cand.append(c)
    return cand

def fit_preprocessor(train_df_raw):
    tr = add_fe(train_df_raw)
    log_cols = detect_log_cols(tr)
    num_cols = NUM_COLS_BASE + FE_NEW_COLS + [f"{c}__log" for c in log_cols]
    for c in log_cols:
        tr[f"{c}__log"] = np.log1p(pd.to_numeric(tr[c], errors="coerce").clip(lower=0))
    miss = tr[base_feature_cols].isna().mean()
    flag_source = miss[miss > HIGH_MISSING_THR].index.tolist()
    median = {c: pd.to_numeric(tr[c], errors="coerce").median() for c in num_cols}
    return {"log_cols": log_cols, "num_cols": num_cols, "cat_cols": CAT_LIKE,
            "flag_source": flag_source, "median": median}

def transform_X(df_raw, pp):
    df = add_fe(df_raw)
    for c in pp["log_cols"]:
        df[f"{c}__log"] = np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0))
    out = pd.DataFrame(index=df.index)
    for c in pp["num_cols"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(pp["median"][c]).astype(float).values
    for c in pp["cat_cols"]:
        fill = AA_UNK if c in CR.AA_COLS else "MISSING"
        out[c] = df[c].astype("object").where(~df[c].isna(), fill).astype(str).values
    for c in pp["flag_source"]:
        out[CR.get_missing_mask_col_name(c)] = df[c].isna().astype(int).values
    return out, list(pp["cat_cols"])

# MASTER'da fit
pp = fit_preprocessor(master)
X_master_df, cat_cols = transform_X(master, pp)
y_master = master[TARGET].values
X_cftr_df, _ = transform_X(cftr, pp)
y_cftr = cftr[TARGET].values

print(f"FE+M3 sonrasi: {X_master_df.shape[1]} sutun (log:{len(pp['log_cols'])}, "
      f"flag:{len(pp['flag_source'])}, cat:{len(cat_cols)}) NaN:{X_master_df.isna().any().any()}")
print(f"CFTR X: {X_cftr_df.shape}  y: {np.bincount(y_cftr.astype(int))}")

# ============================================================
# Cell 4: Degerlendirme Altyapisi (verbatim from NB17)
# ============================================================

# --- 4a. Prior shift duzeltmesi (Saerens et al. 2002) ---
def adjust_prior_shift(proba, pi_train=PI_TRAIN, pi_test=PI_TEST):
    """Model olasiligini pi_train -> pi_test icin kalibre et."""
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    odds = proba / (1 - proba)
    R = (pi_test / (1 - pi_test)) / (pi_train / (1 - pi_train))
    odds_adj = odds * R
    return odds_adj / (1 + odds_adj)

# --- 4b. Bootstrap %80/20 (NB16 ile ayni) ---
def _f1_pos(y, p): return f1_score(y, p, pos_label=1, zero_division=0)

def _resample_8020(y, prob, rng):
    y = np.asarray(y); prob = np.asarray(prob)
    neg = np.where(y == 0)[0]; pos = np.where(y == 1)[0]
    if len(neg) == 0 or len(pos) == 0: return y, prob
    npos = max(1, int(round(len(neg) * (1 - FINAL_BENIGN_FRAC) / FINAL_BENIGN_FRAC)))
    keep = np.concatenate([neg, rng.choice(pos, size=npos, replace=True)])
    return y[keep], prob[keep]

def bootstrap_8020(y_te, p_te, thr, n=N_BOOT):
    rng = np.random.RandomState(BOOT_SEED); f1s = []
    for _ in range(n):
        yb, pb = _resample_8020(y_te, p_te, rng)
        f1s.append(_f1_pos(yb, (pb >= thr).astype(int)))
    f1s = np.array(f1s)
    return {"mean": float(f1s.mean()), "std": float(f1s.std()),
            "lo": float(np.percentile(f1s, 2.5)), "hi": float(np.percentile(f1s, 97.5))}

def select_threshold_8020(y, prob):
    """Tek 8020 yeniden orneklemede F1-max threshold sec."""
    rng = np.random.RandomState(BOOT_SEED)
    yb, pb = _resample_8020(y, prob, rng)
    best, best_thr = -1.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f = _f1_pos(yb, (pb >= thr).astype(int))
        if f > best: best, best_thr = f, thr
    return float(best_thr)

def select_threshold_8020_robust(y, prob, n_resample=N_BOOT):
    """N farkli %80/20 yeniden-ornekleme uzerinde ORTALAMA pathogenic-F1'i maksimize
    eden esik. Tek-resample gurultusunu giderir -> master_cv_f1/memorization_gap guvenilir."""
    thr_grid = np.arange(0.05, 0.95, 0.01)
    acc = np.zeros(len(thr_grid))
    for s in range(n_resample):
        rng = np.random.RandomState(BOOT_SEED + s)
        yb, pb = _resample_8020(y, prob, rng)
        for j, thr in enumerate(thr_grid):
            acc[j] += _f1_pos(yb, (pb >= thr).astype(int))
    return float(thr_grid[int(np.argmax(acc))])

# --- 4c. LOO-CV degerlendirme (tum 111 ornek kullanilir) ---
def loo_metrics(y_true, oof_proba, prior_shift=False):
    """LOO-CV olasiliklarindan MCC, F1, AUC-ROC hesapla.
    prior_shift=True ise adjust_prior_shift uygular."""
    prob = adjust_prior_shift(oof_proba) if prior_shift else oof_proba
    thr = select_threshold_8020(y_true, prob)
    y_pred = (prob >= thr).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    f1  = _f1_pos(y_true, y_pred)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.0
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    boot = bootstrap_8020(y_true, prob, thr)
    return {"mcc": mcc, "f1": f1, "auc": auc, "precision": prec, "recall": rec,
            "thr": thr, "boot8020": boot,
            "y_pred": y_pred, "y_true": y_true, "prob": prob}

# --- 4d. Multi-seed 50/50 degerlendirme ---
def multiseed_eval(fit_fn, n_seeds=N_MULTISEED):
    """n_seeds farkli 50/50 split uzerinde fit_fn(X_tr,y_tr,X_te,y_te)->proba cagir.
    fit_fn: (X_train_df, y_train, X_test_df, y_test) -> (proba_train, proba_test)
    Dondurulen: f1 ortalama, std, mcc ortalama."""
    f1s, mccs = [], []
    for seed in range(n_seeds):
        pos_idx = np.where(y_cftr == 1)[0]
        neg_idx = np.where(y_cftr == 0)[0]
        rng = np.random.RandomState(seed)
        rng.shuffle(pos_idx); rng.shuffle(neg_idx)
        nhalf_pos = len(pos_idx) // 2; nhalf_neg = len(neg_idx) // 2
        tr_idx = np.concatenate([pos_idx[:nhalf_pos], neg_idx[:nhalf_neg]])
        te_idx = np.concatenate([pos_idx[nhalf_pos:], neg_idx[nhalf_neg:]])
        X_tr = X_cftr_df.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_cftr[tr_idx]
        X_te = X_cftr_df.iloc[te_idx].reset_index(drop=True)
        y_te = y_cftr[te_idx]
        try:
            p_tr, p_te = fit_fn(X_tr, y_tr, X_te, y_te)
            thr = select_threshold_8020(y_tr, p_tr)
            yp = (p_te >= thr).astype(int)
            f1s.append(_f1_pos(y_te, yp))
            mccs.append(matthews_corrcoef(y_te, yp) if len(np.unique(y_te)) > 1 else 0.0)
        except Exception:
            pass
    return {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "mcc_mean": float(np.mean(mccs))}

# --- Train metrigi (overfit kontrolu icin) ---
def train_metrics_at(y_tr, p_tr, thr):
    yp = (p_tr >= thr).astype(int)
    return {
        "train_f1": _f1_pos(y_tr, yp),
        "train_mcc": matthews_corrcoef(y_tr, yp) if len(np.unique(y_tr)) > 1 else 0.0,
        "train_precision": precision_score(y_tr, yp, pos_label=1, zero_division=0),
        "train_recall": recall_score(y_tr, yp, pos_label=1, zero_division=0),
    }

# --- Cell 5 (yalniz LOO-label-encode yardimcisi gerekli) ---
def _le_encode_for_loo(X_df):
    """Kategorik sutunlari label-encode et (LOO icin categorical_feature verilemez)."""
    Xn = X_df.copy()
    le_maps = {}
    for c in cat_cols:
        le = LabelEncoder()
        Xn[c] = le.fit_transform(Xn[c].astype(str))
        le_maps[c] = le
    return Xn.astype(float), le_maps

X_cftr_le, le_maps_cftr = _le_encode_for_loo(X_cftr_df)
X_master_le, _          = _le_encode_for_loo(X_master_df)

print("Degerlendirme altyapisi hazir.")


# ============================================================
# Cell 11: Regularizasyon Sweep -- Tasarim + Yardimci Fonksiyon
# (NEW -- mirrors notebook Cell 11)
# ============================================================

def _lgbm_reg(**ov):
    base = {**LGBM_FIXED, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31}
    base.update(ov)
    return lgb.LGBMClassifier(**base)

AXIS_A_VALUES = [5, 10, 20, 40, 80, 160, 320]   # min_child_samples sweep @ num_leaves=31
AXIS_B_VALUES = [63, 31, 15, 7, 3]               # num_leaves sweep @ min_child_samples=20

BASELINE_MCS = 20
BASELINE_NL  = 31

sweep_configs = []
for v in AXIS_A_VALUES:
    sweep_configs.append(("min_child_samples", v, {"min_child_samples": v, "num_leaves": BASELINE_NL}))
for v in AXIS_B_VALUES:
    sweep_configs.append(("num_leaves", v, {"min_child_samples": BASELINE_MCS, "num_leaves": v}))

print(f"Sweep konfigurasyon sayisi: {len(sweep_configs)}")


# ============================================================
# Cell 12: Regularizasyon Sweep -- Calistir
# (NEW -- mirrors notebook Cell 12)
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

sweep_rows = []
for axis, value, ov in sweep_configs:
    print(f"[{axis}={value}] egitiliyor...")
    params = ov

    # 1) Tum MASTER uzerinde fit -> train_f1
    m_full = _lgbm_reg(**params)
    m_full.fit(X_master_le, y_master)
    p_master_train = m_full.predict_proba(X_master_le)[:, 1]
    thr_train = select_threshold_8020_robust(y_master, p_master_train)
    train_m = train_metrics_at(y_master, p_master_train, thr_train)
    train_f1 = train_m["train_f1"]

    # 2) 5-fold StratifiedKFold OOF on MASTER -> master_cv_f1 / master_cv_mcc
    oof_master = np.zeros(len(y_master), dtype=float)
    for tr_idx, va_idx in skf.split(X_master_le, y_master):
        m_cv = _lgbm_reg(**params)
        m_cv.fit(X_master_le.iloc[tr_idx], y_master[tr_idx])
        oof_master[va_idx] = m_cv.predict_proba(X_master_le.iloc[va_idx])[:, 1]
    thr_cv = select_threshold_8020_robust(y_master, oof_master)
    yp_cv = (oof_master >= thr_cv).astype(int)
    master_cv_f1 = _f1_pos(y_master, yp_cv)
    master_cv_mcc = matthews_corrcoef(y_master, yp_cv)

    memorization_gap = train_f1 - master_cv_f1

    # 3) CFTR transfer (S6-style, prior-shift ON)
    p_cftr = m_full.predict_proba(X_cftr_le)[:, 1]
    res = loo_metrics(y_cftr, p_cftr, prior_shift=True)
    loo_mcc = res["mcc"]; loo_f1 = res["f1"]; loo_auc = res["auc"]
    boot_mean = res["boot8020"]["mean"]; boot_std = res["boot8020"]["std"]

    transfer_gap = master_cv_f1 - loo_f1

    # 4) Multi-seed (S6 _s6_fit_fn pattern, swept params)
    def _fit_fn(Xtr_df, ytr, Xte_df, yte, _ov=params):
        m = _lgbm_reg(**_ov); m.fit(X_master_le, y_master)
        Xte_le, _ = _le_encode_for_loo(Xte_df)
        p_tr_raw = m.predict_proba(X_master_le)[:, 1][:len(ytr)]
        p_te_raw = m.predict_proba(Xte_le)[:, 1]
        return adjust_prior_shift(p_tr_raw), adjust_prior_shift(p_te_raw)
    ms = multiseed_eval(_fit_fn)
    ms_f1_mean = ms["f1_mean"]; ms_f1_std = ms["f1_std"]

    sweep_rows.append({
        "axis": axis, "value": value,
        "train_f1": round(train_f1, 4),
        "master_cv_f1": round(master_cv_f1, 4),
        "master_cv_mcc": round(master_cv_mcc, 4),
        "memorization_gap": round(memorization_gap, 4),
        "loo_mcc": round(loo_mcc, 4),
        "loo_f1": round(loo_f1, 4),
        "loo_auc": round(loo_auc, 4),
        "boot_mean": round(boot_mean, 4),
        "boot_std": round(boot_std, 4),
        "transfer_gap": round(transfer_gap, 4),
        "ms_f1_mean": round(ms_f1_mean, 4),
        "ms_f1_std": round(ms_f1_std, 4),
    })
    print(f"  train_f1={train_f1:.4f}  master_cv_f1={master_cv_f1:.4f}  "
          f"loo_mcc={loo_mcc:.4f}  loo_f1={loo_f1:.4f}  "
          f"mem_gap={memorization_gap:.4f}  transfer_gap={transfer_gap:.4f}")

sweep_df = pd.DataFrame(sweep_rows)
sweep_csv = os.path.join(RESULTS_DIR, "reg_sweep.csv")
sweep_df.to_csv(sweep_csv, index=False)
print(f"\nSweep sonuclari kaydedildi: {sweep_csv}")
print(sweep_df.to_string(index=False))


# ============================================================
# Cell 13: Figur -- fig6_reg_sweep.png
# (NEW -- mirrors notebook Cell 13)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

axis_specs = [
    ("min_child_samples", AXIS_A_VALUES, BASELINE_MCS),
    ("num_leaves", AXIS_B_VALUES, BASELINE_NL),
]

for ax, (axis_name, values, baseline_val) in zip(axes, axis_specs):
    sub = sweep_df[sweep_df["axis"] == axis_name].sort_values("value")
    x = sub["value"].values

    # Left axis: loo_mcc + ms_f1_mean band
    l1, = ax.plot(x, sub["loo_mcc"], "o-", color="#1f77b4", label="CFTR LOO-MCC (prior-shift)", linewidth=2)
    l2, = ax.plot(x, sub["ms_f1_mean"], "s--", color="#2ca02c", label="Multi-seed F1 (mean)", linewidth=1.5)
    ax.fill_between(x, sub["ms_f1_mean"] - sub["ms_f1_std"], sub["ms_f1_mean"] + sub["ms_f1_std"],
                    color="#2ca02c", alpha=0.15)
    ax.set_xlabel(f"{axis_name} (artan regularizasyon ->)", fontsize=10)
    ax.set_ylabel("CFTR Skoru (MCC / F1)", fontsize=10)
    ax.set_ylim(-0.2, 1.05)

    # Right axis: gaps
    axr = ax.twinx()
    l3, = axr.plot(x, sub["memorization_gap"], "^:", color="#d62728", label="memorization_gap (train-mastercv)", linewidth=1.5)
    l4, = axr.plot(x, sub["transfer_gap"], "v:", color="#9467bd", label="transfer_gap (mastercv-loo)", linewidth=1.5)
    axr.set_ylabel("Gap (F1 farki)", fontsize=10)
    axr.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # Baseline marker
    ax.axvline(baseline_val, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(baseline_val, ax.get_ylim()[1] * 0.97, " baseline\n (S0)", fontsize=8,
            ha="left" if axis_name == "min_child_samples" else "right", va="top")

    if axis_name == "num_leaves":
        ax.invert_xaxis()  # increasing regularization = decreasing num_leaves

    ax.set_title(axis_name, fontweight="bold")
    ax.grid(alpha=0.2)

    lines = [l1, l2, l3, l4]
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="lower left")

fig.suptitle("NB17 -- CFTR: Regularizasyon Gucu vs Transfer Basarisi\n"
              "(Overfit'i cozmek basari kaybettiriyor mu?)", fontweight="bold")
plt.tight_layout()
fig6_path = os.path.join(RESULTS_DIR, "fig6_reg_sweep.png")
fig.savefig(fig6_path, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Figur kaydedildi: {fig6_path}")


# ============================================================
# Cell 14: Verdict -- Otomatik Karar
# (NEW -- mirrors notebook Cell 14)
# ============================================================

baseline_row = sweep_df[(sweep_df["axis"] == "min_child_samples") & (sweep_df["value"] == BASELINE_MCS)]
if len(baseline_row) == 0:
    baseline_row = sweep_df[(sweep_df["axis"] == "num_leaves") & (sweep_df["value"] == BASELINE_NL)]
baseline_row = baseline_row.iloc[0]
baseline_loo_mcc = baseline_row["loo_mcc"]
baseline_mem_gap = baseline_row["memorization_gap"]
baseline_transfer_gap = baseline_row["transfer_gap"]

# "More regularized" configs: higher min_child_samples than baseline, OR lower num_leaves than baseline
more_reg_mask = (
    ((sweep_df["axis"] == "min_child_samples") & (sweep_df["value"] > BASELINE_MCS)) |
    ((sweep_df["axis"] == "num_leaves") & (sweep_df["value"] < BASELINE_NL))
)
more_reg_df = sweep_df[more_reg_mask]

best_idx = sweep_df["loo_mcc"].idxmax()
best_row = sweep_df.loc[best_idx]
best_loo_mcc = best_row["loo_mcc"]

best_more_reg_idx = more_reg_df["loo_mcc"].idxmax() if len(more_reg_df) > 0 else None
best_more_reg_row = sweep_df.loc[best_more_reg_idx] if best_more_reg_idx is not None else None

print("="*70)
print("NB17 -- REGULARIZASYON SWEEP VERDICT")
print("="*70)
print(f"\nBaseline (S0: min_child_samples={BASELINE_MCS}, num_leaves={BASELINE_NL}):")
print(f"  loo_mcc           = {baseline_loo_mcc:.4f}")
print(f"  memorization_gap  = {baseline_mem_gap:.4f}  (train_f1 - master_cv_f1, GERCEK overfit)")
print(f"  transfer_gap      = {baseline_transfer_gap:.4f}  (master_cv_f1 - loo_f1, MASTER->CFTR domain shift)")
if abs(baseline_mem_gap) >= abs(baseline_transfer_gap):
    print(f"  -> Baseline'da MEMORIZATION_GAP daha buyuk/esit ({baseline_mem_gap:.4f} vs {baseline_transfer_gap:.4f})")
else:
    print(f"  -> Baseline'da TRANSFER_GAP daha buyuk ({baseline_transfer_gap:.4f} vs {baseline_mem_gap:.4f}) "
          f"-> 'overfit' gap'inin cogu domain-shift kaynakli, gercek memorization kucuk.")

print(f"\nSweep genelinde en iyi loo_mcc: {best_loo_mcc:.4f}  ({best_row['axis']}={best_row['value']})")
if best_more_reg_row is not None:
    print(f"Daha regularize konfigurasyonlar icinde en iyi loo_mcc: "
          f"{best_more_reg_row['loo_mcc']:.4f}  ({best_more_reg_row['axis']}={best_more_reg_row['value']}, "
          f"memorization_gap={best_more_reg_row['memorization_gap']:.4f})")

print(f"\n[UYARI] n=21 benign -- yalniz |loo_mcc farki| > 0.05 olan kaymalara gore aksiyon al; "
      f"0.02-0.05 'oneri' niteliginde, < 0.02 gurultu.")

verdict = None
if best_more_reg_row is not None:
    delta = best_more_reg_row["loo_mcc"] - baseline_loo_mcc
    if delta > 0.02 and best_more_reg_row["memorization_gap"] < baseline_mem_gap:
        verdict = ("A", f"{best_more_reg_row['axis']}={best_more_reg_row['value']}")
    else:
        # check overall best regularized configs <= baseline + 0.02
        delta_all = (more_reg_df["loo_mcc"] - baseline_loo_mcc)
        if (delta_all <= 0.02).all():
            verdict = ("B/C", None)
        else:
            verdict = ("B/C", None)  # default conservative
else:
    verdict = ("B/C", None)

print("\n" + "-"*70)
if verdict[0] == "A":
    print(f"VERDICT A: {verdict[1]} hem overfit'i azaltiyor hem CFTR LOO-MCC'yi artiriyor "
          f"-> bedava kazanc, S6'yi bu config ile guncelle.")
else:
    print("VERDICT B/C: Regularizasyon CFTR LOO-MCC'yi artirmiyor -> mevcut overfit 'gap'i "
          "buyuk olcude domain-shift; sikilastirmak gercek basari kaybettirir. "
          "S6 mevcut haliyle birak.")
print("-"*70)

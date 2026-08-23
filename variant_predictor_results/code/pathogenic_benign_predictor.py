#!/usr/bin/env python3
"""
=============================================================================
END-TO-END PATHOGENIC / BENIGN VARIANT PREDICTOR
=============================================================================
Dataset  : open_cravat_curation_v3.csv
Target   : Pathogenic (1) vs Benign (0)
           Derived from clinvar__sig:
             Pathogenic | Likely pathogenic  → 1
             Benign     | Likely benign      → 0

Encoding experiments : 2-mer and 3-mer k-mer encoding for 11-mer sequences
One-hot encoding     : DNA base ref/alt, amino-acid ref/alt
Label encoding       : Categorical tool-prediction columns

Models               : XGBoost · LightGBM · 3-hidden-layer MLP (PyTorch)
Hyper-param opt      : Grid search (small fixed grids, 3-fold CV)

Experiments
  1. All panels  → 5-fold stratified CV
  2. Leave-one-panel-out  → train on 3 panels, test on held-out panel
  3. Single-panel generalisation → train on one panel, test on each other

Outputs
  feature_registry.json  – complete audit of feature decisions
  results.json           – all experiment results

Note: Run on Google Colab A100 (High-RAM). GPU is auto-detected.
=============================================================================

# ---- Google Colab install block (uncomment if needed) -------------------
# !pip install xgboost lightgbm torch --quiet
# -------------------------------------------------------------------------
"""

# =============================================================================
# 0.  IMPORTS
# =============================================================================
import os
import sys
import json
import time
import warnings
import logging
from itertools import product
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report,
)

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # DataLoader kept for CPU fallback

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# 1.  CONFIGURATION
# =============================================================================

DATA_PATH          = "open_cravat_curation_v3.csv"
FEATURE_LOG_PATH   = "feature_registry.json"
RESULTS_PATH       = "results.json"

RANDOM_SEED        = 42
N_CV_SPLITS        = 5          # folds for Experiment 1
N_GRID_CV_SPLITS   = 3          # folds used inside grid search (speed)
EARLY_STOP_ROUNDS  = 40         # for XGBoost / LightGBM early stopping
NN_PATIENCE        = 15         # epochs without val-loss improvement → stop
# Parallel workers: grid combos run concurrently, models run concurrently.
# 80 GB A100 can easily hold 3 simultaneous models + multiple grid combos.
N_PARALLEL_GRID    = 8          # all grid combos at once — 80 GB can hold them
N_PARALLEL_MODELS  = 3          # XGBoost + LightGBM + NN in parallel

KMER_SIZES         = [2, 3]

# Exact panel names as they appear in the dataset
PANELS             = ["General", "PAH", "Hereditary_Cancer", "CFTR"]

USE_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if USE_GPU else "cpu")

# A100 / modern GPU: let cuDNN auto-tune fastest conv/matmul algorithms
if USE_GPU:
    torch.backends.cudnn.benchmark = True

# ---- Build XGBoost device kwargs using version-based detection ----
# XGBoost >= 2.0 uses device="cuda"; older versions use tree_method="gpu_hist"
_xgb_major = int(xgb.__version__.split(".")[0])
if USE_GPU:
    XGB_DEVICE_KWARGS = {"device": "cuda"} if _xgb_major >= 2 else {"tree_method": "gpu_hist"}
else:
    XGB_DEVICE_KWARGS = {"device": "cpu"}  if _xgb_major >= 2 else {"tree_method": "hist"}

LGBM_DEVICE = "gpu" if USE_GPU else "cpu"

def _log_gpu_config():
    log.info("=" * 50)
    log.info(f"Device        : {DEVICE}")
    if USE_GPU:
        log.info(f"GPU           : {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info(f"XGBoost       : v{xgb.__version__}  kwargs={XGB_DEVICE_KWARGS}")
    log.info(f"LightGBM      : v{lgb.__version__}  device={LGBM_DEVICE}")
    log.info(f"AMP (autocast): {'enabled' if USE_GPU else 'disabled (CPU)'}")
    log.info("=" * 50)

_log_gpu_config()


# =============================================================================
# 2.  FEATURE DEFINITIONS  (audit trail for feature_registry.json)
# =============================================================================

# ---------- features to drop (with documented reason) ----------
COLS_TO_DROP: dict[str, list[str]] = {
    "target_source__data_leakage": [
        "clinvar__sig",          # used to derive the label — must be removed
    ],
    "identifier_columns": [
        "clinvar__id",           # ClinVar variant ID
        "base__chrom",           # chromosome (genomic coordinate)
        "base__pos",             # genomic position (coordinate)
        "base__hugo",            # gene symbol (hundreds of categories)
        "base__cchange",         # HGVS cDNA notation (text ID)
        "base__achange",         # HGVS protein notation (text ID)
        "alphamissense__protein_variant",  # variant notation (e.g. P2T)
    ],
    "long_raw_sequences__replaced_by_11mers": [
        "Ref_Sequence",          # full-length flanking sequence
        "Alt_Sequence",          # full-length flanking sequence (alt)
    ],
    "review_metadata__not_predictive": [
        "Germline review status",  # ClinVar curation status text
        "Stars",                   # ClinVar review star rating
    ],
    "experiment_split_column": [
        "Panel",                 # used to define train/test splits
    ],
    "existing_target__re_derived": [
        "target",                # original column; we re-derive per spec
    ],
}

# ---------- 11-mer sequence columns → k-mer encoding ----------
DNA_SEQ_COLS  = ["DNA_11mer_Ref",  "DNA_11mer_Alt"]
PROT_SEQ_COLS = ["Prot_11mer_Ref", "Prot_11mer_Alt"]

DNA_ALPHABET  = list("ACGT")
PROT_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY_")   # 20 AA + gap/pad char

# ---------- one-hot encoding columns (fixed alphabets) ----------
OHE_COLS: dict[str, list[str]] = {
    "base__ref_base": list("ACGT"),
    "base__alt_base": list("ACGT"),
    "ref_amino":      list("ACDEFGHIKLMNPQRSTVWY"),
    "alt_amino":      list("ACDEFGHIKLMNPQRSTVWY"),
}

# ---------- binary bool columns → 0 / 1 ----------
BOOL_COLS = ["polarity_change", "chirality_shift"]

# ---------- categorical tool-prediction columns → label-encoded ----------
LABEL_ENC_COLS = [
    "esm1b__prediction",         # Tolerated / Deleterious
    "metalr__pred",              # Tolerated / Damaging
    "metarnn__pred",             # Tolerated / Damaging
    "metasvm__pred",             # Tolerated / Damaging
    "mistic__pred",              # Benign / Deleterious
    "mutationtaster__prediction",# Damaging / Polymorphism / Automatic ...
    "phdsnpg__prediction",       # Pathogenic / Benign
    "provean__prediction",       # Damaging / Neutral
    "sift__prediction",          # Damaging / Tolerated
    "alphamissense__am_class",   # likely_benign / likely_pathogenic / ambiguous
]


# =============================================================================
# 3.  DATA LOADING & TARGET DERIVATION
# =============================================================================

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Loaded  {df.shape[0]} rows × {df.shape[1]} columns from '{path}'")
    return df


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-derive binary label from clinvar__sig:
        Pathogenic | Likely pathogenic  → 1
        Benign     | Likely benign      → 0
    All four values are present in the dataset; no rows need to be dropped.
    """
    mapping = {
        "Pathogenic":        1,
        "Likely pathogenic": 1,
        "Benign":            0,
        "Likely benign":     0,
    }
    df = df.copy()
    df["label"] = df["clinvar__sig"].map(mapping)
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    log.info(
        f"Label derived: {before}→{len(df)} rows  "
        f"| Pos={df['label'].sum()}  Neg={(df['label']==0).sum()}"
    )
    return df


# =============================================================================
# 4.  FEATURE ENGINEERING
# =============================================================================

# ---- 4a.  K-mer encoding ------------------------------------------------

def _build_kmer_vocab(sequences: pd.Series, k: int, alphabet: list[str]) -> list[str]:
    """
    Build vocabulary of k-mers that actually appear in the dataset.
    Using observed vocab (not full combinatorial) keeps protein 3-mers tractable.
    For DNA k=2 (16) and k=3 (64) the full vocab is small and is used directly.
    """
    if set(alphabet) <= set("ACGT"):            # DNA: use full combinatorial vocab
        return ["".join(p) for p in product(alphabet, repeat=k)]
    # Protein: use observed vocab only
    vocab: set[str] = set()
    for seq in sequences.dropna():
        seq = str(seq).upper()
        for i in range(len(seq) - k + 1):
            vocab.add(seq[i : i + k])
    return sorted(vocab)


def _encode_kmer_col(
    series: pd.Series, k: int, vocab: list[str]
) -> pd.DataFrame:
    """Count k-mer occurrences in each sequence → one column per k-mer."""
    rows = []
    for seq in series:
        seq = str(seq).upper() if isinstance(seq, str) else ""
        counts = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
        rows.append({km: counts.get(km, 0) for km in vocab})
    return pd.DataFrame(rows, index=series.index)


def encode_sequences_kmer(
    df: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    K-mer encode all 11-mer sequence columns.
    Returns concatenated feature DataFrame and vocab mapping for the registry.
    """
    parts, vocabs = [], {}

    for col, alphabet in [(c, DNA_ALPHABET) for c in DNA_SEQ_COLS] + \
                         [(c, PROT_ALPHABET) for c in PROT_SEQ_COLS]:
        vocab = _build_kmer_vocab(df[col], k, alphabet)
        kmer_df = _encode_kmer_col(df[col], k, vocab)
        kmer_df.columns = [f"{col}__{k}mer_{v}" for v in vocab]
        parts.append(kmer_df)
        vocabs[col] = vocab

    return pd.concat(parts, axis=1), vocabs


# ---- 4b.  One-hot encoding -----------------------------------------------

def encode_ohe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode DNA base and amino-acid columns with fixed alphabets."""
    parts, col_names = [], []
    for col, cats in OHE_COLS.items():
        for cat in cats:
            col_name = f"{col}__{cat}"
            parts.append((df[col] == cat).astype(float).rename(col_name))
            col_names.append(col_name)
    return pd.concat(parts, axis=1), col_names


# ---- 4c.  Label encoding (categorical prediction columns) ----------------

def encode_label_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical tool-prediction columns.  Fit on full data."""
    df = df.copy()
    encoders = {}
    for col in LABEL_ENC_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {int(i): str(c) for i, c in enumerate(le.classes_)}
    return df, encoders


# ---- 4d.  Full preprocessing pipeline ------------------------------------

def build_feature_matrix(
    df: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Full preprocessing for k-mer size k.
    Returns
        X_df    : feature DataFrame (unscaled)
        y       : label Series
        panel   : Panel Series (for experiment splits)
        registry: dict with feature audit info
    """
    df = df.copy()

    # --- extract meta-columns before dropping ---
    y     = df["label"].copy()
    panel = df["Panel"].copy()

    # --- bool → int ---
    for col in BOOL_COLS:
        df[col] = df[col].astype(int)

    # --- label-encode categorical prediction columns ---
    df, le_mappings = encode_label_cols(df)

    # --- OHE for bases / amino acids ---
    ohe_df, ohe_col_names = encode_ohe(df)

    # --- k-mer encode sequences ---
    kmer_df, kmer_vocabs = encode_sequences_kmer(df, k)

    # --- collect all columns to remove ---
    all_drop = []
    for cols_list in COLS_TO_DROP.values():
        all_drop.extend(cols_list)
    # also drop source columns that we've encoded
    all_drop += DNA_SEQ_COLS + PROT_SEQ_COLS + list(OHE_COLS.keys())
    # also drop derived target and label cols to prevent leakage
    all_drop += ["label", "Panel"]
    # safety: only drop columns that actually exist
    all_drop = [c for c in all_drop if c in df.columns]
    df = df.drop(columns=all_drop)

    # --- concatenate everything ---
    X_df = pd.concat([df, ohe_df, kmer_df], axis=1)

    # --- drop any remaining non-numeric columns, then cast to float ---
    obj_cols = X_df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        log.warning(f"  Dropping unconverted object columns: {obj_cols}")
        X_df = X_df.drop(columns=obj_cols)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    # --- build feature registry ---
    registry = {
        "kmer_size":             k,
        "n_samples":             int(len(X_df)),
        "n_features_total":      int(X_df.shape[1]),
        "dropped_features":      COLS_TO_DROP,
        "bool_features":         BOOL_COLS,
        "ohe_features":          ohe_col_names,
        "label_encoded_features": le_mappings,
        "kmer_vocabs":           kmer_vocabs,
        "numeric_features": [
            c for c in df.columns
            if c not in BOOL_COLS + LABEL_ENC_COLS
        ],
        "final_feature_list":    X_df.columns.tolist(),
    }

    log.info(
        f"[k={k}]  Feature matrix: {X_df.shape}  "
        f"| OHE={len(ohe_col_names)}  "
        f"| k-mer cols={kmer_df.shape[1]}"
    )
    return X_df, y, panel, registry


# =============================================================================
# 5.  MODEL DEFINITIONS
# =============================================================================

# ---- 5a.  3-Layer MLP (PyTorch) -----------------------------------------

class MLP(nn.Module):
    """Shallow 3-hidden-layer MLP with BatchNorm, ReLU, Dropout."""

    def __init__(self, input_dim: int, h1: int, h2: int, h3: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    params:  dict,
) -> MLP:
    # ------------------------------------------------------------------ #
    # Move EVERYTHING to GPU upfront. With 4k samples the whole dataset   #
    # fits in <50 MB — far cheaper than per-batch CPU→GPU transfers.       #
    # Manual GPU-side shuffling replaces the DataLoader entirely.          #
    # ------------------------------------------------------------------ #
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_v  = torch.FloatTensor(X_val).to(DEVICE)
    y_v  = torch.FloatTensor(y_val).to(DEVICE)

    n  = X_tr.shape[0]
    bs = min(params["batch_size"], n - 1)   # guard against tiny panels

    model = MLP(
        X_train.shape[1],
        params["h1"], params["h2"], params["h3"],
        params["dropout"],
    ).to(DEVICE)

    # torch.compile disabled — L4 lacks enough SMs and parallel threads cause
    # recompile-limit corruption; eager mode is stable and fast enough here.

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["epochs"]
    )
    criterion  = nn.BCEWithLogitsLoss()
    amp_ctx    = torch.amp.autocast("cuda") if USE_GPU else torch.amp.autocast("cpu", enabled=False)
    grad_scaler = torch.amp.GradScaler("cuda") if USE_GPU else None

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(params["epochs"]):
        model.train()
        # GPU-side shuffle: no CPU involvement at all
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n - bs, bs):          # drop_last equivalent
            idx = perm[start : start + bs]
            Xb  = X_tr[idx]
            yb  = y_tr[idx]
            # Light Gaussian noise augmentation — keeps GPU busy & regularises
            if USE_GPU:
                Xb = Xb + torch.randn_like(Xb) * 0.005

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = criterion(model(Xb), yb)
            if grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad(), amp_ctx:
            val_loss = criterion(model(X_v), y_v).item()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= NN_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    if USE_GPU:
        torch.cuda.empty_cache()
    return model


def predict_nn(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t     = torch.FloatTensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)).to(DEVICE)
    amp_ctx = torch.amp.autocast("cuda") if USE_GPU else torch.amp.autocast("cpu", enabled=False)
    with torch.no_grad(), amp_ctx:
        probs = torch.sigmoid(model(X_t)).cpu().float().numpy()
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return probs


# ---- 5b.  XGBoost helpers ------------------------------------------------
# XGB_DEVICE_KWARGS is set at the top of section 1 (version-based detection).

def train_xgb(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v:  np.ndarray, y_v:  np.ndarray,
    params: dict,
) -> xgb.XGBClassifier:
    X_tr = np.nan_to_num(np.asarray(X_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_v  = np.nan_to_num(np.asarray(X_v,  dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # early_stopping_rounds moved to constructor in XGBoost >= 1.6
    model = xgb.XGBClassifier(
        **params,
        **XGB_DEVICE_KWARGS,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        verbose=False,
    )
    return model


def train_lgbm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_v:  np.ndarray, y_v:  np.ndarray,
    params: dict,
) -> lgb.LGBMClassifier:
    X_tr = np.nan_to_num(np.asarray(X_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_v  = np.nan_to_num(np.asarray(X_v,  dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    model = lgb.LGBMClassifier(
        **params,
        device_type  = LGBM_DEVICE,
        n_jobs       = -1,           # all CPU threads for histogram building
        random_state = RANDOM_SEED,
        verbose      = -1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    return model


# =============================================================================
# 6.  GRID SEARCH HYPERPARAMETER OPTIMISATION
# =============================================================================
#
# Small, fixed grids: 8 combos per model.  Evaluated with 3-fold CV (AUROC).
# Scaling is done inside each fold so there is no leakage.
# ─────────────────────────────────────────────────────────────────────────────

XGB_GRID = [
    {"n_estimators": ne, "max_depth": md, "learning_rate": lr,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3}
    for ne in [50, 100, 200]
    for md in [4, 6]
    for lr in [0.05, 0.1]
]  # 3 × 2 × 2 = 12 combinations

LGBM_GRID = [
    {"n_estimators": ne, "num_leaves": nl, "learning_rate": lr,
     "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8}
    for ne in [50, 100, 200]
    for nl in [63, 127]
    for lr in [0.05, 0.1]
]  # 12 combinations

NN_GRID = [
    {"h1": h1, "h2": h2, "h3": h3,
     "dropout": dr, "lr": lr,
     "weight_decay": 1e-4, "batch_size": 2048, "epochs": 10}
    for (h1, h2, h3) in [
        (2048, 1024, 512),
        (4096, 2048, 1024),
        (4096, 2048, 1024),   # repeated with different dropout/lr below
    ]
    for dr in [0.2, 0.3]
    for lr in [1e-3, 3e-4]
]  # 12 combinations — large dims saturate A100 CUDA cores; 150 epochs = long GPU runs

GRIDS = {"xgboost": XGB_GRID, "lightgbm": LGBM_GRID, "neural_network": NN_GRID}


def _cv_auroc(
    model_name: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    """Evaluate one param combo with CV; return mean AUROC."""
    scores = []
    for tr_idx, v_idx in cv.split(X, y):
        X_tr, X_v = _clean(X[tr_idx]), _clean(X[v_idx])
        y_tr, y_v = y[tr_idx], y[v_idx]

        X_tr, X_v = _gpu_scale(X_tr, X_v)

        try:
            if model_name == "xgboost":
                m    = train_xgb(X_tr, y_tr, X_v, y_v, params)
                prob = m.predict_proba(X_v)[:, 1]
            elif model_name == "lightgbm":
                m    = train_lgbm(X_tr, y_tr, X_v, y_v, params)
                prob = m.predict_proba(X_v)[:, 1]
            else:
                m    = train_nn(X_tr, y_tr, X_v, y_v, params)
                prob = predict_nn(m, X_v)
            scores.append(roc_auc_score(y_v, prob))
        except Exception as e:
            log.debug(f"Grid combo failed: {e}")
            scores.append(0.0)

    return float(np.mean(scores))


def _clean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN / inf / -inf with 0 and clip to float32 range."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -3.4e38, 3.4e38)


def _gpu_scale(
    X_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit-on-train StandardScaler executed on GPU (torch ops).
    Falls back to sklearn on CPU when no GPU is available.
    Replaces sklearn.StandardScaler in every hot path.
    """
    X_tr = _clean(X_tr)
    X_te = _clean(X_te)
    if USE_GPU:
        Xtr = torch.FloatTensor(X_tr).to(DEVICE)
        mean = Xtr.mean(dim=0)
        std  = Xtr.std(dim=0, unbiased=False).clamp(min=1e-8)
        Xte  = torch.FloatTensor(X_te).to(DEVICE)
        tr_out = ((Xtr - mean) / std).cpu().numpy()
        te_out = ((Xte - mean) / std).cpu().numpy()
        return _clean(tr_out), _clean(te_out)
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def _gpu_scale_fit(X_tr: np.ndarray):
    """Return (X_tr_scaled, mean_tensor, std_tensor) — mean/std live on GPU."""
    X_tr = _clean(X_tr)
    Xtr  = torch.FloatTensor(X_tr).to(DEVICE)
    mean = Xtr.mean(dim=0)
    std  = Xtr.std(dim=0, unbiased=False).clamp(min=1e-8)
    out  = ((Xtr - mean) / std).cpu().numpy()
    return _clean(out), mean, std


def _gpu_scale_transform(X: np.ndarray, mean, std) -> np.ndarray:
    X = _clean(X)
    Xt = torch.FloatTensor(X).to(DEVICE)
    out = ((Xt - mean) / std).cpu().numpy()
    return _clean(out)


def _safe_cv(y: np.ndarray, max_splits: int = N_GRID_CV_SPLITS) -> StratifiedKFold:
    """
    Return a StratifiedKFold whose n_splits is clamped to the size of the
    smallest class.  Prevents the 'least populated class has only N member'
    error on small / imbalanced panels (e.g. CFTR).
    """
    min_class = int(min(Counter(y.tolist()).values()))
    n_splits  = max(2, min(max_splits, min_class))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def _safe_train_test_split(X, y, test_size=0.15):
    """
    Stratified split, falling back to non-stratified when any class has fewer
    than 2 samples (which would make stratify impossible).
    """
    min_class = int(min(Counter(y.tolist()).values()))
    stratify  = y if min_class >= 2 else None
    return train_test_split(X, y, test_size=test_size,
                            stratify=stratify, random_state=RANDOM_SEED)


def grid_search(model_name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Exhaustive search over the small fixed grid with parallel combo evaluation.
    N_PARALLEL_GRID combos run concurrently — keeps the GPU busy across threads.
    Returns the best param dict (highest mean CV AUROC).
    """
    cv   = _safe_cv(y, max_splits=N_GRID_CV_SPLITS)
    grid = GRIDS[model_name]

    def _eval(item):
        idx, params = item
        score = _cv_auroc(model_name, params, X, y, cv)
        log.debug(f"  [{model_name}] combo {idx+1}/{len(grid)}  AUROC={score:.4f}")
        return score, params

    with ThreadPoolExecutor(max_workers=min(N_PARALLEL_GRID, len(grid))) as ex:
        results = list(ex.map(_eval, enumerate(grid)))

    best_score, best_params = max(results, key=lambda r: r[0])
    log.info(f"    [{model_name}] best CV AUROC={best_score:.4f}  params={best_params}")
    return best_params


# =============================================================================
# 7.  EVALUATION
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64),
                           nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auroc":     float(roc_auc_score(y_true, y_prob)),
        "auprc":     float(average_precision_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _fit_and_eval(
    model_name: str,
    best_params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[dict, float]:
    """Train final model with best params; return metrics and inference prob."""
    # Hold-out val split for early stopping (tree models)
    X_trn, X_val, y_trn, y_val = _safe_train_test_split(X_tr, y_tr, test_size=0.15)
    if model_name == "xgboost":
        m = train_xgb(X_trn, y_trn, X_val, y_val, best_params)
        prob = m.predict_proba(X_te)[:, 1]
    elif model_name == "lightgbm":
        m = train_lgbm(X_trn, y_trn, X_val, y_val, best_params)
        prob = m.predict_proba(X_te)[:, 1]
    else:
        m = train_nn(X_trn, y_trn, X_val, y_val, best_params)
        prob = predict_nn(m, X_te)

    return compute_metrics(y_te, prob)


# =============================================================================
# 8.  EXPERIMENT RUNNERS
# =============================================================================

MODEL_NAMES = ["xgboost", "lightgbm", "neural_network"]


def _run_models_parallel(
    X_tr_raw: np.ndarray,
    y_tr:     np.ndarray,
    X_te_raw: np.ndarray,
    y_te:     np.ndarray,
    tag:      str = "",
) -> dict:
    """
    Train all 3 models concurrently (XGBoost + LightGBM + NN in parallel).
    With 80 GB GPU RAM they can all live on the device simultaneously,
    so all three GPU workloads overlap instead of queuing sequentially.
    """
    def _job(model_name):
        return model_name, _run_single_model(
            model_name, X_tr_raw, y_tr, X_te_raw, y_te, tag=tag
        )

    results = {}
    with ThreadPoolExecutor(max_workers=N_PARALLEL_MODELS) as ex:
        for model_name, metrics in ex.map(_job, MODEL_NAMES):
            results[model_name] = metrics
    return results


def _run_single_model(
    model_name: str,
    X_train_raw: np.ndarray,
    y_train:     np.ndarray,
    X_test_raw:  np.ndarray,
    y_test:      np.ndarray,
    tag:         str = "",
) -> dict:
    """
    Grid search → retrain final model → evaluate on test set.
    Scaler is fit on training data only.
    """
    # Guard: need at least 2 classes in both train and test
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        log.warning(
            f"    [{tag}] {model_name} skipped — "
            f"train_classes={np.unique(y_train)}, test_classes={np.unique(y_test)}"
        )
        return {"auroc": 0.0, "f1": 0.0, "accuracy": 0.0,
                "auprc": 0.0, "precision": 0.0, "recall": 0.0,
                "best_params": {}, "gs_time_s": 0.0}

    X_train, X_test = _gpu_scale(_clean(X_train_raw), _clean(X_test_raw))

    log.info(f"    [{tag}] {model_name}  grid search ({len(GRIDS[model_name])} combos)…")
    t0          = time.time()
    best_params = grid_search(model_name, X_train, y_train)
    gs_time     = time.time() - t0

    metrics = _fit_and_eval(model_name, best_params, X_train, y_train, X_test, y_test)
    metrics["best_params"] = best_params
    metrics["gs_time_s"]   = round(gs_time, 1)

    log.info(
        f"    [{tag}] {model_name}  "
        f"AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}  "
        f"({gs_time:.0f}s grid search)"
    )
    return metrics


# ---------- Experiment 1: All Panels, N-fold CV ---------------------------

def experiment_1_all_panels(
    X_raw: np.ndarray,
    y:     np.ndarray,
    panel: pd.Series,
    k:     int,
) -> dict:
    """Train on ALL panels using stratified N-fold CV."""
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT 1  |  All Panels  |  k={k}")
    log.info(f"{'='*60}")

    cv     = _safe_cv(y, max_splits=N_CV_SPLITS)
    result = {"experiment": 1, "kmer": k, "models": {mn: [] for mn in MODEL_NAMES}}

    # All 3 models train in parallel for each fold
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw, X_te_raw = X_raw[tr_idx], X_raw[te_idx]
        y_tr,     y_te     = y[tr_idx],     y[te_idx]

        fold_results = _run_models_parallel(
            X_tr_raw, y_tr, X_te_raw, y_te, tag=f"Exp1 fold={fold+1}"
        )
        for mn, metrics in fold_results.items():
            metrics["fold"] = fold + 1
            result["models"][mn].append(metrics)

    # Aggregate across folds per model
    agg_results = {}
    for model_name, fold_metrics in result["models"].items():
        agg = {}
        for key in ["accuracy", "auroc", "auprc", "f1", "precision", "recall"]:
            vals = [m[key] for m in fold_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals))
        agg_results[model_name] = {"fold_metrics": fold_metrics, "aggregated": agg}
        log.info(
            f"  ✓ {model_name}  "
            f"AUROC={agg['auroc_mean']:.4f}±{agg['auroc_std']:.4f}  "
            f"F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}"
        )

    result["models"] = agg_results
    return result


# ---------- Experiment 2: Leave-One-Panel-Out -----------------------------

def experiment_2_leave_one_out(
    X_raw: np.ndarray,
    y:     np.ndarray,
    panel: pd.Series,
    k:     int,
) -> dict:
    """Train on 3 panels; test on the held-out panel. Repeat for each panel."""
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT 2  |  Leave-One-Panel-Out  |  k={k}")
    log.info(f"{'='*60}")

    panel_arr = panel.values
    result    = {"experiment": 2, "kmer": k, "panels": {}}

    for left_out in PANELS:
        test_mask  = panel_arr == left_out
        train_mask = ~test_mask

        n_test = test_mask.sum()
        if n_test == 0:
            log.warning(f"Panel '{left_out}' not found – skipping.")
            continue
        log.info(
            f"\n  Left-out panel: {left_out}  "
            f"(train={train_mask.sum()}  test={n_test})"
        )

        X_tr_raw, X_te_raw = X_raw[train_mask], X_raw[test_mask]
        y_tr,     y_te     = y[train_mask],     y[test_mask]

        panel_result = {"models": _run_models_parallel(
            X_tr_raw, y_tr, X_te_raw, y_te, tag=f"Exp2 left={left_out}"
        )}
        result["panels"][left_out] = panel_result

    return result


# ---------- Experiment 3: Single-Panel Generalisation ---------------------

def experiment_3_single_panel(
    X_raw: np.ndarray,
    y:     np.ndarray,
    panel: pd.Series,
    k:     int,
) -> dict:
    """
    For each training panel: run grid search on that panel alone,
    then evaluate on every other panel individually.
    """
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT 3  |  Single-Panel Generalisation  |  k={k}")
    log.info(f"{'='*60}")

    panel_arr = panel.values
    result    = {"experiment": 3, "kmer": k, "train_panels": {}}

    for train_panel in PANELS:
        train_mask = panel_arr == train_panel
        n_train    = train_mask.sum()
        if n_train == 0:
            log.warning(f"Train panel '{train_panel}' empty – skipping.")
            continue

        log.info(f"\n  Train panel: {train_panel}  (n={n_train})")
        X_tr_raw = X_raw[train_mask]
        y_tr     = y[train_mask]

        # Scale on train panel — fit on GPU, keep mean/std for test transforms
        X_tr, _sc_mean, _sc_std = _gpu_scale_fit(X_tr_raw)

        train_panel_result = {"test_panels": {}}
        X_trn, X_val, y_trn, y_val = _safe_train_test_split(X_tr, y_tr, test_size=0.15)

        def _train_and_test_panel(model_name):
            # Skip panels where training split has fewer than 2 classes
            n_classes_tr   = len(np.unique(y_trn))
            n_classes_val  = len(np.unique(y_val))
            if n_classes_tr < 2 or n_classes_val < 2:
                log.warning(
                    f"    Skipping {model_name} on '{train_panel}': "
                    f"train classes={n_classes_tr}, val classes={n_classes_val} "
                    f"(need ≥ 2 each)"
                )
                return model_name, {}

            try:
                log.info(f"    Grid search: {model_name} on {train_panel}…")
                best_params = grid_search(model_name, X_tr, y_tr)

                if model_name == "xgboost":
                    final_model = train_xgb(X_trn, y_trn, X_val, y_val, best_params)
                    predict_fn  = lambda Xt, m=final_model: m.predict_proba(Xt)[:, 1]
                elif model_name == "lightgbm":
                    final_model = train_lgbm(X_trn, y_trn, X_val, y_val, best_params)
                    predict_fn  = lambda Xt, m=final_model: m.predict_proba(Xt)[:, 1]
                else:
                    final_model = train_nn(X_trn, y_trn, X_val, y_val, best_params)
                    predict_fn  = lambda Xt, m=final_model: predict_nn(m, Xt)

                model_test_results = {}
                for test_panel in PANELS:
                    if test_panel == train_panel:
                        continue
                    test_mask = panel_arr == test_panel
                    if test_mask.sum() == 0:
                        continue
                    y_te = y[test_mask]
                    if len(np.unique(y_te)) < 2:
                        log.warning(
                            f"      Skipping test panel '{test_panel}': only one class present"
                        )
                        continue
                    X_te    = _gpu_scale_transform(X_raw[test_mask], _sc_mean, _sc_std)
                    prob    = predict_fn(X_te)
                    metrics = compute_metrics(y_te, prob)
                    metrics["best_params"] = best_params
                    model_test_results[test_panel] = metrics
                    log.info(
                        f"      {model_name}  train={train_panel}→test={test_panel}  "
                        f"AUROC={metrics['auroc']:.4f}  F1={metrics['f1']:.4f}"
                    )
                return model_name, model_test_results

            except Exception as exc:
                log.warning(f"    [{model_name}] train_panel='{train_panel}' failed: {exc}")
                return model_name, {}

        with ThreadPoolExecutor(max_workers=N_PARALLEL_MODELS) as ex:
            for mn, res in ex.map(_train_and_test_panel, MODEL_NAMES):
                train_panel_result["test_panels"][mn] = res

        result["train_panels"][train_panel] = train_panel_result

    return result


# =============================================================================
# 9.  UTILITY – JSON serialisation
# =============================================================================

def _json_safe(obj):
    """Recursively convert numpy types so json.dump doesn't fail."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    return obj


# =============================================================================
# 10.  SUMMARY PRINTER
# =============================================================================

def print_summary(all_results: list[dict]):
    sep = "=" * 70
    print(f"\n{sep}")
    print("RESULTS SUMMARY")
    print(sep)

    for run in all_results:
        k = run["k"]
        print(f"\n{'─'*70}")
        print(f"  K-MER SIZE: {k}")
        print(f"{'─'*70}")

        # Experiment 1
        print("\n  EXPERIMENT 1 — All Panels (5-fold CV):")
        for mn, res in run["exp1"]["models"].items():
            agg = res["aggregated"]
            print(
                f"    {mn:<20}  "
                f"AUROC={agg['auroc_mean']:.4f}±{agg['auroc_std']:.4f}  "
                f"F1={agg['f1_mean']:.4f}±{agg['f1_std']:.4f}  "
                f"AUPRC={agg['auprc_mean']:.4f}±{agg['auprc_std']:.4f}"
            )

        # Experiment 2
        print("\n  EXPERIMENT 2 — Leave-One-Panel-Out:")
        for panel_name, panel_res in run["exp2"]["panels"].items():
            print(f"    Left out: {panel_name}")
            for mn, metrics in panel_res["models"].items():
                print(
                    f"      {mn:<20}  "
                    f"AUROC={metrics['auroc']:.4f}  "
                    f"F1={metrics['f1']:.4f}  "
                    f"AUPRC={metrics['auprc']:.4f}"
                )

        # Experiment 3
        print("\n  EXPERIMENT 3 — Single-Panel Generalisation:")
        for tr_panel, tr_res in run["exp3"]["train_panels"].items():
            print(f"    Trained on: {tr_panel}")
            for mn, test_map in tr_res["test_panels"].items():
                for te_panel, metrics in test_map.items():
                    print(
                        f"      {mn:<20}  → {te_panel:<20}  "
                        f"AUROC={metrics['auroc']:.4f}  "
                        f"F1={metrics['f1']:.4f}"
                    )

    print(f"\n{sep}\n")


# =============================================================================
# 11.  MAIN
# =============================================================================

def main(
    data_path:  str = DATA_PATH,
    kmer_sizes: list[int] = KMER_SIZES,
):
    log.info("=" * 60)
    log.info("PATHOGENIC / BENIGN VARIANT PREDICTOR")
    log.info(f"Device    : {DEVICE}")
    log.info(f"K-mer sizes: {kmer_sizes}")
    log.info("=" * 60)

    # ---- Load & label ----
    df = load_raw_data(data_path)
    df = derive_target(df)

    all_results      = []
    feature_registry = {
        "feature_drop_policy": COLS_TO_DROP,
        "ohe_policy":          OHE_COLS,
        "label_enc_policy":    LABEL_ENC_COLS,
        "bool_cols":           BOOL_COLS,
        "kmer_experiments":    {},
    }

    for k in kmer_sizes:
        log.info(f"\n{'#'*60}")
        log.info(f"# K-MER SIZE = {k}")
        log.info(f"{'#'*60}")

        # ---- Feature engineering ----
        X_df, y, panel, registry = build_feature_matrix(df, k)
        X_raw  = X_df.values
        y_arr  = y.values

        feature_registry["kmer_experiments"][str(k)] = registry

        # ---- Run experiments ----
        exp1 = experiment_1_all_panels(X_raw, y_arr, panel, k)
        exp2 = experiment_2_leave_one_out(X_raw, y_arr, panel, k)
        exp3 = experiment_3_single_panel(X_raw, y_arr, panel, k)

        all_results.append({"k": k, "exp1": exp1, "exp2": exp2, "exp3": exp3})

    # ---- Save feature registry ----
    with open(FEATURE_LOG_PATH, "w") as fh:
        json.dump(_json_safe(feature_registry), fh, indent=2)
    log.info(f"Feature registry saved → '{FEATURE_LOG_PATH}'")

    # ---- Save results ----
    with open(RESULTS_PATH, "w") as fh:
        json.dump(_json_safe(all_results), fh, indent=2)
    log.info(f"Results saved          → '{RESULTS_PATH}'")

    # ---- Print summary ----
    print_summary(all_results)

    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # parse_known_args() silently ignores Jupyter/Colab kernel flags
    # like -f kernel-xxxx.json so the script works in both environments.
    # -----------------------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end Pathogenic/Benign predictor"
    )
    parser.add_argument(
        "--data",  default=DATA_PATH, help="Path to CSV dataset"
    )
    parser.add_argument(
        "--kmers", default="2,3",
        help="Comma-separated k-mer sizes (default '2,3')"
    )
    args, _unknown = parser.parse_known_args()

    main(
        data_path  = args.data,
        kmer_sizes = [int(k) for k in args.kmers.split(",")],
    )

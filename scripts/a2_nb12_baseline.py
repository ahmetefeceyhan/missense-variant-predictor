"""NB14 A2 subprocess runner.

XGBoost'un libomp.dylib'i macOS arm64'te notebook'un libomp'u ile cakisip
SIGSEGV uretiyor. Bu script izole bir Python sureci icinde NB12 XGBoost+OHE
baseline'ini calistirir ve sonuclari JSON olarak stdout'un son satirina yazar.

Cagri: python a2_subprocess_runner.py <project_root>
"""
import os
# THREAD'LI lib import edilmeden ONCE single-thread'e zorla
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS']  = '1'
os.environ['MKL_NUM_THREADS']  = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys, json, time
PROJECT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config import SEED, TEST_SIZE
from src.models import grid_search_xgboost, XGB_FIXED
from src.metrics import compute_all_metrics

# Single-thread XGBoost
XGB_FIXED['n_jobs']  = 1
XGB_FIXED['nthread'] = 1

HIGH_NAN_THRESHOLD = 0.50
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_data')


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'YARISMA_TRAIN_MASTER.csv'))
    df = df.drop(columns=['Variant_ID'])
    y  = df['Label'].astype(int)
    X  = df.drop(columns=['Label'])
    nan_pct = X.isnull().mean()
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    high_nan = [c for c in num_cols if nan_pct[c] > HIGH_NAN_THRESHOLD]
    low_nan  = [c for c in num_cols if nan_pct[c] <= HIGH_NAN_THRESHOLD]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    def add_flags(Xdf):
        present = [c for c in high_nan if c in Xdf.columns]
        flags = Xdf[present].isna().astype(np.float32)
        flags.columns = [f'is_missing_{c}' for c in present]
        return pd.concat([Xdf, flags], axis=1).copy()

    Xtr = add_flags(Xtr); Xte = add_flags(Xte)
    col_medians  = Xtr[low_nan].median()
    high_medians = Xtr[high_nan].median()
    Xtr[low_nan]  = Xtr[low_nan].fillna(col_medians)
    Xte[low_nan]  = Xte[low_nan].fillna(col_medians)
    Xtr[high_nan] = Xtr[high_nan].fillna(high_medians)
    Xte[high_nan] = Xte[high_nan].fillna(high_medians)
    for c in cat_cols:
        Xtr[c] = Xtr[c].fillna('MISSING').astype(str)
        Xte[c] = Xte[c].fillna('MISSING').astype(str)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
    ohe.fit(Xtr[cat_cols])
    names = ohe.get_feature_names_out(cat_cols)
    tr_ohe = pd.DataFrame(ohe.transform(Xtr[cat_cols]), columns=names, index=Xtr.index)
    te_ohe = pd.DataFrame(ohe.transform(Xte[cat_cols]), columns=names, index=Xte.index)
    all_num = low_nan + [c for c in high_nan if c in Xtr.columns]
    flag_cols = [f'is_missing_{c}' for c in high_nan if f'is_missing_{c}' in Xtr.columns]
    X_tr_ohe = pd.concat([Xtr[all_num + flag_cols].astype(np.float32), tr_ohe], axis=1)
    X_te_ohe = pd.concat([Xte[all_num + flag_cols].astype(np.float32), te_ohe], axis=1)

    print(f'NB12 OHE matrix: train={X_tr_ohe.shape} test={X_te_ohe.shape}',
          file=sys.stderr)

    t0 = time.time()
    model, combo, thr, prob = grid_search_xgboost(
        X_tr_ohe, ytr, X_te_ohe, yte, cat_features=[]
    )
    elapsed = time.time() - t0
    y_pred = (prob >= thr).astype(int)
    metrics = compute_all_metrics(yte.values, y_pred, prob)

    out = {
        'f1':            float(metrics['f1']),
        'auc':           float(metrics['auc_roc']),
        'precision':     float(metrics['precision']),
        'recall':        float(metrics['recall']),
        'thr':           float(thr),
        'best_combo':    {k: (v if isinstance(v, (int, float, str)) else str(v))
                          for k, v in combo.items()},
        'elapsed_sec':   round(elapsed, 1),
        'n_train':       int(len(ytr)),
        'n_test':        int(len(yte)),
        'n_features':    int(X_tr_ohe.shape[1]),
    }
    print('==A2_RESULT_JSON==', flush=True)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()

"""NB14 NN/DNN subprocess runner.

stdin'den pickled bir dict bekler:
    {
        'mode':        'fit' veya 'predict',
        'model_type':  'mlp3' veya 'deep_mlp',
        # fit modu icin:
        'X_train':     pandas DataFrame (LE-encoded raw -- inside we re-encode),
        'y_train':     pandas Series,
        'X_test':      pandas DataFrame,
        'y_test':      pandas Series,
        'cat_cols':    list[str],
        # predict modu icin: model_state_pkl + encoders + scaler + input_dim + hidden_combo
    }

stdout'a pickled bir dict yazar:
    fit:     {'test_proba', 'oof_proba', 'model_state_pkl', 'enc_pkl', 'scaler_pkl',
              'input_dim', 'best_combo'}
    predict: {'panel_proba'}
"""
import os
# main process default'lar burada uygulanmaz; subprocess kendi env'ini Cell 7'den alir
import sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.getcwd())

import pickle, time
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from config import SEED
from src.models import MLP3Layer, DeepMLP

N_OOF_FOLDS = 3
NN_FIXED = {
    'h1': 1024, 'h2': 512, 'h3': 256,
    'weight_decay': 1e-4, 'batch_size': 1024,
    'max_epochs': 25, 'patience': 5,
}
NN_GRID = {'dropout': [0.2], 'lr': [1e-3, 3e-4]}
DNN_FIXED = {
    'weight_decay': 1e-4, 'batch_size': 1024,
    'max_epochs': 25, 'patience': 5,
}
DNN_GRID = {'n_layers': [3], 'hidden_dim': [256], 'dropout': [0.3], 'lr': [1e-3, 3e-4]}


def _le_encode(X, cat_cols):
    X = X.copy()
    encoders = {}
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            known = set(le.classes_)
            X[col] = X[col].astype(str).map(
                lambda v, le=le, k=known: le.transform([v])[0] if v in k else -1
            )
            encoders[col] = le
    return X, encoders


def _le_transform(X, encoders):
    X = X.copy()
    for col, le in encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            X[col] = X[col].astype(str).map(
                lambda v, le=le, k=known: le.transform([v])[0] if v in k else -1
            )
    return X


def _prepare_tensors(X_train, X_test, y_train, y_test, cat_cols):
    X_tr_enc, enc = _le_encode(X_train, cat_cols)
    X_te_enc      = _le_transform(X_test, enc)
    X_tr_enc = X_tr_enc.astype(np.float32).fillna(0)
    X_te_enc = X_te_enc.astype(np.float32).fillna(0)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_enc.values).astype(np.float32)
    X_te_s = sc.transform(X_te_enc.values).astype(np.float32)
    X_tr_t = torch.FloatTensor(X_tr_s)
    X_te_t = torch.FloatTensor(X_te_s)
    y_tr_t = torch.FloatTensor(y_train.values.astype(np.float32))
    y_te_t = torch.FloatTensor(y_test.values.astype(np.float32))
    n_pos = float((y_train == 1).sum()); n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    input_dim = X_tr_t.shape[1]
    return X_tr_t, X_te_t, y_tr_t, y_te_t, input_dim, pos_weight, sc, enc


def _make_model(model_type, input_dim, combo, fixed):
    if model_type == 'mlp3':
        return MLP3Layer(input_dim, fixed['h1'], fixed['h2'], fixed['h3'],
                         combo['dropout'])
    return DeepMLP(input_dim, combo['hidden_dim'], combo['n_layers'],
                   combo['dropout'])


def _fit_one(model, X_tr_t, y_tr_t, X_val_t, y_val_t, lr, fixed, pos_weight):
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=fixed['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed['max_epochs'])
    bs = min(fixed['batch_size'], len(X_tr_t) - 1)
    n  = len(X_tr_t)
    best_state, best_val, patience = None, -1.0, 0
    model.train()
    for ep in range(fixed['max_epochs']):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            if len(idx) < 2: continue
            opt.zero_grad()
            loss = crit(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(X_val_t)).numpy()
            vf = f1_score(y_val_t.numpy(), (vp >= 0.5).astype(int), zero_division=0)
        if vf > best_val:
            best_val, best_state, patience = vf, deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= fixed['patience']: break
        model.train()
    return best_state, best_val


def fit_mode(payload):
    mt = payload['model_type']
    fixed = NN_FIXED if mt == 'mlp3' else DNN_FIXED
    grid  = NN_GRID  if mt == 'mlp3' else DNN_GRID

    X_tr_t, X_te_t, y_tr_t, y_te_t, input_dim, pos_w, sc, enc = _prepare_tensors(
        payload['X_train'], payload['X_test'],
        payload['y_train'], payload['y_test'],
        payload['cat_cols'],
    )

    # Build combos
    combos = []
    for d in grid['dropout']:
        for lr in grid['lr']:
            if mt == 'mlp3':
                combos.append({'dropout': d, 'lr': lr})
            else:
                for nl in grid['n_layers']:
                    for hd in grid['hidden_dim']:
                        combos.append({'dropout': d, 'lr': lr,
                                        'n_layers': nl, 'hidden_dim': hd})
    print(f'[runner] {mt} grid: {len(combos)} combos', file=sys.stderr, flush=True)

    # Inner CV grid search (2-fold)
    skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    best_f1, best_combo = -1.0, None
    y_tr_np = y_tr_t.numpy()
    t0 = time.time()
    for ci, combo in enumerate(combos):
        fold_f1s = []
        for fi, (tr_idx, val_idx) in enumerate(skf2.split(X_tr_t.numpy(), y_tr_np)):
            m = _make_model(mt, input_dim, combo, fixed)
            _, vf = _fit_one(m, X_tr_t[tr_idx], y_tr_t[tr_idx],
                              X_tr_t[val_idx], y_tr_t[val_idx],
                              combo['lr'], fixed, pos_w)
            fold_f1s.append(vf)
            print(f'[runner] combo {ci} fold {fi}: vf={vf:.4f} '
                  f'elapsed={time.time()-t0:.1f}s',
                  file=sys.stderr, flush=True)
        mf = float(np.mean(fold_f1s))
        if mf > best_f1:
            best_f1, best_combo = mf, combo
    print(f'[runner] best={best_combo} cv_f1={best_f1:.4f}',
          file=sys.stderr, flush=True)

    # Final fit on full train
    final = _make_model(mt, input_dim, best_combo, fixed)
    best_state, _ = _fit_one(final, X_tr_t, y_tr_t, X_te_t, y_te_t,
                              best_combo['lr'], fixed, pos_w)
    if best_state is not None: final.load_state_dict(best_state)
    final.eval()
    with torch.no_grad():
        test_proba = torch.sigmoid(final(X_te_t)).numpy()

    # OOF (3-fold)
    skf3 = StratifiedKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_tr_t))
    for fi, (tr_idx, val_idx) in enumerate(skf3.split(X_tr_t.numpy(), y_tr_np)):
        m = _make_model(mt, input_dim, best_combo, fixed)
        st, _ = _fit_one(m, X_tr_t[tr_idx], y_tr_t[tr_idx],
                          X_tr_t[val_idx], y_tr_t[val_idx],
                          best_combo['lr'], fixed, pos_w)
        if st is not None: m.load_state_dict(st)
        m.eval()
        with torch.no_grad():
            oof[val_idx] = torch.sigmoid(m(X_tr_t[val_idx])).numpy()
        print(f'[runner] oof fold {fi} done elapsed={time.time()-t0:.1f}s',
              file=sys.stderr, flush=True)

    return {
        'test_proba':      test_proba,
        'oof_proba':       oof,
        'model_state_pkl': pickle.dumps(final.state_dict()),
        'enc_pkl':         pickle.dumps(enc),
        'scaler_pkl':      pickle.dumps(sc),
        'input_dim':       int(input_dim),
        'best_combo':      best_combo,
        'model_type':      mt,
    }


def predict_mode(payload):
    """Tek-shot panel predict: model_state'i yeniden ina edip X_panel'i scale et."""
    mt = payload['model_type']
    fixed = NN_FIXED if mt == 'mlp3' else DNN_FIXED
    enc = pickle.loads(payload['enc_pkl'])
    sc  = pickle.loads(payload['scaler_pkl'])
    combo = payload['best_combo']
    input_dim = payload['input_dim']

    X_panel = payload['X_panel']
    Xe = _le_transform(X_panel, enc).astype(np.float32).fillna(0)
    Xs = sc.transform(Xe.values).astype(np.float32)
    m = _make_model(mt, input_dim, combo, fixed)
    m.load_state_dict(pickle.loads(payload['model_state_pkl']))
    m.eval()
    with torch.no_grad():
        proba = torch.sigmoid(m(torch.FloatTensor(Xs))).numpy()
    return {'panel_proba': proba}


def main():
    payload = pickle.load(sys.stdin.buffer)
    if payload['mode'] == 'fit':
        result = fit_mode(payload)
    elif payload['mode'] == 'predict':
        result = predict_mode(payload)
    else:
        raise ValueError(f'unknown mode {payload["mode"]}')
    sys.stdout.buffer.write(pickle.dumps(result))


if __name__ == '__main__':
    main()

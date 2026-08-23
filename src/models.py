# -*- coding: utf-8 -*-
"""
Model builder fonksiyonlari: LightGBM, XGBoost, NN (Shallow MLP), DNN (Deep MLP).
Focal Loss entegrasyonu desteklenir.
"""
import numpy as np
import pandas as pd
# import torch  # Optional: only needed for NN/DNN models
from copy import deepcopy

import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
# from torch.utils.data import DataLoader, TensorDataset  # Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, TRIALS_TREE, TRIALS_NN
from src.metrics import optimize_threshold
from src.utils import prepare_for_xgb, prepare_for_nn
from src.focal_loss import FocalLoss, focal_objective_lgbm, focal_objective_xgb


# ============================================================================
# SINIR AGI MIMARILERI
# ============================================================================

class ShallowMLP(torch.nn.Module):
    """1-2 katmanli MLP: Input -> [Hidden+BN+ReLU+Dropout] x N -> Output(1)"""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            ])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class DeepMLP(torch.nn.Module):
    """3-5 katmanli MLP: Residual connections + BN + Dropout."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.input_bn = torch.nn.BatchNorm1d(hidden_dim)

        self.blocks = torch.nn.ModuleList()
        for i in range(n_layers):
            block = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            )
            self.blocks.append(block)

        self.output = torch.nn.Linear(hidden_dim, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate * 0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.input_bn(self.input_proj(x))))
        for i, block in enumerate(self.blocks):
            residual = x
            x = block(x)
            if i % 2 == 1:  # Her 2 katmanda bir residual connection
                x = x + residual
        return self.output(x).squeeze(-1)


# ============================================================================
# LIGHTGBM BUILDER
# ============================================================================

def build_lightgbm(X_train, y_train, X_test, y_test, cat_features,
                   n_trials=TRIALS_TREE, use_focal=False):
    """
    Optuna ile optimize edilmis LightGBM modeli.

    Args:
        use_focal: True ise Focal Loss custom objective kullanir,
                   gamma Optuna ile optimize edilir.
    """
    def objective(trial):
        params = {
            'verbosity': -1, 'random_state': SEED,
            'num_leaves': trial.suggest_int('num_leaves', 10, 120),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 50),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 800),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'max_bin': trial.suggest_int('max_bin', 63, 511),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        }

        if use_focal:
            focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
            focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
            params['objective'] = focal_objective_lgbm(focal_alpha, focal_gamma)
        else:
            params['objective'] = 'binary'
            params['class_weight'] = 'balanced'

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, categorical_feature=cat_features)

            # Custom objective ile predict_proba raw score doner
            if use_focal:
                raw_scores = model.predict_proba(X_val)
                y_val_prob = 1.0 / (1.0 + np.exp(-raw_scores))
            else:
                y_val_prob = model.predict_proba(X_val)[:, 1]
            _, f1_val = optimize_threshold(y_val, y_val_prob)
            f1_scores.append(f1_val)
        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # En iyi parametrelerle final model
    best_params = study.best_params
    focal_gamma = best_params.pop('focal_gamma', None)
    focal_alpha = best_params.pop('focal_alpha', None)

    if use_focal and focal_gamma is not None:
        best_params['objective'] = focal_objective_lgbm(focal_alpha, focal_gamma)
    else:
        best_params.update({'objective': 'binary', 'class_weight': 'balanced'})
    best_params.update({'verbosity': -1, 'random_state': SEED})

    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train, categorical_feature=cat_features)

    # Custom objective ile predict_proba raw score doner
    if use_focal:
        raw_scores = best_model.predict_proba(X_test)
        y_pred_proba = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = optimize_threshold(y_test, y_pred_proba)

    return best_model, study, best_thr, y_pred_proba


# ============================================================================
# XGBOOST BUILDER
# ============================================================================

def build_xgboost(X_train, y_train, X_test, y_test, cat_features,
                  n_trials=TRIALS_TREE, use_focal=False):
    """Optuna ile optimize edilmis XGBoost modeli."""

    X_tr_enc, X_te_enc, _ = prepare_for_xgb(X_train, X_test, cat_features)
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            'eval_metric': 'logloss',
            'random_state': SEED, 'verbosity': 0,
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 800),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        if not use_focal:
            params['objective'] = 'binary:logistic'
            params['scale_pos_weight'] = scale_pos

        focal_gamma = None
        focal_alpha = None
        if use_focal:
            focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
            focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        for train_idx, val_idx in skf.split(X_tr_enc, y_train):
            X_tr_fold = X_tr_enc.iloc[train_idx]
            X_val_fold = X_tr_enc.iloc[val_idx]
            y_tr_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params, use_label_encoder=False)

            if use_focal:
                model.set_params(objective=focal_objective_xgb(focal_alpha, focal_gamma))

            model.fit(X_tr_fold, y_tr_fold)
            y_val_prob = model.predict_proba(X_val_fold)[:, 1]
            _, f1_val = optimize_threshold(y_val_fold, y_val_prob)
            f1_scores.append(f1_val)
        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    focal_gamma = best_params.pop('focal_gamma', None)
    focal_alpha = best_params.pop('focal_alpha', None)

    if not use_focal:
        best_params.update({
            'objective': 'binary:logistic',
            'scale_pos_weight': scale_pos,
        })
    best_params.update({
        'eval_metric': 'logloss', 'random_state': SEED,
        'verbosity': 0, 'use_label_encoder': False,
    })

    best_model = xgb.XGBClassifier(**best_params)
    if use_focal and focal_gamma is not None:
        best_model.set_params(objective=focal_objective_xgb(focal_alpha, focal_gamma))

    best_model.fit(X_tr_enc, y_train)

    y_pred_proba = best_model.predict_proba(X_te_enc)[:, 1]
    best_thr, best_f1 = optimize_threshold(y_test, y_pred_proba)

    return best_model, study, best_thr, y_pred_proba


# ============================================================================
# NN/DNN ORTAK EGITIM FONKSIYONU
# ============================================================================

def _train_nn_model(model_class, X_train, y_train, X_test, y_test, cat_features,
                    n_trials, layer_range, hidden_range, dropout_range, use_focal=False):
    """NN/DNN icin ortak egitim fonksiyonu. Focal Loss destegi mevcut."""

    nn_data = prepare_for_nn(X_train, X_test, y_train, y_test, cat_features)
    X_tr_t, X_te_t, y_tr_t, y_te_t, input_dim, pos_weight, scaler = nn_data

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', layer_range[0], layer_range[1])
        hidden_dim = trial.suggest_int('hidden_dim', hidden_range[0], hidden_range[1], step=32)
        dropout = trial.suggest_float('dropout', dropout_range[0], dropout_range[1])
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        # Focal Loss parametreleri
        if use_focal:
            focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
            focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_tr_t.numpy(), y_tr_t.numpy()):
            X_fold_tr = X_tr_t[train_idx]
            y_fold_tr = y_tr_t[train_idx]
            X_fold_val = X_tr_t[val_idx]
            y_fold_val = y_tr_t[val_idx]

            model = model_class(input_dim, hidden_dim, n_layers, dropout)

            if use_focal:
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
            else:
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            dataset = TensorDataset(X_fold_tr, y_fold_tr)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            best_val_f1 = 0
            patience_counter = 0

            model.train()
            for epoch in range(100):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_fold_val)
                    val_probs = torch.sigmoid(val_logits).numpy()
                    val_preds = (val_probs >= 0.5).astype(int)
                    val_f1 = f1_score(y_fold_val.numpy(), val_preds)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        break
                model.train()

            f1_scores.append(best_val_f1)

        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # En iyi parametrelerle final model
    bp = study.best_params
    final_model = model_class(input_dim, bp['hidden_dim'], bp['n_layers'], bp['dropout'])

    if use_focal:
        criterion = FocalLoss(
            alpha=bp.get('focal_alpha', 0.25),
            gamma=bp.get('focal_gamma', 2.0),
            pos_weight=pos_weight
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=bp['lr'], weight_decay=bp['weight_decay'])

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=bp['batch_size'], shuffle=True)

    final_model.train()
    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(150):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = final_model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        final_model.eval()
        with torch.no_grad():
            val_logits = final_model(X_te_t)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_f1 = f1_score(y_te_t.numpy(), (val_probs >= 0.5).astype(int))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
        final_model.train()

    if best_state is not None:
        final_model.load_state_dict(best_state)

    final_model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(final_model(X_te_t)).numpy()

    best_thr, best_f1 = optimize_threshold(y_test, y_pred_proba)

    return final_model, study, best_thr, y_pred_proba


# ============================================================================
# NN/DNN BUILDER WRAPPERS
# ============================================================================

def build_nn(X_train, y_train, X_test, y_test, cat_features,
             n_trials=TRIALS_NN, use_focal=False):
    """Optuna ile optimize edilmis Shallow MLP (1-2 katman)."""
    return _train_nn_model(
        ShallowMLP, X_train, y_train, X_test, y_test, cat_features,
        n_trials=n_trials, layer_range=(1, 2), hidden_range=(32, 256),
        dropout_range=(0.1, 0.5), use_focal=use_focal
    )


def build_dnn(X_train, y_train, X_test, y_test, cat_features,
              n_trials=TRIALS_NN, use_focal=False):
    """Optuna ile optimize edilmis Deep MLP (3-5 katman, residual connections)."""
    return _train_nn_model(
        DeepMLP, X_train, y_train, X_test, y_test, cat_features,
        n_trials=n_trials, layer_range=(3, 5), hidden_range=(64, 512),
        dropout_range=(0.2, 0.6), use_focal=use_focal
    )


# ============================================================================
# MODEL BUILDER REGISTRY
# ============================================================================

MODEL_BUILDERS = {
    'lightgbm': build_lightgbm,
    'xgboost': build_xgboost,
    'nn': build_nn,
    'dnn': build_dnn,
}


# ============================================================================
# GRID SEARCH FONKSIYONLARI (v3 pipeline icin)
# ============================================================================

from itertools import product as _itertools_product
from sklearn.model_selection import train_test_split

# Grid tanimlari
LGBM_GRID = {
    'n_estimators': [50, 100, 200],
    'num_leaves': [63, 127],
    'learning_rate': [0.05, 0.1],
}
LGBM_FIXED = {
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': -1,
    'random_state': SEED,
    'objective': 'binary',
    'class_weight': 'balanced',
}

XGB_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
}
XGB_FIXED = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'eval_metric': 'logloss',
    'random_state': SEED,
    'verbosity': 0,
    'objective': 'binary:logistic',
}

NN_GRID = {
    'h1': [2048, 4096],
    'h2': [1024, 2048],
    'h3': [512, 1024],
    'dropout': [0.2, 0.3],
    'lr': [1e-3, 3e-4],
}
NN_FIXED = {
    'weight_decay': 1e-4,
    'batch_size': 2048,
}


class MLP3Layer(torch.nn.Module):
    """3 hidden katmanli MLP (danisman mimarisi)."""
    def __init__(self, input_dim, h1, h2, h3, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, h1),
            torch.nn.BatchNorm1d(h1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h1, h2),
            torch.nn.BatchNorm1d(h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h2, h3),
            torch.nn.BatchNorm1d(h3),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h3, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _grid_combos(grid_dict):
    """Grid dict'ten tum kombinasyonlari uret."""
    keys = list(grid_dict.keys())
    vals = list(grid_dict.values())
    for combo in _itertools_product(*vals):
        yield dict(zip(keys, combo))


def grid_search_lightgbm(X_train, y_train, X_test, y_test, cat_features=None):
    """
    Grid search ile LightGBM egitimi.
    3-fold CV ile en iyi hyperparameter secimi, sonra full train.
    """
    if cat_features is None:
        cat_features = []

    print(f"  LightGBM Grid Search: {len(list(_grid_combos(LGBM_GRID)))} kombinasyon")
    best_f1 = -1
    best_combo = None

    for combo in _grid_combos(LGBM_GRID):
        params = {**LGBM_FIXED, **combo}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, categorical_feature=cat_features)
            y_prob = model.predict_proba(X_val)[:, 1]
            _, f1_val = optimize_threshold(y_val, y_prob)
            fold_f1s.append(f1_val)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    # Final model: full train
    final_params = {**LGBM_FIXED, **best_combo}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train, categorical_feature=cat_features)

    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba


def grid_search_xgboost(X_train, y_train, X_test, y_test, cat_features=None):
    """
    Grid search ile XGBoost egitimi.
    3-fold CV ile en iyi hyperparameter secimi, sonra full train.
    """
    if cat_features is None:
        cat_features = []

    # XGBoost icin LabelEncode
    X_tr_enc, X_te_enc, _ = prepare_for_xgb(X_train, X_test, cat_features)
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"  XGBoost Grid Search: {len(list(_grid_combos(XGB_GRID)))} kombinasyon")
    best_f1 = -1
    best_combo = None

    for combo in _grid_combos(XGB_GRID):
        params = {**XGB_FIXED, **combo, 'scale_pos_weight': scale_pos}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_enc, y_train):
            X_tr = X_tr_enc.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_tr_enc.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            _, f1_val = optimize_threshold(y_val, y_prob)
            fold_f1s.append(f1_val)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    # Final model: full train
    final_params = {**XGB_FIXED, **best_combo, 'scale_pos_weight': scale_pos}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_tr_enc, y_train)

    y_pred_proba = final_model.predict_proba(X_te_enc)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba


def grid_search_nn(X_train, y_train, X_test, y_test, cat_features=None):
    """
    Grid search ile 3-katmanli MLP egitimi.
    3-fold CV ile en iyi hyperparameter secimi, sonra full train.
    """
    if cat_features is None:
        cat_features = []

    nn_data = prepare_for_nn(X_train, X_test, y_train, y_test, cat_features)
    X_tr_t, X_te_t, y_tr_t, y_te_t, input_dim, pos_weight, scaler = nn_data

    print(f"  NN Grid Search: {len(list(_grid_combos(NN_GRID)))} kombinasyon")
    best_f1 = -1
    best_combo = None

    for combo in _grid_combos(NN_GRID):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_t.numpy(), y_tr_t.numpy()):
            X_fold_tr = X_tr_t[train_idx]
            y_fold_tr = y_tr_t[train_idx]
            X_fold_val = X_tr_t[val_idx]
            y_fold_val = y_tr_t[val_idx]

            model = MLP3Layer(input_dim, combo['h1'], combo['h2'], combo['h3'], combo['dropout'])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=combo['lr'],
                weight_decay=NN_FIXED['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

            bs = min(NN_FIXED['batch_size'], len(X_fold_tr) - 1)
            n = len(X_fold_tr)
            best_val_f1 = 0
            patience = 0

            model.train()
            for epoch in range(150):
                # GPU-side shuffling
                perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    if len(idx) < 2:
                        continue
                    batch_X = X_fold_tr[idx]
                    batch_y = y_fold_tr[idx]
                    optimizer.zero_grad()
                    out = model(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(X_fold_val)).numpy()
                    val_f1 = f1_score(y_fold_val.numpy(), (val_probs >= 0.5).astype(int))
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience = 0
                else:
                    patience += 1
                    if patience >= 15:
                        break
                model.train()

            fold_f1s.append(best_val_f1)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    # Final model: full train
    final_model = MLP3Layer(input_dim, best_combo['h1'], best_combo['h2'],
                            best_combo['h3'], best_combo['dropout'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=best_combo['lr'],
        weight_decay=NN_FIXED['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    bs = min(NN_FIXED['batch_size'], len(X_tr_t) - 1)
    n = len(X_tr_t)
    best_val_f1 = 0
    patience = 0
    best_state = None

    final_model.train()
    for epoch in range(150):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            out = final_model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        final_model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(final_model(X_te_t)).numpy()
            val_f1 = f1_score(y_te_t.numpy(), (val_probs >= 0.5).astype(int))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(final_model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
        final_model.train()

    if best_state is not None:
        final_model.load_state_dict(best_state)

    final_model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(final_model(X_te_t)).numpy()

    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba


# ============================================================================
# FAST NN GRID SEARCH (azaltilmis grid, 50 epoch)
# ============================================================================

NN_GRID_FAST = {
    'dropout': [0.2, 0.3],
    'lr': [1e-3, 3e-4],
}
NN_FIXED_FAST = {
    'h1': 2048, 'h2': 1024, 'h3': 512,
    'weight_decay': 1e-4,
    'batch_size': 2048,
    'max_epochs': 50,
    'patience': 10,
}


def grid_search_nn_fast(X_train, y_train, X_test, y_test, cat_features=None):
    """
    Hizli NN grid search: 4 kombinasyon, 50 epoch, patience=10.
    """
    if cat_features is None:
        cat_features = []

    nn_data = prepare_for_nn(X_train, X_test, y_train, y_test, cat_features)
    X_tr_t, X_te_t, y_tr_t, y_te_t, input_dim, pos_weight, scaler = nn_data

    print(f"  NN-Fast Grid Search: {len(list(_grid_combos(NN_GRID_FAST)))} kombinasyon")
    best_f1 = -1
    best_combo = None
    max_ep = NN_FIXED_FAST['max_epochs']
    pat_limit = NN_FIXED_FAST['patience']

    for combo in _grid_combos(NN_GRID_FAST):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_t.numpy(), y_tr_t.numpy()):
            X_fold_tr = X_tr_t[train_idx]
            y_fold_tr = y_tr_t[train_idx]
            X_fold_val = X_tr_t[val_idx]
            y_fold_val = y_tr_t[val_idx]

            model = MLP3Layer(input_dim, NN_FIXED_FAST['h1'], NN_FIXED_FAST['h2'],
                              NN_FIXED_FAST['h3'], combo['dropout'])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=combo['lr'],
                weight_decay=NN_FIXED_FAST['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_ep)

            bs = min(NN_FIXED_FAST['batch_size'], len(X_fold_tr) - 1)
            n = len(X_fold_tr)
            best_val_f1 = 0
            patience = 0

            model.train()
            for epoch in range(max_ep):
                perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    if len(idx) < 2:
                        continue
                    optimizer.zero_grad()
                    out = model(X_fold_tr[idx])
                    loss = criterion(out, y_fold_tr[idx])
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(X_fold_val)).numpy()
                    val_f1 = f1_score(y_fold_val.numpy(), (val_probs >= 0.5).astype(int))
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience = 0
                else:
                    patience += 1
                    if patience >= pat_limit:
                        break
                model.train()

            fold_f1s.append(best_val_f1)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    # Final model: full train
    final_model = MLP3Layer(input_dim, NN_FIXED_FAST['h1'], NN_FIXED_FAST['h2'],
                            NN_FIXED_FAST['h3'], best_combo['dropout'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=best_combo['lr'],
        weight_decay=NN_FIXED_FAST['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_ep)

    bs = min(NN_FIXED_FAST['batch_size'], len(X_tr_t) - 1)
    n = len(X_tr_t)
    best_val_f1 = 0
    patience = 0
    best_state = None

    final_model.train()
    for epoch in range(max_ep):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            out = final_model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        final_model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(final_model(X_te_t)).numpy()
            val_f1 = f1_score(y_te_t.numpy(), (val_probs >= 0.5).astype(int))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(final_model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= pat_limit + 5:
                break
        final_model.train()

    if best_state is not None:
        final_model.load_state_dict(best_state)

    final_model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(final_model(X_te_t)).numpy()

    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba


# ============================================================================
# FAST DNN GRID SEARCH (DeepMLP, azaltilmis grid, 50 epoch)
# ============================================================================

DNN_GRID_FAST = {
    'n_layers': [3, 4],
    'hidden_dim': [256, 512],
    'dropout': [0.3],
    'lr': [1e-3],
}
DNN_FIXED_FAST = {
    'weight_decay': 1e-4,
    'batch_size': 2048,
    'max_epochs': 50,
    'patience': 10,
}


def grid_search_dnn_fast(X_train, y_train, X_test, y_test, cat_features=None):
    """
    Hizli DNN grid search: DeepMLP ile 4 kombinasyon, 50 epoch, patience=10.
    """
    if cat_features is None:
        cat_features = []

    nn_data = prepare_for_nn(X_train, X_test, y_train, y_test, cat_features)
    X_tr_t, X_te_t, y_tr_t, y_te_t, input_dim, pos_weight, scaler = nn_data

    print(f"  DNN-Fast Grid Search: {len(list(_grid_combos(DNN_GRID_FAST)))} kombinasyon")
    best_f1 = -1
    best_combo = None
    max_ep = DNN_FIXED_FAST['max_epochs']
    pat_limit = DNN_FIXED_FAST['patience']

    for combo in _grid_combos(DNN_GRID_FAST):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_t.numpy(), y_tr_t.numpy()):
            X_fold_tr = X_tr_t[train_idx]
            y_fold_tr = y_tr_t[train_idx]
            X_fold_val = X_tr_t[val_idx]
            y_fold_val = y_tr_t[val_idx]

            model = DeepMLP(input_dim, combo['hidden_dim'], combo['n_layers'], combo['dropout'])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=combo['lr'],
                weight_decay=DNN_FIXED_FAST['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_ep)

            bs = min(DNN_FIXED_FAST['batch_size'], len(X_fold_tr) - 1)
            n = len(X_fold_tr)
            best_val_f1 = 0
            patience = 0

            model.train()
            for epoch in range(max_ep):
                perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    if len(idx) < 2:
                        continue
                    optimizer.zero_grad()
                    out = model(X_fold_tr[idx])
                    loss = criterion(out, y_fold_tr[idx])
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(X_fold_val)).numpy()
                    val_f1 = f1_score(y_fold_val.numpy(), (val_probs >= 0.5).astype(int))
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience = 0
                else:
                    patience += 1
                    if patience >= pat_limit:
                        break
                model.train()

            fold_f1s.append(best_val_f1)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    # Final model: full train
    final_model = DeepMLP(input_dim, best_combo['hidden_dim'], best_combo['n_layers'],
                          best_combo['dropout'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=best_combo['lr'],
        weight_decay=DNN_FIXED_FAST['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_ep)

    bs = min(DNN_FIXED_FAST['batch_size'], len(X_tr_t) - 1)
    n = len(X_tr_t)
    best_val_f1 = 0
    patience = 0
    best_state = None

    final_model.train()
    for epoch in range(max_ep):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            out = final_model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        final_model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(final_model(X_te_t)).numpy()
            val_f1 = f1_score(y_te_t.numpy(), (val_probs >= 0.5).astype(int))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(final_model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= pat_limit + 5:
                break
        final_model.train()

    if best_state is not None:
        final_model.load_state_dict(best_state)

    final_model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(final_model(X_te_t)).numpy()

    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba


# ============================================================================
# SVM GRID SEARCH
# ============================================================================
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as _StandardScaler

SVM_GRID = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
}
SVM_FIXED = {
    'kernel': 'rbf',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': SEED,
    'cache_size': 1000,
}


def grid_search_svm(X_train, y_train, X_test, y_test, cat_features=None):
    """
    Grid search ile SVM (RBF kernel) egitimi.
    3-fold CV ile en iyi hyperparameter secimi, sonra full train.
    StandardScaler zorunlu. Return tuple'a scaler dahil.
    """
    if cat_features is None:
        cat_features = []

    # Kategorik sutunlari label-encode et
    if cat_features:
        X_tr_enc, X_te_enc, _ = prepare_for_xgb(X_train, X_test, cat_features)
    else:
        X_tr_enc = X_train.copy()
        X_te_enc = X_test.copy()

    # NaN temizle
    X_tr_enc = X_tr_enc.fillna(0).astype(float)
    X_te_enc = X_te_enc.fillna(0).astype(float)

    # StandardScaler
    scaler = _StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc)
    X_te_scaled = scaler.transform(X_te_enc)

    print(f"  SVM Grid Search: {len(list(_grid_combos(SVM_GRID)))} kombinasyon")
    best_f1 = -1
    best_combo = None

    for combo in _grid_combos(SVM_GRID):
        params = {**SVM_FIXED, **combo}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_scaled, y_train):
            X_tr = X_tr_scaled[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_tr_scaled[val_idx]
            y_val = y_train.iloc[val_idx]

            model = SVC(**params)
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            _, f1_val = optimize_threshold(y_val, y_prob)
            fold_f1s.append(f1_val)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    # Final model: full train
    final_params = {**SVM_FIXED, **best_combo}
    final_model = SVC(**final_params)
    final_model.fit(X_tr_scaled, y_train)

    y_pred_proba = final_model.predict_proba(X_te_scaled)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba, scaler


GRID_BUILDERS = {
    'lightgbm': grid_search_lightgbm,
    'xgboost': grid_search_xgboost,
    'nn': grid_search_nn,
    'nn_fast': grid_search_nn_fast,
    'dnn_fast': grid_search_dnn_fast,
    'svm': grid_search_svm,
}


# ============================================================================
# LABEL ENCODER + XGBOOST GRID SEARCH
# ============================================================================

from sklearn.preprocessing import LabelEncoder as _LabelEncoder

LE_XGB_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
}
LE_XGB_FIXED = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'eval_metric': 'logloss',
    'random_state': SEED,
    'verbosity': 0,
    'objective': 'binary:logistic',
}


def _label_encode_cats(X_train, X_test, cat_cols):
    """
    Kategorik sutunlari LabelEncoder ile encode eder.
    Fit sadece X_train uzerinde; X_test transform ile islenir.
    Bilinmeyen kategoriler -1 ile gosterilir.
    Returns: X_tr_enc, X_te_enc, encoders dict
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    encoders = {}
    for col in cat_cols:
        if col not in X_tr.columns:
            continue
        le = _LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        known = set(le.classes_)
        X_te[col] = X_te[col].astype(str).apply(
            lambda v, known=known, le=le: le.transform([v])[0] if v in known else -1
        )
        encoders[col] = le
    return X_tr, X_te, encoders


def grid_search_le_xgboost(X_train, y_train, X_test, y_test, cat_cols=None):
    """
    LabelEncoder on ile isleme + XGBoost grid search.
    Returns: (model, best_combo, best_thr, y_prob_ho, encoders)
    """
    if cat_cols is None:
        cat_cols = []

    X_tr_enc, X_te_enc, encoders = _label_encode_cats(X_train, X_test, cat_cols)
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"  LE+XGBoost Grid Search: {len(list(_grid_combos(LE_XGB_GRID)))} kombinasyon")
    best_f1 = -1
    best_combo = None

    for combo in _grid_combos(LE_XGB_GRID):
        params = {**LE_XGB_FIXED, **combo, 'scale_pos_weight': scale_pos}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_enc, y_train):
            X_tr_f = X_tr_enc.iloc[train_idx]
            y_tr_f = y_train.iloc[train_idx]
            X_val_f = X_tr_enc.iloc[val_idx]
            y_val_f = y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr_f, y_tr_f)
            y_prob = model.predict_proba(X_val_f)[:, 1]
            _, f1_val = optimize_threshold(y_val_f, y_prob)
            fold_f1s.append(f1_val)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    final_params = {**LE_XGB_FIXED, **best_combo, 'scale_pos_weight': scale_pos}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_tr_enc, y_train)

    y_pred_proba = final_model.predict_proba(X_te_enc)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba, encoders


def grid_search_le_lightgbm(X_train, y_train, X_test, y_test, cat_cols=None):
    """
    LabelEncoder on isle + LightGBM grid search.
    Returns: (model, best_combo, best_thr, y_prob_ho, encoders)
    """
    if cat_cols is None:
        cat_cols = []

    X_tr_enc, X_te_enc, encoders = _label_encode_cats(X_train, X_test, cat_cols)

    print(f"  LE+LightGBM Grid Search: {len(list(_grid_combos(LGBM_GRID)))} kombinasyon")
    best_f1, best_combo = -1, None

    for combo in _grid_combos(LGBM_GRID):
        params = {**LGBM_FIXED, **combo}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []
        for train_idx, val_idx in skf.split(X_tr_enc, y_train):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr_enc.iloc[train_idx], y_train.iloc[train_idx])
            y_prob = model.predict_proba(X_tr_enc.iloc[val_idx])[:, 1]
            _, f1_val = optimize_threshold(y_train.iloc[val_idx], y_prob)
            fold_f1s.append(f1_val)
        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1, best_combo = mean_f1, combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")
    final_model = lgb.LGBMClassifier(**{**LGBM_FIXED, **best_combo})
    final_model.fit(X_tr_enc, y_train)
    y_pred_proba = final_model.predict_proba(X_te_enc)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)
    return final_model, best_combo, best_thr, y_pred_proba, encoders


def grid_search_le_svm(X_train, y_train, X_test, y_test, cat_cols=None):
    """
    LabelEncoder on isle + SVM grid search.
    Returns: (model, best_combo, best_thr, y_prob_ho, encoders, scaler)
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    if cat_cols is None:
        cat_cols = []

    X_tr_enc, X_te_enc, encoders = _label_encode_cats(X_train, X_test, cat_cols)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_enc.fillna(0))
    X_te_scaled = scaler.transform(X_te_enc.fillna(0))

    SVM_LE_GRID  = {'C': [0.1, 1.0], 'gamma': ['scale', 'auto']}
    SVM_LE_FIXED = {'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced', 'random_state': SEED}

    print(f"  LE+SVM Grid Search: {len(list(_grid_combos(SVM_LE_GRID)))} kombinasyon")
    best_f1, best_combo = -1, None

    for combo in _grid_combos(SVM_LE_GRID):
        params = {**SVM_LE_FIXED, **combo}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []
        for train_idx, val_idx in skf.split(X_tr_scaled, y_train):
            model = SVC(**params)
            model.fit(X_tr_scaled[train_idx], y_train.iloc[train_idx])
            y_prob = model.predict_proba(X_tr_scaled[val_idx])[:, 1]
            _, f1_val = optimize_threshold(y_train.iloc[val_idx], y_prob)
            fold_f1s.append(f1_val)
        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1, best_combo = mean_f1, combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")
    final_model = SVC(**{**SVM_LE_FIXED, **best_combo})
    final_model.fit(X_tr_scaled, y_train)
    y_pred_proba = final_model.predict_proba(X_te_scaled)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)
    return final_model, best_combo, best_thr, y_pred_proba, encoders, scaler


# ============================================================================
# CATBOOST GRID SEARCH (native categorical support)
# ============================================================================

try:
    from catboost import CatBoostClassifier as _CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

CB_GRID = {
    'iterations': [100, 200, 300],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1],
}
CB_FIXED = {
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'random_seed': SEED,
    'verbose': False,
    'auto_class_weights': 'Balanced',
}


def grid_search_catboost(X_train, y_train, X_test, y_test, cat_cols=None):
    """
    CatBoost grid search -- kategorik sutunlari native olarak alir (OHE gerekmez).
    Kategorik sutunlar string olarak gonderilir; CatBoost kendi encoding'ini uygular.
    Returns: (model, best_combo, best_thr, y_prob_ho)
    """
    if not _CATBOOST_AVAILABLE:
        raise ImportError("catboost paketi yuklu degil: pip install catboost")
    if cat_cols is None:
        cat_cols = []

    def _prep_cb(X, cols):
        X_c = X.copy()
        for col in X_c.columns:
            if col in cols:
                X_c[col] = X_c[col].astype(str)
            else:
                X_c[col] = pd.to_numeric(X_c[col], errors='coerce').fillna(0).astype(float)
        return X_c

    X_tr_cb = _prep_cb(X_train, cat_cols)
    X_te_cb = _prep_cb(X_test, cat_cols)
    cat_indices = [list(X_tr_cb.columns).index(c) for c in cat_cols if c in X_tr_cb.columns]

    print(f"  CatBoost Grid Search: {len(list(_grid_combos(CB_GRID)))} kombinasyon  "
          f"(cat_features={len(cat_indices)})")
    best_f1 = -1
    best_combo = None

    for combo in _grid_combos(CB_GRID):
        params = {**CB_FIXED, **combo}
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_tr_cb, y_train):
            X_tr_f = X_tr_cb.iloc[train_idx]
            y_tr_f = y_train.iloc[train_idx]
            X_val_f = X_tr_cb.iloc[val_idx]
            y_val_f = y_train.iloc[val_idx]

            model = _CatBoostClassifier(**params)
            model.fit(X_tr_f, y_tr_f, cat_features=cat_indices, silent=True)
            y_prob = model.predict_proba(X_val_f)[:, 1]
            _, f1_val = optimize_threshold(y_val_f, y_prob)
            fold_f1s.append(f1_val)

        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = combo

    print(f"  En iyi combo: {best_combo} -> CV F1={best_f1:.4f}")

    final_params = {**CB_FIXED, **best_combo}
    final_model = _CatBoostClassifier(**final_params)
    final_model.fit(X_tr_cb, y_train, cat_features=cat_indices, silent=True)

    y_pred_proba = final_model.predict_proba(X_te_cb)[:, 1]
    best_thr, _ = optimize_threshold(y_test, y_pred_proba)

    return final_model, best_combo, best_thr, y_pred_proba

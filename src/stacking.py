# -*- coding: utf-8 -*-
"""
Stacking Ensemble modulu.
Base modeller: LightGBM + XGBoost + SklearnMLP
Meta-learner: LogisticRegression veya LightGBM
"""
import numpy as np
import torch
from copy import deepcopy

import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED
from src.metrics import optimize_threshold
from src.utils import prepare_for_xgb
from src.focal_loss import FocalLoss


# ============================================================================
# SKLEARN-UYUMLU MLP WRAPPER
# ============================================================================

class SklearnMLP(BaseEstimator, ClassifierMixin):
    """
    PyTorch ShallowMLP'yi sklearn arayuzune saran wrapper.
    StackingClassifier icinde kullanilabilir.
    """
    def __init__(self, hidden_dim=128, n_layers=2, dropout=0.3,
                 lr=1e-3, weight_decay=1e-4, epochs=100,
                 batch_size=64, patience=15, use_focal=False,
                 focal_alpha=0.25, focal_gamma=2.0):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Egitim sonrasi
        self.model_ = None
        self.scaler_ = None
        self.classes_ = np.array([0, 1])

    def _build_model(self, input_dim):
        """Shallow MLP olustur."""
        layers = []
        in_dim = input_dim
        for _ in range(self.n_layers):
            layers.extend([
                torch.nn.Linear(in_dim, self.hidden_dim),
                torch.nn.BatchNorm1d(self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
            ])
            in_dim = self.hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):
        """Modeli egit."""
        # Veriyi hazirla
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)

        # NaN doldur
        X_arr = np.nan_to_num(X_arr, nan=0.0)

        # Olcekle
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        # Tensorler
        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_arr)

        input_dim = X_scaled.shape[1]
        pos_weight = torch.FloatTensor([(y_arr == 0).sum() / max((y_arr == 1).sum(), 1)])

        # Model olustur
        self.model_ = self._build_model(input_dim)

        # Loss fonksiyonu
        if self.use_focal:
            criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma,
                                  pos_weight=pos_weight)
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.model_.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Egitim dongusu
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model_(batch_X).squeeze(-1)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        """Tahmin olasiliklari dondur (2 sutunlu array)."""
        X_arr = np.array(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        X_scaled = self.scaler_.transform(X_arr)
        X_t = torch.FloatTensor(X_scaled)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()

        # 2 sutunlu format (sklearn uyumlu)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        """Sinif tahminleri dondur."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# ============================================================================
# STACKING BUILDER: LogisticRegression meta-learner
# ============================================================================

def build_stacking_lr(X_train, y_train, X_test, y_test, cat_features,
                      n_trials=None, use_focal=False,
                      lgbm_params=None, xgb_params=None, nn_params=None):
    """
    Stacking ensemble: LightGBM + XGBoost + SklearnMLP
    Meta-learner: LogisticRegression

    Args:
        lgbm_params: LightGBM icin pre-tuned parametreler (dict). None ise default.
        xgb_params: XGBoost icin pre-tuned parametreler (dict). None ise default.
        nn_params: SklearnMLP icin parametreler (dict). None ise default.
    """
    # XGBoost icin kategorik encoding
    X_tr_enc, X_te_enc, _ = prepare_for_xgb(X_train, X_test, cat_features)

    # Sinif orani
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Base model parametreleri
    _lgbm_params = lgbm_params or {
        'objective': 'binary', 'verbosity': -1, 'random_state': SEED,
        'class_weight': 'balanced', 'n_estimators': 300,
        'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 8,
        'colsample_bytree': 0.8, 'subsample': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
    }

    _xgb_params = xgb_params or {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': SEED, 'verbosity': 0,
        'scale_pos_weight': scale_pos, 'n_estimators': 300,
        'learning_rate': 0.05, 'max_depth': 6,
        'colsample_bytree': 0.8, 'subsample': 0.8,
        'use_label_encoder': False,
    }

    _nn_params = nn_params or {
        'hidden_dim': 128, 'n_layers': 2, 'dropout': 0.3,
        'lr': 1e-3, 'weight_decay': 1e-4, 'epochs': 100,
        'batch_size': 64, 'patience': 15,
        'use_focal': use_focal, 'focal_alpha': 0.25, 'focal_gamma': 2.0,
    }

    # Base estimators
    base_estimators = [
        ('lgbm', lgb.LGBMClassifier(**_lgbm_params)),
        ('xgb', xgb.XGBClassifier(**_xgb_params)),
        ('nn', SklearnMLP(**_nn_params)),
    ]

    # Meta-learner
    meta_learner = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=SEED, C=1.0
    )

    # Stacking
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=1,  # NN thread-safe degil, n_jobs=1
    )

    print("  Stacking (LR meta-learner) egitiliyor...")

    # XGBoost encoded veri kullanmali -- tum veriyi encode et
    # NOT: StackingClassifier icindeki XGBoost kendi encode edecek
    # Bu nedenle kategorik sutunlari onceden encode etmemiz gerekiyor
    X_tr_all_enc = X_tr_enc.copy()
    X_te_all_enc = X_te_enc.copy()

    stack.fit(X_tr_all_enc, y_train)

    y_pred_proba = stack.predict_proba(X_te_all_enc)[:, 1]
    best_thr, best_f1 = optimize_threshold(y_test, y_pred_proba)

    print(f"  Stacking F1: {best_f1:.4f} (threshold: {best_thr:.2f})")

    # Optuna study yok -- None dondur
    return stack, None, best_thr, y_pred_proba


# ============================================================================
# STACKING BUILDER: LightGBM meta-learner
# ============================================================================

def build_stacking_lgbm(X_train, y_train, X_test, y_test, cat_features,
                        n_trials=None, use_focal=False,
                        lgbm_params=None, xgb_params=None, nn_params=None):
    """
    Stacking ensemble: LightGBM + XGBoost + SklearnMLP
    Meta-learner: LightGBM
    """
    X_tr_enc, X_te_enc, _ = prepare_for_xgb(X_train, X_test, cat_features)
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    _lgbm_params = lgbm_params or {
        'objective': 'binary', 'verbosity': -1, 'random_state': SEED,
        'class_weight': 'balanced', 'n_estimators': 300,
        'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 8,
        'colsample_bytree': 0.8, 'subsample': 0.8,
    }

    _xgb_params = xgb_params or {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': SEED, 'verbosity': 0,
        'scale_pos_weight': scale_pos, 'n_estimators': 300,
        'learning_rate': 0.05, 'max_depth': 6,
        'use_label_encoder': False,
    }

    _nn_params = nn_params or {
        'hidden_dim': 128, 'n_layers': 2, 'dropout': 0.3,
        'lr': 1e-3, 'weight_decay': 1e-4, 'epochs': 100,
        'batch_size': 64, 'patience': 15,
        'use_focal': use_focal,
    }

    base_estimators = [
        ('lgbm', lgb.LGBMClassifier(**_lgbm_params)),
        ('xgb', xgb.XGBClassifier(**_xgb_params)),
        ('nn', SklearnMLP(**_nn_params)),
    ]

    # LightGBM meta-learner
    meta_learner = lgb.LGBMClassifier(
        objective='binary', verbosity=-1, random_state=SEED,
        class_weight='balanced', n_estimators=100,
        learning_rate=0.05, num_leaves=15, max_depth=4,
    )

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=1,
    )

    print("  Stacking (LightGBM meta-learner) egitiliyor...")
    stack.fit(X_tr_enc, y_train)

    y_pred_proba = stack.predict_proba(X_te_enc)[:, 1]
    best_thr, best_f1 = optimize_threshold(y_test, y_pred_proba)

    print(f"  Stacking F1: {best_f1:.4f} (threshold: {best_thr:.2f})")

    return stack, None, best_thr, y_pred_proba


# ============================================================================
# STACKING MODEL REGISTRY
# ============================================================================

STACKING_BUILDERS = {
    'stacking_lr': build_stacking_lr,
    'stacking_lgbm': build_stacking_lgbm,
}

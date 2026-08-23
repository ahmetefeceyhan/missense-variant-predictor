# -*- coding: utf-8 -*-
"""
Yardimci fonksiyonlar: veri bolme, encoding, tensor donusumu.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, TEST_SIZE


def get_train_test_data(df_prepared, mode, panel=None):
    """
    Verilen mod/panel icin train/test split yap.

    Args:
        df_prepared: Hazirlanmis DataFrame (target sutunu icermeli)
        mode: 'single_panel' veya 'multi_panel'
        panel: single_panel modunda panel ismi

    Returns:
        (X_train, X_test, y_train, y_test, cat_features)
    """
    df = df_prepared.copy()

    if mode == 'single_panel':
        df = df[df['Panel'] == panel].copy()
        df.drop(columns=['Panel'], inplace=True)
    elif mode == 'multi_panel':
        # Panel'i one-hot encode et
        panel_dummies = pd.get_dummies(df['Panel'], prefix='panel', dtype=int)
        df = pd.concat([df.drop(columns=['Panel']), panel_dummies], axis=1)

    X = df.drop(columns=['target'])
    y = df['target']

    # Kategorik sutunlari tespit et
    cat_features = X.select_dtypes(include=['category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    return X_train, X_test, y_train, y_test, cat_features


def prepare_for_xgb(X_train, X_test, cat_features):
    """
    XGBoost icin kategorik sutunlari LabelEncode et.

    Returns:
        (X_train_encoded, X_test_encoded, label_encoders)
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    label_encoders = {}

    for col in cat_features:
        if col in X_tr.columns:
            le = LabelEncoder()
            all_vals = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)]).unique()
            le.fit(all_vals)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))
            label_encoders[col] = le

    return X_tr, X_te, label_encoders


def prepare_for_nn(X_train, X_test, y_train, y_test, cat_features):
    """
    NN/DNN icin: LabelEncode + StandardScaler + tensor donusumu.

    Returns:
        (X_tr_tensor, X_te_tensor, y_tr_tensor, y_te_tensor,
         input_dim, pos_weight, scaler)
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Kategorik sutunlari LabelEncode et
    for col in cat_features:
        if col in X_tr.columns:
            le = LabelEncoder()
            all_vals = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)]).unique()
            le.fit(all_vals)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))

    # Tum sutunlari float'a cevir + NaN doldur
    X_tr = X_tr.astype(np.float32).fillna(0)
    X_te = X_te.astype(np.float32).fillna(0)

    # StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # Tensor donusumu
    X_tr_tensor = torch.FloatTensor(X_tr_scaled)
    X_te_tensor = torch.FloatTensor(X_te_scaled)
    y_tr_tensor = torch.FloatTensor(y_train.values)
    y_te_tensor = torch.FloatTensor(y_test.values)

    input_dim = X_tr_scaled.shape[1]
    pos_weight = torch.FloatTensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])

    return X_tr_tensor, X_te_tensor, y_tr_tensor, y_te_tensor, input_dim, pos_weight, scaler


def get_device():
    """AMD GPU (DirectML) veya CPU cihazini dondur."""
    try:
        import torch_directml
        device = torch_directml.device()
        return device
    except Exception:
        return torch.device('cpu')

# -*- coding: utf-8 -*-
"""
Degerlendirme metrikleri ve threshold optimizasyonu.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_score, recall_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix
)


def optimize_threshold(y_true, y_prob):
    """
    F1 score'u maximize eden threshold'u bul.
    0.10 - 0.90 arasi 0.01 adimlarla tarar.

    Returns:
        (best_threshold, best_f1)
    """
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.10, 0.90, 0.01):
        f1_val = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thr = thr
    return best_thr, best_f1


def compute_all_metrics(y_true, y_pred, y_prob):
    """
    9 kapsamli metrik hesapla.

    Returns:
        dict: f1, auc_roc, auc_pr, mcc, precision, recall,
              specificity, balanced_accuracy, cohens_kappa
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        'auc_pr': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'cohens_kappa': cohen_kappa_score(y_true, y_pred),
    }
    return metrics

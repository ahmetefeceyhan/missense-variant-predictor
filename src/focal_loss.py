# -*- coding: utf-8 -*-
"""
Focal Loss implementasyonlari.
- PyTorch: NN/DNN modelleri icin
- LightGBM/XGBoost: Custom objective fonksiyonlari
"""
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# PyTorch Focal Loss (NN/DNN icin)
# ============================================================================
class FocalLoss(torch.nn.Module):
    """
    Focal Loss: sinif dengesizligi olan veri setlerinde zor orneklere odaklanir.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Pozitif sinif agirligi (0-1 arasi). Default: 0.25
        gamma: Odaklanma parametresi. gamma=0 -> standart BCE. Default: 2.0
        pos_weight: Pozitif sinif icin ek agirlik (BCEWithLogitsLoss uyumlu). Optional.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Standart BCE loss (reduction=none)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t: dogru sinif icin tahmin olasiliği
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)

        # Alpha agirligi: pozitif sinifa alpha, negatife (1-alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulasyon: kolay orneklerin katkisini azalt
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Pos_weight (opsiyonel, BCEWithLogitsLoss uyumlu)
        if self.pos_weight is not None:
            pw = targets * self.pos_weight + (1 - targets) * 1.0
            focal_weight = focal_weight * pw

        loss = focal_weight * bce
        return loss.mean()


# ============================================================================
# LightGBM Focal Loss Custom Objective (sklearn API)
# sklearn API imzasi: fobj(y_true, y_pred) -> (grad, hess)
# NOT: LGBMClassifier.predict_proba() custom objective ile raw score doner,
#      sigmoid ile donusturulmesi gerekir.
# ============================================================================
def focal_objective_lgbm(alpha=0.25, gamma=2.0):
    """
    LightGBM sklearn API icin Focal Loss objective fonksiyonu factory.
    Imza: (y_true, y_pred) -> (grad, hess)
    """
    def fobj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid

        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * p + (1 - y_true) * (1 - p)

        factor = alpha_t * (1 - p_t) ** gamma
        grad = factor * (p - y_true)

        hess = factor * p * (1 - p)
        hess = np.maximum(hess, 1e-7)  # Numerik stabilite

        return grad, hess

    return fobj


def focal_eval_lgbm(alpha=0.25, gamma=2.0):
    """
    LightGBM sklearn API icin Focal Loss evaluation metric factory.
    Imza: (y_true, y_pred) -> (eval_name, eval_result, is_higher_better)
    """
    def feval(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

        loss = -alpha_t * (1 - p_t) ** gamma * np.log(np.clip(p_t, 1e-8, 1.0))
        return 'focal_loss', np.mean(loss), False  # is_higher_better=False

    return feval


# ============================================================================
# XGBoost Focal Loss Custom Objective (sklearn API)
# sklearn API imzasi: fobj(y_true, y_pred) -> (grad, hess)
# ============================================================================
def focal_objective_xgb(alpha=0.25, gamma=2.0):
    """
    XGBoost sklearn API icin Focal Loss objective fonksiyonu factory.
    Imza: (y_true, y_pred) -> (grad, hess)
    """
    def fobj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))

        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * p + (1 - y_true) * (1 - p)

        factor = alpha_t * (1 - p_t) ** gamma
        grad = factor * (p - y_true)
        hess = factor * p * (1 - p)
        hess = np.maximum(hess, 1e-7)

        return grad, hess

    return fobj

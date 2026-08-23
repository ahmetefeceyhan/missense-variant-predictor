# -*- coding: utf-8 -*-
"""
Teknofest Genetik Varyant Patojenite Tahmini - Proje Sabitleri
"""
import os

# Tekrarlanabilirlik
SEED = 42

# Optuna trial sayilari
TRIALS_TREE = 100    # LightGBM ve XGBoost
TRIALS_NN = 50      # NN ve DNN

# Train/test bolme orani
TEST_SIZE = 0.2

# Panel ayarlari (CFTR haric -- tek sinifli)
PANELS_SINGLE = ['General', 'Hereditary_Cancer', 'PAH']

# Dosya yollari
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'open_cravat_curation_v3.csv')
RESULTS_V1_DIR = os.path.join(PROJECT_ROOT, 'results', 'v1_leaked')
RESULTS_V2_DIR = os.path.join(PROJECT_ROOT, 'results', 'v2_clean')
MODELS_V1_DIR = os.path.join(PROJECT_ROOT, 'models', 'v1_leaked')
MODELS_V2_DIR = os.path.join(PROJECT_ROOT, 'models', 'v2_clean')
STUDIES_DIR = os.path.join(PROJECT_ROOT, 'models', 'studies')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
RESULTS_NO_INSIL_DIR = os.path.join(PROJECT_ROOT, 'results', 'v3_no_insil_panels')
RESULTS_WITH_INSIL_DIR = os.path.join(PROJECT_ROOT, 'results', 'v3_with_insil_panels')
RESULTS_STACKING_DIR = os.path.join(PROJECT_ROOT, 'results', 'v4_stacking')
MODELS_STACKING_DIR  = os.path.join(PROJECT_ROOT, 'models',  'v4_stacking')

# FE modlari
FE_MODES = ['no_fe', 'with_fe', 'clean', 'clean_fe', 'full_fe', 'v3']
FE_MODES_LEGACY = ['no_fe', 'with_fe']         # Baseline
FE_MODES_CLEAN = ['clean', 'clean_fe']          # Temiz modlar
FE_MODES_FULL = ['full_fe']                     # In-silico + gelismis FE
FE_MODES_V3 = ['v3']                            # Optimize pipeline + grid search

# Model tipleri
MODEL_TYPES = ['lightgbm', 'xgboost', 'nn', 'dnn']
STACKING_TYPES = ['stacking_lr', 'stacking_lgbm']

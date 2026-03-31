"""
Configuration module with all paths, constants, and settings.
FIXED VERSION - Addresses data leakage concerns.
"""

import os
import sys
import torch
from pathlib import Path
import warnings

# --- 1. GLOBAL SETTINGS ---
RANDOM_SEED = 42
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Configuration loaded. Device: {DEVICE}")

# --- 2. PATHS ---
def _get_base_dir():
    """Dynamically detect project base directory."""
    # Try environment variable first
    if 'THESIS_BASE_DIR' in os.environ:
        return Path(os.environ['THESIS_BASE_DIR'])
    
    # Try to detect from __file__
    try:
        # If this file is in ThesisCoding/src/config.py
        current_file = Path(__file__).resolve()
        if current_file.parent.name == 'src':
            return current_file.parent.parent
    except:
        pass
    
    # Fallback to current working directory
    return Path.cwd()

BASE_DIR = _get_base_dir()

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'  # ✅ NEW: For final DataFrames
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Source file paths
PATH_GOLD_RAW = RAW_DIR / 'GOLD_data_raw.jsonl'
PATH_SILVER_RAW = RAW_DIR / 'SILVER_data_raw.jsonl'

# Interim processed paths (with embeddings but still JSONL)
PATH_GOLD_INTERIM = INTERIM_DIR / 'GOLD_data_processed.jsonl'
PATH_SILVER_INTERIM = INTERIM_DIR / 'SILVER_data_processed.jsonl'

# ✅ NEW: Final processed paths (DataFrame pickles with metadata)
PATH_GOLD_PROCESSED = PROCESSED_DIR / 'GOLD_data_final.pkl'
PATH_SILVER_PROCESSED = PROCESSED_DIR / 'SILVER_data_final.pkl'

# --- 3. MODEL AND TOKENIZATION ---
MODEL_NAME = "ufal/robeczech-base"
MAX_LENGTH = 128

# --- 4. POS FILTERING DEFINITIONS ---
POS_ALLOWED_AGGRESSIVE = {'NOUN', 'ADJ', 'VERB', 'ADV'}
POS_FORBIDDEN_MILD = {'ADP', 'CCONJ', 'SCONJ', 'PUNCT', 'SYM', 'X', 'PART', 'DET', 'PRON', 'NUM', 'AUX'}

# ✅ NEW: Data splitting configuration
SPLIT_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,  # Validation set for threshold optimization
    'random_state': RANDOM_SEED,
}

# ✅ NEW: Embedding computation modes
EMBEDDING_MODES = {
    'token_independent': 'Compute each sentence independently, extract token embeddings',
    'sentence_cls': 'Compute sentence, extract CLS token',
    'sentence_mean': 'Compute sentence, average all token embeddings',
}

# --- 5. VISUALIZATION SETTINGS ---

# Palety
PALETTE_CATEGORICAL_NAME = "Set2"
PALETTE_CONTINUOUS_NAME = "coolwarm"

# Primární barvy (Set2-inspired, akademicky tlumené)
COLORS = {
    # Třídy (label)
    'l0': '#66C2A5',      # Set2 Teal (Neutral)
    'l1': '#8DA0CB',      # Set2  (Anomaly)

    # Datasety
    'gold': '#E78AC3',    # Set2 Pink
    'silver': '#B3B3B3',  # Neutrální šedá

    # Error analysis
    'TP': '#66C2A5',      # Teal  — správně detekovaná anomálie
    'FP': '#FC8D62',      # Coral — falešný poplach
    'TN': '#D9D9D9',      # Světle šedá — správně neutrální
    'FN': '#8DA0CB',      # Set2 Blue — zmeškaná anomálie

    # Utility
    'grid': '#E0E0E0',
}

# Explicitní barvy pro data-splity (Train / Val / Test)
DATASET_COLORS = {
    'Train': '#8DA0CB',   # Set2 Blue
    'Val':   '#E5C494',   # Set2 
    'Test':  '#66C2A5',   # Set2 Teal
}

# Barvy scénářů
SCENARIO_COLORS = {
    'S1a': '#66C2A5',     # Teal
    'S1b': '#FC8D62',     # Coral
    'S1d': '#8DA0CB',     # Blue
    'S1e': '#E78AC3',     # Pink
}

# Centralizovaná vizualizační konfigurace
VIZ_CONFIG = {
    'figure_sizes': {
        'small':  (8, 6),
        'medium': (10, 6),
        'large':  (12, 6),
        'wide':   (14, 5),
        'square': (10, 10),
    },
    'font': {
        'family': 'sans-serif',
        'title':  14,
        'label':  12,
        'tick':   10,
        'legend': 10,
        'annot':  10,
    },
    'style': {
        'sns_context': 'paper',
        'font_scale':  1.1,
        'bg_color':    '#EAEAF2',
        'grid_color':  '#FFFFFF',
        'text_color':  '#333333',
    },
    'projection': {
        'max_samples': 3000,
        'tsne_perplexity': 30,
        'tsne_init': 'pca',
        'tsne_learning_rate': 'auto',
        'umap_n_neighbors': 15,
        'umap_min_dist': 0.1,
    },
    'dpi': {
        'screen': 100,
        'print':  300,
    },
    'alpha': {
        'grid':    0.3,
        'scatter': 0.7,
        'fill':    0.2,
    },
}

# --- 6. MODEL DEFAULTS ---
MODEL_DEFAULTS = {
    'mahalanobis': { 
        'method': 'empirical', # Empirical je bezpečnější pro vysoké dimenze než robust
    },
    'isolation_forest': {
        'contamination': 'auto',
        'n_estimators': 100,
        'n_jobs': -1,
    },
    'ocsvm': {
        'kernel': 'rbf',
        'gamma': 'scale',
        'cache_size': 500,
    },
    'logistic_regression': {
        'max_iter': 1000,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
    },
    'svm_linear': {
        'kernel': 'linear',
        'class_weight': 'balanced',
        'probability': True,
    },
    'svm_rbf': {
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'probability': True,
    },
    'random_forest': {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'n_jobs': -1,
    },
    'xgboost': {
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'use_label_encoder': False,
    },
    'naive_bayes': {},
    'mlp': {
        'hidden_layer_sizes': (100,),
        'max_iter': 500,
    },
}

# --- 7. ANALYSIS CONFIGURATION ---
ANALYSIS_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,
    'test_size_tolerance': 0.2,
    'bootstrap_iterations': 1000,
}

# --- 8. LABELS ---
LABEL_NEUTRAL = 0
LABEL_ANOMALY = 1

# --- 9. MODEL DISPLAY NAMES ---
# Mapping from internal model names (stored in CSVs) to publication-ready labels.
MODEL_DISPLAY_NAMES = {
    'Dummy (Majority)':   'Majority Baseline',
    'Dummy':              'Majority Baseline',
    'Dummy (None)':       'Majority Baseline (None)',
    'Dummy (Mild)':       'Majority Baseline (Mild)',
    'Dummy (Aggressive)': 'Majority Baseline (Aggressive)',
}

# --- 10. XGBOOST HYPERPARAMETER TUNING ---
XGBOOST_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200, 300],
}

XGBOOST_TUNING_CONFIG = {
    'n_iter': 30,
    'cv': 3,
    'scoring': 'average_precision',
    'random_state': RANDOM_SEED,
}

# --- 9. VALIDATION FUNCTION ---
def validate_config():
    """Validate configuration and check critical paths."""
    errors = []
    warnings_list = []
    
    # Check if base directory exists
    if not BASE_DIR.exists():
        errors.append(f"Base directory does not exist: {BASE_DIR}")
    
    # Check if raw data files exist
    if not PATH_GOLD_RAW.exists():
        warnings_list.append(f"Gold raw data not found: {PATH_GOLD_RAW}")
    
    if not PATH_SILVER_RAW.exists():
        warnings_list.append(f"Silver raw data not found: {PATH_SILVER_RAW}")
    
    # Check write permissions
    try:
        test_file = DATA_DIR / '.write_test'
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        errors.append(f"No write permission in DATA_DIR: {e}")
    
    # Report results
    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for err in errors:
            print(f"   - {err}")
        raise RuntimeError("Configuration validation failed")
    
    if warnings_list:
        print("⚠️  CONFIGURATION WARNINGS:")
        for warn in warnings_list:
            print(f"   - {warn}")
    
    print("✅ Configuration validated successfully")
    return True

# Optionally validate on import
# validate_config()  # Uncomment to auto-validate

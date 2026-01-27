import os
import sys
import torch
from pathlib import Path
import warnings

# --- 1. GLOBÁLNÍ NASTAVENÍ ---
RANDOM_SEED = 42
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Konfigurace načtena. Používám zařízení: {DEVICE}")

# --- 2. CESTY K SOUBORŮM (PATHS) ---
# BASE_DIR je kořen projektu (složka ThesisCoding)
BASE_DIR = Path(r'C:\Users\dobes\Documents\UniversityCodingProject\ThesisCoding')

# Datové složky
DATA_DIR = BASE_DIR / 'data'        # C:\...\ThesisCoding\data
RAW_DIR = DATA_DIR / 'raw'          # C:\...\ThesisCoding\data\raw
INTERIM_DIR = DATA_DIR / 'interim'  # Zde se uloží data s embeddingy
VECTORS_DIR = DATA_DIR / 'vectors'  # C:\...\ThesisCoding\data\vectors
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Vytvoření složek, pokud neexistují (včetně interim!)
for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, VECTORS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Konkrétní cesty k datasetům (Zdrojová data)
PATH_GOLD_RAW = RAW_DIR / 'GOLD_data_raw.jsonl'
PATH_SILVER_RAW = RAW_DIR / 'SILVER_data_raw.jsonl'

# --- 3. MODEL A TOKENIZACE ---
MODEL_NAME = "ufal/robeczech-base"
MAX_LENGTH = 128

# --- 4. DEFINICE FILTRACE (POS TAGY) ---
POS_ALLOWED_AGGRESSIVE = {'NOUN', 'ADJ', 'VERB', 'ADV'}
POS_FORBIDDEN_MILD = {'ADP', 'CCONJ', 'SCONJ', 'PUNCT', 'SYM', 'X', 'PART', 'DET', 'PRON', 'NUM', 'AUX'}

# --- 5. VIZUALIZACE A STYL ---
PALETTE_CATEGORICAL_NAME = "pastel"
PALETTE_CONTINUOUS_NAME = "coolwarm"

COLORS = {
    'l0': '#a1c9f4',      # Pastel Blue (Neutrální)
    'l1': '#ff9f9a',      # Pastel Red (Anomálie)
    'gold': '#ffb482',    # Pastel Orange
    'silver': '#c6c6c6',  # Pastel Grey
    'train': '#b3de69',   # Pastel Green
    'test': '#bc80bd',    # Pastel Purple
    'grid': '#e0e0e0',

    # Barvy pro Error Analysis (Confusion Matrix kategorie)
    'TP': '#ff9f9a',      
    'FP': '#f39c12',      
    'TN': '#e0e0e0',      
    'FN': '#9b59b6'       
}

SNS_STYLE = "whitegrid"
SNS_CONTEXT = "paper"
FONT_SCALE = 1.1

# --- 6. ŠTÍTKY PRO KLASIFIKACI ---
LABEL_NEUTRAL = 0
LABEL_ANOMALY = 1
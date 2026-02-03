# Code Review: `src/` Directory - NLP/CL Project

## Executive Summary

This is a well-structured academic research project with clear separation of concerns. However, there are several security risks, configuration issues, and opportunities for improvement in code quality and maintainability. Below is a comprehensive analysis organized by category.

---

## 1. Security Review

### 🔴 **CRITICAL: Hardcoded Absolute Path with Username**

**File:** `src/config.py`

**Issue:**
```python
BASE_DIR = Path(r'C:\Users\dobes\Documents\UniversityCodingProject\ThesisCoding')
```

**Why it matters:**
- Exposes username (`dobes`) - potential privacy/security concern if shared publicly
- Makes code non-portable (won't work on other machines/OS)
- Could expose internal directory structure if published

**Recommendation:**
```python
# Option 1: Relative to this file's location
BASE_DIR = Path(__file__).resolve().parent.parent

# Option 2: Environment variable with fallback
BASE_DIR = Path(os.getenv('THESIS_BASE_DIR', Path.cwd()))

# Option 3: For notebooks - detect if running in notebook
import sys
if 'ipykernel' in sys.modules:
    BASE_DIR = Path.cwd()  # Notebook directory
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
```

### 🟡 **MEDIUM: Unsafe Pickle Loading**

**File:** `src/load_preprocess_data.py`

**Issue:**
```python
with open(save_dir / filename, 'wb') as f:
    pickle.dump(arr, f)
```

**Why it matters:**
- Pickle files can execute arbitrary code when loaded
- No integrity verification
- Vulnerable to malicious file injection

**Recommendation:**
```python
# Add integrity checks and use safer formats when possible
import hashlib

def _dump(data, filename):
    if not data: return
    arr = np.array(data, dtype=np.float32)
    
    filepath = save_dir / filename
    
    # Save with integrity hash
    with open(filepath, 'wb') as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create checksum file
    with open(filepath, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    
    with open(f"{filepath}.sha256", 'w') as f:
        f.write(checksum)
    
    del arr

# For loading - verify integrity
def _load_with_verification(filepath):
    """Load pickle with integrity verification."""
    checksum_path = f"{filepath}.sha256"
    
    if os.path.exists(checksum_path):
        with open(filepath, 'rb') as f:
            data_bytes = f.read()
        
        computed = hashlib.sha256(data_bytes).hexdigest()
        
        with open(checksum_path, 'r') as f:
            expected = f.read().strip()
        
        if computed != expected:
            raise ValueError(f"Checksum mismatch for {filepath}")
        
        return pickle.loads(data_bytes)
    else:
        # Fallback for old files
        with open(filepath, 'rb') as f:
            return pickle.load(f)
```

### 🟡 **MEDIUM: Error Handling Silently Swallows Exceptions**

**Files:** `src/load_preprocess_data.py`, `src/analysis.py`

**Issue:**
```python
try:
    data.append(json.loads(line))
except:  # Bare except - catches everything
    continue
```

**Why it matters:**
- Hides actual errors (JSON syntax, encoding issues, memory problems)
- Makes debugging nearly impossible
- Could skip valid data due to transient errors

**Recommendation:**
```python
import logging

logger = logging.getLogger(__name__)

# Specific exception handling
try:
    data.append(json.loads(line))
except json.JSONDecodeError as e:
    logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
    continue
except UnicodeDecodeError as e:
    logger.error(f"Encoding error on line {line_num}: {e}")
    continue
# Don't catch unexpected exceptions - let them fail loudly
```

### 🟡 **MEDIUM: No Input Validation**

**Files:** Multiple functions across `src/`

**Issue:**
Functions don't validate inputs (e.g., file paths exist, parameters are in valid ranges)

**Recommendation:**
```python
def generate_predictions_csv(model, X_test, y_test, model_name, scenario, 
                              level='sentence', filter_type='aggressive'):
    """Generate predictions CSV with validation."""
    
    # Validate inputs
    if level not in ['sentence', 'token']:
        raise ValueError(f"Invalid level: {level}. Must be 'sentence' or 'token'")
    
    if filter_type not in ['none', 'mild', 'aggressive']:
        raise ValueError(f"Invalid filter_type: {filter_type}")
    
    if X_test.shape[0] != len(y_test):
        raise ValueError(f"Shape mismatch: X_test={X_test.shape[0]}, y_test={len(y_test)}")
    
    # ... rest of function
```

---

## 2. Configuration & Environment Management

### 🔴 **CRITICAL: Magic Numbers and Hardcoded Values**

**Files:** Multiple

**Issues Found:**

1. **`src/analysis.py`**
```python
# Line 134: Hardcoded tolerance
is_silver_missing = len(X_test) < (expected_test_full * 0.8)  # 20% tolerance

# Line 153: Hardcoded random seed (should use config.RANDOM_SEED)
np.random.seed(42)
```

2. **`src/visualization.py`**
```python
# Line 79: Hardcoded plot sizes
plt.figure(figsize=(10, 6))
plt.figure(figsize=(12, 6))
plt.figure(figsize=(8, 6))

# Line 124: Hardcoded max_samples
def compute_projections(X, methods=['PCA', 't-SNE', 'UMAP'], max_samples=3000, random_state=42):

# Line 143: Hardcoded perplexity
projections['t-SNE'] = TSNE(n_components=2, perplexity=30, ...)
```

3. **`src/models.py`**
```python
# Hardcoded hyperparameters
IsolationForest(contamination='auto', n_estimators=100, ...)
OneClassSVM(nu=nu, kernel='rbf', gamma='scale', cache_size=500)
LogisticRegression(max_iter=1000, ...)
```

**Recommendation:**

**Create `src/config.py` additions:**
```python
# --- ANALYSIS CONFIGURATION ---
ANALYSIS = {
    'test_size_tolerance': 0.2,  # For Silver data detection
    'random_seed_override': None,  # If None, use RANDOM_SEED
}

# --- VISUALIZATION CONFIGURATION ---
VIZ = {
    'figure_sizes': {
        'small': (8, 6),
        'medium': (10, 6),
        'large': (12, 6),
        'square': (8, 8),
    },
    'projection': {
        'max_samples': 3000,
        'tsne_perplexity': 30,
        'umap_n_neighbors': 15,
        'umap_min_dist': 0.1,
    },
    'dpi': 150,
}

# --- MODEL HYPERPARAMETERS ---
MODEL_DEFAULTS = {
    'isolation_forest': {
        'contamination': 'auto',
        'n_estimators': 100,
    },
    'ocsvm': {
        'kernel': 'rbf',
        'gamma': 'scale',
        'cache_size': 500,
    },
    'logistic_regression': {
        'max_iter': 1000,
        'solver': 'lbfgs',
    },
    # ... etc
}
```

**Then update usage:**
```python
# In visualization.py
from config import VIZ

plt.figure(figsize=VIZ['figure_sizes']['medium'])

# In models.py
from config import MODEL_DEFAULTS

class IsolationForestWrapper(BaseDetector):
    def __init__(self, contamination='auto', n_estimators=100, random_state=42):
        defaults = MODEL_DEFAULTS['isolation_forest']
        self.model = IsolationForest(
            contamination=contamination or defaults['contamination'],
            n_estimators=n_estimators or defaults['n_estimators'],
            n_jobs=-1, 
            random_state=random_state
        )
```

### 🟡 **MEDIUM: Missing Environment Variable Support**

**File:** `src/config.py`

**Issue:**
No support for environment variables for sensitive/deployment-specific values

**Recommendation:**
```python
import os
from pathlib import Path
from dotenv import load_dotenv  # Add python-dotenv to requirements.txt

# Load .env file if it exists
load_dotenv()

# Environment-aware configuration
BASE_DIR = Path(os.getenv('THESIS_BASE_DIR', Path(__file__).resolve().parent.parent))
MODEL_NAME = os.getenv('BERT_MODEL', "ufal/robeczech-base")
MAX_LENGTH = int(os.getenv('MAX_SEQ_LENGTH', 128))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Device configuration with override
DEVICE_OVERRIDE = os.getenv('TORCH_DEVICE', None)
if DEVICE_OVERRIDE:
    DEVICE = torch.device(DEVICE_OVERRIDE)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Create `.env.example`:**
```bash
# Example environment configuration
# Copy to .env and customize

# Base directory (auto-detected if not set)
# THESIS_BASE_DIR=/path/to/ThesisCoding

# Model configuration
# BERT_MODEL=ufal/robeczech-base
# MAX_SEQ_LENGTH=128

# Computation
# TORCH_DEVICE=cuda
# RANDOM_SEED=42
```

---

## 3. Code Quality & Cleanliness

### 🔴 **HIGH: Significant Code Duplication**

#### **Issue 1: Duplicate `_prepare_long_data` Functions**

**File:** `src/visualization.py`

**Lines 175 and 318** contain nearly identical functions:
- `_prepare_long_data(df_results, metric='auprc')`
- `_prepare_long_data_s2(df_results, metric='auprc')`

**Recommendation:**
```python
def _prepare_long_data(df_results, metric='auprc', extra_id_vars=None):
    """
    Unified function for melting train/test data.
    
    Args:
        df_results: DataFrame with train/test columns
        metric: Metric name (default: 'auprc')
        extra_id_vars: Additional columns to preserve (e.g., ['pooling'])
    """
    col_test = f'test_{metric}'
    col_train = f'train_{metric}'
    
    if col_test not in df_results.columns:
        print(f"⚠️ Warning: Column {col_test} not found. Available: {df_results.columns.tolist()}")
        return pd.DataFrame()
    
    # Base ID vars
    id_vars = ['scenario', 'model']
    
    # Add extra columns if provided
    if extra_id_vars:
        id_vars.extend(extra_id_vars)
    
    # Ensure all id_vars exist in dataframe
    id_vars = [col for col in id_vars if col in df_results.columns]
    
    df_long = df_results.melt(
        id_vars=id_vars,
        value_vars=[col_train, col_test],
        var_name='dataset_col',
        value_name='score'
    )
    
    df_long['Dataset'] = df_long['dataset_col'].apply(
        lambda x: 'Train' if 'train' in x else 'Test'
    )
    return df_long

# Usage:
# For S1: _prepare_long_data(df_results, 'auprc')
# For S2: _prepare_long_data(df_results, 'auprc', extra_id_vars=['pooling', 'filter'])
```

#### **Issue 2: Duplicate Filter Logic**

**Files:** `src/analysis.py` (line 28) and `src/load_preprocess_data.py` (line 172)

Both have `_is_token_kept()` function - should be centralized.

**Recommendation:**

**Create `src/utils.py`:**
```python
"""
Shared utility functions across modules.
"""
import config

def is_token_kept(pos_tag, filter_type):
    """
    Determine if a token passes the specified POS filter.
    
    Args:
        pos_tag: POS tag string (e.g., 'NOUN', 'ADP')
        filter_type: Filter level ('none', 'mild', 'aggressive')
    
    Returns:
        bool: True if token should be kept
    
    Raises:
        ValueError: If filter_type is invalid
    """
    if filter_type == 'none':
        return True
    elif filter_type == 'mild':
        return pos_tag not in config.POS_FORBIDDEN_MILD
    elif filter_type == 'aggressive':
        return pos_tag in config.POS_ALLOWED_AGGRESSIVE
    else:
        raise ValueError(f"Invalid filter_type: '{filter_type}'. "
                        f"Must be one of: 'none', 'mild', 'aggressive'")
```

**Then update imports:**
```python
# In analysis.py and load_preprocess_data.py
from utils import is_token_kept
```

### 🟡 **MEDIUM: Commented-Out Code**

**File:** `src/visualization.py`

**Line 104:** Entire function `plot_supervised_results()` is marked as "Nepouívaná funkce" (Unused function)

**Recommendation:**
```python
# Either:
# 1. Delete if truly unused
# 2. Move to a deprecated.py module with clear documentation
# 3. Keep but add @deprecated decorator

from warnings import warn

def deprecated(reason):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            warn(f"{func.__name__} is deprecated: {reason}", 
                 DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated("Use plot_scenario_breakdown() instead")
def plot_supervised_results(df_results, metric='auprc', title="Srovnání modelů"):
    # ... existing code
```

### 🟡 **MEDIUM: Inconsistent Error Handling Patterns**

**Files:** Multiple

**Issue:**
Mix of `print()` statements, silent failures, and exceptions

**Recommendation:**

**Create `src/logger.py`:**
```python
"""
Centralized logging configuration.
"""
import logging
import sys

def setup_logger(name=None, level=logging.INFO, log_file=None):
    """
    Configure logger with consistent formatting.
    
    Args:
        name: Logger name (default: root)
        level: Logging level
        log_file: Optional file path for logging
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format with emoji for visibility (matching existing style)
    formatter = logging.Formatter(
        '%(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Usage in modules:
# from logger import setup_logger
# logger = setup_logger(__name__)
# logger.info("✅ Models loaded successfully")
# logger.warning("⚠️ Silver data missing")
# logger.error("❌ Critical error")
```

**Replace print statements:**
```python
# Before:
print(f"⚠️ VAROVÁNÍ: Detekován nesoulad dat!")

# After:
logger.warning("Detekován nesoulad dat!")
```

---

## 4. Performance & Optimization

### 🟡 **MEDIUM: Inefficient Data Loading in Loops**

**File:** `src/analysis.py`

**Issue (Lines 91-96):**
```python
def generate_predictions_csv(...):
    # Loads ENTIRE datasets every time function is called
    raw_gold = load_jsonl_texts(path_gold)
    raw_silver = load_jsonl_texts(path_silver)
```

**Why it matters:**
- If called in a loop (e.g., testing multiple models), data is loaded repeatedly
- JSONL parsing is slow
- Memory inefficient

**Recommendation:**
```python
# Option 1: Cache decorator
from functools import lru_cache

@lru_cache(maxsize=4)
def load_jsonl_texts_cached(file_path):
    """Cached version of load_jsonl_texts."""
    return load_jsonl_texts(file_path)

# Option 2: Load once, pass as parameter
def generate_predictions_csv(model, X_test, y_test, model_name, scenario,
                              raw_gold=None, raw_silver=None,  # ADD THESE
                              level='sentence', filter_type='aggressive'):
    """
    Generate predictions CSV.
    
    Args:
        ...
        raw_gold: Pre-loaded Gold JSONL data (optional, loads if None)
        raw_silver: Pre-loaded Silver JSONL data (optional, loads if None)
    """
    # Load only if not provided
    if raw_gold is None:
        raw_gold = load_jsonl_texts(config.INTERIM_DIR / 'GOLD_data_processed.jsonl')
    if raw_silver is None:
        raw_silver = load_jsonl_texts(config.INTERIM_DIR / 'SILVER_data_processed.jsonl')
    
    # ... rest of function

# Usage in notebooks:
# Load once at top of notebook
raw_gold = load_jsonl_texts(...)
raw_silver = load_jsonl_texts(...)

# Pass to all calls
for model in models:
    generate_predictions_csv(model, X, y, name, scenario, 
                            raw_gold=raw_gold, raw_silver=raw_silver)
```

### 🟡 **MEDIUM: Unnecessary Array Copies**

**File:** `src/load_preprocess_data.py`

**Issue (Line 240):**
```python
def _dump(data, filename):
    arr = np.array(data, dtype=np.float32)  # Creates copy
    with open(save_dir / filename, 'wb') as f:
        pickle.dump(arr, f)
    del arr  # Manual cleanup
```

**Recommendation:**
```python
def _dump(data, filename):
    """Save data efficiently."""
    if not data:
        return
    
    # Convert once if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    
    filepath = save_dir / filename
    
    # Use numpy's native save (faster, smaller files)
    np.save(filepath.with_suffix('.npy'), data, allow_pickle=False)
    
    # Or if pickle is required, use protocol 4+
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# For loading:
def _load(filename):
    """Load saved vectors."""
    filepath = save_dir / filename
    
    if filepath.with_suffix('.npy').exists():
        return np.load(filepath.with_suffix('.npy'))
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
```

### 🟢 **LOW: Potential Memory Optimization**

**File:** `src/load_preprocess_data.py`

**Issue (Line 324):**
Streaming JSONL but keeping results in memory during processing

**Recommendation:**
```python
# For very large datasets, consider generator pattern
def generate_vector_artifacts_streaming(interim_path, dataset_name, batch_size=1000):
    """
    Memory-efficient version using batching.
    Processes and saves in chunks.
    """
    store = _initialize_storage()
    batch_count = 0
    
    with open(interim_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing"), 1):
            # ... process line ...
            
            # Save every batch_size lines
            if line_num % batch_size == 0:
                _save_batch(store, dataset_name, batch_count)
                batch_count += 1
                store = _initialize_storage()  # Reset
    
    # Save final batch
    if store:
        _save_batch(store, dataset_name, batch_count)
```

---

## 5. Architecture & Reusability

### 🔴 **HIGH: Analysis Logic Should Be in `src/`, Not Notebooks**

**Issue:**
Based on document structure, notebooks likely contain significant analysis logic that should be extracted.

**Recommendation:**

**Create `src/experiment.py`:**
```python
"""
High-level experiment orchestration.
Encapsulates common patterns from notebooks.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

class ExperimentRunner:
    """
    Manages end-to-end ML experiment workflow.
    
    Example:
        >>> runner = ExperimentRunner(
        ...     data_config={'scenario': 'S1a', 'filter': 'aggressive'},
        ...     model_config={'name': 'LogReg', 'params': {...}}
        ... )
        >>> results = runner.run()
    """
    
    def __init__(self, data_config: Dict, model_config: Dict, random_state: int = 42):
        self.data_config = data_config
        self.model_config = model_config
        self.random_state = random_state
        self.results = {}
    
    def load_data(self):
        """Load and prepare data based on config."""
        from load_preprocess_data import load_data
        return load_data(
            strategy=self.data_config['scenario'],
            level=self.data_config.get('level', 'token'),
            filter_type=self.data_config.get('filter', 'aggressive')
        )
    
    def train_model(self, X_train, y_train):
        """Train model based on config."""
        from models import get_supervised_model
        model = get_supervised_model(
            self.model_config['name'],
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate(self, model, X_test, y_test):
        """Comprehensive evaluation."""
        from evaluation import calculate_metrics, plot_pr_curve
        
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = calculate_metrics(y_test, y_pred, y_probs)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_probs
        }
    
    def run(self, visualize=True, save_results=True):
        """Execute complete experiment pipeline."""
        # 1. Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # 2. Train
        model = self.train_model(X_train, y_train)
        
        # 3. Evaluate
        results = self.evaluate(model, X_test, y_test)
        
        # 4. Visualize
        if visualize:
            from visualization import plot_pr_curve, plot_confusion_matrix_heatmap
            plot_pr_curve(y_test, results['probabilities'])
            plot_confusion_matrix_heatmap(y_test, results['predictions'])
        
        # 5. Save
        if save_results:
            self._save_results(results)
        
        return results

# Usage in notebooks becomes simple:
# runner = ExperimentRunner(
#     data_config={'scenario': 'S1a', 'filter': 'aggressive'},
#     model_config={'name': 'LogReg'}
# )
# results = runner.run()
```

### 🟡 **MEDIUM: Missing Evaluation Module**

**Issue:**
`src/evaluation.py` is referenced but not provided in documents

**Recommendation:**

**Create `src/evaluation.py`:**
```python
"""
Unified evaluation metrics and utilities.
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from typing import Dict, Optional

def calculate_metrics(y_true, y_pred, y_probs=None, 
                      average='binary', pos_label=1) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional, for ROC/PR curves)
        average: Averaging strategy for multiclass
        pos_label: Positive class label
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
    }
    
    if y_probs is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_probs)
            metrics['auprc'] = average_precision_score(y_true, y_probs)
        except ValueError:
            # Handle edge cases (e.g., only one class present)
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        'tp': int(tp), 'fp': int(fp), 
        'tn': int(tn), 'fn': int(fn)
    })
    
    # Additional derived metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    return metrics

def find_optimal_threshold(y_true, y_scores, metric='f1', beta=1.0):
    """
    Find optimal decision threshold for a given metric.
    
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores (higher = more anomalous)
        metric: Metric to optimize ('f1', 'precision', 'recall', 'f_beta')
        beta: Beta parameter for F-beta score
    
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    if metric == 'f1' or metric == 'f_beta':
        numerator = (1 + beta**2) * precisions * recalls
        denominator = (beta**2 * precisions) + recalls
        scores = np.divide(numerator, denominator, 
                          out=np.zeros_like(denominator), 
                          where=denominator!=0)
    elif metric == 'precision':
        scores = precisions
    elif metric == 'recall':
        scores = recalls
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # sklearn returns one extra value for precision/recall
    scores = scores[:-1]
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

def bootstrap_evaluation(model, X, y, n_iterations=100, test_size=0.2, 
                         metric_fn=None, random_state=42):
    """
    Bootstrap resampling for model stability assessment.
    
    Args:
        model: Fitted model (must have predict method)
        X: Feature matrix
        y: Labels
        n_iterations: Number of bootstrap samples
        test_size: Fraction for test split in each iteration
        metric_fn: Function(y_true, y_pred) -> score (default: F1)
        random_state: Random seed
    
    Returns:
        Array of scores from each iteration
    """
    from sklearn.model_selection import train_test_split
    
    if metric_fn is None:
        metric_fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    
    np.random.seed(random_state)
    scores = []
    
    for i in range(n_iterations):
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=random_state + i
        )
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = metric_fn(y_test, y_pred)
        scores.append(score)
    
    return np.array(scores)
```

### 🟡 **MEDIUM: Scenario-Specific Logic Hardcoded**

**File:** `src/analysis.py`

**Issue (Lines 105-170):**
Complex nested if/elif blocks for scenarios make adding new scenarios difficult

**Recommendation:**

**Use Strategy Pattern:**
```python
# Create src/scenario_strategies.py

from abc import ABC, abstractmethod

class ScenarioStrategy(ABC):
    """Base class for dataset composition strategies."""
    
    @abstractmethod
    def compose_dataset(self, gold_meta, silver_meta):
        """
        Compose final dataset from Gold and Silver metadata.
        
        Returns:
            List of metadata dictionaries
        """
        pass

class BaselineStrategy(ScenarioStrategy):
    """S1a/S2a: Gold only, balanced."""
    
    def compose_dataset(self, gold_meta, silver_meta):
        g_l0 = [m for m in gold_meta if m['label'] == 0 and m.get('sent_id') == 1]
        g_l1 = [m for m in gold_meta if m['label'] == 1]
        
        # Undersample L0
        n_anomalies = len(g_l1)
        if len(g_l0) >= n_anomalies:
            np.random.seed(42)
            idx = np.random.choice(len(g_l0), n_anomalies, replace=False)
            g_l0 = [g_l0[i] for i in idx]
        
        return g_l0 + g_l1

class HybridStrategy(ScenarioStrategy):
    """S1e/S2b: Gold + Silver anomalies, balanced Gold L0."""
    
    def compose_dataset(self, gold_meta, silver_meta):
        g_l0 = [m for m in gold_meta if m['label'] == 0 and m.get('sent_id') == 1]
        g_l1 = [m for m in gold_meta if m['label'] == 1]
        s_l1 = [m for m in silver_meta if m['label'] == 1]
        
        all_l1 = g_l1 + s_l1
        n_anomalies = len(all_l1)
        
        # Undersample
        if len(g_l0) >= n_anomalies:
            np.random.seed(42)
            idx = np.random.choice(len(g_l0), n_anomalies, replace=False)
            g_l0 = [g_l0[i] for i in idx]
        
        return g_l0 + all_l1

# Registry
SCENARIO_STRATEGIES = {
    'baseline': BaselineStrategy(),
    'S1a': BaselineStrategy(),
    'S1b': BaselineStrategy(),
    'S1e': HybridStrategy(),
    'S2a': BaselineStrategy(),
    'S2b': HybridStrategy(),
}

# Usage in analysis.py:
def generate_predictions_csv(...):
    # ...
    strategy = SCENARIO_STRATEGIES.get(scenario)
    if not strategy:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    meta_final = strategy.compose_dataset(g_meta, s_meta)
```

---

## 6. Consistency & Style

### 🟡 **MEDIUM: Inconsistent Naming Conventions**

**Issues:**

1. **Function names:**
   - Mix of `snake_case` (good) with overly long names
   - `generate_predictions_csv` vs `plot_pr_curve` (inconsistent verbosity)

2. **Variable names:**
   - Mix of abbreviations: `df`, `g_l0`, `s_l1`, `meta_final`
   - Some very short: `g`, `f`, `i`, `t`

3. **Constants:**
   - `COLORS`, `RANDOM_SEED` (good) vs `MODEL_NAME` (should be `DEFAULT_MODEL_NAME`)

**Recommendation:**

**Create style guide document and follow consistently:**

```python
# Good naming examples:

# Functions: verb_noun pattern
def load_dataset(...)
def calculate_metrics(...)
def plot_confusion_matrix(...)

# Variables: descriptive nouns
# Bad: g_l0, s_l1
# Good: gold_neutral, silver_anomalies

# Constants: ALL_CAPS
DEFAULT_MODEL_NAME = "ufal/robeczech-base"
MAX_SEQUENCE_LENGTH = 128

# Private functions: leading underscore
def _validate_input(...)
def _compute_embeddings(...)

# Classes: PascalCase
class ExperimentRunner:
    pass
```

### 🟡 **MEDIUM: Inconsistent Docstrings**

**Issue:**
Mix of docstring styles and completeness

**Current state:**
```python
# Some functions have full docstrings
def calculate_metrics(...):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        ...
    Returns:
        ...
    """

# Others have minimal or none
def _dump(data, filename):
    if not data: return
```

**Recommendation:**

**Adopt Google Style consistently:**
```python
def generate_vector_artifacts(interim_path: Path, dataset_name: str) -> None:
    """
    Generate and save vector representations from processed JSONL data.
    
    Processes token and sentence embeddings with various filtering strategies,
    saving results as pickle files for downstream ML tasks.
    
    Args:
        interim_path: Path to processed JSONL file with embeddings
        dataset_name: Dataset identifier ('gold' or 'silver')
    
    Returns:
        None. Artifacts are saved to disk at config.VECTORS_DIR
    
    Raises:
        FileNotFoundError: If interim_path doesn't exist
        ValueError: If dataset_name is invalid
    
    Example:
        >>> generate_vector_artifacts(
        ...     Path("data/interim/GOLD_data_processed.jsonl"),
        ...     "gold"
        ... )
    """
```

### 🟡 **MEDIUM: Missing Type Hints**

**Issue:**
Inconsistent use of type annotations

**Recommendation:**

```python
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Before:
def extract_aligned_metadata(raw_data, level='sentence', scenario='baseline', filter_type='aggressive'):
    pass

# After:
def extract_aligned_metadata(
    raw_data: List[Dict],
    level: str = 'sentence',
    scenario: str = 'baseline',
    filter_type: str = 'aggressive'
) -> List[Dict]:
    """Extract metadata aligned with vector representations."""
    pass

# Complex types:
def process_row(row: Dict) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Process single data row into tokens and sentence embeddings."""
    pass

# Use TypedDict for structured dictionaries:
from typing import TypedDict

class TokenMetadata(TypedDict):
    form: str
    lemma: str
    pos: str
    sent_id: int
    embedding: List[float]

def process_tokens(row: Dict) -> List[TokenMetadata]:
    pass
```

---

## 7. Testing & Reliability

### 🔴 **HIGH: No Unit Tests**

**Issue:**
No test suite found

**Recommendation:**

**Create `tests/` directory:**
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_config.py           # Config validation
├── test_load_data.py        # Data loading
├── test_models.py           # Model interfaces
├── test_utils.py            # Utility functions
└── test_evaluation.py       # Metrics
```

**Example test file (`tests/test_utils.py`):**
```python
import pytest
import numpy as np
from src.utils import is_token_kept

class TestIsTokenKept:
    """Tests for POS filtering logic."""
    
    def test_none_filter_keeps_all(self):
        """Filter 'none' should keep all POS tags."""
        assert is_token_kept('NOUN', 'none') is True
        assert is_token_kept('ADP', 'none') is True
        assert is_token_kept('PUNCT', 'none') is True
    
    def test_aggressive_filter_content_words(self):
        """Filter 'aggressive' should keep only content words."""
        assert is_token_kept('NOUN', 'aggressive') is True
        assert is_token_kept('ADJ', 'aggressive') is True
        assert is_token_kept('ADP', 'aggressive') is False
        assert is_token_kept('PUNCT', 'aggressive') is False
    
    def test_mild_filter_removes_function_words(self):
        """Filter 'mild' should remove function words."""
        assert is_token_kept('NOUN', 'mild') is True
        assert is_token_kept('ADP', 'mild') is False
        assert is_token_kept('CONJ', 'mild') is False
    
    def test_invalid_filter_raises_error(self):
        """Invalid filter type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid filter_type"):
            is_token_kept('NOUN', 'invalid')

# Run with: pytest tests/ -v
```

**Create `tests/conftest.py` for fixtures:**
```python
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_embeddings():
    """Generate sample embedding vectors."""
    return np.random.randn(100, 768).astype(np.float32)

@pytest.fixture
def sample_labels():
    """Generate sample binary labels."""
    return np.random.randint(0, 2, size=100)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "vectors").mkdir(parents=True)
    return data_dir
```

### 🟡 **MEDIUM: Brittle Code - Hard to Test**

**Issue:**
Functions have side effects (file I/O, print statements) making unit testing difficult

**Recommendation:**

**Separate I/O from logic:**
```python
# Bad - hard to test
def calculate_and_save_metrics(model, X_test, y_test, output_path):
    """Calculate metrics and save to CSV."""
    y_pred = model.predict(X_test)
    metrics = {...}  # calculation
    
    df = pd.DataFrame([metrics])
    df.to_csv(output_path)  # Side effect - hard to test
    
    print(f"Saved to {output_path}")  # Side effect
    return metrics

# Good - testable
def calculate_metrics(y_true, y_pred):
    """Pure function - easy to test."""
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        # ...
    }

def save_metrics(metrics, output_path):
    """Separate I/O concern."""
    df = pd.DataFrame([metrics])
    df.to_csv(output_path)

# Now you can test calculate_metrics without file system!
def test_calculate_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 0.5
```

---

## 8. Documentation & Maintainability

### 🟡 **MEDIUM: Missing Module-Level Documentation**

**Recommendation:**

**Add comprehensive module docstrings:**
```python
"""
models.py - Model Definitions and Wrappers

This module provides unified interfaces for unsupervised anomaly detection
and supervised classification models used in lexical bias detection.

Key Components:
    Unsupervised Models (M1):
        - MahalanobisDetector: Statistical distance-based detection
        - IsolationForestWrapper: Tree-based isolation
        - OCSVMWrapper: One-class SVM with RBF kernel
    
    Supervised Models (M2):
        - Logistic Regression, SVM (linear/RBF), Random Forest, XGBoost
        - All configured with class_weight='balanced' for imbalanced data

Usage:
    # Unsupervised
    >>> detector = get_unsupervised_model('mahalanobis')
    >>> detector.fit(X_train)
    >>> scores = detector.decision_function(X_test)
    
    # Supervised
    >>> clf = get_supervised_model('LogReg')
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)

Notes:
    - All models follow sklearn API conventions
    - Decision functions return anomaly scores (higher = more anomalous)
    - Random states are managed through config.RANDOM_SEED

Author: Michal Tobiáš Dobeš
Date: 2025-01
License: MIT
"""
```

### 🟡 **MEDIUM: Missing Inline Comments for Complex Logic**

**File:** `src/analysis.py`

**Issue (Lines 127-170):**
Complex scenario logic lacks explanatory comments

**Recommendation:**
```python
elif scenario == 'S1e' or 'S1' in scenario:
    # === HYBRID STRATEGY: Maximize anomaly data while maintaining balance ===
    #
    # Goal: Use ALL available LJMPNIK examples (Gold + Silver) but keep
    #       1:1 ratio with clean L0 examples from Gold only.
    #
    # Challenge: Silver embeddings might be missing if preprocessing failed.
    #            We detect this by comparing expected vs actual test set size.
    
    # --- Step 1: Detect missing Silver data ---
    # Calculate expected test size IF Silver was included
    total_anomalies = len(g_l1) + len(s_l1)
    total_balanced = total_anomalies * 2  # L0 + L1
    expected_test_size = int(total_balanced * 0.2)  # 20% for testing
    
    # If actual test set is significantly smaller, Silver is missing
    tolerance = 0.2  # Allow 20% variance
    is_silver_missing = len(X_test) < (expected_test_size * (1 - tolerance))
    
    if is_silver_missing:
        logger.warning(
            f"Silver data detection failed: "
            f"Expected ~{expected_test_size} test samples, got {len(X_test)}. "
            f"Falling back to Gold-only mode (equivalent to S1b)."
        )
        # Fallback: Use only Gold anomalies
        all_l1 = g_l1
    else:
        # Normal path: Combine Gold + Silver
        all_l1 = g_l1 + s_l1
    
    # --- Step 2: Balance with Gold L0 ---
    # Randomly select L0 tokens to match anomaly count (undersampling)
    n_anomalies = len(all_l1)
    
    if len(g_l0) >= n_anomalies:
        np.random.seed(42)  # CRITICAL: Must match training seed
        idx = np.random.choice(len(g_l0), n_anomalies, replace=False)
        g_l0_selected = [g_l0[i] for i in idx]
    else:
        # Edge case: Fewer L0 than L1 (shouldn't happen with real data)
        logger.warning(f"Insufficient L0 samples ({len(g_l0)}) for balancing ({n_anomalies} anomalies)")
        g_l0_selected = g_l0
    
    meta_final = g_l0_selected + all_l1
```

---


### Create `setup.py` for Package Installation

```python
from setuptools import setup, find_packages

setup(
    name="thesis_ljmpnik_detection",
    version="0.1.0",
    author="Michal Tobiáš Dobeš",
    description="ML-based detection of lexical bias in Czech news",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        # ... other dependencies
    ],
)

# Install with: pip install -e .
# Enables: from thesis_ljmpnik_detection.models import get_supervised_model
```

---

## Priority Action Items

### 🔴 **Critical (Do Immediately)**
1. Fix hardcoded absolute path in `config.py`
2. Add input validation to all public functions
3. Replace bare `except:` with specific exception handling
4. Add logging framework to replace print statements

### 🟡 **High Priority (This Week)**
5. Centralize duplicate code (`_prepare_long_data`, `_is_token_kept`)
6. Extract analysis logic from notebooks to `src/experiment.py`
7. Create `src/utils.py` and `src/evaluation.py`
8. Add type hints to all functions
9. Create comprehensive `requirements.txt`

### 🟢 **Medium Priority (This Month)**
10. Write unit tests (start with utility functions)
11. Add integrity checking to pickle files
12. Refactor scenario logic using Strategy pattern
13. Optimize data loading (caching)
14. Standardize docstrings across all modules

---

## Conclusion

This is a solid research codebase with good separation of concerns and clear module responsibilities. The main areas needing attention are:

1. **Security**: Path handling and pickle safety
2. **Configuration**: Externalize hardcoded values
3. **Code Quality**: Reduce duplication, add tests
4. **Maintainability**: Consistent style, better documentation

Implementing these recommendations will make the code more robust, portable, and suitable for publication alongside your thesis.
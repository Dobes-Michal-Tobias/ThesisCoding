"""
experiments.py - Experiment Running Logic

Encapsulates training loops to keep notebooks clean.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Any

# Import internal modules
import config
import models
import evaluation
import data_splitting

logger = logging.getLogger(__name__)

def run_unsupervised_benchmark(
    scenarios: List[Dict[str, Any]],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run a batch of M1 (Unsupervised) experiments.
    
    Logic:
    1. Load Data (Train=L0, Val=Mixed, Test=Mixed)
    2. Train on Train set
    3. Tune threshold on Validation set
    4. Evaluate on Test set
    """
    results = []
    
    for run_cfg in tqdm(scenarios, desc="Running M1 Experiments"):
        model_name = run_cfg['model']
        filter_type = run_cfg.get('filter', 'mild')
        pooling = run_cfg.get('pooling', 'mean') # Default for S2, ignored in S1
        level = run_cfg.get('level', 'token')
        
        # 1. Load Data
        # We use the convenience function from data_splitting
        # Note: M1 usually implies 'baseline' scenario logic (Train on L0)
        data = data_splitting.get_unsupervised_splits( 
            scenario='baseline',
            level=level,
            filter_type=filter_type,
            pooling=pooling,
            random_state=random_state
        )
        
        X_train = data['X_train'] 
        X_val = data['X_val']     
        y_val = data['y_val']
        X_test = data['X_test']   
        y_test = data['y_test']
        
        # 2. Train Model (Only on Label 0)
        try:
            # Safety check: Ensure we only train on L0
            # (data_splitting should handle this, but double check)
            train_mask = (data['y_train'] == 0)
            X_train_clean = X_train[train_mask]
            
            clf = models.get_unsupervised_model(model_name, random_state=random_state)
            clf.fit(X_train_clean)
            
            # 3. Validation (Threshold Tuning)
            # We need mixed data here (L0 + L1) to find F1
            scores_val = clf.decision_function(X_val)
            best_thresh, best_val_f1 = evaluation.find_optimal_threshold(
                y_val, scores_val, metric='f1'
            )
            
            # 4. Test Evaluation
            scores_test = clf.decision_function(X_test)
            y_pred_test = (scores_test > best_thresh).astype(int)
            
            # Calculate all metrics
            metrics = evaluation.calculate_metrics(y_test, y_pred_test, scores_test)
            
            # 5. Store Result
            result_row = {
                'model': model_name,
                'filter': filter_type,
                'level': level,
                'threshold': best_thresh,
                'val_f1': best_val_f1,
                **metrics # Unpack test metrics
            }
            results.append(result_row)
            
        except Exception as e:
            logger.error(f"Failed run {model_name} / {filter_type}: {e}")
            continue
            
    return pd.DataFrame(results)
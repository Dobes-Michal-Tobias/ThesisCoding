"""
experiments.py - Experiment Running Logic
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Any
from sklearn.model_selection import ParameterGrid

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
    Run experiments and return ALL results (not just the best one).
    """
    results = []
    
    for run_cfg in tqdm(scenarios, desc="Running M1 Experiments"):
        model_name = run_cfg['model']
        filter_type = run_cfg.get('filter', 'mild')
        pooling = run_cfg.get('pooling', 'mean')
        level = run_cfg.get('level', 'token')
        scenario_type = run_cfg.get('scenario', 'baseline') # baseline vs robustness
        
        # Grid Search Parameters
        param_grid = run_cfg.get('param_grid', {})
        param_list = list(ParameterGrid(param_grid)) if param_grid else [{}]

        # 1. Load Data
        try:
            # Voláme naši vylepšenou funkci (viz níže)
            data = data_splitting.get_unsupervised_splits(
                scenario=scenario_type, # Předáváme typ scénáře
                level=level,
                filter_type=filter_type,
                pooling=pooling,
                random_state=random_state
            )
        except Exception as e:
            logger.error(f"❌ Data load failed for {model_name}: {e}")
            continue
            
        X_train = data['X_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

        # 2. Iterate ALL parameters
        for params in param_list:
            try:
                # A. Train (on L0 only)
                clf = models.get_unsupervised_model(model_name, random_state=random_state, **params)
                clf.fit(X_train)
                
                # B. Validation (Find Threshold)
                scores_val = clf.decision_function(X_val)
                val_threshold, val_f1 = evaluation.find_optimal_threshold(
                    y_val, scores_val, metric='f1'
                )
                
                # C. Test Evaluation (Using VAL threshold)
                # To je důležité: Neměníme práh podle testovacích dat!
                scores_test = clf.decision_function(X_test)
                y_pred_test = (scores_test > val_threshold).astype(int)
                
                metrics = evaluation.calculate_metrics(y_test, y_pred_test, scores_test)
                
                # D. Store Result (EVERY RUN)
                result_row = {
                    'scenario': scenario_type,
                    'model': model_name,
                    'filter': filter_type,
                    'level': level,
                    'params': str(params),         # Aby to šlo uložit do CSV
                    **params,                      # Rozbalit params do sloupců pro snadnější filtraci
                    'threshold': val_threshold,
                    'val_f1': val_f1,              # F1 na validaci (podle toho budeme vybírat vítěze)
                    'test_f1': metrics['f1'],      # F1 na testu (to nás zajímá finálně)
                    'test_precision': metrics['precision'],
                    'test_recall': metrics['recall'],
                    'test_auroc': metrics['roc_auc']
                }
                results.append(result_row)
                
            except Exception as e:
                print(f"⚠️ Failed: {model_name} {params} -> {e}")
                continue
            
    return pd.DataFrame(results)
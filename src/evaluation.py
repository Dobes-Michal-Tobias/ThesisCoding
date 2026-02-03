"""
evaluation.py - Metrics and Evaluation Utilities

Calculates comprehensive performance metrics and handles 
threshold optimization for unsupervised methods.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score
)
import logging

logger = logging.getLogger(__name__)

def find_optimal_threshold(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold based on validation data.
    
    Args:
        y_true: True binary labels
        y_scores: Anomaly scores (higher = more anomalous)
        metric: Metric to optimize ('f1')
        
    Returns:
        (best_threshold, best_score)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 for each threshold
    numerator = 2 * precision * recall
    denominator = precision + recall
    
    # Handle division by zero
    f1_scores = np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(numerator), 
        where=denominator != 0
    )
    
    # Find index of max score
    best_idx = np.argmax(f1_scores)
    
    # thresholds array is usually 1 shorter than p/r arrays
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = thresholds[-1]
        
    return best_thresh, f1_scores[best_idx]

def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive dictionary of metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'confusion_tn': tn,
        'confusion_fp': fp,
        'confusion_fn': fn,
        'confusion_tp': tp,
    }
    
    # Advanced metrics requiring scores
    if y_scores is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_scores)
            metrics['auprc'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
            
    return metrics
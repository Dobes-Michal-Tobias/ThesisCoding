"""
Modul pro kvalitativní analýzu a inspekci chyb.
"""
import pandas as pd
import numpy as np

def get_error_analysis_table(metadata, y_true, y_scores, threshold):
    """
    Vytvoří přehledný DataFrame s texty a typem chyby.
    
    Args:
        metadata: Pole slovníků (form, context, source...)
        y_true: Skutečné labely (0/1)
        y_scores: Skóre z modelu
        threshold: Zvolený práh
        
    Returns:
        pd.DataFrame seřazený podle "překvapivosti" (nejhorší chyby nahoře)
    """
    y_pred = (y_scores > threshold).astype(int)
    
    results = []
    
    for i in range(len(y_true)):
        true_lbl = y_true[i]
        pred_lbl = y_pred[i]
        score = y_scores[i]
        meta = metadata[i] # Slovník s textem
        
        # Určení typu chyby
        if true_lbl == 1 and pred_lbl == 1:
            err_type = 'TP' # Correct Anomaly
        elif true_lbl == 0 and pred_lbl == 1:
            err_type = 'FP' # False Alarm (Zajímavé!)
        elif true_lbl == 0 and pred_lbl == 0:
            err_type = 'TN' # Correct Normal
        elif true_lbl == 1 and pred_lbl == 0:
            err_type = 'FN' # Missed Anomaly
            
        results.append({
            'Word': meta['form'],
            'Lemma': meta['lemma'],
            'POS': meta['pos'],
            'Error_Type': err_type,
            'Score': round(score, 2),
            'True_Label': int(true_lbl),
            'Pred_Label': int(pred_lbl),
            'Context': meta['context'],
            'Source': meta['source']
        })
        
    df = pd.DataFrame(results)
    
    # Seřadíme to chytře:
    # 1. FP (Největší omyly - slova co vypadají jako anomálie, ale nejsou) s nejvyšším skóre
    # 2. FN (Chyby - slova co jsou anomálie, ale model je nenašel) s nejnižším skóre
    # 3. TP (Co jsme našli)
    
    # Pomocný sloupec pro řazení
    df['sort_priority'] = df['Error_Type'].map({'FP': 0, 'FN': 1, 'TP': 2, 'TN': 3})
    
    # FP chceme řadit podle skóre sestupně (největší úlety nahoře)
    # FN chceme řadit podle skóre vzestupně (nejvíce "skryté" anomálie nahoře)
    df = df.sort_values(by=['sort_priority', 'Score'], ascending=[True, False])
    
    return df.drop(columns=['sort_priority'])
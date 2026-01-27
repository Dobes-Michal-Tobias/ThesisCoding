"""
Modul pro kvalitativní analýzu modelů.
Spojuje predikce modelu zpět s původním textem (slova/věty) a identifikuje chyby.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config

# --- A. NAČÍTÁNÍ TEXTŮ ---

def load_jsonl_texts(file_path):
    """
    Načte texty z JSONL souboru.
    Vrací seznam slovníků: [{'text': '...', 'label': ...}, ...]
    """
    data = []
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ Soubor nenalezen: {path}")
        return []
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            # Uložíme si to, co potřebujeme (text, lemma, label)
            # Pokud jde o token level, 'text' je pole tokenů. 
            # Pokud sentence level, 'text' je string.
            data.append(rec)
    return data

def get_flat_texts_and_labels(jsonl_data, level='sentence', filter_type='aggressive'):
    """
    Převede složitou JSONL strukturu na plochý seznam textů, který odpovídá vektorům.
    """
    l0_items = []
    l1_items = []
        
    for entry in jsonl_data:
        # --- SENTENCE LEVEL (M2/S2) ---
        if level == 'sentence':
            # ZDE BYLA CHYBA: Správný klíč je 'target_sentence'
            txt = entry.get('target_sentence')
            lbl = entry.get('label')
            
            # Pokud by náhodou klíč chyběl, zkusíme fallback
            if txt is None:
                txt = entry.get('text') or entry.get('sentence')
            
            if txt is not None and lbl is not None:
                if lbl == 0:
                    l0_items.append(txt)
                else:
                    l1_items.append(txt)
                
        # --- TOKEN LEVEL (M2/S1) ---
        elif level == 'token':
            # Tady budeme muset být opatrní, až to spustíte pro S1.
            # Podle vašich klíčů tam vidím 'tokens', ale nevidím 'pos_tags' nebo 'labels' (seznam pro tokeny).
            # Pokud tyto klíče v JSONL chybí, budeme muset logiku pro Token level upravit.
            # Zatím nechávám původní logiku, pokud tam ty klíče někde jsou:
            
            tokens = entry.get('tokens', [])
            pos_tags = entry.get('pos_tags', []) # Pokud chybí, vrátí prázdný list
            labels = entry.get('labels', [])     # Pokud chybí, vrátí prázdný list
            
            # Pokud nemáme tagy/labely pro tokeny, nemůžeme filtrovat -> přeskočíme
            if not tokens or not pos_tags or not labels:
                continue

            for t, pos, lbl in zip(tokens, pos_tags, labels):
                keep = False
                if filter_type == 'none':
                    keep = True
                elif filter_type == 'aggressive':
                    if pos in config.POS_ALLOWED_AGGRESSIVE: keep = True
                elif filter_type == 'mild':
                    if pos not in config.POS_FORBIDDEN_MILD: keep = True
                
                if keep:
                    if lbl == 0:
                        l0_items.append(t)
                    else:
                        l1_items.append(t)
                        
    return np.array(l0_items), np.array(l1_items)

# --- B. TVORBA ANALYTICKÉHO DATAFRAME ---

def create_analysis_df(model, X_test, y_test, texts_test):
    """
    Vytvoří DataFrame spojující text, ground truth, predikci a pravděpodobnost.
    
    Args:
        model: Natrénovaný model (musí mít predict_proba nebo decision_function)
        X_test: Testovací vektory
        y_test: Testovací labely
        texts_test: Testovací texty (musí mít stejnou délku jako X_test!)
    """
    # 1. Získání pravděpodobností
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback pro SVM/Linear
        d = model.decision_function(X_test)
        probs = (d - d.min()) / (d.max() - d.min()) # Normalizace 0-1
        
    # 2. Predikce
    preds = model.predict(X_test)
    
    # 3. Tvorba DF
    df = pd.DataFrame({
        'Text': texts_test,
        'True_Label': y_test,
        'Pred_Label': preds,
        'Prob_Anomaly': probs
    })
    
    # 4. Kategorizace chyby
    def get_category(row):
        if row['True_Label'] == 1 and row['Pred_Label'] == 1: return 'TP'
        if row['True_Label'] == 0 and row['Pred_Label'] == 0: return 'TN'
        if row['True_Label'] == 0 and row['Pred_Label'] == 1: return 'FP'
        if row['True_Label'] == 1 and row['Pred_Label'] == 0: return 'FN'
        return 'Unknown'

    df['Error_Type'] = df.apply(get_category, axis=1)
    
    return df

# --- C. REPORTING FUNKCE ---

def show_top_anomalies(df, n=15):
    """Ukáže texty, které model s nejvyšší jistotou označil za anomálie."""
    print(f"\n🔥 TOP {n} DETEKOVANÝCH ANOMÁLIÍ (Nejvyšší jistota modelu):")
    # Filtrujeme jen ty, co model označil jako 1 (Anomaly)
    subset = df[df['Pred_Label'] == 1].sort_values(by='Prob_Anomaly', ascending=False).head(n)
    
    # Formátovaný výpis
    print(f"{'Prob':<8} | {'Type':<4} | {'Text'}")
    print("-" * 60)
    for _, row in subset.iterrows():
        # Zvýraznění chyby (FP) hvězdičkou
        mark = "⚠️" if row['Error_Type'] == 'FP' else "✅"
        print(f"{row['Prob_Anomaly']:.4f} | {row['Error_Type']:<4} | {row['Text']} {mark}")

def show_worst_errors(df, error_type='FP', n=15):
    """
    Ukáže nejhorší chyby.
    FP (False Positive): Neutrální texty, kde si byl model nejjistější, že jsou anomálie.
    FN (False Negative): Anomální texty, které model s jistotou označil za neutrální.
    """
    print(f"\n❌ TOP {n} CHYB TYPU {error_type} (Kde se model nejvíc spletl):")
    
    if error_type == 'FP':
        # Chceme FP s nejvyšším Prob_Anomaly (model si myslel, že je to anomálie)
        subset = df[df['Error_Type'] == 'FP'].sort_values(by='Prob_Anomaly', ascending=False).head(n)
    else:
        # Chceme FN s nejnižším Prob_Anomaly (model si myslel, že je to určitě Neutral)
        subset = df[df['Error_Type'] == 'FN'].sort_values(by='Prob_Anomaly', ascending=True).head(n)
        
    if subset.empty:
        print("   (Žádné chyby tohoto typu nenalezeny. Skvělá práce modelu!)")
        return

    print(f"{'Prob':<8} | {'Text'}")
    print("-" * 60)
    for _, row in subset.iterrows():
        print(f"{row['Prob_Anomaly']:.4f} | {row['Text']}")
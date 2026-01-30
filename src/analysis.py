"""
Modul pro kvalitativní analýzu modelů.
Slouží k propojení vektorových predikcí s původním textem.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import config

# --- POMOCNÉ FUNKCE PRO NAČÍTÁNÍ ---

def load_jsonl_texts(file_path):
    """Načte surová JSONL data."""
    data = []
    path = Path(file_path)
    if not path.exists():
        print(f"Soubor nenalezen: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def _is_token_kept(pos, filter_type):
    """Kopie logiky filtrace z preprocessingu (musí být 1:1)."""
    if filter_type == 'none': return True
    # Definice sad tagů
    
    if filter_type == 'aggressive':
        return pos in config.POS_ALLOWED_AGGRESSIVE
    if filter_type == 'mild':
        return pos not in config.POS_FORBIDDEN_MILD
    return False

# --- HLAVNÍ EXTRAKČNÍ LOGIKA ---

def extract_aligned_metadata(raw_data, level='sentence', scenario='baseline', filter_type='aggressive'):
    """
    Projdede raw data a extrahuje metadata (texty, tokeny, labely)
    PŘESNĚ v tom pořadí, v jakém byly generovány vektory.
    
    Returns:
        list of dict: [{'text': ..., 'token': ..., 'label': ...}, ...]
    """
    metadata_list = []
    
    # 1. SENTENCE LEVEL LOGIKA
    if level == 'sentence':
        # Procházíme data
        for entry in raw_data:
            s_vecs = entry.get('sentence_vectors', {})
            
            # A) Target Sentence (ID '1') - Vždy přítomna v datech
            if '1' in s_vecs:
                txt = entry.get('target_sentence') or entry.get('text')
                lbl = entry.get('label')
                if txt is not None:
                    metadata_list.append({
                        'text': txt,
                        'analyzed_token': None, # U vět nedává smysl
                        'label': lbl,
                        'origin': 'target'
                    })

            # B) Context Sentences (ID '0', '2') - Pouze pro ROBUSTNESS a jen z GOLD dat
            # Pozor: Logika musí odpovídat tomu, jak jsme skládali X_train/X_test
            # Pokud voláme tuto funkci pro Silver data, kontext obvykle ignorujeme (v load_data_s2 se context bere jen z Gold)
            if scenario == 'robustness' and 'label' in entry and entry['label'] is not None: 
                 # Jednoduchá heuristika: Context bereme jen pokud je zdroj 'gold' (což poznáme podle entry, nebo to vyřešíme vně)
                 # Zde pro jistotu extrahujeme vše, filtrování proběhne při skládání datasetu
                 pass 

    # 2. TOKEN LEVEL LOGIKA
    elif level == 'token':
        for entry in raw_data:
            tokens = entry.get('tokens', [])
            # Pokud nemáme tokeny s pos_tags, nemůžeme filtrovat
            if not tokens: continue
            
            # Musíme iterovat přes tokeny a aplikovat filtr
            for t in tokens:
                pos = t.get('pos')
                form = t.get('form')
                sent_id = t.get('sent_id') # 0, 1, 2
                emb = t.get('embedding') # Jen kontrola, že existuje
                
                # Filtr: Bereme jen tokeny, které mají embedding a projdou POS filtrem
                if emb is None: continue
                
                # Logika výběru tokenů (musí sedět s load_preprocess_data.py -> _process_token_level)
                # V M2/S1 bereme anomálie (L1) a Target L0 (L0). Context tokeny (sent_id != 1) se braly jen do 'gold_context'
                
                # Zjednodušení: Vracíme všechno, co projde POS filtrem. 
                # Následná selekce (L0 vs L1) proběhne vně funkce.
                if _is_token_kept(pos, filter_type):
                    
                    # Určení labelu pro daný token
                    # V datasetu je label pro celou větu.
                    # Token je anomálie jen pokud (label_věty==1 AND token==target_token AND sent_id==1)
                    row_label = entry.get('label')
                    target_token_str = entry.get('target_token')
                    
                    is_anomaly = (row_label == 1 and form == target_token_str and sent_id == 1)
                    token_label = 1 if is_anomaly else 0
                    
                    # Ukládáme
                    metadata_list.append({
                        'text': entry.get('target_sentence'), # Kontext (celá věta)
                        'analyzed_token': form,               # Co analyzujeme
                        'label': token_label,
                        'sent_id': sent_id
                    })
                    
    return metadata_list

# --- FUNKCE PRO GENEROVÁNÍ VÝSLEDKŮ (PUBLIC) ---

def generate_predictions_csv(model, X_test, y_test, model_name, scenario, level='sentence', filter_type='aggressive'):
    """
    Kompletní pipeline s automatickou detekcí chybějících Silver dat (Fallback S1e -> S1b).
    """
    print(f"🔄 Generuji CSV report pro: {model_name} ({scenario}, {level})")
    
    # 1. Načtení Raw Dat
    path_gold = config.INTERIM_DIR / 'GOLD_data_processed.jsonl'
    path_silver = config.INTERIM_DIR / 'SILVER_data_processed.jsonl'
    
    raw_gold = load_jsonl_texts(path_gold)
    raw_silver = load_jsonl_texts(path_silver)
    
    meta_final = [] 
    
    # ---------------------------------------------------------
    # A) REKONSTRUKCE DAT
    # ---------------------------------------------------------
    
    if level == 'sentence':
        # ... (Logika pro věty - beze změny) ...
        g_meta = extract_aligned_metadata(raw_gold, level='sentence')
        s_meta = extract_aligned_metadata(raw_silver, level='sentence')
        
        g_l0 = [m for m in g_meta if m['label'] == 0]
        g_l1 = [m for m in g_meta if m['label'] == 1]
        s_l1 = [m for m in s_meta if m['label'] == 1]
        
        g_context = []
        if scenario == 'robustness':
            for entry in raw_gold:
                s_vecs = entry.get('sentence_vectors', {})
                if '0' in s_vecs: g_context.append({'text': entry.get('context_prev'), 'label': 0, 'origin': 'context'})
                if '2' in s_vecs: g_context.append({'text': entry.get('context_next'), 'label': 0, 'origin': 'context'})
        
        if scenario == 'baseline':
            meta_final = g_l0 + g_l1
        elif scenario == 'robustness':
            meta_final = g_l0 + g_context + g_l1 + s_l1
            
    elif level == 'token':
        # -- Token Level Logic --
        g_meta = extract_aligned_metadata(raw_gold, level='token', filter_type=filter_type)
        s_meta = extract_aligned_metadata(raw_silver, level='token', filter_type=filter_type)
        
        # 1. Rozdělení
        g_l0 = [m for m in g_meta if m['label'] == 0 and m['sent_id'] == 1]
        g_l1 = [m for m in g_meta if m['label'] == 1]
        s_l1 = [m for m in s_meta if m['label'] == 1]
        
        # 2. Logika skládání
        if scenario == 'S1a': 
            meta_final = g_l0 + g_l1
            
        elif scenario == 'S1b':
            n_anomalies = len(g_l1)
            if len(g_l0) > n_anomalies:
                np.random.seed(42)
                idx = np.random.choice(len(g_l0), n_anomalies, replace=False)
                g_l0_selected = [g_l0[i] for i in idx]
                meta_final = g_l0_selected + g_l1
            else:
                meta_final = g_l0 + g_l1

        elif scenario == 'S1d':
            # Train Noisy -> Test Clean (pouze Gold)
            meta_final = g_l0 + g_l1
            
        elif scenario == 'S1e' or 'S1' in scenario:
            # Hybrid (Gold + Silver L1)
            
            # --- DETEKCE CHYBĚJÍCÍCH SILVER VEKTORŮ ---
            # Vypočítáme očekávanou velikost testovací sady, kdybychom měli i Silver
            total_full = (len(g_l1) + len(s_l1)) * 2 # Anomálie + Balance L0
            expected_test_full = int(total_full * 0.2)
            
            # Pokud je X_test výrazně menší, než čekáme se Silver daty,
            # znamená to, že se model natrénoval bez nich (Fallback na Gold Only).
            is_silver_missing = len(X_test) < (expected_test_full * 0.8) # 20% tolerance
            
            if is_silver_missing:
                print(f"⚠️ VAROVÁNÍ: Detekován nesoulad dat!")
                print(f"   Očekáváno cca {expected_test_full} testovacích vzorků (se Silver), ale máme jen {len(X_test)}.")
                print(f"   -> Model byl pravděpodobně trénován BEZ Silver dat (chybí vektory?).")
                print(f"   -> Přepínám analýzu na 'Gold Only' režim (jako S1b), aby to prošlo.")
                
                # Fallback: Použijeme jen Gold anomálie
                all_l1 = g_l1 
            else:
                all_l1 = g_l1 + s_l1

            # --- VÝBĚR L0 (UNDERSAMPLING) ---
            # Musí odpovídat počtu anomálií (ať už s nebo bez Silver)
            n_anomalies = len(all_l1)
            
            if len(g_l0) >= n_anomalies:
                np.random.seed(42) # DŮLEŽITÉ: Musí sedět s tréninkem
                idx = np.random.choice(len(g_l0), n_anomalies, replace=False)
                g_l0_selected = [g_l0[i] for i in idx]
            else:
                g_l0_selected = g_l0

            meta_final = g_l0_selected + all_l1

    # ---------------------------------------------------------
    # B) SPLIT
    # ---------------------------------------------------------
    if not meta_final:
        return None

    labels_all = [m['label'] for m in meta_final]
    
    # Split
    _, meta_test, _, _ = train_test_split(
        meta_final, labels_all, 
        test_size=0.2, 
        stratify=labels_all, 
        random_state=42
    )
    
    # KONTROLA POČTŮ
    if len(meta_test) != len(X_test):
        print(f"❌ KRIITICKÁ CHYBA: Ani po korekci počty nesedí.")
        print(f"   Metadata: {len(meta_test)}, Vektory: {len(X_test)}")
        return None

    # ... (Zbytek beze změny) ...
    
    # C) PREDIKCE
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        d = model.decision_function(X_test)
        probs = (d - d.min()) / (d.max() - d.min())

    preds = model.predict(X_test)
    
    df = pd.DataFrame(meta_test)
    df['prob_anomaly'] = probs
    df['pred_label'] = preds
    df['true_label'] = y_test
    
    conditions = [
        (df['true_label'] == 1) & (df['pred_label'] == 1),
        (df['true_label'] == 0) & (df['pred_label'] == 0),
        (df['true_label'] == 0) & (df['pred_label'] == 1),
        (df['true_label'] == 1) & (df['pred_label'] == 0)
    ]
    df['error_category'] = np.select(conditions, ['TP', 'TN', 'FP', 'FN'], default='Unknown')
    df['model'] = model_name
    df['scenario'] = scenario
    
    cols = ['text', 'analyzed_token', 'true_label', 'pred_label', 'prob_anomaly', 'error_category', 'model', 'scenario']
    cols = [c for c in cols if c in df.columns]
    
    return df[cols]
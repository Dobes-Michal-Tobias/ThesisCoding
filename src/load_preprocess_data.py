"""
Modul pro načítání a preprocessing dat.
Obsahuje funkce pro NLP pipeline (spaCy + RobeCzech) a generování vektorů.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import spacy_udpipe
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import pickle
import json

# Import konfigurace
import config

nlp = None
tokenizer = None
model = None

def initialize_models():
    """
    Inicializuje spaCy a BERT modely, pokud ještě nejsou načteny.
    """
    global nlp, tokenizer, model
    
    if nlp is None:
        print("⏳ Načítám spaCy-UDPipe model ('cs-pdt')...")
        # Stažení modelu, pokud není (bezpečné volání)
        try:
            spacy_udpipe.load("cs-pdt")
        except:
            spacy_udpipe.download("cs-pdt")
        nlp = spacy_udpipe.load("cs-pdt")
    
    if model is None:
        print(f"⏳ Načítám RobeCzech model ('{config.MODEL_NAME}')...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, add_prefix_space=True)
        model = AutoModel.from_pretrained(config.MODEL_NAME)
        model.to(config.DEVICE)
        model.eval()
        
    print("✅ Modely připraveny.")

def get_bert_embeddings(text_list):
    """
    Získá embeddingy pro seznam slov pomocí BERTa.
    Vrací: (aligned_embeddings, cls_token, sent_mean_embedding)
    """
    # Tokenizace pro BERT
    inputs = tokenizer(
        text_list,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH
    ).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Získání hidden states [batch, seq_len, hidden]
    last_hidden_states = outputs.last_hidden_state
    
    # --- A) CLS Token ---
    # První token v sekvenci (index 0)
    cls_embedding = last_hidden_states[0, 0, :].cpu().numpy()
    
    # --- B) Sentence Mean Pooling (NOVÉ) ---
    # Musíme vzít průměr všech tokenů, ale IGNOROVAT padding!
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sent_mean_embedding = (sum_embeddings / sum_mask)[0].cpu().numpy()

    # --- C) Word Alignment (Původní logika) ---
    # Zde pracujeme s [0], protože batch_size=1
    sequence_output = last_hidden_states[0].cpu()
    word_ids = inputs.word_ids()
    aligned_embeddings = []
    
    current_word_idx = None
    current_subtokens = []
    
    for i, word_id in enumerate(word_ids):
        if word_id is None: # Skip special tokens [CLS], [SEP]
            continue
            
        if word_id != current_word_idx:
            if current_subtokens:
                avg_emb = torch.stack(current_subtokens).mean(dim=0).numpy()
                aligned_embeddings.append(avg_emb)
            current_word_idx = word_id
            current_subtokens = []
        
        current_subtokens.append(sequence_output[i])
        
    if current_subtokens:
        avg_emb = torch.stack(current_subtokens).mean(dim=0).numpy()
        aligned_embeddings.append(avg_emb)
        
    # VRACÍME TŘI HODNOTY (Tokens, CLS, Mean)
    return aligned_embeddings, cls_embedding, sent_mean_embedding

def process_row(row):
    """
    Zpracuje jeden řádek datasetu.
    Vrací: (final_tokens, sentence_embeddings_dict)
    """
    texts_parts = [
        (0, row.get('context_prev')), 
        (1, row.get('target_sentence')), 
        (2, row.get('context_next'))
    ]
    
    final_tokens = []
    
    # Slovník pro všechny věty (klíč bude string '0', '1', '2')
    all_sent_embs = {}
    
    for sent_id, text in texts_parts:
        if not text or pd.isna(text):
            continue
            
        # 1. SpaCy
        doc = nlp(text)
        words_for_bert = []
        spacy_tokens_info = []
        
        for token in doc:
            if token.is_space: continue
            words_for_bert.append(token.text)
            spacy_tokens_info.append({
                "form": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "sent_id": sent_id
            })
            
        if not words_for_bert: continue
            
        # 2. BERT Embeddingy
        embeddings, cls_emb, mean_emb = get_bert_embeddings(words_for_bert)
        
        # 3. ZMĚNA: Ukládáme embeddingy pro KAŽDOU větu (nejen pro target)
        all_sent_embs[str(sent_id)] = {
            'cls': cls_emb.tolist(),
            'mean': mean_emb.tolist()
        }
        
        # 4. Spojení info pro slova
        for i, info in enumerate(spacy_tokens_info):
            if i < len(embeddings):
                info['embedding'] = embeddings[i].tolist()
                final_tokens.append(info)
                
    # Vracíme tokens a slovník se všemi větnými vektory
    return final_tokens, all_sent_embs

def create_interim_jsonl(input_path, output_path):
    """
    Načte RAW JSONL, přidá embeddingy a uloží do Interim JSONL.
    """
    initialize_models()
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
                
    df = pd.DataFrame(data)
    print(f"🚀 Zpracovávám {len(df)} řádků z {input_path}...")
    
    # Aplikace process_row s progress barem
    tqdm.pandas(desc="NLP Pipeline")
    
    # --- ZMĚNA ZDE ---
    # process_row nyní vrací (tokens, sent_embs_dict)
    # Použijeme pomocnou lambda funkci pro aplikaci a výsledek uložíme do dočasné Series
    processed_series = df.progress_apply(process_row, axis=1)
    
    # Rozbalení výsledků do sloupců DataFrame
    # 1. Sloupec 'tokens' (obsahuje seznam slov s embeddingy)
    df['tokens'] = processed_series.apply(lambda x: x[0])

    # 2. ZMĚNA: Ukládáme celý slovník větných embeddingů
    # Bude obsahovat keys "0", "1", "2" (pokud existují)
    df['sentence_vectors'] = processed_series.apply(lambda x: x[1])
    # (Staré sloupce 'sent_cls_emb' už nepotřebujeme, vše je v 'sentence_vectors')
    
    # Uložení
    print(f"💾 Ukládám Interim dataset do {output_path}...")
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print("✅ Hotovo.")
    return df

# --- FUNKCE PRO GENEROVÁNÍ VEKTOROVÝCH ARTEFAKTŮ ---
def _initialize_storage():
    """Připraví prázdné kontejnery pro data."""
    store = {
        'token': {
            'none': {'l0': [], 'l1': []},
            'mild': {'l0': [], 'l1': []},
            'aggressive': {'l0': [], 'l1': []}
        },
        'sentence': {
            'mean': {'l0': [], 'l1': []},
            'cls': {'l0': [], 'l1': []}
        },
        # Augmentace pro Gold (Context)
        'gold_context': {
            'token': {'none': [], 'mild': [], 'aggressive': []},
            'sentence': {'mean': [], 'cls': []}
        }
    }
    return store

def _is_token_kept(pos_tag, filter_type):
    """Rozhodne, zda token projde filtrem."""
    if filter_type == 'none':
        return True
    if filter_type == 'mild':
        return pos_tag not in config.POS_FORBIDDEN_MILD
    if filter_type == 'aggressive':
        return pos_tag in config.POS_ALLOWED_AGGRESSIVE
    return False

def _process_sentence_level(row, store, dataset_name):
    """
    Zpracuje embeddingy celých vět.
    Čte 'True CLS' a 'True Mean' z připraveného slovníku sentence_vectors.
    """
    label = row['label']
    
    # Načteme slovník s vektory (pokud chybí, je prázdný)
    sent_vecs = row.get('sentence_vectors', {})
    if not sent_vecs:
        return # Nemáme data, končíme

    # --- A. TARGET VĚTA (ID '1') ---
    target_data = sent_vecs.get('1')
    
    if target_data:
        mean_emb = target_data['mean']
        cls_emb = target_data['cls']
        
        # Ukládání (Logic: Všechno bereme, Gold i Silver)
        key = 'l1' if label == 1 else 'l0'
        
        # Ukládáme do store
        store['sentence']['mean'][key].append(mean_emb)
        store['sentence']['cls'][key].append(cls_emb)

    # --- B. KONTEXTOVÉ VĚTY (ID '0', '2') - Pouze pro Gold Augmentaci ---
    if dataset_name == 'gold':
        for ctx_id in ['0', '2']: # Pozor: klíče v JSON jsou stringy
            ctx_data = sent_vecs.get(ctx_id)
            
            if ctx_data:
                store['gold_context']['sentence']['mean'].append(ctx_data['mean'])
                store['gold_context']['sentence']['cls'].append(ctx_data['cls'])

def _process_token_level(row, store, dataset_name):
    """Zpracuje embeddingy jednotlivých tokenů."""
    label = row['label']
    target_token_form = row.get('target_token')
    
    for t in row['tokens']:
        emb = t['embedding']
        pos = t['pos']
        form = t['form']
        sent_id = t['sent_id']
        
        # Identifikace role tokenu
        is_anomaly = (label == 1 and form == target_token_form and sent_id == 1)
        is_target_l0 = (label == 0 and sent_id == 1) # Čisté slovo v target větě
        is_context = (sent_id != 1)
        
        # Pokud je to Silver a není to anomálie, token nás nezajímá (Silver L0 zahazujeme)
        # if dataset_name == 'silver' and not is_anomaly: continue

        for f_type in ['none', 'mild', 'aggressive']:
            if _is_token_kept(pos, f_type):
                if is_anomaly:
                    store['token'][f_type]['l1'].append(emb)
                
                elif is_target_l0: # Odstraněno: and dataset_name == 'gold'
                    store['token'][f_type]['l0'].append(emb)
                
                elif is_context and dataset_name == 'gold':
                    store['gold_context']['token'][f_type].append(emb)

def _save_to_disk(store, dataset_name):
    """Uloží nasbíraná data do .pkl souborů."""
    save_dir = config.VECTORS_DIR / dataset_name.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving artifacts to {save_dir}...")

    def _dump(data, filename):
        if not data: return
        # Konverze na numpy array pro efektivitu
        arr = np.array(data, dtype=np.float32)
        with open(save_dir / filename, 'wb') as f:
            pickle.dump(arr, f)
        del arr # Úklid paměti

    # 1. Uložení Tokenů
    for f_type, dict_l0_l1 in store['token'].items():
        _dump(dict_l0_l1['l0'], f"{dataset_name}_token_{f_type}_l0.pkl")
        _dump(dict_l0_l1['l1'], f"{dataset_name}_token_{f_type}_l1.pkl")
        
    # 2. Uložení Vět
    for pool_type, dict_l0_l1 in store['sentence'].items():
        _dump(dict_l0_l1['l0'], f"{dataset_name}_sent_{pool_type}_l0.pkl")
        _dump(dict_l0_l1['l1'], f"{dataset_name}_sent_{pool_type}_l1.pkl")

    # 3. Uložení Gold Augmentace - ZMĚNA ZDE
    if dataset_name == 'gold':
        # Nyní ukládáme oba typy vektorů zvlášť
        _dump(store['gold_context']['sentence']['mean'], "gold_context_sent_mean.pkl")
        _dump(store['gold_context']['sentence']['cls'], "gold_context_sent_cls.pkl")
        
        for f_type, data in store['gold_context']['token'].items():
            _dump(data, f"gold_context_token_{f_type}.pkl")
            
    print("✅ Artifacts saved successfully.")

# --- HLAVNÍ FUNKCE (Public) ---

def generate_vector_artifacts(interim_path, dataset_name):
    """
    Hlavní řídící funkce. Čte soubor řádek po řádku a volá procesory.
    """
    print(f"\n🔨 Generuji vektorové artefakty pro: {dataset_name.upper()}")
    
    if not os.path.exists(interim_path):
        print(f"❌ Error: File not found: {interim_path}")
        return

    # 1. Inicializace
    store = _initialize_storage()

    # Zjištění počtu řádků (pro progress bar)
    total_lines = sum(1 for _ in open(interim_path, 'r', encoding='utf-8'))
    
    # 2. Hlavní smyčka (Streaming)
    print(f"   -> Processing {total_lines} lines stream-wise...")
    
    with open(interim_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Extracting Vectors"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Delegování práce na pod-funkce
            _process_sentence_level(row, store, dataset_name)
            _process_token_level(row, store, dataset_name)
            
    # 3. Uložení výsledků
    _save_to_disk(store, dataset_name)
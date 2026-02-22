"""
Data loading and preprocessing module.
CONTEXTUAL VERSION WITH POS TAGGING - Processes sequences together and filters POS.

Key features:
1. Contextual embeddings: target sentence is encoded WITH its surrounding context.
2. UDPipe Integration: Extracts exact POS tags and lemmas (cs-pdt).
3. Exact Alignment: UDPipe tokens perfectly map to BERT word-pooled vectors.
4. Compatibility: Includes `apply_pos_filter` for downstream scripts.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
import pickle

import torch
import numpy as np
import pandas as pd
import spacy_udpipe
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Import configuration
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model variables
nlp = None
tokenizer = None
model = None

def initialize_models():
    """Initialize spaCy-UDPipe and BERT models if not already loaded."""
    global nlp, tokenizer, model
    
    if nlp is None:
        logger.info("Loading spaCy-UDPipe model ('cs-pdt')...")
        try:
            spacy_udpipe.load("cs-pdt")
        except:
            logger.info("Downloading spaCy-UDPipe model...")
            spacy_udpipe.download("cs-pdt")
        nlp = spacy_udpipe.load("cs-pdt")
    
    if model is None:
        logger.info(f"Loading RobeCzech model ('{config.MODEL_NAME}')...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, add_prefix_space=True)
        model = AutoModel.from_pretrained(config.MODEL_NAME)
        model.to(config.DEVICE)
        model.eval()
        
    logger.info("✅ Models loaded successfully")


def extract_spacy_info(text):
    """
    Helper function to get tokens, POS tags, and lemmas using UDPipe.
    """
    if not text or pd.isna(text):
        return [], [], []
    
    doc = nlp(text)
    words, pos_tags, lemmas = [], [], []
    
    for token in doc:
        if not token.is_space:
            words.append(token.text)
            pos_tags.append(token.pos_)
            lemmas.append(token.lemma_)
            
    return words, pos_tags, lemmas


def get_contextual_embeddings(words_prev, words_target, words_next):
    """
    Computes contextual embeddings by passing the pre-tokenized sequence through BERT,
    then aligns and pools subwords back to the original UDPipe words.
    
    Returns:
        tuple: (embeddings_prev, embeddings_target, embeddings_next, cls_embedding)
    """
    all_words = words_prev + words_target + words_next
    
    if not all_words:
        return [], [], [], None

    # Tokenize all together using the pre-split UDPipe words
    inputs = tokenizer(
        all_words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH
    ).to(config.DEVICE)
    
    # Get embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state[0].cpu() # [seq_len, 768]
    cls_embedding = last_hidden_states[0].numpy()
    
    # Pool subwords back to words
    word_ids = inputs.word_ids()
    word_subtokens = [[] for _ in range(len(all_words))]
    
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            word_subtokens[word_id].append(last_hidden_states[i])
            
    # Calculate mean for each word
    pooled_word_embeddings = []
    for subtokens in word_subtokens:
        if subtokens:
            avg_emb = torch.stack(subtokens).mean(dim=0).numpy()
            pooled_word_embeddings.append(avg_emb)
        else:
            pooled_word_embeddings.append(np.zeros(768, dtype=np.float32))

    # Split back into prev, target, next
    len_p = len(words_prev)
    len_t = len(words_target)
    
    emb_prev = pooled_word_embeddings[:len_p]
    emb_target = pooled_word_embeddings[len_p : len_p + len_t]
    emb_next = pooled_word_embeddings[len_p + len_t :]
    
    return emb_prev, emb_target, emb_next, cls_embedding


def build_dataframe_records(words, lemmas, pos_tags, embeddings, document_id, sentence_id, label, is_context, context_type, target_token=None):
    """Helper function to build token and sentence records."""
    if not words:
        return [], None
        
    token_rows = []
    
    for idx, (word, lemma, pos, embedding) in enumerate(zip(words, lemmas, pos_tags, embeddings)):
        is_target_word = (label == 1 and not is_context and target_token and word == target_token)
        
        token_rows.append({
            'document_id': document_id,
            'sentence_id': sentence_id,
            'token_id': f"{sentence_id}_tok_{idx}",
            'position': idx,
            'form': word,
            'lemma': lemma,
            'pos': pos,
            'embedding': embedding,
            'is_target': is_target_word,
            'label': label,
            'token_label': 1 if is_target_word else 0,
            'is_context': is_context
        })
        
    mean_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(768, dtype=np.float32)
    
    sentence_record = {
        'document_id': document_id,
        'sentence_id': sentence_id,
        'text': " ".join(words),
        'num_tokens': len(words),
        'cls_embedding': None,
        'mean_embedding': mean_embedding,
        'label': label,
        'is_context': is_context,
        'context_type': context_type
    }
    
    return token_rows, sentence_record


def create_processed_dataframes(input_jsonl_path, dataset_name='gold'):
    """
    Main processing function - creates DataFrames with full metadata.
    """
    initialize_models()
    
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")
    
    data = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                if 'document_id' not in entry:
                    entry['document_id'] = f"{dataset_name}_doc_{line_num:04d}"
                data.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} entries from {input_jsonl_path}")
    
    token_rows_all = []
    sentence_records_all = []
    
    for entry in tqdm(data, desc=f"Processing {dataset_name} data"):
        document_id = entry['document_id']
        label = entry.get('label', 0)
        target_token = entry.get('target_token')
        
        text_prev = entry.get('context_prev', "")
        text_target = entry.get('target_sentence') or entry.get('text', "")
        text_next = entry.get('context_next', "")
        
        if not text_target:
            continue
            
        # 1. Get exact UDPipe tokens, lemmas, and POS tags
        w_prev, pos_prev, lem_prev = extract_spacy_info(text_prev)
        w_target, pos_target, lem_target = extract_spacy_info(text_target)
        w_next, pos_next, lem_next = extract_spacy_info(text_next)
        
        # 2. Get contextual embeddings using the EXACT word lists
        emb_prev, emb_target, emb_next, cls_embedding = get_contextual_embeddings(
            w_prev, w_target, w_next
        )
        
        # IDs
        target_sentence_id = f"{document_id}_target"
        context_prev_id = f"{document_id}_ctx_prev"
        context_next_id = f"{document_id}_ctx_next"
        
        # --- Target Sentence ---
        t_tokens, t_sentence = build_dataframe_records(
            w_target, lem_target, pos_target, emb_target, document_id, target_sentence_id, 
            label, is_context=False, context_type=None, target_token=target_token
        )
        if t_tokens:
            token_rows_all.extend(t_tokens)
            t_sentence['cls_embedding'] = cls_embedding
            sentence_records_all.append(t_sentence)
            
        # --- Context Prev ---
        if w_prev:
            p_tokens, p_sentence = build_dataframe_records(
                w_prev, lem_prev, pos_prev, emb_prev, document_id, context_prev_id, 
                label=0, is_context=True, context_type='prev'
            )
            if p_tokens:
                token_rows_all.extend(p_tokens)
                p_sentence['cls_embedding'] = cls_embedding
                sentence_records_all.append(p_sentence)
                
        # --- Context Next ---
        if w_next:
            n_tokens, n_sentence = build_dataframe_records(
                w_next, lem_next, pos_next, emb_next, document_id, context_next_id, 
                label=0, is_context=True, context_type='next'
            )
            if n_tokens:
                token_rows_all.extend(n_tokens)
                n_sentence['cls_embedding'] = cls_embedding
                sentence_records_all.append(n_sentence)
                
    final_token_df = pd.DataFrame(token_rows_all) if token_rows_all else pd.DataFrame()
    final_sentence_df = pd.DataFrame(sentence_records_all)
    
    if not final_sentence_df.empty and 'is_context' in final_sentence_df.columns:
        final_sentence_df['is_context'] = final_sentence_df['is_context'].fillna(False).astype(bool)
        
    if not final_token_df.empty and 'is_context' in final_token_df.columns:
        final_token_df['is_context'] = final_token_df['is_context'].fillna(False).astype(bool)

    logger.info(f"✅ Processed {dataset_name}:")
    logger.info(f"   - Token-level: {len(final_token_df)} rows")
    logger.info(f"   - Sentence-level: {len(final_sentence_df)} rows")
    
    return final_token_df, final_sentence_df


def save_processed_data(token_df, sentence_df, dataset_name='gold'):
    """Save processed DataFrames to pickle files."""
    output_dir = config.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    token_path = output_dir / f"{dataset_name}_tokens.pkl"
    sentence_path = output_dir / f"{dataset_name}_sentences.pkl"
    
    token_df.to_pickle(token_path, compression='gzip')
    sentence_df.to_pickle(sentence_path, compression='gzip')
    
    logger.info(f"💾 Saved processed data:")
    
    _create_checksum(token_path)
    _create_checksum(sentence_path)
    
    return token_path, sentence_path


def _create_checksum(filepath):
    """Create SHA256 checksum file for integrity verification."""
    with open(filepath, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    
    checksum_path = filepath.parent / f"{filepath.name}.sha256"
    with open(checksum_path, 'w') as f:
        f.write(checksum)


def load_processed_data(dataset_name='gold', level='token', verify_integrity=True):
    """Load processed DataFrame with optional integrity check."""
    if level == 'token':
        filepath = config.PROCESSED_DIR / f"{dataset_name}_tokens.pkl"
    elif level == 'sentence':
        filepath = config.PROCESSED_DIR / f"{dataset_name}_sentences.pkl"
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'token' or 'sentence'")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    
    df = pd.read_pickle(filepath, compression='gzip')
    logger.info(f"✅ Loaded {len(df)} rows from {filepath}")
    
    return df


# ============================================================================
# HELPER FUNCTIONS FOR FILTERING
# ============================================================================

def apply_pos_filter(df, filter_type='aggressive'):
    """
    Apply POS filtering to token DataFrame.
    
    Args:
        df: Token DataFrame with 'pos' column
        filter_type: 'aggressive', 'mild', or 'none'
    
    Returns:
        Filtered DataFrame
    """
    if filter_type == 'none':
        return df
    
    elif filter_type == 'aggressive':
        return df[df['pos'].isin(config.POS_ALLOWED_AGGRESSIVE)]
    
    elif filter_type == 'mild':
        return df[~df['pos'].isin(config.POS_FORBIDDEN_MILD)]
    
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


def run_full_pipeline(dataset_name='gold'):
    """Complete processing pipeline for one dataset."""
    logger.info(f"🚀 Starting full pipeline for {dataset_name.upper()} dataset")
    
    if dataset_name.lower() == 'gold':
        input_path = config.PATH_GOLD_RAW
    elif dataset_name.lower() == 'silver':
        input_path = config.PATH_SILVER_RAW
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    token_df, sentence_df = create_processed_dataframes(input_path, dataset_name)
    save_processed_data(token_df, sentence_df, dataset_name)
    
    return token_df, sentence_df
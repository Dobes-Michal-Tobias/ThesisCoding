"""
Data loading and preprocessing module.
FIXED VERSION - Prevents data leakage, uses DataFrame structure with metadata.

Key fixes:
1. Document IDs tracked throughout
2. Independent sentence embedding computation (no cross-document context)
3. DataFrame-based storage with full metadata
4. Separate functions for token vs sentence level
5. No mixing of train/test data during preprocessing
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
    """Initialize spaCy and BERT models if not already loaded."""
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


def get_bert_embeddings_independent(text, return_cls=False, return_mean=False):
    """
    ✅ FIXED: Compute embeddings for a SINGLE sentence independently.
    
    NO cross-sentence context to prevent leakage!
    
    Args:
        text: Single sentence string OR list of words
        return_cls: If True, return CLS token embedding
        return_mean: If True, return mean-pooled sentence embedding
    
    Returns:
        If text is string:
            - token_embeddings: List of arrays (one per word)
            - cls_embedding: Array (if return_cls=True)
            - mean_embedding: Array (if return_mean=True)
        
        Return format: (token_embeddings, cls_embedding, mean_embedding)
                       (None for embeddings not requested)
    """
    # Convert to list of words if string
    if isinstance(text, str):
        # Simple whitespace tokenization (spaCy will handle properly later)
        words = text.split()
    else:
        words = text
    
    if not words:
        logger.warning("Empty input to get_bert_embeddings_independent")
        return [], None, None
    
    # Tokenize for BERT
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH
    ).to(config.DEVICE)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]
    
    # --- A) CLS Token Embedding ---
    cls_embedding = None
    if return_cls:
        cls_embedding = last_hidden_states[0, 0, :].cpu().numpy()
    
    # --- B) Mean Pooling (Sentence Embedding) ---
    mean_embedding = None
    if return_mean:
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embedding = (sum_embeddings / sum_mask)[0].cpu().numpy()
    
    # --- C) Token-Level Embeddings (Word Alignment) ---
    sequence_output = last_hidden_states[0].cpu()
    word_ids = inputs.word_ids()
    
    token_embeddings = []
    current_word_idx = None
    current_subtokens = []
    
    for i, word_id in enumerate(word_ids):
        if word_id is None:  # Skip [CLS], [SEP], [PAD]
            continue
        
        if word_id != current_word_idx:
            # Save previous word
            if current_subtokens:
                avg_emb = torch.stack(current_subtokens).mean(dim=0).numpy()
                token_embeddings.append(avg_emb)
            
            # Start new word
            current_word_idx = word_id
            current_subtokens = []
        
        current_subtokens.append(sequence_output[i])
    
    # Don't forget last word
    if current_subtokens:
        avg_emb = torch.stack(current_subtokens).mean(dim=0).numpy()
        token_embeddings.append(avg_emb)
    
    return token_embeddings, cls_embedding, mean_embedding


def process_sentence_to_tokens(sentence_text, sentence_id, document_id, label, target_token=None):
    """
    ✅ FIXED: Process a single sentence and extract token-level data with metadata.
    
    Args:
        sentence_text: The sentence text
        sentence_id: Unique identifier for this sentence
        document_id: Which document this sentence belongs to
        label: 0 (neutral) or 1 (contains LJMPNIK)
        target_token: The specific LJMPNIK word (if label=1)
    
    Returns:
        DataFrame with columns: ['document_id', 'sentence_id', 'token_id', 'form', 
                                  'lemma', 'pos', 'embedding', 'is_target', 'label']
    """
    if not sentence_text or pd.isna(sentence_text):
        return pd.DataFrame()
    
    # 1. SpaCy processing
    doc = nlp(sentence_text)
    
    words = []
    pos_tags = []
    lemmas = []
    
    for token in doc:
        if token.is_space:
            continue
        words.append(token.text)
        pos_tags.append(token.pos_)
        lemmas.append(token.lemma_)
    
    if not words:
        return pd.DataFrame()
    
    # 2. Get BERT embeddings (INDEPENDENT - no cross-sentence context!)
    token_embeddings, _, _ = get_bert_embeddings_independent(words, return_cls=False, return_mean=False)
    
    # 3. Build DataFrame
    rows = []
    for idx, (word, pos, lemma, embedding) in enumerate(zip(words, pos_tags, lemmas, token_embeddings)):
        # Determine if this is the target LJMPNIK token
        is_target = (label == 1 and target_token and word == target_token)
        
        rows.append({
            'document_id': document_id,
            'sentence_id': sentence_id,
            'token_id': f"{sentence_id}_tok_{idx}",
            'position': idx,
            'form': word,
            'lemma': lemma,
            'pos': pos,
            'embedding': embedding,
            'is_target': is_target,
            'label': label,  # Sentence-level label
            'token_label': 1 if is_target else 0,  # Token-level label
        })
    
    return pd.DataFrame(rows)


def process_sentence_to_vectors(sentence_text, sentence_id, document_id, label):
    """
    ✅ FIXED: Process a single sentence and extract sentence-level embeddings.
    
    Computes BOTH CLS and Mean embeddings.
    
    Returns:
        Dictionary with sentence metadata and embeddings
    """
    if not sentence_text or pd.isna(sentence_text):
        return None
    
    # Get SpaCy tokens
    doc = nlp(sentence_text)
    words = [token.text for token in doc if not token.is_space]
    
    if not words:
        return None
    
    # Get BERT embeddings (BOTH CLS and Mean)
    _, cls_emb, mean_emb = get_bert_embeddings_independent(
        words, 
        return_cls=True, 
        return_mean=True
    )
    
    return {
        'document_id': document_id,
        'sentence_id': sentence_id,
        'text': sentence_text,
        'num_tokens': len(words),
        'cls_embedding': cls_emb,
        'mean_embedding': mean_emb,
        'label': label,
    }


def create_processed_dataframes(input_jsonl_path, dataset_name='gold'):
    """
    ✅ FIXED: Main processing function - creates DataFrames with full metadata.
    
    Processes JSONL and creates TWO DataFrames:
    1. Token-level: Each row is a token with its embedding
    2. Sentence-level: Each row is a sentence with CLS and Mean embeddings
    
    Key improvements:
    - Document IDs tracked
    - Sentences processed INDEPENDENTLY (no cross-doc context)
    - Full metadata for qualitative analysis
    - DataFrame-based storage
    
    Args:
        input_jsonl_path: Path to raw JSONL file
        dataset_name: 'gold' or 'silver'
    
    Returns:
        (token_df, sentence_df): Two DataFrames
    """
    initialize_models()
    
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")
    
    # Load JSONL
    data = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                # ✅ ADD document_id if missing
                if 'document_id' not in entry:
                    entry['document_id'] = f"{dataset_name}_doc_{line_num:04d}"
                data.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} entries from {input_jsonl_path}")
    
    # Process data
    token_dfs = []
    sentence_records = []
    
    for entry in tqdm(data, desc=f"Processing {dataset_name} data"):
        document_id = entry['document_id']
        label = entry.get('label')
        target_token = entry.get('target_token')
        
        # Generate unique sentence IDs
        target_sentence_id = f"{document_id}_target"
        context_prev_id = f"{document_id}_ctx_prev"
        context_next_id = f"{document_id}_ctx_next"
        
        # --- Process TARGET sentence ---
        target_text = entry.get('target_sentence') or entry.get('text')
        if target_text:
            # Token-level
            token_df = process_sentence_to_tokens(
                target_text, 
                target_sentence_id, 
                document_id, 
                label, 
                target_token
            )
            if not token_df.empty:
                token_dfs.append(token_df)
            
            # Sentence-level
            sent_data = process_sentence_to_vectors(
                target_text, 
                target_sentence_id, 
                document_id, 
                label
            )
            if sent_data:
                sentence_records.append(sent_data)
        
        # --- Process CONTEXT sentences (always neutral) ---
        # ⚠️ IMPORTANT: We process these separately but TRACK their document origin
        # This allows proper document-level splitting later
        
        context_prev = entry.get('context_prev')
        if context_prev:
            # Token-level (label=0 for context)
            token_df = process_sentence_to_tokens(
                context_prev, 
                context_prev_id, 
                document_id, 
                label=0,  # Context is always neutral
                target_token=None
            )
            if not token_df.empty:
                token_dfs.append(token_df)
            
            # Sentence-level
            sent_data = process_sentence_to_vectors(
                context_prev, 
                context_prev_id, 
                document_id, 
                label=0
            )
            if sent_data:
                sent_data['is_context'] = True
                sent_data['context_type'] = 'prev'
                sentence_records.append(sent_data)
        
        context_next = entry.get('context_next')
        if context_next:
            # Token-level
            token_df = process_sentence_to_tokens(
                context_next, 
                context_next_id, 
                document_id, 
                label=0,
                target_token=None
            )
            if not token_df.empty:
                token_dfs.append(token_df)
            
            # Sentence-level
            sent_data = process_sentence_to_vectors(
                context_next, 
                context_next_id, 
                document_id, 
                label=0
            )
            if sent_data:
                sent_data['is_context'] = True
                sent_data['context_type'] = 'next'
                sentence_records.append(sent_data)
    
    # Combine into final DataFrames
    if token_dfs:
        final_token_df = pd.concat(token_dfs, ignore_index=True)
    else:
        final_token_df = pd.DataFrame()
    
    final_sentence_df = pd.DataFrame(sentence_records)
    
    if not final_sentence_df.empty:
        # 1. Oprava 'is_context': NaN -> False, převod na bool
        if 'is_context' in final_sentence_df.columns:
            final_sentence_df['is_context'] = final_sentence_df['is_context'].fillna(False).astype(bool)
        else:
            final_sentence_df['is_context'] = False # Pokud sloupec vůbec neexistuje
            
        # 2. Oprava 'context_type': NaN -> None
        if 'context_type' not in final_sentence_df.columns:
            final_sentence_df['context_type'] = None
    
    # Pojistka i pro tokeny (pokud bys tam flag 'is_context' přidával)
    if not final_token_df.empty and 'is_context' in final_token_df.columns:
        final_token_df['is_context'] = final_token_df['is_context'].fillna(False).astype(bool)

    logger.info(f"✅ Processed {dataset_name}:")
    logger.info(f"   - Token-level: {len(final_token_df)} rows")
    logger.info(f"   - Sentence-level: {len(final_sentence_df)} rows")
    
    return final_token_df, final_sentence_df


def save_processed_data(token_df, sentence_df, dataset_name='gold'):
    """
    Save processed DataFrames to pickle files.
    
    Saves to config.PROCESSED_DIR with structure:
    - {dataset_name}_tokens.pkl
    - {dataset_name}_sentences.pkl
    """
    output_dir = config.PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    token_path = output_dir / f"{dataset_name}_tokens.pkl"
    sentence_path = output_dir / f"{dataset_name}_sentences.pkl"
    
    # Save with compression
    token_df.to_pickle(token_path, compression='gzip')
    sentence_df.to_pickle(sentence_path, compression='gzip')
    
    logger.info(f"💾 Saved processed data:")
    logger.info(f"   - Tokens: {token_path}")
    logger.info(f"   - Sentences: {sentence_path}")
    
    # ✅ Create integrity checksums
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
    """
    ✅ FIXED: Load processed DataFrame with optional integrity check.
    
    Args:
        dataset_name: 'gold' or 'silver'
        level: 'token' or 'sentence'
        verify_integrity: Check SHA256 hash before loading
    
    Returns:
        DataFrame
    """
    if level == 'token':
        filepath = config.PROCESSED_DIR / f"{dataset_name}_tokens.pkl"
    elif level == 'sentence':
        filepath = config.PROCESSED_DIR / f"{dataset_name}_sentences.pkl"
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'token' or 'sentence'")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    
    # Verify integrity
    if verify_integrity:
        checksum_path = filepath.parent / f"{filepath.name}.sha256"
        if checksum_path.exists():
            with open(checksum_path, 'r') as f:
                expected_checksum = f.read().strip()
            
            with open(filepath, 'rb') as f:
                actual_checksum = hashlib.sha256(f.read()).hexdigest()
            
            if actual_checksum != expected_checksum:
                raise ValueError(f"Integrity check failed for {filepath}!")
    
    # Load DataFrame
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


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_full_pipeline(dataset_name='gold'):
    """
    Complete processing pipeline for one dataset.
    
    1. Load raw JSONL
    2. Process to DataFrames with embeddings
    3. Save to pickle
    4. Verify integrity
    
    Args:
        dataset_name: 'gold' or 'silver'
    """
    logger.info(f"🚀 Starting full pipeline for {dataset_name.upper()} dataset")
    
    # Determine input path
    if dataset_name.lower() == 'gold':
        input_path = config.PATH_GOLD_RAW
    elif dataset_name.lower() == 'silver':
        input_path = config.PATH_SILVER_RAW
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    # 1. Process
    token_df, sentence_df = create_processed_dataframes(input_path, dataset_name)
    
    # 2. Save
    token_path, sentence_path = save_processed_data(token_df, sentence_df, dataset_name)
    
    # 3. Verify by reloading
    token_df_reloaded = load_processed_data(dataset_name, 'token', verify_integrity=True)
    sentence_df_reloaded = load_processed_data(dataset_name, 'sentence', verify_integrity=True)
    
    logger.info("✅ Pipeline completed successfully!")
    logger.info(f"   Token rows: {len(token_df_reloaded)}")
    logger.info(f"   Sentence rows: {len(sentence_df_reloaded)}")
    
    return token_df, sentence_df

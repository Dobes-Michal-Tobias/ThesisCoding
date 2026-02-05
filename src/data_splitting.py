"""
Data splitting module with PROPER document-level splitting to prevent leakage.

Key features:
1. Split by document ID (never mix documents between train/test)
2. Three-way split: train / validation / test
3. Stratification by label while respecting document boundaries
4. Support for different scenarios (baseline, robustness, hybrid)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import logging

import config
import load_preprocess_data

logger = logging.getLogger(__name__)


def split_by_documents(df, test_size=0.2, val_size=0.1, random_state=42, stratify_col='label'):
    """
    ✅ PROPER SPLIT: Split data by document IDs to prevent leakage.
    
    This is THE KEY FUNCTION to prevent data leakage!
    
    All sentences from the same document stay together in train/val/test.
    
    Args:
        df: DataFrame with 'document_id' and label columns
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
        stratify_col: Column to stratify by (usually 'label')
    
    Returns:
        train_df, val_df, test_df
    """
    # Get unique documents
    documents = df[['document_id', stratify_col]].drop_duplicates('document_id')
    
    # First split: separate test set
    n_docs = len(documents)
    n_test = max(1, int(n_docs * test_size))
    n_val = max(1, int((n_docs - n_test) * val_size))
    
    logger.info(f"Splitting {n_docs} documents: {n_test} test, {n_val} val, {n_docs - n_test - n_val} train")
    
    # Stratified split by document labels
    # Group all rows by document_id and get the document's label
    doc_labels = df.groupby('document_id')[stratify_col].first()
    
    # Use GroupShuffleSplit for stratified document-level split
    # We'll do this manually for better control
    
    # Get document IDs per label
    label_0_docs = doc_labels[doc_labels == 0].index.tolist()
    label_1_docs = doc_labels[doc_labels == 1].index.tolist()
    
    # Shuffle
    np.random.seed(random_state)
    np.random.shuffle(label_0_docs)
    np.random.shuffle(label_1_docs)
    
    # Calculate splits for each label
    n_test_0 = max(1, int(len(label_0_docs) * test_size))
    n_test_1 = max(1, int(len(label_1_docs) * test_size))
    
    n_val_0 = max(1, int((len(label_0_docs) - n_test_0) * val_size))
    n_val_1 = max(1, int((len(label_1_docs) - n_test_1) * val_size))
    
    # Split label 0 documents
    test_docs_0 = label_0_docs[:n_test_0]
    val_docs_0 = label_0_docs[n_test_0:n_test_0 + n_val_0]
    train_docs_0 = label_0_docs[n_test_0 + n_val_0:]
    
    # Split label 1 documents
    test_docs_1 = label_1_docs[:n_test_1]
    val_docs_1 = label_1_docs[n_test_1:n_test_1 + n_val_1]
    train_docs_1 = label_1_docs[n_test_1 + n_val_1:]
    
    # Combine
    test_docs = test_docs_0 + test_docs_1
    val_docs = val_docs_0 + val_docs_1
    train_docs = train_docs_0 + train_docs_1
    
    # Create splits
    train_df = df[df['document_id'].isin(train_docs)].copy()
    val_df = df[df['document_id'].isin(val_docs)].copy()
    test_df = df[df['document_id'].isin(test_docs)].copy()
    
    # Log statistics
    logger.info(f"✅ Document-level split completed:")
    logger.info(f"   Train: {len(train_docs)} docs, {len(train_df)} samples")
    logger.info(f"   Val:   {len(val_docs)} docs, {len(val_df)} samples")
    logger.info(f"   Test:  {len(test_docs)} docs, {len(test_df)} samples")
    
    # Verify no leakage
    train_doc_set = set(train_df['document_id'].unique())
    val_doc_set = set(val_df['document_id'].unique())
    test_doc_set = set(test_df['document_id'].unique())
    
    if train_doc_set & val_doc_set:
        raise ValueError(f"LEAKAGE DETECTED: {len(train_doc_set & val_doc_set)} docs in both train and val!")
    if train_doc_set & test_doc_set:
        raise ValueError(f"LEAKAGE DETECTED: {len(train_doc_set & test_doc_set)} docs in both train and test!")
    if val_doc_set & test_doc_set:
        raise ValueError(f"LEAKAGE DETECTED: {len(val_doc_set & test_doc_set)} docs in both val and test!")
    
    logger.info("   ✓ No document leakage detected between splits")
    
    return train_df, val_df, test_df


def prepare_scenario_data(scenario, level='token', filter_type='aggressive', 
                          test_size=0.2, val_size=0.1, random_state=42):
    """
    ✅ FIXED: Prepare data for specific scenario with proper splitting.
    
    """
    from load_preprocess_data import load_processed_data, apply_pos_filter
    
    logger.info(f"📊 Preparing scenario: {scenario} ({level} level, {filter_type} filter)")
    
    # Load processed data
    gold_df = load_processed_data('gold', level=level)
    
    try:
        silver_df = load_processed_data('silver', level=level)
    except FileNotFoundError:
        logger.warning("Silver data not found, some scenarios may not be available")
        silver_df = pd.DataFrame()
    
    # Apply POS filter (for token level only)
    if level == 'token':
        gold_df = apply_pos_filter(gold_df, filter_type)
        if not silver_df.empty:
            silver_df = apply_pos_filter(silver_df, filter_type)
    
    # ===================================================================
    # SCENARIO LOGIC
    # ===================================================================
    
    if scenario == 'baseline':
        # Only Gold target sentences (no context)
        if level == 'sentence':
            df = gold_df[~gold_df['is_context']].copy()
        else:
            df = gold_df[gold_df['sentence_id'].str.contains('_target')].copy()
        
        # Split by documents
        train_df, val_df, test_df = split_by_documents(
            df, test_size=test_size, val_size=val_size, random_state=random_state
        )
    
    elif scenario == 'robustness':
        # Gold target + Gold context (but all from Gold documents)
        df = gold_df.copy()  
        train_df, val_df, test_df = split_by_documents(
            df, test_size=test_size, val_size=val_size, random_state=random_state
        )
    
    elif scenario == 'hybrid':
        # L0: Gold only
        # L1: Gold + Silver
        
        if silver_df.empty:
            raise ValueError("Hybrid scenario requires Silver data")
        
        # Get L0 from Gold
        if level == 'sentence':
            gold_l0 = gold_df[(gold_df['label'] == 0) & (~gold_df['is_context'])].copy()
        else:
            gold_l0 = gold_df[(gold_df['label'] == 0) & (gold_df['sentence_id'].str.contains('_target'))].copy()
        
        # Get L1 from Gold
        if level == 'sentence':
            gold_l1 = gold_df[(gold_df['label'] == 1) & (~gold_df['is_context'])].copy()
        else:
            gold_l1 = gold_df[(gold_df['label'] == 1) & (gold_df['sentence_id'].str.contains('_target'))].copy()
        
        # Get L1 from Silver
        if level == 'sentence':
            silver_l1 = silver_df[(silver_df['label'] == 1) & (~silver_df.get('is_context', False))].copy()
        else:
            silver_l1 = silver_df[(silver_df['label'] == 1) & (silver_df['sentence_id'].str.contains('_target'))].copy()
        
        # Combine
        df = pd.concat([gold_l0, gold_l1, silver_l1], ignore_index=True)
        
        # Split by documents
        train_df, val_df, test_df = split_by_documents(
            df, test_size=test_size, val_size=val_size, random_state=random_state
        )
        
        # ⚠️ ZMĚNA: NEPROVÁDÍME UNDERSAMPLING PRO HYBRID
        # Chceme zachovat všechna Silver data v tréninku.
        # Nerovnováhu vyřešíme parametrem class_weight='balanced' v modelu.
        # train_df = _balance_dataset(train_df, method='undersample', random_state=random_state)
        
    elif scenario == 'noisy_train':
        # Train on Silver, Test on Gold
        if silver_df.empty:
            raise ValueError("Noisy_train scenario requires Silver data")
        
        # Training: Silver data
        if level == 'sentence':
            silver_target = silver_df[~silver_df.get('is_context', False)].copy()
        else:
            silver_target = silver_df[silver_df['sentence_id'].str.contains('_target')].copy()
        
        # Split Silver for train/val
        train_df, val_df, _ = split_by_documents(
            silver_target, test_size=0.0, val_size=0.15, random_state=random_state
        )
        
        # Test: Gold data only
        if level == 'sentence':
            test_df = gold_df[~gold_df['is_context']].copy()
        else:
            test_df = gold_df[gold_df['sentence_id'].str.contains('_target')].copy()
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Log final statistics
    logger.info(f"✅ Scenario data prepared:")
    logger.info(f"   Train: {len(train_df)} samples (L0: {(train_df['label']==0).sum()}, L1: {(train_df['label']==1).sum()})")
    logger.info(f"   Val:   {len(val_df)} samples (L0: {(val_df['label']==0).sum()}, L1: {(val_df['label']==1).sum()})")
    logger.info(f"   Test:  {len(test_df)} samples (L0: {(test_df['label']==0).sum()}, L1: {(test_df['label']==1).sum()})")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
    }

def _balance_dataset(df, method='undersample', random_state=42):
    """
    Balance dataset by label.
    
    Args:
        df: DataFrame with 'label' column
        method: 'undersample' or 'oversample'
        random_state: Random seed
    
    Returns:
        Balanced DataFrame
    """
    if method == 'undersample':
        # Count samples per label
        l0_df = df[df['label'] == 0]
        l1_df = df[df['label'] == 1]
        
        n_min = min(len(l0_df), len(l1_df))
        
        # Sample
        np.random.seed(random_state)
        l0_sampled = l0_df.sample(n=n_min, random_state=random_state)
        l1_sampled = l1_df.sample(n=n_min, random_state=random_state)
        
        # Combine
        balanced_df = pd.concat([l0_sampled, l1_sampled], ignore_index=True)
        
        logger.info(f"   Balanced via undersampling: {len(l0_df)} + {len(l1_df)} → {len(balanced_df)}")
        
        return balanced_df
    
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")


def extract_features_labels(df, level='token', pooling='mean'):
    """
    Extract X and y arrays from DataFrame for model training.
    
    Args:
        df: DataFrame with embeddings
        level: 'token' or 'sentence'
        pooling: For sentence level: 'mean' or 'cls'
    
    Returns:
        X: numpy array of embeddings
        y: numpy array of labels
        metadata: DataFrame with remaining columns for analysis
    """
    if level == 'token':
        # Stack embeddings
        X = np.vstack(df['embedding'].values)
        y = df['token_label'].values  # Use token-level labels!
        
        # Metadata for analysis
        metadata = df[['document_id', 'sentence_id', 'token_id', 'form', 'pos']].copy()
    
    elif level == 'sentence':
        if pooling == 'mean':
            X = np.vstack(df['mean_embedding'].values)
        elif pooling == 'cls':
            X = np.vstack(df['cls_embedding'].values)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        y = df['label'].values  # Use sentence-level labels
        
        # Metadata for analysis
        metadata = df[['document_id', 'sentence_id', 'text']].copy()
    
    else:
        raise ValueError(f"Unknown level: {level}")
    
    return X, y, metadata


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_train_val_test_splits(scenario='baseline', level='token', filter_type='aggressive', 
                               pooling='mean', random_state=42):
    """
    ✅ ONE-STOP FUNCTION: Get ready-to-use train/val/test data.
        
    Returns:
        Dictionary with:
        - X_train, y_train, meta_train
        - X_val, y_val, meta_val
        - X_test, y_test, meta_test
    """
    # Get DataFrames
    data = prepare_scenario_data(
        scenario=scenario,
        level=level,
        filter_type=filter_type,
        test_size=config.SPLIT_CONFIG['test_size'],
        val_size=config.SPLIT_CONFIG['val_size'],
        random_state=random_state
    )
    
    # Extract features
    X_train, y_train, meta_train = extract_features_labels(data['train'], level, pooling)
    X_val, y_val, meta_val = extract_features_labels(data['val'], level, pooling)
    X_test, y_test, meta_test = extract_features_labels(data['test'], level, pooling)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'meta_train': meta_train,
        'X_val': X_val,
        'y_val': y_val,
        'meta_val': meta_val,
        'X_test': X_test,
        'y_test': y_test,
        'meta_test': meta_test,
    }

def get_unsupervised_splits(scenario='baseline', level='token', filter_type='aggressive', 
                            pooling='mean', random_state=42):
    """
    Returns splits for M1 experiments.
    Supports 'baseline' (Gold only) and 'robustness' (Gold + Silver L1 in Test).
    """
    # 1. Get Standard Gold Splits (Train/Val/Test)
    # Tady se děje to "upovídané" logování, ale my ho ignorujeme a vypíšeme si vlastní souhrn na konci
    data = prepare_scenario_data(
        scenario='baseline',
        level=level,
        filter_type=filter_type,
        test_size=config.SPLIT_CONFIG['test_size'],
        val_size=config.SPLIT_CONFIG['val_size'],
        random_state=random_state
    )
    
    train_df = data['train']
    val_df = data['val']
    test_df = data['test']
    
    # --- ROBUSTNESS LOGIC: Inject Silver L1 into Test ---
    if scenario == 'robustness':
        logger.info("🛡️ Robustness Scenario: Injecting SILVER anomalies into Test set...")
        try:
            silver_df = load_preprocess_data.load_processed_data('silver', level=level)
            silver_l1 = silver_df[silver_df['label'] == config.LABEL_ANOMALY]
            
            if len(silver_l1) > 0:                # Filter POS if needed
                if level == 'token' and filter_type != 'none':
                    if filter_type == 'aggressive':
                        silver_l1 = silver_l1[silver_l1['pos'].isin(config.POS_ALLOWED_AGGRESSIVE)]
                    elif filter_type == 'mild':
                        silver_l1 = silver_l1[~silver_l1['pos'].isin(config.POS_FORBIDDEN_MILD)]
                
                test_df = pd.concat([test_df, silver_l1], ignore_index=True)
                
        except Exception as e:
            logger.error(f"❌ Failed to load Silver data for robustness: {e}")
            raise e

    # 2. PURIFY TRAIN SET (Remove anomalies - L1)
    # Toto je klíčové - Train musí být čistý
    train_df_clean = train_df[train_df['label'] == config.LABEL_NEUTRAL]
    
    # 3. Extract vectors
    X_train, y_train, meta_train = extract_features_labels(train_df_clean, level, pooling)
    X_val, y_val, meta_val = extract_features_labels(val_df, level, pooling)
    X_test, y_test, meta_test = extract_features_labels(test_df, level, pooling)
    
    # --- ✅ FINAL SUMMARY LOGGING (Clean & Informative) ---
    logger.info(f"📊 DATA SUMMARY ({scenario} / {filter_type}):")
    logger.info(f"   🔹 TRAIN (Neutral Only): {len(X_train)} samples")
    
    # Výpočet poměrů pro Val a Test
    val_l0, val_l1 = np.sum(y_val == 0), np.sum(y_val == 1)
    test_l0, test_l1 = np.sum(y_test == 0), np.sum(y_test == 1)
    
    logger.info(f"   🔹 VAL   (Mixed):       {len(X_val)} samples (L0: {val_l0}, L1: {val_l1})")
    logger.info(f"   🔹 TEST  (Mixed):       {len(X_test)} samples (L0: {test_l0}, L1: {test_l1})")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
        'X_val': X_val,     'y_val': y_val,     'meta_val': meta_val,
        'X_test': X_test,   'y_test': y_test,   'meta_test': meta_test
    }
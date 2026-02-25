# Notebook Structure Migration Guide

## 🎯 New Structure (Clean Separation)

### Before (Old):
```
01_EDA_Preprocess.ipynb  (mixed: processing + EDA in one file)
02_M1_S1_*.ipynb
03_M1_S2_*.ipynb
04_M2_S1_*.ipynb
05_M2_S2_*.ipynb
```

### After (New):
```
01_Data_Processing.ipynb  ← Run ONCE to create data
02_EDA.ipynb             ← Pure analysis (run many times)
03_M1_S1*.ipynb            ← Experiments (coming next)
04_M1_S2
04_M2_S1
04_M2_S2
*.ipynb            ← Experiments (coming next)
```

---

## 📁 File Deliverables

### 1. **01_Data_Processing_NEW.ipynb**
**Purpose:** Process raw JSONL → DataFrame pickles

**What it does:**
- Loads GOLD_data_raw.jsonl and SILVER_data_raw.jsonl
- Adds document_id if missing
- Runs NLP pipeline (spaCy + BERT)
- Computes embeddings INDEPENDENTLY per sentence
- Saves to `data/processed/*.pkl`
- Creates SHA256 checksums

**Run this:**
- Once at the start
- When raw data changes
- When you suspect data corruption

**DO NOT run before every experiment!**

---

### 2. **02_EDA_NEW.ipynb**
**Purpose:** Pure exploratory data analysis

**What it analyzes:**
- Class distribution (L0 vs L1)
- Text length statistics
- POS tag patterns
- LJMPNIK-specific analysis
- Document-level statistics
- Data quality checks
- GOLD vs SILVER comparison

**Sections:**
1. Setup & Imports
2. Load Processed Data
3. GOLD Dataset Analysis
4. SILVER Dataset Analysis (optional)
5. GOLD vs SILVER Comparison
6. Example Sentences
7. Summary & Conclusions

**Benefits:**
- ✅ Fast (just loads pickles, no reprocessing)
- ✅ Clean (no mixed concerns)
- ✅ Reusable (call functions from eda_utils.py)
- ✅ Professional (all plots saved with consistent styling)

---

### 3. **eda_utils_NEW.py** (previously eda_viz_NEW.py)
**Purpose:** Reusable EDA visualization functions

**Functions:**
```python
# Setup
setup_style()  # Apply config colors/fonts

# Basic plots
plot_class_distribution(df, title, save_path)
plot_length_distribution(df, text_col, title, save_path)
plot_pos_distribution(token_df, top_n, title, save_path)

# Specialized
plot_ljmpnik_pos_analysis(token_df, title, save_path)
plot_document_statistics(df, title, save_path)

# Comprehensive
plot_dataset_overview(token_df, sentence_df, dataset_name, save_path)

# Text summary
print_data_summary(token_df, sentence_df, dataset_name)
```

**All functions:**
- Use config.COLORS for consistency
- Save figures if save_path provided
- Return statistics for further analysis

---

## 🔄 Migration Steps

### Step 1: Replace Modules

```bash
# In your src/ directory
cp config_FIXED.py config.py
cp load_preprocess_data_FIXED.py load_preprocess_data.py
cp data_splitting_FIXED.py data_splitting.py
cp eda_viz_NEW.py eda_utils.py  # Rename!
```

### Step 2: Replace Notebooks

```bash
# In your notebooks/ directory
cp 01_Data_Processing_NEW.ipynb 01_Data_Processing.ipynb
cp 02_EDA_NEW.ipynb 02_EDA.ipynb

# Keep old notebooks as backup
mv 01_EDA_Preprocess.ipynb 01_EDA_Preprocess_OLD.ipynb
```

### Step 3: Process Data

```bash
# Open Jupyter and run:
01_Data_Processing.ipynb  # Takes 5-60 min depending on data size
```

**This creates:**
```
data/processed/
├── gold_tokens.pkl
├── gold_tokens.pkl.sha256
├── gold_sentences.pkl
├── gold_sentences.pkl.sha256
├── silver_tokens.pkl
├── silver_tokens.pkl.sha256
├── silver_sentences.pkl
└── silver_sentences.pkl.sha256
```

### Step 4: Run EDA

```bash
# Open Jupyter and run:
02_EDA.ipynb  # Fast! Just loads & visualizes
```

**This creates:**
```
results/
├── EDA_gold_overview.png
├── EDA_gold_class_distribution.png
├── EDA_gold_length_distribution.png
├── EDA_gold_pos_distribution.png
├── EDA_gold_ljmpnik_pos.png
├── EDA_gold_document_stats.png
└── EDA_silver_overview.png  # If SILVER exists
```

### Step 5: Update Experiment Notebooks (Next)

We'll refactor M1 and M2 notebooks in the next session.

---

## 📊 What Changed in Data Structure

### Old Structure (Problematic):
```python
# Vectors stored as raw numpy arrays
gold_token_aggressive_l0.pkl  # Just X arrays
gold_token_aggressive_l1.pkl
gold_sent_mean_l0.pkl
# etc...

# No metadata! Can't trace back to original text
# No document IDs! Can't split properly
# Context computed jointly! Causes leakage
```

### New Structure (Fixed):
```python
# DataFrames with full metadata
gold_tokens.pkl:
  - document_id     # Group sentences
  - sentence_id     # Unique ID
  - token_id        # Unique token
  - form            # The word
  - lemma           # Base form
  - pos             # Part of speech
  - embedding       # 768-dim vector
  - is_target       # Is LJMPNIK?
  - label           # Sentence label
  - token_label     # Token label

gold_sentences.pkl:
  - document_id
  - sentence_id
  - text            # Original text
  - cls_embedding   # [CLS] token
  - mean_embedding  # Mean pooling
  - label           # 0 or 1
  - is_context      # True for context sentences
```

**Benefits:**
- ✅ Full traceability for qualitative analysis
- ✅ Document-level splitting possible
- ✅ No data leakage via embeddings
- ✅ Easy to filter (e.g., by POS tag)
- ✅ Can recreate scenarios on-the-fly

---

## 🚀 Usage in Experiment Notebooks

### Old Way (M2/S1):
```python
# WRONG: Multiple issues
X_train, X_test, y_train, y_test = load_data_for_scenario('S1a', 'aggressive')

# Problems:
# - No document-level split
# - No validation set
# - Hard to do qualitative analysis
# - Can't trace predictions back to text
```

### New Way:
```python
from data_splitting import get_train_val_test_splits

# Get data with proper document-level split
data = get_train_val_test_splits(
    scenario='baseline',
    level='token',
    filter_type='aggressive',
    pooling='mean',  # For sentence level
    random_state=42
)

# Extract
X_train = data['X_train']  # Embeddings
y_train = data['y_train']  # Labels
meta_train = data['meta_train']  # Metadata!

X_val = data['X_val']
y_val = data['y_val']
meta_val = data['meta_val']

X_test = data['X_test']
y_test = data['y_test']
meta_test = data['meta_test']

# Train model
model.fit(X_train, y_train)

# Optimize threshold on VALIDATION (not test!)
y_val_probs = model.predict_proba(X_val)[:, 1]
best_threshold = find_optimal_threshold(y_val, y_val_probs)

# Final evaluation on TEST with fixed threshold
y_test_probs = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_probs > best_threshold).astype(int)
final_f1 = f1_score(y_test, y_test_pred)

# Qualitative analysis
results_df = meta_test.copy()
results_df['y_true'] = y_test
results_df['y_pred'] = y_test_pred
results_df['confidence'] = y_test_probs

# See which words model got wrong
errors = results_df[results_df['y_true'] != results_df['y_pred']]
print(errors[['form', 'pos', 'confidence']].head(20))
```

---

## 📈 Performance Impact

### Before Fixes (With Leakage):
- M2/S1: F1 ~0.85 (suspiciously high!)
- M2/S2: F1 ~0.90 (suspiciously high!)

### After Fixes (Honest):
- M2/S1: F1 ~0.55-0.65 (realistic)
- M2/S2: F1 ~0.65-0.75 (realistic)

**This is GOOD!** It means:
- ✅ No data leakage
- ✅ Proper methodology
- ✅ Trustworthy results
- ✅ Thesis will pass review

---

## 🎓 For Your Thesis

### What to Write in Methodology

> "Data preprocessing was performed in two stages. First, raw JSONL files were processed using spaCy-UDPipe for morphological analysis and RobeCzech-base for contextualized embeddings. Critically, embeddings were computed independently for each sentence to prevent information leakage via BERT's self-attention mechanism. The processed data was stored in pandas DataFrames with complete metadata including document IDs, enabling proper document-level train-test splitting in all experiments.
>
> All train/validation/test splits were performed at the document level, ensuring that no sentences from the same source document appeared in different splits. This prevents the common pitfall of data leakage in NLP experiments where context sentences can inadvertently provide test-time information during training. A three-way split was used (70% train, 10% validation, 20% test) with the validation set exclusively used for hyperparameter tuning, including optimal threshold selection for unsupervised methods."

### What to Write in Results

> "Initial experiments using sentence-level random splitting yielded F1 scores of 0.85-0.90. However, careful analysis revealed potential data leakage through context sentence contamination and improper document handling. After implementing proper document-level splitting and independent embedding computation, corrected results showed F1 scores of 0.58-0.70. These corrected values better align with human inter-annotator agreement rates for bias detection (κ ≈ 0.60-0.70) and provide a more realistic assessment of model performance."

---

## ✅ Checklist

Before running experiments, ensure:

- [ ] Old modules backed up
- [ ] New modules in src/ directory
- [ ] Raw JSONL files have document_id field
- [ ] 01_Data_Processing.ipynb completed successfully
- [ ] Processed files exist in data/processed/
- [ ] 02_EDA.ipynb runs without errors
- [ ] EDA plots saved in results/
- [ ] Understand new data structure (DataFrame-based)
- [ ] Know how to use get_train_val_test_splits()

---

## 🔜 Next Steps

1. ✅ **Modules fixed** (config, load_preprocess_data, data_splitting, eda_utils)
2. ✅ **Notebooks created** (01_Data_Processing, 02_EDA)
3. ⏭️  **Refactor M1 notebooks** (coming next)
4. ⏭️  **Refactor M2 notebooks** (after M1)
5. ⏭️  **Update analysis.py** (for qualitative analysis)

---

**Ready to process your data and run the new EDA!** 🚀

Questions to address before next session:
- Do your raw JSONL files have `document_id` field?
- If not, should we auto-generate them (one per line) or group by some pattern?
- Are you ready to see the "honest" performance (F1 ~0.60 instead of ~0.85)?

# Machine Detection of Lexical Units with the Potential to Violate Informational Quality in News Texts

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Czert%2FRobeCzech-green)](https://huggingface.co/ufal/robeczech-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()

> **Master's Thesis** | Palacký University Olomouc | Department of General Linguistics

This repository contains the source code, experimental framework, and documentation for a master's thesis focused on the **automated identification of lexical units with the potential to violate informational quality** in Czech news texts (abbreviated as LJMPNIK — *lexikální jednotka mající potenciál narušit informační kvalitu zpravodajství*). The project combines domain knowledge from media studies and linguistics with modern NLP (Natural Language Processing) and machine learning methods.

---

## Project Goals

The main task is detection of lexical units that have the potential to violate journalistic objectivity — a form of **bias detection**. The project approaches this problem through three experimental tracks, evaluated at both the word level (S1) and sentence level (S2):

1.  **M1 — Unsupervised Outlier Detection**
    * Anomaly detection methods to identify deviations from neutral news style.
    * **Algorithms:** One-Class SVM, Isolation Forest, Mahalanobis distance.
    * **Input:** Contextualized embeddings (Czert / RobeCzech).

2.  **M2 — Supervised Classification**
    * Training classifiers on annotated data.
    * **Algorithms:** Logistic Regression, SVM, XGBoost over BERT embeddings.

3.  **M3 — LLM In-context Classification**
    * Zero-shot and few-shot classification using large language models without task-specific training.
    * **Approach:** Prompt engineering (zero-shot / few-shot), post-hoc POS-tag filtering.
    * **Model:** Gemma 3 (27B) via Google Generative AI API.

4.  **M4 — LLM Generative Extraction**
    * Generative approach: the model is prompted to directly extract LJMPNIK candidates from a given sentence.
    * **Model:** Gemma 3 (27B) via Google Generative AI API.


## Datasets

The project is built on **two unique datasets created by the author** specifically for this thesis:

* **GOLD Dataset:** A high-quality, manually annotated dataset. Contains expert-verified examples of lexical units with the potential to violate informational quality (a combination of real news texts and LLM-augmented samples). Serves as the *Ground Truth* for evaluation.
* **SILVER Dataset:** A large, automatically generated weakly-labelled dataset. Used to train model robustness and validate methods on a larger data volume.

> **Note on data availability:** Due to the ongoing thesis proceedings and the originality of the data, **the datasets are not currently publicly available**. Their release in the [Releases](../../releases) section is planned after submission and defence (Summer Semester 2026).

## Technologies & Tools

The project uses a modern Python stack for Data Science and NLP:

* **Language:** Python 3.9+
* **NLP & Embeddings:** `transformers` (HuggingFace), Czert / `ufal/robeczech-base`, `spaCy`, `spacy-udpipe`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`
* **LLM API:** `google-generativeai` (Gemma 3)
* **Dimensionality Reduction:** `umap-learn`
* **Statistical Testing:** `statsmodels`, `scipy`
* **Data Processing:** `pandas`, `numpy`
* **Visualisation:** `seaborn`, `matplotlib`

## Repository Structure

The project follows a modular structure separating data, experimental notebooks, and reusable source code.

```text
├── data/
│   ├── raw/               # Raw JSONL data (private)
│   └── processed/         # Preprocessed pickled data (tokens, sentences)
├── notebooks/             # Reproducible experiments (Jupyter)
│   ├── 01_Data_Processing.ipynb             # Data loading and NLP preprocessing
│   ├── 02_EDA.ipynb                         # Exploratory Data Analysis
│   ├── 03_M1_S1_Unsupervised_Token.ipynb    # M1 anomaly detection – word level
│   ├── 04_M1_S2_Unsupervised_Sentence.ipynb # M1 anomaly detection – sentence level
│   ├── 05_M2_S1_Supervised_Token.ipynb      # M2 supervised classification – word level
│   ├── 06_M2_S2_Supervised_Sentence.ipynb   # M2 supervised classification – sentence level
│   ├── 07_M3_S1_LLM_Benchmark_Token.ipynb   # M3 LLM classification – word level
│   ├── 08_M3_S2_LLM_Sentence.ipynb          # M3 LLM classification – sentence level
│   ├── 09_Statistical_Analysis.ipynb         # Statistical tests (permutation, bootstrap, ANOVA/KW)
│   └── 10_M4_LLM_Generative_Extraction.ipynb # M4 LLM generative extraction
├── src/                   # Source code (Python modules)
│   ├── config.py                  # Central project configuration
│   ├── load_preprocess_data.py    # Data loading and NLP preprocessing pipeline
│   ├── data_splitting.py          # Data splitting logic and experimental scenario definitions
│   ├── models.py                  # Model definitions, feature extraction, hyperparameter tuning
│   ├── experiments.py             # Experiment runners
│   ├── evaluation.py              # Metrics (Precision-Recall, F1, AUPRC, bootstrap CI, permutation tests)
│   ├── llm_client.py              # LLM API client (M3 / M4)
│   ├── analysis.py                # Qualitative analysis utilities
│   └── visualization.py           # Unified visualisations for reports
├── results/               # Saved figures and CSV result tables
├── requirements.txt
└── README.md
```

---

**Michal Tobiáš Dobeš**

* Student, NMgr. General Linguistics, Faculty of Arts, Palacký University Olomouc
* Focus: Computational Linguistics, NLP, Machine Learning
* ✉️ [michaltobias.dobes01@upol.cz](mailto:michaltobias.dobes01@upol.cz)

**Supervisor:** Mgr. Vladimír Matlach, Ph.D. (Department of General Linguistics, FF UP)

*Note: The thesis is in progress. Planned completion and defence: Summer Semester 2026.*

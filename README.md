# Strojová detekce lexikálních jednotek s potenciálem narušit informační kvalitu ve zpravodajských textech
 

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-RobeCzech-green)](https://huggingface.co/ufal/robeczech-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()

> **Diplomová práce** | Univerzita Palackého v Olomouci | Katedra obecné lingvistiky

Tento repozitář obsahuje zdrojový kód, experimentální framework a dokumentaci k diplomové práci zaměřené na **automatizovanou identifikaci slov porušujících objektivu** v českých zpravodajských textech. Projekt kombinuje doménové znalosti mediálních studií a lingvistiky s moderními metodami NLP (Natural Language Processing) a strojového učení.

---

## Cíle projektu

Hlavním úkolem je detekce lexikálních jednotek, které mají potenciál narušovat objektivitu zpravodajství – tzv. **bias detection**. Projekt řeší tento problém čtyřmi přístupy (na úrovni vět i slov):

1.  **Unsupervised Outlier Detection**
    * Využití metod detekce anomálií pro identifikaci "odchylek" od neutrálního zpravodajského stylu.
    * **Algoritmy:** One-Class SVM, Isolation Forest, Mahalanobisova vzdálenost.
    * **Vstup:** Contextualized embeddings (BERT/RobeCzech).

2.  **Supervised Classification (Klasifikace)**
    * Trénování klasifikátorů na anotovaných datech.
    * **Algoritmy:** Logistic Regression, SVM, XGBoost nad BERT embeddingy.

3.  **Neural Networks (Hluboké učení)**
    * Implementace vlastních architektur neuronových sítí.
    * **Supervised:** MLP (Multilayer Perceptron) a BiLSTM pro klasifikaci sekvencí.
    * **Unsupervised:** Autoenkodéry pro detekci anomálií na základě rekonstrukční chyby.

4.  **Large Language Models (LLM)**
    * Využití state-of-the-art generativních modelů.
    * **In-context Learning:** Zero-shot / Few-shot learning, pokročilý Prompt Engineering.
    * **Fine-tuning:** Doménová adaptace open-source modelů (Llama 3, Mistral…).
  

## Vlastní datasety

Projekt je postaven na **dvou unikátních datasetech vytvořených autorem** speciálně pro účely této práce:

* **GOLD Dataset:** Ručně anotovaný dataset vysoké kvality. Obsahuje expertně ověřené příklady lexikálních jednotek s potenciálem narušit inforamční kvalitu zpravodajství (kombinace reálných textů a LLM augmentace). Slouží jako *Ground Truth* pro evaluaci.
* **SILVER Dataset:** Rozsáhlý, automaticky generovaný dataset (weakly labeled). Slouží k trénování robustnosti modelů a ověření metod na větším objemu dat.

> **Poznámka k dostupnosti dat:** Vzhledem k probíhajícímu řízení a originalitě dat **nejsou datasety momentálně veřejně dostupné**. Jejich zveřejnění v sekci [Releases](../../releases) je plánováno po odevzdání a obhajobě práce (LS 2026).

## Technologie a nástroje

Projekt využívá moderní Python stack pro Data Science:

* **Jazyk:** Python 3.9+
* **NLP & Embeddings:** `transformers` (HuggingFace), `ufal/robeczech-base`, `spaCy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Data Processing:** `pandas`, `numpy`
* **Vizualizace:** `seaborn`, `matplotlib`

## Struktura repozitáře

Projekt dodržuje modulární strukturu oddělující data, experimentální notebooky a znovupoužitelný kód.

```text
├── data/                  # Složka pro data
│   ├── raw/               # (Surová JSONL data - neveřejné)
│   ├── vectors/           # (Předpočítané embeddingy - neveřejné)
├── notebooks/             # Reprodukovatelné experimenty (Jupyter)
│   ├── 01_EDA_Preprocess.ipynb           # Exploratorní analýza a NLP pipeline
│   ├── 02_M1_S1_Unsupervised_Token.ipynb # Detekce anomálií na úrovni slov
│   ├── 03_M1_S2_Unsupervised_Sentence.ipynb
│   ├── 04_M2_S1_Supervised_Token.ipynb   # Supervised Klasifikace 
│   └── ...
├── src/                   # Zdrojový kód (Python moduly)
│   ├── config.py          # Centrální konfigurace projektu
│   ├── load_data.py       # Data loading a preprocessing pipeline
│   ├── visualization.py   # Unifikované vizualizace pro reporty
│   └── evaluation.py      # Metriky (Precision-Recall, F1, AUPRC)
├── requirements.txt       
└── README.md              
```

---

**Michal Tobiáš Dobeš**

* Student NMgr. Obecná lingvistika, FF UP
* Zaměření: Computational Linguistics, NLP, Machine Learning
* ✉️ [michaltobias.dobes01@upol.cz](mailto:michaltobias.dobes01@upol.cz)

**Vedoucí práce:** Mgr. Vladimír Matlach, Ph.D. (Katedra obecné lingvistiky FF UP)

*Poznámka: Práce je ve vývoji. Plánované dokončení a obhajoba: Letní semestr 2026.*

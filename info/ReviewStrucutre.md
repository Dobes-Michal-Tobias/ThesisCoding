# Struktura experimentů

### Filtrace
* **`filter_type`:**
    * `'aggressive'` (Jen NOUN, ADJ, VERB, ADV)
    * `'mild'` (Vše kromě ADP, CONJ, PUNCT, SYM)
    * `'none'` (Všechny tokeny)

### Reprezentace věty 
* **`pooling_method`:**
    * `'mean'` (Průměr embeddingů tokenů)
    * `'cls'` (Embedding speciálního [CLS] tokenu)

### Datasety
1.  **I - Malý čistý dataset** $\rightarrow$ **`GOLD` Dataset**
    * Ručně anotovaný, kotrolovaný, vysoká kvalita, důvěryhodný pro testování. Kombinace REAL, LLM, AUTHOR.
2.  **II - Velký dataset** $\rightarrow$ **`SILVER` Dataset**
    * Automaticky generovaný, významně větší, ale může obsahovat šum (není 100% perfektní), "weakly labeled" data.
3.  **III - Hybridní dataset** $\rightarrow$ **`HYBRID` Dataset**
    * Kombinuje anomálie ze SILVER a GOLD a neutralitu z GOLD.

---
Každý níže uvedený experiment se provádí v kombinacích:
`'aggressive'` vs `'mild'` vs `'none'`

### **---> M1: Unsupervised Outlier Detection**
*Cíl: Modelování normality. Trénujeme pouze na neutrální třídě (L0).

* **S1 - Token Level (Slova)**
* *Poznámka:* Počítáme závilost F1 na parametrech/tresholdech: *nu*, *contamination*, *p-value*
* *Poznámka:* *`'Gold Baseline'`:** Trénink na GOLD L0, test na GOLD L1+L0. * `'Combined Robustness'`:** Trénink na GOLD L0, test na GOLD+SILVER L1 a GOLD L0.

    * **S1a - ONSVM (One-Class SVM):** Geometrická metoda pro nelineární hranice.
        * **S1a-I (Gold Baseline)** 
        * **S1a-II (Combined Robustness)**
    * **S1b - Isolation Forest (IF):** Stromová metoda izolující odlehlé body.
        * **S1b-I (Gold Baseline)** 
        * **S1b-II (Combined Robustness)** 
    * **S1c - Mahalanobis Distance (MD):** Statistická metoda (vzdálenost od centroidu distribuce).
        * **S1c-I (Gold Baseline)**
        * **S1c-II (Combined Robustness)**
* *Otázka:* Na trénovat nejlepší model znovu na augmentovaném datasetu… L1 z Gold a Silver, L0 z target Gold a z Gold kontextových vět

* **S2 - Sentence Level (Věty)**
    * *Poznámka:* U všech S2 experimentů navíc porovnáváme vliv reprezentace věty (**Mean Pooling** vs **[CLS] Token**).
    * *Poznámka:* Počítáme závilost F1 na parametrech/tresholdech: *nu*, *contamination*, *p-value*
    * **S2a - ONSVM (Věty):** Klasifikace celých vět jako anomálií.
        * **S2a-I (Gold Baseline)**
        * **S2a-II (Combined Robustness)**
    * **S2b - Isolation Forest (Věty):**
        * **S2b-I (Gold Baseline)** 
        * **S2b-II (Combined Robustness)** 
    * **S2c - Mahalanobis Distance (Věty):**
        * **S2c-I (Gold Baseline)** 
        * **S2c-II (Combined Robustness)**
 
### **---> M2: Supervised Classification**

* **S1 - Token Level (Slova)**
    * **S1a - Baseline (Imbalanced):** Všechna data, nev    yvážené.
    * **S1b - Gold Balanced:** Undersampling L0 z Gold datasetu.
    * **S1c - Bootstrap Validation:** Ověření stability S1b (100 iterací).
    * **S1d - Train Noisy, Test Clean:** Trénink na Silver, Test na Gold.
    * **S1e - Hybrid:** Max L1 (Gold+Silver) + Clean Balanced L0 (Gold).

* **S2 - Sentence Level (Věty)**
* *Poznámka:* U všech S2 experimentů navíc porovnáváme vliv reprezentace věty (**Mean Pooling** vs **[CLS] Token**).
    * **S2a - Gold Balanced:** Trénink na větách z malého datasetu.
    * **S2b - Hybrid:** Trénink na L1 větách (GOLD + SILVER) + L0 větách (Gold).
  

### --> M3 (LLM)
* **S1 - Token Level**
* **S2 - Sentence Level**

### --> M4 (NN):
* **S1 - Token Level**
* **S2 - Sentence Level**
  
### --> Kombinace pro praxi (prvně poznat větu, pak konkrétní LJMPNIK)


---
---

## Struktura projektu

```text
diploma_thesis_project/
│
├── data/                   # (V .gitignore, na GitHub se nedává)
│   ├── raw/                # Původní JSONL
|   ├── interim/            # Zprocesované JSONL s embed. a postagy pro lepší čtení a procházení.
│   └── vectors/            # Zpracované .pkl
│
├── src/  
|   ├── src_servis/                 # pomocné skripty 
│       ├── check_models.py
│       ├── data_generate.py        # Generátor SILVER datasetu      
│   ├── __init__.py
│   ├── config.py           		# Konstanty, cesty, palety
│   ├── load_preprocess_data.py	    # Načítání a preprocessing textu (embeddingy, postagy, tokenizace…)
│   ├── evaluation.py       		# Metriky, PR Curve, Bootstrap (zatím nemám!!)
│   └── visualization.py    		# Unifikované funkce pro grafy (zatím nemám!!)
│
├── notebooks/              
│   ├── 01_EDA_preprocess.ipynb
│   ├── 02_M1_S1_Unsupervised_Detection-Token_level.ipynb
│   ├── 03_M1_S2_Unsupervised_Detection-Senc_level.ipynb
│   ├── 04_M2_S1_Supervised_Classification-Token_level.ipynb
│   ├── 05_M2_S1_Supervised_Classification-Token_level.ipynb
│
├── requirements.txt        # Seznam knihoven (pip freeze)
├── README.md               # Popis projektu, jak to spustit
└── venv      
```

* Možná časem přidat `└── main.py   # (Volitelné) Jeden skript, který by to pustil celé naráz`
 
---
---

#📂 1. Logika Dat a Souborů:

1. **RAW JSONL** jsou nedotknutelné archivy.
2. **PKL Artefakty** jsou "cihly" (vektory), které si předpřipravíme (jednou).
3. **Trénovací sety (Hybrid)** se staví **"on-the-fly"** v paměti spojováním cihel.

###Adresářová struktura dat (`data/`)* `data/raw/` -> `dataset_gold.jsonl`, `dataset_silver.jsonl`
* `data/vectors/gold/` -> `gold_token_mild_l0.pkl`, `gold_sent_mean_l1.pkl`, ...
* `data/vectors/silver/` -> `silver_token_mild_l1.pkl`, ...

---

#🐍 2. Obsah složky `src/` (Python Moduly)Toto je tvá knihovna. Notebooky budou volat funkce odsud.

###A. `src/config.py` (Mozek)* **Cesty:** Definované přes `pathlib` (BASE_DIR, DATA_DIR, VECTORS_DIR).
* **Model:** `ufal/robeczech-base`, `MAX_LENGTH`.
* **Filtry (Sety):**
* `POS_ALLOWED_AGGRESSIVE = {'NOUN', 'ADJ', ...}`
* `POS_FORBIDDEN_MILD = {'ADP', 'PUNCT', ...}`


* **Barvy (Design):**
* `COLORS`: Dictionary s hex kódy (Pastel pro třídy, Coolwarm pro heatmapy).
* `L0 = Blue`, `L1 = Red`.


* **Konstanty:** `RANDOM_SEED = 42`.

###B. `src/load_preprocess_data.py` (Motor)Tento soubor bude mít dvě hlavní části:

1. **Pipeline pro generování artefaktů (spouští se v 01_EDA):**
* Fce `process_jsonl_to_pkl(input_path, dataset_name)`:
* Načte JSONL.
* Spustí BERT a spaCy.
* Aplikuje filtry (loop přes none/mild/agg).
* Uloží `.pkl` soubory do `data/vectors/`.
* **Důležité:** Ukládá zvlášť L0 a L1. Ukládá zvlášť Tokeny a Věty (Mean/CLS).


2. **DataLoader pro experimenty (spouští se v M1/M2):**
* Fce `load_data(strategy, level, filter_type, pooling=None)`:
* Pokud `strategy == 'GOLD_BASELINE'`: Načte jen Gold `.pkl`.
* Pokud `strategy == 'HYBRID'`: Načte Gold `.pkl` + Silver `.pkl` a spojí je (`np.concatenate`).
* Vrací hotové `X_train`, `X_test`, `y_train`, `y_test` připravené pro `model.fit()`.


###C. `src/evaluation.py` (Rozhodčí)* Fce `calculate_metrics(y_true, y_pred)`: Vrací prec, rec, f1.
* Fce `find_optimal_threshold(y_true, anomaly_scores)`:
* Pro M1 (Unsupervised).
* Projede PR křivku a najde threshold s nejvyšším F1.


* Fce `bootstrap_evaluation(model, X, y)`: Pro M2/S1c.

###D. `src/visualization.py` (Malíř)* Fce `setup_style()`: Nastaví Seaborn theme a paletu z configu.
* Fce `plot_pr_curve()`: Vykreslí křivku a bod optima.
* Fce `plot_confusion_matrix()`: Heatmapa s `coolwarm` (nebo Blues/Reds).
* Fce `plot_metric_dependency()`: Pro závislost F1 na parametrech (contamination/nu).

---

#📓 3. Struktura Notebooků (Experimenty)Notebooky budou čisté, s minimem kódu a maximem Markdownu (příběh).

###`01_EDA_preprocess.ipynb`* **Účel:** Příprava půdy.
* **Akce:**
1. Import `src.load_preprocess_data`.
2. Spustí generování vektorů pro GOLD i SILVER.
3. **EDA:** Vykreslí počty slov, histogramy délek vět, poměry tříd (použije `src.visualization`).
4. 


###`02_M1_Unsupervised_Detection.ipynb` (Sjednoceno Token & Sent)* **Účel:** Experimenty M1 (S1 i S2).
* **Struktura kódu:**
* Loop přes **Level** (`Token`, `Sentence`).
* Loop přes **Algo** (`ONSVM`, `IF`, `MD`).
* Loop přes **Filter** (`Agg`, `Mild`, `None`).
* **Exp 1 (Baseline):** Train Gold L0 -> Test Gold Mix.
* **Exp 2 (Robustness):** Train Gold L0 -> Test Hybrid Mix.
* *Výpočet:* Hledání optimálního `p-value` / `nu`.
* *Vizualizace:* PR Křivka.
* 

###`03_M2_Supervised_Classification.ipynb` (Sjednoceno Token & Sent)* **Účel:** Experimenty M2.
* **Struktura kódu:**
* **Část S1 (Tokeny):**
* S1a (Baseline) -> S1b (Balanced) -> S1c (Bootstrap) -> S1d (Noisy Train) -> **S1e (Hybrid Master)**.


* **Část S2 (Věty):**
* Srovnání `Mean` vs `CLS`.
* S2a (Gold Balanced) -> S2b (Hybrid Master).





---

#🛠 4. Implementační "Cheat Sheet"Tady je návod, co teď udělat krok za krokem:

1. **Vytvoř složky:** `src`, `data/raw`, `data/vectors`.
2. **Vytvoř `src/config.py`:** Definuj tam cesty a barvičky.
3. **Vytvoř `src/load_preprocess_data.py`:**
* Zkopíruj tam logiku z našeho posledního chatu (BERT + ukládání PKL).
* Dopiš funkci `get_data_for_experiment(...)`, která bude umět spojit Gold a Silver vektory.


4. **Spusť `01_EDA_preprocess.ipynb`:** Nech to chroustat (vytvoří to `.pkl` soubory).
5. **Vytvoř `src/visualization.py`:** Aby grafy v dalších noteboocích vypadaly stejně.
6. **Začni experimentovat v `02_M1...`:** Teď už jen importuješ `get_data_for_experiment`, zavoláš model a kreslíš grafy.

Tato struktura je **modulární**. Až budeš dělat M3 (LLM), jen přidáš `04_M3_LLM.ipynb` a využiješ stejná data a stejné vizualizace.

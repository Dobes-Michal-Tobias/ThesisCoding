# Strojová detekce lexikálních jednotek s potenciálem narušit informační kvalitu ve zpravodajských textech

**Autor:** Bc. Michal Tobiáš Dobeš (student NMgr. Obecná lingvistika na KOL FF UP, michaltobias.dobes01@upol.cz)

**Vedoucí práce:** Mgr. Vladimír Matlach, Ph.D. (Katedra obecné lingvistiky FF UP, vladimir.matlach@upol.cz)

**❗Práce k 16.1.2026 není ještě dokončena. Plánovaná doba dokončení je LS 2026**

V diplomové práci se zabývám  aplikací strojového učení s cílem identifikace jazykových lexikálních jednotek, které mají potenciál porušit informační kvalitu typicky zpravodajských textů. Využíváme metody čítající detekce anomálií, supervised/unsupervised techniky, velké jazykové modely (LLM) a další modely umělé inteligence (AI).
Veškeré tyto technické aplikace budu rámovat do žurnalistické a lingvistické doménové znalosti a poskytnu kvantitativní a kvalitativní resumé pro využití výsledků v praxi.

Tento repozitář obsahuje zdrojový kód a experimentální framework pro moji diplomovou práci. 



![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NLP](https://img.shields.io/badge/NLP-RobeCzech-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


## Cíl práce

Hlavním úkolem je identifikovat slova nebo celé věty, které mohou nauršovat *objektivitu* zpravodajství. Projekt zkoumá dva hlavní přístupy:
1.  **Unsupervised Outlier Detection (M1):** Modelování normality na základě neutrálních textů (ONSVM, IF, Mahalanobisova vzdálenost).
2.  **Supervised Classification (M2):** Klasifikace pomocí natrénovaných modelů (LogReg, SVM, XGBoost…) nad BERT embeddingy.

## Struktura repozitáře

Projekt je rozdělen do modulů (`src`) a Jupyter notebooků (`notebooks`), které reprezentují jednotlivé fáze experimentů.

```text
├── data/                  # Složka pro data (stáhněte z Releases)
│   ├── raw/               # Surová JSONL data
│   ├── interim/           # Předzpracovaná data
│   └── vectors/           # Vypočítané embeddingy (.pkl)
├── notebooks/             # Experimentální notebooky
│   ├── 01_EDA_Preprocess.ipynb               # Čištění dat, statistiky, embeddingy
│   ├── 02_M1_S1_Unsupervised_Token.ipynb     # Detekce anomálií (slova)
│   ├── 03_M1_S2_Unsupervised_Sentence.ipynb  # Detekce anomálií (věty)
│   ├── 04_M2_S1_Supervised_Token.ipynb       # Klasifikace (slova) + Bootstrap
│   └── 05_M2_S2_Supervised_Sentence.ipynb    # Klasifikace (věty)
├── src/                   # Zdrojový kód a pomocné moduly
│   ├── config.py          # Konfigurace cest a parametrů
│   ├── models.py          # Wrappery pro ML modely
│   ├── visualization.py   # Vizualizační funkce (PCA, t-SNE, metriky)
│   └── ...
├── info/                  # Dokumentace k datasetům a struktuře
└── README.md              # Tento soubor
```

### Stažení dat

Vzhledem k velikosti nejsou dataset a vypočítané vektory (embeddingy) součástí repozitáře.

1. Přejděte do sekce **[Releases]((https://github.com/Dobes-Michal-Tobias/ThesisCoding/releases))** v tomto repozitáři.
2. Stáhněte archiv `data.zip`.
3. Obsah rozbalte do kořenové složky projektu tak, aby vznikla struktura podle popisu (viz výše).
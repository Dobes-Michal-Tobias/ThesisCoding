
-----

# Report: Detekce lexikálních anomálií (LJMPNIK) pomocí embeddingů

**Cíl výzkumu:** Ověřit, do jaké míry lze automaticky identifikovat lexikální jednotky s potenciálem narušit informační kvalitu (LJMPNIK) ve zpravodajských textech bez využití trénovacích dat s anotacemi (Unsupervised Learning). Předpokladem je, že tyto jednotky představují sémantickou a stylistickou odchylku (outlier) od neutrálního zpravodajského standardu.

**Použitý model:** `ufal/robeczech-base` (BERT-based model pro češtinu), dimenze embeddingů $d=768$.

-----

## \---\> M1: Unsupervised Outlier Detection (Metoda 1)

V této fázi nezná model labely (0/1) během trénování. Učí se pouze charakteristiky "normálních" (neutrálních) dat a následně testujeme, zda se LJMPNIK statisticky vymykají této normě.

### \-\> S1: Detekce na úrovni tokenů (Token-Level)

  * **Fokus:** Jednotlivá slova (tokeny).
  * **Hypotéza:** Samotný vektorový prostor slov nese informaci o stylistické vhodnosti. LJMPNIK by mělo ležet v jiné části prostoru než běžná neutrální slova.
  * **Vstup:** Embeddingy jednotlivých tokenů ($\mathbf{x} \in \mathbb{R}^{768}$).
  * **Trénovací data:** 80 % neutrálních tokenů (cca 10 000 vzorků).

#### \-\> S1-ONSVM (One-Class SVM)

  * **Metoda:** Geometrický přístup. Model hledá hyperplochu v 768-dimenzionálním prostoru, která odděluje neutrální data od počátku (origin).
  * **Výsledek:** **F1-Score $\approx$ 0.05**.
  * **Interpretace:** Selhání. Prostor embeddingů je příliš komplexní a "zašuměný". Geometrické oddělení bez zohlednění korelace dimenzí nefunguje; model nedokázal nalézt hranici mezi neutrálním slovem a anomálií.

#### \-\> S1-IF (Isolation Forest)

  * **Metoda:** Algoritmus založený na náhodném dělení prostoru (random cuts). Předpokládá, že anomálie se izolují rychleji (méně řezy).
  * **Výsledek:** **F1-Score $\approx$ 0.05**.
  * **Interpretace:** Selhání. Stejný problém jako u ONSVM. Všechna slova jsou v tak vysoké dimenzi "daleko od sebe", algoritmus v šumu nenachází vzor.

#### \-\> S1-MD (Mahalanobis Distance)

  * **Klíčový experiment (Návrh vedoucího):** Statistický přístup. Modelujeme neutrální data jako mnohorozměrné normální rozdělení definované průměrem $\boldsymbol{\mu}$ a kovarianční maticí $\boldsymbol{\Sigma}$.
  * **Výpočet:**
    1.  Natrénování `EmpiricalCovariance` na neutrálních tokenech.
    2.  Výpočet Mahalanobisovy vzdálenosti pro každý token: $D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$.
    3.  Převod na P-hodnotu pomocí $\chi^2$ rozdělení.
    4.  Hledání optimálního thresholdu P-value (alfa) pro klasifikaci outlierů.
  * **Výsledek:** **F1-Score $\approx$ 0.34**.
  * **Interpretace:** **Výrazné zlepšení oproti geometrickým metodám.** Zohlednění kovariance (vztahů mezi dimenzemi) umožňuje modelu lépe chápat strukturu dat. Nicméně F1 0.34 je pro praktické využití stále nedostatečné. Potvrzuje se hypotéza, že *samotné embeddingy slov bez širší agregace kontextu nestačí*.

-----

### \-\> S2: Detekce pomocí lokálního kontextu (Centroidy)

  * **Fokus:** Vztah tokenu k jeho bezprostřednímu okolí.
  * **Hypotéza:** LJMPNIK by mělo být sémanticky vzdálené od průměru (centroidu) okolních slov v odstavci.
  * **Vstup:** 1D skalár (Kosinová vzdálenost tokenu od centroidu kontextu).
  * **Výsledek:** **F1-Score $\approx$ 0.15**.
  * **Interpretace:** Slepá ulička. Variabilita uvnitř neutrálních vět je příliš vysoká. I neutrální slova mohou být sémanticky daleko od "tématu" věty. Redukce 768 dimenzí na jediné číslo (vzdálenost) vedla ke ztrátě klíčové informace.

-----

### \-\> S3: Detekce na úrovni vět (Sentence-Level)

  * **Fokus:** Celé věty.
  * **Hypotéza:** Přítomnost silně zabarveného slova (LJMPNIK) "otráví" sémantický vektor celé věty a vychýlí ho mimo distribuci neutrálních vět.
  * **Metodologie:**
    1.  **Mean Pooling:** Embedding věty získáme jako průměr embeddingů jejích tokenů.
    2.  **Data Augmentation (Využití kontextu):** Abychom měli dostatek dat pro výpočet kovarianční matice v 768D, využili jsme fakt, že kontextové věty (`context_prev`, `context_next`) jsou v datasetu vždy neutrální. Tím jsme zvětšili datovou sadu (Původně: L1=332 a L2= 188 --> Celkem neutrálních vět po augmentaci: 1228, Celkem anomálních vět: 332, Trénujeme na: 982 neutrálních větách, Testujeme na: 246 neutrálních + 332 anomálních větách).
    3.  **Dimensionality Reduction (PCA):** Optimalizace dimenze pro odstranění šumu.
    4.  **Mahalanobis Distance:** Detekce outlierů na úrovni celých vět.

**Klíčová část kódu (Augmentace dat):**

```python
# Ukázka logiky augmentace pro S3
sentence_embeddings = []
labels = []

for row in dataset:
    # 1. Věta Před (Vždy Neutrální) -> Přidat do tréninku jako Label 0
    if row['context_prev_tokens']:
        sentence_embeddings.append(np.mean(row['context_prev_tokens'], axis=0))
        labels.append(0)
        
    # 2. Věta Po (Vždy Neutrální) -> Přidat do tréninku jako Label 0
    if row['context_next_tokens']:
        sentence_embeddings.append(np.mean(row['context_next_tokens'], axis=0))
        labels.append(0)

    # 3. Cílová věta -> Původní Label (0 nebo 1)
    sentence_embeddings.append(np.mean(row['target_tokens'], axis=0))
    labels.append(row['label'])
```

#### \-\> S3a: Agresivní filtrace (Aggressive Filtering)

  * **Princip:** Před průměrováním věty odstraníme všechna funkční slova (předložky, spojky, zájmena, částice). Ponecháme jen obsahová slova (NOUN, ADJ, VERB, ADV) a samotné LJMPNIK.
  * **Výsledek:** **Validované F1-Score $\approx$ 0.75**.
      * **Recall:** \~1.0 (Detekuje téměř všechny anomálie).
      * **Precision:** \~0.60 (Vytváří falešné poplachy u složitějších neutrálních vět).

#### \-\> S3b: Jemná filtrace (Mild Filtering)

  * **Princip:** Odstranění pouze technických elementů (interpunkce) a čistých konektorů (předložky, spojky). Zájmena a částice ponechány.
  * **Výsledek:** F1-Score $\approx$ 0.75.

#### \-\> S3c: Žádná filtrace (No Filtering)

  * **Princip:** Průměrování všech tokenů ve větě.
  * **Výsledek:** F1-Score $\approx$ 0.75.
  * **Interpretace:** Asi tolik na filtraci nezávisí… 


-----

## \--\> M2: Supervised Learning (Plánovaná Metoda 2)

V této fázi využijeme anotace (Labely 0/1) pro trénink klasifikátoru. Cílem je vyřešit nízkou Precision z M1 a vrátit se k detekci na úrovni slov.

### \-\> S1: Klasifikace na úrovni slov

  * **Cíl:** Natrénovat model (Logistická regrese / SVM), který pro každý token rozhodne $P(LJMPNIK | \text{embedding})$.
  * **Řešení problému s daty:**
      * Máme silně nevyvážený dataset (tisíce neutrálních slov vs. 332 anomálií).
      * Využijeme **Undersampling** neutrální třídy (náhodný výběr podstatných jmen/sloves z kontextu) pro vytvoření vyváženého trénovacího setu 1:1.
  * **Očekávání:** Díky přímé supervizi by měl model být schopen identifikovat anomální slova s mnohem vyšší přesností než unsupervised metody.

### \-\> S2: Klasifikace na úrovni věty

  * **Cíl:** Supervised klasifikace celých vět (obsahuje/neobsahuje LJMPNIK).
  * **Využití:** Pro srovnání s M1/S3. Očekáváme, že supervised model dokáže lépe odlišit falešné poplachy (False Positives).
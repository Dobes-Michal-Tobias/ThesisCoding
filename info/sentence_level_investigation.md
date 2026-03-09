# Investigace: Podezřele Vysoké Výsledky na Sentence-Level (M1/S2 & M2/S2)

**Datum analýzy:** 2026-03-09
**Analyzoval:** Claude (Senior ML/NLP review)
**Status:** Hypotézy identifikovány — kód NEOPRAVOVÁN, čeká na prodiskutování

---

## 0. Shrnutí (TL;DR)

Bylo identifikováno **5 hypotéz** vysvětlujících vysoké výsledky. Jedna z nich
(**H1: CLS Embedding Leakage**) je potvrzen kódový problém — jde o **architektonický
defekt**, ne o klasický "document leakage". Zbývající 4 jsou statisticko-lingvistické
fenomény, které výsledky zkreslují i bez chyby v kódu.

---

## 1. Přehled Výsledků, které Vyvolaly Podezření

### M1/S2 — Unsupervised (Mahalanobis, IF, OCSVM)

| Scénář | Pooling | Nejlepší model | Test F1 | Test AUPRC |
|---|---|---|---|---|
| S2a Baseline (136 L0 v train) | CLS | IsolationForest | **0.7848** | 0.6761 |
| S2a Baseline (136 L0 v train) | Mean | IsolationForest | 0.7355 | 0.6180 |
| S2c Robustness (888 L0 v train) | CLS | OCSVM | 0.3495 | 0.1930 |
| S2c Robustness (888 L0 v train) | Mean | IsolationForest | 0.2484 | 0.2110 |

> Záhadný jev: přidáním **6x více trénovacích dat** (robustness: 888 vs baseline: 136)
> výkon **KLESNE** z ~0.78 na ~0.35. U F1 to je více než 2× zhoršení.

### M2/S2 — Supervised (SVM, LogReg, XGBoost…)

| Scénář | Pooling | Model | Test F1 (L1) | Test AUPRC | Train F1 |
|---|---|---|---|---|---|
| S2b Hybrid (L0=136, L1=926) | Mean | SVM (Lin) | **0.9669** | **0.9934** | 0.9978 |
| S2b Hybrid | Mean | LogReg | ~0.96 | ~0.99 | ~0.99 |
| S2a Baseline | Mean | různé | ~0.85–0.90 | ~0.87 | — |

> Záhadný jev: téměř dokonalé výsledky na testu (0.9669 F1), ale zároveň
> Recall na třídu L0 (neutralní) = 0.757. Jak může být model "super dobrý"
> a zároveň plést každý čtvrtý neutralní vzorek?

---

## 2. Hypotézy

---

### H1: CLS Embedding Architektonický Defekt — SDÍLENÉ EMBEDDINGY PŘES VĚTY

**Závažnost: KRITICKÁ** | **Pravděpodobnost: VYSOKÁ**

#### Popis problému

V souboru [load_preprocess_data.py](../src/load_preprocess_data.py), funkce
`create_processed_dataframes()`, řádky 234–263:

```python
# --- Target Sentence ---
t_sentence['cls_embedding'] = cls_embedding   # <-- CLS z celého [PREV+TARGET+NEXT]

# --- Context Prev ---
p_sentence['cls_embedding'] = cls_embedding   # <-- STEJNÝ CLS!

# --- Context Next ---
n_sentence['cls_embedding'] = cls_embedding   # <-- STEJNÝ CLS!
```

CLS embedding je vypočten JEDNOU z celého bloku `[PREV + TARGET + NEXT]` a
pak **přiřazen všem třem větám** (target i oběma kontextovým). To znamená:

- Věta `context_prev` má label=0 (neutrální), ale její `cls_embedding` je
  zakódovaný celý blok — včetně případné anomálie v `target_sentence`.
- Věta `context_next` má label=0 (neutrální), ale nese stejný CLS.
- To jsou tři záznamy v DataFrame s různými labely, ale sdíleným vektorem.

#### Proč to poškozuje výsledky

**Scénář Robustness (M1/S2):**
Tréninková data v robustness scénáři obsahují **všechny věty** — target i
kontextové (`df = gold_df.copy()`). Model M1 se pak učí "normalitu" (L0) na
kontextových větách, které nesou CLS embeddingy L1 bloků. Model se naučí, že
"CLS pattern přítomný u anomálií je normální", a proto pak na testovacích datech
selhává. **Toto přesně vysvětluje záhadný pokles výkonu při přidání dat!**

```
TRAIN (Robustness, L0 po filtraci):
  - context_prev z L0 dokumentu → cls_embedding = f(L0 blok)  ✅ OK
  - context_next z L0 dokumentu → cls_embedding = f(L0 blok)  ✅ OK
  - target z L0 dokumentu → cls_embedding = f(L0 blok)         ✅ OK
  - context_prev z L1 dokumentu → cls_embedding = f(L1 blok)  ❌ KONTAMINACE
  - context_next z L1 dokumentu → cls_embedding = f(L1 blok)  ❌ KONTAMINACE
  (target z L1 dokumentu → zahozen, label=1, nezahrnut do L0 train) ✅
```

Pro L1 dokumenty jsou jejich kontextové věty (label=0) zahrnuty do tréninku,
ale jejich CLS embedding kóduje přítomnost anomálie. Model se naučí, že "CLS
vektory anomálních bloků jsou normální" → při testování selže v rozlišení.

**Scénář Baseline (M1/S2 a M2/S2):**
Baseline scénář používá POUZE target věty (`~gold_df['is_context']`).
Kontextové věty v train setu NEJSOU. Ale na test setu jsou **target věty L0 i
L1**, kde CLS embedding každé věty obsahuje celý blokový kontext. Pro L1 target
větu je CLS = embedding bloku se anomálním slovem. Model se tedy učí z čistých
L0 CLS a testuje na L1 CLS, které jsou skutečně jiné — výsledky jsou tedy
**validní** pro baseline CLS variantu, ale z **jiného důvodu**, než předpokládáme:
model nemusí detekovat LJMPNIK, ale může reagovat na celkový "tón" bloku zakódovaný
v CLS.

#### Důkaz z výsledků

| Pooling | Baseline F1 | Robustness F1 | Δ |
|---|---|---|---|
| **CLS** | 0.759 | **0.347** | **−0.41** |
| **Mean** | 0.717 | **0.248** | **−0.47** |

Obě metody trpí stejným problémem při přidání kontextových vět. U Mean pooling
není CLS problém přímý, ale Mean embeddingy kontextových vět z L1 bloků jsou
také "kontaminovány" blízkostí anomálního tokenu přes self-attention.

---

### H2: Clever Hans Effect — LLM "Otisk" (Watermark)

**Závažnost: STŘEDNÍ** | **Pravděpodobnost: STŘEDNÍ-VYSOKÁ**

#### Popis problému

Dataset je generovaný LLM s explicitním promptem:
> "Vlož do cílové věty jedno silně hodnotící slovo (např. *skandální, absurdní*)..."

To znamená:
- **L1 věty** mají strukturu: `[neutrální kontext] + [expresivní adjektivum/adverbium] + [neutrální zbytek]`
- **L0 věty** jsou čistě neutrální

RobeCzech model byl natrénovaný na obrovském korpusu českého textu a jeho
reprezentace "expresivních adjektiv" a "neutrálních adjektiv" jsou fundamentálně
odlišné v embedding prostoru — i bez jakéhokoliv tréninku.

#### Proč to zkresluje výsledky

Model (zejména Supervised M2/S2) de facto detekuje **hodnotící lexémy** v
embedding prostoru. To je správná úloha pro token-level, ale pro sentence-level
to znamená, že model nepotřebuje "chápat" anomálii — stačí mu, že LLM konzistentně
vložil do L1 vět slova s odlišnou semantikou.

**Kritický test:** Jsou výsledky nadhodnocené proto, že LLM generátor byl
příliš konzistentní? Pokud bychom přemíchali (shuffled) L1 věty nebo odebrali
podezřelá slova, výkony by mohly prudce klesnout.

#### Evidence z Qualitative Analysis (M2/S2)

TOP FP (model vidí anomálii kde není):
```
"V norimberském procesu byla organizace SS označena..."  → L0, ale model říká L1
"Za jízdu na červenou hrozí vyšší pokuta."              → L0, ale model říká L1
```
Tyto věty mají silná slova ("SS", "označena za zločineckou", "pokuta") — model
reaguje na "negativní" sémantiku, ne na LJMPNIK jako takový.

TOP FN (model přehlédl anomálii):
```
"Omezení se dotkne linek jedoucích přes Karlák."        → L1, model říká L0
"Policejní prezidium povolalo i těžkooděnce."           → L1, model říká L0
```
Tyto L1 věty mají jemné anomálie (hovorové výrazy "Karlák", "těžkooděnci"),
které model nepovažuje za dostatečně anomální v embedding prostoru.

---

### H3: Trivialní Baseline je Příliš Dobrá (Class Imbalance)

**Závažnost: STŘEDNÍ** | **Pravděpodobnost: JISTÁ**

#### Popis problému

Výsledky M2/S2 Hybrid jsou reportovány jako F1=0.9669. Ale toto je F1 **pro
třídu L1 (anomálie)**, ne makro-F1. Podívejme se na "stupidní" baseline:

**Majority Classifier (vždy predikuje L1):**
- Test set Hybrid: 37 L0, 256 L1 (celkem 293)
- Precision = 256/293 = **0.874**
- Recall = 256/256 = **1.000**
- F1(L1) = 2 × (0.874 × 1.0) / (0.874 + 1.0) = **0.933**

Model SVM Lin dosáhl F1(L1) = **0.9669** — to je zlepšení oproti stupidní
baseline o pouhých **+0.034** (3.4 procentního bodu).

**Makro-F1 je daleko střízlivějších 0.867**, z čehož L0 třída má jen F1=0.767.

#### Doporučení

Reportovat výsledky jako **Macro-F1** a zahrnout Dummy Classifier jako baseline.

---

### H4: Threshold Overfitting na Malý Validační Set

**Závažnost: NIŽŠÍ** | **Pravděpodobnost: STŘEDNÍ**

#### Popis problému

Validační set pro Baseline scénář obsahuje pouze **41 vzorků** (15 L0, 26 L1).
Threshold pro převod skóre na binární predikci je optimalizován na tomto setu.

S tak malým validačním setem:
- Optimalizace thresholdu má vysokou varianci
- Výsledky na test setu jsou ovlivněné "lucky" thresholdem
- Konfidenci intervalů pro F1 jsou velmi široké (pravděpodobně ±0.05–0.10)

Příklad: Pro M1/S2 Baseline CLS dostaneme na val F1 (přes threshold) hodnotu,
ze které pak extrapolujeme na test. Ale s 15 L0 vzorky je odhad distribuce
"normality" v CLS prostoru velmi nestabilní.

---

### H5: Mean Embedding NENÍ Sentence Embedding (Architektonická limitace)

**Závažnost: METODOLOGICKÁ** | **Pravděpodobnost: JISTÁ**

#### Popis problému

`mean_embedding` je vypočten jako průměr embeddingů **UDPipe tokenů cílové věty**
(nikoliv všech sub-tokenů BERT). Viz `build_dataframe_records()`, řádek 163:

```python
mean_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(768)
```

Kde `embeddings` jsou již **word-pooled** vektory (Mean přes BERT sub-tokeny
každého UDPipe slova). Výsledný `mean_embedding` je tedy průměr průměrů.

Toto vede k tomu, že anomální slovo (LJMPNIK) — jedno z ~11 slov ve větě —
přispívá k výslednému vektoru pouze 1/11 svého vlivu. Přitom kontextové tokeny
jsou kódovány **s vědomím** anomálního tokenu (přes BERT self-attention), takže
"rozmažou" jeho signal do celé věty.

Tento jev je metodologicky zajímavý: výsledky sentence-level nejsou způsobeny
modelováním anomálie jako celku, ale zachycením "reziduálního" vlivu anomálního
slova na všechny okolní tokeny přes self-attention.

---

## 3. Analýza Datového Toku — Kódové Chyby vs. Metodologické Volby

### 3.1 `data_splitting.py` — Výsledek: ČISTÝ

Funkce `split_by_documents()` je implementována korektně:
- Rozdělení probíhá na úrovni `document_id`
- Explicitně ověřuje, že žádný dokument není ve dvou setech najednou
- Stratifikace podle labelu je provedena správně

**Závěr:** Klasický document-level data leakage ZDE NENÍ.

### 3.2 `load_preprocess_data.py` — Výsledek: PROBLÉM v CLS

| Řádek | Problém | Závažnost |
|---|---|---|
| 240: `t_sentence['cls_embedding'] = cls_embedding` | OK — target dostane správný CLS | ✅ |
| 250: `p_sentence['cls_embedding'] = cls_embedding` | **context_prev dostane CLS z celého bloku** | ⚠️ |
| 262: `n_sentence['cls_embedding'] = cls_embedding` | **context_next dostane CLS z celého bloku** | ⚠️ |

**Přesná specifikace problému:**
Při použití scénáře **Robustness** (kde context sentences jdou do tréninku),
kontextové věty z **L1 dokumentů** (L1 target, ale kontext je label=0) nesou
CLS embedding zakódovaného L1 bloku. Model M1 se tak učí "normalitu" z vektorů,
které fakticky kódují přítomnost anomálie.

**Pro scénář Baseline** (pouze target věty) tento problém NEEXISTUJE — ale
CLS pak kóduje celý blok, ne jen target větu, což je metodologicky diskutabilní.

### 3.3 Notebooky M1/S2 a M2/S2 — Výsledek: KOREKTNÍ IMPLEMENTACE

Logika výběru dat je v noteboocích správná:
- M1 správně filtruje `X_train_L0 = X_train_full[y_train_full == 0]`
- Threshold je optimalizován na val setu, nikoliv test setu
- PCA je fitován pouze na train datech

---

## 4. Konkrétní Návrhy pro Verifikaci Hypotéz

### Test 1: Isolace CLS problému (pro H1)

**Cíl:** Potvrdit/vyvrátit, že sdílené CLS embeddingy škodí robustness scénáři.

**Postup:**
1. Přepočítat embeddingy pro context věty tak, aby CLS byl vypočten
   **samostatně** z každé věty (bez kontextu), nebo
2. Pro sentence-level experimenty vyloučit `cls_embedding` pro kontextové
   věty (vždy používat pouze `mean_embedding` pro context věty).
3. Spustit M1/S2 Robustness a porovnat výsledky.

**Predikce:** Pokud H1 platí, Robustness s opravenými CLS se výrazně zlepší.

### Test 2: Permutační Test Labelů (pro H2)

**Cíl:** Ověřit, zda výsledky jsou statisticky signifikantní nebo náhodné.

**Postup:**
1. Vzít testovací set (103 vzorků pro baseline).
2. Náhodně permutovat labels (shuffle y_test).
3. Spočítat F1 pro permutovaný test.
4. Opakovat 1000×, vytvořit distribuci "náhodného F1".
5. Porovnat skutečné F1 s touto distribucí (p-value).

**Interpretace:** Pokud skutečné F1 > 95. percentil permutační distribuce,
výsledky jsou statisticky signifikantní. Pokud ne, jde o náhodu.

### Test 3: Délková Analýza (pro H2)

**Cíl:** Ověřit, zda model nereaguje na délku věty místo obsahu.

**Postup:**
```python
df_test['num_tokens'] = df_test['text'].apply(lambda x: len(x.split()))
# Korelace délky s anomaly_score
correlation = df_test['num_tokens'].corr(df_test['anomaly_score'])
# Vizualizace
sns.boxplot(data=df_test, x='true_label', y='num_tokens')
```

**Predikce:** Pokud korelace > 0.3, délka je confounding faktor.
ReviewDataset.md říká, že průměrná délka L0 = 11.0 a L1 = 11.5 tokenů —
to by mělo být OK, ale ověřte i v processed datech.

### Test 4: Dummy Classifier Baseline (pro H3)

**Cíl:** Etablovat správnou dolní hranici výkonu.

**Postup:**
```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
print(classification_report(y_test, y_pred_dummy))
```

Toto je **povinné** zahrnout do každého reportu výsledků.

### Test 5: Ablace — Single Sentence vs. Context (pro H1 a H5)

**Cíl:** Ověřit, zda výsledky jsou způsobeny kontextovým encodingem nebo
samotnou target větou.

**Postup:**
1. Vytvořit novou verzi embeddingů, kde každá věta je enkódována **samostatně**
   (bez context_prev a context_next).
2. Porovnat výsledky M1/S2 a M2/S2 s původní kontextovou variantou.

**Predikce:** Pokud standalone enkódování dává podobné nebo lepší výsledky,
kontextový blok přidává spíše šum než informaci (zvláště pro CLS).

### Test 6: Lexikální Analýza — Co Model Detekuje?

**Cíl:** Ověřit, zda model detekuje LJMPNIK nebo jiný artefakt.

**Postup:**
1. Vzít TOP FP (false positives — model říká L1, ale je to L0) a TOP FN
   (false negatives — model říká L0, ale je to L1).
2. Ručně zkontrolovat text.
3. Spočítat průměrný "hodnotící skóre" vět pomocí lexikonu (SentiLex-PL/CZ
   nebo ručně anotovat 50 vzorků).

**Predikce:** FP věty budou mít vysoké lexikální "negativní" nebo "expresivní"
skóre i přes label=0. FN budou mít jemné anomálie (ironie, hovorové výrazy).

### Test 7: Bootstrap Confidence Intervals (pro H4)

**Cíl:** Odhadnout reálnou nejistotu výsledků.

**Postup:**
```python
from sklearn.utils import resample
bootstrap_f1 = []
for _ in range(1000):
    idx = resample(range(len(y_test)))
    m = evaluation.calculate_metrics(y_test[idx], y_pred[idx], scores[idx])
    bootstrap_f1.append(m['f1'])
ci_low, ci_high = np.percentile(bootstrap_f1, [2.5, 97.5])
print(f"F1 = {np.mean(bootstrap_f1):.4f} ± [{ci_low:.4f}, {ci_high:.4f}]")
```

S testovacím setem 103 vzorků očekávám 95% CI šířku cca ±0.08–0.12.

---

## 5. Rekapitulace: Co Je a Co NENÍ Data Leakage

| Fenomén | Typ | Existuje? | Závažnost |
|---|---|---|---|
| Document-level leakage (stejný dokument v train+test) | Klasický leakage | **NE** | — |
| CLS embedding kóduje L1 blok pro context věty v tréninku | Architektonická kontaminace | **ANO** (robustness) | Kritická |
| Silver věty v testu robustness (M1) | Záměrné rozšíření testu | **ANO** | OK (záměrné) |
| Threshold optimalizace na val setu | Správný postup | — | OK |
| Imbalanced class reportování (F1 pouze L1) | Metodologická volba | **ANO** | Střední |
| LLM "watermark" detekce místo LJMPNIK | Clever Hans / Confound | **Možné** | Střední |

---

## 6. Interpretace Výsledků pro Diplomovou Práci

### Co Výsledky Reálně Říkají

**M2/S2 Hybrid F1(L1)=0.967 říká:** SVM dokáže v 768-dimenzionálním CLS/Mean
embedding prostoru RobeCzech spolehlivě separovat věty s expresivními lexémy
(LJMPNIK) od neutrálních. Zda tato separace probíhá díky "porozumění" anomálii
nebo díky statistické odlišnosti LLM-generovaných expresivních vs. neutrálních
vět, je otevřená otázka.

**M1/S2 Baseline F1=0.78 říká:** Mahalanobis/IF/OCSVM natrénovaný na 136
neutrálních větách dokáže detekovat anomálie s F1≈0.78. S 41 vzorky na val
setu je ale CI tohoto čísla velmi široký.

**M1/S2 Robustness F1=0.35 říká:** Přidání kontextových vět do tréninku
škodí — nejpravděpodobnější příčina je sdílený CLS embedding (H1).

### Doporučená Struktura Závěru v Diplomce

Výsledky prezentovat v pořadí:
1. Dummy baseline (F1 majority classifier)
2. Token-level výsledky (referenční bod — bez CLS problému)
3. Sentence-level výsledky s diskusí:
   - Uvedení CLS architektonické limitace
   - Diskuse o tom, zda model detekuje LJMPNIK nebo "expresivní styl"
   - Bootstrap CI pro všechny klíčové hodnoty

---

## 7. Prioritizace Dalších Kroků

| Priorita | Akce | Odhadovaná práce |
|---|---|---|
| 🔴 P1 | Spustit Test 4 (Dummy Baseline) — okamžitě | 10 min |
| 🔴 P1 | Opravit CLS pro context věty v load_preprocess_data.py (nebo excludovat) | 2–4 hod |
| 🟡 P2 | Spustit Test 2 (Permutační test) | 30 min |
| 🟡 P2 | Spustit Test 7 (Bootstrap CI) | 30 min |
| 🟢 P3 | Délková analýza (Test 3) | 20 min |
| 🟢 P3 | Lexikální analýza FP/FN (Test 6) | 1–2 hod |
| 🔵 P4 | Standalone embeddingy (Test 5) | 3–5 hod |

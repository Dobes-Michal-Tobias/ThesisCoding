
---

# Report: Tvorba a Charakteristika Datasetu

**Cíl tvorby datasetu:** Vytvořit robustní, anotovaný korpus vět simulujících zpravodajský styl, který obsahuje kontrolované příklady jak neutrálních vět, tak vět s lexikálními anomáliemi (LJMPNIK - Lexikální jednotky mající potenciál narušit informační kvalitu). Dataset musí být dostatečně variabilní, aby umožnil testování supervised i unsupervised algoritmů.

---

## 1. Teoretické Východisko a Definice LJMPNIK

Návrh datasetu vychází přímo z lingvistické a žurnalistické teorie, která definuje objektivitu ve zpravodajství.

* **Definice LJMPNIK:** Jedná se o slova (lexémy), která do textu vnášejí jiné funkce než čistě informativní. Typicky jde o slova s **hodnotícím** (např. *katastrofální, geniální*), **emotivním** (např. *šokující, dojemný*) nebo **expresivním** (např. *fiasko, tragédie*) nábojem.
* **Předpoklad:** Přítomnost LJMPNIK narušuje neutralitu zpravodajské věty a posouvá její sémantiku směrem k subjektivitě. Úkolem modelu je tuto odchylku detekovat.
* **Kontextuální povaha:** Slovo samo o sobě nemusí být vždy "zlé" (např. *útok* ve sportu vs. ve válce). Jeho anomálnost se projevuje až v kontextu věty a tématu. Proto dataset pracuje s celými větnými bloky.

---

## 2. Metodika Generování (Syntetická data)

Vzhledem k absenci velkého anotovaného korpusu českých "manipulativních vět" byl zvolen přístup **generování syntetických dat** pomocí velkých jazykových modelů (LLM). Tento přístup umožňuje precizní kontrolu nad obsahem a strukturou dat.

### 2.1. Struktura Datového Bodu
Každý datový bod v datasetu (jeden řádek v JSONL) představuje **trojici vět** (kontextový blok):

1.  **`context_prev`**: Předchozí věta. Vždy striktně **neutrální/faktická**.
2.  **`target_sentence`**: Cílová věta.
    * U třídy **Label 0**: Striktně **neutrální**.
    * U třídy **Label 1**: Obsahuje právě jednu **LJMPNIK** (anomálii).
3.  **`context_next`**: Následující věta. Vždy striktně **neutrální/faktická**.

**Výhoda této struktury:**
* Umožňuje modelu vidět slovo v širším kontextu.
* Poskytuje garantovaný zdroj "čistých" dat (kontextové věty) pro trénování unsupervised metod (viz Metoda S3).

### 2.2. Prompt Engineering (Instrukce pro LLM)
Pro zajištění kvality byly LLM instruovány následovně:
* **Role:** "Jsi nestranný editor zpravodajství..."
* **Omezení:** Věty musí být faktické, stručné (10-25 slov) a týkat se běžných zpravodajských témat (politika, ekonomika, společnost).
* **Pro Label 1:** "Vlož do cílové věty jedno silně hodnotící slovo (např. *skandální, absurdní*), které narušuje objektivitu, ale zbytek věty nech formální."

### 2.3. Zdroje Dat
Dataset kombinuje tři zdroje pro zajištění variability:
1.  **LLM (Syntetické):** Většina datasetu. Zajišťuje konzistentní strukturu a jasné anomálie.
2.  **Author (Ruční):** Věty psané autorem práce pro kontrolu specifických lingvistických jevů.
3.  **Real (Reálné zpravodajství):** Vybrané věty z médií, anotované jako neutrální nebo manipulativní (pro ověření "reality check").

---

## 3. Statistika a Vlastnosti Datasetu

**Celkový rozsah:** 520 datových bodů (bloků).

### 3.1. Rozdělení podle Tříd (Labelů)
* **Label 1 (LJMPNIK):** 332 vzorků (63.85 %)
    * *Účel:* Dostatek příkladů anomálií pro supervised učení.
* **Label 0 (Neutrální):** 188 vzorků (36.15 %)
    * *Poznámka:* Ačkoliv je neutrálních cílových vět méně, díky kontextovým větám (`context_prev/next`) máme ve skutečnosti **přes 1000 neutrálních vět** pro unsupervised trénink (Data Augmentation).

### 3.2. Délky Vět
Analýza délek (počet tokenů) potvrdila, že data jsou technicky konzistentní a model se nemůže "učit podle délky":
* **Průměrná délka věty:** ~11 tokenů.
* **Rozptyl:** Neutrální i anomální věty mají velmi podobnou distribuci délek (Mean 11.0 vs 11.5). Neexistuje zde bias (např. že by anomální věty byly podezřele dlouhé).

### 3.3. Lingvistická Variabilita (POS Tagy)
Analýza slovních druhů (POS) u cílových anomálií (LJMPNIK) ukázala očekávané rozložení:
* Dominují **Adjektiva (ADJ)** (např. *katastrofální*) a **Adverbia (ADV)** (např. *údajně*).
* Méně častá jsou **Substantiva (NOUN)** (*fiasko*) a **Slovesa (VERB)**.
* Toto rozložení odpovídá teorii, že hodnocení se nejčastěji vyjadřuje přívlastky a příslovečným určením.

---

## 4. Zpracování Dat (Preprocessing Pipeline)

Surová data (text) byla převedena do strojové podoby v několika krocích:

1.  **Tokenizace a Tagging:** Použit nástroj **UDPipe** (via `spacy-udpipe`) s modelem `cs-pdt` pro češtinu. Každému slovu byl přiřazen POS tag (Slovní druh) a Lemma.
2.  **Vektorizace (Embeddingy):**
    * Model: **`ufal/robeczech-base`** (State-of-the-art BERT model pro češtinu).
    * Postup: Celý trojvětný blok byl poslán do modelu najednou.
    * Výstup: Kontextualizovaný vektor (768 dimenzí) pro každý token. Díky Self-Attention mechanismu v BERTu vektor každého slova již obsahuje informaci o svém kontextu.
3.  **Filtrace (Noise Reduction):**
    * Pro metody M1 i M2 byla aplikována **agresivní filtrace**: Odstranění funkčních slov (předložky, spojky, interpunkce), která nenesou lexikální význam a v prostoru embeddingů tvoří šum. Ponechána byla pouze obsahová slova (NOUN, ADJ, VERB, ADV) a samotná LJMPNIK.

---

## 5. Závěr k Datasetu

Vytvořený dataset představuje **vysoce kvalitní, kontrolované prostředí** pro experimenty.
* **Silné stránky:** Jasná anotace, izolace zkoumaného jevu (LJMPNIK), dostatek dat pro unsupervised metody (díky kontextu).
* **Limitace:** Syntetický původ může postrádat některé nuance "špinavých" reálných dat, což je ale pro ověření základní hypotézy (Proof of Concept) spíše výhodou.
* **Využití:** Dataset plně postačuje pro validaci metod M1 (Unsupervised Outlier Detection) i M2 (Supervised Classification).

---
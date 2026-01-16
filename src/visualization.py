"""
Modul pro vizualizaci výsledků.
Obsahuje funkce pro P-R křivky, ladění thresholdu, histogramy, matice záměn a projekce embeddingů.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import matplotlib.patches as mpatches
import matplotlib.colors as mc
import colorsys
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

# Import konfigurace
import config

def setup_style():
    """Aplikuje globální nastavení stylu."""
    sns.set_theme(
        style=config.SNS_STYLE, 
        context=config.SNS_CONTEXT, 
        font_scale=config.FONT_SCALE
    )
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[config.COLORS['l0'], config.COLORS['l1']])
    print(f"🎨 Vizualizační styl nastaven: {config.SNS_STYLE}")

# --- A. METRIKY A VÝKONNOST ---

def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve"):
    """
    Vykreslí klasickou P-R křivku (Recall na ose X, Precision na ose Y).
    Slouží pro ukázku celkové kvality modelu (AUPRC).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=recall, y=precision, linewidth=3, color=config.COLORS['l0'])
    plt.fill_between(recall, precision, alpha=0.2, color=config.COLORS['l0'])
    
    plt.title(f"{title} (AUPRC = {auprc:.3f})", pad=15)
    plt.xlabel("Recall (Záchyt LJMPNIK)")
    plt.ylabel("Precision (Přesnost detekce)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

def plot_threshold_tuning(y_true, y_scores, title="Optimalizace prahu (Threshold Tuning)"):
    """
    Vykreslí vývoj metrik (Precision, Recall, F1) v závislosti na Thresholdu.
    Toto je ideální graf pro volbu pracovního bodu modelu.
    Vrací: best_threshold, best_f1
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Výpočet F1
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    # Sklearn vrací prec/rec o 1 delší než threshold (poslední hodnota je pro th=inf), musíme zkrátit
    # Abychom mohli plotovat, zahodíme poslední hodnotu prec/rec/f1
    f1_scores = f1_scores[:-1]
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Najít nejlepší
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # 1. Příprava dat pro Seaborn (Long format)
    df_metrics = pd.DataFrame({
        'Threshold': thresholds,
        'F1 Score': f1_scores,
        'Precision': precisions,
        'Recall': recalls
    })
    df_long = df_metrics.melt(id_vars='Threshold', var_name='Metric', value_name='Score')

    # 2. Vykreslení
    plt.figure(figsize=(12, 6))
    
    # Použijeme barvy z configu: Precision=Modrá(L0), Recall=Šedá/Jiná, F1=Červená(L1) nebo podobně
    # Ale pro jednoduchost a čitelnost použijeme fixní paletu 'pastel' nebo custom mapping
    custom_palette = {'F1 Score': config.COLORS['l1'], 'Precision': config.COLORS['l0'], 'Recall': '#95a5a6'}

    sns.lineplot(
        data=df_long,
        x='Threshold', y='Score', hue='Metric', style='Metric',
        palette=custom_palette,
        linewidth=3,
        dashes={'F1 Score': (None, None), 'Precision': (3, 3), 'Recall': (1, 1)}
    )

    # 3. Vertikální čára
    plt.axvline(x=best_th, color=config.COLORS['l1'], linestyle='-', linewidth=2, alpha=0.6)
    
    # Popisek
    plt.text(
        best_th, 0.5, 
        f'  Best Th: {best_th:.2f}\n  Max F1: {best_f1:.2f}', 
        color=config.COLORS['l1'], 
        fontweight='bold',
        verticalalignment='center'
    )

    # 4. Kosmetika
    plt.title(title, fontsize=15, pad=15)
    plt.xlabel("Threshold (Decision Score)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    sns.despine()
    plt.legend(title='Metrika', loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    plt.tight_layout()
    plt.show()
    
    return best_th, best_f1

def plot_anomaly_histogram(y_true, y_scores, threshold=None, title="Rozložení skóre anomálie"):
    """
    Vykreslí histogram skóre rozdělený podle skutečné třídy.
    """
    df_scores = pd.DataFrame({'score': y_scores, 'label': y_true})
    df_scores['label_name'] = df_scores['label'].map({0: 'Neutral (L0)', 1: 'LJMPNIK (L1)'})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_scores, x='score', hue='label_name', 
        element="step", stat="density", common_norm=False, bins=50,
        palette=[config.COLORS['l0'], config.COLORS['l1']]
    )
    
    if threshold is not None:
        plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.legend()
        
    plt.title(title, pad=15)
    plt.xlabel("Anomaly Score (Vyšší = Podezřelejší)")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, title="Confusion Matrix"):
    """
    Vykreslí matici záměn jako heatmapu.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    
    plt.title(title, pad=15)
    plt.xlabel("Predikce modelu")
    plt.ylabel("Skutečnost")
    plt.xticks([0.5, 1.5], ['Neutral', 'LJMPNIK'])
    plt.yticks([0.5, 1.5], ['Neutral', 'LJMPNIK'])
    plt.tight_layout()
    plt.show()

# --- B. EMBEDDINGS & PROJEKCE ---

def compute_projections(X, methods=['PCA', 't-SNE', 'UMAP'], max_samples=3000, random_state=42):
    """
    Vypočítá 2D projekce pro zadané metody.
    Vrací slovník: {'PCA': coords, 't-SNE': coords, ...}
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap.umap_ as umap
    except ImportError:
        umap = None
        if 'UMAP' in methods:
            print("⚠️ UMAP není nainstalován, přeskakuji.")
            methods = [m for m in methods if m != 'UMAP']

    # Subsampling pro rychlost
    if len(X) > max_samples:
        print(f"⚠️ Dataset je velký ({len(X)}). Vzorkuji na {max_samples} bodů.")
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_viz = X[indices]
        sampled_indices = indices
    else:
        X_viz = X
        sampled_indices = np.arange(len(X))

    projections = {}
    
    if 'PCA' in methods:
        print("1️⃣ Počítám PCA...")
        projections['PCA'] = PCA(n_components=2).fit_transform(X_viz)
        
    if 't-SNE' in methods:
        print("2️⃣ Počítám t-SNE...")
        projections['t-SNE'] = TSNE(n_components=2, perplexity=30, random_state=random_state, init='pca', learning_rate='auto').fit_transform(X_viz)
        
    if 'UMAP' in methods and umap:
        print("3️⃣ Počítám UMAP...")
        projections['UMAP'] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state).fit_transform(X_viz)
        
    return projections, sampled_indices

def plot_embedding_projection(coords, labels, palette, title, hue_order=None, alpha=0.7):
    """
    Obecná funkce pro vykreslení jednoho scatter plotu (např. UMAP).
    """
    plt.figure(figsize=(10, 8))
    
    sizes = {k: 40 for k in palette.keys()} 
    if 'TN' in palette: sizes['TN'] = 15 
    if 'Neutral' in palette: sizes['Neutral'] = 15
    
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1], 
        hue=labels, hue_order=hue_order,
        palette=palette, alpha=alpha, s=25, legend='full'
    )
    
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title=None, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

# --- C. SUPERVISED VIZUALIZACE ---

def plot_supervised_results(df_results, metric='auprc', title="Srovnání modelů"):       # Nepouívaná funkce
    """
    Vykreslí barplot porovnávající Train a Test skóre pro různé modely a filtry.
    
    Args:
        df_results: DataFrame se sloupci ['model', 'filter', 'train_score', 'test_score']
        metric: Název metriky pro popisek osy Y
    """
    # 1. Melt Data (převod na long format pro Seaborn: sloupec 'Dataset' = Train/Test)
    # Předpokládáme, že v df_results jsou sloupce 'train_auprc', 'test_auprc' (nebo jiné metriky)
    # Najdeme názvy sloupců odpovídající metrice
    col_train = f"train_{metric}"
    col_test = f"test_{metric}"
    
    if col_train not in df_results.columns:
        # Fallback, pokud se sloupce jmenují jinak (např. jen 'train_score')
        col_train = 'train_score'
        col_test = 'test_score'

    df_melt = df_results.melt(
        id_vars=['model', 'filter'], 
        value_vars=[col_train, col_test],
        var_name='dataset_type', 
        value_name='score'
    )
    
    # Přejmenování hodnot pro legendu (train_auprc -> Train)
    df_melt['Dataset'] = df_melt['dataset_type'].apply(lambda x: 'Train' if 'train' in x else 'Test')

    # 2. Vykreslení (FacetGrid: Sloupce=Modely)
    # Tím získáme přehled: Každý model má svůj graf, uvnitř vidíme vliv filtru
    g = sns.catplot(
        data=df_melt, 
        kind="bar",
        x="filter", 
        y="score", 
        hue="Dataset",
        col="model", 
        col_wrap=3, # Maximálně 3 grafy vedle sebe
        palette={'Train': config.COLORS['train'], 'Test': config.COLORS['test']},
        height=4, 
        aspect=0.8,
        sharey=True
    )
    
    g.fig.suptitle(title, y=1.02, fontsize=16)
    g.set_axis_labels("Filter Strategy", f"Score ({metric.upper()})")
    g.set_titles("{col_name}")
    
    # Přidání hodnot nad sloupce
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
            
    plt.show()

    # ... (existující importy)

def _prepare_long_data(df_results, metric='auprc'):
    """
    Pomocná funkce: Převede DataFrame z wide (sloupce train_x, test_x) 
    do long formátu (sloupec 'score' a 'dataset_type').
    """
    # Zjištění správných názvů sloupců
    col_test = f'test_{metric}'
    col_train = f'train_{metric}'
    
    if col_test not in df_results.columns:
        raise ValueError(f"Dataframe neobsahuje sloupec {col_test}")

    # Melt (převedení na řádky)
    df_long = df_results.melt(
        id_vars=['scenario', 'filter', 'model'],
        value_vars=[col_train, col_test],
        var_name='dataset_col',
        value_name='score'
    )
    
    # Vytvoření hezčích popisků (train_auprc -> Train)
    df_long['Dataset'] = df_long['dataset_col'].apply(
        lambda x: 'Train' if 'train' in x else 'Test'
    )
    
    return df_long

def plot_scenario_breakdown(df_results, metric='auprc', save_dir=None):
    """
    Vykreslí samostatný graf pro každý scénář.
    Porovnává Modely (osa X) a jejich Train vs Test skóre.
    """
    scenarios = df_results['scenario'].unique()
    df_long = _prepare_long_data(df_results, metric)

    # Barvy: Train = Šedá (pozadí), Test = Barevná (podle Configu nebo jednotná)
    # Zde použijeme klasické rozlišení: Train (Světlá), Test (Tmavá)
    palette = {"Train": "#b0bec5", "Test": "#2c3e50"} # Neutrální šedá vs Tmavě modrá

    for scen in scenarios:
        data_scen = df_long[df_long['scenario'] == scen]
        
        g = sns.catplot(
            data=data_scen,
            kind="bar",
            x="model",
            y="score",
            hue="Dataset",
            col="filter", # Filtry vedle sebe
            palette=palette,
            height=4,
            aspect=1,
            sharey=True
        )
        
        # Titulky a popisky
        g.fig.suptitle(f"Scénář: {scen} (Train vs Test - {metric.upper()})", y=1.05, fontsize=16, weight='bold')
        g.set_axis_labels("", f"{metric.upper()} Score")
        g.set_titles("{col_name}")
        
        # Přidání hodnot nad sloupce
        for ax in g.axes.flat:
            # Grid pro lepší čitelnost
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)

        if save_dir:
            plt.savefig(save_dir / f"breakdown_{scen}_{metric}.png", bbox_inches='tight', dpi=150)
            plt.close() # Zavřít, ať se nehromadí v paměti
        else:
            plt.show()

# --- 2. GLOBÁLNÍ SROVNÁNÍ SCÉNÁŘŮ ---

def plot_global_comparison(df_results, metric='auprc'):
    """
    Vykreslí vše v jednom. 
    Osa X = Scénáře. 
    Barvy = Scénáře (Sytá = Test, Světlá = Train).
    Facet = Model (každý model má svůj graf).
    """
    df_long = _prepare_long_data(df_results, metric)
    
    # --- Magie s barvami ---
    # Chceme, aby S1a_Test byla Červená a S1a_Train byla Světle Červená.
    # Musíme vytvořit custom paletu pro kombinaci (Scenario, Dataset)
    
    # 1. Získáme základní barvy z configu
    base_colors = config.SCENARIO_COLORS
    
    # 2. Vytvoříme paletu pro Hue (což bude 'Dataset')
    # Ale pozor, Seaborn neumí jednoduše "Hue=Scenario+Dataset".
    # Trik: Obarvíme to ručně nebo použijeme 'hue=scenario' a 'style=dataset' (pattern).
    # Nejlepší vizuální varianta pro "vedle sebe": 
    # Vytvoříme sloupec 'Hue_Label' = "S1a (Train)" / "S1a (Test)"
    
    df_long['Scenario_Type'] = df_long['scenario'] + " (" + df_long['Dataset'] + ")"
    
    # Generování palety
    final_palette = {}
    ordered_hue = [] # Abychom zachovali pořadí v legendě
    
    def lighten_color(color, amount=0.5):
        """
        Zesvětlí barvu o dané množství (amount).
        """
        try:
            c = mc.to_rgb(color)
            # Použití colorsys pro převod na HLS
            c = colorsys.rgb_to_hls(*c)
            
            # Zvýšíme světlost (L je na indexu 1), max 1.0
            new_l = min(1.0, c[1] + (1.0 - c[1]) * amount)
            
            # Zpět na RGB
            return colorsys.hls_to_rgb(c[0], new_l, c[2])
        except ValueError:
            return color
            
    scenarios = sorted(df_results['scenario'].unique())
    
    for scen in scenarios:
        base = base_colors.get(scen, "#333333") # Fallback černá
        
        # Test = Sytá (Originál)
        label_test = f"{scen} (Test)"
        final_palette[label_test] = base
        
        # Train = Světlá (Vybledlá)
        label_train = f"{scen} (Train)"
        final_palette[label_train] = lighten_color(base, amount=0.6) # O 60% světlejší
        
        # Pořadí pro legendu: Train, pak Test (nebo naopak)
        ordered_hue.append(label_train)
        ordered_hue.append(label_test)

    # Vykreslení
    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="scenario",
        y="score",
        hue="Scenario_Type", # Tím docílíme barev podle scénářů + odstínů
        col="model",         # Každý model zvlášť
        col_wrap=2,          # Po 2 modelech na řádek
        palette=final_palette,
        hue_order=ordered_hue, # Vynutíme pořadí barev
        height=3.5,
        aspect=1.5,
        sharey=True,
        dodge=True           # Sloupce vedle sebe
    )
    
    g.fig.suptitle(f"Globální srovnání scénářů ({metric.upper()})", y=1.02, fontsize=16)
    g.set_axis_labels("Scénář", f"{metric.upper()}")
    g.set_titles("{col_name}")
    
    # Legenda je teď obrovská, zkusíme ji vyčistit nebo přesunout
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.05), ncol=len(scenarios), title=None)
    
    # Přidání gap (mezery) mezi Train/Test vizuálně? 
    # Seaborn to dává hned vedle sebe. Barvy to odliší.
    
    plt.show()

# --- NOVÉ FCE ---
from sklearn.calibration import calibration_curve

def plot_calibration_curve(y_true, y_probs, title="Calibration Curve (Reliability Diagram)"):
    """
    Vykreslí kalibrační křivku.
    Ideální model (Perfectly Calibrated) by měl jít po diagonále.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model', color=config.COLORS['l1'])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.title(title, pad=15)
    plt.xlabel("Průměrná predikovaná pravděpodobnost")
    plt.ylabel("Podíl pozitivních tříd (Fraction of Positives)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_error_categories(y_true, y_pred):
    """
    Vrátí seznam kategorií (TP, FP, TN, FN) pro každý bod.
    """
    categories = []
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            categories.append("TP (Correct Anomaly)")
        elif true == 0 and pred == 0:
            categories.append("TN (Correct Neutral)")
        elif true == 0 and pred == 1:
            categories.append("FP (False Alarm)")
        elif true == 1 and pred == 0:
            categories.append("FN (Missed Anomaly)")
    return np.array(categories)

def plot_model_calibration(y_true, y_probs, title="Kalibrační křivka"):
    """
    Vykreslí Reliability Diagram.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model', color=config.COLORS['l1'])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.title(title)
    plt.xlabel("Průměrná predikovaná pravděpodobnost")
    plt.ylabel("Skutečný podíl pozitivních (Fraction of Positives)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, top_n=20):
    """
    Vykreslí důležitost rysů (pokud to model podporuje).
    Pozor: U embeddingů (768 dimenzí) to ukazuje důležitost dimenzí, ne slov.
    """
    importances = None
    
    # 1. Tree-based modely (RandomForest, XGBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
    # 2. Lineární modely (LogisticRegression, SVM linear)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        # Seřadíme indexy
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 5))
        plt.title(f"Top {top_n} Feature Importances (Dimensions)")
        plt.bar(range(top_n), importances[indices], color=config.COLORS['l1'], align="center")
        plt.xticks(range(top_n), indices, rotation=45)
        plt.xlim([-1, top_n])
        plt.xlabel("Index dimenze embeddingu")
        plt.ylabel("Důležitost")
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Model {type(model).__name__} nepodporuje feature importance nebo není natrénovaný.")

def plot_error_analysis_projection(coords, y_true, y_pred, method_name="Projection"):
    """
    Wrapper, který automaticky vypočítá kategorie chyb (TP, FP, TN, FN)
    a vykreslí je do projekce.
    """
    # 1. Určení kategorií
    categories = []
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: categories.append('TP (Correct Anomaly)')
        elif t == 0 and p == 0: categories.append('TN (Correct Neutral)')
        elif t == 0 and p == 1: categories.append('FP (False Alarm)')
        elif t == 1 and p == 0: categories.append('FN (Missed Anomaly)')
    
    # 2. Barvy (musí odpovídat klíčům výše)
    palette_err = {
        'TP (Correct Anomaly)': config.COLORS['TP'],
        'TN (Correct Neutral)': config.COLORS['TN'],
        'FP (False Alarm)':     config.COLORS['FP'],
        'FN (Missed Anomaly)':  config.COLORS['FN']
    }
    
    # 3. Pořadí pro legendu (aby FP a FN byly vidět)
    order = ['TN (Correct Neutral)', 'TP (Correct Anomaly)', 'FP (False Alarm)', 'FN (Missed Anomaly)']
    
    # 4. Vykreslení pomocí existující funkce
    plot_embedding_projection(
        coords, 
        labels=categories, 
        palette=palette_err, 
        title=f"{method_name} - Error Analysis", 
        hue_order=order,
        alpha=0.7
    )

def plot_bootstrap_results(bootstrap_scores, metric_name="F1 Score", title="Stabilita modelu (Bootstrap)"):
    """
    Vykreslí distribuci výsledků z bootstrapování (KDE plot + statistiky).
    """
    plt.figure(figsize=(10, 6))
    
    stats_text = []
    
    # Iterujeme přes modely
    for model_name, scores in bootstrap_scores.items():
        if len(scores) == 0: continue
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        # 95% Confidence Interval
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)
        
        # Label do legendy
        label = f"{model_name}\nμ={mean_score:.3f}, σ={std_score:.3f}"
        
        # KDE Plot (Hustota)
        sns.kdeplot(scores, label=label, fill=True, alpha=0.3)
        
        # Uložení textu pro výpis
        stats_text.append(f"🔹 {model_name}: Mean={mean_score:.4f} | Std={std_score:.4f} | 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(metric_name)
    plt.ylabel("Hustota pravděpodobnosti")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legenda bokem, aby nezavazela
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Textový výpis pod graf
    print("\n📊 STATISTICKÉ VÝSLEDKY:")
    for line in stats_text:
        print(line)

def _prepare_long_data_s2(df_results, metric='auprc'):
    """
    Pomocná funkce pro S2 data: Meltuje Train/Test sloupce, zachovává 'pooling'.
    """
    col_test = f'test_{metric}'
    col_train = f'train_{metric}'
    
    # Ošetření chybějících sloupců
    if col_test not in df_results.columns:
        print(f"⚠️ Varování: Sloupec {col_test} nenalezen. Dostupné: {df_results.columns}")
        return pd.DataFrame()

    df_long = df_results.melt(
        id_vars=['scenario', 'pooling', 'model'],
        value_vars=[col_train, col_test],
        var_name='dataset_col',
        value_name='score'
    )
    
    df_long['Dataset'] = df_long['dataset_col'].apply(
        lambda x: 'Train' if 'train' in x else 'Test'
    )
    return df_long

def plot_pooling_breakdown(df_results, metric='auprc'):
    """
    Vykreslí breakdown graf pro S2 experimenty.
    Osa X: Modely
    Sloupce (Facet): Pooling (Mean vs CLS)
    Barvy (Hue): Train vs Test
    Generuje zvlášť graf pro Baseline a Robustness.
    """
    scenarios = df_results['scenario'].unique()
    df_long = _prepare_long_data_s2(df_results, metric)
    
    # Barvy: Train (Světlý), Test (Tmavý)
    palette = {"Train": "#b0bec5", "Test": "#2c3e50"} 

    for scen in scenarios:
        data_scen = df_long[df_long['scenario'] == scen]
        
        if data_scen.empty:
            continue
            
        g = sns.catplot(
            data=data_scen,
            kind="bar",
            x="model",
            y="score",
            hue="Dataset",
            col="pooling", # ZDE JE ZMĚNA: Iterujeme přes pooling
            palette=palette,
            height=5,
            aspect=1.2,
            sharey=True
        )
        
        # Titulky a popisky
        g.fig.suptitle(f"Scénář: {scen.upper()} (Train vs Test - {metric.upper()})", y=1.05, fontsize=16, weight='bold')
        g.set_axis_labels("", f"{metric.upper()} Score")
        g.set_titles("Pooling: {col_name}")
        
        # Přidání hodnot nad sloupce
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') # Natočení popisků modelů
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)

        plt.show()
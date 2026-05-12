"""
Script to illustrate the phenomenon described in Chapter 5.2.2:
Anomalous lexemes (LJMPNIK) concentration in full-meaning parts of speech
and syntactic noise created by functional words in the embedding space.

Author: Gemini CLI (NLP Expert)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Ensure we can import from the current directory
sys.path.append(str(Path(__file__).parent))

import config
import visualization
import load_preprocess_data

def generate_chapter_5_illustration():
    """
    Generates 2D projection and Dendrogram for a specific sentence to illustrate
    the embedding space distribution of LJMPNIK vs. functional words.
    """
    # 1. Setup Style
    visualization.setup_style()
    
    # 2. Example Sentence
    # Sentence containing an adherent expressive/pejorative 'prasata' (LJMPNIK)
    sentence = (
        "Na jednání schůze Poslanecké sněmovny se poslanci chovali jako prasata, "
        "vládní koalice zákon neprosadila, pokračovat se bude muset zítra, "
        "až se vládní představitelé dohodnou s opozičními lídry na zmírnění obstrukcí."
    )
    target_word = "prasata"
    
    print(f"📝 Processing sentence: {sentence}")
    
    # 3. Initialize Models
    load_preprocess_data.initialize_models()
    
    # 4. Process Sentence
    # We use extract_spacy_info to get tokens and POS tags
    words, pos_tags, lemmas = load_preprocess_data.extract_spacy_info(sentence)
    
    # Get contextual embeddings
    # We pass empty prev and next context for this illustration
    _, emb_target, _, _ = load_preprocess_data.get_contextual_embeddings([], words, [])
    embeddings = np.array(emb_target)
    
    # 5. Categorize and Color Tokens
    # LJMPNIK: Coral (#FC8D62)
    # Functional: Light Gray (#D9D9D9)
    # Other: Teal (#66C2A5)
    
    categories = []
    colors = []
    
    functional_pos = config.POS_FORBIDDEN_MILD
    
    for word, pos in zip(words, pos_tags):
        if word.lower().strip(",.") == target_word:
            categories.append("LJMPNIK")
            colors.append(config.COLORS['FP']) # Coral
        elif pos in functional_pos:
            categories.append("Funkční slova (Šum)")
            colors.append(config.COLORS['TN']) # Light Gray
        else:
            categories.append("Plnovýznamová slova")
            colors.append(config.COLORS['l0']) # Teal
            
    # 6. Dimensionality Reduction (PCA)
    print("📉 Computing PCA projection...")
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    coords = pca.fit_transform(embeddings)
    
    # 7. Create 2D Plot
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with categories
    df_plot = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'word': words,
        'category': categories
    })
    
    palette = {
        "LJMPNIK": config.COLORS['FP'],
        "Funkční slova (Šum)": config.COLORS['TN'],
        "Plnovýznamová slova": config.COLORS['l0']
    }
    
    sns.scatterplot(
        data=df_plot, x='x', y='y', hue='category', 
        palette=palette, s=100, alpha=0.8, edgecolor='w'
    )
    
    # Annotate points with words
    for i, txt in enumerate(words):
        plt.annotate(
            txt, (coords[i, 0], coords[i, 1]), 
            xytext=(5, 5), textcoords='offset points',
            fontsize=9, alpha=0.8
        )
        
    plt.title("2D Projekce embeddingů (PCA): LJMPNIK vs. Funkční šum", pad=20)
    plt.xlabel("První hlavní komponenta (PC1)")
    plt.ylabel("Druhá hlavní komponenta (PC2)")
    plt.legend(title="Kategorie lexémů", loc='upper right')
    
    save_path_pca = config.RESULTS_DIR / "chapter_5_embedding_projection.png"
    plt.savefig(save_path_pca, dpi=config.VIZ_CONFIG['dpi']['print'])
    print(f"✅ PCA plot saved to: {save_path_pca}")
    plt.close()
    
    # 8. Create Dendrogram (Hclust)
    print("🌳 Computing Hierarchical Clustering...")
    linked = linkage(embeddings, 'ward')
    
    plt.figure(figsize=(14, 8))
    dendrogram(
        linked,
        orientation='top',
        labels=words,
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_font_size=10,
        leaf_rotation=90
    )
    
    plt.title("Hierarchické shlukování lexémů (Dendrogram)", pad=20)
    plt.ylabel("Eukleidovská vzdálenost")
    
    save_path_dendro = config.RESULTS_DIR / "chapter_5_dendrogram.png"
    plt.savefig(save_path_dendro, dpi=config.VIZ_CONFIG['dpi']['print'])
    print(f"✅ Dendrogram saved to: {save_path_dendro}")
    plt.close()

if __name__ == "__main__":
    generate_chapter_5_illustration()

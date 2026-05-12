"""
Script for Chapter 5.2.2 — Embedding space visualization of LJMPNIK vs. functional words.

Produces six publication-ready figures for the thesis (PCA + dendrogram for each
of three illustrative sentences), showing that LJMPNIK tokens occupy a distinct
region in the embedding space while functional words form syntactic noise.

Uses RobeCzech contextual embeddings (via load_preprocess_data).
Colors follow the project-wide Set2-inspired config.

Author: Michal Tobiáš Dobeš
"""

import sys
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

# Ensure we can import project modules
sys.path.append(str(Path(__file__).parent))

import config
import visualization
import load_preprocess_data

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SENTENCES = [
    (
        "Na jednání schůze Poslanecké sněmovny se poslanci chovali jako prasata, "
        "vládní koalice zákon neprosadila, pokračovat se bude muset zítra, "
        "až se vládní představitelé dohodnou s opozičními lídry "
        "na zmírnění obstrukcí.",
        "prasata",
    ),
    (
        "V průběhu tohoto týdne se totiž v rámci oné diskuse o daních ukázalo, "
        "že se vládní představitelé k celému problému postavili alibisticky, "
        "ačkoli se původně předpokládalo, že k tomu dojde zcela jinak "
        "a přinesou již hotový zákon.",
        "alibisticky",
    ),
    (
        "Podle zprávy od ministerstva se sice v naší přenosové soustavě "
        "i nadále počítá s stabilitou, ale vlivem těchto změn se v ní "
        "brzy začne projevovat fenomén prosumpce, což by mohlo mít "
        "dle odborníků z ČVUT za následek to, že se skoro vše "
        "v budoucnu bude muset opravit.",
        "prosumpce",
    ),
]

# Category definitions matching config.POS_FORBIDDEN_MILD
FUNCTIONAL_POS = config.POS_FORBIDDEN_MILD

# Color assignment (from config.COLORS)
COLOR_LJMPNIK  = config.COLORS['FP']     # Coral   #FC8D62
COLOR_FUNCTION = config.COLORS['TN']     # Light gray  #D9D9D9
COLOR_CONTENT  = config.COLORS['l0']     # Teal    #66C2A5

CATEGORY_MAP = {
    'LJMPNIK':         COLOR_LJMPNIK,
    'Funkční slova':   COLOR_FUNCTION,
    'Plnovýznamová':   COLOR_CONTENT,
}

# ============================================================================
# HELPERS
# ============================================================================

def _warn_no_tokens(words, target_word):
    """Log a warning if expected content tokens are missing."""
    if target_word not in words:
        logger.warning("Target word '%s' not found in tokens!", target_word)


def _classify_tokens(words, pos_tags, target_word):
    """Return (categories, colors) arrays aligned 1:1 with words."""
    categories = []
    for word, pos in zip(words, pos_tags):
        if word.lower().strip(",.;:!?") == target_word.lower():
            categories.append('LJMPNIK')
        elif pos in FUNCTIONAL_POS:
            categories.append('Funkční slova')
        else:
            categories.append('Plnovýznamová')
    colors = [CATEGORY_MAP[c] for c in categories]
    return categories, colors


def _build_legend():
    """Return a list of matplotlib Patch objects for the figure legend."""
    return [
        Patch(facecolor=COLOR_LJMPNIK,  edgecolor='white',
              label='LJMPNIK'),
        Patch(facecolor=COLOR_FUNCTION, edgecolor='white',
              label='Funkční slova (šum)'),
        Patch(facecolor=COLOR_CONTENT,  edgecolor='white',
              label='Plnovýznamová slova'),
    ]


def _make_safe_name(target_word):
    """Sanitize target word for use in a filename."""
    return target_word.strip().lower().replace(' ', '_')


# ============================================================================
# FIGURE GENERATORS
# ============================================================================

def _plot_pca(X, words, categories, colors, target_word):
    """Generate PCA projection figure. Returns path to saved file."""
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['square'])
    ax.set_facecolor(config.VIZ_CONFIG['style']['bg_color'])

    # Scatter per category (to control legend order)
    for cat_name in ['Funkční slova', 'Plnovýznamová', 'LJMPNIK']:
        mask = np.array([c == cat_name for c in categories])
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CATEGORY_MAP[cat_name],
            label=cat_name,
            s=120 if cat_name == 'LJMPNIK' else 80,
            edgecolors='white',
            linewidth=0.6,
            alpha=0.85,
            zorder=3 if cat_name == 'LJMPNIK' else 2,
        )

    # Annotate every token
    for i, word in enumerate(words):
        ax.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=8,
            alpha=0.9,
            fontfamily='sans-serif',
        )

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f} % rozptylu)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f} % rozptylu)')
    ax.legend(
        handles=_build_legend(),
        loc='upper right',
        framealpha=0.9,
        fontsize=config.VIZ_CONFIG['font']['legend'],
    )
    fig.tight_layout()

    safe = _make_safe_name(target_word)
    save_path = config.RESULTS_DIR / f'chapter_5_LJMPNIK_{safe}_pca.png'
    fig.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'])
    plt.close(fig)
    logger.info("Saved PCA plot → %s", save_path)
    return save_path


def _plot_dendrogram(X, words, categories, colors, target_word):
    """Generate hierarchical clustering dendrogram. Returns path to saved file."""
    Z = linkage(X, method='ward', metric='euclidean')

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['wide'])

    dn = dendrogram(
        Z,
        labels=words,
        orientation='top',
        leaf_font_size=10,
        leaf_rotation=90,
        ax=ax,
        color_threshold=0,
    )

    # Colour the x-tick labels according to category
    tick_labels = ax.get_xticklabels()
    for tick in tick_labels:
        leaf_text = tick.get_text()
        try:
            orig_idx = words.index(leaf_text)
            tick.set_color(colors[orig_idx])
            tick.set_fontweight('bold' if categories[orig_idx] == 'LJMPNIK'
                                else 'normal')
            tick.set_fontsize(11 if categories[orig_idx] == 'LJMPNIK' else 9)
        except ValueError:
            pass

    ax.set_ylabel('Eukleidovská vzdálenost (Ward)')
    ax.legend(
        handles=_build_legend(),
        loc='upper right',
        framealpha=0.9,
        fontsize=config.VIZ_CONFIG['font']['legend'],
    )
    fig.tight_layout()

    safe = _make_safe_name(target_word)
    save_path = config.RESULTS_DIR / f'chapter_5_LJMPNIK_{safe}_dendrogram.png'
    fig.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'])
    plt.close(fig)
    logger.info("Saved dendrogram → %s", save_path)
    return save_path


# ============================================================================
# MAIN DEMO
# ============================================================================

def run_demo():
    """Entry point: process all sentences and save figures."""
    # Stage 0 — style
    visualization.setup_style()

    # Stage 1 — models
    logger.info("Initialising RobeCzech + UDPipe …")
    load_preprocess_data.initialize_models()

    all_paths = []

    for idx, (sentence, target_word) in enumerate(SENTENCES, start=1):
        logger.info("=" * 60)
        logger.info("Sentence %d/3 | LJMPNIK = '%s'", idx, target_word)
        logger.info("=" * 60)

        # Tokenise & tag
        words, pos_tags, _ = load_preprocess_data.extract_spacy_info(sentence)
        _warn_no_tokens(words, target_word)
        logger.info("  Tokens: %d", len(words))

        # Embeddings
        _, emb_target, _, _ = load_preprocess_data.get_contextual_embeddings(
            words_prev=[], words_target=words, words_next=[]
        )
        X = np.array(emb_target)
        logger.info("  Embedding matrix: %s", X.shape)

        # Classify
        categories, colors = _classify_tokens(words, pos_tags, target_word)
        n_ljmpnik = sum(1 for c in categories if c == 'LJMPNIK')
        n_func = sum(1 for c in categories if c == 'Funkční slova')
        n_content = sum(1 for c in categories if c == 'Plnovýznamová')
        logger.info("  Categories — LJMPNIK: %d, Funkční: %d, Plnovýznamová: %d",
                    n_ljmpnik, n_func, n_content)

        # Generate figures
        pca_path = _plot_pca(X, words, categories, colors, target_word)
        dendro_path = _plot_dendrogram(X, words, categories, colors, target_word)
        all_paths.extend([pca_path, dendro_path])

    # Summary
    logger.info("=" * 60)
    logger.info("✅ Demo finished — %d figures saved to %s",
                len(all_paths), config.RESULTS_DIR)
    for p in all_paths:
        logger.info("   • %s", p.name)
    logger.info("=" * 60)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s — %(levelname)s — %(message)s',
    )
    run_demo()
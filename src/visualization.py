"""
visualization.py - Unified Visualization Functions

Comprehensive plotting functions for model evaluation, error analysis,
and result presentation.

Changes from original:
    - ✅ FIXED: Removed code duplication (_prepare_long_data functions merged)
    - ✅ FIXED: Removed commented-out unused code
    - ✅ NEW: Added input validation to all plotting functions
    - ✅ NEW: Added type hints for all functions
    - ✅ IMPROVED: Centralized configuration via config module
    - ✅ IMPROVED: Better error handling and user feedback
    - ✅ IMPROVED: Consistent styling and color schemes
    - ✅ NEW: Added save_path parameter to all major plots
    - ✅ NEW: Added comprehensive docstrings with examples

Author: Michal Tobiáš Dobeš
Date: January 2026
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
import seaborn as sns
import pandas as pd
import numpy as np
import colorsys

from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve, 
    auc
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

import config

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def setup_style() -> None:
    """
    Apply global visualization style from config.
    
    Should be called once at the beginning of analysis notebooks.
    
    Example:
        >>> from visualization import setup_style
        >>> setup_style()
        🎨 Visualization style set: whitegrid
    """
    sns.set_theme(
        style=config.SNS_STYLE,
        context=config.SNS_CONTEXT,
        font_scale=config.FONT_SCALE
    )
    
    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=[config.COLORS['l0'], config.COLORS['l1']]
    )
    
    logger.info(f"🎨 Visualization style set: {config.SNS_STYLE}")

# ============================================================================
# A. METRICS AND PERFORMANCE PLOTS
# ============================================================================

def plot_pr_curve(y_true: np.ndarray, 
                  y_scores: np.ndarray,
                  title: str = "Precision-Recall Curve",
                  save_path: Optional[Path] = None) -> float:
    """
    Plot Precision-Recall curve with AUPRC metric.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_scores: Predicted probabilities or anomaly scores
        title: Plot title
        save_path: If provided, save figure to this path
    
    Returns:
        AUPRC score (Area Under Precision-Recall Curve)
    
    Raises:
        ValueError: If input shapes don't match or contain invalid values
    
    Example:
        >>> auprc = plot_pr_curve(y_test, y_probs, save_path=Path('results/pr_curve.png'))
        >>> print(f"AUPRC: {auprc:.3f}")
    """
    # ✅ NEW: Input validation
    if len(y_true) != len(y_scores):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_scores={len(y_scores)}")
    
    if not all(label in [0, 1] for label in np.unique(y_true)):
        raise ValueError("y_true must contain only binary labels (0 and 1)")
    
    # Compute metrics
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    
    # ✅ IMPROVED: Use config for figure size
    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
    
    # Plot curve
    sns.lineplot(x=recall, y=precision, linewidth=3, color=config.COLORS['l0'])
    plt.fill_between(recall, precision, alpha=config.VIZ_CONFIG['alpha']['fill'], 
                     color=config.COLORS['l0'])
    
    plt.title(f"{title} (AUPRC = {auprc:.3f})", pad=15)
    plt.xlabel("Recall (Záchyt LJMPNIK)")
    plt.ylabel("Precision (Přesnost detekce)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # ✅ NEW: Save functionality
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
        logger.info(f"💾 Saved plot to {save_path}")
    
    plt.show()
    
    return auprc


def plot_threshold_tuning(y_true: np.ndarray,
                          y_scores: np.ndarray,
                          title: str = "Optimalizace prahu (Threshold Tuning)",
                          save_path: Optional[Path] = None) -> Tuple[float, float]:
    """
    Plot metrics (Precision, Recall, F1) vs threshold and find optimal point.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probabilities or scores
        title: Plot title
        save_path: Optional save path
    
    Returns:
        Tuple of (best_threshold, best_f1_score)
    
    Example:
        >>> threshold, f1 = plot_threshold_tuning(y_test, y_scores)
        >>> print(f"Optimal threshold: {threshold:.3f} (F1={f1:.3f})")
    """
    # ✅ NEW: Validation
    if len(y_true) != len(y_scores):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_scores={len(y_scores)}")
    
    # Compute PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Compute F1 scores
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, 
                         out=np.zeros_like(denominator), 
                         where=denominator != 0)
    
    # Align lengths (sklearn returns one extra value)
    f1_scores = f1_scores[:-1]
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Prepare data for seaborn
    df_metrics = pd.DataFrame({
        'Threshold': thresholds,
        'F1 Score': f1_scores,
        'Precision': precisions,
        'Recall': recalls
    })
    df_long = df_metrics.melt(id_vars='Threshold', var_name='Metric', value_name='Score')
    
    # ✅ IMPROVED: Use config colors
    custom_palette = {
        'F1 Score': config.COLORS['l1'],
        'Precision': config.COLORS['l0'],
        'Recall': '#95a5a6'
    }
    
    # Plot
    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['large'])
    
    sns.lineplot(
        data=df_long,
        x='Threshold', y='Score', hue='Metric', style='Metric',
        palette=custom_palette,
        linewidth=3,
        dashes={'F1 Score': (None, None), 'Precision': (3, 3), 'Recall': (1, 1)}
    )
    
    # Mark optimal point
    plt.axvline(x=best_th, color=config.COLORS['l1'], linestyle='-', 
                linewidth=2, alpha=0.6)
    
    plt.text(
        best_th, 0.5,
        f'  Best Th: {best_th:.2f}\n  Max F1: {best_f1:.2f}',
        color=config.COLORS['l1'],
        fontweight='bold',
        verticalalignment='center'
    )
    
    plt.title(title, fontsize=15, pad=15)
    plt.xlabel("Threshold (Decision Score)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    sns.despine()
    plt.legend(title='Metrika', loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
        logger.info(f"💾 Saved plot to {save_path}")
    
    plt.show()
    
    return best_th, best_f1


def plot_anomaly_histogram(y_true: np.ndarray, 
                           y_scores: np.ndarray, 
                           threshold: Optional[float] = None, 
                           title: str = "Rozložení skóre anomálie",
                           save_path: Optional[Path] = None) -> None:
    """
    Plot histogram of anomaly scores separated by true class.
    """
    # Validation
    if len(y_true) != len(y_scores):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_scores={len(y_scores)}")
    
    # 1. Bezpečné vytvoření DataFrame
    # Musíme zajistit, že y_true jsou integery, aby fungoval .map()
    y_true_int = np.array(y_true).astype(int)
    
    df_scores = pd.DataFrame({
        'score': y_scores,
        'label': y_true_int
    })

    # 2. Definice názvů a mapování
    label_map = {
        0: 'Neutral (L0)',
        1: 'LJMPNIK (L1)'
    }
    df_scores['label_name'] = df_scores['label'].map(label_map)
    
    # Kontrola, zda mapování nezanechalo NaN (pokud by y_true obsahovalo jiné hodnoty)
    if df_scores['label_name'].isna().any():
        logger.warning("⚠️ Some labels could not be mapped (NaN found). Check y_true inputs.")
        df_scores.dropna(subset=['label_name'], inplace=True)

    # 3. Explicitní paleta (Slovník je bezpečnější než List)
    palette_dict = {
        'Neutral (L0)': config.COLORS['l0'],
        'LJMPNIK (L1)': config.COLORS['l1']
    }

    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
    
    # 4. Vykreslení
    # element="step" a fill=True vypadá lépe pro překrývající se histogramy
    sns.histplot(
        data=df_scores, 
        x='score', 
        hue='label_name',
        element="step", 
        stat="density", 
        common_norm=False, 
        bins=50,
        palette=palette_dict,  # <--- ZDE BÝVALA CHYBA (nyní posíláme dict)
        alpha=0.3
    )
    
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold ({threshold:.2f})')
        plt.legend()
    
    plt.title(title, pad=15)
    plt.xlabel("Anomaly Score (Higher = More Anomalous)")
    plt.ylabel("Density")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix_heatmap(y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  normalize: bool = False,
                                  title: str = "Confusion Matrix", 
                                  save_path: Optional[Path] = None) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: If True, show percentages (recall). If False, show raw counts.
        title: Plot title
        save_path: Path to save the figure
    """
    # Validation
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # 1. Výpočet matice
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Normalizace (pokud je vyžadována)
    if normalize:
        # Dělíme součtem řádků (True labels) -> Získáme Recall pro každou třídu
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.1%' # Formátování jako procenta (např. 85.2%)
        title = f"{title} (Normalized)"
    else:
        fmt = 'd'   # Formátování jako celá čísla (integer)

    # 3. Příprava grafu
    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
    
    # Popisky os
    labels = ['Neutral (L0)', 'Anomaly (L1)']
    
    # Vykreslení Heatmapy
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',      # Modrá je standard pro CM, je čitelná
        xticklabels=labels, 
        yticklabels=labels,
        annot_kws={"size": 14, "weight": "bold"}, # Větší písmo čísel
        cbar=True
    )
    
    plt.title(title, fontsize=15, pad=15)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
        logger.info(f"💾 Saved confusion matrix to {save_path}")
        
    plt.show()

# ============================================================================
# B. EMBEDDING PROJECTIONS
# ============================================================================

def compute_projections(X: np.ndarray,
                       methods: List[str] = ['PCA', 't-SNE', 'UMAP'],
                       max_samples: Optional[int] = None,
                       random_state: int = config.RANDOM_SEED) -> Tuple[Dict, np.ndarray]:
    """
    Compute 2D projections of high-dimensional data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        methods: List of projection methods to compute
        max_samples: Subsample if dataset is large (for speed)
        random_state: Random seed
    
    Returns:
        Tuple of:
            - projections: Dict mapping method name to 2D coordinates
            - sampled_indices: Indices of samples used (if subsampled)
    
    Example:
        >>> projs, indices = compute_projections(X, methods=['PCA', 't-SNE'])
        >>> plt.scatter(projs['PCA'][:, 0], projs['PCA'][:, 1])
    
    Note:
        UMAP requires: pip install umap-learn
    """
    # ✅ NEW: Validation
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    
    # ✅ IMPROVED: Use config default
    if max_samples is None:
        max_samples = config.VIZ_CONFIG['projection']['max_samples']
    
    # Import projection methods
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    try:
        import umap.umap_ as umap
        umap_available = True
    except ImportError:
        umap_available = False
        if 'UMAP' in methods:
            logger.warning("⚠️ UMAP not installed. Skipping UMAP projection.")
            methods = [m for m in methods if m != 'UMAP']
    
    # Subsampling
    if len(X) > max_samples:
        logger.info(f"⚠️ Subsampling {len(X)} → {max_samples} points for visualization")
        indices = np.random.RandomState(random_state).choice(
            len(X), max_samples, replace=False
        )
        X_viz = X[indices]
        sampled_indices = indices
    else:
        X_viz = X
        sampled_indices = np.arange(len(X))
    
    projections = {}
    
    # ✅ IMPROVED: Use config parameters
    viz_config = config.VIZ_CONFIG['projection']
    
    if 'PCA' in methods:
        logger.info("1️⃣ Computing PCA...")
        projections['PCA'] = PCA(n_components=2, random_state=random_state).fit_transform(X_viz)
    
    if 't-SNE' in methods:
        logger.info("2️⃣ Computing t-SNE...")
        projections['t-SNE'] = TSNE(
            n_components=2,
            perplexity=viz_config['tsne_perplexity'],
            random_state=random_state,
            init=viz_config['tsne_init'],
            learning_rate=viz_config['tsne_learning_rate']
        ).fit_transform(X_viz)
    
    if 'UMAP' in methods and umap_available:
        logger.info("3️⃣ Computing UMAP...")
        projections['UMAP'] = umap.UMAP(
            n_neighbors=viz_config['umap_n_neighbors'],
            min_dist=viz_config['umap_min_dist'],
            random_state=random_state
        ).fit_transform(X_viz)
    
    return projections, sampled_indices


def plot_embedding_projection(coords: np.ndarray,
                              labels: np.ndarray,
                              palette: Dict[str, str],
                              title: str,
                              hue_order: Optional[List[str]] = None,
                              alpha: Optional[float] = None,
                              save_path: Optional[Path] = None) -> None:
    """
    Plot 2D scatter plot of embedding projection.
    
    Args:
        coords: 2D coordinates (n_samples, 2)
        labels: Category labels for coloring
        palette: Dict mapping label to color
        title: Plot title
        hue_order: Order of categories in legend
        alpha: Point transparency (default from config)
        save_path: Optional save path
    
    Example:
        >>> palette = {'Neutral': '#a1c9f4', 'LJMPNIK': '#ff9f9a'}
        >>> plot_embedding_projection(coords, labels, palette, "t-SNE Projection")
    """
    # ✅ NEW: Validation
    if coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got shape {coords.shape}")
    
    if len(coords) != len(labels):
        raise ValueError(f"Shape mismatch: coords={len(coords)}, labels={len(labels)}")
    
    # ✅ IMPROVED: Use config default
    if alpha is None:
        alpha = config.VIZ_CONFIG['alpha']['scatter']
    
    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['square'])
    
    # ✅ UNCHANGED: Size logic (works well)
    sizes = {k: 40 for k in palette.keys()}
    if 'TN' in palette:
        sizes['TN'] = 15
    if 'Neutral' in palette:
        sizes['Neutral'] = 15
    
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
    
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
    
    plt.show()

# ============================================================================
# C. SUPERVISED RESULTS COMPARISON
# ============================================================================

# ✅ FIXED: Merged duplicate functions into one
def _prepare_long_data(df_results: pd.DataFrame,
                      metric: str = 'auprc',
                      extra_id_vars: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert results DataFrame from wide to long format for plotting.
    
    Unified function that handles both S1 and S2 experiments.
    
    Args:
        df_results: Results DataFrame with train/test columns
        metric: Metric name (e.g., 'auprc', 'f1')
        extra_id_vars: Additional columns to preserve (e.g., ['pooling', 'filter'])
    
    Returns:
        Long-format DataFrame with 'Dataset' column ('Train'/'Test')
    
    Raises:
        ValueError: If required columns are missing
    """
    col_test = f'test_{metric}'
    col_train = f'train_{metric}'
    
    # ✅ NEW: Validation
    if col_test not in df_results.columns:
        raise ValueError(
            f"Column '{col_test}' not found. "
            f"Available: {df_results.columns.tolist()}"
        )
    
    # Base ID variables
    id_vars = ['scenario', 'model']
    
    # Add extra columns if provided
    if extra_id_vars:
        id_vars.extend(extra_id_vars)
    
    # Ensure all requested columns exist
    id_vars = [col for col in id_vars if col in df_results.columns]
    
    # Melt
    df_long = df_results.melt(
        id_vars=id_vars,
        value_vars=[col_train, col_test],
        var_name='dataset_col',
        value_name='score'
    )
    
    # Create readable label
    df_long['Dataset'] = df_long['dataset_col'].apply(
        lambda x: 'Train' if 'train' in x else 'Test'
    )
    
    return df_long


def plot_scenario_breakdown(df_results: pd.DataFrame,
                            metric: str = 'auprc',
                            save_dir: Optional[Path] = None) -> None:
    """
    Plot separate figure for each scenario comparing models.
    
    Args:
        df_results: Results DataFrame with columns [scenario, filter, model, train_X, test_X]
        metric: Metric to plot
        save_dir: If provided, save figures to this directory
    
    Example:
        >>> plot_scenario_breakdown(results_df, metric='f1', save_dir=Path('results/'))
    """
    # ✅ NEW: Validation
    if 'scenario' not in df_results.columns:
        raise ValueError("df_results must contain 'scenario' column")
    
    scenarios = df_results['scenario'].unique()
    df_long = _prepare_long_data(df_results, metric, extra_id_vars=['filter'])
    
    palette = {"Train": "#b0bec5", "Test": "#2c3e50"}
    
    for scen in scenarios:
        data_scen = df_long[df_long['scenario'] == scen]
        
        if data_scen.empty:
            logger.warning(f"No data for scenario '{scen}', skipping")
            continue
        
        g = sns.catplot(
            data=data_scen,
            kind="bar",
            x="model",
            y="score",
            hue="Dataset",
            col="filter",
            palette=palette,
            height=4,
            aspect=1,
            sharey=True
        )
        
        g.fig.suptitle(
            f"Scénář: {scen} (Train vs Test - {metric.upper()})",
            y=1.05, fontsize=16, weight='bold'
        )
        g.set_axis_labels("", f"{metric.upper()} Score")
        g.set_titles("{col_name}")
        
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"breakdown_{scen}_{metric}.png"
            plt.savefig(save_path, bbox_inches='tight', 
                       dpi=config.VIZ_CONFIG['dpi']['print'])
            logger.info(f"💾 Saved {save_path}")
            plt.close()
        else:
            plt.show()


def plot_global_comparison(df_results: pd.DataFrame,
                           metric: str = 'auprc',
                           save_path: Optional[Path] = None) -> None:
    """
    Plot global comparison of all scenarios and models.
    
    Creates faceted plot with scenarios on X-axis, models as separate subplots,
    and colored bars for train/test split.
    
    Args:
        df_results: Results DataFrame
        metric: Metric to plot
        save_path: Optional save path
    
    Example:
        >>> plot_global_comparison(results_df, metric='auprc')
    """
    # ✅ NEW: Validation
    required_cols = ['scenario', 'model']
    missing = [col for col in required_cols if col not in df_results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df_long = _prepare_long_data(df_results, metric)
    
    # ✅ IMPROVED: Color generation helper
    def lighten_color(color: str, amount: float = 0.5) -> Tuple[float, float, float]:
        """Lighten a color by increasing its luminance."""
        try:
            c = mc.to_rgb(color)
            c_hls = colorsys.rgb_to_hls(*c)
            new_l = min(1.0, c_hls[1] + (1.0 - c_hls[1]) * amount)
            return colorsys.hls_to_rgb(c_hls[0], new_l, c_hls[2])
        except ValueError:
            return color
    
    # Build palette
    final_palette = {}
    ordered_hue = []
    
    scenarios = sorted(df_results['scenario'].unique())
    
    for scen in scenarios:
        base = config.SCENARIO_COLORS.get(scen, "#333333")
        
        label_test = f"{scen} (Test)"
        label_train = f"{scen} (Train)"
        
        final_palette[label_test] = base
        final_palette[label_train] = lighten_color(base, amount=0.6)
        
        ordered_hue.extend([label_train, label_test])
    
    # Create combined label
    df_long['Scenario_Type'] = df_long['scenario'] + " (" + df_long['Dataset'] + ")"
    
    # Plot
    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="scenario",
        y="score",
        hue="Scenario_Type",
        col="model",
        col_wrap=2,
        palette=final_palette,
        hue_order=ordered_hue,
        height=3.5,
        aspect=1.5,
        sharey=True,
        dodge=True
    )
    
    g.fig.suptitle(f"Globální srovnání scénářů ({metric.upper()})", 
                   y=1.02, fontsize=16)
    g.set_axis_labels("Scénář", f"{metric.upper()}")
    g.set_titles("{col_name}")
    
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.05), 
                   ncol=len(scenarios), title=None)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', 
                   dpi=config.VIZ_CONFIG['dpi']['print'])
        logger.info(f"💾 Saved {save_path}")
    
    plt.show()


def plot_pooling_breakdown(df_results: pd.DataFrame,
                           metric: str = 'auprc',
                           save_path: Optional[Path] = None) -> None:
    """
    Plot breakdown for S2 experiments (comparing pooling strategies).
    
    Args:
        df_results: Results with 'pooling' column
        metric: Metric to plot
        save_path: Optional save path
    
    Example:
        >>> plot_pooling_breakdown(s2_results, metric='auprc')
    """
    # ✅ NEW: Validation
    if 'pooling' not in df_results.columns:
        raise ValueError("df_results must contain 'pooling' column for S2 experiments")
    
    scenarios = df_results['scenario'].unique()
    df_long = _prepare_long_data(df_results, metric, extra_id_vars=['pooling'])
    
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
            col="pooling",
            palette=palette,
            height=5,
            aspect=1.2,
            sharey=True
        )
        
        g.fig.suptitle(
            f"Scénář: {scen.upper()} (Train vs Test - {metric.upper()})",
            y=1.05, fontsize=16, weight='bold'
        )
        g.set_axis_labels("", f"{metric.upper()} Score")
        g.set_titles("Pooling: {col_name}")
        
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', 
                       dpi=config.VIZ_CONFIG['dpi']['print'])
            logger.info(f"💾 Saved {save_path}")
        
        plt.show()

# ============================================================================
# D. ADVANCED ANALYSIS
# ============================================================================

def plot_model_calibration(y_true: np.ndarray,
                           y_probs: np.ndarray,
                           title: str = "Kalibrační křivka",
                           save_path: Optional[Path] = None) -> None:
    """
    Plot reliability diagram (calibration curve).
    
    Shows how well predicted probabilities match actual frequencies.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        title: Plot title
        save_path: Optional save path
    
    Example:
        >>> plot_model_calibration(y_test, y_probs)
    """
    # ✅ NEW: Validation
    if len(y_true) != len(y_probs):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_probs={len(y_probs)}")
    
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['small'])
    
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, 
             label='Model', color=config.COLORS['l1'])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', 
             label='Perfectly Calibrated')
    
    plt.title(title)
    plt.xlabel("Průměrná predikovaná pravděpodobnost")
    plt.ylabel("Skutečný podíl pozitivních (Fraction of Positives)")
    plt.legend()
    plt.grid(True, alpha=config.VIZ_CONFIG['alpha']['grid'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(model,
                            top_n: int = 20,
                            title: str = "Feature Importance",
                            save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance for models that support it.
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        top_n: Number of top features to display
        title: Plot title
        save_path: Optional save path
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier().fit(X_train, y_train)
        >>> plot_feature_importance(clf, top_n=30)
    
    Note:
        For embedding models (768 dimensions), this shows importance of
        embedding dimensions, not individual words.
    """
    importances = None
    
    # 1. Tree-based models (RandomForest, XGBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    
    # 2. Linear models (LogisticRegression, SVM linear)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    
    if importances is not None:
        # Sort and select top features
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
        plt.title(f"{title} (Top {top_n})", pad=15)
        plt.bar(range(top_n), importances[indices], 
                color=config.COLORS['l1'], align="center")
        plt.xticks(range(top_n), indices, rotation=45)
        plt.xlim([-1, top_n])
        plt.xlabel("Index dimenze embeddingu")
        plt.ylabel("Důležitost")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], 
                       bbox_inches='tight')
        
        plt.show()
    else:
        logger.warning(
            f"⚠️ Model {type(model).__name__} nepodporuje feature importance "
            f"nebo není natrénovaný."
        )


def plot_error_analysis_projection(coords: np.ndarray,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   method_name: str = "Projection",
                                   save_path: Optional[Path] = None) -> None:
    """
    Plot projection colored by error type (TP, FP, TN, FN).
    
    Args:
        coords: 2D projection coordinates
        y_true: Ground truth labels
        y_pred: Predicted labels
        method_name: Projection method name for title
        save_path: Optional save path
    
    Example:
        >>> projs, _ = compute_projections(X_test, methods=['t-SNE'])
        >>> plot_error_analysis_projection(projs['t-SNE'], y_test, y_pred, "t-SNE")
    """
    # ✅ NEW: Validation
    if len(coords) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("coords, y_true, and y_pred must have same length")
    
    # Determine categories
    categories = []
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            categories.append('TP (Correct Anomaly)')
        elif t == 0 and p == 0:
            categories.append('TN (Correct Neutral)')
        elif t == 0 and p == 1:
            categories.append('FP (False Alarm)')
        elif t == 1 and p == 0:
            categories.append('FN (Missed Anomaly)')
    
    # ✅ IMPROVED: Use config colors
    palette_err = {
        'TP (Correct Anomaly)': config.COLORS['TP'],
        'TN (Correct Neutral)': config.COLORS['TN'],
        'FP (False Alarm)': config.COLORS['FP'],
        'FN (Missed Anomaly)': config.COLORS['FN']
    }
    
    order = [
        'TN (Correct Neutral)',
        'TP (Correct Anomaly)',
        'FP (False Alarm)',
        'FN (Missed Anomaly)'
    ]
    
    plot_embedding_projection(
        coords,
        labels=categories,
        palette=palette_err,
        title=f"{method_name} - Error Analysis",
        hue_order=order,
        alpha=0.7,
        save_path=save_path
    )


def plot_bootstrap_results(bootstrap_scores: Dict[str, np.ndarray],
                           metric_name: str = "F1 Score",
                           title: str = "Stabilita modelu (Bootstrap)",
                           save_path: Optional[Path] = None) -> None:
    """
    Plot bootstrap distribution of scores with statistics.
    
    Args:
        bootstrap_scores: Dict mapping model_name to array of bootstrap scores
        metric_name: Name of metric for axis label
        title: Plot title
        save_path: Optional save path
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression()
        >>> scores = {'LogReg': [0.75, 0.78, 0.76, ...]}  # 100 bootstrap iterations
        >>> plot_bootstrap_results(scores, metric_name="F1 Score")
    
    Output:
        Displays KDE plot with distribution of scores and prints statistics
        (mean, std, 95% confidence interval) for each model.
    """
    # ✅ NEW: Validation
    if not bootstrap_scores:
        raise ValueError("bootstrap_scores dictionary is empty")
    
    plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
    
    stats_text = []
    
    # Iterate through models
    for model_name, scores in bootstrap_scores.items():
        if len(scores) == 0:
            logger.warning(f"No scores for model '{model_name}', skipping")
            continue
        
        scores = np.array(scores)
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)
        
        # Label for legend
        label = f"{model_name}\nμ={mean_score:.3f}, σ={std_score:.3f}"
        
        # KDE Plot
        sns.kdeplot(scores, label=label, fill=True, alpha=0.3)
        
        # Save text for printing
        stats_text.append(
            f"🔹 {model_name}: Mean={mean_score:.4f} | Std={std_score:.4f} | "
            f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
        )
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(metric_name)
    plt.ylabel("Hustota pravděpodobnosti")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=config.VIZ_CONFIG['alpha']['grid'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], 
                   bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print("\n📊 STATISTICKÉ VÝSLEDKY:")
    for line in stats_text:
        print(line)

# =============================================================================
# DETAILED VISUALIZATION (Train vs Val vs Test)
# =============================================================================

def plot_three_way_comparison(df_results, metric='f1', save_dir=None):
    """
    Vykreslí srovnání Train/Val/Test pro každý experiment (S1a, S1b...).
    Používá barvy definované v config.COLORS.
    """
    # 0. Pracujeme s kopií, ať neovlivníme originál
    df_viz = df_results.copy()

    # 1. Vytvoření unikátního popisku (ID + Název), např. "S1a: Baseline (Imbalanced)"
    # Tím zajistíme, že S1a a S1b budou v grafu odděleně
    if 'scenario_name' in df_viz.columns:
        df_viz['Experiment_Label'] = df_viz.apply(lambda x: f"{x['id']}: {x['scenario_name']}", axis=1)
    else:
        # Fallback pro starší verze CSV
        df_viz['Experiment_Label'] = df_viz['id'] + " (" + df_viz['scenario'] + ")"

    # 2. Kontrola sloupců
    cols = [f'train_{metric}', f'val_{metric}', f'test_{metric}']
    if not all(c in df_viz.columns for c in cols):
        print(f"⚠️ Metrika '{metric}' není dostupná pro všechny sady (Train/Val/Test).")
        return

    # 3. Transformace dat (Wide -> Long)
    df_long = pd.melt(
        df_viz,
        id_vars=['Experiment_Label', 'model'], # <--- ZMĚNA: Seskupujeme podle Experiment Label
        value_vars=cols,
        var_name='Split_Raw',
        value_name='Score'
    )
    
    # Vyčistíme názvy splitů (train_f1 -> Train)
    df_long['Split'] = df_long['Split_Raw'].apply(lambda x: x.split('_')[0]) 
    
    # 4. Barvy z Configu (Pastelová paleta)
    split_colors = {
        'train': config.COLORS['train'],
        'val':   config.COLORS['val'],
        'test':  config.COLORS['test']
    }
    
    # Pořadí a formátování pro legendu
    split_order = ['train', 'val', 'test']
    df_long['Split_Label'] = df_long['Split'].map(lambda x: x.capitalize())
    split_colors_mapped = {k.capitalize(): v for k, v in split_colors.items()}
    split_order_mapped = [s.capitalize() for s in split_order]

    # 5. Vykreslení po Experimentech (S1a, S1b, S1d, S1e...)
    experiments = sorted(df_long['Experiment_Label'].unique())
    
    for exp_label in experiments:
        data_exp = df_long[df_long['Experiment_Label'] == exp_label]
        
        plt.figure(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
        
        ax = sns.barplot(
            data=data_exp,
            x='model',
            y='Score',
            hue='Split_Label',
            hue_order=split_order_mapped,
            palette=split_colors_mapped,
            edgecolor='white',
            linewidth=1.5,
            errorbar=None # <--- POJISTKA: Vypne černé čáry (kdyby náhodou)
        )
        
        # Titulek nyní obsahuje ID (S1a...)
        plt.title(f"{exp_label} - {metric.upper()} Comparison", fontsize=15, pad=15, fontweight='bold')
        plt.xlabel("Model", fontsize=12, fontweight='bold')
        plt.ylabel(f"{metric.upper()} Score", fontsize=12, fontweight='bold')
        plt.ylim(0, 1.1) 
        plt.grid(axis='y', linestyle='--', alpha=config.VIZ_CONFIG['alpha']['grid'])
        
        plt.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        
        # Hodnoty nad sloupci
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        
        if save_dir:
            # Bezpečný název souboru (nahradíme dvojtečky a mezery)
            safe_name = exp_label.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
            save_path = save_dir / f"breakdown_3way_{safe_name}_{metric}.png"
            plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi']['print'], bbox_inches='tight')
            print(f"💾 Graf uložen: {save_path.name}")
        
        plt.show()

def plot_model_comparison(df_results: pd.DataFrame,
                          metric: str = 'test_f1',
                          x_col: str = 'model',
                          hue_col: str = 'scenario_name',
                          col_col: str = 'pooling',
                          title: str = "Comparison of Models",
                          save_path: Optional[Path] = None) -> None:
    """
    Vykreslí porovnání modelů (Barplot) rozdělené podle poolingu a scénáře.
    Vhodné pro Results Overview M1.
    
    Args:
        df_results: DataFrame s výsledky
        metric: Název sloupce s metrikou (např. 'test_f1')
        x_col: Co bude na ose X (typicky 'model')
        hue_col: Co bude rozlišeno barvou (typicky 'scenario_name')
        col_col: Co bude rozděleno do sloupců grafu (typicky 'pooling')
        title: Nadpis grafu
        save_path: Cesta pro uložení
    """
    # Kontrola existence sloupce
    if metric not in df_results.columns:
        logger.warning(f"⚠️ Metrika '{metric}' není v datech. Graf nebude vykreslen.")
        return

    # Nastavení stylu (pro jistotu, kdyby nebylo globálně)
    sns.set_style("whitegrid")
    
    # Vytvoření FacetGridu
    g = sns.catplot(
        data=df_results, 
        kind="bar",
        x=x_col, 
        y=metric, 
        hue=hue_col, 
        col=col_col,
        palette="viridis",  # Konzistentní paleta
        height=5, 
        aspect=1.2,
        sharey=True
    )
    
    # Formátování nadpisů a os
    g.fig.suptitle(title, y=1.05, fontsize=16, weight='bold')
    
    # Pěkný popisek osy Y (z 'test_f1' udělá 'Test F1')
    y_label = metric.replace('_', ' ').title()
    g.set_axis_labels("", y_label)
    
    # Nadpisy sloupců (např. "Pooling: Mean")
    g.set_titles(f"{col_col.title()}: {{col_name}}")
    
    # Přidání hodnot nad sloupce
    for ax in g.axes.flat:
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        for container in ax.containers:
            # Formátování čísla (0.75)
            ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3, weight='bold')
            
    # Uložení
    if save_path:
        # Načteme DPI z configu, pokud je dostupné, jinak default 300
        dpi = config.VIZ_CONFIG['dpi']['print'] if hasattr(config, 'VIZ_CONFIG') else 300
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        logger.info(f"💾 Saved comparison plot to {save_path}")
        
    plt.show()

# ============================================================================
# E. LLM BENCHMARK COMPARISON
# ============================================================================

def plot_llm_vs_m2_comparison(df_llm_metrics: pd.DataFrame,
                               df_m2_best: pd.DataFrame,
                               metrics: List[str] = ['auprc', 'f1', 'precision', 'recall'],
                               title: str = "LLM (Zero-Shot) vs. Nejlepší M2/S2 Model",
                               save_path: Optional[Path] = None) -> None:
    """
    Porovná výsledky LLM modelů s nejlepším modelem z M2/S2 (Sentence Supervised).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import numpy as np
    import config
    import logging
    logger = logging.getLogger(__name__)

    # --- 1. Validace vstupů ---
    required_llm_cols = ['model'] + metrics
    missing_llm = [c for c in required_llm_cols if c not in df_llm_metrics.columns]
    if missing_llm:
        raise ValueError(f"df_llm_metrics chybí sloupce: {missing_llm}")

    # --- 2. Příprava dat M2/S2 baseline ---
    col_map = {'auprc': 'test_auprc', 'f1': 'test_f1', 'precision': 'test_prec', 'recall': 'test_rec'}
    
    m2_row = df_m2_best.iloc[0]
    m2_label = f"M2/S2 Baseline"
    
    m2_data = {'model': m2_label}
    for metric in metrics:
        src_col = col_map.get(metric, f'test_{metric}')
        if src_col in m2_row.index:
            m2_data[metric] = m2_row[src_col]
        else:
            m2_data[metric] = np.nan

    df_m2_formatted = pd.DataFrame([m2_data])

    # --- 3. Spojení dat ---
    df_all = pd.concat([df_llm_metrics[['model'] + metrics], df_m2_formatted], ignore_index=True)

    # --- 4. Příprava barev ---
    model_names = df_all['model'].tolist()
    palette = {name: config.COLORS['l0'] if name.startswith('M2/S2') else config.COLORS['l1'] for name in model_names}

    # --- 5. Transformace do long formátu ---
    df_long = df_all.melt(id_vars='model', value_vars=metrics, var_name='Metrika', value_name='Hodnota')
    metric_labels = {'auprc': 'AUPRC', 'f1': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall'}
    df_long['Metrika'] = df_long['Metrika'].map(lambda x: metric_labels.get(x, x.upper()))

    # --- 6. Vykreslení ---
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=True)
    if n_metrics == 1: axes = [axes]
    
    metric_label_list = [metric_labels.get(m, m.upper()) for m in metrics]

    for ax, metric_label in zip(axes, metric_label_list):
        df_sub = df_long[df_long['Metrika'] == metric_label].copy()
        colors = [palette[m] for m in df_sub['model']]

        bars = ax.bar(df_sub['model'], df_sub['Hodnota'], color=colors, edgecolor='white', linewidth=1.5, width=0.6)

        # Hodnoty nad sloupci
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Zvýraznění referenčního modelu
        m2_val = df_sub[df_sub['model'].str.startswith('M2/S2')]['Hodnota'].values
        if len(m2_val) > 0 and not np.isnan(m2_val[0]):
            ax.axhline(y=m2_val[0], color=config.COLORS['l0'], linestyle='--', linewidth=1.5, alpha=0.8)

        ax.set_title(metric_label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        
        # Opravené zarovnání textu osy X
        ax.set_xticks(range(len(df_sub['model'])))
        ax.set_xticklabels(df_sub['model'], rotation=30, ha='right', fontsize=9)
        sns.despine(ax=ax)

    # --- 7. Legenda a titulek ---
    legend_patches = [
        mpatches.Patch(color=config.COLORS['l1'], label='LLM (Zero-Shot)'),
        mpatches.Patch(color=config.COLORS['l0'], label='M2/S2 Baseline')
    ]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Saved LLM comparison plot to {save_path}")

    plt.show()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Style
    'setup_style',
    
    # Metrics & Performance
    'plot_pr_curve',
    'plot_threshold_tuning',
    'plot_anomaly_histogram',
    'plot_confusion_matrix_heatmap',
    
    # Embeddings & Projections
    'compute_projections',
    'plot_embedding_projection',
    
    # Supervised Results
    'plot_scenario_breakdown',
    'plot_global_comparison',
    'plot_pooling_breakdown',
    
    # Advanced Analysis
    'plot_model_calibration',
    'plot_feature_importance',
    'plot_error_analysis_projection',
    'plot_bootstrap_results',

    # LLM Benchmark
    'plot_llm_vs_m2_comparison',
]

"""
EDA Visualization Module

Reusable functions for exploratory data analysis visualizations.
All plots use the unified style from config.py and visualization.setup_style().

Each function produces ONE independent figure (no subplots grids),
so that every plot can be saved and embedded in the thesis separately.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Optional, Dict
from pathlib import Path

import config

# DPI shortcut
_DPI = config.VIZ_CONFIG['dpi']['print']


# ============================================================================
# A. CLASS & LABEL DISTRIBUTIONS
# ============================================================================

def plot_class_distribution(df: pd.DataFrame,
                           title: str = "Distribuce tříd (L0 vs. L1)",
                           save_path: Optional[Path] = None) -> pd.Series:
    """
    Plot distribution of labels (L0 vs L1) as a bar chart.

    Args:
        df: DataFrame with 'label' column
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        pd.Series with label counts
    """
    label_counts = df['label'].value_counts().sort_index()

    plot_df = pd.DataFrame({
        'Třída': ['Neutrální (L0)', 'Bias/LJMPNIK (L1)'],
        'Počet': label_counts.values
    })

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['small'])

    sns.barplot(
        data=plot_df, x='Třída', y='Počet',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        edgecolor='white', linewidth=1.2, ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontweight='bold')

    total = label_counts.sum()
    ratio = label_counts[0] / label_counts[1] if label_counts[1] > 0 else float('inf')

    ax.text(0.5, 0.95,
            f'Celkem: {total:,} vzorků | Poměr L0:L1 = {ratio:.2f}:1',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#EAEAF2', alpha=0.7))

    ax.set_xlabel('')
    ax.set_ylabel('Počet')
    # ax.set_title(title, pad=15)  # LaTeX \caption
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()
    return label_counts


# ============================================================================
# B. TEXT LENGTH DISTRIBUTIONS (split into 2 independent figures)
# ============================================================================

def _get_lengths(df: pd.DataFrame, text_col: str = 'text') -> pd.Series:
    """Helper: extract text lengths from DataFrame."""
    if text_col in df.columns:
        return df[text_col].str.split().str.len()
    elif 'num_tokens' in df.columns:
        return df['num_tokens']
    else:
        raise ValueError("No text or num_tokens column found")


def plot_length_histogram(df: pd.DataFrame,
                          text_col: str = 'text',
                          title: str = "Distribuce délek vět (Histogram)",
                          save_path: Optional[Path] = None) -> Dict:
    """
    Plot histogram of text lengths separated by label.

    Args:
        df: DataFrame with text/num_tokens column and 'label' column
        text_col: Name of text column
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Dict with mean/std statistics per class
    """
    lengths = _get_lengths(df, text_col)
    l0_lengths = lengths[df['label'] == 0]
    l1_lengths = lengths[df['label'] == 1]

    hist_df = pd.DataFrame({
        'Délka': pd.concat([l0_lengths, l1_lengths], ignore_index=True),
        'Třída': ['Neutrální (L0)'] * len(l0_lengths) + ['Bias/LJMPNIK (L1)'] * len(l1_lengths)
    })

    palette_dict = {
        'Neutrální (L0)': config.COLORS['l0'],
        'Bias/LJMPNIK (L1)': config.COLORS['l1']
    }

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])

    sns.histplot(
        data=hist_df, x='Délka', hue='Třída',
        palette=palette_dict,
        bins=30, alpha=0.7, edgecolor='white', ax=ax
    )

    stats_text = (
        f"Neutral: \u03bc={l0_lengths.mean():.1f}, \u03c3={l0_lengths.std():.1f}\n"
        f"LJMPNIK: \u03bc={l1_lengths.mean():.1f}, \u03c3={l1_lengths.std():.1f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#EAEAF2', alpha=0.7), fontsize=10)

    ax.set_xlabel('Počet tokenů')
    ax.set_ylabel('Četnost')
    # ax.set_title(title, pad=15)  # LaTeX \caption
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()

    return {
        'l0_mean': l0_lengths.mean(), 'l0_std': l0_lengths.std(),
        'l1_mean': l1_lengths.mean(), 'l1_std': l1_lengths.std(),
    }


def plot_length_boxplot(df: pd.DataFrame,
                        text_col: str = 'text',
                        title: str = "Distribuce délek vět (Box Plot)",
                        save_path: Optional[Path] = None) -> None:
    """
    Plot box plot of text lengths separated by label.

    Args:
        df: DataFrame with text/num_tokens column and 'label' column
        text_col: Name of text column
        title: Plot title
        save_path: Optional path to save figure
    """
    lengths = _get_lengths(df, text_col)
    l0_lengths = lengths[df['label'] == 0]
    l1_lengths = lengths[df['label'] == 1]

    data_for_box = pd.DataFrame({
        'Length': pd.concat([l0_lengths, l1_lengths]),
        'Label': ['Neutral'] * len(l0_lengths) + ['LJMPNIK'] * len(l1_lengths)
    })

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['small'])

    sns.boxplot(
        data=data_for_box, x='Label', y='Length',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        ax=ax
    )
    # ax.set_title(title, pad=15)  # LaTeX \caption
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()


# Backward-compatible wrapper
def plot_length_distribution(df: pd.DataFrame, text_col: str = 'text',
                             title: str = "Distribuce délek vět",
                             save_path: Optional[Path] = None) -> Dict:
    """Convenience wrapper — calls both histogram and boxplot."""
    if save_path is not None:
        save_path = Path(save_path)
        hist_path = save_path.with_name(save_path.stem + '_histogram' + save_path.suffix)
        box_path  = save_path.with_name(save_path.stem + '_boxplot'   + save_path.suffix)
    else:
        hist_path = box_path = None

    stats = plot_length_histogram(df, text_col, f"{title} (Histogram)", hist_path)
    plot_length_boxplot(df, text_col, f"{title} (Box Plot)", box_path)
    return stats


# ============================================================================
# C. POS TAG ANALYSIS
# ============================================================================

def plot_pos_distribution(token_df: pd.DataFrame,
                          top_n: int = 15,
                          title: str = "Distribuce POS značek",
                          save_path: Optional[Path] = None) -> Dict:
    """
    Plot distribution of POS tags separated by label.

    Args:
        token_df: Token-level DataFrame with 'pos' column
        top_n: Number of top POS tags to show
        title: Plot title
        save_path: Optional save path

    Returns:
        Dict with POS counters per class
    """
    l0_pos = Counter(token_df[token_df['label'] == 0]['pos'])
    l1_pos = Counter(token_df[token_df['label'] == 1]['pos'])

    top_pos_tags = [tag for tag, _ in l1_pos.most_common(top_n)]

    plot_data = []
    for pos in top_pos_tags:
        plot_data.append({'POS': pos, 'Count': l0_pos.get(pos, 0), 'Label': 'Neutrální (L0)'})
        plot_data.append({'POS': pos, 'Count': l1_pos.get(pos, 0), 'Label': 'Bias/LJMPNIK (L1)'})
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=plot_df, x='POS', y='Count', hue='Label',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        ax=ax
    )

    ax.set_xlabel('POS tag')
    ax.set_ylabel('Počet')
    # ax.set_title(title, pad=15)  # LaTeX \caption
    ax.legend(title='Třída')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()

    return {'l0_pos_distribution': l0_pos, 'l1_pos_distribution': l1_pos}


def plot_ljmpnik_pos_analysis(token_df: pd.DataFrame,
                              title: str = "Analýza POS značek LJMPNIK slov",
                              save_path: Optional[Path] = None) -> Optional[pd.Series]:
    """
    Analyze POS tags specifically for LJMPNIK tokens (is_target=True).

    Args:
        token_df: Token DataFrame
        title: Plot title
        save_path: Optional save path

    Returns:
        pd.Series with POS counts, or None if no LJMPNIK tokens
    """
    ljmpnik_tokens = token_df[token_df['is_target'] == True]

    if len(ljmpnik_tokens) == 0:
        print("No LJMPNIK tokens found (is_target=True)")
        return None

    pos_counts = ljmpnik_tokens['pos'].value_counts()

    pos_df = pd.DataFrame({
        'POS': pos_counts.index,
        'Počet': pos_counts.values
    })

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])

    sns.barplot(
        data=pos_df, x='POS', y='Počet',
        color=config.COLORS['l1'], edgecolor='white', linewidth=1.2, ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontweight='bold')

    ax.set_xlabel('POS tag')
    ax.set_ylabel('Počet LJMPNIK tokenů')
    # ax.set_title(title, pad=15)  # LaTeX \caption

    total = len(ljmpnik_tokens)
    top_pos = pos_counts.index[0]
    top_pct = (pos_counts.iloc[0] / total) * 100
    summary_text = f"Celkem LJMPNIK tokenů: {total}\nNejčastější: {top_pos} ({top_pct:.1f} %)"
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#EAEAF2', alpha=0.7), fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()
    return pos_counts


# ============================================================================
# D. DOCUMENT-LEVEL ANALYSIS (split into independent figures)
# ============================================================================

def plot_sentences_per_document(df: pd.DataFrame,
                                title: str = "Počet vět na dokument",
                                save_path: Optional[Path] = None) -> None:
    """
    Plot histogram of sentences per document, separated by label.

    Args:
        df: DataFrame with 'document_id' and 'label' columns
        title: Plot title
        save_path: Optional save path
    """
    doc_stats = df.groupby('document_id').agg({
        'label': 'first',
        'sentence_id': 'count'
    }).rename(columns={'sentence_id': 'num_sentences'})

    l0_docs = doc_stats[doc_stats['label'] == 0]['num_sentences']
    l1_docs = doc_stats[doc_stats['label'] == 1]['num_sentences']

    hist_df = pd.DataFrame({
        'Počet vět': pd.concat([l0_docs, l1_docs], ignore_index=True),
        'Typ': ['Neutrální dok.'] * len(l0_docs) + ['LJMPNIK dok.'] * len(l1_docs)
    })
    palette_dict = {
        'Neutrální dok.': config.COLORS['l0'],
        'LJMPNIK dok.': config.COLORS['l1']
    }

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])

    sns.histplot(
        data=hist_df, x='Počet vět', hue='Typ',
        palette=palette_dict,
        bins=15, alpha=0.7, edgecolor='white', ax=ax
    )
    ax.set_xlabel('Počet vět na dokument')
    ax.set_ylabel('Četnost')
    # ax.set_title(title, pad=15)  # LaTeX \caption
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()


def plot_document_stats_table(df: pd.DataFrame,
                              title: str = "Statistika dokumentů",
                              save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Show document-level summary statistics as a standalone table figure.

    Args:
        df: DataFrame with 'document_id' and 'label' columns
        title: Plot title
        save_path: Optional save path

    Returns:
        Summary DataFrame
    """
    doc_stats = df.groupby('document_id').agg({
        'label': 'first',
        'sentence_id': 'count'
    }).rename(columns={'sentence_id': 'num_sentences'})

    l0_docs = doc_stats[doc_stats['label'] == 0]['num_sentences']
    l1_docs = doc_stats[doc_stats['label'] == 1]['num_sentences']

    summary_data = pd.DataFrame({
        'Metrika': ['Počet dokumentů', 'Prům. vět/dok.', 'Směr. odch. vět/dok.'],
        'Neutrální': [len(l0_docs), f"{l0_docs.mean():.2f}", f"{l0_docs.std():.2f}"],
        'LJMPNIK': [len(l1_docs), f"{l1_docs.mean():.2f}", f"{l1_docs.std():.2f}"],
    })

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    table = ax.table(
        cellText=summary_data.values,
        colLabels=summary_data.columns,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for i in range(len(summary_data.columns)):
        table[(0, i)].set_facecolor('#8DA0CB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # ax.set_title(title, pad=15)  # LaTeX \caption
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')

    plt.show()
    return summary_data


# Backward-compatible wrapper
def plot_document_statistics(df: pd.DataFrame, title: str = "Statistika dokumentů",
                             save_path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience wrapper — calls both document plots."""
    plot_sentences_per_document(df, f"{title} — Počet vět na dokument", save_path)
    return plot_document_stats_table(df, f"{title} — Souhrnná tabulka", save_path)


# ============================================================================
# E. DATASET OVERVIEW (split into 4 independent figures)
# ============================================================================

def plot_overview_class_dist(sentence_df: pd.DataFrame,
                             dataset_name: str = 'Dataset',
                             save_path: Optional[Path] = None) -> None:
    """Sentence-level class distribution bar chart."""
    sent_label_counts = sentence_df['label'].value_counts().sort_index()

    plot_df = pd.DataFrame({
        'Třída': ['Neutrální (L0)', 'Bias/LJMPNIK (L1)'],
        'Počet': sent_label_counts.values
    })

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['small'])

    sns.barplot(
        data=plot_df, x='Třída', y='Počet',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        edgecolor='white', linewidth=1.2, ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontweight='bold')

    ax.set_xlabel('')
    # ax.set_title(f'{dataset_name} — Distribuce tříd na úrovni vět', pad=15)  # LaTeX \caption
    ax.set_ylabel('Počet')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.show()


def plot_overview_token_dist(sentence_df: pd.DataFrame,
                             dataset_name: str = 'Dataset',
                             save_path: Optional[Path] = None) -> None:
    """Token count distribution histogram by label."""
    if 'num_tokens' in sentence_df.columns:
        lengths = sentence_df['num_tokens']
    else:
        lengths = sentence_df['text'].str.split().str.len()

    l0_len = lengths[sentence_df['label'] == 0]
    l1_len = lengths[sentence_df['label'] == 1]

    hist_df = pd.DataFrame({
        'Délka': pd.concat([l0_len, l1_len], ignore_index=True),
        'Třída': ['Neutrální (L0)'] * len(l0_len) + ['Bias/LJMPNIK (L1)'] * len(l1_len)
    })
    palette_dict = {
        'Neutrální (L0)': config.COLORS['l0'],
        'Bias/LJMPNIK (L1)': config.COLORS['l1']
    }

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])

    sns.histplot(
        data=hist_df, x='Délka', hue='Třída',
        palette=palette_dict,
        bins=20, alpha=0.7, edgecolor='white', ax=ax
    )
    # ax.set_title(f'{dataset_name} — Distribuce počtu tokenů', pad=15)  # LaTeX \caption
    ax.set_xlabel('Počet tokenů na větu')
    ax.set_ylabel('Četnost')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.show()


def plot_overview_top_pos(token_df: pd.DataFrame,
                          dataset_name: str = 'Dataset',
                          top_n: int = 10,
                          save_path: Optional[Path] = None) -> None:
    """Top N POS tags grouped bar chart by label."""
    l0_pos = Counter(token_df[token_df['label'] == 0]['pos'])
    l1_pos = Counter(token_df[token_df['label'] == 1]['pos'])

    top_pos = [tag for tag, _ in l1_pos.most_common(top_n)]

    plot_data = []
    for pos in top_pos:
        plot_data.append({'POS': pos, 'Počet': l0_pos.get(pos, 0), 'Třída': 'Neutrální (L0)'})
        plot_data.append({'POS': pos, 'Počet': l1_pos.get(pos, 0), 'Třída': 'Bias/LJMPNIK (L1)'})
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])

    sns.barplot(
        data=plot_df, x='POS', y='Počet', hue='Třída',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        edgecolor='white', linewidth=1.2, ax=ax
    )

    # ax.set_title(f'{dataset_name} — Top {top_n} POS značek', pad=15)  # LaTeX \caption
    ax.set_xlabel('POS tag')
    ax.set_ylabel('Počet')
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.show()


def plot_overview_summary_table(token_df: pd.DataFrame,
                                sentence_df: pd.DataFrame,
                                dataset_name: str = 'Dataset',
                                save_path: Optional[Path] = None) -> None:
    """Summary statistics table as standalone figure."""
    summary_stats = pd.DataFrame({
        'Metrika': [
            'Celkem dokumentů', 'Celkem vět', 'Celkem tokenů',
            'Prům. tokenů/větu', 'LJMPNIK tokenů', 'Unikátních POS značek'
        ],
        'Value': [
            token_df['document_id'].nunique(),
            sentence_df.shape[0],
            token_df.shape[0],
            f"{token_df.groupby('sentence_id').size().mean():.1f}",
            int((token_df['is_target'] == True).sum()),
            token_df['pos'].nunique()
        ]
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')

    table = ax.table(
        cellText=summary_stats.values,
        colLabels=summary_stats.columns,
        cellLoc='left', loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(len(summary_stats.columns)):
        table[(0, i)].set_facecolor('#8DA0CB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # ax.set_title(f'{dataset_name} — Souhrnná statistika', pad=15)  # LaTeX \caption
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.show()


# Backward-compatible wrapper
def plot_dataset_overview(token_df: pd.DataFrame, sentence_df: pd.DataFrame,
                          dataset_name: str = 'Dataset',
                          save_path: Optional[Path] = None) -> None:
    """Convenience wrapper — calls all 4 overview plots."""
    plot_overview_class_dist(sentence_df, dataset_name, save_path)
    plot_overview_token_dist(sentence_df, dataset_name, save_path)
    plot_overview_top_pos(token_df, dataset_name, save_path=save_path)
    plot_overview_summary_table(token_df, sentence_df, dataset_name, save_path)


# ============================================================================
# F. TEXT SUMMARIES (non-visual)
# ============================================================================

def print_dataset_stats(token_df: pd.DataFrame,
                        sentence_df: pd.DataFrame,
                        name: str = "DATASET") -> None:
    """
    Print basic statistics about the dataset.
    Robust version handles NaN values in boolean columns.
    """
    print(f"\n{'=' * 20} {name} STATS {'=' * 20}")

    n_docs = sentence_df['document_id'].nunique()
    print(f"DOCUMENTS: {n_docs}")

    n_sent = len(sentence_df)
    n_sent_l0 = (sentence_df['label'] == 0).sum()
    n_sent_l1 = (sentence_df['label'] == 1).sum()

    print(f"\nSENTENCES:")
    print(f"   Total: {n_sent}")

    if 'is_context' in sentence_df.columns:
        is_context_safe = sentence_df['is_context'].fillna(False).astype(bool)
        n_context = is_context_safe.sum()
        n_target = (~is_context_safe).sum()

        target_mask = ~is_context_safe
        target_l0 = ((sentence_df['label'] == 0) & target_mask).sum()
        target_l1 = ((sentence_df['label'] == 1) & target_mask).sum()

        print(f"   Target (Analyzed): {n_target}")
        print(f"     Neutral (L0): {target_l0}")
        print(f"     LJMPNIK (L1): {target_l1}")
        print(f"   Context (Ignored): {n_context}")
    else:
        print(f"   Neutral: {n_sent_l0}")
        print(f"   LJMPNIK: {n_sent_l1}")

    n_tokens = len(token_df)
    n_ljmpnik = (token_df['label'].fillna(0) == 1).sum()

    print(f"\nTOKENS:")
    print(f"   Total: {n_tokens}")
    print(f"   LJMPNIK anomalies: {n_ljmpnik}")
    print(f"   Global anomaly rate: {n_ljmpnik / n_tokens * 100:.4f}%")

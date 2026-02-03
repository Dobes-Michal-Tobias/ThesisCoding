"""
EDA Visualization Module

Reusable functions for exploratory data analysis visualizations.
All plots use config settings for consistency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import config


def setup_style():
    """Apply global visualization style from config."""
    sns.set_theme(
        style=config.SNS_STYLE,
        context=config.SNS_CONTEXT,
        font_scale=config.FONT_SCALE
    )
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=[config.COLORS['l0'], config.COLORS['l1']]
    )
    print(f"🎨 EDA visualization style set: {config.SNS_STYLE}")


def plot_class_distribution(df, title="Class Distribution", save_path=None):
    """
    Plot distribution of labels (L0 vs L1).
    
    Args:
        df: DataFrame with 'label' column
        title: Plot title
        save_path: Optional path to save figure
    """
    # Count labels
    label_counts = df['label'].value_counts().sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['small'])
    
    # Bar plot
    bars = ax.bar(
        ['Neutral (L0)', 'LJMPNIK (L1)'],
        label_counts.values,
        color=[config.COLORS['l0'], config.COLORS['l1']],
        alpha=0.8,
        edgecolor='black'
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Calculate percentage
    total = label_counts.sum()
    ratio = label_counts[0] / label_counts[1]
    
    # Add ratio text
    ax.text(
        0.5, 0.95,
        f'Total: {total:,} samples | Ratio L0:L1 = {ratio:.2f}:1',
        transform=ax.transAxes,
        ha='center',
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_ylabel('Count')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return label_counts


def plot_length_distribution(df, text_col='text', title="Text Length Distribution", 
                            save_path=None):
    """
    Plot distribution of text lengths (in tokens/words).
    
    Args:
        df: DataFrame with text column
        text_col: Name of text column
        title: Plot title
        save_path: Optional path to save
    """
    # Calculate lengths
    if text_col in df.columns:
        lengths = df[text_col].str.split().str.len()
    elif 'num_tokens' in df.columns:
        lengths = df['num_tokens']
    else:
        raise ValueError("No text or num_tokens column found")
    
    # Separate by label
    l0_lengths = lengths[df['label'] == 0]
    l1_lengths = lengths[df['label'] == 1]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=config.VIZ_CONFIG['figure_sizes']['wide'])
    
    # Histogram
    axes[0].hist(
        [l0_lengths, l1_lengths],
        bins=30,
        label=['Neutral (L0)', 'LJMPNIK (L1)'],
        color=[config.COLORS['l0'], config.COLORS['l1']],
        alpha=0.7,
        edgecolor='black'
    )
    axes[0].set_xlabel('Number of Tokens')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Length Distribution (Histogram)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_for_box = pd.DataFrame({
        'Length': pd.concat([l0_lengths, l1_lengths]),
        'Label': ['Neutral'] * len(l0_lengths) + ['LJMPNIK'] * len(l1_lengths)
    })
    
    sns.boxplot(
        data=data_for_box,
        x='Label',
        y='Length',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        ax=axes[1]
    )
    axes[1].set_title('Length Distribution (Box Plot)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Calculate statistics
    stats_text = (
        f"Neutral: μ={l0_lengths.mean():.1f}, σ={l0_lengths.std():.1f}\\n"
        f"LJMPNIK: μ={l1_lengths.mean():.1f}, σ={l1_lengths.std():.1f}"
    )
    
    fig.text(
        0.5, 0.02,
        stats_text,
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    fig.suptitle(title, fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'l0_mean': l0_lengths.mean(),
        'l0_std': l0_lengths.std(),
        'l1_mean': l1_lengths.mean(),
        'l1_std': l1_lengths.std()
    }


def plot_pos_distribution(token_df, top_n=15, title="POS Tag Distribution", 
                         save_path=None):
    """
    Plot distribution of POS tags.
    
    Args:
        token_df: Token-level DataFrame with 'pos' column
        top_n: Number of top POS tags to show
        title: Plot title
        save_path: Optional save path
    """
    # Count POS tags by label
    l0_pos = Counter(token_df[token_df['label'] == 0]['pos'])
    l1_pos = Counter(token_df[token_df['label'] == 1]['pos'])
    
    # Get top N from L1 (anomalies)
    top_pos_tags = [tag for tag, _ in l1_pos.most_common(top_n)]
    
    # Create DataFrame for plotting
    plot_data = []
    for pos in top_pos_tags:
        plot_data.append({
            'POS': pos,
            'Count': l0_pos.get(pos, 0),
            'Label': 'Neutral (L0)'
        })
        plot_data.append({
            'POS': pos,
            'Count': l1_pos.get(pos, 0),
            'Label': 'LJMPNIK (L1)'
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(
        data=plot_df,
        x='POS',
        y='Count',
        hue='Label',
        palette=[config.COLORS['l0'], config.COLORS['l1']],
        ax=ax
    )
    
    ax.set_xlabel('POS Tag', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(title='Class')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'l0_pos_distribution': l0_pos,
        'l1_pos_distribution': l1_pos
    }


def plot_ljmpnik_pos_analysis(token_df, title="LJMPNIK POS Tag Analysis", 
                              save_path=None):
    """
    Analyze POS tags specifically for LJMPNIK tokens (where is_target=True).
    
    Args:
        token_df: Token DataFrame
        title: Plot title
        save_path: Optional save path
    """
    # Filter to only LJMPNIK tokens
    ljmpnik_tokens = token_df[token_df['is_target'] == True]
    
    if len(ljmpnik_tokens) == 0:
        print("⚠️ No LJMPNIK tokens found (is_target=True)")
        return None
    
    # Count POS tags
    pos_counts = ljmpnik_tokens['pos'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.VIZ_CONFIG['figure_sizes']['medium'])
    
    # Bar plot
    bars = ax.bar(
        pos_counts.index,
        pos_counts.values,
        color=config.COLORS['l1'],
        alpha=0.8,
        edgecolor='black'
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    ax.set_xlabel('POS Tag', fontweight='bold')
    ax.set_ylabel('Count of LJMPNIK Tokens', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary text
    total = len(ljmpnik_tokens)
    top_pos = pos_counts.index[0]
    top_pct = (pos_counts.iloc[0] / total) * 100
    
    summary_text = (
        f"Total LJMPNIK tokens: {total}\\n"
        f"Most common: {top_pos} ({top_pct:.1f}%)"
    )
    
    ax.text(
        0.98, 0.98,
        summary_text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return pos_counts


def plot_document_statistics(df, title="Document Statistics", save_path=None):
    """
    Show statistics at document level.
    
    Args:
        df: DataFrame with 'document_id' and 'label' columns
        title: Plot title
        save_path: Optional save path
    """
    # Group by document
    doc_stats = df.groupby('document_id').agg({
        'label': 'first',  # Document label (same for all sentences in doc)
        'sentence_id': 'count'  # Count sentences per document
    }).rename(columns={'sentence_id': 'num_sentences'})
    
    # Separate by label
    l0_docs = doc_stats[doc_stats['label'] == 0]['num_sentences']
    l1_docs = doc_stats[doc_stats['label'] == 1]['num_sentences']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=config.VIZ_CONFIG['figure_sizes']['wide'])
    
    # Histogram of sentences per document
    axes[0].hist(
        [l0_docs, l1_docs],
        bins=15,
        label=['Neutral Docs', 'LJMPNIK Docs'],
        color=[config.COLORS['l0'], config.COLORS['l1']],
        alpha=0.7,
        edgecolor='black'
    )
    axes[0].set_xlabel('Sentences per Document')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sentences per Document', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Summary stats
    summary_data = pd.DataFrame({
        'Metric': ['Documents', 'Avg Sentences/Doc', 'Std Sentences/Doc'],
        'Neutral': [
            len(l0_docs),
            l0_docs.mean(),
            l0_docs.std()
        ],
        'LJMPNIK': [
            len(l1_docs),
            l1_docs.mean(),
            l1_docs.std()
        ]
    })
    
    axes[1].axis('off')
    table = axes[1].table(
        cellText=summary_data.values,
        colLabels=summary_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(summary_data.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Document Statistics', fontweight='bold')
    
    fig.suptitle(title, fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return summary_data


def plot_dataset_overview(token_df, sentence_df, dataset_name='Dataset', save_path=None):
    """
    Comprehensive overview of a dataset.
    
    Args:
        token_df: Token-level DataFrame
        sentence_df: Sentence-level DataFrame
        dataset_name: Name for title
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} Overview', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Class distribution (sentences)
    sent_label_counts = sentence_df['label'].value_counts().sort_index()
    axes[0, 0].bar(
        ['Neutral', 'LJMPNIK'],
        sent_label_counts.values,
        color=[config.COLORS['l0'], config.COLORS['l1']],
        alpha=0.8,
        edgecolor='black'
    )
    for i, v in enumerate(sent_label_counts.values):
        axes[0, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    axes[0, 0].set_title('Sentence-Level Class Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Token count distribution
    if 'num_tokens' in sentence_df.columns:
        lengths = sentence_df['num_tokens']
    else:
        lengths = sentence_df['text'].str.split().str.len()
    
    l0_len = lengths[sentence_df['label'] == 0]
    l1_len = lengths[sentence_df['label'] == 1]
    
    axes[0, 1].hist(
        [l0_len, l1_len],
        bins=20,
        label=['Neutral', 'LJMPNIK'],
        color=[config.COLORS['l0'], config.COLORS['l1']],
        alpha=0.7,
        edgecolor='black'
    )
    axes[0, 1].set_title('Token Count Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Tokens per Sentence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. POS distribution (top 10)
    l0_pos = Counter(token_df[token_df['label'] == 0]['pos'])
    l1_pos = Counter(token_df[token_df['label'] == 1]['pos'])
    
    top_pos = [tag for tag, _ in l1_pos.most_common(10)]
    l0_counts = [l0_pos.get(tag, 0) for tag in top_pos]
    l1_counts = [l1_pos.get(tag, 0) for tag in top_pos]
    
    x = np.arange(len(top_pos))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, l0_counts, width, label='Neutral', 
                   color=config.COLORS['l0'], alpha=0.8, edgecolor='black')
    axes[1, 0].bar(x + width/2, l1_counts, width, label='LJMPNIK', 
                   color=config.COLORS['l1'], alpha=0.8, edgecolor='black')
    
    axes[1, 0].set_title('Top 10 POS Tags', fontweight='bold')
    axes[1, 0].set_xlabel('POS Tag')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(top_pos, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics table
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Documents',
            'Total Sentences',
            'Total Tokens',
            'Avg Tokens/Sentence',
            'LJMPNIK Tokens',
            'Unique POS Tags'
        ],
        'Value': [
            token_df['document_id'].nunique(),
            sentence_df.shape[0],
            token_df.shape[0],
            f"{token_df.groupby('sentence_id').size().mean():.1f}",
            (token_df['is_target'] == True).sum(),
            token_df['pos'].nunique()
        ]
    })
    
    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=summary_stats.values,
        colLabels=summary_stats.columns,
        cellLoc='left',
        loc='center',
        bbox=[0, 0.1, 1, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(summary_stats.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def print_data_summary(token_df, sentence_df, dataset_name='Dataset'):
    """
    # UNUSED FUNCTION! New one is print_dataset_stats()

    Print textual summary of dataset statistics.
    
    Args:
        token_df: Token DataFrame
        sentence_df: Sentence DataFrame
        dataset_name: Name for display
    """
    print(f"\\n{'='*60}")
    print(f"{dataset_name} SUMMARY")
    print(f"{'='*60}\\n")
    
    # Document level
    n_docs = token_df['document_id'].nunique()
    n_docs_l0 = sentence_df[sentence_df['label'] == 0]['document_id'].nunique()
    n_docs_l1 = sentence_df[sentence_df['label'] == 1]['document_id'].nunique()
    
    print(f"📁 DOCUMENTS:")
    print(f"   Total: {n_docs}")
    print(f"   Neutral: {n_docs_l0}")
    print(f"   LJMPNIK: {n_docs_l1}")
    
    # Sentence level
    n_sent = len(sentence_df)
    n_sent_l0 = (sentence_df['label'] == 0).sum()
    n_sent_l1 = (sentence_df['label'] == 1).sum()
    
    # Filter out context if exists
    if 'is_context' in sentence_df.columns:
        n_context = sentence_df['is_context'].sum()
        n_target = (~sentence_df['is_context']).sum()
        print(f"\\n📝 SENTENCES:")
        print(f"   Total: {n_sent}")
        print(f"   Target: {n_target} (Neutral: {n_sent_l0}, LJMPNIK: {n_sent_l1})")
        print(f"   Context: {n_context}")
    else:
        print(f"\\n📝 SENTENCES:")
        print(f"   Total: {n_sent}")
        print(f"   Neutral: {n_sent_l0}")
        print(f"   LJMPNIK: {n_sent_l1}")
        print(f"   Ratio L0:L1 = {n_sent_l0/n_sent_l1:.2f}:1")
    
    # Token level
    n_tokens = len(token_df)
    n_ljmpnik = (token_df['is_target'] == True).sum()
    avg_tokens_per_sent = token_df.groupby('sentence_id').size().mean()
    
    print(f"\\n🔤 TOKENS:")
    print(f"   Total: {n_tokens:,}")
    print(f"   LJMPNIK: {n_ljmpnik}")
    print(f"   Avg per sentence: {avg_tokens_per_sent:.1f}")
    
    # POS distribution
    top_pos = token_df['pos'].value_counts().head(5)
    print(f"\\n📊 TOP 5 POS TAGS:")
    for pos, count in top_pos.items():
        pct = (count / n_tokens) * 100
        print(f"   {pos}: {count:,} ({pct:.1f}%)")
    
    print(f"\\n{'='*60}\\n")

def print_dataset_stats(token_df, sentence_df, name="DATASET"):
    """
    Print basic statistics about the dataset.
    Robust version handles NaN values in boolean columns.

    Args:
        token_df: Token DataFrame
        sentence_df: Sentence DataFrame
        name: Name for display
    
    """
    print(f"\n{'='*20} {name} STATS {'='*20}")
    
    # Document level
    n_docs = sentence_df['document_id'].nunique()
    print(f"📄 DOCUMENTS: {n_docs}")

    # Sentence level
    n_sent = len(sentence_df)
    n_sent_l0 = (sentence_df['label'] == 0).sum()
    n_sent_l1 = (sentence_df['label'] == 1).sum()
    
    print(f"\n📝 SENTENCES:")
    print(f"   Total: {n_sent}")
    
    # --- OPRAVA ZDE ---
    if 'is_context' in sentence_df.columns:
        # 1. Ošetření NaN hodnot (vyplníme False) a převod na bool
        is_context_safe = sentence_df['is_context'].fillna(False).astype(bool)
        
        n_context = is_context_safe.sum()
        # Teď už můžeme použít vlnovku ~, protože je to bezpečný bool
        n_target = (~is_context_safe).sum()
        
        # Pro L0/L1 počítáme jen TARGET věty (bez kontextu)
        target_mask = ~is_context_safe
        target_l0 = ((sentence_df['label'] == 0) & target_mask).sum()
        target_l1 = ((sentence_df['label'] == 1) & target_mask).sum()

        print(f"   Target (Analyzed): {n_target}")
        print(f"     ├─ Neutral (L0): {target_l0}")
        print(f"     └─ LJMPNIK (L1): {target_l1}")
        print(f"   Context (Ignored): {n_context}")
    else:
        print(f"   Neutral: {n_sent_l0}")
        print(f"   LJMPNIK: {n_sent_l1}")
    
    # Token level
    n_tokens = len(token_df)
    # Zde také ošetříme label, kdyby tam byly NaN
    n_ljmpnik = (token_df['label'].fillna(0) == 1).sum()
    
    print(f"\n🔤 TOKENS:")
    print(f"   Total: {n_tokens}")
    print(f"   LJMPNIK anomalies: {n_ljmpnik}")
    print(f"   Global anomaly rate: {n_ljmpnik/n_tokens*100:.4f}%")
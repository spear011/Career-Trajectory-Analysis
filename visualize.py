"""
Network visualization script
Generates Sankey diagrams and transition matrix heatmap comparisons
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.network_visualizer import (
    build_transition_matrix,
    normalize_transition_matrix,
    compare_transition_matrices,
    prepare_sankey_data,
    get_period_ranges,
    aggregate_transitions_by_period,
    calculate_flow_statistics
)
from src.config import RESULTS_DIR


def create_sankey_diagram(sankey_data, period_name, results_dir, max_label_length=40):
    """
    Create Sankey diagram using plotly
    
    Args:
        sankey_data: Dictionary with source, target, value, labels
        period_name: Name of the period (for title)
        results_dir: Directory to save results
        max_label_length: Maximum length for labels
    """
    # Truncate labels
    labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
              for label in sankey_data['labels']]
    
    # Create color palette
    n_nodes = len(labels)
    colors = [f'rgba({int(255*i/n_nodes)}, {int(100+155*(1-i/n_nodes))}, {int(150+105*i/n_nodes)}, 0.8)' 
              for i in range(n_nodes)]
    
    # Link colors (semi-transparent versions)
    link_colors = [colors[src].replace('0.8', '0.3') for src in sankey_data['source']]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sankey_data['source'],
            target=sankey_data['target'],
            value=sankey_data['value'],
            color=link_colors
        )
    )])
    
    total_flow = sum(sankey_data['value'])
    
    fig.update_layout(
        title={
            'text': f"Occupational Mobility Flows: {period_name}<br><sub>Total transitions: {total_flow:,}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        font=dict(size=11),
        height=800,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    filepath = os.path.join(results_dir, f'sankey_{period_name.lower().replace(" ", "_")}.html')
    fig.write_html(filepath)
    print(f"✓ Saved Sankey diagram: {filepath}")
    
    return fig


def plot_transition_matrix_heatmap(matrix, period_name, results_dir, 
                                   normalize=True, cmap='YlOrRd', 
                                   figsize=(14, 12), max_label_length=35):
    """
    Plot transition matrix as heatmap
    
    Args:
        matrix: Transition matrix
        period_name: Name of the period
        results_dir: Directory to save results
        normalize: Whether to normalize (show probabilities)
        cmap: Colormap name
        figsize: Figure size
        max_label_length: Maximum length for labels
    """
    if normalize:
        matrix = normalize_transition_matrix(matrix)
        fmt = '.2%'
        cbar_label = 'Transition Probability'
    else:
        fmt = '.0f'
        cbar_label = 'Number of Transitions'
    
    # Truncate labels
    row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in matrix.index]
    col_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in matrix.columns]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for diagonal
    mask = np.zeros_like(matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Plot heatmap
    sns.heatmap(
        matrix, 
        annot=False,  # Too cluttered with annotations
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': cbar_label},
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor='lightgray',
        mask=mask if normalize else None  # Mask diagonal for normalized view
    )
    
    ax.set_xlabel('To Occupation', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Occupation', fontsize=12, fontweight='bold')
    ax.set_title(f'Occupation Transition Matrix: {period_name}', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    norm_str = 'normalized' if normalize else 'counts'
    filepath = os.path.join(results_dir, f'transition_matrix_{period_name.lower().replace(" ", "_")}_{norm_str}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved transition matrix: {filepath}")


def plot_transition_matrix_comparison(matrix_pre, matrix_during, matrix_post, 
                                     results_dir, max_label_length=35):
    """
    Create comparison plot of transition matrices across periods
    
    Args:
        matrix_pre: Pre-COVID matrix (normalized)
        matrix_during: During-COVID matrix (normalized)
        matrix_post: Post-COVID matrix (normalized)
        results_dir: Directory to save results
        max_label_length: Maximum length for labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    matrices = [matrix_pre, matrix_during, matrix_post]
    titles = ['Pre-COVID (2015-2019)', 'During COVID (2020-2021)', 'Post-COVID (2022-2023)']
    
    # Find common occupations
    common_occs = sorted(set(matrix_pre.index) & set(matrix_during.index) & set(matrix_post.index))
    
    # Truncate labels
    row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in common_occs]
    
    for ax, matrix, title in zip(axes, matrices, titles):
        # Align to common occupations
        matrix_aligned = matrix.reindex(index=common_occs, columns=common_occs, fill_value=0)
        
        # Mask diagonal
        mask = np.zeros_like(matrix_aligned, dtype=bool)
        np.fill_diagonal(mask, True)
        
        sns.heatmap(
            matrix_aligned,
            annot=False,
            cmap='YlOrRd',
            vmin=0,
            vmax=0.3,  # Cap for better visualization
            cbar_kws={'label': 'Transition Probability'},
            xticklabels=row_labels,
            yticklabels=row_labels,
            ax=ax,
            linewidths=0.3,
            linecolor='lightgray',
            mask=mask
        )
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('To Occupation', fontsize=10)
        ax.set_ylabel('From Occupation', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    filepath = os.path.join(results_dir, 'transition_matrices_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot: {filepath}")


def plot_transition_difference_heatmap(diff_matrix, period_comparison, 
                                      results_dir, max_label_length=35):
    """
    Plot difference between two transition matrices
    
    Args:
        diff_matrix: Difference matrix
        period_comparison: String describing comparison (e.g., "COVID vs Pre-COVID")
        results_dir: Directory to save results
        max_label_length: Maximum length for labels
    """
    # Truncate labels
    row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in diff_matrix.index]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Mask diagonal
    mask = np.zeros_like(diff_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Use diverging colormap
    vmax = max(abs(diff_matrix.min().min()), abs(diff_matrix.max().max()))
    
    sns.heatmap(
        diff_matrix,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': 'Probability Change'},
        xticklabels=row_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor='lightgray',
        mask=mask
    )
    
    ax.set_xlabel('To Occupation', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Occupation', fontsize=12, fontweight='bold')
    ax.set_title(f'Transition Probability Changes: {period_comparison}', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    filepath = os.path.join(results_dir, f'transition_difference_{period_comparison.lower().replace(" ", "_").replace("-", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved difference heatmap: {filepath}")


def create_all_network_visualizations(all_transitions_df, results_dir):
    """
    Create all network visualizations
    
    Args:
        all_transitions_df: All transitions dataframe
        results_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("CREATING NETWORK VISUALIZATIONS")
    print("="*80)
    
    periods = get_period_ranges()
    
    # Calculate and save flow statistics
    print("\nCalculating flow statistics...")
    flow_stats = calculate_flow_statistics(all_transitions_df)
    stats_path = os.path.join(results_dir, 'flow_statistics_by_period.csv')
    flow_stats.to_csv(stats_path, index=False)
    print(f"✓ Saved flow statistics: {stats_path}")
    print("\nFlow Statistics:")
    print(flow_stats.to_string(index=False))
    
    # Build transition matrices for each period
    print("\nBuilding transition matrices...")
    matrices = {}
    matrices_normalized = {}
    
    for period_name, (year_from, year_to) in periods.items():
        print(f"  Processing {period_name} ({year_from}-{year_to})...")
        matrix, occupations, total = build_transition_matrix(
            all_transitions_df, 
            period_filter=(year_from, year_to),
            top_n=15
        )
        matrices[period_name] = matrix
        matrices_normalized[period_name] = normalize_transition_matrix(matrix)
        print(f"    Total transitions: {total:,}")
    
    # Create Sankey diagrams
    print("\nGenerating Sankey diagrams...")
    for period_name, (year_from, year_to) in periods.items():
        sankey_data = prepare_sankey_data(
            all_transitions_df,
            period_filter=(year_from, year_to),
            top_n=12,
            min_flow=20
        )
        create_sankey_diagram(sankey_data, period_name.replace('_', ' ').title(), results_dir)
    
    # Create heatmaps for each period
    print("\nGenerating transition matrix heatmaps...")
    for period_name in periods.keys():
        plot_transition_matrix_heatmap(
            matrices[period_name],
            period_name.replace('_', ' ').title(),
            results_dir,
            normalize=True
        )
    
    # Create comparison plot
    print("\nGenerating comparison plot...")
    plot_transition_matrix_comparison(
        matrices_normalized['pre_covid'],
        matrices_normalized['covid'],
        matrices_normalized['post_covid'],
        results_dir
    )
    
    # Create difference heatmaps
    print("\nGenerating difference heatmaps...")
    
    # COVID vs Pre-COVID
    diff_covid_pre = compare_transition_matrices(
        matrices_normalized['pre_covid'],
        matrices_normalized['covid']
    )
    plot_transition_difference_heatmap(
        diff_covid_pre,
        "COVID vs Pre-COVID",
        results_dir
    )
    
    # Post-COVID vs COVID
    diff_post_covid = compare_transition_matrices(
        matrices_normalized['covid'],
        matrices_normalized['post_covid']
    )
    plot_transition_difference_heatmap(
        diff_post_covid,
        "Post-COVID vs COVID",
        results_dir
    )
    
    # Post-COVID vs Pre-COVID
    diff_post_pre = compare_transition_matrices(
        matrices_normalized['pre_covid'],
        matrices_normalized['post_covid']
    )
    plot_transition_difference_heatmap(
        diff_post_pre,
        "Post-COVID vs Pre-COVID",
        results_dir
    )
    
    print("\n" + "="*80)
    print("NETWORK VISUALIZATION COMPLETE")
    print("="*80)
    print("\n📊 Generated files:")
    print("  - flow_statistics_by_period.csv")
    print("  - sankey_*.html (interactive Sankey diagrams)")
    print("  - transition_matrix_*_normalized.png (heatmaps by period)")
    print("  - transition_matrices_comparison.png (side-by-side comparison)")
    print("  - transition_difference_*.png (difference heatmaps)")


if __name__ == "__main__":
    import sys
    
    # Load transitions data
    results_dir = RESULTS_DIR
    transitions_file = os.path.join(results_dir, 'all_transitions.csv')
    
    if not os.path.exists(transitions_file):
        print(f"Error: {transitions_file} not found.")
        print("Please run main.py first to generate transition data.")
        sys.exit(1)
    
    print("Loading transitions data...")
    all_transitions_df = pd.read_csv(transitions_file)
    print(f"Loaded {len(all_transitions_df):,} transitions")
    
    # Create visualizations
    create_all_network_visualizations(all_transitions_df, results_dir)
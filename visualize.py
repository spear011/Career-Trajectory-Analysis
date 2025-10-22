"""
Network visualization script
Generates Sankey diagrams and transition matrix heatmap comparisons
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.sankey import Sankey

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
    Create professional alluvial flow diagram
    
    Args:
        sankey_data: Dictionary with source, target, value, labels
        period_name: Name of the period (for title)
        results_dir: Directory to save results
        max_label_length: Maximum length for labels
    """
    flow_counts = sankey_data['flow_counts']
    
    # Check if data is empty
    if len(flow_counts) == 0:
        print(f"  ⚠ No data for {period_name}, skipping Sankey diagram")
        return
    
    # Sort by total flow volume and get top flows
    top_flows = flow_counts.nlargest(min(25, len(flow_counts)), 'count')
    
    # Remove self-loops
    top_flows = top_flows[top_flows['From_Occupation'] != top_flows['To_Occupation']]
    
    # Check if we have any valid transitions after filtering
    if len(top_flows) == 0:
        print(f"  ⚠ No valid transitions for {period_name} (all self-loops), skipping Sankey diagram")
        return
    
    # Get unique occupations from flows
    from_occs = top_flows['From_Occupation'].unique()
    to_occs = top_flows['To_Occupation'].unique()
    
    # Create position mappings
    from_positions = {occ: i for i, occ in enumerate(sorted(from_occs))}
    to_positions = {occ: i for i, occ in enumerate(sorted(to_occs))}
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Define x positions for source and target
    x_from = 0
    x_to = 10
    
    # Calculate heights for each occupation based on total flow
    from_heights = {}
    to_heights = {}
    
    for occ in from_occs:
        from_heights[occ] = top_flows[top_flows['From_Occupation'] == occ]['count'].sum()
    
    for occ in to_occs:
        to_heights[occ] = top_flows[top_flows['To_Occupation'] == occ]['count'].sum()
    
    # Normalize heights for better visualization
    max_height = max(max(from_heights.values()), max(to_heights.values()))
    height_scale = 30 / max_height  # Scale to fit in plot
    
    # Calculate y positions for FROM occupations
    from_y_positions = {}
    current_y = 0
    spacing = 0.5
    
    for occ in sorted(from_occs):
        height = from_heights[occ] * height_scale
        from_y_positions[occ] = (current_y, height)
        current_y += height + spacing
    
    # Calculate y positions for TO occupations  
    to_y_positions = {}
    current_y = 0
    
    for occ in sorted(to_occs):
        height = to_heights[occ] * height_scale
        to_y_positions[occ] = (current_y, height)
        current_y += height + spacing
    
    # Draw flows with bezier curves
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    import matplotlib.patches as mpatches
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_flows)))
    
    # Sort flows by count for better layering
    top_flows_sorted = top_flows.sort_values('count', ascending=True)
    
    for idx, (_, row) in enumerate(top_flows_sorted.iterrows()):
        from_occ = row['From_Occupation']
        to_occ = row['To_Occupation']
        flow_count = row['count']
        
        # Calculate flow height
        flow_height = flow_count * height_scale
        
        # Get positions
        from_y_base, from_total_height = from_y_positions[from_occ]
        to_y_base, to_total_height = to_y_positions[to_occ]
        
        # Calculate proportion of this flow in source and target
        from_prop = flow_count / from_heights[from_occ]
        to_prop = flow_count / to_heights[to_occ]
        
        # Calculate y offset based on previous flows
        from_offset = 0
        to_offset = 0
        
        for _, prev_row in top_flows_sorted.iterrows():
            if prev_row['From_Occupation'] == from_occ and prev_row.name < row.name:
                from_offset += prev_row['count'] * height_scale
            if prev_row['To_Occupation'] == to_occ and prev_row.name < row.name:
                to_offset += prev_row['count'] * height_scale
        
        # Define bezier curve points
        y_from_start = from_y_base + from_offset
        y_from_end = y_from_start + flow_height
        
        y_to_start = to_y_base + to_offset
        y_to_end = y_to_start + flow_height
        
        # Control points for smooth curve
        ctrl_x = (x_from + x_to) / 2
        
        # Create path for flow
        verts = [
            (x_from, y_from_start),  # Start bottom
            (ctrl_x, y_from_start),  # Control point 1
            (ctrl_x, y_to_start),    # Control point 2
            (x_to, y_to_start),      # End bottom
            (x_to, y_to_end),        # End top
            (ctrl_x, y_to_end),      # Control point 3
            (ctrl_x, y_from_end),    # Control point 4
            (x_from, y_from_end),    # Start top
            (x_from, y_from_start),  # Close
        ]
        
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor=colors[idx % len(colors)], 
                         edgecolor='none', alpha=0.6)
        ax.add_patch(patch)
    
    # Draw rectangles for occupations
    # FROM side
    for occ in sorted(from_occs):
        y_base, height = from_y_positions[occ]
        rect = mpatches.Rectangle((x_from - 0.3, y_base), 0.3, height,
                                  linewidth=1, edgecolor='black',
                                  facecolor='steelblue', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = occ[:max_label_length]
        if len(occ) > max_label_length:
            label += '...'
        ax.text(x_from - 0.4, y_base + height/2, label,
               ha='right', va='center', fontsize=9, fontweight='bold')
        
        # Add count
        count_text = f"{int(from_heights[occ]):,}"
        ax.text(x_from - 0.35, y_base + height/2, count_text,
               ha='right', va='center', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # TO side
    for occ in sorted(to_occs):
        y_base, height = to_y_positions[occ]
        rect = mpatches.Rectangle((x_to, y_base), 0.3, height,
                                  linewidth=1, edgecolor='black',
                                  facecolor='coral', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = occ[:max_label_length]
        if len(occ) > max_label_length:
            label += '...'
        ax.text(x_to + 0.4, y_base + height/2, label,
               ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add count
        count_text = f"{int(to_heights[occ]):,}"
        ax.text(x_to + 0.35, y_base + height/2, count_text,
               ha='left', va='center', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(-5, 15)
    ax.set_ylim(-1, max(current_y for _, (current_y, _) in from_y_positions.items()) + 1)
    
    ax.axis('off')
    
    # Add title and labels
    total_transitions = top_flows['count'].sum()
    title = f'Occupational Mobility Flows: {period_name}\n'
    title += f'Total transitions shown: {total_transitions:,} people across {len(top_flows)} pathways'
    
    ax.text(x_from - 0.5, ax.get_ylim()[1] + 1, 'FROM\n(Origin)', 
           ha='right', va='bottom', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='steelblue', alpha=0.3))
    
    ax.text(x_to + 0.5, ax.get_ylim()[1] + 1, 'TO\n(Destination)',
           ha='left', va='bottom', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='coral', alpha=0.3))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filepath = os.path.join(results_dir, f'sankey_{period_name.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved Sankey diagram: {filepath}")


def plot_transition_matrix_heatmap(matrix, period_name, results_dir, 
                                   normalize=True, cmap='YlOrRd', 
                                   figsize=(16, 14), max_label_length=40):
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
        matrix_plot = normalize_transition_matrix(matrix)
        fmt = '.1%'
        cbar_label = 'Transition Probability'
        # Remove diagonal for better visualization
        matrix_plot = matrix_plot.copy()
        np.fill_diagonal(matrix_plot.values, 0)
    else:
        matrix_plot = matrix.copy()
        fmt = '.0f'
        cbar_label = 'Number of Transitions'
    
    # Truncate labels
    row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in matrix_plot.index]
    col_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in matrix_plot.columns]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Only annotate cells with significant values
    annot_array = matrix_plot.values.copy()
    annot_mask = annot_array < (annot_array.max() * 0.05)  # Only show top 5%
    
    # Create annotation strings
    if normalize:
        annot = [[f'{val:.1%}' if not mask else '' 
                 for val, mask in zip(row, mask_row)]
                for row, mask_row in zip(annot_array, annot_mask)]
    else:
        annot = [[f'{int(val)}' if not mask and val > 0 else '' 
                 for val, mask in zip(row, mask_row)]
                for row, mask_row in zip(annot_array, annot_mask)]
    
    # Plot heatmap
    sns.heatmap(
        matrix_plot, 
        annot=annot,
        fmt='s',
        cmap=cmap,
        cbar_kws={'label': cbar_label, 'shrink': 0.8},
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    
    ax.set_xlabel('To Occupation', fontsize=13, fontweight='bold')
    ax.set_ylabel('From Occupation', fontsize=13, fontweight='bold')
    
    title_text = f'Occupation Transition Matrix: {period_name}'
    if normalize:
        title_text += '\n(Diagonal removed for clarity)'
    
    ax.set_title(title_text, fontsize=15, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    norm_str = 'normalized' if normalize else 'counts'
    filepath = os.path.join(results_dir, f'transition_matrix_{period_name.lower().replace(" ", "_")}_{norm_str}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved transition matrix: {filepath}")


def plot_transition_matrix_comparison(matrix_pre, matrix_during, matrix_post, 
                                     results_dir, max_label_length=30):
    """
    Create comparison plot of transition matrices across periods
    
    Args:
        matrix_pre: Pre-COVID matrix (normalized)
        matrix_during: During-COVID matrix (normalized)
        matrix_post: Post-COVID matrix (normalized)
        results_dir: Directory to save results
        max_label_length: Maximum length for labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    
    matrices = [matrix_pre, matrix_during, matrix_post]
    titles = ['Pre-COVID\n(2015-2019)', 'During COVID\n(2020-2021)', 'Post-COVID\n(2022-2023)']
    
    # Find common occupations
    common_occs = sorted(set(matrix_pre.index) & set(matrix_during.index) & set(matrix_post.index))
    
    # Truncate labels
    row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                  for label in common_occs]
    
    for ax, matrix, title in zip(axes, matrices, titles):
        # Align to common occupations
        matrix_aligned = matrix.reindex(index=common_occs, columns=common_occs, fill_value=0)
        
        # Remove diagonal
        matrix_plot = matrix_aligned.copy()
        np.fill_diagonal(matrix_plot.values, 0)
        
        sns.heatmap(
            matrix_plot,
            annot=False,
            cmap='YlOrRd',
            vmin=0,
            vmax=0.15,  # Cap for better visualization
            cbar_kws={'label': 'Transition\nProbability', 'shrink': 0.8},
            xticklabels=row_labels,
            yticklabels=row_labels,
            ax=ax,
            linewidths=0.3,
            linecolor='white',
            square=True
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('To Occupation', fontsize=11, fontweight='bold')
        ax.set_ylabel('From Occupation', fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Occupation Transition Matrices: Period Comparison\n(Diagonals removed)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(results_dir, 'transition_matrices_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved comparison plot: {filepath}")


def plot_transition_difference_heatmap(diff_matrix, period_comparison, 
                                      results_dir, max_label_length=30):
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
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Remove diagonal
    diff_plot = diff_matrix.copy()
    np.fill_diagonal(diff_plot.values, 0)
    
    # Use symmetric scale around zero
    vmax = max(abs(diff_plot.min().min()), abs(diff_plot.max().max()))
    vmax = min(vmax, 0.10)  # Cap at 10% for better visualization
    
    # Create custom diverging colormap
    sns.heatmap(
        diff_plot,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': 'Probability Change\n(Later - Earlier)', 'shrink': 0.8},
        xticklabels=row_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    
    ax.set_xlabel('To Occupation', fontsize=13, fontweight='bold')
    ax.set_ylabel('From Occupation', fontsize=13, fontweight='bold')
    ax.set_title(f'Transition Probability Changes: {period_comparison}\n' + 
                'Red = Increased transitions | Blue = Decreased transitions',
                fontsize=15, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    filepath = os.path.join(results_dir, f'transition_difference_{period_comparison.lower().replace(" ", "_").replace("-", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
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

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
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
            min_flow=5  # Lowered from 20 to 5
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
    print("  - flow_statistics_by_period.csv (통계 요약)")
    print("\n🔄 SANKEY DIAGRAMS (PNG):")
    print("  - sankey_pre_covid.png")
    print("  - sankey_covid.png")
    print("  - sankey_post_covid.png")
    print("\n🔥 HEATMAPS BY PERIOD:")
    print("  - transition_matrix_pre_covid_normalized.png")
    print("  - transition_matrix_covid_normalized.png")
    print("  - transition_matrix_post_covid_normalized.png")
    print("\n📊 COMPARISON:")
    print("  - transition_matrices_comparison.png (side-by-side)")
    print("\n🔀 DIFFERENCE HEATMAPS:")
    print("  - transition_difference_covid_vs_pre_covid.png")
    print("  - transition_difference_post_covid_vs_covid.png")
    print("  - transition_difference_post_covid_vs_pre_covid.png")


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
    results_dir = os.path.join(RESULTS_DIR, 'viz')
    create_all_network_visualizations(all_transitions_df, results_dir)
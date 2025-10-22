"""
Network Visualizer Module
Visualizes temporal occupation transition networks
"""

import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def load_networks(network_dir):
    """Load saved networks"""
    print("Loading networks...")
    
    with open(f'{network_dir}/networks_all.pkl', 'rb') as f:
        networks = pickle.load(f)
    
    stats_df = pd.read_csv(f'{network_dir}/network_statistics.csv')
    
    with open(f'{network_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  Loaded {len(networks)} temporal networks")
    print(f"  Year range: {metadata['years'][0]}-{metadata['years'][-1]}")
    
    return networks, stats_df, metadata


def plot_network_evolution(stats_df, output_dir):
    """Visualize network evolution over time"""
    print("\nPlotting network evolution...")
    
    # Remove NaN and sort
    stats_df = stats_df[~stats_df['year'].isna()].sort_values('year')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Occupation Transition Network Evolution', fontsize=16, fontweight='bold')
    
    # 1. Nodes and Edges
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(stats_df['year'], stats_df['n_nodes'], 'o-', color='#2ecc71', 
             linewidth=2, markersize=6, label='Nodes (Occupations)')
    ax1_twin.plot(stats_df['year'], stats_df['n_edges'], 's-', color='#3498db', 
                  linewidth=2, markersize=6, label='Edges (Transitions)')
    
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Number of Nodes', fontweight='bold', color='#2ecc71')
    ax1_twin.set_ylabel('Number of Edges', fontweight='bold', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1_twin.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_title('Network Size', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. Network Density
    ax2 = axes[0, 1]
    ax2.plot(stats_df['year'], stats_df['density'], 'o-', color='#e74c3c', 
             linewidth=2, markersize=6)
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Network Density', fontweight='bold')
    ax2.set_title('Network Density Over Time', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Total Transitions
    ax3 = axes[1, 0]
    ax3.bar(stats_df['year'], stats_df['total_transitions'], color='#9b59b6', alpha=0.7)
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.set_ylabel('Total Transitions', fontweight='bold')
    ax3.set_title('Total Job Transitions per Year', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Upward Mobility Ratio
    ax4 = axes[1, 1]
    ax4.plot(stats_df['year'], stats_df['upward_ratio'] * 100, 'o-', 
             color='#f39c12', linewidth=2, markersize=6)
    ax4.set_xlabel('Year', fontweight='bold')
    ax4.set_ylabel('Upward Mobility (%)', fontweight='bold')
    ax4.set_title('Upward Mobility Ratio (Wage-based)', fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax4.legend()
    
    plt.tight_layout()
    
    filepath = Path(output_dir) / 'network_evolution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def visualize_single_network(G, year, output_dir):
    """Visualize a single network"""
    print(f"\nVisualizing network for year {year}...")
    
    if G.number_of_nodes() == 0:
        print(f"  Skipping year {year} (empty network)")
        return
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Layout (spring layout)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node size: transition_volume
    node_sizes = [G.nodes[node].get('transition_volume', 1) * 100 for node in G.nodes()]
    
    # Node color: mean_wage
    node_colors = []
    for node in G.nodes():
        wage = G.nodes[node].get('mean_wage')
        if wage and wage != 'None':
            node_colors.append(wage)
        else:
            node_colors.append(0)
    
    # Edge width: transition_count
    edge_widths = [G.edges[edge]['transition_count'] * 0.5 for edge in G.edges()]
    
    # Edge color: upward_mobility
    edge_colors = []
    for u, v in G.edges():
        mobility = G.edges[u, v].get('upward_mobility')
        if mobility == 'True' or mobility is True:
            edge_colors.append('#2ecc71')  # green for upward
        elif mobility == 'False' or mobility is False:
            edge_colors.append('#e74c3c')  # red for downward
        else:
            edge_colors.append('#95a5a6')  # gray for unknown
    
    # Draw network
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                          alpha=0.6, arrows=True, arrowsize=15, 
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap='RdYlGn',
                                   alpha=0.8, ax=ax)
    
    # Node labels (abbreviated occupation names)
    labels = {}
    for node in G.nodes():
        name = G.nodes[node].get('occupation_name', node)
        if len(name) > 30:
            labels[node] = name[:27] + '...'
        else:
            labels[node] = name
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Colorbar
    if max(node_colors) > 0:
        plt.colorbar(nodes, ax=ax, label='Mean Annual Wage ($)', shrink=0.8)
    
    ax.set_title(f'Occupation Transition Network - {int(year)}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=3, label='Upward Mobility'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='Downward Mobility'),
        Line2D([0], [0], color='#95a5a6', linewidth=3, label='Unknown')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    filepath = Path(output_dir) / f'network_viz_{int(year)}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {filepath}")


def analyze_key_occupations(networks, top_n=10, output_dir=None):
    """Analyze top occupations by centrality"""
    print(f"\nAnalyzing top {top_n} occupations by centrality...")
    
    results = []
    
    for year, G in sorted(networks.items()):
        if pd.isna(year) or G.number_of_nodes() == 0:
            continue
        
        # Calculate centrality measures
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # PageRank (considering transition weights)
        weights = {(u, v): d['transition_count'] for u, v, d in G.edges(data=True)}
        nx.set_edge_attributes(G, weights, 'weight')
        pagerank = nx.pagerank(G, weight='weight')
        
        # Collect data for each occupation
        for node in G.nodes():
            results.append({
                'year': int(year),
                'occupation_code': node,
                'occupation_name': G.nodes[node].get('occupation_name', ''),
                'in_degree': in_degree[node],
                'out_degree': out_degree[node],
                'total_degree': in_degree[node] + out_degree[node],
                'pagerank': pagerank[node],
                'mean_wage': G.nodes[node].get('mean_wage'),
                'transition_volume': G.nodes[node].get('transition_volume', 0)
            })
    
    results_df = pd.DataFrame(results)
    
    # Extract top occupations for each year
    top_occupations = []
    
    for year in sorted(results_df['year'].unique()):
        year_data = results_df[results_df['year'] == year]
        top_year = year_data.nlargest(top_n, 'pagerank')
        top_occupations.append(top_year)
    
    top_occupations_df = pd.concat(top_occupations, ignore_index=True)
    
    # Save if output directory provided
    if output_dir:
        filepath = Path(output_dir) / 'occupation_centrality_analysis.csv'
        results_df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")
        
        filepath_top = Path(output_dir) / f'top_{top_n}_occupations_by_year.csv'
        top_occupations_df.to_csv(filepath_top, index=False)
        print(f"  Saved: {filepath_top}")
    
    return results_df, top_occupations_df


def create_summary_report(networks, stats_df, metadata, output_dir):
    """Generate summary report"""
    print("\nGenerating summary report...")
    
    stats_df_clean = stats_df[~stats_df['year'].isna()].sort_values('year')
    
    report_lines = [
        "="*80,
        "OCCUPATION TRANSITION NETWORK ANALYSIS REPORT",
        "="*80,
        "",
        "1. OVERVIEW",
        "-"*80,
        f"Total networks created: {len([y for y in networks.keys() if not pd.isna(y)])}",
        f"Year range: {int(stats_df_clean['year'].min())}-{int(stats_df_clean['year'].max())}",
        f"Total unique occupations: {metadata['total_nodes']}",
        f"Total transitions: {metadata['total_edges']}",
        "",
        "2. NETWORK STATISTICS SUMMARY",
        "-"*80,
        f"Average nodes per network: {stats_df_clean['n_nodes'].mean():.1f}",
        f"Average edges per network: {stats_df_clean['n_edges'].mean():.1f}",
        f"Average network density: {stats_df_clean['density'].mean():.4f}",
        f"Average transitions per year: {stats_df_clean['total_transitions'].mean():.1f}",
        "",
        "3. LARGEST NETWORKS (by number of edges)",
        "-"*80
    ]
    
    top_5_networks = stats_df_clean.nlargest(5, 'n_edges')
    for _, row in top_5_networks.iterrows():
        report_lines.append(
            f"Year {int(row['year'])}: {int(row['n_nodes'])} nodes, "
            f"{int(row['n_edges'])} edges, {row['density']:.4f} density"
        )
    
    report_lines.extend([
        "",
        "4. UPWARD MOBILITY TRENDS",
        "-"*80,
        f"Overall upward mobility rate: {stats_df_clean['upward_ratio'].mean()*100:.1f}%",
        f"Years with data: {len(stats_df_clean[stats_df_clean['upward_edges'] > 0])}",
        "",
        "5. DATA FILES GENERATED",
        "-"*80,
        "- network_YYYY.graphml: Individual year networks (GraphML format)",
        "- networks_all.pkl: All networks (Python pickle format)",
        "- network_statistics.csv: Statistical summary",
        "- occupation_centrality_analysis.csv: Centrality metrics",
        "- top_N_occupations_by_year.csv: Top occupations by PageRank",
        "- network_evolution.png: Temporal evolution plots",
        "- network_viz_YYYY.png: Individual network visualizations",
        "",
        "="*80,
        "ANALYSIS COMPLETE",
        "="*80
    ])
    
    report_text = "\n".join(report_lines)
    
    filepath = Path(output_dir) / 'ANALYSIS_REPORT.txt'
    with open(filepath, 'w') as f:
        f.write(report_text)
    
    print(f"  Saved: {filepath}")
    print("\n" + report_text)
    
    return report_text
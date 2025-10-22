"""
Visualize Occupation Transition Networks
Main script for visualizing and analyzing temporal networks
"""

from src.config import NETWORK_OUTPUT_DIR
from src.network_visualizer import (
    load_networks,
    plot_network_evolution,
    visualize_single_network,
    analyze_key_occupations,
    create_summary_report
)


def main():
    """Main execution function"""
    print("="*80)
    print("OCCUPATION TRANSITION NETWORK VISUALIZATION & ANALYSIS")
    print("="*80)
    
    output_dir = NETWORK_OUTPUT_DIR
    
    # 1. Load networks
    networks, stats_df, metadata = load_networks(output_dir)
    
    # 2. Plot temporal evolution
    plot_network_evolution(stats_df, output_dir)
    
    # 3. Visualize key years (2019, 2020, 2021)
    key_years = [2019, 2020, 2021]
    for year in key_years:
        if year in networks:
            visualize_single_network(networks[year], year, output_dir)
    
    # 4. Analyze occupation centrality
    centrality_df, top_occupations_df = analyze_key_occupations(
        networks, top_n=10, output_dir=output_dir
    )
    
    # 5. Generate summary report
    report = create_summary_report(networks, stats_df, metadata, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION & ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    
    return networks, stats_df, centrality_df


if __name__ == '__main__':
    networks, stats, centrality = main()
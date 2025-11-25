"""
Build Occupation Transition Networks with Extended Analysis
Complete pipeline: construction + metrics + visualization
Updated to include occupation-level time series and centrality analysis
"""

import os
import pandas as pd
from pathlib import Path

from src.network import NetworkBuilder, NetworkVisualizer
from src.utils import get_config
from src.benchmark_utils import PipelineBenchmark


def main():
    """Main execution function"""
    benchmark = PipelineBenchmark()
    config = get_config()
    
    DATA_DIR = config.results_dir
    NETWORK_OUTPUT_DIR = config.network_output_dir
    STUDY_START_YEAR = config.study_start_year
    STUDY_END_YEAR = config.study_end_year
    OCCUPATION_COL = config.occupation_column
    
    print("="*80)
    print("OCCUPATION TRANSITION NETWORK PIPELINE")
    print("="*80)
    print(f"Study Period: {STUDY_START_YEAR}-{STUDY_END_YEAR}")
    print("Phases:")
    print("  - Pre-Pandemic: 2017-2019 (Baseline mobility patterns)")
    print("  - COVID Shock: 2020-2021 (Labor market volatility)")
    print("  - Post-Pandemic Recovery: 2022-2024 (Persistence of changes)")
    print("="*80)
    
    # ========================================
    # Stage 1: Setup
    # ========================================
    benchmark.start_stage("Setup")
    os.makedirs(NETWORK_OUTPUT_DIR, exist_ok=True)
    
    builder = NetworkBuilder(
        study_start_year=STUDY_START_YEAR,
        study_end_year=STUDY_END_YEAR,
        occupation_col=OCCUPATION_COL
    )
    visualizer = NetworkVisualizer()
    benchmark.end_stage("Setup")
    
    # ========================================
    # Stage 2: Load data
    # ========================================
    benchmark.start_stage("Data Loading")
    print("\nLoading data...")
    
    trajectory_path = os.path.join(DATA_DIR, 'career_trajectories.parquet')
    trajectory_df = pd.read_parquet(trajectory_path)
    print(f"  Trajectory records: {len(trajectory_df):,}")
    print(f"  Unique users: {trajectory_df['ID'].nunique():,}")
    
    if 'annual_state_wage' in trajectory_df.columns:
        print(f"  Wage data: Available in trajectory (annual_state_wage column)")
    else:
        print(f"  Wage data: Not available")
    
    benchmark.end_stage("Data Loading")
    
    # ========================================
    # Stage 3: Build networks
    # ========================================
    benchmark.start_stage("Network Construction")
    networks, stats_df, network_output_path, transitions_df = builder.build_all(
        trajectory_df, None, NETWORK_OUTPUT_DIR
    )
    benchmark.end_stage("Network Construction")
    
    # ========================================
    # Stage 4: Calculate Extended Metrics
    # ========================================
    benchmark.start_stage("Extended Metrics Calculation")
    print("\n" + "="*80)
    print("CALCULATING EXTENDED NETWORK METRICS")
    print("="*80)
    
    # Occupation-level time series
    occ_time_series = builder.calculate_occupation_time_series(
        transitions_df, networks
    )
    
    # Node centrality metrics
    node_centrality = builder.calculate_node_centrality(networks)
    
    # Period comparison
    periods = {
        'Pre-COVID': (2017, 2019),
        'COVID': (2020, 2021),
        'Post-COVID': (2022, 2024)
    }
    period_comparison = builder.analyze_period_comparison(
        occ_time_series, periods
    )
    
    # Rank occupations by different metrics
    rankings = {}
    for metric in ['pagerank', 'betweenness', 'in_degree', 'out_degree']:
        rankings[metric] = builder.rank_occupations(
            node_centrality, occ_time_series, metric=metric, top_n=10
        )
    
    # Save extended metrics
    builder.save_extended_metrics(
        occ_time_series, node_centrality, period_comparison,
        rankings, Path(NETWORK_OUTPUT_DIR)
    )
    
    benchmark.end_stage("Extended Metrics Calculation")
    
    # ========================================
    # Stage 5: Save transitions
    # ========================================
    benchmark.start_stage("Save Transitions")
    print("\nSaving transition data...")
    
    transitions_path = os.path.join(NETWORK_OUTPUT_DIR, 'all_transitions.csv')
    transitions_df.to_csv(transitions_path, index=False)
    print(f"  ✓ Saved transitions: {transitions_path}")
    
    benchmark.end_stage("Save Transitions")
    
    # ========================================
    # Stage 6: Create visualizations
    # ========================================
    benchmark.start_stage("Visualizations")
    viz_output_dir = os.path.join(NETWORK_OUTPUT_DIR, 'viz')
    visualizer.visualize_all(transitions_df, viz_output_dir)
    benchmark.end_stage("Visualizations")
    
    # ========================================
    # Stage 7: Print Summary
    # ========================================
    print("\n" + "="*80)
    print("NETWORK PIPELINE COMPLETE")
    print("="*80)
    print(f"\nNetwork Output: {network_output_path}")
    print(f"Number of temporal networks: {len([y for y in networks.keys() if not pd.isna(y)])}")
    valid_years = [y for y in networks.keys() if not pd.isna(y)]
    if valid_years:
        print(f"Year range: {str(min(valid_years))}-{str(max(valid_years))}")
    
    print(f"\nVisualization Output: {viz_output_dir}")
    
    print("\n📊 GENERATED FILES:")
    print("\n[Network Structure]")
    print("  - network_YYYY.graphml (one per year)")
    print("  - networks_all.pkl (all networks)")
    print("  - network_statistics.csv (graph-level metrics)")
    print("  - metadata.json")
    
    print("\n[Transitions]")
    print("  - all_transitions.csv (raw transition records)")
    
    print("\n[Extended Metrics] ⭐ NEW")
    print("  - occupation_time_series.csv (inflow/outflow rates per occupation)")
    print("  - node_centrality.csv (SNA centrality metrics)")
    print("  - period_comparison.csv (Pre/COVID/Post comparison)")
    print("  - ranking_by_pagerank.csv (top occupations by PageRank)")
    print("  - ranking_by_betweenness.csv (top occupations by betweenness)")
    print("  - ranking_by_in_degree.csv (top occupations by in-degree)")
    print("  - ranking_by_out_degree.csv (top occupations by out-degree)")
    
    print("\n[Visualizations]")
    print("  - flow_statistics_by_period.csv")
    print("  - sankey_pre_covid.png")
    print("  - sankey_covid.png")
    print("  - sankey_post_covid.png")
    print("  - transition_matrix_*.png (heatmaps)")
    print("  - transition_difference_*.png")
    
    print("\n" + "="*80)
    print("EXTENDED METRICS SUMMARY")
    print("="*80)
    
    # Print occupation time series summary
    print("\nOccupation Time Series:")
    print(f"  Total occupation-windows: {len(occ_time_series):,}")
    print(f"  Unique occupations: {occ_time_series['occupation'].nunique()}")
    print(f"  Windows: {occ_time_series['window'].nunique()}")
    
    # Print top occupations by employment
    print("\nTop 5 Occupations by Employment (latest window):")
    latest_window = occ_time_series['window'].max()
    top_emp = occ_time_series[occ_time_series['window'] == latest_window].nlargest(5, 'employment_count')
    for i, row in top_emp.iterrows():
        print(f"  {row['occupation']}: {row['employment_count']:,} workers, "
              f"inflow_rate={row['inflow_rate']:.1f}%, "
              f"outflow_rate={row['outflow_rate']:.1f}%")
    
    # Print period comparison summary
    print("\nPeriod Comparison:")
    for period in ['Pre-COVID', 'COVID', 'Post-COVID']:
        period_data = period_comparison[period_comparison['period'] == period]
        if len(period_data) > 0:
            avg_inflow = period_data['inflow_rate'].mean()
            avg_outflow = period_data['outflow_rate'].mean()
            avg_upward = period_data['upward_move_rate'].mean()
            print(f"  {period}: avg_inflow={avg_inflow:.1f}%, "
                  f"avg_outflow={avg_outflow:.1f}%, "
                  f"avg_upward_move={avg_upward:.1f}%")
    
    print("="*80)
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(NETWORK_OUTPUT_DIR, 'benchmark_report.json'))
    
    return networks, stats_df, transitions_df, occ_time_series, node_centrality


if __name__ == '__main__':
    results = main()
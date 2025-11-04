"""
Build Occupation Transition Networks
Complete pipeline: construction + visualization
Updated to use career_trajectories.parquet with embedded wage data
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
    
    # Stage 1: Setup
    benchmark.start_stage("Setup")
    os.makedirs(NETWORK_OUTPUT_DIR, exist_ok=True)
    
    builder = NetworkBuilder(
        study_start_year=STUDY_START_YEAR,
        study_end_year=STUDY_END_YEAR,
        occupation_col=OCCUPATION_COL
    )
    visualizer = NetworkVisualizer()
    benchmark.end_stage("Setup")
    
    # Stage 2: Load data
    benchmark.start_stage("Data Loading")
    print("\nLoading data...")
    
    # Load career trajectories (contains wage data)
    trajectory_path = os.path.join(DATA_DIR, 'career_trajectories.parquet')
    trajectory_df = pd.read_parquet(trajectory_path)
    print(f"  Trajectory records: {len(trajectory_df):,}")
    print(f"  Unique users: {trajectory_df['ID'].nunique():,}")
    
    # Check if wage data exists in trajectory
    if 'annual_state_wage' in trajectory_df.columns:
        print(f"  Wage data: Available in trajectory (annual_state_wage column)")
    else:
        print(f"  Wage data: Not available")
    
    benchmark.end_stage("Data Loading")
    
    # Stage 3: Build networks
    benchmark.start_stage("Network Construction")
    networks, stats_df, network_output_path, transitions_df = builder.build_all(
        trajectory_df, None, NETWORK_OUTPUT_DIR
    )
    benchmark.end_stage("Network Construction")
    
    # Stage 4: Save transitions
    benchmark.start_stage("Save Transitions")
    print("\nSaving transition data...")
    
    transitions_path = os.path.join(NETWORK_OUTPUT_DIR, 'all_transitions.csv')
    transitions_df.to_csv(transitions_path, index=False)
    print(f"  ✓ Saved transitions: {transitions_path}")
    
    benchmark.end_stage("Save Transitions")
    
    # Stage 5: Create visualizations
    benchmark.start_stage("Visualizations")
    viz_output_dir = os.path.join(NETWORK_OUTPUT_DIR, 'viz')
    visualizer.visualize_all(transitions_df, viz_output_dir)
    benchmark.end_stage("Visualizations")
    
    # Print summary
    print("\n" + "="*80)
    print("NETWORK PIPELINE COMPLETE")
    print("="*80)
    print(f"\nNetwork Output: {network_output_path}")
    print(f"Number of temporal networks: {len([y for y in networks.keys() if not pd.isna(y)])}")
    valid_years = [y for y in networks.keys() if not pd.isna(y)]
    if valid_years:
        print(f"Year range: {int(min(valid_years))}-{int(max(valid_years))}")
    
    print(f"\nVisualization Output: {viz_output_dir}")
    print("\n📊 GENERATED FILES:")
    print("\n[Network Structure]")
    print("  - network_YYYY.graphml (one per year)")
    print("  - networks_all.pkl (all networks)")
    print("  - network_statistics.csv")
    print("  - metadata.json")
    print("\n[Transitions]")
    print("  - all_transitions.csv")
    print("\n[Visualizations]")
    print("  - flow_statistics_by_period.csv")
    print("  - sankey_pre_covid.png")
    print("  - sankey_covid.png")
    print("  - sankey_post_covid.png")
    print("  - transition_matrix_*.png (heatmaps)")
    print("  - transition_difference_*.png")
    print("="*80)
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(NETWORK_OUTPUT_DIR, 'benchmark_report.json'))
    
    return networks, stats_df, transitions_df


if __name__ == '__main__':
    networks, stats, transitions = main()
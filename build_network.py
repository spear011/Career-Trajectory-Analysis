"""
Build Occupation Transition Networks
Complete pipeline: construction + visualization
"""

import os
import pandas as pd
from pathlib import Path

from src.network import NetworkBuilder, NetworkVisualizer

from src.utils import get_config
config = get_config()
DATA_DIR = config.paths.dataset_base_dir
NETWORK_OUTPUT_DIR = config.paths.network_output_dir
STUDY_START_YEAR = config.date_ranges.study_start_year
STUDY_END_YEAR = config.date_ranges.study_end_year
NETWORK_OCCUPATION_COLUMN = config.network_analysis.occupation_column
NETWORK_OCCUPATION_NAME_COLUMN = config.network_analysis.occupation_name_column


from src.benchmark_utils import PipelineBenchmark


def main():
    """Main execution function"""
    benchmark = PipelineBenchmark()
    
    print("="*80)
    print("OCCUPATION TRANSITION NETWORK PIPELINE")
    print("="*80)
    print(f"Study Period: {STUDY_START_YEAR}-{STUDY_END_YEAR}")
    print("Phases:")
    print("  - Pre-Pandemic: 2017-2019 (Baseline mobility patterns)")
    print("  - COVID Shock: Mar 2020-2021 (Labor market volatility)")
    print("  - Post-Pandemic Recovery: 2022-2024 (Persistence of changes)")
    print("="*80)
    
    # Stage 1: Setup
    benchmark.start_stage("Setup")
    os.makedirs(NETWORK_OUTPUT_DIR, exist_ok=True)
    
    # Initialize builder and visualizer
    builder = NetworkBuilder(
        study_start_year=STUDY_START_YEAR,
        study_end_year=STUDY_END_YEAR,
        occupation_col=NETWORK_OCCUPATION_COLUMN,
        occupation_name_col=NETWORK_OCCUPATION_NAME_COLUMN
    )
    visualizer = NetworkVisualizer()
    benchmark.end_stage("Setup")
    
    # Stage 2: Load data
    benchmark.start_stage("Data Loading")
    print("\nLoading data...")
    job_df = pd.read_csv(os.path.join(DATA_DIR, 'job_group_0.csv'))
    wage_df = pd.read_csv(os.path.join(DATA_DIR, 'wage_interpolated_1999_2022_soc2019_unique.csv'))
    
    print(f"  Job records: {len(job_df):,}")
    print(f"  Wage records: {len(wage_df):,}")
    benchmark.end_stage("Data Loading")
    
    # Stage 3: Build networks
    benchmark.start_stage("Network Construction")
    networks, stats_df, network_output_path = builder.build_all(job_df, wage_df, NETWORK_OUTPUT_DIR)
    benchmark.end_stage("Network Construction")
    
    # Stage 4: Prepare transition data for visualization
    benchmark.start_stage("Transition Data Preparation")
    print("\nPreparing transition data for visualization...")
    
    # We need to reload transitions with additional fields for visualization
    job_df_prep = builder.prepare_job_data(job_df)
    user_paths = builder.build_user_career_paths(job_df_prep)
    transitions_df = builder.extract_transitions(user_paths)
    
    # Enrich with additional fields needed for visualization
    # Add From/To occupation names, industries, states
    job_df_prep_indexed = job_df_prep.set_index('ID')
    
    transitions_enriched = []
    for _, row in transitions_df.iterrows():
        user_id = row['user_id']
        
        transitions_enriched.append({
            'ID': user_id,
            'Year_From': row['from_year'],
            'Year_To': row['to_year'],
            'From_Occupation': row['from_occupation_name'],
            'To_Occupation': row['to_occupation_name'],
            'From_Industry': 'Unknown',  # We don't have this in current structure
            'To_Industry': 'Unknown',
            'From_State': 'Unknown',
            'To_State': 'Unknown',
            'Occupation_Changed': row['from_occupation'] != row['to_occupation'],
            'Industry_Changed': False,
            'State_Changed': False
        })
    
    transitions_viz_df = pd.DataFrame(transitions_enriched)
    
    # Save transitions for future use
    transitions_path = os.path.join(NETWORK_OUTPUT_DIR, 'all_transitions.csv')
    transitions_viz_df.to_csv(transitions_path, index=False)
    print(f"  ✓ Saved transitions: {transitions_path}")
    
    benchmark.end_stage("Transition Data Preparation")
    
    # Stage 5: Create visualizations
    benchmark.start_stage("Visualizations")
    viz_output_dir = os.path.join(NETWORK_OUTPUT_DIR, 'viz')
    visualizer.visualize_all(transitions_viz_df, viz_output_dir)
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
    print("  - transition_matrices_comparison.png")
    print("  - transition_difference_*.png")
    print("="*80)
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(NETWORK_OUTPUT_DIR, 'benchmark_report.json'))
    
    return networks, stats_df


if __name__ == '__main__':
    networks, stats = main()
"""
Integrated Career Trajectory Analysis Pipeline
Combines Dataset Construction preprocessing with network analysis and visualization
"""
import os
import pandas as pd
from pathlib import Path

from src.preprocessing import DataPreprocessor
from src.enrichment import TrajectoryEnricher
from src.network import NetworkBuilder, NetworkVisualizer
from src.utils import get_config, get_dataset_file_path, ensure_dir, get_occupation_file_path, get_enrich_file_path
from src.benchmark_utils import PipelineBenchmark


def main():
    """Execute complete integrated pipeline"""
    benchmark = PipelineBenchmark()
    config = get_config()
    
    print("="*80)
    print("INTEGRATED CAREER TRAJECTORY ANALYSIS PIPELINE")
    print("="*80)
    print(f"Data directory: {config.data_dir}")
    print(f"Results directory: {config.results_dir}")
    print("\nPhases:")
    print("  1. Data Preprocessing (Dataset Construction)")
    print("  2. Trajectory Enrichment")
    print("  3. Network Analysis")
    print("  4. Visualization")
    print("="*80)
    
    # ========================================
    # STAGE 1: Data Preprocessing
    # ========================================
    benchmark.start_stage("Data Preprocessing")
    print("\n" + "="*80)
    print("STAGE 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor(verbose=True)
    
    # Get data paths from config
    job_path = get_dataset_file_path('job', 0)
    edu_path = get_dataset_file_path('education', 0)
    occ_path = get_occupation_file_path(0)
    
    print(f"\nData files:")
    print(f"  Job: {job_path}")
    print(f"  Education: {edu_path}")
    print(f"  Occupation: {occ_path}")

    # Run preprocessing pipeline
    trajectory_df = preprocessor.run_full_pipeline(
        job_path=job_path,
        edu_path=edu_path,
        occ_path=occ_path,
        start_year=1999,
        end_year=2022,
        trajectory_years=5
    )
    
    linear_job_df = preprocessor.linear_job_df
    edu_df = preprocessor.edu_df
    
    benchmark.end_stage("Data Preprocessing")
    
    # ========================================
    # STAGE 2: Trajectory Enrichment
    # ========================================
    benchmark.start_stage("Trajectory Enrichment")
    print("\n" + "="*80)
    print("STAGE 2: TRAJECTORY ENRICHMENT")
    print("="*80)

    attribute_path = get_dataset_file_path('attribute', 0)
    attribute_df = pd.read_csv(attribute_path)

    gdp_path = get_enrich_file_path('gdp')
    gdp_df = pd.read_csv(gdp_path)

    wage_path = get_enrich_file_path('wage')
    wage_df = pd.read_csv(wage_path)
    
    enricher = TrajectoryEnricher(trajectory_df, linear_job_df)
    enricher.estimate_birth_year(edu_df)
    
    enricher.add_demographics(attribute_df)
    enricher.add_state_gdp(gdp_df)
    enricher.add_occupational_wage(wage_df)

    enricher.add_upward_mobility(wage_df)
    enricher.add_job_change_types()

    enricher.final_cleanup(top_code_percentile=95)
    
    final_trajectory_df = enricher.get_trajectory_df()
    
    # Save enriched data
    ensure_dir(config.results_dir)
    output_path = os.path.join(config.results_dir, 'career_trajectories.parquet')
    enricher.save(output_path)
    
    benchmark.end_stage("Trajectory Enrichment")
    
    # ========================================
    # STAGE 3: Network Analysis
    # ========================================
    benchmark.start_stage("Network Analysis")
    print("\n" + "="*80)
    print("STAGE 3: NETWORK ANALYSIS")
    print("="*80)
    
    # Load job data for network construction
    job_df_full = pd.read_csv(
        job_path, 
        compression='gzip' if job_path.endswith('.gz') else None
    )
    
    network_output_dir = config.network_output_dir
    ensure_dir(network_output_dir)
    
    builder = NetworkBuilder(
        study_start_year=config.study_start_year,
        study_end_year=config.study_end_year,
        occupation_col=config.occupation_column,
        occupation_name_col=config.occupation_name_column
    )
    
    job_df_prep = builder.prepare_job_data(job_df_full)
    user_paths = builder.build_user_career_paths(job_df_prep)
    transitions_df = builder.extract_transitions(user_paths)
    
    # Build networks
    print("\nBuilding occupation transition networks...")
    networks = {}
    for year in range(config.study_start_year, config.study_end_year + 1):
        year_transitions = transitions_df[
            (transitions_df['from_year'] == year) |
            (transitions_df['to_year'] == year)
        ]
        
        if len(year_transitions) > 0:
            G = builder._build_single_network(year_transitions, year)
            networks[year] = G
    
    builder._save_networks(networks, network_output_dir)
    
    stats_df = builder._compute_network_statistics(networks)
    stats_df.to_csv(
        os.path.join(network_output_dir, 'network_statistics.csv'), 
        index=False
    )
    
    transitions_df.to_csv(
        os.path.join(network_output_dir, 'all_transitions.csv'), 
        index=False
    )
    
    print(f"\n  Network files saved to: {network_output_dir}")
    print(f"  Number of yearly networks: {len(networks)}")
    
    benchmark.end_stage("Network Analysis")
    
    # ========================================
    # STAGE 4: Visualization
    # ========================================
    benchmark.start_stage("Visualization")
    print("\n" + "="*80)
    print("STAGE 4: VISUALIZATION")
    print("="*80)
    
    visualizer = NetworkVisualizer()
    
    # Prepare transitions for visualization
    transitions_enriched = []
    for _, row in transitions_df.iterrows():
        transitions_enriched.append({
            'ID': row['user_id'],
            'Year_From': row['from_year'],
            'Year_To': row['to_year'],
            'From_Occupation': row['from_occupation_name'],
            'To_Occupation': row['to_occupation_name'],
            'From_Industry': 'Unknown',
            'To_Industry': 'Unknown',
            'From_State': 'Unknown',
            'To_State': 'Unknown',
            'Occupation_Changed': row['from_occupation'] != row['to_occupation'],
            'Industry_Changed': False,
            'State_Changed': False
        })
    
    transitions_viz_df = pd.DataFrame(transitions_enriched)
    
    viz_output_dir = os.path.join(network_output_dir, 'viz')
    ensure_dir(viz_output_dir)
    
    visualizer.visualize_all(transitions_viz_df, viz_output_dir)
    
    print(f"\n  Visualizations saved to: {viz_output_dir}")
    
    benchmark.end_stage("Visualization")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    
    print("\n📊 OUTPUTS:")
    print(f"\n[Trajectory Data]")
    print(f"  - {output_path}")
    print(f"  - Shape: {final_trajectory_df.shape}")
    
    print(f"\n[Network Structure]")
    print(f"  - {network_output_dir}/")
    print(f"  - Network files: network_YYYY.graphml")
    print(f"  - Statistics: network_statistics.csv")
    print(f"  - Transitions: all_transitions.csv")
    
    print(f"\n[Visualizations]")
    print(f"  - {viz_output_dir}/")
    print(f"  - Sankey diagrams by period")
    print(f"  - Transition matrices")
    print(f"  - Flow statistics")
    
    benchmark.print_report()
    benchmark.save_report(
        os.path.join(config.results_dir, 'benchmark_report.json')
    )
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
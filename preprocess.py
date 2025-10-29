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
    # END
    # ========================================

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    benchmark.report()
"""
Integrated Career Trajectory Analysis Pipeline
Combines Dataset Construction preprocessing with network analysis and visualization
MODIFIED: Process groups 0-19 and merge into single parquet file
"""
import os
import pandas as pd
from pathlib import Path

from src.preprocessing import DataPreprocessor
from src.enrichment import TrajectoryEnricher
from src.utils import get_config, get_dataset_file_path, ensure_dir, get_occupation_file_path, get_enrich_file_path
from src.benchmark_utils import PipelineBenchmark


def main():
    """Execute complete integrated pipeline for all groups"""
    benchmark = PipelineBenchmark()
    config = get_config()

    start_year = config.analysis_start_year - 1
    end_year = config.analysis_end_year
    
    group_nums = range(0, 20)

    print("="*80)
    print("INTEGRATED CAREER TRAJECTORY ANALYSIS PIPELINE")
    print("="*80)
    print(f"Data directory: {config.data_dir}")
    print(f"Results directory: {config.results_dir}")
    print(f"Processing groups: {list(group_nums)}")
    print("\nPhases:")
    print("  1. Data Preprocessing (Dataset Construction)")
    print("  2. Trajectory Enrichment")
    print("="*80)
    
    # Load shared enrichment data once
    gdp_path = get_enrich_file_path('gdp')
    gdp_df = pd.read_csv(gdp_path)
    wage_path = get_enrich_file_path('wage')
    wage_df = pd.read_csv(wage_path)
    
    all_trajectories = []
    
    # ========================================
    # PROCESS EACH GROUP
    # ========================================
    for group_num in group_nums:
        print("\n" + "="*80)
        print(f"PROCESSING GROUP {group_num}")
        print("="*80)
        
        try:
            # ========================================
            # STAGE 1: Data Preprocessing
            # ========================================
            benchmark.start_stage(f"Group {group_num} - Preprocessing")
            print("\n" + "-"*80)
            print(f"STAGE 1: DATA PREPROCESSING - GROUP {group_num}")
            print("-"*80)
            
            preprocessor = DataPreprocessor(verbose=True)
            
            job_path = get_dataset_file_path('job', group_num)
            edu_path = get_dataset_file_path('education', group_num)
            occ_path = get_occupation_file_path(group_num)
            
            if not os.path.exists(job_path) or not os.path.exists(edu_path):
                print(f"Skipping group {group_num}: Missing data files")
                benchmark.end_stage(f"Group {group_num} - Preprocessing")
                continue
            
            print(f"Job: {job_path}")
            print(f"Education: {edu_path}")
            print(f"Occupation: {occ_path if os.path.exists(occ_path) else 'N/A'}")

            trajectory_df = preprocessor.run_full_pipeline(
                job_path=job_path,
                edu_path=edu_path,
                occ_path=occ_path if os.path.exists(occ_path) else None,
                start_year=start_year,
                end_year=end_year,
            )
            
            linear_job_df = preprocessor.linear_job_df
            edu_df = preprocessor.edu_df
            
            benchmark.end_stage(f"Group {group_num} - Preprocessing")
            
            # ========================================
            # STAGE 2: Trajectory Enrichment
            # ========================================
            benchmark.start_stage(f"Group {group_num} - Enrichment")
            print("\n" + "-"*80)
            print(f"STAGE 2: TRAJECTORY ENRICHMENT - GROUP {group_num}")
            print("-"*80)

            attribute_path = get_dataset_file_path('attribute', group_num)
            if not os.path.exists(attribute_path):
                print(f"Warning: Missing attribute file for group {group_num}")
                attribute_df = None
            else:
                attribute_df = pd.read_csv(attribute_path)
            
            enricher = TrajectoryEnricher(trajectory_df, linear_job_df)
            enricher.estimate_birth_year(edu_df)
            
            if attribute_df is not None:
                enricher.add_demographics(attribute_df)
            enricher.add_state_gdp(gdp_df)
            enricher.add_occupational_wage(wage_df)
            enricher.add_upward_mobility(wage_df)
            enricher.add_job_change_types()
            enricher.final_cleanup(top_code_percentile=95)
            
            group_trajectory_df = enricher.get_trajectory_df()
            
            print(f"Group {group_num} processed: {len(group_trajectory_df):,} trajectories")
            all_trajectories.append(group_trajectory_df)
            
            benchmark.end_stage(f"Group {group_num} - Enrichment")
            
        except Exception as e:
            print(f"Error processing group {group_num}: {str(e)}")
            benchmark.end_stage(f"Group {group_num} - Preprocessing")
            continue
    
    # ========================================
    # MERGE ALL GROUPS
    # ========================================
    benchmark.start_stage("Merging All Groups")
    print("\n" + "="*80)
    print("MERGING ALL GROUPS")
    print("="*80)
    
    if not all_trajectories:
        print("Error: No trajectories processed successfully")
        return
    
    final_trajectory_df = pd.concat(all_trajectories, ignore_index=True)
    
    print(f"Total trajectories: {len(final_trajectory_df):,}")
    print(f"Unique users: {final_trajectory_df['ID'].nunique():,}")
    
    # Save merged data
    ensure_dir(config.results_dir)
    output_path = os.path.join(config.results_dir, 'career_trajectories.parquet')
    final_trajectory_df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")
    
    benchmark.end_stage("Merging All Groups")
    
    # ========================================
    # END
    # ========================================

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Final output: {output_path}")
    print(f"Total trajectories: {len(final_trajectory_df):,}")
    print(f"Total users: {final_trajectory_df['ID'].nunique():,}")
    
    benchmark.print_report()
    benchmark.save_report(os.path.join(config.results_dir, 'benchmark_report.json'))


if __name__ == "__main__":
    main()
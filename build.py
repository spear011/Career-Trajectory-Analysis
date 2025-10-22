"""
Build Occupation Transition Networks
Main script for constructing temporal networks from job transition data
"""

import os
import pandas as pd
from pathlib import Path

from src.config import (
    DATA_DIR, 
    NETWORK_OUTPUT_DIR,
    STUDY_START_YEAR,
    STUDY_END_YEAR
)
from src.network_builder import (
    prepare_job_data,
    prepare_wage_data,
    build_user_career_paths,
    extract_transitions,
    build_temporal_network,
    calculate_network_statistics,
    save_networks
)


def main():
    """Main execution function"""
    print("="*80)
    print("OCCUPATION TRANSITION NETWORK CONSTRUCTION")
    print("="*80)
    print(f"Study Period: {STUDY_START_YEAR}-{STUDY_END_YEAR}")
    print("Based on Research Proposal:")
    print("  - Pre-Pandemic: 2017-2019 (Baseline mobility patterns)")
    print("  - COVID Shock: Mar 2020-2021 (Labor market volatility)")
    print("  - Post-Pandemic Recovery: 2022-2024 (Persistence of changes)")
    print("="*80)
    
    # 1. Load data (from project files)
    print("\nLoading data...")
    job_df = pd.read_csv(os.path.join(DATA_DIR, 'job_group_0.csv'))
    wage_df = pd.read_csv(os.path.join(DATA_DIR, 'wage_interpolated_1999_2022_soc2019_unique.csv'))
    
    print(f"  Job records: {len(job_df):,}")
    print(f"  Wage records: {len(wage_df):,}")
    
    # 2. Prepare data
    job_df = prepare_job_data(job_df)
    wage_df = prepare_wage_data(wage_df)
    
    # 3. Build user career paths
    user_paths = build_user_career_paths(job_df)
    
    # 4. Extract transitions
    transitions_df = extract_transitions(user_paths)
    
    # 5. Build temporal network
    networks = build_temporal_network(transitions_df, wage_df, period='annual')
    
    # 6. Calculate network statistics
    stats_df = calculate_network_statistics(networks)
    
    # 7. Save
    output_path = save_networks(networks, stats_df, NETWORK_OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("NETWORK CONSTRUCTION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_path}")
    print(f"Number of temporal networks: {len([y for y in networks.keys() if not pd.isna(y)])}")
    valid_years = [y for y in networks.keys() if not pd.isna(y)]
    if valid_years:
        print(f"Year range: {int(min(valid_years))}-{int(max(valid_years))}")
    
    return networks, stats_df


if __name__ == '__main__':
    networks, stats = main()
"""
Labor Market Trend Analysis with Education Data
Extends the original pipeline to include education analysis
"""
import os
import pandas as pd

from src.config import get_windows, RESULTS_DIR, YEARS, OCCUPATION_COLUMN
from src.utils import ensure_dir
from src.data_loader import load_job_data, get_window_users, get_occupation_distributions
from src.education_loader import (
    load_education_data,
    merge_job_education,
    analyze_education_by_occupation,
    analyze_major_by_occupation,
    get_education_level_distribution,
    get_major_distribution
)
from src.transitions import build_transitions, calculate_transition_rates
from src.workforce_flow import analyze_workforce_flow
from src.visualization import create_workforce_flow_dashboard, create_occupation_evolution_plot
from src.occupation_flow_analysis import (
    analyze_occupation_outflows, 
    analyze_occupation_inflows,
    find_critical_transitions,
    plot_occupation_turnover_heatmap,
    plot_occupation_exit_analysis,
    create_transition_flow_summary
)
from src.benchmark_utils import PipelineBenchmark


def export_education_results(ed_level_dist, major_dist, edu_by_occ, major_by_occ, results_dir):
    """Export education analysis results"""
    print("\n" + "="*80)
    print("EXPORTING EDUCATION ANALYSIS")
    print("="*80)
    
    ed_level_dist.to_csv(os.path.join(results_dir, 'education_level_distribution.csv'), index=False)
    major_dist.to_csv(os.path.join(results_dir, 'major_distribution.csv'), index=False)
    edu_by_occ.to_csv(os.path.join(results_dir, 'education_by_occupation.csv'), index=False)
    major_by_occ.to_csv(os.path.join(results_dir, 'major_by_occupation.csv'), index=False)
    
    print(f"✓ Education level distribution: education_level_distribution.csv")
    print(f"✓ Major distribution: major_distribution.csv")
    print(f"✓ Education by occupation: education_by_occupation.csv")
    print(f"✓ Major by occupation: major_by_occupation.csv")


def generate_education_summary(ed_level_dist, major_dist, edu_by_occ, results_dir):
    """Generate human-readable education summary"""
    summary_path = os.path.join(results_dir, 'education_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EDUCATION DATA SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Education level distribution
        f.write("EDUCATION LEVEL DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        total = ed_level_dist['count'].sum()
        for _, row in ed_level_dist.iterrows():
            pct = (row['count'] / total) * 100
            f.write(f"{row['EDULEVEL_NAME']}: {row['count']:,} ({pct:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 MAJORS\n")
        f.write("="*80 + "\n")
        for idx, row in major_dist.iterrows():
            f.write(f"{idx+1}. {row['CIP6_2020_NAME']}: {row['count']:,}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("EDUCATION REQUIREMENTS BY OCCUPATION (Top 10)\n")
        f.write("="*80 + "\n\n")
        
        # Get top 10 occupations by total count
        occ_totals = edu_by_occ.groupby(edu_by_occ.columns[0])['count'].sum().sort_values(ascending=False).head(10)
        
        for occ in occ_totals.index:
            f.write(f"\n{occ}:\n")
            occ_data = edu_by_occ[edu_by_occ[edu_by_occ.columns[0]] == occ].sort_values('percentage', ascending=False)
            for _, row in occ_data.iterrows():
                f.write(f"  - {row['Education_Level']}: {row['percentage']:.1f}%\n")
    
    print(f"✓ Education summary: education_summary.txt")


def main():
    """Main analysis pipeline with education data"""
    benchmark = PipelineBenchmark()
    
    print("="*80)
    print("LABOR MARKET TREND ANALYSIS WITH EDUCATION DATA")
    print("="*80)
    print(f"Period: {YEARS[0]}-{YEARS[-1]}")
    print(f"Occupation field: {OCCUPATION_COLUMN}")
    print("="*80)
    
    ensure_dir(RESULTS_DIR)
    
    # Stage 1: Load data
    benchmark.start_stage("Data Loading")
    job_df = load_job_data()
    ed_df = load_education_data()
    benchmark.end_stage("Data Loading")
    
    # Stage 2: Merge job and education
    benchmark.start_stage("Job-Education Merge")
    merged_df = merge_job_education(job_df, ed_df)
    benchmark.end_stage("Job-Education Merge")
    
    # Stage 3: Education distribution analysis
    benchmark.start_stage("Education Analysis")
    ed_level_dist = get_education_level_distribution(ed_df)
    major_dist = get_major_distribution(ed_df, top_n=20)
    benchmark.end_stage("Education Analysis")
    
    # Stage 4: Education by occupation
    benchmark.start_stage("Education-Occupation Analysis")
    edu_by_occ = analyze_education_by_occupation(merged_df, OCCUPATION_COLUMN)
    major_by_occ = analyze_major_by_occupation(merged_df, OCCUPATION_COLUMN)
    benchmark.end_stage("Education-Occupation Analysis")
    
    # Stage 5: Window processing (with education data)
    benchmark.start_stage("Window Processing")
    windows = get_windows()
    window_users = get_window_users(merged_df, windows)  # Now includes education
    benchmark.end_stage("Window Processing")
    
    # Stage 6: Transitions (same as before)
    benchmark.start_stage("Transitions")
    all_transitions = build_transitions(window_users, windows)
    transition_rates = calculate_transition_rates(all_transitions)
    benchmark.end_stage("Transitions")
    
    # Stage 7: Workforce flow (same as before)
    benchmark.start_stage("Workforce Flow")
    yearly_flow_df, user_details_df = analyze_workforce_flow(
        window_users, windows, RESULTS_DIR, OCCUPATION_COLUMN
    )
    benchmark.end_stage("Workforce Flow")
    
    # Stage 8: Occupation flow (same as before)
    benchmark.start_stage("Occupation Flow")
    outflow_df = analyze_occupation_outflows(user_details_df, window_users, windows, OCCUPATION_COLUMN)
    inflow_df = analyze_occupation_inflows(user_details_df, window_users, windows)
    critical_transitions = find_critical_transitions(outflow_df)
    benchmark.end_stage("Occupation Flow")
    
    # Stage 9: Export results
    benchmark.start_stage("Export")
    
    # Export original results
    transition_rates.to_csv(os.path.join(RESULTS_DIR, 'transition_rates.csv'), index=False)
    yearly_flow_df.to_csv(os.path.join(RESULTS_DIR, 'yearly_workforce_flow.csv'), index=False)
    all_transitions.to_csv(os.path.join(RESULTS_DIR, 'all_transitions.csv'), index=False)
    user_details_df.to_csv(os.path.join(RESULTS_DIR, 'user_detailed_paths.csv'), index=False)
    outflow_df.to_csv(os.path.join(RESULTS_DIR, 'occupation_outflows.csv'), index=False)
    inflow_df.to_csv(os.path.join(RESULTS_DIR, 'occupation_inflows.csv'), index=False)
    critical_transitions.to_csv(os.path.join(RESULTS_DIR, 'critical_transitions.csv'), index=False)
    
    # Export education results
    export_education_results(ed_level_dist, major_dist, edu_by_occ, major_by_occ, RESULTS_DIR)
    
    # Generate summaries
    generate_education_summary(ed_level_dist, major_dist, edu_by_occ, RESULTS_DIR)
    
    benchmark.end_stage("Export")
    
    # Print benchmark report
    benchmark.print_report()
    benchmark.save_report(os.path.join(RESULTS_DIR, 'benchmark_education.json'))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nKey outputs:")
    print("  - user_detailed_paths.csv (with education data)")
    print("  - education_by_occupation.csv")
    print("  - major_by_occupation.csv")
    print("  - education_summary.txt")

if __name__ == "__main__":
    main()
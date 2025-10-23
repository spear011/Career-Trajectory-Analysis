"""
Labor Market Trend Analysis - Refactored Version
Analyzes workforce dynamics from 2017-2024 with study period focus
Enhanced with Pre-Pandemic, COVID Shock, and Post-Pandemic period analysis
"""
import os
import pandas as pd

from src.config import get_windows, RESULTS_DIR, YEARS, OCCUPATION_COLUMN
from src.utils import ensure_dir
from src.data_loader import DataLoader
from src.analyzer import MobilityAnalyzer
from src.benchmark_utils import PipelineBenchmark

# Import visualization - we'll keep these as is for now
from src.visualization import create_all_visualizations


def export_results(results, occ_dist_df, results_dir):
    """
    Export analysis results to CSV
    
    Args:
        results: Dictionary with all analysis results
        occ_dist_df: Occupation distribution dataframe
        results_dir: Results directory
    """
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    results['transition_rates'].to_csv(os.path.join(results_dir, 'transition_rates.csv'), index=False)
    results['workforce_flow'].to_csv(os.path.join(results_dir, 'yearly_workforce_flow.csv'), index=False)
    results['transitions'].to_csv(os.path.join(results_dir, 'all_transitions.csv'), index=False)
    occ_dist_df.to_csv(os.path.join(results_dir, 'occupation_distributions.csv'), index=False)
    results['user_details'].to_csv(os.path.join(results_dir, 'user_detailed_paths.csv'), index=False)
    
    # Export occupation flow analysis results
    if 'outflows' in results:
        results['outflows'].to_csv(os.path.join(results_dir, 'occupation_outflows.csv'), index=False)
    if 'inflows' in results:
        results['inflows'].to_csv(os.path.join(results_dir, 'occupation_inflows.csv'), index=False)
    if 'critical_transitions' in results:
        results['critical_transitions'].to_csv(os.path.join(results_dir, 'critical_transitions.csv'), index=False)
    
    print("✓ All results exported to CSV files")


def generate_summary_report(results, results_dir):
    """
    Generate text summary report
    
    Args:
        results: Dictionary with all analysis results
        results_dir: Results directory
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    yearly_flow_df = results['workforce_flow']
    transition_rates = results['transition_rates']
    
    report_path = os.path.join(results_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LABOR MARKET TREND ANALYSIS - SUMMARY REPORT\n")
        f.write("Study Period Analysis (2017-2024)\n")
        f.write("="*80 + "\n\n")
        
        f.write("STUDY PERIODS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Pre-Pandemic: 2017-2019 (Baseline patterns)\n")
        f.write("2. COVID Shock: 2020-2021 (Labor market disruption)\n")
        f.write("3. Post-Pandemic: 2022-2024 (Recovery & restructuring)\n\n")
        
        f.write("KEY METRICS BY PERIOD:\n")
        f.write("-" * 80 + "\n\n")
        
        # Add period column
        yearly_flow_df['Period'] = yearly_flow_df['Year_To'].apply(
            lambda y: 'Pre-Pandemic' if 2017 <= y <= 2019 
                     else 'COVID Shock' if 2020 <= y <= 2021
                     else 'Post-Pandemic' if 2022 <= y <= 2024
                     else None
        )
        
        for period in ['Pre-Pandemic', 'COVID Shock', 'Post-Pandemic']:
            period_data = yearly_flow_df[yearly_flow_df['Period'] == period]
            
            if len(period_data) > 0:
                f.write(f"{period}:\n")
                f.write(f"  Avg Dropout Rate: {period_data['Dropout_Rate'].mean():.2f}%\n")
                f.write(f"  Avg Permanent Exit Rate: {period_data['Permanent_Exit_Rate'].mean():.2f}%\n")
                f.write(f"  Avg Comeback Rate: {period_data['Comeback_Rate'].mean():.2f}%\n")
                f.write(f"  Total Permanent Exits: {period_data['Permanent_Exits'].sum():,}\n")
                f.write(f"  Total Comebacks: {period_data['Comebacks'].sum():,}\n\n")
        
        f.write("="*80 + "\n")
        f.write("YEAR-BY-YEAR PERMANENT EXITS\n")
        f.write("="*80 + "\n\n")
        
        for _, row in yearly_flow_df.iterrows():
            f.write(f"{int(row['Year_From'])}→{int(row['Year_To'])}: {int(row['Permanent_Exits']):,} people ")
            f.write(f"({row['Permanent_Exit_Rate']:.2f}% of workforce)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("For detailed visualizations, see PNG files in results directory\n")
        f.write("For detailed career paths, see: user_detailed_paths.csv\n")
    
    print(f"✓ Summary report saved: {report_path}")


def main():
    """Main analysis pipeline - refactored version"""
    benchmark = PipelineBenchmark()
    
    print("="*80)
    print("LABOR MARKET TREND ANALYSIS (REFACTORED)")
    print("Study Period Focus: Pre-Pandemic, COVID Shock, Post-Pandemic")
    print("="*80)
    print(f"\nAnalyzing period: {YEARS[0]}-{YEARS[-1]}")
    print(f"Using {OCCUPATION_COLUMN}")
    print("\nStudy Periods:")
    print("  - Pre-Pandemic: 2017-2019")
    print("  - COVID Shock: 2020-2021")
    print("  - Post-Pandemic: 2022-2024")
    print("="*80)
    
    ensure_dir(RESULTS_DIR)
    
    # Stage 1: Setup
    benchmark.start_stage("Setup")
    windows = get_windows()
    print(f"\nCreated {len(windows)} yearly windows")
    benchmark.end_stage("Setup")
    
    # Stage 2: Data Loading (using DataLoader class)
    benchmark.start_stage("Data Loading")
    loader = DataLoader(data_dir='data')
    job_df = loader.load_job_data()
    benchmark.end_stage("Data Loading")
    
    # Stage 3: Window Processing
    benchmark.start_stage("Window Processing")
    window_users = loader.get_window_users(job_df, windows)
    benchmark.end_stage("Window Processing")
    
    # Stage 4: Complete Mobility Analysis (using MobilityAnalyzer class)
    benchmark.start_stage("Mobility Analysis")
    analyzer = MobilityAnalyzer(results_dir=RESULTS_DIR, occupation_col=OCCUPATION_COLUMN)
    results = analyzer.analyze_all(window_users, windows)
    benchmark.end_stage("Mobility Analysis")
    
    # Stage 5: Occupation Distributions
    benchmark.start_stage("Occupation Distributions")
    occ_dist_df = loader.get_occupation_distributions(window_users, windows, OCCUPATION_COLUMN)
    benchmark.end_stage("Occupation Distributions")
    
    # Stage 6: Visualizations
    benchmark.start_stage("Visualizations")
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_all_visualizations(
        results['workforce_flow'], 
        results['transition_rates'], 
        occ_dist_df, 
        RESULTS_DIR
    )
    
    benchmark.end_stage("Visualizations")
    
    # Stage 7: Export Results
    benchmark.start_stage("Export Results")
    export_results(results, occ_dist_df, RESULTS_DIR)
    generate_summary_report(results, RESULTS_DIR)
    benchmark.end_stage("Export Results")
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(RESULTS_DIR, 'benchmark_report.json'))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nFiles created in 'results/' directory:")
    print("\n📊 VERSION 1: FULL TIMELINE VISUALIZATIONS (1990-2023):")
    print("  - full_timeline_1990_2023.png ⭐ Complete historical trends")
    print("  - occupation_evolution_full_timeline.png ⭐ Long-term occupation changes")
    print("\n📊 VERSION 2: STUDY PERIOD VISUALIZATIONS (2017-2024):")
    print("  - period_based_analysis.png ⭐ Pre/COVID/Post period analysis")
    print("  - workforce_flow_by_period.png ⭐ Workforce flows by period")
    print("  - period_summary_statistics.csv ⭐ Statistical summary by period")
    print("  - top_occupations_evolution.png ⭐ Occupation trends with periods")
    print("\n📈 DATA FILES:")
    print("  - transition_rates.csv")
    print("  - yearly_workforce_flow.csv")
    print("  - all_transitions.csv")
    print("  - occupation_distributions.csv")
    print("  - user_detailed_paths.csv")
    print("  - occupation_outflows.csv")
    print("  - occupation_inflows.csv")
    print("  - critical_transitions.csv")
    print("  - summary_report.txt")
    print("\n⚡ PERFORMANCE:")
    print("  - benchmark_report.json")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
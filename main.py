"""
Labor Market Trend Analysis - Main Pipeline
MODIFIED: Uses preprocessed trajectory data from preprocess.py
Analyzes workforce dynamics with study period focus
Enhanced with Pre-Pandemic, COVID Shock, and Post-Pandemic period analysis
"""
import os
import pandas as pd

from src.utils import get_config, set_config, parse_args, ensure_dir
from src.data_loader import DataLoader
from src.analyzer import MobilityAnalyzer
from src.benchmark_utils import PipelineBenchmark
from src.visualization import create_all_visualizations


def export_results(results, results_dir):
    """
    Export analysis results to CSV
    
    Args:
        results: Dictionary with all analysis results
        results_dir: Results directory
    """
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    # Export trajectory summary
    if 'trajectory_summary' in results:
        results['trajectory_summary'].to_csv(
            os.path.join(results_dir, 'trajectory_summary_stats.csv'), 
            index=False
        )
        print("✓ Exported trajectory_summary_stats.csv")
    
    # Export user details (full enriched trajectory data)
    if 'user_details' in results:
        results['user_details'].to_csv(
            os.path.join(results_dir, 'user_detailed_trajectories.csv'), 
            index=False
        )
        print("✓ Exported user_detailed_trajectories.csv")
    
    # Export mobility analysis results
    if 'mobility_analysis' in results:
        mobility = results['mobility_analysis']
        
        # Demographic analysis
        if 'mobility_by_demographics' in mobility:
            demo = mobility['mobility_by_demographics']
            
            if 'by_gender' in demo:
                demo['by_gender'].to_csv(
                    os.path.join(results_dir, 'mobility_by_gender.csv'), 
                    index=False
                )
                print("✓ Exported mobility_by_gender.csv")
            
            if 'by_race' in demo:
                demo['by_race'].to_csv(
                    os.path.join(results_dir, 'mobility_by_race.csv'), 
                    index=False
                )
                print("✓ Exported mobility_by_race.csv")
            
            if 'by_generation' in demo:
                demo['by_generation'].to_csv(
                    os.path.join(results_dir, 'mobility_by_generation.csv'), 
                    index=False
                )
                print("✓ Exported mobility_by_generation.csv")
            
            if 'by_education' in demo:
                demo['by_education'].to_csv(
                    os.path.join(results_dir, 'mobility_by_education.csv'), 
                    index=False
                )
                print("✓ Exported mobility_by_education.csv")
        
        # Job change analysis
        if 'job_change_analysis' in mobility:
            job_change = mobility['job_change_analysis']
            
            if 'change_distribution' in job_change:
                job_change['change_distribution'].to_csv(
                    os.path.join(results_dir, 'job_change_distribution.csv'), 
                    index=False
                )
                print("✓ Exported job_change_distribution.csv")
            
            if 'job_change_types_by_year' in job_change:
                job_change['job_change_types_by_year'].to_csv(
                    os.path.join(results_dir, 'job_change_types_by_year.csv'), 
                    index=False
                )
                print("✓ Exported job_change_types_by_year.csv")
            
            if 'mobility_by_num_changes' in job_change:
                job_change['mobility_by_num_changes'].to_csv(
                    os.path.join(results_dir, 'mobility_by_num_changes.csv'), 
                    index=False
                )
                print("✓ Exported mobility_by_num_changes.csv")
        
        # Wage mobility analysis
        if 'wage_mobility' in mobility:
            wage = mobility['wage_mobility']
            
            if 'upward_mobility_by_year' in wage:
                wage['upward_mobility_by_year'].to_csv(
                    os.path.join(results_dir, 'upward_mobility_by_year.csv'), 
                    index=False
                )
                print("✓ Exported upward_mobility_by_year.csv")
            
            if 'wage_distribution_by_year' in wage:
                wage['wage_distribution_by_year'].to_csv(
                    os.path.join(results_dir, 'wage_distribution_by_year.csv'), 
                    index=False
                )
                print("✓ Exported wage_distribution_by_year.csv")
            
            if 'upward_mobility_by_occupation' in wage:
                wage['upward_mobility_by_occupation'].to_csv(
                    os.path.join(results_dir, 'upward_mobility_by_occupation.csv'), 
                    index=False
                )
                print("✓ Exported upward_mobility_by_occupation.csv")
            
            if 'upward_mobility_by_gender' in wage:
                wage['upward_mobility_by_gender'].to_csv(
                    os.path.join(results_dir, 'upward_mobility_by_gender.csv'), 
                    index=False
                )
                print("✓ Exported upward_mobility_by_gender.csv")
            
            if 'upward_mobility_by_race' in wage:
                wage['upward_mobility_by_race'].to_csv(
                    os.path.join(results_dir, 'upward_mobility_by_race.csv'), 
                    index=False
                )
                print("✓ Exported upward_mobility_by_race.csv")
    
    print("\n✓ All results exported to CSV files")


def generate_summary_report(results, trajectory_df, results_dir, config):
    """
    Generate text summary report
    
    Args:
        results: Dictionary with all analysis results
        trajectory_df: Enriched trajectory dataframe
        results_dir: Results directory
        config: Config instance
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report_path = os.path.join(results_dir, 'summary_report.txt')
    
    # Get study periods from config
    periods = config.get_study_periods()
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LABOR MARKET TREND ANALYSIS - SUMMARY REPORT\n")
        f.write(f"Study Period Analysis ({config.study_start_year}-{config.study_end_year})\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATA SOURCE:\n")
        f.write("-" * 80 + "\n")
        f.write("Preprocessed trajectory data from preprocess.py\n")
        f.write(f"Total trajectories analyzed: {len(trajectory_df):,}\n")
        f.write(f"Features: {len(trajectory_df.columns)}\n\n")
        
        f.write("STUDY PERIODS:\n")
        f.write("-" * 80 + "\n")
        for i, (period_key, period_info) in enumerate(periods.items(), 1):
            f.write(f"{i}. {period_info['name']}: {period_info['start_year']}-{period_info['end_year']} "
                   f"({period_info['description']})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TRAJECTORY DATA FEATURES\n")
        f.write("="*80 + "\n")
        
        # Demographics
        if 'gender' in trajectory_df.columns:
            f.write("\nGender Distribution:\n")
            gender_dist = trajectory_df['gender'].value_counts(normalize=True) * 100
            for gender, pct in gender_dist.items():
                f.write(f"  {gender}: {pct:.1f}%\n")
        
        if 'race' in trajectory_df.columns:
            f.write("\nRace Distribution:\n")
            race_dist = trajectory_df['race'].value_counts(normalize=True) * 100
            for race, pct in race_dist.items():
                f.write(f"  {race}: {pct:.1f}%\n")
        
        if 'generation' in trajectory_df.columns:
            f.write("\nGeneration Distribution:\n")
            gen_dist = trajectory_df['generation'].value_counts(normalize=True) * 100
            for gen, pct in gen_dist.items():
                f.write(f"  {gen}: {pct:.1f}%\n")
        
        # Job mobility
        if 'num_job_changes' in trajectory_df.columns:
            f.write("\nJob Mobility:\n")
            f.write(f"  Average job changes: {trajectory_df['num_job_changes'].mean():.2f}\n")
            f.write(f"  Median job changes: {trajectory_df['num_job_changes'].median():.1f}\n")
        
        if 'up_move' in trajectory_df.columns:
            up_move_rate = trajectory_df['up_move'].mean() * 100
            f.write(f"  Upward mobility rate: {up_move_rate:.1f}%\n")
        
        # Job change types
        move_cols = ['move_1_1', 'move_1_2', 'move_2_1', 'move_2_2']
        if all(col in trajectory_df.columns for col in move_cols):
            f.write("\nJob Change Types:\n")
            f.write(f"  Type 1-1 (Diff company, diff occupation): {trajectory_df['move_1_1'].sum():,}\n")
            f.write(f"  Type 1-2 (Diff company, same occupation): {trajectory_df['move_1_2'].sum():,}\n")
            f.write(f"  Type 2-1 (Same company, diff occupation): {trajectory_df['move_2_1'].sum():,}\n")
            f.write(f"  Type 2-2 (Same company, same occupation): {trajectory_df['move_2_2'].sum():,}\n")
        
        # Wages
        if 'annual_state_wage_x' in trajectory_df.columns:
            f.write("\nWage Statistics (First Job):\n")
            f.write(f"  Mean: ${trajectory_df['annual_state_wage_x'].mean():,.0f}\n")
            f.write(f"  Median: ${trajectory_df['annual_state_wage_x'].median():,.0f}\n")
            f.write(f"  25th percentile: ${trajectory_df['annual_state_wage_x'].quantile(0.25):,.0f}\n")
            f.write(f"  75th percentile: ${trajectory_df['annual_state_wage_x'].quantile(0.75):,.0f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("For detailed visualizations, see PNG files in results directory\n")
        f.write("For detailed analysis results, see CSV files\n")
    
    print(f"✓ Summary report saved: {report_path}")


def compute_trajectory_summary(trajectory_df, config):
    """
    Compute summary statistics from trajectory data
    
    Args:
        trajectory_df: Enriched trajectory dataframe
        config: Config instance
    
    Returns:
        Summary statistics dataframe
    """
    print("\nComputing trajectory summary statistics...")
    
    summary_stats = []
    
    # Filter by study period
    study_trajectories = trajectory_df[
        (trajectory_df['job_start_year_x'] >= config.study_start_year) &
        (trajectory_df['job_start_year_x'] <= config.study_end_year)
    ]
    
    # Group by year and compute statistics
    for year in range(config.study_start_year, config.study_end_year + 1):
        year_data = study_trajectories[study_trajectories['job_start_year_x'] == year]
        
        if len(year_data) == 0:
            continue
        
        stats = {
            'year': year,
            'n_trajectories': len(year_data),
            'avg_job_changes': year_data['num_job_changes'].mean() if 'num_job_changes' in year_data.columns else None,
            'upward_mobility_rate': year_data['up_move'].mean() * 100 if 'up_move' in year_data.columns else None,
            'avg_wage': year_data['annual_state_wage_x'].mean() if 'annual_state_wage_x' in year_data.columns else None,
            'median_wage': year_data['annual_state_wage_x'].median() if 'annual_state_wage_x' in year_data.columns else None,
        }
        
        # Add demographic breakdowns if available
        if 'gender' in year_data.columns:
            stats['pct_female'] = (year_data['gender'] == 2).mean() * 100
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    print(f"✓ Computed summary for {len(summary_df)} years")
    
    return summary_df


def main():
    """Main analysis pipeline using preprocessed trajectory data"""
    
    # Parse CLI arguments and load config
    args = parse_args()
    config = get_config() if args.config is None else get_config()
    
    # Override config with CLI args if provided
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.study_start_year:
        config.study_start_year = args.study_start_year
    if args.study_end_year:
        config.study_end_year = args.study_end_year
    if args.no_cache:
        config.cache_enabled = False
    
    set_config(config)
    
    benchmark = PipelineBenchmark()
    
    print("="*80)
    print("LABOR MARKET TREND ANALYSIS (TRAJECTORY-BASED)")
    print("Study Period Focus: Pre-Pandemic, COVID Shock, Post-Pandemic")
    print("="*80)
    print(f"\nData Source: Preprocessed trajectory data")
    print(f"Analyzing period: {config.study_start_year}-{config.study_end_year}")
    print(f"Using occupation column: {config.analysis_occupation_column}")
    
    # Display study periods
    periods = config.get_study_periods()
    print("\nStudy Periods:")
    for period_key, period_info in periods.items():
        print(f"  - {period_info['name']}: {period_info['start_year']}-{period_info['end_year']} "
              f"({period_info['description']})")
    print("="*80)
    
    ensure_dir(config.results_dir)
    
    # Stage 1: Setup
    benchmark.start_stage("Setup")
    print("\nInitializing analysis pipeline...")
    benchmark.end_stage("Setup")
    
    # Stage 2: Load Preprocessed Trajectory Data
    benchmark.start_stage("Load Preprocessed Data")
    loader = DataLoader(config)
    trajectory_df = loader.load_preprocessed_trajectories()
    benchmark.end_stage("Load Preprocessed Data")
    
    # Stage 3: Compute Trajectory Summary Statistics
    benchmark.start_stage("Trajectory Summary")
    trajectory_summary = compute_trajectory_summary(trajectory_df, config)
    benchmark.end_stage("Trajectory Summary")
    
    # Stage 4: Mobility Analysis (using trajectory data)
    benchmark.start_stage("Mobility Analysis")
    print("\n" + "="*80)
    print("MOBILITY ANALYSIS FROM TRAJECTORY DATA")
    print("="*80)
    
    analyzer = MobilityAnalyzer(
        results_dir=config.results_dir,
        occupation_col='onet_major_x'  # Use trajectory_df occupation column
    )
    
    analysis_results = analyzer.analyze_all(trajectory_df, config)
    
    # Combine with trajectory summary
    results = {
        'trajectory_summary': trajectory_summary,
        'mobility_analysis': analysis_results,
        'user_details': trajectory_df.copy(),  # Full enriched trajectory data
    }
    
    print("✓ Mobility analysis complete")
    benchmark.end_stage("Mobility Analysis")
    
    # Stage 5: Visualizations
    benchmark.start_stage("Visualizations")
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # TODO: Create trajectory-based visualizations
    print("⚠ Visualization generation to be implemented for trajectory data")
    
    benchmark.end_stage("Visualizations")
    
    # Stage 6: Export Results
    benchmark.start_stage("Export Results")
    export_results(results, config.results_dir)
    generate_summary_report(results, trajectory_df, config.results_dir, config)
    benchmark.end_stage("Export Results")
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(config.results_dir, 'benchmark_report.json'))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFiles created in '{config.results_dir}/' directory:")
    print("\n📊 TRAJECTORY DATA:")
    print("  - trajectory_summary_stats.csv")
    print("  - user_detailed_trajectories.csv (enriched trajectory data)")
    print("\n📈 DEMOGRAPHIC ANALYSIS:")
    print("  - mobility_by_gender.csv")
    print("  - mobility_by_race.csv")
    print("  - mobility_by_generation.csv")
    print("  - mobility_by_education.csv")
    print("\n🔄 JOB CHANGE ANALYSIS:")
    print("  - job_change_distribution.csv")
    print("  - job_change_types_by_year.csv")
    print("  - mobility_by_num_changes.csv")
    print("\n💰 WAGE MOBILITY ANALYSIS:")
    print("  - upward_mobility_by_year.csv")
    print("  - wage_distribution_by_year.csv")
    print("  - upward_mobility_by_occupation.csv")
    print("  - upward_mobility_by_gender.csv")
    print("  - upward_mobility_by_race.csv")
    print("\n📋 REPORTS:")
    print("  - summary_report.txt")
    print("\n⚡ PERFORMANCE:")
    print("  - benchmark_report.json")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
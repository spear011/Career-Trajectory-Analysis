"""
Labor Market Trend Analysis - Main Pipeline with Windowed Analysis
MODIFIED: Integrated WindowedAnalyzer for temporal workforce tracking
"""
import os
import pandas as pd

from src.utils import get_config, set_config, parse_args, ensure_dir
from src.data_loader import DataLoader
from src.analyzer import WindowedAnalyzer
from src.benchmark_utils import PipelineBenchmark


def export_windowed_results(results: dict, results_dir: str):
    """
    Export windowed analysis results to CSV
    
    Args:
        results: Dictionary with analysis results
        results_dir: Results directory
    """
    print("\n" + "="*80)
    print("EXPORTING WINDOWED ANALYSIS RESULTS")
    print("="*80)
    
    # Export window statistics
    if 'window_stats' in results:
        results['window_stats'].to_csv(
            os.path.join(results_dir, 'window_statistics.csv'),
            index=False
        )
        print("✓ Exported window_statistics.csv")
    
    # Export occupation dynamics per window
    if 'occupation_dynamics' in results:
        for window_label, occ_df in results['occupation_dynamics'].items():
            filename = f'occupation_dynamics_{window_label}.csv'
            occ_df.to_csv(
                os.path.join(results_dir, filename),
                index=False
            )
        print(f"✓ Exported occupation dynamics for {len(results['occupation_dynamics'])} windows")
    
    # Export period comparisons
    if 'period_comparisons' in results:
        comparison_df = pd.DataFrame([results['period_comparisons']])
        comparison_df.to_csv(
            os.path.join(results_dir, 'period_comparisons.csv'),
            index=False
        )
        print("✓ Exported period_comparisons.csv")


def generate_windowed_summary_report(results: dict, 
                                     trajectory_df: pd.DataFrame,
                                     results_dir: str,
                                     config):
    """
    Generate text summary report for windowed analysis
    
    Args:
        results: Analysis results dictionary
        trajectory_df: Full trajectory dataframe
        results_dir: Results directory
        config: Config instance
    """
    report_path = os.path.join(results_dir, 'windowed_summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WINDOWED WORKFORCE ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Period: {config.study_start_year}-{config.study_end_year}\n")
        f.write(f"Window Size: {config.window_size} year(s)\n")
        f.write(f"Hop Size: {config.hop_size} year(s)\n")
        f.write(f"Occupation Column: {config.windowed_occupation_column}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total trajectory records: {len(trajectory_df):,}\n")
        f.write(f"Unique users: {trajectory_df['ID'].nunique():,}\n")
        f.write(f"Year range: {trajectory_df['job_start_year'].min():.0f}-{trajectory_df['job_start_year'].max():.0f}\n\n")
        
        if 'window_stats' in results:
            window_stats = results['window_stats']
            f.write("-"*80 + "\n")
            f.write("WINDOW STATISTICS SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total windows analyzed: {len(window_stats)}\n")
            f.write(f"Average active users per window: {window_stats['total_active_users'].mean():,.0f}\n")
            f.write(f"Average occupations per window: {window_stats['num_occupations'].mean():.1f}\n")
            f.write(f"Average new entrants per window: {window_stats['new_entrants'].mean():,.0f}\n")
            f.write(f"Average exits per window: {window_stats['exits'].mean():,.0f}\n\n")
            
            # Peak and trough windows
            peak_users_idx = window_stats['total_active_users'].idxmax()
            trough_users_idx = window_stats['total_active_users'].idxmin()
            
            f.write(f"Peak workforce window: {window_stats.loc[peak_users_idx, 'window_label']} "
                   f"({window_stats.loc[peak_users_idx, 'total_active_users']:,} users)\n")
            f.write(f"Trough workforce window: {window_stats.loc[trough_users_idx, 'window_label']} "
                   f"({window_stats.loc[trough_users_idx, 'total_active_users']:,} users)\n\n")
        
        if 'period_comparisons' in results:
            comp = results['period_comparisons']
            f.write("-"*80 + "\n")
            f.write("PERIOD COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"{comp['period1_name']} ({comp['period1_years']}): {comp['period1_users']:,} users\n")
            f.write(f"{comp['period2_name']} ({comp['period2_years']}): {comp['period2_users']:,} users\n")
            f.write(f"Users in both periods: {comp['users_in_both']:,} ({comp['retention_rate']:.1f}% retention)\n")
            f.write(f"New entrants in {comp['period2_name']}: {comp['users_only_period2']:,}\n")
            f.write(f"Exits after {comp['period1_name']}: {comp['users_only_period1']:,}\n\n")
            
            f.write(f"Occupations in {comp['period1_name']}: {comp['period1_occupations']}\n")
            f.write(f"Occupations in {comp['period2_name']}: {comp['period2_occupations']}\n")
            f.write(f"New occupations: {comp['new_occupations']}\n")
            f.write(f"Disappeared occupations: {comp['disappeared_occupations']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Generated summary report: {report_path}")


def main():
    """Main analysis pipeline with windowed analysis"""
    
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
    if hasattr(args, 'window_size') and args.window_size:
        config.window_size = args.window_size
    if hasattr(args, 'hop_size') and args.hop_size:
        config.hop_size = args.hop_size
    if args.no_cache:
        config.cache_enabled = False
    
    set_config(config)
    
    benchmark = PipelineBenchmark()
    
    print("="*80)
    print("WINDOWED WORKFORCE ANALYSIS PIPELINE")
    print("="*80)
    print(f"\nData Source: Preprocessed trajectory data")
    print(f"Analysis Period: {config.study_start_year}-{config.study_end_year}")
    print(f"Window Configuration:")
    print(f"  - Window size: {config.window_size} year(s)")
    print(f"  - Hop size: {config.hop_size} year(s)")
    print(f"  - Occupation column: {config.windowed_occupation_column}")
    
    # Display study periods
    periods = config.get_study_periods()
    print("\nStudy Periods for Comparison:")
    for period_key, period_info in periods.items():
        print(f"  - {period_info['name']}: {period_info['start_year']}-{period_info['end_year']}")
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
    
    # Stage 3: Windowed Analysis
    benchmark.start_stage("Windowed Analysis")
    print("\n" + "="*80)
    print("RUNNING WINDOWED WORKFORCE ANALYSIS")
    print("="*80)
    
    analyzer = WindowedAnalyzer(
        window_size=config.window_size,
        hop_size=config.hop_size,
        occupation_col=config.windowed_occupation_column
    )
    
    # Run main windowed analysis
    window_stats = analyzer.analyze_all_windows(
        trajectory_df,
        config.study_start_year,
        config.study_end_year
    )
    
    results = {
        'window_stats': window_stats,
        'occupation_dynamics': {},
        'trajectory_data': trajectory_df.copy()
    }
    
    # Detailed occupation dynamics for key windows
    print("\nAnalyzing occupation dynamics for key periods...")
    for period_key, period_info in periods.items():
        period_start = period_info['start_year']
        period_end = period_info['end_year']
        period_name = period_info['name'].replace(' ', '_')
        
        occ_dynamics = analyzer.analyze_occupation_dynamics(
            trajectory_df,
            period_start,
            period_end
        )
        results['occupation_dynamics'][period_name] = occ_dynamics
        print(f"  ✓ {period_info['name']}: {len(occ_dynamics)} occupations analyzed")
    
    # Compare Pre-Pandemic vs Post-Pandemic
    print("\nComparing Pre-Pandemic vs Post-Pandemic periods...")
    pre_pandemic = periods['pre_pandemic']
    post_pandemic = periods['post_pandemic']
    
    period_comparison = analyzer.compare_periods(
        trajectory_df,
        pre_pandemic['start_year'],
        pre_pandemic['end_year'],
        post_pandemic['start_year'],
        post_pandemic['end_year'],
        pre_pandemic['name'],
        post_pandemic['name']
    )
    results['period_comparisons'] = period_comparison
    
    print("\n✓ Windowed analysis complete")
    benchmark.end_stage("Windowed Analysis")
    
    # Stage 4: Export Results
    benchmark.start_stage("Export Results")
    export_windowed_results(results, config.results_dir)
    generate_windowed_summary_report(results, trajectory_df, config.results_dir, config)
    benchmark.end_stage("Export Results")
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(config.results_dir, 'benchmark_report.json'))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFiles created in '{config.results_dir}/' directory:")
    print("\n📊 WINDOWED ANALYSIS:")
    print("  - window_statistics.csv (main time series)")
    print("  - occupation_dynamics_*.csv (per period)")
    print("  - period_comparisons.csv (Pre vs Post pandemic)")
    print("\n📋 REPORTS:")
    print("  - windowed_summary_report.txt")
    print("\n⚡ PERFORMANCE:")
    print("  - benchmark_report.json")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
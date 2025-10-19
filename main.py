"""
Labor Market Trend Analysis - Detecting Sudden Changes
Analyzes workforce dynamics from 2010-2023 to identify sudden market shifts
Now includes detailed user-level tracking and performance optimizations
"""
import os
import pandas as pd

from src.config import get_windows, RESULTS_DIR, YEARS, OCCUPATION_COLUMN
from src.utils import ensure_dir
from src.data_loader import load_job_data, get_window_users, get_occupation_distributions
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


def export_results(transition_rates, yearly_flow_df, all_transitions_df, 
                  occ_dist_df, user_details_df, outflow_df, inflow_df, 
                  transitions_df, results_dir):
    """Export all results to CSV including detailed user paths and occupation flows"""
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    transition_rates.to_csv(os.path.join(results_dir, 'transition_rates.csv'), index=False)
    yearly_flow_df.to_csv(os.path.join(results_dir, 'yearly_workforce_flow.csv'), index=False)
    all_transitions_df.to_csv(os.path.join(results_dir, 'all_transitions.csv'), index=False)
    occ_dist_df.to_csv(os.path.join(results_dir, 'occupation_distributions.csv'), index=False)
    
    # Export detailed user paths
    user_details_df.to_csv(os.path.join(results_dir, 'user_detailed_paths.csv'), index=False)
    
    # Export occupation flow analysis
    outflow_df.to_csv(os.path.join(results_dir, 'occupation_outflows.csv'), index=False)
    inflow_df.to_csv(os.path.join(results_dir, 'occupation_inflows.csv'), index=False)
    transitions_df.to_csv(os.path.join(results_dir, 'occupation_transitions.csv'), index=False)
    
    print("\n✓ CSV files exported")
    print(f"  - user_detailed_paths.csv: {len(user_details_df):,} user records with career paths")
    print(f"  - occupation_outflows.csv: {len(outflow_df):,} occupation outflow records")
    print(f"  - occupation_inflows.csv: {len(inflow_df):,} occupation inflow records")
    print(f"  - occupation_transitions.csv: {len(transitions_df):,} occupation-to-occupation transitions")


def generate_path_analysis_examples(user_details_df, results_dir):
    """Generate example queries showing how to analyze user paths"""
    
    examples = []
    
    # Example 1: Occupation changes among comebacks
    comebacks = user_details_df[user_details_df['Status'] == 'comeback']
    occ_changed = comebacks[comebacks['Occupation_Changed'] == True]
    examples.append(
        f"Comebacks who changed occupation: {len(occ_changed):,} / {len(comebacks):,} "
        f"({len(occ_changed)/len(comebacks)*100:.1f}%)"
    )
    
    # Example 2: Top destination occupations for new entrants
    new_entrants = user_details_df[user_details_df['Status'] == 'new_entrant']
    top_entry_occs = new_entrants['To_Occupation'].value_counts().head(5)
    
    # Example 3: Top source occupations for permanent exits
    perm_exits = user_details_df[user_details_df['Status'] == 'dropout_permanent_exit']
    top_exit_occs = perm_exits['From_Occupation'].value_counts().head(5)
    
    # Write to file
    with open(os.path.join(results_dir, 'path_analysis_guide.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("USER PATH ANALYSIS GUIDE\n")
        f.write("="*80 + "\n\n")
        
        f.write("File: user_detailed_paths.csv\n")
        f.write(f"Total records: {len(user_details_df):,}\n\n")
        
        f.write("COLUMNS:\n")
        f.write("- ID: User identifier\n")
        f.write("- Year_From, Year_To: Transition period\n")
        f.write("- Status: dropout_career_break | dropout_permanent_exit | comeback | new_entrant | stayed\n")
        f.write("- From_Occupation, To_Occupation: Career path\n")
        f.write("- From_Industry, To_Industry: Industry path\n")
        f.write("- From_State, To_State: Geographic movement\n")
        f.write("- Occupation_Changed, Industry_Changed, State_Changed: Change flags\n\n")
        
        f.write("="*80 + "\n")
        f.write("QUICK STATS\n")
        f.write("="*80 + "\n\n")
        
        for example in examples:
            f.write(f"• {example}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 5 OCCUPATIONS FOR NEW ENTRANTS\n")
        f.write("="*80 + "\n\n")
        for occ, count in top_entry_occs.items():
            f.write(f"{count:,} people → {occ}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 5 OCCUPATIONS FOR PERMANENT EXITS\n")
        f.write("="*80 + "\n\n")
        for occ, count in top_exit_occs.items():
            f.write(f"{count:,} people ← {occ}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("EXAMPLE ANALYSIS QUERIES (using pandas)\n")
        f.write("="*80 + "\n\n")
        
        f.write("# Load data\n")
        f.write("df = pd.read_csv('results/user_detailed_paths.csv')\n\n")
        
        f.write("# 1. Which occupations had highest permanent exit rates during COVID?\n")
        f.write("covid_exits = df[(df['Year_From'] == 2020) & (df['Status'] == 'dropout_permanent_exit')]\n")
        f.write("covid_exits['From_Occupation'].value_counts()\n\n")
        
        f.write("# 2. Career path analysis for comebacks\n")
        f.write("comebacks = df[df['Status'] == 'comeback']\n")
        f.write("career_changes = comebacks[comebacks['Occupation_Changed'] == True]\n")
        f.write("print(f'Career change rate: {len(career_changes)/len(comebacks)*100:.1f}%')\n\n")
        
        f.write("# 3. Most common occupation transitions\n")
        f.write("stayed = df[df['Status'] == 'stayed']\n")
        f.write("transitions = stayed[stayed['Occupation_Changed'] == True]\n")
        f.write("transitions.groupby(['From_Occupation', 'To_Occupation']).size().sort_values(ascending=False)\n\n")
        
        f.write("# 4. Geographic mobility by status\n")
        f.write("df.groupby('Status')['State_Changed'].mean() * 100\n\n")
        
        f.write("# 5. Industry transitions for new entrants by year\n")
        f.write("new_entrants = df[df['Status'] == 'new_entrant']\n")
        f.write("new_entrants.groupby(['Year_To', 'To_Industry']).size().unstack(fill_value=0)\n\n")
    
    print("✓ Path analysis guide generated")


def generate_summary_report(transition_rates, yearly_flow_df, results_dir):
    """Generate summary report"""
    with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("LABOR MARKET TREND ANALYSIS (2010-2023)\n")
        f.write("Detecting Sudden Changes and Market Shifts\n")
        f.write("="*80 + "\n\n")
        
        f.write("ANALYSIS APPROACH:\n")
        f.write("- Consecutive year pairs (includes new entrants & exits)\n")
        f.write("- Tracks: Dropouts, Permanent Exits, Comebacks, New Entrants\n")
        f.write("- Full workforce dynamics captured\n")
        f.write("- Detailed user-level paths tracked (occupation/industry/location)\n")
        f.write("- Analyzes all years to detect ANY sudden changes\n\n")
        
        f.write("="*80 + "\n")
        f.write("YEAR-BY-YEAR METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Period | Dropout% | PermExit% | Comeback% | Entry% | OccChange%\n")
        f.write("-" * 80 + "\n")
        
        for _, row in yearly_flow_df.iterrows():
            period = row['Period']
            trans_row = transition_rates[transition_rates['Period'] == period]
            occ_rate = trans_row['Occupation_Change_Rate'].values[0] if len(trans_row) > 0 else 0
            
            f.write(f"{period} | {row['Dropout_Rate']:5.2f}% | {row['Permanent_Exit_Rate']:7.2f}% | ")
            f.write(f"{row['Comeback_Rate']:7.2f}% | {row['Entry_Rate']:5.2f}% | {occ_rate:9.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("YEAR-BY-YEAR PERMANENT EXITS (Retirement/Exit Year)\n")
        f.write("="*80 + "\n\n")
        for _, row in yearly_flow_df.iterrows():
            f.write(f"{int(row['Year_From'])}→{int(row['Year_To'])}: {int(row['Permanent_Exits']):,} people ")
            f.write(f"({row['Permanent_Exit_Rate']:.2f}% of workforce)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETECTING SUDDEN CHANGES\n")
        f.write("="*80 + "\n\n")
        f.write("Look for:\n")
        f.write("- Spikes in Dropout or Permanent Exit rates\n")
        f.write("- Sharp drops in Comeback rates\n")
        f.write("- Sudden changes in Occupation transition rates\n")
        f.write("- Year-over-year deviations > 2-3 percentage points\n\n")
        
        f.write("For detailed career paths, see: user_detailed_paths.csv\n")
    
    print("✓ Summary report generated")


def main():
    """Main analysis pipeline with performance benchmarking"""
    benchmark = PipelineBenchmark()
    
    print("="*80)
    print("LABOR MARKET TREND ANALYSIS (2010-2023)")
    print("Detecting Sudden Changes in Workforce Dynamics")
    print("="*80)
    print(f"\nAnalyzing consecutive year pairs (includes new entrants & exits)")
    print(f"Using {OCCUPATION_COLUMN} (Broad Occupation Categories)")
    print(f"Period: {YEARS[0]}-{YEARS[-1]}")
    print(f"NEW: Tracking detailed user-level career paths")
    print(f"NEW: Performance optimizations (10-20x faster)")
    print("="*80)
    
    ensure_dir(RESULTS_DIR)
    
    # Stage 1: Setup
    benchmark.start_stage("Setup")
    windows = get_windows()
    print(f"\nCreated {len(windows)} yearly windows")
    benchmark.end_stage("Setup")
    
    # Stage 2: Data Loading
    benchmark.start_stage("Data Loading")
    job_df = load_job_data()
    benchmark.end_stage("Data Loading")
    
    # Stage 3: Window Processing
    benchmark.start_stage("Window Processing")
    window_users = get_window_users(job_df, windows)
    benchmark.end_stage("Window Processing")
    
    # Stage 4: Transitions
    benchmark.start_stage("Transition Analysis")
    all_transitions_df = build_transitions(window_users, windows)
    transition_rates = calculate_transition_rates(all_transitions_df)
    benchmark.end_stage("Transition Analysis")
    
    # Stage 5: Workforce Flow (OPTIMIZED)
    benchmark.start_stage("Workforce Flow Analysis (Optimized)")
    yearly_flow_df, user_details_df = analyze_workforce_flow(
        window_users, windows, RESULTS_DIR, OCCUPATION_COLUMN
    )
    benchmark.end_stage("Workforce Flow Analysis (Optimized)")
    
    # Stage 6: Occupation Distributions
    benchmark.start_stage("Occupation Distributions")
    occ_dist_df = get_occupation_distributions(window_users, windows, OCCUPATION_COLUMN)
    benchmark.end_stage("Occupation Distributions")
    
    # Stage 7: Occupation Flow Analysis (OPTIMIZED)
    benchmark.start_stage("Occupation Flow Analysis (Optimized)")
    print("\n" + "="*80)
    print("OCCUPATION-LEVEL FLOW ANALYSIS")
    print("="*80)
    
    outflow_df = analyze_occupation_outflows(user_details_df, window_users, windows, OCCUPATION_COLUMN)
    inflow_df = analyze_occupation_inflows(user_details_df, window_users, windows)
    transitions_df = find_critical_transitions(outflow_df, min_count=50, top_n=15)
    benchmark.end_stage("Occupation Flow Analysis (Optimized)")
    
    # Stage 8: Visualizations
    benchmark.start_stage("Visualizations")
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_workforce_flow_dashboard(yearly_flow_df, transition_rates, RESULTS_DIR)
    create_occupation_evolution_plot(occ_dist_df, YEARS, RESULTS_DIR)
    
    plot_occupation_turnover_heatmap(outflow_df, RESULTS_DIR, top_n=15)
    plot_occupation_exit_analysis(outflow_df, RESULTS_DIR, year_to=2021, top_n=15)
    if len(transitions_df) > 0:
        create_transition_flow_summary(transitions_df, RESULTS_DIR, year_to=2021, top_n=20)
    benchmark.end_stage("Visualizations")
    
    # Stage 9: Export
    benchmark.start_stage("Export Results")
    export_results(transition_rates, yearly_flow_df, all_transitions_df, 
                  occ_dist_df, user_details_df, outflow_df, inflow_df,
                  transitions_df, RESULTS_DIR)
    generate_summary_report(transition_rates, yearly_flow_df, RESULTS_DIR)
    generate_path_analysis_examples(user_details_df, RESULTS_DIR)
    benchmark.end_stage("Export Results")
    
    # Print performance report
    benchmark.print_report()
    benchmark.save_report(os.path.join(RESULTS_DIR, 'benchmark_report.json'))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nFiles created in 'results/' directory:")
    print("\n📊 SUMMARY DATA:")
    print("  - transition_rates.csv")
    print("  - yearly_workforce_flow.csv")
    print("  - all_transitions.csv")
    print("  - occupation_distributions.csv")
    print("  - summary_report.txt")
    print("\n🔍 DETAILED TRACKING:")
    print("  - user_detailed_paths.csv ⭐ Detailed career paths")
    print("  - path_analysis_guide.txt ⭐ Analysis examples")
    print("\n🎯 OCCUPATION FLOW ANALYSIS:")
    print("  - occupation_outflows.csv ⭐ Where people go when leaving each occupation")
    print("  - occupation_inflows.csv ⭐ Where people come from when entering each occupation")
    print("  - occupation_transitions.csv ⭐ Occupation-to-occupation transition volumes")
    print("\n📈 VISUALIZATIONS:")
    print("  - comprehensive_workforce_flow.png")
    print("  - top_occupations_evolution.png")
    print("  - occupation_turnover_heatmap.png ⭐")
    print("  - occupation_exit_analysis_2021.png ⭐")
    print("  - occupation_transitions_2021.png ⭐")
    print("\n⚡ PERFORMANCE:")
    print("  - benchmark_report.json ⭐ Performance metrics")
    print("\nNote: COVID (2020) marked for reference, all years analyzed for sudden changes")


if __name__ == "__main__":
    main()
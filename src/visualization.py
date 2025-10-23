"""
Visualization functions for labor market trend analysis
Enhanced with study period analysis (Pre-Pandemic, COVID Shock, Post-Pandemic)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


# Study periods based on research framework
STUDY_PERIODS = {
    'Pre-Pandemic': (2017, 2019),
    'COVID Shock': (2020, 2021),
    'Post-Pandemic': (2022, 2024)
}

PERIOD_COLORS = {
    'Pre-Pandemic': '#2ecc71',
    'COVID Shock': '#e74c3c',
    'Post-Pandemic': '#3498db'
}


def setup_plot_style():
    """Set up matplotlib style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10


def assign_period(year):
    """Assign study period to a given year"""
    for period_name, (start, end) in STUDY_PERIODS.items():
        if start <= year <= end:
            return period_name
    return None


def create_period_based_visualization(yearly_flow_df, transition_rates, results_dir):
    """
    Create comprehensive visualization based on study periods
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
        results_dir: Directory to save results
    """
    print("\nGenerating period-based analysis visualization...")
    setup_plot_style()
    
    # Add period column
    yearly_flow_df['Period'] = yearly_flow_df['Year_To'].apply(assign_period)
    transition_rates['Period'] = transition_rates['Year_To'].apply(assign_period)
    
    # Filter valid periods
    yearly_flow_df = yearly_flow_df[yearly_flow_df['Period'].notna()].copy()
    transition_rates = transition_rates[transition_rates['Period'].notna()].copy()
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Dropout Rate by Period
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_by_period(ax1, yearly_flow_df, 'Dropout_Rate', 
                         'Dropout Rate Over Time', 'Dropout Rate (%)')
    
    # 2. Permanent Exit Rate by Period
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_by_period(ax2, yearly_flow_df, 'Permanent_Exit_Rate',
                         'Permanent Exit Rate Over Time', 'Permanent Exit Rate (%)')
    
    # 3. Comeback Rate by Period
    ax3 = fig.add_subplot(gs[1, 0])
    plot_metric_by_period(ax3, yearly_flow_df, 'Comeback_Rate',
                         'Comeback Rate Over Time', 'Comeback Rate (%)')
    
    # 4. Occupation Change Rate by Period
    ax4 = fig.add_subplot(gs[1, 1])
    plot_metric_by_period(ax4, transition_rates, 'Occupation_Change_Rate',
                         'Occupation Change Rate Over Time', 'Change Rate (%)')
    
    # 5. Period Comparison - Summary Statistics
    ax5 = fig.add_subplot(gs[2, :])
    plot_period_comparison(ax5, yearly_flow_df, transition_rates)
    
    plt.savefig(os.path.join(results_dir, 'period_based_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: period_based_analysis.png")


def plot_metric_by_period(ax, df, metric_col, title, ylabel):
    """
    Plot a metric over time with period highlighting
    
    Args:
        ax: Matplotlib axis
        df: Dataframe with Period and Year_To columns
        metric_col: Column to plot
        title: Plot title
        ylabel: Y-axis label
    """
    years = df['Year_To'].values
    values = df[metric_col].values
    periods = df['Period'].values
    
    # Plot line
    ax.plot(years, values, marker='o', linewidth=2.5, markersize=8, 
            color='#34495e', alpha=0.7, zorder=3)
    
    # Highlight periods with background colors
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        mask = (years >= start_year) & (years <= end_year)
        if mask.any():
            period_years = years[mask]
            period_values = values[mask]
            
            ax.fill_between(period_years, 0, period_values.max() * 1.2,
                           alpha=0.15, color=PERIOD_COLORS[period_name],
                           label=period_name, zorder=1)
    
    # Add value labels
    for x, y in zip(years, values):
        ax.text(x, y + values.max() * 0.02, f'{y:.1f}%', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(y)) for y in years], rotation=45, ha='right')


def plot_period_comparison(ax, yearly_flow_df, transition_rates):
    """
    Create bar chart comparing average metrics across periods
    
    Args:
        ax: Matplotlib axis
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
    """
    # Calculate period averages
    period_stats = []
    
    for period_name in ['Pre-Pandemic', 'COVID Shock', 'Post-Pandemic']:
        flow_period = yearly_flow_df[yearly_flow_df['Period'] == period_name]
        trans_period = transition_rates[transition_rates['Period'] == period_name]
        
        if len(flow_period) > 0 and len(trans_period) > 0:
            period_stats.append({
                'Period': period_name,
                'Dropout': flow_period['Dropout_Rate'].mean(),
                'Permanent Exit': flow_period['Permanent_Exit_Rate'].mean(),
                'Comeback': flow_period['Comeback_Rate'].mean(),
                'Occupation Change': trans_period['Occupation_Change_Rate'].mean()
            })
    
    stats_df = pd.DataFrame(period_stats)
    
    # Plot grouped bar chart
    x = np.arange(len(stats_df))
    width = 0.2
    
    metrics = ['Dropout', 'Permanent Exit', 'Comeback', 'Occupation Change']
    colors_metrics = ['#e74c3c', '#9b59b6', '#2ecc71', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, stats_df[metric], width, 
                     label=metric, color=color, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Rates by Study Period', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['Period'], fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)


def create_transition_flow_by_period(yearly_flow_df, results_dir):
    """
    Create visualization showing absolute numbers of workforce flows by period
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        results_dir: Directory to save results
    """
    print("\nGenerating workforce flow by period...")
    setup_plot_style()
    
    yearly_flow_df['Period'] = yearly_flow_df['Year_To'].apply(assign_period)
    yearly_flow_df = yearly_flow_df[yearly_flow_df['Period'].notna()].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    # Chart 1: Stacked area chart
    ax1 = axes[0]
    
    for period_name, color in PERIOD_COLORS.items():
        period_data = yearly_flow_df[yearly_flow_df['Period'] == period_name]
        
        if len(period_data) > 0:
            years = period_data['Year_To'].values
            exits = period_data['Permanent_Exits'].values
            
            ax1.plot(years, exits, marker='o', linewidth=2.5, markersize=8,
                    label=period_name, color=color, alpha=0.8)
            ax1.fill_between(years, 0, exits, alpha=0.2, color=color)
    
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of People', fontsize=12, fontweight='bold')
    ax1.set_title('Permanent Exits Over Time by Study Period', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Chart 2: Grouped bar chart - Exits vs Comebacks vs New Entrants
    ax2 = axes[1]
    
    years = yearly_flow_df['Year_To'].values
    x_pos = np.arange(len(years))
    width = 0.25
    
    exits = yearly_flow_df['Permanent_Exits'].values
    comebacks = yearly_flow_df['Comebacks'].values
    new_entrants = yearly_flow_df['New_Entrants'].values
    
    ax2.bar(x_pos - width, exits, width, label='Permanent Exits', 
           color='#e74c3c', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos, comebacks, width, label='Comebacks',
           color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos + width, new_entrants, width, label='New Entrants',
           color='#3498db', alpha=0.8, edgecolor='black')
    
    # Period background coloring
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        period_mask = (years >= start_year) & (years <= end_year)
        if period_mask.any():
            period_indices = np.where(period_mask)[0]
            if len(period_indices) > 0:
                ax2.axvspan(period_indices[0] - 0.5, period_indices[-1] + 0.5,
                           alpha=0.1, color=PERIOD_COLORS[period_name], zorder=0)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(int(y)) for y in years], rotation=45, ha='right')
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of People', fontsize=12, fontweight='bold')
    ax2.set_title('Workforce Flow Comparison by Year', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'workforce_flow_by_period.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: workforce_flow_by_period.png")


def create_period_summary_table(yearly_flow_df, transition_rates, results_dir):
    """
    Create summary statistics table for each period
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
        results_dir: Directory to save results
    """
    print("\nGenerating period summary table...")
    
    yearly_flow_df['Period'] = yearly_flow_df['Year_To'].apply(assign_period)
    transition_rates['Period'] = transition_rates['Year_To'].apply(assign_period)
    
    summary_records = []
    
    for period_name in ['Pre-Pandemic', 'COVID Shock', 'Post-Pandemic']:
        flow_period = yearly_flow_df[yearly_flow_df['Period'] == period_name]
        trans_period = transition_rates[transition_rates['Period'] == period_name]
        
        if len(flow_period) > 0:
            summary_records.append({
                'Period': period_name,
                'Years': f"{STUDY_PERIODS[period_name][0]}-{STUDY_PERIODS[period_name][1]}",
                'Avg Dropout Rate (%)': f"{flow_period['Dropout_Rate'].mean():.2f}",
                'Avg Permanent Exit Rate (%)': f"{flow_period['Permanent_Exit_Rate'].mean():.2f}",
                'Avg Comeback Rate (%)': f"{flow_period['Comeback_Rate'].mean():.2f}",
                'Avg Occupation Change Rate (%)': f"{trans_period['Occupation_Change_Rate'].mean():.2f}" if len(trans_period) > 0 else "N/A",
                'Total Permanent Exits': f"{flow_period['Permanent_Exits'].sum():,}",
                'Total Comebacks': f"{flow_period['Comebacks'].sum():,}",
                'Total New Entrants': f"{flow_period['New_Entrants'].sum():,}"
            })
    
    summary_df = pd.DataFrame(summary_records)
    
    # Save to CSV
    output_path = os.path.join(results_dir, 'period_summary_statistics.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: period_summary_statistics.csv")
    
    return summary_df


def create_full_timeline_visualization(yearly_flow_df, transition_rates, results_dir):
    """
    Create visualization for full timeline (1990-2023)
    Shows long-term trends without period segmentation
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
        results_dir: Directory to save results
    """
    print("\nGenerating full timeline visualization (1990-2023)...")
    setup_plot_style()
    
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    years_flow = yearly_flow_df['Year_To'].values
    years_trans = transition_rates['Year_To'].values
    
    # 1. Dropout Rate - Full Timeline
    ax1 = fig.add_subplot(gs[0, 0])
    plot_full_timeline_metric(ax1, years_flow, yearly_flow_df['Dropout_Rate'].values,
                             'Dropout Rate Over Time (1990-2023)', 'Dropout Rate (%)',
                             '#e74c3c')
    
    # 2. Permanent Exit Rate - Full Timeline
    ax2 = fig.add_subplot(gs[0, 1])
    plot_full_timeline_metric(ax2, years_flow, yearly_flow_df['Permanent_Exit_Rate'].values,
                             'Permanent Exit Rate Over Time (1990-2023)', 'Permanent Exit Rate (%)',
                             '#9b59b6')
    
    # 3. Comeback Rate - Full Timeline
    ax3 = fig.add_subplot(gs[1, 0])
    plot_full_timeline_metric(ax3, years_flow, yearly_flow_df['Comeback_Rate'].values,
                             'Comeback Rate Over Time (1990-2023)', 'Comeback Rate (%)',
                             '#2ecc71')
    
    # 4. Occupation Change Rate - Full Timeline
    ax4 = fig.add_subplot(gs[1, 1])
    plot_full_timeline_metric(ax4, years_trans, transition_rates['Occupation_Change_Rate'].values,
                             'Occupation Change Rate Over Time (1990-2023)', 'Change Rate (%)',
                             '#f39c12')
    
    # 5. All Metrics Combined - Long-term Trends
    ax5 = fig.add_subplot(gs[2, :])
    plot_combined_metrics_timeline(ax5, yearly_flow_df, transition_rates)
    
    # 6. Permanent Exits - Absolute Numbers Over Time
    ax6 = fig.add_subplot(gs[3, :])
    plot_permanent_exits_timeline(ax6, yearly_flow_df)
    
    plt.savefig(os.path.join(results_dir, 'full_timeline_1990_2023.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: full_timeline_1990_2023.png")


def plot_full_timeline_metric(ax, years, values, title, ylabel, color):
    """
    Plot a single metric over full timeline
    
    Args:
        ax: Matplotlib axis
        years: Array of years
        values: Array of values
        title: Plot title
        ylabel: Y-axis label
        color: Line color
    """
    # Main line
    ax.plot(years, values, marker='o', linewidth=2, markersize=5,
            color=color, alpha=0.8, zorder=2)
    
    # Add study period highlighting (subtle)
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        if start_year >= years.min() and end_year <= years.max():
            ax.axvspan(start_year, end_year, alpha=0.08,
                      color=PERIOD_COLORS[period_name], zorder=1)
    
    # Highlight COVID shock year
    if 2020 in years:
        covid_idx = list(years).index(2020)
        ax.plot(2020, values[covid_idx], 'r*', markersize=15, 
               label='COVID-19 Shock', zorder=3)
    
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    # X-axis ticks - show every 5 years for readability
    tick_years = [y for y in years if int(y) % 5 == 0 or int(y) == 2020]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')
    
    if 2020 in years:
        ax.legend(loc='best', fontsize=9)


def plot_combined_metrics_timeline(ax, yearly_flow_df, transition_rates):
    """
    Plot all key metrics on same chart for comparison
    
    Args:
        ax: Matplotlib axis
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
    """
    years_flow = yearly_flow_df['Year_To'].values
    years_trans = transition_rates['Year_To'].values
    
    # Plot each metric
    ax.plot(years_flow, yearly_flow_df['Dropout_Rate'].values,
           marker='o', linewidth=2, markersize=4, label='Dropout Rate',
           color='#e74c3c', alpha=0.8)
    
    ax.plot(years_flow, yearly_flow_df['Permanent_Exit_Rate'].values,
           marker='s', linewidth=2, markersize=4, label='Permanent Exit Rate',
           color='#9b59b6', alpha=0.8)
    
    ax.plot(years_flow, yearly_flow_df['Comeback_Rate'].values,
           marker='^', linewidth=2, markersize=4, label='Comeback Rate',
           color='#2ecc71', alpha=0.8)
    
    ax.plot(years_trans, transition_rates['Occupation_Change_Rate'].values,
           marker='d', linewidth=2, markersize=4, label='Occupation Change Rate',
           color='#f39c12', alpha=0.8)
    
    # Highlight COVID period
    ax.axvspan(2020, 2021, alpha=0.15, color='red', label='COVID Shock Period', zorder=0)
    
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('All Key Metrics Over Time (1990-2023)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    # X-axis ticks
    tick_years = [y for y in years_flow if int(y) % 5 == 0 or int(y) in [2020, 2021]]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')


def plot_permanent_exits_timeline(ax, yearly_flow_df):
    """
    Plot permanent exits as bars over full timeline
    
    Args:
        ax: Matplotlib axis
        yearly_flow_df: Yearly flow dataframe
    """
    years = yearly_flow_df['Year_To'].values
    exits = yearly_flow_df['Permanent_Exits'].values
    
    # Create color array - highlight COVID period
    colors = ['#e74c3c' if 2020 <= y <= 2021 else '#95a5a6' for y in years]
    
    bars = ax.bar(years, exits, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels for key years
    for i, (year, exit_count) in enumerate(zip(years, exits)):
        if int(year) % 10 == 0 or int(year) in [2020, 2021]:
            ax.text(year, exit_count, f'{int(exit_count):,}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Number of People', fontsize=12, fontweight='bold')
    ax.set_title('Permanent Exits Over Time (1990-2023)',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    # X-axis ticks
    tick_years = [y for y in years if int(y) % 5 == 0 or int(y) in [2020, 2021]]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='COVID Period (2020-2021)'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Other Years')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)


def create_occupation_evolution_full_timeline(occ_dist_df, results_dir):
    """
    Create occupation evolution plot for full timeline (1990-2023)
    
    Args:
        occ_dist_df: Occupation distribution dataframe
        results_dir: Directory to save results
    """
    print("\nGenerating occupation evolution plot (full timeline)...")
    setup_plot_style()
    
    # Get top occupations based on average across all years
    avg_occ_size = occ_dist_df.groupby('Occupation')['Count'].mean().sort_values(ascending=False)
    top_occupations = list(avg_occ_size.head(10).index)
    
    fig, ax = plt.subplots(figsize=(22, 12))
    colors = sns.color_palette("husl", len(top_occupations))
    
    # Subtle study period backgrounds
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        ax.axvspan(start_year, end_year, alpha=0.05,
                  color=PERIOD_COLORS[period_name], zorder=0)
    
    # COVID shock line
    ax.axvline(x=2020, color='red', linestyle='--', linewidth=2, 
              alpha=0.5, label='COVID-19 Shock', zorder=1)
    
    # Plot each occupation
    for idx, occ in enumerate(top_occupations):
        occ_data = occ_dist_df[occ_dist_df['Occupation'] == occ].sort_values('Year')
        years_occ = occ_data['Year'].values
        percentages = occ_data['Percentage'].values
        
        ax.plot(years_occ, percentages, marker='o', linewidth=2,
               markersize=4, label=occ[:50], color=colors[idx],
               alpha=0.8, zorder=2)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Workforce (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Occupations Evolution Over Time (1990-2023)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # X-axis ticks
    all_years = occ_dist_df['Year'].unique()
    tick_years = [y for y in all_years if int(y) % 5 == 0 or int(y) == 2020]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'occupation_evolution_full_timeline.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_all_period_visualizations(yearly_flow_df, transition_rates, results_dir):
    """
    Create all period-based visualizations
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe  
        results_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("GENERATING PERIOD-BASED VISUALIZATIONS")
    print("="*80)
    
    # Main analysis visualization
    create_period_based_visualization(yearly_flow_df, transition_rates, results_dir)
    
    # Workforce flow by period
    create_transition_flow_by_period(yearly_flow_df, results_dir)
    
    # Summary table
    summary_df = create_period_summary_table(yearly_flow_df, transition_rates, results_dir)
    
    print("\n" + "="*80)
    print("PERIOD SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)


def create_all_visualizations(yearly_flow_df, transition_rates, occ_dist_df, results_dir):
    """
    Create ALL visualizations - both full timeline and period-based
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
        occ_dist_df: Occupation distribution dataframe
        results_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)
    
    # VERSION 1: Full Timeline (1990-2023)
    print("\n[VERSION 1] Full Timeline Analysis (1990-2023)")
    print("-" * 80)
    create_full_timeline_visualization(yearly_flow_df, transition_rates, results_dir)
    create_occupation_evolution_full_timeline(occ_dist_df, results_dir)
    
    # VERSION 2: Study Period Focus (2017-2024)
    print("\n[VERSION 2] Study Period Analysis (2017-2024)")
    print("-" * 80)
    create_all_period_visualizations(yearly_flow_df, transition_rates, results_dir)
    create_occupation_evolution_plot(occ_dist_df, None, results_dir)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)


# Keep original functions for backward compatibility
def create_workforce_flow_dashboard(yearly_flow_df, transition_rates, results_dir):
    """Legacy function - redirects to all visualizations"""
    # This will be called from main.py with occ_dist_df
    pass


def create_occupation_evolution_plot(occ_dist_df, years, results_dir):
    """
    Create occupation evolution plot with period highlighting
    
    Args:
        occ_dist_df: Occupation distribution dataframe
        years: List of years
        results_dir: Directory to save results
    """
    print("\nGenerating occupation evolution plot...")
    setup_plot_style()
    
    # Get top occupations
    avg_occ_size = occ_dist_df.groupby('Occupation')['Count'].mean().sort_values(ascending=False)
    top_occupations = list(avg_occ_size.head(10).index)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    colors = sns.color_palette("husl", len(top_occupations))
    
    # Add period backgrounds
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        ax.axvspan(start_year, end_year, alpha=0.1, 
                  color=PERIOD_COLORS[period_name], label=period_name, zorder=0)
    
    # Plot each occupation
    for idx, occ in enumerate(top_occupations):
        occ_data = occ_dist_df[occ_dist_df['Occupation'] == occ].sort_values('Year')
        years_occ = occ_data['Year'].values
        percentages = occ_data['Percentage'].values
        
        ax.plot(years_occ, percentages, marker='o', linewidth=2.5, 
               markersize=6, label=occ[:50], color=colors[idx], 
               alpha=0.8, zorder=2)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Workforce (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Occupations Evolution Over Time', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'top_occupations_evolution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")
"""
Visualization functions for labor market trend analysis
Enhanced with study period analysis (Pre-Pandemic, COVID Shock, Post-Pandemic)
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    """
    print("\nGenerating period-based analysis visualization...")
    setup_plot_style()
    
    yearly_flow_df['Period'] = yearly_flow_df['Year_To'].apply(assign_period)
    transition_rates['Period'] = transition_rates['Year_To'].apply(assign_period)
    
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
    
    # 5. Period Comparison
    ax5 = fig.add_subplot(gs[2, :])
    plot_period_comparison(ax5, yearly_flow_df, transition_rates)
    
    plt.savefig(os.path.join(results_dir, 'period_based_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: period_based_analysis.png")


def plot_metric_by_period(ax, df, metric_col, title, ylabel):
    """Plot a metric over time with period highlighting"""
    years = df['Year_To'].values
    values = df[metric_col].values
    
    ax.plot(years, values, marker='o', linewidth=2.5, markersize=8, 
            color='#34495e', alpha=0.7, zorder=3)
    
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        mask = (years >= start_year) & (years <= end_year)
        if mask.any():
            period_years = years[mask]
            period_values = values[mask]
            
            ax.fill_between(period_years, 0, period_values.max() * 1.2,
                           alpha=0.15, color=PERIOD_COLORS[period_name],
                           label=period_name, zorder=1)
    
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
    """Create bar chart comparing average metrics across periods"""
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
    
    x = np.arange(len(stats_df))
    width = 0.2
    
    metrics = ['Dropout', 'Permanent Exit', 'Comeback', 'Occupation Change']
    colors_metrics = ['#e74c3c', '#9b59b6', '#2ecc71', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, stats_df[metric], width, 
                     label=metric, color=color, alpha=0.8, edgecolor='black')
        
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
    """Create visualization showing workforce flows by period"""
    print("\nGenerating workforce flow by period...")
    setup_plot_style()
    
    yearly_flow_df['Period'] = yearly_flow_df['Year_To'].apply(assign_period)
    yearly_flow_df = yearly_flow_df[yearly_flow_df['Period'].notna()].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    # Chart 1: Permanent exits over time
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
    
    # Chart 2: Grouped bar chart
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
    """Create summary statistics table for each period"""
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
    
    output_path = os.path.join(results_dir, 'period_summary_statistics.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: period_summary_statistics.csv")
    
    return summary_df


def create_full_timeline_visualization(yearly_flow_df, transition_rates, results_dir):
    """Create visualization for full timeline (2000-2023)"""
    print("\nGenerating full timeline visualization (2000-2023)...")
    setup_plot_style()
    
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    years_flow = yearly_flow_df['Year_To'].values
    years_trans = transition_rates['Year_To'].values
    
    # Various charts...
    ax1 = fig.add_subplot(gs[0, 0])
    plot_full_timeline_metric(ax1, years_flow, yearly_flow_df['Dropout_Rate'].values,
                             'Dropout Rate Over Time (2000-2023)', 'Dropout Rate (%)',
                             '#e74c3c')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_full_timeline_metric(ax2, years_flow, yearly_flow_df['Permanent_Exit_Rate'].values,
                             'Permanent Exit Rate Over Time (2000-2023)', 'Permanent Exit Rate (%)',
                             '#9b59b6')
    
    ax3 = fig.add_subplot(gs[1, 0])
    plot_full_timeline_metric(ax3, years_flow, yearly_flow_df['Comeback_Rate'].values,
                             'Comeback Rate Over Time (2000-2023)', 'Comeback Rate (%)',
                             '#2ecc71')
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_full_timeline_metric(ax4, years_trans, transition_rates['Occupation_Change_Rate'].values,
                             'Occupation Change Rate Over Time (2000-2023)', 'Change Rate (%)',
                             '#f39c12')
    
    ax5 = fig.add_subplot(gs[2, :])
    plot_combined_metrics_timeline(ax5, yearly_flow_df, transition_rates)
    
    ax6 = fig.add_subplot(gs[3, :])
    plot_permanent_exits_timeline(ax6, yearly_flow_df)
    
    plt.savefig(os.path.join(results_dir, 'full_timeline_1990_2023.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: full_timeline_1990_2023.png")


def plot_full_timeline_metric(ax, years, values, title, ylabel, color):
    """Plot a single metric over full timeline"""
    ax.plot(years, values, marker='o', linewidth=2, markersize=5,
            color=color, alpha=0.8, zorder=2)
    
    # Highlight COVID
    if 2020 in years:
        covid_idx = list(years).index(2020)
        ax.plot(2020, values[covid_idx], 'r*', markersize=15, 
               label='COVID-19 Shock', zorder=3)
    
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    tick_years = [y for y in years if int(y) % 5 == 0 or int(y) == 2020]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')
    
    if 2020 in years:
        ax.legend(loc='best', fontsize=9)


def plot_combined_metrics_timeline(ax, yearly_flow_df, transition_rates):
    """Plot all key metrics on same chart"""
    years_flow = yearly_flow_df['Year_To'].values
    years_trans = transition_rates['Year_To'].values
    
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
    
    ax.axvspan(2020, 2021, alpha=0.15, color='red', label='COVID Shock Period', zorder=0)
    
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('All Key Metrics Over Time (2000-2023)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    tick_years = [y for y in years_flow if int(y) % 5 == 0 or int(y) in [2020, 2021]]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')


def plot_permanent_exits_timeline(ax, yearly_flow_df):
    """Plot permanent exits as bars over full timeline"""
    years = yearly_flow_df['Year_To'].values
    exits = yearly_flow_df['Permanent_Exits'].values
    
    colors = ['#e74c3c' if 2020 <= y <= 2021 else '#95a5a6' for y in years]
    
    bars = ax.bar(years, exits, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Number of People', fontsize=12, fontweight='bold')
    ax.set_title('Permanent Exits Over Time (2000-2023)',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    tick_years = [y for y in years if int(y) % 5 == 0 or int(y) in [2020, 2021]]
    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')


def create_occupation_evolution_full_timeline(occ_dist_df, results_dir):
    """Create occupation evolution plot for full timeline"""
    print("\nGenerating occupation evolution plot (full timeline)...")
    setup_plot_style()
    
    avg_occ_size = occ_dist_df.groupby('Occupation')['Count'].mean().sort_values(ascending=False)
    top_occupations = list(avg_occ_size.head(10).index)
    
    fig, ax = plt.subplots(figsize=(22, 12))
    colors = sns.color_palette("husl", len(top_occupations))
    
    ax.axvline(x=2020, color='red', linestyle='--', linewidth=2, 
              alpha=0.5, label='COVID-19 Shock', zorder=1)
    
    for idx, occ in enumerate(top_occupations):
        occ_data = occ_dist_df[occ_dist_df['Occupation'] == occ].sort_values('Year')
        years_occ = occ_data['Year'].values
        percentages = occ_data['Percentage'].values
        
        ax.plot(years_occ, percentages, marker='o', linewidth=2,
               markersize=4, label=occ[:50], color=colors[idx],
               alpha=0.8, zorder=2)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Workforce (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Occupations Evolution Over Time (2000-2023)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'occupation_evolution_full_timeline.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_occupation_evolution_plot(occ_dist_df, years, results_dir):
    """Create occupation evolution plot with period highlighting"""
    print("\nGenerating occupation evolution plot...")
    setup_plot_style()
    
    avg_occ_size = occ_dist_df.groupby('Occupation')['Count'].mean().sort_values(ascending=False)
    top_occupations = list(avg_occ_size.head(10).index)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    colors = sns.color_palette("husl", len(top_occupations))
    
    for period_name, (start_year, end_year) in STUDY_PERIODS.items():
        ax.axvspan(start_year, end_year, alpha=0.1, 
                  color=PERIOD_COLORS[period_name], label=period_name, zorder=0)
    
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


def create_all_period_visualizations(yearly_flow_df, transition_rates, results_dir):
    """Create all period-based visualizations"""
    print("\n" + "="*80)
    print("GENERATING PERIOD-BASED VISUALIZATIONS")
    print("="*80)
    
    create_period_based_visualization(yearly_flow_df, transition_rates, results_dir)
    create_transition_flow_by_period(yearly_flow_df, results_dir)
    summary_df = create_period_summary_table(yearly_flow_df, transition_rates, results_dir)
    
    print("\n" + "="*80)
    print("PERIOD SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)


def create_all_visualizations(yearly_flow_df, transition_rates, occ_dist_df, results_dir):
    """Create ALL visualizations - both full timeline and period-based"""
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)
    
    # VERSION 1: Full Timeline
    print("\n[VERSION 1] Full Timeline Analysis (2000-2023)")
    print("-" * 80)
    create_full_timeline_visualization(yearly_flow_df, transition_rates, results_dir)
    create_occupation_evolution_full_timeline(occ_dist_df, results_dir)
    
    # VERSION 2: Study Period Focus
    print("\n[VERSION 2] Study Period Analysis (2017-2024)")
    print("-" * 80)
    create_all_period_visualizations(yearly_flow_df, transition_rates, results_dir)
    create_occupation_evolution_plot(occ_dist_df, None, results_dir)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
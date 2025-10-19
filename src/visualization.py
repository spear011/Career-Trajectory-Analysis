"""
Visualization functions for labor market trend analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from .config import COVID_YEAR, COLORS, RESULTS_DIR, TOP_N_OCCUPATIONS


def setup_plot_style():
    """Set up matplotlib style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'


def create_workforce_flow_dashboard(yearly_flow_df, transition_rates, results_dir):
    """
    Create comprehensive workforce flow dashboard
    
    Args:
        yearly_flow_df: Yearly flow dataframe
        transition_rates: Transition rates dataframe
        results_dir: Directory to save results
    """
    print("\nGenerating workforce flow dashboard...")
    setup_plot_style()
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    years_flow = yearly_flow_df['Year_To'].values
    covid_idx = list(years_flow).index(COVID_YEAR) if COVID_YEAR in list(years_flow) else len(years_flow)
    
    # 1a. Dropout Rate
    _plot_rate_over_time(
        fig.add_subplot(gs[0, 0]),
        years_flow, yearly_flow_df['Dropout_Rate'].values,
        covid_idx, 'Annual Dropout Rate', 'Dropout Rate (%)',
        COLORS['dropout'], COLORS['permanent_exit']
    )
    
    # 1b. Permanent Exit Rate
    _plot_rate_over_time(
        fig.add_subplot(gs[0, 1]),
        years_flow, yearly_flow_df['Permanent_Exit_Rate'].values,
        covid_idx, 'Permanent Exit Rate (Retirement Year)', 'Permanent Exit Rate (%)',
        '#8e44ad', COLORS['permanent_exit']
    )
    
    # 1c. Comeback Rate
    _plot_rate_over_time(
        fig.add_subplot(gs[1, 0]),
        years_flow, yearly_flow_df['Comeback_Rate'].values,
        covid_idx, 'Career Comeback Rate (Return from Break)', 'Comeback Rate (%)',
        COLORS['entry'], COLORS['comeback']
    )
    
    # 1d. Occupation Change Rate
    years_trans = transition_rates['Year_To'].values
    covid_idx_t = list(years_trans).index(COVID_YEAR) if COVID_YEAR in list(years_trans) else len(years_trans)
    _plot_rate_over_time(
        fig.add_subplot(gs[1, 1]),
        years_trans, transition_rates['Occupation_Change_Rate'].values,
        covid_idx_t, 'Occupation Change Rate', 'Change Rate (%)',
        COLORS['occupation_change'], COLORS['covid']
    )
    
    # 1e. Absolute Numbers: Permanent Exits by Year
    ax5 = fig.add_subplot(gs[2, :])
    _plot_permanent_exits_bars(ax5, yearly_flow_df, years_flow)
    
    # 1f. Workforce Flow Comparison
    ax6 = fig.add_subplot(gs[3, :])
    _plot_workforce_comparison(ax6, yearly_flow_df, years_flow)
    
    filepath = os.path.join(results_dir, 'comprehensive_workforce_flow.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_occupation_evolution_plot(occ_dist_df, years, results_dir):
    """
    Create top occupations evolution plot
    
    Args:
        occ_dist_df: Occupation distribution dataframe
        years: List of years
        results_dir: Directory to save results
    """
    print("Generating occupation evolution plot...")
    setup_plot_style()
    
    # Get top occupations
    avg_occ_size = occ_dist_df.groupby('Occupation')['Count'].mean().sort_values(ascending=False)
    top_occupations = list(avg_occ_size.head(TOP_N_OCCUPATIONS).index)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = sns.color_palette("husl", len(top_occupations))
    
    covid_x_pos = COVID_YEAR
    
    for idx, occ in enumerate(top_occupations):
        occ_data = occ_dist_df[occ_dist_df['Occupation'] == occ].sort_values('Year')
        years_occ = occ_data['Year'].values
        percentages = occ_data['Percentage'].values
        
        covid_idx = np.where(years_occ == COVID_YEAR)[0][0] if COVID_YEAR in years_occ else len(years_occ)
        
        ax.plot(years_occ[:covid_idx+1], percentages[:covid_idx+1], 
                marker='o', linewidth=2.5, markersize=6, 
                label=occ[:50], color=colors[idx], alpha=0.8)
        
        if covid_idx < len(years_occ) - 1:
            ax.plot(years_occ[covid_idx:], percentages[covid_idx:], 
                    marker='s', linewidth=2.5, markersize=6, 
                    color=colors[idx], alpha=0.8, linestyle='--')
    
    ax.axvline(x=covid_x_pos, color='red', linestyle=':', linewidth=3, 
              alpha=0.6, label='COVID Start', zorder=1)
    ax.fill_between([covid_x_pos - 0.5, max(years) + 0.5], 0, ax.get_ylim()[1], 
                    alpha=0.1, color='red', zorder=0)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Workforce (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {TOP_N_OCCUPATIONS} Occupations: Evolution Over Time ({years[0]}-{years[-1]})', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    
    filepath = os.path.join(results_dir, 'top_occupations_evolution.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def _plot_rate_over_time(ax, years, values, covid_idx, title, ylabel, 
                         pre_color, covid_color):
    """Helper function to plot rate over time"""
    ax.plot(years[:covid_idx], values[:covid_idx], 
            marker='o', linewidth=3, markersize=8, color=pre_color, 
            label='Pre-COVID', zorder=3)
    
    if covid_idx < len(years):
        ax.plot(years[covid_idx-1:], values[covid_idx-1:], 
                marker='s', linewidth=3, markersize=8, color=covid_color, 
                label='COVID Era', linestyle='--', zorder=3)
    
    ax.axvline(x=COVID_YEAR, color='red', linestyle=':', linewidth=2.5, alpha=0.7, zorder=2)
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    
    for x, y in zip(years, values):
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                   xytext=(0, 6), ha='center', fontsize=8, fontweight='bold')


def _plot_permanent_exits_bars(ax, yearly_flow_df, years_flow):
    """Helper function to plot permanent exits bar chart"""
    x_pos = np.arange(len(yearly_flow_df))
    perm_exits = yearly_flow_df['Permanent_Exits'].values
    
    bars = ax.bar(x_pos, perm_exits, color=COLORS['permanent_exit'], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    covid_periods = [i for i, y in enumerate(years_flow) if y >= COVID_YEAR]
    for idx in covid_periods:
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.15, color='red', zorder=0)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(y)}" for y in years_flow], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of People', fontsize=11, fontweight='bold')
    ax.set_title('Permanent Exits by Retirement Year (Absolute Numbers)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')


def _plot_workforce_comparison(ax, yearly_flow_df, years_flow):
    """Helper function to plot workforce flow comparison"""
    x_pos = np.arange(len(yearly_flow_df))
    width = 0.25
    
    perm_exits = yearly_flow_df['Permanent_Exits'].values
    comebacks = yearly_flow_df['Comebacks'].values
    new_entrants = yearly_flow_df['New_Entrants'].values
    
    ax.bar(x_pos - width, perm_exits, width, 
           label='Permanent Exits', color=COLORS['permanent_exit'], 
           alpha=0.8, edgecolor='black')
    ax.bar(x_pos, comebacks, width, 
           label='Comebacks', color=COLORS['comeback'], 
           alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width, new_entrants, width, 
           label='New Entrants', color=COLORS['pre_covid'], 
           alpha=0.8, edgecolor='black')
    
    covid_periods = [i for i, y in enumerate(years_flow) if y >= COVID_YEAR]
    for idx in covid_periods:
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.15, color='red', zorder=0)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(y)}" for y in years_flow], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of People', fontsize=12, fontweight='bold')
    ax.set_title('Workforce Flow: Exits vs Comebacks vs New Entrants', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
"""
Occupation-level flow analysis
Tracks where people go when they leave specific occupations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


def analyze_occupation_outflows(user_details_df, window_users, windows, 
                                occupation_col='SOC_EMSI_2019_3_NAME'):
    """
    Analyze where people go when they leave each occupation
    OPTIMIZED: Use groupby for faster aggregation
    
    Args:
        user_details_df: Detailed user paths dataframe
        window_users: Dictionary of window users (not used but kept for API consistency)
        windows: List of windows
        occupation_col: Occupation column name (not used but kept for API consistency)
    
    Returns:
        Dataframe with occupation outflow analysis
    """
    print("\n" + "="*80)
    print("OCCUPATION OUTFLOW ANALYSIS (OPTIMIZED)")
    print("="*80)
    
    outflow_records = []
    
    for i in range(len(windows) - 1):
        _, _, _, year_from = windows[i]
        _, _, _, year_to = windows[i + 1]
        
        period_data = user_details_df[
            (user_details_df['Year_From'] == int(year_from)) & 
            (user_details_df['Year_To'] == int(year_to))
        ].copy()
        
        # Validate data exists
        if len(period_data) == 0:
            print(f"  Warning: No data for {year_from}→{year_to}")
            continue
        
        # OPTIMIZATION: Group by occupation once
        grouped = period_data.groupby('From_Occupation', dropna=True)
        
        for occ, occ_data in grouped:
            total_from_occ = len(occ_data)
            
            # OPTIMIZATION: Vectorized counting
            stayed_same = ((occ_data['Status'] == 'stayed') & 
                          (occ_data['Occupation_Changed'] == False)).sum()
            
            changed_mask = ((occ_data['Status'] == 'stayed') & 
                           (occ_data['Occupation_Changed'] == True))
            changed_occ_count = changed_mask.sum()
            
            career_break = (occ_data['Status'] == 'dropout_career_break').sum()
            perm_exit = (occ_data['Status'] == 'dropout_permanent_exit').sum()
            
            # VALIDATION: Check totals match
            total_check = stayed_same + changed_occ_count + career_break + perm_exit
            if total_check != total_from_occ:
                print(f"  ⚠️  Warning: Total mismatch for {occ}: {total_check} != {total_from_occ}")
            
            # Get destination occupations for those who changed
            if changed_occ_count > 0:
                dest_occupations = occ_data[changed_mask]['To_Occupation'].value_counts().to_dict()
            else:
                dest_occupations = {}
            
            outflow_records.append({
                'Year_From': int(year_from),
                'Year_To': int(year_to),
                'Period': f'{year_from}→{year_to}',
                'From_Occupation': occ,
                'Total_Count': total_from_occ,
                'Stayed_Same_Occ': int(stayed_same),
                'Changed_Occupation': int(changed_occ_count),
                'Career_Break': int(career_break),
                'Permanent_Exit': int(perm_exit),
                'Retention_Rate': (stayed_same / total_from_occ) * 100,
                'Turnover_Rate': ((changed_occ_count + career_break + perm_exit) / total_from_occ) * 100,
                'Permanent_Exit_Rate': (perm_exit / total_from_occ) * 100,
                'Top_Destination_Occupations': dest_occupations
            })
    
    print(f"  ✓ Analyzed {len(outflow_records):,} occupation-period combinations")
    return pd.DataFrame(outflow_records)


def analyze_occupation_inflows(user_details_df, window_users, windows):
    """
    Analyze where people come from when entering each occupation
    OPTIMIZED: Use groupby for faster aggregation
    
    Returns:
        Dataframe with occupation inflow analysis
    """
    print("\nOCCUPATION INFLOW ANALYSIS (OPTIMIZED)")
    print("="*80)
    
    inflow_records = []
    
    for i in range(len(windows) - 1):
        _, _, _, year_from = windows[i]
        _, _, _, year_to = windows[i + 1]
        
        period_data = user_details_df[
            (user_details_df['Year_From'] == int(year_from)) & 
            (user_details_df['Year_To'] == int(year_to))
        ].copy()
        
        # OPTIMIZATION: Group by target occupation once
        grouped = period_data.groupby('To_Occupation', dropna=True)
        
        for occ, occ_data in grouped:
            total_to_occ = len(occ_data)
            
            # OPTIMIZATION: Vectorized counting
            stayed_same = ((occ_data['Status'] == 'stayed') & 
                          (occ_data['Occupation_Changed'] == False)).sum()
            
            from_diff_mask = ((occ_data['Status'] == 'stayed') & 
                             (occ_data['Occupation_Changed'] == True))
            from_diff_count = from_diff_mask.sum()
            
            new_entrants = (occ_data['Status'] == 'new_entrant').sum()
            comebacks = (occ_data['Status'] == 'comeback').sum()
            
            # Get source occupations
            if from_diff_count > 0:
                source_occupations = occ_data[from_diff_mask]['From_Occupation'].value_counts().to_dict()
            else:
                source_occupations = {}
            
            inflow_records.append({
                'Year_From': int(year_from),
                'Year_To': int(year_to),
                'Period': f'{year_from}→{year_to}',
                'To_Occupation': occ,
                'Total_Count': total_to_occ,
                'Stayed_Same_Occ': stayed_same,
                'From_Other_Occ': from_diff_count,
                'New_Entrants': new_entrants,
                'Comebacks': comebacks,
                'Internal_Retention_Rate': (stayed_same / total_to_occ) * 100,
                'External_Recruitment_Rate': ((from_diff_count + new_entrants) / total_to_occ) * 100,
                'Top_Source_Occupations': source_occupations
            })
    
    print(f"  ✓ Analyzed {len(inflow_records):,} occupation-period combinations")
    return pd.DataFrame(inflow_records)


def find_critical_transitions(outflow_df, min_count=100, top_n=10):
    """
    Find the most significant occupation-to-occupation transitions
    OPTIMIZED: Vectorized processing
    
    Args:
        outflow_df: Occupation outflow dataframe
        min_count: Minimum count to consider
        top_n: Number of top transitions to return
    
    Returns:
        Dataframe with top transitions
    """
    print("\nFINDING CRITICAL OCCUPATION TRANSITIONS (OPTIMIZED)")
    print("="*80)
    
    transition_records = []
    
    # OPTIMIZATION: Process all at once instead of row-by-row
    for _, row in outflow_df.iterrows():
        if not row['Top_Destination_Occupations']:
            continue
            
        from_occ = row['From_Occupation']
        period = row['Period']
        year_from = row['Year_From']
        year_to = row['Year_To']
        from_total = row['Total_Count']
        
        for to_occ, count in row['Top_Destination_Occupations'].items():
            if count >= min_count:
                transition_records.append({
                    'Period': period,
                    'Year_From': year_from,
                    'Year_To': year_to,
                    'From_Occupation': from_occ,
                    'To_Occupation': to_occ,
                    'Count': count,
                    'From_Total': from_total,
                    'Transition_Rate': (count / from_total) * 100
                })
    
    if not transition_records:
        print("  No transitions found with minimum count threshold")
        return pd.DataFrame()
    
    transitions_df = pd.DataFrame(transition_records)
    transitions_df = transitions_df.sort_values('Count', ascending=False)
    
    print(f"  Found {len(transitions_df):,} significant transitions")
    print(f"\n  Top {top_n} transitions by volume:")
    for idx, row in transitions_df.head(top_n).iterrows():
        print(f"    {row['Period']}: {row['From_Occupation'][:40]} → {row['To_Occupation'][:40]}")
        print(f"      {row['Count']:,} people ({row['Transition_Rate']:.1f}% of source occupation)\n")
    
    return transitions_df


def plot_occupation_turnover_heatmap(outflow_df, results_dir, top_n=15):
    """
    Create heatmap of occupation turnover rates over time
    
    Args:
        outflow_df: Occupation outflow dataframe
        results_dir: Directory to save results
        top_n: Number of occupations to include
    """
    print("\nGenerating occupation turnover heatmap...")
    
    # Get top occupations by average turnover
    avg_turnover = outflow_df.groupby('From_Occupation')['Turnover_Rate'].mean().sort_values(ascending=False)
    top_occupations = list(avg_turnover.head(top_n).index)
    
    # Create pivot table
    pivot_data = outflow_df[outflow_df['From_Occupation'].isin(top_occupations)].pivot_table(
        values='Turnover_Rate',
        index='From_Occupation',
        columns='Period',
        aggfunc='mean'
    )
    
    # Reorder by average turnover
    pivot_data = pivot_data.reindex(top_occupations)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
               linewidths=0.5, ax=ax, cbar_kws={'label': 'Turnover Rate (%)'})
    
    ax.set_xlabel('Period', fontsize=12, fontweight='bold')
    ax.set_ylabel('Occupation', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Occupations by Turnover Rate Over Time', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    filepath = os.path.join(results_dir, 'occupation_turnover_heatmap.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filepath}")


def plot_occupation_exit_analysis(outflow_df, results_dir, year_to=2021, top_n=15):
    """
    Create bar chart showing occupation exit patterns for a specific year
    
    Args:
        outflow_df: Occupation outflow dataframe
        results_dir: Directory to save results
        year_to: Year to analyze
        top_n: Number of occupations to show
    """
    print(f"\nGenerating occupation exit analysis for {year_to}...")
    
    # Filter for specific year
    year_data = outflow_df[outflow_df['Year_To'] == year_to].copy()
    
    if len(year_data) == 0:
        print(f"  No data for year {year_to}")
        return
    
    # Sort by permanent exit rate
    year_data = year_data.sort_values('Permanent_Exit_Rate', ascending=False).head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Chart 1: Permanent Exit Rate
    y_pos = np.arange(len(year_data))
    ax1.barh(y_pos, year_data['Permanent_Exit_Rate'].values, color='#c0392b', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([occ[:45] for occ in year_data['From_Occupation'].values], fontsize=9)
    ax1.set_xlabel('Permanent Exit Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Occupations by Permanent Exit Rate ({year_to})', 
                 fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(year_data.iterrows()):
        ax1.text(row['Permanent_Exit_Rate'] + 0.2, i, 
                f"{row['Permanent_Exit_Rate']:.1f}%\n({int(row['Permanent_Exit']):,})", 
                va='center', fontsize=8, fontweight='bold')
    
    # Chart 2: Breakdown of outflows
    year_data_sorted = year_data.sort_values('Total_Count', ascending=False)
    y_pos2 = np.arange(len(year_data_sorted))
    
    width = 0.8
    ax2.barh(y_pos2, year_data_sorted['Stayed_Same_Occ'].values, 
            label='Stayed', color='#27ae60', alpha=0.8, height=width)
    ax2.barh(y_pos2, year_data_sorted['Changed_Occupation'].values, 
            left=year_data_sorted['Stayed_Same_Occ'].values,
            label='Changed Occupation', color='#3498db', alpha=0.8, height=width)
    ax2.barh(y_pos2, year_data_sorted['Career_Break'].values,
            left=year_data_sorted['Stayed_Same_Occ'].values + year_data_sorted['Changed_Occupation'].values,
            label='Career Break', color='#f39c12', alpha=0.8, height=width)
    ax2.barh(y_pos2, year_data_sorted['Permanent_Exit'].values,
            left=year_data_sorted['Stayed_Same_Occ'].values + year_data_sorted['Changed_Occupation'].values + year_data_sorted['Career_Break'].values,
            label='Permanent Exit', color='#c0392b', alpha=0.8, height=width)
    
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([occ[:45] for occ in year_data_sorted['From_Occupation'].values], fontsize=9)
    ax2.set_xlabel('Number of People', fontsize=11, fontweight='bold')
    ax2.set_title(f'Occupation Outflow Breakdown ({year_to})', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, f'occupation_exit_analysis_{year_to}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filepath}")


def create_transition_flow_summary(transitions_df, results_dir, year_to=2021, top_n=20):
    """
    Create summary of top occupation transitions for a specific year
    
    Args:
        transitions_df: Transitions dataframe
        results_dir: Directory to save results
        year_to: Year to analyze
        top_n: Number of transitions to show
    """
    print(f"\nGenerating transition flow summary for {year_to}...")
    
    year_transitions = transitions_df[transitions_df['Year_To'] == year_to].copy()
    
    if len(year_transitions) == 0:
        print(f"  No transitions for year {year_to}")
        return
    
    year_transitions = year_transitions.sort_values('Count', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    y_pos = np.arange(len(year_transitions))
    bars = ax.barh(y_pos, year_transitions['Count'].values, color='#3498db', alpha=0.8)
    
    # Create labels
    labels = []
    for _, row in year_transitions.iterrows():
        from_short = row['From_Occupation'][:35]
        to_short = row['To_Occupation'][:35]
        labels.append(f"{from_short} → {to_short}")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of People', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Occupation Transitions ({year_to})', 
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(year_transitions.iterrows()):
        ax.text(row['Count'] + max(year_transitions['Count']) * 0.01, i,
               f"{int(row['Count']):,}\n({row['Transition_Rate']:.1f}%)",
               va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, f'occupation_transitions_{year_to}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filepath}")
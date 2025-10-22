"""
Network visualization module for occupational mobility analysis
Implements Sankey diagrams and transition matrix comparisons
"""
import pandas as pd
import numpy as np
from collections import defaultdict


def build_transition_matrix(all_transitions_df, period_filter=None, top_n=15):
    """
    Build transition matrix from transitions dataframe
    
    Args:
        all_transitions_df: All transitions dataframe
        period_filter: Tuple of (year_from, year_to) or None for all periods
        top_n: Number of top occupations to include
    
    Returns:
        Tuple of (transition_matrix, occupation_list, total_transitions)
    """
    df = all_transitions_df.copy()
    
    # Filter by period if specified
    if period_filter:
        year_from, year_to = period_filter
        df = df[(df['Year_From'] >= year_from) & (df['Year_To'] <= year_to)]
    
    # Get top occupations by total transition volume
    from_counts = df['From_Occupation'].value_counts()
    to_counts = df['To_Occupation'].value_counts()
    total_counts = (from_counts + to_counts).fillna(0)
    top_occupations = list(total_counts.nlargest(top_n).index)
    
    # Filter transitions to only include top occupations
    df = df[
        df['From_Occupation'].isin(top_occupations) & 
        df['To_Occupation'].isin(top_occupations)
    ]
    
    # Build matrix
    matrix = pd.crosstab(
        df['From_Occupation'], 
        df['To_Occupation'], 
        values=df['ID'], 
        aggfunc='count',
        dropna=False
    ).fillna(0)
    
    # Ensure all top occupations are in both dimensions
    for occ in top_occupations:
        if occ not in matrix.index:
            matrix.loc[occ] = 0
        if occ not in matrix.columns:
            matrix[occ] = 0
    
    # Reindex to maintain order
    matrix = matrix.reindex(index=top_occupations, columns=top_occupations, fill_value=0)
    
    total_transitions = df.shape[0]
    
    return matrix, top_occupations, total_transitions


def normalize_transition_matrix(matrix):
    """
    Normalize transition matrix (row-wise probabilities)
    
    Args:
        matrix: Transition count matrix
    
    Returns:
        Normalized matrix (each row sums to 1)
    """
    row_sums = matrix.sum(axis=1)
    # Avoid division by zero
    row_sums = row_sums.replace(0, 1)
    normalized = matrix.div(row_sums, axis=0)
    return normalized


def compare_transition_matrices(matrix1, matrix2):
    """
    Compare two transition matrices (difference)
    
    Args:
        matrix1: First transition matrix (normalized)
        matrix2: Second transition matrix (normalized)
    
    Returns:
        Difference matrix (matrix2 - matrix1)
    """
    # Ensure same shape
    all_occupations = sorted(set(matrix1.index) | set(matrix2.index))
    
    matrix1_aligned = matrix1.reindex(
        index=all_occupations, 
        columns=all_occupations, 
        fill_value=0
    )
    matrix2_aligned = matrix2.reindex(
        index=all_occupations, 
        columns=all_occupations, 
        fill_value=0
    )
    
    diff_matrix = matrix2_aligned - matrix1_aligned
    
    return diff_matrix


def prepare_sankey_data(all_transitions_df, period_filter=None, top_n=10, min_flow=10):
    """
    Prepare data for Sankey diagram
    
    Args:
        all_transitions_df: All transitions dataframe
        period_filter: Tuple of (year_from, year_to) or None
        top_n: Number of top occupations to show
        min_flow: Minimum flow count to include
    
    Returns:
        Dictionary with source, target, value, and labels
    """
    df = all_transitions_df.copy()
    
    # Filter by period
    if period_filter:
        year_from, year_to = period_filter
        df = df[(df['Year_From'] >= year_from) & (df['Year_To'] <= year_to)]
    
    # Check if we have any data
    if len(df) == 0:
        return {
            'source': [],
            'target': [],
            'value': [],
            'labels': [],
            'flow_counts': pd.DataFrame()
        }
    
    # Get top occupations
    from_counts = df['From_Occupation'].value_counts()
    to_counts = df['To_Occupation'].value_counts()
    total_counts = (from_counts + to_counts).fillna(0)
    
    if len(total_counts) == 0:
        return {
            'source': [],
            'target': [],
            'value': [],
            'labels': [],
            'flow_counts': pd.DataFrame()
        }
    
    top_occupations = list(total_counts.nlargest(min(top_n, len(total_counts))).index)
    
    # Filter to top occupations
    df = df[
        df['From_Occupation'].isin(top_occupations) & 
        df['To_Occupation'].isin(top_occupations)
    ]
    
    if len(df) == 0:
        return {
            'source': [],
            'target': [],
            'value': [],
            'labels': [],
            'flow_counts': pd.DataFrame()
        }
    
    # Aggregate transitions
    flow_counts = df.groupby(['From_Occupation', 'To_Occupation']).size().reset_index(name='count')
    flow_counts = flow_counts[flow_counts['count'] >= min_flow]
    
    if len(flow_counts) == 0:
        # Try with lower threshold
        flow_counts = df.groupby(['From_Occupation', 'To_Occupation']).size().reset_index(name='count')
        flow_counts = flow_counts[flow_counts['count'] >= max(1, min_flow // 2)]
    
    # Create node labels (unique occupations)
    all_occupations = sorted(set(flow_counts['From_Occupation']) | set(flow_counts['To_Occupation']))
    
    if len(all_occupations) == 0:
        return {
            'source': [],
            'target': [],
            'value': [],
            'labels': [],
            'flow_counts': pd.DataFrame()
        }
    
    node_dict = {occ: idx for idx, occ in enumerate(all_occupations)}
    
    # Prepare Sankey data
    source = [node_dict[occ] for occ in flow_counts['From_Occupation']]
    target = [node_dict[occ] for occ in flow_counts['To_Occupation']]
    value = flow_counts['count'].tolist()
    
    return {
        'source': source,
        'target': target,
        'value': value,
        'labels': all_occupations,
        'flow_counts': flow_counts
    }


def get_period_ranges():
    """
    Define pre/during/post COVID period ranges
    
    Returns:
        Dictionary with period definitions
    """
    return {
        'pre_covid': (2015, 2019),
        'covid': (2020, 2021),
        'post_covid': (2022, 2023)
    }


def aggregate_transitions_by_period(all_transitions_df):
    """
    Aggregate transitions into pre/during/post COVID periods
    
    Args:
        all_transitions_df: All transitions dataframe
    
    Returns:
        Dictionary with DataFrames for each period
    """
    periods = get_period_ranges()
    period_data = {}
    
    for period_name, (year_from, year_to) in periods.items():
        df = all_transitions_df[
            (all_transitions_df['Year_From'] >= year_from) & 
            (all_transitions_df['Year_To'] <= year_to)
        ].copy()
        period_data[period_name] = df
    
    return period_data


def calculate_flow_statistics(all_transitions_df, periods=None):
    """
    Calculate flow statistics for each period
    
    Args:
        all_transitions_df: All transitions dataframe
        periods: Dictionary of period ranges or None for default
    
    Returns:
        DataFrame with statistics by period
    """
    if periods is None:
        periods = get_period_ranges()
    
    stats = []
    
    for period_name, (year_from, year_to) in periods.items():
        df = all_transitions_df[
            (all_transitions_df['Year_From'] >= year_from) & 
            (all_transitions_df['Year_To'] <= year_to)
        ]
        
        total_transitions = len(df)
        occupation_changes = df['Occupation_Changed'].sum()
        industry_changes = df['Industry_Changed'].sum()
        state_changes = (df['From_State'] != df['To_State']).sum()
        
        unique_from_occ = df['From_Occupation'].nunique()
        unique_to_occ = df['To_Occupation'].nunique()
        
        stats.append({
            'Period': period_name,
            'Year_Range': f"{year_from}-{year_to}",
            'Total_Transitions': total_transitions,
            'Occupation_Changes': occupation_changes,
            'Occupation_Change_Rate': (occupation_changes / total_transitions * 100) if total_transitions > 0 else 0,
            'Industry_Changes': industry_changes,
            'Industry_Change_Rate': (industry_changes / total_transitions * 100) if total_transitions > 0 else 0,
            'State_Changes': state_changes,
            'State_Change_Rate': (state_changes / total_transitions * 100) if total_transitions > 0 else 0,
            'Unique_From_Occupations': unique_from_occ,
            'Unique_To_Occupations': unique_to_occ
        })
    
    return pd.DataFrame(stats)
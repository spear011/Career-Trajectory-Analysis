"""
Occupation and industry transition analysis
Year-over-year tracking of career changes
"""
import pandas as pd
from .config import OCCUPATION_COLUMN


def build_transitions(window_users, windows):
    """
    Build year-to-year transitions for users who stayed
    
    Args:
        window_users: Dictionary of window users
        windows: List of windows
    
    Returns:
        Combined dataframe of all transitions
    """
    print("\n" + "="*80)
    print("BUILDING YEAR-TO-YEAR TRANSITIONS")
    print("="*80)
    
    transitions = []
    
    for i in range(len(windows) - 1):
        w_from_name, _, _, w_from_label = windows[i]
        w_to_name, _, _, w_to_label = windows[i + 1]
        
        from_users = window_users[w_from_name]
        to_users = window_users[w_to_name]
        
        from_ids = set(from_users['ID'])
        to_ids = set(to_users['ID'])
        
        # Find transitions (users in both years)
        transition_ids = from_ids & to_ids
        
        print(f"{w_from_label}→{w_to_label}: {len(transition_ids):,} users stayed")
        
        # Build transition dataframe for users who stayed
        if len(transition_ids) > 0:
            from_indexed = from_users[from_users['ID'].isin(transition_ids)].set_index('ID')
            to_indexed = to_users[to_users['ID'].isin(transition_ids)].set_index('ID')
            
            merged = from_indexed.merge(to_indexed, left_index=True, right_index=True, 
                                        suffixes=('_from', '_to'))
            
            trans_df = pd.DataFrame({
                'ID': merged.index,
                'Year_From': int(w_from_label),
                'Year_To': int(w_to_label),
                'Period': f'{w_from_label}→{w_to_label}',
                'From_Occupation': merged[f'{OCCUPATION_COLUMN}_from'],
                'To_Occupation': merged[f'{OCCUPATION_COLUMN}_to'],
                'From_Industry': merged['NAICS6_NAME_from'],
                'To_Industry': merged['NAICS6_NAME_to'],
                'From_State': merged['STATE_RAW_from'],
                'To_State': merged['STATE_RAW_to'],
                'Occupation_Changed': merged[f'{OCCUPATION_COLUMN}_from'] != merged[f'{OCCUPATION_COLUMN}_to'],
                'Industry_Changed': merged['NAICS6_NAME_from'] != merged['NAICS6_NAME_to'],
                'State_Changed': merged['STATE_RAW_from'] != merged['STATE_RAW_to']
            }).reset_index(drop=True)
            
            transitions.append(trans_df)
    
    return pd.concat(transitions, ignore_index=True)


def calculate_transition_rates(all_transitions_df):
    """
    Calculate transition rates by period
    
    Args:
        all_transitions_df: Combined transitions dataframe
    
    Returns:
        Dataframe with transition rates by period
    """
    transition_rates = all_transitions_df.groupby('Period').agg({
        'Occupation_Changed': 'mean',
        'Industry_Changed': 'mean',
        'State_Changed': 'mean',
        'ID': 'count'
    }).reset_index()
    
    transition_rates.columns = ['Period', 'Occupation_Change_Rate', 'Industry_Change_Rate', 
                                'State_Change_Rate', 'Users']
    transition_rates['Occupation_Change_Rate'] *= 100
    transition_rates['Industry_Change_Rate'] *= 100
    transition_rates['State_Change_Rate'] *= 100
    transition_rates['Year_To'] = [int(p.split('→')[1]) for p in transition_rates['Period']]
    
    return transition_rates
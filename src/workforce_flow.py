"""
Workforce flow analysis: dropouts, comebacks, permanent exits
OPTIMIZED VERSION with pre-computed user presence mapping
"""
import pandas as pd
import os
from tqdm import tqdm


def _get_cache_filepath(results_dir, year_from, year_to):
    """Get cache file path for a transition"""
    from .utils import ensure_dir
    cache_dir = os.path.join(results_dir, 'workforce_flow_cache')
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, f'transition_{year_from}_to_{year_to}.csv')


def _get_users_cache_filepath(results_dir, year_from, year_to):
    """Get user-level cache file path"""
    from .utils import ensure_dir
    cache_dir = os.path.join(results_dir, 'workforce_flow_cache')
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, f'users_{year_from}_to_{year_to}.csv')


def _load_cached_transition(results_dir, year_from, year_to):
    """Load cached transition if exists"""
    filepath = _get_cache_filepath(results_dir, year_from, year_to)
    users_filepath = _get_users_cache_filepath(results_dir, year_from, year_to)
    
    if os.path.exists(filepath) and os.path.exists(users_filepath):
        try:
            summary = pd.read_csv(filepath).iloc[0].to_dict()
            # Fix DtypeWarning: specify dtypes for mixed-type columns
            users = pd.read_csv(
                users_filepath,
                dtype={
                    'Occupation_Changed': 'object',  # Allow None/bool mix
                    'Industry_Changed': 'object',
                    'State_Changed': 'object'
                }
            )
            return summary, users
        except Exception as e:
            print(f"  Warning: Cache corrupted, recomputing...")
            return None, None
    return None, None


def _save_transition_cache(results_dir, year_from, year_to, transition_data, user_data):
    """Save transition and user-level data to cache"""
    filepath = _get_cache_filepath(results_dir, year_from, year_to)
    pd.DataFrame([transition_data]).to_csv(filepath, index=False)
    
    users_filepath = _get_users_cache_filepath(results_dir, year_from, year_to)
    user_data.to_csv(users_filepath, index=False)


def _build_user_presence_map(window_users, windows):
    """
    Pre-compute which windows each user appears in
    This is the KEY optimization - O(1) lookups instead of O(n) searches
    
    Returns:
        dict: user_id -> set of window indices where user appears
    """
    print("  Building user presence map (optimization)...")
    user_presence = {}
    
    for window_idx, (window_name, _, _, _) in enumerate(windows):
        user_ids = set(window_users[window_name]['ID'])
        for user_id in user_ids:
            if user_id not in user_presence:
                user_presence[user_id] = set()
            user_presence[user_id].add(window_idx)
    
    print(f"  ✓ Mapped {len(user_presence):,} users across {len(windows)} windows")
    return user_presence


def _build_user_info_cache(window_users, windows, occupation_col):
    """
    Pre-build user info cache for fast lookups
    
    Returns:
        dict: (window_idx, user_id) -> user_info dict
    """
    print("  Building user info cache...")
    user_info_cache = {}
    
    for window_idx, (window_name, _, _, _) in enumerate(windows):
        users_df = window_users[window_name].set_index('ID')
        
        for user_id in users_df.index:
            user_row = users_df.loc[user_id]
            user_info_cache[(window_idx, user_id)] = {
                'Occupation': user_row[occupation_col],
                'Industry': user_row['NAICS6_NAME'],
                'State': user_row['STATE_RAW'],
                'Job_Start_Date': user_row['JOB_START_DATE']
            }
    
    print(f"  ✓ Cached info for {len(user_info_cache):,} user-window pairs")
    return user_info_cache


def analyze_workforce_flow(window_users, windows, results_dir='results', 
                          occupation_col='SOC_EMSI_2019_3_NAME'):
    """
    Comprehensive workforce flow analysis with optimizations
    
    OPTIMIZATIONS:
    - Pre-compute user presence across all windows (O(1) lookups)
    - Cache user info to avoid repeated dataframe lookups
    - Vectorize where possible
    
    Args:
        window_users: Dictionary of window users
        windows: List of windows
        results_dir: Directory for caching results
        occupation_col: Occupation column name
    
    Returns:
        Tuple of (summary dataframe, detailed user dataframe)
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE WORKFORCE FLOW ANALYSIS (OPTIMIZED)")
    print("="*80)
    
    # OPTIMIZATION: Pre-compute user presence map once for all transitions
    user_presence = _build_user_presence_map(window_users, windows)
    user_info_cache = _build_user_info_cache(window_users, windows, occupation_col)
    
    yearly_flow = []
    all_user_details = []
    
    for i in range(len(windows) - 1):
        w_from_name, _, _, w_from_label = windows[i]
        w_to_name, _, _, w_to_label = windows[i + 1]
        
        # Check cache first
        cached, cached_users = _load_cached_transition(results_dir, w_from_label, w_to_label)
        if cached is not None:
            print(f"\n{w_from_label}→{w_to_label}: Loading from cache...")
            yearly_flow.append(cached)
            all_user_details.append(cached_users)
            print(f"  Dropouts: {cached['Dropouts']:,} ({cached['Dropout_Rate']:.2f}%)")
            print(f"    └─ Career breaks: {cached['Career_Breaks']:,}")
            print(f"    └─ Permanent exits: {cached['Permanent_Exits']:,}")
            print(f"  Comebacks: {cached['Comebacks']:,} ({cached['Comeback_Rate']:.2f}%)")
            print(f"  New entrants: {cached['New_Entrants']:,} ({cached['Entry_Rate']:.2f}%)")
            continue
        
        print(f"\n{w_from_label}→{w_to_label}: Computing (optimized)...")
        
        from_users = window_users[w_from_name]
        to_users = window_users[w_to_name]
        
        from_ids = set(from_users['ID'])
        to_ids = set(to_users['ID'])
        
        # 1. DROPOUTS
        dropout_ids = from_ids - to_ids
        
        # 2. OPTIMIZED: Classify dropouts using pre-computed presence map
        career_breaks = set()
        permanent_exits = set()
        
        for dropout_id in dropout_ids:
            # O(1) lookup instead of O(n) search!
            if dropout_id in user_presence:
                future_windows = user_presence[dropout_id]
                # Check if appears in any future window
                if any(window_idx > i + 1 for window_idx in future_windows):
                    career_breaks.add(dropout_id)
                else:
                    permanent_exits.add(dropout_id)
            else:
                permanent_exits.add(dropout_id)
        
        # 3. OPTIMIZED: Comebacks using pre-computed presence map
        comeback_ids = set()
        comeback_last_seen = {}
        
        for user_id in to_ids:
            if user_id not in from_ids and user_id in user_presence:
                # FIXED: Use w < i for clarity (though w <= i would work due to user_id not in from_ids)
                past_windows = [w for w in user_presence[user_id] if w < i]
                if past_windows:
                    comeback_ids.add(user_id)
                    comeback_last_seen[user_id] = max(past_windows)
        
        # 4. OPTIMIZED: New entrants
        # Check if user_id appears in any window <= i
        new_entrants = set()
        for user_id in to_ids:
            if user_id not in from_ids:
                if user_id not in user_presence or all(w > i for w in user_presence[user_id]):
                    new_entrants.add(user_id)
        
        # 5. STAYED
        stayed_ids = from_ids & to_ids
        
        # Build detailed user-level data using cached info
        user_records = []
        
        # Career breaks
        for user_id in career_breaks:
            from_info = user_info_cache.get((i, user_id))
            user_records.append({
                'ID': user_id,
                'Year_From': int(w_from_label),
                'Year_To': int(w_to_label),
                'Status': 'dropout_career_break',
                'From_Occupation': from_info['Occupation'] if from_info else None,
                'From_Industry': from_info['Industry'] if from_info else None,
                'From_State': from_info['State'] if from_info else None,
                'To_Occupation': None,
                'To_Industry': None,
                'To_State': None,
                'Occupation_Changed': None,
                'Industry_Changed': None,
                'State_Changed': None
            })
        
        # Permanent exits
        for user_id in permanent_exits:
            from_info = user_info_cache.get((i, user_id))
            user_records.append({
                'ID': user_id,
                'Year_From': int(w_from_label),
                'Year_To': int(w_to_label),
                'Status': 'dropout_permanent_exit',
                'From_Occupation': from_info['Occupation'] if from_info else None,
                'From_Industry': from_info['Industry'] if from_info else None,
                'From_State': from_info['State'] if from_info else None,
                'To_Occupation': None,
                'To_Industry': None,
                'To_State': None,
                'Occupation_Changed': None,
                'Industry_Changed': None,
                'State_Changed': None
            })
        
        # Comebacks
        for user_id in comeback_ids:
            to_info = user_info_cache.get((i + 1, user_id))
            last_seen_idx = comeback_last_seen[user_id]
            from_info = user_info_cache.get((last_seen_idx, user_id))
            
            user_records.append({
                'ID': user_id,
                'Year_From': int(w_from_label),
                'Year_To': int(w_to_label),
                'Status': 'comeback',
                'From_Occupation': from_info['Occupation'] if from_info else None,
                'From_Industry': from_info['Industry'] if from_info else None,
                'From_State': from_info['State'] if from_info else None,
                'To_Occupation': to_info['Occupation'] if to_info else None,
                'To_Industry': to_info['Industry'] if to_info else None,
                'To_State': to_info['State'] if to_info else None,
                'Occupation_Changed': (from_info['Occupation'] != to_info['Occupation']) if (from_info and to_info) else None,
                'Industry_Changed': (from_info['Industry'] != to_info['Industry']) if (from_info and to_info) else None,
                'State_Changed': (from_info['State'] != to_info['State']) if (from_info and to_info) else None
            })
        
        # New entrants
        for user_id in new_entrants:
            to_info = user_info_cache.get((i + 1, user_id))
            user_records.append({
                'ID': user_id,
                'Year_From': int(w_from_label),
                'Year_To': int(w_to_label),
                'Status': 'new_entrant',
                'From_Occupation': None,
                'From_Industry': None,
                'From_State': None,
                'To_Occupation': to_info['Occupation'] if to_info else None,
                'To_Industry': to_info['Industry'] if to_info else None,
                'To_State': to_info['State'] if to_info else None,
                'Occupation_Changed': None,
                'Industry_Changed': None,
                'State_Changed': None
            })
        
        # Stayed
        for user_id in stayed_ids:
            from_info = user_info_cache.get((i, user_id))
            to_info = user_info_cache.get((i + 1, user_id))
            
            user_records.append({
                'ID': user_id,
                'Year_From': int(w_from_label),
                'Year_To': int(w_to_label),
                'Status': 'stayed',
                'From_Occupation': from_info['Occupation'] if from_info else None,
                'From_Industry': from_info['Industry'] if from_info else None,
                'From_State': from_info['State'] if from_info else None,
                'To_Occupation': to_info['Occupation'] if to_info else None,
                'To_Industry': to_info['Industry'] if to_info else None,
                'To_State': to_info['State'] if to_info else None,
                'Occupation_Changed': (from_info['Occupation'] != to_info['Occupation']) if (from_info and to_info) else None,
                'Industry_Changed': (from_info['Industry'] != to_info['Industry']) if (from_info and to_info) else None,
                'State_Changed': (from_info['State'] != to_info['State']) if (from_info and to_info) else None
            })
        
        user_data = pd.DataFrame(user_records)
        
        # VALIDATION: Check totals
        status_counts = user_data['Status'].value_counts()
        total_check = (
            status_counts.get('stayed', 0) +
            status_counts.get('dropout_career_break', 0) +
            status_counts.get('dropout_permanent_exit', 0) +
            status_counts.get('comeback', 0) +
            status_counts.get('new_entrant', 0)
        )
        
        if total_check != len(from_ids) + len(comeback_ids) + len(new_entrants):
            print(f"  ⚠️  Warning: User count mismatch!")
            print(f"     Total records: {total_check}")
            print(f"     Expected: {len(from_ids) + len(comeback_ids) + len(new_entrants)}")
        
        # Validate from/to balance
        from_check = len(stayed_ids) + len(dropout_ids)
        to_check = len(stayed_ids) + len(comeback_ids) + len(new_entrants)
        
        if from_check != len(from_ids):
            print(f"  ⚠️  Warning: From balance error! {from_check} != {len(from_ids)}")
        if to_check != len(to_ids):
            print(f"  ⚠️  Warning: To balance error! {to_check} != {len(to_ids)}")
        
        transition_data = {
            'Year_From': int(w_from_label),
            'Year_To': int(w_to_label),
            'Period': f'{w_from_label}→{w_to_label}',
            'Total_From': len(from_ids),
            'Total_To': len(to_ids),
            'Dropouts': len(dropout_ids),
            'Dropout_Rate': (len(dropout_ids) / len(from_ids)) * 100 if len(from_ids) > 0 else 0,
            'Career_Breaks': len(career_breaks),
            'Permanent_Exits': len(permanent_exits),
            'Permanent_Exit_Rate': (len(permanent_exits) / len(from_ids)) * 100 if len(from_ids) > 0 else 0,
            'Comebacks': len(comeback_ids),
            'Comeback_Rate': (len(comeback_ids) / len(to_ids)) * 100 if len(to_ids) > 0 else 0,
            'New_Entrants': len(new_entrants),
            'Entry_Rate': (len(new_entrants) / len(to_ids)) * 100 if len(to_ids) > 0 else 0,
        }
        
        _save_transition_cache(results_dir, w_from_label, w_to_label, transition_data, user_data)
        
        yearly_flow.append(transition_data)
        all_user_details.append(user_data)
        
        print(f"  Dropouts: {len(dropout_ids):,} ({len(dropout_ids)/len(from_ids)*100:.2f}%)")
        print(f"    └─ Career breaks: {len(career_breaks):,}")
        print(f"    └─ Permanent exits: {len(permanent_exits):,}")
        print(f"  Comebacks: {len(comeback_ids):,} ({len(comeback_ids)/len(to_ids)*100:.2f}%)")
        print(f"  New entrants: {len(new_entrants):,} ({len(new_entrants)/len(to_ids)*100:.2f}%)")
        print(f"  Stayed: {len(stayed_ids):,}")
        print(f"  ✓ Processed {len(user_records):,} users (optimized)")
    
    combined_user_details = pd.concat(all_user_details, ignore_index=True)
    
    return pd.DataFrame(yearly_flow), combined_user_details
"""
Unified mobility analyzer
Combines transitions, workforce flow, and occupation flow analysis
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict


class MobilityAnalyzer:
    """
    Labor market mobility analyzer
    Combines transitions, workforce flow, and occupation flow analysis into a single class
    """
    
    def __init__(self, results_dir='results', occupation_col='ONET_2019_NAME'):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory for caching and results
            occupation_col: Occupation column name to use
        """
        self.results_dir = results_dir
        self.occupation_col = occupation_col
        
        # Optimization caches
        self.user_presence_map = None
        self.user_info_cache = None
    
    # ========================================
    # PUBLIC API - Main Analysis Method
    # ========================================
    
    def analyze_all(self, window_users, windows):
        """
        Run complete mobility analysis pipeline
        
        Args:
            window_users: Dictionary of window users
            windows: List of windows
        
        Returns:
            Dictionary with all analysis results:
                - transitions: Year-to-year transition dataframe
                - transition_rates: Transition rates by period
                - workforce_flow: Workforce flow summary
                - user_details: Detailed user paths
                - outflows: Occupation outflow analysis
                - inflows: Occupation inflow analysis
                - critical_transitions: Critical occupation transitions
        """
        print("\n" + "="*80)
        print("RUNNING COMPLETE MOBILITY ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Build transitions
        print("\n[1/5] Building transitions...")
        results['transitions'] = self.build_transitions(window_users, windows)
        results['transition_rates'] = self.calculate_transition_rates(results['transitions'])
        
        # 2. Workforce flow
        print("\n[2/5] Analyzing workforce flow...")
        flow_summary, user_details = self.analyze_workforce_flow(window_users, windows)
        results['workforce_flow'] = flow_summary
        results['user_details'] = user_details
        
        # 3. Occupation outflows
        print("\n[3/5] Analyzing occupation outflows...")
        results['outflows'] = self.analyze_occupation_outflows(user_details, window_users, windows)
        
        # 4. Occupation inflows
        print("\n[4/5] Analyzing occupation inflows...")
        results['inflows'] = self.analyze_occupation_inflows(user_details, window_users, windows)
        
        # 5. Critical transitions
        print("\n[5/5] Finding critical transitions...")
        results['critical_transitions'] = self.find_critical_transitions(results['outflows'])
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return results
    
    # ========================================
    # TRANSITIONS ANALYSIS
    # ========================================
    
    def build_transitions(self, window_users, windows):
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
                    'From_Occupation': merged[f'{self.occupation_col}_from'],
                    'To_Occupation': merged[f'{self.occupation_col}_to'],
                    'From_Industry': merged['NAICS6_NAME_from'],
                    'To_Industry': merged['NAICS6_NAME_to'],
                    'From_State': merged['STATE_RAW_from'],
                    'To_State': merged['STATE_RAW_to'],
                    'Occupation_Changed': merged[f'{self.occupation_col}_from'] != merged[f'{self.occupation_col}_to'],
                    'Industry_Changed': merged['NAICS6_NAME_from'] != merged['NAICS6_NAME_to'],
                    'State_Changed': merged['STATE_RAW_from'] != merged['STATE_RAW_to']
                }).reset_index(drop=True)
                
                transitions.append(trans_df)
        
        return pd.concat(transitions, ignore_index=True)
    
    def calculate_transition_rates(self, all_transitions_df):
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
    
    # ========================================
    # WORKFORCE FLOW ANALYSIS
    # ========================================
    
    def analyze_workforce_flow(self, window_users, windows):
        """
        Comprehensive workforce flow analysis with optimizations
        
        Args:
            window_users: Dictionary of window users
            windows: List of windows
        
        Returns:
            Tuple of (summary dataframe, detailed user dataframe)
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE WORKFORCE FLOW ANALYSIS (OPTIMIZED)")
        print("="*80)
        
        # Build optimization caches
        if self.user_presence_map is None:
            self.user_presence_map = self._build_user_presence_map(window_users, windows)
        if self.user_info_cache is None:
            self.user_info_cache = self._build_user_info_cache(window_users, windows)
        
        yearly_flow = []
        all_user_details = []
        
        for i in range(len(windows) - 1):
            w_from_name, _, _, w_from_label = windows[i]
            w_to_name, _, _, w_to_label = windows[i + 1]
            
            # Check cache first
            cached, cached_users = self._load_cached_transition(w_from_label, w_to_label)
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
            
            # 2. Classify dropouts using pre-computed presence map
            career_breaks = set()
            permanent_exits = set()
            
            for dropout_id in dropout_ids:
                if dropout_id in self.user_presence_map:
                    future_windows = self.user_presence_map[dropout_id]
                    if any(window_idx > i + 1 for window_idx in future_windows):
                        career_breaks.add(dropout_id)
                    else:
                        permanent_exits.add(dropout_id)
                else:
                    permanent_exits.add(dropout_id)
            
            # 3. Comebacks using pre-computed presence map
            comeback_ids = set()
            comeback_last_seen = {}
            
            for user_id in to_ids:
                if user_id not in from_ids and user_id in self.user_presence_map:
                    past_windows = [w for w in self.user_presence_map[user_id] if w < i]
                    if past_windows:
                        comeback_ids.add(user_id)
                        comeback_last_seen[user_id] = max(past_windows)
            
            # 4. New entrants
            new_entrants = set()
            for user_id in to_ids:
                if user_id not in from_ids:
                    if user_id not in self.user_presence_map or all(w > i for w in self.user_presence_map[user_id]):
                        new_entrants.add(user_id)
            
            # 5. Stayed
            stayed_ids = from_ids & to_ids
            
            # Build detailed user-level data
            user_records = []
            
            # Career breaks
            for user_id in career_breaks:
                from_info = self.user_info_cache.get((i, user_id))
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
                from_info = self.user_info_cache.get((i, user_id))
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
                to_info = self.user_info_cache.get((i + 1, user_id))
                last_seen_idx = comeback_last_seen[user_id]
                from_info = self.user_info_cache.get((last_seen_idx, user_id))
                
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
                to_info = self.user_info_cache.get((i + 1, user_id))
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
                from_info = self.user_info_cache.get((i, user_id))
                to_info = self.user_info_cache.get((i + 1, user_id))
                
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
            
            self._save_transition_cache(w_from_label, w_to_label, transition_data, user_data)
            
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
    
    # ========================================
    # OCCUPATION FLOW ANALYSIS
    # ========================================
    
    def analyze_occupation_outflows(self, user_details_df, window_users, windows):
        """
        Analyze where people go when they leave each occupation
        
        Args:
            user_details_df: Detailed user paths dataframe
            window_users: Dictionary of window users (for API consistency)
            windows: List of windows
        
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
            
            if len(period_data) == 0:
                print(f"  Warning: No data for {year_from}→{year_to}")
                continue
            
            # Group by occupation once
            grouped = period_data.groupby('From_Occupation', dropna=True)
            
            for occ, occ_data in grouped:
                total_from_occ = len(occ_data)
                
                # Vectorized counting
                stayed_same = ((occ_data['Status'] == 'stayed') & 
                              (occ_data['Occupation_Changed'] == False)).sum()
                
                changed_mask = ((occ_data['Status'] == 'stayed') & 
                               (occ_data['Occupation_Changed'] == True))
                changed_occ_count = changed_mask.sum()
                
                career_break = (occ_data['Status'] == 'dropout_career_break').sum()
                perm_exit = (occ_data['Status'] == 'dropout_permanent_exit').sum()
                
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
    
    def analyze_occupation_inflows(self, user_details_df, window_users, windows):
        """
        Analyze where people come from when entering each occupation
        
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
            
            # Group by target occupation once
            grouped = period_data.groupby('To_Occupation', dropna=True)
            
            for occ, occ_data in grouped:
                total_to_occ = len(occ_data)
                
                # Vectorized counting
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
    
    def find_critical_transitions(self, outflow_df, min_count=100, top_n=10):
        """
        Find the most significant occupation-to-occupation transitions
        
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
    
    # ========================================
    # PRIVATE - Optimization Helpers
    # ========================================
    
    def _build_user_presence_map(self, window_users, windows):
        """
        Pre-compute which windows each user appears in
        KEY optimization - O(1) lookups instead of O(n) searches
        
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
    
    def _build_user_info_cache(self, window_users, windows):
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
                    'Occupation': user_row[self.occupation_col],
                    'Industry': user_row['NAICS6_NAME'],
                    'State': user_row['STATE_RAW'],
                    'Job_Start_Date': user_row['JOB_START_DATE']
                }
        
        print(f"  ✓ Cached info for {len(user_info_cache):,} user-window pairs")
        return user_info_cache
    
    # ========================================
    # CACHING
    # ========================================
    
    def _get_cache_filepath(self, year_from, year_to):
        """Get cache file path for a transition"""
        from .utils import ensure_dir
        cache_dir = os.path.join(self.results_dir, 'workforce_flow_cache')
        ensure_dir(cache_dir)
        return os.path.join(cache_dir, f'transition_{year_from}_to_{year_to}.csv')
    
    def _get_users_cache_filepath(self, year_from, year_to):
        """Get user-level cache file path"""
        from .utils import ensure_dir
        cache_dir = os.path.join(self.results_dir, 'workforce_flow_cache')
        ensure_dir(cache_dir)
        return os.path.join(cache_dir, f'users_{year_from}_to_{year_to}.csv')
    
    def _load_cached_transition(self, year_from, year_to):
        """Load cached transition if exists"""
        filepath = self._get_cache_filepath(year_from, year_to)
        users_filepath = self._get_users_cache_filepath(year_from, year_to)
        
        if os.path.exists(filepath) and os.path.exists(users_filepath):
            try:
                summary = pd.read_csv(filepath).iloc[0].to_dict()
                users = pd.read_csv(
                    users_filepath,
                    dtype={
                        'Occupation_Changed': 'object',
                        'Industry_Changed': 'object',
                        'State_Changed': 'object'
                    }
                )
                return summary, users
            except Exception as e:
                print(f"  Warning: Cache corrupted, recomputing...")
                return None, None
        return None, None
    
    def _save_transition_cache(self, year_from, year_to, transition_data, user_data):
        """Save transition and user-level data to cache"""
        filepath = self._get_cache_filepath(year_from, year_to)
        pd.DataFrame([transition_data]).to_csv(filepath, index=False)
        
        users_filepath = self._get_users_cache_filepath(year_from, year_to)
        user_data.to_csv(users_filepath, index=False)


# ========================================
# LEGACY FUNCTIONS (for backward compatibility)
# ========================================

def build_transitions(window_users, windows):
    """Legacy function - use MobilityAnalyzer class instead"""
    analyzer = MobilityAnalyzer()
    return analyzer.build_transitions(window_users, windows)


def calculate_transition_rates(all_transitions_df):
    """Legacy function - use MobilityAnalyzer class instead"""
    analyzer = MobilityAnalyzer()
    return analyzer.calculate_transition_rates(all_transitions_df)


def analyze_workforce_flow(window_users, windows, results_dir='results', 
                          occupation_col='SOC_EMSI_2019_3_NAME'):
    """Legacy function - use MobilityAnalyzer class instead"""
    analyzer = MobilityAnalyzer(results_dir, occupation_col)
    return analyzer.analyze_workforce_flow(window_users, windows)


def analyze_occupation_outflows(user_details_df, window_users, windows, 
                                occupation_col='SOC_EMSI_2019_3_NAME'):
    """Legacy function - use MobilityAnalyzer class instead"""
    analyzer = MobilityAnalyzer(occupation_col=occupation_col)
    return analyzer.analyze_occupation_outflows(user_details_df, window_users, windows)


def analyze_occupation_inflows(user_details_df, window_users, windows):
    """Legacy function - use MobilityAnalyzer class instead"""
    analyzer = MobilityAnalyzer()
    return analyzer.analyze_occupation_inflows(user_details_df, window_users, windows)


def find_critical_transitions(outflow_df, min_count=100, top_n=10):
    """Legacy function - use MobilityAnalyzer class instead"""
    analyzer = MobilityAnalyzer()
    return analyzer.find_critical_transitions(outflow_df, min_count, top_n)
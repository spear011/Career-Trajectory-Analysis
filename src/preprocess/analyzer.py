"""
Windowed Workforce Analyzer 3-Cohort Design

Three-cohort structure:

Any Employment: Individuals who worked at any point within the window (overall labor force size)

Attached (≥27 weeks/year): Individuals employed at least 27 weeks in each year (main analytical sample)

Full-time Year-round: Individuals working 50+ weeks every year (stable core workforce)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class WindowedAnalyzer:
    """
    Windowed analysis of career trajectories with 3-cohort framework
    """
    
    def __init__(self,
                 window_size: int = 1,
                 hop_size: int = 1,
                 occupation_col: str = 'onet_major'):
        """
        Initialize windowed analyzer
        
        Args:
            window_size: Window size in years (default: 1)
            hop_size: Hop size in years for sliding window (default: 1)
            occupation_col: Column name for occupation classification
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.occupation_col = occupation_col
    
    def generate_windows(self, start_year: int, end_year: int) -> List[Tuple[int, int]]:
        """Generate sliding windows"""
        windows = []
        current_start = start_year
        
        while current_start + self.window_size - 1 <= end_year:
            window_end = current_start + self.window_size - 1
            windows.append((current_start, window_end))
            current_start += self.hop_size
        
        return windows
    
    def get_active_users_in_window(self,
                                    trajectory_df: pd.DataFrame,
                                   window_start: int,
                                   window_end: int) -> pd.DataFrame:
        """Get all users with active jobs within the window"""
        window_start_date = pd.Timestamp(f'{window_start}-01-01')
        window_end_date = pd.Timestamp(f'{window_end}-12-31')
        
        df = trajectory_df.copy()
        
        if 'job_start_date' in df.columns:
            if df['job_start_date'].dtype == 'object':
                df['job_start_date'] = pd.to_datetime(df['job_start_date'], errors='coerce')
        else:
            df['job_start_date'] = pd.to_datetime(df['job_start_year'].astype(str) + '-01-01', errors='coerce')
        
        if 'job_end_date' in df.columns:
            if df['job_end_date'].dtype == 'object':
                df['job_end_date'] = pd.to_datetime(df['job_end_date'], errors='coerce')
        else:
            df['job_end_date'] = pd.to_datetime(df['job_end_year'].astype(str) + '-12-31', errors='coerce')
        
        active = df[
            (df['job_start_date'] <= window_end_date) &
            ((df['job_end_date'] >= window_start_date) | (df['job_end_date'].isna()))
        ].copy()
        
        return active
    
    def build_user_lifecycle(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """Build complete lifecycle tracking for all users"""
        user_agg = trajectory_df.groupby('ID').agg({
            'job_start_year': 'min',
            'job_end_year': ['max', lambda x: x.isna().any()]
        }).reset_index()
        
        user_agg.columns = ['ID', 'first_job_year', 'last_job_year', 'is_currently_employed']
        user_agg.loc[user_agg['is_currently_employed'], 'last_job_year'] = np.nan
        
        return user_agg
    
    # ========== 3 COHORT DEFINITIONS ==========
    
    def get_any_employment_cohort(self,
                                   trajectory_df: pd.DataFrame,
                                   window_start: int,
                                   window_end: int) -> set:
        """
        Any Employment Cohort: Worked at Least Once Within the Window

        Definition: Included if the individual worked even a single day during the window (BLS “worked at some point”).
        Purpose: Used to estimate overall labor market size.
        """
        active_df = self.get_active_users_in_window(trajectory_df, window_start, window_end)
        return set(active_df['ID'].unique())
    
    def get_attached_cohort(self,
                           trajectory_df: pd.DataFrame,
                           window_start: int,
                           window_end: int) -> set:
        """
        Attached Cohort (≥27 Weeks/Year): Individuals Continuously Attached to the Labor Market

        Definition: Employed at least 27 weeks in every year (aligned with the BLS "working poor" threshold).
        Purpose: Main analytical group for transition rates and industry mobility.
        Optimized: Uses vectorized operations for speed.
        """
        window_start_date = pd.Timestamp(f'{window_start}-01-01')
        window_end_date = pd.Timestamp(f'{window_end}-12-31')
        
        df = trajectory_df.copy()
        
        # Ensure datetime
        if 'job_start_date' not in df.columns or df['job_start_date'].dtype == 'object':
            df['job_start_date'] = pd.to_datetime(df.get('job_start_date', df['job_start_year'].astype(str) + '-01-01'), errors='coerce')
        
        if 'job_end_date' not in df.columns or df['job_end_date'].dtype == 'object':
            df['job_end_date'] = pd.to_datetime(df.get('job_end_date', df['job_end_year'].astype(str) + '-12-31'), errors='coerce')
        
        # Filter to active jobs
        active = df[
            (df['job_start_date'] <= window_end_date) &
            ((df['job_end_date'] >= window_start_date) | (df['job_end_date'].isna()))
        ].copy()
        
        # For each year, calculate weeks worked per user (vectorized)
        attached_users_per_year = []
        
        for year in range(window_start, window_end + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31')
            
            # Clip job dates to year boundaries
            year_active = active.copy()
            year_active['effective_start'] = year_active['job_start_date'].clip(lower=year_start)
            year_active['effective_end'] = year_active['job_end_date'].fillna(year_end).clip(upper=year_end)
            
            # Calculate weeks for each job
            year_active['weeks'] = (year_active['effective_end'] - year_active['effective_start']).dt.days / 7
            
            # Sum weeks per user
            user_weeks = year_active.groupby('ID')['weeks'].sum()
            
            # Users with ≥27 weeks this year
            attached_this_year = set(user_weeks[user_weeks >= 27].index)
            attached_users_per_year.append(attached_this_year)
        
        # Users must meet criteria in ALL years
        if len(attached_users_per_year) == 0:
            return set()
        
        attached_users = attached_users_per_year[0]
        for year_set in attached_users_per_year[1:]:
            attached_users = attached_users & year_set
        
        return attached_users
    
    def get_fulltime_yearround_cohort(self,
                                      trajectory_df: pd.DataFrame,
                                      window_start: int,
                                      window_end: int) -> set:
        """
        Full-time Year-round Cohort: Stable Core Workforce

        Definition: Worked 50-52 weeks in every year (Census/BLS definition of full-time year-round).
        Purpose: Used to analyze structural mobility among stable, long-term employees.
        Optimized: Uses vectorized operations for speed.
        
        """
        window_start_date = pd.Timestamp(f'{window_start}-01-01')
        window_end_date = pd.Timestamp(f'{window_end}-12-31')
        
        df = trajectory_df.copy()
        
        # Ensure datetime
        if 'job_start_date' not in df.columns or df['job_start_date'].dtype == 'object':
            df['job_start_date'] = pd.to_datetime(df.get('job_start_date', df['job_start_year'].astype(str) + '-01-01'), errors='coerce')
        
        if 'job_end_date' not in df.columns or df['job_end_date'].dtype == 'object':
            df['job_end_date'] = pd.to_datetime(df.get('job_end_date', df['job_end_year'].astype(str) + '-12-31'), errors='coerce')
        
        # Filter to active jobs
        active = df[
            (df['job_start_date'] <= window_end_date) &
            ((df['job_end_date'] >= window_start_date) | (df['job_end_date'].isna()))
        ].copy()
        
        # For each year, calculate weeks worked per user (vectorized)
        fulltime_users_per_year = []
        
        for year in range(window_start, window_end + 1):
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31')
            
            # Clip job dates to year boundaries
            year_active = active.copy()
            year_active['effective_start'] = year_active['job_start_date'].clip(lower=year_start)
            year_active['effective_end'] = year_active['job_end_date'].fillna(year_end).clip(upper=year_end)
            
            # Calculate weeks for each job
            year_active['weeks'] = (year_active['effective_end'] - year_active['effective_start']).dt.days / 7
            
            # Sum weeks per user
            user_weeks = year_active.groupby('ID')['weeks'].sum()
            
            # Users with ≥50 weeks this year
            fulltime_this_year = set(user_weeks[user_weeks >= 50].index)
            fulltime_users_per_year.append(fulltime_this_year)
        
        # Users must meet criteria in ALL years
        if len(fulltime_users_per_year) == 0:
            return set()
        
        fulltime_users = fulltime_users_per_year[0]
        for year_set in fulltime_users_per_year[1:]:
            fulltime_users = fulltime_users & year_set
        
        return fulltime_users
    
    # ========== ENTRY/EXIT TRACKING ==========
    
    def get_new_entrants_in_window(self,
                                    user_lifecycle: pd.DataFrame,
                                    window_start: int,
                                    window_end: int) -> set:
        new_entrants_mask = (
            (user_lifecycle['first_job_year'] >= window_start) &
            (user_lifecycle['first_job_year'] <= window_end)
        )
        return set(user_lifecycle[new_entrants_mask]['ID'])
    
    def get_permanent_exits_in_window(self,
                                      trajectory_df: pd.DataFrame,
                                      user_lifecycle: pd.DataFrame,
                                      window_start: int,
                                      window_end: int,
                                      dataset_end_year: int) -> set:
        last_job_in_window_mask = (
            (~user_lifecycle['is_currently_employed']) &
            (user_lifecycle['last_job_year'] >= window_start) &
            (user_lifecycle['last_job_year'] <= window_end)
        )
        
        potential_exits = set(user_lifecycle[last_job_in_window_mask]['ID'])
        
        future_jobs = trajectory_df[
            (trajectory_df['ID'].isin(potential_exits)) &
            (trajectory_df['job_start_year'] > window_end)
        ]
        
        users_with_future_jobs = set(future_jobs['ID'].unique())
        permanent_exits = potential_exits - users_with_future_jobs
        
        return permanent_exits

    def analyze_window(self,
                      trajectory_df: pd.DataFrame,
                      window_start: int,
                      window_end: int,
                      user_lifecycle: pd.DataFrame,
                      dataset_end_year: int,
                      previous_attached: Optional[set] = None,
                      previous_fulltime: Optional[set] = None) -> Tuple[Dict, set, set]:
        """
        Analyze workforce composition for a single window using 3-cohort framework
        
        Returns:
            Tuple of (statistics, current_attached, current_fulltime)
        """
        # ===== COHORT CALCULATIONS =====
        
        any_employment = self.get_any_employment_cohort(trajectory_df, window_start, window_end)
        attached = self.get_attached_cohort(trajectory_df, window_start, window_end)
        fulltime = self.get_fulltime_yearround_cohort(trajectory_df, window_start, window_end)
        
        # ===== ENTRY/EXIT CALCULATIONS =====
        
        new_labor_entrants = self.get_new_entrants_in_window(user_lifecycle, window_start, window_end)
        permanent_exits = self.get_permanent_exits_in_window(
            trajectory_df, user_lifecycle, window_start, window_end, dataset_end_year
        )
        
        # ===== TRANSITION CALCULATIONS (Attached cohort) =====
        
        new_attached = set()
        exits_from_attached = set()
        attached_retention_rate = None
        attached_exit_rate = None
        attached_entry_rate = None
        
        if previous_attached is not None:
            new_attached = attached - previous_attached
            exits_from_attached = previous_attached - attached
            
            if len(previous_attached) > 0:
                retained = attached & previous_attached
                attached_retention_rate = len(retained) / len(previous_attached) * 100
                attached_exit_rate = len(exits_from_attached) / len(previous_attached) * 100
            
            if len(attached) > 0:
                attached_entry_rate = len(new_attached) / len(attached) * 100
        
        # ===== TRANSITION CALCULATIONS (Full-time cohort) =====
        
        new_fulltime = set()
        exits_from_fulltime = set()
        fulltime_retention_rate = None
        fulltime_exit_rate = None
        fulltime_entry_rate = None
        
        if previous_fulltime is not None:
            new_fulltime = fulltime - previous_fulltime
            exits_from_fulltime = previous_fulltime - fulltime
            
            if len(previous_fulltime) > 0:
                retained_ft = fulltime & previous_fulltime
                fulltime_retention_rate = len(retained_ft) / len(previous_fulltime) * 100
                fulltime_exit_rate = len(exits_from_fulltime) / len(previous_fulltime) * 100
            
            if len(fulltime) > 0:
                fulltime_entry_rate = len(new_fulltime) / len(fulltime) * 100
        
        # ===== OCCUPATION & DEMOGRAPHICS =====
        
        active_df = self.get_active_users_in_window(trajectory_df, window_start, window_end)
        occupations = active_df[self.occupation_col].dropna().unique()
        
        occ_counts = active_df[self.occupation_col].value_counts()
        top_occ = occ_counts.index[0] if len(occ_counts) > 0 else None
        top_occ_count = occ_counts.iloc[0] if len(occ_counts) > 0 else 0
        
        # ===== BUILD STATISTICS =====
        
        stats = {
            'window_start': window_start,
            'window_end': window_end,
            'window_label': f"{window_start}" if window_start == window_end else f"{window_start}-{window_end}",
            
            # COHORT SIZES
            'any_employment': len(any_employment),
            'attached_cohort': len(attached),
            'fulltime_cohort': len(fulltime),
            
            # COHORT PERCENTAGES
            'attached_pct': (len(attached) / len(any_employment) * 100) if len(any_employment) > 0 else 0,
            'fulltime_pct': (len(fulltime) / len(any_employment) * 100) if len(any_employment) > 0 else 0,
            
            # ENTRY/EXIT
            'new_labor_entrants': len(new_labor_entrants),
            'permanent_exits': len(permanent_exits),
            
            # ATTACHED TRANSITIONS
            'new_attached': len(new_attached),
            'exits_from_attached': len(exits_from_attached),
            'attached_retention_rate': attached_retention_rate,
            'attached_exit_rate': attached_exit_rate,
            'attached_entry_rate': attached_entry_rate,
            
            # FULLTIME TRANSITIONS
            'new_fulltime': len(new_fulltime),
            'exits_from_fulltime': len(exits_from_fulltime),
            'fulltime_retention_rate': fulltime_retention_rate,
            'fulltime_exit_rate': fulltime_exit_rate,
            'fulltime_entry_rate': fulltime_entry_rate,
            
            # OCCUPATION & JOBS
            'total_active_jobs': len(active_df),
            'num_occupations': len(occupations),
            'top_occupation': top_occ,
            'top_occupation_count': top_occ_count,
        }
        
        # Demographics
        if 'gender' in active_df.columns:
            stats['pct_female'] = (active_df['gender'] == 2).mean() * 100
        
        # Wages
        if 'annual_state_wage' in active_df.columns:
            stats['avg_wage'] = active_df['annual_state_wage'].mean()
            stats['median_wage'] = active_df['annual_state_wage'].median()
        
        return stats, attached, fulltime
    
    def analyze_all_windows(self,
                           trajectory_df: pd.DataFrame,
                           start_year: int,
                           end_year: int) -> pd.DataFrame:
        """Run windowed analysis across entire time range"""
        print(f"\nRunning 3-Cohort Windowed Analysis:")
        print(f"  Window size: {self.window_size} year(s)")
        print(f"  Hop size: {self.hop_size} year(s)")
        print(f"  Period: {start_year}-{end_year}")
        
        print(f"\n  Data diagnostics:")
        print(f"    Total records: {len(trajectory_df):,}")
        print(f"    Unique users: {trajectory_df['ID'].nunique():,}")
        print(f"    Year range (start): {trajectory_df['job_start_year'].min():.0f}-{trajectory_df['job_start_year'].max():.0f}")
        
        if 'job_end_year' in trajectory_df.columns:
            print(f"    Year range (end): {trajectory_df['job_end_year'].min():.0f}-{trajectory_df['job_end_year'].max():.0f}")
            null_end_pct = trajectory_df['job_end_year'].isna().sum() / len(trajectory_df) * 100
            print(f"    Jobs with no end date: {trajectory_df['job_end_year'].isna().sum():,} ({null_end_pct:.1f}%)")
        
        print(f"\n  Building user lifecycle tracking...")
        user_lifecycle = self.build_user_lifecycle(trajectory_df)
        
        dataset_end_year = int(trajectory_df['job_start_year'].max())
        if 'job_end_year' in trajectory_df.columns:
            dataset_end_year = max(dataset_end_year, int(trajectory_df['job_end_year'].max()))
        
        windows = self.generate_windows(start_year, end_year)
        print(f"\n  Total windows: {len(windows)}")
        print(f"\n  Cohort definitions:")
        print(f"    Any Employment: worked at any point in window")
        print(f"    Attached (≥27wk/yr): ≥27 weeks per year")
        print(f"    Full-time (≥50wk/yr): ≥50 weeks per year\n")
        
        results = []
        previous_attached = None
        previous_fulltime = None
        
        for window_start, window_end in windows:
            stats, current_attached, current_fulltime = self.analyze_window(
                trajectory_df,
                window_start,
                window_end,
                user_lifecycle,
                dataset_end_year,
                previous_attached,
                previous_fulltime
            )
            results.append(stats)
            previous_attached = current_attached
            previous_fulltime = current_fulltime
            
            print(f"  ✓ {stats['window_label']}: "
                  f"Any={stats['any_employment']:,} | "
                  f"Attached={stats['attached_cohort']:,} ({stats['attached_pct']:.1f}%) | "
                  f"FT={stats['fulltime_cohort']:,} ({stats['fulltime_pct']:.1f}%)")
        
        results_df = pd.DataFrame(results)
        print(f"\n✓ Analysis complete: {len(results_df)} windows")
        
        return results_df
    
    # ========== ADDITIONAL METHODS ==========
    
    def analyze_occupation_dynamics(self,
                                   trajectory_df: pd.DataFrame,
                                   window_start: int,
                                   window_end: int) -> pd.DataFrame:
        """Detailed occupation-level analysis"""
        active_df = self.get_active_users_in_window(trajectory_df, window_start, window_end)
        occupation_stats = []
        
        for occ in active_df[self.occupation_col].dropna().unique():
            occ_df = active_df[active_df[self.occupation_col] == occ]
            
            stats = {
                'window_start': window_start,
                'window_end': window_end,
                'occupation': occ,
                'total_workers': len(occ_df),
                'unique_users': occ_df['ID'].nunique(),
            }
            
            if 'annual_state_wage' in occ_df.columns:
                stats['avg_wage'] = occ_df['annual_state_wage'].mean()
                stats['median_wage'] = occ_df['annual_state_wage'].median()
            
            if 'gender' in occ_df.columns:
                stats['pct_female'] = (occ_df['gender'] == 2).mean() * 100
            
            occupation_stats.append(stats)
        
        return pd.DataFrame(occupation_stats)
    
    def compare_periods(self,
                       trajectory_df: pd.DataFrame,
                       period1_start: int,
                       period1_end: int,
                       period2_start: int,
                       period2_end: int,
                       period1_name: str = "Period 1",
                       period2_name: str = "Period 2") -> Dict:
        """
        Compare workforce composition between two periods
        
        Args:
            trajectory_df: Career trajectory dataframe
            period1_start: Period 1 start year
            period1_end: Period 1 end year
            period2_start: Period 2 start year
            period2_end: Period 2 end year
            period1_name: Name for period 1
            period2_name: Name for period 2
        
        Returns:
            Dictionary with comparison statistics
        """
        # Get cohorts for each period
        p1_any = self.get_any_employment_cohort(trajectory_df, period1_start, period1_end)
        p1_attached = self.get_attached_cohort(trajectory_df, period1_start, period1_end)
        p1_fulltime = self.get_fulltime_yearround_cohort(trajectory_df, period1_start, period1_end)
        
        p2_any = self.get_any_employment_cohort(trajectory_df, period2_start, period2_end)
        p2_attached = self.get_attached_cohort(trajectory_df, period2_start, period2_end)
        p2_fulltime = self.get_fulltime_yearround_cohort(trajectory_df, period2_start, period2_end)
        
        # Get occupations
        p1_df = self.get_active_users_in_window(trajectory_df, period1_start, period1_end)
        p2_df = self.get_active_users_in_window(trajectory_df, period2_start, period2_end)
        
        p1_occs = set(p1_df[self.occupation_col].dropna().unique())
        p2_occs = set(p2_df[self.occupation_col].dropna().unique())
        
        comparison = {
            'period1_name': period1_name,
            'period1_years': f"{period1_start}-{period1_end}",
            'period2_name': period2_name,
            'period2_years': f"{period2_start}-{period2_end}",
            
            # Any employment comparison
            'period1_any_employment': len(p1_any),
            'period2_any_employment': len(p2_any),
            'any_in_both': len(p1_any & p2_any),
            'any_only_period1': len(p1_any - p2_any),
            'any_only_period2': len(p2_any - p1_any),
            
            # Attached cohort comparison
            'period1_attached': len(p1_attached),
            'period2_attached': len(p2_attached),
            'attached_in_both': len(p1_attached & p2_attached),
            'attached_only_period1': len(p1_attached - p2_attached),
            'attached_only_period2': len(p2_attached - p1_attached),
            'attached_retention_rate': len(p1_attached & p2_attached) / len(p1_attached) * 100 if len(p1_attached) > 0 else 0,
            
            # Full-time cohort comparison
            'period1_fulltime': len(p1_fulltime),
            'period2_fulltime': len(p2_fulltime),
            'fulltime_in_both': len(p1_fulltime & p2_fulltime),
            'fulltime_only_period1': len(p1_fulltime - p2_fulltime),
            'fulltime_only_period2': len(p2_fulltime - p1_fulltime),
            'fulltime_retention_rate': len(p1_fulltime & p2_fulltime) / len(p1_fulltime) * 100 if len(p1_fulltime) > 0 else 0,
            
            # Occupation comparison
            'period1_occupations': len(p1_occs),
            'period2_occupations': len(p2_occs),
            'new_occupations': len(p2_occs - p1_occs),
            'disappeared_occupations': len(p1_occs - p2_occs),
        }
        
        return comparison
    
    def track_cohort_flow(self,
                         trajectory_df: pd.DataFrame,
                         cohort_users: set,
                         start_year: int,
                         end_year: int,
                         cohort_type: str = 'attached') -> pd.DataFrame:
        """
        Track a specific cohort of users through time windows
        
        Args:
            trajectory_df: Career trajectory dataframe
            cohort_users: Set of user IDs to track
            start_year: First year to track
            end_year: Last year to track
            cohort_type: 'any', 'attached', or 'fulltime'
        
        Returns:
            Dataframe tracking cohort statistics over windows
        """
        windows = self.generate_windows(start_year, end_year)
        cohort_stats = []
        
        for window_start, window_end in windows:
            # Get appropriate cohort for this window
            if cohort_type == 'any':
                window_cohort = self.get_any_employment_cohort(trajectory_df, window_start, window_end)
            elif cohort_type == 'attached':
                window_cohort = self.get_attached_cohort(trajectory_df, window_start, window_end)
            elif cohort_type == 'fulltime':
                window_cohort = self.get_fulltime_yearround_cohort(trajectory_df, window_start, window_end)
            else:
                raise ValueError(f"Unknown cohort_type: {cohort_type}")
            
            # Track original cohort members
            cohort_active = cohort_users & window_cohort
            
            # Get job details for active cohort members
            active_df = self.get_active_users_in_window(trajectory_df, window_start, window_end)
            cohort_jobs = active_df[active_df['ID'].isin(cohort_active)]
            
            stats = {
                'window_start': window_start,
                'window_end': window_end,
                'window_label': f"{window_start}" if window_start == window_end else f"{window_start}-{window_end}",
                'cohort_size': len(cohort_users),
                'active_in_window': len(cohort_active),
                'retention_rate': len(cohort_active) / len(cohort_users) * 100,
                'total_jobs': len(cohort_jobs),
            }
            
            if len(cohort_active) > 0:
                stats['avg_jobs_per_person'] = len(cohort_jobs) / len(cohort_active)
                
                if 'annual_state_wage' in cohort_jobs.columns:
                    stats['avg_wage'] = cohort_jobs['annual_state_wage'].mean()
            
            cohort_stats.append(stats)
        
        return pd.DataFrame(cohort_stats)
    
    def run_temporal_diagnostics(self,
                                 trajectory_df: pd.DataFrame,
                                start_year: int,
                                end_year: int) -> Dict:
        """Run diagnostic checks"""
        diagnostics = {}
        
        print(f"\n{'='*80}")
        print("TEMPORAL DATA QUALITY DIAGNOSTICS")
        print(f"{'='*80}")
        
        print(f"\nYear-by-Year Job Starts ({start_year}-{end_year}):")
        year_starts = {}
        for year in range(start_year, end_year + 1):
            count = (trajectory_df['job_start_year'] == year).sum()
            year_starts[year] = count
            print(f"  {year}: {count:,} jobs")
        diagnostics['year_starts'] = year_starts
        
        return diagnostics
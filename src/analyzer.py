"""
Windowed Workforce Analyzer
Analyzes workforce dynamics using sliding windows to track:
- Active workers per window
- New entrants
- Exits
- Occupation counts
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class WindowedAnalyzer:
    """
    Windowed analysis of career trajectories
    Tracks workforce composition changes over time using sliding windows
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
        """
        Generate sliding windows
        
        Args:
            start_year: First year to analyze
            end_year: Last year to analyze
        
        Returns:
            List of (window_start_year, window_end_year) tuples
        """
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
        """
        Get all users with active jobs within the window
        Uses date-based filtering for accurate overlap detection
        
        A job is considered active in a window if:
        - Job start date <= Window end date (Dec 31), AND
        - Job end date >= Window start date (Jan 1) OR job has no end date
        
        Args:
            trajectory_df: Career trajectory dataframe
            window_start: Window start year
            window_end: Window end year
        
        Returns:
            Dataframe of active users/jobs in window
        """
        # Convert window years to datetime boundaries
        window_start_date = pd.Timestamp(f'{window_start}-01-01')
        window_end_date = pd.Timestamp(f'{window_end}-12-31')
        
        df = trajectory_df.copy()
        
        # Parse date columns if they exist and are strings
        if 'job_start_date' in df.columns:
            if df['job_start_date'].dtype == 'object':
                df['job_start_date'] = pd.to_datetime(df['job_start_date'], errors='coerce')
        else:
            # Fallback: create date from year (assuming January 1st)
            df['job_start_date'] = pd.to_datetime(df['job_start_year'].astype(str) + '-01-01', errors='coerce')
        
        if 'job_end_date' in df.columns:
            if df['job_end_date'].dtype == 'object':
                df['job_end_date'] = pd.to_datetime(df['job_end_date'], errors='coerce')
        else:
            # Fallback: create date from year (assuming December 31st)
            df['job_end_date'] = pd.to_datetime(df['job_end_year'].astype(str) + '-12-31', errors='coerce')
        
        # Filter for jobs that overlap with window period
        # Job is active if: start_date <= window_end_date AND (end_date >= window_start_date OR end_date is NaN)
        active = df[
            (df['job_start_date'] <= window_end_date) &
            ((df['job_end_date'] >= window_start_date) | (df['job_end_date'].isna()))
        ].copy()
        
        return active
    
    def analyze_window(self, 
                      trajectory_df: pd.DataFrame,
                      window_start: int,
                      window_end: int,
                      previous_active_users: Optional[set] = None) -> Dict:
        """
        Analyze workforce composition for a single window
        
        Args:
            trajectory_df: Career trajectory dataframe
            window_start: Window start year
            window_end: Window end year
            previous_active_users: Set of user IDs active in previous window
        
        Returns:
            Dictionary with window statistics
        """
        # Get active jobs in this window
        active_df = self.get_active_users_in_window(trajectory_df, window_start, window_end)
        
        # Get unique users and occupations
        current_users = set(active_df['ID'].unique())
        occupations = active_df[self.occupation_col].dropna().unique()
        
        # Calculate entry/exit if previous window exists
        new_entrants = set()
        exits = set()
        
        if previous_active_users is not None:
            new_entrants = current_users - previous_active_users
            exits = previous_active_users - current_users
        
        # Compile statistics
        stats = {
            'window_start': window_start,
            'window_end': window_end,
            'window_label': f"{window_start}" if window_start == window_end else f"{window_start}-{window_end}",
            'total_active_users': len(current_users),
            'total_active_jobs': len(active_df),
            'num_occupations': len(occupations),
            'new_entrants': len(new_entrants),
            'exits': len(exits),
            'retention_rate': None if previous_active_users is None else 
                            len(current_users & previous_active_users) / len(previous_active_users) * 100 
                            if len(previous_active_users) > 0 else 0.0
        }
        
        # Add occupation-level statistics
        occ_counts = active_df[self.occupation_col].value_counts()
        stats['top_occupation'] = occ_counts.index[0] if len(occ_counts) > 0 else None
        stats['top_occupation_count'] = occ_counts.iloc[0] if len(occ_counts) > 0 else 0
        
        # Add demographic breakdowns if available
        if 'gender' in active_df.columns:
            stats['pct_female'] = (active_df['gender'] == 2).mean() * 100
        
        if 'annual_state_wage' in active_df.columns:
            stats['avg_wage'] = active_df['annual_state_wage'].mean()
            stats['median_wage'] = active_df['annual_state_wage'].median()
        
        return stats, current_users
    
    def analyze_all_windows(self, 
                           trajectory_df: pd.DataFrame,
                           start_year: int,
                           end_year: int) -> pd.DataFrame:
        """
        Run windowed analysis across entire time range
        
        Args:
            trajectory_df: Career trajectory dataframe
            start_year: First year to analyze
            end_year: Last year to analyze
        
        Returns:
            Dataframe with statistics for each window
        """
        print(f"\nRunning windowed analysis:")
        print(f"  Window size: {self.window_size} year(s)")
        print(f"  Hop size: {self.hop_size} year(s)")
        print(f"  Period: {start_year}-{end_year}")
        
        # Data diagnostics
        print(f"\n  Data diagnostics:")
        print(f"    Total records: {len(trajectory_df):,}")
        print(f"    Unique users: {trajectory_df['ID'].nunique():,}")
        print(f"    Year range (start): {trajectory_df['job_start_year'].min():.0f}-{trajectory_df['job_start_year'].max():.0f}")
        if 'job_end_year' in trajectory_df.columns:
            print(f"    Year range (end): {trajectory_df['job_end_year'].min():.0f}-{trajectory_df['job_end_year'].max():.0f}")
            null_end_pct = trajectory_df['job_end_year'].isna().sum() / len(trajectory_df) * 100
            print(f"    Jobs with no end date: {trajectory_df['job_end_year'].isna().sum():,} ({null_end_pct:.1f}%)")
        
        # Check recent years distribution
        print(f"\n  Recent year distribution (job_start_year):")
        for year in range(max(start_year, 2018), end_year + 1):
            count = (trajectory_df['job_start_year'] == year).sum()
            print(f"    {year}: {count:,} jobs")
        
        # Generate windows
        windows = self.generate_windows(start_year, end_year)
        print(f"\n  Total windows: {len(windows)}")
        
        # Analyze each window
        results = []
        previous_users = None
        
        for window_start, window_end in windows:
            stats, current_users = self.analyze_window(
                trajectory_df, 
                window_start, 
                window_end,
                previous_users
            )
            results.append(stats)
            previous_users = current_users
            
            print(f"  ✓ Window {stats['window_label']}: "
                  f"{stats['total_active_users']:,} users, "
                  f"{stats['num_occupations']} occupations"
                  f"{' [LOW DATA WARNING]' if stats['total_active_users'] < 1000 else ''}")
        
        results_df = pd.DataFrame(results)
        print(f"\n✓ Windowed analysis complete: {len(results_df)} windows analyzed")
        
        # Identify potential data quality issues
        low_count_windows = results_df[results_df['total_active_users'] < 1000]
        if len(low_count_windows) > 0:
            print(f"\n⚠️  WARNING: {len(low_count_windows)} windows have <1000 active users")
            print(f"   This may indicate:")
            print(f"   1. Limited data collection in recent years")
            print(f"   2. Preprocessing filters excluding recent data")
            print(f"   3. Jobs without end dates not properly handled")
            print(f"\n   Low-count windows: {', '.join(low_count_windows['window_label'].tolist())}")
        
        return results_df
    
    def analyze_occupation_dynamics(self,
                                   trajectory_df: pd.DataFrame,
                                   window_start: int,
                                   window_end: int) -> pd.DataFrame:
        """
        Detailed occupation-level analysis for a window
        
        Args:
            trajectory_df: Career trajectory dataframe
            window_start: Window start year
            window_end: Window end year
        
        Returns:
            Dataframe with per-occupation statistics
        """
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
            
            # Add wage info if available
            if 'annual_state_wage' in occ_df.columns:
                stats['avg_wage'] = occ_df['annual_state_wage'].mean()
                stats['median_wage'] = occ_df['annual_state_wage'].median()
            
            # Add demographic info if available
            if 'gender' in occ_df.columns:
                stats['pct_female'] = (occ_df['gender'] == 2).mean() * 100
            
            if 'age_at_job_start' in occ_df.columns:
                stats['avg_age'] = occ_df['age_at_job_start'].mean()
            
            occupation_stats.append(stats)
        
        return pd.DataFrame(occupation_stats)
    
    def track_cohort_flow(self,
                         trajectory_df: pd.DataFrame,
                         cohort_users: set,
                         start_year: int,
                         end_year: int) -> pd.DataFrame:
        """
        Track a specific cohort of users through time windows
        
        Args:
            trajectory_df: Career trajectory dataframe
            cohort_users: Set of user IDs to track
            start_year: First year to track
            end_year: Last year to track
        
        Returns:
            Dataframe tracking cohort statistics over windows
        """
        windows = self.generate_windows(start_year, end_year)
        cohort_stats = []
        
        for window_start, window_end in windows:
            active_df = self.get_active_users_in_window(trajectory_df, window_start, window_end)
            cohort_active = active_df[active_df['ID'].isin(cohort_users)]
            
            stats = {
                'window_start': window_start,
                'window_end': window_end,
                'window_label': f"{window_start}" if window_start == window_end else f"{window_start}-{window_end}",
                'cohort_size': len(cohort_users),
                'active_in_window': cohort_active['ID'].nunique(),
                'retention_rate': cohort_active['ID'].nunique() / len(cohort_users) * 100,
                'total_jobs': len(cohort_active),
            }
            
            if len(cohort_active) > 0:
                stats['avg_jobs_per_person'] = len(cohort_active) / cohort_active['ID'].nunique()
                
                if 'annual_state_wage' in cohort_active.columns:
                    stats['avg_wage'] = cohort_active['annual_state_wage'].mean()
            
            cohort_stats.append(stats)
        
        return pd.DataFrame(cohort_stats)
    
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
            period1_name: Label for period 1
            period2_name: Label for period 2
        
        Returns:
            Dictionary with comparison statistics
        """
        # Get active users in each period
        p1_active = self.get_active_users_in_window(trajectory_df, period1_start, period1_end)
        p2_active = self.get_active_users_in_window(trajectory_df, period2_start, period2_end)
        
        p1_users = set(p1_active['ID'].unique())
        p2_users = set(p2_active['ID'].unique())
        
        # Calculate overlap and changes
        comparison = {
            'period1_name': period1_name,
            'period1_years': f"{period1_start}-{period1_end}",
            'period1_users': len(p1_users),
            'period2_name': period2_name,
            'period2_years': f"{period2_start}-{period2_end}",
            'period2_users': len(p2_users),
            'users_in_both': len(p1_users & p2_users),
            'users_only_period1': len(p1_users - p2_users),
            'users_only_period2': len(p2_users - p1_users),
            'retention_rate': len(p1_users & p2_users) / len(p1_users) * 100 if len(p1_users) > 0 else 0.0
        }
        
        # Occupation comparison
        p1_occs = set(p1_active[self.occupation_col].dropna().unique())
        p2_occs = set(p2_active[self.occupation_col].dropna().unique())
        
        comparison['period1_occupations'] = len(p1_occs)
        comparison['period2_occupations'] = len(p2_occs)
        comparison['occupations_in_both'] = len(p1_occs & p2_occs)
        comparison['new_occupations'] = len(p2_occs - p1_occs)
        comparison['disappeared_occupations'] = len(p1_occs - p2_occs)
        
        return comparison
    
    def diagnose_data_quality(self, 
                             trajectory_df: pd.DataFrame,
                             start_year: int,
                             end_year: int) -> Dict:
        """
        Diagnose potential data quality issues affecting windowed analysis
        
        Args:
            trajectory_df: Career trajectory dataframe
            start_year: Analysis start year
            end_year: Analysis end year
        
        Returns:
            Dictionary with diagnostic information
        """
        print("\n" + "="*80)
        print("DATA QUALITY DIAGNOSTICS")
        print("="*80)
        
        diagnostics = {}
        
        # Overall statistics
        diagnostics['total_records'] = len(trajectory_df)
        diagnostics['unique_users'] = trajectory_df['ID'].nunique()
        diagnostics['start_year_min'] = trajectory_df['job_start_year'].min()
        diagnostics['start_year_max'] = trajectory_df['job_start_year'].max()
        
        print(f"\nOverall Dataset:")
        print(f"  Total records: {diagnostics['total_records']:,}")
        print(f"  Unique users: {diagnostics['unique_users']:,}")
        print(f"  Job start year range: {diagnostics['start_year_min']:.0f}-{diagnostics['start_year_max']:.0f}")
        
        # End year analysis
        if 'job_end_year' in trajectory_df.columns:
            diagnostics['end_year_min'] = trajectory_df['job_end_year'].min()
            diagnostics['end_year_max'] = trajectory_df['job_end_year'].max()
            diagnostics['null_end_years'] = trajectory_df['job_end_year'].isna().sum()
            diagnostics['null_end_years_pct'] = diagnostics['null_end_years'] / len(trajectory_df) * 100
            
            print(f"  Job end year range: {diagnostics['end_year_min']:.0f}-{diagnostics['end_year_max']:.0f}")
            print(f"  Jobs with no end date: {diagnostics['null_end_years']:,} ({diagnostics['null_end_years_pct']:.1f}%)")
        
        # Year-by-year breakdown
        print(f"\nYear-by-Year Job Starts ({start_year}-{end_year}):")
        year_starts = {}
        for year in range(start_year, end_year + 1):
            count = (trajectory_df['job_start_year'] == year).sum()
            year_starts[year] = count
            print(f"  {year}: {count:,} jobs started")
        diagnostics['year_starts'] = year_starts
        
        if 'job_end_year' in trajectory_df.columns:
            print(f"\nYear-by-Year Job Ends ({start_year}-{end_year}):")
            year_ends = {}
            for year in range(start_year, end_year + 1):
                count = (trajectory_df['job_end_year'] == year).sum()
                year_ends[year] = count
                print(f"  {year}: {count:,} jobs ended")
            diagnostics['year_ends'] = year_ends
        
        # Identify drop-off pattern
        print(f"\nData Quality Warnings:")
        warnings = []
        
        # Check for sudden drops in job starts
        year_list = sorted(year_starts.keys())
        for i in range(1, len(year_list)):
            prev_year = year_list[i-1]
            curr_year = year_list[i]
            prev_count = year_starts[prev_year]
            curr_count = year_starts[curr_year]
            
            if prev_count > 0:
                pct_change = ((curr_count - prev_count) / prev_count) * 100
                if pct_change < -30:  # 30% drop
                    warning = f"  ⚠️  Sudden drop {prev_year}→{curr_year}: {prev_count:,} → {curr_count:,} ({pct_change:.1f}%)"
                    warnings.append(warning)
                    print(warning)
        
        if len(warnings) == 0:
            print("  No major data quality issues detected")
        
        diagnostics['warnings'] = warnings
        
        # Sample recent records
        print(f"\nSample of Recent Jobs (2020+):")
        recent = trajectory_df[trajectory_df['job_start_year'] >= 2020].head(10)
        if len(recent) > 0:
            print(recent[['ID', 'job_start_year', 'job_end_year', self.occupation_col]].to_string())
        else:
            print("  ⚠️  No jobs found starting in 2020 or later!")
        
        print("\n" + "="*80)
        
        return diagnostics
"""
Unified mobility analyzer - MODIFIED FOR TRAJECTORY DATA
Analyzes preprocessed trajectory data instead of raw job data
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict


class MobilityAnalyzer:
    """
    Labor market mobility analyzer for trajectory data
    Analyzes enriched trajectory dataframe from preprocess.py
    """
    
    def __init__(self, results_dir='results', occupation_col='onet_major'):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory for caching and results
            occupation_col: Occupation column name to use (default: onet_major from trajectory_df)
        """
        self.results_dir = results_dir
        self.occupation_col = occupation_col
    
    # ========================================
    # PUBLIC API - Main Analysis Method
    # ========================================
    
    def analyze_all(self, trajectory_df, config):
        """
        Run complete mobility analysis pipeline using trajectory data
        
        Args:
            trajectory_df: Enriched trajectory dataframe from preprocess.py
            config: Config instance with study period information
        
        Returns:
            Dictionary with all analysis results:
                - trajectory_summary: Summary statistics by year
                - mobility_by_demographics: Mobility metrics by demographic groups
                - job_change_analysis: Analysis of job change types
                - wage_mobility: Wage-based mobility analysis
        """
        print("\n" + "="*80)
        print("RUNNING TRAJECTORY-BASED MOBILITY ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Trajectory summary by year
        print("\n[1/4] Computing trajectory summary statistics...")
        results['trajectory_summary'] = self.compute_trajectory_summary(trajectory_df, config)
        
        # 2. Demographic analysis
        print("\n[2/4] Analyzing mobility by demographics...")
        results['mobility_by_demographics'] = self.analyze_by_demographics(trajectory_df, config)
        
        # 3. Job change type analysis
        print("\n[3/4] Analyzing job change patterns...")
        results['job_change_analysis'] = self.analyze_job_changes(trajectory_df, config)
        
        # 4. Wage mobility analysis
        print("\n[4/4] Analyzing wage-based mobility...")
        results['wage_mobility'] = self.analyze_wage_mobility(trajectory_df, config)
        
        print("\n" + "="*80)
        print("TRAJECTORY ANALYSIS COMPLETE")
        print("="*80)
        
        return results
    
    # ========================================
    # TRAJECTORY SUMMARY
    # ========================================
    
    def compute_trajectory_summary(self, trajectory_df, config):
        """
        Compute summary statistics from trajectory data
        
        Args:
            trajectory_df: Enriched trajectory dataframe
            config: Config instance
        
        Returns:
            Summary statistics dataframe grouped by year
        """
        print("\nComputing trajectory summary by year...")
        
        # Debug: Check year column
        print(f"DEBUG: job_start_year dtype: {trajectory_df['job_start_year'].dtype}")
        print(f"DEBUG: job_start_year unique values (first 20): {sorted(trajectory_df['job_start_year'].dropna().unique())[:20]}")
        print(f"DEBUG: job_start_year range: {trajectory_df['job_start_year'].min():.0f} - {trajectory_df['job_start_year'].max():.0f}")
        
        summary_stats = []
        
        # Convert year to int if needed
        trajectory_df = trajectory_df.copy()
        if trajectory_df['job_start_year'].dtype in ['float64', 'float32']:
            trajectory_df['job_start_year'] = trajectory_df['job_start_year'].astype('Int64')
        
        # Filter by study period
        study_trajectories = trajectory_df[
            (trajectory_df['job_start_year'] >= config.study_start_year) &
            (trajectory_df['job_start_year'] <= config.study_end_year)
        ].copy()
        
        print(f"Analyzing {len(study_trajectories):,} trajectories in study period "
              f"({config.study_start_year}-{config.study_end_year})")
        
        # Debug: Show year distribution in filtered data
        if len(study_trajectories) > 0:
            year_counts = study_trajectories['job_start_year'].value_counts().sort_index()
            print(f"DEBUG: Years in filtered data: {year_counts.to_dict()}")
        
        # Group by year and compute statistics
        for year in range(config.study_start_year, config.study_end_year + 1):
            year_data = study_trajectories[study_trajectories['job_start_year'] == year]
            
            if len(year_data) == 0:
                continue
            
            stats = {
                'year': year,
                'n_trajectories': len(year_data),
            }
            
            # Job mobility metrics
            if 'num_job_changes' in year_data.columns:
                stats['avg_job_changes'] = year_data['num_job_changes'].mean()
                stats['median_job_changes'] = year_data['num_job_changes'].median()
                stats['pct_no_change'] = (year_data['num_job_changes'] == 0).mean() * 100
                stats['pct_multiple_changes'] = (year_data['num_job_changes'] >= 2).mean() * 100
            
            # Upward mobility
            if 'up_move' in year_data.columns:
                stats['upward_mobility_rate'] = year_data['up_move'].mean() * 100
            
            # Wage statistics
            if 'annual_state_wage' in year_data.columns:
                stats['avg_wage'] = year_data['annual_state_wage'].mean()
                stats['median_wage'] = year_data['annual_state_wage'].median()
                stats['wage_25th_pct'] = year_data['annual_state_wage'].quantile(0.25)
                stats['wage_75th_pct'] = year_data['annual_state_wage'].quantile(0.75)
            
            # Demographics
            if 'gender' in year_data.columns:
                stats['pct_female'] = (year_data['gender'] == 2).mean() * 100
            
            if 'race' in year_data.columns:
                race_dist = year_data['race'].value_counts(normalize=True) * 100
                for race_code, pct in race_dist.items():
                    stats[f'pct_race_{int(race_code)}'] = pct
            
            # Job change types
            move_cols = ['move_1_1', 'move_1_2', 'move_2_1', 'move_2_2']
            if all(col in year_data.columns for col in move_cols):
                stats['pct_move_1_1'] = year_data['move_1_1'].mean() * 100  # Diff company, diff occ
                stats['pct_move_1_2'] = year_data['move_1_2'].mean() * 100  # Diff company, same occ
                stats['pct_move_2_1'] = year_data['move_2_1'].mean() * 100  # Same company, diff occ
                stats['pct_move_2_2'] = year_data['move_2_2'].mean() * 100  # Same company, same occ
            
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        print(f"✓ Computed summary for {len(summary_df)} years")
        
        return summary_df
    
    # ========================================
    # DEMOGRAPHIC ANALYSIS
    # ========================================
    
    def analyze_by_demographics(self, trajectory_df, config):
        """
        Analyze mobility patterns by demographic groups
        
        Args:
            trajectory_df: Enriched trajectory dataframe
            config: Config instance
        
        Returns:
            Dictionary with demographic breakdowns
        """
        print("\nAnalyzing mobility by demographic groups...")
        
        # Convert year to int if needed
        trajectory_df = trajectory_df.copy()
        if trajectory_df['job_start_year'].dtype in ['float64', 'float32']:
            trajectory_df['job_start_year'] = trajectory_df['job_start_year'].astype('Int64')
        
        study_trajectories = trajectory_df[
            (trajectory_df['job_start_year'] >= config.study_start_year) &
            (trajectory_df['job_start_year'] <= config.study_end_year)
        ].copy()
        
        results = {}
        
        # By gender
        if 'gender' in study_trajectories.columns:
            gender_analysis = []
            for gender in study_trajectories['gender'].dropna().unique():
                gender_data = study_trajectories[study_trajectories['gender'] == gender]
                
                stats = {
                    'gender': int(gender),
                    'n': len(gender_data),
                    'avg_job_changes': gender_data['num_job_changes'].mean() if 'num_job_changes' in gender_data.columns else None,
                    'upward_mobility_rate': gender_data['up_move'].mean() * 100 if 'up_move' in gender_data.columns else None,
                    'avg_wage': gender_data['annual_state_wage'].mean() if 'annual_state_wage' in gender_data.columns else None,
                }
                gender_analysis.append(stats)
            
            results['by_gender'] = pd.DataFrame(gender_analysis)
            print(f"  ✓ Gender analysis: {len(gender_analysis)} groups")
        
        # By race
        if 'race' in study_trajectories.columns:
            race_analysis = []
            for race in study_trajectories['race'].dropna().unique():
                race_data = study_trajectories[study_trajectories['race'] == race]
                
                stats = {
                    'race': int(race),
                    'n': len(race_data),
                    'avg_job_changes': race_data['num_job_changes'].mean() if 'num_job_changes' in race_data.columns else None,
                    'upward_mobility_rate': race_data['up_move'].mean() * 100 if 'up_move' in race_data.columns else None,
                    'avg_wage': race_data['annual_state_wage'].mean() if 'annual_state_wage' in race_data.columns else None,
                }
                race_analysis.append(stats)
            
            results['by_race'] = pd.DataFrame(race_analysis)
            print(f"  ✓ Race analysis: {len(race_analysis)} groups")
        
        # By generation
        if 'generation' in study_trajectories.columns:
            gen_analysis = []
            for gen in study_trajectories['generation'].dropna().unique():
                gen_data = study_trajectories[study_trajectories['generation'] == gen]
                
                stats = {
                    'generation': gen,
                    'n': len(gen_data),
                    'avg_job_changes': gen_data['num_job_changes'].mean() if 'num_job_changes' in gen_data.columns else None,
                    'upward_mobility_rate': gen_data['up_move'].mean() * 100 if 'up_move' in gen_data.columns else None,
                    'avg_wage': gen_data['annual_state_wage'].mean() if 'annual_state_wage' in gen_data.columns else None,
                }
                gen_analysis.append(stats)
            
            results['by_generation'] = pd.DataFrame(gen_analysis)
            print(f"  ✓ Generation analysis: {len(gen_analysis)} groups")
        
        # By education
        if 'max_edu_name' in study_trajectories.columns:
            edu_analysis = []
            for edu in study_trajectories['max_edu_name'].dropna().unique():
                edu_data = study_trajectories[study_trajectories['max_edu_name'] == edu]
                
                stats = {
                    'education': edu,
                    'n': len(edu_data),
                    'avg_job_changes': edu_data['num_job_changes'].mean() if 'num_job_changes' in edu_data.columns else None,
                    'upward_mobility_rate': edu_data['up_move'].mean() * 100 if 'up_move' in edu_data.columns else None,
                    'avg_wage': edu_data['annual_state_wage'].mean() if 'annual_state_wage' in edu_data.columns else None,
                }
                edu_analysis.append(stats)
            
            results['by_education'] = pd.DataFrame(edu_analysis)
            print(f"  ✓ Education analysis: {len(edu_analysis)} groups")
        
        return results
    
    # ========================================
    # JOB CHANGE ANALYSIS
    # ========================================
    
    def analyze_job_changes(self, trajectory_df, config):
        """
        Analyze job change patterns and types
        
        Args:
            trajectory_df: Enriched trajectory dataframe
            config: Config instance
        
        Returns:
            Dictionary with job change analysis results
        """
        print("\nAnalyzing job change patterns...")
        
        # Convert year to int if needed
        trajectory_df = trajectory_df.copy()
        if trajectory_df['job_start_year'].dtype in ['float64', 'float32']:
            trajectory_df['job_start_year'] = trajectory_df['job_start_year'].astype('Int64')
        
        study_trajectories = trajectory_df[
            (trajectory_df['job_start_year'] >= config.study_start_year) &
            (trajectory_df['job_start_year'] <= config.study_end_year)
        ].copy()
        
        results = {}
        
        # Overall distribution of job changes
        if 'num_job_changes' in study_trajectories.columns:
            change_dist = study_trajectories['num_job_changes'].value_counts().sort_index()
            results['change_distribution'] = pd.DataFrame({
                'num_changes': change_dist.index,
                'count': change_dist.values,
                'percentage': (change_dist.values / len(study_trajectories) * 100).round(2)
            })
            print(f"  ✓ Job change distribution computed")
        
        # Job change types over time
        move_cols = ['move_1_1', 'move_1_2', 'move_2_1', 'move_2_2']
        if all(col in study_trajectories.columns for col in move_cols):
            move_by_year = []
            for year in range(config.study_start_year, config.study_end_year + 1):
                year_data = study_trajectories[study_trajectories['job_start_year'] == year]
                if len(year_data) == 0:
                    continue
                
                move_stats = {
                    'year': year,
                    'n': len(year_data),
                    'move_1_1_count': year_data['move_1_1'].sum(),
                    'move_1_2_count': year_data['move_1_2'].sum(),
                    'move_2_1_count': year_data['move_2_1'].sum(),
                    'move_2_2_count': year_data['move_2_2'].sum(),
                    'move_1_1_pct': year_data['move_1_1'].mean() * 100,
                    'move_1_2_pct': year_data['move_1_2'].mean() * 100,
                    'move_2_1_pct': year_data['move_2_1'].mean() * 100,
                    'move_2_2_pct': year_data['move_2_2'].mean() * 100,
                }
                move_by_year.append(move_stats)
            
            results['job_change_types_by_year'] = pd.DataFrame(move_by_year)
            print(f"  ✓ Job change types by year computed")
        
        # Relationship between job changes and upward mobility
        if 'num_job_changes' in study_trajectories.columns and 'up_move' in study_trajectories.columns:
            mobility_by_changes = []
            for n_changes in sorted(study_trajectories['num_job_changes'].unique()):
                subset = study_trajectories[study_trajectories['num_job_changes'] == n_changes]
                
                mobility_by_changes.append({
                    'num_changes': n_changes,
                    'n': len(subset),
                    'upward_mobility_rate': subset['up_move'].mean() * 100,
                    'avg_wage': subset['annual_state_wage'].mean() if 'annual_state_wage' in subset.columns else None
                })
            
            results['mobility_by_num_changes'] = pd.DataFrame(mobility_by_changes)
            print(f"  ✓ Mobility by number of changes computed")
        
        return results
    
    # ========================================
    # WAGE MOBILITY ANALYSIS
    # ========================================
    
    def analyze_wage_mobility(self, trajectory_df, config):
        """
        Analyze wage-based mobility patterns
        
        Args:
            trajectory_df: Enriched trajectory dataframe
            config: Config instance
        
        Returns:
            Dictionary with wage mobility analysis
        """
        print("\nAnalyzing wage-based mobility...")
        
        # Convert year to int if needed
        trajectory_df = trajectory_df.copy()
        if trajectory_df['job_start_year'].dtype in ['float64', 'float32']:
            trajectory_df['job_start_year'] = trajectory_df['job_start_year'].astype('Int64')
        
        study_trajectories = trajectory_df[
            (trajectory_df['job_start_year'] >= config.study_start_year) &
            (trajectory_df['job_start_year'] <= config.study_end_year)
        ].copy()
        
        results = {}
        
        # Upward mobility by year
        if 'up_move' in study_trajectories.columns:
            up_move_by_year = []
            for year in range(config.study_start_year, config.study_end_year + 1):
                year_data = study_trajectories[study_trajectories['job_start_year'] == year]
                if len(year_data) == 0:
                    continue
                
                up_move_by_year.append({
                    'year': year,
                    'n': len(year_data),
                    'upward_mobility_count': year_data['up_move'].sum(),
                    'upward_mobility_rate': year_data['up_move'].mean() * 100
                })
            
            results['upward_mobility_by_year'] = pd.DataFrame(up_move_by_year)
            print(f"  ✓ Upward mobility by year computed")
        
        # Wage distribution by year
        if 'annual_state_wage' in study_trajectories.columns:
            wage_by_year = []
            for year in range(config.study_start_year, config.study_end_year + 1):
                year_data = study_trajectories[study_trajectories['job_start_year'] == year]
                if len(year_data) == 0:
                    continue
                
                wage_by_year.append({
                    'year': year,
                    'n': len(year_data),
                    'mean_wage': year_data['annual_state_wage'].mean(),
                    'median_wage': year_data['annual_state_wage'].median(),
                    'p10_wage': year_data['annual_state_wage'].quantile(0.10),
                    'p25_wage': year_data['annual_state_wage'].quantile(0.25),
                    'p75_wage': year_data['annual_state_wage'].quantile(0.75),
                    'p90_wage': year_data['annual_state_wage'].quantile(0.90),
                })
            
            results['wage_distribution_by_year'] = pd.DataFrame(wage_by_year)
            print(f"  ✓ Wage distribution by year computed")
        
        # Upward mobility by occupation
        if 'up_move' in study_trajectories.columns and self.occupation_col in study_trajectories.columns:
            up_move_by_occ = []
            for occ in study_trajectories[self.occupation_col].dropna().unique():
                occ_data = study_trajectories[study_trajectories[self.occupation_col] == occ]
                
                if len(occ_data) < 30:  # Minimum threshold
                    continue
                
                up_move_by_occ.append({
                    'occupation': occ,
                    'n': len(occ_data),
                    'upward_mobility_rate': occ_data['up_move'].mean() * 100,
                    'avg_wage': occ_data['annual_state_wage'].mean() if 'annual_state_wage' in occ_data.columns else None
                })
            
            if len(up_move_by_occ) > 0:
                results['upward_mobility_by_occupation'] = pd.DataFrame(up_move_by_occ).sort_values(
                    'upward_mobility_rate', ascending=False
                )
                print(f"  ✓ Upward mobility by occupation computed ({len(up_move_by_occ)} occupations)")
            else:
                print(f"  ⚠ No occupations met minimum threshold for upward mobility analysis")
        
        # Wage mobility by demographics
        if 'up_move' in study_trajectories.columns:
            if 'gender' in study_trajectories.columns:
                results['upward_mobility_by_gender'] = study_trajectories.groupby('gender')['up_move'].agg([
                    ('n', 'count'),
                    ('upward_count', 'sum'),
                    ('upward_rate', lambda x: x.mean() * 100)
                ]).reset_index()
                print(f"  ✓ Upward mobility by gender computed")
            
            if 'race' in study_trajectories.columns:
                results['upward_mobility_by_race'] = study_trajectories.groupby('race')['up_move'].agg([
                    ('n', 'count'),
                    ('upward_count', 'sum'),
                    ('upward_rate', lambda x: x.mean() * 100)
                ]).reset_index()
                print(f"  ✓ Upward mobility by race computed")
        
        return results


# ========================================
# LEGACY FUNCTIONS (for backward compatibility)
# ========================================

def build_transitions(window_users, windows):
    """Legacy function - not compatible with trajectory data"""
    raise NotImplementedError(
        "build_transitions() requires window_users format. "
        "Use MobilityAnalyzer.analyze_all() with trajectory_df instead."
    )


def calculate_transition_rates(all_transitions_df):
    """Legacy function - not compatible with trajectory data"""
    raise NotImplementedError(
        "calculate_transition_rates() requires transition dataframe. "
        "Use MobilityAnalyzer.analyze_all() with trajectory_df instead."
    )


def analyze_workforce_flow(window_users, windows, results_dir='results', 
                          occupation_col='SOC_EMSI_2019_3_NAME'):
    """Legacy function - not compatible with trajectory data"""
    raise NotImplementedError(
        "analyze_workforce_flow() requires window_users format. "
        "Use MobilityAnalyzer.analyze_all() with trajectory_df instead."
    )
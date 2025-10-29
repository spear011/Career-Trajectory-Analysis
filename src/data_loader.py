"""
Unified data loading and preprocessing for labor market analysis
MODIFIED: Added support for loading preprocessed trajectory data
"""
import pandas as pd
import os
from src.utils import get_config, get_active_jobs, get_latest_job_per_user


class DataLoader:
    """
    Unified data loader for all labor market data sources
    """
    
    def __init__(self, config=None):
        """
        Initialize DataLoader
        
        Args:
            config: Config instance (loads default if None)
        """
        if config is None:
            config = get_config()
        self.config = config
        self.data_dir = config.data_dir
    
    # ========================================
    # PREPROCESSED TRAJECTORY DATA
    # ========================================
    
    def load_preprocessed_trajectories(self, trajectory_path=None):
        """
        Load preprocessed trajectory data from preprocess.py output
        
        Args:
            trajectory_path: Path to trajectory parquet file.
                           If None, uses default path from config.
        
        Returns:
            Trajectory dataframe with enriched features including:
            - Demographics (gender, race, generation)
            - Education (max_edu_name)
            - First job info (onet_major_x, naics6_major_x, company_x, state_x, job_start_year_x)
            - Wages (annual_state_wage_x, log_wage_x)
            - Mobility indicators (num_job_changes, up_move)
            - Job change types (move_1_1, move_1_2, move_2_1, move_2_2)
            - State GDP metrics (state_gdp_decile_x)
        """
        if trajectory_path is None:
            trajectory_path = os.path.join(
                self.config.results_dir, 
                'career_trajectories.parquet'
            )
        
        print(f"Loading preprocessed trajectory data from: {trajectory_path}")
        
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(
                f"Preprocessed trajectory file not found: {trajectory_path}\n"
                f"Please run preprocess.py first to generate the trajectory data."
            )
        
        trajectory_df = pd.read_parquet(trajectory_path)
        
        print(f"Loaded trajectory data:")
        print(f"  Total trajectories: {len(trajectory_df):,}")
        print(f"  Features: {len(trajectory_df.columns)}")
        
        # Display key columns
        key_columns = [
            'onet_major_x', 'naics6_major_x', 'job_start_year_x', 
            'num_job_changes', 'gender', 'race', 'generation',
            'annual_state_wage_x', 'up_move'
        ]
        available_keys = [col for col in key_columns if col in trajectory_df.columns]
        print(f"  Key features available: {available_keys}")
        
        return trajectory_df
    
    # ========================================
    # JOB DATA (Original methods preserved)
    # ========================================
    
    def load_job_data(self):
        """
        Load and preprocess job data
        
        Returns:
            Preprocessed job dataframe
        """
        print("Loading job data...")
        job_df = pd.read_csv(os.path.join(self.data_dir, 'job', 'job_group_0.csv'))
        
        # Convert date columns
        job_df['JOB_START_DATE'] = pd.to_datetime(job_df['JOB_START_DATE'], format='%Y-%m', errors='coerce')
        job_df['JOB_END_DATE'] = pd.to_datetime(job_df['JOB_END_DATE'], format='%Y-%m', errors='coerce')
        
        print(f"Total job records: {len(job_df):,}")
        return job_df
    
    def get_window_users(self, job_df, windows):
        """
        Get users for each window
        
        Args:
            job_df: Job dataframe
            windows: List of (name, start, end, label) tuples
        
        Returns:
            Dictionary mapping window name to user dataframe
        """
        print("\nProcessing yearly windows...")
        window_users = {}
        
        for name, start, end, label in windows:
            jobs = get_active_jobs(job_df, start, end)
            users = get_latest_job_per_user(jobs)
            window_users[name] = users
            print(f"{label}: {len(users):,} users")
        
        return window_users
    
    def get_occupation_distributions(self, window_users, windows, occupation_col=None):
        """
        Compute occupation distributions across windows
        
        Args:
            window_users: Dictionary mapping window name to user dataframe
            windows: List of (name, start, end, label) tuples
            occupation_col: Column name for occupation (uses config default if None)
        
        Returns:
            Long-form dataframe with occupation distributions over time
        """
        if occupation_col is None:
            occupation_col = self.config.analysis_occupation_column
        
        print(f"\nComputing occupation distributions (using {occupation_col})...")
        
        all_dists = []
        for name, start, end, label in windows:
            users = window_users[name]
            year = int(name.split('_')[0])
            
            # Count occupations
            occ_counts = users[occupation_col].value_counts()
            total = len(users)
            
            # Create distribution dataframe
            dist_df = pd.DataFrame({
                'year': year,
                'occupation': occ_counts.index,
                'count': occ_counts.values,
                'percentage': (occ_counts.values / total * 100).round(2)
            })
            
            all_dists.append(dist_df)
        
        result_df = pd.concat(all_dists, ignore_index=True)
        print(f"✓ Computed distributions for {len(windows)} time windows")
        
        return result_df
    
    # ========================================
    # EDUCATION DATA
    # ========================================
    
    def load_education_data(self):
        """
        Load education data
        
        Returns:
            Education dataframe
        """
        print("Loading education data...")
        edu_df = pd.read_csv(os.path.join(self.data_dir, 'education', 'education_group_0.csv'))
        
        # Convert date columns
        edu_df['START_DATE'] = pd.to_datetime(edu_df['START_DATE'], errors='coerce')
        edu_df['END_DATE'] = pd.to_datetime(edu_df['END_DATE'], errors='coerce')
        
        print(f"Total education records: {len(edu_df):,}")
        return edu_df
    
    def merge_job_education(self, job_df, edu_df):
        """
        Merge job and education data
        
        Args:
            job_df: Job dataframe
            edu_df: Education dataframe
        
        Returns:
            Merged dataframe
        """
        print("\nMerging job and education data...")
        merged = job_df.merge(edu_df, on='ID', how='inner', suffixes=('', '_edu'))
        print(f"Merged records: {len(merged):,}")
        return merged
    
    def get_occupation_major_mappings(self, job_df, ed_df, occupation_col=None, min_count=100):
        """
        Get occupation to major mappings
        
        Args:
            job_df: Job dataframe
            ed_df: Education dataframe
            occupation_col: Occupation column name (uses config default if None)
            min_count: Minimum count threshold for valid mappings
        
        Returns:
            Dataframe with occupation-major mappings
        """
        if occupation_col is None:
            occupation_col = self.config.analysis_occupation_column
        
        print(f"\nMapping occupations to majors (min_count={min_count})...")
        
        merged_df = self.merge_job_education(job_df, ed_df)
        
        # Filter valid data
        valid_data = merged_df[
            (merged_df[occupation_col].notna()) & 
            (merged_df['Major'].notna())
        ].copy()
        
        # Group by occupation and major
        major_occ = valid_data.groupby([occupation_col, 'Major']).size().reset_index(name='count')
        
        # Filter occupations with sufficient data
        occ_counts = major_occ.groupby(occupation_col)['count'].sum()
        valid_occs = occ_counts[occ_counts >= min_count].index
        major_occ = major_occ[major_occ[occupation_col].isin(valid_occs)]
        
        # Get top 5 majors per occupation
        top_majors = major_occ.sort_values(['count'], ascending=False).groupby(occupation_col).head(5)
        
        return top_majors
    
    # ========================================
    # SKILLS DATA
    # ========================================
    
    def load_skills_data(self):
        """
        Load skills data
        
        Returns:
            Skills dataframe
        """
        print("Loading skills data...")
        skills_df = pd.read_csv(os.path.join(self.data_dir, 'skill_group_0.csv'))
        print(f"Total skill records: {len(skills_df):,}")
        return skills_df
    
    # ========================================
    # WAGE DATA
    # ========================================
    
    def load_wage_data(self, filename='wages.csv'):
        """
        Load wage data
        
        Args:
            filename: Wage data filename
        
        Returns:
            Wage dataframe
        """
        print("Loading wage data...")
        wage_df = pd.read_csv(os.path.join(self.data_dir, filename))
        print(f"Total wage records: {len(wage_df):,}")
        return wage_df
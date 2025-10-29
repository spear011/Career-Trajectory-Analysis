"""
Unified data loading and preprocessing for labor market analysis
"""
import pandas as pd
import os
from .utils import get_config, get_active_jobs, get_latest_job_per_user


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
    # JOB DATA
    # ========================================
    
    def load_job_data(self):
        """
        Load and preprocess job data
        
        Returns:
            Preprocessed job dataframe
        """
        print("Loading job data...")
        job_df = pd.read_csv(os.path.join(self.data_dir, 'job' ,'job_group_0.csv'))
        
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
        Calculate occupation distributions for each window
        
        Args:
            window_users: Dictionary of window users
            windows: List of windows
            occupation_col: Name of occupation column (uses config if None)
        
        Returns:
            Dataframe with occupation distributions by year
        """
        if occupation_col is None:
            occupation_col = self.config.analysis_occupation_column
        
        print("\nAnalyzing occupation distributions...")
        occ_distributions = []
        
        for name, _, _, label in windows:
            occ_counts = window_users[name][occupation_col].value_counts()
            total = len(window_users[name])
            
            for occ, count in occ_counts.items():
                occ_distributions.append({
                    'Year': int(label),
                    'Occupation': occ,
                    'Count': count,
                    'Percentage': (count / total) * 100
                })
        
        return pd.DataFrame(occ_distributions)
    
    # ========================================
    # EDUCATION DATA
    # ========================================
    
    def load_education_data(self):
        """
        Load and preprocess education data
        
        Returns:
            Preprocessed education dataframe
        """
        print("Loading education data...")
        ed_df = pd.read_csv(os.path.join(self.data_dir, 'education_group_0.csv'))
        
        # Convert date columns
        ed_df['START_DATE'] = pd.to_datetime(ed_df['START_DATE'], format='%Y-%m', errors='coerce')
        ed_df['END_DATE'] = pd.to_datetime(ed_df['END_DATE'], format='%Y-%m', errors='coerce')
        
        # Handle graduation year
        ed_df['GRAD_YEAR'] = pd.to_numeric(ed_df['GRAD_YEAR'], errors='coerce')
        
        print(f"Total education records: {len(ed_df):,}")
        print(f"Unique individuals: {ed_df['ID'].nunique():,}")
        
        return ed_df
    
    def get_education_level_distribution(self, ed_df):
        """
        Get distribution of education levels
        
        Args:
            ed_df: Education dataframe
        
        Returns:
            DataFrame with education level counts
        """
        level_dist = ed_df.groupby(['EDULEVEL_NAME']).size().reset_index(name='count')
        level_dist = level_dist.sort_values('count', ascending=False)
        return level_dist
    
    def get_major_distribution(self, ed_df, top_n=20):
        """
        Get distribution of majors (CIP6)
        
        Args:
            ed_df: Education dataframe
            top_n: Number of top majors to return
        
        Returns:
            DataFrame with major counts
        """
        major_dist = ed_df.groupby(['CIP6_2020_NAME']).size().reset_index(name='count')
        major_dist = major_dist.sort_values('count', ascending=False).head(top_n)
        return major_dist
    
    def merge_job_education(self, job_df, ed_df):
        """
        Merge job and education data
        
        Args:
            job_df: Job dataframe
            ed_df: Education dataframe
        
        Returns:
            Merged dataframe
        """
        print("Merging job and education data...")
        merged_df = job_df.merge(
            ed_df[['ID', 'CIP6_2020_NAME', 'EDULEVEL_NAME']], 
            on='ID', 
            how='left'
        )
        merged_df = merged_df.rename(columns={'CIP6_2020_NAME': 'Major'})
        print(f"Merged records: {len(merged_df):,}")
        return merged_df
    
    def get_occupation_major_mapping(self, job_df, ed_df, occupation_col=None, min_count=50):
        """
        Map occupations to common majors
        
        Args:
            job_df: Job dataframe
            ed_df: Education dataframe
            occupation_col: Occupation column name (uses config if None)
            min_count: Minimum occupation count
        
        Returns:
            DataFrame with occupation-major mappings
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
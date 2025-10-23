"""
Unified data loading and preprocessing for labor market analysis
Combines job, education, skills, and wage data loading
"""
import pandas as pd
import os


class DataLoader:
    """
    Unified data loader for all labor market data sources
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
    
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
        job_df = pd.read_csv(os.path.join(self.data_dir, 'job_group_0.csv'))
        
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
        from .utils import get_active_jobs, get_latest_job_per_user
        
        print("\nProcessing yearly windows...")
        window_users = {}
        
        for name, start, end, label in windows:
            jobs = get_active_jobs(job_df, start, end)
            users = get_latest_job_per_user(jobs)
            window_users[name] = users
            print(f"{label}: {len(users):,} users")
        
        return window_users
    
    def get_occupation_distributions(self, window_users, windows, occupation_col):
        """
        Calculate occupation distributions for each window
        
        Args:
            window_users: Dictionary of window users
            windows: List of windows
            occupation_col: Name of occupation column
        
        Returns:
            Dataframe with occupation distributions by year
        """
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
        major_dist = ed_df[ed_df['CIP6_2020_NAME'].notna()].groupby('CIP6_2020_NAME').size().reset_index(name='count')
        major_dist = major_dist.sort_values('count', ascending=False).head(top_n)
        return major_dist
    
    def get_user_highest_education(self, ed_df):
        """
        For each user, get their highest education level
        Priority: Doctoral > Master's > Bachelor's > Associate's > Other
        
        Args:
            ed_df: Education dataframe
        
        Returns:
            DataFrame with one row per user (their highest education)
        """
        # Define education level priority
        level_priority = {
            "Doctoral Degree": 1,
            "Master's Degree": 2,
            "Bachelor's Degree": 3,
            "Associate's Degree": 4,
            "Post-Baccalaureate Certificate": 5,
            "Certificate": 6,
        }
        
        # Add priority column
        ed_df_copy = ed_df.copy()
        ed_df_copy['edu_priority'] = ed_df_copy['EDULEVEL_NAME'].map(level_priority)
        ed_df_copy['edu_priority'] = ed_df_copy['edu_priority'].fillna(99)
        
        # Sort by priority and graduation year (most recent first)
        ed_df_copy = ed_df_copy.sort_values(['edu_priority', 'GRAD_YEAR'], 
                                            ascending=[True, False])
        
        # Get first (highest) education per user
        highest_ed = ed_df_copy.groupby('ID').first().reset_index()
        
        return highest_ed
    
    def merge_job_education(self, job_df, ed_df):
        """
        Merge job and education data
        
        Args:
            job_df: Job dataframe with ID column
            ed_df: Education dataframe
        
        Returns:
            Merged dataframe
        """
        print("\nMerging job and education data...")
        
        # Get highest education per user
        highest_ed = self.get_user_highest_education(ed_df)
        
        # Select key education columns
        ed_cols = ['ID', 'EDULEVEL_NAME', 'CIP6_2020_NAME', 'SCHOOL_NAME', 
                   'GRAD_YEAR', 'ED_STATE', 'ED_COUNTRY']
        highest_ed_subset = highest_ed[ed_cols].copy()
        
        # Rename for clarity
        highest_ed_subset = highest_ed_subset.rename(columns={
            'EDULEVEL_NAME': 'Education_Level',
            'CIP6_2020_NAME': 'Major',
            'SCHOOL_NAME': 'School',
            'GRAD_YEAR': 'Graduation_Year',
            'ED_STATE': 'Education_State',
            'ED_COUNTRY': 'Education_Country'
        })
        
        # Merge with job data
        merged = job_df.merge(highest_ed_subset, on='ID', how='left')
        
        print(f"Jobs with education data: {merged['Education_Level'].notna().sum():,} / {len(merged):,}")
        
        return merged
    
    def analyze_education_by_occupation(self, merged_df, occupation_col='SOC_EMSI_2019_3_NAME'):
        """
        Analyze education distribution by occupation
        
        Args:
            merged_df: Merged job-education dataframe
            occupation_col: Name of occupation column
        
        Returns:
            DataFrame with education stats by occupation
        """
        print("\nAnalyzing education by occupation...")
        
        # Filter for records with both occupation and education
        valid_data = merged_df[
            (merged_df[occupation_col].notna()) & 
            (merged_df['Education_Level'].notna())
        ].copy()
        
        # Group by occupation and education level
        edu_occ = valid_data.groupby([occupation_col, 'Education_Level']).size().reset_index(name='count')
        
        # Calculate percentage within each occupation
        occ_totals = edu_occ.groupby(occupation_col)['count'].transform('sum')
        edu_occ['percentage'] = (edu_occ['count'] / occ_totals) * 100
        
        return edu_occ
    
    def analyze_major_by_occupation(self, merged_df, occupation_col='SOC_EMSI_2019_3_NAME', min_count=50):
        """
        Analyze major distribution by occupation
        
        Args:
            merged_df: Merged job-education dataframe
            occupation_col: Name of occupation column
            min_count: Minimum count to include occupation
        
        Returns:
            DataFrame with top majors by occupation
        """
        print("\nAnalyzing majors by occupation...")
        
        # Filter for records with both occupation and major
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
    
    def load_wage_data(self):
        """
        Load wage data
        
        Returns:
            Wage dataframe
        """
        print("Loading wage data...")
        wage_df = pd.read_csv(os.path.join(self.data_dir, 'wages.csv'))
        print(f"Total wage records: {len(wage_df):,}")
        return wage_df


# ========================================
# LEGACY FUNCTIONS (for backward compatibility)
# ========================================

def load_job_data():
    """Legacy function - use DataLoader class instead"""
    loader = DataLoader()
    return loader.load_job_data()


def get_window_users(job_df, windows):
    """Legacy function - use DataLoader class instead"""
    loader = DataLoader()
    return loader.get_window_users(job_df, windows)


def get_occupation_distributions(window_users, windows, occupation_col):
    """Legacy function - use DataLoader class instead"""
    loader = DataLoader()
    return loader.get_occupation_distributions(window_users, windows, occupation_col)
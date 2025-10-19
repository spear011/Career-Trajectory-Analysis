"""
Data loading and preprocessing for labor market analysis
"""
import pandas as pd
import os
from .config import DATA_DIR
from .utils import get_active_jobs, get_latest_job_per_user


def load_job_data():
    """
    Load and preprocess job data
    
    Returns:
        Preprocessed job dataframe
    """
    print("Loading job data...")
    job_df = pd.read_csv(os.path.join(DATA_DIR, 'job_group_0.csv'))
    
    # Convert date columns
    job_df['JOB_START_DATE'] = pd.to_datetime(job_df['JOB_START_DATE'], format='%Y-%m', errors='coerce')
    job_df['JOB_END_DATE'] = pd.to_datetime(job_df['JOB_END_DATE'], format='%Y-%m', errors='coerce')
    
    print(f"Total job records: {len(job_df):,}")
    return job_df


def get_window_users(job_df, windows):
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


def get_occupation_distributions(window_users, windows, occupation_col):
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
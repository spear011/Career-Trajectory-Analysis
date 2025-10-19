"""
Utility functions for data processing
"""
import os

def get_active_jobs(df, period_start, period_end):
    """
    Get jobs that were active during the specified period
    
    Args:
        df: Job dataframe
        period_start: Start date of period
        period_end: End date of period
    
    Returns:
        Filtered dataframe with active jobs
    """
    active = df[
        (df['JOB_START_DATE'] <= period_end) &
        ((df['JOB_END_DATE'] >= period_start) | (df['JOB_END_DATE'].isna()))
    ].copy()
    return active


def get_latest_job_per_user(df):
    """
    For each user, get their most recent job (by start date)
    
    Args:
        df: Job dataframe
    
    Returns:
        Dataframe with one row per user (their latest job)
    """
    df_sorted = df.sort_values('JOB_START_DATE', ascending=False)
    return df_sorted.groupby('ID').first().reset_index()


def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)
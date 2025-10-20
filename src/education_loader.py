"""
Education data loading and preprocessing
"""
import pandas as pd
import os
from .config import DATA_DIR


def load_education_data():
    """
    Load and preprocess education data
    
    Returns:
        Preprocessed education dataframe
    """
    print("Loading education data...")
    ed_df = pd.read_csv(os.path.join(DATA_DIR, 'education_group_0.csv'))
    
    # Convert date columns
    ed_df['START_DATE'] = pd.to_datetime(ed_df['START_DATE'], format='%Y-%m', errors='coerce')
    ed_df['END_DATE'] = pd.to_datetime(ed_df['END_DATE'], format='%Y-%m', errors='coerce')
    
    # Handle graduation year
    ed_df['GRAD_YEAR'] = pd.to_numeric(ed_df['GRAD_YEAR'], errors='coerce')
    
    print(f"Total education records: {len(ed_df):,}")
    print(f"Unique individuals: {ed_df['ID'].nunique():,}")
    
    return ed_df


def get_education_level_distribution(ed_df):
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


def get_major_distribution(ed_df, top_n=20):
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


def get_user_highest_education(ed_df):
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


def merge_job_education(job_df, ed_df):
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
    highest_ed = get_user_highest_education(ed_df)
    
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


def analyze_education_by_occupation(merged_df, occupation_col='SOC_EMSI_2019_3_NAME'):
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


def analyze_major_by_occupation(merged_df, occupation_col='SOC_EMSI_2019_3_NAME', min_count=50):
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
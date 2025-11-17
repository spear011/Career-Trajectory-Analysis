"""
Data Loader Module
Loads preprocessed trajectory data for analysis
"""
import os
import pandas as pd

class DataLoader:
    """
    Data loader for preprocessed career trajectory data
    """
    
    def __init__(self, config):
        """
        Initialize data loader
        
        Args:
            config: Config instance
        """
        self.config = config
    
    def load_preprocessed_trajectories(self) -> pd.DataFrame:
        """
        Load preprocessed career trajectory data
        
        Returns:
            Trajectory dataframe
        """
        print("\n" + "="*80)
        print("LOADING PREPROCESSED TRAJECTORY DATA")
        print("="*80)
        
        trajectory_path = os.path.join(self.config.results_dir, 'career_trajectories.parquet')
        
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(
                f"Preprocessed trajectory file not found: {trajectory_path}\n"
                f"Please run preprocess.py first to generate trajectory data."
            )
        
        print(f"Loading from: {trajectory_path}")
        df = pd.read_parquet(trajectory_path)
        
        print(f"Loaded {len(df):,} trajectory records")
        print(f"Unique users: {df['ID'].nunique():,}")
        print(f"Year range: {df['job_start_year'].min():.0f}-{df['job_start_year'].max():.0f}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        print("="*80 + "\n")
        
        return df
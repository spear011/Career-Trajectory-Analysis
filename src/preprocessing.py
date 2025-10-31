"""
Data Preprocessing Pipeline
Integrates Dataset Construction notebook pipeline from the original research
MODIFIED: flatten_to_trajectory_df now creates one row per job instead of one row per user
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from tabulate import tabulate


class DataPreprocessor:
    """
    Complete preprocessing pipeline following Dataset Construction notebook
    Converts raw Lightcast data into clean trajectory dataset
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize preprocessor
        
        Args:
            verbose: Print detailed progress information
        """
        self.verbose = verbose
        self.job_df = None
        self.edu_df = None
        self.linear_job_df = None
        self.trajectory_df = None
        
    def _log(self, message: str):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    # ========================================
    # STEP 0: Load and Preprocess Raw Data
    # ========================================
    
    def load_data(self, job_path: str, edu_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load job and education data files
        
        Args:
            job_path: Path to job CSV file
            edu_path: Path to education CSV file
        
        Returns:
            Tuple of (job_df, edu_df)
        """
        self._log("="*80)
        self._log("STEP 0: Loading Raw Data")
        self._log("="*80)
        
        # Load datasets
        self.job_df = pd.read_csv(job_path, encoding="utf-8")
        self.edu_df = pd.read_csv(edu_path, encoding="utf-8")
        
        self._log(f"Job records loaded: {len(self.job_df):,}")
        self._log(f"Education records loaded: {len(self.edu_df):,}")
        
        # Convert date columns
        self._ensure_datetime(self.job_df, ['JOB_START_DATE', 'JOB_END_DATE'])
        self._ensure_datetime(self.edu_df, ['START_DATE', 'END_DATE'])
        
        return self.job_df, self.edu_df
    
    def _ensure_datetime(self, df: pd.DataFrame, cols: list):
        """Convert specified columns to datetime if not already"""
        for col in cols:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # ========================================
    # STEP 1: Job Records Filtering
    # ========================================
    
    def filter_job_records(self):
        """
        Filter job records
        - Keep jobs with valid titles, company, city, state, and country
        - Filter for valid start/end dates
        - Sort jobs chronologically within each individual
        """
        self._log("\n" + "="*80)
        self._log("STEP 1: Job Records Filtering")
        self._log("="*80)
        
        initial_count = len(self.job_df)
        initial_users = self.job_df['ID'].nunique()
        
        # Filter required fields
        self.job_df = self.job_df[
            (~self.job_df['TITLE_RAW'].isna()) &
            (~self.job_df['COMPANY_RAW'].isna()) &
            (~self.job_df['CITY_RAW'].isna()) &
            (~self.job_df['STATE_RAW'].isna()) &
            (~self.job_df['COUNTRY_RAW'].isna()) &
            (~self.job_df['JOB_START_DATE'].isna())
        ].copy()
        
        # Sort jobs
        self.job_df.sort_values(
            ['ID', 'JOB_START_DATE', 'JOB_END_DATE'], 
            ascending=[True, True, False], 
            inplace=True
        )
        
        self._log(f"Records: {initial_count:,} → {len(self.job_df):,}")
        self._log(f"Users: {initial_users:,} → {self.job_df['ID'].nunique():,}")
    
    # ========================================
    # STEP 2: Merge Occupation Codes
    # ========================================
    
    def merge_occupation_codes(self, occ_path: Optional[str] = None):
        """
        Merge occupation codes from predictions file
        
        Args:
            occ_path: Path to occupation predictions CSV (optional)
        """
        if occ_path is None:
            self._log("\n" + "="*80)
            self._log("STEP 2: Skipping occupation code merge (no file provided)")
            self._log("="*80)
            return
        
        self._log("\n" + "="*80)
        self._log("STEP 2: Merging Occupation Codes")
        self._log("="*80)
        
        occ_df = pd.read_csv(occ_path, encoding="utf-8")
        
        # Merge occupation predictions into job_df
        if 'ONET_2019' in occ_df.columns and 'ONET_2019_NAME' in occ_df.columns:
            # Merge on ID and job title
            self.job_df = self.job_df.merge(
                occ_df[['ID', 'TITLE_RAW', 'ONET_2019', 'ONET_2019_NAME', 'NAICS6', 'COMPANY_NAME']],
                on=['ID', 'TITLE_RAW'],
                how='left'
            )
            self._log(f"Occupation codes merged. Records with ONET: {self.job_df['ONET_2019'].notna().sum():,}")
        else:
            self._log("Warning: Expected columns not found in occupation file")
    
    # ========================================
    # STEP 3: Education Records Filtering
    # ========================================
    
    def filter_education_records(self):
        """
        Filter education records
        - Retain only Bachelor's and higher degrees
        - Drop records with missing fields
        - Sort degrees chronologically
        """
        self._log("\n" + "="*80)
        self._log("STEP 3: Education Records Filtering")
        self._log("="*80)
        
        initial_count = len(self.edu_df)
        initial_users = self.edu_df['ID'].nunique()
        
        # Filter required fields
        self.edu_df = self.edu_df[
            (~self.edu_df['EDUCATION_RAW'].isna()) &
            (~self.edu_df['SCHOOL_RAW'].isna()) &
            (~self.edu_df['START_DATE'].isna()) &
            (~self.edu_df['END_DATE'].isna())
        ].copy()
        
        # Keep only BA and higher
        valid_degrees = ["Bachelor's Degree", "Master's Degree", "Doctorate"]
        self.edu_df = self.edu_df[self.edu_df['EDULEVEL_NAME'].isin(valid_degrees)]
        
        # Sort education by individual and chronological order
        self.edu_df.sort_values(
            ['ID', 'START_DATE', 'END_DATE'], 
            ascending=[True, True, True], 
            inplace=True
        )
        
        self._log(f"Records: {initial_count:,} → {len(self.edu_df):,}")
        self._log(f"Users: {initial_users:,} → {self.edu_df['ID'].nunique():,}")
    
    # ========================================
    # STEP 3b: User Intersection
    # ========================================
    
    def filter_user_intersection(self):
        """Keep only users that appear in both job and education datasets"""
        self._log("\n" + "="*80)
        self._log("STEP 3b: User Intersection Filtering")
        self._log("="*80)
        
        job_users = set(self.job_df['ID'].unique())
        edu_users = set(self.edu_df['ID'].unique())
        valid_users = job_users & edu_users
        
        self._log(f"Job users: {len(job_users):,}")
        self._log(f"Education users: {len(edu_users):,}")
        self._log(f"Intersection: {len(valid_users):,}")
        
        self.job_df = self.job_df[self.job_df['ID'].isin(valid_users)]
        self.edu_df = self.edu_df[self.edu_df['ID'].isin(valid_users)]
    
    # ========================================
    # STEP 4: Post-Graduation Gap Filtering
    # ========================================
    
    def filter_post_graduation_gap(self):
        """
        Filter users by post-graduation gap
        - Compute gap between BA graduation and first job
        - Drop users exceeding maximum allowable gap
        
        Args:
            max_gap_years: Maximum gap in years (default: 3)
        """
        self._log("\n" + "="*80)
        self._log(f"STEP 4: Post-Graduation Gap Filtering")
        self._log("="*80)

        job_df = self.job_df.copy()
        edu_df = self.edu_df.copy()
        initial_users = job_df['ID'].nunique()
        
        # Sort education table deterministically
        edu_df = edu_df.sort_values(['ID', 'EDULEVEL_NAME', 'END_DATE'], 
                                    key=lambda col: col.map({"Bachelor's Degree":1, "Master's Degree":2, "Doctorate":3}) 
                                                    if col.name=='EDULEVEL_NAME' else col)

        # Compute BA graduation date per user
        ba_df = edu_df[edu_df['EDULEVEL_NAME'] == "Bachelor's Degree"]
        ba_grad_dates = ba_df.groupby('ID')['END_DATE'].min().rename('BA_GRAD_DATE')

        # First job start date
        first_job_dates = job_df.groupby('ID')['JOB_START_DATE'].min().rename('FIRST_JOB_DATE')

        # Highest degree per user
        degree_order = {"Bachelor's Degree": 1, "Master's Degree": 2, "Doctorate": 3}
        edu_df['DEGREE_ORDER'] = edu_df['EDULEVEL_NAME'].map(degree_order)
        # Take the row with max degree order per user
        idx = edu_df.groupby('ID')['DEGREE_ORDER'].idxmax()
        highest_degree = edu_df.loc[idx, ['ID','EDULEVEL_NAME']].set_index('ID')['EDULEVEL_NAME'].rename('HIGHEST_DEGREE')

        # Combine into gap dataframe
        gap_df = pd.concat([ba_grad_dates, first_job_dates, highest_degree], axis=1).dropna()

        # Remove negative gaps (first job before BA graduation)
        gap_df = gap_df[gap_df['FIRST_JOB_DATE'] >= gap_df['BA_GRAD_DATE']]

        # Compute post-grad gap in years (rounded to 4 decimals)
        gap_df['POST_GRAD_GAP_YEARS'] = ((gap_df['FIRST_JOB_DATE'] - gap_df['BA_GRAD_DATE']).dt.days / 365.25).round(4)

        # pply maximum allowed gap thresholds according to highest degree
        gap_thresholds = {"Bachelor's Degree": 3.75, "Master's Degree": 5.59, "Doctorate": 8.25}
        valid_ids_gap = gap_df[gap_df['POST_GRAD_GAP_YEARS'] <= gap_df['HIGHEST_DEGREE'].map(gap_thresholds)].index

        # Apply filter to main tables
        self.job_df = job_df[job_df['ID'].isin(valid_ids_gap)].copy()
        self.edu_df = edu_df[edu_df['ID'].isin(valid_ids_gap)].copy()
        valid_ids = set(self.job_df['ID'].unique())
        
        self._log(f"Users: {initial_users:,} → {len(valid_ids):,}")
        self._log(f"Dropped: {initial_users - len(valid_ids):,} users")
    
    # ========================================
    # STEP 5: Timeframe Filtering
    # ========================================
    
    def filter_timeframe(self, start_year: int = 1999, end_year: int = 2022):
        """
        Filter users by timeframe
        - First job year >= start_year
        - Last job year <= end_year
        
        Args:
            start_year: Minimum first job year (default: 1999)
            end_year: Maximum last job year (default: 2022)
        """
        self._log("\n" + "="*80)
        self._log(f"STEP 5: Timeframe Filtering ({start_year} - {end_year})")
        self._log("="*80)
        
        # Compute first and last job years per user
        user_years = self.job_df.groupby('ID').agg({
            'JOB_START_DATE': ['min', 'max']
        }).reset_index()
        user_years.columns = ['ID', 'first_job_date', 'last_job_date']
        user_years['first_job_year'] = user_years['first_job_date'].dt.year
        user_years['last_job_year'] = user_years['last_job_date'].dt.year
        
        # Filter valid users
        valid_ids_time = user_years[
            (user_years['first_job_year'] >= start_year) &
            (user_years['last_job_year'] <= end_year)
        ]['ID'].unique()
        
        initial_users = self.job_df['ID'].nunique()
        
        # Apply filter to all tables
        self.job_df = self.job_df[self.job_df['ID'].isin(valid_ids_time)].copy()
        self.edu_df = self.edu_df[self.edu_df['ID'].isin(valid_ids_time)].copy()
        
        self._log(f"Users: {initial_users:,} → {len(valid_ids_time):,}")
    
    # ========================================
    # STEP 6: Construct Linear Career Trajectories
    # ========================================
    
    def construct_linear_trajectories(self, trajectory_years: int = 5):
        """
        Construct linear career trajectories per user
        - Remove overlapping jobs
        - Sort by (start_date asc, end_date desc)
        - Truncate to specified years after BA graduation
        
        Args:
            trajectory_years: Years to keep in trajectory
        """
        self._log("\n" + "="*80)
        self._log(f"STEP 6: Constructing Linear Trajectories ({trajectory_years} years)")
        self._log("="*80)
        
        df = self.job_df.copy()
        df = df.sort_values(
            by=['ID', 'JOB_START_DATE', 'JOB_END_DATE'], 
            ascending=[True, True, False]
        )
        
        linear_jobs = []
        
        for uid, group in df.groupby('ID'):
            group = group.reset_index(drop=True)
            if len(group) == 0:
                continue
            
            # Build non-overlapping trajectory
            trajectory = [group.iloc[0]]
            
            for i in range(1, len(group)):
                last_job = trajectory[-1]
                current_job = group.iloc[i]
                
                if pd.notna(current_job['JOB_START_DATE']) and pd.notna(last_job['JOB_END_DATE']):
                    if current_job['JOB_START_DATE'] >= last_job['JOB_END_DATE']:
                        trajectory.append(current_job)
            
            if len(trajectory) == 0:
                continue
            
            traj_df = pd.DataFrame(trajectory)
            traj_df['TRAJECTORY_ORDER'] = range(1, len(traj_df) + 1)
            
            # Compute trajectory duration
            traj_start = traj_df['JOB_START_DATE'].min()
            traj_end = traj_df['JOB_END_DATE'].max()
            if pd.isna(traj_end):
                traj_end = traj_df['JOB_START_DATE'].max()
            duration_years = (traj_end - traj_start).days / 365.25
            
            # Truncate to first N years from trajectory start
            cutoff_date = traj_start + pd.DateOffset(years=trajectory_years)
            traj_df = traj_df[traj_df['JOB_START_DATE'] <= cutoff_date]
            
            linear_jobs.append(traj_df)
        
        self.linear_job_df = pd.concat(linear_jobs, ignore_index=True)
        
        self._log(f"Original job records: {len(self.job_df):,}")
        self._log(f"Linear trajectory records: {len(self.linear_job_df):,}")
        self._log(f"Users with valid trajectories: {self.linear_job_df['ID'].nunique():,}")
    
    # ========================================
    # STEP 7: Flatten to Trajectory DataFrame
    # ========================================
    
    def flatten_to_trajectory_df(self):
        """
        Flatten linear job history to one row per job
        Each row represents a single job with enrichment information
        """
        self._log("\n" + "="*80)
        self._log("STEP 7: Flattening to Trajectory DataFrame (One Row Per Job)")
        self._log("="*80)
        
        trajectory_records = []
        degree_order = {"Bachelor's Degree": 1, "Master's Degree": 2, "Doctorate": 3}
        
        for uid, group in self.linear_job_df.groupby('ID'):
            group = group.sort_values(by='TRAJECTORY_ORDER')
            if len(group) == 0:
                continue
            
            # Calculate trajectory-level information
            last_job_end = group['JOB_END_DATE'].max()
            if pd.isna(last_job_end):
                last_job_end = group['JOB_START_DATE'].max()
            
            # Get maximum degree within trajectory period
            edu_sub = self.edu_df[
                (self.edu_df['ID'] == uid) & 
                (self.edu_df['END_DATE'] <= last_job_end)
            ]
            
            if len(edu_sub) > 0:
                edu_sub = edu_sub.copy()
                edu_sub['degree_num'] = edu_sub['EDULEVEL_NAME'].map(degree_order).fillna(0)
                max_row = edu_sub.sort_values(
                    ['degree_num', 'END_DATE'], 
                    ascending=[False, False]
                ).iloc[0]
                max_edu_name = max_row['EDULEVEL_NAME']
            else:
                max_edu_name = None
            
            # Create one row per job
            for idx, job_row in group.iterrows():
                onet_major = str(job_row['ONET_2019'])[:2] if pd.notna(job_row.get('ONET_2019')) else None
                naics6_major = str(job_row['NAICS6'])[:2] if pd.notna(job_row.get('NAICS6')) else None
                onet_detailed = str(job_row['ONET_2019'])[:7] if pd.notna(job_row.get('ONET_2019')) else None
                company = job_row.get('COMPANY_NAME', None)
                state = job_row.get('STATE_RAW', None)
                job_start_date = job_row['JOB_START_DATE']
                job_end_date = job_row['JOB_END_DATE']
                job_start_year = job_start_date.year if pd.notna(job_start_date) else None
                job_end_year = job_end_date.year if pd.notna(job_end_date) else None
                
                trajectory_records.append({
                    'ID': uid,
                    'max_edu_name': max_edu_name,
                    'onet_major': onet_major,
                    'onet_detailed': onet_detailed,
                    'naics6_major': naics6_major,
                    'company': company,
                    'state': state,
                    'job_start_date': job_start_date,
                    'job_end_date': job_end_date if pd.notna(job_end_date) else None,
                    'job_start_year': job_start_year,
                    'job_end_year': job_end_year,
                    'trajectory_order': job_row.get('TRAJECTORY_ORDER', None)
                })
        
        self.trajectory_df = pd.DataFrame(trajectory_records)
        
        self._log(f"Trajectory records created: {len(self.trajectory_df):,}")
        self._log(f"Users: {self.trajectory_df['ID'].nunique():,}")
        self._log(f"Jobs per user (mean): {len(self.trajectory_df) / self.trajectory_df['ID'].nunique():.2f}")
    
    # ========================================
    # VALIDATION & DIAGNOSTICS
    # ========================================
    
    def run_diagnostics(self):
        """Run diagnostic checks on current state of data"""
        self._log("\n" + "="*80)
        self._log("RUNNING DIAGNOSTICS")
        self._log("="*80)
        
        if self.job_df is not None:
            self._log(f"\nJob DataFrame:")
            self._log(f"  Records: {len(self.job_df):,}")
            self._log(f"  Users: {self.job_df['ID'].nunique():,}")
            
        if self.edu_df is not None:
            self._log(f"\nEducation DataFrame:")
            self._log(f"  Records: {len(self.edu_df):,}")
            self._log(f"  Users: {self.edu_df['ID'].nunique():,}")
            
        if self.linear_job_df is not None:
            self._log(f"\nLinear Job DataFrame:")
            self._log(f"  Records: {len(self.linear_job_df):,}")
            self._log(f"  Users: {self.linear_job_df['ID'].nunique():,}")
            
        if self.trajectory_df is not None:
            self._log(f"\nTrajectory DataFrame:")
            self._log(f"  Records: {len(self.trajectory_df):,}")
            self._log(f"  Users: {self.trajectory_df['ID'].nunique():,}")
    
    # ========================================
    # PIPELINE EXECUTION
    # ========================================
    
    def run_full_pipeline(
        self, 
        job_path: str, 
        edu_path: str,
        occ_path: Optional[str] = None,
        start_year: int = 1999,
        end_year: int = 2022,
        trajectory_years: int = 5
    ) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline
        
        Args:
            job_path: Path to job CSV
            edu_path: Path to education CSV
            occ_path: Path to occupation predictions CSV (optional)
            start_year: Minimum first job year
            end_year: Maximum last job year
            trajectory_years: Years to keep in trajectory
        
        Returns:
            Processed trajectory dataframe
        """
        # Step 0: Load data
        self.load_data(job_path, edu_path)
        
        # Step 1: Filter job records
        self.filter_job_records()
        
        # Step 2: Merge occupation codes (optional)
        self.merge_occupation_codes(occ_path)
        
        # Step 3: Filter education records
        self.filter_education_records()
        
        # Step 3b: User intersection
        self.filter_user_intersection()
        
        # Step 4: Post-graduation gap filtering
        self.filter_post_graduation_gap()
        
        # Step 5: Timeframe filtering
        self.filter_timeframe(start_year, end_year)
        
        # Step 6: Construct linear trajectories
        self.construct_linear_trajectories(trajectory_years)
        
        # Step 7: Flatten to trajectory df
        self.flatten_to_trajectory_df()
        
        # Final diagnostics
        self.run_diagnostics()
        
        self._log("\n" + "="*80)
        self._log("PREPROCESSING PIPELINE COMPLETE")
        self._log("="*80)
        
        return self.trajectory_df
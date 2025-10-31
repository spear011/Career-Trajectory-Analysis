"""
Data Enrichment Module
Add demographic attributes, wages, and derived features to trajectory data
MODIFIED: Updated to work with new preprocessing output (one row per job)
"""
import re
import pandas as pd
import numpy as np
from typing import Optional


class TrajectoryEnricher:
    """
    Enrich trajectory data with contextual variables
    - Demographic attributes (gender, race)
    - Birth year and generation cohort
    - State GDP and decile
    - Occupational wages
    - Job change type indicators
    - Upward mobility indicators
    """
    
    def __init__(self, trajectory_df: pd.DataFrame, linear_job_df: pd.DataFrame):
        """
        Initialize enricher
        
        Args:
            trajectory_df: Flattened trajectory dataframe (one row per job)
            linear_job_df: Linear job history dataframe
        """
        self.trajectory_df = trajectory_df.copy()
        self.linear_job_df = linear_job_df.copy()
    
    # ========================================
    # DEMOGRAPHIC ATTRIBUTES
    # ========================================
    
    def add_demographics(self, attribute_df: pd.DataFrame):
        """
        Add demographic attributes (gender, race)
        
        Args:
            attribute_df: DataFrame with demographic predictions
                          Must have columns: ID, gender (1=Male, 2=Female),
                          L1-L4 (race indicators: 1=White, 2=Black, 3=Asian, 4=Hispanic)
        """
        print("Adding demographic attributes...")
        
        attribute_df = attribute_df.copy()
        attribute_df['race'] = attribute_df.apply(self._choose_race, axis=1)
        attribute_df.rename(columns={'L5': 'gender'}, inplace=True)
        attribute_df = attribute_df.query('~race.isna() and gender > 0 and gender < 3')[['ID', 'race', 'gender']]

        self.trajectory_df = self.trajectory_df.merge(attribute_df, on='ID', how='left')
        
        print(f"  Demographics added. Missing gender: {self.trajectory_df['gender'].isna().sum()}")
        if 'race' in self.trajectory_df.columns:
            print(f"  Missing race: {self.trajectory_df['race'].isna().sum()}")
    
    @staticmethod
    def _choose_race(row):
        """
        Choose race from L1-L4 indicators
        1=White, 2=Black, 3=Asian, 4=Hispanic
        """
        if row['L1'] == 1:
            return 1
        elif row['L2'] == 1:
            return 2
        elif row['L3'] == 1:
            return 3
        elif row['L4'] == 1:
            return 4
        else:
            return None
    
    # ========================================
    # BIRTH YEAR & GENERATION
    # ========================================
    
    def estimate_birth_year(self, edu_df: pd.DataFrame):
        """
        Estimate birth year from BA graduation date
        Assumes typical BA graduation age is 22
        
        Args:
            edu_df: Education dataframe with END_DATE
        """
        print("Estimating birth year and generation...")
        
        # Get BA graduation dates
        ba_degrees = edu_df[
            edu_df['EDULEVEL_NAME'] == "Bachelor's Degree"
        ].sort_values('END_DATE')
        ba_grad = ba_degrees.groupby('ID')['END_DATE'].first()
        
        # Estimate birth year
        birth_year = ba_grad.apply(lambda x: x.year - 22 if pd.notna(x) else None)
        
        # Assign generation cohort
        generation = birth_year.apply(self._assign_generation)
        
        # Merge into trajectory df
        birth_gen_df = pd.DataFrame({
            'ID': birth_year.index,
            'birth_year': birth_year.values,
            'generation': generation.values
        })
        
        self.trajectory_df = self.trajectory_df.merge(birth_gen_df, on='ID', how='left')
        
        print(f"  Birth year estimated. Missing: {self.trajectory_df['birth_year'].isna().sum()}")
    
    @staticmethod
    def _assign_generation(birth_year):
        """Assign generation cohort based on birth year"""
        if pd.isna(birth_year):
            return None
        if birth_year < 1946:
            return "Silent Generation"
        elif 1946 <= birth_year <= 1964:
            return "Baby Boomer"
        elif 1965 <= birth_year <= 1980:
            return "Generation X"
        elif 1981 <= birth_year <= 1996:
            return "Millennial"
        elif birth_year >= 1997:
            return "Generation Z"
        else:
            return None
    
    # ========================================
    # STATE GDP
    # ========================================

    def normalize_state_name(self, name):
        if pd.isna(name):
            return None
        name = name.lower().strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'[^\w\s]', '', name)
        return name
    
    def add_state_gdp(self, gdp_df: pd.DataFrame):
        """
        Add state GDP at job start year and compute GDP decile
        
        Args:
            gdp_df: GDP dataframe with columns GeoName and year columns
        """
        print("Adding state GDP and decile...")
        
        trajectory_df = self.trajectory_df.copy()
        
        # Extract year from job_start_date
        trajectory_df['job_start_year'] = trajectory_df['job_start_date'].dt.year

        # Melt GDP to long format
        gdp_long = gdp_df.melt(id_vars=['GeoName'], var_name='Year', value_name='GDP')
        gdp_long = gdp_long[gdp_long['Year'].str.isdigit()]
        gdp_long['Year'] = gdp_long['Year'].astype(int)

        # Normalize state names
        gdp_long['GeoName_norm'] = gdp_long['GeoName'].apply(self.normalize_state_name)
        trajectory_df['state_norm'] = trajectory_df['state'].apply(self.normalize_state_name)

        # Build lookup dict
        state_year_to_gdp = {
            (row['GeoName_norm'], row['Year']): row['GDP']
            for _, row in gdp_long.iterrows()
        }

        # Map GDP to each row
        def lookup_gdp(row):
            return state_year_to_gdp.get((row['state_norm'], row['job_start_year']), None)

        trajectory_df['state_gdp'] = trajectory_df.apply(lookup_gdp, axis=1)

        # Compute deciles
        trajectory_df['state_gdp_decile'] = pd.qcut(
            trajectory_df['state_gdp'], 10, labels=False, duplicates='drop'
        ) + 1

        # Drop temporary columns (keep job_start_year for final output)
        trajectory_df.drop(columns=['state_norm', 'state_gdp'], inplace=True)
        self.trajectory_df = trajectory_df
        
        print(f"  State GDP decile added. Missing: {self.trajectory_df['state_gdp_decile'].isna().sum()}")
   
    # ========================================
    # OCCUPATIONAL WAGES
    # ========================================
    
    def add_occupational_wage(self, wage_df: pd.DataFrame):
        """
        Add occupational wage at job start year
        
        Args:
            wage_df: Wage dataframe with AREA_TITLE, year, OCC_CODE, A_MEAN
        """
        print("Adding occupational wages...")
        
        trajectory_df = self.trajectory_df.copy()
        wage_df = wage_df.copy()
        
        # Use existing job_start_year or create it
        if 'job_start_year' not in trajectory_df.columns:
            trajectory_df['job_start_year'] = trajectory_df['job_start_date'].dt.year
        
        # Normalize strings
        wage_df['OCC_CODE'] = wage_df['OCC_CODE'].astype(str).str[:7]
        wage_df['AREA_TITLE'] = wage_df['AREA_TITLE'].str.strip().str.lower()

        # Prepare MultiIndex Series
        wage_series = wage_df.set_index(['AREA_TITLE', 'year', 'OCC_CODE'])['A_MEAN']

        # Normalize trajectory keys
        trajectory_df['state_norm'] = trajectory_df['state'].str.strip().str.lower()
        trajectory_df['onet_norm'] = trajectory_df['onet_detailed'].astype(str).str[:7]

        # Build MultiIndex for lookup
        trajectory_index = pd.MultiIndex.from_arrays([
            trajectory_df['state_norm'],
            trajectory_df['job_start_year'],
            trajectory_df['onet_norm']
        ])

        # Vectorized lookup
        trajectory_df['annual_state_wage'] = wage_series.reindex(trajectory_index).to_numpy()

        # Cleanup (keep job_start_year for final output)
        trajectory_df.drop(columns=['state_norm', 'onet_norm'], inplace=True)
        self.trajectory_df = trajectory_df
        print(f"  Annual wage added. Missing: {self.trajectory_df['annual_state_wage'].isna().sum()}")
    
    # ========================================
    # JOB CHANGE TYPES
    # ========================================
    
    def add_job_change_types(self):
        """
        Add job movement type indicators per user:
        - move_1_1: Different company, different occupation
        - move_1_2: Different company, same occupation  
        - move_2_1: Same company, different occupation
        - move_2_2: Same company, same occupation (promotion)
        """
        print("Adding job change type indicators...")
        
        # Initialize columns
        self.trajectory_df[['move_1_1', 'move_1_2', 'move_2_1', 'move_2_2']] = 0
        
        # Ensure jobs are sorted
        linear_job_df_sorted = self.linear_job_df.sort_values(
            ['ID', 'TRAJECTORY_ORDER']
        ).copy()
        
        # For each user
        for uid, group in linear_job_df_sorted.groupby('ID'):
            if len(group) < 2:
                continue
            
            # Shift columns to compare consecutive jobs
            prev_company = group['COMPANY_NAME'].shift(0)
            next_company = group['COMPANY_NAME'].shift(-1)
            prev_onet = group['ONET_2019'].shift(0)
            next_onet = group['ONET_2019'].shift(-1)
            
            # Determine job change types
            type_1_1 = ((prev_company != next_company) & (prev_onet != next_onet)).any()
            type_1_2 = ((prev_company != next_company) & (prev_onet == next_onet)).any()
            type_2_1 = ((prev_company == next_company) & (prev_onet != next_onet)).any()
            type_2_2 = ((prev_company == next_company) & (prev_onet == next_onet)).any()
            
            # Update trajectory_df
            mask = self.trajectory_df['ID'] == uid
            if type_1_1:
                self.trajectory_df.loc[mask, 'move_1_1'] = 1
            if type_1_2:
                self.trajectory_df.loc[mask, 'move_1_2'] = 1
            if type_2_1:
                self.trajectory_df.loc[mask, 'move_2_1'] = 1
            if type_2_2:
                self.trajectory_df.loc[mask, 'move_2_2'] = 1
        
        print("  Job change types added")
        print(f"    move_1_1: {self.trajectory_df['move_1_1'].sum():,}")
        print(f"    move_1_2: {self.trajectory_df['move_1_2'].sum():,}")
        print(f"    move_2_1: {self.trajectory_df['move_2_1'].sum():,}")
        print(f"    move_2_2: {self.trajectory_df['move_2_2'].sum():,}")
    
    # ========================================
    # UPWARD MOBILITY
    # ========================================
    
    def add_upward_mobility(self, wage_df: pd.DataFrame, threshold: float = 0.05):
        """
        Add upward mobility indicator
        Compares last job wage to first job wage per user
        
        Args:
            wage_df: Wage dataframe
            threshold: Wage increase threshold (default 5%)
        """
        print(f"Adding upward mobility indicator (threshold={threshold})...")
        
        wage_df = wage_df.copy()
        wage_df['OCC_CODE'] = wage_df['OCC_CODE'].astype(str)
        wage_df['AREA_TITLE'] = wage_df['AREA_TITLE'].str.strip().str.lower()
        wage_df = wage_df[['AREA_TITLE', 'year', 'OCC_CODE', 'A_MEAN']]
        
        # Get first and last job per user
        self.trajectory_df['ID'] = self.trajectory_df['ID'].astype(str).str.strip()
        self.linear_job_df['ID'] = self.linear_job_df['ID'].astype(str).str.strip()
        
        first_jobs = self.linear_job_df.sort_values(
            ['ID', 'TRAJECTORY_ORDER']
        ).groupby('ID').first().reset_index()
        
        last_jobs = self.linear_job_df.sort_values(
            ['ID', 'TRAJECTORY_ORDER']
        ).groupby('ID').last().reset_index()
        
        # Get wages for first jobs (already in trajectory_df as 'annual_state_wage')
        first_wage = self.trajectory_df.sort_values(
            ['ID', 'trajectory_order']
        ).groupby('ID').first()[['annual_state_wage']].reset_index()
        first_wage.rename(columns={'annual_state_wage': 'first_job_wage'}, inplace=True)
        
        # Get wages for last jobs
        last_jobs['state_norm'] = last_jobs['STATE_RAW'].astype(str).str.strip().str.lower()
        last_jobs['onet_norm'] = last_jobs['ONET_2019'].astype(str).str[:7]
        last_jobs['year'] = last_jobs['JOB_START_DATE'].dt.year
        
        # Merge with wage table
        last_jobs = last_jobs.merge(
            wage_df.rename(columns={
                'AREA_TITLE': 'state_norm',
                'OCC_CODE': 'onet_norm',
                'A_MEAN': 'last_job_wage'
            }),
            on=['state_norm', 'year', 'onet_norm'],
            how='left'
        )
        last_jobs = last_jobs[['ID', 'last_job_wage']]
        
        # Merge first and last wages
        mobility = first_wage.merge(last_jobs, on='ID', how='left')
        
        # Compute up_move indicator
        mobility['up_move'] = (
            (mobility['last_job_wage'] - mobility['first_job_wage'])
            / mobility['first_job_wage'] > threshold
        ).astype(int)
        
        # Merge into trajectory df
        self.trajectory_df = self.trajectory_df.merge(
            mobility[['ID', 'up_move']], 
            on='ID', 
            how='left'
        )
        
        matched = last_jobs['last_job_wage'].notna().sum()
        total = len(last_jobs)
        print(f"  Last job wage matched: {matched}/{total} ({matched/total*100:.1f}%)")
        print(f"  Upward mobility cases: {self.trajectory_df['up_move'].sum():,}")
    
    # ========================================
    # FINAL CLEANUP
    # ========================================
    
    def final_cleanup(self, top_code_percentile: int = 95):
        """
        Final data cleaning steps
        - Drop rows with null values
        - Log transform wages
        - Drop unnecessary columns
        
        Args:
            top_code_percentile: Percentile for top-coding (not used in per-job format)
        """
        print("Final cleanup...")
        
        initial_rows = len(self.trajectory_df)
        
        # Drop rows with null values
        self.trajectory_df = self.trajectory_df.dropna().reset_index(drop=True)
        
        print(f"  Rows: {initial_rows:,} → {len(self.trajectory_df):,}")
        print(f"  Dropped {initial_rows - len(self.trajectory_df):,} rows with missing values")
        
        # Log transform wage
        if 'annual_state_wage' in self.trajectory_df.columns:
            self.trajectory_df['log_wage'] = np.log(
                self.trajectory_df['annual_state_wage']
            )
            print(f"  Log wage computed")
    
    # ========================================
    # EXPORT
    # ========================================
    
    def save(self, output_path: str):
        """
        Save enriched trajectory data
        
        Args:
            output_path: Output file path (supports .parquet, .csv)
        """
        print(f"\nSaving enriched trajectory data...")
        
        if output_path.endswith('.parquet'):
            self.trajectory_df.to_parquet(output_path, index=False)
        elif output_path.endswith('.csv'):
            self.trajectory_df.to_csv(output_path, index=False)
        else:
            raise ValueError("Output format must be .parquet or .csv")
        
        print(f"  Saved to: {output_path}")
        print(f"  Shape: {self.trajectory_df.shape}")
    
    def get_trajectory_df(self) -> pd.DataFrame:
        """Return enriched trajectory dataframe"""
        return self.trajectory_df
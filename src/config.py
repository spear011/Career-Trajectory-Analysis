"""
Configuration settings for labor market trend analysis
"""
from datetime import datetime

# ============================================
# Paths
# ============================================
DATA_DIR = 'data'
RESULTS_DIR = 'results'
NETWORK_OUTPUT_DIR = 'results/network_output'

# ============================================
# Date ranges
# ============================================
COVID_START_DATE = datetime(2020, 3, 11)

# Yearly windows from 2000 to 2023
YEARS = list(range(2000, 2024))

def get_windows():
    """Generate yearly windows"""
    windows = []
    for year in YEARS:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        windows.append((f'{year}', start, end, str(year)))
    return windows

# ============================================
# Network Analysis Parameters
# ============================================
STUDY_START_YEAR = 2017
STUDY_END_YEAR = 2024

NETWORK_OCCUPATION_COLUMN = 'ONET_2019'
NETWORK_OCCUPATION_NAME_COLUMN = 'ONET_2019_NAME'
NETWORK_TEMPORAL_GRANULARITY = 'annual'

# ============================================
# Analysis parameters
# ============================================
OCCUPATION_COLUMN = 'ONET_2019_NAME'

# Visualization parameters
TOP_N_OCCUPATIONS = 12
COVID_YEAR = 2020

# Color scheme
COLORS = {
    'pre_covid': '#3498db',
    'covid': '#e74c3c',
    'dropout': '#e67e22',
    'permanent_exit': '#c0392b',
    'comeback': '#27ae60',
    'entry': '#16a085',
    'occupation_change': '#3498db',
    'industry_change': '#9b59b6',
}
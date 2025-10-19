"""
Configuration settings for labor market trend analysis
"""
from datetime import datetime

# ============================================
# Paths
# ============================================
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# ============================================
# Date ranges
# ============================================
COVID_START_DATE = datetime(2020, 3, 11)

# Yearly windows from 2000 to 2023 (analyze full range for all sudden changes)
YEARS = list(range(1990, 2024))

def get_windows():
    """Generate yearly windows"""
    windows = []
    for year in YEARS:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        windows.append((f'{year}', start, end, str(year)))
    return windows

# ============================================
# Analysis parameters
# ============================================
# Occupation classification level
OCCUPATION_COLUMN = 'SOC_EMSI_2019_3_NAME'  # Level 3 - broad categories

# Visualization parameters
TOP_N_OCCUPATIONS = 12
COVID_YEAR = 2020  # Marked for reference, but analyzing all years

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
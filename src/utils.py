"""
Utility functions for data processing and configuration management
UPDATED: Added windowed analysis configuration support
"""
import os
import yaml
import torch
import numpy as np
import argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


# ========================================
# Configuration Class
# ========================================

@dataclass
class Config:
    """Configuration class with attribute access"""
    
    # Paths
    onet_dir: str = ""
    data_dir: str = "data"
    enrich_dir: str = "data/enrich"
    results_dir: str = "results"
    network_output_dir: str = "results/network_output"
    
    # Date ranges
    covid_start_date: str = "2020-03-11"
    study_start_year: int = 2017
    study_end_year: int = 2024
    analysis_start_year: int = 2000
    analysis_end_year: int = 2024
    
    # Network analysis
    occupation_column: str = "ONET_2019"
    occupation_name_column: str = "ONET_2019_NAME"
    temporal_granularity: str = "annual"
    
    # Windowed analysis
    window_size: int = 1
    hop_size: int = 1
    windowed_occupation_column: str = "onet_major"
    
    # Analysis parameters
    analysis_occupation_column: str = "ONET_2019_NAME"
    top_n_occupations: int = 12
    covid_year: int = 2020
    
    # Cache
    cache_enabled: bool = True
    cache_dir: str = "results/workforce_flow_cache"
    
    # Raw config dict
    _raw_config: Dict = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_yaml(cls, yaml_path: str = None) -> "Config":
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to YAML config file (default: config.yaml in same dir as utils.py)
        
        Returns:
            Config instance
        """
        if yaml_path is None:
            # Get path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(current_dir, "config.yaml")
        
        with open(yaml_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        config = cls()
        config._raw_config = raw_config
        
        # Flatten nested config into attributes
        config.onet_dir = raw_config['paths']['onet_dir']
        config.data_dir = raw_config['paths']['data_dir']
        config.enrich_dir = raw_config['paths']['enrich_dir']
        config.results_dir = raw_config['paths']['results_dir']
        config.network_output_dir = raw_config['paths']['network_output_dir']
        
        config.covid_start_date = raw_config['date_ranges']['covid_start_date']
        config.study_start_year = raw_config['date_ranges']['study_start_year']
        config.study_end_year = raw_config['date_ranges']['study_end_year']
        config.analysis_start_year = raw_config['date_ranges']['analysis_start_year']
        config.analysis_end_year = raw_config['date_ranges']['analysis_end_year']
        
        config.occupation_column = raw_config['network_analysis']['occupation_column']
        config.occupation_name_column = raw_config['network_analysis']['occupation_name_column']
        config.temporal_granularity = raw_config['network_analysis']['temporal_granularity']
        
        # Load windowed analysis config
        config.window_size = raw_config['windowed_analysis']['window_size']
        config.hop_size = raw_config['windowed_analysis']['hop_size']
        config.windowed_occupation_column = raw_config['windowed_analysis']['occupation_column']
        
        config.analysis_occupation_column = raw_config['analysis_parameters']['occupation_column']
        config.top_n_occupations = raw_config['analysis_parameters']['top_n_occupations']
        config.covid_year = raw_config['analysis_parameters']['covid_year']
        
        config.cache_enabled = raw_config['cache']['enabled']
        config.cache_dir = raw_config['cache']['cache_dir']
        
        return config
    
    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> "Config":
        """
        Create config from command line arguments (with YAML as base)
        
        Args:
            args: Parsed arguments (parses if None)
        
        Returns:
            Config instance
        """
        if args is None:
            args = parse_args()
        
        # Load base config from YAML
        if args.config:
            config = cls.from_yaml(args.config)
        else:
            config = cls.from_yaml()  # Use default path
        
        # Override with command line arguments
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.results_dir:
            config.results_dir = args.results_dir
        if args.study_start_year:
            config.study_start_year = args.study_start_year
        if args.study_end_year:
            config.study_end_year = args.study_end_year
        if hasattr(args, 'window_size') and args.window_size:
            config.window_size = args.window_size
        if hasattr(args, 'hop_size') and args.hop_size:
            config.hop_size = args.hop_size
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested value from raw config using dot notation
        
        Args:
            key_path: Path like 'study_periods.pre_pandemic.name'
            default: Default value if not found
        
        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self._raw_config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_study_periods(self) -> Dict:
        """Get study periods info"""
        return self._raw_config.get('study_periods', {})
    
    def get_visualization_colors(self) -> Dict:
        """Get visualization color scheme"""
        return self._raw_config.get('visualization', {}).get('colors', {})
    
    def get_period_colors(self) -> Dict:
        """Get period-specific colors"""
        return self._raw_config.get('visualization', {}).get('period_colors', {})


# ========================================
# Argument Parser
# ========================================

class Namespace:
    """Simple object for holding attributes"""
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Career Trajectory Analysis Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: config.yaml in src/)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Override data directory'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Override results directory'
    )
    
    parser.add_argument(
        '--study-start-year',
        type=int,
        help='Override study start year'
    )
    
    parser.add_argument(
        '--study-end-year',
        type=int,
        help='Override study end year'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        help='Window size in years for windowed analysis'
    )
    
    parser.add_argument(
        '--hop-size',
        type=int,
        help='Hop size in years for sliding window'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


# ========================================
# Global Config Instance
# ========================================

_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global config instance (loads if not exists)
    
    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config.from_yaml()
    return _global_config


def set_config(config: Config):
    """Set global config instance"""
    global _global_config
    _global_config = config


# ========================================
# File Path Utilities
# ========================================

def create_dirs_if_not_exists(paths):
    """Create directories if they don't exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_dataset_file_path(folder: str, group_num: int) -> str:
    """
    Get path to specific dataset file
    
    Args:
        folder: Folder name (e.g., 'job', 'education', 'skill')
        group_num: Group number
    
    Returns:
        Full path to dataset file
    """
    config = get_config()
    data_dir = config.data_dir
    filename = f"{folder}_group_{group_num}.csv"
    
    return os.path.join(data_dir, folder, filename)


def get_occupation_file_path(group_num: int) -> str:
    """
    Get path to occupation prediction file
    
    Args:
        group_num: Group number
    
    Returns:
        Full path to occupation file
    """
    config = get_config()
    onet_dir = config.onet_dir
    filename = f"lc{group_num}_pred.csv"
    return os.path.join(onet_dir, filename)


def get_enrich_file_path(filename: str) -> str:
    """
    Get path to enrichment data file
    
    Args:
        filename: 'gdp' or 'wage'
    
    Returns:
        Full path to enrichment file
    """
    config = get_config()
    enrich_dir = config.enrich_dir
    
    if filename == 'gdp':
        return os.path.join(enrich_dir, '1998_2022_real_gdp_by_state.csv')
    elif filename == 'wage':
        return os.path.join(enrich_dir, 'wage_interpolated_1999_2022_soc_2019.csv')


def list_available_groups(folder: str) -> List[int]:
    """
    List available group numbers for a given folder
    
    Args:
        folder: Folder name (e.g., 'job', 'education')
    
    Returns:
        List of available group numbers
    """
    config = get_config()
    data_dir = config.data_dir
    folder_path = os.path.join(data_dir, folder)
    
    if not os.path.exists(folder_path):
        return []
    
    # Extract group numbers from filenames
    pattern_prefix = f"{folder}_group_"
    groups = []
    
    for filename in os.listdir(folder_path):
        if filename.startswith(pattern_prefix) and filename.endswith('.csv'):
            try:
                # Extract number between 'group_' and '.csv'
                group_num = int(filename[len(pattern_prefix):-4])
                groups.append(group_num)
            except ValueError:
                continue
    
    return sorted(groups)


# ========================================
# Period Classification
# ========================================

def get_period_for_year(year: int) -> Optional[str]:
    """
    Get study period name for a given year
    
    Args:
        year: Year to classify
    
    Returns:
        Period name or None if year is outside study periods
    """
    config = get_config()
    periods = config.get_study_periods()
    
    for period_key, period_info in periods.items():
        if period_info['start_year'] <= year <= period_info['end_year']:
            return period_info['name']
    
    return None


def get_all_period_info() -> Dict[str, Dict]:
    """
    Get information about all study periods
    
    Returns:
        Dictionary of period information
    """
    config = get_config()
    return config.get_study_periods()
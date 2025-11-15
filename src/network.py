"""
Unified Network Module for Career Trajectories
Builds occupation transition networks from career_trajectories.parquet
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pickle
import json
import os
from pathlib import Path
from collections import defaultdict
from matplotlib.patches import PathPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches


# ============================================
# NETWORK BUILDER
# ============================================

class NetworkBuilder:
    """
    Builds temporal occupation transition networks from career trajectories
    """
    
    def __init__(self, 
                 study_start_year=2017,
                 study_end_year=2024,
                 occupation_col='onet_detailed',
                 occupation_name_col=None):
        """
        Initialize network builder
        
        Args:
            study_start_year: Start year of study period
            study_end_year: End year of study period
            occupation_col: Occupation code column (default: onet_detailed)
            occupation_name_col: Occupation name column (not used for trajectories)
        """
        self.study_start_year = study_start_year
        self.study_end_year = study_end_year
        self.occupation_col = occupation_col
        self.occupation_name_col = occupation_name_col
    
    def prepare_trajectory_data(self, trajectory_df):
        """
        Prepare trajectory data for network construction
        
        Args:
            trajectory_df: Career trajectories dataframe
            
        Returns:
            Prepared dataframe
        """
        print("\nPreparing trajectory data...")
        
        df = trajectory_df.copy()
        
        # Ensure year columns are integers
        if 'job_start_year' not in df.columns:
            df['job_start_year'] = pd.to_datetime(df['job_start_date']).dt.year
        if 'job_end_year' not in df.columns:
            df['job_end_year'] = pd.to_datetime(df['job_end_date']).dt.year
        
        # Filter by study period
        df = df[
            (df['job_start_year'] >= self.study_start_year) & 
            (df['job_start_year'] <= self.study_end_year)
        ].copy()
        
        # Sort by ID and trajectory_order
        df = df.sort_values(['ID', 'trajectory_order']).copy()
        
        print(f"  Valid trajectory records: {len(df):,}")
        print(f"  Unique users: {df['ID'].nunique():,}")
        print(f"  Unique occupations: {df[self.occupation_col].nunique():,}")
        print(f"  Year range: {df['job_start_year'].min()}-{df['job_start_year'].max()}")
        
        return df
    
    def prepare_wage_data(self, wage_df):
        """
        Prepare wage data (kept for compatibility, not required for trajectories)
        
        Args:
            wage_df: Wage dataframe
            
        Returns:
            Prepared wage summary
        """
        print("\nPreparing wage data...")
        
        # Convert OCC_CODE to O*NET format
        wage_df['ONET_CODE'] = wage_df['OCC_CODE'].apply(
            lambda x: f"{x}.00" if pd.notna(x) and '.' not in str(x) else x
        )
        
        # Filter to study period
        wage_df = wage_df[
            (wage_df['year'] >= self.study_start_year) & 
            (wage_df['year'] <= self.study_end_year)
        ].copy()
        
        # Aggregate by year and occupation
        wage_summary = wage_df.groupby(['year', 'ONET_CODE']).agg({
            'A_MEAN': 'mean',
            'A_MEDIAN': 'mean',
            'TOT_EMP': 'sum'
        }).reset_index()
        
        wage_summary.columns = ['year', 'occupation', 'mean_wage', 'median_wage', 'employment']
        
        print(f"  Wage records: {len(wage_summary):,}")
        print(f"  Year range: {wage_summary['year'].min()}-{wage_summary['year'].max()}")
        print(f"  Occupations with wage data: {wage_summary['occupation'].nunique():,}")
        
        return wage_summary
    
    def build_user_career_paths(self, trajectory_df):
        """
        Build career paths from trajectory data
        
        Args:
            trajectory_df: Prepared trajectory dataframe
            
        Returns:
            Dictionary of user_id -> list of job dictionaries
        """
        print("\nBuilding user career paths...")
        
        user_paths = defaultdict(list)
        
        for user_id, group in trajectory_df.groupby('ID'):
            for _, row in group.iterrows():
                user_paths[user_id].append({
                    'occupation': row[self.occupation_col],
                    'occupation_name': row.get('onet_major', row[self.occupation_col]),
                    'start_year': row['job_start_year'],
                    'end_year': row['job_end_year'],
                    'trajectory_order': row['trajectory_order'],
                    'state': row.get('state', 'Unknown'),
                    'naics6_major': row.get('naics6_major', 'Unknown'),
                    'wage': row.get('annual_state_wage', None),
                    'up_move': row.get('up_move', False),
                    'move_1_1': row.get('move_1_1', False),
                    'move_1_2': row.get('move_1_2', False),
                    'move_2_1': row.get('move_2_1', False),
                    'move_2_2': row.get('move_2_2', False)
                })
        
        print(f"  Career paths built for {len(user_paths):,} users")
        return user_paths
    
    def extract_transitions(self, user_paths):
        """
        Extract occupation transitions from career paths
        
        Args:
            user_paths: Dictionary of user career paths
            
        Returns:
            DataFrame of transitions
        """
        print("\nExtracting transitions...")
        print(f"  Study Period: {self.study_start_year}-{self.study_end_year}")
        
        transitions = []
        
        for user_id, path in user_paths.items():
            # Already sorted by trajectory_order
            path_sorted = [p for p in path if p['start_year'] is not None]
            
            for i in range(len(path_sorted) - 1):
                from_job = path_sorted[i]
                to_job = path_sorted[i + 1]
                
                transition_year = to_job['start_year']
                
                # Filter by study period
                if transition_year < self.study_start_year or transition_year > self.study_end_year:
                    continue
                
                transitions.append({
                    'user_id': user_id,
                    'from_occupation': from_job['occupation'],
                    'to_occupation': to_job['occupation'],
                    'from_occupation_name': from_job['occupation_name'],
                    'to_occupation_name': to_job['occupation_name'],
                    'from_year': from_job['start_year'],
                    'to_year': to_job['start_year'],
                    'from_state': from_job['state'],
                    'to_state': to_job['state'],
                    'from_industry': from_job['naics6_major'],
                    'to_industry': to_job['naics6_major'],
                    'from_wage': from_job['wage'],
                    'to_wage': to_job['wage'],
                    'up_move': to_job['up_move'],
                    'move_1_1': to_job['move_1_1'],
                    'move_1_2': to_job['move_1_2'],
                    'move_2_1': to_job['move_2_1'],
                    'move_2_2': to_job['move_2_2']
                })
        
        transitions_df = pd.DataFrame(transitions)
        print(f"  Total transitions extracted: {len(transitions_df):,}")
        
        if len(transitions_df) > 0:
            print(f"  Unique from-occupations: {transitions_df['from_occupation'].nunique():,}")
            print(f"  Unique to-occupations: {transitions_df['to_occupation'].nunique():,}")
            print(f"  Occupation changes: {(transitions_df['from_occupation'] != transitions_df['to_occupation']).sum():,}")
            print(f"  Upward moves: {transitions_df['up_move'].sum():,}")
        
        return transitions_df
    
    def build_temporal_networks(self, transitions_df, wage_df=None, window_size=2):
        """
        Build temporal occupation transition networks with sliding windows
        
        Args:
            transitions_df: Transitions dataframe
            wage_df: Optional wage summary dataframe
            window_size: Window size in years (default: 2)
            
        Returns:
            Dictionary of window_label -> NetworkX graph
        """
        print("\nBuilding temporal networks with sliding windows...")
        print(f"  Window size: {window_size} years")
        print(f"  Hop size: 1 year")
        
        networks = {}
        
        # Get year range
        min_year = int(transitions_df['to_year'].min())
        max_year = int(transitions_df['to_year'].max())
        
        # Generate sliding windows
        for start_year in range(min_year, max_year - window_size + 2):
            end_year = start_year + window_size - 1
            window_label = f"{start_year}-{end_year}"
            
            # Filter transitions within window
            window_transitions = transitions_df[
                (transitions_df['to_year'] >= start_year) &
                (transitions_df['to_year'] <= end_year)
            ].copy()
            
            if len(window_transitions) == 0:
                continue
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Count transitions between occupations
            transition_counts = window_transitions.groupby(
                ['from_occupation', 'to_occupation']
            ).size().reset_index(name='count')
            
            # Add edges
            for _, row in transition_counts.iterrows():
                from_occ = row['from_occupation']
                to_occ = row['to_occupation']
                count = row['count']
                
                G.add_edge(from_occ, to_occ, weight=count)
            
            # Add node attributes
            for node in G.nodes():
                # Get transitions involving this node
                node_from = window_transitions[window_transitions['from_occupation'] == node]
                node_to = window_transitions[window_transitions['to_occupation'] == node]
                node_all = window_transitions[
                    (window_transitions['from_occupation'] == node) |
                    (window_transitions['to_occupation'] == node)
                ]
                
                # Employment count: unique users who had this occupation
                employment_count = len(node_all['user_id'].unique())
                G.nodes[node]['employment_count'] = employment_count
                
                # Calculate wage metrics from outgoing transitions
                if len(node_from) > 0:
                    avg_wage = node_from['from_wage'].mean()
                    G.nodes[node]['avg_wage'] = avg_wage if not pd.isna(avg_wage) else None
                    
                    # Up-move rate
                    up_move_rate = node_from['up_move'].mean()
                    G.nodes[node]['up_move_rate'] = up_move_rate if not pd.isna(up_move_rate) else None
                else:
                    G.nodes[node]['avg_wage'] = None
                    G.nodes[node]['up_move_rate'] = None
            
            # Add wage data if provided (optional external data)
            if wage_df is not None:
                # Average wage data over the window
                window_wages = wage_df[
                    (wage_df['year'] >= start_year) & 
                    (wage_df['year'] <= end_year)
                ]
                
                if len(window_wages) > 0:
                    wage_summary = window_wages.groupby('occupation').agg({
                        'mean_wage': 'mean',
                        'employment': 'mean'
                    }).to_dict('index')
                    
                    for node in G.nodes():
                        if node in wage_summary:
                            G.nodes[node]['bls_mean_wage'] = wage_summary[node]['mean_wage']
                            G.nodes[node]['bls_employment'] = wage_summary[node]['employment']
            
            networks[window_label] = G
            print(f"  {window_label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return networks
    
    def calculate_statistics(self, networks):
        """
        Calculate network statistics for each window
        
        Args:
            networks: Dictionary of window_label -> NetworkX graph
            
        Returns:
            DataFrame of statistics
        """
        print("\nCalculating network statistics...")
        
        stats = []
        
        for window_label, G in networks.items():
            # Basic network metrics
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G) if n_nodes > 0 else 0
            
            # Connectivity
            if n_nodes > 0:
                largest_wcc = max(nx.weakly_connected_components(G), key=len)
                largest_wcc_size = len(largest_wcc)
                largest_wcc_pct = largest_wcc_size / n_nodes * 100
            else:
                largest_wcc_size = 0
                largest_wcc_pct = 0
            
            # Degree statistics
            if n_nodes > 0:
                in_degrees = [d for n, d in G.in_degree()]
                out_degrees = [d for n, d in G.out_degree()]
                avg_in_degree = np.mean(in_degrees)
                avg_out_degree = np.mean(out_degrees)
            else:
                avg_in_degree = 0
                avg_out_degree = 0
            
            # Employment statistics (from node attributes)
            employment_counts = [
                G.nodes[node].get('employment_count', 0) 
                for node in G.nodes()
            ]
            total_employment = sum(employment_counts)
            avg_employment_per_occupation = np.mean(employment_counts) if employment_counts else 0
            
            stats.append({
                'window': window_label,
                'nodes': n_nodes,
                'edges': n_edges,
                'density': density,
                'avg_in_degree': avg_in_degree,
                'avg_out_degree': avg_out_degree,
                'largest_wcc_size': largest_wcc_size,
                'largest_wcc_pct': largest_wcc_pct,
                'total_employment': total_employment,
                'avg_employment_per_occupation': avg_employment_per_occupation
            })
        
        stats_df = pd.DataFrame(stats)
        return stats_df
    
    def save_networks(self, networks, stats_df, output_dir):
        """
        Save networks and statistics to disk
        
        Args:
            networks: Dictionary of networks
            stats_df: Statistics dataframe
            output_dir: Output directory
            
        Returns:
            Path to output directory
        """
        print("\nSaving networks...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual networks as GraphML
        for year, G in networks.items():
            if pd.isna(year):
                continue
            
            # Create a copy to clean for GraphML export
            G_clean = G.copy()
            
            # Remove None values from node attributes (GraphML doesn't support None)
            for node in G_clean.nodes():
                attrs = G_clean.nodes[node]
                clean_attrs = {k: v for k, v in attrs.items() if v is not None and not pd.isna(v)}
                G_clean.nodes[node].clear()
                G_clean.nodes[node].update(clean_attrs)
            
            # Remove None values from edge attributes
            for u, v in G_clean.edges():
                attrs = G_clean.edges[u, v]
                clean_attrs = {k: v for k, v in attrs.items() if v is not None and not pd.isna(v)}
                G_clean.edges[u, v].clear()
                G_clean.edges[u, v].update(clean_attrs)
            
            filename = output_path / f"network_{str(year)}.graphml"
            nx.write_graphml(G_clean, filename)
            print(f"  Saved: {filename}")
        
        # Save all networks as pickle (can handle None values)
        with open(output_path / 'networks_all.pkl', 'wb') as f:
            pickle.dump(networks, f)
        print(f"  Saved: {output_path / 'networks_all.pkl'}")
        
        # Save statistics
        stats_df.to_csv(output_path / 'network_statistics.csv', index=False)
        print(f"  Saved: {output_path / 'network_statistics.csv'}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'study_period': f"{self.study_start_year}-{self.study_end_year}",
            'occupation_column': self.occupation_col,
            'n_years': len([y for y in networks.keys() if not pd.isna(y)]),
            'total_nodes': sum(G.number_of_nodes() for y, G in networks.items() if not pd.isna(y)),
            'total_edges': sum(G.number_of_edges() for y, G in networks.items() if not pd.isna(y))
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {output_path / 'metadata.json'}")
        
        return output_path
    
    def build_all(self, trajectory_df, wage_df, output_dir):
        """
        Complete pipeline: build and save networks
        
        Args:
            trajectory_df: Career trajectories dataframe
            wage_df: Wage dataframe (optional)
            output_dir: Output directory
        
        Returns:
            Tuple of (networks, stats_df, output_path)
        """
        trajectory_df = self.prepare_trajectory_data(trajectory_df)
        
        if wage_df is not None:
            wage_df = self.prepare_wage_data(wage_df)
        
        user_paths = self.build_user_career_paths(trajectory_df)
        transitions_df = self.extract_transitions(user_paths)
        networks = self.build_temporal_networks(transitions_df, wage_df)
        stats_df = self.calculate_statistics(networks)
        output_path = self.save_networks(networks, stats_df, output_dir)
        
        return networks, stats_df, output_path, transitions_df


# ============================================
# NETWORK VISUALIZER
# ============================================

class NetworkVisualizer:
    """
    Visualizes occupation transition networks
    Creates Sankey diagrams, transition matrices, and comparison plots
    """
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Set up matplotlib style"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    # ========================================
    # TRANSITION MATRIX METHODS
    # ========================================
    
    def build_transition_matrix(self, transitions_df, period_filter=None, top_n=15):
        """
        Build transition matrix from transitions dataframe
        
        Args:
            transitions_df: Transitions dataframe
            period_filter: Tuple of (year_from, year_to) to filter
            top_n: Number of top occupations to include
            
        Returns:
            Tuple of (matrix, occupations, total_transitions)
        """
        df = transitions_df.copy()
        
        if period_filter:
            year_from, year_to = period_filter
            df = df[(df['from_year'] >= year_from) & (df['to_year'] <= year_to)]
        
        # Get top occupations by frequency
        from_counts = df['from_occupation'].value_counts()
        to_counts = df['to_occupation'].value_counts()
        total_counts = (from_counts + to_counts).fillna(0)
        top_occupations = list(total_counts.nlargest(top_n).index)
        
        # Filter to top occupations
        df = df[
            df['from_occupation'].isin(top_occupations) & 
            df['to_occupation'].isin(top_occupations)
        ]
        
        # Build transition matrix
        matrix = pd.crosstab(
            df['from_occupation'], 
            df['to_occupation']
        )
        
        # Ensure all top occupations are present
        for occ in top_occupations:
            if occ not in matrix.index:
                matrix.loc[occ] = 0
            if occ not in matrix.columns:
                matrix[occ] = 0
        
        matrix = matrix.loc[top_occupations, top_occupations]
        
        return matrix, top_occupations, len(df)
    
    def normalize_transition_matrix(self, matrix):
        """
        Normalize transition matrix by row sums
        
        Args:
            matrix: Transition count matrix
            
        Returns:
            Normalized matrix (row percentages)
        """
        row_sums = matrix.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # Avoid division by zero
        return matrix.div(row_sums, axis=0) * 100
    
    def plot_transition_matrix(self, matrix, title, filename, normalized=False):
        """
        Plot transition matrix as heatmap
        
        Args:
            matrix: Transition matrix
            title: Plot title
            filename: Output filename
            normalized: Whether matrix is normalized
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Format values
        if normalized:
            fmt = '.1f'
            cbar_label = 'Transition Probability (%)'
        else:
            fmt = 'd'
            cbar_label = 'Number of Transitions'
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap='YlOrRd',
            linewidths=0.5,
            cbar_kws={'label': cbar_label},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('To Occupation', fontsize=12)
        ax.set_ylabel('From Occupation', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    def plot_transition_difference(self, matrix1, matrix2, period1_name, period2_name, 
                                   filename, top_n=15):
        """
        Plot difference between two transition matrices
        
        Args:
            matrix1: First transition matrix
            matrix2: Second transition matrix
            period1_name: Name of first period
            period2_name: Name of second period
            filename: Output filename
            top_n: Number of top occupations
        """
        # Normalize matrices
        norm1 = self.normalize_transition_matrix(matrix1)
        norm2 = self.normalize_transition_matrix(matrix2)
        
        # Calculate difference
        diff = norm2 - norm1
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        
        sns.heatmap(
            diff,
            annot=True,
            fmt='.1f',
            cmap='RdBu_r',
            center=0,
            linewidths=0.5,
            cbar_kws={'label': 'Percentage Point Change'},
            vmin=-10,
            vmax=10,
            ax=ax
        )
        
        ax.set_title(
            f'Change in Transition Patterns: {period2_name} vs {period1_name}',
            fontsize=14,
            pad=20
        )
        ax.set_xlabel('To Occupation', fontsize=12)
        ax.set_ylabel('From Occupation', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    # ========================================
    # SANKEY DIAGRAM METHODS
    # ========================================
    
    def create_sankey_diagram(self, transitions_df, period_filter, title, filename, top_n=10):
        """
        Create Sankey diagram for occupation transitions
        
        Args:
            transitions_df: Transitions dataframe
            period_filter: Tuple of (year_from, year_to)
            title: Plot title
            filename: Output filename
            top_n: Number of top occupations to show
        """
        df = transitions_df.copy()
        year_from, year_to = period_filter
        
        # Filter by period
        df = df[(df['from_year'] >= year_from) & (df['to_year'] <= year_to)]
        
        if len(df) == 0:
            print(f"  Warning: No transitions in period {year_from}-{year_to}")
            return
        
        # Get top occupations
        from_counts = df['from_occupation'].value_counts()
        to_counts = df['to_occupation'].value_counts()
        total_counts = (from_counts + to_counts).fillna(0)
        top_occupations = list(total_counts.nlargest(top_n).index)
        
        # Filter to top occupations
        df = df[
            df['from_occupation'].isin(top_occupations) & 
            df['to_occupation'].isin(top_occupations)
        ]
        
        # Aggregate transitions
        flow_data = df.groupby(['from_occupation', 'to_occupation']).size().reset_index(name='value')
        
        # Create simple bar plot as placeholder
        # (Full Sankey diagram requires plotly or additional libraries)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        top_flows = flow_data.nlargest(20, 'value')
        top_flows['flow'] = top_flows['from_occupation'] + ' → ' + top_flows['to_occupation']
        
        ax.barh(range(len(top_flows)), top_flows['value'])
        ax.set_yticks(range(len(top_flows)))
        ax.set_yticklabels(top_flows['flow'], fontsize=8)
        ax.set_xlabel('Number of Transitions', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    # ========================================
    # FLOW STATISTICS
    # ========================================
    
    def calculate_flow_statistics(self, transitions_df):
        """
        Calculate flow statistics by period
        
        Args:
            transitions_df: Transitions dataframe
            
        Returns:
            DataFrame of flow statistics
        """
        periods = {
            'Pre-COVID (2017-2019)': (2017, 2019),
            'COVID (2020-2021)': (2020, 2021),
            'Post-COVID (2022-2024)': (2022, 2024)
        }
        
        stats = []
        
        for period_name, (year_from, year_to) in periods.items():
            df = transitions_df[
                (transitions_df['from_year'] >= year_from) & 
                (transitions_df['to_year'] <= year_to)
            ]
            
            if len(df) == 0:
                continue
            
            total_transitions = len(df)
            occupation_changes = (df['from_occupation'] != df['to_occupation']).sum()
            state_changes = (df['from_state'] != df['to_state']).sum()
            industry_changes = (df['from_industry'] != df['to_industry']).sum()
            upward_moves = df['up_move'].sum()
            
            unique_from_occ = df['from_occupation'].nunique()
            unique_to_occ = df['to_occupation'].nunique()
            
            stats.append({
                'Period': period_name,
                'Year_Range': f"{year_from}-{year_to}",
                'Total_Transitions': total_transitions,
                'Occupation_Changes': occupation_changes,
                'Occupation_Change_Rate': (occupation_changes / total_transitions * 100),
                'State_Changes': state_changes,
                'State_Change_Rate': (state_changes / total_transitions * 100),
                'Industry_Changes': industry_changes,
                'Industry_Change_Rate': (industry_changes / total_transitions * 100),
                'Upward_Moves': upward_moves,
                'Upward_Move_Rate': (upward_moves / total_transitions * 100),
                'Unique_From_Occupations': unique_from_occ,
                'Unique_To_Occupations': unique_to_occ
            })
        
        return pd.DataFrame(stats)
    
    # ========================================
    # COMPLETE VISUALIZATION PIPELINE
    # ========================================
    
    def visualize_all(self, transitions_df, output_dir):
        """
        Create all network visualizations
        
        Args:
            transitions_df: All transitions dataframe
            output_dir: Directory to save results
        """
        print("\n" + "="*80)
        print("CREATING NETWORK VISUALIZATIONS")
        print("="*80)
        
        periods = {
            'pre_covid': (2017, 2019),
            'covid': (2020, 2021),
            'post_covid': (2022, 2024)
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate flow statistics
        print("\nCalculating flow statistics...")
        flow_stats = self.calculate_flow_statistics(transitions_df)
        stats_path = os.path.join(output_dir, 'flow_statistics_by_period.csv')
        flow_stats.to_csv(stats_path, index=False)
        print(f"✓ Saved flow statistics: {stats_path}")
        print("\nFlow Statistics:")
        print(flow_stats.to_string(index=False))
        
        # Build transition matrices
        print("\nBuilding transition matrices...")
        matrices = {}
        matrices_normalized = {}
        
        for period_name, (year_from, year_to) in periods.items():
            print(f"  Processing {period_name} ({year_from}-{year_to})...")
            matrix, occupations, total = self.build_transition_matrix(
                transitions_df, 
                period_filter=(year_from, year_to),
                top_n=15
            )
            matrices[period_name] = matrix
            matrices_normalized[period_name] = self.normalize_transition_matrix(matrix)
        
        # Plot transition matrices
        print("\nCreating transition matrix heatmaps...")
        for period_name in periods.keys():
            # Raw counts
            self.plot_transition_matrix(
                matrices[period_name],
                f'Occupation Transitions: {period_name.replace("_", " ").title()}',
                os.path.join(output_dir, f'transition_matrix_{period_name}.png'),
                normalized=False
            )
            
            # Normalized
            self.plot_transition_matrix(
                matrices_normalized[period_name],
                f'Occupation Transition Probabilities: {period_name.replace("_", " ").title()}',
                os.path.join(output_dir, f'transition_matrix_{period_name}_normalized.png'),
                normalized=True
            )
        
        # Plot differences
        print("\nCreating transition difference plots...")
        self.plot_transition_difference(
            matrices['pre_covid'],
            matrices['covid'],
            'Pre-COVID (2017-2019)',
            'COVID (2020-2021)',
            os.path.join(output_dir, 'transition_difference_covid_vs_pre.png')
        )
        
        self.plot_transition_difference(
            matrices['covid'],
            matrices['post_covid'],
            'COVID (2020-2021)',
            'Post-COVID (2022-2024)',
            os.path.join(output_dir, 'transition_difference_post_vs_covid.png')
        )
        
        # Create Sankey diagrams
        print("\nCreating Sankey diagrams...")
        for period_name, (year_from, year_to) in periods.items():
            self.create_sankey_diagram(
                transitions_df,
                (year_from, year_to),
                f'Top Occupation Flows: {period_name.replace("_", " ").title()} ({year_from}-{year_to})',
                os.path.join(output_dir, f'sankey_{period_name}.png')
            )
        
        print("\n✓ All visualizations completed")
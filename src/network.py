"""
Unified Network Module
Combines network construction and visualization into a single module
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
    Builds temporal occupation transition networks from job history data
    """
    
    def __init__(self, 
                 study_start_year=2017,
                 study_end_year=2024,
                 occupation_col='ONET_2019',
                 occupation_name_col='ONET_2019_NAME'):
        """
        Initialize network builder
        
        Args:
            study_start_year: Start year of study period
            study_end_year: End year of study period
            occupation_col: Occupation code column
            occupation_name_col: Occupation name column
        """
        self.study_start_year = study_start_year
        self.study_end_year = study_end_year
        self.occupation_col = occupation_col
        self.occupation_name_col = occupation_name_col
    
    def parse_date(self, date_str):
        """Parse date string (YYYY-MM format)"""
        if pd.isna(date_str) or date_str == '':
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m')
        except:
            return None
    
    def get_year_from_date(self, date_str):
        """Extract year from date string"""
        date = self.parse_date(date_str)
        return date.year if date else None
    
    def prepare_job_data(self, job_df):
        """Prepare job data"""
        print("\nPreparing job data...")
        
        # Parse dates
        job_df['start_year'] = job_df['JOB_START_DATE'].apply(self.get_year_from_date)
        job_df['end_year'] = job_df['JOB_END_DATE'].apply(self.get_year_from_date)
        
        # Remove Unclassified occupations
        job_df = job_df[job_df[self.occupation_col] != '99-9999.00'].copy()
        
        # Select required columns only
        cols = ['ID', self.occupation_col, self.occupation_name_col, 
                'start_year', 'end_year', 'IS_CURRENT',
                'SOC_EMSI_2019_2', 'SOC_EMSI_2019_2_NAME']
        job_df = job_df[cols].copy()
        
        print(f"  Valid job records: {len(job_df):,}")
        print(f"  Unique users: {job_df['ID'].nunique():,}")
        print(f"  Unique occupations: {job_df[self.occupation_col].nunique():,}")
        
        return job_df
    
    def prepare_wage_data(self, wage_df):
        """Prepare wage data - convert to O*NET code format"""
        print("\nPreparing wage data...")
        
        # Convert OCC_CODE to O*NET format (SOC -> O*NET)
        wage_df['ONET_CODE'] = wage_df['OCC_CODE'].apply(
            lambda x: f"{x}.00" if pd.notna(x) and '.' not in str(x) else x
        )
        
        # Filter wage data to study period
        wage_df = wage_df[
            (wage_df['year'] >= self.study_start_year) & 
            (wage_df['year'] <= self.study_end_year)
        ].copy()
        
        # Calculate average wages by year and occupation
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
    
    def build_user_career_paths(self, job_df):
        """Build career paths for each user"""
        print("\nBuilding user career paths...")
        
        job_df_sorted = job_df.sort_values(['ID', 'start_year']).copy()
        user_paths = defaultdict(list)
        
        for user_id, group in job_df_sorted.groupby('ID'):
            for _, row in group.iterrows():
                user_paths[user_id].append({
                    'occupation': row[self.occupation_col],
                    'occupation_name': row[self.occupation_name_col],
                    'start_year': row['start_year'],
                    'end_year': row['end_year'],
                    'soc_2': row['SOC_EMSI_2019_2'],
                    'soc_2_name': row['SOC_EMSI_2019_2_NAME']
                })
        
        print(f"  Career paths built for {len(user_paths):,} users")
        return user_paths
    
    def extract_transitions(self, user_paths):
        """Extract transitions from career paths"""
        print("\nExtracting transitions...")
        print(f"  Study Period: {self.study_start_year}-{self.study_end_year}")
        
        transitions = []
        
        for user_id, path in user_paths.items():
            path_sorted = sorted([p for p in path if p['start_year'] is not None], 
                                key=lambda x: x['start_year'])
            
            for i in range(len(path_sorted) - 1):
                from_job = path_sorted[i]
                to_job = path_sorted[i + 1]
                
                transition_year = to_job['start_year']
                
                if transition_year < self.study_start_year or transition_year > self.study_end_year:
                    continue
                
                transitions.append({
                    'user_id': user_id,
                    'from_occupation': from_job['occupation'],
                    'from_occupation_name': from_job['occupation_name'],
                    'to_occupation': to_job['occupation'],
                    'to_occupation_name': to_job['occupation_name'],
                    'transition_year': transition_year,
                    'from_year': from_job['start_year'],
                    'to_year': to_job['start_year'],
                    'from_soc_2': from_job['soc_2'],
                    'to_soc_2': to_job['soc_2']
                })
        
        transitions_df = pd.DataFrame(transitions)
        
        if len(transitions_df) > 0:
            print(f"  Total transitions: {len(transitions_df):,}")
            print(f"  Unique users: {transitions_df['user_id'].nunique():,}")
            print(f"  Year range: {transitions_df['transition_year'].min()}-{transitions_df['transition_year'].max()}")
        else:
            print("  No transitions found")
        
        return transitions_df
    
    def build_temporal_networks(self, transitions_df, wage_df):
        """Build temporal networks"""
        print("\nBuilding temporal networks...")
        
        networks = {}
        
        for year in sorted(transitions_df['transition_year'].unique()):
            year_transitions = transitions_df[transitions_df['transition_year'] == year]
            year_wage = wage_df[wage_df['year'] == year]
            
            G = nx.DiGraph()
            G.graph['year'] = year
            G.graph['period'] = 'annual'
            
            # Add nodes
            all_occupations = set(year_transitions['from_occupation'].unique()) | \
                             set(year_transitions['to_occupation'].unique())
            
            for occ in all_occupations:
                occ_name = year_transitions[
                    (year_transitions['from_occupation'] == occ) | 
                    (year_transitions['to_occupation'] == occ)
                ].iloc[0]
                
                if year_transitions[year_transitions['from_occupation'] == occ].empty:
                    name = occ_name['to_occupation_name']
                else:
                    name = occ_name['from_occupation_name']
                
                node_attrs = {
                    'occupation_code': occ,
                    'occupation_name': name,
                    'year': year
                }
                
                # Add wage information
                wage_info = year_wage[year_wage['occupation'] == occ]
                if not wage_info.empty:
                    node_attrs['mean_wage'] = float(wage_info.iloc[0]['mean_wage'])
                    node_attrs['median_wage'] = float(wage_info.iloc[0]['median_wage'])
                    node_attrs['employment'] = float(wage_info.iloc[0]['employment'])
                else:
                    node_attrs['mean_wage'] = None
                    node_attrs['median_wage'] = None
                    node_attrs['employment'] = None
                
                from_count = len(year_transitions[year_transitions['from_occupation'] == occ])
                to_count = len(year_transitions[year_transitions['to_occupation'] == occ])
                node_attrs['transition_volume'] = from_count + to_count
                
                G.add_node(occ, **node_attrs)
            
            # Add edges
            edge_counts = year_transitions.groupby(['from_occupation', 'to_occupation']).size()
            
            for (from_occ, to_occ), count in edge_counts.items():
                from_wage = G.nodes[from_occ].get('mean_wage')
                to_wage = G.nodes[to_occ].get('mean_wage')
                
                upward_mobility = None
                if from_wage is not None and to_wage is not None:
                    upward_mobility = to_wage > from_wage
                    wage_change = to_wage - from_wage
                    wage_change_pct = (wage_change / from_wage) * 100 if from_wage > 0 else 0
                else:
                    wage_change = None
                    wage_change_pct = None
                
                edge_attrs = {
                    'transition_count': int(count),
                    'year': year,
                    'upward_mobility': upward_mobility,
                    'wage_change': wage_change,
                    'wage_change_pct': wage_change_pct
                }
                
                G.add_edge(from_occ, to_occ, **edge_attrs)
            
            networks[year] = G
            print(f"  Year {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return networks
    
    def calculate_statistics(self, networks):
        """Calculate network statistics"""
        print("\nCalculating network statistics...")
        
        stats = []
        
        for year, G in sorted(networks.items()):
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G) if n_nodes > 1 else 0
            
            upward_edges = sum(1 for _, _, d in G.edges(data=True) 
                              if d.get('upward_mobility') is True)
            downward_edges = sum(1 for _, _, d in G.edges(data=True) 
                                if d.get('upward_mobility') is False)
            upward_ratio = upward_edges / n_edges if n_edges > 0 else 0
            
            total_transitions = sum(d['transition_count'] for _, _, d in G.edges(data=True))
            
            avg_in_degree = sum(d for _, d in G.in_degree()) / n_nodes if n_nodes > 0 else 0
            avg_out_degree = sum(d for _, d in G.out_degree()) / n_nodes if n_nodes > 0 else 0
            
            stats.append({
                'year': year,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
                'total_transitions': total_transitions,
                'upward_edges': upward_edges,
                'downward_edges': downward_edges,
                'upward_ratio': upward_ratio,
                'avg_in_degree': avg_in_degree,
                'avg_out_degree': avg_out_degree
            })
        
        stats_df = pd.DataFrame(stats)
        
        print("\nNetwork Statistics Summary:")
        print(stats_df.to_string(index=False))
        
        return stats_df
    
    def save_networks(self, networks, stats_df, output_dir):
        """Save networks"""
        print("\nSaving networks...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save each year's network as GraphML
        for year, G in networks.items():
            if pd.isna(year):
                continue
            
            G_copy = G.copy()
            for node in G_copy.nodes():
                for key, value in G_copy.nodes[node].items():
                    if value is None:
                        G_copy.nodes[node][key] = 'None'
            
            for u, v in G_copy.edges():
                for key, value in G_copy.edges[u, v].items():
                    if value is None:
                        G_copy.edges[u, v][key] = 'None'
            
            filename = output_path / f'network_{int(year)}.graphml'
            nx.write_graphml(G_copy, filename)
            print(f"  Saved: {filename}")
        
        # Save all networks as pickle
        with open(output_path / 'networks_all.pkl', 'wb') as f:
            pickle.dump(networks, f)
        print(f"  Saved: {output_path / 'networks_all.pkl'}")
        
        # Save statistics
        stats_df.to_csv(output_path / 'network_statistics.csv', index=False)
        print(f"  Saved: {output_path / 'network_statistics.csv'}")
        
        # Save metadata
        metadata = {
            'occupation_column': self.occupation_col,
            'occupation_name_column': self.occupation_name_col,
            'years': sorted([y for y in networks.keys() if not pd.isna(y)]),
            'n_networks': len([y for y in networks.keys() if not pd.isna(y)]),
            'total_nodes': sum(G.number_of_nodes() for y, G in networks.items() if not pd.isna(y)),
            'total_edges': sum(G.number_of_edges() for y, G in networks.items() if not pd.isna(y))
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {output_path / 'metadata.json'}")
        
        return output_path
    
    def build_all(self, job_df, wage_df, output_dir):
        """
        Complete pipeline: build and save networks
        
        Args:
            job_df: Job dataframe
            wage_df: Wage dataframe
            output_dir: Output directory
        
        Returns:
            Tuple of (networks, stats_df, output_path)
        """
        job_df = self.prepare_job_data(job_df)
        wage_df = self.prepare_wage_data(wage_df)
        user_paths = self.build_user_career_paths(job_df)
        transitions_df = self.extract_transitions(user_paths)
        networks = self.build_temporal_networks(transitions_df, wage_df)
        stats_df = self.calculate_statistics(networks)
        output_path = self.save_networks(networks, stats_df, output_dir)
        
        return networks, stats_df, output_path


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
        """Build transition matrix from transitions dataframe"""
        df = transitions_df.copy()
        
        if period_filter:
            year_from, year_to = period_filter
            df = df[(df['Year_From'] >= year_from) & (df['Year_To'] <= year_to)]
        
        from_counts = df['From_Occupation'].value_counts()
        to_counts = df['To_Occupation'].value_counts()
        total_counts = (from_counts + to_counts).fillna(0)
        top_occupations = list(total_counts.nlargest(top_n).index)
        
        df = df[
            df['From_Occupation'].isin(top_occupations) & 
            df['To_Occupation'].isin(top_occupations)
        ]
        
        matrix = pd.crosstab(
            df['From_Occupation'], 
            df['To_Occupation'], 
            values=df['ID'], 
            aggfunc='count',
            dropna=False
        ).fillna(0)
        
        for occ in top_occupations:
            if occ not in matrix.index:
                matrix.loc[occ] = 0
            if occ not in matrix.columns:
                matrix[occ] = 0
        
        matrix = matrix.reindex(index=top_occupations, columns=top_occupations, fill_value=0)
        total_transitions = df.shape[0]
        
        return matrix, top_occupations, total_transitions
    
    def normalize_transition_matrix(self, matrix):
        """Normalize transition matrix (row-wise probabilities)"""
        row_sums = matrix.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        normalized = matrix.div(row_sums, axis=0)
        return normalized
    
    def compare_transition_matrices(self, matrix1, matrix2):
        """Compare two transition matrices (difference)"""
        all_occupations = sorted(set(matrix1.index) | set(matrix2.index))
        
        matrix1_aligned = matrix1.reindex(
            index=all_occupations, 
            columns=all_occupations, 
            fill_value=0
        )
        matrix2_aligned = matrix2.reindex(
            index=all_occupations, 
            columns=all_occupations, 
            fill_value=0
        )
        
        diff_matrix = matrix2_aligned - matrix1_aligned
        return diff_matrix
    
    # ========================================
    # SANKEY DIAGRAM METHODS
    # ========================================
    
    def prepare_sankey_data(self, transitions_df, period_filter=None, top_n=10, min_flow=10):
        """Prepare data for Sankey diagram"""
        df = transitions_df.copy()
        
        if period_filter:
            year_from, year_to = period_filter
            df = df[(df['Year_From'] >= year_from) & (df['Year_To'] <= year_to)]
        
        if len(df) == 0:
            return {
                'source': [],
                'target': [],
                'value': [],
                'labels': [],
                'flow_counts': pd.DataFrame()
            }
        
        from_counts = df['From_Occupation'].value_counts()
        to_counts = df['To_Occupation'].value_counts()
        total_counts = (from_counts + to_counts).fillna(0)
        
        if len(total_counts) == 0:
            return {
                'source': [],
                'target': [],
                'value': [],
                'labels': [],
                'flow_counts': pd.DataFrame()
            }
        
        top_occupations = list(total_counts.nlargest(min(top_n, len(total_counts))).index)
        
        df = df[
            df['From_Occupation'].isin(top_occupations) & 
            df['To_Occupation'].isin(top_occupations)
        ]
        
        if len(df) == 0:
            return {
                'source': [],
                'target': [],
                'value': [],
                'labels': [],
                'flow_counts': pd.DataFrame()
            }
        
        flow_counts = df.groupby(['From_Occupation', 'To_Occupation']).size().reset_index(name='count')
        flow_counts = flow_counts[flow_counts['count'] >= min_flow]
        
        if len(flow_counts) == 0:
            flow_counts = df.groupby(['From_Occupation', 'To_Occupation']).size().reset_index(name='count')
            flow_counts = flow_counts[flow_counts['count'] >= max(1, min_flow // 2)]
        
        all_occupations = sorted(set(flow_counts['From_Occupation']) | set(flow_counts['To_Occupation']))
        
        if len(all_occupations) == 0:
            return {
                'source': [],
                'target': [],
                'value': [],
                'labels': [],
                'flow_counts': pd.DataFrame()
            }
        
        node_dict = {occ: idx for idx, occ in enumerate(all_occupations)}
        
        source = [node_dict[occ] for occ in flow_counts['From_Occupation']]
        target = [node_dict[occ] for occ in flow_counts['To_Occupation']]
        value = flow_counts['count'].tolist()
        
        return {
            'source': source,
            'target': target,
            'value': value,
            'labels': all_occupations,
            'flow_counts': flow_counts
        }
    
    def create_sankey_diagram(self, sankey_data, period_name, output_path, max_label_length=40):
        """Create Sankey diagram"""
        flow_counts = sankey_data['flow_counts']
        
        if len(flow_counts) == 0:
            print(f"  ⚠ No data for {period_name}, skipping Sankey diagram")
            return
        
        top_flows = flow_counts.nlargest(min(25, len(flow_counts)), 'count')
        top_flows = top_flows[top_flows['From_Occupation'] != top_flows['To_Occupation']]
        
        if len(top_flows) == 0:
            print(f"  ⚠ No valid transitions for {period_name}, skipping Sankey diagram")
            return
        
        from_occs = top_flows['From_Occupation'].unique()
        to_occs = top_flows['To_Occupation'].unique()
        
        fig, ax = plt.subplots(figsize=(20, 14))
        
        x_from = 0
        x_to = 10
        
        from_heights = {}
        to_heights = {}
        
        for occ in from_occs:
            from_heights[occ] = top_flows[top_flows['From_Occupation'] == occ]['count'].sum()
        
        for occ in to_occs:
            to_heights[occ] = top_flows[top_flows['To_Occupation'] == occ]['count'].sum()
        
        max_height = max(max(from_heights.values()), max(to_heights.values()))
        height_scale = 30 / max_height
        
        from_y_positions = {}
        current_y = 0
        spacing = 0.5
        
        for occ in sorted(from_occs):
            height = from_heights[occ] * height_scale
            from_y_positions[occ] = (current_y, height)
            current_y += height + spacing
        
        to_y_positions = {}
        current_y = 0
        
        for occ in sorted(to_occs):
            height = to_heights[occ] * height_scale
            to_y_positions[occ] = (current_y, height)
            current_y += height + spacing
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_flows)))
        top_flows_sorted = top_flows.sort_values('count', ascending=True)
        
        for idx, (_, row) in enumerate(top_flows_sorted.iterrows()):
            from_occ = row['From_Occupation']
            to_occ = row['To_Occupation']
            flow_count = row['count']
            
            flow_height = flow_count * height_scale
            
            from_y_base, from_total_height = from_y_positions[from_occ]
            to_y_base, to_total_height = to_y_positions[to_occ]
            
            from_offset = 0
            to_offset = 0
            
            for _, prev_row in top_flows_sorted.iterrows():
                if prev_row['From_Occupation'] == from_occ and prev_row.name < row.name:
                    from_offset += prev_row['count'] * height_scale
                if prev_row['To_Occupation'] == to_occ and prev_row.name < row.name:
                    to_offset += prev_row['count'] * height_scale
            
            y_from_start = from_y_base + from_offset
            y_from_end = y_from_start + flow_height
            
            y_to_start = to_y_base + to_offset
            y_to_end = y_to_start + flow_height
            
            ctrl_x = (x_from + x_to) / 2
            
            verts = [
                (x_from, y_from_start),
                (ctrl_x, y_from_start),
                (ctrl_x, y_to_start),
                (x_to, y_to_start),
                (x_to, y_to_end),
                (ctrl_x, y_to_end),
                (ctrl_x, y_from_end),
                (x_from, y_from_end),
                (x_from, y_from_start),
            ]
            
            codes = [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            
            path = MplPath(verts, codes)
            patch = PathPatch(path, facecolor=colors[idx % len(colors)], 
                             edgecolor='none', alpha=0.6)
            ax.add_patch(patch)
        
        # Draw rectangles for occupations
        for occ in sorted(from_occs):
            y_base, height = from_y_positions[occ]
            rect = mpatches.Rectangle((x_from - 0.3, y_base), 0.3, height,
                                      linewidth=1, edgecolor='black',
                                      facecolor='steelblue', alpha=0.8)
            ax.add_patch(rect)
            
            label = occ[:max_label_length]
            if len(occ) > max_label_length:
                label += '...'
            ax.text(x_from - 0.4, y_base + height/2, label,
                   ha='right', va='center', fontsize=9, fontweight='bold')
            
            count_text = f"{int(from_heights[occ]):,}"
            ax.text(x_from - 0.35, y_base + height/2, count_text,
                   ha='right', va='center', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        for occ in sorted(to_occs):
            y_base, height = to_y_positions[occ]
            rect = mpatches.Rectangle((x_to, y_base), 0.3, height,
                                      linewidth=1, edgecolor='black',
                                      facecolor='coral', alpha=0.8)
            ax.add_patch(rect)
            
            label = occ[:max_label_length]
            if len(occ) > max_label_length:
                label += '...'
            ax.text(x_to + 0.4, y_base + height/2, label,
                   ha='left', va='center', fontsize=9, fontweight='bold')
            
            count_text = f"{int(to_heights[occ]):,}"
            ax.text(x_to + 0.35, y_base + height/2, count_text,
                   ha='left', va='center', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-5, 15)
        ax.set_ylim(-1, max(current_y for _, (current_y, _) in from_y_positions.items()) + 1)
        ax.axis('off')
        
        total_transitions = top_flows['count'].sum()
        title = f'Occupational Mobility Flows: {period_name}\n'
        title += f'Total transitions: {total_transitions:,} people across {len(top_flows)} pathways'
        
        ax.text(x_from - 0.5, ax.get_ylim()[1] + 1, 'FROM\n(Origin)', 
               ha='right', va='bottom', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='steelblue', alpha=0.3))
        
        ax.text(x_to + 0.5, ax.get_ylim()[1] + 1, 'TO\n(Destination)',
               ha='left', va='bottom', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='coral', alpha=0.3))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved Sankey diagram: {output_path}")
    
    # ========================================
    # HEATMAP METHODS
    # ========================================
    
    def plot_transition_heatmap(self, matrix, period_name, output_path, 
                                normalize=True, max_label_length=40):
        """Plot transition matrix as heatmap"""
        if normalize:
            matrix_plot = self.normalize_transition_matrix(matrix)
            fmt = '.1%'
            cbar_label = 'Transition Probability'
            matrix_plot = matrix_plot.copy()
            np.fill_diagonal(matrix_plot.values, 0)
        else:
            matrix_plot = matrix.copy()
            fmt = '.0f'
            cbar_label = 'Number of Transitions'
        
        row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                      for label in matrix_plot.index]
        col_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                      for label in matrix_plot.columns]
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        annot_array = matrix_plot.values.copy()
        annot_mask = annot_array < (annot_array.max() * 0.05)
        
        if normalize:
            annot = [[f'{val:.1%}' if not mask else '' 
                     for val, mask in zip(row, mask_row)]
                    for row, mask_row in zip(annot_array, annot_mask)]
        else:
            annot = [[f'{int(val)}' if not mask and val > 0 else '' 
                     for val, mask in zip(row, mask_row)]
                    for row, mask_row in zip(annot_array, annot_mask)]
        
        sns.heatmap(
            matrix_plot, 
            annot=annot,
            fmt='s',
            cmap='YlOrRd',
            cbar_kws={'label': cbar_label, 'shrink': 0.8},
            xticklabels=col_labels,
            yticklabels=row_labels,
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            square=True
        )
        
        ax.set_xlabel('To Occupation', fontsize=13, fontweight='bold')
        ax.set_ylabel('From Occupation', fontsize=13, fontweight='bold')
        
        title_text = f'Occupation Transition Matrix: {period_name}'
        if normalize:
            title_text += '\n(Diagonal removed for clarity)'
        
        ax.set_title(title_text, fontsize=15, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved transition matrix: {output_path}")
    
    def plot_comparison(self, matrix_pre, matrix_during, matrix_post, 
                       output_path, max_label_length=30):
        """Create comparison plot of transition matrices"""
        fig, axes = plt.subplots(1, 3, figsize=(28, 10))
        
        matrices = [matrix_pre, matrix_during, matrix_post]
        titles = ['Pre-COVID\n(2015-2019)', 'During COVID\n(2020-2021)', 'Post-COVID\n(2022-2023)']
        
        common_occs = sorted(set(matrix_pre.index) & set(matrix_during.index) & set(matrix_post.index))
        
        row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                      for label in common_occs]
        
        for ax, matrix, title in zip(axes, matrices, titles):
            matrix_aligned = matrix.reindex(index=common_occs, columns=common_occs, fill_value=0)
            matrix_plot = matrix_aligned.copy()
            np.fill_diagonal(matrix_plot.values, 0)
            
            sns.heatmap(
                matrix_plot,
                annot=False,
                cmap='YlOrRd',
                vmin=0,
                vmax=0.15,
                cbar_kws={'label': 'Transition\nProbability', 'shrink': 0.8},
                xticklabels=row_labels,
                yticklabels=row_labels,
                ax=ax,
                linewidths=0.3,
                linecolor='white',
                square=True
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('To Occupation', fontsize=11, fontweight='bold')
            ax.set_ylabel('From Occupation', fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', labelsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Occupation Transition Matrices: Period Comparison\n(Diagonals removed)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved comparison plot: {output_path}")
    
    def plot_difference_heatmap(self, diff_matrix, period_comparison, 
                               output_path, max_label_length=30):
        """Plot difference between two transition matrices"""
        row_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label 
                      for label in diff_matrix.index]
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        diff_plot = diff_matrix.copy()
        np.fill_diagonal(diff_plot.values, 0)
        
        vmax = max(abs(diff_plot.min().min()), abs(diff_plot.max().max()))
        vmax = min(vmax, 0.10)
        
        sns.heatmap(
            diff_plot,
            annot=False,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={'label': 'Probability Change\n(Later - Earlier)', 'shrink': 0.8},
            xticklabels=row_labels,
            yticklabels=row_labels,
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            square=True
        )
        
        ax.set_xlabel('To Occupation', fontsize=13, fontweight='bold')
        ax.set_ylabel('From Occupation', fontsize=13, fontweight='bold')
        ax.set_title(f'Transition Probability Changes: {period_comparison}\n' + 
                    'Red = Increased transitions | Blue = Decreased transitions',
                    fontsize=15, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved difference heatmap: {output_path}")
    
    # ========================================
    # FLOW STATISTICS
    # ========================================
    
    def calculate_flow_statistics(self, transitions_df, periods=None):
        """Calculate flow statistics for each period"""
        if periods is None:
            periods = {
                'pre_covid': (2015, 2019),
                'covid': (2020, 2021),
                'post_covid': (2022, 2023)
            }
        
        stats = []
        
        for period_name, (year_from, year_to) in periods.items():
            df = transitions_df[
                (transitions_df['Year_From'] >= year_from) & 
                (transitions_df['Year_To'] <= year_to)
            ]
            
            total_transitions = len(df)
            occupation_changes = df['Occupation_Changed'].sum()
            industry_changes = df['Industry_Changed'].sum()
            state_changes = (df['From_State'] != df['To_State']).sum()
            
            unique_from_occ = df['From_Occupation'].nunique()
            unique_to_occ = df['To_Occupation'].nunique()
            
            stats.append({
                'Period': period_name,
                'Year_Range': f"{year_from}-{year_to}",
                'Total_Transitions': total_transitions,
                'Occupation_Changes': occupation_changes,
                'Occupation_Change_Rate': (occupation_changes / total_transitions * 100) if total_transitions > 0 else 0,
                'Industry_Changes': industry_changes,
                'Industry_Change_Rate': (industry_changes / total_transitions * 100) if total_transitions > 0 else 0,
                'State_Changes': state_changes,
                'State_Change_Rate': (state_changes / total_transitions * 100) if total_transitions > 0 else 0,
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
            'pre_covid': (2015, 2019),
            'covid': (2020, 2021),
            'post_covid': (2022, 2023)
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate and save flow statistics
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
            print(f"    Total transitions: {total:,}")
        
        # Create Sankey diagrams
        print("\nGenerating Sankey diagrams...")
        for period_name, (year_from, year_to) in periods.items():
            sankey_data = self.prepare_sankey_data(
                transitions_df,
                period_filter=(year_from, year_to),
                top_n=12,
                min_flow=5
            )
            output_path = os.path.join(output_dir, f'sankey_{period_name}.png')
            self.create_sankey_diagram(sankey_data, period_name.replace('_', ' ').title(), output_path)
        
        # Create heatmaps
        print("\nGenerating transition matrix heatmaps...")
        for period_name in periods.keys():
            output_path = os.path.join(output_dir, f'transition_matrix_{period_name}_normalized.png')
            self.plot_transition_heatmap(
                matrices[period_name],
                period_name.replace('_', ' ').title(),
                output_path,
                normalize=True
            )
        
        # Create comparison plot
        print("\nGenerating comparison plot...")
        output_path = os.path.join(output_dir, 'transition_matrices_comparison.png')
        self.plot_comparison(
            matrices_normalized['pre_covid'],
            matrices_normalized['covid'],
            matrices_normalized['post_covid'],
            output_path
        )
        
        # Create difference heatmaps
        print("\nGenerating difference heatmaps...")
        
        diff_covid_pre = self.compare_transition_matrices(
            matrices_normalized['pre_covid'],
            matrices_normalized['covid']
        )
        output_path = os.path.join(output_dir, 'transition_difference_covid_vs_pre_covid.png')
        self.plot_difference_heatmap(diff_covid_pre, "COVID vs Pre-COVID", output_path)
        
        diff_post_covid = self.compare_transition_matrices(
            matrices_normalized['covid'],
            matrices_normalized['post_covid']
        )
        output_path = os.path.join(output_dir, 'transition_difference_post_covid_vs_covid.png')
        self.plot_difference_heatmap(diff_post_covid, "Post-COVID vs COVID", output_path)
        
        diff_post_pre = self.compare_transition_matrices(
            matrices_normalized['pre_covid'],
            matrices_normalized['post_covid']
        )
        output_path = os.path.join(output_dir, 'transition_difference_post_covid_vs_pre_covid.png')
        self.plot_difference_heatmap(diff_post_pre, "Post-COVID vs Pre-COVID", output_path)
        
        print("\n" + "="*80)
        print("NETWORK VISUALIZATION COMPLETE")
        print("="*80)
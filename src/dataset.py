"""
Career Trajectory Temporal Data Loader
Loads and processes career trajectory data for link prediction with EvolveGCN-H
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import utils as u


class CareerTrajectoryDataset:
    """Dataset for career trajectory temporal graphs"""
    
    def __init__(self, args):
        """
        Initialize dataset
        
        Args:
            args: Arguments containing data paths and parameters
        """
        self.args = args
        self.data_dir = Path(args.data_dir)
        
        # Load preprocessed trajectory data
        self.trajectory_df = self._load_trajectory_data()
        
        # Build temporal graphs
        self.num_nodes = None
        self.feats_per_node = args.feats_per_node
        self.adj_matrices = []
        self.node_features = []
        self.masks = []
        self.edges = []
        
        self._build_temporal_graphs()
        
    def _load_trajectory_data(self):
        """Load preprocessed career trajectory data"""
        trajectory_path = self.data_dir / 'career_trajectories.parquet'
        
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory data not found: {trajectory_path}")
        
        df = pd.read_parquet(trajectory_path)
        return df
    
    def _build_temporal_graphs(self):
        """Build temporal graphs from career trajectory data"""
        # Get unique occupations (nodes)
        all_occupations = sorted(self.trajectory_df['onet_major_x'].unique())
        self.occupation_to_idx = {occ: idx for idx, occ in enumerate(all_occupations)}
        self.idx_to_occupation = {idx: occ for occ, idx in self.occupation_to_idx.items()}
        self.num_nodes = len(all_occupations)
        
        # Get year range
        df = self.trajectory_df.copy()
        df['year'] = df['job_start_year'].astype(int)
        years = sorted(df['year'].unique())
        self.min_year = years[0]
        self.max_year = years[-1]
        self.num_timesteps = len(years)
        
        # Build user trajectories for transition tracking
        user_trajectories = defaultdict(list)
        for _, row in df.iterrows():
            user_trajectories[row['ID']].append({
                'occupation': row['onet_major_x'],
                'year': row['year'],
                'order': row['trajectory_order']
            })
        
        # Sort each user's trajectory by order
        for user_id in user_trajectories:
            user_trajectories[user_id] = sorted(
                user_trajectories[user_id], 
                key=lambda x: x['order']
            )
        
        # Build graph for each year
        for year in years:
            year_df = df[df['year'] == year]
            
            # Build adjacency matrix from transitions
            adj_matrix = self._build_adjacency_matrix(year, user_trajectories)
            self.adj_matrices.append(adj_matrix)
            
            # Build node features
            node_feats = self._build_node_features(year_df)
            self.node_features.append(node_feats)
            
            # Build node mask (all nodes active)
            mask = torch.ones(self.num_nodes, 1)
            self.masks.append(mask)
            
            # Extract edges for link prediction
            edges = self._extract_edges(adj_matrix)
            self.edges.append(edges)
    
    def _build_adjacency_matrix(self, year, user_trajectories):
        """
        Build adjacency matrix from transitions in given year
        
        Args:
            year: Target year
            user_trajectories: Dictionary of user trajectories
        
        Returns:
            Normalized adjacency matrix
        """
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        
        # For each user, find transitions that occur in this year
        for user_id, trajectory in user_trajectories.items():
            for i in range(len(trajectory) - 1):
                # Check if transition occurs in target year
                if trajectory[i]['year'] == year:
                    src_occ = trajectory[i]['occupation']
                    dst_occ = trajectory[i + 1]['occupation']
                    
                    if src_occ in self.occupation_to_idx and dst_occ in self.occupation_to_idx:
                        src_idx = self.occupation_to_idx[src_occ]
                        dst_idx = self.occupation_to_idx[dst_occ]
                        adj[src_idx, dst_idx] += 1
        
        # Normalize adjacency matrix
        adj_normalized = u.normalize_adj(adj)
        
        return adj_normalized
    
    def _build_node_features(self, year_df):
        """
        Build node features for given year
        
        Args:
            year_df: DataFrame for specific year
        
        Returns:
            Node feature matrix
        """
        # Initialize features
        features = torch.zeros((self.num_nodes, self.feats_per_node))
        
        # Aggregate features per occupation
        for occ in self.occupation_to_idx.keys():
            occ_df = year_df[year_df['onet_major_x'] == occ]
            
            if len(occ_df) == 0:
                continue
            
            idx = self.occupation_to_idx[occ]
            
            # Feature 1: Average wage (log scale)
            if 'log_wage_x' in occ_df.columns:
                features[idx, 0] = occ_df['log_wage_x'].mean()
            
            # Feature 2: Number of workers
            features[idx, 1] = len(occ_df)
            
            # Feature 3-6: Move type indicators
            if self.feats_per_node >= 7:
                features[idx, 2] = occ_df['move_1_1'].mean() if 'move_1_1' in occ_df.columns else 0
                features[idx, 3] = occ_df['move_1_2'].mean() if 'move_1_2' in occ_df.columns else 0
                features[idx, 4] = occ_df['move_2_1'].mean() if 'move_2_1' in occ_df.columns else 0
                features[idx, 5] = occ_df['move_2_2'].mean() if 'move_2_2' in occ_df.columns else 0
                features[idx, 6] = occ_df['up_move'].mean() if 'up_move' in occ_df.columns else 0
        
        return features
    
    def _extract_edges(self, adj_matrix):
        """
        Extract edges from adjacency matrix
        
        Args:
            adj_matrix: Adjacency matrix
        
        Returns:
            Edge list as tensor of shape (2, num_edges)
        """
        edge_indices = (adj_matrix > 0).nonzero(as_tuple=False)
        if len(edge_indices) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return edge_indices.t()
    
    def get_sample(self, idx, test_idx):
        """
        Get sample for training/validation/testing
        
        Args:
            idx: Time step index
            test_idx: Index for test time step
        
        Returns:
            Dictionary containing sample data
        """
        # Use history up to idx
        hist_adj_list = self.adj_matrices[:idx+1]
        hist_ndFeats_list = self.node_features[:idx+1]
        hist_mask_list = self.masks[:idx+1]
        
        # Get current snapshot for evaluation
        label_adj = self.adj_matrices[idx]
        
        return {
            'hist_adj_list': hist_adj_list,
            'hist_ndFeats_list': hist_ndFeats_list,
            'hist_mask_list': hist_mask_list,
            'label_adj': label_adj,
            'idx': idx
        }
    
    def get_num_nodes(self):
        """Get number of nodes"""
        return self.num_nodes
    
    def get_num_classes(self):
        """Get number of classes (for compatibility)"""
        return 2  # Binary link prediction
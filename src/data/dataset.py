"""
Career Trajectory Temporal Data Loader
Loads pre-built networks from build_network.py output for EvolveGCN-H training
"""
import torch
import numpy as np
import pandas as pd
import pickle
import json
import networkx as nx
from pathlib import Path
from collections import defaultdict

def make_sparse_eye(size):
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye

def normalize_adj(adj, num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by: 
        - adding an identity matrix, 
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    '''
    idx = adj['idx']
    vals = adj['vals']
    
    sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes]))
    
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor,dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)
    
    return {'idx': idx.t(), 'vals': vals}


class CareerTrajectoryDataset:
    """Dataset that loads pre-built temporal graphs from build_network.py"""
    
    def __init__(self, args):
        """
        Initialize dataset by loading pre-built networks
        
        Args:
            args: Arguments containing network paths and parameters
        """
        self.args = args
        self.network_dir = Path(args.network_dir)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load all networks
        self.networks_dict = self._load_networks()
        
        # Sort windows and build temporal sequence
        self.windows = sorted(self.networks_dict.keys())
        self.num_timesteps = len(self.windows)
        
        # Build occupation mapping
        self._build_occupation_mapping()
        
        # Convert NetworkX graphs to tensors
        self.feats_per_node = args.feats_per_node
        self.adj_matrices = []
        self.node_features = []
        self.masks = []
        self.edges = []
        
        self._convert_networks_to_tensors()
        
    def _load_metadata(self):
        """Load metadata.json from network output"""
        metadata_path = self.network_dir / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded network metadata:")
        print(f"  Created: {metadata['created_at']}")
        print(f"  Study period: {metadata['study_period']}")
        print(f"  Occupation column: {metadata['occupation_column']}")
        print(f"  Number of temporal networks: {metadata['n_years']}")
        
        return metadata
    
    def _load_networks(self):
        """Load all networks from pickle file"""
        networks_path = self.network_dir / 'networks_all.pkl'
        
        if not networks_path.exists():
            raise FileNotFoundError(f"Networks file not found: {networks_path}")
        
        with open(networks_path, 'rb') as f:
            networks = pickle.load(f)
        
        # Filter out None values (non-window entries)
        networks_dict = {k: v for k, v in networks.items() if not pd.isna(k) and v is not None}
        
        print(f"Loaded {len(networks_dict)} temporal networks")
        print(f"  Windows: {sorted(networks_dict.keys())}")
        
        return networks_dict
    
    def _build_occupation_mapping(self):
        """Build occupation to index mapping from all networks"""
        all_occupations = set()
        
        for window, G in self.networks_dict.items():
            all_occupations.update(G.nodes())
        
        all_occupations = sorted(list(all_occupations))
        
        self.occupation_to_idx = {occ: idx for idx, occ in enumerate(all_occupations)}
        self.idx_to_occupation = {idx: occ for occ, idx in self.occupation_to_idx.items()}
        self.num_nodes = len(all_occupations)
        
        print(f"Total unique occupations: {self.num_nodes}")
    
    def _convert_networks_to_tensors(self):
        """Convert NetworkX graphs to PyTorch tensors"""
        print("\nConverting networks to tensors...")
        
        for window in self.windows:
            G = self.networks_dict[window]
            
            # Build adjacency matrix
            adj_matrix = self._graph_to_adjacency_matrix(G)
            self.adj_matrices.append(adj_matrix)
            
            # Build node features
            node_feats = self._graph_to_node_features(G)
            self.node_features.append(node_feats)
            
            # Build node mask (all nodes active)
            mask = torch.ones(self.num_nodes, 1)
            self.masks.append(mask)
            
            # Extract edges
            edges = self._extract_edges(adj_matrix)
            self.edges.append(edges)
            
            print(f"  {window}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    def _graph_to_adjacency_matrix(self, G):
        """
        Convert NetworkX graph to adjacency matrix
        
        Args:
            G: NetworkX graph
        
        Returns:
            Normalized adjacency matrix tensor
        """
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        
        # Fill adjacency matrix from graph edges
        for u, v, data in G.edges(data=True):
            if u in self.occupation_to_idx and v in self.occupation_to_idx:
                u_idx = self.occupation_to_idx[u]
                v_idx = self.occupation_to_idx[v]
                weight = data.get('weight', 1.0)
                adj[u_idx, v_idx] = weight
        
        # Normalize adjacency matrix
        adj_normalized = normalize_adj(adj, self.num_nodes)
        
        return adj_normalized
    
    def _graph_to_node_features(self, G):
        """
        Extract node features from NetworkX graph
        
        Args:
            G: NetworkX graph
        
        Returns:
            Node feature matrix tensor
        """
        features = torch.zeros((self.num_nodes, self.feats_per_node))
        
        for node in G.nodes():
            if node not in self.occupation_to_idx:
                continue
            
            idx = self.occupation_to_idx[node]
            attrs = G.nodes[node]
            
            # Feature 0: Employment count (log scale)
            employment_count = attrs.get('employment_count', 0)
            features[idx, 0] = np.log1p(employment_count)
            
            # Feature 1: Average wage (log scale)
            if self.feats_per_node >= 2:
                avg_wage = attrs.get('avg_wage', None)
                if avg_wage is not None and not pd.isna(avg_wage):
                    features[idx, 1] = np.log1p(avg_wage)
        
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
            test_idx: Index for test time step (unused, for compatibility)
        
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
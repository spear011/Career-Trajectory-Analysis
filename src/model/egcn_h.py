"""
EvolveGCN-H model implementation
Adapted from original EvolveGCN repository for Career Trajectory link prediction
"""
import src.utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math


class EGCN_H(torch.nn.Module):
    """EvolveGCN-H model for temporal graph learning"""
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({'in_feats': feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        """
        Forward pass through EvolveGCN-H
        
        Args:
            A_list: List of adjacency matrices over time
            Nodes_list: List of node embeddings over time
            nodes_mask_list: List of node masks over time
        
        Returns:
            Final node embeddings
        """
        node_feats = Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)
        return out


class GRCU(torch.nn.Module):
    """Graph Recurrent Convolutional Unit"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats, self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        """Initialize parameters"""
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list, mask_list):
        """
        Forward pass through GRCU
        
        Args:
            A_list: List of adjacency matrices
            node_embs_list: List of node embeddings
            mask_list: List of node masks
        
        Returns:
            List of evolved node embeddings
        """
        GCN_weights = self.GCN_init_weights
        out_seq = []
        
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            node_mask = mask_list[t]
            
            # Evolve weights using GRU
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, node_mask)
            
            # Apply GCN with evolved weights
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    """Matrix GRU cell for evolving GCN weights"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q, prev_Z, mask):
        """
        Forward pass through matrix GRU cell
        
        Args:
            prev_Q: Previous weight matrix
            prev_Z: Previous node embeddings
            mask: Node mask
        
        Returns:
            New weight matrix
        """
        z_topk = self.choose_topk(prev_Z, mask)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    """Matrix GRU gate"""
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        """Initialize parameters"""
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        """Forward pass through gate"""
        out = self.activation(self.W.matmul(x) + 
                              self.U.matmul(hidden) + 
                              self.bias)
        return out


class TopK(torch.nn.Module):
    """TopK selection for node embeddings"""
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self, t):
        """Initialize parameters"""
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        """
        Select top-k nodes
        
        Args:
            node_embs: Node embeddings
            mask: Node mask
        
        Returns:
            Top-k node embeddings
        """
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + -1e9 * (1 - mask)
        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -1e8]
        
        if topk_indices.shape[0] < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)
            
        tanh = torch.nn.Tanh()
        
        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))
        
        return out.t()
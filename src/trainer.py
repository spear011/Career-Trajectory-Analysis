"""
Link Prediction Trainer for EvolveGCN-H on Career Trajectory Data
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import utils as u


class LinkPredictionTrainer:
    """Trainer for link prediction task"""
    
    def __init__(self, args, model, dataset, device):
        """
        Initialize trainer
        
        Args:
            args: Training arguments
            model: EvolveGCN-H model
            dataset: CareerTrajectoryDataset
            device: Device to use
        """
        self.args = args
        self.model = model
        self.dataset = dataset
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Early stopping
        self.early_stopping = u.EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=0
        )
        
    def _build_optimizer(self):
        """Build optimizer"""
        params = self.model.parameters()
        
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        return optimizer
    
    def _negative_sampling(self, pos_edges, num_nodes, num_neg_samples):
        """
        Generate negative samples for link prediction
        
        Args:
            pos_edges: Positive edges (2, num_pos_edges)
            num_nodes: Number of nodes
            num_neg_samples: Number of negative samples per positive edge
        
        Returns:
            Negative edges (2, num_neg_edges)
        """
        num_pos = pos_edges.shape[1]
        
        # Create set of positive edges
        pos_edge_set = set()
        for i in range(num_pos):
            pos_edge_set.add((pos_edges[0, i].item(), pos_edges[1, i].item()))
        
        # Sample negative edges
        neg_edges = []
        while len(neg_edges) < num_pos * num_neg_samples:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            
            if (src, dst) not in pos_edge_set and src != dst:
                neg_edges.append([src, dst])
        
        neg_edges = torch.tensor(neg_edges[:num_pos * num_neg_samples]).t()
        
        return neg_edges
    
    def _get_edge_scores(self, node_embs, edges):
        """
        Compute edge scores using dot product
        
        Args:
            node_embs: Node embeddings (num_nodes, embedding_dim)
            edges: Edges (2, num_edges)
        
        Returns:
            Edge scores (num_edges,)
        """
        src_embs = node_embs[edges[0]]
        dst_embs = node_embs[edges[1]]
        
        scores = (src_embs * dst_embs).sum(dim=1)
        
        return scores
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        # Train on all time steps except last few (reserved for validation/test)
        train_end = self.dataset.num_timesteps - self.args.num_val_steps - self.args.num_test_steps
        
        for t in range(self.args.num_hist_steps, train_end):
            sample = self.dataset.get_sample(t, None)
            
            # Move to device
            hist_adj_list = [adj.to(self.device) for adj in sample['hist_adj_list']]
            hist_ndFeats_list = [feats.to(self.device) for feats in sample['hist_ndFeats_list']]
            hist_mask_list = [mask.to(self.device) for mask in sample['hist_mask_list']]
            
            # Forward pass
            self.optimizer.zero_grad()
            node_embs = self.model(hist_adj_list, hist_ndFeats_list, hist_mask_list)
            
            # Get positive edges from label adjacency
            label_adj = sample['label_adj'].to(self.device)
            pos_edges = (label_adj > 0).nonzero(as_tuple=False).t()
            
            if pos_edges.shape[1] == 0:
                continue
            
            # Generate negative edges
            neg_edges = self._negative_sampling(
                pos_edges,
                self.dataset.num_nodes,
                self.args.neg_sample_ratio
            ).to(self.device)
            
            # Compute scores
            pos_scores = self._get_edge_scores(node_embs, pos_edges)
            neg_scores = self._get_edge_scores(node_embs, neg_edges)
            
            # Combine and compute loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).to(self.device)
            
            loss = self.loss_fn(scores, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_samples += 1
        
        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss
    
    def evaluate(self, split='val'):
        """
        Evaluate model on validation or test set
        
        Args:
            split: 'val' or 'test'
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Determine evaluation range
        if split == 'val':
            eval_start = self.dataset.num_timesteps - self.args.num_val_steps - self.args.num_test_steps
            eval_end = self.dataset.num_timesteps - self.args.num_test_steps
        else:  # test
            eval_start = self.dataset.num_timesteps - self.args.num_test_steps
            eval_end = self.dataset.num_timesteps
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for t in range(eval_start, eval_end):
                sample = self.dataset.get_sample(t, None)
                
                # Move to device
                hist_adj_list = [adj.to(self.device) for adj in sample['hist_adj_list']]
                hist_ndFeats_list = [feats.to(self.device) for feats in sample['hist_ndFeats_list']]
                hist_mask_list = [mask.to(self.device) for mask in sample['hist_mask_list']]
                
                # Forward pass
                node_embs = self.model(hist_adj_list, hist_ndFeats_list, hist_mask_list)
                
                # Get positive edges
                label_adj = sample['label_adj'].to(self.device)
                pos_edges = (label_adj > 0).nonzero(as_tuple=False).t()
                
                if pos_edges.shape[1] == 0:
                    continue
                
                # Generate negative edges
                neg_edges = self._negative_sampling(
                    pos_edges,
                    self.dataset.num_nodes,
                    self.args.neg_sample_ratio
                ).to(self.device)
                
                # Compute scores
                pos_scores = self._get_edge_scores(node_embs, pos_edges)
                neg_scores = self._get_edge_scores(node_embs, neg_edges)
                
                # Collect predictions
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([
                    torch.ones(pos_scores.shape[0]),
                    torch.zeros(neg_scores.shape[0])
                ])
                
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        return {
            'auc': auc,
            'ap': ap
        }
    
    def train(self):
        """
        Main training loop
        
        Returns:
            Best validation metrics
        """
        best_val_auc = 0
        best_epoch = 0
        
        for epoch in range(self.args.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate('val')
            
            # Check early stopping
            self.early_stopping(-val_metrics['auc'])
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                
                # Save best model
                if self.args.save_model:
                    u.save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        f"{self.args.save_dir}/best_model.pt"
                    )
            
            # Log progress
            if epoch % self.args.log_interval == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | Val AP: {val_metrics['ap']:.4f}")
            
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test
        if self.args.save_model:
            u.load_checkpoint(
                self.model,
                self.optimizer,
                f"{self.args.save_dir}/best_model.pt"
            )
        
        test_metrics = self.evaluate('test')
        
        print(f"\nBest epoch: {best_epoch}")
        print(f"Test AUC: {test_metrics['auc']:.4f} | Test AP: {test_metrics['ap']:.4f}")
        
        return {
            'best_epoch': best_epoch,
            'val_auc': best_val_auc,
            'test_auc': test_metrics['auc'],
            'test_ap': test_metrics['ap']
        }
"""
Link Prediction Trainer for EvolveGCN-H on Career Trajectory Data
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving"""
    
    def __init__(self, patience=7, min_delta=0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Call this method after each epoch with validation loss
        
        Args:
            val_loss: Current validation loss (lower is better)
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        filepath: Path to save checkpoint
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
    
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


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
        self.early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=0
        )
    
    def _move_adj_to_device(self, adj_dict):
        """
        Move sparse adjacency matrix dictionary to device
        
        Args:
            adj_dict: Dictionary with 'idx' and 'vals' keys
        
        Returns:
            Dictionary with tensors moved to device
        """
        return {
            'idx': adj_dict['idx'].to(self.device),
            'vals': adj_dict['vals'].to(self.device)
        }
        
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
    
    def _compute_ranking_metrics(self, scores, labels):
        """
        Compute ranking metrics: MAP and MRR
        
        Args:
            scores: Predicted scores
            labels: True labels
        
        Returns:
            Dictionary with MAP and MRR
        """
        # Sort by scores (descending)
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        # Mean Average Precision (MAP)
        precisions = []
        num_positive = 0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                num_positive += 1
                precision_at_i = num_positive / (i + 1)
                precisions.append(precision_at_i)
        
        map_score = np.mean(precisions) if precisions else 0.0
        
        # Mean Reciprocal Rank (MRR)
        first_positive_idx = np.where(sorted_labels == 1)[0]
        mrr_score = 1.0 / (first_positive_idx[0] + 1) if len(first_positive_idx) > 0 else 0.0
        
        return {
            'map': map_score,
            'mrr': mrr_score
        }
    
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
        batch_losses = []
        
        # Train on all time steps except last few (reserved for validation/test)
        train_end = self.dataset.num_timesteps - self.args.num_val_steps - self.args.num_test_steps
        
        print(f"\nEpoch {epoch:03d} Training:")
        print(f"  Training timesteps: {self.args.num_hist_steps} to {train_end-1}")
        
        for t in range(self.args.num_hist_steps, train_end):
            sample = self.dataset.get_sample(t, None)
            
            # Move to device - handle sparse adjacency matrices
            hist_adj_list = [self._move_adj_to_device(adj) for adj in sample['hist_adj_list']]
            hist_ndFeats_list = [feats.to(self.device) for feats in sample['hist_ndFeats_list']]
            hist_mask_list = [mask.to(self.device) for mask in sample['hist_mask_list']]
            
            # Forward pass
            self.optimizer.zero_grad()
            node_embs = self.model(hist_adj_list, hist_ndFeats_list, hist_mask_list)
            
            # Get positive edges from label adjacency
            label_adj = self._move_adj_to_device(sample['label_adj'])
            
            # Reconstruct dense adjacency for edge extraction
            label_idx = label_adj['idx']
            label_vals = label_adj['vals']
            if label_idx.shape[1] == 0:
                continue
            
            # Use indices as positive edges
            pos_edges = label_idx
            
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
            batch_losses.append(loss.item())
            num_samples += 1
            
            # Log every 10 timesteps
            if (t - self.args.num_hist_steps) % 10 == 0:
                print(f"    Timestep {t}: Loss={loss.item():.4f}, Pos edges={pos_edges.shape[1]}, Neg edges={neg_edges.shape[1]}")
        
        avg_loss = total_loss / max(num_samples, 1)
        print(f"  Epoch {epoch:03d} Summary: Avg Loss={avg_loss:.4f}, Batches={num_samples}")
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
        
        print(f"\n  Evaluating on {split} set:")
        print(f"    Timesteps: {eval_start} to {eval_end-1}")
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for t in range(eval_start, eval_end):
                sample = self.dataset.get_sample(t, None)
                
                # Move to device - handle sparse adjacency matrices
                hist_adj_list = [self._move_adj_to_device(adj) for adj in sample['hist_adj_list']]
                hist_ndFeats_list = [feats.to(self.device) for feats in sample['hist_ndFeats_list']]
                hist_mask_list = [mask.to(self.device) for mask in sample['hist_mask_list']]
                
                # Forward pass
                node_embs = self.model(hist_adj_list, hist_ndFeats_list, hist_mask_list)
                
                # Get positive edges from label adjacency
                label_adj = self._move_adj_to_device(sample['label_adj'])
                pos_edges = label_adj['idx']
                
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
        
        # Compute ranking metrics
        ranking_metrics = self._compute_ranking_metrics(all_scores, all_labels)
        map_score = ranking_metrics['map']
        mrr_score = ranking_metrics['mrr']
        
        print(f"    AUC: {auc:.4f}, AP: {ap:.4f}, MAP: {map_score:.4f}, MRR: {mrr_score:.4f}")
        print(f"    Total samples: {len(all_labels)}, Positives: {all_labels.sum():.0f}")
        
        return {
            'auc': auc,
            'ap': ap,
            'map': map_score,
            'mrr': mrr_score
        }
    
    def train(self):
        """
        Main training loop
        
        Returns:
            Best validation metrics
        """
        best_val_auc = 0
        best_epoch = 0
        
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80)
        print(f"Total epochs: {self.args.num_epochs}")
        print(f"Optimizer: {self.args.optimizer}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Early stopping patience: {self.args.early_stop_patience}")
        print("="*80)
        
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
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        f"{self.args.save_dir}/best_model.pt"
                    )
                    print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")
            
            # Log progress
            print(f"\nEpoch {epoch:03d} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUC: {val_metrics['auc']:.4f} | AP: {val_metrics['ap']:.4f} | MAP: {val_metrics['map']:.4f} | MRR: {val_metrics['mrr']:.4f}")
            print(f"  Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
            print(f"  Early Stopping Counter: {self.early_stopping.counter}/{self.early_stopping.patience}")
            print("-" * 80)
            
            if self.early_stopping.early_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model and evaluate on test
        print("\n" + "="*80)
        print("EVALUATING BEST MODEL ON TEST SET")
        print("="*80)
        
        if self.args.save_model:
            print(f"Loading best model from epoch {best_epoch}...")
            load_checkpoint(
                self.model,
                self.optimizer,
                f"{self.args.save_dir}/best_model.pt"
            )
        
        test_metrics = self.evaluate('test')
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best Epoch: {best_epoch}")
        print(f"Val AUC: {best_val_auc:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test AP: {test_metrics['ap']:.4f}")
        print(f"Test MAP: {test_metrics['map']:.4f}")
        print(f"Test MRR: {test_metrics['mrr']:.4f}")
        print("="*80)
        
        return {
            'best_epoch': best_epoch,
            'val_auc': best_val_auc,
            'test_auc': test_metrics['auc'],
            'test_ap': test_metrics['ap'],
            'test_map': test_metrics['map'],
            'test_mrr': test_metrics['mrr']
        }
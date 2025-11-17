"""
Main experiment runner for EvolveGCN-H on Career Trajectory data
"""
import argparse
import torch
import yaml
from pathlib import Path
import src.utils as u
from src.model.egcn_h import EGCN_H
from src.data.dataset import CareerTrajectoryDataset
from src.trainer import LinkPredictionTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EvolveGCN-H for Career Trajectory Link Prediction')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to configuration yaml file')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return u.Namespace(config)


def main():
    """Main experiment function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Set random seed
    u.set_random_seed(config.seed)
    
    # Set device
    device = torch.device('cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    u.create_dirs_if_not_exists([config.save_dir, config.log_dir])
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CareerTrajectoryDataset(config)
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of timesteps: {dataset.num_timesteps}")
    print(f"Time range: {dataset.min_year}-{dataset.max_year}")
    
    # Initialize model
    print("\nInitializing model...")
    model = EGCN_H(
        args=config,
        activation=torch.nn.ReLU(),
        device=device,
        skipfeats=config.skipfeats
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = LinkPredictionTrainer(
        args=config,
        model=model,
        dataset=dataset,
        device=device
    )
    
    # Train model
    print("\nStarting training...")
    print("="*80)
    results = trainer.train()
    print("="*80)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Validation AUC: {results['val_auc']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(f"Test AP: {results['test_ap']:.4f}")
    
    # Save results
    if config.save_results:
        results_path = Path(config.save_dir) / 'results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Training script for the upgraded RecessionGATTransformer model.

This script trains the new architecture with:
1. GAT spatial encoder for correlation graphs
2. Transformer temporal encoder for long-range dependencies
3. Regime detection for structural breaks
4. Early-warning loss for timely predictions
5. Macro anchor integration

Usage:
    python train_recession_gat_transformer.py --epochs 50 --seq-len 60
    python train_recession_gat_transformer.py --quick  # Fast experiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model.recession_gat_transformer import (
    RecessionGATTransformer,
    RecessionSequenceDataset,
    EarlyWarningLoss,
    train_recession_model,
    extract_attention_saliency,
    visualize_attention,
)
from model.gnn_correlation_learner import SpreadGraph


# NBER recession dates for labeling
NBER_RECESSIONS = [
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def create_recession_labels(dates: pd.DatetimeIndex) -> np.ndarray:
    """Create binary recession labels from NBER dates."""
    labels = np.zeros(len(dates), dtype=np.float32)
    
    for start, end in NBER_RECESSIONS:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        mask = (dates >= start_date) & (dates <= end_date)
        labels[mask] = 1.0
    
    return labels


def load_data(
    corr_npz: Path,
    anchor_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], np.ndarray]:
    """Load correlation tensor, macro anchors, and create recession labels."""
    # Load correlations
    corr_data = np.load(corr_npz, allow_pickle=True)
    dates = pd.to_datetime(corr_data["dates"].astype(str))
    spreads = corr_data["spreads"].astype(str).tolist()
    correlations = corr_data["corr"].astype(float)
    
    # Load macro anchors
    anchor_df = pd.read_csv(anchor_csv, parse_dates=["date"])
    anchor_df = anchor_df.set_index("date").sort_index()
    anchor_aligned = anchor_df.reindex(dates).ffill().bfill()
    anchors = anchor_aligned.values.astype(float)
    
    # Create recession labels
    recession_labels = create_recession_labels(dates)
    
    print(f"Loaded data:")
    print(f"  Dates: {len(dates)} ({dates.min().date()} to {dates.max().date()})")
    print(f"  Spreads: {len(spreads)}")
    print(f"  Correlations: {correlations.shape}")
    print(f"  Macro features: {anchors.shape[1]}")
    print(f"  Recession days: {recession_labels.sum():.0f} ({100*recession_labels.mean():.1f}%)")
    
    return correlations, anchors, dates, spreads, recession_labels


def train_and_evaluate(
    correlations: np.ndarray,
    anchors: np.ndarray,
    dates: pd.DatetimeIndex,
    spreads: List[str],
    recession_labels: np.ndarray,
    output_dir: Path,
    config: Dict,
    device: str = "cpu",
) -> Tuple[RecessionGATTransformer, Dict]:
    """
    Train the RecessionGATTransformer and evaluate.
    
    Returns: (trained_model, results_dict)
    """
    print("\n" + "=" * 60)
    print("TRAINING RECESSION GAT-TRANSFORMER")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Get feature dimensions from sample graph
    graph = SpreadGraph(spreads)
    sample_node_feat = graph.get_node_features(correlations[0])
    sample_edge_feat = graph.get_edge_features(correlations[0])
    
    n_node_features = sample_node_feat.shape[1]
    n_edge_features = sample_edge_feat.shape[1]
    n_macro_features = anchors.shape[1]
    
    print(f"\nFeature dimensions:")
    print(f"  Node features: {n_node_features}")
    print(f"  Edge features: {n_edge_features}")
    print(f"  Macro features: {n_macro_features}")
    
    # Create datasets
    horizons = [3, 6, 12, 24]
    
    # Split: 80% train, 20% val (time-based)
    split_idx = int(len(correlations) * 0.8)
    
    train_dataset = RecessionSequenceDataset(
        correlations=correlations[:split_idx],
        macro_anchors=anchors[:split_idx],
        recession_labels=recession_labels[:split_idx],
        spread_names=spreads,
        seq_len=config['seq_len'],
        horizons_months=horizons,
        stride=config['stride'],
    )
    
    val_dataset = RecessionSequenceDataset(
        correlations=correlations[split_idx:],
        macro_anchors=anchors[split_idx:],
        recession_labels=recession_labels[split_idx:],
        spread_names=spreads,
        seq_len=config['seq_len'],
        horizons_months=horizons,
        stride=config['stride'],
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create model
    model = RecessionGATTransformer(
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        n_macro_features=n_macro_features,
        hidden_dim=config['hidden_dim'],
        n_gat_heads=config['n_gat_heads'],
        n_gat_layers=config['n_gat_layers'],
        n_transformer_heads=config['n_transformer_heads'],
        n_transformer_layers=config['n_transformer_layers'],
        n_regimes=config['n_regimes'],
        n_horizons=len(horizons),
        dropout=config['dropout'],
        device=device,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Train
    model_path = output_dir / "recession_gat_transformer.pt"
    model, history = train_recession_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        device=device,
        save_path=model_path,
        patience=config['patience'],
    )
    
    # Save history
    history_path = output_dir / "gat_transformer_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Evaluate lead-time performance
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = evaluate_lead_time(
        model, val_dataset, dates[split_idx:], recession_labels[split_idx:],
        horizons, device
    )
    
    results['n_params'] = n_params
    results['final_train_loss'] = history['train_loss'][-1]
    results['final_val_loss'] = history['val_loss'][-1]
    results['best_val_loss'] = min(history['val_loss'])
    results['config'] = config
    
    # Save results
    results_path = output_dir / "gat_transformer_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")
    
    return model, results


def evaluate_lead_time(
    model: RecessionGATTransformer,
    dataset: RecessionSequenceDataset,
    dates: pd.DatetimeIndex,
    recession_labels: np.ndarray,
    horizons: List[int],
    device: str,
) -> Dict:
    """
    Evaluate model's lead-time performance.
    
    Measures:
    - How many months before each recession did P(recession) > 0.5?
    - AUC at each horizon
    - Precision/recall at each horizon
    """
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    
    model.eval()
    all_probs = []
    all_labels = []
    
    # Collect predictions
    with torch.no_grad():
        for i in range(min(len(dataset), 500)):  # Limit for speed
            sample = dataset[i]
            graphs = sample['graphs']
            macro = sample['macro'].unsqueeze(0).to(device)
            
            output = model(graphs, macro)
            probs = output['probs'][0, -1].cpu().numpy()  # Last timestep
            labels = sample['labels'].numpy()
            
            all_probs.append(probs)
            all_labels.append(labels)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    results = {}
    
    # Per-horizon metrics
    for h, horizon in enumerate(horizons):
        y_true = all_labels[:, h]
        y_pred = all_probs[:, h]
        
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auc = roc_auc_score(y_true, y_pred)
            
            y_pred_binary = (y_pred > 0.5).astype(int)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            
            results[f'auc_{horizon}mo'] = float(auc)
            results[f'precision_{horizon}mo'] = float(precision)
            results[f'recall_{horizon}mo'] = float(recall)
            
            print(f"  {horizon}-month horizon: AUC={auc:.3f}, P={precision:.3f}, R={recall:.3f}")
    
    # Average lead time for detected recessions
    # (This would require more complex analysis of when P > 0.5 before each recession)
    results['mean_auc'] = np.mean([v for k, v in results.items() if k.startswith('auc')])
    results['mean_recall'] = np.mean([v for k, v in results.items() if k.startswith('recall')])
    
    print(f"\n  Mean AUC: {results['mean_auc']:.3f}")
    print(f"  Mean Recall: {results['mean_recall']:.3f}")
    
    return results


def run_hyperparameter_experiment(
    correlations: np.ndarray,
    anchors: np.ndarray,
    dates: pd.DatetimeIndex,
    spreads: List[str],
    recession_labels: np.ndarray,
    output_dir: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run a small hyperparameter search experiment.
    """
    configs = [
        # Baseline
        {'hidden_dim': 32, 'n_gat_heads': 2, 'n_gat_layers': 2, 
         'n_transformer_heads': 2, 'n_transformer_layers': 2},
        # Larger
        {'hidden_dim': 64, 'n_gat_heads': 4, 'n_gat_layers': 2, 
         'n_transformer_heads': 4, 'n_transformer_layers': 3},
        # Deeper
        {'hidden_dim': 48, 'n_gat_heads': 4, 'n_gat_layers': 3, 
         'n_transformer_heads': 4, 'n_transformer_layers': 4},
        # More heads
        {'hidden_dim': 32, 'n_gat_heads': 8, 'n_gat_layers': 2, 
         'n_transformer_heads': 8, 'n_transformer_layers': 2},
    ]
    
    base_config = {
        'seq_len': 30,
        'stride': 10,
        'epochs': 20,  # Shorter for experiment
        'batch_size': 8,
        'lr': 1e-4,
        'n_regimes': 3,
        'dropout': 0.1,
        'patience': 10,
    }
    
    results_list = []
    
    for i, exp_config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        config = {**base_config, **exp_config}
        exp_dir = output_dir / f"experiment_{i+1}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            model, results = train_and_evaluate(
                correlations, anchors, dates, spreads, recession_labels,
                exp_dir, config, device
            )
            
            results['experiment'] = i + 1
            results_list.append(results)
            
        except Exception as e:
            print(f"Experiment {i+1} failed: {e}")
            results_list.append({
                'experiment': i + 1,
                'error': str(e),
                'config': config,
            })
    
    # Create comparison DataFrame
    df = pd.DataFrame(results_list)
    df.to_csv(output_dir / "experiment_comparison.csv", index=False)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    if 'mean_auc' in df.columns:
        print(df[['experiment', 'mean_auc', 'mean_recall', 'best_val_loss', 'n_params']].to_string())
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Train RecessionGATTransformer")
    parser.add_argument("--corr-npz", type=Path, 
                        default=Path("outputs/correlation_tensor_usa.npz"))
    parser.add_argument("--anchor-csv", type=Path,
                        default=Path("outputs/macro_anchors_daily.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gat_transformer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Training config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    
    # Model config
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-gat-heads", type=int, default=4)
    parser.add_argument("--n-gat-layers", type=int, default=2)
    parser.add_argument("--n-transformer-heads", type=int, default=4)
    parser.add_argument("--n-transformer-layers", type=int, default=3)
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Experiment modes
    parser.add_argument("--quick", action="store_true", help="Quick training for testing")
    parser.add_argument("--experiment", action="store_true", help="Run hyperparameter experiment")
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    correlations, anchors, dates, spreads, recession_labels = load_data(
        args.corr_npz, args.anchor_csv
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiment:
        # Run hyperparameter experiment
        run_hyperparameter_experiment(
            correlations, anchors, dates, spreads, recession_labels,
            args.output_dir, args.device
        )
    else:
        # Single training run
        config = {
            'seq_len': args.seq_len if not args.quick else 20,
            'stride': args.stride if not args.quick else 20,
            'epochs': args.epochs if not args.quick else 10,
            'batch_size': args.batch_size if not args.quick else 8,
            'lr': args.lr,
            'hidden_dim': args.hidden_dim if not args.quick else 32,
            'n_gat_heads': args.n_gat_heads if not args.quick else 2,
            'n_gat_layers': args.n_gat_layers,
            'n_transformer_heads': args.n_transformer_heads if not args.quick else 2,
            'n_transformer_layers': args.n_transformer_layers if not args.quick else 2,
            'n_regimes': args.n_regimes,
            'dropout': args.dropout,
            'patience': args.patience if not args.quick else 5,
        }
        
        train_and_evaluate(
            correlations, anchors, dates, spreads, recession_labels,
            args.output_dir, config, args.device
        )


if __name__ == "__main__":
    main()

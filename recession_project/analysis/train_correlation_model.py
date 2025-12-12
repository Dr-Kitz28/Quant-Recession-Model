#!/usr/bin/env python3
"""Training script for correlation learners.

Supports two architectures:
1. MLP (baseline): Flattens correlation matrix, loses graph structure
2. GNN (new): Treats spreads as nodes, correlations as edges

Usage:
    python train_correlation_model.py --model mlp  # baseline
    python train_correlation_model.py --model gnn  # graph-based
    python train_correlation_model.py --model both # compare both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model.correlation_weight_learner import (
    CorrelationSequenceDataset,
    CorrelationWeightLearner,
    train_model as train_mlp_model,
)
from model.gnn_correlation_learner import (
    GNNCorrelationLearner,
    GNNCorrelationDataset,
    train_gnn_correlation_learner,
    SpreadGraph,
)


def load_data(
    corr_npz: Path,
    anchor_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list]:
    """Load correlation tensor and macro anchors."""
    corr_data = np.load(corr_npz, allow_pickle=True)
    dates = pd.to_datetime(corr_data["dates"].astype(str))
    spreads = corr_data["spreads"].astype(str).tolist()
    correlations = corr_data["corr"].astype(float)

    anchor_df = pd.read_csv(anchor_csv, parse_dates=["date"])
    anchor_df = anchor_df.set_index("date").sort_index()
    anchor_aligned = anchor_df.reindex(dates).ffill().bfill()
    anchors = anchor_aligned.values.astype(float)

    print(f"Loaded data:")
    print(f"  Dates: {len(dates)} ({dates.min()} to {dates.max()})")
    print(f"  Spreads: {len(spreads)}")
    print(f"  Correlation matrix: {correlations.shape}")
    print(f"  Anchor features: {anchors.shape[1]}")
    
    return correlations, anchors, dates, spreads


def train_and_evaluate_mlp(
    correlations: np.ndarray,
    anchors: np.ndarray,
    spreads: list,
    output_dir: Path,
    device: str = "cpu",
) -> Dict:
    """Train MLP model and return metrics."""
    n_spreads = len(spreads)
    triu_idx = np.triu_indices(n_spreads, k=1)
    n_corr_features = len(triu_idx[0])
    n_anchor_features = anchors.shape[1]
    
    print(f"\n{'='*60}")
    print(f"Training MLP Correlation Learner")
    print(f"  Input features: {n_corr_features} correlations + {n_anchor_features} anchors = {n_corr_features + n_anchor_features}")
    print(f"  Output features: {n_corr_features} correlations")
    print(f"{'='*60}")
    
    model, history = train_mlp_model(
        correlations=correlations,
        anchors=anchors,
        n_epochs=50,
        batch_size=32,
        lr=1e-3,
        device=device,
        save_path=output_dir / "mlp_weight_learner.pt",
    )
    
    # Save history
    with open(output_dir / "mlp_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return {
        "model_type": "mlp",
        "n_params": sum(p.numel() for p in model.parameters()),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_mae": history["val_mae"][-1],
        "best_val_loss": min(history["val_loss"]),
    }


def train_and_evaluate_gnn(
    correlations: np.ndarray,
    anchors: np.ndarray,
    spreads: list,
    output_dir: Path,
    device: str = "cpu",
) -> Dict:
    """Train GNN model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training GNN Correlation Learner")
    print(f"  Nodes: {len(spreads)} spreads")
    print(f"  Edges: {len(spreads) * (len(spreads) - 1)} directed")
    print(f"  Anchor features: {anchors.shape[1]}")
    print(f"{'='*60}")
    
    model, history = train_gnn_correlation_learner(
        correlations=correlations,
        spread_names=spreads,
        anchors=anchors,
        n_epochs=50,
        lr=1e-3,
        device=device,
        save_path=output_dir / "gnn_weight_learner.pt",
    )
    
    # Save history
    with open(output_dir / "gnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return {
        "model_type": "gnn",
        "n_params": sum(p.numel() for p in model.parameters()),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_mae": history["val_mae"][-1],
        "best_val_loss": min(history["val_loss"]),
    }


def compare_models(mlp_metrics: Dict, gnn_metrics: Dict) -> None:
    """Print comparison of MLP vs GNN."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON: MLP vs GNN")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<25} {'MLP':<15} {'GNN':<15} {'Winner':<10}")
    print("-" * 65)
    
    metrics = [
        ("Parameters", "n_params", False),  # False = lower is better
        ("Final Train Loss", "final_train_loss", True),
        ("Final Val Loss", "final_val_loss", True),
        ("Final Val MAE", "final_val_mae", True),
        ("Best Val Loss", "best_val_loss", True),
    ]
    
    for name, key, lower_better in metrics:
        mlp_val = mlp_metrics[key]
        gnn_val = gnn_metrics[key]
        
        if lower_better:
            winner = "GNN ✓" if gnn_val < mlp_val else "MLP ✓"
        else:
            winner = "-"
        
        if isinstance(mlp_val, int):
            print(f"{name:<25} {mlp_val:<15,} {gnn_val:<15,} {winner:<10}")
        else:
            print(f"{name:<25} {mlp_val:<15.4f} {gnn_val:<15.4f} {winner:<10}")
    
    # Summary
    print(f"\n{'='*60}")
    val_improvement = (mlp_metrics["best_val_loss"] - gnn_metrics["best_val_loss"]) / mlp_metrics["best_val_loss"] * 100
    if val_improvement > 0:
        print(f"GNN improves validation loss by {val_improvement:.1f}%")
    else:
        print(f"MLP has {-val_improvement:.1f}% lower validation loss")
    
    print(f"\nGNN Advantages:")
    print(f"  - Captures tenor-based graph structure")
    print(f"  - Message passing propagates correlation shocks")
    print(f"  - Permutation equivariant (order of spreads doesn't matter)")
    print(f"\nMLP Advantages:")
    print(f"  - Faster training (no message passing)")
    print(f"  - Simpler architecture")


def main():
    parser = argparse.ArgumentParser(description="Train correlation learners")
    parser.add_argument(
        "--model", 
        choices=["mlp", "gnn", "both"], 
        default="both",
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--corr-npz",
        type=Path,
        default=Path("outputs/correlation_tensor_usa.npz"),
        help="Path to correlation tensor"
    )
    parser.add_argument(
        "--anchor-csv",
        type=Path,
        default=Path("outputs/macro_anchors.csv"),
        help="Path to macro anchors CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for models and history"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training"
    )
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    
    # Load data
    correlations, anchors, dates, spreads = load_data(args.corr_npz, args.anchor_csv)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    mlp_metrics = None
    gnn_metrics = None
    
    if args.model in ["mlp", "both"]:
        mlp_metrics = train_and_evaluate_mlp(
            correlations, anchors, spreads, args.output_dir, args.device
        )
        print(f"\nMLP Results: {mlp_metrics}")
    
    if args.model in ["gnn", "both"]:
        gnn_metrics = train_and_evaluate_gnn(
            correlations, anchors, spreads, args.output_dir, args.device
        )
        print(f"\nGNN Results: {gnn_metrics}")
    
    if args.model == "both" and mlp_metrics and gnn_metrics:
        compare_models(mlp_metrics, gnn_metrics)
        
        # Save comparison
        comparison = {
            "mlp": mlp_metrics,
            "gnn": gnn_metrics,
        }
        with open(args.output_dir / "model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()

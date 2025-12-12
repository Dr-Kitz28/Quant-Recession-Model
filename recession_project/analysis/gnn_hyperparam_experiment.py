#!/usr/bin/env python3
"""Run a small hyperparameter grid search for the GNN correlation learner.

This script runs short trainings (few epochs) across a small grid of
node_embed_dim, hidden_dim, and n_layers and records validation losses.

Usage:
    python analysis/gnn_hyperparam_experiment.py --corr-npz ../outputs/correlation_tensor_usa.npz --anchor-csv ../outputs/macro_anchors_daily.csv --output-dir ../outputs/experiments --epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from model.gnn_correlation_learner import train_gnn_correlation_learner


def load_data(corr_npz: Path, anchor_csv: Path):
    data = np.load(corr_npz, allow_pickle=True)
    dates = pd.to_datetime(data['dates'].astype(str))
    spreads = data['spreads'].astype(str).tolist()
    correlations = data['corr'].astype(float)

    anchor_df = pd.read_csv(anchor_csv, parse_dates=['date'])
    anchor_df = anchor_df.set_index('date').sort_index()
    anchor_aligned = anchor_df.reindex(dates).ffill().bfill()
    anchors = anchor_aligned.values.astype(float)

    return correlations, anchors, dates, spreads


def run_grid(correlations, anchors, spreads, output_dir: Path, epochs: int = 20, device: str = 'cpu') -> List[Dict]:
    grid = {
        'node_embed_dim': [32, 64],
        'hidden_dim': [64, 128],
        'n_layers': [2, 3],
    }

    combos = []
    for ned in grid['node_embed_dim']:
        for hd in grid['hidden_dim']:
            for nl in grid['n_layers']:
                combos.append({'node_embed_dim': ned, 'hidden_dim': hd, 'n_layers': nl})

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for combo in combos:
        name = f"gnn_ned{combo['node_embed_dim']}_hd{combo['hidden_dim']}_nl{combo['n_layers']}"
        print(f"Running {name}...")
        save_path = output_dir / f"{name}.pt"

        model, history = train_gnn_correlation_learner(
            correlations=correlations,
            spread_names=spreads,
            anchors=anchors,
            n_epochs=epochs,
            batch_size=1,
            lr=1e-3,
            val_split=0.2,
            device=device,
            save_path=save_path,
            node_embed_dim=combo['node_embed_dim'],
            hidden_dim=combo['hidden_dim'],
            n_layers=combo['n_layers'],
        )

        res = {
            'name': name,
            'node_embed_dim': combo['node_embed_dim'],
            'hidden_dim': combo['hidden_dim'],
            'n_layers': combo['n_layers'],
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
            'final_val_mae': history['val_mae'][-1] if history['val_mae'] else None,
        }
        results.append(res)

        # save per-experiment history
        with open(output_dir / f"{name}_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        # save summary so far
        with open(output_dir / "gnn_grid_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corr-npz', type=Path, default=Path('../outputs/correlation_tensor_usa.npz'))
    parser.add_argument('--anchor-csv', type=Path, default=Path('../outputs/macro_anchors_daily.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('../outputs/experiments'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    correlations, anchors, dates, spreads = load_data(args.corr_npz, args.anchor_csv)
    results = run_grid(correlations, anchors, spreads, args.output_dir, epochs=args.epochs, device=args.device)

    print('\nGrid search complete. Results:')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

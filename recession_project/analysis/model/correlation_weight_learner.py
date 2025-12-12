#!/usr/bin/env python3
"""Neural Network Weight Learner for Correlation Matrix Dynamics.

This module implements an MLP that predicts tomorrow's correlation matrix
from today's correlation matrix and macro anchor features. The learned
weights can be used to refine the AFNS forecast.

Architecture:
    Input:  [flattened upper-tri correlations (t)] + [macro anchors (t)]
    Hidden: 2-layer MLP with ReLU
    Output: [flattened upper-tri correlations (t+1)]

The model maintains an internal "weights table" that can be serialized
and used for inference without retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Dataset
# ============================================================================
class CorrelationSequenceDataset(Dataset):
    """Dataset for (corr_t, anchors_t) -> corr_{t+1} prediction."""

    def __init__(
        self,
        correlations: np.ndarray,  # (n_dates, n_spreads, n_spreads)
        anchors: np.ndarray,       # (n_dates, n_anchor_features)
        horizon: int = 1,
    ):
        self.correlations = correlations
        self.anchors = anchors
        self.horizon = horizon
        self.n_spreads = correlations.shape[1]

        # upper triangular indices (excluding diagonal)
        self.triu_idx = np.triu_indices(self.n_spreads, k=1)
        self.n_corr_features = len(self.triu_idx[0])

        # valid indices: need t and t+horizon both valid
        self.valid_indices = []
        for t in range(len(correlations) - horizon):
            corr_t = correlations[t][self.triu_idx]
            corr_next = correlations[t + horizon][self.triu_idx]
            if not (np.isnan(corr_t).any() or np.isnan(corr_next).any()):
                self.valid_indices.append(t)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.valid_indices[idx]

        # flatten upper-tri of correlation matrix at time t
        corr_t = self.correlations[t][self.triu_idx]
        anchor_t = self.anchors[t]
        anchor_t = np.nan_to_num(anchor_t, nan=0.0)

        # target: upper-tri at t+horizon
        corr_next = self.correlations[t + self.horizon][self.triu_idx]

        x = np.concatenate([corr_t, anchor_t]).astype(np.float32)
        y = corr_next.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================================
# Model
# ============================================================================
class CorrelationWeightLearner(nn.Module):
    """MLP that learns correlation matrix dynamics."""

    def __init__(
        self,
        n_corr_features: int,
        n_anchor_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_corr_features = n_corr_features
        self.n_anchor_features = n_anchor_features
        input_dim = n_corr_features + n_anchor_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_corr_features),
            nn.Tanh(),  # correlations are in [-1, 1]
        )

        # internal weights table for interpretability
        self._weights_table: Optional[Dict[str, np.ndarray]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def extract_weights_table(self) -> Dict[str, np.ndarray]:
        """Extract learned weights for analysis/serialization."""
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        self._weights_table = weights
        return weights

    def save_weights_table(self, path: Path) -> None:
        """Save weights table to NPZ file."""
        weights = self.extract_weights_table()
        np.savez_compressed(path, **weights)
        print(f"[weight_learner] Saved weights to {path}")

    def load_weights_table(self, path: Path) -> None:
        """Load weights from NPZ and update model parameters."""
        data = np.load(path)
        state = {}
        for key in data.files:
            state[key] = torch.from_numpy(data[key])
        self.load_state_dict(state)
        print(f"[weight_learner] Loaded weights from {path}")


# ============================================================================
# Trainer
# ============================================================================
class CorrelationWeightTrainer:
    """Training loop for the weight learner."""

    def __init__(
        self,
        model: CorrelationWeightLearner,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                mae = F.l1_loss(pred, y)
                total_loss += loss.item()
                total_mae += mae.item()
                n_batches += 1
        n = max(n_batches, 1)
        return total_loss / n, total_mae / n

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            })

            if verbose and epoch % 5 == 0:
                print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae={val_mae:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"[weight_learner] Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_loss": [h["train_loss"] for h in self.history],
            "val_loss": [h["val_loss"] for h in self.history],
            "val_mae": [h["val_mae"] for h in self.history],
        }


# ============================================================================
# Predictor (inference)
# ============================================================================
class CorrelationPredictor:
    """Inference wrapper for trained weight learner."""

    def __init__(self, model: CorrelationWeightLearner, n_spreads: int, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.n_spreads = n_spreads
        self.triu_idx = np.triu_indices(n_spreads, k=1)

    def predict_next_matrix(
        self,
        current_corr: np.ndarray,  # (n_spreads, n_spreads)
        anchors: np.ndarray,        # (n_anchor_features,)
    ) -> np.ndarray:
        """Predict the next correlation matrix."""
        self.model.eval()

        corr_flat = current_corr[self.triu_idx].astype(np.float32)
        anchors = np.nan_to_num(anchors, nan=0.0).astype(np.float32)
        x = np.concatenate([corr_flat, anchors])

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            pred = self.model(x_t).squeeze(0).cpu().numpy()

        # reconstruct full matrix
        next_corr = np.eye(self.n_spreads, dtype=np.float32)
        next_corr[self.triu_idx] = pred
        next_corr = next_corr + next_corr.T - np.diag(np.diag(next_corr))

        # clamp to valid correlation range
        next_corr = np.clip(next_corr, -1.0, 1.0)

        return next_corr

    def predict_sequence(
        self,
        initial_corr: np.ndarray,
        anchors_sequence: np.ndarray,  # (steps, n_anchor_features)
        steps: int,
    ) -> np.ndarray:
        """Predict a sequence of correlation matrices."""
        predictions = []
        current = initial_corr.copy()

        for t in range(min(steps, len(anchors_sequence))):
            next_corr = self.predict_next_matrix(current, anchors_sequence[t])
            predictions.append(next_corr)
            current = next_corr

        return np.array(predictions)


# ============================================================================
# Convenience training function
# ============================================================================
def train_model(
    correlations: np.ndarray,
    anchors: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
    save_path: Optional[Path] = None,
) -> Tuple[CorrelationWeightLearner, Dict]:
    """
    Convenience function to train MLP correlation learner.
    
    Returns: (trained_model, training_history)
    """
    n_spreads = correlations.shape[1]
    n_corr_features = len(np.triu_indices(n_spreads, k=1)[0])
    n_anchor_features = anchors.shape[1]
    
    # Create dataset
    dataset = CorrelationSequenceDataset(correlations, anchors)
    
    # Split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = CorrelationWeightLearner(
        n_corr_features=n_corr_features,
        n_anchor_features=n_anchor_features,
    )
    
    # Train
    trainer = CorrelationWeightTrainer(model, device=device, lr=lr)
    history = trainer.fit(train_loader, val_loader, epochs=n_epochs, verbose=True)
    
    # Save if path provided
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[weight_learner] Saved model to {save_path}")
    
    return model, history


# ============================================================================
# CLI / Main
# ============================================================================
def load_data(
    corr_npz_path: Path,
    anchor_csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]]:
    """Load correlation tensor and macro anchors."""
    # load correlation tensor
    corr_data = np.load(corr_npz_path, allow_pickle=True)
    dates = pd.to_datetime(corr_data["dates"].astype(str))
    spreads = corr_data["spreads"].astype(str).tolist()
    correlations = corr_data["corr"].astype(float)

    # load anchors
    anchor_df = pd.read_csv(anchor_csv_path, parse_dates=["date"])
    anchor_df = anchor_df.set_index("date").sort_index()

    # align to correlation dates
    anchor_aligned = anchor_df.reindex(dates).ffill().bfill()
    anchor_names = list(anchor_aligned.columns)
    anchors = anchor_aligned.values.astype(float)

    return correlations, anchors, dates, spreads, anchor_names


def main():
    parser = argparse.ArgumentParser(description="Train correlation weight learner")
    parser.add_argument("--corr-npz", type=Path, required=True, help="Path to correlation tensor NPZ")
    parser.add_argument("--anchor-csv", type=Path, required=True, help="Path to macro anchors CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (days)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    print(f"[weight_learner] Loading data...")
    correlations, anchors, dates, spreads, anchor_names = load_data(
        args.corr_npz, args.anchor_csv
    )
    print(f"[weight_learner] Correlations: {correlations.shape}, Anchors: {anchors.shape}")
    print(f"[weight_learner] Spreads: {len(spreads)}, Anchor features: {len(anchor_names)}")

    # create dataset
    dataset = CorrelationSequenceDataset(correlations, anchors, horizon=args.horizon)
    print(f"[weight_learner] Valid samples: {len(dataset)}")

    # train/val split
    n_train = int(len(dataset) * args.train_ratio)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # create model
    n_spreads = correlations.shape[1]
    n_corr_features = len(np.triu_indices(n_spreads, k=1)[0])
    n_anchor_features = anchors.shape[1]

    model = CorrelationWeightLearner(
        n_corr_features=n_corr_features,
        n_anchor_features=n_anchor_features,
        hidden_dim=args.hidden_dim,
    )
    print(f"[weight_learner] Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # train
    trainer = CorrelationWeightTrainer(model, device=args.device, lr=args.lr)
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs, verbose=True)

    # save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "correlation_weight_learner.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[weight_learner] Saved model to {model_path}")

    weights_path = args.output_dir / "correlation_weights_table.npz"
    model.save_weights_table(weights_path)

    history_path = args.output_dir / "weight_learner_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[weight_learner] Saved training history to {history_path}")

    # quick validation summary
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    final_val_mae = history["val_mae"][-1] if history["val_mae"] else float("nan")
    print(f"[weight_learner] Final val_loss={final_val_loss:.6f}, val_mae={final_val_mae:.6f}")


if __name__ == "__main__":
    main()

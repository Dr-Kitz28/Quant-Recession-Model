#!/usr/bin/env python3
"""Graph Neural Network for Correlation Matrix Learning.

Instead of flattening the correlation matrix to a 1D vector,
we model spreads as nodes in a graph where correlations are edge weights.

This captures:
1. Structural relationships (spreads sharing tenors are connected)
2. Message passing between related spreads
3. Permutation equivariance (order of spreads doesn't matter)

Architecture:
    Node features: [spread_mean, spread_std, tenor_short, tenor_long, position_in_curve]
    Edge features: [correlation_t, correlation_change, shared_tenor_flag]
    
    Graph → GNN layers → Node embeddings → Edge predictions (correlation_{t+1})
"""
from __future__ import annotations

import re
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Tenor Parsing Utilities
# =============================================================================

def parse_tenor_months(tenor_str: str) -> int:
    """Convert tenor string like '3M', '1Y', '10Y' to months."""
    tenor_str = tenor_str.upper().strip()
    
    if tenor_str.endswith('M'):
        return int(tenor_str[:-1])
    elif tenor_str.endswith('Y'):
        return int(tenor_str[:-1]) * 12
    else:
        # Try to parse as number (assume years if > 12, else months)
        try:
            val = int(tenor_str)
            return val * 12 if val <= 30 else val
        except ValueError:
            return 12  # default to 1Y


def parse_spread_tenors(spread_name: str) -> Tuple[int, int]:
    """Parse spread like '10Y-3M' into (long_months, short_months)."""
    parts = spread_name.split('-')
    if len(parts) != 2:
        return (12, 1)  # default
    
    long_tenor = parse_tenor_months(parts[0])
    short_tenor = parse_tenor_months(parts[1])
    
    return (long_tenor, short_tenor)


def compute_tenor_overlap(spread1: str, spread2: str) -> int:
    """Count shared tenors between two spreads. Returns 0, 1, or 2."""
    t1_long, t1_short = parse_spread_tenors(spread1)
    t2_long, t2_short = parse_spread_tenors(spread2)
    
    tenors1 = {t1_long, t1_short}
    tenors2 = {t2_long, t2_short}
    
    return len(tenors1 & tenors2)


# =============================================================================
# Graph Construction
# =============================================================================

class SpreadGraph:
    """
    Represents the spread correlation network as a graph.
    
    Nodes: Spreads (e.g., '10Y-3M')
    Edges: All pairwise connections (fully connected for correlation matrix)
    """
    
    def __init__(self, spread_names: List[str]):
        self.spread_names = spread_names
        self.n_nodes = len(spread_names)
        self.n_edges = self.n_nodes * (self.n_nodes - 1)  # directed edges
        
        # Parse tenor structure
        self.tenor_info = []
        for name in spread_names:
            long_m, short_m = parse_spread_tenors(name)
            self.tenor_info.append({
                'name': name,
                'long_months': long_m,
                'short_months': short_m,
                'duration': long_m - short_m,  # spread duration
                'midpoint': (long_m + short_m) / 2,  # curve position
            })
        
        # Build edge index (source, target) for all pairs
        # For undirected correlation, we need both directions for message passing
        src, dst = [], []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Precompute structural edge features (static)
        self.structural_edge_features = self._compute_structural_edge_features()
        
        # Upper triangular indices for output
        self.triu_idx = np.triu_indices(self.n_nodes, k=1)
        self.n_upper_tri = len(self.triu_idx[0])
    
    def _compute_structural_edge_features(self) -> torch.Tensor:
        """Compute static structural features for each edge."""
        features = []
        
        for src, dst in zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()):
            info_src = self.tenor_info[src]
            info_dst = self.tenor_info[dst]
            
            # Shared tenor count (0, 1, or 2)
            overlap = compute_tenor_overlap(info_src['name'], info_dst['name'])
            
            # Duration difference
            dur_diff = abs(info_src['duration'] - info_dst['duration'])
            dur_diff_norm = dur_diff / 360  # normalize by 30 years
            
            # Curve position difference
            pos_diff = abs(info_src['midpoint'] - info_dst['midpoint'])
            pos_diff_norm = pos_diff / 180  # normalize
            
            # Same short tenor (e.g., both X-1M spreads)
            same_short = float(info_src['short_months'] == info_dst['short_months'])
            
            # Same long tenor
            same_long = float(info_src['long_months'] == info_dst['long_months'])
            
            features.append([
                overlap / 2,  # normalize to [0, 1]
                dur_diff_norm,
                pos_diff_norm,
                same_short,
                same_long,
            ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_node_features(
        self, 
        correlations: np.ndarray,
        historical_spreads: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Compute node features from correlation matrix.
        
        Args:
            correlations: (n_nodes, n_nodes) correlation matrix
            historical_spreads: Optional (n_nodes,) spread values
        
        Returns:
            (n_nodes, n_node_features) tensor
        """
        features = []
        
        for i, info in enumerate(self.tenor_info):
            # Degree centrality: average correlation with others
            row = correlations[i, :]
            valid = ~np.isnan(row) & (np.arange(len(row)) != i)
            mean_corr = np.nanmean(row[valid]) if valid.sum() > 0 else 0.0
            
            # Correlation std (how variable are this node's correlations)
            std_corr = np.nanstd(row[valid]) if valid.sum() > 0 else 0.0
            
            # Max absolute correlation
            max_corr = np.nanmax(np.abs(row[valid])) if valid.sum() > 0 else 0.0
            
            # Tenor features
            long_norm = info['long_months'] / 360
            short_norm = info['short_months'] / 360
            duration_norm = info['duration'] / 360
            midpoint_norm = info['midpoint'] / 180
            
            node_feat = [
                mean_corr,
                std_corr,
                max_corr,
                long_norm,
                short_norm,
                duration_norm,
                midpoint_norm,
            ]
            
            # Add spread value if available
            if historical_spreads is not None:
                spread_val = historical_spreads[i] if not np.isnan(historical_spreads[i]) else 0.0
                node_feat.append(spread_val / 5.0)  # normalize (spreads typically < 5%)
            
            features.append(node_feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_edge_features(
        self, 
        correlations: np.ndarray,
        prev_correlations: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Compute dynamic edge features from correlation matrix.
        
        Args:
            correlations: (n_nodes, n_nodes) current correlation matrix
            prev_correlations: Optional (n_nodes, n_nodes) previous correlation
        
        Returns:
            (n_edges, n_edge_features) tensor
        """
        dynamic_features = []
        
        for src, dst in zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()):
            corr = correlations[src, dst]
            if np.isnan(corr):
                corr = 0.0
            
            # Correlation change
            if prev_correlations is not None:
                prev_corr = prev_correlations[src, dst]
                if np.isnan(prev_corr):
                    prev_corr = corr
                corr_change = corr - prev_corr
            else:
                corr_change = 0.0
            
            dynamic_features.append([
                corr,           # current correlation
                corr_change,    # momentum
                abs(corr),      # absolute magnitude
            ])
        
        dynamic = torch.tensor(dynamic_features, dtype=torch.float32)
        
        # Concatenate with structural features
        return torch.cat([self.structural_edge_features, dynamic], dim=1)
    
    def correlation_matrix_to_edge_targets(self, corr_matrix: np.ndarray) -> torch.Tensor:
        """Extract target correlations for all directed edges."""
        targets = []
        for src, dst in zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()):
            corr = corr_matrix[src, dst]
            targets.append(corr if not np.isnan(corr) else 0.0)
        return torch.tensor(targets, dtype=torch.float32)
    
    def edge_predictions_to_matrix(self, edge_preds: torch.Tensor) -> np.ndarray:
        """Convert edge predictions back to correlation matrix."""
        matrix = np.eye(self.n_nodes)
        
        preds = edge_preds.detach().cpu().numpy()
        
        for idx, (src, dst) in enumerate(zip(self.edge_index[0].tolist(), 
                                              self.edge_index[1].tolist())):
            matrix[src, dst] = preds[idx]
        
        # Symmetrize (average of both directions)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        
        return np.clip(matrix, -1, 1)


# =============================================================================
# GNN Model Components
# =============================================================================

class EdgeConv(nn.Module):
    """
    Edge-conditioned convolution layer.
    
    Updates node embeddings by aggregating messages from neighbors,
    where messages are conditioned on edge features.
    """
    
    def __init__(
        self, 
        node_dim: int, 
        edge_dim: int, 
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Message MLP: combines source node, target node, and edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update MLP: updates target node with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.node_dim = node_dim
    
    def forward(
        self, 
        x: torch.Tensor,           # (n_nodes, node_dim)
        edge_index: torch.Tensor,  # (2, n_edges)
        edge_attr: torch.Tensor,   # (n_edges, edge_dim)
    ) -> torch.Tensor:
        """
        Perform message passing.
        
        Returns: (n_nodes, node_dim) updated node embeddings
        """
        src, dst = edge_index
        n_nodes = x.size(0)
        n_edges = edge_index.size(1)
        
        # Gather source and target node features for each edge
        x_src = x[src]  # (n_edges, node_dim)
        x_dst = x[dst]  # (n_edges, node_dim)
        
        # Compute messages
        edge_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        messages = self.message_mlp(edge_input)  # (n_edges, hidden_dim)
        
        # Aggregate messages to target nodes (mean aggregation)
        aggregated = torch.zeros(n_nodes, messages.size(1), device=x.device)
        counts = torch.zeros(n_nodes, 1, device=x.device)
        
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand(-1, messages.size(1)), messages)
        counts.scatter_add_(0, dst.unsqueeze(1), torch.ones(n_edges, 1, device=x.device))
        
        aggregated = aggregated / (counts + 1e-6)
        
        # Update node embeddings
        update_input = torch.cat([x, aggregated], dim=1)
        x_new = self.update_mlp(update_input)
        
        # Residual connection
        return x + x_new


class EdgePredictor(nn.Module):
    """
    Predicts edge values (correlations) from node embeddings.
    
    Uses source and target node embeddings plus edge features.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Correlations in [-1, 1]
        )
    
    def forward(
        self, 
        x: torch.Tensor,           # (n_nodes, node_dim)
        edge_index: torch.Tensor,  # (2, n_edges)
        edge_attr: torch.Tensor,   # (n_edges, edge_dim)
    ) -> torch.Tensor:
        """Predict correlation for each edge."""
        src, dst = edge_index
        
        x_src = x[src]
        x_dst = x[dst]
        
        edge_input = torch.cat([x_src, x_dst, edge_attr], dim=1)
        pred = self.mlp(edge_input).squeeze(-1)  # (n_edges,)
        
        return pred


# =============================================================================
# Main GNN Model
# =============================================================================

class GNNCorrelationLearner(nn.Module):
    """
    Graph Neural Network for learning correlation matrix dynamics.
    
    Architecture:
        1. Embed node features
        2. N layers of message passing (EdgeConv)
        3. Predict next-day correlations using edge predictor
        
    The model learns:
        - How correlation shocks propagate through the tenor structure
        - Which spreads are leading indicators for others
        - Systemic vs. idiosyncratic correlation changes
    """
    
    def __init__(
        self,
        n_node_features: int = 7,
        n_edge_features: int = 8,  # 5 structural + 3 dynamic
        node_embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        n_anchor_features: int = 0,
    ):
        super().__init__()
        
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_anchor_features = n_anchor_features
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(n_node_features, node_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Optional: incorporate macro anchors via global conditioning
        if n_anchor_features > 0:
            self.anchor_embed = nn.Sequential(
                nn.Linear(n_anchor_features, node_embed_dim),
                nn.ReLU(),
            )
        else:
            self.anchor_embed = None
        
        # GNN layers
        self.conv_layers = nn.ModuleList([
            EdgeConv(node_embed_dim, n_edge_features, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_embed_dim)
            for _ in range(n_layers)
        ])
        
        # Edge predictor
        self.edge_predictor = EdgePredictor(
            node_embed_dim, n_edge_features, hidden_dim
        )
    
    def forward(
        self,
        node_features: torch.Tensor,   # (n_nodes, n_node_features)
        edge_index: torch.Tensor,      # (2, n_edges)
        edge_attr: torch.Tensor,       # (n_edges, n_edge_features)
        anchors: Optional[torch.Tensor] = None,  # (n_anchor_features,)
    ) -> torch.Tensor:
        """
        Forward pass to predict next-day correlations.
        
        Returns: (n_edges,) predicted correlations
        """
        # Embed nodes
        x = self.node_embed(node_features)
        
        # Add global macro conditioning if available
        if anchors is not None and self.anchor_embed is not None:
            anchor_embed = self.anchor_embed(anchors)
            x = x + anchor_embed.unsqueeze(0)  # broadcast to all nodes
        
        # Message passing layers
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
        
        # Predict edges
        edge_preds = self.edge_predictor(x, edge_index, edge_attr)
        
        return edge_preds


# =============================================================================
# Dataset for Training
# =============================================================================

class GNNCorrelationDataset(Dataset):
    """
    Dataset for training GNN correlation learner.
    
    Each sample: (current graph state) → (next-day correlations)
    """
    
    def __init__(
        self,
        correlations: np.ndarray,  # (n_dates, n_spreads, n_spreads)
        spread_names: List[str],
        anchors: Optional[np.ndarray] = None,  # (n_dates, n_anchor_features)
        lookback: int = 1,  # Use previous correlation for change feature
    ):
        self.correlations = correlations
        self.anchors = anchors
        self.lookback = lookback
        self.n_dates = len(correlations)
        
        # Build graph structure
        self.graph = SpreadGraph(spread_names)
        
        # Valid indices (need lookback history and next-day target)
        self.valid_indices = list(range(lookback, self.n_dates - 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.valid_indices[idx]
        
        current_corr = self.correlations[t]
        prev_corr = self.correlations[t - 1]
        next_corr = self.correlations[t + 1]
        
        # Node features
        node_features = self.graph.get_node_features(current_corr)
        
        # Edge features (structural + dynamic)
        edge_features = self.graph.get_edge_features(current_corr, prev_corr)
        
        # Target: next-day correlations
        target = self.graph.correlation_matrix_to_edge_targets(next_corr)
        
        # Macro anchors
        if self.anchors is not None:
            anchors = torch.tensor(self.anchors[t], dtype=torch.float32)
        else:
            anchors = torch.zeros(1)
        
        return {
            'node_features': node_features,
            'edge_index': self.graph.edge_index,
            'edge_features': edge_features,
            'target': target,
            'anchors': anchors,
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_gnn_correlation_learner(
    correlations: np.ndarray,
    spread_names: List[str],
    anchors: Optional[np.ndarray] = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    node_embed_dim: int = 64,
    hidden_dim: int = 128,
    n_layers: int = 3,
) -> Tuple[GNNCorrelationLearner, Dict]:
    """
    Train the GNN correlation learner.
    
    Returns: (trained_model, training_history)
    """
    from pathlib import Path
    
    print(f"[GNN] Training on {len(correlations)} samples, {len(spread_names)} spreads")
    
    # Create dataset
    dataset = GNNCorrelationDataset(correlations, spread_names, anchors)
    
    # Train/val split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    
    # Custom collate (graphs have fixed structure, so just stack)
    def collate_fn(batch):
        # For simplicity, process one graph at a time (no batching across graphs)
        # In production, you'd want to batch multiple time steps
        return batch[0]  # Just return single sample
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Get feature dimensions from first sample
    sample = dataset[0]
    n_node_features = sample['node_features'].shape[1]
    n_edge_features = sample['edge_features'].shape[1]
    n_anchor_features = sample['anchors'].shape[0] if anchors is not None else 0
    
    print(f"  Node features: {n_node_features}")
    print(f"  Edge features: {n_edge_features}")
    print(f"  Anchor features: {n_anchor_features}")
    
    # Initialize model
    model = GNNCorrelationLearner(
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        node_embed_dim=node_embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_anchor_features=n_anchor_features if n_anchor_features > 1 else 0,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            node_feat = batch['node_features'].to(device)
            edge_idx = batch['edge_index'].to(device)
            edge_feat = batch['edge_features'].to(device)
            target = batch['target'].to(device)
            anch = batch['anchors'].to(device) if n_anchor_features > 1 else None
            
            pred = model(node_feat, edge_idx, edge_feat, anch)
            
            # MSE loss
            loss = F.mse_loss(pred, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for batch in val_loader:
                node_feat = batch['node_features'].to(device)
                edge_idx = batch['edge_index'].to(device)
                edge_feat = batch['edge_features'].to(device)
                target = batch['target'].to(device)
                anch = batch['anchors'].to(device) if n_anchor_features > 1 else None
                
                pred = model(node_feat, edge_idx, edge_feat, anch)
                
                loss = F.mse_loss(pred, target)
                mae = F.l1_loss(pred, target)
                
                val_losses.append(loss.item())
                val_maes.append(mae.item())
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_mae = np.mean(val_maes)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_mae'].append(avg_mae)
        
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, val_mae={avg_mae:.4f}")
        
        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    
    return model, history


# =============================================================================
# Predictor Class for Inference
# =============================================================================

class GNNCorrelationPredictor:
    """
    Wrapper for making predictions with trained GNN model.
    """
    
    def __init__(
        self,
        model: GNNCorrelationLearner,
        spread_names: List[str],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.graph = SpreadGraph(spread_names)
    
    def predict_next_matrix(
        self,
        current_corr: np.ndarray,
        prev_corr: Optional[np.ndarray] = None,
        anchors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict next-day correlation matrix.
        
        Args:
            current_corr: (n_spreads, n_spreads) current correlation matrix
            prev_corr: Optional previous correlation for momentum feature
            anchors: Optional macro anchor features
        
        Returns:
            (n_spreads, n_spreads) predicted correlation matrix
        """
        if prev_corr is None:
            prev_corr = current_corr
        
        with torch.no_grad():
            node_feat = self.graph.get_node_features(current_corr).to(self.device)
            edge_idx = self.graph.edge_index.to(self.device)
            edge_feat = self.graph.get_edge_features(current_corr, prev_corr).to(self.device)
            
            if anchors is not None:
                anch = torch.tensor(anchors, dtype=torch.float32).to(self.device)
            else:
                anch = None
            
            edge_preds = self.model(node_feat, edge_idx, edge_feat, anch)
            
            matrix = self.graph.edge_predictions_to_matrix(edge_preds)
        
        return matrix


if __name__ == "__main__":
    # Quick test
    print("Testing GNN Correlation Learner...")
    
    # Fake data
    n_dates = 100
    n_spreads = 10
    spread_names = [f"{t}Y-1M" for t in [1, 2, 3, 5, 7, 10, 20, 30]] + ["10Y-2Y", "30Y-10Y"]
    
    # Random correlation matrices
    correlations = np.random.randn(n_dates, n_spreads, n_spreads)
    for t in range(n_dates):
        correlations[t] = (correlations[t] + correlations[t].T) / 2
        np.fill_diagonal(correlations[t], 1)
        correlations[t] = np.clip(correlations[t], -1, 1)
    
    # Test graph construction
    graph = SpreadGraph(spread_names)
    print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"Structural edge features shape: {graph.structural_edge_features.shape}")
    
    # Test dataset
    dataset = GNNCorrelationDataset(correlations, spread_names)
    sample = dataset[0]
    print(f"Sample node features: {sample['node_features'].shape}")
    print(f"Sample edge features: {sample['edge_features'].shape}")
    print(f"Sample target: {sample['target'].shape}")
    
    # Test model
    model = GNNCorrelationLearner(
        n_node_features=sample['node_features'].shape[1],
        n_edge_features=sample['edge_features'].shape[1],
    )
    
    pred = model(
        sample['node_features'],
        sample['edge_index'],
        sample['edge_features'],
    )
    print(f"Model output: {pred.shape}")
    
    print("\nTest passed!")

"""Recession predictor (CSV input) - LSTM / MLP approach.

This module implements a small training & inference pipeline that consumes a
correlation-tensor CSV (long or wide) and trains a model to predict upcoming
recession flags. The implementation is intentionally compact and self-contained
– it's a working baseline you can expand later.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


class CorrelationTensorDataset(Dataset):
    def __init__(self, correlations: np.ndarray, labels: np.ndarray, window_size: int = 20, stride: int = 1):
        self.correlations = correlations
        self.labels = labels
        self.window_size = window_size
        self.stride = stride

        self.indices = []
        for i in range(0, len(correlations) - window_size, stride):
            if not np.isnan(labels[i + window_size]):
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        corr_window = self.correlations[i : i + self.window_size]
        corr_window = np.nan_to_num(corr_window, nan=0.0)

        n_spreads = corr_window.shape[1]
        triu = np.triu_indices(n_spreads, k=1)
        features = np.stack([m[triu] for m in corr_window], axis=0)
        label = float(self.labels[i + self.window_size])
        return torch.FloatTensor(features), torch.FloatTensor([label])


class LSTMRecessionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        x = F.relu(self.fc1(last_hidden))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x


class RecessionPredictor:
    def __init__(self, model_type: str = "lstm", window_size: int = 20, forecast_horizon: int = 6):
        self.model_type = model_type
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CSV loaders
    def load_csv_long_format(self, csv_path: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        df = pd.read_csv(csv_path)
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        spread1_col = [c for c in df.columns if 'spread1' in c.lower() or c.lower() in ['spread1', 'spread_1', 'from']][0]
        spread2_col = [c for c in df.columns if 'spread2' in c.lower() or c.lower() in ['spread2', 'spread_2', 'to']][0]
        corr_col = [c for c in df.columns if 'corr' in c.lower() or c.lower() in ['correlation', 'value', 'corr']][0]

        df[date_col] = pd.to_datetime(df[date_col])
        dates = pd.DatetimeIndex(sorted(df[date_col].unique()))

        spread_names = sorted(list(set(df[spread1_col].unique()) | set(df[spread2_col].unique())))
        n_spreads = len(spread_names)
        spread_to_idx = {s: i for i, s in enumerate(spread_names)}

        correlations = np.full((len(dates), n_spreads, n_spreads), np.nan)

        for _, row in df.iterrows():
            date_idx = dates.get_loc(pd.Timestamp(row[date_col]))
            s1 = row[spread1_col]
            s2 = row[spread2_col]
            corr_value = float(row[corr_col])
            i1 = spread_to_idx[s1]
            i2 = spread_to_idx[s2]
            correlations[date_idx, i1, i2] = corr_value
            correlations[date_idx, i2, i1] = corr_value

        for i in range(len(dates)):
            np.fill_diagonal(correlations[i], 1.0)

        return correlations, dates, spread_names

    def load_csv_wide_format(self, csv_path: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        df = pd.read_csv(csv_path)
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col])
        dates = pd.DatetimeIndex(df[date_col])

        corr_cols = [c for c in df.columns if c != date_col]

        spread_pairs = []
        for col in corr_cols:
            if '_vs_' in col:
                s1, s2 = col.split('_vs_')
            elif '|' in col:
                s1, s2 = col.split('|')
            elif '_' in col:
                parts = col.split('_')
                s1, s2 = '_'.join(parts[:len(parts)//2]), '_'.join(parts[len(parts)//2:])
            else:
                continue
            spread_pairs.append((s1.strip(), s2.strip()))

        spread_names = sorted(list(set([s for pair in spread_pairs for s in pair])))
        n_spreads = len(spread_names)
        spread_to_idx = {s: i for i, s in enumerate(spread_names)}

        correlations = np.full((len(dates), n_spreads, n_spreads), np.nan)

        for date_idx in range(len(dates)):
            for col_idx, (s1, s2) in enumerate(spread_pairs):
                try:
                    corr_value = float(df.iloc[date_idx][corr_cols[col_idx]])
                except Exception:
                    continue
                i1 = spread_to_idx[s1]
                i2 = spread_to_idx[s2]
                correlations[date_idx, i1, i2] = corr_value
                correlations[date_idx, i2, i1] = corr_value

        for i in range(len(dates)):
            np.fill_diagonal(correlations[i], 1.0)

        return correlations, dates, spread_names

    def load_csv_auto_detect(self, csv_path: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        df = pd.read_csv(csv_path, nrows=5)
        has_spread_cols = any('spread' in c.lower() for c in df.columns)
        if has_spread_cols and len(df.columns) <= 5:
            return self.load_csv_long_format(csv_path)
        return self.load_csv_wide_format(csv_path)

    def create_labels(self, dates: pd.DatetimeIndex, recession_periods: list) -> np.ndarray:
        labels = np.zeros(len(dates))
        for start, end in recession_periods:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            for i, date in enumerate(dates):
                months_until_recession = (s - date).days / 30.44
                if 0 <= months_until_recession <= self.forecast_horizon:
                    labels[i] = 1
                if s <= date <= e:
                    labels[i] = 1
        return labels

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, lr: float = 1e-3):
        sample_batch, _ = next(iter(train_loader))
        input_dim = sample_batch.shape[2]

        if self.model_type == 'lstm':
            self.model = LSTMRecessionModel(input_dim)
        else:
            flattened_dim = sample_batch.shape[1] * sample_batch.shape[2]
            self.model = SimpleMLPModel(flattened_dim)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_val = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(1, len(train_loader))

            # validation
            self.model.eval()
            val_loss = 0.0
            preds = []
            labs = []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    out = self.model(X)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    preds.extend(out.cpu().numpy())
                    labs.extend(y.cpu().numpy())
            val_loss /= max(1, len(val_loader))
            try:
                auc = roc_auc_score(np.array(labs).flatten(), np.array(preds).flatten())
            except Exception:
                auc = 0.0

            print(f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={auc:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'best_recession_model.pt')

        self.model.load_state_dict(torch.load('best_recession_model.pt'))

    def predict(self, correlations: np.ndarray) -> np.ndarray:
        self.model.eval()
        dummy_labels = np.zeros(len(correlations))
        ds = CorrelationTensorDataset(correlations, dummy_labels, self.window_size)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                out = self.model(X)
                preds.extend(out.cpu().numpy())

        full = np.full(len(correlations), np.nan)
        full[self.window_size : self.window_size + len(preds)] = np.array(preds).flatten()
        return full


def get_us_recessions() -> list:
    return [
        ('2001-03-01', '2001-11-01'),
        ('2007-12-01', '2009-06-01'),
        ('2020-02-01', '2020-04-01'),
    ]


def parse_args():
    p = argparse.ArgumentParser(description='CSV-based recession predictor')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--format', choices=['auto', 'long', 'wide'], default='auto')
    p.add_argument('--model', choices=['lstm', 'mlp'], default='lstm')
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', type=str, default='recession_model.pt')
    return p.parse_args()


def main():
    args = parse_args()
    predictor = RecessionPredictor(model_type=args.model, window_size=args.window, forecast_horizon=args.horizon)
    if args.format == 'auto':
        correlations, dates, spread_names = predictor.load_csv_auto_detect(str(args.data))
    elif args.format == 'long':
        correlations, dates, spread_names = predictor.load_csv_long_format(str(args.data))
    else:
        correlations, dates, spread_names = predictor.load_csv_wide_format(str(args.data))

    labels = predictor.create_labels(dates, get_us_recessions())

    print(f"Data shape: {correlations.shape}, spreads: {len(spread_names)}")
    print(f"Recession positive labels: {labels.sum()}/{len(labels)}")

    dataset = CorrelationTensorDataset(correlations, labels, window_size=args.window)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    predictor.train(train_loader, val_loader, epochs=args.epochs)
    torch.save(predictor.model.state_dict(), args.output)
    print(f"Saved model to {args.output}")

    preds = predictor.predict(correlations)
    out_df = pd.DataFrame({'date': dates, 'recession_prob': preds, 'actual': labels})
    out_df.to_csv('recession_predictions.csv', index=False)
    print('Wrote recession_predictions.csv')


if __name__ == '__main__':
    main()

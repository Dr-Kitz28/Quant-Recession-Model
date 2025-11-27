#!/usr/bin/env python3
"""Macro-anchor conditioned KnewImp-style imputation for spread correlations."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from catalog_knewimp_folders import build_catalog, write_catalog  # type: ignore
from sklearn.neighbors import NearestNeighbors


def slugify(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text.lower()


def load_macro_series(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(root.rglob("*.csv")):
        rel_parts = csv_path.relative_to(root).parts
        if not rel_parts:
            continue
        category = slugify(rel_parts[0])
        series = slugify(csv_path.stem)
        alias_prefix = f"{category}__{series}"

        raw = pd.read_csv(csv_path)
        cols = [c.strip() for c in raw.columns]
        raw.columns = cols
        date_col = next((c for c in cols if "date" in c.lower()), None)
        if date_col is None:
            continue
        value_cols = [c for c in cols if c != date_col]
        if not value_cols:
            continue
        raw[date_col] = pd.to_datetime(raw[date_col])
        for value in value_cols:
            series_name = alias_prefix if len(value_cols) == 1 else f"{alias_prefix}__{slugify(value)}"
            subset = raw[[date_col, value]].rename(columns={date_col: "date", value: series_name})
            subset[series_name] = pd.to_numeric(subset[series_name], errors="coerce")
            frames.append(subset)

    if not frames:
        raise RuntimeError(f"No macro anchor CSVs found under {root}")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date")
    merged = merged.drop_duplicates(subset="date")
    merged = merged.set_index("date")

    full_index = pd.date_range(merged.index.min(), merged.index.max(), freq="D")
    merged = merged.reindex(full_index)
    merged.index.name = "date"
    merged = merged.ffill().bfill()
    merged = merged.dropna(axis=1, how="all")
    return merged


def add_yield_curve_anchors(macro_df: pd.DataFrame, bond_csv: Path) -> pd.DataFrame:
    yields = pd.read_csv(bond_csv)
    usa = yields[yields["country"].str.upper() == "USA"].copy()
    usa["date"] = pd.to_datetime(usa["date"])
    pivot = (
        usa.pivot_table(index="date", columns="tenor", values="yield_pct", aggfunc="mean")
        .sort_index()
    )
    factors = pd.DataFrame(index=pivot.index)
    long_cols = [c for c in ["10Y", "20Y", "30Y"] if c in pivot.columns]
    if long_cols:
        factors["anchor_level"] = pivot[long_cols].mean(axis=1)
    if {"10Y", "3M"}.issubset(pivot.columns):
        factors["anchor_slope_10y_3m"] = pivot["10Y"] - pivot["3M"]
    if {"5Y", "3M", "10Y"}.issubset(pivot.columns):
        factors["anchor_curvature"] = 2 * pivot["5Y"] - pivot["3M"] - pivot["10Y"]

    combined = macro_df.join(factors, how="outer")
    combined = combined.sort_index().ffill().bfill()
    combined = combined.dropna(axis=1, how="all")
    return combined


def load_correlation_dataframe(npz_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    data = np.load(npz_path, allow_pickle=True)
    dates = pd.to_datetime(data["dates"].astype(str))
    spreads = data["spreads"].astype(str).tolist()
    corr = data["corr"].astype(float)
    feature_names = [f"corr::{row}__{col}" for row in spreads for col in spreads]
    flat = corr.reshape(len(dates), -1)
    df = pd.DataFrame(flat, columns=feature_names)
    df.insert(0, "date", dates)
    df = df.sort_values("date").set_index("date")
    return df, spreads, feature_names


def assemble_dataset(
    macro_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    macro_aligned = macro_df.reindex(corr_df.index).ffill().bfill()
    dataset = pd.concat([macro_aligned, corr_df], axis=1)
    return dataset, macro_aligned


class KnewImpImputer:
    """Kernelised entropy-regularised imputation guided by anchor features."""

    def __init__(
        self,
        bandwidth: float = 5.0,
        epsilon: float = 1e-4,
        max_iter: int = 50,
        tol: float = 1e-4,
        n_neighbors: int = 64,
    ) -> None:
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors

    def fit_transform(self, anchors: np.ndarray, data: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        X = data.astype(np.float32).copy()
        mask = observed_mask.astype(bool)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(~mask)
        X[inds] = np.take(col_means, inds[1])
        if np.isnan(X).any():
            raise RuntimeError("Failed to initialise KnewImp imputations; column means contained NaN")

        n_samples = X.shape[0]
        if n_samples < 2:
            return X
        neighbor_count = min(self.n_neighbors, n_samples - 1)
        if neighbor_count <= 0:
            neighbor_count = 1

        nn = NearestNeighbors(n_neighbors=neighbor_count + 1)
        nn.fit(anchors)
        distances, indices = nn.kneighbors(anchors)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        weights = np.exp(-distances ** 2 / (2 * self.bandwidth ** 2))
        weights = weights + self.epsilon
        weights = weights / weights.sum(axis=1, keepdims=True)

        for iteration in range(self.max_iter):
            updated = np.empty_like(X)
            for row in range(n_samples):
                neighbor_vals = X[indices[row]]
                updated[row] = weights[row] @ neighbor_vals
            prev = X.copy()
            X[~mask] = updated[~mask]
            delta = np.nanmax(np.abs(X - prev))
            print(f"[knewimp] iter={iteration+1:02d} delta={delta:.6f}")
            if delta < self.tol:
                break
        return X.astype(np.float64)


def ensure_psd(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def reconstruct_and_write(
    imputed_corr: pd.DataFrame,
    spreads: Sequence[str],
    dates: Sequence[pd.Timestamp],
    knewimp_root: Path,
) -> None:
    knewimp_root = knewimp_root.expanduser().resolve()
    for idx, date in enumerate(dates, start=1):
        folder = knewimp_root / f"{idx:05d}_{date.date()}"
        folder.mkdir(parents=True, exist_ok=True)
        row = imputed_corr.iloc[idx - 1].to_numpy(dtype=float)
        mat = row.reshape(len(spreads), len(spreads))
        mat = np.clip(mat, -1.0, 1.0)
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        mat = ensure_psd(mat)
        mat = np.clip(mat, -1.0, 1.0)
        df = pd.DataFrame(mat, index=spreads, columns=spreads)
        df.to_csv(folder / "imputed_correlation_matrix.csv", float_format="%.6f")


def standardise_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = df.mean(axis=0)
    std = df.std(axis=0).replace(0, 1.0)
    scaled = (df - mean) / std
    return scaled, mean, std


def destandardise_frame(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return df * std + mean


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Impute spread correlation gaps using MacroAnchor-informed KnewImp")
    parser.add_argument("--macro-root", type=Path, default=Path("MacroAnchors"))
    parser.add_argument("--bond-data", type=Path, default=Path("bond_market_data.csv"))
    parser.add_argument("--correlation-npz", type=Path, default=Path("outputs/correlation_tensor_usa.npz"))
    parser.add_argument("--knewimp-root", type=Path, default=Path("analysis/KnewIMP_Spreads"))
    parser.add_argument("--macro-cache", type=Path, default=Path("outputs/macro_anchors_daily.csv"))
    parser.add_argument("--dataset-cache", type=Path, default=Path("outputs/macro_anchor_correlation_dataset.pkl"))
    parser.add_argument("--catalog-csv", type=Path, default=Path("analysis/KnewIMP_Spreads_catalog.csv"))
    parser.add_argument("--bandwidth", type=float, default=7.5, help="RBF bandwidth for the kernel weights")
    parser.add_argument("--neighbors", type=int, default=64, help="Number of macro-similar days to blend per update")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tol", type=float, default=5e-5)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--date-limit", type=int, default=0, help="Optional cap on the number of trading days to process (for smoke tests)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    macro_df = load_macro_series(args.macro_root)
    macro_df = add_yield_curve_anchors(macro_df, args.bond_data)
    args.macro_cache.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_csv(args.macro_cache)
    print(f"[macro] Wrote anchor table to {args.macro_cache}")

    corr_df, spreads, corr_cols = load_correlation_dataframe(args.correlation_npz)
    if args.date_limit and args.date_limit > 0:
        corr_df = corr_df.iloc[: args.date_limit]
        spreads = spreads  # unchanged but keep comment? (no change needed)
    dataset, macro_aligned = assemble_dataset(macro_df, corr_df)
    args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pickle(args.dataset_cache)
    print(f"[dataset] Cached macro+correlation frame to {args.dataset_cache}")

    anchor_cols = macro_df.columns.tolist()
    corr_cols_full = [c for c in dataset.columns if c.startswith("corr::")]

    imputer = KnewImpImputer(
        bandwidth=args.bandwidth,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        tol=args.tol,
        n_neighbors=args.neighbors,
    )
    anchor_scaled, _, _ = standardise_frame(macro_aligned)
    corr_scaled, corr_mean, corr_std = standardise_frame(corr_df)
    observed_mask = ~corr_df.isna().to_numpy()

    completed_corr = imputer.fit_transform(
        anchor_scaled.to_numpy(),
        corr_scaled.to_numpy(),
        observed_mask,
    )
    completed_corr = destandardise_frame(
        pd.DataFrame(completed_corr, index=corr_df.index, columns=corr_df.columns),
        corr_mean,
        corr_std,
    )

    imputed_corr = completed_corr[corr_cols_full]
    reconstruct_and_write(imputed_corr, spreads, dataset.index.to_list(), args.knewimp_root)
    print(f"[knewimp] Wrote imputed matrices into {args.knewimp_root}")

    summaries = build_catalog(args.knewimp_root)
    write_catalog(summaries, args.catalog_csv)
    print(f"[catalog] Updated catalog at {args.catalog_csv}")


if __name__ == "__main__":
    main()

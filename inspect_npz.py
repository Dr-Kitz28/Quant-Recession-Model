#!/usr/bin/env python3
"""Detailed inspection of the correlations NPZ file.

This script loads the NPZ archive and provides detailed statistics, slices,
and visualizations of the correlation tensor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

def inspect_npz(npz_path: str) -> None:
    """Load and inspect the NPZ file contents."""
    if not Path(npz_path).exists():
        print(f"Error: {npz_path} not found.")
        return

    data = np.load(npz_path, allow_pickle=True)
    print(f"Inspecting: {npz_path}")
    print("=" * 50)

    # Dates
    dates = pd.to_datetime(data["dates"].astype(str))
    print(f"Dates: {len(dates)} entries")
    print(f"  Range: {dates.min()} to {dates.max()}")
    print(f"  Sample: {dates[:5].tolist()} ... {dates[-5:].tolist()}")
    print()

    # Spreads
    spreads = data["spreads"].astype(str)
    print(f"Spreads: {len(spreads)} entries")
    print(f"  Sample: {spreads[:10].tolist()} ...")
    print()

    # Correlation tensors
    corr_raw = data["corr"]
    corr_scaled = data["corr_scaled"]
    print(f"Correlation tensor shape: {corr_raw.shape} (dates × spreads × spreads)")
    print(f"  Raw corr dtype: {corr_raw.dtype}, range: [{np.nanmin(corr_raw):.3f}, {np.nanmax(corr_raw):.3f}]")
    print(f"  Scaled corr dtype: {corr_scaled.dtype}, range: [{np.nanmin(corr_scaled):.3f}, {np.nanmax(corr_scaled):.3f}]")
    print()

    # NaN statistics
    nan_count_raw = np.isnan(corr_raw).sum()
    nan_count_scaled = np.isnan(corr_scaled).sum()
    total_elements = corr_raw.size
    print(f"NaN statistics:")
    print(f"  Raw: {nan_count_raw}/{total_elements} ({100*nan_count_raw/total_elements:.1f}%)")
    print(f"  Scaled: {nan_count_scaled}/{total_elements} ({100*nan_count_scaled/total_elements:.1f}%)")
    print()

    # Latest valid slice
    valid_mask = ~np.isnan(corr_raw)
    last_valid_idx = np.where(valid_mask.any(axis=(1,2)))[0]
    if last_valid_idx.size > 0:
        idx = last_valid_idx[-1]
        print(f"Latest valid date: {dates[idx]} (index {idx})")
        slice_raw = corr_raw[idx]
        slice_scaled = corr_scaled[idx]
        print(f"  Raw slice stats: mean={np.nanmean(slice_raw):.3f}, std={np.nanstd(slice_raw):.3f}")
        print(f"  Scaled slice stats: mean={np.nanmean(slice_scaled):.3f}, std={np.nanstd(slice_scaled):.3f}")
        print(f"  Sample correlations (first 5x5):")
        print(slice_raw[:5, :5])
    else:
        print("No valid correlations found.")
    print()

    # Most correlated pairs (latest slice)
    if last_valid_idx.size > 0:
        idx = last_valid_idx[-1]
        slice_raw = corr_raw[idx]
        triu_mask = np.triu(np.ones_like(slice_raw, dtype=bool), k=1)
        flat_corr = slice_raw[triu_mask]
        flat_pairs = [(i, j) for i in range(len(spreads)) for j in range(i+1, len(spreads))]
        
        valid_pairs = [(pair, corr) for pair, corr in zip(flat_pairs, flat_corr) if not np.isnan(corr)]
        if valid_pairs:
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
            print("Top 5 most positive correlations (latest slice):")
            for (i, j), corr in valid_pairs[:5]:
                print(f"  {spreads[i]} vs {spreads[j]}: {corr:.3f}")
            print()
            print("Top 5 most negative correlations (latest slice):")
            valid_pairs.sort(key=lambda x: x[1])
            for (i, j), corr in valid_pairs[:5]:
                print(f"  {spreads[i]} vs {spreads[j]}: {corr:.3f}")
        else:
            print("No valid pairs in latest slice.")

if __name__ == "__main__":
    inspect_npz("outputs/correlations_usa.npz")
#!/usr/bin/env python3
"""Create correlation_tensor_usa.npz from spreads_usa.csv if the NPZ is missing.

This script reads `outputs/spreads_usa.csv`, pivots to wide, computes a rolling
correlation tensor using the same logic as the main generator, and writes the
compressed NPZ used by the visualization script.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUTS = Path(__file__).resolve().parents[1] / 'outputs'
SPREADS_CSV = OUTPUTS / 'spreads_usa.csv'
NPZ_OUT = OUTPUTS / 'correlation_tensor_usa.npz'

WINDOW = 60
MIN_PERIODS = 20


def build_correlation_cube(spreads: pd.DataFrame, window: int, min_periods: int):
    spreads = spreads.sort_index()
    dates = spreads.index
    spread_names = list(spreads.columns)
    n_dates = len(dates)
    n_spreads = len(spread_names)

    corr_cube = np.full((n_dates, n_spreads, n_spreads), np.nan, dtype=float)

    for idx in range(n_dates):
        start = max(0, idx - window + 1)
        window_df = spreads.iloc[start: idx + 1]
        window_df = window_df.dropna(how='all')
        if window_df.shape[0] < min_periods:
            continue
        corr = window_df.corr()
        corr_cube[idx, :, :] = corr.to_numpy(dtype=float)

    corr_scaled = (corr_cube + 1.0) / 2.0
    corr_scaled = np.clip(corr_scaled, 0.0, 1.0, out=corr_scaled)
    return corr_cube, corr_scaled


if __name__ == '__main__':
    if not SPREADS_CSV.exists():
        raise SystemExit(f"spreads CSV not found: {SPREADS_CSV}")

    print(f"Reading spreads from {SPREADS_CSV}")
    df = pd.read_csv(SPREADS_CSV, parse_dates=['date'])
    # only USA rows
    if 'country' in df.columns:
        df = df[df['country'] == 'USA']

    # pivot to wide
    wide = df.pivot_table(index='date', columns='spread_name', values='spread_bp', aggfunc='mean')
    wide = wide.sort_index()

    print(f"Pivoted to wide: {wide.shape[0]} dates x {wide.shape[1]} spreads")

    corr_cube, corr_scaled = build_correlation_cube(wide, WINDOW, MIN_PERIODS)

    dates = wide.index.astype(str).to_numpy(dtype=object)
    spreads = np.array(list(wide.columns), dtype=object)

    NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_OUT, dates=dates, spreads=spreads, corr=corr_cube, corr_scaled=corr_scaled)
    print(f"Wrote {NPZ_OUT} (dates={len(dates)}, spreads={len(spreads)})")

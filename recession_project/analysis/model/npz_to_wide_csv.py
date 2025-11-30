#!/usr/bin/env python3
"""Convert a correlation NPZ (dates, spreads, corr or corr_scaled) into
one wide CSV per date row with columns like '2Y-3M_vs_5Y-3M'.

This is a tiny utility to bridge the NPZ artefacts produced by the
pipeline to the CSV-based recession predictor.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def npz_to_wide(npz_path: Path, out_csv: Path, use_scaled: bool = False):
    data = np.load(npz_path, allow_pickle=True)
    dates = pd.to_datetime(data["dates"].astype(str))
    spreads = data["spreads"].astype(str)
    corr = data["corr_scaled"] if use_scaled else data["corr"]

    # Upper triangle pairs
    n = len(spreads)
    pairs = []
    cols = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            cols.append(f"{spreads[i]}_vs_{spreads[j]}")

    rows = []
    for t in range(corr.shape[0]):
        row = {}
        for (i, j), col in zip(pairs, cols):
            val = corr[t, i, j]
            # keep NaNs as-is
            row[col] = float(val) if not np.isnan(val) else ''
        row["Date"] = str(dates[t].date())
        rows.append(row)

    df = pd.DataFrame(rows)
    # place date first
    cols_out = ["Date"] + [c for c in df.columns if c != "Date"]
    df = df[cols_out]
    df.to_csv(out_csv, index=False)
    return out_csv


def parse_args():
    p = argparse.ArgumentParser(description="Convert correlation NPZ to wide CSV for predictor.")
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--use-scaled', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    out = npz_to_wide(args.npz, args.output, use_scaled=args.use_scaled)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

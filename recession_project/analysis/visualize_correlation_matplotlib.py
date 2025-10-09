#!/usr/bin/env python3
"""Static correlation visualisation using Matplotlib.

This script complements the Plotly-based viewer by emitting PNG images for daily
heatmaps, full-timeline heatmaps, or 3D scatter volumes (spread i × spread j ×
date) using Matplotlib. Colour mapping matches the red↔blue gradient used in the
interactive version, and NaN entries are rendered in white.

Examples
--------
# Single-day correlation heatmap
python analysis/visualize_correlation_matplotlib.py \
    --npz outputs/correlations_usa.npz \
    --mode daily-heatmap \
    --from-date 2020-03-16 --to-date 2020-03-16 \
    --output outputs/corr_usa_2020-03-16.png

# Full timeline heatmap
python analysis/visualize_correlation_matplotlib.py \
    --npz outputs/correlations_usa.npz \
    --mode timeline-heatmap \
    --output outputs/corr_usa_timeline.png

# 3D scatter volume (static)
python analysis/visualize_correlation_matplotlib.py \
    --npz outputs/correlations_usa.npz \
    --mode volume \
    --from-date 2020-01-01 --to-date 2020-12-31 \
    --output outputs/corr_usa_volume.png
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# Build red (0) → blue (1) gradient, matching the Plotly version.
RB_COLORS = LinearSegmentedColormap.from_list("RedBlue", [(0.0, "#ff0000"), (1.0, "#0000ff")])
RB_COLORS.set_bad(color="white")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render correlation artefacts with Matplotlib.")
    parser.add_argument("--npz", type=Path, required=True, help="Correlation archive (.npz).")
    parser.add_argument(
        "--mode",
        choices=["auto", "daily-heatmap", "timeline-heatmap", "volume"],
        default="auto",
        help="Visualisation mode. 'auto' chooses daily heatmap for single date else volume.",
    )
    parser.add_argument("--from-date", dest="from_date", type=str, default=None, help="Start date (yyyy-mm-dd).")
    parser.add_argument("--to-date", dest="to_date", type=str, default=None, help="End date (yyyy-mm-dd).")
    parser.add_argument("--output", type=Path, required=True, help="Output image path (PNG).")
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots-per-inch for saved figure (default 200).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=10.0,
        help="Marker size for 3D scatter volume (default 10).",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=500,
        help="Maximum number of timeline samples to render in 3D volume (default 500).",
    )
    return parser.parse_args(argv)


def load_tensor(path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    dates = pd.to_datetime(data["dates"].astype(str))
    spreads = data["spreads"].astype(str)
    corr = data["corr"]
    corr_scaled = data["corr_scaled"]
    return dates, spreads, corr, corr_scaled


def subset_tensor(
    dates: pd.DatetimeIndex,
    corr: np.ndarray,
    corr_scaled: np.ndarray,
    start: datetime | None,
    end: datetime | None,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    if not mask.any():
        raise ValueError("Date filter excludes all observations.")
    return dates[mask], corr[mask], corr_scaled[mask]


def render_daily_heatmap(
    output: Path,
    dpi: int,
    date: pd.Timestamp,
    spreads: np.ndarray,
    corr_scaled: np.ndarray,
    corr_raw: np.ndarray,
) -> None:
    matrix = corr_scaled[0]
    raw = corr_raw[0]
    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(masked, cmap=RB_COLORS, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr (scaled)")
    ax.set_xticks(np.arange(len(spreads)))
    ax.set_yticks(np.arange(len(spreads)))
    ax.set_xticklabels(spreads, rotation=45, ha="right")
    ax.set_yticklabels(spreads)
    ax.set_title(f"Correlation matrix on {date.strftime('%Y-%m-%d')}")

    if len(spreads) <= 12:
        for i in range(len(spreads)):
            for j in range(len(spreads)):
                val = raw[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def render_timeline_heatmap(
    output: Path,
    dpi: int,
    dates: pd.DatetimeIndex,
    spreads: np.ndarray,
    corr_scaled: np.ndarray,
    corr_raw: np.ndarray,
) -> None:
    n_spreads = len(spreads)
    tri_i, tri_j = np.triu_indices(n_spreads)
    pair_labels = [f"{spreads[i]} vs {spreads[j]}" for i, j in zip(tri_i, tri_j)]

    max_pairs = 2000
    if len(pair_labels) > max_pairs:
        print(f"[warn] Reducing spread pairs from {len(pair_labels)} to {max_pairs} for timeline heatmap.")
        col_idx = np.linspace(0, len(pair_labels) - 1, max_pairs).round().astype(int)
        tri_i = tri_i[col_idx]
        tri_j = tri_j[col_idx]
        pair_labels = [pair_labels[i] for i in col_idx]

    values_scaled = corr_scaled[:, tri_i, tri_j].astype(np.float32, copy=False)

    max_dates = 2000
    if len(dates) > max_dates:
        print(f"[warn] Reducing timeline from {len(dates)} to {max_dates} samples for render.")
        row_idx = np.linspace(0, len(dates) - 1, max_dates).round().astype(int)
        values_scaled = values_scaled[row_idx]
        dates = dates[row_idx]

    labels = pair_labels

    masked = np.ma.masked_invalid(values_scaled)

    fig_width = max(10.0, len(labels) * 0.35)
    fig_height = 12.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(masked, aspect="auto", cmap=RB_COLORS, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr (scaled)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    if len(dates) <= 40:
        positions = np.arange(len(dates))
        labels_y = [d.strftime("%Y-%m-%d") for d in dates]
    else:
        tick_count = min(40, len(dates))
        positions = np.linspace(0, len(dates) - 1, tick_count)
        labels_y = [dates[int(round(p))].strftime("%Y-%m-%d") for p in positions]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_ylabel("Date")
    ax.set_xlabel("Spread pair")
    ax.set_title("Correlation timeline heatmap")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def render_volume(
    output: Path,
    dpi: int,
    dates: pd.DatetimeIndex,
    spreads: np.ndarray,
    corr_scaled: np.ndarray,
    corr_raw: np.ndarray,
    marker_size: float,
    max_dates: int,
) -> None:
    n_dates = len(dates)
    n_spreads = len(spreads)

    if n_dates > max_dates:
        print(f"[warn] Reducing volume timeline from {n_dates} to {max_dates} samples for render.")
        idx = np.linspace(0, n_dates - 1, max_dates).round().astype(int)
        dates = dates.take(idx)
        corr_scaled = corr_scaled[idx]
        corr_raw = corr_raw[idx]
        n_dates = len(dates)

    spread_idx = np.arange(n_spreads)
    date_idx = np.arange(n_dates)
    grid_x, grid_y = np.meshgrid(spread_idx, spread_idx, indexing="ij")
    grid_x = np.broadcast_to(grid_x, (n_dates, n_spreads, n_spreads))
    grid_y = np.broadcast_to(grid_y, (n_dates, n_spreads, n_spreads))
    grid_z = np.broadcast_to(date_idx[:, None, None], (n_dates, n_spreads, n_spreads))

    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)
    flat_z = grid_z.reshape(-1)
    values_scaled = corr_scaled.reshape(-1).astype(np.float32, copy=False)

    mask_nan = np.isnan(values_scaled)
    mask_valid = ~mask_nan

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        flat_x[mask_valid],
        flat_y[mask_valid],
        flat_z[mask_valid],
        c=values_scaled[mask_valid],
        cmap=RB_COLORS,
        vmin=0,
        vmax=1,
        s=marker_size,
        depthshade=False,
    )

    if mask_nan.any():
        ax.scatter(
            flat_x[mask_nan],
            flat_y[mask_nan],
            flat_z[mask_nan],
            c="white",
            edgecolors="grey",
            s=marker_size,
            depthshade=False,
        )

    ax.set_xticks(spread_idx)
    ax.set_xticklabels(spreads, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(spread_idx)
    ax.set_yticklabels(spreads, fontsize=8)

    tick_positions = np.linspace(0, n_dates - 1, min(10, n_dates)).round().astype(int)
    tick_positions = np.unique(tick_positions)
    ax.set_zticks(tick_positions)
    ax.set_zticklabels([dates[i].strftime("%Y-%m-%d") for i in tick_positions], rotation=45, fontsize=8)
    ax.set_xlabel("Spread i")
    ax.set_ylabel("Spread j")
    ax.set_zlabel("Date")
    ax.set_title("Correlation volume (static)")

    mappable = plt.cm.ScalarMappable(cmap=RB_COLORS)
    mappable.set_array([0, 1])
    fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1, label="corr (scaled)")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dates, spreads, corr_raw, corr_scaled = load_tensor(args.npz)
    start = pd.to_datetime(args.from_date) if args.from_date else None
    end = pd.to_datetime(args.to_date) if args.to_date else None
    sub_dates, sub_corr_raw, sub_corr_scaled = subset_tensor(dates, corr_raw, corr_scaled, start, end)

    mode = args.mode
    if mode == "auto":
        mode = "daily-heatmap" if len(sub_dates) == 1 else "volume"

    if mode == "daily-heatmap":
        if len(sub_dates) != 1:
            raise ValueError("daily-heatmap mode requires from-date == to-date and a single observation.")
        render_daily_heatmap(args.output, args.dpi, sub_dates[0], spreads, sub_corr_scaled, sub_corr_raw)
    elif mode == "timeline-heatmap":
        render_timeline_heatmap(args.output, args.dpi, sub_dates, spreads, sub_corr_scaled, sub_corr_raw)
    elif mode == "volume":
        render_volume(
            args.output,
            args.dpi,
            sub_dates,
            spreads,
            sub_corr_scaled,
            sub_corr_raw,
            args.marker_size,
            args.max_dates,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    print(f"[info] Figure written to {args.output}")


if __name__ == "__main__":
    main()

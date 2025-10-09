#!/usr/bin/env python3
"""Visualise rolling correlation tensors in 3D or as individual heatmaps.

Usage examples
--------------
# render a 3D cube for a given date range (written to outputs/correlation_usa.html)
python analysis/visualize_correlation_volume.py \
    --npz outputs/correlations_usa.npz \
    --from-date 2020-01-01 \
    --to-date 2020-12-31 \
    --output outputs/correlation_usa_2020.html

# render a single-day heatmap
python analysis/visualize_correlation_volume.py \
    --npz outputs/correlations_usa.npz \
    --from-date 2020-03-16 \
    --to-date 2020-03-16 \
    --output outputs/correlation_usa_2020-03-16.html
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CORR_COLORSCALE = [
    (0.0, "#ff0000"),  # red at 0
    (1.0, "#0000ff"),  # blue at 1
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise rolling correlation matrices as 3D volume or 2D heatmap."
    )
    parser.add_argument(
        "--npz",
        required=True,
        type=Path,
        help="Path to the compressed correlation tensor produced by generate_spreads_and_correlations.py.",
    )
    parser.add_argument(
        "--from-date",
        dest="from_date",
        type=str,
        default=None,
        help="Start date (inclusive, yyyy-mm-dd).",
    )
    parser.add_argument(
        "--to-date",
        dest="to_date",
        type=str,
        default=None,
        help="End date (inclusive, yyyy-mm-dd).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/correlation_visualisation.html"),
        help="Destination HTML file for the interactive figure.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "daily-heatmap", "timeline-heatmap", "volume"],
        default="auto",
        help=(
            "Visualisation mode: 'auto' chooses 2D heatmap for single date, 3D volume otherwise."
            " 'daily-heatmap' forces single-matrix heatmap (requires from=to)."
            " 'timeline-heatmap' creates a 2D heatmap of correlation values across the entire date range."
            " 'volume' forces 3D point cloud."
        ),
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=4.0,
        help="Marker size for the 3D scatter plot.",
    )
    return parser.parse_args(argv)


def load_tensor(path: Path) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Correlation archive not found: {path}")
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


def make_heatmap(
    date: pd.Timestamp,
    spreads: np.ndarray,
    corr_scaled: np.ndarray,
    corr_raw: np.ndarray,
) -> go.Figure:
    matrix = corr_scaled[0]
    raw = corr_raw[0]
    z = np.where(np.isnan(matrix), None, matrix)
    hover_text = []
    for i, row in enumerate(raw):
        hover_row = []
        for j, value in enumerate(row):
            if np.isnan(value):
                hover_row.append("No data")
            else:
                hover_row.append(
                    f"{spreads[i]} vs {spreads[j]}<br>corr={value:.3f}"
                )
        hover_text.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=spreads,
            y=spreads,
            colorscale=CORR_COLORSCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(title="corr (scaled)", ticksuffix=""),
            hoverinfo="text",
            text=hover_text,
        )
    )
    fig.update_layout(
        title=f"Correlation matrix on {date.date()}",
        xaxis_title="Spread i",
        yaxis_title="Spread j",
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True),
        template="plotly_white",
    )
    return fig


def make_3d_scatter(
    dates: pd.DatetimeIndex,
    spreads: np.ndarray,
    corr_scaled: np.ndarray,
    corr_raw: np.ndarray,
    marker_size: float,
) -> go.Figure:
    n_dates = len(dates)
    n_spreads = len(spreads)
    if n_dates == 0 or n_spreads == 0:
        raise ValueError("Empty tensor supplied to 3D renderer.")

    grid_x, grid_y = np.meshgrid(np.arange(n_spreads), np.arange(n_spreads), indexing="ij")
    grid_x = np.repeat(grid_x[None, :, :], n_dates, axis=0).reshape(-1)
    grid_y = np.repeat(grid_y[None, :, :], n_dates, axis=0).reshape(-1)
    grid_date_index = np.repeat(np.arange(n_dates), n_spreads * n_spreads)

    values_scaled = corr_scaled.reshape(-1)
    values_raw = corr_raw.reshape(-1)

    valid_mask = ~np.isnan(values_scaled)
    nan_mask = ~valid_mask

    def build_customdata(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.empty((0, 4))
        return np.column_stack(
            [
                spreads[grid_x[mask]].astype(str),
                spreads[grid_y[mask]].astype(str),
                dates[grid_date_index[mask]].astype(str),
                values_raw[mask],
            ]
        )

    custom_valid = build_customdata(valid_mask)
    custom_nan = build_customdata(nan_mask)

    # Map dates to integer axis values for plotting.
    z_axis_values = dates.view("int64") // 86400000000000  # days since epoch
    z_coords = z_axis_values[grid_date_index]

    fig = go.Figure()
    if valid_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=grid_x[valid_mask],
                y=grid_y[valid_mask],
                z=z_coords[valid_mask],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=values_scaled[valid_mask],
                    colorscale=CORR_COLORSCALE,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="corr (scaled)"),
                ),
                customdata=custom_valid,
                hovertemplate=(
                    "Spread i: %{customdata[0]}<br>"
                    "Spread j: %{customdata[1]}<br>"
                    "Date: %{customdata[2]}<br>"
                    "Correlation: %{customdata[3]:.3f}<extra></extra>"
                ),
                name="Correlation",
            )
        )

    if nan_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=grid_x[nan_mask],
                y=grid_y[nan_mask],
                z=z_coords[nan_mask],
                mode="markers",
                marker=dict(size=marker_size, color="rgba(255,255,255,1)", line=dict(color="#cccccc", width=0.5)),
                customdata=custom_nan,
                hovertemplate=(
                    "Spread i: %{customdata[0]}<br>"
                    "Spread j: %{customdata[1]}<br>"
                    "Date: %{customdata[2]}<br>"
                    "No data<extra></extra>"
                ),
                name="NaN",
            )
        )

    tickvals = np.arange(n_spreads)
    ticktext = spreads.tolist()
    date_ticks = z_axis_values
    # Reduce date ticks to avoid overcrowding (max 12)
    if len(date_ticks) > 12:
        step = max(1, len(date_ticks) // 12)
        date_tick_vals = date_ticks[::step]
        date_tick_text = [d.strftime("%Y-%m-%d") for d in dates[::step]]
    else:
        date_tick_vals = date_ticks
        date_tick_text = [d.strftime("%Y-%m-%d") for d in dates]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Spread i",
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=8),
            ),
            yaxis=dict(
                title="Spread j",
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=8),
            ),
            zaxis=dict(
                title="Date",
                tickvals=date_tick_vals,
                ticktext=date_tick_text,
                tickfont=dict(size=8),
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_white",
        title="Rolling correlation volume",
    )
    return fig


def make_timeline_heatmap(
    dates: pd.DatetimeIndex,
    spreads: np.ndarray,
    corr_scaled: np.ndarray,
    corr_raw: np.ndarray,
) -> go.Figure:
    if len(dates) == 0:
        raise ValueError("No observations to plot.")
    n_spreads = len(spreads)
    tri_i, tri_j = np.triu_indices(n_spreads)
    col_labels = [f"{spreads[i]} vs {spreads[j]}" for i, j in zip(tri_i, tri_j)]
    values_scaled = corr_scaled[:, tri_i, tri_j]
    values_raw = corr_raw[:, tri_i, tri_j]

    z = np.where(np.isnan(values_scaled), None, values_scaled)
    hover_text = []
    for row_idx in range(len(dates)):
        row_text = []
        for col_idx in range(len(col_labels)):
            if np.isnan(values_raw[row_idx, col_idx]):
                row_text.append("No data")
            else:
                row_text.append(
                    f"Date: {dates[row_idx].strftime('%Y-%m-%d')}<br>"
                    f"{col_labels[col_idx]}<br>corr={values_raw[row_idx, col_idx]:.3f}"
                )
        hover_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=col_labels,
            y=[d.strftime("%Y-%m-%d") for d in dates],
            colorscale=CORR_COLORSCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(title="corr (scaled)"),
            hoverinfo="text",
            text=hover_text,
        )
    )
    fig.update_layout(
        title="Correlation timeline heatmap",
        xaxis_title="Spread pair",
        yaxis_title="Date",
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True, autorange="reversed"),
        template="plotly_white",
    )
    return fig


def main() -> None:
    args = parse_args()
    dates, spreads, corr_raw, corr_scaled = load_tensor(args.npz)

    start = pd.to_datetime(args.from_date) if args.from_date else None
    end = pd.to_datetime(args.to_date) if args.to_date else None
    sub_dates, sub_corr_raw, sub_corr_scaled = subset_tensor(
        dates, corr_raw, corr_scaled, start, end
    )

    mode = args.mode
    if mode == "auto":
        mode = "daily-heatmap" if len(sub_dates) == 1 else "volume"

    if mode == "daily-heatmap":
        if len(sub_dates) != 1:
            raise ValueError("daily-heatmap mode requires a single date (from=to).")
        fig = make_heatmap(sub_dates[0], spreads, sub_corr_scaled, sub_corr_raw)
    elif mode == "timeline-heatmap":
        fig = make_timeline_heatmap(sub_dates, spreads, sub_corr_scaled, sub_corr_raw)
    elif mode == "volume":
        fig = make_3d_scatter(
            sub_dates,
            spreads,
            sub_corr_scaled,
            sub_corr_raw,
            marker_size=args.marker_size,
        )
    else:
        raise ValueError(f"Unknown visualisation mode: {mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output), include_plotlyjs="cdn")
    print(f"[info] Figure written to {args.output}")


if __name__ == "__main__":
    main()

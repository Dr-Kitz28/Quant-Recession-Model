import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def parse_args():
    parser = argparse.ArgumentParser(description="Render an interactive 2D correlation heatmap with a time slider using Plotly.")
    parser.add_argument("--npz", type=Path, required=True, help="Path to the correlation tensor .npz file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the output HTML file.")
    parser.add_argument("--max-dates", type=int, default=500, help="Maximum number of timeline samples for the slider.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data from the .npz file
    data = np.load(args.npz, allow_pickle=True)
    dates = pd.to_datetime(data["dates"].astype(str))
    spreads = data["spreads"].astype(str)
    corr_scaled = data["corr_scaled"]

    n_dates, _, _ = corr_scaled.shape

    # Subsample dates to keep the visualization performant
    if n_dates > args.max_dates:
        print(f"[info] Reducing timeline from {n_dates} to {args.max_dates} samples for render.")
        idx = np.linspace(0, n_dates - 1, args.max_dates).round().astype(int)
        dates = dates[idx]
        corr_scaled = corr_scaled[idx]

    # Create the figure
    fig = go.Figure()

    # Add a heatmap trace for each date
    for i, date in enumerate(dates):
        fig.add_trace(
            go.Heatmap(
                z=corr_scaled[i],
                x=spreads,
                y=spreads,
                colorscale=[[0, 'red'], [1, 'blue']],
                zmin=0,
                zmax=1,
                visible=(i == len(dates) - 1), # Make only the last date visible initially
                colorbar=dict(title='Correlation')
            )
        )

    # Create the slider steps
    steps = []
    for i, date in enumerate(dates):
        step = dict(
            method="update",
            args=[{"visible": [d == date for d in dates]}],
            label=date.strftime('%Y-%m-%d')
        )
        steps.append(step)

    sliders = [dict(
        active=len(dates) - 1,
        currentvalue={"prefix": "Date: "},
        pad={"t": 50},
        steps=steps
    )]

    # Configure the layout
    fig.update_layout(
        title='Interactive Correlation Heatmap',
        xaxis_title="Spread (i)",
        yaxis_title="Spread (j)",
        sliders=sliders,
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font_color="#e0e0e0",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'), # Show origin at top-left
    )

    # Save the figure to an HTML file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, full_html=False, include_plotlyjs='cdn')
    print(f"[info] Interactive heatmap figure written to {args.output}")

if __name__ == "__main__":
    main()
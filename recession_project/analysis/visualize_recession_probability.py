#!/usr/bin/env python3
"""Recession Probability Distribution Visualizer.

This module creates visualizations of recession probability predictions,
including:
    1. Time-series plot with confidence bands
    2. Probability heatmap across different time horizons
    3. Distribution density plots
    4. Comparison with actual recession periods

Outputs both static (matplotlib) and interactive (HTML) visualizations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# Data Loading
# ============================================================================
def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load recession predictions from CSV."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def load_npz_predictions(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    dates = pd.to_datetime(data["dates"].astype(str))
    probs = data["probabilities"]
    confidence = data.get("confidence", np.zeros_like(probs))
    return dates, probs, confidence


# ============================================================================
# Time-Series Plot with Confidence Bands
# ============================================================================
def plot_recession_probability(
    dates: pd.DatetimeIndex,
    probabilities: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    actual_recessions: Optional[List[Tuple[str, str]]] = None,
    threshold: float = 0.5,
    title: str = "Recession Probability Over Time",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Create time-series plot of recession probability with confidence bands.

    Args:
        dates: DatetimeIndex of prediction dates
        probabilities: Array of recession probabilities [0, 1]
        lower_bound: Lower confidence bound (optional)
        upper_bound: Upper confidence bound (optional)
        actual_recessions: List of (start_date, end_date) tuples for actual recessions
        threshold: Probability threshold for recession call
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # plot confidence band if provided
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            dates, lower_bound, upper_bound,
            alpha=0.2, color="steelblue", label="95% Confidence"
        )

    # plot probability line
    ax.plot(dates, probabilities, color="steelblue", linewidth=1.5, label="P(Recession)")

    # plot threshold line
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({threshold})")

    # shade actual recession periods (NBER official dates)
    if actual_recessions:
        first_labeled = False
        for start, end in actual_recessions:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            # Only shade if within data range
            if end_dt >= dates.min() and start_dt <= dates.max():
                if not first_labeled:
                    ax.axvspan(start_dt, end_dt, alpha=0.3, color="lightgray", 
                              edgecolor="gray", linewidth=1, label="NBER Recession")
                    first_labeled = True
                else:
                    ax.axvspan(start_dt, end_dt, alpha=0.3, color="lightgray", 
                              edgecolor="gray", linewidth=1)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Recession Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualizer] Saved time-series plot to {output_path}")

    return fig


# ============================================================================
# Probability Heatmap
# ============================================================================
def plot_probability_heatmap(
    dates: pd.DatetimeIndex,
    probabilities_multi: np.ndarray,  # (n_dates, n_horizons)
    horizons: List[str],
    title: str = "Recession Probability by Forecast Horizon",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6),
    forecast_start: Optional[pd.Timestamp] = None,
    forecast_peak: Optional[pd.Timestamp] = None,
) -> plt.Figure:
    """
    Create heatmap of recession probabilities across different horizons.

    Args:
        dates: DatetimeIndex of prediction dates
        probabilities_multi: Array of shape (n_dates, n_horizons)
        horizons: List of horizon labels (e.g., ["1M", "3M", "6M", "12M"])
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
        forecast_start: Date where forecast begins (for visual marker)
        forecast_peak: Date of forecast peak (for annotation)

    Returns:
        matplotlib Figure object
    """
    # create custom colormap (green -> yellow -> red)
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("recession", colors, N=256)

    fig, ax = plt.subplots(figsize=figsize)

    # downsample dates for readability
    step = max(1, len(dates) // 50)
    date_labels = [d.strftime("%Y-%m") if i % step == 0 else "" for i, d in enumerate(dates)]

    im = ax.imshow(probabilities_multi.T, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels(horizons)
    ax.set_ylabel("Forecast Horizon", fontsize=12)

    # x-axis ticks
    tick_positions = [i for i in range(0, len(dates), max(1, len(dates) // 10))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in tick_positions], rotation=45)
    ax.set_xlabel("Date", fontsize=12)

    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add forecast zone marker
    if forecast_start is not None:
        try:
            # Find index of forecast start
            forecast_idx = dates.get_indexer([forecast_start], method='nearest')[0]
            ax.axvline(x=forecast_idx, color="white", linestyle="--", linewidth=2, alpha=0.8)
            ax.text(forecast_idx + 5, len(horizons) - 0.5, "← Forecast", color="white", 
                   fontsize=10, fontweight="bold", verticalalignment="center")
        except:
            pass
    
    # Add forecast peak marker
    if forecast_peak is not None:
        try:
            peak_idx = dates.get_indexer([forecast_peak], method='nearest')[0]
            ax.axvline(x=peak_idx, color="white", linestyle=":", linewidth=2, alpha=0.9)
            ax.scatter([peak_idx], [0], color="white", s=100, marker="v", zorder=5)
            ax.text(peak_idx, -0.7, f"Peak\n{forecast_peak.strftime('%Y-%m')}", 
                   color="darkred", fontsize=9, fontweight="bold", ha="center")
        except:
            pass

    # colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Recession Probability", fontsize=11)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualizer] Saved heatmap to {output_path}")

    return fig


# ============================================================================
# Distribution Density Plot
# ============================================================================
def plot_probability_distribution(
    probabilities: np.ndarray,
    title: str = "Distribution of Recession Probabilities",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot the distribution of recession probabilities.

    Args:
        probabilities: Array of recession probabilities
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # histogram
    ax1.hist(probabilities, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Recession Probability", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Probability Histogram", fontsize=12)
    ax1.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
    ax1.legend()

    # cumulative distribution
    sorted_probs = np.sort(probabilities)
    cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax2.plot(sorted_probs, cdf, color="steelblue", linewidth=2)
    ax2.fill_between(sorted_probs, 0, cdf, alpha=0.2, color="steelblue")
    ax2.set_xlabel("Recession Probability", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.set_title("Cumulative Distribution", fontsize=12)
    ax2.axvline(x=0.5, color="red", linestyle="--")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualizer] Saved distribution plot to {output_path}")

    return fig


# ============================================================================
# Summary Statistics
# ============================================================================
def compute_summary_stats(
    dates: pd.DatetimeIndex,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute summary statistics for recession predictions."""
    high_prob_days = (probabilities >= threshold).sum()
    total_days = len(probabilities)

    stats = {
        "date_range": f"{dates.min().date()} to {dates.max().date()}",
        "total_days": int(total_days),
        "mean_probability": float(np.nanmean(probabilities)),
        "std_probability": float(np.nanstd(probabilities)),
        "median_probability": float(np.nanmedian(probabilities)),
        "min_probability": float(np.nanmin(probabilities)),
        "max_probability": float(np.nanmax(probabilities)),
        "high_probability_days": int(high_prob_days),
        "high_probability_pct": float(high_prob_days / total_days * 100),
        "threshold": threshold,
    }

    # find high-probability periods
    in_period = False
    periods = []
    start = None

    for i, (date, prob) in enumerate(zip(dates, probabilities)):
        if prob >= threshold and not in_period:
            start = date
            in_period = True
        elif prob < threshold and in_period:
            periods.append({"start": str(start.date()), "end": str(dates[i - 1].date())})
            in_period = False

    if in_period:
        periods.append({"start": str(start.date()), "end": str(dates[-1].date())})

    stats["high_probability_periods"] = periods

    return stats


# ============================================================================
# Generate All Visualizations
# ============================================================================
def generate_all_visualizations(
    dates: pd.DatetimeIndex,
    probabilities: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    output_dir: Path = Path("outputs/visuals"),
    actual_recessions: Optional[List[Tuple[str, str]]] = None,
) -> Dict:
    """Generate all visualization outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"figures": [], "stats": None}

    # 1. Time-series plot
    fig1 = plot_recession_probability(
        dates, probabilities,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        actual_recessions=actual_recessions,
        output_path=output_dir / "recession_probability_timeseries.png",
    )
    results["figures"].append("recession_probability_timeseries.png")
    plt.close(fig1)

    # 2. Distribution plot
    fig2 = plot_probability_distribution(
        probabilities,
        output_path=output_dir / "recession_probability_distribution.png",
    )
    results["figures"].append("recession_probability_distribution.png")
    plt.close(fig2)

    # 3. Multi-horizon heatmap (simulate different horizons by shifting)
    horizons = ["1M", "3M", "6M", "12M"]
    shifts = [22, 66, 132, 264]  # approximate trading days
    probs_multi = []

    for shift in shifts:
        if shift < len(probabilities):
            shifted = np.roll(probabilities, -shift)
            shifted[-shift:] = np.nan
            probs_multi.append(shifted)
        else:
            probs_multi.append(np.full_like(probabilities, np.nan))

    probs_multi = np.column_stack(probs_multi)

    fig3 = plot_probability_heatmap(
        dates, probs_multi, horizons,
        output_path=output_dir / "recession_probability_heatmap.png",
    )
    results["figures"].append("recession_probability_heatmap.png")
    plt.close(fig3)

    # 4. Summary statistics
    stats = compute_summary_stats(dates, probabilities)
    results["stats"] = stats

    import json
    stats_path = output_dir / "recession_probability_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[visualizer] Saved stats to {stats_path}")

    return results


# ============================================================================
# CLI / Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Recession probability visualizer")
    parser.add_argument("--input", type=Path, help="Input CSV with predictions")
    parser.add_argument("--input-npz", type=Path, help="Input NPZ with predictions")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visuals"), help="Output directory")
    parser.add_argument("--prob-col", type=str, default="recession_probability", help="Probability column name")
    parser.add_argument("--lower-col", type=str, default=None, help="Lower bound column")
    parser.add_argument("--upper-col", type=str, default=None, help="Upper bound column")
    args = parser.parse_args()

    print(f"[visualizer] Starting visualization generation...")

    if args.input_npz:
        dates, probs, confidence = load_npz_predictions(args.input_npz)
        lower_bound = probs - 2 * confidence if confidence.any() else None
        upper_bound = probs + 2 * confidence if confidence.any() else None
    elif args.input:
        df = load_predictions(args.input)
        dates = df.index
        probs = df[args.prob_col].values

        lower_bound = df[args.lower_col].values if args.lower_col and args.lower_col in df.columns else None
        upper_bound = df[args.upper_col].values if args.upper_col and args.upper_col in df.columns else None
    else:
        # demo mode: generate synthetic data
        print("[visualizer] No input provided, generating demo data...")
        dates = pd.date_range("2000-01-01", "2024-01-01", freq="D")

        # create realistic-looking recession probability curve
        np.random.seed(42)
        t = np.linspace(0, 8 * np.pi, len(dates))
        base = 0.3 + 0.2 * np.sin(t) + 0.1 * np.sin(2 * t)
        noise = np.random.normal(0, 0.05, len(dates))
        probs = np.clip(base + noise, 0, 1)

        # add recession spikes
        recession_periods = [
            ("2001-03-01", "2001-11-01"),
            ("2007-12-01", "2009-06-01"),
            ("2020-02-01", "2020-04-01"),
        ]
        for start, end in recession_periods:
            mask = (dates >= start) & (dates <= end)
            probs[mask] = np.clip(probs[mask] + 0.4, 0, 1)

        lower_bound = np.clip(probs - 0.1, 0, 1)
        upper_bound = np.clip(probs + 0.1, 0, 1)

    print(f"[visualizer] Loaded {len(dates)} data points from {dates.min().date()} to {dates.max().date()}")

    # US recession periods for reference
    us_recessions = [
        ("2001-03-01", "2001-11-01"),  # Dot-com bust
        ("2007-12-01", "2009-06-01"),  # Great Recession
        ("2020-02-01", "2020-04-01"),  # COVID-19
    ]

    results = generate_all_visualizations(
        dates, probs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        output_dir=args.output_dir,
        actual_recessions=us_recessions,
    )

    print(f"[visualizer] Generated {len(results['figures'])} visualizations")
    print(f"[visualizer] Output directory: {args.output_dir}")

    # print summary
    if results["stats"]:
        stats = results["stats"]
        print(f"\n--- Summary Statistics ---")
        print(f"Date range: {stats['date_range']}")
        print(f"Mean probability: {stats['mean_probability']:.3f}")
        print(f"High-prob days: {stats['high_probability_days']} ({stats['high_probability_pct']:.1f}%)")
        print(f"High-prob periods: {len(stats['high_probability_periods'])}")


if __name__ == "__main__":
    main()

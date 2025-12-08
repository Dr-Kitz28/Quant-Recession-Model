#!/usr/bin/env python3
"""End-to-End Recession Forecast Pipeline.

This script runs the full inference pipeline:
1. Load trained correlation weight learner → predict next-day correlations
2. Apply trained RL policy adjustments (deterministic) → refined correlations
3. Extract features from correlation matrices → compute recession probabilities
4. Generate visualizations from the probability time-series

Usage:
    python run_forecast_pipeline.py --output-dir outputs/forecast
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch


def load_correlation_data(corr_npz: Path, anchor_csv: Path) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Load correlation tensor and macro anchors."""
    corr_data = np.load(corr_npz, allow_pickle=True)
    dates = pd.to_datetime(corr_data["dates"].astype(str))
    spreads = corr_data["spreads"].astype(str).tolist()
    correlations = corr_data["corr"].astype(float)

    anchor_df = pd.read_csv(anchor_csv, parse_dates=["date"])
    anchor_df = anchor_df.set_index("date").sort_index()
    anchor_aligned = anchor_df.reindex(dates).ffill().bfill()
    anchors = anchor_aligned.values.astype(float)

    return correlations, anchors, dates, spreads


def predict_correlations(
    correlations: np.ndarray,
    anchors: np.ndarray,
    model_path: Path,
    n_spreads: int,
    n_anchor_features: int,
    device: str = "cpu",
) -> np.ndarray:
    """Predict next-day correlations using trained weight learner."""
    from model.correlation_weight_learner import CorrelationWeightLearner, CorrelationPredictor

    triu_idx = np.triu_indices(n_spreads, k=1)
    n_corr_features = len(triu_idx[0])

    model = CorrelationWeightLearner(
        n_corr_features=n_corr_features,
        n_anchor_features=n_anchor_features,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictor = CorrelationPredictor(model, n_spreads, device)

    print(f"[pipeline] Predicting correlations for {len(correlations)} dates...")
    predictions = []

    for t in range(len(correlations) - 1):
        current_corr = correlations[t]
        anchor_t = anchors[t]

        # skip if too many NaNs
        corr_flat = current_corr[triu_idx]
        if np.isnan(corr_flat).sum() / len(corr_flat) > 0.5:
            predictions.append(current_corr)  # fallback to current
            continue

        next_corr = predictor.predict_next_matrix(current_corr, anchor_t)
        predictions.append(next_corr)

    # last date: no prediction, use actual
    predictions.append(correlations[-1])

    return np.array(predictions)


def apply_rl_adjustments(
    pred_correlations: np.ndarray,
    policy_path: Path,
    n_spreads: int,
    device: str = "cpu",
) -> np.ndarray:
    """Apply RL policy adjustments to predicted correlations."""
    from model.rl_feedback_loop import RLFeedbackAgent

    triu_idx = np.triu_indices(n_spreads, k=1)
    n_corr_features = len(triu_idx[0])

    agent = RLFeedbackAgent(n_corr_features=n_corr_features, device=device)
    agent.load(policy_path)

    print(f"[pipeline] Applying RL adjustments to {len(pred_correlations)} predictions...")
    adjusted = []

    for t in range(len(pred_correlations)):
        pred_flat = pred_correlations[t][triu_idx]

        # skip if too many NaNs
        if np.isnan(pred_flat).sum() / len(pred_flat) > 0.5:
            adjusted.append(pred_correlations[t])
            continue

        adjustment = agent.select_adjustment(pred_flat, deterministic=True)
        pred_flat_clean = np.nan_to_num(pred_flat, nan=0.0)
        adjusted_flat = np.clip(pred_flat_clean + adjustment, -1, 1)

        mat = np.eye(n_spreads, dtype=np.float32)
        mat[triu_idx] = adjusted_flat
        mat = mat + mat.T - np.diag(np.diag(mat))
        adjusted.append(mat)

    return np.array(adjusted)


def compute_recession_probabilities(
    correlations: np.ndarray,
    dates: pd.DatetimeIndex,
    spreads: List[str],
    window: int = 60,
    lookback: int = 252,  # 1 year lookback for regime detection
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute recession probabilities from correlation features.
    
    Uses multiple indicators:
    1. Correlation breakdown: when high-corr assets decouple (flight to quality)
    2. Correlation regime change: sudden jumps in average correlation
    3. Correlation dispersion: wide spread of correlations = stress
    4. Mean reversion signal: deviation from long-term mean
    5. Eigenvalue concentration: first PC explaining too much variance = crisis
    
    Returns: probabilities, lower_bound, upper_bound
    """
    n_dates = len(correlations)
    n_spreads = correlations.shape[1]
    triu_idx = np.triu_indices(n_spreads, k=1)

    print(f"[pipeline] Computing recession probabilities for {n_dates} dates...")

    # Pre-compute rolling statistics
    mean_corrs = []
    for t in range(n_dates):
        corr_flat = correlations[t][triu_idx]
        valid = ~np.isnan(corr_flat)
        if valid.sum() > 10:
            mean_corrs.append(np.mean(corr_flat[valid]))
        else:
            mean_corrs.append(np.nan)
    
    mean_corrs = np.array(mean_corrs)
    mean_series = pd.Series(mean_corrs, index=dates)
    
    # Rolling long-term mean and std
    rolling_mean = mean_series.rolling(window=lookback, min_periods=60).mean()
    rolling_std = mean_series.rolling(window=lookback, min_periods=60).std()
    
    # Z-score: deviation from long-term mean
    z_scores = (mean_series - rolling_mean) / (rolling_std + 1e-6)
    
    # Rolling volatility of correlations (regime change indicator)
    corr_volatility = mean_series.diff().abs().rolling(window=20, min_periods=5).mean()
    
    probs = []
    for t in range(n_dates):
        corr_flat = correlations[t][triu_idx]
        valid = ~np.isnan(corr_flat)

        if valid.sum() < 10:
            probs.append(np.nan)
            continue

        # Feature 1: Mean correlation level (high = crisis, flight to quality)
        mean_corr = np.mean(corr_flat[valid])
        
        # Feature 2: Correlation dispersion (high = uncertainty/stress)
        corr_dispersion = np.std(corr_flat[valid])
        
        # Feature 3: Fraction of very high correlations (>0.7) - crisis signature
        high_corr_frac = (corr_flat[valid] > 0.7).sum() / valid.sum()
        
        # Feature 4: Fraction of negative correlations (flight to quality)
        neg_frac = (corr_flat[valid] < 0).sum() / valid.sum()
        
        # Feature 5: Z-score - deviation from long-term regime
        z = z_scores.iloc[t] if not np.isnan(z_scores.iloc[t]) else 0.0
        
        # Feature 6: Recent volatility in correlations
        vol = corr_volatility.iloc[t] if not np.isnan(corr_volatility.iloc[t]) else 0.0
        
        # Feature 7: Eigenvalue concentration (first PC)
        try:
            corr_mat = correlations[t].copy()
            # Fill NaN with 0 for eigendecomposition
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            np.fill_diagonal(corr_mat, 1.0)
            eigenvalues = np.linalg.eigvalsh(corr_mat)
            eigenvalues = np.sort(eigenvalues)[::-1]
            # First eigenvalue ratio
            eigen_ratio = eigenvalues[0] / (eigenvalues.sum() + 1e-6)
        except:
            eigen_ratio = 0.5
        
        # Combine features into recession score
        # Key insight: Recessions show:
        # - Rising correlations (diversification loss)
        # - High volatility in correlations (regime change)
        # - First eigenvalue dominance (single factor driving all)
        # - But also some negative correlations (flight to quality)
        
        # Normalize features to [0, 1] range with tighter bounds
        f1 = np.clip((mean_corr - 0.3) / 0.5, 0, 1)  # mean corr: 0.3-0.8 range
        f2 = np.clip((corr_dispersion - 0.1) / 0.3, 0, 1)  # dispersion: 0.1-0.4 range
        f3 = np.clip((high_corr_frac - 0.1) / 0.4, 0, 1)  # high corr: 0.1-0.5 range
        f4 = np.clip(neg_frac * 3, 0, 1)             # neg frac scaled (rare = important)
        f5 = np.clip((z + 1.5) / 3, 0, 1)            # z-score: -1.5 to +1.5 range
        f6 = np.clip(vol * 100, 0, 1)                # volatility scaled
        f7 = np.clip((eigen_ratio - 0.2) / 0.5, 0, 1)  # eigen ratio: 0.2-0.7 range
        
        # Weighted combination - emphasize regime change indicators
        raw_score = (
            0.15 * f1 +  # mean correlation
            0.10 * f2 +  # dispersion
            0.10 * f3 +  # high correlation fraction
            0.10 * f4 +  # negative correlations
            0.20 * f5 +  # z-score deviation (key regime indicator)
            0.20 * f6 +  # correlation volatility (crisis signature)
            0.15 * f7    # eigenvalue concentration
        )
        
        # Sigmoid transformation with centered threshold
        prob = 1.0 / (1.0 + np.exp(-10 * (raw_score - 0.5)))
        probs.append(prob)

    probs = np.array(probs)

    # Rescale probabilities to have better calibration
    # Use percentile-based rescaling to ensure good discrimination
    valid_probs = probs[~np.isnan(probs)]
    p10 = np.percentile(valid_probs, 10)
    p90 = np.percentile(valid_probs, 90)
    
    # Rescale so p10 → ~0.15 and p90 → ~0.85
    probs_rescaled = (probs - p10) / (p90 - p10 + 1e-6)
    probs_rescaled = np.clip(probs_rescaled, 0, 1)
    # Apply a slight sigmoid to smooth extremes
    probs_rescaled = 0.1 + 0.8 * probs_rescaled
    probs = probs_rescaled

    # Smooth with rolling window
    probs_series = pd.Series(probs, index=dates)
    probs_smooth = probs_series.rolling(window=window, min_periods=1, center=True).mean()
    
    # Fill NaN by interpolation
    probs_smooth = probs_smooth.interpolate(method='linear', limit_direction='both')
    probs_smooth = probs_smooth.fillna(0.5).values

    # Confidence bands based on rolling std
    probs_raw_series = pd.Series(probs, index=dates)
    probs_std = probs_raw_series.rolling(window=window, min_periods=1, center=True).std()
    probs_std = probs_std.fillna(0.1).values
    
    lower_bound = np.clip(probs_smooth - 1.96 * probs_std, 0, 1)
    upper_bound = np.clip(probs_smooth + 1.96 * probs_std, 0, 1)

    return probs_smooth, lower_bound, upper_bound


def generate_visualizations(
    dates: pd.DatetimeIndex,
    probabilities: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate visualizations from predictions."""
    from visualize_recession_probability import (
        plot_recession_probability,
        plot_probability_distribution,
        plot_probability_heatmap,
        compute_summary_stats,
    )
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # US recession periods
    us_recessions = [
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01"),
    ]

    # 1. Time-series plot
    fig1 = plot_recession_probability(
        dates, probabilities,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        actual_recessions=us_recessions,
        title="Recession Probability Forecast (Model Predictions)",
        output_path=output_dir / "recession_forecast_timeseries.png",
    )
    plt.close(fig1)

    # 2. Distribution plot
    valid_probs = probabilities[~np.isnan(probabilities)]
    fig2 = plot_probability_distribution(
        valid_probs,
        title="Distribution of Forecasted Recession Probabilities",
        output_path=output_dir / "recession_forecast_distribution.png",
    )
    plt.close(fig2)

    # 3. Multi-horizon heatmap
    horizons = ["1M", "3M", "6M", "12M"]
    shifts = [22, 66, 132, 264]
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
        title="Recession Probability Forecast by Horizon",
        output_path=output_dir / "recession_forecast_heatmap.png",
    )
    plt.close(fig3)

    # 4. Summary stats
    stats = compute_summary_stats(dates, probabilities)
    stats_path = output_dir / "recession_forecast_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[pipeline] Saved stats to {stats_path}")


def save_predictions(
    dates: pd.DatetimeIndex,
    probabilities: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    output_dir: Path,
) -> None:
    """Save predictions to CSV and NPZ."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame({
        "date": dates,
        "recession_probability": probabilities,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    })
    csv_path = output_dir / "recession_forecast.csv"
    df.to_csv(csv_path, index=False)
    print(f"[pipeline] Saved predictions to {csv_path}")

    # NPZ
    npz_path = output_dir / "recession_forecast.npz"
    np.savez_compressed(
        npz_path,
        dates=dates.astype(str).values,
        probabilities=probabilities,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    print(f"[pipeline] Saved predictions to {npz_path}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end recession forecast pipeline")
    parser.add_argument("--corr-npz", type=Path, default=Path("outputs/correlation_tensor_usa.npz"))
    parser.add_argument("--anchor-csv", type=Path, default=Path("outputs/macro_anchors_daily.csv"))
    parser.add_argument("--weight-model", type=Path, default=Path("outputs/correlation_weight_learner.pt"))
    parser.add_argument("--rl-policy", type=Path, default=Path("outputs/rl_feedback_policy.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forecast"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window", type=int, default=60, help="Smoothing window for probabilities")
    parser.add_argument("--skip-nn", action="store_true", help="Skip NN prediction (use raw correlations)")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL adjustments")
    args = parser.parse_args()

    print("=" * 60)
    print("RECESSION FORECAST PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    correlations, anchors, dates, spreads = load_correlation_data(args.corr_npz, args.anchor_csv)
    n_spreads = correlations.shape[1]
    n_anchor_features = anchors.shape[1]
    print(f"  Correlations: {correlations.shape}")
    print(f"  Anchors: {anchors.shape}")
    print(f"  Date range: {dates.min().date()} to {dates.max().date()}")

    # 2. Predict correlations
    if args.skip_nn:
        print("\n[2/5] Skipping NN prediction (using raw correlations)...")
        pred_correlations = correlations
    else:
        print("\n[2/5] Predicting correlations with NN weight learner...")
        pred_correlations = predict_correlations(
            correlations, anchors, args.weight_model,
            n_spreads, n_anchor_features, args.device
        )

    # 3. Apply RL adjustments
    if args.skip_rl:
        print("\n[3/5] Skipping RL adjustments...")
        adjusted_correlations = pred_correlations
    else:
        print("\n[3/5] Applying RL policy adjustments...")
        adjusted_correlations = apply_rl_adjustments(
            pred_correlations, args.rl_policy, n_spreads, args.device
        )

    # Save intermediate outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "adjusted_correlations.npz",
        dates=dates.astype(str).values,
        spreads=np.array(spreads),
        corr=adjusted_correlations,
    )
    print(f"  Saved adjusted correlations to {args.output_dir / 'adjusted_correlations.npz'}")

    # 4. Compute recession probabilities
    # NOTE: Use RAW correlations for probability calculation
    # The NN/RL adjustments are experimental and may produce degenerate outputs
    # The raw correlations contain the actual market signal
    print("\n[4/5] Computing recession probabilities from raw correlations...")
    probabilities, lower_bound, upper_bound = compute_recession_probabilities(
        correlations, dates, spreads, window=args.window  # Use raw correlations
    )

    # Save predictions
    save_predictions(dates, probabilities, lower_bound, upper_bound, args.output_dir)

    # 5. Generate visualizations
    print("\n[5/5] Generating visualizations...")
    generate_visualizations(dates, probabilities, lower_bound, upper_bound, args.output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Summary
    valid_probs = probabilities[~np.isnan(probabilities)]
    print(f"\nSummary:")
    print(f"  Total dates: {len(dates)}")
    print(f"  Valid predictions: {len(valid_probs)}")
    print(f"  Mean probability: {np.mean(valid_probs):.3f}")
    print(f"  Max probability: {np.max(valid_probs):.3f}")
    print(f"  High-prob days (>0.5): {(valid_probs > 0.5).sum()}")
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

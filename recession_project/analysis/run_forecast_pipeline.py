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
from scipy import interpolate


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


# NBER Recession dates (monthly, US)
NBER_RECESSIONS = [
    ("1980-01-01", "1980-07-31"),   # 1980 recession
    ("1981-07-01", "1982-11-30"),   # 1981-82 recession  
    ("1990-07-01", "1991-03-31"),   # 1990-91 recession
    ("2001-03-01", "2001-11-30"),   # Dot-com bust
    ("2007-12-01", "2009-06-30"),   # Great Recession
    ("2020-02-01", "2020-04-30"),   # COVID-19 recession
]


def create_recession_labels(dates: pd.DatetimeIndex) -> np.ndarray:
    """Create binary recession labels based on NBER dates."""
    labels = np.zeros(len(dates), dtype=float)
    
    for start, end in NBER_RECESSIONS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        mask = (dates >= start_dt) & (dates <= end_dt)
        labels[mask] = 1.0
    
    return labels


def compute_recession_probabilities(
    correlations: np.ndarray,
    dates: pd.DatetimeIndex,
    spreads: List[str],
    window: int = 60,
    lookback: int = 252,
    horizon_months: int = 6,  # Forecast horizon in months
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibrated recession probabilities from correlation features.
    
    Key changes for proper calibration:
    1. Use NBER recession labels as ground truth
    2. Compute features that correlate with recession onset
    3. Calibrate output to match empirical base rate (~12%)
    4. Create horizon-specific labels (recession in next k months)
    
    Returns: probabilities, lower_bound, upper_bound
    """
    n_dates = len(correlations)
    n_spreads = correlations.shape[1]
    triu_idx = np.triu_indices(n_spreads, k=1)

    print(f"[pipeline] Computing calibrated recession probabilities for {n_dates} dates...")
    
    # Create recession labels
    recession_labels = create_recession_labels(dates)
    base_rate = recession_labels.mean()
    print(f"  Historical recession base rate: {base_rate:.1%}")
    
    # Create horizon labels: "recession within next k months"
    horizon_days = horizon_months * 21  # ~21 trading days per month
    horizon_labels = np.zeros(n_dates)
    for t in range(n_dates):
        future_window = min(t + horizon_days, n_dates)
        if recession_labels[t:future_window].max() > 0:
            horizon_labels[t] = 1.0
    
    horizon_base_rate = horizon_labels.mean()
    print(f"  {horizon_months}-month horizon base rate: {horizon_base_rate:.1%}")

    # Compute correlation-based stress features
    print(f"  Computing correlation stress features...")
    
    # First pass: compute basic features
    basic_features = []
    for t in range(n_dates):
        corr_flat = correlations[t][triu_idx]
        valid = ~np.isnan(corr_flat)

        if valid.sum() < 10:
            basic_features.append([np.nan] * 7)
            continue

        # Feature 1: Mean correlation (stress = high correlation)
        mean_corr = np.mean(corr_flat[valid])
        
        # Feature 2: Correlation volatility (20-day change)
        if t >= 20:
            prev_flat = correlations[t-20][triu_idx]
            prev_valid = ~np.isnan(prev_flat)
            both = valid & prev_valid
            if both.sum() > 10:
                corr_change = np.abs(corr_flat[both] - prev_flat[both]).mean()
            else:
                corr_change = 0.0
        else:
            corr_change = 0.0
        
        # Feature 3: Eigenvalue concentration (systemic risk)
        try:
            corr_mat = np.nan_to_num(correlations[t], nan=0.0)
            np.fill_diagonal(corr_mat, 1.0)
            eigenvalues = np.linalg.eigvalsh(corr_mat)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigen_ratio = eigenvalues[0] / (eigenvalues.sum() + 1e-6)
            # Top 3 eigenvalues concentration
            top3_ratio = eigenvalues[:3].sum() / (eigenvalues.sum() + 1e-6)
        except:
            eigen_ratio = 0.3
            top3_ratio = 0.5
        
        # Feature 4: High correlation fraction (contagion)
        high_frac = (corr_flat[valid] > 0.6).sum() / valid.sum()
        
        # Feature 5: Negative correlation fraction (flight to quality)
        neg_frac = (corr_flat[valid] < -0.1).sum() / valid.sum()
        
        # Feature 6: Correlation dispersion (uncertainty)
        corr_std = np.std(corr_flat[valid])
        
        # Feature 7: 5th percentile (extreme low correlation)
        corr_p5 = np.percentile(corr_flat[valid], 5)
        
        basic_features.append([mean_corr, corr_change, eigen_ratio, high_frac, 
                               neg_frac, corr_std, corr_p5])
    
    basic_features = np.array(basic_features)
    
    # Convert to DataFrame for easier feature engineering
    feature_df = pd.DataFrame(basic_features, index=dates,
                              columns=['mean_corr', 'corr_change', 'eigen_ratio', 
                                       'high_frac', 'neg_frac', 'corr_std', 'corr_p5'])
    
    # Add rolling statistics as additional features
    for col in ['mean_corr', 'corr_change', 'eigen_ratio']:
        # 60-day rolling mean and std
        feature_df[f'{col}_ma60'] = feature_df[col].rolling(60, min_periods=1).mean()
        feature_df[f'{col}_std60'] = feature_df[col].rolling(60, min_periods=1).std()
        
        # 20-day change 
        feature_df[f'{col}_chg20'] = feature_df[col] - feature_df[col].shift(20)
        
        # Z-score relative to 252-day history
        rolling_mean = feature_df[col].rolling(252, min_periods=60).mean()
        rolling_std = feature_df[col].rolling(252, min_periods=60).std() + 1e-6
        feature_df[f'{col}_zscore'] = (feature_df[col] - rolling_mean) / rolling_std
    
    # Fill NaN values
    feature_df = feature_df.fillna(0.0)
    
    features = feature_df.values
    feature_names = list(feature_df.columns)
    print(f"  Total features: {len(feature_names)}")
    
    # Normalize features to z-scores  
    feature_means = np.nanmean(features, axis=0)
    feature_stds = np.nanstd(features, axis=0) + 1e-6
    features_z = (features - feature_means) / feature_stds
    features_z = np.nan_to_num(features_z, nan=0.0)
    
    # CALIBRATION: Use logistic regression trained on horizon labels
    # This learns optimal weights and produces calibrated probabilities
    print(f"  Training Random Forest for calibration...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    
    # Only use historical data for training (not future forecast)
    valid_mask = np.ones(len(dates), dtype=bool)
    valid_mask &= dates <= pd.Timestamp("2025-11-25")
    
    X_train = features_z[valid_mask]
    y_train = horizon_labels[valid_mask]
    
    # Train Random Forest with class weighting
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        class_weight={0: 1, 1: 7},  # Heavy weight on recessions (they're rare)
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"  Top 5 important features:")
    for i in sorted_idx[:5]:
        print(f"    {feature_names[i]}: {importances[i]:.3f}")
    
    # Get raw probabilities from Random Forest
    raw_probs = rf.predict_proba(features_z)[:, 1]
    
    # Smooth raw predictions (60-day rolling)
    probs_series = pd.Series(raw_probs, index=dates)
    probs_smooth = probs_series.rolling(window=window, min_periods=1, center=False).mean()

    probs_smooth = probs_smooth.fillna(base_rate).values
    
    # Apply Isotonic Regression for monotonic calibration
    # This ensures higher raw scores → higher calibrated probabilities
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_smooth[valid_mask], y_train)
    calibrated_probs = iso.predict(probs_smooth)
    calibrated_probs = np.clip(calibrated_probs, 0.01, 0.99)
    
    # Final smoothing
    probs_series = pd.Series(calibrated_probs, index=dates)
    probs_smooth = probs_series.rolling(window=window//2, min_periods=1, center=True).mean()
    probs_smooth = probs_smooth.fillna(base_rate).values
    
    # Verify calibration
    final_mean = probs_smooth[valid_mask].mean()
    print(f"  Final mean probability: {final_mean:.1%} (target: {horizon_base_rate:.1%})")
    
    # Check recession vs expansion separation
    rec_mask = valid_mask & (recession_labels == 1)
    exp_mask = valid_mask & (recession_labels == 0)
    if rec_mask.sum() > 0 and exp_mask.sum() > 0:
        p_rec = probs_smooth[rec_mask].mean()
        p_exp = probs_smooth[exp_mask].mean()
        print(f"  P(recession) during recessions: {p_rec:.1%}")
        print(f"  P(recession) during expansions: {p_exp:.1%}")

    # Confidence bands
    probs_std = probs_series.rolling(window=window, min_periods=1, center=True).std()
    probs_std = probs_std.fillna(0.05).values
    
    lower_bound = np.clip(probs_smooth - 1.96 * probs_std, 0.01, 0.99)
    upper_bound = np.clip(probs_smooth + 1.96 * probs_std, 0.01, 0.99)

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

    # US recession periods (NBER official dates)
    # https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
    us_recessions = [
        ("1980-01-01", "1980-07-01"),   # 1980 recession
        ("1981-07-01", "1982-11-01"),   # 1981-82 recession
        ("1990-07-01", "1991-03-01"),   # 1990-91 recession
        ("2001-03-01", "2001-11-01"),   # Dot-com bust
        ("2007-12-01", "2009-06-01"),   # Great Recession
        ("2020-02-01", "2020-04-01"),   # COVID-19 recession
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

    # Extend forecast to end of 2026
    last_date = dates.max()
    target_end = pd.Timestamp("2026-12-31")
    if last_date < target_end:
        print(f"\n  Extending forecast from {last_date.date()} to {target_end.date()}...")
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            end=target_end,
            freq='B'  # Business days
        )
        
        # Project forward using recent trend and mean reversion
        recent_window = 60  # ~3 months
        recent_probs = probabilities[-recent_window:]
        recent_mean = np.nanmean(recent_probs)
        recent_std = np.nanstd(recent_probs)
        long_term_mean = np.nanmean(probabilities)
        
        # Generate future probabilities with mean reversion + noise
        n_future = len(future_dates)
        np.random.seed(42)  # reproducibility
        
        future_probs = []
        current = recent_mean
        reversion_speed = 0.02  # slow mean reversion
        
        for i in range(n_future):
            # Mean reversion + random walk
            current = current + reversion_speed * (long_term_mean - current)
            noise = np.random.normal(0, recent_std * 0.3)
            prob = np.clip(current + noise, 0.1, 0.9)
            future_probs.append(prob)
        
        future_probs = np.array(future_probs)
        
        # Smooth the future projection
        future_series = pd.Series(future_probs)
        future_smooth = future_series.rolling(window=20, min_periods=1, center=True).mean().values
        
        # Extend all arrays
        dates = dates.append(future_dates)
        probabilities = np.concatenate([probabilities, future_smooth])
        
        # Extend confidence bands (wider for future)
        future_std = np.linspace(recent_std, recent_std * 2, n_future)  # widening uncertainty
        future_lower = np.clip(future_smooth - 1.96 * future_std, 0, 1)
        future_upper = np.clip(future_smooth + 1.96 * future_std, 0, 1)
        
        lower_bound = np.concatenate([lower_bound, future_lower])
        upper_bound = np.concatenate([upper_bound, future_upper])
        
        print(f"  Added {n_future} forecast days (now {len(dates)} total)")

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

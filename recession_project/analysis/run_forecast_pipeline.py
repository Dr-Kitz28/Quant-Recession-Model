#!/usr/bin/env python3
"""End-to-End Recession Forecast Pipeline.

This script runs the full inference pipeline:
1. Load trained correlation weight learner → predict next-day correlations
2. Apply trained RL policy adjustments (deterministic) → refined correlations
3. Extend correlations forward using NN + RL (diagram-compliant projection)
4. Extract features from (historical + projected) correlation matrices → compute recession probabilities
5. Generate visualizations from the probability time-series

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
    model_type: str = "mlp",
    spread_names: list = None,
) -> np.ndarray:
    """Predict next-day correlations using trained weight learner.
    
    Args:
        model_type: "mlp" for flattened MLP, "gnn" for graph neural network
        spread_names: Required for GNN model (list of spread names like '10Y-3M')
    """
    triu_idx = np.triu_indices(n_spreads, k=1)
    n_corr_features = len(triu_idx[0])
    
    if model_type == "gnn":
        from model.gnn_correlation_learner import GNNCorrelationLearner, GNNCorrelationPredictor, SpreadGraph
        
        if spread_names is None:
            raise ValueError("spread_names required for GNN model")
        
        # Build graph to get feature dimensions
        graph = SpreadGraph(spread_names)
        sample_node_feat = graph.get_node_features(correlations[0])
        sample_edge_feat = graph.get_edge_features(correlations[0])
        
        model = GNNCorrelationLearner(
            n_node_features=sample_node_feat.shape[1],
            n_edge_features=sample_edge_feat.shape[1],
            n_anchor_features=n_anchor_features if n_anchor_features > 1 else 0,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        predictor = GNNCorrelationPredictor(model, spread_names, device)
        
        print(f"[pipeline] Predicting correlations with GNN for {len(correlations)} dates...")
        predictions = []
        
        for t in range(len(correlations) - 1):
            current_corr = correlations[t]
            prev_corr = correlations[t-1] if t > 0 else current_corr
            anchor_t = anchors[t]
            
            # skip if too many NaNs
            corr_flat = current_corr[triu_idx]
            if np.isnan(corr_flat).sum() / len(corr_flat) > 0.5:
                predictions.append(current_corr)
                continue
            
            next_corr = predictor.predict_next_matrix(current_corr, prev_corr, anchor_t)
            predictions.append(next_corr)
        
        predictions.append(correlations[-1])
        return np.array(predictions)
    
    else:  # MLP model
        from model.correlation_weight_learner import CorrelationWeightLearner, CorrelationPredictor

        model = CorrelationWeightLearner(
            n_corr_features=n_corr_features,
            n_anchor_features=n_anchor_features,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        predictor = CorrelationPredictor(model, n_spreads, device)

        print(f"[pipeline] Predicting correlations with MLP for {len(correlations)} dates...")
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


def project_macro_anchors(
    anchors: np.ndarray,
    dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    cycle_aware: bool = True,
) -> np.ndarray:
    """
    Project macro anchors into the future using historical patterns.
    
    This is critical for the diagram flow: MacroAnchors → M' substantiation.
    Without evolving anchors, the NN has no forward-looking information.
    
    Methods:
    1. Trend extrapolation from recent history
    2. Mean reversion toward long-term average
    3. Cycle-aware stress scenarios based on business cycle position
    """
    n_future = len(future_dates)
    n_features = anchors.shape[1]
    
    if n_future == 0:
        return np.array([]).reshape(0, n_features)
    
    # Use last 2 years of data for trend estimation
    lookback = min(504, len(anchors))  # ~2 years of trading days
    recent = anchors[-lookback:]
    
    # Long-term statistics
    long_term_mean = np.nanmean(anchors, axis=0)
    long_term_std = np.nanstd(anchors, axis=0) + 1e-8
    
    # Recent trend (linear regression over last 6 months)
    trend_lookback = min(126, len(anchors))
    x = np.arange(trend_lookback)
    trends = np.zeros(n_features)
    for j in range(n_features):
        y = anchors[-trend_lookback:, j]
        valid = ~np.isnan(y)
        if valid.sum() > 10:
            slope, _ = np.polyfit(x[valid], y[valid], 1)
            trends[j] = slope
    
    # Current values
    current = anchors[-1].copy()
    
    # Mean-reversion speed (half-life ~2 years = 504 trading days)
    mean_reversion_rate = np.log(2) / 504
    
    # Business cycle position (years since last recession)
    last_recession_end = pd.Timestamp("2020-04-30")  # COVID end
    last_date = dates.max()
    years_since_recession = (last_date - last_recession_end).days / 365.25
    
    # Yield curve un-inversion signal (same as in probability computation)
    yield_curve_uninversion_date = pd.Timestamp("2025-10-16")
    
    future_anchors = np.zeros((n_future, n_features))
    
    for i, future_date in enumerate(future_dates):
        t = i + 1  # days into future
        
        # Base: mean reversion + trend continuation (with decay)
        trend_decay = np.exp(-t / 252)  # trend influence decays over 1 year
        mean_rev_factor = 1 - np.exp(-mean_reversion_rate * t)
        
        # Interpolate between current and long-term mean
        base = current * (1 - mean_rev_factor) + long_term_mean * mean_rev_factor
        # Add decaying trend
        base += trends * t * trend_decay
        
        if cycle_aware:
            # Compute stress based on yield curve signal + cycle position
            months_since_uninversion = (future_date - yield_curve_uninversion_date).days / 30.44
            years_elapsed = years_since_recession + t / 252
            
            # Yield curve-based stress (peaks 6-18 months post-uninversion)
            if months_since_uninversion < 0:
                yc_stress = 0.2
            elif months_since_uninversion < 6:
                yc_stress = 0.2 + 0.4 * (months_since_uninversion / 6)
            elif months_since_uninversion < 12:
                yc_stress = 0.6 + 0.3 * ((months_since_uninversion - 6) / 6)
            elif months_since_uninversion < 18:
                yc_stress = 0.9 + 0.1 * ((months_since_uninversion - 12) / 6)
            elif months_since_uninversion < 24:
                yc_stress = 1.0 - 0.2 * ((months_since_uninversion - 18) / 6)
            else:
                yc_stress = max(0.3, 0.8 - 0.05 * (months_since_uninversion - 24) / 6)
            
            # Cycle-based stress (long-term)
            if years_elapsed <= 5.0:
                cycle_stress = 0.1 * years_elapsed / 5.0
            elif years_elapsed <= 7.0:
                cycle_stress = 0.1 + 0.4 * (years_elapsed - 5.0) / 2.0
            elif years_elapsed <= 9.0:
                cycle_stress = 0.5 + 0.4 * (years_elapsed - 7.0) / 2.0
            else:
                cycle_stress = min(1.0, 0.9 + 0.05 * (years_elapsed - 9.0))
            
            # Combined stress factor (YC dominates near-term)
            yc_weight = max(0.2, 1.0 - 0.02 * months_since_uninversion)
            stress_factor = yc_weight * yc_stress + (1 - yc_weight) * cycle_stress
            
            # Apply stress to key indicators (credit spreads, VIX, etc.)
            # Indices 5-11 are credit/financial conditions in the anchor CSV
            stress_shift = stress_factor * long_term_std
            # Increase credit spreads and volatility
            base[5:12] += stress_shift[5:12] * 0.75
        
        future_anchors[i] = base
    
    return future_anchors


def extend_correlations_with_model(
    base_correlations: np.ndarray,
    anchors: np.ndarray,
    dates: pd.DatetimeIndex,
    target_end: pd.Timestamp,
    weight_model: Path,
    rl_policy: Path,
    n_spreads: int,
    n_anchor_features: int,
    device: str = "cpu",
    use_nn: bool = True,
    use_rl: bool = True,
    model_type: str = "gnn",
    spread_names: list = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Extend correlation forecasts to target_end using NN + RL per architecture diagram.
    
    Key fixes:
    1. Project macro anchors forward (not just repeat last value)
    2. Use projected anchors to substantiate M' (diagram requirement)
    3. Apply RL adjustments with increasing uncertainty over time
    4. Add mean-reversion to prevent runaway drift
    
    Args:
        model_type: "mlp" or "gnn" - which correlation model to use
        spread_names: List of spread names (required for GNN)
    """
    triu_idx = np.triu_indices(n_spreads, k=1)
    n_corr_features = len(triu_idx[0])

    predictor = None
    gnn_predictor = None
    
    if use_nn:
        if model_type == "gnn":
            from model.gnn_correlation_learner import GNNCorrelationLearner, GNNCorrelationPredictor, SpreadGraph
            
            if spread_names is None:
                raise ValueError("spread_names required for GNN model")
            
            graph = SpreadGraph(spread_names)
            sample_node_feat = graph.get_node_features(base_correlations[0])
            sample_edge_feat = graph.get_edge_features(base_correlations[0])
            
            model = GNNCorrelationLearner(
                n_node_features=sample_node_feat.shape[1],
                n_edge_features=sample_edge_feat.shape[1],
                n_anchor_features=n_anchor_features if n_anchor_features > 1 else 0,
            )
            model.load_state_dict(torch.load(weight_model, map_location=device))
            model.eval()
            gnn_predictor = GNNCorrelationPredictor(model, spread_names, device)
            print(f"  Using GNN model for correlation forecasting")
        else:
            from model.correlation_weight_learner import CorrelationWeightLearner, CorrelationPredictor

            model = CorrelationWeightLearner(
                n_corr_features=n_corr_features,
                n_anchor_features=n_anchor_features,
            )
            model.load_state_dict(torch.load(weight_model, map_location=device))
            model.eval()
            predictor = CorrelationPredictor(model, n_spreads, device)
            print(f"  Using MLP model for correlation forecasting")

    agent = None
    if use_rl:
        from model.rl_feedback_loop import RLFeedbackAgent

        agent = RLFeedbackAgent(n_corr_features=n_corr_features, device=device)
        agent.load(rl_policy)

    # Future business-day dates
    future_dates = pd.date_range(
        start=dates.max() + pd.Timedelta(days=1),
        end=target_end,
        freq="B",
    )

    if len(future_dates) == 0:
        return base_correlations, dates

    # === KEY FIX: Project anchors forward instead of repeating ===
    print(f"  Projecting macro anchors for {len(future_dates)} future dates...")
    future_anchors = project_macro_anchors(anchors, dates, future_dates, cycle_aware=True)

    # Historical correlation statistics for mean reversion
    hist_corr_mean = np.nanmean(base_correlations, axis=0)
    hist_corr_std = np.nanstd(base_correlations, axis=0) + 1e-8
    
    # Mean reversion rate (half-life ~1 year)
    corr_mean_rev_rate = np.log(2) / 252

    forecasts = []
    current = base_correlations[-1].copy()
    prev = base_correlations[-2].copy() if len(base_correlations) > 1 else current

    for i in range(len(future_dates)):
        anchor_t = future_anchors[i]

        # NN forecast using projected anchors
        if gnn_predictor is not None:
            next_corr = gnn_predictor.predict_next_matrix(current, prev, anchor_t)
        elif predictor is not None:
            next_corr = predictor.predict_next_matrix(current, anchor_t)
        else:
            next_corr = current.copy()

        # Apply mean reversion to prevent unrealistic drift
        mean_rev_factor = 1 - np.exp(-corr_mean_rev_rate)
        next_corr = next_corr * (1 - mean_rev_factor) + hist_corr_mean * mean_rev_factor

        # RL refinement with time-varying adjustment magnitude
        if agent is not None:
            pred_flat = next_corr[triu_idx]
            adjustment = agent.select_adjustment(pred_flat, deterministic=True)
            
            # Scale RL adjustment based on forecast horizon
            # Larger adjustments further in future (more uncertainty)
            horizon_scale = 1.0 + 0.5 * min(i / 252, 2.0)  # up to 2x after 2 years
            adjustment = adjustment * horizon_scale
            
            pred_flat_clean = np.nan_to_num(pred_flat, nan=0.0)
            adjusted_flat = np.clip(pred_flat_clean + adjustment, -1, 1)
            
            mat = np.eye(n_spreads, dtype=np.float32)
            mat[triu_idx] = adjusted_flat
            mat = mat + mat.T - np.diag(np.diag(mat))
            next_corr = mat

        # Ensure valid correlation matrix
        next_corr = np.clip(next_corr, -1, 1)
        np.fill_diagonal(next_corr, 1.0)

        forecasts.append(next_corr)
        prev = current
        current = next_corr

    extended_corr = np.concatenate([base_correlations, np.array(forecasts)])
    extended_dates = dates.append(future_dates)

    print(f"  Extended correlations: {base_correlations.shape[0]} → {extended_corr.shape[0]} dates")

    return extended_corr, extended_dates


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
    historical_cutoff: Optional[pd.Timestamp] = None,
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
    if historical_cutoff is None:
        historical_cutoff = dates.max()
    valid_mask &= dates <= historical_cutoff  # Only historical
    
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
    
    # === FIX: Apply cycle-aware adjustment for forecast period ===
    # The RF/Isotonic model extrapolates poorly to future dates because:
    # 1. Forecasted correlations tend toward steady state
    # 2. No new information beyond macro anchor projections
    # We need to inject business cycle prior knowledge using a proper hazard model
    forecast_mask = dates > historical_cutoff
    if forecast_mask.sum() > 0:
        print(f"  Applying cycle-aware adjustment to {forecast_mask.sum()} forecast dates...")
        
        # === YIELD CURVE INVERSION SIGNAL ===
        # Critical: The yield curve was inverted from Oct 2022 to Oct 2025
        # Historical pattern: Un-inversion typically precedes recession by 6-18 months
        # This is a MUCH stronger signal than simple cycle timing
        yield_curve_uninversion_date = pd.Timestamp("2025-10-16")  # When curve un-inverted
        inversion_duration_years = 3.0  # How long it was inverted (very long!)
        
        # Recession typically follows 6-18 months after un-inversion
        # Peak hazard window: 6-18 months post-uninversion
        recession_window_start = yield_curve_uninversion_date + pd.DateOffset(months=6)  # Apr 2026
        recession_window_end = yield_curve_uninversion_date + pd.DateOffset(months=18)  # Apr 2027
        
        print(f"  Yield curve signal: uninverted {yield_curve_uninversion_date.date()}")
        print(f"  High-risk window: {recession_window_start.date()} to {recession_window_end.date()}")
        
        # Business cycle baseline (still useful for long-term)
        last_recession_end = pd.Timestamp("2020-04-30")  # COVID end
        
        for idx in np.where(forecast_mask)[0]:
            future_date = dates[idx]
            years_since_recession = (future_date - last_recession_end).days / 365.25
            months_since_uninversion = (future_date - yield_curve_uninversion_date).days / 30.44
            
            # === YIELD CURVE HAZARD MODEL ===
            # Based on empirical pattern: recession follows uninversion by 6-18 months
            if months_since_uninversion < 0:
                # Before uninversion - use basic cycle model
                yc_hazard = 0.15  # Elevated due to ongoing inversion
            elif months_since_uninversion < 6:
                # 0-6 months post-uninversion: rising rapidly
                yc_hazard = 0.20 + 0.30 * (months_since_uninversion / 6)
            elif months_since_uninversion < 12:
                # 6-12 months post-uninversion: PEAK DANGER ZONE
                yc_hazard = 0.50 + 0.25 * ((months_since_uninversion - 6) / 6)
            elif months_since_uninversion < 18:
                # 12-18 months: still very high
                yc_hazard = 0.75 + 0.10 * ((months_since_uninversion - 12) / 6)
            elif months_since_uninversion < 24:
                # 18-24 months: declining but still elevated
                yc_hazard = 0.85 - 0.10 * ((months_since_uninversion - 18) / 6)
            else:
                # After 24 months: signal fades, use cycle model
                yc_hazard = max(0.30, 0.75 - 0.05 * (months_since_uninversion - 24) / 6)
            
            # === CYCLE-BASED HAZARD (long-term baseline) ===
            if years_since_recession <= 3.0:
                cycle_hazard = 0.05 + 0.05 * years_since_recession
            elif years_since_recession <= 5.0:
                cycle_hazard = 0.20 + 0.10 * (years_since_recession - 3.0)
            elif years_since_recession <= 7.0:
                cycle_hazard = 0.40 + 0.15 * (years_since_recession - 5.0)
            elif years_since_recession <= 9.0:
                cycle_hazard = 0.70 + 0.10 * (years_since_recession - 7.0)
            else:
                cycle_hazard = min(0.95, 0.90 + 0.02 * (years_since_recession - 9.0))
            
            # Combine yield curve signal with cycle model
            # YC signal is dominant in near-term, cycle dominates long-term
            yc_weight = max(0.2, 1.0 - 0.02 * months_since_uninversion)  # Fades over time
            cycle_prior = yc_weight * yc_hazard + (1 - yc_weight) * cycle_hazard
            
            # Blend model output with combined prior
            days_into_forecast = (future_date - historical_cutoff).days
            # Aggressive blending for near-term (high confidence in YC signal)
            if days_into_forecast < 252:  # First year
                prior_weight = min(0.80, 0.50 + 0.30 * (days_into_forecast / 252))
            else:
                prior_weight = min(0.85, 0.80 + 0.05 * ((days_into_forecast - 252) / 252))
            
            model_prob = calibrated_probs[idx]
            blended_prob = (1 - prior_weight) * model_prob + prior_weight * cycle_prior
            calibrated_probs[idx] = blended_prob
    
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

    # Confidence bands - wider for forecast period
    probs_std = probs_series.rolling(window=window, min_periods=1, center=True).std()
    probs_std = probs_std.fillna(0.05).values
    
    # Increase uncertainty for forecast dates (farther = more uncertain)
    if historical_cutoff is not None:
        for idx in np.where(forecast_mask)[0]:
            days_into_forecast = (dates[idx] - historical_cutoff).days
            # Uncertainty grows with sqrt of time (random walk scaling)
            uncertainty_scale = 1.0 + 0.5 * np.sqrt(days_into_forecast / 252)
            probs_std[idx] = min(0.25, probs_std[idx] * uncertainty_scale)
    
    lower_bound = np.clip(probs_smooth - 1.96 * probs_std, 0.01, 0.99)
    upper_bound = np.clip(probs_smooth + 1.96 * probs_std, 0.01, 0.99)

    return probs_smooth, lower_bound, upper_bound


def generate_visualizations(
    dates: pd.DatetimeIndex,
    probabilities: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    output_dir: Path,
    forecast_peak_date: Optional[pd.Timestamp] = None,
    forecast_peak_prob: Optional[float] = None,
) -> None:
    """Generate visualizations from predictions."""
    from visualize_recession_probability import (
        plot_recession_probability,
        plot_probability_distribution,
        plot_probability_heatmap,
        compute_summary_stats,
    )
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

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

    # 1. Time-series plot with forecast peak annotation
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    
    # Split into historical and forecast
    current_date = pd.Timestamp("2025-11-25")
    hist_mask = dates <= current_date
    fore_mask = dates > current_date
    
    # Confidence band
    ax1.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color="steelblue", label="95% Confidence")
    
    # Historical line (solid)
    ax1.plot(dates[hist_mask], probabilities[hist_mask], color="steelblue", linewidth=1.5, label="P(Recession) - Historical")
    
    # Forecast line (dashed)
    if fore_mask.sum() > 0:
        ax1.plot(dates[fore_mask], probabilities[fore_mask], color="darkorange", linewidth=2, linestyle="--", label="P(Recession) - Forecast")
    
    # Threshold
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    
    # Shade NBER recessions
    first_labeled = False
    for start, end in us_recessions:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if end_dt >= dates.min() and start_dt <= dates.max():
            if not first_labeled:
                ax1.axvspan(start_dt, end_dt, alpha=0.3, color="lightgray", label="NBER Recession")
                first_labeled = True
            else:
                ax1.axvspan(start_dt, end_dt, alpha=0.3, color="lightgray")
    
    # Add forecast peak marker if provided
    if forecast_peak_date is not None and forecast_peak_prob is not None:
        ax1.axvline(x=forecast_peak_date, color="red", linestyle=":", alpha=0.8, linewidth=2)
        ax1.scatter([forecast_peak_date], [forecast_peak_prob], color="red", s=100, zorder=5, marker="^")
        ax1.annotate(f"Forecast Peak\n{forecast_peak_date.strftime('%b %Y')}\nP={forecast_peak_prob:.0%}",
                     xy=(forecast_peak_date, forecast_peak_prob),
                     xytext=(forecast_peak_date + pd.DateOffset(months=2), forecast_peak_prob + 0.15),
                     fontsize=10, color="red", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))
    
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Recession Probability", fontsize=12)
    ax1.set_title("Recession Probability Forecast (Model Predictions)", fontsize=14, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(output_dir / "recession_forecast_timeseries.png", dpi=150, bbox_inches="tight")
    print(f"[visualizer] Saved time-series plot to {output_dir / 'recession_forecast_timeseries.png'}")
    plt.close(fig1)

    # 2. Distribution plot
    valid_probs = probabilities[~np.isnan(probabilities)]
    fig2 = plot_probability_distribution(
        valid_probs,
        title="Distribution of Forecasted Recession Probabilities",
        output_path=output_dir / "recession_forecast_distribution.png",
    )
    plt.close(fig2)

    # 3. Multi-horizon heatmap with forecast zone
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
        forecast_start=current_date,
        forecast_peak=forecast_peak_date,
    )
    plt.close(fig3)

    # 4. Summary stats
    stats = compute_summary_stats(dates, probabilities)
    if forecast_peak_date is not None:
        stats["forecast_peak_date"] = forecast_peak_date.strftime("%Y-%m-%d")
        stats["forecast_peak_probability"] = float(forecast_peak_prob) if forecast_peak_prob else None
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


def compute_recession_probabilities_gat(
    correlations: np.ndarray,
    anchors: np.ndarray,
    dates: pd.DatetimeIndex,
    spreads: List[str],
    model_path: Path,
    device: str = "cpu",
    seq_len: int = 60,
    historical_cutoff: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute recession probabilities using trained GAT-Transformer model.
    
    This is the upgraded approach using Graph Attention Networks + Transformer.
    Falls back to RF-based method if model not found.
    """
    from model.recession_gat_transformer import RecessionGATTransformer
    
    print(f"[pipeline] Computing recession probabilities with GAT-Transformer...")
    
    n_dates = len(correlations)
    n_spreads = len(spreads)
    
    # Check if model exists
    if not model_path.exists():
        print(f"  WARNING: GAT-Transformer model not found at {model_path}")
        print(f"  Falling back to Random Forest method")
        return compute_recession_probabilities(correlations, dates, spreads)
    
    # Build node/edge features matching training using SpreadGraph (directed edges)
    print(f"  Building graph features for {n_dates} dates...")
    try:
        from model.gnn_correlation_learner import SpreadGraph
        sg = SpreadGraph(spreads)
        node_features = []
        edge_features = []

        for t in range(n_dates):
            corr_mat = correlations[t]
            prev_corr = correlations[t - 1] if t > 0 else corr_mat
            nf = sg.get_node_features(corr_mat)
            ef = sg.get_edge_features(corr_mat, prev_corr)
            node_features.append(nf.cpu().numpy())
            edge_features.append(ef.cpu().numpy())

        node_features = np.array(node_features)
        edge_features = np.array(edge_features)
    except Exception as e:
        # Fallback to simple features if SpreadGraph isn't available
        print(f"  WARNING: SpreadGraph unavailable ({e}), falling back to simple features")
        node_features = []
        edge_features = []
        for t in range(n_dates):
            corr_mat = correlations[t]
            nf = np.zeros((n_spreads, 7), dtype=np.float32)
            ef = np.zeros((n_spreads * (n_spreads - 1), 8), dtype=np.float32)
            node_features.append(nf)
            edge_features.append(ef)
        node_features = np.array(node_features)
        edge_features = np.array(edge_features)

    macro_features = anchors
    
    # Load model
    try:
        model = RecessionGATTransformer(
            n_node_features=node_features.shape[-1],
            n_edge_features=edge_features.shape[-1],
            n_macro_features=macro_features.shape[-1],
            hidden_dim=64,
            n_gat_heads=4,
            n_gat_layers=3,
            n_transformer_heads=4,
            n_transformer_layers=4,
            n_regimes=3,
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  Loaded GAT-Transformer from {model_path}")
    except Exception as e:
        print(f"  WARNING: Failed to load GAT-Transformer: {e}")
        print(f"  Falling back to Random Forest method")
        return compute_recession_probabilities(correlations, dates, spreads)
    
    # Run inference in sliding windows using the GAT-Transformer
    n_dates = len(dates)
    probabilities = np.full(n_dates, np.nan)

    seq_stride = max(1, seq_len // 2)

    # Build a SpreadGraph helper once
    spread_graph = None
    try:
        from model.gnn_correlation_learner import SpreadGraph
        spread_graph = SpreadGraph(spreads)
    except Exception:
        # Fall back to creating edge_index dynamically
        spread_graph = None

    with torch.no_grad():
        for start in range(0, n_dates - seq_len + 1, seq_stride):
            end = start + seq_len

            # Build graph_sequences as list of dicts per timestep
            graph_sequences = []
            if spread_graph is None:
                sg = None
            else:
                sg = spread_graph
                edge_index = sg.edge_index.to(device)

            for t in range(start, end):
                nf = torch.tensor(node_features[t], dtype=torch.float32).to(device)
                ef = torch.tensor(edge_features[t], dtype=torch.float32).to(device)

                if sg is not None:
                    ei = edge_index
                else:
                    # Construct full directed edge_index for n_spreads
                    src = []
                    dst = []
                    for i in range(n_spreads):
                        for j in range(n_spreads):
                            if i != j:
                                src.append(i)
                                dst.append(j)
                    ei = torch.tensor([src, dst], dtype=torch.long).to(device)

                graph_sequences.append({
                    'node_features': nf,
                    'edge_index': ei,
                    'edge_attr': ef,
                })

            # Macro features for the window
            mf_seq = torch.tensor(macro_features[start:end], dtype=torch.float32).unsqueeze(0).to(device)

            output = model(graph_sequences, mf_seq, return_interpretability=False)
            # output['probs']: (batch=1, seq_len, n_horizons)
            probs = output.get('probs')
            if probs is None:
                continue
            probs_np = probs.squeeze(0).cpu().numpy()  # (seq_len, n_horizons)

            # Use 6-month horizon (index 1) by convention
            if probs_np.shape[1] < 2:
                horizon_pred = probs_np[:, 0]
            else:
                horizon_pred = probs_np[:, 1]

            for i, p in enumerate(horizon_pred):
                idx = start + i
                if np.isnan(probabilities[idx]):
                    probabilities[idx] = p
                else:
                    probabilities[idx] = (probabilities[idx] + p) / 2.0

    # Fill edges (start and tail) using nearest valid
    valid_idx = ~np.isnan(probabilities)
    if valid_idx.any():
        first_valid = np.where(valid_idx)[0][0]
        last_valid = np.where(valid_idx)[0][-1]
        probabilities[:first_valid] = probabilities[first_valid]
        probabilities[last_valid + 1:] = probabilities[last_valid]

    # Calibration: train an isotonic regressor on historical portion only
    try:
        from sklearn.isotonic import IsotonicRegression

        # Build horizon labels from NBER for calibration
        full_labels = create_recession_labels(dates)
        horizon_days = 6 * 21
        horizon_labels = np.zeros(len(dates), dtype=float)
        for t in range(len(dates)):
            future_window = min(t + horizon_days, len(dates))
            if full_labels[t:future_window].max() > 0:
                horizon_labels[t] = 1.0

        # Determine historical mask by anchors length (assumes anchors contains historical+projected)
        historical_cutoff = dates[len(anchors) - 1] if len(anchors) < len(dates) else dates[-1]
        hist_mask = dates <= historical_cutoff
        if hist_mask.sum() > 30 and horizon_labels[hist_mask].sum() > 0:
            iso = IsotonicRegression(out_of_bounds='clip')
            try:
                iso.fit(probabilities[hist_mask], horizon_labels[hist_mask])
                probabilities_cal = iso.transform(probabilities)
                probabilities = np.clip(probabilities_cal, 0.0, 1.0)
                print("  Applied isotonic calibration to GAT probabilities using historical data.")
            except Exception:
                print("  Isotonic calibration failed; leaving raw GAT probabilities.")
        else:
            print("  Not enough historical positives for isotonic calibration; leaving raw GAT probabilities.")
    except Exception:
        print("  sklearn not available or calibration failed; returning raw GAT probabilities.")

    lower_bound = np.clip(probabilities - 0.1, 0, 1)
    upper_bound = np.clip(probabilities + 0.1, 0, 1)

    return probabilities, lower_bound, upper_bound


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end recession forecast pipeline")
    parser.add_argument("--corr-npz", type=Path, default=Path("outputs/correlation_tensor_usa.npz"))
    parser.add_argument("--anchor-csv", type=Path, default=Path("outputs/macro_anchors_daily.csv"))
    parser.add_argument("--weight-model", type=Path, default=Path("outputs/gnn_weight_learner.pt"))
    parser.add_argument("--rl-policy", type=Path, default=Path("outputs/rl_feedback_policy.pt"))
    parser.add_argument("--gat-model", type=Path, default=Path("outputs/gat_transformer/gat_transformer_best.pt"),
                        help="Path to trained GAT-Transformer model for recession prediction")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forecast"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window", type=int, default=60, help="Smoothing window for probabilities")
    parser.add_argument("--skip-nn", action="store_true", help="Skip NN prediction (use raw correlations)")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL adjustments")
    parser.add_argument("--model-type", type=str, choices=["mlp", "gnn"], default="gnn",
                        help="Correlation model type: 'mlp' (baseline) or 'gnn' (graph-based, default)")
    parser.add_argument("--recession-model", type=str, choices=["rf", "gat"], default="rf",
                        help="Recession probability model: 'rf' (Random Forest) or 'gat' (GAT-Transformer)")
    args = parser.parse_args()

    print("=" * 60)
    print("RECESSION FORECAST PIPELINE")
    print(f"  Correlation model: {args.model_type.upper()}")
    print(f"  Recession model: {args.recession_model.upper()}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    correlations, anchors, dates, spreads = load_correlation_data(args.corr_npz, args.anchor_csv)
    n_spreads = correlations.shape[1]
    n_anchor_features = anchors.shape[1]
    print(f"  Correlations: {correlations.shape}")
    print(f"  Anchors: {anchors.shape}")
    print(f"  Date range: {dates.min().date()} to {dates.max().date()}")

    # 2. Predict correlations
    if args.skip_nn:
        print("\n[2/6] Skipping NN prediction (using raw correlations)...")
        pred_correlations = correlations
    else:
        print(f"\n[2/6] Predicting correlations with {args.model_type.upper()} model...")
        pred_correlations = predict_correlations(
            correlations, anchors, args.weight_model,
            n_spreads, n_anchor_features, args.device,
            model_type=args.model_type,
            spread_names=spreads,
        )

    # 3. Apply RL adjustments
    if args.skip_rl:
        print("\n[3/6] Skipping RL adjustments...")
        adjusted_correlations = pred_correlations
    else:
        print("\n[3/6] Applying RL policy adjustments...")
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

    # 4. Extend correlations using NN + RL (diagram-compliant pipeline)
    print(f"\n[4/6] Extending correlations into forecast horizon using {args.model_type.upper()} + RL...")
    last_actual_date = dates.max()
    target_end = pd.Timestamp("2029-12-31")
    extended_correlations, extended_dates = extend_correlations_with_model(
        adjusted_correlations,
        anchors,
        dates,
        target_end,
        args.weight_model,
        args.rl_policy,
        n_spreads,
        n_anchor_features,
        device=args.device,
        use_nn=not args.skip_nn,
        use_rl=not args.skip_rl,
        model_type=args.model_type,
        spread_names=spreads,
    )

    # 5. Compute recession probabilities on NN/RL + forecasted correlations
    print(f"\n[5/6] Computing recession probabilities with {args.recession_model.upper()} model...")
    
    if args.recession_model == "gat":
        # Extend anchors for forecast period
        extended_anchors = project_macro_anchors(anchors, dates, extended_dates[len(dates):])
        if len(extended_anchors) > 0:
            all_anchors = np.vstack([anchors, extended_anchors])
        else:
            all_anchors = anchors
        # Pad if needed
        if len(all_anchors) < len(extended_correlations):
            pad = np.tile(all_anchors[-1:], (len(extended_correlations) - len(all_anchors), 1))
            all_anchors = np.vstack([all_anchors, pad])
        
        probabilities, lower_bound, upper_bound = compute_recession_probabilities_gat(
            extended_correlations, all_anchors, extended_dates, spreads,
            args.gat_model, args.device, seq_len=60, historical_cutoff=last_actual_date
        )
    else:
        probabilities, lower_bound, upper_bound = compute_recession_probabilities(
            extended_correlations, extended_dates, spreads, window=args.window, historical_cutoff=last_actual_date
        )

    # Identify forecast peak beyond the historical window
    forecast_mask = extended_dates > last_actual_date
    if forecast_mask.any():
        forecast_probs = probabilities[forecast_mask]
        if len(forecast_probs) > 0:
            max_idx = np.argmax(forecast_probs)
            forecast_peak_prob = forecast_probs[max_idx]
            forecast_peak_date = extended_dates[forecast_mask][max_idx]
        else:
            forecast_peak_prob = None
            forecast_peak_date = None
    else:
        forecast_peak_prob = None
        forecast_peak_date = None

    # Save predictions
    save_predictions(extended_dates, probabilities, lower_bound, upper_bound, args.output_dir)

    # 6. Generate visualizations
    print("\n[6/6] Generating visualizations...")
    generate_visualizations(
        extended_dates, probabilities, lower_bound, upper_bound, args.output_dir,
        forecast_peak_date=forecast_peak_date,
        forecast_peak_prob=forecast_peak_prob,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Summary
    valid_probs = probabilities[~np.isnan(probabilities)]
    print(f"\nSummary:")
    print(f"  Total dates: {len(extended_dates)}")
    print(f"  Valid predictions: {len(valid_probs)}")
    print(f"  Mean probability: {np.mean(valid_probs):.3f}")
    print(f"  Max probability: {np.max(valid_probs):.3f}")
    print(f"  High-prob days (>0.5): {(valid_probs > 0.5).sum()}")
    if forecast_peak_date:
        print(f"  Forecast peak: {forecast_peak_date.date()} (P={forecast_peak_prob:.1%})")
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

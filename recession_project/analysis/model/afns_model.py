"""arbitrage_free_nelson_siegel (AFNS) model implementation

This module implements a lightweight three-factor AFNS model for extracting
Level, Slope and Curvature factors from cross-sectional yields and performing
simple VAR forecasting / scenario generation.

The implementation is intentionally simple and focuses on being easy to test
and integrate with the repo's existing spread/correlation pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AFNSParameters:
    lambda_param: float
    kappa: np.ndarray
    theta: np.ndarray
    sigma: np.ndarray


class AFNSModel:
    """Small, test-friendly AFNS model.

    The model provides:
    - loading_factors: Nelson-Siegel loadings for a given set of maturities
    - fit_factors_ols: fast OLS extraction of level/slope/curvature
    - fit_var: basic VAR(1) fit to the extracted factors
    - forecast_factors: iterate VAR to produce multi-step forecasts
    - generate_scenarios: Monte-Carlo sampling using the VAR residual covariance
    """

    def __init__(self, lambda_param: float = 0.0609):
        self.lambda_param = float(lambda_param)
        self.params: Optional[AFNSParameters] = None
        self.factors_history: Optional[pd.DataFrame] = None
        self.var_params: Optional[dict] = None

    def loading_factors(self, maturities: np.ndarray) -> np.ndarray:
        """Nelson–Siegel loadings for an array of maturities (years)."""
        lam = float(self.lambda_param)
        tau = np.asarray(maturities, dtype=float)
        tau = np.maximum(tau, 1e-8)

        level = np.ones_like(tau)
        slope = (1 - np.exp(-lam * tau)) / (lam * tau)
        curvature = slope - np.exp(-lam * tau)

        return np.column_stack([level, slope, curvature])

    def fit_factors_ols(self, yields: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """Extract Level/Slope/Curvature via OLS for each cross-section.

        yields: shape (n_dates, n_maturities)
        maturities: shape (n_maturities,) (years)
        returns: (n_dates, 3)
        """
        yields = np.asarray(yields, dtype=float)
        n_dates = yields.shape[0]
        loadings = self.loading_factors(maturities)

        factors = np.full((n_dates, 3), np.nan, dtype=float)

        for i in range(n_dates):
            y = yields[i]
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                continue
            L = loadings[mask]
            yv = y[mask]
            # solve least squares
            coeffs, *_ = np.linalg.lstsq(L, yv, rcond=None)
            factors[i, :] = coeffs

        return factors

    def reconstruct_yields(self, factors: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        load = self.loading_factors(maturities)  # (n_mats, 3)
        factors = np.asarray(factors)
        if factors.ndim == 1:
            return load @ factors
        else:
            return factors @ load.T

    def fit_var(self, factors: pd.DataFrame, order: int = 1) -> dict:
        f = factors.dropna()
        if len(f) < order + 5:
            raise ValueError("not enough data to fit VAR")

        n = len(f)
        k = f.shape[1]

        Y = f.values[order:]
        X = np.ones((n - order, 1))
        for lag in range(1, order + 1):
            X = np.hstack([X, f.values[order - lag : -lag]])

        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        residuals = Y - X @ B
        sigma = np.cov(residuals.T)

        # parse coefficients back into list of (k,k) matrices
        const = B[0]
        coeffs = []
        for p in range(order):
            start = 1 + p * k
            coeffs.append(B[start : start + k].T)

        params = {"order": order, "const": const, "coeffs": coeffs, "sigma_eps": sigma}
        self.var_params = params
        return params

    def forecast_factors(self, factors_history: pd.DataFrame, var_params: dict, steps: int = 12) -> pd.DataFrame:
        order = var_params["order"]
        const = var_params["const"]
        coeffs = var_params["coeffs"]

        last_obs = factors_history.dropna().iloc[-order:].values
        if last_obs.shape[0] < order:
            raise ValueError("insufficient history for forecasting")

        out = []
        for _ in range(steps):
            next_vals = const.copy()
            for lag in range(order):
                next_vals += coeffs[lag] @ last_obs[-(lag + 1), :]
            out.append(next_vals)
            last_obs = np.vstack([last_obs[1:], next_vals])

        # build index: if factors_history indexed by Timestamp, use monthly periods
        idx0 = factors_history.index[-1]
        if isinstance(idx0, pd.Timestamp):
            dates = pd.date_range(start=idx0, periods=steps + 1, freq="M")[1:]
        else:
            dates = range(len(factors_history), len(factors_history) + steps)

        return pd.DataFrame(out, index=dates, columns=factors_history.columns)

    def generate_scenarios(self, n_scenarios: int = 1000, steps: int = 12, shock_scale: float = 1.0) -> np.ndarray:
        if self.var_params is None or self.factors_history is None:
            raise ValueError("must fit VAR and store factors_history before scenario generation")

        order = self.var_params["order"]
        const = self.var_params["const"]
        coeffs = self.var_params["coeffs"]
        sigma = self.var_params["sigma_eps"]

        last_obs = self.factors_history.dropna().iloc[-order:].values
        if last_obs.shape[0] < order:
            raise ValueError("insufficient history for scenario generation")

        rng = np.random.default_rng()
        scenarios = np.zeros((n_scenarios, steps, last_obs.shape[1]), dtype=float)

        for s in range(n_scenarios):
            cur = last_obs.copy()
            for t in range(steps):
                nxt = const.copy()
                for lag in range(order):
                    nxt += coeffs[lag] @ cur[-(lag + 1), :]
                shock = rng.multivariate_normal(np.zeros(nxt.shape[0]), sigma) * shock_scale
                nxt = nxt + shock
                scenarios[s, t] = nxt
                cur = np.vstack([cur[1:], nxt])

        return scenarios


def example_usage() -> None:
    """Demonstrative local run using random data (keeps file self-contained)."""
    import pandas as pd

    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=48, freq="M")
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    n = len(dates)
    y = np.random.randn(n, len(mats)) * 0.3 + 2.5

    m = AFNSModel(lambda_param=0.0609)
    factors = m.fit_factors_ols(y, mats)
    fac_df = pd.DataFrame(factors, index=dates, columns=["level", "slope", "curvature"])
    m.factors_history = fac_df
    varp = m.fit_var(fac_df, order=1)
    fc = m.forecast_factors(fac_df, varp, steps=12)
    print("example forecasts (head):")
    print(fc.head())


if __name__ == "__main__":
    example_usage()

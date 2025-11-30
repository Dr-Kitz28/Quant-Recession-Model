#!/usr/bin/env python3
"""AFNS → spreads → rolling correlation pipeline

Creates a rolling correlation tensor from AFNS-reconstructed yield curves and
saves it in the same NPZ format used by the project's visualization API
(keys: dates, spreads, corr, corr_scaled).
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .afns_model import AFNSModel


def parse_tenor_to_years(tenor: str) -> float:
    t = str(tenor).strip().upper()
    if t.endswith("M"):
        return float(t[:-1]) / 12.0
    if t.endswith("Y"):
        return float(t[:-1])
    # try direct float
    try:
        return float(t)
    except Exception:
        raise ValueError(f"cannot parse tenor: {tenor}")


def build_correlation_cube(spreads: pd.DataFrame, window: int, min_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    spreads = spreads.sort_index()
    dates = spreads.index
    spread_names = list(spreads.columns)
    n_dates = len(dates)
    n_spreads = len(spread_names)

    corr_cube = np.full((n_dates, n_spreads, n_spreads), np.nan, dtype=float)

    for idx in range(n_dates):
        start = max(0, idx - window + 1)
        window_df = spreads.iloc[start : idx + 1]
        window_df = window_df.dropna(how="all")
        if window_df.shape[0] < min_periods:
            continue
        corr = window_df.corr()
        corr_cube[idx, :, :] = corr.to_numpy(dtype=float)

    corr_scaled = (corr_cube + 1.0) / 2.0
    corr_scaled = np.clip(corr_scaled, 0.0, 1.0, out=corr_scaled)

    return corr_cube, corr_scaled


class AFNSSpreadPipeline:
    def __init__(self, lambda_param: float = 0.0609):
        self.model = AFNSModel(lambda_param=lambda_param)

    def run(self,
            input_csv: Path,
            output_dir: Path,
            country: str = "USA",
            forecast_steps: int = 12,
            rolling_window: int = 60,
            min_periods: int = 20):
        df = pd.read_csv(input_csv)
        df["date"] = pd.to_datetime(df["date"])

        # filter yield rows and specific country
        df = df[df["data_type"].str.lower() == "yield"]
        country_df = df[df["country"].str.upper() == country.upper()]
        if country_df.empty:
            raise ValueError(f"no data for country {country}")

        pivot = (
            country_df.pivot_table(index="date", columns="tenor", values="yield_pct", aggfunc="mean").sort_index()
        )

        tenors = list(pivot.columns)
        maturities = np.array([parse_tenor_to_years(t) for t in tenors], dtype=float)

        # fit AFNS to the full panel
        yields = pivot.values  # shape (n_dates, n_mats)
        factors = self.model.fit_factors_ols(yields, maturities)
        hist_index = pivot.index
        self.model.factors_history = pd.DataFrame(factors, index=hist_index, columns=["level","slope","curvature"])

        # fit VAR on factors (simple VAR(1))
        try:
            varp = self.model.fit_var(self.model.factors_history, order=1)
        except Exception:
            varp = None

        # forecast factors
        if varp is not None:
            fcst_factors = self.model.forecast_factors(self.model.factors_history, varp, steps=forecast_steps)
        else:
            # fall back: use last observed factors repeated
            last = self.model.factors_history.dropna().iloc[-1].values
            fcst_factors = pd.DataFrame(np.tile(last, (forecast_steps,1)), index=pd.date_range(start=hist_index[-1], periods=forecast_steps+1,freq='M')[1:], columns=self.model.factors_history.columns)

        # reconstruct yields
        fcst_yields = pd.DataFrame(self.model.reconstruct_yields(fcst_factors.values, maturities), index=fcst_factors.index, columns=tenors)

        # combine historical and forecasted yields (note: yields are in %)
        all_yields = pd.concat([pivot, fcst_yields])

        # compute pairwise spreads (long - short) in basis points (like generate_spreads_and_correlations)
        tenor_pairs = [(short, long) for short, long in combinations(tenors, 2)]
        spread_names = [f"{long}-{short}" for short, long in tenor_pairs]
        spread_frames = {name: (all_yields[long] - all_yields[short]) * 100.0 for (short, long), name in zip(tenor_pairs, spread_names)}

        spreads_wide = pd.DataFrame(spread_frames, index=all_yields.index)

        # compute rolling correlation cube
        corr_cube, corr_scaled = build_correlation_cube(spreads_wide, window=rolling_window, min_periods=min_periods)

        # save NPZ in same format used elsewhere
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "correlation_tensor_afns.npz"

        np.savez_compressed(
            out_path,
            dates=np.array(spreads_wide.index.astype(str), dtype=object),
            spreads=np.array(spread_names, dtype=object),
            corr=corr_cube,
            corr_scaled=corr_scaled,
        )

        meta = {
            "n_dates": len(spreads_wide),
            "n_spreads": len(spread_names),
            "tenors": tenors,
            "maturities_years": maturities.tolist(),
            "forecast_steps": forecast_steps,
            "rolling_window": rolling_window,
            "created": pd.Timestamp.now().isoformat(),
        }
        (output_dir / "correlation_tensor_afns_meta.json").write_text(json.dumps(meta, indent=2))

        return out_path, spreads_wide, corr_cube, corr_scaled


def parse_args():
    p = argparse.ArgumentParser(description="AFNS spread → rolling correlation pipeline")
    p.add_argument("--input", type=Path, default=Path("bond_market_data.csv"))
    p.add_argument("--country", type=str, default="USA")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--lambda-param", type=float, default=0.0609)
    p.add_argument("--forecast-steps", type=int, default=12)
    p.add_argument("--rolling-window", type=int, default=60)
    p.add_argument("--min-periods", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = AFNSSpreadPipeline(lambda_param=args.lambda_param)
    out, spreads_wide, corr_cube, corr_scaled = pipeline.run(
        input_csv=args.input, output_dir=args.output_dir, country=args.country, forecast_steps=args.forecast_steps, rolling_window=args.rolling_window, min_periods=args.min_periods
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

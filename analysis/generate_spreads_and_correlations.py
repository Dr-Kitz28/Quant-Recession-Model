#!/usr/bin/env python3
"""Generate intra-day spread panels and rolling correlation matrices.

This script ingests the long-form `bond_market_data.csv`, computes all pairwise
yield spreads between distinct tenors for each country, saves the long-form
spread panel to CSV, and builds a rolling correlation tensor (date × spread ×
spread). The correlation tensor is persisted as a compressed NumPy archive for
subsequent visualisation.

Example
-------
python analysis/generate_spreads_and_correlations.py \
    --input bond_market_data.csv \
    --output-dir outputs \
    --window 60 \
    --min-periods 20
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_TENORS = [
    "1M",
    "3M",
    "6M",
    "1Y",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "15Y",
    "20Y",
    "30Y",
]


def load_config_tenors() -> List[str]:
    """Return tenor order from config.py if available."""
    try:
        import config  # type: ignore

        configured = list(getattr(config, "STANDARD_TENORS", []))
        cleaned = [t.strip().upper() for t in configured if isinstance(t, str)]
        if cleaned:
            # Preserve order but append any defaults not present.
            remainder = [t for t in DEFAULT_TENORS if t not in cleaned]
            return cleaned + remainder
    except Exception:
        pass
    return DEFAULT_TENORS.copy()


TENOR_ORDER = load_config_tenors()


def tenor_to_months(tenor: str) -> float:
    tenor = tenor.strip().upper()
    if tenor.endswith("M"):
        return float(tenor[:-1])
    if tenor.endswith("Y"):
        return float(tenor[:-1]) * 12.0
    if tenor.endswith("D"):
        return float(tenor[:-1]) / 30.0
    raise ValueError(f"Unrecognised tenor format: {tenor}")


def prepare_spreads(
    df: pd.DataFrame,
    window: int,
    min_periods: int,
    output_dir: Path,
) -> Dict[str, Dict[str, object]]:
    """Compute spreads and rolling correlations for every country.

    Returns a dictionary keyed by country containing metadata and paths to
    generated artefacts.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df[df["data_type"].str.lower() == "yield"].dropna(subset=["yield_pct"])
    df.sort_values(["country", "date"], inplace=True)

    outputs: Dict[str, Dict[str, object]] = {}

    for country, df_country in df.groupby("country"):
        pivot = (
            df_country.pivot_table(
                index="date", columns="tenor", values="yield_pct", aggfunc="mean"
            )
            .sort_index()
        )

        available_tenors = [t for t in TENOR_ORDER if t in pivot.columns]
        if len(available_tenors) < 2:
            print(f"[warn] Skipping {country}: <2 tenors available for spreads.")
            continue

        pivot = pivot[available_tenors]

        tenor_pairs: List[Tuple[str, str]] = [
            (short, long)
            for short, long in combinations(available_tenors, 2)
        ]
        spread_columns = [f"{long}-{short}" for short, long in tenor_pairs]

        spread_frames = {}
        for (short, long), name in zip(tenor_pairs, spread_columns):
            spread_frames[name] = (pivot[long] - pivot[short]) * 100.0  # basis pts

        spreads_wide = pd.DataFrame(spread_frames, index=pivot.index)
        spreads_wide.index.name = "date"

        spread_long = (
            spreads_wide.reset_index()
            .assign(country=country)
            .melt(
                id_vars=["date", "country"],
                var_name="spread_name",
                value_name="spread_bp",
            )
        )
        split = spread_long["spread_name"].str.split("-", expand=True)
        spread_long["tenor_long"] = split[0]
        spread_long["tenor_short"] = split[1]
        spread_long = spread_long[
            [
                "date",
                "country",
                "tenor_long",
                "tenor_short",
                "spread_name",
                "spread_bp",
            ]
        ].sort_values(["date", "spread_name"])

        spread_csv = output_dir / f"spreads_{country.lower()}.csv"
        spread_long.to_csv(spread_csv, index=False)

        corr_cube, corr_scaled = build_correlation_cube(
            spreads_wide, window=window, min_periods=min_periods
        )

        meta = {
            "country": country,
            "tenor_order": available_tenors,
            "spread_names": spread_columns,
            "tenor_pairs": tenor_pairs,
            "dates": spreads_wide.index.astype(str).tolist(),
            "window": window,
            "min_periods": min_periods,
        }

        corr_path = output_dir / f"correlations_{country.lower()}.npz"
        np.savez_compressed(
            corr_path,
            dates=np.array(meta["dates"], dtype=object),
            spreads=np.array(spread_columns, dtype=object),
            corr=corr_cube,
            corr_scaled=corr_scaled,
        )

        outputs[country] = {
            "spread_csv": str(spread_csv),
            "correlation_npz": str(corr_path),
            "metadata": meta,
        }

        print(
            f"[info] {country}: {len(spread_columns)} spreads, "
            f"{len(spreads_wide)} dates, window={window}, min_periods={min_periods}."
        )

    return outputs


def build_correlation_cube(
    spreads: pd.DataFrame,
    window: int,
    min_periods: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a rolling correlation tensor (date × spread × spread)."""
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

    corr_scaled = (corr_cube + 1.0) / 2.0  # map [-1, 1] → [0, 1]
    corr_scaled = np.clip(corr_scaled, 0.0, 1.0, out=corr_scaled)

    return corr_cube, corr_scaled


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spreads and rolling correlation tensors."
    )
    parser.add_argument(
        "--input",
        default="bond_market_data.csv",
        type=Path,
        help="Path to the long-form bond market CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory to write spreads and correlation artefacts.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window size (days) for correlation computation.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=20,
        help="Minimum number of observations needed to compute correlation.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    outputs = prepare_spreads(
        df=df, window=args.window, min_periods=args.min_periods, output_dir=args.output_dir
    )

    if not outputs:
        print("[warn] No outputs generated – check input coverage.")
        return

    summary = {
        country: {
            "spread_csv": info["spread_csv"],
            "correlation_npz": info["correlation_npz"],
            "n_spreads": len(info["metadata"]["spread_names"]),
            "n_dates": len(info["metadata"]["dates"]),
            "window": info["metadata"]["window"],
            "min_periods": info["metadata"]["min_periods"],
        }
        for country, info in outputs.items()
    }

    summary_path = args.output_dir / "spread_correlation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[info] Summary written to {summary_path}")


if __name__ == "__main__":
    main()

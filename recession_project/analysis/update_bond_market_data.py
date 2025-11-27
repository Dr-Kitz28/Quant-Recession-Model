#!/usr/bin/env python3
"""Refresh bond_market_data.csv with daily US Treasury yields from FRED."""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests

FRED_SERIES: Dict[str, str] = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "15Y": "DGS15",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
COLUMN_ORDER = [
    "date",
    "country",
    "tenor",
    "data_type",
    "yield_pct",
    "auction_amount_mn",
    "secondary_volume_mn",
    "outstanding_amount_mn",
    "bid_to_cover",
    "auction_type",
    "source",
    "timestamp",
]


def fetch_series(series_id: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(FRED_OBS_URL, params=params, timeout=60)
    if resp.status_code == 400:
        print(
            f"[warn] {series_id}: FRED returned 400 for {start}->{end}. Skipping series.")
        return pd.DataFrame(columns=["date", "value"])
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df = df[["date", "value"]]
    df = df[df["value"].ne(".")]
    df["value"] = df["value"].astype(float)
    return df


def build_long_panel(api_key: str, start: str, end: str) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for tenor, series_id in FRED_SERIES.items():
        series_df = fetch_series(series_id, api_key, start, end)
        print(f"[info] {tenor}: fetched {len(series_df)} observations")
        for row in series_df.itertuples(index=False):
            records.append({
                "date": row.date,
                "country": "USA",
                "tenor": tenor,
                "data_type": "yield",
                "yield_pct": row.value,
                "source": "FRED",
            })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No FRED observations retrieved. Check API key and parameters.")
    tenor_order = list(FRED_SERIES.keys())
    df["tenor"] = pd.Categorical(df["tenor"], categories=tenor_order, ordered=True)
    df.sort_values(["date", "tenor"], inplace=True)
    df["tenor"] = df["tenor"].astype(str)
    return df


def merge_with_existing(new_df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    if not output_csv.exists():
        return new_df
    existing = pd.read_csv(output_csv)
    keep = existing[existing["country"].ne("USA")]
    if keep.empty:
        return new_df
    combined = pd.concat([keep, new_df], ignore_index=True)
    combined.sort_values(["country", "date", "tenor"], inplace=True)
    return combined


def finalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["auction_amount_mn"] = pd.NA
    df["secondary_volume_mn"] = pd.NA
    df["outstanding_amount_mn"] = pd.NA
    df["bid_to_cover"] = pd.NA
    df["auction_type"] = pd.NA
    df["timestamp"] = pd.Timestamp.utcnow().isoformat()
    return df[COLUMN_ORDER]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update bond_market_data.csv from FRED")
    parser.add_argument("--api-key", dest="api_key", type=str, help="FRED API key")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bond_market_data.csv"),
        help="Destination CSV (will be overwritten for USA rows)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1985-01-01",
        help="Observation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=date.today().isoformat(),
        help="Observation end date (YYYY-MM-DD)",
    )
    return parser.parse_args(argv)


def resolve_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    env_key = os.getenv("FRED_API_KEY")
    if env_key:
        return env_key
    try:
        import config  # type: ignore

        cfg_key = getattr(config, "FRED_API_KEY", None)
        if cfg_key:
            return cfg_key
    except Exception:
        pass
    raise RuntimeError("FRED API key not provided via --api-key, env var, or config.py")


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    output_csv = args.output.expanduser().resolve()

    df = build_long_panel(api_key=api_key, start=args.start_date, end=args.end_date)
    df = finalise_columns(df)
    df = merge_with_existing(df, output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df):,} rows to {output_csv}")


if __name__ == "__main__":
    main()

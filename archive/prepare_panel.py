"""
Prepare a clean bond data panel from raw long-form CSV.
- One row per (date, country, tenor)
- Columns: yield_pct, auction_amount_mn, had_auction, secondary_volume_mn, outstanding_amount_mn
- NaN handling policy:
  * yield_pct: forward-fill within (country, tenor) by date
  * auction_amount_mn: NaN -> 0, had_auction flag (1 if auction_amount_mn>0 else 0)
  * secondary_volume_mn: NaN -> 0 (assume no recorded volume that day)
  * outstanding_amount_mn: forward-fill within (country, tenor) by date
Usage:
  python prepare_panel.py [input_csv] [output_csv]
Defaults:
  input_csv = bond_market_data.csv (or test_bond_data.csv if present and bond_market_data.csv missing)
  output_csv = bond_panel_clean.csv
"""
import sys
import os
import pandas as pd
import numpy as np

from datetime import datetime

DEFAULT_OUTPUT = "bond_panel_clean.csv"


def pick_default_input() -> str:
    if os.path.exists("bond_market_data.csv"):
        return "bond_market_data.csv"
    if os.path.exists("test_bond_data.csv"):
        return "test_bond_data.csv"
    return "bond_market_data.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize dtypes
    df['date'] = pd.to_datetime(df['date'])
    num_cols = ['yield_pct','auction_amount_mn','secondary_volume_mn','outstanding_amount_mn','bid_to_cover']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Standardize country casing
    df['country'] = df['country'].str.title()
    return df


def pivot_by_type(df: pd.DataFrame, dtype: str, value_col: str) -> pd.DataFrame:
    sub = df[df['data_type'] == dtype][['date','country','tenor', value_col]].copy()
    # multiple rows per key possible -> aggregate sensibly
    agg = {'yield_pct':'mean', 'auction_amount_mn':'sum', 'secondary_volume_mn':'sum', 'outstanding_amount_mn':'last'}
    vagg = agg.get(value_col, 'last')
    sub = sub.groupby(['date','country','tenor'], as_index=False).agg({value_col: vagg})
    sub = sub.sort_values(['country','tenor','date'])
    return sub


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    # Base keys: all unique combinations appearing anywhere
    keys = df[['date','country','tenor']].drop_duplicates()
    panel = keys.sort_values(['country','tenor','date']).copy()

    # Merge yields
    y = pivot_by_type(df, 'yield', 'yield_pct')
    panel = panel.merge(y, on=['date','country','tenor'], how='left')

    # Merge auctions
    a = pivot_by_type(df, 'auction', 'auction_amount_mn')
    panel = panel.merge(a, on=['date','country','tenor'], how='left')

    # had_auction flag
    panel['had_auction'] = (panel['auction_amount_mn'].fillna(0) > 0).astype(int)

    # Merge secondary volumes
    v = pivot_by_type(df, 'volume', 'secondary_volume_mn')
    panel = panel.merge(v, on=['date','country','tenor'], how='left')

    # Merge outstanding
    o = pivot_by_type(df, 'outstanding', 'outstanding_amount_mn')
    panel = panel.merge(o, on=['date','country','tenor'], how='left')

    # Sort for fills
    panel = panel.sort_values(['country','tenor','date']).reset_index(drop=True)

    # Forward-fill yields and outstanding within (country, tenor)
    panel['yield_pct'] = panel.groupby(['country','tenor'])['yield_pct'].ffill()
    panel['outstanding_amount_mn'] = panel.groupby(['country','tenor'])['outstanding_amount_mn'].ffill()

    # Fill 0 for amounts when absent that day
    panel['auction_amount_mn'] = panel['auction_amount_mn'].fillna(0.0)
    panel['secondary_volume_mn'] = panel['secondary_volume_mn'].fillna(0.0)

    # Reorder columns
    cols = ['date','country','tenor','yield_pct','auction_amount_mn','had_auction','secondary_volume_mn','outstanding_amount_mn']
    # Preserve any unknown columns at end if present
    for c in cols:
        if c not in panel.columns:
            panel[c] = np.nan
    panel = panel[cols]

    # Final types
    panel['date'] = panel['date'].dt.strftime('%Y-%m-%d')
    return panel


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else pick_default_input()
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not os.path.exists(in_path):
        print(f"❌ Input file not found: {in_path}")
        sys.exit(1)

    print(f"📥 Reading: {in_path}")
    df = load_data(in_path)

    print("🔧 Building clean panel and handling NaNs...")
    panel = build_panel(df)

    print(f"📤 Writing: {out_path}")
    panel.to_csv(out_path, index=False)

    # Quick summary
    print("\n📈 Output summary:")
    print(f"   Rows: {len(panel):,}")
    print(f"   Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"   Countries: {sorted(panel['country'].unique())}")
    print(f"   Tenors: {sorted(panel['tenor'].unique())}")

if __name__ == "__main__":
    main()

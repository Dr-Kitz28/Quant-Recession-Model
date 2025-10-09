# Handling NaNs in the Bond Data

Your raw CSV is a long-form event log: each date/tenor can have 0–N records across types (yield, auction, volume, outstanding). Some types simply don't happen daily (e.g., auctions), so NaNs are normal in the raw log. This guide shows how to transform it into a clean daily panel and what NaNs mean.

## Recommended Policy

- yield_pct
  - Forward-fill within (country, tenor) over time. Yields are available most days, so ffill preserves a continuous series when a day is missing.
- auction_amount_mn
  - Fill NaN with 0 and add a boolean flag had_auction (1 if auction_amount_mn > 0). Auctions occur only on certain days.
- secondary_volume_mn
  - Fill NaN with 0 (interpreted as no volume recorded or data source not reporting that day).
- outstanding_amount_mn
  - Forward-fill within (country, tenor). Outstanding rarely changes daily and is often reported at lower frequency.

## One-Row-Per-Day Panel

Use `prepare_panel.py` to build a tidy panel with one row per date/country/tenor and apply the NaN policy above.

### Usage

```bash
# Default: uses bond_market_data.csv if present; else test_bond_data.csv
python prepare_panel.py

# Explicit paths
python prepare_panel.py bond_market_data.csv bond_panel_clean.csv
```

### Output Columns
- date
- country
- tenor
- yield_pct (ffill within group)
- auction_amount_mn (0 when no auction)
- had_auction (0/1)
- secondary_volume_mn (0 when absent)
- outstanding_amount_mn (ffill within group)

## Why NaNs Happen in the Raw Log

- Auction rows contain only auction fields; yield/volume/outstanding are blank by design.
- Volume rows contain only volume fields; yields/auction are blank.
- Outstanding rows contain only outstanding; other fields are blank.
- On non-business days or when a source is down, that type may be missing entirely for a date.

## Tips

- Keep the raw log as an append-only audit trail.
- Build analytical tables (like the daily panel) from the raw log using repeatable scripts.
- If you need stricter completeness, restrict analyses to business days where yield is present after ffill.

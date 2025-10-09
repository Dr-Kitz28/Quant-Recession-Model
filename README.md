# Recession Warning System for India

We propose a **recession-warning system** that learns from how bond yield spreads move together across the Indian sovereign curve.

---

## Overview
- Starting with `n` distinct maturities (T-Bills and G-Secs), we compute all `nC2` spreads.  
- For each day, we form a **symmetric correlation matrix** summarizing their comovement.  
- Stacking these matrices over 20–25 years yields a **time-ordered tensor** — effectively a *"movie"* of the curve's internal dynamics.  
- Hypothesis: **pattern shifts in this movie** (changes in network structure & correlation intensity) precede downturns in India's real economy.  

---

## Methodology Used
1. **Yield Curve Modeling**  
   - Use an Arbitrage-Free Nelson–Siegel (AFNS) term-structure model.  
   - AFNS factors (level, slope, curvature) evolve via a small VAR/VARX that incorporates Indian macro-financial drivers:  
     - Policy rate  
     - Inflation  
     - Growth proxies  
     - Funding/FX markets  
     - Oil prices  

2. **Simulation & Stress Testing**  
   - Simulated curves → spreads → correlation matrices.  
   - Enables *what-if* stress tests around policy or macro shocks.  

3. **Machine Learning**  
   - A lightweight **temporal neural model** trained on rolling windows of correlation matrices.  
   - Outputs probabilities of **recession/slowdown** (defined by standard macro criteria).  
   - Attribution tools highlight the most influential spreads and subnetworks.  

---

## Benchmarking & Evaluation
- Compare against **canonical term-spread models**.  
- Evaluate on **lead time, stability, and calibration**.  

---

## Contribution
This project aims to provide an **earlier and more robust recession signal for India** by exploiting **higher-order comovement** across the yield curve, not just traditional level/slope metrics.

---

# Bond Market Data Logger

## Overview
Automated webscraping system for collecting bond market data from India and USA sources.

## Files Overview

### Data Templates (CSV)
- `TenorBucketDef.csv` - Editable tenor bucket definitions (3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 30Y)
- `YieldDaily.csv` - Daily yield data by tenor bucket (empty template)
- `AuctionDaily.csv` - Daily auction results (empty template) 
- `SecurityMaster.csv` - Bond security master data (empty template)
- `SecondaryTrade.csv` - Individual secondary market trades (empty template)
- `SecondaryVolumeDaily.csv` - Pre-aggregated daily volumes by bucket (empty template)
- `OutstandingBySecurityMonthly.csv` - Monthly outstanding by ISIN (empty template)
- `OutstandingDaily.csv` - Daily outstanding by bucket (empty template)
- `DailyCurveFeatures.csv` - Final wide-format output template

### Python Scripts
- `utils_bucket_mapper.py` - Utility functions for maturity→bucket mapping and aggregation
- `build_daily_features.py` - Main script to build the wide daily features table

# QRM (minimal)

This workspace contains only the minimal scripts and data for the three-step
goal:

1. Ingest US bond market data: `bond_market_data.csv`.
2. Compute spreads and serialise outputs: `analysis/generate_spreads_and_correlations.py`.
3. Render correlation visualisations (Matplotlib PNGs) with red→blue gradient
   over range [-1, 1] and white for NaNs: `analysis/visualize_correlation_matplotlib.py`.

Advanced features include prototyping time-ordered rolling-correlation tensors
across sovereign yield-spread pairs for evolving tenor-network analysis with
graph/eigen features; integrating Arbitrage-Free Nelson-Siegel with VAR/VARX
for macro-consistent yield-curve scenarios propagated to spreads and correlations
for stress testing; and building regime-detection on yield-curve factors using
change-point and Markov-switching methods to identify tenor-curve regime shifts.

Usage (examples):

Render a single-day heatmap:

```powershell
& D:\Downloads\QRM\.venv\Scripts\python.exe analysis\visualize_correlation_matplotlib.py \
  --npz outputs\correlations_usa.npz \
  --mode daily-heatmap --from-date 2020-03-16 --to-date 2020-03-16 \
  --output outputs\corr_usa_2020-03-16.png
```

Render a timeline heatmap (downsampling applied automatically for very long histories):

```powershell
& D:\Downloads\QRM\.venv\Scripts\python.exe analysis\visualize_correlation_matplotlib.py \
  --npz outputs\correlations_usa.npz --mode timeline-heatmap \
  --output outputs\corr_usa_timeline.png
```

Render a capped 3D volume (use `--max-dates` to control date slices):

```powershell
& D:\Downloads\QRM\.venv\Scripts\python.exe analysis\visualize_correlation_matplotlib.py \
  --npz outputs\correlations_usa.npz --mode volume --max-dates 365 \
  --output outputs\corr_usa_volume.png
```

Requirements are in `requirements.txt`.

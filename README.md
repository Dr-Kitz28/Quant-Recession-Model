# Recession Warning System for India

We propose a **recession-warning system** that learns from how bond yield spreads move together across the Indian sovereign curve.

---

## Overview
- Starting with `n` distinct maturities (T-Bills and G-Secs), we compute all `nC2` spreads.  
- For each day, we form a **symmetric correlation matrix** summarizing their comovement.  
- Stacking these matrices over 20–25 years yields a **time-ordered tensor** — effectively a *“movie”* of the curve’s internal dynamics.  
- Hypothesis: **pattern shifts in this movie** (changes in network structure & correlation intensity) precede downturns in India’s real economy.  

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

# Recession Warning System

We propose a **recession-warning system** that learns from how bond yield spreads move together.

---

## Overview
- Starting with `n` distinct maturities, we compute all `nC2` spreads.  
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

Prerequisites & setup (step-by-step)

Before you begin, make sure you have:
- Python 3.10+ installed (the project venv was created with Python 3.12)
- Node.js (18+) and npm
- Git
- (Windows) WSL2 if you prefer running the backend inside a Linux-style venv

1) Clone the repository

PowerShell (Windows):
```powershell
git clone https://github.com/AdityaVardhanGandi/Quant-Recession-Model.git
cd Quant-Recession-Model
```

WSL / Bash:
```bash
git clone https://github.com/AdityaVardhanGandi/Quant-Recession-Model.git
cd Quant-Recession-Model
```

2) Backend — install Python dependencies

Recommended: run the backend from WSL so the included Unix-style `.venv` works without modification.

WSL (recommended):
```bash
cd /mnt/d/Downloads/QRM/recession_project/server
python3 -m venv .venv        # create venv if missing
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ../../requirements.txt
```

Windows PowerShell (alternative):
```powershell
cd D:\Downloads\QRM\recession_project\server
python -m venv .venv
\.venv\Scripts\Activate.ps1   # or use Activate.bat on cmd.exe
python -m pip install --upgrade pip
python -m pip install -r ..\..\requirements.txt
```

3) Frontend — install Node dependencies

PowerShell (or WSL):
```powershell
cd D:\Downloads\QRM\recession_project\frontend\heatmap-client
npm install
```

4) Data files (important)

- Large binary outputs such as `correlation_tensor_usa.npz` are intentionally excluded from the repo (see `.gitignore`).
- Place required data files into one of the expected locations before running the backend:
  - `recession_project/outputs/` (preferred) or `outputs/` at repo root — make sure paths match the server configuration.

If the NPZ is missing the server may start but endpoints that read it will fail with a FileNotFoundError.

5) Environment variables / .env

- The frontend includes a placeholder `.env` in `heatmap-client` for client-side configuration (e.g. backend URL). Edit that file locally if you need to override settings.

6) Start the app (quick summary)

Backend (WSL recommended):
```bash
cd /mnt/d/Downloads/QRM/recession_project/server
source .venv/bin/activate
python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info
```

Frontend (PowerShell):
```powershell
cd D:\Downloads\QRM\recession_project\frontend\heatmap-client
npm run dev
# open http://localhost:5173
```

Helper scripts

Two convenience scripts are provided in the repo root if you prefer one-command startup:
- `start-dev.ps1` — PowerShell script that launches the backend in WSL and then runs the frontend via npm
- `start-wsl.sh` — WSL/Bash script that starts the backend and frontend in sequence

Run the script that matches your environment and adjust paths if your checkout is in a different location.

## Run the website (development) — two terminals

Use two terminals: one for the backend (FastAPI / Uvicorn) and one for the frontend (Vite React dev server). The backend venv in this repository is a Unix-style virtualenv under `recession_project/server/.venv`, so running the backend from WSL (recommended) avoids Windows/venv mismatches.

### Backend (recommended: WSL)

Open a WSL bash shell and run:

```bash
cd /mnt/d/Downloads/QRM/recession_project/server
source .venv/bin/activate
python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info
```

If you want the server to run in the background (detach) from the same WSL shell:

```bash
cd /mnt/d/Downloads/QRM/recession_project/server
. .venv/bin/activate
nohup python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info > uvicorn.log 2>&1 &
echo $! > /tmp/uvicorn.pid
tail -f uvicorn.log
```

### Backend (alternative: from Windows PowerShell using WSL)

If you prefer PowerShell, run the server in WSL from PowerShell with one command (careful with quoting):

```powershell
wsl -e bash -lc "cd /mnt/d/Downloads/QRM/recession_project/server && . .venv/bin/activate && python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info"
```

This will run the server in the WSL environment (recommended because the `.venv` here is Unix-style).

### Frontend (PowerShell)

Open a separate PowerShell terminal and run the Vite dev server:

```powershell
cd D:\Downloads\QRM\recession_project\frontend\heatmap-client
npm install
npm run dev
# open http://localhost:5173 in your browser (Vite default)
```

### Quick health checks

From any terminal you can check the backend metadata endpoint:

Linux/WSL or Git Bash:
```bash
curl "http://127.0.0.1:8001/meta?order=clustered"
```

PowerShell:
```powershell
Invoke-RestMethod "http://127.0.0.1:8001/meta?order=clustered"
```

Expected: a JSON object with keys like `n_dates`, `n_spreads`, and `spreads`.

If the frontend fails to load data, open the browser DevTools console to see fetch errors (status codes and URLs) and verify the `/meta` endpoint directly.

### Troubleshooting

- If `/meta` returns HTTP 500: run the backend in the foreground (no nohup, no daemon) so you can see the full Python traceback in the terminal. Copy-paste the traceback if you need help.
- If `curl` or the browser cannot connect (connection refused): ensure the backend is running and listening on 127.0.0.1:8001. Confirm there are no other services bound to that port and that the server didn't exit right after starting (watch the server terminal or `uvicorn.log`).
- Use the WSL-based backend run when the project `.venv` contains `bin/` and Unix executables. Running the server with a Windows Python interpreter (outside WSL) will usually fail to activate that venv.
- If you see warnings like `All-NaN slice encountered` during startup: these are warnings from the clustering routine when some rows are all-NaN for a particular chunk; they are non-fatal. If you see exceptions (tracebacks), paste them in a message and I'll help fix them.
## Convenience scripts (optional)

Added two small helper scripts to start the backend (WSL) and the frontend (Vite) together. Place the chosen script in the repo root and run it (adjust paths if your checkout is in a different location).

start-dev.ps1 (PowerShell)
```powershell
# start-dev.ps1 — run from Windows PowerShell (repo root)
wsl -e bash -lc "cd /mnt/d/Downloads/QRM/recession_project/server && . .venv/bin/activate && nohup python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info > uvicorn.log 2>&1 & echo $! > /tmp/uvicorn.pid"
Push-Location D:\Downloads\QRM\recession_project\frontend\heatmap-client
npm install
npm run dev
Pop-Location
```

start-wsl.sh (Bash / WSL)
```bash
#!/usr/bin/env bash
# start-wsl.sh — run from WSL/bash (repo root)
set -e
cd /mnt/d/Downloads/QRM/recession_project/server
source .venv/bin/activate
nohup python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info > uvicorn.log 2>&1 &
echo $! > /tmp/uvicorn.pid

cd /mnt/d/Downloads/QRM/recession_project/frontend/heatmap-client
npm install
npm run dev
```

Usage:
- PowerShell: .\start-dev.ps1
- WSL/Bash: chmod +x ./start-wsl.sh && ./start-wsl.sh

Adjust the D:/ or /mnt/d/ paths to match your local checkout if needed.

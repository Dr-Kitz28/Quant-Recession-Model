"""FastAPI service serving correlation heatmap frames on demand.

Run with:
    uvicorn frame_api:app --host 0.0.0.0 --port 8001 --reload

The service exposes:
    * GET /meta  -> metadata with dates, spread names and shape
    * GET /frame/{index} -> single frame as binary Float32 (row-major)
    * GET /frames?start=<a>&end=<b> -> batch frames (inclusive start, exclusive end)

The correlation tensor is loaded once (memory-mapped) so requests are fast
without exhausting memory, even for 10k+ frames.
"""

from __future__ import annotations

import json
import traceback
import logging
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
DEFAULT_NPZ = OUTPUTS_DIR / "correlation_tensor_usa.npz"

app = FastAPI(title="Correlation Heatmap Frame API", version="0.1.0")
app.add_middleware(GZipMiddleware, minimum_size=1_000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _compute_diff_order(spreads: np.ndarray) -> np.ndarray:
    """Order by maturity difference (long-short in months)."""
    def parse_maturity(s: str) -> int:
        if s.endswith('Y'):
            return int(s[:-1]) * 12
        if s.endswith('M'):
            return int(s[:-1])
        return 0

    def get_spread_diff(spread_name: str) -> int:
        parts = spread_name.split('-')
        if len(parts) == 2:
            return parse_maturity(parts[0]) - parse_maturity(parts[1])
        return 0

    diffs = [(i, get_spread_diff(str(s))) for i, s in enumerate(spreads)]
    diffs.sort(key=lambda x: x[1])
    return np.array([i for i, _ in diffs], dtype=int)


def _compute_cluster_order(avg_corr: np.ndarray) -> np.ndarray:
    """Compute hierarchical clustering order from an average/symmetric correlation matrix.

    Uses a simple agglomerative approach without external deps to avoid SciPy requirement.
    This is a greedy single-linkage approximation sufficient for ordering visualization.
    Input: avg_corr shape (n, n), values in [0,1], diagonal ~1.
    Output: permutation indices of length n.
    """
    n = avg_corr.shape[0]
    remaining = set(range(n))
    # start from the node with highest average correlation to others
    avg_strength = np.where(np.eye(n, dtype=bool), 0.0, avg_corr).mean(axis=1)
    current = int(np.argmax(avg_strength))
    order = [current]
    remaining.remove(current)
    # greedy nearest-neighbor path
    while remaining:
        # pick next with max correlation to any already in order (single-linkage)
        best_j = None
        best_val = -1.0
        for j in remaining:
            # use nanmax to be robust to any NaNs in the averaged matrix
            try:
                val = float(np.nanmax(avg_corr[j, order]))
            except Exception:
                # if something unexpected happens, treat as very low affinity
                val = -1.0
            if np.isnan(val):
                # ignore NaNs
                continue
            if val > best_val:
                best_val = val
                best_j = j
        # defensive fallback: if no best found (all NaNs or other issue), pick any remaining
        if best_j is None:
            best_j = next(iter(remaining))
        order.append(best_j)
        remaining.remove(best_j)
    return np.array(order, dtype=int)


@lru_cache(maxsize=1)
def _load_npz(npz_path: Path = DEFAULT_NPZ):
    if not npz_path.exists():
        raise FileNotFoundError(f"Correlation tensor not found at {npz_path}")
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    spreads = data["spreads"]
    corr = data["corr"]
    corr_scaled = data["corr_scaled"]

    # Precompute two orderings: maturity-diff and hierarchical clustering (from average corr)
    diff_order = _compute_diff_order(spreads)

    # Compute average correlation over time in a memory-friendly way
    # Avoid loading all dates into RAM: iteratively accumulate
    n_dates = corr.shape[0]
    n = corr.shape[1]
    avg_corr = np.zeros((n, n), dtype=np.float64)
    # sample or full accumulate; full accumulate but keep float64 for precision
    chunk = 256
    for start in range(0, n_dates, chunk):
        end = min(n_dates, start + chunk)
        # accumulate sum over the chunk to avoid weighting bias
        avg_corr += np.sum(corr[start:end], axis=0)
    avg_corr /= float(n_dates)
    # Ensure symmetry and unit diagonal
    avg_corr = (avg_corr + avg_corr.T) / 2.0
    np.fill_diagonal(avg_corr, 1.0)

    cluster_order = _compute_cluster_order(avg_corr)

    # Prepare both reordered variants
    def reorder(order: np.ndarray):
        return (
            spreads[order],
            corr[:, order, :][:, :, order],
            corr_scaled[:, order, :][:, :, order],
        )

    spreads_diff, corr_diff, corr_scaled_diff = reorder(diff_order)
    spreads_cluster, corr_cluster, corr_scaled_cluster = reorder(cluster_order)

    return {
        "path": npz_path,
        "dates": data["dates"],
        # store both orderings
        "spreads_diff": spreads_diff,
        "corr_diff": corr_diff,
        "corr_scaled_diff": corr_scaled_diff,
        "spreads_cluster": spreads_cluster,
        "corr_cluster": corr_cluster,
        "corr_scaled_cluster": corr_scaled_cluster,
    }


def _frame_shape(npz, order: str) -> Tuple[int, int, int]:
    corr = npz["corr_scaled_cluster"] if order == "clustered" else npz["corr_scaled_diff"]
    return corr.shape  # (n_dates, n_spreads, n_spreads)


@app.get("/meta")
def get_meta(order: str = Query("clustered", pattern="^(clustered|diff)$")):
    try:
        npz = _load_npz()
        n_dates, n_spreads, _ = _frame_shape(npz, order)
        return {
            "dates": npz["dates"].tolist(),
            "order": order,
            "spreads": (npz["spreads_cluster"] if order == "clustered" else npz["spreads_diff"]).tolist(),
            "n_dates": n_dates,
            "n_spreads": n_spreads,
            "tensor": {
                "path": str(npz["path"].as_posix()),
                "dtype": str((npz["corr_scaled_cluster"] if order == "clustered" else npz["corr_scaled_diff"]).dtype),
            },
        }
    except Exception as exc:
        logging.exception("Error in get_meta")
        # return a JSON-friendly error with traceback for local debugging
        raise HTTPException(status_code=500, detail={
            "error": str(exc),
            "trace": traceback.format_exc(),
        })


def _frame_bytes(npz, index: int, order: str) -> Tuple[bytes, int, int]:
    corr_scaled = npz["corr_scaled_cluster"] if order == "clustered" else npz["corr_scaled_diff"]
    n_dates, rows, cols = corr_scaled.shape
    if index < 0 or index >= n_dates:
        raise IndexError(f"index {index} out of range (0, {n_dates - 1})")
    frame = np.asarray(corr_scaled[index], dtype=np.float32)
    return frame.tobytes(), rows, cols


@app.get("/frame/{index}")
def get_frame(index: int, order: str = Query("clustered", pattern="^(clustered|diff)$")):
    npz = _load_npz()
    try:
        payload, rows, cols = _frame_bytes(npz, index, order)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    headers = {
        "X-Frame-Index": str(index),
        "X-Order": order,
        "X-Rows": str(rows),
        "X-Cols": str(cols),
        "X-DType": "float32",
        "Cache-Control": "public, max-age=604800",  # one week
    }
    return Response(content=payload, media_type="application/octet-stream", headers=headers)


@app.get("/frames")
def get_frames(
    start: int = Query(0, ge=0),
    end: int = Query(..., gt=0),
    order: str = Query("clustered", pattern="^(clustered|diff)$"),
):
    npz = _load_npz()
    n_dates, rows, cols = _frame_shape(npz, order)
    if end <= start:
        raise HTTPException(status_code=400, detail="end must be greater than start")
    if start >= n_dates:
        raise HTTPException(status_code=416, detail="start out of range")
    end = min(end, n_dates)

    corr_scaled = npz["corr_scaled_cluster"] if order == "clustered" else npz["corr_scaled_diff"]
    # Avoid large intermediates: slice and copy once into a pre-allocated buffer
    batch_np = np.empty((end - start, rows, cols), dtype=np.float32)
    # iterate in small chunks to reduce peak RAM
    chunk = 128
    out_pos = 0
    for s in range(start, end, chunk):
        e = min(end, s + chunk)
        batch_np[out_pos : out_pos + (e - s)] = corr_scaled[s:e].astype(np.float32, copy=False)
        out_pos += (e - s)
    headers = {
        "X-Start": str(start),
        "X-End": str(end),
        "X-Order": order,
        "X-Rows": str(rows),
        "X-Cols": str(cols),
        "X-DType": "float32",
        "Cache-Control": "public, max-age=86400",  # one day
    }
    return Response(content=batch_np.tobytes(), media_type="application/octet-stream", headers=headers)


@app.get("/tiles")
def get_tiles(level: int = Query(0, ge=0), tile: int = Query(0, ge=0), tile_size: int = Query(256, gt=0)):
    """
    Return a temporally-decimated tile of frames.

    - level: temporal downsample level (0 = original, 1 = every 2nd frame, 2 = every 4th frame, ...)
    - tile: tile index (0-based)
    - tile_size: number of frames per tile at the requested level

    The server computes step = 2**level, then returns frames for indices in [start, end) sampled every `step`.
    """
    npz = _load_npz()
    n_dates, rows, cols = _frame_shape(npz, order="clustered")

    step = 1 << int(level)
    # compute start/end in the original timeline
    start = tile * tile_size * step
    if start >= n_dates:
        raise HTTPException(status_code=416, detail="tile start out of range")
    end = min(n_dates, start + tile_size * step)

    # sample indices at the requested step
    indices = np.arange(start, end, step, dtype=int)
    batch = np.asarray(npz["corr_scaled"][indices], dtype=np.float32)

    headers = {
        "X-Tile-Level": str(level),
        "X-Tile-Index": str(tile),
        "X-Start": str(start),
        "X-End": str(end),
        "X-Step": str(step),
        "X-Count": str(batch.shape[0]),
        "X-Rows": str(rows),
        "X-Cols": str(cols),
        "X-DType": "float32",
        "Cache-Control": "public, max-age=86400",
    }

    return Response(content=batch.tobytes(), media_type="application/octet-stream", headers=headers)


@app.get("/")
def root():
    npz = _load_npz()
    n_dates, n_spreads, _ = _frame_shape(npz)
    return {
        "status": "ok",
        "npz": str(npz["path"].as_posix()),
        "n_dates": n_dates,
        "n_spreads": n_spreads,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("frame_api:app", host="0.0.0.0", port=8001, reload=True)

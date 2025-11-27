"""Export per-day correlation matrices for KnewIMP starter pipeline.

Usage:
    python export_knewimp_spreads.py \
        --npz ../outputs/correlation_tensor_usa.npz \
        --out KnewIMP_Spreads

Creates a subfolder per day named "00001_YYYY-MM-DD" containing
correlation_matrix.csv with values clipped into [-1, 1].
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def export(npz_path: Path, out_dir: Path, overwrite: bool = False) -> None:
    npz_path = npz_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {out_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    dates = data["dates"].astype(str)
    spreads = data["spreads"].astype(str)
    corr = np.clip(data["corr"], -1.0, 1.0)

    for idx, (date, mat) in enumerate(zip(dates, corr), start=1):
        folder = out_dir / f"{idx:05d}_{date}"
        folder.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(mat, index=spreads, columns=spreads)
        df.to_csv(folder / "correlation_matrix.csv", float_format="%.6f")

    print(f"Created {len(dates)} folders in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-day correlation CSVs for KnewIMP")
    parser.add_argument("--npz", type=Path, required=True, help="Path to correlation tensor npz")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.npz, args.out, overwrite=args.overwrite)

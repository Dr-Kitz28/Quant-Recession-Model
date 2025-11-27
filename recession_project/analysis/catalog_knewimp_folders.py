#!/usr/bin/env python3
"""Summarise KnewIMP export folders and imputation status.

This utility scans every sub-folder under KnewIMP_Spreads (or a user supplied
root), inspects the `correlation_matrix.csv`, and emits a catalog CSV that can
serve as ground-truth for downstream KnewIMP pipelines. The catalog keeps
pointers to both the raw correlation CSV and the expected location of an
imputed CSV so that batch jobs can discover which days still need attention.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

RAW_FILENAME = "correlation_matrix.csv"
IMPUTED_FILENAME = "imputed_correlation_matrix.csv"


@dataclass
class FolderSummary:
    folder_name: str
    as_of_date: str
    raw_csv: Path
    imputed_csv: Path
    imputed_exists: bool
    n_spreads: int
    n_cells: int
    n_missing: int

    @property
    def needs_imputation(self) -> bool:
        return self.n_missing > 0 and not self.imputed_exists


def scan_folder(folder: Path) -> FolderSummary:
    raw_csv = folder / RAW_FILENAME
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing {RAW_FILENAME} in {folder}")

    df = pd.read_csv(raw_csv, index_col=0)
    n_spreads = df.shape[0]
    n_cells = df.size
    n_missing = int(df.isna().sum().sum())

    imputed_csv = folder / IMPUTED_FILENAME
    return FolderSummary(
        folder_name=folder.name,
        as_of_date=folder.name.split("_", 1)[-1],
        raw_csv=raw_csv,
        imputed_csv=imputed_csv,
        imputed_exists=imputed_csv.exists(),
        n_spreads=n_spreads,
        n_cells=n_cells,
        n_missing=n_missing,
    )


def build_catalog(root: Path) -> List[FolderSummary]:
    summaries: List[FolderSummary] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        summaries.append(scan_folder(child))
    return summaries


def write_catalog(rows: Iterable[FolderSummary], output_csv: Path) -> None:
    fieldnames = [
        "folder_name",
        "as_of_date",
        "raw_csv",
        "imputed_csv",
        "imputed_exists",
        "needs_imputation",
        "n_spreads",
        "n_cells",
        "n_missing",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "folder_name": row.folder_name,
                    "as_of_date": row.as_of_date,
                    "raw_csv": str(row.raw_csv),
                    "imputed_csv": str(row.imputed_csv),
                    "imputed_exists": row.imputed_exists,
                    "needs_imputation": row.needs_imputation,
                    "n_spreads": row.n_spreads,
                    "n_cells": row.n_cells,
                    "n_missing": row.n_missing,
                }
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Catalog correlation exports and expected KnewIMP outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("KnewIMP_Spreads"),
        help="Directory containing per-day KnewIMP folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("KnewIMP_Spreads_catalog.csv"),
        help="Path to write the catalog CSV",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    summaries = build_catalog(root)
    if not summaries:
        raise RuntimeError(f"No folders found under {root}")

    output_csv = args.output.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_catalog(summaries, output_csv)
    print(f"Cataloged {len(summaries)} folders into {output_csv}")


if __name__ == "__main__":
    main()

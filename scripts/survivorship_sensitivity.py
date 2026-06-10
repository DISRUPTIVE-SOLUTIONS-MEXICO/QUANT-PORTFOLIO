"""Survivorship sensitivity: PIT universe vs current-constituents universe.

Compares two runner summary CSVs (one produced with ``--pit-universe``, one
without) and writes ``survivorship_sensitivity.csv`` with the active-return
delta per objective — an explicit upper bound on how much of the headline
performance could be a survivorship artifact.

    python scripts/survivorship_sensitivity.py --pit pit_summary.csv \
        --current current_summary.csv --out research_artifacts/survivorship_sensitivity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COMPARE_COLS = (
    "ann_return",
    "active_ann_return",
    "downside_capture",
    "maxdd_loss",
    "cvar_loss",
)


def build_sensitivity(pit: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    cols = ["objective", *[c for c in COMPARE_COLS if c in pit.columns and c in current.columns]]
    merged = pit[cols].merge(current[cols], on="objective", suffixes=("_pit", "_current"))
    for col in cols[1:]:
        merged[f"{col}_delta_current_minus_pit"] = merged[f"{col}_current"] - merged[f"{col}_pit"]
    if "active_ann_return_delta_current_minus_pit" in merged.columns:
        merged["survivorship_inflation_flag"] = merged["active_ann_return_delta_current_minus_pit"] > 0.01
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pit", required=True, help="summary CSV from the --pit-universe run")
    parser.add_argument("--current", required=True, help="summary CSV from the current-constituents run")
    parser.add_argument("--out", default="research_artifacts/survivorship_sensitivity.csv")
    args = parser.parse_args()
    pit = pd.read_csv(args.pit)
    current = pd.read_csv(args.current)
    out = build_sensitivity(pit, current)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

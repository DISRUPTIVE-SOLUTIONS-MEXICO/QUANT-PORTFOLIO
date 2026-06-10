"""Cross-source price reconciliation.

Robustness at zero cost comes from verifying free sources against each other
instead of trusting any single one. The reconciler compares overlapping daily
closes from a primary and a secondary provider, flags tickers whose relative
discrepancies exceed tolerance, and produces a consensus frame that prefers
the primary source while filling gaps from the secondary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconcile_price_frames(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    tolerance_bps: float = 50.0,
    max_discrepancy_rate: float = 0.01,
    min_overlap_days: int = 20,
    primary_name: str = "yfinance",
    secondary_name: str = "stooq",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (consensus_prices, reconciliation_report).

    A ticker is flagged when more than ``max_discrepancy_rate`` of overlapping
    days differ by more than ``tolerance_bps`` (relative, in basis points).
    Flagged tickers keep the primary series (auditable choice) but carry a
    ``Price_Quality_Warning`` so downstream freshness reports surface them.
    """
    primary = pd.DataFrame(primary).copy()
    secondary = pd.DataFrame(secondary).copy()
    if primary.empty and secondary.empty:
        return pd.DataFrame(), pd.DataFrame()
    if primary.empty:
        report = pd.DataFrame(
            [
                {
                    "Ticker": col,
                    "Primary_Source": primary_name,
                    "Secondary_Source": secondary_name,
                    "Overlap_Days": 0,
                    "Discrepancy_Rate": np.nan,
                    "Max_Rel_Diff_Bps": np.nan,
                    "Price_Quality_Warning": False,
                    "Consensus_Source": secondary_name,
                }
                for col in secondary.columns
            ]
        )
        return secondary, report

    tol = float(tolerance_bps) / 1e4
    rows = []
    consensus = primary.copy()
    secondary_only = [c for c in secondary.columns if c not in primary.columns]
    for col in primary.columns:
        if col not in secondary.columns:
            rows.append(
                {
                    "Ticker": col,
                    "Primary_Source": primary_name,
                    "Secondary_Source": secondary_name,
                    "Overlap_Days": 0,
                    "Discrepancy_Rate": np.nan,
                    "Max_Rel_Diff_Bps": np.nan,
                    "Price_Quality_Warning": False,
                    "Consensus_Source": primary_name,
                }
            )
            continue
        both = pd.concat([primary[col].rename("p"), secondary[col].rename("s")], axis=1).dropna()
        overlap = len(both)
        if overlap < int(min_overlap_days):
            disc_rate, max_diff, warn = np.nan, np.nan, False
        else:
            rel = (both["s"] / both["p"] - 1.0).abs()
            disc_rate = float((rel > tol).mean())
            max_diff = float(rel.max() * 1e4)
            warn = bool(disc_rate > float(max_discrepancy_rate))
        # Fill primary gaps from the secondary source (extension, not override).
        gap_fill = secondary[col].reindex(consensus.index)
        consensus[col] = consensus[col].fillna(gap_fill)
        rows.append(
            {
                "Ticker": col,
                "Primary_Source": primary_name,
                "Secondary_Source": secondary_name,
                "Overlap_Days": overlap,
                "Discrepancy_Rate": disc_rate,
                "Max_Rel_Diff_Bps": max_diff,
                "Price_Quality_Warning": warn,
                "Consensus_Source": primary_name,
            }
        )
    for col in secondary_only:
        consensus[col] = secondary[col].reindex(consensus.index)
        rows.append(
            {
                "Ticker": col,
                "Primary_Source": primary_name,
                "Secondary_Source": secondary_name,
                "Overlap_Days": 0,
                "Discrepancy_Rate": np.nan,
                "Max_Rel_Diff_Bps": np.nan,
                "Price_Quality_Warning": False,
                "Consensus_Source": secondary_name,
            }
        )
    report = pd.DataFrame(rows)
    return consensus.sort_index(), report

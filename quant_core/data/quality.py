"""Ingest-time data-quality diagnostics for price/volume panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_run_length(mask: pd.Series) -> int:
    """Longest run of consecutive True values."""
    if mask.empty:
        return 0
    groups = (~mask).cumsum()
    runs = mask.groupby(groups).sum()
    return int(runs.max()) if len(runs) else 0


def detect_price_anomalies(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    *,
    jump_sigma: float = 8.0,
    max_stale_days: int = 10,
    max_calendar_gap_days: int = 7,
) -> pd.DataFrame:
    """Per-ticker anomaly flags raised at ingest time.

    - ``Jump_Flag``: a daily return beyond ``jump_sigma`` standard deviations
      (possible split/adjustment error rather than a market move).
    - ``Stale_Flag``: more than ``max_stale_days`` consecutive identical
      prices (dead feed or delisted ticker forward-filled).
    - ``Gap_Flag``: a calendar hole longer than ``max_calendar_gap_days``.
    - ``Zero_Volume_Run``: longest stretch of zero/missing volume.
    """
    px = pd.DataFrame(prices).sort_index()
    if px.empty:
        return pd.DataFrame()
    rows = []
    index_gaps = px.index.to_series().diff().dt.days.fillna(1)
    max_gap = float(index_gaps.max()) if len(index_gaps) else np.nan
    for ticker in px.columns:
        series = px[ticker].dropna()
        if len(series) < 10:
            rows.append(
                {
                    "Ticker": ticker,
                    "Obs": len(series),
                    "Max_Abs_Return_Sigma": np.nan,
                    "Jump_Flag": False,
                    "Stale_Run_Days": np.nan,
                    "Stale_Flag": False,
                    "Calendar_Gap_Days_Max": max_gap,
                    "Gap_Flag": bool(max_gap > max_calendar_gap_days),
                    "Zero_Volume_Run": np.nan,
                    "Anomaly_Flag": False,
                }
            )
            continue
        rets = series.pct_change().dropna()
        sd = float(rets.std(ddof=1))
        max_sigma = float(rets.abs().max() / sd) if sd > 1e-12 else np.nan
        jump = bool(np.isfinite(max_sigma) and max_sigma > jump_sigma)
        stale_run = _max_run_length(series.diff().abs() < 1e-12)
        stale = bool(stale_run > max_stale_days)
        zero_vol_run = np.nan
        if volumes is not None and ticker in getattr(volumes, "columns", []):
            vol = volumes[ticker].reindex(series.index).fillna(0.0)
            zero_vol_run = float(_max_run_length(vol <= 0))
        gap = bool(max_gap > max_calendar_gap_days)
        rows.append(
            {
                "Ticker": ticker,
                "Obs": len(series),
                "Max_Abs_Return_Sigma": max_sigma,
                "Jump_Flag": jump,
                "Stale_Run_Days": float(stale_run),
                "Stale_Flag": stale,
                "Calendar_Gap_Days_Max": max_gap,
                "Gap_Flag": gap,
                "Zero_Volume_Run": zero_vol_run,
                "Anomaly_Flag": bool(jump or stale or gap),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["Anomaly_Flag", "Max_Abs_Return_Sigma"], ascending=[False, False]).reset_index(drop=True)

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_COLUMNS = (
    "ROIC",
    "EV_EBITDA",
    "FCF_Yield",
    "NetDebt_EBITDA",
    "Piotroski",
    "Asset_Turnover",
    "Altman_Z",
    "Interest_Coverage",
    "Retention_Ratio",
    "Earnings_Yield",
    "PB",
    "PE",
    "EPS",
    "Solvency",
    "ROE",
)


def add_pit_confidence(
    panel: pd.DataFrame, asof_date=None, ratio_cols: tuple[str, ...] = RATIO_COLUMNS
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()
    out = panel.copy()
    asof = pd.Timestamp(asof_date) if asof_date is not None else pd.Timestamp.utcnow().tz_localize(None)
    availability = pd.to_datetime(out.get("Availability_Date"), errors="coerce")
    staleness_days = (asof - availability).dt.days.clip(lower=0)
    has_sec = out.get("SEC_Facts_Coverage", pd.Series(0, index=out.index)).fillna(0).astype(float) > 0
    has_accept = (
        pd.to_datetime(out.get("SEC_Accepted_At"), errors="coerce").notna()
        if "SEC_Accepted_At" in out
        else pd.Series(False, index=out.index)
    )
    coverage = out.get("Valid_Fundamental_Ratios", pd.Series(np.nan, index=out.index))
    coverage = pd.to_numeric(coverage, errors="coerce").fillna(0.0).clip(0, 15) / 15.0
    stale_penalty = (staleness_days.fillna(999) / 540.0).clip(0, 1)
    base = (
        0.20 + 0.35 * has_sec.astype(float) + 0.20 * has_accept.astype(float) + 0.20 * coverage - 0.20 * stale_penalty
    )
    out["PIT_Confidence"] = base.clip(0.05, 0.95)
    out["PIT_Data_Class"] = np.select(
        [has_sec & has_accept, has_sec, availability.notna()],
        ["SEC accepted timestamp", "SEC companyfacts", "Yahoo/PIT approximation"],
        default="Unknown",
    )
    for ratio in ratio_cols:
        if ratio in out:
            out[f"{ratio}_PIT_Confidence"] = np.where(
                pd.to_numeric(out[ratio], errors="coerce").notna(), out["PIT_Confidence"], 0.0
            )
            out[f"{ratio}_Missing_Reason"] = np.where(
                pd.to_numeric(out[ratio], errors="coerce").notna(), "", "missing_or_not_applicable"
            )
            out[f"{ratio}_Source_Class"] = out["PIT_Data_Class"]
    return out

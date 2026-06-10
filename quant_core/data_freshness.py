from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE_TTL_HOURS = {
    "prices_daily": 24,
    "volume_daily": 24,
    "fundamentals_yfinance": 24,
    "fundamentals_sec_companyfacts": 168,
    "sec_nlp_filings": 168,
    "macro_fred": 24,
    "macro_country_direct": 24,
    "options_yahoo_snapshot": 24,
    "gdelt_timeline": 24,
    "gdelt_articles": 24,
    "google_news_rss_articles": 24,
    "forex_factory_calendar": 24,
    "interbank_reference_rates": 24,
    "public_news_translation_en": 24,
    "universe_sec_company_tickers": 168,
    "universe_nasdaq_trader": 168,
    "universe_sp500_wikipedia": 168,
    "fx_usd_value_series": 24,
}


SOURCE_LABELS = {
    "prices_daily": "Yahoo Finance prices",
    "volume_daily": "Yahoo Finance volume",
    "fundamentals_yfinance": "Yahoo Finance fundamentals",
    "fundamentals_sec_companyfacts": "SEC EDGAR companyfacts",
    "sec_nlp_filings": "SEC filings NLP",
    "macro_fred": "FRED macro",
    "macro_country_direct": "Direct sovereign-rate APIs",
    "options_yahoo_snapshot": "Yahoo options snapshot",
    "gdelt_timeline": "GDELT timeline",
    "gdelt_articles": "GDELT articles",
    "google_news_rss_articles": "Google News RSS fallback",
    "forex_factory_calendar": "ForexFactory / FairEconomy calendar",
    "interbank_reference_rates": "SOFR / SONIA / ESTR / TONAR",
    "public_news_translation_en": "Public news translation cache",
    "universe_sec_company_tickers": "SEC company tickers universe",
    "universe_nasdaq_trader": "Nasdaq public universe",
    "universe_sp500_wikipedia": "Wikipedia S&P 500 universe",
    "fx_usd_value_series": "Public FX proxy series",
}


def build_data_freshness_report(
    cache_inventory: pd.DataFrame, now=None, timezone: str = "America/Mexico_City"
) -> pd.DataFrame:
    """Convert raw cache inventory into a source-level freshness contract."""
    if cache_inventory is None or cache_inventory.empty:
        return pd.DataFrame(
            columns=[
                "Source",
                "Namespace",
                "Last_Update_CT",
                "Age_Hours",
                "TTL_Hours",
                "Status",
                "Rows",
                "Cache_Keys",
                "Fallback_Used",
            ]
        )
    inv = cache_inventory.copy()
    if "Namespace" not in inv:
        return pd.DataFrame()
    inv["Created_At"] = pd.to_datetime(inv.get("Created_At"), errors="coerce", utc=True)
    inv["Age_Hours"] = pd.to_numeric(inv.get("Age_Hours"), errors="coerce")
    inv["Rows"] = pd.to_numeric(inv.get("Rows"), errors="coerce")
    rows = []
    for namespace, group in inv.dropna(subset=["Namespace"]).groupby("Namespace"):
        latest = group.sort_values("Created_At").tail(1)
        latest_dt = latest["Created_At"].iloc[0] if not latest.empty else pd.NaT
        age = (
            float(latest["Age_Hours"].iloc[0]) if not latest.empty and pd.notna(latest["Age_Hours"].iloc[0]) else np.nan
        )
        ttl = float(SOURCE_TTL_HOURS.get(str(namespace), 24))
        if pd.isna(age):
            status = "unknown"
        elif age <= ttl:
            status = "fresh"
        elif age <= 2.0 * ttl:
            status = "stale"
        else:
            status = "expired"
        try:
            last_ct = latest_dt.tz_convert(timezone).strftime("%Y-%m-%d %H:%M:%S %Z") if pd.notna(latest_dt) else None
        except Exception:
            last_ct = str(latest_dt) if pd.notna(latest_dt) else None
        ns = str(namespace)
        rows.append(
            {
                "Source": SOURCE_LABELS.get(ns, ns.replace("_", " ").title()),
                "Namespace": ns,
                "Dataset": ns,
                "Last_Update_CT": last_ct,
                "Last_Update_Central": last_ct,
                "Age_Hours": age,
                "TTL_Hours": ttl,
                "Status": status,
                "Rows": int(group["Rows"].fillna(0).sum()) if "Rows" in group else 0,
                "Cache_Keys": int(len(group)),
                "Fallback_Used": bool("fallback" in ns or "google_news_rss" in ns),
                "Fallback": bool("fallback" in ns or "google_news_rss" in ns),
                "Errors": "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"expired": 0, "stale": 1, "unknown": 2, "fresh": 3}
    out["_order"] = out["Status"].map(order).fillna(2)
    return out.sort_values(["_order", "Source"]).drop(columns=["_order"]).reset_index(drop=True)

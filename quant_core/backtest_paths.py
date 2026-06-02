from __future__ import annotations

import numpy as np
import pandas as pd


def aligned_observed_price(curve: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.Series:
    if curve is None or curve.empty or prices is None or prices.empty or benchmark_ticker not in prices:
        return pd.Series(dtype=float)
    date_source = "Date" if "Date" in curve else "Period_End" if "Period_End" in curve else None
    if date_source is None:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(curve[date_source], errors="coerce")
    px = pd.to_numeric(prices[benchmark_ticker], errors="coerce").sort_index().ffill()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index()
    aligned = px.reindex(dates, method="ffill")
    aligned.index = curve.index
    return aligned


def period_holdings(holdings: pd.DataFrame, period_row: pd.Series, date_col_candidates: tuple[str, ...]) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    for col in date_col_candidates:
        if col in holdings and col in period_row:
            target = pd.Timestamp(period_row[col])
            hdates = pd.to_datetime(holdings[col], errors="coerce")
            sub = holdings[hdates.eq(target)].copy()
            if not sub.empty:
                return sub
    return pd.DataFrame()


def daily_synthetic_nav_from_holdings(
    schedule: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    holding_date_cols: tuple[str, ...],
    weight_col: str = "Effective_Weight",
    label: str = "Optimized synthetic NAV price",
    cost_cols: tuple[str, ...] = ("Fixed_TC", "Impact_TC"),
) -> pd.DataFrame:
    """Reconstruct a daily synthetic NAV using only OOS holdings and daily prices."""
    if schedule is None or schedule.empty or holdings is None or holdings.empty or prices is None or prices.empty:
        return pd.DataFrame()
    if benchmark_ticker not in prices:
        return pd.DataFrame()
    px = prices.copy()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index().ffill()
    rows: list[dict] = []
    nav = 1.0
    prior_date = None
    schedule = schedule.copy()
    for col in ["OOS_Start", "OOS_End", "Period_End", "Rebalance_Date"]:
        if col in schedule:
            schedule[col] = pd.to_datetime(schedule[col], errors="coerce")
    date_sort = "OOS_Start" if "OOS_Start" in schedule else "Period_End"
    for _, period in schedule.sort_values(date_sort).iterrows():
        start = pd.Timestamp(period.get("OOS_Start", period.get("Rebalance_Date", pd.NaT)))
        end = pd.Timestamp(period.get("OOS_End", period.get("Period_End", pd.NaT)))
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        h = period_holdings(holdings, period, holding_date_cols)
        if h.empty or "Ticker" not in h:
            continue
        wc = weight_col if weight_col in h else "Weight" if "Weight" in h else None
        if wc is None:
            continue
        weights = (
            h[~h["Ticker"].astype(str).str.upper().eq("CASH")]
            .groupby("Ticker")[wc]
            .sum()
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
        )
        tickers = [t for t in weights.index if t in px.columns]
        if not tickers:
            continue
        interval = px.loc[(px.index >= start) & (px.index <= end), list(dict.fromkeys(tickers + [benchmark_ticker]))].copy()
        if len(interval) < 2:
            continue
        asset_ret = interval[tickers].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weights = weights.reindex(tickers).fillna(0.0).astype(float)
        daily_ret = asset_ret @ weights
        period_cost = 0.0
        for c in cost_cols:
            val = pd.to_numeric(pd.Series([period.get(c, 0.0)]), errors="coerce").iloc[0]
            if pd.notna(val) and np.isfinite(float(val)):
                period_cost += float(val)
        first_return_idx = daily_ret.index[1] if len(daily_ret) > 1 else daily_ret.index[0]
        if np.isfinite(period_cost) and abs(period_cost) > 0:
            daily_ret.loc[first_return_idx] = daily_ret.loc[first_return_idx] - period_cost
        bench = interval[benchmark_ticker]
        for dt in daily_ret.index:
            ts = pd.Timestamp(dt)
            if prior_date is not None and ts <= prior_date:
                continue
            nav *= 1.0 + float(daily_ret.loc[dt])
            rows.append({"Date": ts, "_Synthetic_NAV": nav, f"{benchmark_ticker} observed price": float(bench.loc[dt])})
            prior_date = ts
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["Date", f"{benchmark_ticker} observed price"]).sort_values("Date")
    benchmark_first = out[f"{benchmark_ticker} observed price"].dropna()
    nav_first = out["_Synthetic_NAV"].dropna()
    if benchmark_first.empty or nav_first.empty:
        return pd.DataFrame()
    anchor = float(benchmark_first.iloc[0])
    out[label] = anchor * out["_Synthetic_NAV"] / float(nav_first.iloc[0])
    return out.drop(columns=["_Synthetic_NAV"])


def daily_backtest_price_frame(perf: pd.DataFrame, holdings: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    return daily_synthetic_nav_from_holdings(
        perf,
        holdings,
        prices,
        benchmark_ticker,
        holding_date_cols=("Rebalance_Date", "OOS_Start", "Period_End"),
        weight_col="Effective_Weight",
        label="Sortino optimized synthetic NAV price",
    )


def daily_side_price_frame(side_perf: pd.DataFrame, side_holdings: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    return daily_synthetic_nav_from_holdings(
        side_perf,
        side_holdings,
        prices,
        benchmark_ticker,
        holding_date_cols=("OOS_Start", "Rebalance_Date", "Period_End"),
        weight_col="Weight",
        label="Private Side Alpha synthetic NAV price",
        cost_cols=(),
    )


def backtest_price_frame(curve: pd.DataFrame, benchmark_ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()
    benchmark_price = aligned_observed_price(curve, prices, benchmark_ticker)
    if benchmark_price.empty or benchmark_price.dropna().empty:
        return pd.DataFrame()
    anchor_price = float(benchmark_price.dropna().iloc[0])
    if not np.isfinite(anchor_price) or anchor_price <= 0:
        return pd.DataFrame()
    out = pd.DataFrame({"Period_End": pd.to_datetime(curve["Period_End"], errors="coerce")})
    out[f"{benchmark_ticker} observed price"] = benchmark_price.values
    for src, label in [
        ("Portfolio_Equity", "Sortino optimized synthetic NAV price"),
        ("Side_Boom_Equity", "Private Side Alpha synthetic NAV price"),
    ]:
        if src in curve and pd.to_numeric(curve[src], errors="coerce").notna().any():
            equity = pd.to_numeric(curve[src], errors="coerce")
            first_equity = float(equity.dropna().iloc[0]) if not equity.dropna().empty else np.nan
            if np.isfinite(first_equity) and abs(first_equity) > 1e-12:
                out[label] = anchor_price * equity / first_equity
    return out.dropna(subset=["Period_End"]).dropna(axis=1, how="all")


def price_drawdown_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    if price_frame is None or price_frame.empty:
        return pd.DataFrame()
    date_col = "Date" if "Date" in price_frame else "Period_End" if "Period_End" in price_frame else None
    if date_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({date_col: pd.to_datetime(price_frame[date_col], errors="coerce")})
    for col in price_frame.columns:
        if col == date_col:
            continue
        px = pd.to_numeric(price_frame[col], errors="coerce")
        if px.notna().sum() < 2:
            continue
        out[col] = px / px.cummax() - 1.0
    return out.dropna(subset=[date_col]).dropna(axis=1, how="all")


def max_drawdown_table(price_frame: pd.DataFrame) -> pd.DataFrame:
    dd = price_drawdown_frame(price_frame)
    if dd.empty:
        return pd.DataFrame()
    date_col = "Date" if "Date" in dd else "Period_End"
    rows = []
    for col in dd.columns:
        if col == date_col:
            continue
        y = pd.to_numeric(dd[col], errors="coerce")
        if y.dropna().empty:
            continue
        idx = y.idxmin()
        rows.append({"Series": col, "Max_Drawdown": float(y.loc[idx]), "Max_Drawdown_Date": dd.loc[idx, date_col]})
    return pd.DataFrame(rows).sort_values("Max_Drawdown")


def build_backtest_path_bundle(
    perf: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    equity_curve: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict]:
    price_paths = daily_backtest_price_frame(perf, holdings, prices, benchmark_ticker)
    if price_paths.empty and equity_curve is not None:
        price_paths = backtest_price_frame(equity_curve, benchmark_ticker, prices)
    drawdowns = price_drawdown_frame(price_paths)
    return {
        "price_paths": price_paths,
        "drawdowns": drawdowns,
        "max_drawdown_table": max_drawdown_table(price_paths),
        "path_metadata": {
            "benchmark": benchmark_ticker,
            "source": "daily_oos_holdings" if not price_paths.empty and "Date" in price_paths else "period_equity_fallback",
            "drawdown_formula": "P_t / running_max(P_t) - 1",
        },
    }

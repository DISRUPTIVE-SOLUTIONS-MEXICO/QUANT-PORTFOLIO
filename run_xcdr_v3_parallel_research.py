from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from quant_core.uncertainty_state import (
    upside_downside_diagnostics,
    xcdr_v3_growth_control_score,
    xodr_v1_omega_dominance_score,
)


TRADING_DAYS = 252
SEED = 1729
RNG = np.random.default_rng(SEED)
CACHE_DIR = Path(r"C:\Users\chris\Downloads\qpk_market_cache")
OUT_DIR = Path(r"C:\Users\chris\Downloads")

BENCHMARKS = [
    "SPY",
    "QQQ",
    "ACWI",
    "VT",
    "VTI",
    "IWM",
    "USMV",
    "SPLV",
    "MTUM",
    "QUAL",
    "VLUE",
    "XLK",
    "XLV",
    "XLE",
    "XLU",
    "XLF",
    "XLI",
    "XLY",
    "XLP",
    "XLB",
    "XLRE",
]
REFERENCE_ASSETS = ["SHY", "IEF", "TLT", "GLD", "DBC", "UUP"]
DEFAULT_PROMOTION_OBJECTIVES = (
    "capital_preservation_policy",
    "downside_preserving_growth_policy",
    "enhanced_growth_anchor_dd_budget_policy",
    "fundamental_upside_convex_anchor_dd_budget_policy",
)


@dataclass(frozen=True)
class BatchConfig:
    train_days: int = 756
    validation_days: int = 126
    test_days: int = 42
    purge_days: int = 5
    embargo_days: int = 1
    universe_limit: int = 90
    max_weight: float = 0.08
    max_windows: int = 18
    workers: int = max(2, min(8, (os.cpu_count() or 4) - 1))
    bootstrap_n: int = 300
    pso_particles: int = 18
    pso_iterations: int = 18
    promotion_objectives: tuple[str, ...] = DEFAULT_PROMOTION_OBJECTIVES
    min_promotion_windows: int = 12
    bootstrap_block_length: int = 0


def parse_objective_list(raw: str | None, default: tuple[str, ...] = DEFAULT_PROMOTION_OBJECTIVES) -> tuple[str, ...]:
    if not raw:
        return default
    items = tuple(dict.fromkeys(x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()))
    return items or default


def _latest_cache_key() -> str:
    candidates = []
    for meta_path in CACHE_DIR.glob("market_*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(meta.get("period", "")).lower() == "10y" and int(meta.get("columns", 0)) >= 200:
            candidates.append((meta_path.stat().st_mtime, meta_path.stem.replace("market_", "")))
    if not candidates:
        raise FileNotFoundError(f"No usable 10y market cache found in {CACHE_DIR}")
    return sorted(candidates, reverse=True)[0][1]


def load_cached_returns(cache_key: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    key = cache_key or _latest_cache_key()
    meta = json.loads((CACHE_DIR / f"market_{key}.json").read_text(encoding="utf-8"))
    prices = pd.read_parquet(CACHE_DIR / f"prices_{key}.parquet").sort_index().ffill()
    volumes = pd.read_parquet(CACHE_DIR / f"volumes_{key}.parquet").reindex(prices.index).ffill()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all").fillna(0.0)
    return returns, volumes, meta


def annual_return(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() * TRADING_DAYS) if len(r) else np.nan


def annual_vol(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 else np.nan


def downside_ann(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(TRADING_DAYS))


def cvar_loss(r: pd.Series, alpha: float = 0.95) -> float:
    losses = -pd.Series(r).dropna()
    if losses.empty:
        return np.nan
    q = losses.quantile(alpha)
    tail = losses[losses >= q]
    return float(tail.mean()) if len(tail) else float(q)


def max_dd_loss(r: pd.Series) -> float:
    nav = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    dd = nav / nav.cummax() - 1.0
    return float(-dd.min())


def drawdown_loss_path(r: pd.Series) -> pd.Series:
    nav = (1.0 + pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)).cumprod()
    if nav.empty:
        return pd.Series(dtype=float)
    return (1.0 - nav / nav.cummax()).clip(lower=0.0)


def benchmark_relative_drawdown_diagnostics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    abs_buffer: float = 0.008,
    rel_buffer: float = 0.15,
) -> dict[str, float]:
    """Pathwise drawdown excess versus a benchmark drawdown budget.

    The budget is not a point estimate of max drawdown. It is a daily path:
    portfolio drawdown is allowed to exceed benchmark drawdown only by a small
    preregistered absolute and relative buffer.
    """
    p = pd.Series(portfolio_returns, dtype=float).dropna()
    x = pd.Series(benchmark_returns, dtype=float).reindex(p.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pdd = drawdown_loss_path(p)
    xdd = drawdown_loss_path(x).reindex(pdd.index).fillna(0.0)
    if pdd.empty:
        return {
            "path_dd_max_excess": np.nan,
            "path_dd_breach_area": np.nan,
            "path_dd_breach_rate": np.nan,
            "path_dd_ratio_xi": np.nan,
        }
    budget = xdd + abs_buffer + rel_buffer * xdd
    excess = (pdd - budget).clip(lower=0.0)
    return {
        "path_dd_max_excess": float(excess.max()),
        "path_dd_breach_area": float(excess.mean()),
        "path_dd_breach_rate": float((excess > 1e-12).mean()),
        "path_dd_ratio_xi": float(pdd.max() / (xdd.max() + 1e-12)),
    }


def causal_drawdown_vol_overlay(
    risk_returns: pd.Series,
    anchor_returns: pd.Series,
    state: dict | None = None,
    *,
    benchmark_returns: pd.Series | None = None,
    min_exposure: float = 0.25,
    rerisk_step: float = 0.12,
    dd_soft_shift: float = 0.0,
    dd_hard_gap: float | None = None,
    vol_target_shift: float = 0.0,
    relative_dd_buffer: float = 0.008,
    relative_dd_multiplier: float = 0.15,
) -> tuple[pd.Series, dict[str, float]]:
    """Causal exposure throttle using only information available before day t.

    The overlay blends a risky sleeve with a defensive anchor. Exposure for day
    t is a deterministic function of adjusted returns observed through t-1:
    realized drawdown, realized volatility, and the pre-test state estimated on
    train. It therefore does not inspect the return it is about to trade.
    """
    raw = pd.Series(risk_returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    anchor = pd.Series(anchor_returns, dtype=float).reindex(raw.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark = None
    if benchmark_returns is not None:
        benchmark = (
            pd.Series(benchmark_returns, dtype=float)
            .reindex(raw.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    stress = float((state or {}).get("state_stress", 0.50))
    risk_on = float((state or {}).get("state_risk_on", 0.0))
    recovery = float((state or {}).get("state_recovery", 0.0))
    dd_soft = float(np.clip(0.055 - 0.025 * stress + 0.010 * risk_on + dd_soft_shift, 0.012, 0.080))
    hard_gap = float(dd_hard_gap) if dd_hard_gap is not None else float(0.055 - 0.015 * stress)
    dd_hard = float(np.clip(dd_soft + hard_gap, dd_soft + 0.010, dd_soft + 0.080))
    vol_target = float(np.clip(0.145 + 0.045 * risk_on + 0.025 * recovery - 0.045 * stress + vol_target_shift, 0.070, 0.210))

    nav = 1.0
    peak = 1.0
    bench_nav = 1.0
    bench_peak = 1.0
    exposure = 1.0
    adjusted = []
    exposures = []
    history = []
    triggers = 0
    path_budget_triggers = 0
    prev_benchmark_return = 0.0
    for dt in raw.index:
        dd_loss = max(1.0 - nav / max(peak, 1e-12), 0.0)
        effective_dd_soft = dd_soft
        effective_dd_hard = dd_hard
        path_scale = 1.0
        if benchmark is not None:
            bench_dd_loss = max(1.0 - bench_nav / max(bench_peak, 1e-12), 0.0)
            path_soft = bench_dd_loss + relative_dd_buffer + relative_dd_multiplier * bench_dd_loss
            path_hard = path_soft + max(0.010, 0.50 * hard_gap)
            effective_dd_soft = max(effective_dd_soft, float(np.clip(path_soft, 0.008, 0.095)))
            effective_dd_hard = max(
                effective_dd_hard,
                float(np.clip(path_hard, effective_dd_soft + 0.008, 0.125)),
            )
            if dd_loss > path_soft:
                path_budget_triggers += 1
                if dd_loss >= path_hard:
                    path_scale = min_exposure
                else:
                    path_frac = (dd_loss - path_soft) / max(path_hard - path_soft, 1e-12)
                    path_scale = 1.0 - path_frac * (1.0 - min_exposure)

        if dd_loss <= effective_dd_soft:
            dd_scale = 1.0
        elif dd_loss >= effective_dd_hard:
            dd_scale = min_exposure
        else:
            frac = (dd_loss - effective_dd_soft) / max(effective_dd_hard - effective_dd_soft, 1e-12)
            dd_scale = 1.0 - frac * (1.0 - min_exposure)
        dd_scale = min(dd_scale, path_scale)

        if len(history) >= 21:
            realized_vol = float(pd.Series(history[-21:]).std(ddof=1) * np.sqrt(TRADING_DAYS))
            vol_scale = min(1.0, vol_target / max(realized_vol, 1e-12))
        else:
            vol_scale = 1.0

        target_exposure = float(np.clip(min(dd_scale, vol_scale), min_exposure, 1.0))
        if target_exposure > exposure:
            step = rerisk_step
            if benchmark is not None and dd_loss <= effective_dd_soft and prev_benchmark_return > 0.0:
                step = min(0.35, rerisk_step + 0.10 + 0.05 * risk_on + 0.04 * recovery)
            exposure = min(target_exposure, exposure + step)
        else:
            exposure = target_exposure
        if exposure < 0.999:
            triggers += 1
        ret = exposure * float(raw.loc[dt]) + (1.0 - exposure) * float(anchor.loc[dt])
        adjusted.append(ret)
        exposures.append(exposure)
        history.append(ret)
        nav *= 1.0 + ret
        peak = max(peak, nav)
        if benchmark is not None:
            prev_benchmark_return = float(benchmark.loc[dt])
            bench_nav *= 1.0 + prev_benchmark_return
            bench_peak = max(bench_peak, bench_nav)
    out = pd.Series(adjusted, index=raw.index)
    diag = {
        "overlay_avg_exposure": float(np.mean(exposures)) if exposures else np.nan,
        "overlay_min_exposure": float(np.min(exposures)) if exposures else np.nan,
        "overlay_trigger_rate": float(triggers / max(len(exposures), 1)),
        "overlay_dd_soft": dd_soft,
        "overlay_dd_hard": dd_hard,
        "overlay_vol_target": vol_target,
        "overlay_min_exposure_budget": float(min_exposure),
        "overlay_rerisk_step": float(rerisk_step),
        "overlay_dd_soft_shift": float(dd_soft_shift),
        "overlay_dd_hard_gap": float(hard_gap),
        "overlay_vol_target_shift": float(vol_target_shift),
        "overlay_path_budget_trigger_rate": float(path_budget_triggers / max(len(exposures), 1)),
    }
    if benchmark is not None:
        diag.update(
            {
                f"overlay_{k}": v
                for k, v in benchmark_relative_drawdown_diagnostics(
                    out,
                    benchmark,
                    abs_buffer=relative_dd_buffer,
                    rel_buffer=relative_dd_multiplier,
                ).items()
            }
        )
    return out, diag


def robust_z(x: pd.Series) -> pd.Series:
    x = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    med = float(x.median())
    scale = 1.4826 * float(np.median(np.abs(x - med)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(x.std(ddof=0)) if x.std(ddof=0) > 1e-12 else 1.0
    return ((x - med) / scale).clip(-3.0, 3.0)


def sector_robust_z(values: pd.Series, sectors: pd.Series) -> pd.Series:
    values = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    sectors = pd.Series(sectors, index=values.index).fillna("Unknown").astype(str)
    global_z = robust_z(values)
    out = pd.Series(index=values.index, dtype=float)
    for _, idx in sectors.groupby(sectors).groups.items():
        idx = list(idx)
        if len(idx) >= 5 and values.reindex(idx).notna().sum() >= 3:
            out.loc[idx] = robust_z(values.reindex(idx))
        else:
            out.loc[idx] = global_z.reindex(idx)
    return out.fillna(0.0).clip(-3.0, 3.0)


@lru_cache(maxsize=1)
def load_fundamental_cache() -> pd.DataFrame:
    """Load latest local public fundamental caches.

    The research script remains cost-zero and offline-fast by consuming the
    existing parquet cache produced by the main pipeline. Availability_Date is
    respected later at each train window.
    """
    root = Path(r"C:\Users\chris\FINANZAS\.quant_cache")
    frames = []
    for namespace in ["fundamentals_sec_companyfacts", "fundamentals_yfinance"]:
        cache_dir = root / namespace
        files = sorted(cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            continue
        try:
            frame = pd.read_parquet(files[0])
        except Exception:
            continue
        if "Ticker" not in frame:
            continue
        frame = frame.copy()
        frame["Fundamental_Cache_Namespace"] = namespace
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["Availability_Date", "Period_End", "SEC_Accepted_At", "SEC_Filing_Date"]:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper()
    return out


def safe_div(a, b) -> float:
    a = float(a) if pd.notna(a) else np.nan
    b = float(b) if pd.notna(b) else np.nan
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return np.nan
    return a / b


def fundamental_scores_asof(tickers: list[str], asof_date) -> pd.DataFrame:
    panel = load_fundamental_cache()
    tickers = [str(t).upper() for t in tickers]
    if panel.empty or not tickers:
        return pd.DataFrame(index=tickers)
    asof = pd.Timestamp(asof_date)
    if "Availability_Date" not in panel:
        return pd.DataFrame(index=tickers)
    eligible = panel[panel["Ticker"].isin(tickers) & (panel["Availability_Date"] <= asof)].copy()
    if eligible.empty:
        return pd.DataFrame(index=tickers)
    eligible = eligible.sort_values(["Ticker", "Availability_Date", "Period_End"], na_position="first")
    rows = []
    for ticker, grp in eligible.groupby("Ticker", sort=False):
        latest = grp.iloc[-1]
        prev = grp.iloc[-2] if len(grp) >= 2 else pd.Series(dtype=float)
        revenue = latest.get("_revenue")
        prev_revenue = prev.get("_revenue", np.nan)
        net_income = latest.get("_net_income")
        prev_net_income = prev.get("_net_income", np.nan)
        shares = latest.get("_shares")
        prev_shares = prev.get("_shares", np.nan)
        cfo = latest.get("_cfo")
        capex = latest.get("_capex")
        fcf_stmt = latest.get("_fcf_statement")
        fcf = fcf_stmt if pd.notna(fcf_stmt) else (cfo + capex if pd.notna(cfo) and pd.notna(capex) else np.nan)
        ebit = latest.get("_ebit")
        gross_profit = latest.get("_gross_profit")
        nopat = latest.get("_nopat")
        assets = latest.get("_assets")
        liabilities = latest.get("_liabilities")
        debt = latest.get("_debt")
        cash = latest.get("_cash")
        equity = latest.get("_equity")
        ebitda = latest.get("_ebitda")
        sector = latest.get("Sector", "Unknown")
        if not isinstance(sector, str) or sector == "Unknown":
            known_sector = grp["Sector"].dropna().astype(str)
            known_sector = known_sector[known_sector.ne("Unknown")]
            sector = known_sector.iloc[-1] if len(known_sector) else "Unknown"
        eps = safe_div(net_income, shares)
        prev_eps = safe_div(prev_net_income, prev_shares)
        invested_capital = np.nan
        if pd.notna(debt) or pd.notna(equity) or pd.notna(cash):
            invested_capital = (0.0 if pd.isna(debt) else float(debt)) + (0.0 if pd.isna(equity) else float(equity)) - (0.0 if pd.isna(cash) else float(cash))
        rows.append(
            {
                "Ticker": ticker,
                "Sector": sector,
                "Fundamental_Availability_Date": latest.get("Availability_Date"),
                "Fundamental_Source": latest.get("Fundamental_Source", latest.get("Fundamental_Cache_Namespace", "cache")),
                "Fundamental_Staleness_Days": float((asof - pd.Timestamp(latest.get("Availability_Date"))).days)
                if pd.notna(latest.get("Availability_Date"))
                else np.nan,
                "Revenue_Growth": safe_div(float(revenue) - float(prev_revenue), abs(float(prev_revenue))) if pd.notna(revenue) and pd.notna(prev_revenue) else np.nan,
                "EPS_Growth": safe_div(eps - prev_eps, abs(prev_eps)) if pd.notna(eps) and pd.notna(prev_eps) else np.nan,
                "Gross_Margin": safe_div(gross_profit, revenue),
                "EBIT_Margin": safe_div(ebit, revenue),
                "FCF_Margin": safe_div(fcf, revenue),
                "ROIC": safe_div(nopat, invested_capital),
                "ROE": safe_div(net_income, equity),
                "NetDebt_EBITDA": safe_div((0.0 if pd.isna(debt) else float(debt)) - (0.0 if pd.isna(cash) else float(cash)), ebitda),
                "Solvency": safe_div(assets, liabilities),
                "Fundamental_Coverage": float(
                    sum(pd.notna(x) for x in [revenue, net_income, cfo, ebit, gross_profit, nopat, assets, liabilities, equity])
                    / 9.0
                ),
            }
        )
    df = pd.DataFrame(rows).set_index("Ticker")
    if df.empty:
        return pd.DataFrame(index=tickers)
    sectors = df["Sector"].fillna("Unknown")
    growth_parts = []
    for col in ["Revenue_Growth", "EPS_Growth", "Gross_Margin", "EBIT_Margin", "FCF_Margin"]:
        if col in df:
            growth_parts.append(sector_robust_z(df[col], sectors))
    quality_parts = []
    for col in ["ROIC", "ROE", "Solvency"]:
        if col in df:
            quality_parts.append(sector_robust_z(df[col], sectors))
    leverage_penalty = sector_robust_z(df["NetDebt_EBITDA"], sectors) if "NetDebt_EBITDA" in df else pd.Series(0.0, index=df.index)
    df["Fundamental_Growth_Score"] = pd.concat(growth_parts, axis=1).mean(axis=1, skipna=True) if growth_parts else 0.0
    df["Fundamental_Quality_Score"] = pd.concat(quality_parts, axis=1).mean(axis=1, skipna=True) if quality_parts else 0.0
    source = df["Fundamental_Source"].fillna("").astype(str).str.lower()
    source_confidence = pd.Series(0.55, index=df.index)
    source_confidence[source.str.contains("sec|companyfacts", regex=True)] = 1.00
    source_confidence[source.str.contains("yfinance|yahoo", regex=True)] = 0.62
    freshness = np.exp(-df["Fundamental_Staleness_Days"].fillna(720.0).clip(lower=0.0) / 540.0)
    coverage = df["Fundamental_Coverage"].fillna(0.0).clip(0.0, 1.0)
    df["Fundamental_PIT_Confidence"] = (source_confidence * freshness * coverage).clip(0.0, 1.0)
    raw_upside_score = (
        0.50 * df["Fundamental_Growth_Score"].fillna(0.0)
        + 0.35 * df["Fundamental_Quality_Score"].fillna(0.0)
        - 0.25 * leverage_penalty.fillna(0.0)
        + 0.15 * robust_z(df["Fundamental_Coverage"].fillna(0.0))
    ).clip(-3.0, 3.0)
    confidence_shrink = 0.25 + 0.75 * df["Fundamental_PIT_Confidence"]
    df["Fundamental_Upside_Score"] = (raw_upside_score * confidence_shrink).clip(-3.0, 3.0)
    df["Fundamental_Data_Quality_Flag"] = np.where(
        df["Fundamental_PIT_Confidence"] >= 0.35,
        "usable",
        "low_confidence",
    )
    return df.reindex(tickers)


def project_weights(raw: pd.Series, max_weight: float) -> pd.Series:
    w = pd.Series(raw, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if w.sum() <= 1e-12:
        w[:] = 1.0
    w = w / w.sum()
    w = w.clip(upper=max_weight)
    for _ in range(50):
        resid = 1.0 - float(w.sum())
        if abs(resid) < 1e-12:
            break
        room = (max_weight - w).clip(lower=0.0)
        if room.sum() <= 1e-12:
            break
        w += min(resid, float(room.sum())) * room / room.sum()
        w = w.clip(upper=max_weight)
    return w / w.sum()


def make_schedule(index: pd.Index, cfg: BatchConfig) -> list[dict]:
    idx = pd.DatetimeIndex(index).sort_values()
    start = cfg.train_days + cfg.validation_days + cfg.purge_days + cfg.embargo_days
    rows = []
    for t in range(start, len(idx) - cfg.test_days, cfg.test_days):
        rows.append(
            {
                "train_start": idx[t - cfg.validation_days - cfg.purge_days - cfg.train_days],
                "train_end": idx[t - cfg.validation_days - cfg.purge_days - 1],
                "validation_start": idx[t - cfg.validation_days],
                "validation_end": idx[t - 1],
                "test_start": idx[t],
                "test_end": idx[min(t + cfg.test_days - 1, len(idx) - 1)],
            }
        )
    return rows[-cfg.max_windows :] if cfg.max_windows and len(rows) > cfg.max_windows else rows


def select_xi(train: pd.DataFrame, omega: pd.DataFrame, max_weight: float) -> tuple[str, pd.DataFrame]:
    vols = train.std(ddof=1).replace(0, np.nan)
    inv = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proxy_w = project_weights(inv, max_weight=max_weight)
    proxy = train @ proxy_w
    rows = []
    for b in omega.columns:
        s = omega[b].reindex(proxy.index).fillna(0.0)
        if len(s) < 60 or s.std(ddof=1) <= 1e-12:
            continue
        corr = float(proxy.corr(s))
        beta = float(np.cov(proxy, s)[0, 1] / (np.var(s, ddof=1) + 1e-12))
        te = annual_vol(proxy - s)
        score = 0.65 * corr - 0.25 * abs(beta - 1.0) - 0.70 * te
        rows.append({"Benchmark": b, "Correlation": corr, "Beta": beta, "TE": te, "FitScore": score})
    tab = pd.DataFrame(rows).sort_values("FitScore", ascending=False)
    if tab.empty:
        return "SPY", tab
    return str(tab.iloc[0]["Benchmark"]), tab


def rmt_crowding_penalty(train: pd.DataFrame) -> pd.Series:
    x = train - train.mean()
    if x.shape[1] < 3 or x.shape[0] < 60:
        return pd.Series(0.0, index=train.columns)
    cov = x.cov().values * TRADING_DAYS
    vols = np.sqrt(np.diag(cov)).clip(1e-12)
    corr = np.nan_to_num(cov / np.outer(vols, vols))
    corr = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(corr)
    q = train.shape[1] / max(train.shape[0], 1)
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    signal = vals > lam_plus
    if not signal.any():
        return pd.Series(0.0, index=train.columns)
    raw = (vecs[:, signal] ** 2) @ vals[signal]
    return robust_z(pd.Series(raw, index=train.columns)).clip(lower=0.0)


def convex_opportunity_universe_builder(
    train: pd.DataFrame,
    volumes: pd.DataFrame,
    candidate_cols: list[str],
    xi_train: pd.Series,
    fundamental_frame: pd.DataFrame,
    limit: int,
) -> tuple[list[str], pd.DataFrame]:
    """Causal prefilter for convex upside opportunities before portfolio construction.

    The builder operates only on the training window. It widens the initial
    candidate set beyond simple momentum/downside ranking and then filters on
    benchmark-relative upside convexity, residual momentum, sector-relative
    fundamentals, liquidity and RMT crowding.
    """
    cols = [c for c in candidate_cols if c in train.columns]
    if not cols:
        return [], pd.DataFrame()
    x = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xi = xi_train.reindex(x.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    up_mask = xi > 0.0
    down_mask = xi < 0.0
    top_threshold = float(xi.quantile(0.65)) if len(xi.dropna()) else 0.0
    top_mask = xi >= top_threshold
    tail_threshold = float(xi.quantile(0.10)) if len(xi.dropna()) else 0.0
    tail_mask = xi <= tail_threshold
    X = np.column_stack([np.ones(len(xi)), xi.to_numpy()])
    crowd = rmt_crowding_penalty(x)
    fundamental = (
        fundamental_frame["Fundamental_Upside_Score"].reindex(cols).fillna(0.0)
        if "Fundamental_Upside_Score" in fundamental_frame
        else pd.Series(0.0, index=cols)
    )
    fundamental_conf = (
        fundamental_frame["Fundamental_PIT_Confidence"].reindex(cols).fillna(0.0)
        if "Fundamental_PIT_Confidence" in fundamental_frame
        else pd.Series(0.0, index=cols)
    )
    if volumes is not None and not volumes.empty:
        vol_proxy = (
            volumes.reindex(index=x.index, columns=cols)
            .replace([np.inf, -np.inf], np.nan)
            .tail(63)
            .mean()
        )
        liquidity = np.log1p(vol_proxy.fillna(0.0))
    else:
        liquidity = pd.Series(0.0, index=cols)
    rows = []
    xi_up_mean = float(xi.loc[up_mask].mean()) if up_mask.sum() > 5 else np.nan
    xi_down_mean = float(xi.loc[down_mask].mean()) if down_mask.sum() > 5 else np.nan
    xi_top_mean = float(xi.loc[top_mask].mean()) if top_mask.sum() > 5 else np.nan
    xi_tail_mean = float(xi.loc[tail_mask].mean()) if tail_mask.sum() > 5 else np.nan
    for col in cols:
        y = x[col]
        try:
            b = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
            resid = pd.Series(y.to_numpy() - X @ b, index=x.index)
        except Exception:
            resid = y - y.mean()
        resid_tail = resid.tail(min(126, len(resid)))
        resid_mom = float(resid_tail.mean() * TRADING_DAYS)
        resid_ir = float(resid_tail.mean() / (resid_tail.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS))
        upside_beta = (
            float(y.loc[up_mask].mean() / (xi_up_mean + 1e-12))
            if up_mask.sum() > 5 and np.isfinite(xi_up_mean) and abs(xi_up_mean) > 1e-12
            else 0.0
        )
        downside_beta = (
            float(y.loc[down_mask].mean() / (xi_down_mean + 1e-12))
            if down_mask.sum() > 5 and np.isfinite(xi_down_mean) and abs(xi_down_mean) > 1e-12
            else 1.0
        )
        large_up_beta = (
            float(y.loc[top_mask].mean() / (xi_top_mean + 1e-12))
            if top_mask.sum() > 5 and np.isfinite(xi_top_mean) and abs(xi_top_mean) > 1e-12
            else upside_beta
        )
        tail_beta = (
            float(y.loc[tail_mask].mean() / (xi_tail_mean + 1e-12))
            if tail_mask.sum() > 5 and np.isfinite(xi_tail_mean) and abs(xi_tail_mean) > 1e-12
            else 1.0
        )
        upside_hit = float((y.loc[top_mask] > xi.loc[top_mask]).mean()) if top_mask.sum() > 5 else 0.0
        rows.append(
            {
                "Ticker": col,
                "Residual_Momentum": resid_mom,
                "Residual_IR": resid_ir,
                "Upside_Beta": upside_beta,
                "Large_Up_Beta": large_up_beta,
                "Downside_Beta": downside_beta,
                "Tail_Beta": tail_beta,
                "Upside_Hit_Rate": upside_hit,
                "Momentum_126": float(y.tail(min(126, len(y))).mean() * TRADING_DAYS),
                "Downside": downside_ann(y),
                "CVaR": cvar_loss(y),
                "MaxDD": max_dd_loss(y),
                "Liquidity_Proxy": float(liquidity.get(col, 0.0)),
                "RMT_Crowding": float(crowd.get(col, 0.0)),
                "Fundamental_Upside_Score": float(fundamental.get(col, 0.0)),
                "Fundamental_PIT_Confidence": float(fundamental_conf.get(col, 0.0)),
            }
        )
    table = pd.DataFrame(rows).set_index("Ticker")
    if table.empty:
        return [], table
    table["Convexity"] = table["Large_Up_Beta"] + 0.35 * table["Upside_Beta"] - table["Downside_Beta"]
    table["Tail_Admissible"] = (
        (table["Tail_Beta"] < 1.35)
        & (table["Downside_Beta"] < 1.30)
        & (table["CVaR"] <= table["CVaR"].quantile(0.85))
    )
    liquid_floor = table["Liquidity_Proxy"].quantile(0.10) if table["Liquidity_Proxy"].nunique() > 3 else -np.inf
    table["Liquidity_Admissible"] = table["Liquidity_Proxy"] >= liquid_floor
    table["Opportunity_Score"] = (
        0.22 * robust_z(table["Residual_Momentum"])
        + 0.18 * robust_z(table["Residual_IR"])
        + 0.26 * robust_z(table["Convexity"])
        + 0.16 * robust_z(table["Large_Up_Beta"])
        + 0.12 * robust_z(table["Upside_Hit_Rate"])
        + 0.20 * robust_z(table["Fundamental_Upside_Score"])
        + 0.08 * robust_z(table["Fundamental_PIT_Confidence"])
        + 0.08 * robust_z(table["Liquidity_Proxy"])
        - 0.20 * robust_z(table["Tail_Beta"].clip(-2.0, 4.0))
        - 0.16 * robust_z(table["CVaR"])
        - 0.12 * robust_z(table["MaxDD"])
        - 0.14 * table["RMT_Crowding"].fillna(0.0)
    )
    table["Opportunity_Score"] = table["Opportunity_Score"].where(table["Tail_Admissible"], table["Opportunity_Score"] - 2.0)
    table["Opportunity_Score"] = table["Opportunity_Score"].where(table["Liquidity_Admissible"], table["Opportunity_Score"] - 1.0)
    base_n = min(len(table), max(limit, int(np.ceil(limit * 0.55))))
    sector_fundamental_n = min(len(table), max(5, int(np.ceil(limit * 0.20))))
    low_tail_n = min(len(table), max(5, int(np.ceil(limit * 0.15))))
    selected = list(table.sort_values("Opportunity_Score", ascending=False).head(base_n).index)
    selected += list(
        table.sort_values(["Fundamental_Upside_Score", "Fundamental_PIT_Confidence"], ascending=False)
        .head(sector_fundamental_n)
        .index
    )
    selected += list(
        table[table["Tail_Admissible"]].sort_values(["Tail_Beta", "CVaR"], ascending=[True, True]).head(low_tail_n).index
    )
    selected = list(dict.fromkeys(selected))[:limit]
    table["Selected_By_ConvexOpportunityBuilder"] = table.index.isin(selected)
    return selected, table.sort_values("Opportunity_Score", ascending=False)


def market_state_optimizer(train: pd.DataFrame, xi_train: pd.Series) -> dict:
    """Causal state optimizer for growth/alpha exposure.

    The state is estimated only from the training filtration. It does not choose
    assets directly; it sets admissible exposure budgets for validation.
    """
    x = train.tail(min(126, len(train))).fillna(0.0)
    xi = xi_train.reindex(x.index).fillna(0.0)
    xi_21 = xi.tail(min(21, len(xi)))
    xi_63 = xi.tail(min(63, len(xi)))
    xi_126 = xi.tail(min(126, len(xi)))

    rolling_vol = xi_train.rolling(21).std(ddof=1).dropna() * np.sqrt(TRADING_DAYS)
    current_vol = annual_vol(xi_21)
    vol_rank = float((rolling_vol <= current_vol).mean()) if len(rolling_vol) and np.isfinite(current_vol) else 0.50
    dd_126 = max_dd_loss(xi_126)
    trend_63 = annual_return(xi_63)
    trend_126 = annual_return(xi_126)

    if x.shape[1] > 1 and x.shape[0] > 20:
        corr = x.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        corr_crowding = float(np.nanmean(np.abs(corr.values[mask])))
        dispersion = float(x.std(axis=1).median() * np.sqrt(TRADING_DAYS))
    else:
        corr_crowding = 0.0
        dispersion = 0.0

    stress = (
        0.35 * np.clip(vol_rank, 0.0, 1.0)
        + 0.30 * np.clip(dd_126 / 0.12, 0.0, 1.0)
        + 0.20 * np.clip(corr_crowding / 0.65, 0.0, 1.0)
        + 0.15 * (1.0 if trend_63 < 0.0 else 0.0)
    )
    risk_on = float((trend_63 > 0.0) and (trend_126 > 0.0) and (vol_rank < 0.70))
    recovery = float((trend_63 > 0.0) and (trend_126 <= 0.0) and (dd_126 > 0.03))
    state = "expansion" if risk_on else "recovery" if recovery else "stress" if stress > 0.70 else "fragile"

    growth_cap = float(np.clip(0.58 - 0.42 * stress + 0.18 * risk_on + 0.10 * recovery, 0.10, 0.78))
    alpha_cap = float(np.clip(0.24 - 0.16 * stress + 0.08 * risk_on + 0.05 * recovery, 0.00, 0.34))
    total_risk_cap = float(np.clip(growth_cap + alpha_cap, 0.20, 0.82))

    return {
        "state_label": state,
        "state_stress": float(stress),
        "state_risk_on": risk_on,
        "state_recovery": recovery,
        "state_vol_rank": vol_rank,
        "state_drawdown_126": float(dd_126),
        "state_trend_63": float(trend_63),
        "state_trend_126": float(trend_126),
        "state_crowding": corr_crowding,
        "state_dispersion": dispersion,
        "state_growth_cap": growth_cap,
        "state_alpha_cap": alpha_cap,
        "state_total_risk_cap": total_risk_cap,
    }


def signal_weights(
    train: pd.DataFrame,
    xi_train: pd.Series,
    max_weight: float,
    fundamental_score: pd.Series | None = None,
) -> dict[str, pd.Series]:
    up_mask = xi_train > 0
    down_mask = xi_train < 0
    X = np.column_stack([np.ones(len(xi_train)), xi_train.values])
    xi_tail_threshold = float(xi_train.quantile(0.10)) if len(xi_train.dropna()) else 0.0
    xi_tail_mask = xi_train <= xi_tail_threshold
    xi_tail_mean = float(xi_train.loc[xi_tail_mask].mean()) if xi_tail_mask.sum() > 5 else np.nan
    xi_top_threshold = float(xi_train.quantile(0.65)) if len(xi_train.dropna()) else 0.0
    xi_top_mask = xi_train >= xi_top_threshold
    xi_top_mean = float(xi_train.loc[xi_top_mask].mean()) if xi_top_mask.sum() > 5 else np.nan
    (
        resid_mom,
        resid_ir,
        upside_beta,
        downside_beta,
        tail_beta,
        tail_loss,
        large_up_beta,
        upside_hit_rate,
        upside_excess,
        upside_power,
        trend_consistency,
        recovery,
        crash,
    ) = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for col in train.columns:
        y = train[col].values
        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = pd.Series(y - X @ b, index=train.index)
        except Exception:
            resid = train[col] - train[col].mean()
        resid_tail = resid.tail(min(126, len(resid)))
        resid_mom[col] = float(resid_tail.mean() * TRADING_DAYS)
        resid_ir[col] = float(resid_tail.mean() / (resid_tail.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS))
        upside_beta[col] = (
            float(train.loc[up_mask, col].mean() / (xi_train.loc[up_mask].mean() + 1e-12))
            if up_mask.sum() > 5 and abs(float(xi_train.loc[up_mask].mean())) > 1e-12
            else 0.0
        )
        downside_beta[col] = (
            float(train.loc[down_mask, col].mean() / (xi_train.loc[down_mask].mean() + 1e-12))
            if down_mask.sum() > 5 and abs(float(xi_train.loc[down_mask].mean())) > 1e-12
            else 1.0
        )
        if xi_tail_mask.sum() > 5 and np.isfinite(xi_tail_mean) and abs(xi_tail_mean) > 1e-12:
            asset_tail_mean = float(train.loc[xi_tail_mask, col].mean())
            tail_beta[col] = asset_tail_mean / (xi_tail_mean + 1e-12)
            tail_loss[col] = max(-asset_tail_mean, 0.0)
        else:
            tail_beta[col] = 1.0
            tail_loss[col] = float(cvar_loss(train[col]))
        if xi_top_mask.sum() > 5 and np.isfinite(xi_top_mean) and abs(xi_top_mean) > 1e-12:
            asset_top = train.loc[xi_top_mask, col].replace([np.inf, -np.inf], np.nan).dropna()
            xi_top = xi_train.loc[asset_top.index]
            large_up_beta[col] = float(asset_top.mean() / (xi_top.mean() + 1e-12)) if len(asset_top) else 0.0
            upside_hit_rate[col] = float((asset_top > xi_top).mean()) if len(asset_top) else 0.0
            upside_excess[col] = float((asset_top - xi_top).mean() * TRADING_DAYS) if len(asset_top) else 0.0
            upside_power[col] = float(np.sqrt(np.mean(np.maximum(asset_top.to_numpy(), 0.0) ** 2)) * np.sqrt(TRADING_DAYS)) if len(asset_top) else 0.0
        else:
            large_up_beta[col] = float(upside_beta[col])
            upside_hit_rate[col] = 0.0
            upside_excess[col] = 0.0
            upside_power[col] = 0.0
        nav = (1.0 + train[col].fillna(0.0)).cumprod()
        dd_path = nav / nav.cummax() - 1.0
        trend_21 = train[col].fillna(0.0).rolling(21).sum().dropna()
        trend_consistency[col] = float((trend_21 > 0.0).mean()) if len(trend_21) else 0.0
        recovery[col] = float(dd_path.iloc[-1] - dd_path.min()) if len(dd_path) else 0.0
        crash[col] = float(-train[col].quantile(0.03)) if len(train[col].dropna()) > 30 else 0.0
    mom = train.tail(126).mean() * TRADING_DAYS
    mom_63 = train.tail(63).mean() * TRADING_DAYS
    mom_252 = train.tail(min(252, len(train))).mean() * TRADING_DAYS
    total_ret = train.mean() * TRADING_DAYS
    downside = train.apply(downside_ann)
    cvar = train.apply(cvar_loss)
    dd = train.apply(max_dd_loss)
    crowd = rmt_crowding_penalty(train)
    convexity = pd.Series(upside_beta) - pd.Series(downside_beta)
    tail_beta = pd.Series(tail_beta)
    tail_loss = pd.Series(tail_loss)
    large_up_beta_s = pd.Series(large_up_beta)
    upside_hit_s = pd.Series(upside_hit_rate)
    upside_excess_s = pd.Series(upside_excess)
    upside_power_s = pd.Series(upside_power)
    trend_quality = pd.Series(trend_consistency)
    recovery = pd.Series(recovery)
    crash = pd.Series(crash)
    upside_beta_s = pd.Series(upside_beta)
    downside_beta_s = pd.Series(downside_beta)
    resid_ir_s = pd.Series(resid_ir)
    resid_mom_s = pd.Series(resid_mom)
    fundamental_score = (
        pd.Series(fundamental_score, dtype=float).reindex(train.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if fundamental_score is not None
        else pd.Series(0.0, index=train.columns)
    )
    fundamental_z = robust_z(fundamental_score)
    clipped_tail_beta = tail_beta.clip(lower=-2.0, upper=4.0)
    conditional_tail_penalty = (
        0.45 * robust_z(clipped_tail_beta)
        + 0.35 * robust_z(tail_loss)
        + 0.25 * robust_z(cvar)
        + 0.20 * robust_z(crash)
        + 0.15 * robust_z(dd)
    )
    upside_edge = (upside_beta_s - 1.0).clip(lower=-2.0, upper=4.0)
    downside_drag = (downside_beta_s - 0.75).clip(lower=-2.0, upper=4.0)
    tail_convexity = (upside_beta_s - downside_beta_s - 0.35 * clipped_tail_beta).clip(lower=-4.0, upper=4.0)
    pure_upside_convexity = (upside_beta_s - downside_beta_s).clip(lower=-4.0, upper=4.0)
    real_upside_convexity = (
        0.45 * large_up_beta_s
        + 0.25 * upside_beta_s
        + 0.20 * robust_z(upside_hit_s)
        + 0.12 * robust_z(upside_excess_s)
        - 0.32 * downside_beta_s
        - 0.28 * clipped_tail_beta
    ).clip(lower=-5.0, upper=5.0)
    upside_convex_eligible = (
        (upside_beta_s > 1.0)
        & (downside_beta_s < 1.0)
        & (clipped_tail_beta < 1.10)
        & (pure_upside_convexity > 0.0)
    )
    real_upside_eligible = (
        (large_up_beta_s > 1.02)
        & (upside_beta_s > 0.95)
        & (downside_beta_s < 1.03)
        & (clipped_tail_beta < 1.18)
        & (upside_hit_s > 0.44)
        & (real_upside_convexity > 0.0)
    )
    quality_growth_proxy = (
        0.35 * robust_z(trend_quality)
        + 0.25 * robust_z(recovery)
        + 0.20 * robust_z(total_ret)
        - 0.20 * robust_z(downside)
        - 0.15 * robust_z(dd)
    )

    capital_score = -0.45 * robust_z(downside) - 0.35 * robust_z(cvar) - 0.25 * robust_z(dd) + 0.05 * robust_z(total_ret)
    benchmark_tail_capital_score = (
        -0.34 * robust_z(downside)
        -0.30 * robust_z(cvar)
        -0.24 * robust_z(dd)
        -0.26 * robust_z(clipped_tail_beta)
        -0.24 * robust_z(tail_loss)
        -0.18 * robust_z(downside_drag)
        + 0.08 * robust_z(recovery)
        + 0.05 * robust_z(total_ret)
        - 0.12 * crowd
    )
    growth_score = (
        0.30 * robust_z(mom)
        + 0.25 * robust_z(pd.Series(resid_mom))
        + 0.15 * robust_z(pd.Series(upside_beta))
        + 0.15 * robust_z(total_ret)
        - 0.20 * robust_z(cvar)
        - 0.18 * crowd
    )
    alpha_score = growth_score + 0.20 * robust_z(pd.Series(resid_mom)) - 0.10 * crowd
    growth_plus_score = (
        0.20 * robust_z(mom_63)
        + 0.18 * robust_z(mom)
        + 0.12 * robust_z(mom_252)
        + 0.18 * robust_z(pd.Series(resid_ir))
        + 0.16 * robust_z(convexity)
        + 0.12 * robust_z(trend_quality)
        + 0.10 * robust_z(recovery)
        + 0.08 * robust_z(total_ret)
        - 0.18 * robust_z(cvar)
        - 0.15 * robust_z(dd)
        - 0.12 * robust_z(crash)
        - 0.18 * crowd
    )
    alpha_plus_score = (
        growth_plus_score
        + 0.16 * robust_z(pd.Series(resid_mom))
        + 0.10 * robust_z(pd.Series(resid_ir))
        + 0.08 * robust_z(convexity)
        - 0.08 * crowd
    )
    tail_aware_growth_score = (
        0.18 * robust_z(mom_63)
        + 0.16 * robust_z(mom)
        + 0.14 * robust_z(resid_ir_s)
        + 0.20 * robust_z(convexity)
        + 0.10 * robust_z(trend_quality)
        + 0.08 * robust_z(recovery)
        + 0.06 * robust_z(total_ret)
        - 0.22 * robust_z(clipped_tail_beta)
        - 0.20 * robust_z(tail_loss)
        - 0.16 * robust_z(cvar)
        - 0.12 * robust_z(dd)
        - 0.10 * robust_z(crash)
        - 0.16 * crowd
    )
    tail_aware_alpha_score = (
        tail_aware_growth_score
        + 0.14 * robust_z(resid_mom_s)
        + 0.10 * robust_z(resid_ir_s)
        + 0.08 * robust_z(convexity)
        - 0.10 * robust_z(clipped_tail_beta)
        - 0.06 * crowd
    )
    tail_convex_growth_score = (
        0.18 * robust_z(mom_63)
        + 0.15 * robust_z(mom)
        + 0.13 * robust_z(resid_ir_s)
        + 0.13 * robust_z(resid_mom_s)
        + 0.24 * robust_z(tail_convexity)
        + 0.16 * robust_z(upside_edge)
        + 0.16 * quality_growth_proxy
        - 0.18 * robust_z(downside_drag)
        - 0.42 * conditional_tail_penalty
        - 0.22 * crowd
    )
    tail_convex_alpha_score = (
        tail_convex_growth_score
        + 0.16 * robust_z(resid_ir_s)
        + 0.12 * robust_z(resid_mom_s)
        + 0.10 * robust_z(tail_convexity)
        - 0.10 * robust_z(clipped_tail_beta)
        - 0.08 * crowd
    )
    upside_convex_growth_score = (
        0.25 * robust_z(mom_63)
        + 0.18 * robust_z(mom)
        + 0.20 * robust_z(resid_ir_s)
        + 0.18 * robust_z(resid_mom_s)
        + 0.24 * robust_z(pure_upside_convexity)
        + 0.18 * robust_z(upside_edge)
        + 0.15 * quality_growth_proxy
        + 0.10 * robust_z(recovery)
        - 0.22 * robust_z(clipped_tail_beta)
        - 0.18 * robust_z(cvar)
        - 0.14 * robust_z(downside_drag)
        - 0.10 * crowd
    )
    upside_convex_growth_score = upside_convex_growth_score.where(
        upside_convex_eligible,
        float(upside_convex_growth_score.min()) - 3.0,
    )
    upside_convex_alpha_score = (
        upside_convex_growth_score
        + 0.20 * robust_z(resid_ir_s)
        + 0.16 * robust_z(resid_mom_s)
        + 0.12 * robust_z(pure_upside_convexity)
        - 0.12 * robust_z(clipped_tail_beta)
        - 0.08 * crowd
    ).where(upside_convex_eligible, float(upside_convex_growth_score.min()) - 3.0)
    fundamental_upside_growth_score = (
        0.34 * upside_convex_growth_score
        + 0.30 * fundamental_z
        + 0.14 * robust_z(pure_upside_convexity)
        + 0.12 * robust_z(resid_ir_s)
        + 0.08 * robust_z(mom)
        - 0.12 * robust_z(clipped_tail_beta)
        - 0.10 * crowd
    )
    fundamental_upside_alpha_score = (
        0.40 * fundamental_upside_growth_score
        + 0.22 * fundamental_z
        + 0.18 * robust_z(resid_mom_s)
        + 0.14 * robust_z(resid_ir_s)
        + 0.12 * robust_z(pure_upside_convexity)
        - 0.10 * robust_z(clipped_tail_beta)
        - 0.08 * crowd
    )
    real_upside_growth_score = (
        0.22 * robust_z(mom_63)
        + 0.16 * robust_z(mom)
        + 0.18 * robust_z(resid_ir_s)
        + 0.14 * robust_z(resid_mom_s)
        + 0.30 * robust_z(real_upside_convexity)
        + 0.18 * robust_z(large_up_beta_s)
        + 0.14 * robust_z(upside_hit_s)
        + 0.10 * robust_z(upside_power_s)
        + 0.12 * quality_growth_proxy
        - 0.20 * robust_z(clipped_tail_beta)
        - 0.16 * robust_z(cvar)
        - 0.14 * robust_z(downside_drag)
        - 0.12 * crowd
    ).where(real_upside_eligible, float(upside_convex_growth_score.min()) - 3.0)
    real_upside_alpha_score = (
        real_upside_growth_score
        + 0.18 * robust_z(resid_ir_s)
        + 0.14 * robust_z(resid_mom_s)
        + 0.12 * robust_z(real_upside_convexity)
        + 0.08 * robust_z(upside_excess_s)
        - 0.12 * robust_z(clipped_tail_beta)
        - 0.08 * crowd
    ).where(real_upside_eligible, float(upside_convex_growth_score.min()) - 3.0)
    fundamental_real_upside_growth_score = (
        0.38 * real_upside_growth_score
        + 0.26 * fundamental_z
        + 0.18 * robust_z(real_upside_convexity)
        + 0.12 * robust_z(upside_excess_s)
        + 0.10 * robust_z(resid_ir_s)
        - 0.12 * robust_z(clipped_tail_beta)
        - 0.08 * crowd
    ).where(real_upside_eligible, float(upside_convex_growth_score.min()) - 3.0)
    fundamental_real_upside_alpha_score = (
        0.44 * fundamental_real_upside_growth_score
        + 0.20 * fundamental_z
        + 0.16 * robust_z(resid_ir_s)
        + 0.14 * robust_z(real_upside_convexity)
        + 0.10 * robust_z(upside_excess_s)
        - 0.10 * robust_z(clipped_tail_beta)
        - 0.08 * crowd
    ).where(real_upside_eligible, float(upside_convex_growth_score.min()) - 3.0)
    return {
        "capital": project_weights((capital_score - capital_score.min() + 1e-6).clip(lower=1e-6), max_weight),
        "capital_xi_tail": project_weights(
            (benchmark_tail_capital_score - benchmark_tail_capital_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "growth": project_weights((growth_score - growth_score.min() + 1e-6).clip(lower=1e-6), max_weight),
        "alpha": project_weights((alpha_score - alpha_score.min() + 1e-6).clip(lower=1e-6), max_weight),
        "growth_plus": project_weights((growth_plus_score - growth_plus_score.min() + 1e-6).clip(lower=1e-6), max_weight),
        "alpha_plus": project_weights((alpha_plus_score - alpha_plus_score.min() + 1e-6).clip(lower=1e-6), max_weight),
        "growth_tail_aware": project_weights(
            (tail_aware_growth_score - tail_aware_growth_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "alpha_tail_aware": project_weights(
            (tail_aware_alpha_score - tail_aware_alpha_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "growth_tail_convex": project_weights(
            (tail_convex_growth_score - tail_convex_growth_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "alpha_tail_convex": project_weights(
            (tail_convex_alpha_score - tail_convex_alpha_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "growth_upside_convex": project_weights(
            (upside_convex_growth_score - upside_convex_growth_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "alpha_upside_convex": project_weights(
            (upside_convex_alpha_score - upside_convex_alpha_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "growth_fundamental_upside_convex": project_weights(
            (fundamental_upside_growth_score - fundamental_upside_growth_score.min() + 1e-6).clip(lower=1e-6),
            max_weight,
        ),
        "alpha_fundamental_upside_convex": project_weights(
            (fundamental_upside_alpha_score - fundamental_upside_alpha_score.min() + 1e-6).clip(lower=1e-6),
            max_weight,
        ),
        "growth_real_upside_convex": project_weights(
            (real_upside_growth_score - real_upside_growth_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "alpha_real_upside_convex": project_weights(
            (real_upside_alpha_score - real_upside_alpha_score.min() + 1e-6).clip(lower=1e-6), max_weight
        ),
        "growth_fundamental_real_upside_convex": project_weights(
            (fundamental_real_upside_growth_score - fundamental_real_upside_growth_score.min() + 1e-6).clip(lower=1e-6),
            max_weight,
        ),
        "alpha_fundamental_real_upside_convex": project_weights(
            (fundamental_real_upside_alpha_score - fundamental_real_upside_alpha_score.min() + 1e-6).clip(lower=1e-6),
            max_weight,
        ),
    }


def reference_risk_anchor_weights(train_ref: pd.DataFrame, xi_train: pd.Series) -> pd.Series:
    """Build a causal defensive anchor from public reference assets.

    The anchor is estimated only on the training window and is intended to add
    non-equity convexity/ballast without letting the optimizer select a hedge
    using validation or test outcomes.
    """
    refs = train_ref.dropna(axis=1, how="all").fillna(0.0)
    if refs.empty:
        return pd.Series(dtype=float)
    xi = xi_train.reindex(refs.index).fillna(0.0)
    xi_var = float(np.var(xi, ddof=1)) + 1e-12
    beta = refs.apply(lambda s: float(np.cov(s, xi)[0, 1] / xi_var))
    corr = refs.apply(lambda s: float(pd.Series(s).corr(xi)) if pd.Series(s).std(ddof=1) > 1e-12 else 0.0)
    ret = refs.mean() * TRADING_DAYS
    down = refs.apply(downside_ann)
    cvar = refs.apply(cvar_loss)
    dd = refs.apply(max_dd_loss)
    score = (
        0.20 * robust_z(ret)
        - 0.30 * robust_z(down)
        - 0.25 * robust_z(cvar)
        - 0.20 * robust_z(dd)
        - 0.20 * robust_z(beta)
        - 0.10 * robust_z(corr.abs())
    )
    raw = (score - score.min() + 1e-6).clip(lower=1e-6)
    cap = max(0.20, min(0.45, 1.25 / max(len(raw), 1)))
    return project_weights(raw, max_weight=cap)


def add_reference_anchor_to_books(
    books: dict[str, pd.Series],
    reference_anchor: pd.Series,
    *,
    hedge_floor: float,
    max_weight: float,
) -> dict[str, pd.Series]:
    """Blend the capital-preservation book with a fixed training-only anchor."""
    if reference_anchor.empty or hedge_floor <= 1e-12:
        return books
    idx = books["capital"].index.union(reference_anchor.index)
    out = {k: v.reindex(idx).fillna(0.0) for k, v in books.items()}
    anchor = reference_anchor.reindex(idx).fillna(0.0)
    if anchor.sum() <= 1e-12:
        return out
    anchor = anchor / anchor.sum()
    capital = (1.0 - hedge_floor) * out["capital"] + hedge_floor * anchor
    out["capital"] = project_weights(capital, max_weight=max_weight)
    out["reference_anchor"] = anchor[anchor > 1e-12]
    return out


def defensive_overlay_anchor_returns(frame: pd.DataFrame, books: dict[str, pd.Series]) -> tuple[pd.Series, str]:
    anchor = books.get("reference_anchor", pd.Series(dtype=float))
    if not anchor.empty:
        w = anchor.reindex(frame.columns).fillna(0.0)
        if w.sum() > 1e-12:
            return frame @ (w / w.sum()), "reference_anchor"
    return frame @ books["capital"], "capital_book"


def _project_beta_gamma(beta: float, gamma: float, beta_cap: float, gamma_cap: float, total_cap: float) -> tuple[float, float]:
    b = float(np.clip(beta, 0.0, max(beta_cap, 0.0)))
    g = float(np.clip(gamma, 0.0, max(gamma_cap, 0.0)))
    if b + g > total_cap and b + g > 1e-12:
        scale = total_cap / (b + g)
        b *= scale
        g *= scale
    return b, g


def pso_beta_gamma_search(
    score_fn,
    *,
    beta_cap: float,
    gamma_cap: float,
    total_cap: float,
    particles: int = 18,
    iterations: int = 18,
    seed: int = SEED,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Small deterministic PSO over sleeve masses only.

    This is intentionally low-dimensional: the swarm cannot choose tickers,
    assets, dates, benchmarks, or test data. It only proposes validation-time
    sleeve throttles under causal caps.
    """
    rng = np.random.default_rng(seed)
    n = int(max(4, particles))
    iters = int(max(3, iterations))
    beta_cap = float(max(0.0, beta_cap))
    gamma_cap = float(max(0.0, gamma_cap))
    total_cap = float(max(0.0, min(total_cap, beta_cap + gamma_cap, 0.85)))

    pos = np.column_stack([rng.uniform(0.0, beta_cap, n), rng.uniform(0.0, gamma_cap, n)])
    for i in range(n):
        pos[i] = _project_beta_gamma(pos[i, 0], pos[i, 1], beta_cap, gamma_cap, total_cap)
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_score = np.array([float(score_fn(float(b), float(g))) for b, g in pbest])
    if not np.isfinite(pbest_score).any():
        return [(0.0, 0.0)], {"pso_best_score": np.nan, "pso_particles": n, "pso_iterations": iters}
    gbest = pbest[int(np.nanargmax(pbest_score))].copy()
    gbest_score = float(np.nanmax(pbest_score))

    for _ in range(iters):
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = 0.55 * vel + 1.35 * r1 * (pbest - pos) + 1.35 * r2 * (gbest - pos)
        pos = pos + vel
        for i in range(n):
            pos[i] = _project_beta_gamma(pos[i, 0], pos[i, 1], beta_cap, gamma_cap, total_cap)
        scores = np.array([float(score_fn(float(b), float(g))) for b, g in pos])
        improve = scores > pbest_score
        pbest[improve] = pos[improve]
        pbest_score[improve] = scores[improve]
        if np.isfinite(scores).any() and float(np.nanmax(scores)) > gbest_score:
            gbest = pos[int(np.nanargmax(scores))].copy()
            gbest_score = float(np.nanmax(scores))

    candidates = {(round(float(gbest[0]), 6), round(float(gbest[1]), 6)), (0.0, 0.0)}
    for i in np.argsort(pbest_score)[-min(6, len(pbest_score)) :]:
        candidates.add((round(float(pbest[i, 0]), 6), round(float(pbest[i, 1]), 6)))
    diagnostics = {"pso_best_score": gbest_score, "pso_particles": n, "pso_iterations": iters}
    return sorted(candidates), diagnostics


def validation_tail_breach(
    returns: pd.Series,
    xi_returns: pd.Series,
    omega_returns: pd.DataFrame,
    *,
    weights: pd.Series | None = None,
    tolerance: float = 0.00,
) -> dict[str, float | bool]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    xi = pd.Series(xi_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    idx = r.index.intersection(xi.index)
    r = r.reindex(idx)
    xi = xi.reindex(idx)
    if len(idx) < 20:
        return {"tail_breach": np.inf, "tail_pass": False}
    xi_cvar = cvar_loss(xi)
    xi_dd = max_dd_loss(xi)
    xi_down = downside_ann(xi)
    diag = upside_downside_diagnostics(r, xi)
    xodr = xodr_v1_omega_dominance_score(r, xi, omega_returns.reindex(idx), weights=weights)
    dc = float(diag.get("Downside_Capture", np.nan))
    uc = float(diag.get("Upside_Capture", np.nan))
    port_cvar = cvar_loss(r)
    port_dd = max_dd_loss(r)
    port_down = downside_ann(r)
    omega_down = float(xodr.get("XODR_v1_Omega_Downside_Frontier", np.nan))
    omega_cvar = float(xodr.get("XODR_v1_Omega_CVaR_Frontier", np.nan))
    omega_dd = float(xodr.get("XODR_v1_Omega_MaxDD_Frontier", np.nan))
    breach = (
        max(port_cvar / (xi_cvar + 1e-12) - (1.0 + tolerance), 0.0)
        + max(port_dd / (xi_dd + 1e-12) - (1.0 + tolerance), 0.0)
        + max(port_down / (xi_down + 1e-12) - (1.0 + tolerance), 0.0)
        + max(dc - 1.0, 0.0)
    )
    if np.isfinite(omega_down):
        breach += max(port_down / (omega_down + 1e-12) - (1.0 + tolerance), 0.0)
    if np.isfinite(omega_cvar):
        breach += max(port_cvar / (omega_cvar + 1e-12) - (1.0 + tolerance), 0.0)
    if np.isfinite(omega_dd):
        breach += max(port_dd / (omega_dd + 1e-12) - (1.0 + tolerance), 0.0)
    return {
        "tail_breach": float(breach),
        "tail_pass": bool(breach <= 1e-10 and np.isfinite(uc) and uc > 1.0 and np.isfinite(dc) and dc < 1.0),
        "tail_uc": uc,
        "tail_dc": dc,
        "tail_cvar_ratio_xi": float(port_cvar / (xi_cvar + 1e-12)),
        "tail_dd_ratio_xi": float(port_dd / (xi_dd + 1e-12)),
        "tail_downside_ratio_xi": float(port_down / (xi_down + 1e-12)),
        "tail_xodr_pass": bool(xodr.get("XODR_v1_Pass", False)),
    }


def tail_throttle_beta_gamma(
    beta: float,
    gamma: float,
    val: pd.DataFrame,
    xi_val: pd.Series,
    omega_val: pd.DataFrame,
    books: dict[str, pd.Series],
    cfg: BatchConfig,
    *,
    steps: int = 12,
) -> tuple[float, float, dict[str, float | bool]]:
    best = None
    for scale in np.linspace(1.0, 0.0, int(max(3, steps))):
        b = float(beta) * float(scale)
        g = float(gamma) * float(scale)
        if b + g > 0.85:
            continue
        a = 1.0 - b - g
        w = project_weights(a * books["capital"] + b * books["growth"] + g * books["alpha"], cfg.max_weight)
        r = val @ w
        tail = validation_tail_breach(r, xi_val, omega_val, weights=w, tolerance=0.0)
        xodr = xodr_v1_omega_dominance_score(r, xi_val, omega_val, weights=w)
        active = annual_return(r - xi_val)
        key = (float(tail["tail_breach"]), -active, -float(xodr.get("XODR_v1", -1e9)), -scale)
        row = {
            **tail,
            "tail_scale": float(scale),
            "tail_active_ann_return": float(active),
            "tail_xodr_v1": float(xodr.get("XODR_v1", np.nan)),
        }
        if best is None or key < best[0]:
            best = (key, b, g, row)
    if best is None:
        return 0.0, 0.0, {"tail_breach": np.inf, "tail_pass": False, "tail_scale": 0.0}
    return float(best[1]), float(best[2]), best[3]


def optimize_drawdown_budget_overlay(
    raw_returns: pd.Series,
    anchor_returns: pd.Series,
    xi_returns: pd.Series,
    state: dict | None = None,
    *,
    tolerance: float = 0.02,
) -> tuple[dict[str, float], dict[str, float | bool]]:
    """Select causal overlay parameters on validation only.

    The objective is lexicographic: first minimize benchmark-relative downside
    breaches, then preserve active return and upside capture. The returned
    parameters are frozen and later applied to test/OOS.
    """
    raw = pd.Series(raw_returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    anchor = pd.Series(anchor_returns, dtype=float).reindex(raw.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xi = pd.Series(xi_returns, dtype=float).reindex(raw.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xi_down = downside_ann(xi)
    xi_cvar = cvar_loss(xi)
    xi_dd = max_dd_loss(xi)
    grid = []
    # Preregister a deliberately small grid. A wide continuous search here
    # would add degrees of freedom exactly where the strategy is most likely
    # to overfit: downside preservation after observing validation tails.
    for min_exp in [0.15, 0.35]:
        for dd_shift in [-0.020, 0.000]:
            for dd_gap in [0.015, 0.040]:
                for vol_shift in [-0.040, 0.000]:
                    for rerisk_step in [0.06, 0.14]:
                        grid.append(
                            {
                                "budget_min_exposure": float(min_exp),
                                "budget_rerisk_step": float(rerisk_step),
                                "budget_dd_soft_shift": float(dd_shift),
                                "budget_dd_hard_gap": float(dd_gap),
                                "budget_vol_target_shift": float(vol_shift),
                            }
                        )
    best = None
    for params in grid:
        r, diag = causal_drawdown_vol_overlay(
            raw,
            anchor,
            state,
            benchmark_returns=xi,
            min_exposure=params["budget_min_exposure"],
            rerisk_step=params["budget_rerisk_step"],
            dd_soft_shift=params["budget_dd_soft_shift"],
            dd_hard_gap=params["budget_dd_hard_gap"],
            vol_target_shift=params["budget_vol_target_shift"],
        )
        ud = upside_downside_diagnostics(r, xi, baseline_returns=anchor, tolerance=0.0)
        down = downside_ann(r)
        cv = cvar_loss(r)
        dd = max_dd_loss(r)
        path_dd = benchmark_relative_drawdown_diagnostics(r, xi, abs_buffer=0.008, rel_buffer=0.15)
        active = annual_return(r - xi)
        uc = float(ud.get("Upside_Capture", np.nan))
        dc = float(ud.get("Downside_Capture", np.nan))
        down_breach = max(down / (xi_down + 1e-12) - (1.0 + tolerance), 0.0)
        cvar_breach = max(cv / (xi_cvar + 1e-12) - (1.0 + tolerance), 0.0)
        dd_breach = max(dd / (xi_dd + 1e-12) - (1.0 + tolerance), 0.0)
        dc_breach = max(dc - 1.0, 0.0) if np.isfinite(dc) else 1.0
        active_breach = max(-active, 0.0)
        path_breach = 8.0 * max(path_dd["path_dd_max_excess"], 0.0) + 20.0 * max(
            path_dd["path_dd_breach_area"], 0.0
        )
        breach = (
            2.0 * dd_breach
            + 1.5 * cvar_breach
            + 1.2 * down_breach
            + dc_breach
            + active_breach
            + path_breach
        )
        pass_flag = bool(
            active > 0.0
            and breach <= 1e-10
            and path_dd["path_dd_max_excess"] <= 1e-12
            and np.isfinite(uc)
            and uc > 1.0
            and np.isfinite(dc)
            and dc < 1.0
        )
        key = (not pass_flag, breach, -active, -float(uc if np.isfinite(uc) else -1e9), float(diag["overlay_avg_exposure"]))
        row = {
            **params,
            **{f"budget_{k}": v for k, v in diag.items()},
            "budget_breach": float(breach),
            "budget_downside_ratio_xi": float(down / (xi_down + 1e-12)),
            "budget_cvar_ratio_xi": float(cv / (xi_cvar + 1e-12)),
            "budget_maxdd_ratio_xi": float(dd / (xi_dd + 1e-12)),
            "budget_path_dd_max_excess": float(path_dd["path_dd_max_excess"]),
            "budget_path_dd_breach_area": float(path_dd["path_dd_breach_area"]),
            "budget_path_dd_breach_rate": float(path_dd["path_dd_breach_rate"]),
            "budget_path_dd_ratio_xi": float(path_dd["path_dd_ratio_xi"]),
            "budget_active_ann_return": float(active),
            "budget_upside_capture": float(uc) if np.isfinite(uc) else np.nan,
            "budget_downside_capture": float(dc) if np.isfinite(dc) else np.nan,
            "budget_pass": pass_flag,
        }
        if best is None or key < best[0]:
            best = (key, params, row)
    if best is None:
        params = {
            "budget_min_exposure": 0.25,
            "budget_rerisk_step": 0.10,
            "budget_dd_soft_shift": 0.0,
            "budget_dd_hard_gap": 0.04,
            "budget_vol_target_shift": 0.0,
        }
        return params, {"budget_breach": np.inf, "budget_pass": False, **params}
    return dict(best[1]), dict(best[2])


def optimize_upside_recovery_overlay(
    raw_returns: pd.Series,
    anchor_returns: pd.Series,
    xi_returns: pd.Series,
    state: dict | None = None,
    *,
    tolerance: float = 0.04,
) -> tuple[dict[str, float], dict[str, float | bool]]:
    """Validation-only overlay search that tries to recover upside participation.

    This is not allowed to use test returns. It searches a pre-registered grid
    with higher re-risking speed and exposure floors, but candidates remain
    infeasible when validation downside, CVaR, max drawdown, downside capture,
    or benchmark-relative drawdown path breaches exceed the tolerance budget.
    """
    raw = pd.Series(raw_returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    anchor = pd.Series(anchor_returns, dtype=float).reindex(raw.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xi = pd.Series(xi_returns, dtype=float).reindex(raw.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xi_down = downside_ann(xi)
    xi_cvar = cvar_loss(xi)
    xi_dd = max_dd_loss(xi)
    grid = []
    for min_exp in [0.35, 0.50, 0.65]:
        for dd_shift in [-0.010, 0.000, 0.015]:
            for dd_gap in [0.025, 0.050, 0.075]:
                for vol_shift in [0.000, 0.030, 0.060]:
                    for rerisk_step in [0.14, 0.24, 0.34]:
                        grid.append(
                            {
                                "budget_min_exposure": float(min_exp),
                                "budget_rerisk_step": float(rerisk_step),
                                "budget_dd_soft_shift": float(dd_shift),
                                "budget_dd_hard_gap": float(dd_gap),
                                "budget_vol_target_shift": float(vol_shift),
                            }
                        )
    best = None
    for params in grid:
        r, diag = causal_drawdown_vol_overlay(
            raw,
            anchor,
            state,
            benchmark_returns=xi,
            min_exposure=params["budget_min_exposure"],
            rerisk_step=params["budget_rerisk_step"],
            dd_soft_shift=params["budget_dd_soft_shift"],
            dd_hard_gap=params["budget_dd_hard_gap"],
            vol_target_shift=params["budget_vol_target_shift"],
        )
        ud = upside_downside_diagnostics(r, xi, baseline_returns=anchor, tolerance=0.0)
        down = downside_ann(r)
        cv = cvar_loss(r)
        dd = max_dd_loss(r)
        path_dd = benchmark_relative_drawdown_diagnostics(r, xi, abs_buffer=0.008, rel_buffer=0.15)
        active = annual_return(r - xi)
        uc = float(ud.get("Upside_Capture", np.nan))
        dc = float(ud.get("Downside_Capture", np.nan))
        down_ratio = down / (xi_down + 1e-12)
        cvar_ratio = cv / (xi_cvar + 1e-12)
        dd_ratio = dd / (xi_dd + 1e-12)
        down_breach = max(down_ratio - (1.0 + tolerance), 0.0)
        cvar_breach = max(cvar_ratio - (1.0 + tolerance), 0.0)
        dd_breach = max(dd_ratio - (1.0 + tolerance), 0.0)
        dc_breach = max(dc - 0.98, 0.0) if np.isfinite(dc) else 1.0
        active_breach = max(-active, 0.0)
        path_breach = 8.0 * max(path_dd["path_dd_max_excess"], 0.0) + 20.0 * max(
            path_dd["path_dd_breach_area"], 0.0
        )
        breach = (
            2.0 * dd_breach
            + 1.7 * cvar_breach
            + 1.3 * down_breach
            + 1.2 * dc_breach
            + active_breach
            + path_breach
        )
        pass_flag = bool(
            active > 0.0
            and breach <= 1e-10
            and path_dd["path_dd_max_excess"] <= 0.010
            and np.isfinite(dc)
            and dc < 0.98
            and down_ratio <= 1.0 + tolerance
            and cvar_ratio <= 1.0 + tolerance
            and dd_ratio <= 1.0 + tolerance
        )
        key = (
            not pass_flag,
            breach,
            -active,
            -float(uc if np.isfinite(uc) else -1e9),
            -float(diag["overlay_avg_exposure"]),
        )
        row = {
            **params,
            **{f"budget_{k}": v for k, v in diag.items()},
            "budget_breach": float(breach),
            "budget_downside_ratio_xi": float(down_ratio),
            "budget_cvar_ratio_xi": float(cvar_ratio),
            "budget_maxdd_ratio_xi": float(dd_ratio),
            "budget_path_dd_max_excess": float(path_dd["path_dd_max_excess"]),
            "budget_path_dd_breach_area": float(path_dd["path_dd_breach_area"]),
            "budget_path_dd_breach_rate": float(path_dd["path_dd_breach_rate"]),
            "budget_path_dd_ratio_xi": float(path_dd["path_dd_ratio_xi"]),
            "budget_active_ann_return": float(active),
            "budget_upside_capture": float(uc) if np.isfinite(uc) else np.nan,
            "budget_downside_capture": float(dc) if np.isfinite(dc) else np.nan,
            "budget_pass": pass_flag,
            "budget_recovery_mode": True,
        }
        if best is None or key < best[0]:
            best = (key, params, row)
    if best is None:
        return optimize_drawdown_budget_overlay(raw, anchor, xi, state, tolerance=0.02)
    return dict(best[1]), dict(best[2])


def eval_candidate_on_validation(
    val: pd.DataFrame,
    xi_val: pd.Series,
    omega_val: pd.DataFrame,
    books: dict[str, pd.Series],
    variant: str,
    cfg: BatchConfig,
    state: dict | None = None,
) -> tuple[pd.Series, dict]:
    baseline = val @ books["capital"]
    defensive_anchor, defensive_anchor_kind = defensive_overlay_anchor_returns(val, books)
    beta_grid = [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]
    gamma_grid = [0.0, 0.10, 0.20, 0.30]
    if variant == "xcdr_v3_growth_control_policy":
        beta_grid = [0.10, 0.20, 0.35, 0.50, 0.65, 0.75]
        gamma_grid = [0.05, 0.15, 0.25, 0.35]
    if variant == "state_optimized_xcdr_v3_policy":
        state = state or {}
        growth_cap = float(state.get("state_growth_cap", 0.45))
        alpha_cap = float(state.get("state_alpha_cap", 0.20))
        total_cap = float(state.get("state_total_risk_cap", 0.65))
        beta_grid = [b for b in [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.75] if b <= growth_cap + 1e-12]
        gamma_grid = [g for g in [0.00, 0.05, 0.10, 0.15, 0.25, 0.35] if g <= alpha_cap + 1e-12]
        if not beta_grid:
            beta_grid = [0.0]
        if not gamma_grid:
            gamma_grid = [0.0]
    if variant == "state_xodr_v1_policy":
        state = state or {}
        growth_cap = min(float(state.get("state_growth_cap", 0.45)), 0.50)
        alpha_cap = min(float(state.get("state_alpha_cap", 0.20)), 0.20)
        total_cap = min(float(state.get("state_total_risk_cap", 0.65)), 0.60)
        beta_grid = [b for b in [0.00, 0.05, 0.10, 0.20, 0.35, 0.50] if b <= growth_cap + 1e-12]
        gamma_grid = [g for g in [0.00, 0.05, 0.10, 0.15, 0.20] if g <= alpha_cap + 1e-12]
    if variant == "pso_state_xodr_v1_policy":
        state = state or {}
        growth_cap = min(float(state.get("state_growth_cap", 0.45)), 0.55)
        alpha_cap = min(float(state.get("state_alpha_cap", 0.20)), 0.22)
        total_cap = min(float(state.get("state_total_risk_cap", 0.65)), 0.62)
        beta_grid = [0.0, min(0.10, growth_cap), min(0.25, growth_cap), min(0.45, growth_cap)]
        gamma_grid = [0.0, min(0.05, alpha_cap), min(0.12, alpha_cap)]
    if variant in {
        "balanced_anchor_xodr_v1_policy",
        "enhanced_growth_anchor_xodr_v1_policy",
        "enhanced_growth_anchor_dd_control_policy",
        "enhanced_growth_anchor_crash_budget_policy",
        "enhanced_growth_anchor_dd_budget_policy",
        "tail_aware_anchor_dd_budget_policy",
        "tail_convex_anchor_dd_budget_policy",
        "upside_convex_anchor_dd_budget_policy",
        "fundamental_upside_convex_anchor_dd_budget_policy",
        "fundamental_real_upside_anchor_dd_budget_policy",
        "fundamental_upside_recovery_anchor_dd_budget_policy",
        "enhanced_growth_xodr_v1_policy",
    }:
        state = state or {}
        growth_cap = min(float(state.get("state_growth_cap", 0.45)), 0.58)
        alpha_cap = min(float(state.get("state_alpha_cap", 0.20)), 0.24)
        total_cap = min(float(state.get("state_total_risk_cap", 0.65)), 0.66)
        beta_grid = [0.0, min(0.10, growth_cap), min(0.25, growth_cap), min(0.45, growth_cap)]
        gamma_grid = [0.0, min(0.05, alpha_cap), min(0.12, alpha_cap), min(0.18, alpha_cap)]
    if variant in {"tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy"}:
        state = state or {}
        growth_cap = min(float(state.get("state_growth_cap", 0.45)), 0.55)
        alpha_cap = min(float(state.get("state_alpha_cap", 0.20)), 0.22)
        total_cap = min(float(state.get("state_total_risk_cap", 0.65)), 0.62)
        beta_grid = [0.0, min(0.10, growth_cap), min(0.25, growth_cap), min(0.45, growth_cap)]
        gamma_grid = [0.0, min(0.05, alpha_cap), min(0.12, alpha_cap)]
    candidate_pairs = [(float(b), float(g)) for b in beta_grid for g in gamma_grid]
    pso_diag: dict[str, float] = {}
    if variant in {"pso_state_xodr_v1_policy", "tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy", "balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy", "enhanced_growth_xodr_v1_policy"}:
        def pso_score(beta: float, gamma: float) -> float:
            if beta + gamma > total_cap + 1e-12 or beta + gamma > 0.85:
                return -1e9
            alpha = 1.0 - beta - gamma
            w_local = project_weights(alpha * books["capital"] + beta * books["growth"] + gamma * books["alpha"], cfg.max_weight)
            rr = val @ w_local
            diag_local = upside_downside_diagnostics(rr, xi_val, baseline_returns=baseline, tolerance=0.03)
            xodr_local = xodr_v1_omega_dominance_score(rr, xi_val, omega_val, weights=w_local)
            dc_local = float(diag_local.get("Downside_Capture", np.nan))
            xi_cvar_local = cvar_loss(xi_val)
            xi_dd_local = max_dd_loss(xi_val)
            xi_down_local = downside_ann(xi_val)
            breach = (
                max(cvar_loss(rr) / (xi_cvar_local + 1e-12) - 1.01, 0.0)
                + max(max_dd_loss(rr) / (xi_dd_local + 1e-12) - 1.01, 0.0)
                + max(downside_ann(rr) / (xi_down_local + 1e-12) - 1.01, 0.0)
                + max(dc_local - 0.95, 0.0)
            )
            return float(xodr_local.get("XODR_v1", -1e6)) + 0.20 * float(xcdr_v3_growth_control_score(rr, xi_val, weights=w_local).get("XCDR_v3_GrowthControl", 0.0)) - 5.0 * breach

        pso_seed = SEED + int(pd.Timestamp(val.index[0]).strftime("%Y%m%d"))
        pso_pairs, pso_diag = pso_beta_gamma_search(
            pso_score,
            beta_cap=growth_cap,
            gamma_cap=alpha_cap,
            total_cap=total_cap,
            particles=cfg.pso_particles,
            iterations=cfg.pso_iterations,
            seed=pso_seed,
        )
        if variant in {"tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy"}:
            candidate_pairs = sorted(set(pso_pairs + [(0.0, 0.0)]))
        else:
            candidate_pairs = sorted(set(candidate_pairs + pso_pairs))
    best = None
    for beta, gamma in candidate_pairs:
            if beta + gamma > 0.85:
                continue
            if variant in {"state_optimized_xcdr_v3_policy", "state_xodr_v1_policy", "pso_state_xodr_v1_policy", "tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy", "balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy", "enhanced_growth_xodr_v1_policy"} and beta + gamma > total_cap + 1e-12:
                continue
            tail_diag = {}
            if variant in {"tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy"}:
                beta, gamma, tail_diag = tail_throttle_beta_gamma(
                    beta, gamma, val, xi_val, omega_val, books, cfg, steps=8
                )
            alpha = 1.0 - beta - gamma
            w = project_weights(alpha * books["capital"] + beta * books["growth"] + gamma * books["alpha"], cfg.max_weight)
            raw_r = val @ w
            overlay_diag = {}
            if variant in {"enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy"}:
                if variant == "fundamental_upside_recovery_anchor_dd_budget_policy":
                    budget_params, budget_diag = optimize_upside_recovery_overlay(raw_r, defensive_anchor, xi_val, state)
                else:
                    budget_params, budget_diag = optimize_drawdown_budget_overlay(raw_r, defensive_anchor, xi_val, state)
                r, overlay_diag = causal_drawdown_vol_overlay(
                    raw_r,
                    defensive_anchor,
                    state,
                    benchmark_returns=xi_val,
                min_exposure=budget_params["budget_min_exposure"],
                rerisk_step=budget_params["budget_rerisk_step"],
                    dd_soft_shift=budget_params["budget_dd_soft_shift"],
                    dd_hard_gap=budget_params["budget_dd_hard_gap"],
                    vol_target_shift=budget_params["budget_vol_target_shift"],
                )
                overlay_diag.update(budget_diag)
                overlay_diag["overlay_anchor_kind"] = defensive_anchor_kind
            elif variant in {"enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy"}:
                r, overlay_diag = causal_drawdown_vol_overlay(raw_r, baseline, state)
            else:
                r = raw_r
            diag = upside_downside_diagnostics(r, xi_val, baseline_returns=baseline, tolerance=0.05)
            score_v3 = xcdr_v3_growth_control_score(r, xi_val, weights=w)
            score_xodr = xodr_v1_omega_dominance_score(r, xi_val, omega_val, weights=w)
            feasible = bool(diag.get("Downside_Preservation_Pass", False))
            uc = float(diag.get("Upside_Capture", np.nan))
            dc = float(diag.get("Downside_Capture", np.nan))
            capture = bool(np.isfinite(uc) and np.isfinite(dc) and uc > 1.0 and dc < 1.0)
            if beta + gamma > 1e-12 and (not np.isfinite(dc) or dc >= 1.0):
                feasible = False
            xi_cvar = cvar_loss(xi_val)
            xi_dd = max_dd_loss(xi_val)
            xi_downside = downside_ann(xi_val)
            row = {
                "variant": variant,
                "alpha_mass": alpha,
                "growth_mass": beta,
                "alpha_signal_mass": gamma,
                "feasible": feasible,
                "capture_pass": capture,
                "return_gap_to_xi": float(diag.get("Return_Gap_to_Xi", np.nan)),
                "upside_capture": uc,
                "downside_capture": dc,
                "xcdr_v3": float(score_v3.get("XCDR_v3_GrowthControl", np.nan)),
                "xodr_v1": float(score_xodr.get("XODR_v1", np.nan)),
                "xodr_v1_pass": bool(score_xodr.get("XODR_v1_Pass", False)),
                "xodr_v1_uc_omega": float(score_xodr.get("XODR_v1_Upside_Capture_Omega", np.nan)),
                "xodr_v1_dc_omega": float(score_xodr.get("XODR_v1_Downside_Capture_Omega", np.nan)),
                "score": float(score_v3.get("XCDR_v3_GrowthControl", -999.0)),
                "val_ann_return": annual_return(r),
                "val_dd": max_dd_loss(r),
                "val_cvar": cvar_loss(r),
                "val_downside": downside_ann(r),
                "val_xi_dd": xi_dd,
                "val_xi_cvar": xi_cvar,
                "val_xi_downside": xi_downside,
            }
            if variant == "state_optimized_xcdr_v3_policy":
                stress = float((state or {}).get("state_stress", 0.50))
                benchmark_downside_breach = (
                    max(row["val_cvar"] / (xi_cvar + 1e-12) - 1.05, 0.0)
                    + max(row["val_dd"] / (xi_dd + 1e-12) - 1.05, 0.0)
                    + max(row["val_downside"] / (xi_downside + 1e-12) - 1.05, 0.0)
                )
                if benchmark_downside_breach > 0.0:
                    feasible = False
                    row["feasible"] = False
                state_penalty = (
                    stress * max(dc - 0.92, 0.0)
                    + 0.50 * stress * max(beta + gamma - 0.45, 0.0)
                    + 2.50 * benchmark_downside_breach
                )
                row["score"] = row["score"] - state_penalty
                row["state_penalty"] = float(state_penalty)
                row["benchmark_downside_breach"] = float(benchmark_downside_breach)
                row.update({k: v for k, v in (state or {}).items() if k.startswith("state_")})
            if variant == "state_xodr_v1_policy":
                stress = float((state or {}).get("state_stress", 0.50))
                benchmark_downside_breach = (
                    max(row["val_cvar"] / (xi_cvar + 1e-12) - 1.02, 0.0)
                    + max(row["val_dd"] / (xi_dd + 1e-12) - 1.02, 0.0)
                    + max(row["val_downside"] / (xi_downside + 1e-12) - 1.02, 0.0)
                )
                if benchmark_downside_breach > 0.0 or not row["xodr_v1_pass"]:
                    feasible = False
                    row["feasible"] = False
                row["score"] = (
                    row["xodr_v1"]
                    - 2.0 * stress * max(dc - 0.90, 0.0)
                    - 3.0 * benchmark_downside_breach
                    + 0.25 * row["score"]
                )
                row["state_penalty"] = float(2.0 * stress * max(dc - 0.90, 0.0) + 3.0 * benchmark_downside_breach)
                row["benchmark_downside_breach"] = float(benchmark_downside_breach)
                row.update({k: v for k, v in (state or {}).items() if k.startswith("state_")})
            if variant in {"pso_state_xodr_v1_policy", "balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy", "enhanced_growth_xodr_v1_policy"}:
                stress = float((state or {}).get("state_stress", 0.50))
                benchmark_downside_breach = (
                    max(row["val_cvar"] / (xi_cvar + 1e-12) - 1.01, 0.0)
                    + max(row["val_dd"] / (xi_dd + 1e-12) - 1.01, 0.0)
                    + max(row["val_downside"] / (xi_downside + 1e-12) - 1.01, 0.0)
                )
                active_val = annual_return(r - xi_val)
                if variant in {"tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy"}:
                    soft_breach = (
                        max(row["val_cvar"] / (xi_cvar + 1e-12) - 1.05, 0.0)
                        + max(row["val_dd"] / (xi_dd + 1e-12) - 1.05, 0.0)
                        + max(row["val_downside"] / (xi_downside + 1e-12) - 1.05, 0.0)
                    )
                    feasible = bool(
                        diag.get("Downside_Preservation_Pass", False)
                        and np.isfinite(dc)
                        and dc < 1.0
                        and soft_breach <= 0.20
                    )
                    row["feasible"] = feasible
                    row["tail_aware_soft_breach"] = float(soft_breach)
                    row["score"] = (
                        row["xodr_v1"]
                        + 0.25 * row["score"]
                        + 1.10 * active_val
                        + 0.55 * max(uc - 0.90, 0.0)
                        - 1.75 * stress * max(dc - 0.92, 0.0)
                        - 2.50 * soft_breach
                        - 0.15 * (beta + gamma)
                    )
                    row["state_penalty"] = float(
                        1.75 * stress * max(dc - 0.92, 0.0) + 2.50 * soft_breach + 0.15 * (beta + gamma)
                    )
                else:
                    if benchmark_downside_breach > 0.0 or not row["xodr_v1_pass"]:
                        feasible = False
                        row["feasible"] = False
                    row["score"] = (
                        row["xodr_v1"]
                        + 0.20 * row["score"]
                        + (0.50 * active_val if variant in {"balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "enhanced_growth_xodr_v1_policy"} else 0.0)
                        + (0.25 * max(uc - 1.0, 0.0) if variant in {"balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "enhanced_growth_xodr_v1_policy"} else 0.0)
                        - 2.50 * stress * max(dc - 0.90, 0.0)
                        - 4.00 * benchmark_downside_breach
                    )
                    row["state_penalty"] = float(2.50 * stress * max(dc - 0.90, 0.0) + 4.00 * benchmark_downside_breach)
                row["benchmark_downside_breach"] = float(benchmark_downside_breach)
                if variant in {"balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy"}:
                    anchor = books.get("reference_anchor", pd.Series(dtype=float))
                    row["reference_anchor_mass"] = float(w.reindex(anchor.index).fillna(0.0).sum()) if not anchor.empty else 0.0
                row.update(overlay_diag)
                row.update(pso_diag)
                row.update({k: v for k, v in (state or {}).items() if k.startswith("state_")})
            if variant in {"tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy"}:
                stress = float((state or {}).get("state_stress", 0.50))
                tail_breach = float(tail_diag.get("tail_breach", np.inf))
                if tail_breach > 1e-10 or not bool(tail_diag.get("tail_pass", False)):
                    feasible = False
                    row["feasible"] = False
                if variant == "risk_anchor_tail_pso_xodr_v1_policy":
                    active_val = annual_return(r - xi_val)
                    row["score"] = (
                        row["xodr_v1"]
                        + 0.20 * row["score"]
                        + 0.75 * active_val
                        + 0.35 * max(uc - 1.0, 0.0)
                        - 1.75 * stress * max(dc - 0.92, 0.0)
                        - 2.50 * tail_breach
                    )
                else:
                    row["score"] = (
                        row["xodr_v1"]
                        + 0.20 * row["score"]
                        - 2.00 * stress * max(dc - 0.90, 0.0)
                        - 6.00 * tail_breach
                    )
                row.update(pso_diag)
                row.update(tail_diag)
                row["state_penalty"] = float(
                    (1.75 if variant == "risk_anchor_tail_pso_xodr_v1_policy" else 2.00)
                    * stress
                    * max(dc - (0.92 if variant == "risk_anchor_tail_pso_xodr_v1_policy" else 0.90), 0.0)
                    + (2.50 if variant == "risk_anchor_tail_pso_xodr_v1_policy" else 6.00) * tail_breach
                )
                row["benchmark_downside_breach"] = tail_breach
                if variant == "risk_anchor_tail_pso_xodr_v1_policy":
                    anchor = books.get("reference_anchor", pd.Series(dtype=float))
                    row["reference_anchor_mass"] = float(w.reindex(anchor.index).fillna(0.0).sum()) if not anchor.empty else 0.0
                row.update({k: v for k, v in (state or {}).items() if k.startswith("state_")})
            if best is None:
                best = (w, row)
                continue
            if variant in {"tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy"}:
                row_breach = float(row.get("tail_aware_soft_breach", np.inf))
                best_breach = float(best[1].get("tail_aware_soft_breach", np.inf))
                lhs = (
                    not row["feasible"],
                    row_breach if not row["feasible"] else 0.0,
                    row["return_gap_to_xi"],
                    -row["score"],
                    not row["capture_pass"],
                )
                rhs = (
                    not best[1]["feasible"],
                    best_breach if not best[1]["feasible"] else 0.0,
                    best[1]["return_gap_to_xi"],
                    -best[1]["score"],
                    not best[1]["capture_pass"],
                )
            elif variant in {"xcdr_v3_growth_control_policy", "state_optimized_xcdr_v3_policy", "state_xodr_v1_policy", "pso_state_xodr_v1_policy", "tail_pso_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy", "balanced_anchor_xodr_v1_policy", "enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy", "enhanced_growth_xodr_v1_policy"}:
                lhs = (not row["feasible"], not row["capture_pass"], -row["score"], row["return_gap_to_xi"])
                rhs = (not best[1]["feasible"], not best[1]["capture_pass"], -best[1]["score"], best[1]["return_gap_to_xi"])
            else:
                lhs = (not row["feasible"], row["return_gap_to_xi"], -row["score"])
                rhs = (not best[1]["feasible"], best[1]["return_gap_to_xi"], -best[1]["score"])
            if lhs < rhs:
                best = (w, row)
    return best


def run_window(
    task: dict,
    returns: pd.DataFrame,
    volumes: pd.DataFrame,
    omega_cols: list[str],
    cfg: BatchConfig,
) -> dict[str, list[dict]]:
    train = returns.loc[task["train_start"] : task["train_end"]]
    val = returns.loc[task["validation_start"] : task["validation_end"]]
    test = returns.loc[task["test_start"] : task["test_end"]]

    omega = train[[c for c in omega_cols if c in train.columns]].copy()
    raw_asset_cols = [c for c in train.columns if c not in omega_cols and c not in REFERENCE_ASSETS]
    obs = train[raw_asset_cols].notna().sum()
    raw_asset_cols = obs[obs >= 252].index.tolist()
    broad_rank = (
        0.65 * robust_z(train[raw_asset_cols].tail(126).mean() * TRADING_DAYS)
        + 0.35 * robust_z(train[raw_asset_cols].tail(min(252, len(train))).mean() * TRADING_DAYS)
        - 0.30 * robust_z(train[raw_asset_cols].apply(downside_ann))
        - 0.20 * robust_z(train[raw_asset_cols].apply(cvar_loss))
    )
    broad_limit = min(len(raw_asset_cols), max(cfg.universe_limit * 4, cfg.universe_limit + 60))
    broad_cols = broad_rank.sort_values(ascending=False).head(broad_limit).index.tolist()
    ref_cols = [c for c in REFERENCE_ASSETS if c in train.columns and train[c].notna().sum() >= 252]
    initial_train_a = train[broad_cols].fillna(0.0)
    xi, xi_table = select_xi(initial_train_a, omega, cfg.max_weight)
    xi_train_initial = train[xi].reindex(initial_train_a.index).fillna(0.0)
    volume_train = volumes.loc[train.index] if volumes is not None and not volumes.empty else pd.DataFrame(index=train.index)
    fundamental_frame_broad = fundamental_scores_asof(broad_cols, task["train_end"])
    asset_cols, universe_table = convex_opportunity_universe_builder(
        train,
        volume_train,
        broad_cols,
        xi_train_initial,
        fundamental_frame_broad,
        cfg.universe_limit,
    )
    if not asset_cols:
        asset_cols = broad_rank.sort_values(ascending=False).head(cfg.universe_limit).index.tolist()
    portfolio_cols = list(dict.fromkeys(asset_cols + ref_cols))
    train_a = train[asset_cols].fillna(0.0)
    val_a = val[portfolio_cols].fillna(0.0)
    test_a = test[portfolio_cols].fillna(0.0)

    xi, xi_table = select_xi(train_a, omega, cfg.max_weight)
    xi_train = train[xi].reindex(train_a.index).fillna(0.0)
    xi_val = val[xi].reindex(val_a.index).fillna(0.0)
    xi_test = test[xi].reindex(test_a.index).fillna(0.0)
    omega_val = val[[c for c in omega_cols if c in val.columns]].reindex(val_a.index).fillna(0.0)
    omega_test = test[[c for c in omega_cols if c in test.columns]].reindex(test_a.index).fillna(0.0)
    fundamental_frame = fundamental_frame_broad.reindex(asset_cols)
    fundamental_score = (
        fundamental_frame["Fundamental_Upside_Score"].reindex(asset_cols).fillna(0.0)
        if "Fundamental_Upside_Score" in fundamental_frame
        else pd.Series(0.0, index=asset_cols)
    )
    universe_diag = {
        "universe_raw_count": float(len(raw_asset_cols)),
        "universe_broad_count": float(len(broad_cols)),
        "universe_selected_count": float(len(asset_cols)),
        "universe_selected_avg_opportunity_score": float(
            universe_table.reindex(asset_cols)["Opportunity_Score"].mean()
            if not universe_table.empty and "Opportunity_Score" in universe_table
            else np.nan
        ),
        "universe_selected_avg_fundamental_confidence": float(
            universe_table.reindex(asset_cols)["Fundamental_PIT_Confidence"].mean()
            if not universe_table.empty and "Fundamental_PIT_Confidence" in universe_table
            else np.nan
        ),
        "universe_selected_tail_admissible_rate": float(
            universe_table.reindex(asset_cols)["Tail_Admissible"].fillna(False).mean()
            if not universe_table.empty and "Tail_Admissible" in universe_table
            else np.nan
        ),
    }
    equity_books = signal_weights(train_a, xi_train, cfg.max_weight, fundamental_score=fundamental_score)
    books = {k: v.reindex(portfolio_cols).fillna(0.0) for k, v in equity_books.items()}
    enhanced_books = {
        "capital": books["capital"],
        "growth": books.get("growth_plus", books["growth"]),
        "alpha": books.get("alpha_plus", books["alpha"]),
    }
    tail_aware_books = {
        "capital": books["capital"],
        "growth": books.get("growth_tail_aware", books.get("growth_plus", books["growth"])),
        "alpha": books.get("alpha_tail_aware", books.get("alpha_plus", books["alpha"])),
    }
    tail_convex_books = {
        "capital": books.get("capital_xi_tail", books["capital"]),
        "growth": books.get("growth_tail_convex", books.get("growth_tail_aware", books.get("growth_plus", books["growth"]))),
        "alpha": books.get("alpha_tail_convex", books.get("alpha_tail_aware", books.get("alpha_plus", books["alpha"]))),
    }
    upside_convex_books = {
        "capital": books.get("capital_xi_tail", books["capital"]),
        "growth": books.get("growth_upside_convex", books.get("growth_tail_convex", books.get("growth_plus", books["growth"]))),
        "alpha": books.get("alpha_upside_convex", books.get("alpha_tail_convex", books.get("alpha_plus", books["alpha"]))),
    }
    fundamental_upside_books = {
        "capital": books.get("capital_xi_tail", books["capital"]),
        "growth": books.get(
            "growth_fundamental_upside_convex",
            books.get("growth_upside_convex", books.get("growth_tail_convex", books.get("growth_plus", books["growth"]))),
        ),
        "alpha": books.get(
            "alpha_fundamental_upside_convex",
            books.get("alpha_upside_convex", books.get("alpha_tail_convex", books.get("alpha_plus", books["alpha"]))),
        ),
    }
    fundamental_real_upside_books = {
        "capital": books.get("capital_xi_tail", books["capital"]),
        "growth": books.get(
            "growth_fundamental_real_upside_convex",
            books.get("growth_fundamental_upside_convex", books.get("growth_upside_convex", books["growth"])),
        ),
        "alpha": books.get(
            "alpha_fundamental_real_upside_convex",
            books.get("alpha_fundamental_upside_convex", books.get("alpha_upside_convex", books["alpha"])),
        ),
    }
    state = market_state_optimizer(train_a, xi_train)
    ref_anchor = reference_risk_anchor_weights(train[ref_cols].fillna(0.0), xi_train) if ref_cols else pd.Series(dtype=float)
    hedge_floor = float(np.clip(0.04 + 0.18 * state.get("state_stress", 0.50), 0.04, 0.24))
    anchor_books = add_reference_anchor_to_books(
        books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    enhanced_anchor_books = add_reference_anchor_to_books(
        enhanced_books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    tail_aware_anchor_books = add_reference_anchor_to_books(
        tail_aware_books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    tail_convex_anchor_books = add_reference_anchor_to_books(
        tail_convex_books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    upside_convex_anchor_books = add_reference_anchor_to_books(
        upside_convex_books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    fundamental_upside_anchor_books = add_reference_anchor_to_books(
        fundamental_upside_books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    fundamental_real_upside_anchor_books = add_reference_anchor_to_books(
        fundamental_real_upside_books,
        ref_anchor.reindex(portfolio_cols).fillna(0.0) if not ref_anchor.empty else ref_anchor,
        hedge_floor=hedge_floor,
        max_weight=cfg.max_weight,
    )
    variants = [
        "capital_preservation_policy",
        "downside_preserving_growth_policy",
        "xcdr_v3_growth_control_policy",
        "state_optimized_xcdr_v3_policy",
        "state_xodr_v1_policy",
        "pso_state_xodr_v1_policy",
        "tail_pso_xodr_v1_policy",
        "enhanced_growth_xodr_v1_policy",
        "balanced_anchor_xodr_v1_policy",
        "enhanced_growth_anchor_xodr_v1_policy",
        "enhanced_growth_anchor_dd_control_policy",
        "enhanced_growth_anchor_crash_budget_policy",
        "enhanced_growth_anchor_dd_budget_policy",
        "tail_aware_anchor_dd_budget_policy",
        "tail_convex_anchor_dd_budget_policy",
        "upside_convex_anchor_dd_budget_policy",
        "fundamental_upside_convex_anchor_dd_budget_policy",
        "fundamental_upside_recovery_anchor_dd_budget_policy",
        "fundamental_real_upside_anchor_dd_budget_policy",
        "risk_anchor_tail_pso_xodr_v1_policy",
    ]
    out = []
    daily_out = []
    weights_out = []
    for variant in variants:
        if variant == "fundamental_real_upside_anchor_dd_budget_policy":
            variant_books = fundamental_real_upside_anchor_books
        elif variant == "fundamental_upside_recovery_anchor_dd_budget_policy":
            variant_books = fundamental_upside_anchor_books
        elif variant == "fundamental_upside_convex_anchor_dd_budget_policy":
            variant_books = fundamental_upside_anchor_books
        elif variant == "upside_convex_anchor_dd_budget_policy":
            variant_books = upside_convex_anchor_books
        elif variant == "tail_convex_anchor_dd_budget_policy":
            variant_books = tail_convex_anchor_books
        elif variant == "tail_aware_anchor_dd_budget_policy":
            variant_books = tail_aware_anchor_books
        elif variant in {"enhanced_growth_anchor_xodr_v1_policy", "enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy", "enhanced_growth_anchor_dd_budget_policy"}:
            variant_books = enhanced_anchor_books
        elif variant == "enhanced_growth_xodr_v1_policy":
            variant_books = enhanced_books
        elif variant in {"balanced_anchor_xodr_v1_policy", "risk_anchor_tail_pso_xodr_v1_policy"}:
            variant_books = anchor_books
        else:
            variant_books = books
        if variant == "capital_preservation_policy":
            w = variant_books["capital"]
            meta = {
                "variant": variant,
                "alpha_mass": 1.0,
                "growth_mass": 0.0,
                "alpha_signal_mass": 0.0,
                "feasible": True,
                "capture_pass": False,
                "return_gap_to_xi": np.nan,
                "upside_capture": np.nan,
                "downside_capture": np.nan,
                "xcdr_v3": np.nan,
                "xodr_v1": np.nan,
                "xodr_v1_pass": False,
            }
        else:
            w, meta = eval_candidate_on_validation(val_a, xi_val, omega_val, variant_books, variant, cfg, state=state)
        raw_test_r = test_a @ w
        if variant in {"enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy"}:
            anchor_test_r, test_anchor_kind = defensive_overlay_anchor_returns(test_a, variant_books)
        else:
            anchor_test_r = test_a @ variant_books["capital"]
            test_anchor_kind = "capital_book"
        if variant in {"enhanced_growth_anchor_dd_budget_policy", "tail_aware_anchor_dd_budget_policy", "tail_convex_anchor_dd_budget_policy", "upside_convex_anchor_dd_budget_policy", "fundamental_upside_convex_anchor_dd_budget_policy", "fundamental_real_upside_anchor_dd_budget_policy", "fundamental_upside_recovery_anchor_dd_budget_policy"}:
            r, test_overlay_diag = causal_drawdown_vol_overlay(
                raw_test_r,
                anchor_test_r,
                state,
                benchmark_returns=xi_test,
                min_exposure=float(meta.get("budget_min_exposure", 0.25)),
                rerisk_step=float(meta.get("budget_rerisk_step", 0.10)),
                dd_soft_shift=float(meta.get("budget_dd_soft_shift", 0.0)),
                dd_hard_gap=float(meta.get("budget_dd_hard_gap", 0.04)),
                vol_target_shift=float(meta.get("budget_vol_target_shift", 0.0)),
            )
        elif variant in {"enhanced_growth_anchor_dd_control_policy", "enhanced_growth_anchor_crash_budget_policy"}:
            r, test_overlay_diag = causal_drawdown_vol_overlay(raw_test_r, anchor_test_r, state)
        else:
            r = raw_test_r
            test_overlay_diag = {}
        active_weights = pd.Series(w, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        active_weights = active_weights[active_weights.abs() > 1e-10].sort_values(ascending=False)
        for ticker, weight in active_weights.items():
            weights_out.append(
                {
                    **task,
                    "objective": variant,
                    "xi": xi,
                    "ticker": str(ticker),
                    "weight": float(weight),
                    "n_assets": int((pd.Series(w) > 1e-8).sum()),
                    "max_weight": float(pd.Series(w).max()),
                    "state_label": state.get("state_label", "unknown"),
                    "state_stress": state.get("state_stress", np.nan),
                    "state_risk_on": state.get("state_risk_on", np.nan),
                    "state_recovery": state.get("state_recovery", np.nan),
                    "val_growth_mass": meta.get("growth_mass", np.nan),
                    "val_alpha_signal_mass": meta.get("alpha_signal_mass", np.nan),
                    "budget_min_exposure": meta.get("budget_min_exposure", np.nan),
                    "budget_rerisk_step": meta.get("budget_rerisk_step", np.nan),
                    "budget_dd_soft_shift": meta.get("budget_dd_soft_shift", np.nan),
                    "budget_dd_hard_gap": meta.get("budget_dd_hard_gap", np.nan),
                    "budget_vol_target_shift": meta.get("budget_vol_target_shift", np.nan),
                    "overlay_anchor_kind": meta.get("overlay_anchor_kind", test_anchor_kind),
                    **universe_diag,
                }
            )
        diag_test = upside_downside_diagnostics(r, xi_test, baseline_returns=test_a @ variant_books["capital"], tolerance=0.05)
        v3_test = xcdr_v3_growth_control_score(r, xi_test, weights=w)
        xodr_test = xodr_v1_omega_dominance_score(r, xi_test, omega_test, weights=w)
        for dt, pr in r.items():
            xr = float(xi_test.reindex(r.index).loc[dt])
            day = test_a.reindex(r.index).loc[dt]
            daily_out.append(
                {
                    **task,
                    "date": dt,
                    "objective": variant,
                    "xi": xi,
                    "portfolio_return": float(pr),
                    "raw_portfolio_return": float(raw_test_r.reindex(r.index).loc[dt]),
                    "anchor_return": float(anchor_test_r.reindex(r.index).loc[dt]),
                    "xi_return": xr,
                    "active_return": float(pr - xr),
                    "state_label": state.get("state_label", "unknown"),
                    "state_stress": state.get("state_stress", np.nan),
                    "state_risk_on": state.get("state_risk_on", np.nan),
                    "state_recovery": state.get("state_recovery", np.nan),
                    "market_down_breadth": float((day < 0.0).mean()),
                    "market_tail_breadth": float((day < -0.02).mean()),
                    "market_dispersion": float(day.std(ddof=0)),
                    "val_growth_mass": meta.get("growth_mass", np.nan),
                    "val_alpha_signal_mass": meta.get("alpha_signal_mass", np.nan),
                    "budget_min_exposure": meta.get("budget_min_exposure", np.nan),
                    "budget_rerisk_step": meta.get("budget_rerisk_step", np.nan),
                    "budget_dd_soft_shift": meta.get("budget_dd_soft_shift", np.nan),
                    "budget_dd_hard_gap": meta.get("budget_dd_hard_gap", np.nan),
                    "budget_vol_target_shift": meta.get("budget_vol_target_shift", np.nan),
                    "overlay_anchor_kind": meta.get("overlay_anchor_kind", test_anchor_kind),
                    **universe_diag,
                }
            )
        out.append(
            {
                **task,
                "objective": variant,
                "xi": xi,
                "test_obs": int(len(r)),
                "test_ann_return": annual_return(r),
                "xi_ann_return": annual_return(xi_test),
                "active_ann_return": annual_return(r - xi_test),
                "test_ann_vol": annual_vol(r),
                "xi_ann_vol": annual_vol(xi_test),
                "test_downside": downside_ann(r),
                "xi_downside": downside_ann(xi_test),
                "test_cvar_loss": cvar_loss(r),
                "xi_cvar_loss": cvar_loss(xi_test),
                "test_maxdd_loss": max_dd_loss(r),
                "xi_maxdd_loss": max_dd_loss(xi_test),
                "test_upside_capture": diag_test.get("Upside_Capture", np.nan),
                "test_downside_capture": diag_test.get("Downside_Capture", np.nan),
                "test_downside_preservation": diag_test.get("Downside_Preservation_Pass", False),
                "test_xcdr_v3": v3_test.get("XCDR_v3_GrowthControl", np.nan),
                "test_xcdr_v3_capture_pass": v3_test.get("XCDR_v3_Capture_Pass", False),
                "test_xodr_v1": xodr_test.get("XODR_v1", np.nan),
                "test_xodr_v1_pass": xodr_test.get("XODR_v1_Pass", False),
                "test_xodr_v1_uc_omega": xodr_test.get("XODR_v1_Upside_Capture_Omega", np.nan),
                "test_xodr_v1_dc_omega": xodr_test.get("XODR_v1_Downside_Capture_Omega", np.nan),
                "n_assets": int((w > 1e-8).sum()),
                "max_weight": float(w.max()),
                **{f"test_{k}": v for k, v in test_overlay_diag.items()},
                "test_overlay_anchor_kind": test_anchor_kind,
                **universe_diag,
                **state,
                **{f"val_{k}": v for k, v in meta.items() if k not in {"variant"}},
            }
        )
    return {"metrics": out, "daily": daily_out, "weights": weights_out}


def white_reality_check(
    candidate_returns: dict[str, pd.Series],
    benchmark: pd.Series,
    n_boot: int,
    min_obs: int = 20,
    block_length: int = 0,
) -> dict:
    aligned = []
    b = pd.Series(benchmark).dropna()
    for name, s in candidate_returns.items():
        idx = pd.Series(s).dropna().index.intersection(b.index)
        if len(idx) >= min_obs:
            aligned.append((pd.Series(s).loc[idx] - b.loc[idx]).rename(name))
    if not aligned:
        return {"WRC_p": np.nan, "WRC_candidates": 0}
    A = pd.concat(aligned, axis=1).dropna()
    obs = float(A.mean().max())
    centered = A - A.mean(axis=0)
    block_length = int(block_length or max(5, round(len(centered) ** (1 / 3))))
    boot = []
    for _ in range(n_boot):
        idx = block_bootstrap_indices(len(centered), block_length)
        boot.append(float(centered.iloc[idx].mean().max()))
    return {
        "WRC_p": float(np.mean(np.array(boot) >= obs)),
        "WRC_candidates": int(A.shape[1]),
        "WRC_block_length": int(block_length),
        "WRC_obs_mean": obs,
    }


def spa(
    candidate_returns: dict[str, pd.Series],
    benchmark: pd.Series,
    n_boot: int,
    min_obs: int = 20,
    block_length: int = 0,
) -> dict:
    aligned = []
    b = pd.Series(benchmark).dropna()
    for name, s in candidate_returns.items():
        idx = pd.Series(s).dropna().index.intersection(b.index)
        if len(idx) >= min_obs:
            aligned.append((pd.Series(s).loc[idx] - b.loc[idx]).rename(name))
    if not aligned:
        return {"SPA_p": np.nan}
    D = pd.concat(aligned, axis=1).dropna()
    se = D.std(ddof=1) / np.sqrt(max(len(D), 1))
    t_obs = float((D.mean() / se.replace(0, np.nan)).max())
    centered = D - D.mean(axis=0)
    block_length = int(block_length or max(5, round(len(centered) ** (1 / 3))))
    boot = []
    for _ in range(n_boot):
        idx = block_bootstrap_indices(len(centered), block_length)
        bd = centered.iloc[idx]
        bse = bd.std(ddof=1) / np.sqrt(max(len(bd), 1))
        boot.append(float((bd.mean() / bse.replace(0, np.nan)).max()))
    return {
        "SPA_p": float(np.mean(np.array(boot) >= t_obs)),
        "SPA_T": t_obs,
        "SPA_block_length": int(block_length),
        "SPA_candidates": int(D.shape[1]),
    }


def block_bootstrap_indices(n: int, block_length: int) -> np.ndarray:
    n = int(n)
    block_length = int(max(1, min(block_length, n))) if n > 0 else 1
    if n <= 0:
        return np.array([], dtype=int)
    starts = RNG.integers(0, n, size=int(np.ceil(n / block_length)))
    idx = []
    for start in starts:
        idx.extend(((int(start) + j) % n) for j in range(block_length))
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def pbo_window_proxy(rows: pd.DataFrame, max_splits: int = 4096, objectives: tuple[str, ...] | None = None) -> float:
    piv = rows.pivot_table(index="test_start", columns="objective", values="active_ann_return", aggfunc="mean")
    if objectives:
        cols = [c for c in objectives if c in piv.columns]
        piv = piv[cols] if cols else pd.DataFrame(index=piv.index)
    piv = piv.dropna(axis=0, how="any")
    if len(piv) < 4 or piv.shape[1] < 2:
        return np.nan
    vals = []
    idx = np.arange(len(piv))
    full_masks = np.arange(1, 2 ** len(idx) - 1, dtype=np.int64)
    if len(full_masks) > max_splits:
        masks = RNG.choice(full_masks, size=max_splits, replace=False)
    else:
        masks = full_masks
    for mask in masks:
        train_idx = [i for i in idx if mask & (1 << i)]
        test_idx = [i for i in idx if i not in train_idx]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        best = piv.iloc[train_idx].mean().idxmax()
        test_ranks = piv.iloc[test_idx].mean().rank(pct=True)
        vals.append(float(test_ranks.get(best, np.nan) < 0.50))
    return float(np.nanmean(vals)) if vals else np.nan


def daily_oos_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for obj, sub in daily.groupby("objective"):
        p = sub.set_index("date")["portfolio_return"].astype(float).sort_index()
        x = sub.set_index("date")["xi_return"].astype(float).sort_index()
        active = p - x
        diag = upside_downside_diagnostics(p, x, baseline_returns=x, tolerance=0.05)
        rows.append(
            {
                "objective": obj,
                "daily_obs": int(len(p)),
                "daily_ann_return": annual_return(p),
                "daily_xi_ann_return": annual_return(x),
                "daily_active_ann_return": annual_return(active),
                "daily_ann_vol": annual_vol(p),
                "daily_xi_ann_vol": annual_vol(x),
                "daily_downside": downside_ann(p),
                "daily_xi_downside": downside_ann(x),
                "daily_cvar_loss": cvar_loss(p),
                "daily_xi_cvar_loss": cvar_loss(x),
                "daily_maxdd_loss": max_dd_loss(p),
                "daily_xi_maxdd_loss": max_dd_loss(x),
                "daily_upside_capture": diag.get("Upside_Capture", np.nan),
                "daily_downside_capture": diag.get("Downside_Capture", np.nan),
                "daily_downside_preservation": diag.get("Downside_Preservation_Pass", False),
            }
        )
    return pd.DataFrame(rows)


def red_team_daily_oos(daily: pd.DataFrame, cost_per_rebalance: float = 0.0005) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    for obj, sub in daily.groupby("objective"):
        s = sub.copy()
        s["date"] = pd.to_datetime(s["date"])
        s = s.sort_values(["test_start", "date"])
        p = s.set_index("date")["portfolio_return"].astype(float)
        x = s.set_index("date")["xi_return"].astype(float)
        active = p - x

        rebalance_cost = pd.Series(0.0, index=s.index)
        first_idx = s.groupby("test_start").head(1).index
        rebalance_cost.loc[first_idx] = cost_per_rebalance
        p_costed = pd.Series(s["portfolio_return"].to_numpy() - rebalance_cost.to_numpy(), index=s["date"])

        top_cut = active.quantile(0.95)
        active_without_best_tail = active.clip(upper=top_cut)
        p_without_best_tail = x + active_without_best_tail

        down_mask = x < 0.0
        p_downside_shock = p.copy()
        p_downside_shock.loc[down_mask] = p_downside_shock.loc[down_mask] * 1.25

        scenarios = {
            "base": p,
            "rebalance_cost_5bps": p_costed,
            "remove_top_5pct_active": p_without_best_tail,
            "downside_returns_25pct_worse": p_downside_shock,
        }
        for name, rr in scenarios.items():
            diag = upside_downside_diagnostics(rr, x, baseline_returns=x, tolerance=0.05)
            rows.append(
                {
                    "objective": obj,
                    "scenario": name,
                    "ann_return": annual_return(rr),
                    "xi_ann_return": annual_return(x),
                    "active_ann_return": annual_return(rr - x),
                    "downside_capture": diag.get("Downside_Capture", np.nan),
                    "downside_preservation": diag.get("Downside_Preservation_Pass", False),
                    "cvar_loss": cvar_loss(rr),
                    "xi_cvar_loss": cvar_loss(x),
                    "maxdd_loss": max_dd_loss(rr),
                    "xi_maxdd_loss": max_dd_loss(x),
                }
            )
    return pd.DataFrame(rows)


def apply_persistent_oos_overlay(daily: pd.DataFrame, objective: str) -> pd.DataFrame:
    """Apply drawdown/vol control across the full daily OOS path.

    Window-level overlays reset at each rebalance. This function models the
    deployable path: NAV, peak, realized volatility and exposure persist across
    OOS windows, while each day's throttle still uses only information available
    before that day.
    """
    if daily.empty or objective not in set(daily["objective"].astype(str)):
        return daily
    out = daily.copy()
    mask = out["objective"].astype(str) == objective
    sub = out.loc[mask].copy().sort_values(["date", "test_start"])
    nav = 1.0
    peak = 1.0
    xi_nav = 1.0
    xi_peak = 1.0
    exposure = 1.0
    history: list[float] = []
    prev_down_breadth = 0.0
    prev_tail_breadth = 0.0
    prev_dispersion = 0.0
    prev_xi_return = 0.0
    use_crash_budget = "crash_budget" in objective or "dd_budget" in objective
    use_drawdown_budget = "dd_budget" in objective
    rows = []
    for idx, row in sub.iterrows():
        stress = float(row.get("state_stress", 0.50))
        risk_on = float(row.get("state_risk_on", 0.0))
        recovery = float(row.get("state_recovery", 0.0))
        min_exposure = 0.20
        rerisk_step = 0.10
        dd_soft_shift = 0.0
        dd_hard_gap = None
        vol_target_shift = 0.0
        if use_drawdown_budget:
            min_exposure = float(row.get("budget_min_exposure", min_exposure))
            rerisk_step = float(row.get("budget_rerisk_step", rerisk_step))
            dd_soft_shift = float(row.get("budget_dd_soft_shift", 0.0))
            dd_hard_gap = float(row.get("budget_dd_hard_gap", 0.04))
            vol_target_shift = float(row.get("budget_vol_target_shift", 0.0))
        dd_soft = float(np.clip(0.045 - 0.025 * stress + 0.010 * risk_on + dd_soft_shift, 0.012, 0.070))
        base_hard_gap = float(dd_hard_gap) if dd_hard_gap is not None else float(0.045 - 0.015 * stress)
        dd_hard = float(np.clip(dd_soft + base_hard_gap, dd_soft + 0.012, dd_soft + 0.090))
        vol_target = float(np.clip(0.135 + 0.045 * risk_on + 0.025 * recovery - 0.045 * stress + vol_target_shift, 0.070, 0.190))
        dd_loss = max(1.0 - nav / max(peak, 1e-12), 0.0)
        xi_dd_loss = max(1.0 - xi_nav / max(xi_peak, 1e-12), 0.0)
        rel_buffer = 0.008 + 0.15 * xi_dd_loss
        dd_soft = max(dd_soft, float(np.clip(xi_dd_loss + rel_buffer, 0.010, 0.095)))
        dd_hard = max(dd_hard, float(np.clip(xi_dd_loss + rel_buffer + 0.018, dd_soft + 0.010, 0.125)))
        path_dd_excess_pre = max(dd_loss - (xi_dd_loss + rel_buffer), 0.0)
        path_hard = xi_dd_loss + rel_buffer + max(0.010, 0.50 * base_hard_gap)
        if path_dd_excess_pre <= 1e-12:
            path_scale = 1.0
        elif dd_loss >= path_hard:
            path_scale = min_exposure
        else:
            path_frac = path_dd_excess_pre / max(path_hard - (xi_dd_loss + rel_buffer), 1e-12)
            path_scale = 1.0 - path_frac * (1.0 - min_exposure)
        if dd_loss <= dd_soft:
            dd_scale = 1.0
        elif dd_loss >= dd_hard:
            dd_scale = min_exposure
        else:
            frac = (dd_loss - dd_soft) / max(dd_hard - dd_soft, 1e-12)
            dd_scale = 1.0 - frac * (1.0 - min_exposure)
        dd_scale = min(dd_scale, path_scale)
        if len(history) >= 21:
            rv = float(pd.Series(history[-21:]).std(ddof=1) * np.sqrt(TRADING_DAYS))
            vol_scale = min(1.0, vol_target / max(rv, 1e-12))
        else:
            vol_scale = 1.0
        crash_pressure = 0.0
        if use_crash_budget:
            crash_pressure = float(
                np.clip(
                    0.42 * prev_down_breadth
                    + 0.28 * np.clip(prev_tail_breadth / 0.12, 0.0, 1.0)
                    + 0.20 * np.clip(prev_dispersion * np.sqrt(TRADING_DAYS) / 0.40, 0.0, 1.0)
                    + 0.10 * (1.0 if prev_xi_return < 0.0 else 0.0),
                    0.0,
                    1.0,
                )
            )
            if crash_pressure > 0.55:
                shock_frac = (crash_pressure - 0.55) / 0.45
                shock_scale = 1.0 - 0.60 * shock_frac
                vol_scale = min(vol_scale, shock_scale)
                dd_scale = min(dd_scale, shock_scale)
        target = float(np.clip(min(dd_scale, vol_scale), min_exposure, 1.0))
        if target > exposure:
            step = rerisk_step
            if path_dd_excess_pre <= 1e-12 and prev_xi_return > 0.0:
                step = min(0.35, rerisk_step + 0.10 + 0.05 * risk_on + 0.04 * recovery)
            exposure = min(target, exposure + step)
        else:
            exposure = target
        raw = float(row.get("raw_portfolio_return", row.get("portfolio_return", 0.0)))
        anchor = float(row.get("anchor_return", 0.0))
        ret = exposure * raw + (1.0 - exposure) * anchor
        xi = float(row.get("xi_return", 0.0))
        nav *= 1.0 + ret
        peak = max(peak, nav)
        xi_nav *= 1.0 + xi
        xi_peak = max(xi_peak, xi_nav)
        prev_down_breadth = float(row.get("market_down_breadth", 0.0))
        prev_tail_breadth = float(row.get("market_tail_breadth", 0.0))
        prev_dispersion = float(row.get("market_dispersion", 0.0))
        prev_xi_return = xi
        history.append(ret)
        rows.append((idx, ret, ret - xi, raw, exposure, crash_pressure, path_dd_excess_pre))
    for idx, ret, active, raw, exp, crash_pressure, path_dd_excess_pre in rows:
        out.loc[idx, "portfolio_return"] = ret
        out.loc[idx, "active_return"] = active
        out.loc[idx, "persistent_raw_portfolio_return"] = raw
        out.loc[idx, "persistent_overlay_exposure"] = exp
        out.loc[idx, "persistent_crash_pressure"] = crash_pressure
        out.loc[idx, "persistent_path_dd_excess_pre"] = path_dd_excess_pre
    return out


def main() -> int:
    cfg = BatchConfig(
        train_days=int(os.getenv("QPK_XCDR3_TRAIN_DAYS", "756")),
        validation_days=int(os.getenv("QPK_XCDR3_VALIDATION_DAYS", "126")),
        test_days=int(os.getenv("QPK_XCDR3_TEST_DAYS", "42")),
        universe_limit=int(os.getenv("QPK_XCDR3_UNIVERSE_LIMIT", "90")),
        max_windows=int(os.getenv("QPK_XCDR3_MAX_WINDOWS", "18")),
        workers=int(os.getenv("QPK_XCDR3_WORKERS", str(BatchConfig().workers))),
        bootstrap_n=int(os.getenv("QPK_XCDR3_BOOTSTRAP_N", "300")),
        pso_particles=int(os.getenv("QPK_XCDR3_PSO_PARTICLES", "18")),
        pso_iterations=int(os.getenv("QPK_XCDR3_PSO_ITERATIONS", "18")),
        promotion_objectives=parse_objective_list(os.getenv("QPK_XCDR3_PROMOTION_OBJECTIVES")),
        min_promotion_windows=int(os.getenv("QPK_XCDR3_MIN_PROMOTION_WINDOWS", "12")),
        bootstrap_block_length=int(os.getenv("QPK_XCDR3_BOOTSTRAP_BLOCK_LENGTH", "0")),
    )
    returns, volumes, meta = load_cached_returns(os.getenv("QPK_XCDR3_CACHE_KEY") or None)
    omega_cols = [c for c in BENCHMARKS if c in returns.columns]
    schedule = make_schedule(returns.index, cfg)
    if not schedule:
        raise RuntimeError("No walk-forward windows available")

    rows = []
    daily_rows = []
    weight_rows = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(run_window, task, returns, volumes, omega_cols, cfg) for task in schedule]
        for fut in as_completed(futures):
            payload = fut.result()
            rows.extend(payload["metrics"])
            daily_rows.extend(payload["daily"])
            weight_rows.extend(payload.get("weights", []))
    results = pd.DataFrame(rows).sort_values(["test_start", "objective"])
    daily_results = pd.DataFrame(daily_rows).sort_values(["date", "objective"])
    weight_results = pd.DataFrame(weight_rows)
    if not weight_results.empty:
        weight_results = weight_results.sort_values(["test_start", "objective", "weight"], ascending=[True, True, False])
    daily_results = apply_persistent_oos_overlay(daily_results, "enhanced_growth_anchor_dd_control_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "enhanced_growth_anchor_crash_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "enhanced_growth_anchor_dd_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "tail_aware_anchor_dd_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "tail_convex_anchor_dd_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "upside_convex_anchor_dd_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "fundamental_upside_convex_anchor_dd_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "fundamental_real_upside_anchor_dd_budget_policy")
    daily_results = apply_persistent_oos_overlay(daily_results, "fundamental_upside_recovery_anchor_dd_budget_policy")

    # Build true daily OOS active-return series for WRC/SPA. Each point is an
    # investable test-day return generated by weights selected before test_start.
    daily_active = {}
    if not daily_results.empty:
        daily_results["date"] = pd.to_datetime(daily_results["date"])
        active_pivot = daily_results.pivot_table(
            index="date", columns="objective", values="active_return", aggfunc="mean"
        ).sort_index()
        daily_active = {col: active_pivot[col].dropna() for col in active_pivot.columns}
    zero_bench = pd.Series(0.0, index=next(iter(daily_active.values())).index) if daily_active else pd.Series(dtype=float)
    min_daily_obs = min(252, max(60, int(0.25 * len(zero_bench)))) if len(zero_bench) else 20
    promotion_objectives = tuple(obj for obj in cfg.promotion_objectives if obj in daily_active)
    daily_active_promotion = {obj: daily_active[obj] for obj in promotion_objectives}
    wrc_all = white_reality_check(daily_active, zero_bench, cfg.bootstrap_n, min_obs=min_daily_obs, block_length=cfg.bootstrap_block_length)
    spa_all = spa(daily_active, zero_bench, cfg.bootstrap_n, min_obs=min_daily_obs, block_length=cfg.bootstrap_block_length)
    pbo_all = pbo_window_proxy(results)
    wrc = white_reality_check(daily_active_promotion, zero_bench, cfg.bootstrap_n, min_obs=min_daily_obs, block_length=cfg.bootstrap_block_length)
    spa_res = spa(daily_active_promotion, zero_bench, cfg.bootstrap_n, min_obs=min_daily_obs, block_length=cfg.bootstrap_block_length)
    pbo = pbo_window_proxy(results, objectives=promotion_objectives)
    daily_summary = daily_oos_summary(daily_results)
    red_team = red_team_daily_oos(daily_results)
    holdout_summary = pd.DataFrame()
    if not daily_results.empty:
        starts = sorted(pd.to_datetime(daily_results["test_start"]).unique())
        holdout_n = max(4, int(np.ceil(0.25 * len(starts)))) if starts else 0
        holdout_starts = set(starts[-holdout_n:]) if holdout_n else set()
        holdout_daily = daily_results[pd.to_datetime(daily_results["test_start"]).isin(holdout_starts)].copy()
        holdout_summary = daily_oos_summary(holdout_daily)

    summary = (
        results.groupby("objective")
        .agg(
            windows=("test_start", "count"),
            ann_return=("test_ann_return", "mean"),
            xi_ann_return=("xi_ann_return", "mean"),
            active_ann_return=("active_ann_return", "mean"),
            ann_vol=("test_ann_vol", "mean"),
            downside=("test_downside", "mean"),
            cvar_loss=("test_cvar_loss", "mean"),
            maxdd_loss=("test_maxdd_loss", "mean"),
            upside_capture=("test_upside_capture", "mean"),
            downside_capture=("test_downside_capture", "mean"),
            downside_preservation_rate=("test_downside_preservation", "mean"),
            xcdr_v3=("test_xcdr_v3", "mean"),
            capture_pass_rate=("test_xcdr_v3_capture_pass", "mean"),
            xodr_v1=("test_xodr_v1", "mean"),
            xodr_pass_rate=("test_xodr_v1_pass", "mean"),
            xodr_uc_omega=("test_xodr_v1_uc_omega", "mean"),
            xodr_dc_omega=("test_xodr_v1_dc_omega", "mean"),
            selected_growth_mass=("val_growth_mass", "mean"),
            selected_alpha_signal_mass=("val_alpha_signal_mass", "mean"),
            selected_reference_anchor_mass=("val_reference_anchor_mass", "mean"),
            val_overlay_avg_exposure=("val_overlay_avg_exposure", "mean"),
            val_overlay_trigger_rate=("val_overlay_trigger_rate", "mean"),
            val_budget_breach=("val_budget_breach", "mean"),
            val_budget_pass_rate=("val_budget_pass", "mean"),
            val_budget_active_ann_return=("val_budget_active_ann_return", "mean"),
            val_budget_path_dd_max_excess=("val_budget_path_dd_max_excess", "mean"),
            val_budget_path_dd_breach_rate=("val_budget_path_dd_breach_rate", "mean"),
            test_overlay_avg_exposure=("test_overlay_avg_exposure", "mean"),
            test_overlay_trigger_rate=("test_overlay_trigger_rate", "mean"),
            pso_best_score=("val_pso_best_score", "mean"),
            pso_particles=("val_pso_particles", "mean"),
            pso_iterations=("val_pso_iterations", "mean"),
            tail_breach=("val_tail_breach", "mean"),
            tail_scale=("val_tail_scale", "mean"),
            tail_pass_rate=("val_tail_pass", "mean"),
            universe_raw_count=("universe_raw_count", "mean"),
            universe_broad_count=("universe_broad_count", "mean"),
            universe_selected_count=("universe_selected_count", "mean"),
            universe_selected_avg_opportunity_score=("universe_selected_avg_opportunity_score", "mean"),
            universe_selected_avg_fundamental_confidence=("universe_selected_avg_fundamental_confidence", "mean"),
            universe_selected_tail_admissible_rate=("universe_selected_tail_admissible_rate", "mean"),
            avg_state_stress=("state_stress", "mean"),
            avg_state_growth_cap=("state_growth_cap", "mean"),
            avg_state_alpha_cap=("state_alpha_cap", "mean"),
        )
        .reset_index()
        .sort_values(["downside_preservation_rate", "capture_pass_rate", "active_ann_return"], ascending=[False, False, False])
    )
    summary["WRC_p"] = wrc["WRC_p"]
    summary["SPA_p"] = spa_res["SPA_p"]
    summary["PBO_proxy"] = pbo
    summary["WRC_AllCandidates_p"] = wrc_all["WRC_p"]
    summary["SPA_AllCandidates_p"] = spa_all["SPA_p"]
    summary["PBO_AllCandidates_proxy"] = pbo_all
    summary["Promotion_Family"] = ",".join(promotion_objectives)
    summary["Promotion_Family_Size"] = len(promotion_objectives)
    summary["Promotion_Min_Windows"] = cfg.min_promotion_windows
    summary["Promotion_Block_Length"] = wrc.get("WRC_block_length")
    if not daily_summary.empty:
        summary = summary.merge(
            daily_summary[
                [
                    "objective",
                    "daily_ann_return",
                    "daily_xi_ann_return",
                    "daily_active_ann_return",
                    "daily_downside_capture",
                    "daily_downside_preservation",
                ]
            ],
            on="objective",
            how="left",
        )
    if not holdout_summary.empty:
        summary = summary.merge(
            holdout_summary[
                [
                    "objective",
                    "daily_ann_return",
                    "daily_xi_ann_return",
                    "daily_active_ann_return",
                    "daily_downside_capture",
                    "daily_downside_preservation",
                ]
            ].rename(
                columns={
                    "daily_ann_return": "holdout_ann_return",
                    "daily_xi_ann_return": "holdout_xi_ann_return",
                    "daily_active_ann_return": "holdout_active_ann_return",
                    "daily_downside_capture": "holdout_downside_capture",
                    "daily_downside_preservation": "holdout_downside_preservation",
                }
            ),
            on="objective",
            how="left",
        )
    summary["research_gate_pass"] = (
        (summary["windows"] >= cfg.min_promotion_windows)
        & (summary["objective"].isin(promotion_objectives))
        & (summary["WRC_p"] < 0.05)
        & (summary["SPA_p"] < 0.05)
        & (summary["PBO_proxy"] < 0.10)
        & (summary["daily_active_ann_return"].fillna(summary["active_ann_return"]) > 0.0)
        & (summary["daily_downside_capture"].fillna(summary["downside_capture"]) < 1.0)
        & (summary["daily_downside_preservation"].fillna(False).astype(bool))
    )
    summary["holdout_gate_pass"] = (
        (summary["holdout_active_ann_return"].fillna(np.nan) > 0.0)
        & (summary["holdout_downside_capture"].fillna(np.inf) < 1.0)
        & (summary["holdout_downside_preservation"].fillna(False).astype(bool))
    )

    out_csv = OUT_DIR / "xcdr_v3_parallel_research_summary.csv"
    out_detail = OUT_DIR / "xcdr_v3_parallel_research_windows.csv"
    out_daily = OUT_DIR / "xcdr_v3_parallel_research_daily_oos.csv"
    out_daily_summary = OUT_DIR / "xcdr_v3_parallel_research_daily_summary.csv"
    out_holdout_summary = OUT_DIR / "xcdr_v3_parallel_research_holdout_summary.csv"
    out_red_team = OUT_DIR / "xcdr_v3_parallel_research_red_team.csv"
    out_weights = OUT_DIR / "xcdr_v3_parallel_research_weights.csv"
    out_json = OUT_DIR / "xcdr_v3_parallel_research_report.json"
    summary.to_csv(out_csv, index=False)
    results.to_csv(out_detail, index=False)
    daily_results.to_csv(out_daily, index=False)
    daily_summary.to_csv(out_daily_summary, index=False)
    holdout_summary.to_csv(out_holdout_summary, index=False)
    red_team.to_csv(out_red_team, index=False)
    weight_results.to_csv(out_weights, index=False)
    report = {
        "config": asdict(cfg),
        "cache_meta": {k: meta.get(k) for k in ["cached_at", "period", "rows", "columns"]},
        "wrc": wrc,
        "spa": spa_res,
        "pbo_proxy": pbo,
        "all_candidate_wrc": wrc_all,
        "all_candidate_spa": spa_all,
        "all_candidate_pbo_proxy": pbo_all,
        "promotion_objectives": promotion_objectives,
        "summary": summary.to_dict("records"),
        "daily_summary": daily_summary.to_dict("records") if not daily_summary.empty else [],
        "holdout_summary": holdout_summary.to_dict("records") if not holdout_summary.empty else [],
        "red_team": red_team.to_dict("records") if not red_team.empty else [],
        "artifacts": {
            "summary_csv": str(out_csv),
            "windows_csv": str(out_detail),
            "daily_oos_csv": str(out_daily),
            "daily_summary_csv": str(out_daily_summary),
            "holdout_summary_csv": str(out_holdout_summary),
            "red_team_csv": str(out_red_team),
            "weights_csv": str(out_weights),
        },
        "note": "Research-only. WRC/SPA use daily OOS active returns; production promotion still requires frozen final holdout and broader universe validation.",
    }
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(summary.round(6).to_string(index=False))
    print("\nArtifacts:")
    print(out_csv)
    print(out_detail)
    print(out_daily)
    print(out_daily_summary)
    print(out_holdout_summary)
    print(out_red_team)
    print(out_weights)
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

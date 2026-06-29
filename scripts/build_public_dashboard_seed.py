from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".quant_cache" / "cloud"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "public_artifacts"
DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "research_artifacts"
PUBLIC_DASHBOARD_UI_SCHEMA_VERSION = "2026.06.29-institutional-terminal-full-artifact-v18"
PUBLIC_DASHBOARD_BUILD_ID = "2026.06.29-bloomberg-zero-cost-terminal-v18"

BLOCKED_KEY_FRAGMENTS = (
    "side_sleeve",
    "side_boom",
    "private_side",
    "mnpi",
)
BLOCKED_EXACT_KEYS = {
    "side_pelt_regime_segments",
    "side_pelt_change_points",
    "side_pelt_timeline",
}
PRIVATE_LABEL_REPLACEMENTS = {
    "Private Side Alpha": "Research strategy",
    "private side alpha": "research strategy",
    "Side Boom": "Research strategy",
    "side boom": "research strategy",
}


def _len_rows(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0


def _publication_completeness(payload: dict[str, Any]) -> dict[str, Any]:
    """Minimum public-seed contract for the hosted analytical terminal.

    The hosted app may run without Supabase or public APIs during first paint.
    This gate prevents a thin market snapshot from masquerading as the
    institutional dashboard that preserves the research evidence.
    """
    market = payload.get("market_intelligence", {}) if isinstance(payload, dict) else {}
    strategy = payload.get("strategy_lab", {}) if isinstance(payload, dict) else {}
    allocation = payload.get("allocation", {}) if isinstance(payload, dict) else {}
    fixed_income = payload.get("fixed_income_intelligence", {}) if isinstance(payload, dict) else {}
    checks = {
        "xcdr_weights": _len_rows(strategy.get("weights")) >= 20,
        "xcdr_oos_prices": _len_rows(strategy.get("oos_price_paths")) >= 40,
        "portfolio_weights": _len_rows(allocation.get("recommended_portfolio")) >= 20,
        "global_yield_curves": _len_rows(market.get("global_yield_curves")) >= 10,
        "rate_history": _len_rows(market.get("global_rate_history")) >= 500,
        "interbank_reference_rates": _len_rows(market.get("interbank_reference_rates")) >= 100,
        "carry_trade_screen": _len_rows(market.get("carry_trade_suggestions")) >= 5,
        "latent_sentiment": _len_rows(market.get("sentiment_timeline")) >= 100,
        "geopolitical_articles": _len_rows(market.get("geopolitical_articles")) >= 10,
        "sector_fundamentals": _len_rows(market.get("fundamentals_snapshot"))
        >= 10
        or _len_rows(payload.get("tables", {}).get("fundamentals") if isinstance(payload.get("tables"), dict) else []) >= 10,
        "fixed_income_country_metrics": _len_rows(fixed_income.get("country_metrics")) >= 5,
    }
    ready = sum(bool(v) for v in checks.values())
    return {
        "ready": ready,
        "total": len(checks),
        "ratio": ready / max(1, len(checks)),
        "checks": checks,
        "blocking_missing": [key for key, ok in checks.items() if not ok],
    }


def _is_nullish(value: Any) -> bool:
    try:
        if value is None:
            return True
        if isinstance(value, float) and not math.isfinite(value):
            return True
        return bool(pd.isna(value))
    except Exception:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_string(value: str) -> str:
    out = value
    for old, new in PRIVATE_LABEL_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def _is_blocked_key(key: str) -> bool:
    low = key.lower()
    if key in BLOCKED_EXACT_KEYS:
        return True
    return any(fragment in low for fragment in BLOCKED_KEY_FRAGMENTS)


def sanitize_public_artifact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            if _is_blocked_key(str(key)):
                continue
            clean[key] = sanitize_public_artifact(child)
        return clean
    if isinstance(value, list):
        return [sanitize_public_artifact(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    if _is_nullish(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return sanitize_public_artifact(frame.where(pd.notna(frame), None).to_dict("records"))


def _read_research_csv(research_dir: Path, name: str) -> pd.DataFrame:
    path = research_dir / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, math.nan)
    out = pd.to_numeric(numerator, errors="coerce") / den
    return out.replace([math.inf, -math.inf], math.nan)


def _latest_cached_yfinance_fundamentals(cache_root: Path | None = None) -> pd.DataFrame:
    cache_root = cache_root or (PROJECT_ROOT / ".quant_cache" / "fundamentals_yfinance")
    if not cache_root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_root.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty and "Ticker" in frame.columns:
            frame["_Cache_File"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    for col in ("Availability_Date", "Period_End"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    sort_cols = [c for c in ("Availability_Date", "Period_End") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.drop_duplicates("Ticker", keep="last").reset_index(drop=True)


def _fundamental_ratio_snapshot(tickers: list[str]) -> list[dict[str, Any]]:
    """Build a public-cache fundamental snapshot for selected research tickers.

    This is not a vendor-grade PIT dataset. It uses the latest cached public
    yfinance/statement approximation, exposes availability dates, and computes
    sector-relative robust z-scores so the UI can preserve the fundamental
    research surface without pretending to have Bloomberg/FactSet data.
    """
    selected = {str(t).upper().strip() for t in tickers if str(t).strip()}
    if not selected:
        return []
    universe = _latest_cached_yfinance_fundamentals()
    if universe.empty:
        return []

    df = universe.copy()
    market_cap = _safe_num(df, "_quote_market_cap")
    enterprise_value = _safe_num(df, "_quote_enterprise_value")
    debt = _safe_num(df, "_debt").combine_first(_safe_num(df, "_quote_debt"))
    cash = _safe_num(df, "_cash").combine_first(_safe_num(df, "_quote_cash"))
    revenue = _safe_num(df, "_revenue").combine_first(_safe_num(df, "_quote_revenue"))
    ebit = _safe_num(df, "_ebit")
    ebitda = _safe_num(df, "_ebitda").combine_first(_safe_num(df, "_quote_ebitda"))
    net_income = _safe_num(df, "_net_income")
    assets = _safe_num(df, "_assets")
    liabilities = _safe_num(df, "_liabilities")
    equity = _safe_num(df, "_equity")
    shares = _safe_num(df, "_shares")
    cfo = _safe_num(df, "_cfo").combine_first(_safe_num(df, "_quote_cfo"))
    capex = _safe_num(df, "_capex")
    fcf = _safe_num(df, "_fcf_statement").combine_first(_safe_num(df, "_quote_fcf")).combine_first(cfo + capex)
    interest = _safe_num(df, "_interest")
    retained = _safe_num(df, "_retained")
    working_capital = _safe_num(df, "_working_capital").combine_first(
        _safe_num(df, "_current_assets") - _safe_num(df, "_current_liabilities")
    )
    nopat = _safe_num(df, "_nopat")
    if nopat.isna().all():
        tax = _safe_num(df, "_tax")
        pretax = _safe_num(df, "_pretax")
        tax_rate = _safe_div(tax, pretax).clip(lower=0, upper=0.45).fillna(0.21)
        nopat = ebit * (1.0 - tax_rate)
    invested_capital = (equity + debt - cash).where(lambda x: x > 0, assets - liabilities)

    df["P/E"] = _safe_num(df, "_quote_pe").combine_first(_safe_div(market_cap, net_income))
    df["P/B"] = _safe_num(df, "_quote_price_to_book").combine_first(_safe_div(market_cap, equity))
    df["EPS"] = _safe_num(df, "_quote_eps").combine_first(_safe_div(net_income, shares))
    df["Solvency_Assets_Liabilities"] = _safe_div(assets, liabilities)
    df["ROE"] = _safe_num(df, "_quote_roe").combine_first(_safe_div(net_income, equity))
    df["ROIC"] = _safe_div(nopat, invested_capital)
    df["EV_EBITDA"] = _safe_div(enterprise_value.combine_first(market_cap + debt - cash), ebitda)
    df["FCF_Yield"] = _safe_div(fcf, market_cap)
    df["Net_Debt_EBITDA"] = _safe_div(debt - cash, ebitda)
    df["Piotroski_F_Score"] = _safe_num(df, "Piotroski")
    df["Asset_Turnover"] = _safe_div(revenue, assets)
    df["Altman_Z"] = (
        1.2 * _safe_div(working_capital, assets)
        + 1.4 * _safe_div(retained, assets)
        + 3.3 * _safe_div(ebit, assets)
        + 0.6 * _safe_div(equity, liabilities)
        + 1.0 * _safe_div(revenue, assets)
    )
    df["Interest_Coverage"] = _safe_div(ebit, interest.abs())
    df["Retention_Ratio"] = _safe_div(net_income - _safe_num(df, "_dividends").abs(), net_income).clip(lower=-2, upper=2)
    df["Earnings_Yield"] = _safe_div(net_income, market_cap).combine_first(_safe_div(pd.Series(1.0, index=df.index), df["P/E"]))

    ratio_cols = [
        "P/E",
        "P/B",
        "EPS",
        "Solvency_Assets_Liabilities",
        "ROE",
        "ROIC",
        "EV_EBITDA",
        "FCF_Yield",
        "Net_Debt_EBITDA",
        "Piotroski_F_Score",
        "Asset_Turnover",
        "Altman_Z",
        "Interest_Coverage",
        "Retention_Ratio",
        "Earnings_Yield",
    ]
    lower_better = {"P/E", "P/B", "EV_EBITDA", "Net_Debt_EBITDA"}
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    z_cols: list[str] = []
    for col in ratio_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        med = values.groupby(df["Sector"].astype(str)).transform("median")
        mad = values.groupby(df["Sector"].astype(str)).transform(lambda x: (x - x.median()).abs().median())
        z = (values - med) / (1.4826 * mad.replace(0, math.nan))
        if col in lower_better:
            z = -z
        z_col = f"Sector_Z_{col.replace('/', '_').replace(' ', '_')}"
        df[z_col] = z.clip(lower=-5, upper=5)
        z_cols.append(z_col)
    df["Sector_Robust_Z_Composite"] = df[z_cols].mean(axis=1, skipna=True)
    df["Fundamental_Ratio_Coverage"] = df[ratio_cols].notna().sum(axis=1)
    df["PIT_Data_Class"] = "Public cache PIT approximation"
    df["PIT_Confidence"] = (df["Fundamental_Ratio_Coverage"] / max(1, len(ratio_cols))).clip(0, 1)

    selected_df = df[df["Ticker"].isin(selected)].copy()
    if selected_df.empty:
        return []
    keep_cols = [
        "Ticker",
        "Sector",
        "Country",
        "Fundamental_Source",
        "Period_End",
        "Availability_Date",
        "PIT_Data_Class",
        "PIT_Confidence",
        "Fundamental_Ratio_Coverage",
        "Sector_Robust_Z_Composite",
        *ratio_cols,
        *z_cols,
    ]
    for col in ("Period_End", "Availability_Date"):
        if col in selected_df.columns:
            selected_df[col] = pd.to_datetime(selected_df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return _records(selected_df[[c for c in keep_cols if c in selected_df.columns]].sort_values("Ticker"))


def _select_objective(summary: pd.DataFrame) -> str | None:
    if not isinstance(summary, pd.DataFrame) or summary.empty or "objective" not in summary.columns:
        return None
    preferred = (
        "enhanced_growth_anchor_dd_budget_policy",
        "state_optimized_xcdr_v3_policy",
        "downside_preserving_growth_policy",
    )
    objectives = summary["objective"].dropna().astype(str).tolist()
    for objective in preferred:
        if objective in objectives:
            return objective
    return objectives[0] if objectives else None


def _drawdown_from_return(returns: pd.Series) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    nav = (1.0 + r).cumprod()
    if nav.empty:
        return pd.Series(dtype=float)
    return nav / nav.cummax() - 1.0


def _build_xcdr_strategy_lab(research_dir: Path) -> dict[str, Any]:
    summary = _read_research_csv(research_dir, "xcdr_v3_parallel_research_summary.csv")
    daily_summary = _read_research_csv(research_dir, "xcdr_v3_parallel_research_daily_summary.csv")
    daily = _read_research_csv(research_dir, "xcdr_v3_parallel_research_daily_oos.csv")
    weights = _read_research_csv(research_dir, "xcdr_v3_parallel_research_weights.csv")
    windows = _read_research_csv(research_dir, "xcdr_v3_parallel_research_windows.csv")
    red_team = _read_research_csv(research_dir, "xcdr_v3_parallel_research_red_team.csv")
    report_path = research_dir / "xcdr_v3_parallel_research_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}

    objective = _select_objective(summary)
    if objective is None:
        return {}
    selected_summary = summary[summary["objective"].astype(str).eq(objective)].copy()
    selected_daily = (
        daily_summary[daily_summary["objective"].astype(str).eq(objective)].copy()
        if "objective" in daily_summary.columns
        else pd.DataFrame()
    )
    selected_daily_path = (
        daily[daily["objective"].astype(str).eq(objective)].copy() if "objective" in daily.columns else daily.copy()
    )
    selected_weights = (
        weights[weights["objective"].astype(str).eq(objective)].copy()
        if "objective" in weights.columns
        else weights.copy()
    )
    xi = "ξ"
    for frame in (selected_weights, selected_daily_path, selected_summary):
        if "xi" in frame.columns and not frame["xi"].dropna().empty:
            xi = str(frame["xi"].dropna().astype(str).iloc[0])
            break

    price_paths = pd.DataFrame()
    drawdowns = pd.DataFrame()
    if not selected_daily_path.empty and {"date", "portfolio_return", "xi_return"}.issubset(selected_daily_path.columns):
        path = selected_daily_path.copy()
        path["Date"] = pd.to_datetime(path["date"], errors="coerce")
        path = path.dropna(subset=["Date"]).sort_values("Date")
        portfolio_nav = (1.0 + pd.to_numeric(path["portfolio_return"], errors="coerce").fillna(0.0)).cumprod() * 100.0
        xi_nav = (1.0 + pd.to_numeric(path["xi_return"], errors="coerce").fillna(0.0)).cumprod() * 100.0
        price_paths = pd.DataFrame(
            {
                "Date": path["Date"].dt.strftime("%Y-%m-%d"),
                "XCDR/XODR synthetic strategy price": portfolio_nav,
                f"{xi} benchmark observed price": xi_nav,
            }
        )
        drawdowns = pd.DataFrame(
            {
                "Date": path["Date"].dt.strftime("%Y-%m-%d"),
                "XCDR/XODR strategy drawdown": _drawdown_from_return(path["portfolio_return"]),
                f"{xi} benchmark drawdown": _drawdown_from_return(path["xi_return"]),
            }
        )

    observation_days = len(selected_daily_path) if not selected_daily_path.empty else 0
    research_pass = False
    if not selected_summary.empty and "research_gate_pass" in selected_summary.columns:
        research_pass = bool(str(selected_summary["research_gate_pass"].iloc[0]).lower() == "true")
    holdout_pass = False
    if not selected_summary.empty and "holdout_gate_pass" in selected_summary.columns:
        holdout_pass = bool(str(selected_summary["holdout_gate_pass"].iloc[0]).lower() == "true")

    validation_rows = pd.DataFrame(
        [
            {
                "Gate": "Research gate",
                "Observed": research_pass,
                "Threshold": "WRC/SPA/PBO, ICIR and downside preservation",
                "Pass": research_pass,
            },
            {
                "Gate": "Holdout gate",
                "Observed": holdout_pass,
                "Threshold": "Frozen final holdout",
                "Pass": holdout_pass,
            },
            {
                "Gate": "Minimum windows",
                "Observed": selected_summary["windows"].iloc[0] if "windows" in selected_summary.columns and not selected_summary.empty else None,
                "Threshold": selected_summary["Promotion_Min_Windows"].iloc[0] if "Promotion_Min_Windows" in selected_summary.columns and not selected_summary.empty else 12,
                "Pass": research_pass,
            },
        ]
    )

    return {
        "generation": "public_seed_repo_xcdr_v3",
        "status": "PROMOTED_RESEARCH" if research_pass else "RESEARCH_ONLY",
        "benchmark_xi": xi,
        "observation_days": observation_days,
        "frozen_candidate": objective,
        "summary": _records(summary),
        "daily_summary": _records(daily_summary),
        "oos_summary": _records(selected_daily if not selected_daily.empty else selected_summary),
        "oos_price_paths": _records(price_paths),
        "oos_drawdowns": _records(drawdowns),
        "walk_forward_windows": _records(windows),
        "weights": _records(selected_weights),
        "validation": _records(validation_rows),
        "red_team": _records(red_team),
        "report": sanitize_public_artifact(report),
    }


def _strategy_lab_has_evidence(strategy_lab: Any) -> bool:
    if not isinstance(strategy_lab, dict):
        return False
    for key in ("oos_price_paths", "oos_drawdowns", "weights", "validation", "summary"):
        value = strategy_lab.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
        if isinstance(value, dict) and bool(value):
            return True
    return False


def _inject_xcdr_research_if_missing(artifact: dict[str, Any], research_dir: Path) -> dict[str, Any]:
    payload = artifact.get("dashboard_payload")
    if not isinstance(payload, dict):
        return artifact
    existing_strategy = payload.get("strategy_lab", {})
    if _strategy_lab_has_evidence(existing_strategy):
        return artifact
    strategy_lab = _build_xcdr_strategy_lab(research_dir)
    if not strategy_lab:
        return artifact

    payload["strategy_lab"] = strategy_lab
    allocation = payload.get("allocation")
    if not isinstance(allocation, dict):
        allocation = {}
    if not allocation.get("recommended_portfolio"):
        allocation["recommended_portfolio"] = strategy_lab.get("weights", [])
    if not allocation.get("weights"):
        allocation["weights"] = strategy_lab.get("weights", [])
    payload["allocation"] = allocation

    charts = payload.get("charts")
    if not isinstance(charts, dict):
        charts = {}
    if not charts.get("price_paths"):
        charts["price_paths"] = strategy_lab.get("oos_price_paths", [])
    if not charts.get("drawdowns"):
        charts["drawdowns"] = strategy_lab.get("oos_drawdowns", [])
    payload["charts"] = charts

    research = payload.get("research")
    if not isinstance(research, dict):
        research = {}
    if not research.get("optimization_grid"):
        research["optimization_grid"] = strategy_lab.get("summary", [])
    if not research.get("overfit_diagnostics"):
        research["overfit_diagnostics"] = strategy_lab.get("validation", [])
    payload["research"] = research

    status = payload.get("status")
    if isinstance(status, dict):
        status.setdefault(
            "promotion",
            [
                {
                    "Status": strategy_lab.get("status", "RESEARCH_ONLY"),
                    "Objective": strategy_lab.get("frozen_candidate"),
                    "Benchmark_Xi": strategy_lab.get("benchmark_xi"),
                    "Reason": "Repository XCDR research artifact injected into public seed fallback.",
                }
            ],
        )
    return artifact


def _stamp_public_seed(artifact: dict[str, Any], *, scope: str, research_dir: Path) -> dict[str, Any]:
    clean = sanitize_public_artifact(artifact)
    if not isinstance(clean, dict):
        return {}
    clean = _inject_xcdr_research_if_missing(clean, research_dir)
    clean["scope"] = scope
    clean["public_seed"] = True
    clean["seed_created_at"] = datetime.now(UTC).isoformat()
    payload = clean.get("dashboard_payload")
    if isinstance(payload, dict):
        payload["schema_version"] = PUBLIC_DASHBOARD_UI_SCHEMA_VERSION
        payload["app_build_id"] = PUBLIC_DASHBOARD_BUILD_ID
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            contract = {}
        contract.update(
            {
                "schema_version": PUBLIC_DASHBOARD_UI_SCHEMA_VERSION,
                "app_build_id": PUBLIC_DASHBOARD_BUILD_ID,
                "public_seed": True,
                "seed_scope": scope,
                "seed_disclaimer": (
                    "Sanitized public-data dashboard seed. Supabase artifacts remain the production source of truth."
                ),
            }
        )
        payload["contract"] = contract
        allocation = payload.get("allocation")
        if isinstance(allocation, dict):
            allocation.pop("side_sleeve", None)
        strategy = payload.get("strategy_lab") if isinstance(payload.get("strategy_lab"), dict) else {}
        strategy_weights = strategy.get("weights") if isinstance(strategy, dict) else []
        allocation_weights = allocation.get("recommended_portfolio") if isinstance(allocation, dict) else []
        tickers: list[str] = []
        for row in strategy_weights if isinstance(strategy_weights, list) else []:
            if isinstance(row, dict) and row.get("ticker"):
                tickers.append(str(row.get("ticker")))
        for row in allocation_weights if isinstance(allocation_weights, list) else []:
            if isinstance(row, dict) and (row.get("Ticker") or row.get("ticker")):
                tickers.append(str(row.get("Ticker") or row.get("ticker")))
        tables = payload.get("tables")
        if not isinstance(tables, dict):
            tables = {}
        market = payload.get("market_intelligence")
        if not isinstance(market, dict):
            market = {}
        existing_fundamentals = tables.get("fundamentals") or market.get("fundamentals_snapshot") or []
        if not existing_fundamentals:
            fundamentals = _fundamental_ratio_snapshot(tickers)
            if fundamentals:
                tables["fundamentals"] = fundamentals
                market["fundamentals_snapshot"] = fundamentals
                payload["tables"] = tables
                payload["market_intelligence"] = market
        research = payload.get("research")
        if isinstance(research, dict):
            for key in list(research):
                if _is_blocked_key(str(key)):
                    research.pop(key, None)
        payload["publication_completeness"] = _publication_completeness(payload)
    return clean


def write_seed(source_path: Path, target_path: Path, *, scope: str, research_dir: Path) -> int:
    artifact = _read_json(source_path)
    if not artifact:
        raise FileNotFoundError(f"Missing source artifact: {source_path}")
    clean = _stamp_public_seed(artifact, scope=scope, research_dir=research_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with gzip.open(target_path, "wb", compresslevel=9) as fh:
        fh.write(encoded)
    return target_path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sanitized public dashboard seed artifacts.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    args = parser.parse_args()

    outputs = {
        "full_analysis": (
            args.source_dir / "latest_full_analysis_payload.json",
            args.target_dir / "latest_full_dashboard_payload.seed.json.gz",
        ),
        "daily_snapshot": (
            args.source_dir / "latest_daily_snapshot_payload.json",
            args.target_dir / "latest_daily_dashboard_payload.seed.json.gz",
        ),
    }
    for scope, (source, target) in outputs.items():
        size = write_seed(source, target, scope=scope, research_dir=args.research_dir)
        print(f"{scope}: {target} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

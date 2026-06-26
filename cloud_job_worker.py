from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from quant_core.contracts import EvidenceScope, PortfolioPositionV2, PortfolioRunV2
from quant_core.execution import approval_status, build_order_intent, evaluate_pretrade
from quant_stockpicker_core import RunConfig, run_pipeline
from supabase_store import (
    get_supabase_client,
    save_order_intent,
    save_pretrade_decision,
    save_run_to_supabase,
    save_versioned_user_portfolio,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())[:420]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _normalize_tickers(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip().upper().replace(".", "-")
            for value in values
            if str(value).strip()
        )
    )


def build_job_run_config(config: dict[str, Any]) -> RunConfig:
    """Map the public API contract into a frozen, auditable engine config."""
    tickers = _normalize_tickers(config.get("tickers"))
    if len(tickers) < 2:
        raise ValueError("An optimization job requires at least two distinct tickers")
    style = str(config.get("filter_style") or "factor").strip().lower()
    if style not in {"growth", "value", "quality", "factor", "custom"}:
        raise ValueError(f"Unsupported fundamental style: {style}")
    period = str(config.get("base_period") or "3y").strip().lower()
    if period not in {"3y", "5y", "10y"}:
        raise ValueError(f"Unsupported base period: {period}")
    benchmark = str(config.get("benchmark_ticker") or "SPY").strip().upper().replace(".", "-")
    compute_mode = str(os.getenv("QPK_JOB_COMPUTE_MODE", "rigorous")).strip().lower()
    workers = max(1, min(int(os.getenv("QPK_JOB_WORKERS", "4")), 8))
    top_n = max(5, min(int(config.get("top_n") or 12), min(25, len(tickers))))
    preselect_n = max(top_n, min(int(config.get("preselect_n") or 30), len(tickers)))
    return RunConfig(
        tickers=tickers,
        benchmark_ticker=benchmark,
        price_period=period,
        fundamental_style=style,
        top_n=top_n,
        preselect_n=preselect_n,
        weight_objective="xcdr_v3",
        compute_mode=compute_mode,
        max_workers=workers,
        use_persistent_cache=True,
        use_side_boom_portfolio=False,
        use_kaizen_bandit=False,
        use_gdelt=False,
        benchmark_auto_select=True,
        portfolio_notional=float(config.get("initial_capital") or 100_000.0),
        investor_initial_capital=float(config.get("initial_capital") or 100_000.0),
        investor_monthly_contribution=float(config.get("monthly_contribution") or 0.0),
        investor_risk_aversion_score=float(config.get("risk_aversion") or 5.0),
        investor_max_drawdown=float(config.get("max_drawdown") or 0.20),
        investor_liquidity_need=str(config.get("liquidity_need") or "Medium"),
        investor_base_currency=str(config.get("base_currency") or "USD").upper(),
        sec_user_agent=str(os.getenv("SEC_USER_AGENT", "QuantPortfolioKaizen/1.0 research@example.com")),
    )


def _first_record(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value.iloc[0].to_dict()
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return dict(value[0])
    if isinstance(value, dict):
        return value
    return {}


def _status_value(container: Any, keys: tuple[str, ...], default: str) -> str:
    safe = container if isinstance(container, dict) else {}
    for key in keys:
        value = safe.get(key)
        if value not in (None, ""):
            return str(value).strip().lower()
    summary = _first_record(safe.get("summary"))
    for key in keys:
        value = summary.get(key)
        if value not in (None, ""):
            return str(value).strip().lower()
    return default


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_portfolio_contract(
    results: dict[str, Any],
    config: RunConfig,
    *,
    run_id: str,
    user_id: str,
    portfolio_name: str,
) -> PortfolioRunV2:
    portfolio = results.get("portfolio")
    if not isinstance(portfolio, pd.DataFrame) or portfolio.empty:
        raise ValueError("The optimization completed without an allocatable portfolio")
    frame = portfolio.copy()
    frame["Weight"] = pd.to_numeric(frame.get("Weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
    frame = frame[frame["Weight"] > 1e-10].copy()
    total = float(frame["Weight"].sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Portfolio weights are not finite and positive")
    frame["Weight"] /= total

    prices = results.get("prices")
    latest_prices = pd.Series(dtype=float)
    as_of = _utc_now()
    if isinstance(prices, pd.DataFrame) and not prices.empty:
        latest_prices = pd.to_numeric(prices.iloc[-1], errors="coerce")
        parsed_as_of = pd.Timestamp(prices.index[-1])
        as_of = parsed_as_of.tz_localize("UTC").to_pydatetime() if parsed_as_of.tzinfo is None else parsed_as_of.tz_convert("UTC").to_pydatetime()

    positions: list[PortfolioPositionV2] = []
    for row in frame.to_dict(orient="records"):
        ticker = str(row.get("Ticker") or "").upper()
        reference_price = float(latest_prices.get(ticker, np.nan))
        positions.append(
            PortfolioPositionV2(
                ticker=ticker,
                sector=str(row.get("Sector")) if pd.notna(row.get("Sector")) else None,
                country=str(row.get("Country")) if pd.notna(row.get("Country")) else None,
                target_weight=float(row["Weight"]),
                reference_price=reference_price if np.isfinite(reference_price) and reference_price > 0 else None,
                adv_usd=float(row.get("Dollar_Volume_63"))
                if pd.notna(row.get("Dollar_Volume_63")) and float(row.get("Dollar_Volume_63")) >= 0
                else None,
                composite_score=float(row.get("Composite_Score"))
                if pd.notna(row.get("Composite_Score"))
                else None,
                pit_confidence=float(np.clip(row.get("PIT_Confidence"), 0.0, 1.0))
                if pd.notna(row.get("PIT_Confidence"))
                else None,
            )
        )

    suitability_raw = _status_value(
        results.get("suitability_gate"),
        ("status", "Gate_Status", "Suitability_Status"),
        "watchlist",
    )
    suitability = "approved" if suitability_raw in {"approved", "pass", "passed"} else "blocked" if suitability_raw in {"blocked", "rejected", "failed"} else "watchlist"
    promotion_raw = _status_value(
        results.get("promotion_gate"),
        ("promotion_status", "Promotion_Status", "status"),
        "research_only",
    )
    if promotion_raw in {"promoted", "approved"}:
        promotion = "promoted"
    elif promotion_raw in {"rejected", "blocked", "failed"}:
        promotion = "rejected"
    elif promotion_raw == "watchlist":
        promotion = "watchlist"
    else:
        promotion = "research_only"

    equity_curve = results.get("equity_curve")
    evidence_scope = EvidenceScope.OUT_OF_SAMPLE if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty else EvidenceScope.LIVE_SNAPSHOT
    if evidence_scope == EvidenceScope.LIVE_SNAPSHOT and promotion == "promoted":
        promotion = "research_only"

    registry = _first_record(results.get("model_registry"))
    config_dict = asdict(config)
    config_hash = str(registry.get("config_hash") or _hash_json(config_dict))
    data_hash = str(registry.get("data_hash") or _hash_json({"as_of": as_of.isoformat(), "tickers": config.tickers}))
    stress_set = _normalize_tickers(registry.get("stress_set_omega") or ())
    return PortfolioRunV2(
        run_id=UUID(run_id),
        user_id=UUID(user_id),
        as_of=as_of,
        portfolio_name=portfolio_name,
        base_currency=config.investor_base_currency,
        objective="xcdr_v3",
        benchmark_xi=config.benchmark_ticker,
        stress_set_omega=stress_set,
        evidence_scope=evidence_scope,
        positions=tuple(positions),
        config_hash=config_hash,
        data_hash=data_hash,
        model_version=str(registry.get("model_version") or registry.get("code_version") or "unregistered"),
        suitability_status=suitability,
        promotion_status=promotion,
    )


def claim_next_job(client, *, job_id: str | None = None, allowed_types: tuple[str, ...] = ("optimization", "paper_pretrade")) -> dict[str, Any]:
    query = client.table("jobs").select("job_id,user_id,job_type,status,config,created_at").eq("status", "queued")
    if job_id:
        query = query.eq("job_id", job_id)
    else:
        query = query.in_("job_type", list(allowed_types)).order("created_at", desc=False).limit(10)
    candidates = query.execute().data or []
    for candidate in candidates:
        claimed = (
            client.table("jobs")
            .update({"status": "running", "started_at": _utc_now().isoformat(), "error": None})
            .eq("job_id", candidate["job_id"])
            .eq("status", "queued")
            .select("job_id,user_id,job_type,status,config,created_at")
            .execute()
            .data
            or []
        )
        if claimed:
            return dict(claimed[0])
    return {}


def _complete_job(client, job_id: str, *, run_id: str | None = None) -> None:
    payload: dict[str, Any] = {"status": "completed", "finished_at": _utc_now().isoformat(), "error": None}
    if run_id:
        payload["result_run_id"] = run_id
    client.table("jobs").update(payload).eq("job_id", job_id).eq("status", "running").execute()


def _fail_job(client, job_id: str, exc: Exception) -> None:
    client.table("jobs").update(
        {"status": "failed", "finished_at": _utc_now().isoformat(), "error": _safe_error(exc)}
    ).eq("job_id", job_id).eq("status", "running").execute()


def process_optimization_job(client, job: dict[str, Any]) -> str:
    config_data = dict(job.get("config") or {})
    config = build_job_run_config(config_data)
    results = run_pipeline(config)
    user_id = str(job["user_id"])
    portfolio_name = str(config_data.get("portfolio_name") or f"{config.fundamental_style.title()} XCDR")[:64]
    run_id = save_run_to_supabase(results, config, user_id=user_id, portfolio_name=portfolio_name)
    contract = build_portfolio_contract(
        results,
        config,
        run_id=run_id,
        user_id=user_id,
        portfolio_name=portfolio_name,
    )
    save_versioned_user_portfolio(contract, client=client)
    return run_id


def process_pretrade_job(client, job: dict[str, Any]) -> str:
    config = dict(job.get("config") or {})
    user_id = str(job["user_id"])
    version_id = str(config.get("portfolio_version_id") or "")
    versions = (
        client.table("portfolio_versions")
        .select("version_id,run_id,contract_json,contract_sha256")
        .eq("version_id", version_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not versions:
        raise ValueError("Portfolio version is unavailable to this user")
    version = versions[0]
    portfolio = PortfolioRunV2.model_validate(version["contract_json"])
    order = build_order_intent(
        portfolio,
        user_id=UUID(user_id),
        portfolio_value=float(config.get("portfolio_value")),
        current_weights=dict(config.get("current_weights") or {}),
    )
    age_hours = max(0.0, (_utc_now() - portfolio.as_of.astimezone(UTC)).total_seconds() / 3600.0)
    decision = evaluate_pretrade(
        order,
        portfolio,
        data_age_hours=age_hours,
        artifact_hash_matches=portfolio.sha256() == str(version.get("contract_sha256") or ""),
    )
    final_order = order.model_copy(update={"status": approval_status(decision, human_approved=False)})
    save_order_intent(final_order, portfolio_version_id=version_id, client=client)
    save_pretrade_decision(decision, user_id=user_id, client=client)
    return str(portfolio.run_id)


def process_claimed_job(client, job: dict[str, Any]) -> str:
    job_type = str(job.get("job_type") or "")
    if job_type == "optimization":
        return process_optimization_job(client, job)
    if job_type == "paper_pretrade":
        return process_pretrade_job(client, job)
    raise ValueError(f"Unsupported job type: {job_type}")


def run_worker(*, max_jobs: int = 1, job_id: str | None = None) -> int:
    client = get_supabase_client()
    processed = 0
    while processed < max(1, int(max_jobs)):
        job = claim_next_job(client, job_id=job_id)
        if not job:
            break
        try:
            run_id = process_claimed_job(client, job)
            _complete_job(client, str(job["job_id"]), run_id=run_id)
        except Exception as exc:
            _fail_job(client, str(job["job_id"]), exc)
        processed += 1
        if job_id:
            break
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Process tenant-scoped Quant Portfolio-Kaizen jobs")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    processed = run_worker(max_jobs=args.max_jobs, job_id=args.job_id)
    print(json.dumps({"processed_jobs": processed, "finished_at": _utc_now().isoformat()}))


if __name__ == "__main__":
    main()

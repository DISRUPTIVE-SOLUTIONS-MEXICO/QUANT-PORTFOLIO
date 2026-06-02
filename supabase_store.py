from __future__ import annotations

import json
import os
import hashlib
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import dotenv_values, load_dotenv
from supabase import Client, create_client


ENV_PATH = Path(__file__).resolve().with_name(".env")
RUN_ARTIFACT_DIR = Path(__file__).resolve().with_name(".quant_cache") / "run_artifacts"
logger = logging.getLogger(__name__)


def load_local_env() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=False, encoding="utf-8-sig")
    for key, value in dotenv_values(ENV_PATH, encoding="utf-8-sig").items():
        if key and value is not None and not os.getenv(key):
            os.environ[key] = value
    try:
        import streamlit as st  # type: ignore

        aliases = {
            "SUPABASE_URL": ("SUPABASE_URL", "url"),
            "SUPABASE_SERVICE_ROLE_KEY": ("SUPABASE_SERVICE_ROLE_KEY", "service_key", "service_role_key"),
        }
        for key, candidates in aliases.items():
            if os.getenv(key):
                continue
            value = None
            for candidate in candidates:
                value = st.secrets.get(candidate)
                if value is not None:
                    break
            if value is None and "supabase" in st.secrets:
                for candidate in candidates:
                    value = st.secrets["supabase"].get(candidate)
                    if value is not None:
                        break
            if value is not None:
                os.environ[key] = str(value)
    except Exception as exc:
        logger.debug("Streamlit secrets are unavailable in this runtime: %s", type(exc).__name__)


def _json_safe(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return value.isoformat()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, pd.Series, pd.DataFrame)) else False:
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return [{str(k): _json_safe(v) for k, v in row.items()} for row in clean.to_dict(orient="records")]


def get_supabase_client() -> Client:
    load_local_env()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en .env.")
    return create_client(url, key)


def config_to_json(config) -> dict:
    if is_dataclass(config):
        return _json_safe(asdict(config))
    if hasattr(config, "__dict__"):
        return _json_safe(dict(config.__dict__))
    return _json_safe(config)


def insert_chunk(client: Client, table: str, rows: list[dict], chunk_size: int = 500):
    if not rows:
        return
    for start in range(0, len(rows), chunk_size):
        client.table(table).insert(rows[start : start + chunk_size]).execute()


def _artifact_shape(value: Any):
    if isinstance(value, pd.DataFrame):
        return {"type": "dataframe", "rows": int(value.shape[0]), "columns": int(value.shape[1])}
    if isinstance(value, dict):
        return {str(k): _artifact_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return {"type": "list", "items": len(value)}
    return {"type": type(value).__name__}


def persist_run_artifacts_local(run_id: str, artifacts: dict[str, Any]) -> tuple[Path, str]:
    RUN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(artifacts)
    serial = json.dumps(safe, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(serial.encode("utf-8")).hexdigest()
    path = RUN_ARTIFACT_DIR / f"{run_id}_{digest[:16]}.json"
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path, digest


def save_run_artifacts(client: Client, run_id: str, artifacts: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    """Persist audit bundles locally and, when available, into Supabase run_artifacts.

    The current Supabase schema may not include run_artifacts. In that case this
    function returns a manifest so risk_diagnostics can still audit local storage.
    """
    local_path, digest = persist_run_artifacts_local(run_id, artifacts)
    manifest = {
        "artifact_local_path": str(local_path),
        "artifact_sha256": digest,
        "artifact_shapes": _artifact_shape(artifacts),
        "supabase_run_artifacts": False,
        "supabase_artifact_error": "",
    }
    rows = []
    for name, artifact in artifacts.items():
        row = {
            "run_id": run_id,
            "artifact_name": name,
            "artifact_json": _json_safe(artifact),
            "artifact_sha256": hashlib.sha256(json.dumps(_json_safe(artifact), sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        }
        if user_id:
            row["user_id"] = user_id
        rows.append(row)
    try:
        insert_chunk(client, "run_artifacts", rows, chunk_size=50)
        manifest["supabase_run_artifacts"] = True
    except Exception as exc:
        if user_id:
            try:
                legacy_rows = [{k: v for k, v in row.items() if k != "user_id"} for row in rows]
                insert_chunk(client, "run_artifacts", legacy_rows, chunk_size=50)
                manifest["supabase_run_artifacts"] = True
                manifest["supabase_artifact_error"] = "Inserted without user_id; apply multiuser run_artifacts migration."
            except Exception as legacy_exc:
                manifest["supabase_artifact_error"] = str(legacy_exc)[:500]
        else:
            manifest["supabase_artifact_error"] = str(exc)[:500]
    return manifest


def save_run_to_supabase(results: dict, config, status: str = "completed", user_id: str | None = None) -> str:
    client = get_supabase_client()
    cfg = config_to_json(config)
    model_registry = results.get("model_registry", pd.DataFrame()).copy()
    registry_row = _records(model_registry)[0] if not model_registry.empty else {}

    run_payload = {
        "config": cfg,
        "benchmark_ticker": cfg.get("benchmark_ticker"),
        "price_period": cfg.get("price_period"),
        "status": status,
    }
    extended_payload = dict(run_payload)
    if user_id:
        extended_payload["user_id"] = user_id
    for key in ["run_hash", "code_version", "config_hash", "universe_hash", "data_hash", "objective", "warnings"]:
        if registry_row.get(key) is not None:
            extended_payload[key] = registry_row.get(key)
    for key in ["app_version", "model_version", "schema_version"]:
        if registry_row.get(key) is not None:
            extended_payload[key] = registry_row.get(key)
    try:
        run_resp = client.table("runs").insert(extended_payload).execute()
    except Exception:
        fallback_payload = dict(run_payload)
        if user_id:
            fallback_payload["user_id"] = user_id
        try:
            run_resp = client.table("runs").insert(fallback_payload).execute()
        except Exception:
            run_resp = client.table("runs").insert(run_payload).execute()
    run_data = run_resp.data or []
    if not run_data:
        raise RuntimeError("Supabase no devolvió run_id al insertar en runs.")
    run_id = run_data[0]["run_id"]
    artifact_manifest = save_run_artifacts(
        client,
        str(run_id),
        {
            "dashboard_payload": results.get("dashboard_payload", {}),
            "backtest_path_bundle": results.get("backtest_path_bundle", {}),
            "suitability_gate": results.get("suitability_gate", {}),
            "promotion_gate": results.get("promotion_gate", {}),
            "data_freshness_report": results.get("data_freshness_report", pd.DataFrame()),
        },
        user_id=user_id,
    )

    portfolio = results.get("portfolio", pd.DataFrame()).copy()
    if not portfolio.empty:
        rows = []
        for row in _records(portfolio):
            rows.append(
                {
                    "run_id": run_id,
                    "ticker": row.get("Ticker"),
                    "sector": row.get("Sector"),
                    "country": row.get("Country"),
                    "weight": row.get("Weight"),
                    "composite_score": row.get("Composite_Score"),
                    "optimization_sortino": row.get("Optimization_Sortino"),
                }
            )
        insert_chunk(client, "portfolio_weights", rows)

    curve = results.get("equity_curve", pd.DataFrame()).copy()
    if not curve.empty:
        rows = []
        for row in _records(curve):
            rows.append(
                {
                    "run_id": run_id,
                    "signal_date": row.get("Signal_Date"),
                    "rebalance_date": row.get("Rebalance_Date"),
                    "period_end": row.get("Period_End"),
                    "net_return": row.get("Net_Return"),
                    "gross_return": row.get("Gross_Return"),
                    "benchmark_return": row.get("Benchmark_Return"),
                    "portfolio_equity": row.get("Portfolio_Equity"),
                    "benchmark_equity": row.get("Benchmark_Equity"),
                    "active_equity": row.get("Active_Equity"),
                    "turnover": row.get("Turnover"),
                }
            )
        insert_chunk(client, "backtest_perf", rows)

    summary = results.get("performance_summary", pd.DataFrame()).copy()
    extra_summary_rows = []
    for metric, value in artifact_manifest.items():
        extra_summary_rows.append(
            {
                "Metric": f"artifact_{metric}",
                "Value": json.dumps(_json_safe(value), ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value,
            }
        )
    freshness = results.get("data_freshness_report", pd.DataFrame()).copy()
    if not freshness.empty:
        for row in _records(freshness):
            ns = row.get("Namespace") or row.get("Dataset") or row.get("Source")
            for metric in ["Status", "Age_Hours", "TTL_Hours", "Rows", "Fallback_Used"]:
                if row.get(metric) is not None:
                    extra_summary_rows.append({"Metric": f"freshness_{ns}_{metric}", "Value": row.get(metric)})
    gate = results.get("suitability_gate", {})
    if isinstance(gate, dict):
        extra_summary_rows.append({"Metric": "suitability_gate_status", "Value": gate.get("status")})
        breaches = gate.get("breaches", pd.DataFrame())
        extra_summary_rows.append({"Metric": "suitability_gate_breach_count", "Value": int(len(breaches)) if isinstance(breaches, pd.DataFrame) else 0})
    promo = results.get("promotion_gate", {})
    if isinstance(promo, dict):
        extra_summary_rows.append({"Metric": "promotion_gate_status", "Value": promo.get("promotion_status")})
        extra_summary_rows.append({"Metric": "promotion_validation_score", "Value": promo.get("validation_score")})
    options_summary = results.get("options_summary", pd.DataFrame()).copy()
    if not options_summary.empty:
        for row in _records(options_summary):
            ticker = row.get("Ticker")
            for metric in ["ATM_IV", "Skew_95P_105C", "Put_Call_OpenInterest", "Median_Rel_BidAsk", "Contracts"]:
                if row.get(metric) is not None:
                    extra_summary_rows.append({"Metric": f"options_{ticker}_{metric}", "Value": row.get(metric)})
    validation = results.get("validation_diagnostics", {})
    if isinstance(validation, dict):
        val_summary = validation.get("summary", pd.DataFrame()).copy()
        if not val_summary.empty:
            for row in _records(val_summary):
                extra_summary_rows.append({"Metric": f"validation_{row.get('Metric')}", "Value": row.get("Value")})
    latent = results.get("latent_regime_diagnostics", {})
    if isinstance(latent, dict):
        latent_summary = latent.get("summary", pd.DataFrame()).copy()
        if not latent_summary.empty:
            for row in _records(latent_summary):
                state = row.get("Latent_State_Label")
                for metric in ["Frequency", "Mean_Prob", "Mean_Entropy", "Mean_Hawkish", "Mean_Bullish", "Mean_Curve", "Mean_Credit"]:
                    if row.get(metric) is not None:
                        extra_summary_rows.append({"Metric": f"latent_{state}_{metric}", "Value": row.get(metric)})
        markov = latent.get("markov_forecast", pd.DataFrame()).copy()
        if not markov.empty:
            last = _records(markov.tail(1))[0]
            for metric in ["Markov_State_Persistence", "Markov_Stress_Prob", "Markov_Risk_On_Prob", "Markov_Transition_Entropy", "Markov_Transition_Obs"]:
                if last.get(metric) is not None:
                    extra_summary_rows.append({"Metric": f"markov_latest_{metric}", "Value": last.get(metric)})
    alt = results.get("alternative_data", {})
    if isinstance(alt, dict):
        alt_summary = alt.get("summary", pd.DataFrame()).copy()
        if not alt_summary.empty:
            for row in _records(alt_summary):
                if row.get("Latest") is not None:
                    extra_summary_rows.append({"Metric": f"alt_{row.get('Signal')}_Latest", "Value": row.get("Latest")})
                if row.get("Z_252") is not None:
                    extra_summary_rows.append({"Metric": f"alt_{row.get('Signal')}_Z_252", "Value": row.get("Z_252")})
    portfolio_for_risk = results.get("portfolio", pd.DataFrame()).copy()
    if not portfolio_for_risk.empty:
        for metric, col in [
            ("bayesian_alpha_mean_avg", "Bayesian_Alpha_Mean"),
            ("bayesian_prob_alpha_positive_avg", "Prob_Alpha_Positive"),
            ("bayesian_posterior_confidence_avg", "Bayesian_Posterior_Confidence"),
            ("bayesian_alpha_ci_width_avg", "Alpha_CI_95_Width"),
            ("hierarchical_sector_reliability_avg", "Hierarchical_Sector_Reliability"),
        ]:
            if col in portfolio_for_risk:
                value = pd.to_numeric(portfolio_for_risk[col], errors="coerce").mean()
                if pd.notna(value):
                    extra_summary_rows.append({"Metric": metric, "Value": value})
    ret_diag = results.get("return_diagnostics", {})
    if isinstance(ret_diag, dict):
        factor_risk = ret_diag.get("factor_model_factor_risk", pd.DataFrame()).copy()
        if not factor_risk.empty:
            for row in _records(factor_risk):
                if row.get("Pct_Total_Variance") is not None:
                    extra_summary_rows.append({"Metric": f"factor_risk_{row.get('Name')}_Pct_Total_Variance", "Value": row.get("Pct_Total_Variance")})
    regime_perf = results.get("regime_performance", pd.DataFrame()).copy()
    if not regime_perf.empty:
        for row in _records(regime_perf):
            state = row.get("State")
            if row.get("Sortino_Approx") is not None:
                extra_summary_rows.append({"Metric": f"regime_{state}_Sortino_Approx", "Value": row.get("Sortino_Approx")})
            if row.get("Total_Return") is not None:
                extra_summary_rows.append({"Metric": f"regime_{state}_Total_Return", "Value": row.get("Total_Return")})
    stress = results.get("stress_tests", pd.DataFrame()).copy()
    if not stress.empty:
        for row in _records(stress):
            if row.get("Estimated_Portfolio_Return") is not None:
                extra_summary_rows.append({"Metric": f"stress_{row.get('Scenario')}_Estimated_Return", "Value": row.get("Estimated_Portfolio_Return")})
    attr = results.get("oos_factor_attribution", pd.DataFrame()).copy()
    if not attr.empty and {"Component", "Contribution"}.issubset(attr.columns):
        for comp, value in attr.groupby("Component")["Contribution"].sum().items():
            extra_summary_rows.append({"Metric": f"oos_attr_{comp}_Total", "Value": value})
    ledger = results.get("capital_ledger", pd.DataFrame()).copy()
    if not ledger.empty:
        for metric, col in [("ledger_final_capital", "End_Capital"), ("ledger_max_drawdown", "Drawdown"), ("ledger_total_net_pnl", "Net_PnL")]:
            if col in ledger:
                value = ledger[col].iloc[-1] if col == "End_Capital" else (ledger[col].min() if col == "Drawdown" else ledger[col].sum())
                extra_summary_rows.append({"Metric": metric, "Value": value})
    if not summary.empty:
        rows = []
        for row in _records(summary):
            rows.append({"run_id": run_id, "metric": row.get("Metric"), "value": row.get("Value")})
        for row in extra_summary_rows:
            rows.append({"run_id": run_id, "metric": row.get("Metric"), "value": row.get("Value")})
        if registry_row:
            for metric in ["run_hash", "code_version", "app_version", "model_version", "schema_version", "config_hash", "universe_hash", "data_hash", "objective", "benchmark"]:
                if registry_row.get(metric) is not None:
                    rows.append({"run_id": run_id, "metric": f"registry_{metric}", "value": registry_row.get(metric)})
            for metric, value in (registry_row.get("data_quality") or {}).items():
                rows.append({"run_id": run_id, "metric": f"data_quality_{metric}", "value": value})
            for metric, value in (registry_row.get("timings") or {}).items():
                rows.append({"run_id": run_id, "metric": f"timing_{metric}", "value": value})
        insert_chunk(client, "risk_diagnostics", rows)
    elif extra_summary_rows or registry_row:
        rows = [{"run_id": run_id, "metric": row.get("Metric"), "value": row.get("Value")} for row in extra_summary_rows]
        if registry_row:
            for metric in ["run_hash", "code_version", "app_version", "model_version", "schema_version", "config_hash", "universe_hash", "data_hash", "objective", "benchmark"]:
                if registry_row.get(metric) is not None:
                    rows.append({"run_id": run_id, "metric": f"registry_{metric}", "value": registry_row.get(metric)})
        insert_chunk(client, "risk_diagnostics", rows)

    variance = pd.DataFrame()
    if isinstance(ret_diag, dict):
        variance = ret_diag.get("variance_model_selection", pd.DataFrame()).copy()
    if not variance.empty:
        rows = []
        for row in _records(variance):
            rows.append(
                {
                    "run_id": run_id,
                    "series": row.get("Series"),
                    "model": row.get("Model"),
                    "log_likelihood": row.get("LogLikelihood"),
                    "aic": row.get("AIC"),
                    "bic": row.get("BIC"),
                    "next_ann_vol": row.get("Next_Ann_Vol"),
                    "best_aic": row.get("Best_AIC"),
                    "best_bic": row.get("Best_BIC"),
                    "params": row.get("Params"),
                }
            )
        insert_chunk(client, "variance_model_selection", rows)

    return str(run_id)


def list_runs(limit: int = 25, user_id: str | None = None) -> pd.DataFrame:
    client = get_supabase_client()
    query = client.table("runs").select("*").order("created_at", desc=True)
    if user_id:
        query = query.eq("user_id", user_id)
    resp = query.limit(limit).execute()
    return pd.DataFrame(resp.data or [])


def load_run_bundle(run_id: str, user_id: str | None = None) -> dict[str, pd.DataFrame]:
    client = get_supabase_client()
    if user_id:
        owner_resp = (
            client.table("runs")
            .select("run_id")
            .eq("run_id", run_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not (owner_resp.data or []):
            return {
                "run": pd.DataFrame(),
                "portfolio": pd.DataFrame(),
                "backtest": pd.DataFrame(),
                "risk": pd.DataFrame(),
                "variance": pd.DataFrame(),
                "artifacts": pd.DataFrame(),
            }
    tables = {
        "run": ("runs", "run_id"),
        "portfolio": ("portfolio_weights", "run_id"),
        "backtest": ("backtest_perf", "run_id"),
        "risk": ("risk_diagnostics", "run_id"),
        "variance": ("variance_model_selection", "run_id"),
    }
    out = {}
    for key, (table, col) in tables.items():
        query = client.table(table).select("*").eq(col, run_id)
        if user_id and table in {"runs", "run_artifacts"}:
            query = query.eq("user_id", user_id)
        resp = query.execute()
        out[key] = pd.DataFrame(resp.data or [])
    try:
        query = client.table("run_artifacts").select("*").eq("run_id", run_id)
        if user_id:
            query = query.eq("user_id", user_id)
        resp = query.execute()
        out["artifacts"] = pd.DataFrame(resp.data or [])
    except Exception:
        out["artifacts"] = pd.DataFrame()
    return out


def supabase_available() -> bool:
    try:
        load_local_env()
        return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    except Exception:
        return False

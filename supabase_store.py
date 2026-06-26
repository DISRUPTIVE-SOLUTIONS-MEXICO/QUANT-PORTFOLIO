from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import dotenv_values, load_dotenv

from quant_core.contracts import (
    CONTRACT_SCHEMA_VERSION,
    ArtifactDescriptor,
    OrderIntentV1,
    PortfolioRunV2,
    PreTradeDecisionV1,
    PublicationManifest,
    PublicationState,
)
from supabase import Client, create_client

ENV_PATH = Path(__file__).resolve().with_name(".env")
RUN_ARTIFACT_DIR = Path(__file__).resolve().with_name(".quant_cache") / "run_artifacts"
ARTIFACT_CHUNK_THRESHOLD_BYTES = 1_500_000
ARTIFACT_CHUNK_SEPARATOR = "::"
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
        if not os.getenv("QPK_PORTFOLIO_OWNER_SECRET") and "auth" in st.secrets:
            cookie_key = st.secrets["auth"].get("cookie_key")
            if cookie_key:
                os.environ["QPK_PORTFOLIO_OWNER_SECRET"] = str(cookie_key)
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
    if isinstance(value, (datetime, date)):
        return value.isoformat()
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


def user_owner_key(username: str) -> str:
    """Return a stable, non-reversible owner key for app-level portfolios."""
    load_local_env()
    secret = os.getenv("QPK_PORTFOLIO_OWNER_SECRET") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not secret:
        raise RuntimeError("A server-side portfolio owner secret is required.")
    normalized = str(username or "").strip().lower()
    if not normalized:
        raise ValueError("username is required")
    return hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def insert_chunk(client: Client, table: str, rows: list[dict], chunk_size: int = 500):
    if not rows:
        return
    for start in range(0, len(rows), chunk_size):
        client.table(table).insert(rows[start : start + chunk_size]).execute()


def _finite_float(value: Any) -> float | None:
    """Return a database-safe numeric diagnostic or None for metadata/text."""
    if value is None or isinstance(value, (dict, list, tuple, pd.Series, pd.DataFrame, Path)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _risk_diagnostic_rows(run_id: str, rows: list[dict]) -> list[dict]:
    """Keep the numeric risk table type-safe; textual metadata lives in artifacts."""
    diagnostics = []
    for row in rows:
        value = _finite_float(row.get("Value"))
        metric = row.get("Metric")
        if metric and value is not None:
            diagnostics.append({"run_id": run_id, "metric": str(metric), "value": value})
    return diagnostics


def _payload_records(value: Any) -> list[dict]:
    if isinstance(value, pd.DataFrame):
        return _records(value)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and value:
        return [value]
    return []


def _payload_row_count(value: Any) -> int:
    return len(_payload_records(value))


def _payload_section(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    return value if isinstance(value, dict) else {}


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _first_record_timestamp(records: Any, *keys: str) -> pd.Timestamp | None:
    for row in _payload_records(records):
        for key in keys:
            timestamp = _parse_timestamp(row.get(key))
            if timestamp is not None:
                return timestamp
    return None


def _dashboard_snapshot_asof(dashboard: dict[str, Any]) -> pd.Timestamp | None:
    contract = _payload_section(dashboard, "contract")
    timestamp = _parse_timestamp(contract.get("as_of") or contract.get("asof"))
    if timestamp is not None:
        return timestamp
    status = _payload_section(dashboard, "status")
    timestamp = _first_record_timestamp(status.get("snapshot_meta"), "As_Of", "as_of", "Date")
    if timestamp is not None:
        return timestamp
    return _first_record_timestamp(status.get("market_context"), "As_Of", "Date")


def _module_timestamp_violations(dashboard: dict[str, Any]) -> tuple[str, ...]:
    r"""Reject explicit module timestamps that are after the snapshot date.

    Daily overlays and full artifacts are allowed to contain historical price
    paths, but an explicit module as-of date cannot be in the future relative
    to the publication snapshot. That is the atomic-publication equivalent of
    \(\mathcal{D}_{module}(t)\subset\mathcal{F}_t\).
    """
    snapshot_asof = _dashboard_snapshot_asof(dashboard)
    if snapshot_asof is None:
        return ()
    checks: list[tuple[str, pd.Timestamp | None]] = []
    for section_name in ("security_intelligence", "fixed_income_intelligence", "market_intelligence", "strategy_lab"):
        section = _payload_section(dashboard, section_name)
        checks.append((f"{section_name}.as_of", _parse_timestamp(section.get("as_of"))))
        contract = _payload_section(section, "contract")
        checks.append((f"{section_name}.contract.as_of", _parse_timestamp(contract.get("as_of"))))
    status = _payload_section(dashboard, "status")
    checks.append(("status.market_context.As_Of", _first_record_timestamp(status.get("market_context"), "As_Of", "Date")))
    violations = []
    for label, timestamp in checks:
        if timestamp is not None and timestamp.normalize() > snapshot_asof.normalize():
            violations.append(label)
    return tuple(violations)


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


def _artifact_storage_rows(run_id: str, name: str, artifact: Any, user_id: str | None = None) -> list[dict]:
    """Recursively encode large dictionaries/lists into bounded JSONB rows."""
    safe = _json_safe(artifact)

    def encode(path: str, value: Any) -> list[dict]:
        serial = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        base = {
            "run_id": run_id,
            "artifact_name": path,
            "artifact_sha256": hashlib.sha256(serial.encode("utf-8")).hexdigest(),
        }
        if user_id:
            base["user_id"] = user_id
        if len(serial.encode("utf-8")) <= ARTIFACT_CHUNK_THRESHOLD_BYTES:
            return [{**base, "artifact_json": value}]
        if isinstance(value, dict):
            keys = list(value)
            output = [
                {
                    **base,
                    "artifact_json": {
                        "_chunked": True,
                        "schema": "dict_v1",
                        "keys": keys,
                    },
                }
            ]
            for key in keys:
                output.extend(encode(f"{path}{ARTIFACT_CHUNK_SEPARATOR}{key}", value.get(key)))
            return output
        if isinstance(value, list) and len(value) > 1:
            estimated_parts = max(
                2,
                int(math.ceil(len(serial.encode("utf-8")) / ARTIFACT_CHUNK_THRESHOLD_BYTES)),
            )
            part_size = max(1, int(math.ceil(len(value) / estimated_parts)))
            parts = [value[start : start + part_size] for start in range(0, len(value), part_size)]
            output = [
                {
                    **base,
                    "artifact_json": {
                        "_chunked": True,
                        "schema": "list_v1",
                        "parts": len(parts),
                    },
                }
            ]
            for idx, part in enumerate(parts):
                output.extend(
                    encode(
                        f"{path}{ARTIFACT_CHUNK_SEPARATOR}part{idx:04d}",
                        part,
                    )
                )
            return output
        raise ValueError(f"Artifact node {path!r} exceeds the JSONB row budget and cannot be split.")

    return encode(name, safe)


def reassemble_artifact_rows(rows: list[dict] | pd.DataFrame, name: str) -> dict:
    """Restore an artifact stored either as one JSONB row or top-level section rows."""
    records = rows.to_dict(orient="records") if isinstance(rows, pd.DataFrame) else list(rows or [])
    payloads = {
        str(row.get("artifact_name")): row.get("artifact_json")
        for row in records
        if row.get("artifact_name") is not None
    }

    def decode(path: str):
        payload = payloads.get(path)
        if not isinstance(payload, dict) or not payload.get("_chunked"):
            return payload
        schema = payload.get("schema")
        if schema in {"top_level_dict_v1", "dict_v1"}:
            keys = payload.get("sections", []) if schema == "top_level_dict_v1" else payload.get("keys", [])
            return {
                key: decode(f"{path}{ARTIFACT_CHUNK_SEPARATOR}{key}")
                for key in keys
                if f"{path}{ARTIFACT_CHUNK_SEPARATOR}{key}" in payloads
            }
        if schema == "list_v1":
            output = []
            for idx in range(int(payload.get("parts", 0))):
                part = decode(f"{path}{ARTIFACT_CHUNK_SEPARATOR}part{idx:04d}")
                if isinstance(part, list):
                    output.extend(part)
            return output
        return {}

    restored = decode(name)
    return restored if isinstance(restored, dict) else {}


def save_run_artifacts(
    client: Client, run_id: str, artifacts: dict[str, Any], user_id: str | None = None
) -> dict[str, Any]:
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
        rows.extend(_artifact_storage_rows(run_id, name, artifact, user_id=user_id))
    try:
        # Dashboard and market-intelligence bundles can each be several MB.
        # PostgREST accepts them individually, while a combined request can
        # exceed the gateway body limit and leave the run in building state.
        insert_chunk(client, "run_artifacts", rows, chunk_size=1)
        manifest["supabase_run_artifacts"] = True
    except Exception as exc:
        manifest["supabase_artifact_error"] = str(exc)[:500]
        if user_id:
            manifest["supabase_artifact_error"] = (
                "Tenant-safe write rejected; user_id was not removed. "
                + manifest["supabase_artifact_error"]
            )
    return manifest


def build_publication_manifest(
    run_id: str,
    artifacts: dict[str, Any],
    *,
    channel: str = "global",
) -> PublicationManifest:
    dashboard = artifacts.get("dashboard_payload")
    contract = dashboard.get("contract", {}) if isinstance(dashboard, dict) else {}
    analytics_scope = str(contract.get("analytics_scope", "")).strip().lower()
    if channel == "user":
        publication_kind = "user_portfolio"
    elif channel == "research":
        publication_kind = "research_evidence"
    elif analytics_scope == "full_analysis":
        publication_kind = "full_analysis"
    else:
        publication_kind = "daily_snapshot"
    descriptors = []
    for name, artifact in artifacts.items():
        safe = _json_safe(artifact)
        serial = json.dumps(safe, sort_keys=True, ensure_ascii=False, default=str)
        descriptors.append(
            ArtifactDescriptor(
                name=str(name),
                content_sha256=hashlib.sha256(serial.encode("utf-8")).hexdigest(),
                bytes=len(serial.encode("utf-8")),
                required=name
                in {
                    "dashboard_payload",
                    "backtest_path_bundle",
                    "suitability_gate",
                    "promotion_gate",
                    "data_freshness_report",
                },
            )
        )
    return PublicationManifest(
        run_id=run_id,
        channel=channel,
        publication_kind=publication_kind,
        state=PublicationState.STAGING,
        artifacts=tuple(descriptors),
        quality_checks={},
    )


def validate_publication_bundle(
    manifest: PublicationManifest,
    artifacts: dict[str, Any],
) -> tuple[bool, dict[str, bool], tuple[str, ...]]:
    names = set(artifacts)
    dashboard = artifacts.get("dashboard_payload")
    dashboard = dashboard if isinstance(dashboard, dict) else {}
    contract = dashboard.get("contract", {}) if isinstance(dashboard.get("contract"), dict) else {}
    status = dashboard.get("status", {}) if isinstance(dashboard.get("status"), dict) else {}
    charts = dashboard.get("charts", {}) if isinstance(dashboard.get("charts"), dict) else {}
    allocation = dashboard.get("allocation", {}) if isinstance(dashboard.get("allocation"), dict) else {}
    tables = dashboard.get("tables", {}) if isinstance(dashboard.get("tables"), dict) else {}
    research = dashboard.get("research", {}) if isinstance(dashboard.get("research"), dict) else {}
    market_intelligence = (
        dashboard.get("market_intelligence", {}) if isinstance(dashboard.get("market_intelligence"), dict) else {}
    )
    strategy_lab = dashboard.get("strategy_lab", {}) if isinstance(dashboard.get("strategy_lab"), dict) else {}
    price_paths = charts.get("price_paths", []) if isinstance(charts.get("price_paths"), list) else []
    drawdown_paths = charts.get("drawdowns", []) if isinstance(charts.get("drawdowns"), list) else []
    portfolio_rows = (
        allocation.get("recommended_portfolio", []) if isinstance(allocation.get("recommended_portfolio"), list) else []
    )

    minimum_history = True
    allocation_present = True
    weights_valid = True
    strategy_lab_present = True
    security_intelligence_present = True
    fixed_income_intelligence_present = True
    full_scope_declared = True
    institutional_full_payload_complete = True
    full_capability_matrix_present = True
    full_capability_core_complete = True
    temporal_coherence = True
    daily_market_overlay_present = True
    temporal_violations = _module_timestamp_violations(dashboard)
    temporal_coherence = not temporal_violations
    schema_version = str(contract.get("schema_version", ""))
    requires_security_intelligence = schema_version in {
        "2026.06.15-market-intelligence-v9",
        "2026.06.15-market-intelligence-v10",
        "2026.06.19-publication-isolation-v11",
    }
    if requires_security_intelligence:
        security = dashboard.get("security_intelligence")
        security = security if isinstance(security, dict) else {}
        security_metrics = security.get("metrics", [])
        security_prices = security.get("price_history", [])
        security_methodology = security.get("methodology", [])
        security_contract = security.get("contract", {})
        benchmark_xi = str(security.get("benchmark_xi", "")).strip()
        security_intelligence_present = (
            isinstance(security_contract, dict)
            and security_contract.get("evidence_scope") == "live_snapshot"
            and str(security_contract.get("benchmark_xi", "")).strip().upper() == benchmark_xi.upper()
            and isinstance(security_metrics, list)
            and len(security_metrics) >= 2
            and isinstance(security_prices, list)
            and len(security_prices) >= 252
            and isinstance(security_methodology, list)
            and bool(security_methodology)
            and bool(benchmark_xi)
        )
    if schema_version in {
        "2026.06.15-market-intelligence-v10",
        "2026.06.19-publication-isolation-v11",
    }:
        fixed_income = dashboard.get("fixed_income_intelligence")
        fixed_income = fixed_income if isinstance(fixed_income, dict) else {}
        fixed_income_contract = fixed_income.get("contract", {})
        country_metrics = fixed_income.get("country_metrics", [])
        factor_history = fixed_income.get("factor_history", [])
        stress_scenarios = fixed_income.get("stress_scenarios", [])
        fixed_income_methodology = fixed_income.get("methodology", [])
        fixed_income_intelligence_present = (
            isinstance(fixed_income_contract, dict)
            and fixed_income_contract.get("evidence_scope") == "live_snapshot"
            and str(fixed_income_contract.get("factor_observation_mode", "")).startswith("native_calendar")
            and isinstance(country_metrics, list)
            and len(country_metrics) >= 2
            and isinstance(factor_history, list)
            and len(factor_history) >= 126
            and isinstance(stress_scenarios, list)
            and len(stress_scenarios) >= 6
            and isinstance(fixed_income_methodology, list)
            and bool(fixed_income_methodology)
        )
    if manifest.publication_kind == "full_analysis":
        full_scope_declared = str(contract.get("analytics_scope", "")).strip().lower() == "full_analysis"
        capability_rows = _payload_records(status.get("capability_completeness", []))
        required_capability_modules = {
            "Market Intelligence",
            "Rates & Fixed Income",
            "Equity Fundamentals",
            "Benchmark xi",
            "XCDR Research",
            "Portfolio Construction",
            "Risk Laboratory",
            "Validation & Governance",
            "Data Quality",
        }
        observed_capability_modules = {
            str(row.get("Module", "")).strip()
            for row in capability_rows
            if str(row.get("Module", "")).strip()
        }
        full_capability_matrix_present = required_capability_modules.issubset(observed_capability_modules)
        full_capability_core_complete = full_capability_matrix_present and all(
            str(row.get("Status", "")).strip().lower() == "complete"
            for row in capability_rows
            if str(row.get("Module", "")).strip() in required_capability_modules
        )
        allocation_present = len(portfolio_rows) >= 2
        if price_paths:
            dates = pd.to_datetime(
                [row.get("Date") for row in price_paths if isinstance(row, dict) and row.get("Date")],
                errors="coerce",
                utc=True,
            )
            valid_dates = dates[~pd.isna(dates)]
            minimum_history = bool(
                len(valid_dates) >= 2 and (valid_dates.max() - valid_dates.min()).days >= (3 * 365 - 7)
            )
        else:
            minimum_history = False
        weights = [
            float(row.get("Weight", row.get("target_weight")))
            for row in portfolio_rows
            if isinstance(row, dict) and row.get("Weight", row.get("target_weight")) is not None
        ]
        weights_valid = (
            bool(weights) and abs(sum(weights) - 1.0) <= 1e-6 and all(0.0 <= value <= 1.0 for value in weights)
        )
        strategy_summary = strategy_lab.get("summary", [])
        strategy_constitution = strategy_lab.get("constitution", [])
        strategy_windows = strategy_lab.get("walk_forward_windows", [])
        strategy_holdout = strategy_lab.get("holdout_summary", [])
        strategy_validation = strategy_lab.get("validation", [])
        strategy_lineage = strategy_lab.get("research_lineage", [])
        strategy_registry = strategy_lab.get("strategy_registry", [])
        strategy_equivalence = strategy_lab.get("candidate_equivalence", [])
        validation_metrics = {
            str(row.get("Metric")): row for row in strategy_validation if isinstance(row, dict) and row.get("Metric")
        }
        strategy_lab_present = (
            isinstance(strategy_summary, list)
            and len(strategy_summary) >= 3
            and isinstance(strategy_constitution, list)
            and bool(strategy_constitution)
            and isinstance(strategy_windows, list)
            and bool(strategy_windows)
            and isinstance(strategy_holdout, list)
            and bool(strategy_holdout)
            and isinstance(strategy_validation, list)
            and "Promotion_Status" in validation_metrics
            and "Holdout_Independence" in validation_metrics
            and isinstance(strategy_lineage, list)
            and bool(strategy_lineage)
            and isinstance(strategy_registry, list)
            and len(strategy_registry) >= len(strategy_summary)
            and isinstance(strategy_equivalence, list)
            and len(strategy_equivalence) == len(strategy_summary)
        )
        fundamentals = tables.get("fundamentals", [])
        risk_rows = tables.get("risk", [])
        validation_rows = tables.get("validation", [])
        data_quality_rows = artifacts.get("data_freshness_report", [])
        benchmark_rows = research.get("benchmark_governance", [])
        registry_rows = research.get("model_registry", [])
        has_benchmark_xi = _payload_row_count(benchmark_rows) > 0 or any(
            row.get("benchmark_ticker") or row.get("benchmark_xi") or row.get("Benchmark")
            for row in _payload_records(registry_rows)
        )
        retained_research_surfaces = all(
            key in research
            for key in (
                "variance_model_selection",
                "pelt_regime_segments",
                "pelt_change_points",
                "options_summary",
                "options_chain",
                "factor_attribution",
                "hedge_suggestions",
            )
        )
        retained_market_surfaces = all(
            key in market_intelligence
            for key in (
                "macro_history",
                "global_yield_curves",
                "global_rate_history",
                "sentiment_timeline",
                "geopolitical_summary",
            )
        )
        non_price_only_sectors = [
            str(row.get("Sector", "")).strip().lower()
            for row in _payload_records(fundamentals)
            if str(row.get("Sector", "")).strip()
        ]
        fundamentals_present = (
            _payload_row_count(fundamentals) >= len(portfolio_rows)
            and bool(non_price_only_sectors)
            and not all(sector == "price-only snapshot" for sector in non_price_only_sectors)
        )
        institutional_full_payload_complete = (
            fundamentals_present
            and _payload_row_count(risk_rows) >= 4
            and _payload_row_count(validation_rows) >= 1
            and _payload_row_count(drawdown_paths) >= max(2, _payload_row_count(price_paths) // 2)
            and has_benchmark_xi
            and retained_research_surfaces
            and retained_market_surfaces
            and _payload_row_count(data_quality_rows) >= 1
        )
    elif manifest.publication_kind == "daily_snapshot":
        market_intelligence = dashboard.get("market_intelligence")
        daily_market_overlay_present = isinstance(market_intelligence, dict) and bool(market_intelligence)

    checks = {
        "required_artifacts_present": all(
            descriptor.name in names for descriptor in manifest.artifacts if descriptor.required
        ),
        "dashboard_contract_present": bool(dashboard),
        "suitability_gate_present": isinstance(artifacts.get("suitability_gate"), dict),
        "promotion_gate_present": isinstance(artifacts.get("promotion_gate"), dict),
        "full_analysis_scope_declared": full_scope_declared,
        "minimum_three_year_price_history": minimum_history,
        "allocation_present": allocation_present,
        "portfolio_weights_valid": weights_valid,
        "strategy_lab_present": strategy_lab_present,
        "security_intelligence_present": security_intelligence_present,
        "fixed_income_intelligence_present": fixed_income_intelligence_present,
        "publication_temporal_coherence": temporal_coherence,
        "institutional_full_payload_complete": institutional_full_payload_complete,
        "full_capability_matrix_present": full_capability_matrix_present,
        "full_capability_core_complete": full_capability_core_complete,
        "daily_market_overlay_present": daily_market_overlay_present,
    }
    rejections = tuple(name for name, passed in checks.items() if not passed)
    if temporal_violations:
        rejections = (*rejections, *tuple(f"future_module_asof:{item}" for item in temporal_violations))
    return all(checks.values()), checks, rejections


def stage_and_promote_publication(
    client: Client,
    run_id: str,
    artifacts: dict[str, Any],
    *,
    user_id: str | None = None,
    channel: str = "global",
) -> dict[str, Any]:
    if channel == "user" and not user_id:
        raise ValueError("user publications require a Supabase auth user_id")
    manifest = build_publication_manifest(run_id, artifacts, channel=channel)
    valid, checks, rejections = validate_publication_bundle(manifest, artifacts)
    state = PublicationState.VALIDATED if valid else PublicationState.REJECTED
    validated_manifest = manifest.model_copy(
        update={
            "state": state,
            "validated_at": datetime.now(UTC) if valid else None,
            "quality_checks": checks,
            "rejection_reasons": rejections,
        }
    )
    payload = {
        "publication_id": str(validated_manifest.publication_id),
        "run_id": run_id,
        "user_id": user_id,
        "channel": channel,
        "publication_kind": validated_manifest.publication_kind,
        "state": validated_manifest.state.value,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "manifest_json": validated_manifest.model_dump(mode="json"),
        "manifest_sha256": validated_manifest.sha256(),
        "validated_at": validated_manifest.validated_at.isoformat() if validated_manifest.validated_at else None,
        "rejection_reason": ", ".join(rejections) if rejections else None,
    }
    client.table("publication_manifests").insert(payload).execute()
    if not valid:
        return {
            "publication_id": str(validated_manifest.publication_id),
            "state": state.value,
            "quality_checks": checks,
            "rejection_reasons": list(rejections),
        }
    client.rpc("promote_publication", {"p_publication_id": str(validated_manifest.publication_id)}).execute()
    return {
        "publication_id": str(validated_manifest.publication_id),
        "state": PublicationState.ACTIVE.value,
        "quality_checks": checks,
        "rejection_reasons": [],
    }


def save_run_to_supabase(
    results: dict,
    config,
    status: str = "completed",
    user_id: str | None = None,
    *,
    owner_username: str | None = None,
    portfolio_name: str | None = None,
) -> str:
    if owner_username and not user_id:
        raise ValueError(
            "Personal portfolio persistence requires a Supabase Auth user_id; refusing global publication."
        )
    client = get_supabase_client()
    cfg = config_to_json(config)
    model_registry = results.get("model_registry", pd.DataFrame()).copy()
    registry_row = _records(model_registry)[0] if not model_registry.empty else {}

    run_payload = {
        "config": cfg,
        "benchmark_ticker": cfg.get("benchmark_ticker"),
        "price_period": cfg.get("price_period"),
        # Publish only after every dependent table and artifact is durable.
        "status": "building",
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
        except Exception as exc:
            if user_id:
                raise RuntimeError("Tenant-safe run persistence failed; user_id was not removed.") from exc
            run_resp = client.table("runs").insert(run_payload).execute()
    run_data = run_resp.data or []
    if not run_data:
        raise RuntimeError("Supabase no devolvió run_id al insertar en runs.")
    run_id = run_data[0]["run_id"]
    owner_key = user_owner_key(owner_username) if owner_username else None
    portfolio = results.get("portfolio", pd.DataFrame()).copy()
    portfolio_manifest = None
    if owner_key:
        portfolio_manifest = {
            "owner_key": owner_key,
            "portfolio_name": str(portfolio_name or "My Portfolio").strip() or "My Portfolio",
            "benchmark_ticker": cfg.get("benchmark_ticker"),
            "objective": cfg.get("weight_objective") or registry_row.get("objective"),
            "price_period": cfg.get("price_period"),
            "position_count": int(len(portfolio)),
            "config": cfg,
        }
    artifact_bundle = {
        "dashboard_payload": results.get("dashboard_payload", {}),
        "backtest_path_bundle": results.get("backtest_path_bundle", {}),
        "suitability_gate": results.get("suitability_gate", {}),
        "promotion_gate": results.get("promotion_gate", {}),
        "data_freshness_report": results.get("data_freshness_report", pd.DataFrame()),
        "performance_summary": results.get("performance_summary", pd.DataFrame()),
    }
    if portfolio_manifest:
        artifact_bundle["user_portfolio_manifest"] = portfolio_manifest
    artifact_manifest = save_run_artifacts(
        client,
        str(run_id),
        artifact_bundle,
        user_id=user_id,
    )
    if not artifact_manifest.get("supabase_run_artifacts"):
        raise RuntimeError(
            "Run artifacts were not persisted in Supabase; publication remains in building state. "
            f"Cause: {artifact_manifest.get('supabase_artifact_error') or 'unknown'}"
        )

    require_atomic_publication = str(
        os.getenv(
            "QPK_REQUIRE_ATOMIC_PUBLICATION",
            os.getenv("QPK_CLOUD_REFRESH_REQUIRE_SUPABASE", "0"),
        )
    ).strip().lower() in {"1", "true", "yes", "on"}
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
                "Value": json.dumps(_json_safe(value), ensure_ascii=False, default=str)
                if isinstance(value, (dict, list))
                else value,
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
        extra_summary_rows.append(
            {
                "Metric": "suitability_gate_breach_count",
                "Value": int(len(breaches)) if isinstance(breaches, pd.DataFrame) else 0,
            }
        )
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
                for metric in [
                    "Frequency",
                    "Mean_Prob",
                    "Mean_Entropy",
                    "Mean_Hawkish",
                    "Mean_Bullish",
                    "Mean_Curve",
                    "Mean_Credit",
                ]:
                    if row.get(metric) is not None:
                        extra_summary_rows.append({"Metric": f"latent_{state}_{metric}", "Value": row.get(metric)})
        markov = latent.get("markov_forecast", pd.DataFrame()).copy()
        if not markov.empty:
            last = _records(markov.tail(1))[0]
            for metric in [
                "Markov_State_Persistence",
                "Markov_Stress_Prob",
                "Markov_Risk_On_Prob",
                "Markov_Transition_Entropy",
                "Markov_Transition_Obs",
            ]:
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
                    extra_summary_rows.append(
                        {
                            "Metric": f"factor_risk_{row.get('Name')}_Pct_Total_Variance",
                            "Value": row.get("Pct_Total_Variance"),
                        }
                    )
    regime_perf = results.get("regime_performance", pd.DataFrame()).copy()
    if not regime_perf.empty:
        for row in _records(regime_perf):
            state = row.get("State")
            if row.get("Sortino_Approx") is not None:
                extra_summary_rows.append(
                    {"Metric": f"regime_{state}_Sortino_Approx", "Value": row.get("Sortino_Approx")}
                )
            if row.get("Total_Return") is not None:
                extra_summary_rows.append({"Metric": f"regime_{state}_Total_Return", "Value": row.get("Total_Return")})
    stress = results.get("stress_tests", pd.DataFrame()).copy()
    if not stress.empty:
        for row in _records(stress):
            if row.get("Estimated_Portfolio_Return") is not None:
                extra_summary_rows.append(
                    {
                        "Metric": f"stress_{row.get('Scenario')}_Estimated_Return",
                        "Value": row.get("Estimated_Portfolio_Return"),
                    }
                )
    attr = results.get("oos_factor_attribution", pd.DataFrame()).copy()
    if not attr.empty and {"Component", "Contribution"}.issubset(attr.columns):
        for comp, value in attr.groupby("Component")["Contribution"].sum().items():
            extra_summary_rows.append({"Metric": f"oos_attr_{comp}_Total", "Value": value})
    ledger = results.get("capital_ledger", pd.DataFrame()).copy()
    if not ledger.empty:
        for metric, col in [
            ("ledger_final_capital", "End_Capital"),
            ("ledger_max_drawdown", "Drawdown"),
            ("ledger_total_net_pnl", "Net_PnL"),
        ]:
            if col in ledger:
                value = (
                    ledger[col].iloc[-1]
                    if col == "End_Capital"
                    else (ledger[col].min() if col == "Drawdown" else ledger[col].sum())
                )
                extra_summary_rows.append({"Metric": metric, "Value": value})
    if not summary.empty:
        diagnostic_inputs = _records(summary) + extra_summary_rows
        if registry_row:
            for metric, value in (registry_row.get("data_quality") or {}).items():
                diagnostic_inputs.append({"Metric": f"data_quality_{metric}", "Value": value})
            for metric, value in (registry_row.get("timings") or {}).items():
                diagnostic_inputs.append({"Metric": f"timing_{metric}", "Value": value})
        rows = _risk_diagnostic_rows(str(run_id), diagnostic_inputs)
        insert_chunk(client, "risk_diagnostics", rows)
    elif extra_summary_rows or registry_row:
        rows = _risk_diagnostic_rows(str(run_id), extra_summary_rows)
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

    # This update is the atomic publication boundary. Readers only resolve
    # completed global dashboards or user_completed personal portfolios.
    client.table("runs").update({"status": status}).eq("run_id", run_id).execute()

    publication_manifest = None
    try:
        publication_manifest = stage_and_promote_publication(
            client,
            str(run_id),
            artifact_bundle,
            user_id=user_id,
            channel="user" if user_id else "global",
        )
    except Exception as exc:
        logger.warning("Atomic publication migration is unavailable: %s", type(exc).__name__)
        if require_atomic_publication:
            raise RuntimeError("Atomic publication failed; the previous active snapshot remains unchanged.") from exc
    if require_atomic_publication and (
        not publication_manifest or publication_manifest.get("state") != PublicationState.ACTIVE.value
    ):
        raise RuntimeError("Publication did not become active; the previous active snapshot remains unchanged.")
    if publication_manifest:
        logger.info(
            "Activated publication %s for run %s",
            publication_manifest.get("publication_id"),
            run_id,
        )
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
            client.table("runs").select("run_id").eq("run_id", run_id).eq("user_id", user_id).limit(1).execute()
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


def list_user_portfolios(username: str, limit: int = 50) -> pd.DataFrame:
    """List only portfolios owned by the authenticated app username."""
    client = get_supabase_client()
    owner_key = user_owner_key(username)
    runs = (
        client.table("runs")
        .select("run_id,created_at,status,benchmark_ticker,price_period,config")
        .eq("status", "user_completed")
        .order("created_at", desc=True)
        .limit(min(1000, max(250, int(limit) * 20)))
        .execute()
        .data
        or []
    )
    if not runs:
        return pd.DataFrame()
    run_ids = [str(row["run_id"]) for row in runs if row.get("run_id")]
    artifacts = (
        client.table("run_artifacts")
        .select("run_id,artifact_json")
        .in_("run_id", run_ids)
        .eq("artifact_name", "user_portfolio_manifest")
        .execute()
        .data
        or []
    )
    manifests = {
        str(row.get("run_id")): row.get("artifact_json")
        for row in artifacts
        if isinstance(row.get("artifact_json"), dict)
        and hmac.compare_digest(str(row["artifact_json"].get("owner_key", "")), owner_key)
    }
    output = []
    for run in runs:
        run_id = str(run.get("run_id"))
        manifest = manifests.get(run_id)
        if not manifest:
            continue
        output.append(
            {
                "run_id": run_id,
                "created_at": run.get("created_at"),
                "portfolio_name": manifest.get("portfolio_name") or "My Portfolio",
                "benchmark_ticker": manifest.get("benchmark_ticker") or run.get("benchmark_ticker"),
                "objective": manifest.get("objective"),
                "price_period": manifest.get("price_period") or run.get("price_period"),
                "position_count": manifest.get("position_count"),
            }
        )
        if len(output) >= limit:
            break
    return pd.DataFrame(output)


def load_user_portfolio(run_id: str, username: str) -> dict[str, pd.DataFrame]:
    """Load a personal portfolio only after verifying its HMAC owner manifest."""
    client = get_supabase_client()
    owner_key = user_owner_key(username)
    response = (
        client.table("run_artifacts")
        .select("artifact_json")
        .eq("run_id", str(run_id))
        .eq("artifact_name", "user_portfolio_manifest")
        .limit(1)
        .execute()
    )
    rows = response.data or []
    manifest = rows[0].get("artifact_json") if rows else {}
    if not isinstance(manifest, dict) or not hmac.compare_digest(str(manifest.get("owner_key", "")), owner_key):
        return {
            "run": pd.DataFrame(),
            "portfolio": pd.DataFrame(),
            "backtest": pd.DataFrame(),
            "risk": pd.DataFrame(),
            "variance": pd.DataFrame(),
            "artifacts": pd.DataFrame(),
        }
    return load_run_bundle(str(run_id))


def supabase_available() -> bool:
    try:
        load_local_env()
        return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    except Exception:
        return False


def save_versioned_user_portfolio(
    portfolio: PortfolioRunV2,
    *,
    client: Client | None = None,
) -> dict[str, str]:
    """Persist one immutable portfolio version and atomically select it as active."""
    if portfolio.user_id is None:
        raise ValueError("A Supabase auth user_id is required for a user portfolio")
    db = client or get_supabase_client()
    user_id = str(portfolio.user_id)
    portfolio_row = (
        db.table("user_portfolios")
        .select("portfolio_id")
        .eq("user_id", user_id)
        .eq("name", portfolio.portfolio_name)
        .limit(1)
        .execute()
        .data
        or []
    )
    if portfolio_row:
        portfolio_id = str(portfolio_row[0]["portfolio_id"])
    else:
        created = (
            db.table("user_portfolios")
            .insert(
                {
                    "user_id": user_id,
                    "name": portfolio.portfolio_name,
                    "base_currency": portfolio.base_currency,
                }
            )
            .execute()
            .data
            or []
        )
        if not created:
            raise RuntimeError("Supabase did not return portfolio_id")
        portfolio_id = str(created[0]["portfolio_id"])

    version_payload = {
        "portfolio_id": portfolio_id,
        "user_id": user_id,
        "run_id": str(portfolio.run_id),
        "contract_json": portfolio.model_dump(mode="json"),
        "contract_sha256": portfolio.sha256(),
    }
    existing_version = (
        db.table("portfolio_versions")
        .select("version_id")
        .eq("portfolio_id", portfolio_id)
        .eq("contract_sha256", version_payload["contract_sha256"])
        .limit(1)
        .execute()
        .data
        or []
    )
    if existing_version:
        version_id = str(existing_version[0]["version_id"])
    else:
        created_version = db.table("portfolio_versions").insert(version_payload).execute().data or []
        if not created_version:
            raise RuntimeError("Supabase did not return version_id")
        version_id = str(created_version[0]["version_id"])
    (
        db.table("user_portfolios")
        .update({"active_version_id": version_id, "updated_at": datetime.now(UTC).isoformat()})
        .eq("portfolio_id", portfolio_id)
        .eq("user_id", user_id)
        .execute()
    )
    return {"portfolio_id": portfolio_id, "version_id": version_id}


def save_order_intent(
    order: OrderIntentV1,
    *,
    portfolio_version_id: str | None = None,
    client: Client | None = None,
) -> str:
    """Persist an immutable paper-order intent through the server-only client."""
    db = client or get_supabase_client()
    payload = {
        "order_intent_id": str(order.order_intent_id),
        "user_id": str(order.user_id),
        "run_id": str(order.run_id),
        "portfolio_version_id": portfolio_version_id,
        "status": order.status.value,
        "contract_json": order.model_dump(mode="json"),
        "contract_sha256": order.sha256(),
        "approved_by": str(order.approved_by) if order.approved_by else None,
        "approved_at": order.approved_at.isoformat() if order.approved_at else None,
    }
    db.table("order_intents").insert(payload).execute()
    return str(order.order_intent_id)


def save_pretrade_decision(
    decision: PreTradeDecisionV1,
    *,
    user_id: str,
    client: Client | None = None,
) -> str:
    """Persist the immutable decision produced by the backend pre-trade engine."""
    db = client or get_supabase_client()
    payload = {
        "decision_id": str(decision.decision_id),
        "order_intent_id": str(decision.order_intent_id),
        "user_id": str(user_id),
        "approved": decision.approved,
        "contract_json": decision.model_dump(mode="json"),
        "contract_sha256": decision.sha256(),
        "evaluated_at": decision.evaluated_at.isoformat(),
    }
    db.table("pretrade_decisions").insert(payload).execute()
    return str(decision.decision_id)

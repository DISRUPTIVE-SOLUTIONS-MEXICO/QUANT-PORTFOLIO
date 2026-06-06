from __future__ import annotations

from typing import Any

import pandas as pd

from supabase_store import get_supabase_client


def _payload_row_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        if not value:
            return 0
        lengths = [len(v) for v in value.values() if isinstance(v, list)]
        return max(lengths, default=1)
    return 0


def dashboard_artifact_scope(payload: dict[str, Any] | None) -> str:
    """Classify a persisted dashboard contract without relying on run ordering."""
    safe = payload if isinstance(payload, dict) else {}
    status = safe.get("status") if isinstance(safe.get("status"), dict) else {}
    tables = safe.get("tables") if isinstance(safe.get("tables"), dict) else {}
    charts = safe.get("charts") if isinstance(safe.get("charts"), dict) else {}

    snapshot_meta = status.get("snapshot_meta")
    if isinstance(snapshot_meta, list) and snapshot_meta:
        mode = str(snapshot_meta[0].get("Snapshot_Mode", "")).lower()
        if mode == "daily_price_snapshot":
            return "daily_snapshot"
    elif isinstance(snapshot_meta, dict):
        mode = str(snapshot_meta.get("Snapshot_Mode", "")).lower()
        if mode == "daily_price_snapshot":
            return "daily_snapshot"

    full_evidence = (
        _payload_row_count(tables.get("validation"))
        + _payload_row_count(tables.get("rejections"))
        + _payload_row_count(charts.get("rate_curves"))
        + _payload_row_count(charts.get("options_surface"))
        + _payload_row_count(status.get("promotion_tests"))
    )
    return "full_analysis" if full_evidence > 0 else "unknown"


def latest_dashboard_artifacts(user_id: str | None = None, scan_limit: int = 25) -> dict[str, Any]:
    """Return separately scoped daily and full dashboard artifacts.

    A lightweight daily prewarm must never displace the most recent full
    research run. The frontend can render the full analysis while overlaying
    the fresher market snapshot.
    """
    client = get_supabase_client()
    runs_query = client.table("runs").select("run_id,created_at").order("created_at", desc=True)
    if user_id:
        runs_query = runs_query.eq("user_id", user_id)
    run_rows = runs_query.limit(max(1, int(scan_limit))).execute().data or []
    if not run_rows:
        return {}

    run_ids = [str(row["run_id"]) for row in run_rows if row.get("run_id")]
    artifact_rows = (
        client.table("run_artifacts")
        .select("run_id,artifact_json,created_at")
        .in_("run_id", run_ids)
        .eq("artifact_name", "dashboard_payload")
        .execute()
        .data
        or []
    )
    artifacts_by_run = {str(row.get("run_id")): row for row in artifact_rows}
    resolved: dict[str, Any] = {}

    for run_row in run_rows:
        run_id = str(run_row.get("run_id"))
        artifact_row = artifacts_by_run.get(run_id)
        if not artifact_row:
            continue
        payload = artifact_row.get("artifact_json") or {}
        artifact = {
            "run_id": run_id,
            "created_at": artifact_row.get("created_at") or run_row.get("created_at"),
            "dashboard_payload": payload,
            "scope": dashboard_artifact_scope(payload),
        }
        resolved.setdefault("latest_any", artifact)
        scope = artifact["scope"]
        if scope == "daily_snapshot":
            resolved.setdefault("daily_snapshot", artifact)
        elif scope == "full_analysis":
            resolved.setdefault("full_analysis", artifact)
        if "daily_snapshot" in resolved and "full_analysis" in resolved:
            break
    return resolved


def create_optimization_job(user_id: str, config: dict[str, Any]) -> str:
    """Create a queued optimization job for a future worker.

    This is intentionally lightweight for Vercel/API usage. Heavy computation
    should happen in a worker that reads queued jobs and writes run artifacts.
    """
    client = get_supabase_client()
    resp = (
        client.table("jobs")
        .insert({"user_id": user_id, "job_type": "optimization", "status": "queued", "config": config})
        .execute()
    )
    data = resp.data or []
    if not data:
        raise RuntimeError("Supabase did not return a job_id.")
    return str(data[0]["job_id"])


def list_user_jobs(user_id: str, limit: int = 25) -> pd.DataFrame:
    client = get_supabase_client()
    resp = (
        client.table("jobs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(resp.data or [])


def get_job_status(job_id: str, user_id: str | None = None) -> dict[str, Any]:
    client = get_supabase_client()
    query = client.table("jobs").select("*").eq("job_id", job_id)
    if user_id:
        query = query.eq("user_id", user_id)
    resp = query.limit(1).execute()
    data = resp.data or []
    return data[0] if data else {}


def latest_dashboard_artifact(user_id: str | None = None) -> dict[str, Any]:
    """Return the latest dashboard_payload artifact.

    In service-role contexts, user_id can be omitted for admin/local testing.
    Production API routes must pass the authenticated user id so run artifacts
    cannot be resolved across tenants by guessed UUIDs.
    """
    return latest_dashboard_artifacts(user_id=user_id, scan_limit=1).get("latest_any", {})

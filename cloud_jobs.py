from __future__ import annotations

from typing import Any

import pandas as pd

from supabase_store import get_supabase_client


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
    client = get_supabase_client()
    runs = client.table("runs").select("run_id,created_at").order("created_at", desc=True)
    if user_id:
        runs = runs.eq("user_id", user_id)
    run_resp = runs.limit(1).execute()
    run_data = run_resp.data or []
    if not run_data:
        return {}
    run_id = run_data[0]["run_id"]
    art_resp = (
        client.table("run_artifacts")
        .select("artifact_json,created_at")
        .eq("run_id", run_id)
        .eq("artifact_name", "dashboard_payload")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    art_data = art_resp.data or []
    if not art_data:
        return {"run_id": run_id, "dashboard_payload": {}}
    return {"run_id": run_id, "dashboard_payload": art_data[0].get("artifact_json") or {}}

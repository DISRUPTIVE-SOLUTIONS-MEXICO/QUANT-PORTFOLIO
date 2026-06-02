from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supabase_store import get_supabase_client


TABLES = [
    "schema_migrations",
    "runs",
    "run_artifacts",
    "portfolio_weights",
    "backtest_perf",
    "risk_diagnostics",
    "variance_model_selection",
    "profiles",
    "user_risk_profiles",
    "user_universes",
    "user_filter_presets",
    "user_run_configs",
    "jobs",
    "chat_sessions",
    "chat_messages",
    "app_knowledge_base",
    "app_knowledge_chunks",
    "assistant_tool_audit",
]


def main() -> int:
    client = get_supabase_client()
    ok = True
    for table in TABLES:
        try:
            resp = client.table(table).select("*").limit(1).execute()
            print(f"{table}: ok rows={len(resp.data or [])}")
        except Exception as exc:
            ok = False
            print(f"{table}: missing/error {type(exc).__name__}: {str(exc)[:300]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

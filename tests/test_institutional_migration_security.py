from __future__ import annotations

from pathlib import Path

MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "supabase"
    / "migrations"
    / "20260612_003_institutional_publication_and_paper_execution.sql"
)


def test_execution_and_portfolio_tables_are_backend_write_only():
    source = MIGRATION.read_text(encoding="utf-8").lower()
    for policy in (
        "user portfolios owner read",
        "portfolio versions owner read",
        "order intents owner read",
        "pretrade decisions owner read",
        "paper fills owner read",
    ):
        assert policy in source
    assert 'create policy "order intents owner access"' not in source
    assert 'create policy "pretrade decisions owner access"' not in source
    assert 'create policy "paper fills owner access"' not in source


def test_atomic_promotion_rpc_is_not_executable_by_frontend_roles():
    source = MIGRATION.read_text(encoding="utf-8").lower()
    assert "revoke all on function public.promote_publication(uuid) from public, anon, authenticated" in source
    assert "grant execute on function public.promote_publication(uuid) to service_role" in source


def test_user_publication_pointer_is_tenant_scoped():
    source = MIGRATION.read_text(encoding="utf-8").lower()
    assert "pointer_key like 'user:' || auth.uid()::text || ':%'" in source
    assert "candidate.publication_kind" in source

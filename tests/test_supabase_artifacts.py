import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from supabase_store import (
    _artifact_storage_rows,
    persist_run_artifacts_local,
    reassemble_artifact_rows,
    save_run_artifacts,
    save_run_to_supabase,
    user_owner_key,
)


class _Response:
    def __init__(self, data=None):
        self.data = data or []


class _Query:
    def __init__(self, client, table):
        self.client = client
        self.table = table
        self.action = None
        self.payload = None

    def insert(self, payload):
        self.action = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.action = "update"
        self.payload = payload
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        self.client.events.append((self.table, self.action, self.payload))
        if self.table == "runs" and self.action == "insert":
            return _Response([{"run_id": "atomic-run"}])
        return _Response()


class _Client:
    def __init__(self):
        self.events = []

    def table(self, name):
        return _Query(self, name)


class SupabaseArtifactTests(unittest.TestCase):
    def test_native_dates_are_serialized_before_supabase_insert(self):
        rows = _artifact_storage_rows(
            "atomic-run",
            "dashboard_payload",
            {"event": {"date": date(2026, 6, 8)}},
        )
        self.assertEqual(rows[0]["artifact_json"]["event"]["date"], "2026-06-08")

    def test_large_dictionary_artifact_round_trips_through_section_rows(self):
        artifact = {
            "status": {"state": "ready"},
            "market_intelligence": {"history": ["x" * 800_000, "y" * 800_000]},
            "charts": {"price_paths": [{"Date": "2026-01-01", "Value": 100.0}]},
        }
        rows = _artifact_storage_rows("atomic-run", "dashboard_payload", artifact)
        self.assertGreater(len(rows), 1)
        self.assertTrue(rows[0]["artifact_json"]["_chunked"])
        self.assertTrue(
            any(
                row["artifact_name"].startswith("dashboard_payload::market_intelligence::history::part") for row in rows
            )
        )
        self.assertEqual(reassemble_artifact_rows(rows, "dashboard_payload"), artifact)

    def test_large_artifact_bundle_is_inserted_one_artifact_per_request(self):
        client = _Client()
        manifest = save_run_artifacts(
            client,
            "atomic-run",
            {
                "dashboard_payload": {"large": "x" * 10_000},
                "backtest_path_bundle": {"paths": [1, 2, 3]},
                "promotion_gate": {"promotion_status": "research-only"},
            },
        )
        inserts = [event for event in client.events if event[0] == "run_artifacts" and event[1] == "insert"]
        self.assertTrue(manifest["supabase_run_artifacts"])
        self.assertEqual(len(inserts), 3)
        self.assertTrue(all(len(event[2]) == 1 for event in inserts))

    def test_local_artifact_manifest_is_written(self):
        path, digest = persist_run_artifacts_local(
            "unit-test",
            {
                "dashboard_payload": {"status": {"x": pd.DataFrame({"A": [1]})}},
                "promotion_gate": {"promotion_status": "watchlist"},
            },
        )
        self.assertTrue(path.exists())
        self.assertGreater(len(digest), 20)
        text = path.read_text(encoding="utf-8")
        self.assertIn("dashboard_payload", text)
        try:
            path.unlink()
        except Exception:
            pass

    def test_owner_key_is_stable_case_insensitive_and_non_plaintext(self):
        with patch.dict(
            "os.environ",
            {"QPK_PORTFOLIO_OWNER_SECRET": "unit-test-secret-with-enough-entropy"},
            clear=False,
        ):
            first = user_owner_key("Chris")
            second = user_owner_key(" chris ")
            other = user_owner_key("analyst")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertNotIn("chris", first)
        self.assertEqual(len(first), 64)

    def test_run_is_published_only_after_artifacts_are_durable(self):
        client = _Client()
        results = {
            "dashboard_payload": {"status": {"promotion": "promoted"}},
            "portfolio": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "performance_summary": pd.DataFrame(),
        }
        with (
            patch("supabase_store.get_supabase_client", return_value=client),
            patch(
                "supabase_store.save_run_artifacts",
                return_value={"supabase_run_artifacts": True},
            ),
        ):
            run_id = save_run_to_supabase(results, {"benchmark_ticker": "SPY"}, status="completed")
        self.assertEqual(run_id, "atomic-run")
        self.assertEqual(client.events[0][2]["status"], "building")
        self.assertEqual(client.events[-1], ("runs", "update", {"status": "completed"}))

    def test_failed_artifact_persistence_never_publishes_run(self):
        client = _Client()
        with (
            patch("supabase_store.get_supabase_client", return_value=client),
            patch(
                "supabase_store.save_run_artifacts",
                return_value={"supabase_run_artifacts": False},
            ),
        ):
            with self.assertRaises(RuntimeError):
                save_run_to_supabase({}, {"benchmark_ticker": "SPY"}, status="completed")
        self.assertFalse(any(event[1] == "update" for event in client.events))


if __name__ == "__main__":
    unittest.main()

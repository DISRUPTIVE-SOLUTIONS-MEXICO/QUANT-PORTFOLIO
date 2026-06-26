import unittest
from pathlib import Path
from unittest.mock import patch

import cloud_jobs


class _Response:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.eq_filters = {}
        self.in_filters = {}
        self.like_filters = {}
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.eq_filters[key] = value
        return self

    def in_(self, key, values):
        self.in_filters[key] = list(values)
        return self

    def like(self, key, value):
        self.like_filters[key] = value
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        return _Response(self.client.rows_for(self))


class _FakeClient:
    def table(self, table_name):
        return _Query(self, table_name)

    def rows_for(self, query):
        if query.table_name == "publication_pointers":
            return [{"pointer_key": "global:daily_snapshot", "publication_id": "pub-daily", "updated_at": "2026-06-15"}]
        if query.table_name == "publication_manifests":
            return [
                {
                    "publication_id": "pub-daily",
                    "run_id": "run-daily",
                    "publication_kind": "daily_snapshot",
                    "activated_at": "2026-06-15",
                }
            ]
        if query.table_name == "runs":
            return [{"run_id": "run-full", "created_at": "2026-06-14"}]
        if query.table_name == "run_artifacts":
            if query.eq_filters.get("run_id") == "run-daily":
                return [
                    {
                        "run_id": "run-daily",
                        "artifact_name": "dashboard_payload",
                        "artifact_json": {"status": {"snapshot_meta": [{"Snapshot_Mode": "daily_price_snapshot"}]}},
                        "created_at": "2026-06-15",
                    }
                ]
            return [
                {
                    "run_id": "run-full",
                    "artifact_name": "dashboard_payload",
                    "artifact_json": {"tables": {"validation": [{"Metric": "WRC_p", "Value": 0.2}]}},
                    "created_at": "2026-06-14",
                }
            ]
        return []


class CloudJobsContractTests(unittest.TestCase):
    def test_cloud_job_helpers_exist(self):
        self.assertTrue(callable(cloud_jobs.create_optimization_job))
        self.assertTrue(callable(cloud_jobs.list_user_jobs))
        self.assertTrue(callable(cloud_jobs.get_job_status))
        self.assertTrue(callable(cloud_jobs.latest_dashboard_artifact))
        self.assertTrue(callable(cloud_jobs.latest_dashboard_artifacts))

    def test_dashboard_artifact_scope_distinguishes_daily_snapshot(self):
        payload = {
            "status": {"snapshot_meta": [{"Snapshot_Mode": "daily_price_snapshot"}]},
            "tables": {"validation": []},
        }
        self.assertEqual(cloud_jobs.dashboard_artifact_scope(payload), "daily_snapshot")

    def test_dashboard_artifact_scope_requires_analytical_evidence(self):
        payload = {
            "status": {"promotion_tests": [{"Test": "PBO", "Pass": True}]},
            "tables": {"validation": [{"Metric": "PBO", "Value": 0.04}]},
            "charts": {},
        }
        self.assertEqual(cloud_jobs.dashboard_artifact_scope(payload), "full_analysis")

    def test_daily_workflow_runs_market_overlay_at_seven_central(self):
        workflow = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily-cloud-refresh.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('if [ "$HOUR" != "07" ]', workflow)
        self.assertIn("--mode rigorous --save-supabase --require-supabase", workflow)
        self.assertNotIn("--full-pipeline --mode rigorous", workflow)
        self.assertIn("QPK_CLOUD_REFRESH_MAX_TICKERS", workflow)

    def test_daily_pointer_does_not_hide_existing_full_research_artifact(self):
        with patch.object(cloud_jobs, "get_supabase_client", return_value=_FakeClient()):
            artifacts = cloud_jobs.latest_dashboard_artifacts(scan_limit=5)
        self.assertEqual(artifacts["daily_snapshot"]["run_id"], "run-daily")
        self.assertEqual(artifacts["full_analysis"]["run_id"], "run-full")
        self.assertEqual(artifacts["latest_any"]["run_id"], "run-full")


if __name__ == "__main__":
    unittest.main()

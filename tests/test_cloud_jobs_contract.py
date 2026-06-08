import unittest
from pathlib import Path

import cloud_jobs


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
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "daily-cloud-refresh.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('if [ "$HOUR" != "07" ]', workflow)
        self.assertIn("--mode rigorous --save-supabase --require-supabase", workflow)
        self.assertNotIn("--full-pipeline --mode rigorous", workflow)
        self.assertIn("QPK_CLOUD_REFRESH_MAX_TICKERS", workflow)


if __name__ == "__main__":
    unittest.main()

import unittest

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


if __name__ == "__main__":
    unittest.main()

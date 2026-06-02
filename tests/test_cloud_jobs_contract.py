import unittest

import cloud_jobs


class CloudJobsContractTests(unittest.TestCase):
    def test_cloud_job_helpers_exist(self):
        self.assertTrue(callable(cloud_jobs.create_optimization_job))
        self.assertTrue(callable(cloud_jobs.list_user_jobs))
        self.assertTrue(callable(cloud_jobs.get_job_status))
        self.assertTrue(callable(cloud_jobs.latest_dashboard_artifact))


if __name__ == "__main__":
    unittest.main()


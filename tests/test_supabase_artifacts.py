import unittest

import pandas as pd

from supabase_store import persist_run_artifacts_local


class SupabaseArtifactTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()


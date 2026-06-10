import unittest
from unittest.mock import patch

import pandas as pd

import quant_stockpicker_core as core


class GeopoliticalThermometerTests(unittest.TestCase):
    def test_default_topics_are_english(self):
        self.assertIn("Trade / Tariffs", core.GEOPOLITICAL_TOPIC_QUERIES)
        self.assertIn("Primary Markets / IPOs", core.GEOPOLITICAL_TOPIC_QUERIES)
        self.assertFalse(any("/Mercado" in topic or "Aranceles" in topic for topic in core.GEOPOLITICAL_TOPIC_QUERIES))

    def test_robust_news_flow_diagnostics_uses_within_topic_baseline(self):
        s = pd.Series([10.0] * 40 + [13.0, 14.0, 15.0, 18.0])
        diag = core.robust_news_flow_diagnostics(s, min_obs=20, min_unique=4)
        self.assertEqual(diag["Score_Type"], "GDELT robust within-topic news-flow shock")
        self.assertGreater(diag["Robust_Z_Score"], 0)
        self.assertGreaterEqual(diag["Positive_Shock_Score"], 0)
        self.assertTrue(diag["Statistical_Admissibility"])
        self.assertTrue(diag["Risk_Overlay_Admissible"])

    def test_degenerate_timeline_is_not_forced_into_fake_z_score(self):
        s = pd.Series([8.0] * 30)
        diag = core.robust_news_flow_diagnostics(s, min_obs=20, min_unique=4)
        self.assertTrue(pd.isna(diag["Robust_Z_Score"]))
        self.assertEqual(diag["Thermometer"], "Insufficient history")
        self.assertFalse(diag["Statistical_Admissibility"])

    def test_rss_fallback_is_not_reported_as_z_score(self):
        topics = {"Trade": "tariffs", "Credit": "credit stress"}

        def fake_articles(query, days, max_records, use_cache, cache_ttl_hours):
            return pd.DataFrame(
                {
                    "Title": [f"{query} {i}" for i in range(max_records)],
                    "Domain": [f"source{i % 2}.com" for i in range(max_records)],
                    "Source": ["Google News RSS fallback"] * max_records,
                }
            )

        with patch.object(core, "fetch_gdelt_timeline", return_value=pd.DataFrame()):
            with patch.object(core, "fetch_gdelt_articles", side_effect=fake_articles):
                out = core.geopolitical_thermometer(topics, use_cache=False)

        summary = out["summary"]
        self.assertFalse(summary.empty)
        self.assertTrue(summary["Z_Score"].isna().all())
        self.assertTrue(summary["News_Flow_Score"].notna().all())
        self.assertTrue(summary["Score_Type"].str.contains("not a Z-score").all())
        self.assertTrue(summary["Latest_Volume"].isna().all())
        self.assertTrue((summary["Article_Count"] == 8).all())
        self.assertFalse(summary["Statistical_Admissibility"].fillna(False).any())

    def test_geopolitical_audit_rejects_raw_cross_topic_comparability(self):
        summary = pd.DataFrame(
            {
                "Topic": ["Primary Markets / IPOs", "Trade / Tariffs"],
                "Robust_Z_Score": [-3.0, pd.NA],
                "Positive_Shock_Score": [0.0, pd.NA],
                "Statistical_Admissibility": [True, False],
                "Risk_Overlay_Admissible": [False, False],
                "Data_Source_Type": ["GDELT_TIMELINE", "RSS_ARTICLE_FALLBACK"],
            }
        )
        audit = core.geopolitical_thermometer_model_audit(summary)
        values = audit.set_index("Metric")["Value"].to_dict()
        self.assertEqual(values["Model_State"], "No positive abnormal-attention shock")
        self.assertEqual(values["Fallback_Only_Topics"], 1)
        self.assertEqual(values["Raw_Cross_Topic_Comparability"], "Rejected")

    def test_country_heatmap_excludes_global_fallback_and_scores_source_countries(self):
        articles = pd.DataFrame(
            {
                "Topic": ["Trade / Tariffs", "Trade / Tariffs", "Wars / Security", "Trade / Tariffs"],
                "SourceCountry": ["United States", "Mexico", "United States", "global"],
                "Domain": ["a.com", "b.com", "c.com", "rss.com"],
                "Source": [
                    "GDELT DOC 2.1 artlist",
                    "GDELT DOC 2.1 artlist",
                    "GDELT DOC 2.1 artlist",
                    "Google News RSS fallback",
                ],
            }
        )
        summary = pd.DataFrame(
            {
                "Topic": ["Trade / Tariffs", "Wars / Security"],
                "Positive_Shock_Score": [1.5, 0.0],
                "News_Flow_Score": [pd.NA, pd.NA],
                "Statistical_Admissibility": [True, True],
                "Risk_Overlay_Admissible": [True, False],
            }
        )
        out = core.geopolitical_country_heatmap(articles, summary)
        self.assertEqual(set(out["Country"]), {"United States", "Mexico"})
        self.assertNotIn("global", set(out["Country"].str.lower()))
        self.assertIn("Geo_News_Attention_Score", out.columns)
        self.assertGreater(out["Geo_News_Attention_Score"].max(), 0)

    def test_country_heatmap_prefers_event_country_regex_over_publisher_country(self):
        articles = pd.DataFrame(
            {
                "Topic": ["Trade / Tariffs", "Trade / Tariffs", "Financial / Credit"],
                "Title": [
                    "Mexico tariffs hit auto supply chains",
                    "China export controls escalate chip tensions",
                    "Funding spreads tighten after central-bank meeting",
                ],
                "Query": ["tariffs", "export control", "credit stress"],
                "SourceCountry": ["global", "United States", "Brazil"],
                "Domain": ["rss.com", "us-source.com", "br-source.com"],
                "Source": ["Google News RSS fallback", "GDELT DOC 2.1 artlist", "GDELT DOC 2.1 artlist"],
            }
        )
        summary = pd.DataFrame(
            {
                "Topic": ["Trade / Tariffs", "Financial / Credit"],
                "Positive_Shock_Score": [1.0, 0.0],
                "News_Flow_Score": [pd.NA, pd.NA],
                "Statistical_Admissibility": [True, True],
                "Risk_Overlay_Admissible": [True, False],
            }
        )
        out = core.geopolitical_country_heatmap(articles, summary)
        countries = set(out["Country"])
        self.assertIn("Mexico", countries)
        self.assertIn("China", countries)
        self.assertIn("Brazil", countries)
        self.assertNotIn("global", {str(c).lower() for c in countries})
        self.assertNotIn("United States", countries)
        china = out.set_index("Country").loc["China"]
        self.assertIn("title_regex", china["Geo_Inference_Methods"])
        self.assertEqual(int(china["SourceCountry_Fallback_Count"]), 0)

    def test_country_regex_does_not_map_lowercase_us_pronoun_to_united_states(self):
        inferred = core.infer_geopolitical_event_countries(
            title="Analysts warn tariffs could hurt us if demand slows",
            query="tariffs",
            source_country="global",
        )
        self.assertEqual(inferred["Event_Countries"], [])
        self.assertEqual(inferred["Geo_Inference_Method"], "unresolved")


if __name__ == "__main__":
    unittest.main()

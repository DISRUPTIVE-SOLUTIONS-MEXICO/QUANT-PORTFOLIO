from __future__ import annotations

import numpy as np
import pandas as pd

from quant_stockpicker_core import objective_metric_name, xcdr_v3_sample_score


def test_xcdr_v3_rewards_upside_convexity_and_downside_control():
    rng = np.random.default_rng(41)
    benchmark = pd.Series(rng.normal(0.0004, 0.011, 504))
    controlled = pd.Series(
        np.where(
            benchmark >= 0.0,
            1.15 * benchmark + 0.00015,
            0.70 * benchmark + 0.00005,
        )
    )
    tail_heavy = pd.Series(
        np.where(
            benchmark >= 0.0,
            1.05 * benchmark,
            1.40 * benchmark - 0.00010,
        )
    )

    controlled_score = xcdr_v3_sample_score(controlled, benchmark)
    tail_heavy_score = xcdr_v3_sample_score(tail_heavy, benchmark)

    assert np.isfinite(controlled_score)
    assert np.isfinite(tail_heavy_score)
    assert controlled_score > tail_heavy_score


def test_xcdr_v3_has_an_explicit_validation_metric():
    assert objective_metric_name("xcdr_v3") == "XCDR_v3"

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


def test_xcdr_v3_constant_gain_series_pins_formula_exactly():
    # b constant and positive, p = 1.1 * b: no downside, no drawdown, zero
    # tracking error and a *negative* daily CVaR loss that must be clipped to
    # zero (an all-gain sample cannot shrink the risk denominator).
    b = pd.Series(np.full(504, 0.001))
    p = 1.1 * b
    # active_ann = 0.0001 * 252; upside_capture = 1.1; betas are NaN on
    # constant series so convexity falls back to uc - dc = 0.1.
    numerator = 0.0001 * 252.0 + 0.15 * 0.1 + 0.10 * 0.1
    expected = numerator / 1e-4
    score = xcdr_v3_sample_score(p, b)
    assert np.isclose(score, expected, rtol=1e-9)


def test_xcdr_v3_denominator_uses_annualized_clipped_cvar():
    from quant_stockpicker_core import beta_to_benchmark, historical_cvar_loss, max_drawdown

    rng = np.random.default_rng(7)
    b = pd.Series(rng.normal(0.0004, 0.011, 504))
    p = b + 0.0002

    active = p - b
    active_ann = float(active.mean() * 252.0)
    up = b > 0.0
    down = b < 0.0
    uc = float(p[up].mean() / b[up].mean())
    dc = float(p[down].mean() / b[down].mean())
    convexity = float(beta_to_benchmark(p[up], b[up]) - beta_to_benchmark(p[down], b[down]))
    downside = float(np.sqrt(np.mean(np.minimum(p.values, 0.0) ** 2)) * np.sqrt(252.0))
    cvar_ann = max(historical_cvar_loss(p.values, alpha=0.95), 0.0) * float(np.sqrt(252.0))
    drawdown = abs(float(max_drawdown(p)))
    te = float((p - b).std(ddof=1) * np.sqrt(252.0))
    denominator = 1e-4 + downside + 0.75 * cvar_ann + 0.50 * drawdown + 0.20 * te
    numerator = active_ann + 0.15 * (uc - 1.0) + 0.10 * convexity
    breach = max(dc - 1.0, 0.0)
    expected = numerator / denominator - 4.0 * breach**2

    assert np.isclose(xcdr_v3_sample_score(p, b), expected, rtol=1e-9)

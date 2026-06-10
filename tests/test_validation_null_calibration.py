"""Calibration of the data-snooping test stack under the null hypothesis.

The platform's value proposition *is* the validation layer, so the most
important unit test in the repository is the one proving that, when there is
no signal at all (i.i.d. Gaussian returns, trials independent of outcomes):

- White Reality Check / Hansen SPA p-values are roughly uniform (no
  systematic false rejections),
- the PBO proxy hovers around 0.5 (a random best trial ranks anywhere OOS),
- the Deflated Sortino with an honest trial count rarely rejects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_core.promotion_gate import effective_trial_count
from quant_stockpicker_core import (
    cpcv_pbo_diagnostics,
    deflated_sortino_diagnostics,
    white_reality_check_spa,
)

N_DATES = 60
N_TRIALS = 20
N_REPS = 25


def _null_opt_grid(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-31", periods=N_DATES, freq="ME")
    rows = []
    for trial in range(N_TRIALS):
        key = f"trial_{trial:02d}"
        for date in dates:
            rows.append(
                {
                    "Rebalance_Date": date,
                    "Trial_Key": key,
                    "Tickers": key,
                    "Sortino": rng.normal(0.0, 1.0),
                    "OOS_Equal_Return": rng.normal(0.0, 0.02),
                }
            )
    return pd.DataFrame(rows)


def test_white_reality_check_and_spa_pvalues_are_calibrated_under_null():
    wrc_pvals, spa_pvals = [], []
    for rep in range(N_REPS):
        grid = _null_opt_grid(seed=1_000 + rep)
        out = white_reality_check_spa(grid, samples=256, seed=42)
        metrics = out.set_index("Metric")["Value"]
        wrc_pvals.append(float(metrics["White_Reality_Check_PValue"]))
        spa_pvals.append(float(metrics["Hansen_SPA_PValue"]))
    wrc = np.asarray(wrc_pvals)
    spa = np.asarray(spa_pvals)
    # Under the null, p-values should be roughly uniform: their mean near 0.5
    # and the 5% rejection rate not wildly above nominal.
    assert 0.30 <= wrc.mean() <= 0.70, f"WRC p-values miscalibrated: mean={wrc.mean():.3f}"
    assert 0.30 <= spa.mean() <= 0.70, f"SPA p-values miscalibrated: mean={spa.mean():.3f}"
    assert (wrc < 0.05).mean() <= 0.20, f"WRC over-rejects under null: {(wrc < 0.05).mean():.2f}"
    assert (spa < 0.05).mean() <= 0.20, f"SPA over-rejects under null: {(spa < 0.05).mean():.2f}"


def test_pbo_proxy_is_near_one_half_under_null():
    pbos = []
    for rep in range(N_REPS):
        grid = _null_opt_grid(seed=2_000 + rep)
        out = cpcv_pbo_diagnostics(grid, n_folds=4)
        if not out.empty:
            pbos.append(float(out["PBO"].iloc[-1]))
    assert len(pbos) >= N_REPS // 2
    mean_pbo = float(np.mean(pbos))
    # With no signal the selected trial's OOS rank is uniform, so the
    # probability of landing in the bottom half is 0.5 in expectation.
    assert 0.25 <= mean_pbo <= 0.75, f"PBO miscalibrated under null: mean={mean_pbo:.3f}"


def test_deflated_sortino_rarely_rejects_under_null_with_honest_trials():
    rejections = []
    n_trials = effective_trial_count(logged_trials=50, lookback_grid_size=3, chunk_grid_size=3, bandit_arms=9)
    for rep in range(40):
        rng = np.random.default_rng(3_000 + rep)
        returns = pd.Series(rng.normal(0.0, 0.01, 24))
        out = deflated_sortino_diagnostics(returns, n_trials=n_trials)
        pval = float(out.set_index("Metric")["Value"]["Deflated_Sortino_PValue"])
        rejections.append(pval < 0.05)
    rate = float(np.mean(rejections))
    assert rate <= 0.10, f"DSR over-rejects under null with honest trial count: {rate:.2f}"


def test_effective_trial_count_is_monotone_and_floored():
    base = effective_trial_count(logged_trials=10)
    assert base == 10
    assert effective_trial_count() == 1
    with_bandit = effective_trial_count(logged_trials=10, bandit_arms=9)
    assert with_bandit == 90
    with_grid = effective_trial_count(logged_trials=0, lookback_grid_size=3, chunk_grid_size=3)
    assert with_grid == 9
    with_pso = effective_trial_count(logged_trials=10, pso_particles=4, pso_iterations=3, lambda_variants=20)
    assert with_pso == 10 + 12 + 20
    # More search effort can never reduce the deflation count.
    assert with_bandit >= base

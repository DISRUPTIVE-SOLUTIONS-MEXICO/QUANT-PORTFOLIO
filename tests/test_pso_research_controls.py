import math

import numpy as np
import pandas as pd

from run_xcdr_v3_parallel_research import (
    BatchConfig,
    _project_beta_gamma,
    apply_persistent_oos_overlay,
    benchmark_relative_drawdown_diagnostics,
    block_bootstrap_indices,
    causal_drawdown_vol_overlay,
    convex_opportunity_universe_builder,
    cvar_loss,
    defensive_overlay_anchor_returns,
    downside_ann,
    max_dd_loss,
    optimize_drawdown_budget_overlay,
    optimize_upside_recovery_overlay,
    parse_objective_list,
    pbo_window_proxy,
    project_weights,
    pso_beta_gamma_search,
    signal_weights,
    tail_throttle_beta_gamma,
    validation_tail_breach,
)


def test_project_beta_gamma_respects_caps_and_total_budget():
    beta, gamma = _project_beta_gamma(0.80, 0.60, beta_cap=0.50, gamma_cap=0.20, total_cap=0.55)
    assert 0.0 <= beta <= 0.50
    assert 0.0 <= gamma <= 0.20
    assert beta + gamma <= 0.55 + 1e-12


def test_parse_objective_list_preserves_frozen_order_and_dedupes():
    parsed = parse_objective_list("a,b,a; c ,,")
    assert parsed == ("a", "b", "c")


def test_block_bootstrap_indices_are_bounded_and_full_length():
    idx = block_bootstrap_indices(37, 5)
    assert len(idx) == 37
    assert idx.min() >= 0
    assert idx.max() < 37


def test_pbo_proxy_can_be_limited_to_frozen_promotion_family():
    rows = []
    for window in range(6):
        for objective, value in {
            "stable": 0.03 + 0.001 * window,
            "anchor": 0.02 + 0.001 * window,
            "unstable_trap": 0.10 if window < 3 else -0.10,
        }.items():
            rows.append({"test_start": f"2025-{window + 1:02d}-01", "objective": objective, "active_ann_return": value})
    data = pd.DataFrame(rows)
    full_pbo = pbo_window_proxy(data, objectives=("stable", "anchor", "unstable_trap"))
    frozen_pbo = pbo_window_proxy(data, objectives=("stable", "anchor"))
    assert frozen_pbo <= full_pbo
    assert frozen_pbo == 0.0


def test_pso_beta_gamma_search_is_bounded_and_deterministic():
    def score(beta: float, gamma: float) -> float:
        return -((beta - 0.20) ** 2 + (gamma - 0.05) ** 2)

    first, diag1 = pso_beta_gamma_search(
        score, beta_cap=0.40, gamma_cap=0.15, total_cap=0.45, particles=8, iterations=8, seed=123
    )
    second, diag2 = pso_beta_gamma_search(
        score, beta_cap=0.40, gamma_cap=0.15, total_cap=0.45, particles=8, iterations=8, seed=123
    )
    assert first == second
    assert diag1["pso_particles"] == diag2["pso_particles"] == 8
    assert diag1["pso_iterations"] == diag2["pso_iterations"] == 8
    assert math.isfinite(diag1["pso_best_score"])
    for beta, gamma in first:
        assert 0.0 <= beta <= 0.40
        assert 0.0 <= gamma <= 0.15
        assert beta + gamma <= 0.45 + 1e-12
    best_beta, best_gamma = min(first, key=lambda x: (x[0] - 0.20) ** 2 + (x[1] - 0.05) ** 2)
    assert abs(best_beta - 0.20) < 0.12
    assert abs(best_gamma - 0.05) < 0.08


def test_tail_throttle_reduces_validation_tail_breach():
    idx = pd.bdate_range("2025-01-01", periods=80)
    xi = pd.Series(np.tile([0.006, -0.005, 0.007, -0.004], 20), index=idx)
    capital = pd.Series(np.where(xi > 0, 0.75 * xi, 0.55 * xi), index=idx)
    growth = pd.Series(np.where(xi > 0, 1.50 * xi, 1.80 * xi), index=idx)
    alpha = pd.Series(np.where(xi > 0, 1.70 * xi, 2.20 * xi), index=idx)
    val = pd.DataFrame({"CAP": capital, "GRO": growth, "ALP": alpha}, index=idx)
    books = {
        "capital": pd.Series({"CAP": 1.0, "GRO": 0.0, "ALP": 0.0}),
        "growth": pd.Series({"CAP": 0.0, "GRO": 1.0, "ALP": 0.0}),
        "alpha": pd.Series({"CAP": 0.0, "GRO": 0.0, "ALP": 1.0}),
    }
    omega = pd.DataFrame({"XI": xi, "DEF": capital, "RISK": growth}, index=idx)
    cfg = BatchConfig(max_weight=1.0)
    raw_w = project_weights(0.20 * books["capital"] + 0.55 * books["growth"] + 0.25 * books["alpha"], cfg.max_weight)
    raw_breach = validation_tail_breach(val @ raw_w, xi, omega, weights=raw_w)["tail_breach"]
    beta, gamma, diag = tail_throttle_beta_gamma(0.55, 0.25, val, xi, omega, books, cfg, steps=10)
    throttled_w = project_weights(
        (1.0 - beta - gamma) * books["capital"] + beta * books["growth"] + gamma * books["alpha"], cfg.max_weight
    )
    throttled_breach = validation_tail_breach(val @ throttled_w, xi, omega, weights=throttled_w)["tail_breach"]
    assert throttled_breach <= raw_breach + 1e-12
    assert float(diag["tail_scale"]) <= 1.0


def test_enhanced_growth_weights_prefer_positive_convexity():
    idx = pd.bdate_range("2025-01-01", periods=120)
    xi = pd.Series(np.tile([0.004, 0.003, -0.003, 0.002, -0.002], 24), index=idx)
    good = pd.Series(np.where(xi > 0, 1.65 * xi + 0.0010, 0.45 * xi), index=idx)
    beta_trap = pd.Series(np.where(xi > 0, 1.80 * xi, 1.85 * xi - 0.0005), index=idx)
    defensive = pd.Series(np.where(xi > 0, 0.70 * xi, 0.45 * xi), index=idx)
    train = pd.DataFrame({"GOOD": good, "TRAP": beta_trap, "DEF": defensive}, index=idx)
    books = signal_weights(train, xi, max_weight=1.0)
    assert "growth_plus" in books
    assert books["growth_plus"]["GOOD"] > books["growth_plus"]["TRAP"]
    assert books["alpha_plus"]["GOOD"] > books["alpha_plus"]["TRAP"]


def test_tail_aware_growth_penalizes_benchmark_tail_beta():
    idx = pd.bdate_range("2025-01-01", periods=160)
    pattern = np.array([0.005, 0.004, 0.003, -0.004, 0.002, -0.018, 0.006, 0.004])
    xi = pd.Series(np.tile(pattern, 20), index=idx)
    convex = pd.Series(np.where(xi > 0, 1.45 * xi + 0.0008, 0.55 * xi + 0.0002), index=idx)
    tail_trap = pd.Series(np.where(xi > 0, 1.60 * xi + 0.0009, 2.40 * xi - 0.0010), index=idx)
    ballast = pd.Series(np.where(xi > 0, 0.45 * xi, 0.30 * xi), index=idx)
    train = pd.DataFrame({"CONVEX": convex, "TAILTRAP": tail_trap, "BALLAST": ballast}, index=idx)
    books = signal_weights(train, xi, max_weight=1.0)
    assert "growth_tail_aware" in books
    assert "alpha_tail_aware" in books
    assert "capital_xi_tail" in books
    assert books["growth_tail_aware"]["CONVEX"] > books["growth_tail_aware"]["TAILTRAP"]
    assert books["alpha_tail_aware"]["CONVEX"] > books["alpha_tail_aware"]["TAILTRAP"]
    assert books["growth_tail_convex"]["CONVEX"] > books["growth_tail_convex"]["TAILTRAP"]
    assert books["alpha_tail_convex"]["CONVEX"] > books["alpha_tail_convex"]["TAILTRAP"]
    assert books["growth_upside_convex"]["CONVEX"] > books["growth_upside_convex"]["TAILTRAP"]
    assert books["alpha_upside_convex"]["CONVEX"] > books["alpha_upside_convex"]["TAILTRAP"]
    assert books["growth_tail_convex"]["CONVEX"] >= books["growth_tail_aware"]["CONVEX"]
    assert books["alpha_tail_convex"]["CONVEX"] >= books["alpha_tail_aware"]["CONVEX"]
    assert books["growth_tail_convex"]["TAILTRAP"] < 1e-5
    assert books["growth_upside_convex"]["TAILTRAP"] < 1e-5
    assert books["capital_xi_tail"]["TAILTRAP"] < 1e-5


def test_fundamental_upside_convex_rewards_quality_without_overriding_tail_gate():
    idx = pd.bdate_range("2025-01-01", periods=160)
    pattern = np.array([0.005, 0.004, 0.003, -0.004, 0.002, -0.018, 0.006, 0.004])
    xi = pd.Series(np.tile(pattern, 20), index=idx)
    convex_high_quality = pd.Series(np.where(xi > 0, 1.45 * xi + 0.0008, 0.55 * xi + 0.0002), index=idx)
    convex_low_quality = pd.Series(np.where(xi > 0, 1.42 * xi + 0.0006, 0.58 * xi + 0.0001), index=idx)
    tail_trap = pd.Series(np.where(xi > 0, 1.60 * xi + 0.0009, 2.40 * xi - 0.0010), index=idx)
    train = pd.DataFrame(
        {
            "HIGHQ": convex_high_quality,
            "LOWQ": convex_low_quality,
            "TAILTRAP": tail_trap,
        },
        index=idx,
    )
    fundamental_score = pd.Series({"HIGHQ": 2.0, "LOWQ": -1.0, "TAILTRAP": 3.0})
    books = signal_weights(train, xi, max_weight=1.0, fundamental_score=fundamental_score)
    assert "growth_fundamental_upside_convex" in books
    assert "alpha_fundamental_upside_convex" in books
    assert books["growth_fundamental_upside_convex"]["HIGHQ"] > books["growth_fundamental_upside_convex"]["LOWQ"]
    assert books["alpha_fundamental_upside_convex"]["HIGHQ"] > books["alpha_fundamental_upside_convex"]["LOWQ"]
    assert books["growth_fundamental_upside_convex"]["TAILTRAP"] < 1e-5


def test_real_upside_convex_prefers_rally_capture_without_tail_trap():
    idx = pd.bdate_range("2025-01-01", periods=180)
    pattern = np.array([0.007, 0.005, 0.004, -0.003, 0.002, -0.012, 0.006, 0.005, -0.002])
    xi = pd.Series(np.tile(pattern, 20), index=idx)
    rally_convex = pd.Series(
        np.where(xi > 0.004, 1.75 * xi + 0.0010, np.where(xi < 0, 0.50 * xi, 1.05 * xi)), index=idx
    )
    plain_upside = pd.Series(np.where(xi > 0, 1.05 * xi + 0.0002, 0.60 * xi), index=idx)
    tail_trap = pd.Series(
        np.where(xi > 0.004, 1.90 * xi + 0.0012, np.where(xi < 0, 2.20 * xi - 0.0010, 1.10 * xi)), index=idx
    )
    train = pd.DataFrame({"RALLY": rally_convex, "PLAIN": plain_upside, "TAILTRAP": tail_trap}, index=idx)
    fundamental_score = pd.Series({"RALLY": 1.0, "PLAIN": 0.5, "TAILTRAP": 3.0})
    books = signal_weights(train, xi, max_weight=1.0, fundamental_score=fundamental_score)
    assert "growth_real_upside_convex" in books
    assert "growth_fundamental_real_upside_convex" in books
    assert books["growth_real_upside_convex"]["RALLY"] > books["growth_real_upside_convex"]["PLAIN"]
    assert (
        books["growth_fundamental_real_upside_convex"]["RALLY"]
        > books["growth_fundamental_real_upside_convex"]["PLAIN"]
    )
    assert books["growth_real_upside_convex"]["TAILTRAP"] < 1e-5
    assert books["growth_fundamental_real_upside_convex"]["TAILTRAP"] < 1e-5


def test_convex_opportunity_universe_builder_prefers_convex_fundamental_liquid_names():
    idx = pd.bdate_range("2024-01-01", periods=260)
    pattern = np.array([0.006, 0.004, 0.003, -0.003, 0.002, -0.012, 0.006, -0.002, 0.005, 0.003])
    xi = pd.Series(np.tile(pattern, 26), index=idx)
    convex = pd.Series(np.where(xi > 0.003, 1.55 * xi + 0.0008, np.where(xi < 0, 0.55 * xi, xi)), index=idx)
    plain = pd.Series(np.where(xi > 0, 0.95 * xi, 0.70 * xi), index=idx)
    tail = pd.Series(np.where(xi > 0.003, 1.70 * xi + 0.0008, np.where(xi < 0, 2.10 * xi - 0.0008, xi)), index=idx)
    noisy = pd.Series(np.sin(np.arange(len(idx)) / 7.0) * 0.002, index=idx)
    train = pd.DataFrame({"CONVEX": convex, "PLAIN": plain, "TAIL": tail, "NOISY": noisy}, index=idx)
    volumes = pd.DataFrame(
        {
            "CONVEX": np.full(len(idx), 2_000_000.0),
            "PLAIN": np.full(len(idx), 1_500_000.0),
            "TAIL": np.full(len(idx), 2_000_000.0),
            "NOISY": np.full(len(idx), 10_000.0),
        },
        index=idx,
    )
    fundamentals = pd.DataFrame(
        {
            "Fundamental_Upside_Score": [2.0, 0.0, 2.5, -1.0],
            "Fundamental_PIT_Confidence": [0.9, 0.6, 0.9, 0.2],
        },
        index=["CONVEX", "PLAIN", "TAIL", "NOISY"],
    )
    selected, table = convex_opportunity_universe_builder(
        train, volumes, list(train.columns), xi, fundamentals, limit=2
    )
    assert "CONVEX" in selected
    assert table.loc["CONVEX", "Opportunity_Score"] > table.loc["PLAIN", "Opportunity_Score"]
    assert bool(table.loc["TAIL", "Tail_Admissible"]) is False


def test_drawdown_overlay_is_causal_with_respect_to_future_returns():
    idx = pd.bdate_range("2025-01-01", periods=60)
    raw = pd.Series(np.r_[np.full(30, 0.001), np.full(30, -0.02)], index=idx)
    anchor = pd.Series(0.0001, index=idx)
    state = {"state_stress": 0.65, "state_risk_on": 0.0, "state_recovery": 0.0}
    base, base_diag = causal_drawdown_vol_overlay(raw, anchor, state)
    shocked = raw.copy()
    shocked.iloc[35:] = 0.03
    shocked_path, shocked_diag = causal_drawdown_vol_overlay(shocked, anchor, state)
    pd.testing.assert_series_equal(base.iloc[:35], shocked_path.iloc[:35])
    assert base_diag["overlay_min_exposure"] < 1.0
    assert 0.0 <= base_diag["overlay_trigger_rate"] <= 1.0


def test_persistent_oos_overlay_updates_only_target_objective():
    idx = pd.bdate_range("2025-01-01", periods=50)
    rows = []
    for obj in ["enhanced_growth_anchor_dd_control_policy", "control"]:
        for i, dt in enumerate(idx):
            raw = -0.02 if i > 20 and obj.startswith("enhanced") else 0.001
            rows.append(
                {
                    "date": dt,
                    "test_start": idx[0],
                    "objective": obj,
                    "portfolio_return": raw,
                    "raw_portfolio_return": raw,
                    "anchor_return": 0.0,
                    "xi_return": 0.0,
                    "active_return": raw,
                    "state_stress": 0.8,
                    "state_risk_on": 0.0,
                    "state_recovery": 0.0,
                }
            )
    daily = pd.DataFrame(rows)
    out = apply_persistent_oos_overlay(daily, "enhanced_growth_anchor_dd_control_policy")
    target = out[out["objective"] == "enhanced_growth_anchor_dd_control_policy"]
    control = out[out["objective"] == "control"]
    assert target["persistent_overlay_exposure"].dropna().min() < 1.0
    assert control["portfolio_return"].eq(0.001).all()


def test_crash_budget_uses_lagged_market_pressure():
    idx = pd.bdate_range("2025-01-01", periods=40)
    rows = []
    for i, dt in enumerate(idx):
        rows.append(
            {
                "date": dt,
                "test_start": idx[0],
                "objective": "enhanced_growth_anchor_crash_budget_policy",
                "portfolio_return": 0.002,
                "raw_portfolio_return": 0.002,
                "anchor_return": 0.0001,
                "xi_return": 0.001,
                "active_return": 0.001,
                "state_stress": 0.5,
                "state_risk_on": 0.0,
                "state_recovery": 0.0,
                "market_down_breadth": 1.0 if i == 10 else 0.0,
                "market_tail_breadth": 0.5 if i == 10 else 0.0,
                "market_dispersion": 0.05 if i == 10 else 0.0,
            }
        )
    daily = pd.DataFrame(rows)
    out = apply_persistent_oos_overlay(daily, "enhanced_growth_anchor_crash_budget_policy")
    exp = out["persistent_overlay_exposure"].dropna().reset_index(drop=True)
    pressure = out["persistent_crash_pressure"].dropna().reset_index(drop=True)
    assert pressure.iloc[10] == 0.0
    assert pressure.iloc[11] > 0.55
    assert exp.iloc[11] < exp.iloc[10]


def test_drawdown_budget_optimizer_reduces_validation_tail_breach():
    idx = pd.bdate_range("2025-01-01", periods=90)
    xi = pd.Series(np.r_[np.full(35, 0.002), np.full(12, -0.006), np.full(43, 0.0015)], index=idx)
    raw = pd.Series(np.r_[np.full(35, 0.003), np.full(12, -0.018), np.full(43, 0.0022)], index=idx)
    anchor = pd.Series(np.r_[np.full(35, 0.0007), np.full(12, -0.001), np.full(43, 0.0006)], index=idx)
    state = {"state_stress": 0.75, "state_risk_on": 0.0, "state_recovery": 0.0}
    params, diag = optimize_drawdown_budget_overlay(raw, anchor, xi, state, tolerance=0.03)
    adjusted, _ = causal_drawdown_vol_overlay(
        raw,
        anchor,
        state,
        benchmark_returns=xi,
        min_exposure=params["budget_min_exposure"],
        rerisk_step=params["budget_rerisk_step"],
        dd_soft_shift=params["budget_dd_soft_shift"],
        dd_hard_gap=params["budget_dd_hard_gap"],
        vol_target_shift=params["budget_vol_target_shift"],
    )
    assert params["budget_min_exposure"] <= 0.45
    assert diag["budget_breach"] <= validation_tail_breach(raw, xi, pd.DataFrame({"XI": xi}, index=idx))["tail_breach"]
    assert max_dd_loss(adjusted) <= max_dd_loss(raw) + 1e-12


def test_upside_recovery_overlay_uses_validation_budget_before_rerisking():
    idx = pd.bdate_range("2025-01-01", periods=90)
    xi = pd.Series(np.tile([0.004, 0.003, -0.003, 0.002, -0.002], 18), index=idx)
    raw = pd.Series(np.where(xi > 0, 1.20 * xi + 0.0004, 0.72 * xi), index=idx)
    anchor = pd.Series(np.where(xi > 0, 0.45 * xi, 0.25 * xi), index=idx)
    state = {"state_stress": 0.35, "state_risk_on": 0.55, "state_recovery": 0.25}
    params, diag = optimize_upside_recovery_overlay(raw, anchor, xi, state, tolerance=0.04)
    adjusted, _ = causal_drawdown_vol_overlay(
        raw,
        anchor,
        state,
        benchmark_returns=xi,
        min_exposure=params["budget_min_exposure"],
        rerisk_step=params["budget_rerisk_step"],
        dd_soft_shift=params["budget_dd_soft_shift"],
        dd_hard_gap=params["budget_dd_hard_gap"],
        vol_target_shift=params["budget_vol_target_shift"],
    )
    assert diag["budget_active_ann_return"] > 0.0
    assert diag["budget_downside_capture"] < 0.98
    assert downside_ann(adjusted) <= downside_ann(xi) * 1.04 + 1e-12
    assert cvar_loss(adjusted) <= cvar_loss(xi) * 1.04 + 1e-12


def test_benchmark_relative_drawdown_budget_is_pathwise():
    idx = pd.bdate_range("2025-01-01", periods=70)
    xi = pd.Series(np.r_[np.full(30, 0.001), np.full(10, -0.006), np.full(30, 0.001)], index=idx)
    raw = pd.Series(np.r_[np.full(30, 0.0012), np.full(10, -0.020), np.full(30, 0.0012)], index=idx)
    anchor = pd.Series(0.0001, index=idx)
    state = {"state_stress": 0.80, "state_risk_on": 0.0, "state_recovery": 0.0}
    plain, _ = causal_drawdown_vol_overlay(raw, anchor, state)
    relative, diag = causal_drawdown_vol_overlay(raw, anchor, state, benchmark_returns=xi)
    plain_path = benchmark_relative_drawdown_diagnostics(plain, xi)
    relative_path = benchmark_relative_drawdown_diagnostics(relative, xi)
    assert relative_path["path_dd_breach_area"] <= plain_path["path_dd_breach_area"] + 1e-12
    assert relative_path["path_dd_max_excess"] <= plain_path["path_dd_max_excess"] + 1e-12
    assert diag["overlay_path_budget_trigger_rate"] > 0.0


def test_defensive_overlay_anchor_prefers_reference_anchor():
    idx = pd.bdate_range("2025-01-01", periods=5)
    frame = pd.DataFrame({"EQ": [0.01, -0.02, 0.01, -0.01, 0.01], "SHY": [0.0001] * 5}, index=idx)
    books = {
        "capital": pd.Series({"EQ": 1.0, "SHY": 0.0}),
        "reference_anchor": pd.Series({"SHY": 1.0}),
    }
    anchor_returns, kind = defensive_overlay_anchor_returns(frame, books)
    assert kind == "reference_anchor"
    assert np.allclose(anchor_returns.to_numpy(), 0.0001)

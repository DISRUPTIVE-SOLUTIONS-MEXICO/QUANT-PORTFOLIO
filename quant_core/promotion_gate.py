from __future__ import annotations

import numpy as np
import pandas as pd


def effective_trial_count(
    *,
    logged_trials: int = 0,
    lookback_grid_size: int = 1,
    chunk_grid_size: int = 1,
    objectives_evaluated: int = 1,
    bandit_arms: int = 0,
    pso_particles: int = 0,
    pso_iterations: int = 0,
    lambda_variants: int = 0,
) -> int:
    """Effective number of strategy evaluations for multiple-testing deflation.

    The Deflated Sortino/Sharpe adjustment is only honest if ``n_trials``
    counts *every* candidate examined, not just the rows that were logged:
    the lookback x chunk grid, each objective the meta-layer may select, each
    Kaizen bandit arm, every PSO particle evaluation, and any scalarization
    lambda variants explored. Bandit arms multiply the search space (the
    bandit chooses among objective configurations); PSO and lambda variants
    add candidate evaluations on top.
    """
    grid = max(int(lookback_grid_size), 1) * max(int(chunk_grid_size), 1)
    base = max(int(logged_trials), grid)
    base = max(base, 1) * max(int(objectives_evaluated), 1) * max(int(bandit_arms), 1)
    extra = max(int(pso_particles), 0) * max(int(pso_iterations), 0) + max(int(lambda_variants), 0)
    return int(max(base + extra, 1))


def _validation_metric(validation: dict[str, pd.DataFrame], metric: str) -> float:
    summary = validation.get("summary", pd.DataFrame()) if isinstance(validation, dict) else pd.DataFrame()
    if summary.empty or not {"Metric", "Value"}.issubset(summary.columns):
        return np.nan
    row = summary[summary["Metric"].astype(str).eq(metric)]
    if row.empty:
        return np.nan
    try:
        return float(row.iloc[0]["Value"])
    except Exception:
        return np.nan


def evaluate_promotion_gate(
    performance_summary: pd.DataFrame,
    validation: dict[str, pd.DataFrame],
    config,
    pbo_max: float = 0.10,
    spa_alpha: float = 0.05,
    wrc_alpha: float = 0.05,
    min_icir: float = 0.0,
) -> dict:
    dsr = _validation_metric(validation, "Deflated_Sortino")
    pbo = _validation_metric(validation, "CPCV_PBO")
    spa = _validation_metric(validation, "Hansen_SPA_PValue")
    wrc = _validation_metric(validation, "White_Reality_Check_PValue")
    icir = _validation_metric(validation, "ICIR")
    if pd.isna(icir):
        icir = _validation_metric(validation, "Mean_IC")
    dxcdr = _validation_metric(validation, "DXCDR")
    qlike_delta = _validation_metric(validation, "OOS_QLIKE_Delta")
    upside_capture = _validation_metric(validation, "Upside_Capture")
    downside_capture = _validation_metric(validation, "Downside_Capture")
    return_gap = _validation_metric(validation, "Return_Gap_to_Xi")
    downside_preservation = _validation_metric(validation, "Downside_Preservation_Pass")
    max_dd = _perf_metric(performance_summary, "Max_Drawdown")
    cvar95 = _perf_metric(performance_summary, "CVaR_95_Daily")
    dd_limit = (
        abs(float(getattr(config, "investor_max_drawdown", np.nan)))
        if pd.notna(getattr(config, "investor_max_drawdown", np.nan))
        else np.nan
    )
    cvar_limit = (
        abs(float(getattr(config, "investor_cvar_max_daily", np.nan)))
        if pd.notna(getattr(config, "investor_cvar_max_daily", np.nan))
        else np.nan
    )
    checks = [
        ("Deflated Sortino", dsr, 0.0, bool(pd.notna(dsr) and dsr > 0.0), "higher"),
        ("PBO", pbo, pbo_max, bool(pd.notna(pbo) and pbo < pbo_max), "lower"),
        ("Hansen SPA p-value", spa, spa_alpha, bool(pd.notna(spa) and spa < spa_alpha), "lower"),
        ("White Reality Check p-value", wrc, wrc_alpha, bool(pd.notna(wrc) and wrc < wrc_alpha), "lower"),
        ("ICIR", icir, min_icir, bool(pd.notna(icir) and icir > min_icir), "higher"),
        ("DXCDR", dxcdr, 0.0, bool(pd.notna(dxcdr) and dxcdr > 0.0), "higher"),
        ("OOS QLIKE delta", qlike_delta, 0.0, bool(pd.notna(qlike_delta) and qlike_delta < 0.0), "lower"),
    ]
    if pd.notna(dd_limit):
        checks.append(
            ("Max drawdown", max_dd, dd_limit, bool(pd.notna(max_dd) and abs(max_dd) <= dd_limit), "lower_abs")
        )
    if pd.notna(cvar_limit):
        checks.append(
            ("Daily CVaR 95%", cvar95, cvar_limit, bool(pd.notna(cvar95) and abs(cvar95) <= cvar_limit), "lower_abs")
        )
    rows = []
    for name, observed, threshold, passed, direction in checks:
        rows.append(
            {
                "Test": name,
                "Observed": observed,
                "Threshold": threshold,
                "Direction": direction,
                "Passed": bool(passed),
            }
        )
    tests = pd.DataFrame(rows)
    valid_tests = tests["Observed"].notna()
    failed = tests[valid_tests & ~tests["Passed"]].copy()
    if valid_tests.sum() == 0:
        status = "watchlist"
    elif failed.empty and valid_tests.sum() >= 2:
        status = "promoted"
    else:
        status = "rejected"
    score = float(tests.loc[valid_tests, "Passed"].mean()) if valid_tests.any() else 0.0
    return {
        "promotion_status": status,
        "tests": tests,
        "failed_tests": failed,
        "validation_score": score,
        "approved_objective": getattr(config, "weight_objective", None) if status == "promoted" else None,
        "summary": pd.DataFrame(
            [
                {
                    "Promotion_Status": status,
                    "Validation_Score": score,
                    "Approved_Objective": getattr(config, "weight_objective", None) if status == "promoted" else None,
                    "Failed_Tests": ", ".join(failed["Test"].astype(str).tolist()) if not failed.empty else "",
                    "Upside_Capture": upside_capture,
                    "Downside_Capture": downside_capture,
                    "Return_Gap_to_Xi": return_gap,
                    "Downside_Preservation_Pass": downside_preservation,
                }
            ]
        ),
        "diagnostics": pd.DataFrame(
            [
                {"Metric": "Upside_Capture", "Value": upside_capture},
                {"Metric": "Downside_Capture", "Value": downside_capture},
                {"Metric": "Return_Gap_to_Xi", "Value": return_gap},
                {"Metric": "Downside_Preservation_Pass", "Value": downside_preservation},
                # Romano-Wolf FWER control over the trial family: informative
                # for now; becomes a blocking check once 12+ windows feed it.
                {
                    "Metric": "RomanoWolf_Rejected_5pct",
                    "Value": _validation_metric(validation, "RomanoWolf_Rejected_5pct"),
                },
                {
                    "Metric": "RomanoWolf_Best_Trial_Adj_PValue",
                    "Value": _validation_metric(validation, "RomanoWolf_Best_Trial_Adj_PValue"),
                },
            ]
        ),
    }


def _perf_metric(performance_summary: pd.DataFrame, metric: str) -> float:
    if performance_summary is None or performance_summary.empty:
        return np.nan
    if {"Metric", "Value"}.issubset(performance_summary.columns):
        row = performance_summary[performance_summary["Metric"].astype(str).eq(metric)]
        if not row.empty:
            try:
                return float(row.iloc[0]["Value"])
            except Exception:
                return np.nan
    if metric in performance_summary.columns:
        try:
            return float(pd.to_numeric(performance_summary[metric], errors="coerce").dropna().iloc[-1])
        except Exception:
            return np.nan
    return np.nan

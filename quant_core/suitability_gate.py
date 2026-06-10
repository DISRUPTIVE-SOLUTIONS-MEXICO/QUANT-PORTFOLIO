from __future__ import annotations

import numpy as np
import pandas as pd


def _metric_value(summary: pd.DataFrame, names: tuple[str, ...]) -> float:
    if summary is None or summary.empty or not {"Metric", "Value"}.issubset(summary.columns):
        return np.nan
    lookup = dict(zip(summary["Metric"].astype(str), summary["Value"], strict=False))
    for name in names:
        if name in lookup:
            try:
                return float(lookup[name])
            except Exception:
                return np.nan
    return np.nan


def evaluate_suitability_gate(
    config, portfolio: pd.DataFrame, performance_summary: pd.DataFrame, diagnostics: pd.DataFrame | None = None
) -> dict:
    vol = _metric_value(performance_summary, ("Annualized_Vol", "Ann_Vol", "Volatility"))
    cvar = _metric_value(performance_summary, ("CVaR_95", "Hist_CVaR_95_Daily", "CVaR"))
    max_dd = _metric_value(performance_summary, ("Max_Drawdown", "Maximum_Drawdown"))
    if pd.isna(vol) and portfolio is not None and not portfolio.empty and "realized_weight_ann_vol" in portfolio:
        vol = (
            pd.to_numeric(portfolio["realized_weight_ann_vol"], errors="coerce").dropna().iloc[0]
            if pd.to_numeric(portfolio["realized_weight_ann_vol"], errors="coerce").notna().any()
            else np.nan
        )
    if pd.isna(cvar) and portfolio is not None and not portfolio.empty and "hist_cvar_95_daily" in portfolio:
        cvar = (
            pd.to_numeric(portfolio["hist_cvar_95_daily"], errors="coerce").dropna().iloc[0]
            if pd.to_numeric(portfolio["hist_cvar_95_daily"], errors="coerce").notna().any()
            else np.nan
        )
    if (
        pd.isna(max_dd)
        and portfolio is not None
        and not portfolio.empty
        and "realized_weight_max_drawdown" in portfolio
    ):
        max_dd = (
            pd.to_numeric(portfolio["realized_weight_max_drawdown"], errors="coerce").dropna().iloc[0]
            if pd.to_numeric(portfolio["realized_weight_max_drawdown"], errors="coerce").notna().any()
            else np.nan
        )

    limits = {
        "Vol_Max": getattr(config, "target_vol", np.nan),
        "CVaR_Max_Daily": getattr(config, "investor_cvar_max_daily", np.nan),
        "DD_Max": getattr(config, "investor_max_drawdown", np.nan),
        "Max_Weight": getattr(config, "max_weight", np.nan),
    }
    weights = (
        pd.to_numeric(portfolio.get("Weight", pd.Series(dtype=float)), errors="coerce")
        if portfolio is not None and not portfolio.empty
        else pd.Series(dtype=float)
    )
    breaches = []
    if pd.notna(limits["Vol_Max"]) and pd.notna(vol) and vol > limits["Vol_Max"]:
        breaches.append({"Constraint": "Volatility", "Observed": vol, "Limit": limits["Vol_Max"], "Severity": "hard"})
    if pd.notna(limits["CVaR_Max_Daily"]) and pd.notna(cvar) and cvar > limits["CVaR_Max_Daily"]:
        breaches.append(
            {"Constraint": "Daily CVaR", "Observed": cvar, "Limit": limits["CVaR_Max_Daily"], "Severity": "hard"}
        )
    if pd.notna(limits["DD_Max"]) and pd.notna(max_dd) and abs(max_dd) > limits["DD_Max"]:
        breaches.append(
            {"Constraint": "Max drawdown", "Observed": max_dd, "Limit": -abs(limits["DD_Max"]), "Severity": "hard"}
        )
    if pd.notna(limits["Max_Weight"]) and not weights.empty and weights.max() > limits["Max_Weight"] + 1e-8:
        breaches.append(
            {
                "Constraint": "Single-name weight",
                "Observed": float(weights.max()),
                "Limit": limits["Max_Weight"],
                "Severity": "hard",
            }
        )
    if bool(getattr(config, "suitability_hard_block", False)):
        breaches.append({"Constraint": "Profile coherence", "Observed": 1.0, "Limit": 0.0, "Severity": "hard"})

    breach_df = pd.DataFrame(breaches)
    status = "blocked" if not breach_df.empty and breach_df["Severity"].eq("hard").any() else "approved"
    if diagnostics is not None and not diagnostics.empty:
        diagnostics = diagnostics.copy()
        diagnostics["Gate_Status"] = status
    summary = pd.DataFrame(
        [
            {
                "Gate_Status": status,
                "Observed_Volatility": vol,
                "Observed_CVaR": cvar,
                "Observed_Max_Drawdown": max_dd,
                "Vol_Max": limits["Vol_Max"],
                "CVaR_Max_Daily": limits["CVaR_Max_Daily"],
                "DD_Max": limits["DD_Max"],
                "Max_Weight": limits["Max_Weight"],
                "Breach_Count": int(len(breach_df)),
            }
        ]
    )
    return {
        "status": status,
        "summary": summary,
        "breaches": breach_df,
        "allowed_constraints": limits,
        "user_safe_summary": (
            "Portfolio blocked by suitability constraints. Review risk limits before presenting an allocation."
            if status == "blocked"
            else "Portfolio passed the current suitability constraints."
        ),
        "diagnostics": diagnostics if diagnostics is not None else pd.DataFrame(),
    }

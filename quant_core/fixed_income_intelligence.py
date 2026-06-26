from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_core.contracts import FixedIncomeIntelligenceV1


@dataclass(frozen=True)
class FixedIncomeIntelligenceConfig:
    """Frozen conventions for public-data term-structure diagnostics."""

    minimum_history_observations: int = 63
    factor_history_rows_per_country: int = 756
    reference_history_days: int = 756
    stale_warning_days: int = 31
    rate_volatility_observations: int = 252


TENOR_LABELS = {
    "POLICY_RATE": "Policy",
    "SOV_SHORT": "Short",
    "SOV_2Y": "2Y",
    "SOV_10Y": "10Y",
}


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _decision_date(
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
    reference_rates: pd.DataFrame,
    as_of: pd.Timestamp | str | None,
) -> pd.Timestamp | None:
    if as_of is not None:
        return _utc_timestamp(as_of)
    candidates: list[pd.Timestamp] = []
    for frame, columns in (
        (snapshot, ("Latest_Date", "Policy_Observation_Date")),
        (history, ("Observation_Date",)),
        (reference_rates, ("Observation_Date", "Latest_Observation_Date")),
    ):
        for column in columns:
            if column not in frame:
                continue
            values = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
            if not values.empty:
                candidates.append(pd.Timestamp(values.max()))
    return max(candidates) if candidates else None


def _normalize_snapshot(snapshot: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    out = _frame(snapshot)
    if out.empty or "Country" not in out:
        return pd.DataFrame()
    for column in ("Latest_Date", "Policy_Observation_Date"):
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
    if "Latest_Date" in out:
        out = out.loc[out["Latest_Date"].isna() | (out["Latest_Date"] <= decision_date)]
    numeric_columns = (
        "Policy_Rate",
        "Yield_Short",
        "Yield_2Y",
        "Yield_10Y",
        "Curve_10Y_2Y",
        "Term_Premium_Proxy",
    )
    for column in numeric_columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("Country").drop_duplicates("Country", keep="last").reset_index(drop=True)


def _normalize_history(history: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    out = _frame(history)
    required = {"Country", "Observation_Date", "Tenor_Code", "Rate"}
    if out.empty or not required.issubset(out.columns):
        return pd.DataFrame()
    out["Observation_Date"] = pd.to_datetime(out["Observation_Date"], errors="coerce", utc=True)
    out["Rate"] = pd.to_numeric(out["Rate"], errors="coerce")
    out = out.dropna(subset=["Country", "Observation_Date", "Tenor_Code", "Rate"])
    out = out.loc[out["Observation_Date"] <= decision_date]
    return out.sort_values(["Country", "Observation_Date", "Tenor_Code"]).reset_index(drop=True)


def _normalize_reference_rates(reference_rates: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    out = _frame(reference_rates)
    if out.empty or "Observation_Date" not in out or "Rate" not in out:
        return pd.DataFrame()
    out["Observation_Date"] = pd.to_datetime(out["Observation_Date"], errors="coerce", utc=True)
    out["Rate"] = pd.to_numeric(out["Rate"], errors="coerce")
    out = out.dropna(subset=["Observation_Date", "Rate"])
    return out.loc[out["Observation_Date"] <= decision_date].sort_values("Observation_Date").reset_index(drop=True)


def _last_observed_change(series: pd.Series, calendar_days: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return np.nan
    target = pd.Timestamp(clean.index[-1]) - pd.Timedelta(days=calendar_days)
    prior = clean.loc[clean.index <= target]
    return float(clean.iloc[-1] - prior.iloc[-1]) if not prior.empty else np.nan


def _curve_state(level: float, slope: float) -> str:
    if not np.isfinite(slope):
        return "Insufficient tenors"
    if slope < -0.10:
        return "Inverted"
    if slope > 1.00:
        return "Steep"
    if slope > 0.10:
        return "Upward sloping"
    return "Flat"


def _latest_curve_value(snapshot_row: pd.Series, factor_row: pd.Series, name: str) -> float:
    candidate = pd.to_numeric(snapshot_row.get(name, np.nan), errors="coerce")
    if not np.isfinite(candidate):
        candidate = pd.to_numeric(factor_row.get(name, np.nan), errors="coerce")
    return float(candidate)


def _factor_history(history: pd.DataFrame, cfg: FixedIncomeIntelligenceConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if history.empty:
        return pd.DataFrame()
    for country, group in history.groupby("Country", sort=True):
        pivot = group.pivot_table(
            index="Observation_Date",
            columns="Tenor_Code",
            values="Rate",
            aggfunc="last",
        ).sort_index()
        # This is event-time last-observation-carried-forward, not daily interpolation.
        pivot = pivot.ffill()
        two_year = pivot.get("SOV_2Y", pd.Series(np.nan, index=pivot.index, dtype=float))
        ten_year = pivot.get("SOV_10Y", pd.Series(np.nan, index=pivot.index, dtype=float))
        policy = pivot.get("POLICY_RATE", pd.Series(np.nan, index=pivot.index, dtype=float))
        short = pivot.get("SOV_SHORT", policy)
        factors = pd.DataFrame(index=pivot.index)
        factors["Country"] = str(country)
        factors["Level_Factor"] = pd.concat([two_year, ten_year], axis=1).mean(axis=1, skipna=False)
        factors["Slope_10Y_2Y"] = ten_year - two_year
        factors["Policy_Gap_2Y"] = two_year - policy
        factors["Curvature_Proxy"] = (2.0 * two_year) - short - ten_year
        factors["Policy_Rate"] = policy
        factors["Yield_2Y"] = two_year
        factors["Yield_10Y"] = ten_year
        factors = factors.loc[factors[["Level_Factor", "Slope_10Y_2Y"]].notna().any(axis=1)]
        if factors.empty:
            continue
        factors["Curve_State"] = [
            _curve_state(float(level), float(slope))
            for level, slope in zip(factors["Level_Factor"], factors["Slope_10Y_2Y"], strict=False)
        ]
        factors["Observation_Mode"] = "event_time_last_observation_carried"
        rows.append(factors.tail(cfg.factor_history_rows_per_country).reset_index())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(columns={"Observation_Date": "Date"})


def _modified_duration(years: float, yield_percent: float) -> float:
    if not np.isfinite(yield_percent):
        return np.nan
    return float(years / (1.0 + yield_percent / 100.0))


def _zero_coupon_convexity(years: float, yield_percent: float) -> float:
    if not np.isfinite(yield_percent):
        return np.nan
    denominator = (1.0 + yield_percent / 100.0) ** 2
    return float(years * (years + 1.0) / denominator)


def _price_impact_percent(duration: float, convexity: float, shock_bp: float) -> float:
    if not np.isfinite(duration) or not np.isfinite(convexity):
        return np.nan
    shock = shock_bp / 10_000.0
    return float((-duration * shock + 0.5 * convexity * shock**2) * 100.0)


def _stress_scenarios(country_metrics: pd.DataFrame) -> pd.DataFrame:
    scenarios = (
        ("Parallel +100 bp", 100.0, 100.0, "Rates rise across the observed curve"),
        ("Parallel -100 bp", -100.0, -100.0, "Rates fall across the observed curve"),
        ("Bear steepener", 25.0, 100.0, "Long rates rise more than front-end rates"),
        ("Bull steepener", -100.0, -25.0, "Front-end rates fall more than long rates"),
        ("Bear flattener", 100.0, 25.0, "Front-end rates rise more than long rates"),
        ("Bull flattener", -25.0, -100.0, "Long rates fall more than front-end rates"),
    )
    rows: list[dict[str, object]] = []
    for metric in country_metrics.to_dict("records"):
        if not np.isfinite(float(metric.get("Yield_2Y", np.nan))) or not np.isfinite(
            float(metric.get("Yield_10Y", np.nan))
        ):
            continue
        duration_2y = float(metric["Modified_Duration_2Y"])
        duration_10y = float(metric["Modified_Duration_10Y"])
        convexity_2y = float(metric["Convexity_2Y"])
        convexity_10y = float(metric["Convexity_10Y"])
        for scenario, shock_2y, shock_10y, interpretation in scenarios:
            impact_2y = _price_impact_percent(duration_2y, convexity_2y, shock_2y)
            impact_10y = _price_impact_percent(duration_10y, convexity_10y, shock_10y)
            rows.append(
                {
                    "Country": metric["Country"],
                    "Scenario": scenario,
                    "Shock_2Y_bp": shock_2y,
                    "Shock_10Y_bp": shock_10y,
                    "Curve_Slope_Change_bp": shock_10y - shock_2y,
                    "Approx_Price_Impact_2Y_pct": impact_2y,
                    "Approx_Price_Impact_10Y_pct": impact_10y,
                    "Equal_Notional_2Y_10Y_Impact_pct": float(np.nanmean([impact_2y, impact_10y])),
                    "Interpretation": interpretation,
                    "Model": "zero_coupon_duration_convexity_proxy",
                }
            )
    return pd.DataFrame(rows)


def _reference_rate_summary(reference_rates: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    if reference_rates.empty:
        return pd.DataFrame()
    key = "Code" if "Code" in reference_rates else "Benchmark"
    rows: list[dict[str, object]] = []
    for code, group in reference_rates.groupby(key, sort=True):
        group = group.sort_values("Observation_Date")
        latest = group.iloc[-1]
        series = group.set_index("Observation_Date")["Rate"]
        rows.append(
            {
                "Code": str(code),
                "Benchmark": latest.get("Benchmark", code),
                "Jurisdiction": latest.get("Jurisdiction"),
                "Currency": latest.get("Currency"),
                "Tenor": latest.get("Tenor"),
                "Latest_Rate": float(latest["Rate"]),
                "Change_1M_bp": _last_observed_change(series, 31) * 100.0,
                "Change_3M_bp": _last_observed_change(series, 92) * 100.0,
                "Change_1Y_bp": _last_observed_change(series, 366) * 100.0,
                "Latest_Observation_Date": latest["Observation_Date"],
                "Data_Staleness_Days": int(
                    max(0, (decision_date.normalize() - pd.Timestamp(latest["Observation_Date"]).normalize()).days)
                ),
                "Observation_Frequency": latest.get("Observation_Frequency"),
                "Status": latest.get("Status"),
                "Source": latest.get("Source"),
            }
        )
    return pd.DataFrame(rows).sort_values(["Currency", "Code"]).reset_index(drop=True)


def _country_metrics(
    snapshot: pd.DataFrame,
    factor_history: pd.DataFrame,
    decision_date: pd.Timestamp,
    cfg: FixedIncomeIntelligenceConfig,
) -> pd.DataFrame:
    countries = set(snapshot.get("Country", pd.Series(dtype=str)).dropna().astype(str))
    countries.update(factor_history.get("Country", pd.Series(dtype=str)).dropna().astype(str))
    rows: list[dict[str, object]] = []
    for country in sorted(countries):
        snapshot_rows = (
            snapshot.loc[snapshot["Country"].astype(str) == country] if not snapshot.empty else pd.DataFrame()
        )
        latest_snapshot = snapshot_rows.iloc[-1] if not snapshot_rows.empty else pd.Series(dtype=object)
        factors = (
            factor_history.loc[factor_history["Country"].astype(str) == country].copy()
            if not factor_history.empty
            else pd.DataFrame()
        )
        if not factors.empty:
            factors["Date"] = pd.to_datetime(factors["Date"], errors="coerce", utc=True)
            factors = factors.dropna(subset=["Date"]).sort_values("Date")
        latest_factor = factors.iloc[-1] if not factors.empty else pd.Series(dtype=object)

        policy = _latest_curve_value(latest_snapshot, latest_factor, "Policy_Rate")
        two_year = _latest_curve_value(latest_snapshot, latest_factor, "Yield_2Y")
        ten_year = _latest_curve_value(latest_snapshot, latest_factor, "Yield_10Y")
        level = float(np.nanmean([two_year, ten_year])) if np.isfinite(two_year) and np.isfinite(ten_year) else np.nan
        slope = ten_year - two_year if np.isfinite(two_year) and np.isfinite(ten_year) else np.nan
        sovereign_tenors = int(np.isfinite(two_year)) + int(np.isfinite(ten_year))
        observed_tenors = sovereign_tenors + int(np.isfinite(policy))
        latest_candidates = []
        for candidate in (
            latest_snapshot.get("Latest_Date"),
            latest_snapshot.get("Policy_Observation_Date"),
            latest_factor.get("Date"),
        ):
            parsed = _utc_timestamp(candidate)
            if parsed is not None:
                latest_candidates.append(parsed)
        latest_date = max(latest_candidates) if latest_candidates else decision_date
        stale_days = int(max(0, (decision_date.normalize() - latest_date.normalize()).days))
        history_observations = int(len(factors))
        quality_score = (
            0.60 * (sovereign_tenors / 2.0)
            + 0.15 * float(np.isfinite(policy))
            + 0.15 * min(history_observations / 252.0, 1.0)
            + 0.10 * (1.0 if stale_days <= 7 else 0.5 if stale_days <= cfg.stale_warning_days else 0.0)
        )
        if sovereign_tenors >= 2 and history_observations >= cfg.minimum_history_observations and stale_days <= 31:
            quality = "High"
        elif sovereign_tenors >= 2:
            quality = "Medium"
        else:
            quality = "Insufficient for curve analytics"
        slope_series = (
            factors.set_index("Date")["Slope_10Y_2Y"]
            if not factors.empty and "Slope_10Y_2Y" in factors
            else pd.Series()
        )
        level_series = (
            factors.set_index("Date")["Level_Factor"]
            if not factors.empty and "Level_Factor" in factors
            else pd.Series()
        )
        rate_changes = level_series.diff().dropna().tail(cfg.rate_volatility_observations) * 100.0
        rows.append(
            {
                "Country": country,
                "As_Of": latest_date,
                "Evidence_Scope": "live_snapshot",
                "Policy_Rate": policy,
                "Yield_2Y": two_year,
                "Yield_10Y": ten_year,
                "Level_Factor": level,
                "Slope_10Y_2Y": slope,
                "Policy_Gap_2Y": two_year - policy if np.isfinite(two_year) and np.isfinite(policy) else np.nan,
                "Curve_State": _curve_state(level, slope),
                "Slope_Change_1M_bp": _last_observed_change(slope_series, 31) * 100.0,
                "Slope_Change_3M_bp": _last_observed_change(slope_series, 92) * 100.0,
                "Level_Change_3M_bp": _last_observed_change(level_series, 92) * 100.0,
                "Observed_Rate_Change_Vol_bp": float(rate_changes.std(ddof=1)) if len(rate_changes) >= 20 else np.nan,
                "Modified_Duration_2Y": _modified_duration(2.0, two_year),
                "Modified_Duration_10Y": _modified_duration(10.0, ten_year),
                "Convexity_2Y": _zero_coupon_convexity(2.0, two_year),
                "Convexity_10Y": _zero_coupon_convexity(10.0, ten_year),
                "Sovereign_Tenor_Count": sovereign_tenors,
                "Observed_Tenor_Count": observed_tenors,
                "History_Observations": history_observations,
                "Stale_Days": stale_days,
                "Curve_Quality_Score": float(np.clip(quality_score, 0.0, 1.0)),
                "Curve_Quality": quality,
                "Rate_Source": latest_snapshot.get("Rate_Source", "public source"),
                "Observation_Mode": "native_calendar_event_time",
            }
        )
    return pd.DataFrame(rows)


def build_fixed_income_intelligence(
    yield_curve_snapshot: pd.DataFrame,
    rate_history: pd.DataFrame,
    *,
    reference_rates: pd.DataFrame | None = None,
    carry_validation: pd.DataFrame | None = None,
    as_of: pd.Timestamp | str | None = None,
    config: FixedIncomeIntelligenceConfig | None = None,
) -> dict:
    """Build causal fixed-income diagnostics from observed public-source tenors."""

    cfg = config or FixedIncomeIntelligenceConfig()
    raw_snapshot = _frame(yield_curve_snapshot)
    raw_history = _frame(rate_history)
    raw_reference = _frame(reference_rates)
    decision_date = _decision_date(raw_snapshot, raw_history, raw_reference, as_of)
    if decision_date is None:
        return {
            "as_of": None,
            "country_metrics": pd.DataFrame(),
            "factor_history": pd.DataFrame(),
            "stress_scenarios": pd.DataFrame(),
            "reference_rate_summary": pd.DataFrame(),
            "carry_candidates": pd.DataFrame(),
            "methodology": pd.DataFrame(),
        }

    snapshot = _normalize_snapshot(raw_snapshot, decision_date)
    history = _normalize_history(raw_history, decision_date)
    references = _normalize_reference_rates(raw_reference, decision_date)
    factors = _factor_history(history, cfg)
    metrics = _country_metrics(snapshot, factors, decision_date, cfg)
    scenarios = _stress_scenarios(metrics)
    reference_summary = _reference_rate_summary(references, decision_date)
    carry = _frame(carry_validation)
    if not carry.empty:
        score_column = (
            "FX_Risk_Adjusted_Carry_Score"
            if "FX_Risk_Adjusted_Carry_Score" in carry
            else "Carry_Trade_Score"
            if "Carry_Trade_Score" in carry
            else None
        )
        if score_column:
            carry[score_column] = pd.to_numeric(carry[score_column], errors="coerce")
            carry = carry.sort_values(score_column, ascending=False, na_position="last")
        carry = carry.head(40).reset_index(drop=True)
        carry["Evidence_Scope"] = "live_snapshot_research_only"
        carry["Execution_Note"] = "Rate differential is not a trade without explicit FX and event-risk controls."

    if metrics.empty:
        return {
            "as_of": decision_date,
            "country_metrics": metrics,
            "factor_history": factors,
            "stress_scenarios": scenarios,
            "reference_rate_summary": reference_summary,
            "carry_candidates": carry,
            "methodology": pd.DataFrame(),
        }

    countries = tuple(metrics.get("Country", pd.Series(dtype=str)).astype(str))
    contract = FixedIncomeIntelligenceV1(
        as_of=decision_date.to_pydatetime(),
        countries=countries,
        minimum_real_tenors=2,
        factor_observation_mode="native_calendar_event_time_last_observation_carried",
        formulas={
            "level": "L_t=(y_t(2Y)+y_t(10Y))/2",
            "slope": "S_t=y_t(10Y)-y_t(2Y)",
            "curvature_proxy": "C_t=2y_t(2Y)-y_t(short)-y_t(10Y)",
            "duration_convexity": "dP/P=-D_mod*dy+0.5*Convexity*dy^2",
            "quality": "weighted tenor, history, recency and source-coverage score",
        },
    )
    methodology = pd.DataFrame(
        [
            {
                "Evidence_Scope": "live_snapshot",
                "Decision_Date": decision_date,
                "Countries": len(countries),
                "Countries_With_Two_Sovereign_Tenors": int(
                    (metrics.get("Sovereign_Tenor_Count", pd.Series(dtype=float)) >= 2).sum()
                ),
                "Factor_Observation_Mode": contract.factor_observation_mode,
                "Curve_Interpolation": "None",
                "Scenario_Model": "zero_coupon_duration_convexity_proxy",
                "Scenario_Limitation": "Illustrative local price sensitivity; not a bond-level valuation or executable quote.",
                "Carry_Limitation": "FX-risk-adjusted research ranking; no claim of covered-interest arbitrage.",
            }
        ]
    )
    return {
        "contract": contract.model_dump(mode="json"),
        "as_of": decision_date,
        "country_metrics": metrics,
        "factor_history": factors,
        "stress_scenarios": scenarios,
        "reference_rate_summary": reference_summary,
        "carry_candidates": carry,
        "methodology": methodology,
    }

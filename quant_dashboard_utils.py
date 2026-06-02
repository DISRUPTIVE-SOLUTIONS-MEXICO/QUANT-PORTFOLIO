from __future__ import annotations

import pandas as pd


COUNTRY_FLAGS = {
    "United States": "🇺🇸",
    "Mexico": "🇲🇽",
    "Canada": "🇨🇦",
    "Brazil": "🇧🇷",
    "China": "🇨🇳",
    "India": "🇮🇳",
    "Japan": "🇯🇵",
    "South Korea": "🇰🇷",
    "Australia": "🇦🇺",
    "New Zealand": "🇳🇿",
    "United Kingdom": "🇬🇧",
    "Germany": "🇩🇪",
    "France": "🇫🇷",
    "Spain": "🇪🇸",
    "Italy": "🇮🇹",
    "Netherlands": "🇳🇱",
    "Switzerland": "🇨🇭",
    "Sweden": "🇸🇪",
    "Norway": "🇳🇴",
    "South Africa": "🇿🇦",
}


def build_drawdown_frame(curve: pd.DataFrame | None = None, perf: pd.DataFrame | None = None) -> pd.DataFrame:
    if curve is not None and not curve.empty and {"Period_End", "Portfolio_Equity"}.issubset(curve.columns):
        eq = pd.to_numeric(curve["Portfolio_Equity"], errors="coerce")
        dates = pd.to_datetime(curve["Period_End"], errors="coerce")
        if eq.notna().sum() > 1 and eq.std(skipna=True) > 1e-12:
            out = pd.DataFrame({"Period_End": dates, "Portfolio_Equity": eq}).dropna()
            out["Drawdown"] = out["Portfolio_Equity"] / out["Portfolio_Equity"].cummax() - 1.0
            return out
    if perf is not None and not perf.empty and {"Period_End", "Net_Return"}.issubset(perf.columns):
        out = perf[["Period_End", "Net_Return"]].copy()
        out["Period_End"] = pd.to_datetime(out["Period_End"], errors="coerce")
        out["Net_Return"] = pd.to_numeric(out["Net_Return"], errors="coerce").fillna(0.0)
        out = out.dropna(subset=["Period_End"]).sort_values("Period_End")
        out["Portfolio_Equity"] = (1.0 + out["Net_Return"]).cumprod()
        out["Drawdown"] = out["Portfolio_Equity"] / out["Portfolio_Equity"].cummax() - 1.0
        return out
    return pd.DataFrame()


def prepare_global_curve_matrix(curves: pd.DataFrame) -> pd.DataFrame:
    if curves is None or curves.empty:
        return pd.DataFrame()
    cols = ["Policy_Rate", "Yield_2Y", "Yield_10Y", "Curve_10Y_2Y", "Term_Premium_Proxy"]
    data = curves.copy()
    for col in cols:
        if col not in data:
            data[col] = pd.NA
        data[col] = pd.to_numeric(data[col], errors="coerce")
    matrix = (
        data.set_index("Country")[cols]
        .rename(
            columns={
                "Policy_Rate": "Policy",
                "Yield_2Y": "2Y",
                "Yield_10Y": "10Y",
                "Curve_10Y_2Y": "10Y-2Y",
                "Term_Premium_Proxy": "10Y-Policy",
            }
        )
        .dropna(how="all")
    )
    if "10Y" in matrix:
        matrix = matrix.sort_values("10Y", ascending=False)
    return matrix


def prepare_discrete_rate_plot_data(
    history: pd.DataFrame,
    tenor_code: str,
    max_countries: int = 10,
    lookback_days: int = 365 * 3,
    normalize_frequency: str = "month_end",
) -> pd.DataFrame:
    """Select comparable country rate histories by calendar window, not observation count."""
    if history is None or history.empty:
        return pd.DataFrame()
    required = {"Country", "Observation_Date", "Tenor_Code", "Rate"}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    data = history[history["Tenor_Code"].eq(tenor_code)].copy()
    if data.empty:
        return pd.DataFrame()
    data["Observation_Date"] = pd.to_datetime(data["Observation_Date"], errors="coerce")
    data["Rate"] = pd.to_numeric(data["Rate"], errors="coerce")
    data = data.dropna(subset=["Country", "Observation_Date", "Rate"])
    if data.empty:
        return pd.DataFrame()
    cutoff = data["Observation_Date"].max() - pd.Timedelta(days=int(lookback_days))
    latest_countries = (
        data.sort_values("Observation_Date")
        .groupby("Country")["Rate"]
        .last()
        .dropna()
        .sort_values(ascending=False)
        .head(int(max_countries))
        .index
    )
    out = (
        data[data["Country"].isin(latest_countries) & data["Observation_Date"].ge(cutoff)]
        .sort_values(["Country", "Observation_Date"])
        .reset_index(drop=True)
    )
    if normalize_frequency == "month_end" and not out.empty:
        frames = []
        for country, sub in out.groupby("Country"):
            sub = sub.sort_values("Observation_Date").set_index("Observation_Date")
            resampled = sub[["Rate"]].resample("ME").last().dropna()
            if resampled.empty:
                continue
            frame = resampled.reset_index()
            frame["Country"] = country
            frame["Tenor_Code"] = tenor_code
            if "Tenor" in sub.columns:
                frame["Tenor"] = sub["Tenor"].dropna().iloc[-1] if sub["Tenor"].dropna().size else tenor_code
            native_freq = sub["Observation_Frequency"].dropna().iloc[-1] if "Observation_Frequency" in sub and sub["Observation_Frequency"].dropna().size else "Native discrete"
            source = sub["Source"].dropna().iloc[-1] if "Source" in sub and sub["Source"].dropna().size else None
            frame["Observation_Frequency"] = "Monthly comparable view"
            frame["Native_Observation_Frequency"] = native_freq
            frame["Source"] = source
            frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not out.empty:
            out = out.sort_values(["Country", "Observation_Date"]).reset_index(drop=True)
    return out


def latest_rate_observations(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required = {"Country", "Observation_Date", "Tenor_Code", "Rate"}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    data = history.copy()
    data["Observation_Date"] = pd.to_datetime(data["Observation_Date"], errors="coerce")
    data["Rate"] = pd.to_numeric(data["Rate"], errors="coerce")
    data = data.dropna(subset=["Country", "Observation_Date", "Tenor_Code", "Rate"])
    if data.empty:
        return pd.DataFrame()
    return (
        data.sort_values("Observation_Date")
        .groupby(["Country", "Tenor_Code"], as_index=False)
        .tail(1)
        .sort_values(["Tenor_Code", "Country"])
        .reset_index(drop=True)
    )


def balanced_rate_history_sample(history: pd.DataFrame, rows_per_group: int = 5) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required = {"Country", "Observation_Date", "Tenor_Code"}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    data = history.copy()
    data["Observation_Date"] = pd.to_datetime(data["Observation_Date"], errors="coerce")
    data = data.dropna(subset=["Country", "Observation_Date", "Tenor_Code"])
    if data.empty:
        return pd.DataFrame()
    return (
        data.sort_values("Observation_Date")
        .groupby(["Country", "Tenor_Code"], as_index=False)
        .tail(int(rows_per_group))
        .sort_values(["Tenor_Code", "Country", "Observation_Date"])
        .reset_index(drop=True)
    )


def country_flag(country: str) -> str:
    return COUNTRY_FLAGS.get(str(country), str(country)[:2].upper())


def spread_label_positions(values: list[float], min_gap: float = 0.18) -> list[float]:
    if not values:
        return []
    indexed = sorted(enumerate(float(v) for v in values), key=lambda x: x[1])
    adjusted: list[tuple[int, float]] = []
    last_y = None
    for idx, y in indexed:
        new_y = y if last_y is None else max(y, last_y + min_gap)
        adjusted.append((idx, new_y))
        last_y = new_y
    drift = sum(y for _, y in adjusted) / len(adjusted) - sum(float(v) for v in values) / len(values)
    adjusted = [(idx, y - drift) for idx, y in adjusted]
    return [y for _, y in sorted(adjusted, key=lambda x: x[0])]

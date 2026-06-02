from __future__ import annotations

import math
import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import defusedxml.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.optimize import minimize
from scipy.stats import genpareto, iqr, kurtosis, norm, skew, t as student_t
try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover - optional scipy submodules in degraded environments.
    leaves_list = linkage = squareform = None

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.covariance import LedoitWolf

from quant_core.backtest_paths import build_backtest_path_bundle
from quant_core.data_freshness import build_data_freshness_report
from quant_core.dashboard_payload import build_dashboard_payload
from quant_core.pit_confidence import add_pit_confidence
from quant_core.promotion_gate import evaluate_promotion_gate
from quant_core.suitability_gate import evaluate_suitability_gate
from quant_core.uncertainty_state import fractional_volterra_variance, qlike_loss

APP_VERSION = "0.2.0"
MODEL_VERSION = "qp-kaizen-core-v0.2.0"
SCHEMA_VERSION = "20260520_001"

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - sklearn is optional in degraded environments.
    GaussianMixture = None


CORE_RATIO_COLS = [
    "ROIC",
    "EV_EBITDA",
    "FCF_Yield",
    "NetDebt_EBITDA",
    "Piotroski",
    "Asset_Turnover",
    "Altman_Z",
    "Interest_Coverage",
    "Retention_Ratio",
    "Earnings_Yield",
    "Price_Book",
    "PE_Ratio",
    "EPS",
    "Solvency",
    "ROE",
]

FUNDAMENTAL_AUDIT_COLS = [
    "Enterprise_Value",
    "Free_Cash_Flow",
    "Net_Debt",
    "NOPAT",
    "Invested_Capital",
    "EBITDA",
    "Book_Value_Per_Share",
    "Price_to_Book_Value",
    "Price_to_Earnings",
    "PER",
]

BENCHMARK_METADATA = {
    "SPY": {"Group": "US Market", "Country": "United States", "Sector": "Broad", "Scope": "US Large Cap"},
    "QQQ": {"Group": "US Market", "Country": "United States", "Sector": "Technology/Growth", "Scope": "US Nasdaq 100"},
    "IWM": {"Group": "US Market", "Country": "United States", "Sector": "Broad", "Scope": "US Small Cap"},
    "DIA": {"Group": "US Market", "Country": "United States", "Sector": "Broad", "Scope": "US Dow"},
    "ACWI": {"Group": "International", "Country": "Global", "Sector": "Broad", "Scope": "Global Equity"},
    "VT": {"Group": "International", "Country": "Global", "Sector": "Broad", "Scope": "Total World"},
    "EFA": {"Group": "International", "Country": "Developed ex-US", "Sector": "Broad", "Scope": "Developed ex-US"},
    "VEA": {"Group": "International", "Country": "Developed ex-US", "Sector": "Broad", "Scope": "Developed ex-US"},
    "EEM": {"Group": "International", "Country": "Emerging Markets", "Sector": "Broad", "Scope": "Emerging Markets"},
    "EWW": {"Group": "Country", "Country": "Mexico", "Sector": "Broad", "Scope": "Mexico Equity"},
    "EWC": {"Group": "Country", "Country": "Canada", "Sector": "Broad", "Scope": "Canada Equity"},
    "EWZ": {"Group": "Country", "Country": "Brazil", "Sector": "Broad", "Scope": "Brazil Equity"},
    "EWU": {"Group": "Country", "Country": "United Kingdom", "Sector": "Broad", "Scope": "UK Equity"},
    "EWG": {"Group": "Country", "Country": "Germany", "Sector": "Broad", "Scope": "Germany Equity"},
    "EWQ": {"Group": "Country", "Country": "France", "Sector": "Broad", "Scope": "France Equity"},
    "EWP": {"Group": "Country", "Country": "Spain", "Sector": "Broad", "Scope": "Spain Equity"},
    "EWJ": {"Group": "Country", "Country": "Japan", "Sector": "Broad", "Scope": "Japan Equity"},
    "MCHI": {"Group": "Country", "Country": "China", "Sector": "Broad", "Scope": "China Equity"},
    "INDA": {"Group": "Country", "Country": "India", "Sector": "Broad", "Scope": "India Equity"},
    "EWA": {"Group": "Country", "Country": "Australia", "Sector": "Broad", "Scope": "Australia Equity"},
    "XLK": {"Group": "US Sector", "Country": "United States", "Sector": "Technology", "Scope": "US Technology"},
    "XLV": {"Group": "US Sector", "Country": "United States", "Sector": "Healthcare", "Scope": "US Health Care"},
    "XLU": {"Group": "US Sector", "Country": "United States", "Sector": "Utilities", "Scope": "US Utilities"},
    "XLRE": {"Group": "US Sector", "Country": "United States", "Sector": "Real Estate", "Scope": "US Real Estate"},
    "XLE": {"Group": "US Sector", "Country": "United States", "Sector": "Energy", "Scope": "US Energy"},
    "XLF": {"Group": "US Sector", "Country": "United States", "Sector": "Financial Services", "Scope": "US Financials"},
    "XLI": {"Group": "US Sector", "Country": "United States", "Sector": "Industrials", "Scope": "US Industrials"},
    "XLY": {"Group": "US Sector", "Country": "United States", "Sector": "Consumer Cyclical", "Scope": "US Consumer Discretionary"},
    "XLP": {"Group": "US Sector", "Country": "United States", "Sector": "Consumer Defensive", "Scope": "US Consumer Staples"},
    "XLB": {"Group": "US Sector", "Country": "United States", "Sector": "Basic Materials", "Scope": "US Materials"},
}

COUNTRY_BENCHMARKS = {
    "United States": "SPY",
    "Mexico": "EWW",
    "Canada": "EWC",
    "Brazil": "EWZ",
    "United Kingdom": "EWU",
    "Germany": "EWG",
    "France": "EWQ",
    "Spain": "EWP",
    "Japan": "EWJ",
    "China": "MCHI",
    "India": "INDA",
    "Australia": "EWA",
}

SECTOR_BENCHMARKS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Basic Materials": "XLB",
    "Materials": "XLB",
}

SIDE_TICKER_ALIASES = {
    "CEREBRAS": "CBRS",
    "CEREBRAS SYSTEMS": "CBRS",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "PFIZER": "PFE",
    "NOVO": "NVO",
    "NOVO NORDISK": "NVO",
    "TSMC": "TSM",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "APPLE": "AAPL",
    "LENOVO": "LNVGY",
    "CISCO": "CSCO",
}

DEFAULT_SIDE_BOOM_TICKERS = (
    "CBRS", "MSFT", "NVDA", "PFE", "NVO", "TSM", "AMD", "XOM", "CVX", "SM",
    "FCX", "RIO", "LIN", "APD", "DD", "ALB", "ECL", "AAPL", "LNVGY", "CSCO",
)

DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS = (
    ("LITE", 0.05),
    ("CIEN", 0.05),
    ("WDC", 0.05),
    ("STX", 0.05),
    ("GILD", 0.05),
    ("PLTR", 0.05),
    ("MO", 0.05),
    ("PM", 0.05),
    ("SATS", 0.05),
    ("CBOE", 0.05),
    ("CASY", 0.05),
    ("WELL", 0.05),
    ("T", 0.05),
    ("EBAY", 0.05),
    ("INSM", 0.05),
    ("CAH", 0.047485566),
    ("GEV", 0.038704298),
    ("CF", 0.036477665),
    ("CHRW", 0.035200363),
    ("GLW", 0.03043802),
    ("WBD", 0.028459801),
    ("APP", 0.021009987),
    ("JNJ", 0.012224285),
)
DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT = max(
    0.0,
    1.0 - float(sum(weight for _, weight in DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS)),
)
DEFAULT_SIDE_ALPHA_TICKERS = tuple(
    dict.fromkeys(["CBRS"] + [ticker for ticker, _ in DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS])
)


@dataclass(frozen=True)
class RunConfig:
    tickers: tuple[str, ...]
    benchmark_ticker: str = "SPY"
    price_period: str = "3y"
    accounting_lag_days: int = 90
    top_n: int = 10
    preselect_n: int = 14
    min_chunk: int = 5
    max_chunk: int = 10
    max_combos: int = 25_000
    max_names_per_sector: int = 3
    max_weight: float = 0.25
    sector_weight_cap: float = 0.40
    risk_aversion: float = 4.0
    alpha_weight: float = 1.0
    weight_objective: str = "sortino"
    entropy_penalty: float = 0.05
    crlb_penalty: float = 0.15
    garch_penalty: float = 0.10
    evt_penalty: float = 0.10
    cvar_penalty: float = 0.25
    cvar_alpha: float = 0.95
    robust_alpha_uncertainty: float = 0.35
    robust_cov_uncertainty: float = 0.10
    use_garch: bool = True
    target_vol: float | None = None
    nested_validation_fraction: float = 0.35
    purge_days: int = 5
    sortino_multistarts: int = 8
    bootstrap_samples: int = 64
    factor_mkt_cap: float = 1.25
    factor_rates_cap: float = 0.25
    factor_credit_cap: float = 0.25
    factor_oil_cap: float = 0.35
    factor_usd_cap: float = 0.35
    factor_cov_blend: float = 0.50
    use_black_litterman: bool = False
    black_litterman_tau: float = 0.05
    portfolio_notional: float = 100_000.0
    max_adv_participation: float = 0.05
    impact_coefficient: float = 0.10
    min_dollar_volume: float = 1_000_000.0
    max_workers: int = 8
    compute_mode: str = "rigorous"
    use_persistent_cache: bool = True
    cache_ttl_hours: int = 24
    rate_country: str = "United States"
    use_sec_edgar: bool = True
    sec_user_agent: str = "QuantStockPicker/1.0 contact@example.com"
    use_sec_nlp: bool = True
    sec_nlp_max_tickers: int = 30
    sec_nlp_max_filings: int = 2
    text_risk_penalty: float = 0.10
    use_kaizen_bandit: bool = True
    kaizen_ucb_alpha: float = 0.75
    kaizen_reward_cvar_lambda: float = 1.0
    kaizen_reward_drawdown_lambda: float = 1.0
    kaizen_reward_turnover_lambda: float = 0.10
    kaizen_reward_cost_lambda: float = 1.0
    universe_source: str = "Manual"
    universe_asof: str | None = None
    use_options_snapshot: bool = True
    option_expiries: int = 3
    options_cache_ttl_hours: int = 24
    garch_candidate_n: int = 30
    validation_bootstrap_samples: int = 512
    reality_check_samples: int = 512
    cpcv_folds: int = 4
    use_gdelt: bool = True
    gdelt_query: str = "(tariff OR sanctions OR geopolitical OR election OR regulation OR fiscal policy)"
    use_forex_factory_calendar: bool = True
    forex_factory_cache_ttl_hours: int = 24
    rebalance_freq: str = "2QE"
    reoptimization_freq: str = "YE"
    tc_bps: float = 10.0
    embargo_days: int = 5
    max_oos_trials_per_rebalance: int = 200
    robust_selection_lambda: float = 0.50
    robust_selection_min_obs: int = 4
    model_confidence_window: int = 6
    model_confidence_min: float = 0.25
    use_latent_macro_regime: bool = True
    latent_regime_states: int = 4
    latent_regime_refit_days: int = 21
    latent_regime_min_train: int = 252
    use_dynamic_sizing: bool = True
    min_dynamic_exposure: float = 0.25
    max_dynamic_exposure: float = 1.00
    regime_entropy_exposure_penalty: float = 0.50
    markov_stress_exposure_penalty: float = 0.50
    markov_transition_min_obs: int = 60
    lookback_grid: tuple[int, ...] = (63, 126, 252)
    chunk_size_grid: tuple[int, ...] = (5, 8, 10)
    suitability_mode: str = "automatic"
    investor_horizon_years: float = 3.0
    investor_initial_capital: float = 100_000.0
    investor_monthly_contribution: float = 0.0
    investor_liquidity_need: str = "Media"
    investor_max_drawdown: float = 0.20
    investor_cvar_max_daily: float = 0.018
    investor_risk_aversion_score: float = 5.0
    investor_objective: str = "Balanced growth"
    investor_base_currency: str = "USD"
    suitability_profile: str = "Balanceado"
    suitability_score: float = 0.50
    suitability_hard_block: bool = False
    suitability_warnings: tuple[str, ...] = ()
    benchmark_group: str = "US Market"
    benchmark_mandate_type: str = "Relative vs benchmark"
    benchmark_auto_select: bool = False
    benchmark_suggested_ticker: str = "SPY"
    benchmark_governance_warnings: tuple[str, ...] = ()
    use_side_boom_portfolio: bool = True
    side_boom_tickers: tuple[str, ...] = DEFAULT_SIDE_ALPHA_TICKERS
    side_boom_fixed_ticker: str = "CBRS"
    side_boom_fixed_weight: float = DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT
    side_boom_fixed_weights: tuple[tuple[str, float], ...] = DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS
    side_boom_mode: str = "private_side_alpha_firewall"
    side_boom_min_obs: int = 60
    side_boom_cash_return: float = 0.0


def build_suitability_constraints(
    horizon_years: float,
    initial_capital: float,
    monthly_contribution: float,
    liquidity_need: str,
    max_drawdown: float,
    risk_aversion_score: float,
    investor_objective: str,
    base_currency: str = "USD",
    parameter_mode: str = "automatic",
) -> dict:
    """Map an investor profile into quantitative portfolio constraints.

    The score is risk capacity net of risk aversion. It is intentionally
    conservative: user-facing suitability must dominate optimizer enthusiasm.
    """
    horizon_years = float(np.clip(to_float(horizon_years), 0.25, 30.0))
    initial_capital = float(max(to_float(initial_capital), 0.0))
    monthly_contribution = float(max(to_float(monthly_contribution), 0.0))
    max_drawdown = float(np.clip(to_float(max_drawdown), 0.03, 0.80))
    risk_aversion_score = float(np.clip(to_float(risk_aversion_score), 0.0, 10.0))
    liquidity_need = str(liquidity_need or "Medium")
    investor_objective = str(investor_objective or "Balanced growth")
    base_currency = str(base_currency or "USD").upper()

    horizon_score = np.clip((horizon_years - 0.5) / 9.5, 0.0, 1.0)
    capital_score = np.clip(np.log1p(initial_capital) / np.log1p(250_000.0), 0.0, 1.0)
    contribution_score = np.clip((12.0 * monthly_contribution) / max(initial_capital, 1.0), 0.0, 1.0)
    drawdown_score = np.clip((max_drawdown - 0.05) / 0.45, 0.0, 1.0)
    liquidity_penalty = {
        "Baja": 0.00,
        "Low": 0.00,
        "Media": 0.14,
        "Medium": 0.14,
        "Alta": 0.30,
        "High": 0.30,
    }.get(liquidity_need, 0.14)
    objective_tilt = {
        "Preservacion de capital": -0.18,
        "Capital preservation": -0.18,
        "Ingreso": -0.08,
        "Income": -0.08,
        "Crecimiento balanceado": 0.00,
        "Balanced growth": 0.00,
        "Crecimiento agresivo": 0.14,
        "Aggressive growth": 0.14,
        "Alta conviccion": 0.20,
        "High conviction": 0.20,
    }.get(investor_objective, 0.0)
    aversion_penalty = risk_aversion_score / 10.0
    score = (
        0.24 * horizon_score
        + 0.18 * capital_score
        + 0.10 * contribution_score
        + 0.28 * drawdown_score
        + 0.20 * (1.0 - aversion_penalty)
        + objective_tilt
        - liquidity_penalty
    )
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.25:
        profile = "Conservador"
    elif score < 0.52:
        profile = "Balanceado"
    elif score < 0.78:
        profile = "Agresivo"
    else:
        profile = "Especulativo"

    first_year_capital = initial_capital + 12.0 * monthly_contribution
    min_ticket_notional = 1_000.0 if first_year_capital >= 25_000 else 500.0
    capital_holdings = int(np.clip(np.floor(max(first_year_capital, 1.0) / min_ticket_notional), 3, 25))
    profile_caps = {
        "Conservador": dict(vol=0.10, cvar=0.012, max_w=0.12, sector=0.25, adv=0.025, min_dv=5_000_000, cvar_penalty=0.55, entropy=0.18, factor_blend=0.80),
        "Balanceado": dict(vol=0.15, cvar=0.018, max_w=0.20, sector=0.35, adv=0.050, min_dv=1_000_000, cvar_penalty=0.35, entropy=0.08, factor_blend=0.55),
        "Agresivo": dict(vol=0.22, cvar=0.027, max_w=0.30, sector=0.50, adv=0.075, min_dv=500_000, cvar_penalty=0.22, entropy=0.04, factor_blend=0.35),
        "Especulativo": dict(vol=0.30, cvar=0.040, max_w=0.40, sector=0.65, adv=0.100, min_dv=0, cvar_penalty=0.15, entropy=0.02, factor_blend=0.20),
    }
    caps = profile_caps[profile].copy()
    dd_vol_cap = max_drawdown / 2.0
    target_vol = float(np.clip(min(caps["vol"], dd_vol_cap), 0.05, caps["vol"]))
    top_n = int(np.clip(min(capital_holdings, {"Conservador": 8, "Balanceado": 12, "Agresivo": 16, "Especulativo": 20}[profile]), 3, 20))
    min_chunk = int(np.clip(min(5, top_n), 3, 12))
    max_chunk = int(np.clip(min(top_n, {"Conservador": 6, "Balanceado": 10, "Agresivo": 14, "Especulativo": 16}[profile]), min_chunk, 16))
    max_sector_names = {"Conservador": 2, "Balanceado": 3, "Agresivo": 4, "Especulativo": 5}[profile]

    warnings = []
    hard_block = False
    if horizon_years < 1.0 and profile in {"Agresivo", "Especulativo"}:
        warnings.append("Investment horizon below 1 year is incompatible with an aggressive/speculative profile.")
        hard_block = True
    if max_drawdown < 0.10 and profile in {"Agresivo", "Especulativo"}:
        warnings.append("A tolerated drawdown below 10% requires a defensive portfolio.")
        hard_block = True
    if liquidity_need in {"Alta", "High"} and profile in {"Agresivo", "Especulativo"}:
        warnings.append("High liquidity needs reduce small/mid caps and concentration.")
    if first_year_capital < 5_000 and top_n > 5:
        warnings.append("Low capital: number of positions is constrained to avoid impracticable weights.")
    if investor_objective in {"Alta conviccion", "Crecimiento agresivo", "High conviction", "Aggressive growth"} and risk_aversion_score >= 8:
        warnings.append("Aggressive objective conflicts with high risk aversion; defensive constraints are prioritized.")
        hard_block = True

    return {
        "Suitability_Mode": parameter_mode,
        "Suitability_Profile": profile,
        "Suitability_Score": score,
        "Horizon_Years": horizon_years,
        "Initial_Capital": initial_capital,
        "Monthly_Contribution": monthly_contribution,
        "First_Year_Capital": first_year_capital,
        "Liquidity_Need": liquidity_need,
        "User_Max_Drawdown": max_drawdown,
        "Risk_Aversion_Score": risk_aversion_score,
        "Investor_Objective": investor_objective,
        "Base_Currency": base_currency,
        "Vol_Max": target_vol,
        "CVaR_Max_Daily": caps["cvar"],
        "DD_Max": max_drawdown,
        "N_Capital_Max": capital_holdings,
        "Top_N_Max": top_n,
        "Min_Chunk": min_chunk,
        "Max_Chunk": max_chunk,
        "Max_Names_Per_Sector": max_sector_names,
        "Max_Weight": caps["max_w"],
        "Sector_Weight_Cap": caps["sector"],
        "Max_ADV_Participation": caps["adv"],
        "Min_Dollar_Volume": caps["min_dv"],
        "Target_Vol": target_vol,
        "CVaR_Penalty": caps["cvar_penalty"],
        "Entropy_Penalty": caps["entropy"],
        "Factor_Cov_Blend": caps["factor_blend"],
        "Hard_Block": hard_block,
        "Warnings": tuple(warnings),
    }


def _mode_value(values: pd.Series, default: str = "Unknown") -> str:
    if values is None or values.empty:
        return default
    clean = values.dropna().astype(str)
    if clean.empty:
        return default
    return str(clean.value_counts().index[0])


def suggest_benchmark(
    mandate_type: str,
    rate_country: str,
    dominant_sector: str | None = None,
    benchmark_group: str | None = None,
    investor_objective: str | None = None,
) -> str:
    mandate_type = str(mandate_type or "")
    benchmark_group = str(benchmark_group or "")
    investor_objective = str(investor_objective or "")
    if "Internacional" in mandate_type or "International" in mandate_type or benchmark_group in {"Internacional", "International"}:
        return "ACWI"
    if "Sector" in mandate_type or benchmark_group in {"Sector USA", "US Sector"}:
        return SECTOR_BENCHMARKS.get(str(dominant_sector or ""), "SPY")
    if "Pais" in mandate_type or "Country" in mandate_type or benchmark_group in {"Pais", "Country"}:
        return COUNTRY_BENCHMARKS.get(str(rate_country or "United States"), "ACWI")
    if investor_objective in {"Crecimiento agresivo", "Alta conviccion", "Aggressive growth", "High conviction"} and str(rate_country) == "United States":
        return "QQQ"
    return COUNTRY_BENCHMARKS.get(str(rate_country or "United States"), "SPY")


def benchmark_governance_diagnostics(
    benchmark_ticker: str,
    benchmark_group: str,
    mandate_type: str,
    rate_country: str,
    investor_objective: str,
    weight_objective: str,
    tickers: Iterable[str],
    cross_section: pd.DataFrame | None = None,
) -> pd.DataFrame:
    benchmark_ticker = str(benchmark_ticker or "SPY").upper()
    meta = BENCHMARK_METADATA.get(benchmark_ticker, {"Group": "Custom", "Country": "Unknown", "Sector": "Unknown", "Scope": "Custom"})
    cs = cross_section.copy() if cross_section is not None and not cross_section.empty else pd.DataFrame()
    dominant_country = _mode_value(cs["Country"]) if "Country" in cs else str(rate_country or "Unknown")
    dominant_sector = _mode_value(cs["Sector"]) if "Sector" in cs else "Unknown"
    sector_share = float(cs["Sector"].astype(str).eq(dominant_sector).mean()) if "Sector" in cs and len(cs) else np.nan
    country_share = float(cs["Country"].astype(str).eq(dominant_country).mean()) if "Country" in cs and len(cs) else np.nan
    suggested = suggest_benchmark(mandate_type, rate_country, dominant_sector, benchmark_group, investor_objective)
    warnings = []

    relative_metric = str(weight_objective).lower() in {"information_ratio", "treynor"} or "Relativo" in str(mandate_type) or "Relative" in str(mandate_type)
    if relative_metric and benchmark_ticker not in BENCHMARK_METADATA:
        warnings.append("IR/Treynor require an observable and representative benchmark; custom benchmark is not recognized.")
    if (str(mandate_type).startswith("Pais") or str(mandate_type).startswith("Country")) and meta.get("Country") not in {dominant_country, rate_country, "Global"}:
        warnings.append(f"Benchmark {benchmark_ticker} does not match dominant country {dominant_country}.")
    if str(mandate_type).startswith("Sector") and meta.get("Sector") not in {dominant_sector, "Broad", "Technology/Growth"}:
        warnings.append(f"Benchmark {benchmark_ticker} does not match dominant sector {dominant_sector}.")
    if (str(mandate_type).startswith("Internacional") or str(mandate_type).startswith("International")) and meta.get("Group") not in {"Internacional", "International"}:
        warnings.append("International mandate with a non-global benchmark; consider ACWI or VT.")
    if benchmark_group in {"Sector USA", "US Sector"} and pd.notna(sector_share) and sector_share < 0.50:
        warnings.append("Sector benchmark selected, but the universe is not mostly single-sector.")
    if benchmark_group in {"Pais", "Country"} and pd.notna(country_share) and country_share < 0.50:
        warnings.append("Country benchmark selected, but the universe is not mostly that country.")
    if relative_metric and benchmark_ticker != suggested and suggested in BENCHMARK_METADATA:
        warnings.append(f"For this mandate, suggested benchmark: {suggested}.")

    is_coherent = len(warnings) == 0
    return pd.DataFrame(
        [
            {
                "Benchmark": benchmark_ticker,
                "Benchmark_Group": benchmark_group,
                "Mandate_Type": mandate_type,
                "Benchmark_Country": meta.get("Country"),
                "Benchmark_Sector": meta.get("Sector"),
                "Benchmark_Scope": meta.get("Scope"),
                "Dominant_Universe_Country": dominant_country,
                "Dominant_Universe_Country_Share": country_share,
                "Dominant_Universe_Sector": dominant_sector,
                "Dominant_Universe_Sector_Share": sector_share,
                "Suggested_Benchmark": suggested,
                "Relative_Metric_Used": relative_metric,
                "Benchmark_Is_Coherent": is_coherent,
                "Warnings": " | ".join(warnings),
                "Universe_Size": len(tuple(tickers or ())),
            }
        ]
    )


CACHE_DIR = Path(__file__).resolve().parent / ".quant_cache"
MODEL_REGISTRY_DIR = CACHE_DIR / "model_registry"


class PersistentCache:
    def __init__(self, root: Path = CACHE_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, namespace: str, payload: dict) -> tuple[Path, Path]:
        serial = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:24]
        folder = self.root / namespace
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.parquet", folder / f"{digest}.json"

    def get_df(self, namespace: str, payload: dict, ttl_hours: int) -> pd.DataFrame | None:
        path, meta_path = self._paths(namespace, payload)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            age_hours = (time.time() - float(meta.get("created_at", 0.0))) / 3600.0
            if ttl_hours > 0 and age_hours > ttl_hours:
                return None
            df = pd.read_parquet(path)
            if "__index__" in df.columns:
                df = df.set_index("__index__")
                try:
                    converted = pd.to_datetime(df.index)
                    if converted.notna().any():
                        df.index = converted
                except Exception:
                    pass
            return df
        except Exception:
            return None

    def set_df(self, namespace: str, payload: dict, df: pd.DataFrame) -> None:
        if df is None:
            return
        path, meta_path = self._paths(namespace, payload)
        try:
            out = df.copy()
            if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
                out = out.reset_index(names="__index__")
            out.to_parquet(path, index=False)
            meta_path.write_text(
                json.dumps(
                    {
                        "namespace": namespace,
                        "payload": payload,
                        "rows": int(len(df)),
                        "columns": list(map(str, df.columns)),
                        "created_at": time.time(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            return

    def inventory(self) -> pd.DataFrame:
        rows = []
        for meta_path in self.root.glob("*/*.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "Namespace": meta.get("namespace"),
                        "Rows": meta.get("rows"),
                        "Created_At": pd.to_datetime(meta.get("created_at"), unit="s"),
                        "Age_Hours": (time.time() - float(meta.get("created_at", 0.0))) / 3600.0,
                        "Key": meta_path.stem,
                    }
                )
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("Created_At", ascending=False) if rows else pd.DataFrame()


PERSISTENT_CACHE = PersistentCache()


def registry_json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return value.isoformat()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): registry_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [registry_json_safe(v) for v in value]
    if isinstance(value, pd.Series):
        return registry_json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return registry_json_safe(value.to_dict(orient="records"))
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def canonical_hash(payload: dict, length: int = 24) -> str:
    serial = json.dumps(registry_json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:length]


def code_version_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ["quant_stockpicker_core.py", "stockpicker_app.py", "supabase_store.py", "requirements.txt"]:
        path = root / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:24]


def dataframe_timestamp_max(df: pd.DataFrame, columns: Iterable[str] = ()) -> str | None:
    if df is None or df.empty:
        return None
    candidates = []
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        candidates.append(df.index.max())
    for col in columns:
        if col in df:
            candidates.append(pd.to_datetime(df[col], errors="coerce").max())
    clean = [pd.Timestamp(x) for x in candidates if pd.notna(x)]
    return max(clean).isoformat() if clean else None


def build_model_registry_record(
    config: RunConfig,
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    cs: pd.DataFrame,
    portfolio: pd.DataFrame,
    perf: pd.DataFrame,
    performance_summary: pd.DataFrame,
    benchmark_governance: pd.DataFrame,
    suitability_diagnostics: pd.DataFrame,
    validation: dict,
    timings: dict,
) -> dict:
    cfg = registry_json_safe(asdict(config))
    data_timestamps = {
        "prices_max": dataframe_timestamp_max(prices),
        "fundamentals_availability_max": dataframe_timestamp_max(panel, ["Availability_Date", "SEC_Accepted_At", "Period_End"]),
        "macro_max": dataframe_timestamp_max(macro, ["Date"]),
        "backtest_oos_max": dataframe_timestamp_max(perf, ["Period_End", "OOS_End"]),
    }
    data_quality = {
        "price_rows": int(len(prices)) if prices is not None else 0,
        "price_columns": int(prices.shape[1]) if prices is not None and not prices.empty else 0,
        "fundamental_rows": int(len(panel)) if panel is not None else 0,
        "cross_section_rows": int(len(cs)) if cs is not None else 0,
        "portfolio_rows": int(len(portfolio)) if portfolio is not None else 0,
        "backtest_periods": int(len(perf)) if perf is not None else 0,
    }
    warning_parts = []
    warning_parts.extend(list(config.suitability_warnings or ()))
    warning_parts.extend(list(config.benchmark_governance_warnings or ()))
    if not benchmark_governance.empty and str(benchmark_governance.iloc[0].get("Warnings", "")).strip():
        warning_parts.append(str(benchmark_governance.iloc[0].get("Warnings")))
    if not suitability_diagnostics.empty and str(suitability_diagnostics.iloc[0].get("Warnings", "")).strip():
        warning_parts.append(str(suitability_diagnostics.iloc[0].get("Warnings")))
    perf_metrics = {}
    if performance_summary is not None and not performance_summary.empty and {"Metric", "Value"}.issubset(performance_summary.columns):
        perf_metrics = dict(zip(performance_summary["Metric"].astype(str), performance_summary["Value"]))
    validation_summary = {}
    if isinstance(validation, dict) and not validation.get("summary", pd.DataFrame()).empty:
        val = validation.get("summary")
        if {"Metric", "Value"}.issubset(val.columns):
            validation_summary = dict(zip(val["Metric"].astype(str), val["Value"]))
    base = {
        "config": cfg,
        "universe": list(config.tickers),
        "benchmark": config.benchmark_ticker,
        "benchmark_group": config.benchmark_group,
        "benchmark_mandate_type": config.benchmark_mandate_type,
        "objective": config.weight_objective,
        "constraints": {
            "top_n": config.top_n,
            "preselect_n": config.preselect_n,
            "min_chunk": config.min_chunk,
            "max_chunk": config.max_chunk,
            "max_weight": config.max_weight,
            "sector_weight_cap": config.sector_weight_cap,
            "target_vol": config.target_vol,
            "max_adv_participation": config.max_adv_participation,
            "min_dollar_volume": config.min_dollar_volume,
            "cvar_alpha": config.cvar_alpha,
            "cvar_penalty": config.cvar_penalty,
            "use_black_litterman": config.use_black_litterman,
            "black_litterman_tau": config.black_litterman_tau,
        },
        "data_timestamps": data_timestamps,
        "data_quality": data_quality,
        "warnings": [w for w in warning_parts if str(w).strip()],
        "code_version": code_version_hash(),
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    run_hash = canonical_hash(base, length=32)
    return {
        "run_hash": run_hash,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "code_version": base["code_version"],
        "app_version": base["app_version"],
        "model_version": base["model_version"],
        "schema_version": base["schema_version"],
        "config_hash": canonical_hash(cfg, length=32),
        "universe_hash": canonical_hash({"tickers": list(config.tickers)}, length=32),
        "data_hash": canonical_hash({"timestamps": data_timestamps, "quality": data_quality}, length=32),
        "config": cfg,
        "universe": list(config.tickers),
        "benchmark": config.benchmark_ticker,
        "objective": config.weight_objective,
        "constraints": base["constraints"],
        "data_timestamps": data_timestamps,
        "data_quality": data_quality,
        "warnings": base["warnings"],
        "performance_metrics": registry_json_safe(perf_metrics),
        "validation_metrics": registry_json_safe(validation_summary),
        "timings": registry_json_safe(timings),
    }


def persist_model_registry_record(record: dict) -> None:
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    run_hash = str(record.get("run_hash", canonical_hash(record, length=32)))
    (MODEL_REGISTRY_DIR / f"{run_hash}.json").write_text(json.dumps(registry_json_safe(record), indent=2, sort_keys=True), encoding="utf-8")
    flat = {
        "run_hash": run_hash,
        "created_at": record.get("created_at"),
        "code_version": record.get("code_version"),
        "config_hash": record.get("config_hash"),
        "universe_hash": record.get("universe_hash"),
        "data_hash": record.get("data_hash"),
        "benchmark": record.get("benchmark"),
        "objective": record.get("objective"),
        "universe_size": len(record.get("universe", []) or []),
        "warnings_count": len(record.get("warnings", []) or []),
    }
    for key, value in (record.get("performance_metrics") or {}).items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            flat[f"perf_{key}"] = value
    path = MODEL_REGISTRY_DIR / "registry.parquet"
    row = pd.DataFrame([registry_json_safe(flat)])
    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old[old.get("run_hash") != run_hash], row], ignore_index=True)
    else:
        out = row
    out.to_parquet(path, index=False)


def load_model_registry(limit: int = 100) -> pd.DataFrame:
    path = MODEL_REGISTRY_DIR / "registry.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "created_at" in df:
        df = df.sort_values("created_at", ascending=False)
    return df.head(limit)


def _validated_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError(f"Unsupported public-data URL scheme or host: {parsed.scheme or '<missing>'}")
    return urllib.parse.urlunparse(parsed)


def http_read_json(url: str, user_agent: str = "QuantStockPicker/1.0", timeout: int = 20) -> dict:
    url = _validated_http_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def http_read_text(url: str, user_agent: str = "QuantStockPicker/1.0", timeout: int = 20) -> str:
    url = _validated_http_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


ENGLISH_LANGUAGE_CODES = {"", "en", "eng", "english", "en-us", "en-gb"}
TRANSLATION_NON_ASCII_THRESHOLD = 0.08


def text_needs_english_translation(text: str, language: str | None = None) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    lang = str(language or "").strip().lower()
    if lang and lang not in ENGLISH_LANGUAGE_CODES:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for ch in letters if ord(ch) > 127)
    return safe_div(non_ascii, len(letters)) >= TRANSLATION_NON_ASCII_THRESHOLD


def translate_text_to_english(
    text: str,
    language: str | None = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 168,
) -> tuple[str, str]:
    """Translate public-news snippets to English with a zero-cost public endpoint.

    The function is deliberately conservative: if the endpoint is unavailable,
    the original text is returned with a failure status so downstream views
    remain non-blocking and auditable.
    """
    raw = str(text or "").strip()
    if not raw:
        return raw, "blank"
    if not text_needs_english_translation(raw, language):
        return raw, "not_required"

    payload = {
        "v": 1,
        "target": "en",
        "language": str(language or "").lower(),
        "text_sha": hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest(),
    }
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("public_news_translation_en", payload, cache_ttl_hours)
        if cached is not None and not cached.empty and "Translated_Text" in cached:
            translated = str(cached["Translated_Text"].iloc[0] or raw).strip()
            status = str(cached.get("Translation_Status", pd.Series(["translated_cache"])).iloc[0] or "translated_cache")
            return translated or raw, status

    try:
        params = urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": "auto",
                "tl": "en",
                "dt": "t",
                "q": raw[:900],
            }
        )
        data = http_read_json(f"https://translate.googleapis.com/translate_a/single?{params}", user_agent="Mozilla/5.0", timeout=6)
        parts = data[0] if isinstance(data, list) and data else []
        translated = "".join(str(part[0]) for part in parts if isinstance(part, list) and part and part[0]).strip()
        status = "translated" if translated and translated.lower() != raw.lower() else "translation_same"
        if use_cache:
            PERSISTENT_CACHE.set_df(
                "public_news_translation_en",
                payload,
                pd.DataFrame([{"Original_Text": raw, "Translated_Text": translated or raw, "Translation_Status": status}]),
            )
        return translated or raw, status
    except Exception as exc:
        return raw, f"translation_unavailable:{type(exc).__name__}"


def add_english_article_titles(
    articles: pd.DataFrame,
    use_cache: bool = True,
    cache_ttl_hours: int = 168,
    max_workers: int = 4,
) -> pd.DataFrame:
    if articles is None or articles.empty or "Title" not in articles:
        return articles if articles is not None else pd.DataFrame()
    out = articles.copy()
    if "Original_Title" not in out:
        out["Original_Title"] = out["Title"]

    def one(row: pd.Series) -> tuple[str, str]:
        return translate_text_to_english(
            row.get("Original_Title", row.get("Title", "")),
            row.get("Language"),
            use_cache=use_cache,
            cache_ttl_hours=cache_ttl_hours,
        )

    rows = [row for _, row in out.iterrows()]
    translated: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(rows)))) as ex:
        futures = [ex.submit(one, row) for row in rows]
        for fut in futures:
            try:
                translated.append(fut.result())
            except Exception as exc:
                translated.append(("", f"translation_unavailable:{type(exc).__name__}"))
    out["Title_EN"] = [t[0] or str(out["Original_Title"].iloc[i]) for i, t in enumerate(translated)]
    out["Translation_Status"] = [t[1] for t in translated]
    out["Title"] = out["Title_EN"]
    return out


def normalize_ticker_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def normalize_side_ticker_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    return SIDE_TICKER_ALIASES.get(raw, normalize_ticker_symbol(raw))


def normalize_side_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys([normalize_side_ticker_symbol(t) for t in tickers if str(t).strip()]))


def side_boom_fixed_weight_map(config: RunConfig) -> dict[str, float]:
    fixed: dict[str, float] = {}
    primary = normalize_side_ticker_symbol(config.side_boom_fixed_ticker)
    if primary:
        fixed[primary] = float(np.clip(config.side_boom_fixed_weight, 0.0, 1.0))
    for ticker, weight in getattr(config, "side_boom_fixed_weights", ()) or ():
        norm = normalize_side_ticker_symbol(ticker)
        if norm:
            fixed[norm] = float(np.clip(to_float(weight), 0.0, 1.0))
    total = sum(weight for weight in fixed.values() if pd.notna(weight))
    if total > 1.0:
        fixed = {ticker: weight / total for ticker, weight in fixed.items()}
    return {ticker: weight for ticker, weight in fixed.items() if weight > 1e-12}


def load_sec_company_tickers(use_cache: bool = True, cache_ttl_hours: int = 168, user_agent: str = "QuantStockPicker/1.0") -> pd.DataFrame:
    payload = {"source": "sec_company_tickers"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("universe_sec_company_tickers", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    url = "https://www.sec.gov/files/company_tickers.json"
    data = http_read_json(url, user_agent=user_agent)
    rows = []
    for item in data.values():
        ticker = normalize_ticker_symbol(item.get("ticker", ""))
        if ticker:
            rows.append(
                {
                    "Ticker": ticker,
                    "CIK": str(item.get("cik_str", "")).zfill(10),
                    "Name": item.get("title"),
                    "Universe_Source": "SEC company_tickers",
                    "Source_Status": "ok",
                }
            )
    df = pd.DataFrame(rows).drop_duplicates("Ticker")
    if use_cache:
        PERSISTENT_CACHE.set_df("universe_sec_company_tickers", payload, df)
    return df


def load_nasdaq_trader_universe(use_cache: bool = True, cache_ttl_hours: int = 24) -> pd.DataFrame:
    payload = {"source": "nasdaqtrader_symboldir"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("universe_nasdaq_trader", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    rows = []
    endpoints = [
        ("NASDAQ", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
        ("OTHER_LISTED", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"),
    ]
    for venue, url in endpoints:
        try:
            text = http_read_text(url, timeout=20)
            lines = [ln for ln in text.splitlines() if "|" in ln and not ln.startswith("File Creation Time")]
            if not lines:
                continue
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
            symbol_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            name_col = "Security Name" if "Security Name" in df.columns else "Security Name"
            for _, row in df.iterrows():
                ticker = normalize_ticker_symbol(row.get(symbol_col, ""))
                if not ticker or ticker == "SYMBOL":
                    continue
                test_issue = str(row.get("Test Issue", row.get("ETF", "N"))).upper()
                if test_issue == "Y":
                    continue
                rows.append(
                    {
                        "Ticker": ticker,
                        "Name": row.get(name_col),
                        "Venue": venue,
                        "Universe_Source": "NasdaqTrader SymDir",
                        "Source_Status": "ok",
                    }
                )
        except Exception:
            continue
    df = pd.DataFrame(rows).drop_duplicates("Ticker") if rows else pd.DataFrame()
    if use_cache and not df.empty:
        PERSISTENT_CACHE.set_df("universe_nasdaq_trader", payload, df)
    return df


def load_sp500_wikipedia_asof(asof_date=None, use_cache: bool = True, cache_ttl_hours: int = 24) -> pd.DataFrame:
    asof = pd.Timestamp(asof_date) if asof_date is not None else pd.Timestamp.today()
    payload = {"source": "wikipedia_sp500", "asof": str(asof.date()), "method": "current_plus_changes_v2"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("universe_sp500_wikipedia", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = http_read_text(url, user_agent="QuantStockPicker/1.0", timeout=30)
    tables = pd.read_html(io.StringIO(html))
    constituents = tables[0].copy()
    current = constituents.rename(columns={"Symbol": "Ticker", "Security": "Name", "GICS Sector": "Sector"})
    current["Ticker"] = current["Ticker"].map(normalize_ticker_symbol)
    current["Universe_Source"] = "Wikipedia S&P 500"
    current["Source_Status"] = "current_constituents"
    current["Universe_AsOf"] = pd.Timestamp.today().normalize()

    if asof >= pd.Timestamp.today().normalize() or len(tables) < 2:
        out = current[["Ticker", "Name", "Sector", "Universe_Source", "Source_Status", "Universe_AsOf"]].drop_duplicates("Ticker")
        if use_cache:
            PERSISTENT_CACHE.set_df("universe_sp500_wikipedia", payload, out)
        return out

    out = current.set_index("Ticker")
    changes = tables[1].copy()
    changes.columns = ["_".join([str(x) for x in col if str(x) != "nan"]).strip("_") if isinstance(col, tuple) else str(col) for col in changes.columns]
    date_col = next((c for c in changes.columns if "Date" in c), None)
    added_col = next((c for c in changes.columns if "Added_Ticker" in c or c == "Added"), None)
    removed_col = next((c for c in changes.columns if "Removed_Ticker" in c or c == "Removed"), None)
    if date_col and added_col and removed_col:
        changes[date_col] = pd.to_datetime(changes[date_col], errors="coerce")
        for _, row in changes[changes[date_col] > asof].sort_values(date_col, ascending=False).iterrows():
            added = normalize_ticker_symbol(row.get(added_col, ""))
            removed = normalize_ticker_symbol(row.get(removed_col, ""))
            if added in out.index:
                out = out.drop(index=added)
            if removed and removed not in out.index:
                out.loc[removed, ["Name", "Sector", "Universe_Source", "Source_Status", "Universe_AsOf"]] = [
                    np.nan,
                    np.nan,
                    "Wikipedia S&P 500 historical changes",
                    "approx_asof_reconstructed",
                    asof.normalize(),
                ]
    out = out.reset_index()
    if use_cache:
        PERSISTENT_CACHE.set_df("universe_sp500_wikipedia", payload, out)
    return out


def load_local_optional_universe(source: str, filename: str) -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "universes" / filename
    if not path.exists():
        return pd.DataFrame(columns=["Ticker", "Universe_Source", "Source_Status"])
    try:
        df = pd.read_csv(path)
        if "Ticker" not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: "Ticker"})
        df["Ticker"] = df["Ticker"].map(normalize_ticker_symbol)
        df["Universe_Source"] = source
        df["Source_Status"] = "local_file"
        return df.drop_duplicates("Ticker")
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Universe_Source", "Source_Status"])


def load_public_universe(
    source: str,
    asof_date=None,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
    user_agent: str = "QuantStockPicker/1.0",
) -> pd.DataFrame:
    if source == "Wikipedia S&P 500 as-of":
        return load_sp500_wikipedia_asof(asof_date, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    if source == "NasdaqTrader listed":
        return load_nasdaq_trader_universe(use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    if source == "SEC company tickers":
        return load_sec_company_tickers(use_cache=use_cache, cache_ttl_hours=cache_ttl_hours, user_agent=user_agent)
    if source == "Stooq local delisted CSV":
        return load_local_optional_universe(source, "stooq_delisted.csv")
    if source == "Kaggle local CSV":
        return load_local_optional_universe(source, "kaggle_universe.csv")
    return pd.DataFrame()


def safe_div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def to_float(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_info(tk):
    try:
        return tk.info if tk.info is not None else {}
    except Exception:
        return {}


def safe_fast_info(tk):
    try:
        return dict(tk.fast_info)
    except Exception:
        return {}


def safe_statement(tk, kind: str, freq: str = "yearly") -> pd.DataFrame:
    try:
        if kind == "income":
            df = tk.get_income_stmt(freq=freq)
        elif kind == "balance":
            df = tk.get_balance_sheet(freq=freq)
        elif kind == "cashflow":
            df = tk.get_cashflow(freq=freq)
        else:
            df = pd.DataFrame()
        return df.copy() if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        try:
            attr_map = {
                ("income", "yearly"): "income_stmt",
                ("income", "quarterly"): "quarterly_income_stmt",
                ("income", "trailing"): "ttm_income_stmt",
                ("balance", "yearly"): "balance_sheet",
                ("balance", "quarterly"): "quarterly_balance_sheet",
                ("cashflow", "yearly"): "cashflow",
                ("cashflow", "quarterly"): "quarterly_cashflow",
                ("cashflow", "trailing"): "ttm_cashflow",
            }
            df = getattr(tk, attr_map.get((kind, freq), ""), pd.DataFrame())
            return df.copy() if df is not None and not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()


def get_statement_value(df: pd.DataFrame, labels: Iterable[str], col):
    if df is None or df.empty or col is None or col not in df.columns:
        return np.nan
    normalized_index = {re.sub(r"[^a-z0-9]", "", str(idx).lower()): idx for idx in df.index}
    for label in labels:
        if label in df.index:
            return to_float(df.loc[label, col])
        normalized_label = re.sub(r"[^a-z0-9]", "", str(label).lower())
        if normalized_label in normalized_index:
            return to_float(df.loc[normalized_index[normalized_label], col])
    return np.nan


def get_info_value(info: dict, labels: Iterable[str]):
    for label in labels:
        value = to_float(info.get(label))
        if pd.notna(value):
            return value
    return np.nan


def statement_cols(*dfs):
    cols = []
    for df in dfs:
        if df is not None and not df.empty:
            cols.extend(df.columns)
    out = []
    for c in cols:
        try:
            out.append(pd.Timestamp(c))
        except Exception:
            pass
    return sorted(pd.Index(out).unique())


def matching_col(df, period_ts):
    if df is None or df.empty or period_ts is None:
        return None
    for c in df.columns:
        if pd.Timestamp(c) == pd.Timestamp(period_ts):
            return c
    return None


def download_prices(
    tickers: Iterable[str],
    period: str = "3y",
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        return pd.DataFrame()
    payload = {"tickers": tickers, "period": period, "auto_adjust": True}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("prices_daily", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached.sort_index().ffill().dropna(axis=1, how="all")
    raw = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.sort_index().ffill().dropna(axis=1, how="all")
    if use_cache:
        PERSISTENT_CACHE.set_df("prices_daily", payload, prices)
    return prices


def download_volume(
    tickers: Iterable[str],
    period: str = "3y",
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        return pd.DataFrame()
    payload = {"tickers": tickers, "period": period, "field": "Volume"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("volume_daily", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached.sort_index().ffill().dropna(axis=1, how="all")
    raw = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        vol = raw["Volume"].copy()
    else:
        vol = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    vol = vol.sort_index().ffill().dropna(axis=1, how="all")
    if use_cache:
        PERSISTENT_CACHE.set_df("volume_daily", payload, vol)
    return vol


def piotroski_score(income, balance, cashflow, col, prev_col):
    if col is None or prev_col is None:
        return np.nan
    def gt(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else bool(a > b)
    def lt(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else bool(a < b)
    def le(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else bool(a <= b)
    def positive(a):
        return np.nan if pd.isna(a) else bool(a > 0)
    try:
        ni_0 = get_statement_value(income, ["Net Income", "Net Income Common Stockholders"], col)
        ni_1 = get_statement_value(income, ["Net Income", "Net Income Common Stockholders"], prev_col)
        ta_0 = get_statement_value(balance, ["Total Assets"], col)
        ta_1 = get_statement_value(balance, ["Total Assets"], prev_col)
        td_0 = get_statement_value(balance, ["Total Debt", "Long Term Debt"], col)
        td_1 = get_statement_value(balance, ["Total Debt", "Long Term Debt"], prev_col)
        ca_0 = get_statement_value(balance, ["Current Assets", "Total Current Assets"], col)
        ca_1 = get_statement_value(balance, ["Current Assets", "Total Current Assets"], prev_col)
        cl_0 = get_statement_value(balance, ["Current Liabilities", "Total Current Liabilities"], col)
        cl_1 = get_statement_value(balance, ["Current Liabilities", "Total Current Liabilities"], prev_col)
        gp_0 = get_statement_value(income, ["Gross Profit"], col)
        gp_1 = get_statement_value(income, ["Gross Profit"], prev_col)
        rev_0 = get_statement_value(income, ["Total Revenue", "Operating Revenue", "Revenue"], col)
        rev_1 = get_statement_value(income, ["Total Revenue", "Operating Revenue", "Revenue"], prev_col)
        cfo_0 = get_statement_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"], col)
        sh_0 = get_statement_value(balance, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], col)
        sh_1 = get_statement_value(balance, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"], prev_col)

        checks = [
            positive(ni_0),
            positive(cfo_0),
            gt(safe_div(ni_0, ta_0), safe_div(ni_1, ta_1)),
            gt(cfo_0, ni_0),
            lt(safe_div(td_0, ta_0), safe_div(td_1, ta_1)),
            gt(safe_div(ca_0, cl_0), safe_div(ca_1, cl_1)),
            le(sh_0, sh_1) if pd.notna(sh_0) and pd.notna(sh_1) else np.nan,
            gt(safe_div(gp_0, rev_0), safe_div(gp_1, rev_1)),
            gt(safe_div(rev_0, ta_0), safe_div(rev_1, ta_1)),
        ]
        valid = [x for x in checks if isinstance(x, (bool, np.bool_))]
        return float(sum(valid)) if len(valid) >= 7 else np.nan
    except Exception:
        return np.nan


INCOME_LABELS = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA", "Ebitda"],
    "gross_profit": ["Gross Profit"],
    "pretax": ["Pretax Income", "Pre Tax Income"],
    "tax": ["Tax Provision", "Income Tax Expense"],
    "interest": ["Interest Expense", "Interest Expense Non Operating", "Net Non Operating Interest Income Expense"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
        "Net Income From Continuing And Discontinued Operation",
    ],
    "depreciation_income": ["Reconciled Depreciation"],
}

BALANCE_LABELS = {
    "assets": ["Total Assets"],
    "liabilities": ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "debt": ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt", "Current Debt And Capital Lease Obligation"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments", "Cash"],
    "retained": ["Retained Earnings"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "shares": ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
    "working_capital": ["Working Capital"],
}

CASHFLOW_LABELS = {
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
    "fcf": ["Free Cash Flow"],
    "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "depreciation_amortization": [
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
        "Depreciation",
        "Depreciation Depletion And Amortization",
    ],
}


def combine_statement_blocks(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in blocks if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[~out.index.duplicated(keep="first")]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
    return out.sort_index(axis=1)


def _build_fundamental_panel_sequential(tickers: Iterable[str], accounting_lag_days: int = 90) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        tk = yf.Ticker(ticker)
        try:
            info = safe_info(tk)
            fast = safe_fast_info(tk)
            income = combine_statement_blocks(
                [
                    safe_statement(tk, "income", "quarterly"),
                    safe_statement(tk, "income", "trailing"),
                    safe_statement(tk, "income", "yearly"),
                ]
            )
            balance = combine_statement_blocks(
                [
                    safe_statement(tk, "balance", "quarterly"),
                    safe_statement(tk, "balance", "yearly"),
                ]
            )
            cashflow = combine_statement_blocks(
                [
                    safe_statement(tk, "cashflow", "quarterly"),
                    safe_statement(tk, "cashflow", "trailing"),
                    safe_statement(tk, "cashflow", "yearly"),
                ]
            )
        except Exception:
            continue

        periods = statement_cols(income, balance, cashflow)
        if not periods:
            continue

        sector = info.get("sector", "Unknown")
        country = info.get("country", "Unknown")
        current_shares = to_float(fast.get("shares") if "shares" in fast else info.get("sharesOutstanding"))
        quote_market_cap = get_info_value(info, ["marketCap"])
        quote_enterprise_value = get_info_value(info, ["enterpriseValue"])
        quote_revenue = get_info_value(info, ["totalRevenue"])
        quote_ebitda = get_info_value(info, ["ebitda"])
        quote_cfo = get_info_value(info, ["operatingCashflow"])
        quote_fcf = get_info_value(info, ["freeCashflow"])
        quote_debt = get_info_value(info, ["totalDebt"])
        quote_cash = get_info_value(info, ["totalCash"])
        quote_eps = get_info_value(info, ["trailingEps", "forwardEps"])
        quote_book_value = get_info_value(info, ["bookValue"])
        quote_price_to_book = get_info_value(info, ["priceToBook"])
        quote_pe = get_info_value(info, ["trailingPE", "forwardPE"])
        quote_roe = get_info_value(info, ["returnOnEquity"])

        for i, period_ts in enumerate(periods):
            col_i = matching_col(income, period_ts)
            bal_col_i = matching_col(balance, period_ts)
            cf_col_i = matching_col(cashflow, period_ts)
            prev_col_i = matching_col(income, periods[i - 1]) if i > 0 else None

            revenue = get_statement_value(income, INCOME_LABELS["revenue"], col_i)
            ebit = get_statement_value(income, INCOME_LABELS["ebit"], col_i)
            ebitda = get_statement_value(income, INCOME_LABELS["ebitda"], col_i)
            gross_profit = get_statement_value(income, INCOME_LABELS["gross_profit"], col_i)
            pretax = get_statement_value(income, INCOME_LABELS["pretax"], col_i)
            tax = get_statement_value(income, INCOME_LABELS["tax"], col_i)
            interest = get_statement_value(income, INCOME_LABELS["interest"], col_i)
            net_income = get_statement_value(income, INCOME_LABELS["net_income"], col_i)
            assets = get_statement_value(balance, BALANCE_LABELS["assets"], bal_col_i)
            liabilities = get_statement_value(balance, BALANCE_LABELS["liabilities"], bal_col_i)
            current_assets = get_statement_value(balance, BALANCE_LABELS["current_assets"], bal_col_i)
            current_liabilities = get_statement_value(balance, BALANCE_LABELS["current_liabilities"], bal_col_i)
            debt = get_statement_value(balance, BALANCE_LABELS["debt"], bal_col_i)
            cash = get_statement_value(balance, BALANCE_LABELS["cash"], bal_col_i)
            retained = get_statement_value(balance, BALANCE_LABELS["retained"], bal_col_i)
            equity = get_statement_value(balance, BALANCE_LABELS["equity"], bal_col_i)
            shares = get_statement_value(balance, BALANCE_LABELS["shares"], bal_col_i)
            working_capital_statement = get_statement_value(balance, BALANCE_LABELS["working_capital"], bal_col_i)
            shares_source = "statement"
            if pd.isna(shares):
                shares = current_shares
                shares_source = "current_snapshot_fallback"
            cfo = get_statement_value(cashflow, CASHFLOW_LABELS["cfo"], cf_col_i)
            capex = get_statement_value(cashflow, CASHFLOW_LABELS["capex"], cf_col_i)
            fcf_statement = get_statement_value(cashflow, CASHFLOW_LABELS["fcf"], cf_col_i)
            dividends = get_statement_value(cashflow, CASHFLOW_LABELS["dividends"], cf_col_i)
            depreciation_amortization = get_statement_value(cashflow, CASHFLOW_LABELS["depreciation_amortization"], cf_col_i)
            if pd.isna(depreciation_amortization):
                depreciation_amortization = get_statement_value(income, INCOME_LABELS["depreciation_income"], col_i)

            if pd.isna(equity) and pd.notna(assets) and pd.notna(liabilities):
                equity = assets - liabilities
            if pd.isna(equity) and pd.notna(quote_book_value) and pd.notna(shares):
                equity = quote_book_value * shares
            tax_rate = safe_div(tax, pretax)
            if pd.isna(tax_rate) or not np.isfinite(tax_rate):
                tax_rate = 0.21
            tax_rate = float(np.clip(tax_rate, 0.0, 0.45))

            rows.append(
                {
                    "Ticker": ticker,
                    "Sector": sector,
                    "Country": country,
                    "Fundamental_Source": "yfinance",
                    "Period_End": pd.Timestamp(period_ts),
                    "Availability_Date": pd.Timestamp(period_ts) + pd.Timedelta(days=accounting_lag_days),
                    "Shares_Source": shares_source,
                    "Piotroski": piotroski_score(income, balance, cashflow, col_i, prev_col_i),
                    "_shares": shares,
                    "_revenue": revenue,
                    "_ebit": ebit,
                    "_ebitda": ebitda,
                    "_gross_profit": gross_profit,
                    "_tax": tax,
                    "_pretax": pretax,
                    "_net_income": net_income,
                    "_assets": assets,
                    "_liabilities": liabilities,
                    "_current_assets": current_assets,
                    "_current_liabilities": current_liabilities,
                    "_debt": debt,
                    "_cash": cash,
                    "_retained": retained,
                    "_equity": equity,
                    "_cfo": cfo,
                    "_capex": capex,
                    "_dividends": dividends,
                    "_depreciation_amortization": depreciation_amortization,
                    "_interest": interest,
                    "_fcf_statement": fcf_statement,
                    "_quote_market_cap": quote_market_cap,
                    "_quote_enterprise_value": quote_enterprise_value,
                    "_quote_revenue": quote_revenue,
                    "_quote_ebitda": quote_ebitda,
                    "_quote_cfo": quote_cfo,
                    "_quote_fcf": quote_fcf,
                    "_quote_debt": quote_debt,
                    "_quote_cash": quote_cash,
                    "_quote_eps": quote_eps,
                    "_quote_book_value": quote_book_value,
                    "_quote_price_to_book": quote_price_to_book,
                    "_quote_pe": quote_pe,
                    "_quote_roe": quote_roe,
                    "_nopat": ebit * (1.0 - tax_rate) if pd.notna(ebit) else np.nan,
                    "_working_capital": working_capital_statement
                    if pd.notna(working_capital_statement)
                    else (current_assets - current_liabilities if pd.notna(current_assets) and pd.notna(current_liabilities) else np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["Ticker", "Availability_Date"]).reset_index(drop=True)


def build_fundamental_panel(
    tickers: Iterable[str],
    accounting_lag_days: int = 90,
    max_workers: int = 8,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    """
    Parallel yfinance fundamental loader.

    yfinance calls are I/O-bound, so threads reduce wall-clock time without
    changing the deterministic financial calculations.
    """
    tickers = list(dict.fromkeys([t for t in tickers if t]))
    if not tickers:
        return pd.DataFrame()
    payload = {"tickers": sorted(tickers), "accounting_lag_days": accounting_lag_days, "schema": "yfinance_v2_quarterly_trailing_quote_fallback"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("fundamentals_yfinance", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            for col in ["Period_End", "Availability_Date"]:
                if col in cached.columns:
                    cached[col] = pd.to_datetime(cached[col], errors="coerce")
            return cached.sort_values(["Ticker", "Availability_Date"]).reset_index(drop=True)
    max_workers = max(1, min(int(max_workers), len(tickers)))
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_build_fundamental_panel_sequential, [ticker], accounting_lag_days): ticker for ticker in tickers}
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Availability_Date"]).reset_index(drop=True)
    if use_cache:
        PERSISTENT_CACHE.set_df("fundamentals_yfinance", payload, panel)
    return panel


SEC_FACT_TAGS = {
    "Revenues": "_revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "_revenue",
    "SalesRevenueNet": "_revenue",
    "GrossProfit": "_gross_profit",
    "OperatingIncomeLoss": "_ebit",
    "OperatingIncomeLossAvailableToCommonStockholdersBasic": "_ebit",
    "EarningsBeforeInterestTaxesDepreciationAmortization": "_ebitda",
    "NetIncomeLoss": "_net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "_net_income",
    "Assets": "_assets",
    "Liabilities": "_liabilities",
    "StockholdersEquity": "_equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "_equity",
    "AssetsCurrent": "_current_assets",
    "LiabilitiesCurrent": "_current_liabilities",
    "LongTermDebt": "_debt",
    "LongTermDebtAndFinanceLeaseObligations": "_debt",
    "DebtCurrent": "_debt_current",
    "CashAndCashEquivalentsAtCarryingValue": "_cash",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "_cash",
    "RetainedEarningsAccumulatedDeficit": "_retained",
    "NetCashProvidedByUsedInOperatingActivities": "_cfo",
    "PaymentsToAcquirePropertyPlantAndEquipment": "_capex",
    "PaymentsOfDividends": "_dividends",
    "DepreciationDepletionAndAmortization": "_depreciation_amortization",
    "DepreciationDepletionAndAmortizationExcludingPropertyPlantAndEquipment": "_depreciation_amortization",
    "DepreciationAmortizationAndAccretionNet": "_depreciation_amortization",
    "InterestExpenseNonOperating": "_interest",
    "InterestExpense": "_interest",
    "IncomeTaxExpenseBenefit": "_tax",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "_pretax",
    "EntityCommonStockSharesOutstanding": "_shares",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "_shares",
}

SEC_TEXT_RISK_TERMS = {
    "litigation": ["litigation", "lawsuit", "legal proceeding", "settlement", "subpoena", "investigation"],
    "impairment": ["impairment", "write-down", "writedown", "goodwill impairment", "asset impairment"],
    "going_concern": ["going concern", "substantial doubt", "continue as a going concern"],
    "debt_covenant": ["debt covenant", "covenant compliance", "default", "cross-default", "credit agreement"],
    "liquidity": ["liquidity risk", "working capital", "cash flow constraints", "capital resources"],
    "cyber": ["cybersecurity", "cyber attack", "data breach", "ransomware"],
    "regulatory": ["regulatory", "sanctions", "tariff", "export control", "antitrust"],
    "ai_automation": ["artificial intelligence", "machine learning", "automation", "generative ai"],
}


def strip_sec_html(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def text_word_counter(text: str, max_words: int = 80_000) -> Counter:
    words = re.findall(r"[a-z][a-z\-]{2,}", str(text).lower())
    if len(words) > max_words:
        words = words[:max_words]
    return Counter(words)


def cosine_counter_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return np.nan
    keys = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na > 0 and nb > 0 else np.nan


def sec_text_features(text: str, previous_text: str | None = None) -> dict:
    clean = strip_sec_html(text)
    words = text_word_counter(clean)
    word_count = max(sum(words.values()), 1)
    rows = {
        "SEC_Text_Word_Count": word_count,
    }
    weighted = 0.0
    for category, terms in SEC_TEXT_RISK_TERMS.items():
        count = 0
        for term in terms:
            if " " in term:
                count += clean.count(term)
            else:
                count += words.get(term, 0)
        freq = 10_000.0 * count / word_count
        rows[f"SEC_Text_{category}_Mentions"] = float(count)
        rows[f"SEC_Text_{category}_Freq10K"] = float(freq)
        if category != "ai_automation":
            weighted += freq
    prev_risk = np.nan
    similarity = np.nan
    if previous_text:
        prev_clean = strip_sec_html(previous_text)
        prev_words = text_word_counter(prev_clean)
        similarity = cosine_counter_similarity(words, prev_words)
        prev_wc = max(sum(prev_words.values()), 1)
        prev_weighted = 0.0
        for category, terms in SEC_TEXT_RISK_TERMS.items():
            if category == "ai_automation":
                continue
            c = 0
            for term in terms:
                c += prev_clean.count(term) if " " in term else prev_words.get(term, 0)
            prev_weighted += 10_000.0 * c / prev_wc
        prev_risk = prev_weighted
    deterioration = max(weighted - prev_risk, 0.0) if pd.notna(prev_risk) else 0.0
    novelty_penalty = max(1.0 - similarity, 0.0) if pd.notna(similarity) else 0.0
    text_risk = float(np.log1p(weighted) + 0.50 * np.log1p(deterioration) + 0.25 * novelty_penalty)
    rows.update(
        {
            "SEC_Text_Risk_Raw": float(weighted),
            "SEC_Text_Risk_Deterioration": float(deterioration),
            "SEC_Text_Similarity_Prior": similarity,
            "TextRisk_Score": text_risk,
            "SEC_TextRisk_Source": "sec_10k_10q_public_filing",
        }
    )
    return rows


def fetch_sec_filing_document(cik: str, accession: str, primary_doc: str, user_agent: str) -> str:
    cik_plain = str(cik).lstrip("0")
    acc_plain = str(accession).replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{acc_plain}/{primary_doc}"
    try:
        return http_read_text(url, user_agent=user_agent, timeout=30)
    except Exception:
        return ""


def build_sec_nlp_panel_for_ticker(ticker: str, cik: str, user_agent: str, max_filings: int = 2) -> pd.DataFrame:
    try:
        sub = http_read_json(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json", user_agent=user_agent, timeout=30)
    except Exception:
        return pd.DataFrame()
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    accepted = recent.get("acceptanceDateTime", [])
    filed = recent.get("filingDate", [])
    primary = recent.get("primaryDocument", [])
    rows, docs = [], []
    for i, form in enumerate(forms):
        form = str(form).upper()
        if form not in {"10-K", "10-Q"}:
            continue
        if i >= len(accessions) or i >= len(primary):
            continue
        docs.append(
            {
                "Ticker": ticker,
                "CIK": str(cik).zfill(10),
                "SEC_Form": form,
                "SEC_Accession": accessions[i],
                "SEC_Accepted_At": accepted[i] if i < len(accepted) else None,
                "SEC_Filing_Date": filed[i] if i < len(filed) else None,
                "Primary_Document": primary[i],
            }
        )
        if len(docs) >= max(max_filings, 1):
            break
    previous_text = None
    for doc in reversed(docs):
        text = fetch_sec_filing_document(doc["CIK"], doc["SEC_Accession"], doc["Primary_Document"], user_agent)
        features = sec_text_features(text, previous_text=previous_text)
        previous_text = text or previous_text
        row = {**doc, **features}
        row["SEC_Accepted_At"] = pd.to_datetime(row["SEC_Accepted_At"], errors="coerce", utc=True)
        row["Availability_Date"] = row["SEC_Accepted_At"].tz_convert(None).normalize() if pd.notna(row["SEC_Accepted_At"]) else pd.to_datetime(row["SEC_Filing_Date"], errors="coerce")
        row["SEC_Accepted_At"] = row["SEC_Accepted_At"].tz_convert(None) if pd.notna(row["SEC_Accepted_At"]) else pd.NaT
        rows.append(row)
    return pd.DataFrame(rows)


def build_sec_nlp_panel(
    tickers: Iterable[str],
    max_tickers: int = 30,
    max_filings: int = 2,
    max_workers: int = 6,
    use_cache: bool = True,
    cache_ttl_hours: int = 168,
    user_agent: str = "QuantStockPicker/1.0",
) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([normalize_ticker_symbol(t) for t in tickers]))[:max_tickers]
    payload = {"tickers": sorted(tickers), "max_filings": max_filings, "source": "sec_nlp"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("sec_nlp_filings", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    mapping = load_sec_company_tickers(use_cache=use_cache, cache_ttl_hours=cache_ttl_hours, user_agent=user_agent)
    if mapping.empty:
        return pd.DataFrame()
    cik_map = mapping.set_index("Ticker")["CIK"].to_dict()
    jobs = [(tk, cik_map.get(tk)) for tk in tickers if cik_map.get(tk)]
    frames = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs) or 1))) as ex:
        futs = {ex.submit(build_sec_nlp_panel_for_ticker, tk, cik, user_agent, max_filings): tk for tk, cik in jobs}
        for fut in as_completed(futs):
            try:
                df = fut.result()
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if use_cache:
        PERSISTENT_CACHE.set_df("sec_nlp_filings", payload, out)
    return out


def merge_sec_nlp_into_panel(panel: pd.DataFrame, nlp: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or nlp.empty:
        return panel
    if "Ticker" not in panel.columns or "Ticker" not in nlp.columns:
        return panel
    keys = [k for k in ["Ticker", "CIK", "SEC_Accession"] if k in panel.columns and k in nlp.columns]
    nlp_cols = [c for c in nlp.columns if c.startswith("SEC_Text") or c.startswith("TextRisk") or c in ["Availability_Date", "SEC_Accepted_At"]]
    nlp_cols = [c for c in nlp_cols if c not in keys + ["Ticker"]]
    if "SEC_Accession" in keys:
        nlp_latest = nlp[keys + nlp_cols].drop_duplicates(subset=keys, keep="last")
        merged = panel.merge(nlp_latest, on=keys, how="left", suffixes=("", "_NLP"))
    else:
        if "Availability_Date" not in panel.columns or "Availability_Date" not in nlp.columns:
            return panel
        panel_tmp = panel.copy()
        panel_tmp["_row_order"] = np.arange(len(panel_tmp))
        panel_tmp["_panel_availability"] = pd.to_datetime(panel_tmp["Availability_Date"], errors="coerce")
        nlp_tmp = nlp[["Ticker"] + [c for c in nlp_cols if c != "Availability_Date"] + ["Availability_Date"]].copy()
        nlp_tmp["_nlp_availability"] = pd.to_datetime(nlp_tmp["Availability_Date"], errors="coerce")
        frames = []
        for ticker, g in panel_tmp.groupby("Ticker", sort=False):
            ng = nlp_tmp[nlp_tmp["Ticker"].eq(ticker)].dropna(subset=["_nlp_availability"]).sort_values("_nlp_availability")
            g_valid = g.dropna(subset=["_panel_availability"])
            g_missing = g[g["_panel_availability"].isna()]
            if ng.empty or g_valid.empty:
                frames.append(g)
                continue
            merged_g = pd.merge_asof(
                g_valid.sort_values("_panel_availability"),
                ng.drop(columns=["Ticker", "Availability_Date"]).sort_values("_nlp_availability"),
                left_on="_panel_availability",
                right_on="_nlp_availability",
                direction="backward",
            )
            if not g_missing.empty:
                merged_g = pd.concat([merged_g, g_missing], ignore_index=True, sort=False)
            frames.append(merged_g)
        merged = pd.concat(frames, ignore_index=True).sort_values("_row_order").drop(columns=["_row_order", "_panel_availability", "_nlp_availability"], errors="ignore")
    for col in ["Availability_Date_NLP", "SEC_Accepted_At_NLP"]:
        if col in merged:
            merged.drop(columns=[col], inplace=True)
    return merged


def build_sec_companyfacts_panel_for_ticker(ticker: str, cik: str, user_agent: str, accounting_lag_days: int = 90) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    try:
        data = http_read_json(url, user_agent=user_agent, timeout=30)
    except Exception:
        return pd.DataFrame()
    accepted_map = {}
    filing_map = {}
    try:
        sub = http_read_json(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json", user_agent=user_agent, timeout=30)
        recent = sub.get("filings", {}).get("recent", {})
        accession = recent.get("accessionNumber", [])
        accepted = recent.get("acceptanceDateTime", [])
        filed = recent.get("filingDate", [])
        accepted_map = {a: accepted[i] for i, a in enumerate(accession) if i < len(accepted)}
        filing_map = {a: filed[i] for i, a in enumerate(accession) if i < len(filed)}
    except Exception:
        accepted_map = {}
        filing_map = {}
    facts = data.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag, out_col in SEC_FACT_TAGS.items():
        if tag not in facts:
            continue
        units = facts[tag].get("units", {})
        unit_key = "USD" if "USD" in units else "shares" if "shares" in units else next(iter(units.keys()), None)
        if unit_key is None:
            continue
        for item in units.get(unit_key, []):
            form = str(item.get("form", "")).upper()
            if form not in {"10-K", "10-Q", "20-F", "40-F"}:
                continue
            end = pd.to_datetime(item.get("end"), errors="coerce")
            start = pd.to_datetime(item.get("start"), errors="coerce")
            filed = pd.to_datetime(item.get("filed"), errors="coerce")
            if pd.isna(end) or pd.isna(filed):
                continue
            duration_days = (end - start).days if pd.notna(start) else np.nan
            rows.append(
                {
                    "Ticker": ticker,
                    "CIK": str(cik).zfill(10),
                    "SEC_Start": start,
                    "Period_End": end,
                    "SEC_Filing_Date": filed,
                    "SEC_Form": form,
                    "SEC_FY": item.get("fy"),
                    "SEC_FP": item.get("fp"),
                    "SEC_Frame": item.get("frame"),
                    "SEC_Duration_Days": duration_days,
                    "SEC_Accession": item.get("accn"),
                    "SEC_Tag": tag,
                    "SEC_Unit": unit_key,
                    "Metric": out_col,
                    "Value": to_float(item.get("val")),
                }
            )
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows).dropna(subset=["Value"])
    for col in ["SEC_FY", "SEC_FP", "SEC_Frame"]:
        long[col] = long[col].fillna("NA")
    long["SEC_Start"] = pd.to_datetime(long["SEC_Start"], errors="coerce").fillna(pd.to_datetime(long["Period_End"], errors="coerce"))
    long["SEC_Duration_Days"] = pd.to_numeric(long["SEC_Duration_Days"], errors="coerce").fillna(0.0)
    long = long.sort_values(["Ticker", "Period_End", "SEC_Filing_Date"])
    base_keys = ["Ticker", "CIK", "Period_End", "SEC_Form", "SEC_FY", "SEC_FP", "SEC_Accession"]
    latest = long.groupby(base_keys + ["SEC_Frame", "Metric"], dropna=False, as_index=False).tail(1)
    pivot = latest.pivot_table(
        index=base_keys,
        columns="Metric",
        values="Value",
        aggfunc="last",
    ).reset_index()
    filing = latest.groupby(base_keys, dropna=False, as_index=False)["SEC_Filing_Date"].max()
    meta = (
        latest.groupby(base_keys, dropna=False)
        .agg(
            SEC_Start=("SEC_Start", "min"),
            SEC_Duration_Days=("SEC_Duration_Days", "max"),
            SEC_Frame=("SEC_Frame", lambda x: ",".join(sorted(set(str(v) for v in x if str(v) != "NA"))) or "NA"),
        )
        .reset_index()
    )
    out = pivot.merge(filing, on=base_keys, how="left").merge(meta, on=base_keys, how="left")
    out["SEC_Accepted_At"] = out["SEC_Accession"].map(accepted_map)
    out["SEC_Submission_Filing_Date"] = pd.to_datetime(out["SEC_Accession"].map(filing_map), errors="coerce")
    out["SEC_Accepted_At"] = pd.to_datetime(out["SEC_Accepted_At"], errors="coerce", utc=True).dt.tz_convert(None)
    out["Availability_Date"] = out["SEC_Accepted_At"].dt.normalize()
    out["Availability_Date"] = out["Availability_Date"].fillna(out["SEC_Submission_Filing_Date"])
    out["Availability_Date"] = out["Availability_Date"].fillna(out["SEC_Filing_Date"])
    out.loc[out["Availability_Date"].isna(), "Availability_Date"] = out["Period_End"] + pd.Timedelta(days=accounting_lag_days)
    out["Availability_Date"] = pd.to_datetime(out["Availability_Date"], errors="coerce").dt.tz_localize(None)
    out["SEC_Filing_Date"] = pd.to_datetime(out["SEC_Filing_Date"], errors="coerce").dt.tz_localize(None)
    out["Fundamental_Source"] = "SEC companyfacts"
    for col in [
        "_revenue", "_ebit", "_ebitda", "_net_income", "_assets", "_liabilities", "_current_assets",
        "_current_liabilities", "_debt", "_debt_current", "_cash", "_retained", "_equity", "_gross_profit", "_cfo", "_capex",
        "_dividends", "_depreciation_amortization", "_interest", "_shares", "_tax", "_pretax",
        "_nopat", "_working_capital", "Piotroski",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["Shares_Source"] = np.where(out["_shares"].notna(), "sec_companyfacts", "missing")
    if "_debt_current" in out.columns:
        out["_debt"] = out["_debt"].fillna(0.0) + out["_debt_current"].fillna(0.0)
    tax_rate = out.get("_tax", pd.Series(np.nan, index=out.index)) / out.get("_pretax", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    tax_rate = tax_rate.replace([np.inf, -np.inf], np.nan).fillna(0.21).clip(0.0, 0.45)
    out["_nopat"] = out["_ebit"] * (1.0 - tax_rate)
    out["_working_capital"] = out["_current_assets"] - out["_current_liabilities"]
    out["SEC_Facts_Coverage"] = out[[c for c in sorted(set(SEC_FACT_TAGS.values())) if c in out.columns]].notna().sum(axis=1)
    out["SEC_Period_Type"] = np.select(
        [
            out["SEC_Duration_Days"].between(70, 115, inclusive="both"),
            out["SEC_Duration_Days"].between(330, 390, inclusive="both"),
            out["SEC_Duration_Days"].isna(),
        ],
        ["quarterly_flow", "annual_flow", "instant_or_missing_start"],
        default="ytd_or_irregular_flow",
    )
    out["Sector"] = "Unknown"
    out["Country"] = "United States"
    return out.sort_values(["Ticker", "Availability_Date"]).reset_index(drop=True)


def build_sec_companyfacts_panel(
    tickers: Iterable[str],
    accounting_lag_days: int = 90,
    max_workers: int = 8,
    use_cache: bool = True,
    cache_ttl_hours: int = 168,
    user_agent: str = "QuantStockPicker/1.0 contact@example.com",
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([normalize_ticker_symbol(t) for t in tickers if t]))
    if not tickers:
        return pd.DataFrame()
    payload = {"tickers": sorted(tickers), "accounting_lag_days": accounting_lag_days, "source": "sec_companyfacts_v1"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("fundamentals_sec_companyfacts", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            for col in ["Period_End", "Availability_Date", "SEC_Filing_Date"]:
                if col in cached.columns:
                    cached[col] = pd.to_datetime(cached[col], errors="coerce")
            return cached
    try:
        mapping = load_sec_company_tickers(use_cache=use_cache, cache_ttl_hours=cache_ttl_hours, user_agent=user_agent)
    except Exception:
        return pd.DataFrame()
    if mapping.empty:
        return pd.DataFrame()
    cik_map = mapping.set_index("Ticker")["CIK"].to_dict()
    jobs = [(tk, cik_map.get(tk)) for tk in tickers if cik_map.get(tk)]
    if not jobs:
        return pd.DataFrame()
    frames = []
    max_workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(build_sec_companyfacts_panel_for_ticker, tk, cik, user_agent, accounting_lag_days): tk
            for tk, cik in jobs
        }
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Availability_Date"]).reset_index(drop=True)
    if use_cache:
        PERSISTENT_CACHE.set_df("fundamentals_sec_companyfacts", payload, panel)
    return panel


def merge_fundamental_sources(yf_panel: pd.DataFrame, sec_panel: pd.DataFrame) -> pd.DataFrame:
    if yf_panel.empty:
        return sec_panel
    out = yf_panel.copy()
    if "Fundamental_Source" not in out.columns:
        out["Fundamental_Source"] = "yfinance"
    if sec_panel.empty:
        return out
    sec = sec_panel.copy()
    meta = out.sort_values("Availability_Date").groupby("Ticker").tail(1)[["Ticker", "Sector", "Country"]]
    sec = sec.drop(columns=[c for c in ["Sector", "Country"] if c in sec.columns], errors="ignore").merge(meta, on="Ticker", how="left")
    sec["Sector"] = sec["Sector"].fillna("Unknown")
    sec["Country"] = sec["Country"].fillna("United States")
    fallback_cols = [
        c for c in out.columns
        if c.startswith("_") or c in {"Piotroski", "Shares_Source"}
    ]
    if fallback_cols:
        yfs = out.sort_values(["Ticker", "Availability_Date"]).copy()
        filled_rows = []
        for _, sec_row in sec.iterrows():
            tk = sec_row.get("Ticker")
            asof = pd.Timestamp(sec_row.get("Availability_Date"))
            hist = yfs[(yfs["Ticker"].eq(tk)) & (pd.to_datetime(yfs["Availability_Date"], errors="coerce") <= asof)]
            if hist.empty:
                filled_rows.append(sec_row)
                continue
            fb = hist.tail(1).iloc[0]
            fallback_count = 0
            for col in fallback_cols:
                if col in sec.columns and pd.isna(sec_row.get(col)) and pd.notna(fb.get(col)):
                    sec_row[col] = fb.get(col)
                    fallback_count += 1
                elif col not in sec.columns and pd.notna(fb.get(col)):
                    sec_row[col] = fb.get(col)
                    fallback_count += 1
            if fallback_count:
                sec_row["Statement_Fallback_Source"] = "yfinance_latest_available_before_sec_asof"
                sec_row["Statement_Fallback_Count"] = fallback_count
            filled_rows.append(sec_row)
        sec = pd.DataFrame(filled_rows)
    all_cols = sorted(set(out.columns).union(sec.columns))
    return pd.concat([out.reindex(columns=all_cols), sec.reindex(columns=all_cols)], ignore_index=True).sort_values(["Ticker", "Availability_Date"]).reset_index(drop=True)


def price_asof(prices: pd.DataFrame, ticker: str, asof_date) -> float:
    if ticker not in prices.columns:
        return np.nan
    s = prices.loc[:asof_date, ticker].dropna()
    return to_float(s.iloc[-1]) if not s.empty else np.nan


def compute_ratios(row: dict, price: float) -> dict:
    out = dict(row)
    shares = to_float(out.get("_shares"))
    quote_market_cap = to_float(out.get("_quote_market_cap"))
    quote_enterprise_value = to_float(out.get("_quote_enterprise_value"))
    quote_eps = to_float(out.get("_quote_eps"))
    quote_book_value = to_float(out.get("_quote_book_value"))
    quote_pe = to_float(out.get("_quote_pe"))
    quote_price_to_book = to_float(out.get("_quote_price_to_book"))
    quote_roe = to_float(out.get("_quote_roe"))
    if pd.isna(shares) and pd.notna(price) and price > 0 and pd.notna(quote_market_cap):
        shares = quote_market_cap / price
        out["_shares"] = shares
        out["Shares_Source"] = "quote_market_cap_implied"
    market_cap = price * shares if pd.notna(price) and pd.notna(shares) else quote_market_cap
    debt = to_float(out.get("_debt"))
    cash = to_float(out.get("_cash"))
    ebitda = to_float(out.get("_ebitda"))
    ebit = to_float(out.get("_ebit"))
    revenue = to_float(out.get("_revenue"))
    net_income = to_float(out.get("_net_income"))
    equity = to_float(out.get("_equity"))
    assets = to_float(out.get("_assets"))
    liabilities = to_float(out.get("_liabilities"))
    cfo = to_float(out.get("_cfo"))
    capex = to_float(out.get("_capex"))
    dividends = to_float(out.get("_dividends"))
    depreciation_amortization = to_float(out.get("_depreciation_amortization"))
    interest = to_float(out.get("_interest"))
    retained = to_float(out.get("_retained"))
    nopat = to_float(out.get("_nopat"))
    tax = to_float(out.get("_tax"))
    pretax = to_float(out.get("_pretax"))
    working_capital = to_float(out.get("_working_capital"))
    if pd.isna(revenue):
        revenue = to_float(out.get("_quote_revenue"))
    if pd.isna(ebitda):
        ebitda = to_float(out.get("_quote_ebitda"))
    if pd.isna(cfo):
        cfo = to_float(out.get("_quote_cfo"))
    if pd.isna(debt):
        debt = to_float(out.get("_quote_debt"))
    if pd.isna(cash):
        cash = to_float(out.get("_quote_cash"))
    if pd.isna(ebitda) and pd.notna(ebit) and pd.notna(depreciation_amortization):
        ebitda = ebit + abs(depreciation_amortization)
    if pd.isna(nopat) and pd.notna(ebit):
        tax_rate = safe_div(tax, pretax)
        if pd.isna(tax_rate) or not np.isfinite(tax_rate):
            tax_rate = 0.21
        tax_rate = float(np.clip(tax_rate, 0.0, 0.45))
        nopat = ebit * (1.0 - tax_rate)

    ev = quote_enterprise_value if pd.notna(quote_enterprise_value) else market_cap
    if pd.notna(ev) and pd.isna(quote_enterprise_value):
        ev += debt if pd.notna(debt) else 0.0
        ev -= cash if pd.notna(cash) else 0.0
    fcf = to_float(out.get("_fcf_statement"))
    if pd.isna(fcf):
        fcf = cfo - abs(capex) if pd.notna(cfo) and pd.notna(capex) else to_float(out.get("_quote_fcf"))
    net_debt = (debt - cash) if pd.notna(debt) and pd.notna(cash) else np.nan
    invested_capital = np.nansum([equity if pd.notna(equity) else 0.0, debt if pd.notna(debt) else 0.0, -(cash if pd.notna(cash) else 0.0)])
    if invested_capital == 0:
        invested_capital = np.nan
    eps = safe_div(net_income, shares)
    if pd.isna(eps):
        eps = quote_eps
    bvps = safe_div(equity, shares)
    if pd.isna(bvps):
        bvps = quote_book_value
    dividends_paid = abs(dividends) if pd.notna(dividends) else 0.0
    retention_ratio = safe_div(net_income - dividends_paid, net_income) if pd.notna(net_income) and net_income > 0 else np.nan
    if pd.notna(retention_ratio):
        retention_ratio = float(np.clip(retention_ratio, -2.0, 2.0))

    altman = np.nan
    if all(pd.notna(x) for x in [working_capital, retained, ebit, market_cap, revenue, assets, liabilities]):
        altman = (
            1.2 * safe_div(working_capital, assets)
            + 1.4 * safe_div(retained, assets)
            + 3.3 * safe_div(ebit, assets)
            + 0.6 * safe_div(market_cap, liabilities)
            + safe_div(revenue, assets)
        )

    out.update(
        {
            "Price_AsOf": price,
            "Market_Cap_AsOf": market_cap,
            "Enterprise_Value": ev,
            "Free_Cash_Flow": fcf,
            "Net_Debt": net_debt,
            "NOPAT": nopat,
            "Invested_Capital": invested_capital,
            "EBITDA": ebitda,
            "Book_Value_Per_Share": bvps,
            "ROIC": safe_div(nopat, invested_capital),
            "EV_EBITDA": safe_div(ev, ebitda),
            "FCF_Yield": safe_div(fcf, market_cap),
            "NetDebt_EBITDA": safe_div(net_debt, ebitda),
            "Asset_Turnover": safe_div(revenue, assets),
            "Altman_Z": altman,
            "Interest_Coverage": safe_div(ebit, abs(interest) if pd.notna(interest) else np.nan),
            "Retention_Ratio": retention_ratio,
            "Earnings_Yield": safe_div(eps, price),
            "Price_Book": safe_div(price, bvps) if pd.notna(safe_div(price, bvps)) else quote_price_to_book,
            "PE_Ratio": safe_div(price, eps) if pd.notna(safe_div(price, eps)) else quote_pe,
            "PER": safe_div(price, eps) if pd.notna(safe_div(price, eps)) else quote_pe,
            "Price_to_Earnings": safe_div(price, eps) if pd.notna(safe_div(price, eps)) else quote_pe,
            "Price_to_Book_Value": safe_div(price, bvps) if pd.notna(safe_div(price, bvps)) else quote_price_to_book,
            "EPS": eps,
            "Solvency": safe_div(assets, liabilities),
            "ROE": safe_div(net_income, equity) if pd.notna(safe_div(net_income, equity)) else quote_roe,
        }
    )
    out["Market_Data_Fallback_Count"] = sum(
        pd.notna(out.get(c))
        for c in [
            "_quote_market_cap",
            "_quote_enterprise_value",
            "_quote_revenue",
            "_quote_ebitda",
            "_quote_cfo",
            "_quote_fcf",
            "_quote_debt",
            "_quote_cash",
            "_quote_eps",
            "_quote_book_value",
            "_quote_price_to_book",
            "_quote_pe",
            "_quote_roe",
        ]
    )
    out["Valid_Fundamental_Ratios"] = sum(pd.notna(out.get(c)) for c in CORE_RATIO_COLS)
    return out


def piotroski_from_panel_rows(row: pd.Series, prev: pd.Series | None) -> float:
    if prev is None or prev.empty:
        return np.nan
    def v(src, key):
        return to_float(src.get(key))
    def gt(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else bool(a > b)
    def lt(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else bool(a < b)
    def le(a, b):
        return np.nan if pd.isna(a) or pd.isna(b) else bool(a <= b)
    ni_0, ni_1 = v(row, "_net_income"), v(prev, "_net_income")
    cfo_0 = v(row, "_cfo")
    ta_0, ta_1 = v(row, "_assets"), v(prev, "_assets")
    debt_0, debt_1 = v(row, "_debt"), v(prev, "_debt")
    ca_0, ca_1 = v(row, "_current_assets"), v(prev, "_current_assets")
    cl_0, cl_1 = v(row, "_current_liabilities"), v(prev, "_current_liabilities")
    gp_0, gp_1 = v(row, "_gross_profit"), v(prev, "_gross_profit")
    rev_0, rev_1 = v(row, "_revenue"), v(prev, "_revenue")
    sh_0, sh_1 = v(row, "_shares"), v(prev, "_shares")
    checks = [
        np.nan if pd.isna(ni_0) else bool(ni_0 > 0),
        np.nan if pd.isna(cfo_0) else bool(cfo_0 > 0),
        gt(safe_div(ni_0, ta_0), safe_div(ni_1, ta_1)),
        gt(cfo_0, ni_0),
        lt(safe_div(debt_0, ta_0), safe_div(debt_1, ta_1)),
        gt(safe_div(ca_0, cl_0), safe_div(ca_1, cl_1)),
        le(sh_0, sh_1) if pd.notna(sh_0) and pd.notna(sh_1) else np.nan,
        gt(safe_div(gp_0, rev_0), safe_div(gp_1, rev_1)),
        gt(safe_div(rev_0, ta_0), safe_div(rev_1, ta_1)),
    ]
    valid = [x for x in checks if isinstance(x, (bool, np.bool_))]
    return float(sum(valid)) if len(valid) >= 7 else np.nan


def fundamentals_asof(panel: pd.DataFrame, prices: pd.DataFrame, asof_date) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    p = panel[pd.to_datetime(panel["Availability_Date"]) <= pd.Timestamp(asof_date)].copy()
    if p.empty:
        return pd.DataFrame()
    p = p.sort_values(["Ticker", "Period_End", "Availability_Date"])
    latest = p.groupby("Ticker").tail(1)
    rows = []
    for _, row in latest.iterrows():
        ticker_hist = p[p["Ticker"].eq(row["Ticker"])]
        prev_hist = ticker_hist[pd.to_datetime(ticker_hist["Period_End"]) < pd.Timestamp(row["Period_End"])]
        prev = prev_hist.tail(1).iloc[0] if not prev_hist.empty else None
        row_dict = row.to_dict()
        if pd.isna(row_dict.get("Piotroski")):
            row_dict["Piotroski"] = piotroski_from_panel_rows(row, prev)
        ratio = compute_ratios(row_dict, price_asof(prices, row["Ticker"], asof_date))
        prev_dict = prev.to_dict() if prev is not None else {}
        revenue = to_float(row_dict.get("_revenue"))
        prev_revenue = to_float(prev_dict.get("_revenue"))
        gross_profit = to_float(row_dict.get("_gross_profit"))
        prev_gross_profit = to_float(prev_dict.get("_gross_profit"))
        ebit = to_float(row_dict.get("_ebit"))
        cfo = to_float(row_dict.get("_cfo"))
        capex = to_float(row_dict.get("_capex"))
        fcf = cfo - abs(capex) if pd.notna(cfo) and pd.notna(capex) else np.nan
        prev_eps = safe_div(to_float(prev_dict.get("_net_income")), to_float(prev_dict.get("_shares")))
        ratio["Revenue_Growth"] = safe_div(revenue - prev_revenue, abs(prev_revenue)) if pd.notna(prev_revenue) and prev_revenue != 0 else np.nan
        ratio["EPS_Growth"] = safe_div(ratio.get("EPS") - prev_eps, abs(prev_eps)) if pd.notna(prev_eps) and prev_eps != 0 else np.nan
        ratio["Gross_Margin"] = safe_div(gross_profit, revenue)
        ratio["EBIT_Margin"] = safe_div(ebit, revenue)
        ratio["FCF_Margin"] = safe_div(fcf, revenue)
        ratio["Gross_Margin_Change"] = safe_div(gross_profit, revenue) - safe_div(prev_gross_profit, prev_revenue)
        rows.append(ratio)
    return pd.DataFrame(rows)


def robust_zscores(df: pd.DataFrame, cols: list[str], group_col: str = "Sector") -> pd.DataFrame:
    out = df.copy()
    out[group_col] = out[group_col].fillna("Unknown")
    for col in cols:
        if col not in out.columns:
            continue
        z_col = f"{col}_z"
        out[z_col] = np.nan
        global_x = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if global_x.notna().sum() == 0:
            out[z_col] = 0.0
            continue
        global_med = np.nanmedian(global_x)
        global_scale = iqr(global_x.dropna()) if global_x.notna().sum() > 1 else 1.0
        if pd.isna(global_scale) or global_scale == 0:
            global_scale = 1.0
        for _, idx in out.groupby(group_col).groups.items():
            x = pd.to_numeric(out.loc[idx, col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() >= 4:
                med = np.nanmedian(x)
                scale = iqr(x.dropna()) if x.notna().sum() > 1 else global_scale
                if pd.isna(scale) or scale == 0:
                    scale = global_scale
            else:
                med = global_med
                scale = global_scale
            out.loc[idx, z_col] = ((x - med) / scale).clip(-8, 8)
    return out


def add_mahalanobis(df: pd.DataFrame, cols: list[str], group_col: str = "Sector") -> pd.DataFrame:
    out = robust_zscores(df, cols, group_col)
    out["Mahalanobis"] = np.nan
    z_cols = [f"{c}_z" for c in cols if f"{c}_z" in out.columns]
    for _, idx in out.groupby(group_col).groups.items():
        z = out.loc[idx, z_cols].fillna(0.0).values.astype(float)
        if z.shape[0] < 2 or z.shape[1] < 2:
            out.loc[idx, "Mahalanobis"] = 0.0
            continue
        cov = np.atleast_2d(np.cov(z, rowvar=False))
        cov = 0.9 * cov + 0.1 * np.eye(cov.shape[0])
        inv = np.linalg.pinv(cov)
        center = z.mean(axis=0)
        out.loc[idx, "Mahalanobis"] = np.sqrt(np.einsum("ij,jk,ik->i", z - center, inv, z - center))
    return out


def price_features(prices: pd.DataFrame, asof_date) -> pd.DataFrame:
    px = prices.loc[:asof_date].sort_index().ffill()
    if len(px) < 130:
        return pd.DataFrame()
    ret = px.pct_change(fill_method=None)
    latest = px.iloc[-1]
    ma50 = px.rolling(50).mean().iloc[-1]
    ma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else pd.Series(np.nan, index=px.columns)
    window = px.iloc[-126:]
    dd = (window / window.cummax() - 1).min()
    return pd.DataFrame(
        {
            "Ticker": latest.index,
            "Price": latest.values,
            "Momentum_21": px.pct_change(21, fill_method=None).iloc[-1].reindex(latest.index).values,
            "Momentum_63": px.pct_change(63, fill_method=None).iloc[-1].reindex(latest.index).values,
            "Momentum_126": px.pct_change(126, fill_method=None).iloc[-1].reindex(latest.index).values,
            "Volatility_63": (ret.iloc[-63:].std() * np.sqrt(252)).reindex(latest.index).values,
            "Max_Drawdown_126": dd.reindex(latest.index).values,
            "Trend_50_200": (ma50 > ma200).astype(float).reindex(latest.index).values,
        }
    )


def liquidity_features(prices: pd.DataFrame, volumes: pd.DataFrame, asof_date, lookback: int = 63) -> pd.DataFrame:
    if volumes is None or volumes.empty:
        return pd.DataFrame()
    common = [c for c in prices.columns if c in volumes.columns]
    if not common:
        return pd.DataFrame()
    px = prices.loc[:asof_date, common].sort_index().ffill().tail(lookback + 1)
    vol = volumes.loc[:asof_date, common].sort_index().ffill().tail(lookback + 1)
    if px.empty or vol.empty:
        return pd.DataFrame()
    ret = px.pct_change(fill_method=None)
    dollar_volume = px * vol
    adv = vol.tail(lookback).mean()
    adv_dollar = dollar_volume.tail(lookback).mean()
    amihud = (ret.abs() / dollar_volume.replace(0, np.nan)).tail(lookback).mean()
    spread_proxy = (ret.abs().rolling(5).mean() / np.log1p(dollar_volume)).tail(lookback).mean()
    out = pd.DataFrame(
        {
            "Ticker": common,
            "ADV_63": adv.reindex(common).values,
            "Dollar_Volume_63": adv_dollar.reindex(common).values,
            "Amihud_ILLIQ_63": amihud.reindex(common).values,
            "Spread_Proxy_63": spread_proxy.reindex(common).values,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def fetch_option_chain_for_ticker(
    ticker: str,
    spot: float | None = None,
    max_expiries: int = 3,
    use_cache: bool = True,
    cache_ttl_hours: int = 6,
) -> pd.DataFrame:
    ticker = normalize_ticker_symbol(ticker)
    payload = {"ticker": ticker, "max_expiries": max_expiries, "snapshot_date": str(pd.Timestamp.today().date())}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("options_yahoo_snapshot", payload, cache_ttl_hours)
        if cached is not None:
            return cached
    rows = []
    try:
        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])[:max_expiries]
        for expiry in expiries:
            try:
                chain = tk.option_chain(expiry)
            except Exception:
                continue
            for opt_type, frame in [("call", chain.calls), ("put", chain.puts)]:
                if frame is None or frame.empty:
                    continue
                df = frame.copy()
                df["Ticker"] = ticker
                df["Option_Type"] = opt_type
                df["Expiry"] = pd.to_datetime(expiry)
                df["Snapshot_Date"] = pd.Timestamp.today().normalize()
                df["DTE"] = (df["Expiry"] - df["Snapshot_Date"]).dt.days
                if spot is not None and pd.notna(spot) and spot > 0:
                    df["Spot"] = float(spot)
                    df["Moneyness"] = df["strike"] / float(spot)
                rows.append(df)
    except Exception:
        return pd.DataFrame()
    if not rows:
        out = pd.DataFrame()
    else:
        out = pd.concat(rows, ignore_index=True)
        rename = {
            "contractSymbol": "Contract",
            "lastTradeDate": "Last_Trade_Date",
            "strike": "Strike",
            "lastPrice": "Last_Price",
            "bid": "Bid",
            "ask": "Ask",
            "change": "Change",
            "percentChange": "Percent_Change",
            "volume": "Volume",
            "openInterest": "Open_Interest",
            "impliedVolatility": "Implied_Vol",
            "inTheMoney": "In_The_Money",
        }
        out = out.rename(columns=rename)
        keep = [
            "Ticker", "Contract", "Option_Type", "Expiry", "DTE", "Strike", "Spot", "Moneyness",
            "Bid", "Ask", "Last_Price", "Implied_Vol", "Open_Interest", "Volume",
            "Last_Trade_Date", "In_The_Money", "Snapshot_Date",
        ]
        out = out[[c for c in keep if c in out.columns]].replace([np.inf, -np.inf], np.nan)
    if use_cache:
        PERSISTENT_CACHE.set_df("options_yahoo_snapshot", payload, out)
    return out


def fetch_options_snapshot(
    tickers: Iterable[str],
    prices: pd.DataFrame,
    max_expiries: int = 3,
    max_workers: int = 8,
    use_cache: bool = True,
    cache_ttl_hours: int = 6,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([normalize_ticker_symbol(t) for t in tickers if t]))
    if not tickers:
        return pd.DataFrame()
    max_workers = max(1, min(max_workers, len(tickers)))
    frames = []
    latest = prices.ffill().iloc[-1] if prices is not None and not prices.empty else pd.Series(dtype=float)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                fetch_option_chain_for_ticker,
                ticker,
                latest.get(ticker, np.nan),
                max_expiries,
                use_cache,
                cache_ttl_hours,
            ): ticker
            for ticker in tickers
        }
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_options_snapshot(options_chain: pd.DataFrame) -> pd.DataFrame:
    if options_chain is None or options_chain.empty:
        return pd.DataFrame()
    rows = []
    chain = options_chain.copy()
    chain["Mid"] = (pd.to_numeric(chain.get("Bid"), errors="coerce") + pd.to_numeric(chain.get("Ask"), errors="coerce")) / 2.0
    chain["Bid_Ask_Spread"] = pd.to_numeric(chain.get("Ask"), errors="coerce") - pd.to_numeric(chain.get("Bid"), errors="coerce")
    chain["Rel_Spread"] = chain["Bid_Ask_Spread"] / chain["Mid"].replace(0, np.nan)
    for ticker, df in chain.groupby("Ticker"):
        spot = pd.to_numeric(df.get("Spot"), errors="coerce").dropna()
        spot_val = float(spot.iloc[-1]) if not spot.empty else np.nan
        near = df[df["DTE"] >= 7].copy()
        if near.empty:
            near = df.copy()
        nearest_expiry = near.sort_values("DTE")["Expiry"].iloc[0] if "Expiry" in near else pd.NaT
        near = near[near["Expiry"] == nearest_expiry].copy() if pd.notna(nearest_expiry) else near
        atm_iv = np.nan
        if pd.notna(spot_val) and "Strike" in near:
            near["ATM_Dist"] = (pd.to_numeric(near["Strike"], errors="coerce") - spot_val).abs()
            atm = near.sort_values("ATM_Dist").groupby("Option_Type").head(1)
            atm_iv = pd.to_numeric(atm.get("Implied_Vol"), errors="coerce").mean()
        put_oi = pd.to_numeric(df.loc[df["Option_Type"] == "put", "Open_Interest"], errors="coerce").sum()
        call_oi = pd.to_numeric(df.loc[df["Option_Type"] == "call", "Open_Interest"], errors="coerce").sum()
        put_call_oi = safe_div(put_oi, call_oi)
        skew = np.nan
        if pd.notna(spot_val) and "Moneyness" in near:
            put_otm = near[near["Option_Type"].eq("put")].copy()
            call_otm = near[near["Option_Type"].eq("call")].copy()
            if not put_otm.empty and not call_otm.empty:
                put_iv = put_otm.iloc[(put_otm["Moneyness"] - 0.95).abs().argsort()[:1]]["Implied_Vol"].mean()
                call_iv = call_otm.iloc[(call_otm["Moneyness"] - 1.05).abs().argsort()[:1]]["Implied_Vol"].mean()
                skew = put_iv - call_iv
        rows.append(
            {
                "Ticker": ticker,
                "Spot": spot_val,
                "Nearest_Expiry": nearest_expiry,
                "Nearest_DTE": int(near["DTE"].dropna().iloc[0]) if "DTE" in near and near["DTE"].notna().any() else np.nan,
                "ATM_IV": atm_iv,
                "Skew_95P_105C": skew,
                "Put_Call_OpenInterest": put_call_oi,
                "Total_Call_OI": call_oi,
                "Total_Put_OI": put_oi,
                "Median_Rel_BidAsk": chain.loc[chain["Ticker"] == ticker, "Rel_Spread"].replace([np.inf, -np.inf], np.nan).median(),
                "Contracts": int(len(df)),
                "Source": "Yahoo options snapshot",
            }
        )
    return pd.DataFrame(rows).sort_values("ATM_IV", ascending=False)


def portfolio_implied_vol_surface(
    options_chain: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empty = {
        "portfolio_vol_surface": pd.DataFrame(),
        "portfolio_vol_surface_matrix": pd.DataFrame(),
        "portfolio_vol_surface_diagnostics": pd.DataFrame(),
    }
    if options_chain is None or options_chain.empty or portfolio is None or portfolio.empty:
        return empty
    if not {"Ticker", "DTE", "Moneyness", "Implied_Vol"}.issubset(options_chain.columns):
        return empty
    weights = portfolio.set_index("Ticker")["Weight"].astype(float)
    weights = weights[weights > 0]
    if weights.empty:
        return empty
    weights = weights / weights.sum()
    ch = options_chain.copy()
    ch["Ticker"] = ch["Ticker"].astype(str)
    ch = ch[ch["Ticker"].isin(weights.index)].copy()
    if ch.empty:
        return empty
    ch["Portfolio_Weight"] = ch["Ticker"].map(weights)
    ch["DTE"] = pd.to_numeric(ch["DTE"], errors="coerce")
    ch["Moneyness"] = pd.to_numeric(ch["Moneyness"], errors="coerce")
    ch["Implied_Vol"] = pd.to_numeric(ch["Implied_Vol"], errors="coerce")
    ch = ch.replace([np.inf, -np.inf], np.nan).dropna(subset=["DTE", "Moneyness", "Implied_Vol", "Portfolio_Weight"])
    ch = ch[(ch["DTE"] > 0) & (ch["Moneyness"].between(0.60, 1.60)) & (ch["Implied_Vol"].between(0.01, 5.0))]
    if ch.empty:
        return empty
    dte_bins = [0, 14, 30, 60, 90, 180, 365, 1095]
    dte_labels = ["0-14D", "15-30D", "31-60D", "61-90D", "91-180D", "181-365D", "1Y+"]
    mon_bins = [0.60, 0.75, 0.90, 0.97, 1.03, 1.10, 1.25, 1.60]
    mon_labels = ["0.60-0.75", "0.75-0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", "1.10-1.25", "1.25-1.60"]
    ch["DTE_Bucket"] = pd.cut(ch["DTE"], bins=dte_bins, labels=dte_labels, include_lowest=True)
    ch["Moneyness_Bucket"] = pd.cut(ch["Moneyness"], bins=mon_bins, labels=mon_labels, include_lowest=True)
    ch = ch.dropna(subset=["DTE_Bucket", "Moneyness_Bucket"])
    if ch.empty:
        return empty
    rows = []
    for (dte_bucket, mon_bucket), x in ch.groupby(["DTE_Bucket", "Moneyness_Bucket"], observed=True):
        w = x["Portfolio_Weight"].astype(float)
        iv = x["Implied_Vol"].astype(float)
        if w.sum() <= 0:
            continue
        rows.append(
            {
                "DTE_Bucket": str(dte_bucket),
                "Moneyness_Bucket": str(mon_bucket),
                "Weighted_IV": float(np.average(iv, weights=w)),
                "Median_IV": float(iv.median()),
                "Contracts": int(len(x)),
                "Tickers_Covered": int(x["Ticker"].nunique()),
                "Weight_Coverage": float(w.drop_duplicates().sum()) if "Ticker" not in x else float(weights.reindex(x["Ticker"].unique()).fillna(0.0).sum()),
                "Model_Note": "Weighted holding-level implied volatility surface; not a correlation-corrected basket option surface.",
            }
        )
    surface = pd.DataFrame(rows)
    if surface.empty:
        return empty
    matrix = surface.pivot_table(index="Moneyness_Bucket", columns="DTE_Bucket", values="Weighted_IV", aggfunc="mean")
    matrix = matrix.reindex(index=mon_labels, columns=dte_labels)
    diagnostics = pd.DataFrame(
        [
            {"Metric": "Portfolio_Weight_With_Option_Coverage", "Value": float(weights.reindex(ch["Ticker"].unique()).fillna(0.0).sum())},
            {"Metric": "Tickers_With_Option_Coverage", "Value": int(ch["Ticker"].nunique())},
            {"Metric": "Surface_Cells", "Value": int(len(surface))},
            {"Metric": "Contracts_Used", "Value": int(len(ch))},
            {"Metric": "Interpretation", "Value": "Approximate weighted implied-vol surface. True basket volatility requires correlation, dispersion, and cross-gamma modeling."},
        ]
    )
    return {
        "portfolio_vol_surface": surface.sort_values(["DTE_Bucket", "Moneyness_Bucket"]),
        "portfolio_vol_surface_matrix": matrix.reset_index(),
        "portfolio_vol_surface_diagnostics": diagnostics,
    }


def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)))


def normalized_weight_entropy(weights: pd.Series) -> float:
    w = pd.Series(weights).dropna()
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    return shannon_entropy(w.values / w.sum()) / np.log(len(w))


def softmax(x: pd.Series, temperature: float = 1.0) -> pd.Series:
    x = pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tau = max(float(temperature), 1e-6)
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum() if e.sum() > 0 else pd.Series(1.0 / len(x), index=x.index)


def ewma_vol_forecast(r: pd.Series, lam: float = 0.94) -> float:
    x = pd.Series(r).dropna().astype(float)
    if len(x) < 20:
        return np.nan
    var = float(x.var())
    for val in x.values:
        var = lam * var + (1.0 - lam) * val * val
    return float(np.sqrt(max(var, 0.0) * 252.0))


def garch11_forecast(r: pd.Series, max_obs: int = 756) -> tuple[float, str]:
    """
    Lightweight GARCH(1,1) QMLE. If optimization is unstable, returns EWMA.
    Returns annualized one-step volatility.
    """
    x = pd.Series(r).dropna().astype(float).tail(max_obs)
    if len(x) < 80:
        return ewma_vol_forecast(x), "ewma_insufficient_obs"
    y = (x - x.mean()).values
    var0 = np.var(y)
    if var0 <= 0 or not np.isfinite(var0):
        return np.nan, "invalid_variance"

    def nll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        h = np.empty_like(y)
        h[0] = var0
        for t in range(1, len(y)):
            h[t] = omega + alpha * y[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0 or not np.isfinite(h[t]):
                return 1e12
        return 0.5 * float(np.sum(np.log(h) + y * y / h))

    try:
        bounds = [(1e-12, var0 * 10), (1e-6, 0.35), (1e-6, 0.995)]
        cons = {"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}
        x0 = np.array([var0 * 0.05, 0.05, 0.90])
        res = minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 250, "ftol": 1e-7})
        if not res.success:
            return ewma_vol_forecast(x), f"ewma_garch_fail:{res.message}"
        omega, alpha, beta = res.x
        h_last = var0
        for t in range(1, len(y)):
            h_last = omega + alpha * y[t - 1] ** 2 + beta * h_last
        h_next = omega + alpha * y[-1] ** 2 + beta * h_last
        return float(np.sqrt(max(h_next, 0.0) * 252.0)), "garch11"
    except Exception as exc:
        return ewma_vol_forecast(x), f"ewma_exception:{type(exc).__name__}"


def _variance_recursion_nll(y: np.ndarray, model: str, params: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Gaussian QMLE negative log-likelihood for variance architectures.

    Models:
    - CONST: h_t = omega
    - ARCH1/ARCH2: h_t = omega + sum alpha_i eps_{t-i}^2
    - GARCH11/GARCH12/GARCH21: h_t = omega + alpha eps^2 + beta h
    """
    var0 = float(np.var(y))
    if var0 <= 0 or not np.isfinite(var0):
        return 1e12, np.full_like(y, np.nan)

    h = np.empty_like(y)
    h[0] = var0
    eps2 = y * y

    if model == "CONST":
        omega = params[0]
        if omega <= 0:
            return 1e12, h
        h[:] = omega
    elif model == "ARCH1":
        omega, a1 = params
        if omega <= 0 or a1 < 0 or a1 >= 0.999:
            return 1e12, h
        for t in range(1, len(y)):
            h[t] = omega + a1 * eps2[t - 1]
    elif model == "ARCH2":
        omega, a1, a2 = params
        if omega <= 0 or a1 < 0 or a2 < 0 or a1 + a2 >= 0.999:
            return 1e12, h
        for t in range(1, len(y)):
            lag2 = eps2[t - 2] if t >= 2 else var0
            h[t] = omega + a1 * eps2[t - 1] + a2 * lag2
    elif model == "GARCH11":
        omega, a1, b1 = params
        if omega <= 0 or a1 < 0 or b1 < 0 or a1 + b1 >= 0.999:
            return 1e12, h
        for t in range(1, len(y)):
            h[t] = omega + a1 * eps2[t - 1] + b1 * h[t - 1]
    elif model == "GARCH12":
        omega, a1, b1, b2 = params
        if omega <= 0 or a1 < 0 or b1 < 0 or b2 < 0 or a1 + b1 + b2 >= 0.999:
            return 1e12, h
        for t in range(1, len(y)):
            h_lag2 = h[t - 2] if t >= 2 else var0
            h[t] = omega + a1 * eps2[t - 1] + b1 * h[t - 1] + b2 * h_lag2
    elif model == "GARCH21":
        omega, a1, a2, b1 = params
        if omega <= 0 or a1 < 0 or a2 < 0 or b1 < 0 or a1 + a2 + b1 >= 0.999:
            return 1e12, h
        for t in range(1, len(y)):
            lag2 = eps2[t - 2] if t >= 2 else var0
            h[t] = omega + a1 * eps2[t - 1] + a2 * lag2 + b1 * h[t - 1]
    else:
        return 1e12, h

    if np.any(h <= 0) or np.any(~np.isfinite(h)):
        return 1e12, h
    nll = 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(h) + eps2 / h))
    return nll, h


def _student_t_garch11_nll(y: np.ndarray, params: np.ndarray) -> tuple[float, np.ndarray]:
    var0 = float(np.var(y))
    if var0 <= 0 or not np.isfinite(var0):
        return 1e12, np.full_like(y, np.nan)
    omega, a1, b1, nu = params
    if omega <= 0 or a1 < 0 or b1 < 0 or a1 + b1 >= 0.999 or nu <= 2.05:
        return 1e12, np.full_like(y, np.nan)
    h = np.empty_like(y)
    h[0] = var0
    eps2 = y * y
    for idx in range(1, len(y)):
        h[idx] = omega + a1 * eps2[idx - 1] + b1 * h[idx - 1]
    if np.any(h <= 0) or np.any(~np.isfinite(h)):
        return 1e12, h
    z = y / np.sqrt(h)
    nll = -float(np.sum(student_t.logpdf(z, df=nu) - 0.5 * np.log(h)))
    return nll, h


def fit_variance_architecture(r: pd.Series, max_obs: int = 756) -> pd.DataFrame:
    """
    Selects conditional variance architecture using Gaussian QMLE.
    Lower AIC/BIC is better; higher log-likelihood is better.
    """
    x = pd.Series(r).dropna().astype(float).tail(max_obs)
    if len(x) < 80:
        return pd.DataFrame()
    y = (x - x.mean()).values
    var0 = float(np.var(y))
    if var0 <= 0 or not np.isfinite(var0):
        return pd.DataFrame()

    specs = {
        "CONST": {
            "x0": np.array([var0]),
            "bounds": [(1e-12, var0 * 100)],
            "stationarity": None,
        },
        "ARCH1": {
            "x0": np.array([var0 * 0.20, 0.20]),
            "bounds": [(1e-12, var0 * 100), (1e-8, 0.999)],
            "stationarity": lambda p: 0.999 - p[1],
        },
        "ARCH2": {
            "x0": np.array([var0 * 0.20, 0.12, 0.08]),
            "bounds": [(1e-12, var0 * 100), (1e-8, 0.999), (1e-8, 0.999)],
            "stationarity": lambda p: 0.999 - p[1] - p[2],
        },
        "GARCH11": {
            "x0": np.array([var0 * 0.05, 0.05, 0.90]),
            "bounds": [(1e-12, var0 * 100), (1e-8, 0.999), (1e-8, 0.999)],
            "stationarity": lambda p: 0.999 - p[1] - p[2],
        },
        "GARCH12": {
            "x0": np.array([var0 * 0.05, 0.05, 0.70, 0.15]),
            "bounds": [(1e-12, var0 * 100), (1e-8, 0.999), (1e-8, 0.999), (1e-8, 0.999)],
            "stationarity": lambda p: 0.999 - p[1] - p[2] - p[3],
        },
        "GARCH21": {
            "x0": np.array([var0 * 0.05, 0.04, 0.02, 0.90]),
            "bounds": [(1e-12, var0 * 100), (1e-8, 0.999), (1e-8, 0.999), (1e-8, 0.999)],
            "stationarity": lambda p: 0.999 - p[1] - p[2] - p[3],
        },
        "StudentT_GARCH11": {
            "x0": np.array([var0 * 0.05, 0.05, 0.90, 8.0]),
            "bounds": [(1e-12, var0 * 100), (1e-8, 0.999), (1e-8, 0.999), (2.05, 80.0)],
            "stationarity": lambda p: 0.999 - p[1] - p[2],
        },
    }

    rows = []
    for model, spec in specs.items():
        constraints = []
        if spec["stationarity"] is not None:
            constraints.append({"type": "ineq", "fun": spec["stationarity"]})
        try:
            objective = (
                (lambda p: _student_t_garch11_nll(y, p)[0])
                if model == "StudentT_GARCH11"
                else (lambda p, model=model: _variance_recursion_nll(y, model, p)[0])
            )
            res = minimize(
                objective,
                spec["x0"],
                method="SLSQP",
                bounds=spec["bounds"],
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-8},
            )
            if not res.success or not np.all(np.isfinite(res.x)):
                rows.append({"Model": model, "Status": f"fail:{res.message}", "LogLikelihood": np.nan, "AIC": np.nan, "BIC": np.nan})
                continue
            nll, h = _student_t_garch11_nll(y, res.x) if model == "StudentT_GARCH11" else _variance_recursion_nll(y, model, res.x)
            ll = -nll
            k = len(res.x) + 1  # includes mean parameter estimated by demeaning
            n = len(y)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            next_vol = float(np.sqrt(h[-1] * 252.0))
            rows.append(
                {
                    "Model": model,
                    "Status": "ok",
                    "LogLikelihood": ll,
                    "AIC": aic,
                    "BIC": bic,
                    "N": n,
                    "Num_Params": k,
                    "Next_Ann_Vol": next_vol,
                    "Params": ",".join(f"{v:.6g}" for v in res.x),
                    "QLIKE": qlike_loss(pd.Series(y), pd.Series(h)),
                }
            )
        except Exception as exc:
            rows.append({"Model": model, "Status": f"exception:{type(exc).__name__}", "LogLikelihood": np.nan, "AIC": np.nan, "BIC": np.nan})
    # Fractional Volterra is deliberately governed as a low-degree grid search:
    # H and kernel length are fixed candidates, not freely optimized against the
    # backtest. It competes with ARCH/GARCH by likelihood diagnostics and QLIKE.
    fv_candidates: list[dict] = []
    for hurst in (0.05, 0.10, 0.20, 0.35, 0.45):
        for kernel_len in (21, 63, 126):
            if kernel_len >= len(x) - 5:
                continue
            try:
                fv = fractional_volterra_variance(x, hurst=hurst, length=kernel_len, min_periods=max(10, min(20, kernel_len // 2)))
                aligned = pd.DataFrame({"r": x, "h": fv}).dropna()
                if len(aligned) < 40:
                    continue
                h = aligned["h"].clip(lower=1e-12).to_numpy()
                yy = (aligned["r"] - aligned["r"].mean()).to_numpy()
                ll = -0.5 * float(np.sum(np.log(2 * np.pi) + np.log(h) + (yy * yy) / h))
                k = 3  # mean, H, kernel length
                n = len(yy)
                fv_candidates.append(
                    {
                        "Model": "FractionalVolterra",
                        "Status": "ok",
                        "LogLikelihood": ll,
                        "AIC": 2 * k - 2 * ll,
                        "BIC": k * np.log(n) - 2 * ll,
                        "N": n,
                        "Num_Params": k,
                        "Next_Ann_Vol": float(np.sqrt(max(fv.dropna().iloc[-1], 1e-12) * 252.0)),
                        "Params": f"H={hurst:.3g},L={kernel_len}",
                        "QLIKE": qlike_loss(aligned["r"], aligned["h"]),
                    }
                )
            except Exception:
                continue
    if fv_candidates:
        rows.append(sorted(fv_candidates, key=lambda row: (row.get("BIC", np.inf), row.get("AIC", np.inf)))[0])
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "QLIKE" not in out.columns:
        out["QLIKE"] = np.nan
    out["Best_AIC"] = out["AIC"] == out["AIC"].min(skipna=True)
    out["Best_BIC"] = out["BIC"] == out["BIC"].min(skipna=True)
    out["Best_QLIKE"] = out["QLIKE"] == out["QLIKE"].min(skipna=True) if out["QLIKE"].notna().any() else False
    return out.sort_values(["BIC", "AIC"], na_position="last").reset_index(drop=True)


def pelt_change_point_analysis(
    r: pd.Series,
    min_size: int = 21,
    penalty: float | None = None,
    max_obs: int = 756,
) -> dict[str, pd.DataFrame]:
    """
    Penalized PELT-style Gaussian segmentation for portfolio return regimes.

    The segment cost is the maximized Gaussian negative log-likelihood under a
    piecewise-constant mean/variance model:

        C(a,b) = n_{a:b} [log(sigma^2_{a:b}) + 1]

    constants are omitted because they do not affect the argmin. The penalty is
    BIC-like by default, discouraging spurious change points in low-SNR returns.
    """
    x = pd.Series(r).dropna().astype(float).tail(max_obs)
    if len(x) < max(3 * min_size, 60):
        return {"pelt_regime_segments": pd.DataFrame(), "pelt_change_points": pd.DataFrame(), "pelt_timeline": pd.DataFrame()}
    min_size = int(np.clip(min_size, 10, max(10, len(x) // 3)))
    y = x.values.astype(float)
    n = len(y)
    penalty = float(penalty) if penalty is not None else float(3.0 * np.log(n))
    csum = np.r_[0.0, np.cumsum(y)]
    csum2 = np.r_[0.0, np.cumsum(y * y)]

    def cost(a: int, b: int) -> float:
        m = b - a
        if m < min_size:
            return 1e18
        s1 = csum[b] - csum[a]
        s2 = csum2[b] - csum2[a]
        var = max((s2 - s1 * s1 / m) / m, 1e-12)
        return float(m * (np.log(var) + 1.0))

    f = np.full(n + 1, np.inf)
    f[0] = -penalty
    partitions: list[list[int]] = [[] for _ in range(n + 1)]
    candidates = [0]
    for t in range(min_size, n + 1):
        valid = [tau for tau in candidates if t - tau >= min_size]
        if not valid:
            candidates.append(t - min_size + 1)
            continue
        vals = np.array([f[tau] + cost(tau, t) + penalty for tau in valid], dtype=float)
        best_pos = int(np.nanargmin(vals))
        best_tau = int(valid[best_pos])
        f[t] = float(vals[best_pos])
        partitions[t] = partitions[best_tau] + ([best_tau] if best_tau > 0 else [])
        # PELT pruning: keep candidates that can still be optimal under an
        # additive penalty; this remains exact for compatible costs and is a
        # conservative speed-up for the Gaussian variance cost used here.
        candidates = [tau for tau in valid if f[tau] + cost(tau, t) <= f[t] + penalty]
        new_candidate = t - min_size + 1
        if new_candidate >= 0:
            candidates.append(new_candidate)

    cps = sorted(set([cp for cp in partitions[n] if 0 < cp < n]))
    bounds = [0] + cps + [n]
    equity = (1.0 + x).cumprod()
    segment_rows = []
    timeline_parts = []
    for seg_id, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        seg = x.iloc[a:b]
        seg_equity = (1.0 + seg).cumprod()
        ann_vol = float(seg.std(ddof=1) * np.sqrt(252.0)) if len(seg) > 1 else np.nan
        downside = seg[seg < 0]
        downside_vol = float(np.sqrt((downside * downside).mean()) * np.sqrt(252.0)) if len(downside) else 0.0
        segment_rows.append(
            {
                "Segment": seg_id,
                "Start_Index": int(a),
                "End_Index": int(b - 1),
                "Start_Date": seg.index.min(),
                "End_Date": seg.index.max(),
                "N": int(len(seg)),
                "Mean_Daily": float(seg.mean()),
                "Ann_Return_Proxy": float(seg.mean() * 252.0),
                "Ann_Vol": ann_vol,
                "Downside_Ann_Vol": downside_vol,
                "Skew": float(seg.skew()) if len(seg) > 2 else np.nan,
                "Excess_Kurtosis": float(seg.kurt()) if len(seg) > 3 else np.nan,
                "Segment_Return": float(seg_equity.iloc[-1] - 1.0) if len(seg_equity) else np.nan,
                "Max_Drawdown": float((seg_equity / seg_equity.cummax() - 1.0).min()) if len(seg_equity) else np.nan,
                "PELT_Penalty": penalty,
            }
        )
        tmp = pd.DataFrame(
            {
                "Date": seg.index,
                "Portfolio_Return": seg.values,
                "Segment": seg_id,
                "Segment_Ann_Vol": ann_vol,
                "Segment_Mean_Daily": float(seg.mean()),
                "Is_Change_Point": False,
            }
        )
        timeline_parts.append(tmp)
    segments = pd.DataFrame(segment_rows)
    if not segments.empty:
        med_vol = segments["Ann_Vol"].median(skipna=True)
        med_mu = segments["Mean_Daily"].median(skipna=True)
        segments["Variance_Regime"] = np.where(segments["Ann_Vol"] >= med_vol, "High variance", "Low variance")
        segments["Drift_Regime"] = np.where(segments["Mean_Daily"] >= med_mu, "High drift", "Low drift")
        segments["Regime_Label"] = segments["Variance_Regime"] + " / " + segments["Drift_Regime"]
    timeline = pd.concat(timeline_parts, ignore_index=True) if timeline_parts else pd.DataFrame()
    if not timeline.empty:
        timeline["Rolling_21D_Ann_Vol"] = (
            pd.Series(timeline["Portfolio_Return"].values)
            .rolling(21, min_periods=10)
            .std()
            .mul(np.sqrt(252.0))
            .values
        )
        timeline["Cumulative_Equity"] = equity.reindex(pd.to_datetime(timeline["Date"])).values
        change_dates = set(pd.Timestamp(x.index[cp]) for cp in cps)
        timeline["Is_Change_Point"] = pd.to_datetime(timeline["Date"]).isin(change_dates)
    event_rows = []
    for cp in cps:
        prev = segments[segments["End_Index"].eq(cp - 1)]
        nxt = segments[segments["Start_Index"].eq(cp)]
        if prev.empty or nxt.empty:
            continue
        p = prev.iloc[0]
        q = nxt.iloc[0]
        vol_ratio = safe_div(q.get("Ann_Vol"), p.get("Ann_Vol"))
        event_rows.append(
            {
                "Change_Date": x.index[cp],
                "Prev_Segment": int(p["Segment"]),
                "Next_Segment": int(q["Segment"]),
                "Prev_Ann_Vol": p.get("Ann_Vol"),
                "Next_Ann_Vol": q.get("Ann_Vol"),
                "Vol_Ratio": vol_ratio,
                "Prev_Mean_Daily": p.get("Mean_Daily"),
                "Next_Mean_Daily": q.get("Mean_Daily"),
                "Change_Type": "Variance up" if pd.notna(vol_ratio) and vol_ratio > 1.25 else "Variance down" if pd.notna(vol_ratio) and vol_ratio < 0.80 else "Mean/variance shift",
            }
        )
    changes = pd.DataFrame(event_rows)
    return {"pelt_regime_segments": segments, "pelt_change_points": changes, "pelt_timeline": timeline}


def evt_tail_metrics(r: pd.Series, threshold_q: float = 0.90) -> tuple[float, float, str]:
    losses = -pd.Series(r).dropna().astype(float)
    if len(losses) < 120:
        var95, cvar95 = historical_var_cvar(pd.Series(r), 0.95)
        return var95, cvar95, "historical_insufficient_evt"
    threshold = losses.quantile(threshold_q)
    excess = losses[losses > threshold] - threshold
    if len(excess) < 20 or excess.std() == 0:
        var95, cvar95 = historical_var_cvar(pd.Series(r), 0.95)
        return var95, cvar95, "historical_fallback"
    try:
        xi, loc, beta = genpareto.fit(excess.values, floc=0)
        p_exceed = len(excess) / len(losses)
        alpha = 0.95
        tail_prob = (1 - alpha) / max(p_exceed, 1e-9)
        if xi == 0:
            var = threshold - beta * np.log(tail_prob)
        else:
            var = threshold + beta / xi * (tail_prob ** (-xi) - 1)
        cvar = var + (beta + xi * (var - threshold)) / max(1 - xi, 1e-6) if xi < 1 else np.nan
        return float(var), float(cvar), "gpd_evt"
    except Exception:
        var95, cvar95 = historical_var_cvar(pd.Series(r), 0.95)
        return var95, cvar95, "historical_exception"


def latent_regime_state(macro_row: pd.Series) -> str:
    model_label = macro_row.get("Latent_State_Label", None)
    if isinstance(model_label, str) and model_label:
        return model_label
    hawkish = macro_row.get("Regime_Hawkish_Dovish", "Dovish")
    bullbear = macro_row.get("Regime_Bull_Bear", "Bull")
    credit = to_float(macro_row.get("CREDIT_SPREAD"))
    curve = to_float(macro_row.get("Curve_10Y_2Y"))
    inflation = to_float(macro_row.get("Inflation_YoY"))
    if pd.notna(credit) and credit > 2.5:
        return "credit_stress"
    if hawkish == "Hawkish" and pd.notna(inflation) and inflation > 0.035:
        return "inflation_shock"
    if bullbear == "Bear" and pd.notna(curve) and curve < -0.25:
        return "liquidity_or_recession"
    if bullbear == "Bull" and hawkish == "Dovish":
        return "risk_on"
    return "mixed_transition"


def regime_alpha_half_life(regime: str) -> float:
    return {
        "risk_on": 63.0,
        "inflation_shock": 42.0,
        "credit_stress": 21.0,
        "liquidity_or_recession": 21.0,
        "mixed_transition": 42.0,
    }.get(regime, 42.0)


LATENT_REGIME_FEATURES = [
    "hawkish_score",
    "bullish_score",
    "CREDIT_SPREAD",
    "Country_Curve_10Y_2Y",
    "Country_Term_Premium_Proxy",
    "VIXCLS",
    "Inflation_YoY",
    "IPMAN_YoY",
]


def _label_latent_components(component_means: pd.DataFrame) -> dict[int, str]:
    labels = {}
    for component, row in component_means.iterrows():
        credit_score = (
            row.get("CREDIT_SPREAD", 0.0)
            + 0.50 * row.get("VIXCLS", 0.0)
            - 0.60 * row.get("bullish_score", 0.0)
        )
        inflation_score = (
            row.get("hawkish_score", 0.0)
            + row.get("Inflation_YoY", 0.0)
            + 0.35 * row.get("Country_Term_Premium_Proxy", 0.0)
        )
        recession_score = (
            -row.get("bullish_score", 0.0)
            - 0.50 * row.get("Country_Curve_10Y_2Y", 0.0)
            - 0.35 * row.get("IPMAN_YoY", 0.0)
            + 0.25 * row.get("CREDIT_SPREAD", 0.0)
        )
        risk_on_score = (
            row.get("bullish_score", 0.0)
            - 0.40 * row.get("hawkish_score", 0.0)
            - 0.35 * row.get("CREDIT_SPREAD", 0.0)
            - 0.25 * row.get("VIXCLS", 0.0)
        )
        scores = {
            "risk_on": risk_on_score,
            "inflation_shock": inflation_score,
            "credit_stress": credit_score,
            "liquidity_or_recession": recession_score,
        }
        label = max(scores, key=scores.get)
        if max(scores.values()) < 0.15:
            label = "mixed_transition"
        labels[int(component)] = label
    return labels


def online_latent_regime_model(
    macro: pd.DataFrame,
    n_states: int = 4,
    min_train: int = 252,
    refit_days: int = 21,
) -> pd.DataFrame:
    if GaussianMixture is None or macro.empty:
        return pd.DataFrame(index=macro.index)
    available = [c for c in LATENT_REGIME_FEATURES if c in macro.columns]
    if len(available) < 3:
        return pd.DataFrame(index=macro.index)
    raw = macro[available].replace([np.inf, -np.inf], np.nan).ffill()
    raw = raw.dropna(how="all")
    if len(raw) < max(min_train, 80):
        return pd.DataFrame(index=macro.index)
    out = pd.DataFrame(index=macro.index)
    refit_days = max(int(refit_days), 5)
    min_train = max(int(min_train), 80)
    state_count = int(np.clip(n_states, 2, 6))
    positions = list(range(min_train, len(raw), refit_days))
    if not positions or positions[-1] != len(raw) - 1:
        positions.append(len(raw) - 1)

    for pos in positions:
        asof = raw.index[pos]
        train = raw.iloc[: pos + 1].tail(max(min_train * 3, min_train)).copy()
        train = train.dropna(axis=1, how="all")
        if len(train) < min_train or train.shape[1] < 3:
            continue
        med = train.median(numeric_only=True)
        train = train.fillna(med)
        mu = train.mean()
        sig = train.std(ddof=0).replace(0, np.nan).fillna(1.0)
        z = ((train - mu) / sig).clip(-5, 5)
        components = min(state_count, max(2, len(z) // 60))
        try:
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=1e-5,
                n_init=3,
                random_state=17,
            )
            model.fit(z.values)
            current = ((raw.loc[[asof], z.columns].fillna(med[z.columns]) - mu[z.columns]) / sig[z.columns]).clip(-5, 5)
            probs = model.predict_proba(current.values)[0]
            state_id = int(np.argmax(probs))
            means = pd.DataFrame(model.means_, columns=z.columns)
            labels = _label_latent_components(means)
            entropy = shannon_entropy(probs) / np.log(len(probs)) if len(probs) > 1 else 0.0
            row = {
                "Latent_State_ID": state_id,
                "Latent_State_Prob": float(np.max(probs)),
                "Latent_State_Entropy": float(entropy),
                "Latent_State_Label": labels.get(state_id, "mixed_transition"),
                "Latent_Regime_Source": "causal_expanding_gaussian_mixture",
            }
            for component, prob in enumerate(probs):
                row[f"Latent_Prob_Component_{component}"] = float(prob)
            for label in sorted(set(labels.values())):
                row[f"Latent_Prob_{label}"] = float(sum(probs[k] for k, v in labels.items() if v == label))
            out.loc[asof, list(row.keys())] = list(row.values())
        except Exception:
            continue
    out = out.ffill()
    return out.reindex(macro.index).ffill()


STRESS_REGIME_LABELS = {"credit_stress", "inflation_shock", "liquidity_or_recession"}


def online_markov_regime_forecast(
    macro: pd.DataFrame,
    min_obs: int = 60,
    laplace: float = 1.0,
) -> pd.DataFrame:
    if macro.empty or "Latent_State_Label" not in macro.columns:
        return pd.DataFrame(index=macro.index)
    labels = macro["Latent_State_Label"].dropna().astype(str)
    if labels.empty:
        return pd.DataFrame(index=macro.index)
    states = sorted(labels.unique().tolist())
    out = pd.DataFrame(index=macro.index)
    min_obs = max(int(min_obs), 5)
    laplace = float(max(laplace, 1e-9))
    for pos in range(len(labels)):
        date = labels.index[pos]
        history = labels.iloc[: pos + 1]
        current = history.iloc[-1]
        if len(history) < min_obs:
            counts = pd.DataFrame(laplace, index=states, columns=states)
        else:
            prev = history.shift(1).dropna()
            curr = history.loc[prev.index]
            counts = pd.crosstab(prev, curr).reindex(index=states, columns=states, fill_value=0).astype(float) + laplace
        probs = counts.div(counts.sum(axis=1), axis=0)
        row_prob = probs.loc[current] if current in probs.index else pd.Series(1.0 / len(states), index=states)
        entropy = shannon_entropy(row_prob.values) / np.log(len(row_prob)) if len(row_prob) > 1 else 0.0
        stress_prob = float(row_prob.reindex([s for s in STRESS_REGIME_LABELS if s in row_prob.index]).fillna(0.0).sum())
        risk_on_prob = float(row_prob.get("risk_on", 0.0))
        mode_state = str(row_prob.idxmax())
        out.loc[date, "Markov_Current_State"] = current
        out.loc[date, "Markov_Next_State_Mode"] = mode_state
        out.loc[date, "Markov_State_Persistence"] = float(row_prob.get(current, np.nan))
        out.loc[date, "Markov_Stress_Prob"] = stress_prob
        out.loc[date, "Markov_Risk_On_Prob"] = risk_on_prob
        out.loc[date, "Markov_Transition_Entropy"] = float(entropy)
        out.loc[date, "Markov_Transition_Obs"] = float(max(len(history) - 1, 0))
        for state, prob in row_prob.items():
            out.loc[date, f"Markov_Prob_Next_{state}"] = float(prob)
    return out.reindex(macro.index).ffill()


def latent_regime_diagnostics(macro: pd.DataFrame) -> dict[str, pd.DataFrame]:
    empty = {"timeline": pd.DataFrame(), "transition_matrix": pd.DataFrame(), "markov_forecast": pd.DataFrame(), "summary": pd.DataFrame()}
    if macro.empty or "Latent_State_Label" not in macro.columns:
        return empty
    cols = [
        "Latent_State_Label",
        "Latent_State_Prob",
        "Latent_State_Entropy",
        "Regime_Hawkish_Dovish",
        "Regime_Bull_Bear",
        "hawkish_score",
        "bullish_score",
        "Country_Curve_10Y_2Y",
        "CREDIT_SPREAD",
        "Markov_Next_State_Mode",
        "Markov_State_Persistence",
        "Markov_Stress_Prob",
        "Markov_Risk_On_Prob",
        "Markov_Transition_Entropy",
        "Markov_Transition_Obs",
    ]
    timeline = macro[[c for c in cols if c in macro.columns]].dropna(subset=["Latent_State_Label"]).copy()
    if timeline.empty:
        return empty
    timeline = timeline.reset_index().rename(columns={timeline.index.name or "index": "Date"})
    seq = timeline["Latent_State_Label"].astype(str)
    transitions = pd.crosstab(seq.shift(1), seq, normalize="index").fillna(0.0)
    transitions.index.name = "From_State"
    summary = timeline.groupby("Latent_State_Label").agg(
        Observations=("Latent_State_Label", "size"),
        Mean_Prob=("Latent_State_Prob", "mean"),
        Mean_Entropy=("Latent_State_Entropy", "mean"),
        Mean_Hawkish=("hawkish_score", "mean"),
        Mean_Bullish=("bullish_score", "mean"),
        Mean_Curve=("Country_Curve_10Y_2Y", "mean"),
        Mean_Credit=("CREDIT_SPREAD", "mean"),
    ).reset_index()
    summary["Frequency"] = summary["Observations"] / max(summary["Observations"].sum(), 1)
    markov_cols = [c for c in timeline.columns if c.startswith("Markov_") or c == "Date"]
    markov_forecast = timeline[markov_cols].copy() if markov_cols else pd.DataFrame()
    return {"timeline": timeline, "transition_matrix": transitions.reset_index(), "markov_forecast": markov_forecast, "summary": summary}


def empirical_bayes_alpha_posterior(
    raw_score: pd.Series,
    crlb_mu: pd.Series,
    rank_entropy: float = 0.0,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    raw = pd.Series(raw_score).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    crlb = pd.Series(crlb_mu).reindex(raw.index).astype(float).replace([np.inf, -np.inf], np.nan)
    groups = pd.Series(groups).reindex(raw.index).fillna("GLOBAL") if groups is not None else pd.Series("GLOBAL", index=raw.index)
    global_mean = float(raw.mean())
    global_var = float(raw.var(ddof=0))
    prior_var = global_var
    if not np.isfinite(prior_var) or prior_var <= 1e-10:
        prior_var = 1.0
    global_std = float(np.sqrt(prior_var))
    group_stats = raw.groupby(groups).agg(["mean", "var", "count"])
    group_mean = groups.map(group_stats["mean"]).reindex(raw.index).astype(float)
    group_var = groups.map(group_stats["var"]).reindex(raw.index).astype(float)
    group_count = groups.map(group_stats["count"]).reindex(raw.index).astype(float)
    sector_reliability = (group_count / (group_count + 5.0)).clip(0.0, 1.0)
    prior_mean = sector_reliability * group_mean.fillna(global_mean) + (1.0 - sector_reliability) * global_mean
    hierarchical_prior_var = (
        sector_reliability * group_var.fillna(prior_var).clip(lower=prior_var * 0.10)
        + (1.0 - sector_reliability) * prior_var
    ).clip(lower=1e-10)
    crlb_scale = crlb[crlb > 0].median()
    crlb_scale = float(crlb_scale) if pd.notna(crlb_scale) and crlb_scale > 0 else 1.0
    obs_var = prior_var * (crlb.fillna(crlb_scale) / crlb_scale).clip(0.05, 25.0)
    obs_var = obs_var * (1.0 + float(np.clip(rank_entropy, 0.0, 1.0)))
    obs_var = obs_var.clip(lower=1e-10)
    post_var = 1.0 / (1.0 / hierarchical_prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / hierarchical_prior_var + raw / obs_var)
    post_std = np.sqrt(post_var.clip(lower=1e-12))
    tstat = post_mean / post_std.replace(0, np.nan)
    prob_pos = pd.Series(norm.cdf(tstat.fillna(0.0)), index=raw.index)
    credible_low = post_mean - 1.96 * post_std
    credible_high = post_mean + 1.96 * post_std
    posterior_confidence = (2.0 * prob_pos - 1.0).abs()
    decision_score = post_mean * posterior_confidence
    return pd.DataFrame(
        {
            "Bayesian_Alpha_Mean": post_mean,
            "Bayesian_Alpha_Std": post_std,
            "Bayesian_Alpha_TStat": tstat,
            "Prob_Alpha_Positive": prob_pos,
            "Alpha_CI_95_Low": credible_low,
            "Alpha_CI_95_High": credible_high,
            "Alpha_CI_95_Width": credible_high - credible_low,
            "Bayesian_Decision_Score": decision_score,
            "Bayesian_Posterior_Confidence": posterior_confidence,
            "Bayesian_Prior_Var": prior_var,
            "Hierarchical_Alpha_Prior_Mean": prior_mean,
            "Hierarchical_Alpha_Prior_Std": np.sqrt(hierarchical_prior_var),
            "Hierarchical_Sector_Reliability": sector_reliability,
            "Hierarchical_Global_Mean": global_mean,
            "Hierarchical_Global_Std": global_std,
            "Hierarchical_Group_Count": group_count,
            "Bayesian_Obs_Var": obs_var,
        }
    )


def add_post_phd_alpha_layer(
    cs: pd.DataFrame,
    prices: pd.DataFrame,
    macro_row: pd.Series,
    asof_date,
    lookback: int = 252,
    use_garch: bool = True,
    garch_candidate_n: int | None = 30,
    crlb_penalty: float = 0.15,
    garch_penalty: float = 0.10,
    evt_penalty: float = 0.10,
) -> pd.DataFrame:
    if cs.empty:
        return cs
    out = cs.copy()
    tickers = [t for t in out["Ticker"] if t in prices.columns]
    ret = prices.loc[:asof_date, tickers].tail(lookback + 1).pct_change(fill_method=None).dropna(how="all")
    regime = latent_regime_state(macro_row)
    half_life = regime_alpha_half_life(regime)
    decay = float(np.exp(-21.0 / half_life))

    raw = out.set_index("Ticker")["Composite_Score"].astype(float)
    garch_names = set(raw.sort_values(ascending=False).head(garch_candidate_n).index) if garch_candidate_n else set(raw.index)
    p = softmax(raw, temperature=max(raw.std(ddof=0), 0.25))
    rank_entropy = shannon_entropy(p.values) / np.log(len(p)) if len(p) > 1 else 0.0

    rows = {}
    for tk in out["Ticker"]:
        r = ret[tk].dropna() if tk in ret.columns else pd.Series(dtype=float)
        sigma = float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 2 else np.nan
        t_obs = max(len(r), 1)
        crlb_mu = (sigma * sigma) / t_obs if pd.notna(sigma) else np.nan
        crlb_sigma2 = 2.0 * (sigma ** 4) / t_obs if pd.notna(sigma) else np.nan
        gvol, gstat = garch11_forecast(r) if use_garch and tk in garch_names else (ewma_vol_forecast(r), "ewma_fast_or_not_preselected")
        evt_var, evt_cvar, evt_status = evt_tail_metrics(r)
        sk = skew(r, bias=False) if len(r) > 20 else np.nan
        ku = kurtosis(r, fisher=True, bias=False) if len(r) > 20 else np.nan
        rows[tk] = {
            "CRLB_Mu": crlb_mu,
            "CRLB_Sigma2": crlb_sigma2,
            "GARCH_Vol_Forecast": gvol,
            "GARCH_Status": gstat,
            "EVT_VaR_95": evt_var,
            "EVT_CVaR_95": evt_cvar,
            "Return_Skew": sk,
            "Return_Excess_Kurtosis": ku,
        }

    diag = pd.DataFrame.from_dict(rows, orient="index")
    out = out.merge(diag, left_on="Ticker", right_index=True, how="left")
    out["Composite_Score_Raw"] = out["Composite_Score"]
    score_scale = out["Composite_Score_Raw"].std(ddof=0)
    if pd.isna(score_scale) or score_scale <= 0:
        score_scale = 1.0
    crlb_scale = out["CRLB_Mu"].replace([np.inf, -np.inf], np.nan).median()
    garch_scale = out["GARCH_Vol_Forecast"].replace([np.inf, -np.inf], np.nan).median()
    evt_scale = out["EVT_CVaR_95"].replace([np.inf, -np.inf], np.nan).median()
    crlb_scale = crlb_scale if pd.notna(crlb_scale) and crlb_scale > 0 else 1.0
    garch_scale = garch_scale if pd.notna(garch_scale) and garch_scale > 0 else 1.0
    evt_scale = evt_scale if pd.notna(evt_scale) and evt_scale > 0 else 1.0

    uncertainty = (
        crlb_penalty * np.sqrt(out["CRLB_Mu"].clip(lower=0).fillna(crlb_scale) / crlb_scale)
        + garch_penalty * (out["GARCH_Vol_Forecast"].fillna(garch_scale) / garch_scale)
        + evt_penalty * (out["EVT_CVaR_95"].fillna(evt_scale) / evt_scale)
    )
    posterior = empirical_bayes_alpha_posterior(
        out.set_index("Ticker")["Composite_Score_Raw"],
        out.set_index("Ticker")["CRLB_Mu"],
        rank_entropy=rank_entropy,
        groups=out.set_index("Ticker")["Sector"] if "Sector" in out.columns else None,
    )
    out = out.merge(posterior, left_on="Ticker", right_index=True, how="left")
    shrink = out["Bayesian_Alpha_Mean"].abs() / (out["Composite_Score_Raw"].abs() + 1e-12)
    out["Bayesian_Alpha_Shrink"] = shrink.clip(0.0, 1.0)
    out["Latent_Regime_State"] = regime
    out["Regime_Alpha_Decay_21D"] = decay
    out["Ranking_Shannon_Entropy"] = rank_entropy
    out["Uncertainty_Penalty"] = uncertainty
    out["Composite_Score"] = out["Bayesian_Decision_Score"].fillna(out["Composite_Score_Raw"]) * decay - uncertainty
    return out


COUNTRY_RATE_SERIES = {
    "United States": {"policy": "FEDFUNDS", "long": "DGS10", "short": "DGS2"},
    "Mexico": {"policy": "IRSTCI01MXM156N", "long": "IRLTLT01MXM156N", "short": "IR3TIB01MXM156N"},
    "Canada": {"policy": "IRSTCI01CAM156N", "long": "IRLTLT01CAM156N", "short": "IR3TIB01CAM156N"},
    "United Kingdom": {"policy": "IRSTCI01GBM156N", "long": "IRLTLT01GBM156N", "short": "IR3TIB01GBM156N"},
    "Germany": {"policy": "IRSTCI01DEM156N", "long": "IRLTLT01DEM156N", "short": "IR3TIB01DEM156N"},
    "France": {"policy": "IRSTCI01FRM156N", "long": "IRLTLT01FRM156N", "short": "IR3TIB01FRM156N"},
    "Spain": {"policy": "IRSTCI01ESM156N", "long": "IRLTLT01ESM156N", "short": "IR3TIB01ESM156N"},
    "Italy": {"policy": "IRSTCI01ITM156N", "long": "IRLTLT01ITM156N", "short": "IR3TIB01ITM156N"},
    "Netherlands": {"policy": "IRSTCI01NLM156N", "long": "IRLTLT01NLM156N", "short": "IR3TIB01NLM156N"},
    "Switzerland": {"policy": "IRSTCI01CHM156N", "long": "IRLTLT01CHM156N", "short": "IR3TIB01CHM156N"},
    "Sweden": {"policy": "IRSTCI01SEM156N", "long": "IRLTLT01SEM156N", "short": "IR3TIB01SEM156N"},
    "Norway": {"policy": "IRSTCI01NOM156N", "long": "IRLTLT01NOM156N", "short": "IR3TIB01NOM156N"},
    "Australia": {"policy": "IRSTCI01AUM156N", "long": "IRLTLT01AUM156N", "short": "IR3TIB01AUM156N"},
    "New Zealand": {"policy": "IRSTCI01NZM156N", "long": "IRLTLT01NZM156N", "short": "IR3TIB01NZM156N"},
    "Japan": {"policy": "IRSTCI01JPM156N", "long": "IRLTLT01JPM156N", "short": "IR3TIB01JPM156N"},
    "Brazil": {"policy": "IRSTCI01BRM156N", "long": "IRLTLT01BRM156N", "short": "IR3TIB01BRM156N"},
    "China": {"policy": "IRSTCI01CNM156N", "long": "IRLTLT01CNM156N", "short": "IR3TIB01CNM156N"},
    "India": {"policy": "IRSTCI01INM156N", "long": "IRLTLT01INM156N", "short": "IR3TIB01INM156N"},
    "South Korea": {"policy": "IRSTCI01KRM156N", "long": "IRLTLT01KRM156N", "short": "IR3TIB01KRM156N"},
    "South Africa": {"policy": "IRSTCI01ZAM156N", "long": "IRLTLT01ZAM156N", "short": "IR3TIB01ZAM156N"},
}

GLOBAL_RATE_COUNTRIES = (
    "United States",
    "Mexico",
    "Canada",
    "Brazil",
    "China",
    "India",
    "Japan",
    "South Korea",
    "Australia",
    "New Zealand",
    "United Kingdom",
    "Germany",
    "France",
    "Spain",
    "Italy",
    "Netherlands",
    "Switzerland",
    "Sweden",
    "Norway",
    "South Africa",
)

RATE_TENOR_LABELS = {
    "POLICY_RATE": "Policy / short policy rate",
    "SOV_2Y": "2Y sovereign or money-market proxy",
    "SOV_10Y": "10Y sovereign",
}

INTERBANK_REFERENCE_SERIES = {
    "SOFR": {
        "Code": "SOFR",
        "Tenor": "Overnight",
        "Benchmark": "SOFR",
        "Jurisdiction": "United States",
        "Currency": "USD",
        "Status": "Active overnight risk-free/reference rate",
        "Source": "FRED / Federal Reserve Bank of New York",
    },
    "IUDSOIA": {
        "Code": "IUDSOIA",
        "Tenor": "Overnight",
        "Benchmark": "SONIA",
        "Jurisdiction": "United Kingdom",
        "Currency": "GBP",
        "Status": "Active overnight risk-free/reference rate",
        "Source": "FRED / Bank of England",
    },
    "ECBESTRVOLWGTTRMDMNRT": {
        "Code": "ECBESTRVOLWGTTRMDMNRT",
        "Tenor": "Overnight",
        "Benchmark": "ESTR",
        "Jurisdiction": "Eurozone",
        "Currency": "EUR",
        "Status": "Active overnight risk-free/reference rate",
        "Source": "FRED / European Central Bank",
    },
    "IRSTCI01JPM156N": {
        "Code": "IRSTCI01JPM156N",
        "Tenor": "Overnight",
        "Benchmark": "TONAR",
        "Jurisdiction": "Japan",
        "Currency": "JPY",
        "Status": "Active overnight call-rate proxy",
        "Source": "FRED / OECD immediate call money rate proxy",
    },
}


def fetch_macro_frame(start, end, country: str = "United States", use_cache: bool = True, cache_ttl_hours: int = 24) -> pd.DataFrame:
    country_rates = COUNTRY_RATE_SERIES.get(country, COUNTRY_RATE_SERIES["United States"])
    series = {
        "CPIAUCSL": "CPI",
        country_rates.get("policy", "FEDFUNDS"): "POLICY_RATE",
        country_rates.get("long", "DGS10"): "SOV_10Y",
        country_rates.get("short", "DGS2"): "SOV_2Y",
        "FEDFUNDS": "FEDFUNDS",
        "DGS10": "US10Y",
        "DGS2": "US2Y",
        "BAA10Y": "CREDIT_SPREAD",
        "DTWEXBGS": "USD_BROAD",
        "DCOILWTICO": "WTI",
        "USEPUINDXD": "EPU",
        "IPMAN": "IPMAN",
        "WALCL": "FED_BALANCE_SHEET",
        "RRPONTSYD": "FED_REVERSE_REPO",
        "NFCI": "NFCI",
        "VIXCLS": "VIX",
        "BAMLH0A0HYM2": "HY_OAS",
    }
    payload = {"start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date()), "country": country, "series": series}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("macro_fred", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    macro = pd.DataFrame()
    for code, name in series.items():
        try:
            s = pdr.DataReader(code, "fred", start, end).rename(columns={code: name})
            macro = pd.concat([macro, s], axis=1)
        except Exception:
            pass
    macro = macro.loc[:, ~macro.columns.duplicated()].copy()
    if use_cache and not macro.empty:
        PERSISTENT_CACHE.set_df("macro_fred", payload, macro)
    return macro


def fetch_banxico_rates(start, end) -> pd.DataFrame:
    token = os.getenv("BANXICO_TOKEN", "").strip()
    if not token:
        return pd.DataFrame()
    series = {"SF61745": "POLICY_RATE"}
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    frames = []
    for code, name in series.items():
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{code}/datos/{start_s}/{end_s}?token={urllib.parse.quote(token)}"
        try:
            data = http_read_json(url, user_agent="QuantStockPicker/1.0", timeout=30)
            datos = data.get("bmx", {}).get("series", [{}])[0].get("datos", [])
            rows = []
            for item in datos:
                val = str(item.get("dato", "")).replace(",", "")
                rows.append({"Date": pd.to_datetime(item.get("fecha"), dayfirst=True, errors="coerce"), name: to_float(val)})
            df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date")
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_bcb_sgs_rates(start, end) -> pd.DataFrame:
    series = {432: "POLICY_RATE"}
    frames = []
    start_s = pd.Timestamp(start).strftime("%d/%m/%Y")
    end_s = pd.Timestamp(end).strftime("%d/%m/%Y")
    for code, name in series.items():
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={urllib.parse.quote(start_s)}&dataFinal={urllib.parse.quote(end_s)}"
        try:
            data = http_read_json(url, timeout=30)
            df = pd.DataFrame(data)
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            df[name] = pd.to_numeric(df["valor"].str.replace(",", ".", regex=False), errors="coerce")
            frames.append(df[["Date", name]].dropna(subset=["Date"]).set_index("Date"))
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_ecb_yield_curve(start, end) -> pd.DataFrame:
    codes = {
        "SR_3M": "POLICY_RATE",
        "SR_2Y": "SOV_2Y",
        "SR_10Y": "SOV_10Y",
    }
    frames = []
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    for maturity, name in codes.items():
        url = (
            "https://data-api.ecb.europa.eu/service/data/YC/"
            f"B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity}?startPeriod={start_s}&endPeriod={end_s}&format=csvdata"
        )
        try:
            df = pd.read_csv(url)
            if df.empty or "TIME_PERIOD" not in df or "OBS_VALUE" not in df:
                continue
            df["Date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
            df[name] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
            frames.append(df[["Date", name]].dropna(subset=["Date"]).set_index("Date"))
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_bank_of_canada_rates(start, end) -> pd.DataFrame:
    codes = {
        "V39079": "POLICY_RATE",
        "BD.CDN.2YR.DQ.YLD": "SOV_2Y",
        "BD.CDN.10YR.DQ.YLD": "SOV_10Y",
    }
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    frames = []
    for code, name in codes.items():
        url = f"https://www.bankofcanada.ca/valet/observations/{urllib.parse.quote(code)}/json?start_date={start_s}&end_date={end_s}"
        try:
            data = http_read_json(url, timeout=30)
            rows = []
            for obs in data.get("observations", []):
                rows.append({"Date": pd.to_datetime(obs.get("d"), errors="coerce"), name: to_float(obs.get(code, {}).get("v"))})
            df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date")
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_direct_country_rates(
    start,
    end,
    country: str,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    payload = {"country": country, "start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date())}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("macro_country_direct", payload, cache_ttl_hours)
        if cached is not None:
            return cached
    if country == "Mexico":
        df = fetch_banxico_rates(start, end)
        source = "Banxico SIE" if not df.empty else "Banxico SIE unavailable; FRED/OECD proxy"
    elif country == "Brazil":
        df = fetch_bcb_sgs_rates(start, end)
        source = "BCB SGS" if not df.empty else "BCB SGS unavailable; FRED/OECD proxy"
    elif country == "Canada":
        df = fetch_bank_of_canada_rates(start, end)
        source = "Bank of Canada Valet" if not df.empty else "Bank of Canada unavailable; FRED/OECD proxy"
    elif country in {"Germany", "France", "Spain", "Eurozone"}:
        df = fetch_ecb_yield_curve(start, end)
        source = "ECB Data Portal YC" if not df.empty else "ECB unavailable; FRED/OECD proxy"
    else:
        df = pd.DataFrame()
        source = "FRED/OECD proxy"
    if not df.empty:
        df["Country_Rate_Source"] = source
        if use_cache:
            PERSISTENT_CACHE.set_df("macro_country_direct", payload, df)
    return df


def market_regime(
    prices: pd.DataFrame,
    country: str = "United States",
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
    use_latent_macro_regime: bool = True,
    latent_regime_states: int = 4,
    latent_regime_min_train: int = 252,
    latent_regime_refit_days: int = 21,
    markov_transition_min_obs: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    start = prices.index.min() - pd.Timedelta(days=500)
    end = prices.index.max() + pd.Timedelta(days=5)
    macro = fetch_macro_frame(start, end, country=country, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    direct_rates = fetch_direct_country_rates(start, end, country, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    if not direct_rates.empty:
        macro = macro.join(direct_rates, how="outer", rsuffix="_direct")
        for col in ["POLICY_RATE", "SOV_10Y", "SOV_2Y"]:
            direct_col = f"{col}_direct"
            if direct_col in macro.columns:
                macro[col] = macro[direct_col].combine_first(macro[col] if col in macro.columns else pd.Series(index=macro.index, dtype=float))
                macro = macro.drop(columns=[direct_col])
        if "Country_Rate_Source_direct" in macro.columns:
            macro["Country_Rate_Source"] = macro["Country_Rate_Source_direct"].combine_first(macro.get("Country_Rate_Source", pd.Series(index=macro.index, dtype=object)))
            macro = macro.drop(columns=["Country_Rate_Source_direct"])
    if macro.empty:
        macro = pd.DataFrame(index=prices.index)
    macro = macro.resample("B").ffill()
    monthly = macro.resample("ME").last()
    if "CPI" in monthly:
        monthly["Inflation_YoY"] = monthly["CPI"].pct_change(12, fill_method=None)
        macro["Inflation_YoY"] = monthly["Inflation_YoY"].reindex(macro.index, method="ffill")
    if "IPMAN" in monthly:
        monthly["IPMAN_YoY"] = monthly["IPMAN"].pct_change(12, fill_method=None)
        macro["IPMAN_YoY"] = monthly["IPMAN_YoY"].reindex(macro.index, method="ffill")

    for col in ["POLICY_RATE", "SOV_10Y", "SOV_2Y", "FEDFUNDS", "US10Y", "US2Y", "CREDIT_SPREAD", "USD_BROAD", "WTI", "EPU", "Inflation_YoY", "IPMAN_YoY"]:
        if col not in macro:
            macro[col] = np.nan
    macro["POLICY_RATE"] = macro["POLICY_RATE"].fillna(macro["FEDFUNDS"])
    macro["SOV_10Y"] = macro["SOV_10Y"].fillna(macro["US10Y"])
    macro["SOV_2Y"] = macro["SOV_2Y"].fillna(macro["US2Y"])
    macro["Curve_10Y_2Y"] = macro["US10Y"] - macro["US2Y"]
    macro["Country_Curve_10Y_2Y"] = macro["SOV_10Y"] - macro["SOV_2Y"]
    macro["Country_Term_Premium_Proxy"] = macro["SOV_10Y"] - macro["POLICY_RATE"]
    macro["Term_Premium_Proxy"] = macro["US10Y"] - macro["FEDFUNDS"]
    macro["Real_10Y_Proxy"] = macro["US10Y"] - 100 * macro["Inflation_YoY"]
    macro["Country_Real_10Y_Proxy"] = macro["SOV_10Y"] - 100 * macro["Inflation_YoY"]
    macro["Credit_20d_Change"] = macro["CREDIT_SPREAD"].diff(20)
    macro["Rate_Country"] = country
    if "Country_Rate_Source" not in macro:
        macro["Country_Rate_Source"] = "FRED/OECD proxy"
    macro["Country_Rate_Source"] = macro["Country_Rate_Source"].ffill().fillna("FRED/OECD proxy")

    def rz(x):
        return (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std().replace(0, np.nan)

    macro["hawkish_score"] = (
        0.30 * rz(macro["POLICY_RATE"])
        + 0.20 * rz(macro["Inflation_YoY"])
        + 0.15 * rz(macro["Country_Real_10Y_Proxy"])
        + 0.20 * rz(macro["CREDIT_SPREAD"])
        - 0.15 * rz(macro["IPMAN_YoY"])
    )

    spy = prices["SPY"] if "SPY" in prices.columns else prices.mean(axis=1)
    macro = macro.join(pd.DataFrame({"SPY": spy}), how="left").ffill()
    macro["SPY_MA_200"] = macro["SPY"].rolling(200).mean()
    macro["SPY_MOM_126"] = macro["SPY"].pct_change(126)
    macro["bullish_score"] = (
        (macro["SPY"] > macro["SPY_MA_200"]).astype(float)
        + (macro["SPY_MOM_126"] > 0).astype(float)
        - (macro["Country_Curve_10Y_2Y"] < -0.50).astype(float) * 0.25
        - (macro["Credit_20d_Change"] > 0.25).astype(float) * 0.25
    )
    macro["Regime_Hawkish_Dovish"] = np.where(macro["hawkish_score"] > 0, "Hawkish", "Dovish")
    macro["Regime_Bull_Bear"] = np.where(macro["bullish_score"] >= 1.25, "Bull", "Bear")
    valid_regime = macro.dropna(subset=["hawkish_score", "bullish_score"])
    if valid_regime.empty:
        macro["hawkish_score"] = macro["hawkish_score"].fillna(0.0)
        macro["bullish_score"] = macro["bullish_score"].fillna(1.0)
        macro["Regime_Hawkish_Dovish"] = np.where(macro["hawkish_score"] > 0, "Hawkish", "Dovish")
        macro["Regime_Bull_Bear"] = np.where(macro["bullish_score"] >= 1.25, "Bull", "Bear")
    if use_latent_macro_regime:
        latent = online_latent_regime_model(
            macro,
            n_states=latent_regime_states,
            min_train=latent_regime_min_train,
            refit_days=latent_regime_refit_days,
        )
        if not latent.empty:
            macro = macro.join(latent, how="left")
            markov = online_markov_regime_forecast(macro, min_obs=markov_transition_min_obs)
            if not markov.empty:
                macro = macro.join(markov, how="left")
    valid_regime = macro.dropna(subset=["hawkish_score", "bullish_score"])
    latest = valid_regime.iloc[-1] if not valid_regime.empty else macro.iloc[-1]
    return macro, latest


def global_yield_curve_snapshot(
    prices: pd.DataFrame,
    countries: Iterable[str] | None = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    """Latest cross-country sovereign curve snapshot using zero-cost public sources."""
    if prices.empty:
        return pd.DataFrame()
    countries = tuple(countries or GLOBAL_RATE_COUNTRIES)
    def one_country(country: str) -> dict:
        try:
            discrete = fetch_discrete_country_rate_frame(
                prices.index.min() - pd.Timedelta(days=500),
                prices.index.max() + pd.Timedelta(days=5),
                country,
                use_cache=use_cache,
                cache_ttl_hours=cache_ttl_hours,
            )
            macro, latest = market_regime(
                prices,
                country=country,
                use_cache=use_cache,
                cache_ttl_hours=cache_ttl_hours,
                use_latent_macro_regime=False,
            )
            def latest_discrete_rate(col: str) -> tuple[float, pd.Timestamp, str, int]:
                if discrete is not None and not discrete.empty and col in discrete.columns:
                    s = pd.to_numeric(discrete[col], errors="coerce").dropna()
                    if not s.empty:
                        return (
                            to_float(s.iloc[-1]),
                            pd.Timestamp(s.index[-1]),
                            infer_rate_frequency(s.index),
                            int(len(s)),
                        )
                return np.nan, pd.NaT, "Unavailable", 0

            policy_rate, policy_dt, policy_freq, policy_n = latest_discrete_rate("POLICY_RATE")
            short_rate, short_dt, short_freq, short_n = latest_discrete_rate("SOV_2Y")
            teny_rate, teny_dt, teny_freq, teny_n = latest_discrete_rate("SOV_10Y")
            source = "public proxy"
            if discrete is not None and not discrete.empty and "Country_Rate_Source" in discrete.columns:
                src = discrete["Country_Rate_Source"].dropna()
                if not src.empty:
                    source = str(src.iloc[-1])
            if not source or source == "public proxy":
                source = str(latest.get("Country_Rate_Source", "public proxy"))

            regime_policy = to_float(latest.get("POLICY_RATE"))
            if pd.isna(policy_rate) and pd.notna(regime_policy):
                policy_rate = regime_policy
            if pd.isna(short_rate) and country == "United States":
                short_rate = to_float(latest.get("SOV_2Y"))
            if pd.isna(teny_rate) and country == "United States":
                teny_rate = to_float(latest.get("SOV_10Y"))

            curve_10y_2y = teny_rate - short_rate if pd.notna(teny_rate) and pd.notna(short_rate) else np.nan
            term_premium = teny_rate - policy_rate if pd.notna(teny_rate) and pd.notna(policy_rate) else np.nan
            row = {
                "Country": country,
                "Policy_Rate": policy_rate,
                "Yield_2Y": short_rate,
                "Yield_Short": short_rate,
                "Short_Rate_Tenor": "2Y sovereign" if country == "United States" else "Money-market / 3M proxy where 2Y is unavailable",
                "Yield_10Y": teny_rate,
                "Curve_10Y_2Y": curve_10y_2y,
                "Term_Premium_Proxy": term_premium,
                "Regime_Hawkish_Dovish": latest.get("Regime_Hawkish_Dovish", "n/a"),
                "Regime_Bull_Bear": latest.get("Regime_Bull_Bear", "n/a"),
                "Rate_Source": source,
                "Latest_Date": macro.index.max() if not macro.empty else pd.NaT,
                "Policy_Observation_Date": policy_dt,
                "Policy_Observation_Frequency": policy_freq,
                "Policy_Observation_Count": policy_n,
                "Short_Observation_Date": short_dt,
                "Short_Observation_Frequency": short_freq,
                "Short_Observation_Count": short_n,
                "TenY_Observation_Date": teny_dt,
                "TenY_Observation_Frequency": teny_freq,
                "TenY_Observation_Count": teny_n,
            }
            if pd.notna(row["Curve_10Y_2Y"]):
                row["Curve_Shape"] = "Inverted" if row["Curve_10Y_2Y"] < 0 else "Steep" if row["Curve_10Y_2Y"] > 1.0 else "Flat/Normal"
            else:
                row["Curve_Shape"] = "Unknown"
            return row
        except Exception as exc:
            return {
                "Country": country,
                "Policy_Rate": np.nan,
                "Yield_2Y": np.nan,
                "Yield_10Y": np.nan,
                "Curve_10Y_2Y": np.nan,
                "Term_Premium_Proxy": np.nan,
                "Regime_Hawkish_Dovish": "n/a",
                "Regime_Bull_Bear": "n/a",
                "Rate_Source": f"unavailable: {exc}",
                "Latest_Date": pd.NaT,
                "Curve_Shape": "Unavailable",
            }

    rows = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(countries)))) as ex:
        futures = {ex.submit(one_country, country): country for country in countries}
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows)


def fetch_gdelt_timeline(
    query: str,
    days: int = 90,
    use_cache: bool = True,
    cache_ttl_hours: int = 6,
    timeout: int = 8,
) -> pd.DataFrame:
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    payload = {"v": 2, "query": query, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "mode": "timelinevolraw"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("gdelt_timeline", payload, cache_ttl_hours)
        if cached is not None:
            return cached
    params = urllib.parse.urlencode(
        {
            "query": query,
            "mode": "timelinevolraw",
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            "maxrecords": 250,
        }
    )
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
    try:
        data = http_read_json(url, timeout=timeout)
        timeline = data.get("timeline", [])
        rows = []
        for item in timeline:
            if isinstance(item, dict) and "data" in item:
                for sub in item.get("data", []) or []:
                    rows.append(
                        {
                            "Date": pd.to_datetime(sub.get("date"), errors="coerce"),
                            "GDELT_Volume": to_float(sub.get("value")),
                            "GDELT_Norm": to_float(sub.get("norm")),
                            "Source": "GDELT DOC 2.1",
                            "Query": query,
                        }
                    )
            else:
                rows.append(
                    {
                        "Date": pd.to_datetime(item.get("date"), errors="coerce") if isinstance(item, dict) else pd.NaT,
                        "GDELT_Volume": to_float(item.get("value")) if isinstance(item, dict) else np.nan,
                        "GDELT_Norm": to_float(item.get("norm")) if isinstance(item, dict) else np.nan,
                        "Source": "GDELT DOC 2.1",
                        "Query": query,
                    }
                )
        df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    except Exception:
        df = pd.DataFrame()
    if use_cache and not df.empty:
        PERSISTENT_CACHE.set_df("gdelt_timeline", payload, df)
    return df


def infer_rate_frequency(index: pd.Index) -> str:
    dates = pd.to_datetime(pd.Index(index), errors="coerce").dropna().sort_values()
    if len(dates) < 3:
        return "Discrete/unknown"
    deltas = pd.Series(dates).diff().dt.days.dropna()
    if deltas.empty:
        return "Discrete/unknown"
    med = float(deltas.median())
    if med <= 4:
        return "Daily/business-day discrete"
    if med <= 10:
        return "Weekly discrete"
    if med <= 45:
        return "Monthly discrete"
    if med <= 120:
        return "Quarterly discrete"
    return "Low-frequency discrete"


def fetch_interbank_reference_rates(
    start,
    end,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    """Fetch zero-cost active overnight reference-rate history: SOFR, SONIA, ESTR and TONAR proxy."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    payload = {
        "version": 3,
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "series": tuple(INTERBANK_REFERENCE_SERIES.keys()),
    }
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("interbank_reference_rates", payload, cache_ttl_hours)
        if cached is not None:
            return cached
    rows = []
    for code, meta in INTERBANK_REFERENCE_SERIES.items():
        try:
            df = pdr.DataReader(code, "fred", start_ts, end_ts)
        except Exception:
            continue
        if df is None or df.empty or code not in df.columns:
            continue
        s = pd.to_numeric(df[code], errors="coerce").dropna()
        if s.empty:
            continue
        frequency = infer_rate_frequency(s.index)
        for dt, value in s.tail(900).items():
            rows.append(
                {
                    "Observation_Date": pd.Timestamp(dt),
                    "Code": code,
                    "Benchmark": meta["Benchmark"],
                    "Jurisdiction": meta.get("Jurisdiction"),
                    "Currency": meta.get("Currency"),
                    "Tenor": meta["Tenor"],
                    "Rate": to_float(value),
                    "Observation_Frequency": frequency,
                    "Status": meta["Status"],
                    "Source": meta["Source"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Benchmark", "Observation_Date"]).reset_index(drop=True)
    sofr = (
        out.loc[out["Code"].eq("SOFR"), ["Observation_Date", "Rate"]]
        .dropna()
        .sort_values("Observation_Date")
        .rename(columns={"Rate": "SOFR_Aligned"})
    )
    latest = (
        out.sort_values("Observation_Date")
        .groupby("Code", as_index=False)
        .tail(1)[["Code", "Rate", "Observation_Date"]]
        .rename(columns={"Rate": "Latest_Rate", "Observation_Date": "Latest_Observation_Date"})
    )
    out = out.merge(latest, on="Code", how="left")
    global_latest_date = pd.to_datetime(out["Observation_Date"], errors="coerce").max()
    out["Data_Staleness_Days"] = (
        global_latest_date - pd.to_datetime(out["Latest_Observation_Date"], errors="coerce")
    ).dt.days if pd.notna(global_latest_date) else np.nan
    stale_limit = np.select(
        [
            out["Observation_Frequency"].astype(str).str.contains("Monthly", case=False, na=False),
            out["Observation_Frequency"].astype(str).str.contains("Quarterly", case=False, na=False),
            out["Observation_Frequency"].astype(str).str.contains("Low-frequency", case=False, na=False),
        ],
        [75, 120, 180],
        default=45,
    )
    out["Comparable_To_Current_Funding"] = out["Status"].astype(str).str.startswith("Active") & out["Data_Staleness_Days"].le(stale_limit)
    if not sofr.empty:
        out = pd.merge_asof(
            out.sort_values("Observation_Date"),
            sofr,
            on="Observation_Date",
            direction="nearest",
            tolerance=pd.Timedelta(days=7),
        ).sort_values(["Benchmark", "Observation_Date"]).reset_index(drop=True)
        out["Level_Diff_vs_SOFR_bps"] = (
            pd.to_numeric(out["Rate"], errors="coerce") - pd.to_numeric(out["SOFR_Aligned"], errors="coerce")
        ) * 100.0
    else:
        out["SOFR_Aligned"] = np.nan
        out["Level_Diff_vs_SOFR_bps"] = np.nan
    if use_cache:
        PERSISTENT_CACHE.set_df("interbank_reference_rates", payload, out)
    return out


def fetch_discrete_country_rate_frame(
    start,
    end,
    country: str,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    raw = fetch_macro_frame(start, end, country=country, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    if country == "United States" and not raw.empty:
        us_aliases = {"FEDFUNDS": "POLICY_RATE", "US10Y": "SOV_10Y", "US2Y": "SOV_2Y"}
        for src, dst in us_aliases.items():
            if src in raw.columns:
                raw[dst] = raw[src].combine_first(raw[dst] if dst in raw.columns else pd.Series(index=raw.index, dtype=float))
        if "Country_Rate_Source" not in raw.columns:
            raw["Country_Rate_Source"] = "FRED US sovereign/policy series"
    direct = fetch_direct_country_rates(start, end, country, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    if not direct.empty:
        raw = raw.join(direct, how="outer", rsuffix="_direct")
        for col in ["POLICY_RATE", "SOV_10Y", "SOV_2Y"]:
            direct_col = f"{col}_direct"
            if direct_col in raw.columns:
                raw[col] = raw[direct_col].combine_first(raw[col] if col in raw.columns else pd.Series(index=raw.index, dtype=float))
                raw = raw.drop(columns=[direct_col])
        if "Country_Rate_Source_direct" in raw.columns:
            raw["Country_Rate_Source"] = raw["Country_Rate_Source_direct"].combine_first(raw.get("Country_Rate_Source", pd.Series(index=raw.index, dtype=object)))
            raw = raw.drop(columns=["Country_Rate_Source_direct"])
    return raw.sort_index()


def global_yield_curve_discrete_history(
    prices: pd.DataFrame,
    countries: Iterable[str] | None = None,
    lookback_days: int = 365 * 3,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    countries = tuple(countries or GLOBAL_RATE_COUNTRIES)
    end = prices.index.max() + pd.Timedelta(days=5)
    start = max(prices.index.min() - pd.Timedelta(days=30), end - pd.Timedelta(days=int(lookback_days)))

    def one_country(country: str) -> pd.DataFrame:
        try:
            raw = fetch_discrete_country_rate_frame(start, end, country, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
            if raw.empty:
                return pd.DataFrame()
            source = str(raw.get("Country_Rate_Source", pd.Series(dtype=object)).dropna().iloc[-1]) if "Country_Rate_Source" in raw and raw["Country_Rate_Source"].dropna().size else "FRED/OECD public proxy"
            rows = []
            for col in ["POLICY_RATE", "SOV_2Y", "SOV_10Y"]:
                if col not in raw:
                    continue
                s = pd.to_numeric(raw[col], errors="coerce").dropna()
                if s.empty:
                    continue
                s = s[s.index >= start]
                if len(s) > 900:
                    s = s.tail(900)
                freq = infer_rate_frequency(s.index)
                for dt, value in s.items():
                    rows.append(
                        {
                            "Country": country,
                            "Observation_Date": pd.Timestamp(dt),
                            "Tenor_Code": col,
                            "Tenor": RATE_TENOR_LABELS.get(col, col),
                            "Rate": to_float(value),
                            "Observation_Frequency": freq,
                            "Source": source,
                        }
                    )
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    frames = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(countries)))) as ex:
        futures = {ex.submit(one_country, country): country for country in countries}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["Country", "Tenor_Code", "Observation_Date"]).reset_index(drop=True)


GEOPOLITICAL_TOPIC_QUERIES = {
    "Trade / Tariffs": "(tariff OR tariffs OR trade war OR export control OR sanctions)",
    "Wars / Security": "(war OR conflict OR missile OR invasion OR military escalation)",
    "Fiscal / Sovereign Debt": "(fiscal deficit OR debt ceiling OR tax reform OR sovereign debt OR budget crisis)",
    "Financial / Credit": "(banking crisis OR credit stress OR liquidity crunch OR default OR refinancing)",
    "Primary Markets / IPOs": "(IPO OR initial public offering OR listing OR new issue OR direct listing)",
    "Regulation / Legal": "(antitrust OR regulation OR lawsuit OR investigation OR compliance OR enforcement)",
}

GEOPOLITICAL_COUNTRY_ALIASES = {
    # Avoid ambiguous lowercase "us" and broad "American" because they create
    # false positives in headlines such as "Latin American markets..." or
    # ordinary English pronouns. The case-sensitive inline groups keep "US/USA"
    # useful without turning every "us" into the United States.
    "United States": [r"United States", r"U\.S\.", r"U\.S\.A\.", r"(?-i:US)", r"(?-i:USA)", r"Federal Reserve", r"White House"],
    "Mexico": [r"Mexico", r"Mexican", r"MXN", r"Pemex", r"AMLO", r"Sheinbaum"],
    "Canada": [r"Canada", r"Canadian", r"CAD", r"BoC"],
    "Brazil": [r"Brazil", r"Brazilian", r"BRL", r"Lula", r"Petrobras"],
    "China": [r"China", r"Chinese", r"Beijing", r"Shanghai", r"Shenzhen", r"yuan", r"renminbi", r"CNY", r"PBOC"],
    "France": [r"France", r"French", r"Paris", r"Macron"],
    "Germany": [r"Germany", r"German", r"Berlin", r"Bundesbank"],
    "Spain": [r"Spain", r"Spanish", r"Madrid"],
    "Italy": [r"Italy", r"Italian", r"Rome", r"Meloni"],
    "Netherlands": [r"Netherlands", r"Dutch", r"Amsterdam"],
    "United Kingdom": [r"United Kingdom", r"UK", r"U\.K\.", r"Britain", r"British", r"England", r"London", r"BoE", r"Sterling"],
    "Japan": [r"Japan", r"Japanese", r"Tokyo", r"yen", r"JPY", r"BOJ"],
    "India": [r"India", r"Indian", r"New Delhi", r"Mumbai", r"rupee", r"INR", r"RBI"],
    "South Korea": [r"South Korea", r"Korea", r"Korean", r"Seoul", r"KRW", r"BOK"],
    "Australia": [r"Australia", r"Australian", r"Sydney", r"AUD", r"RBA"],
    "New Zealand": [r"New Zealand", r"Kiwi", r"NZD", r"RBNZ"],
    "South Africa": [r"South Africa", r"South African", r"Johannesburg", r"rand", r"ZAR"],
    "Switzerland": [r"Switzerland", r"Swiss", r"Zurich", r"CHF", r"SNB"],
    "Sweden": [r"Sweden", r"Swedish", r"Stockholm", r"SEK", r"Riksbank"],
    "Norway": [r"Norway", r"Norwegian", r"Oslo", r"NOK", r"Norges Bank"],
    "Russia": [r"Russia", r"Russian", r"Moscow", r"Kremlin", r"ruble", r"RUB"],
    "Ukraine": [r"Ukraine", r"Ukrainian", r"Kyiv", r"Kiev"],
    "Israel": [r"Israel", r"Israeli", r"Jerusalem", r"Tel Aviv"],
    "Iran": [r"Iran", r"Iranian", r"Tehran"],
    "Saudi Arabia": [r"Saudi Arabia", r"Saudi", r"Riyadh", r"OPEC\+"],
    "Taiwan": [r"Taiwan", r"Taiwanese", r"Taipei", r"TSMC"],
}


def canonical_geo_country(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"global", "unknown", "none", "nan"}:
        return None
    for country, aliases in GEOPOLITICAL_COUNTRY_ALIASES.items():
        if text.lower() == country.lower() or any(re.fullmatch(alias, text, flags=re.IGNORECASE) for alias in aliases):
            return country
    return text


def _country_alias_hits(text: str) -> list[str]:
    haystack = str(text or "")
    hits = []
    for country, aliases in GEOPOLITICAL_COUNTRY_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?<![A-Za-z])(?:{alias})(?![A-Za-z])"
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                hits.append(country)
                break
    return list(dict.fromkeys(hits))


def infer_geopolitical_event_countries(title: str = "", query: str = "", source_country: str | None = None) -> dict:
    """Infer event countries from article text before using publisher metadata.

    This is intentionally transparent and zero-cost: it is not a full NER
    geocoder, but it prevents the map from confusing a U.S. publisher with a
    China/Mexico/Brazil event when the headline names the event country.
    """
    title_hits = _country_alias_hits(title)
    if title_hits:
        return {
            "Event_Countries": title_hits,
            "Geo_Inference_Method": "title_regex",
            "Geo_Inference_Confidence": float(min(0.95, 0.75 + 0.08 * len(title_hits))),
        }
    query_hits = _country_alias_hits(query)
    if query_hits:
        return {
            "Event_Countries": query_hits,
            "Geo_Inference_Method": "query_regex",
            "Geo_Inference_Confidence": float(min(0.80, 0.55 + 0.07 * len(query_hits))),
        }
    src = canonical_geo_country(source_country)
    if src:
        return {
            "Event_Countries": [src],
            "Geo_Inference_Method": "sourcecountry_fallback",
            "Geo_Inference_Confidence": 0.35,
        }
    return {
        "Event_Countries": [],
        "Geo_Inference_Method": "unresolved",
        "Geo_Inference_Confidence": 0.0,
    }


def robust_news_flow_diagnostics(series: pd.Series, min_obs: int = 20, min_unique: int = 4) -> dict:
    """Robust point-in-time news-flow shock diagnostics for one topic.

    GDELT raw counts are not cross-topic comparable. The signal is therefore a
    within-topic abnormal-attention statistic relative to its own historical
    distribution, using a median/MAD estimator to reduce sensitivity to event
    clusters and API count spikes.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return {}
    latest = float(s.iloc[-1])
    out = {
        "Latest_Volume": latest,
        "Baseline_Median": float(s.median()),
        "Sample_Size": int(len(s)),
        "Unique_Observations": int(s.nunique(dropna=True)),
        "Data_Source_Type": "GDELT_TIMELINE",
        "Cross_Topic_Raw_Comparable": False,
    }
    if len(s) < min_obs or s.nunique(dropna=True) < min_unique:
        out.update(
            {
                "Robust_Z_Score": np.nan,
                "Z_Score": np.nan,
                "Percentile": np.nan,
                "Positive_Shock_Score": np.nan,
                "Score_Type": "Insufficient GDELT timeline dispersion",
                "Thermometer": "Insufficient history",
                "Statistical_Admissibility": False,
                "Risk_Overlay_Admissible": False,
                "Quant_Interpretation": "Timeline count has insufficient observations or dispersion; do not use as a portfolio risk penalty.",
            }
        )
        return out
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        std = float(s.std(ddof=1))
        scale = std if np.isfinite(std) and std > 1e-12 else np.nan
    pct = float((s <= latest).mean())
    robust_z = (latest - med) / scale if pd.notna(scale) and scale > 0 else np.nan
    robust_z = float(np.clip(robust_z, -5.0, 5.0)) if pd.notna(robust_z) else np.nan
    positive_shock = max(0.0, robust_z) if pd.notna(robust_z) else np.nan
    if pd.isna(robust_z):
        label = "Insufficient dispersion"
    elif robust_z >= 2.0:
        label = "Extreme abnormal attention"
    elif robust_z >= 1.0:
        label = "Elevated abnormal attention"
    elif robust_z <= -1.0:
        label = "Below baseline"
    else:
        label = "Normal"
    out.update(
        {
            "Baseline_MAD": mad,
            "Robust_Scale": float(scale) if pd.notna(scale) else np.nan,
            "Robust_Z_Score": robust_z,
            "Z_Score": robust_z,
            "Percentile": pct,
            "Positive_Shock_Score": positive_shock,
            "Score_Type": "GDELT robust within-topic news-flow shock",
            "Thermometer": label,
            "Statistical_Admissibility": True,
            "Risk_Overlay_Admissible": bool(pd.notna(positive_shock) and positive_shock > 0.0),
            "Quant_Interpretation": (
                "Positive abnormal attention; eligible as a risk overlay input."
                if pd.notna(positive_shock) and positive_shock > 0.0
                else "Below/near baseline attention; report diagnostically but do not increase risk penalty."
            ),
        }
    )
    return out


def fetch_google_news_rss_articles(
    query: str,
    days: int = 1,
    max_records: int = 12,
    use_cache: bool = True,
    cache_ttl_hours: int = 3,
) -> pd.DataFrame:
    rss_query = re.sub(r"[()]", " ", str(query or ""))
    rss_query = re.sub(r"\bOR\b", " OR ", rss_query, flags=re.IGNORECASE)
    rss_query = f"{rss_query.strip()} when:{max(int(days), 1)}d"
    payload = {"v": 1, "query": rss_query, "days": days, "max_records": max_records, "source": "google_news_rss"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("google_news_rss_articles", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    params = urllib.parse.urlencode({"q": rss_query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    url = f"https://news.google.com/rss/search?{params}"
    rows = []
    try:
        xml = http_read_text(url, timeout=8)
        root = ET.fromstring(xml)
        for item in root.findall(".//item")[:max_records]:
            source = item.find("source")
            rows.append(
                {
                    "Title": item.findtext("title"),
                    "URL": item.findtext("link"),
                    "Domain": source.text if source is not None else None,
                    "SeenDate": item.findtext("pubDate"),
                    "Language": "en",
                    "SourceCountry": "global",
                    "Query": query,
                    "Source": "Google News RSS fallback",
                }
            )
    except Exception:
        rows = []
    out = pd.DataFrame(rows)
    if use_cache and not out.empty:
        PERSISTENT_CACHE.set_df("google_news_rss_articles", payload, out)
    return out


def fetch_gdelt_articles(
    query: str,
    days: int = 1,
    max_records: int = 12,
    use_cache: bool = True,
    cache_ttl_hours: int = 3,
    timeout: int = 8,
    use_rss_fallback: bool = True,
) -> pd.DataFrame:
    payload = {"v": 2, "query": query, "days": days, "max_records": max_records, "mode": "artlist"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("gdelt_articles", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    params = urllib.parse.urlencode(
        {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "sort": "hybridrel",
            "timespan": f"{max(days, 1)}d",
        }
    )
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
    try:
        data = http_read_json(url, timeout=timeout)
        rows = []
        for item in data.get("articles", []) or []:
            rows.append(
                {
                    "Title": item.get("title"),
                    "URL": item.get("url"),
                    "Domain": item.get("domain"),
                    "SeenDate": item.get("seendate"),
                    "Language": item.get("language"),
                    "SourceCountry": item.get("sourcecountry"),
                    "Query": query,
                    "Source": "GDELT DOC 2.1 artlist",
                }
            )
        out = pd.DataFrame(rows)
    except Exception:
        out = pd.DataFrame()
    if out.empty and use_rss_fallback:
        out = fetch_google_news_rss_articles(query, days=days, max_records=max_records, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    if use_cache and not out.empty and "GDELT" in str(out.get("Source", pd.Series(dtype=str)).astype(str).iloc[0] if not out.empty and "Source" in out else ""):
        PERSISTENT_CACHE.set_df("gdelt_articles", payload, out)
    return out


def geopolitical_thermometer(
    topic_queries: dict[str, str] | None = None,
    days: int = 90,
    article_days: int = 1,
    use_cache: bool = True,
    cache_ttl_hours: int = 3,
) -> dict[str, pd.DataFrame]:
    topic_queries = topic_queries or GEOPOLITICAL_TOPIC_QUERIES
    summary_rows, timelines, articles = [], [], []

    def one_topic(topic: str, query: str) -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
        timeline = fetch_gdelt_timeline(query, days=days, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours, timeout=8)
        summary = None
        if not timeline.empty:
            tmp = timeline.reset_index(names="Date").copy()
            tmp["Topic"] = topic
            s = pd.to_numeric(timeline["GDELT_Volume"], errors="coerce").dropna()
            diag = robust_news_flow_diagnostics(s)
            if diag:
                summary = {
                    "Topic": topic,
                    **diag,
                    "Article_Count": np.nan,
                    "Unique_Domains": np.nan,
                    "News_Flow_Score": np.nan,
                    "Source": "GDELT DOC 2.1 timeline",
                }
        else:
            tmp = pd.DataFrame()
        art = fetch_gdelt_articles(query, days=article_days, max_records=8, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
        if not art.empty:
            art = art.copy()
            art.insert(0, "Topic", topic)
            if summary is None:
                n = len(art)
                domains = art["Domain"].dropna().astype(str).str.lower().nunique() if "Domain" in art else 0
                source_diversity = safe_div(domains, n)
                proxy = float(np.clip(np.log1p(n) * (0.5 + 0.5 * source_diversity), 0.0, 3.0))
                summary = {
                    "Topic": topic,
                    "Latest_Volume": np.nan,
                    "Baseline_Median": np.nan,
                    "Robust_Z_Score": np.nan,
                    "Z_Score": np.nan,
                    "Percentile": np.nan,
                    "Article_Count": int(n),
                    "Unique_Domains": int(domains),
                    "News_Flow_Score": proxy,
                    "Score_Type": "Article-flow proxy; not a Z-score",
                    "Thermometer": "RSS News Flow" if str(art["Source"].iloc[0]).startswith("Google") else "Article Flow",
                    "Source": str(art["Source"].iloc[0]),
                    "Sample_Size": np.nan,
                    "Unique_Observations": np.nan,
                    "Positive_Shock_Score": np.nan,
                    "Data_Source_Type": "RSS_ARTICLE_FALLBACK" if str(art["Source"].iloc[0]).startswith("Google") else "GDELT_ARTICLE_FALLBACK",
                    "Cross_Topic_Raw_Comparable": False,
                    "Statistical_Admissibility": False,
                    "Risk_Overlay_Admissible": False,
                    "Quant_Interpretation": "Qualitative article fallback only; capped article count is not a time-series shock and must not be compared to GDELT timeline volumes.",
                }
        return summary, tmp, art

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(topic_queries)))) as ex:
        futures = {ex.submit(one_topic, topic, query): topic for topic, query in topic_queries.items()}
        for fut in as_completed(futures):
            try:
                summary, timeline, art = fut.result()
            except Exception:
                continue
            if summary is not None:
                summary_rows.append(summary)
            if timeline is not None and not timeline.empty:
                timelines.append(timeline)
            if art is not None and not art.empty:
                articles.append(art)
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        sort_col = "Positive_Shock_Score" if summary_df.get("Positive_Shock_Score", pd.Series(dtype=float)).notna().any() else "News_Flow_Score"
        summary_df = summary_df.sort_values(sort_col, ascending=False)
    articles_df = pd.concat(articles, ignore_index=True) if articles else pd.DataFrame()
    if not articles_df.empty:
        articles_df = add_english_article_titles(articles_df, use_cache=use_cache, cache_ttl_hours=max(cache_ttl_hours, 24))
    return {
        "summary": summary_df,
        "timeline": pd.concat(timelines, ignore_index=True) if timelines else pd.DataFrame(),
        "articles": articles_df,
        "country_heatmap": geopolitical_country_heatmap(articles_df, summary_df),
    }


def geopolitical_thermometer_model_audit(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame(
            [
                {"Metric": "Model_Status", "Value": "No geopolitical data", "Interpretation": "No quantitative risk overlay can be inferred."},
            ]
        )
    s = summary.copy()
    robust = pd.to_numeric(s.get("Robust_Z_Score", pd.Series(dtype=float)), errors="coerce")
    positive = pd.to_numeric(s.get("Positive_Shock_Score", pd.Series(dtype=float)), errors="coerce")
    stat_ok = s.get("Statistical_Admissibility", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    risk_ok = s.get("Risk_Overlay_Admissible", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    fallback = s.get("Data_Source_Type", pd.Series("", index=s.index)).astype(str).str.contains("FALLBACK", case=False, na=False)
    max_pos = positive[risk_ok].max() if risk_ok.any() else 0.0
    if max_pos >= 2.0:
        state = "High abnormal geopolitical/news attention"
    elif max_pos >= 1.0:
        state = "Elevated abnormal geopolitical/news attention"
    elif stat_ok.any():
        state = "No positive abnormal-attention shock"
    else:
        state = "Qualitative fallback only"
    return pd.DataFrame(
        [
            {
                "Metric": "Model_State",
                "Value": state,
                "Interpretation": "Only positive robust within-topic shocks are eligible for risk-overlay escalation.",
            },
            {
                "Metric": "Statistically_Admissible_Topics",
                "Value": int(stat_ok.sum()),
                "Interpretation": "Topics with enough GDELT timeline history and dispersion for robust z-score inference.",
            },
            {
                "Metric": "Risk_Overlay_Admissible_Topics",
                "Value": int(risk_ok.sum()),
                "Interpretation": "Topics with positive abnormal attention; negative z-scores are below baseline and should not raise risk.",
            },
            {
                "Metric": "Fallback_Only_Topics",
                "Value": int(fallback.sum()),
                "Interpretation": "Article fallback is qualitative evidence; capped counts are not comparable with timeline volumes.",
            },
            {
                "Metric": "Max_Positive_Shock",
                "Value": float(max_pos) if pd.notna(max_pos) else np.nan,
                "Interpretation": "Maximum positive robust z-score across admissible topics.",
            },
            {
                "Metric": "Raw_Cross_Topic_Comparability",
                "Value": "Rejected",
                "Interpretation": "Raw GDELT/RSS counts are query-dependent and cannot be linearly compared across topics.",
            },
        ]
    )


def geopolitical_country_heatmap(articles: pd.DataFrame, summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Country-level geo-news attention proxy from public article metadata.

    The primary layer is a transparent regex event-country inference over
    title/query text. GDELT `sourcecountry` is used only as a fallback, because
    the publishing country can differ from the country discussed in the article.
    """
    if articles is None or articles.empty:
        return pd.DataFrame()
    art = articles.copy()
    if "SourceCountry" not in art:
        art["SourceCountry"] = None
    inference = art.apply(
        lambda row: infer_geopolitical_event_countries(
            f"{row.get('Title_EN', row.get('Title', ''))} {row.get('Original_Title', '')}",
            row.get("Query", ""),
            row.get("SourceCountry"),
        ),
        axis=1,
    )
    art["Event_Countries"] = inference.map(lambda x: x.get("Event_Countries", []))
    art["Geo_Inference_Method"] = inference.map(lambda x: x.get("Geo_Inference_Method", "unresolved"))
    art["Geo_Inference_Confidence"] = inference.map(lambda x: x.get("Geo_Inference_Confidence", 0.0))
    art = art[art["Event_Countries"].map(len) > 0].copy()
    if art.empty:
        return pd.DataFrame()
    art = art.explode("Event_Countries").rename(columns={"Event_Countries": "Geo_Country"})
    art["Geo_Country"] = art["Geo_Country"].map(canonical_geo_country)
    art = art.dropna(subset=["Geo_Country"])
    if art.empty:
        return pd.DataFrame()
    topic_weight = {}
    topic_admissible = {}
    if summary is not None and not summary.empty and "Topic" in summary:
        for _, row in summary.iterrows():
            topic = str(row.get("Topic"))
            shock = to_float(row.get("Positive_Shock_Score"))
            news_flow = to_float(row.get("News_Flow_Score"))
            stat_ok = bool(row.get("Statistical_Admissibility", False)) if pd.notna(row.get("Statistical_Admissibility", np.nan)) else False
            risk_ok = bool(row.get("Risk_Overlay_Admissible", False)) if pd.notna(row.get("Risk_Overlay_Admissible", np.nan)) else False
            if stat_ok and pd.notna(shock):
                topic_weight[topic] = 1.0 + max(0.0, shock)
            elif pd.notna(news_flow):
                topic_weight[topic] = 0.25 + 0.25 * min(max(news_flow, 0.0), 3.0)
            else:
                topic_weight[topic] = 0.50
            topic_admissible[topic] = risk_ok
    art["_Topic_Weight"] = art.get("Topic", pd.Series("", index=art.index)).astype(str).map(topic_weight).fillna(0.50)
    rows = []
    for country, sub in art.groupby("Geo_Country"):
        domains = sub["Domain"].dropna().astype(str).str.lower().nunique() if "Domain" in sub else 0
        n = len(sub)
        topic_count = sub["Topic"].nunique() if "Topic" in sub else 0
        domain_diversity = safe_div(domains, n)
        confidence = pd.to_numeric(sub["Geo_Inference_Confidence"], errors="coerce").fillna(0.0)
        weighted = float((sub["_Topic_Weight"] * confidence.clip(lower=0.0, upper=1.0)).sum())
        score = float(np.log1p(n) * (0.50 + 0.50 * domain_diversity) * max(1.0, weighted / max(n, 1)))
        admissible_topics = int(sub.get("Topic", pd.Series("", index=sub.index)).astype(str).map(topic_admissible).fillna(False).sum())
        methods = ", ".join(sorted(sub["Geo_Inference_Method"].dropna().astype(str).unique()))
        rows.append(
            {
                "Country": country,
                "Geo_News_Attention_Score": score,
                "Article_Count": int(n),
                "Unique_Domains": int(domains),
                "Topic_Count": int(topic_count),
                "Weighted_Topic_Intensity": weighted,
                "Risk_Overlay_Article_Count": admissible_topics,
                "Dominant_Topic": sub["Topic"].mode().iloc[0] if "Topic" in sub and not sub["Topic"].mode().empty else None,
                "Geo_Inference_Methods": methods,
                "Mean_Geo_Inference_Confidence": float(confidence.mean()) if len(confidence) else np.nan,
                "Regex_Inferred_Article_Count": int(sub["Geo_Inference_Method"].astype(str).str.contains("regex", na=False).sum()),
                "SourceCountry_Fallback_Count": int(sub["Geo_Inference_Method"].eq("sourcecountry_fallback").sum()),
                "Data_Source": "GDELT/RSS article metadata with regex country inference",
                "Quant_Interpretation": "Regex-inferred event-country attention proxy; source-country is fallback only.",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Percentile"] = out["Geo_News_Attention_Score"].rank(pct=True)
    out["Heat_Level"] = np.select(
        [out["Percentile"] >= 0.90, out["Percentile"] >= 0.70, out["Percentile"] >= 0.40],
        ["Extreme", "Elevated", "Moderate"],
        default="Low",
    )
    return out.sort_values("Geo_News_Attention_Score", ascending=False).reset_index(drop=True)


FOREX_FACTORY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FOREX_FACTORY_IMPACT_WEIGHTS = {"Holiday": 0.10, "Low": 0.25, "Medium": 0.60, "High": 1.00}
CENTRAL_TIMEZONE = "America/Mexico_City"
CURRENCY_BY_COUNTRY = {
    "United States": "USD",
    "Mexico": "MXN",
    "Brazil": "BRL",
    "China": "CNY",
    "France": "EUR",
    "Germany": "EUR",
    "Spain": "EUR",
    "Italy": "EUR",
    "Netherlands": "EUR",
    "Canada": "CAD",
    "United Kingdom": "GBP",
    "Japan": "JPY",
    "India": "INR",
    "South Korea": "KRW",
    "Australia": "AUD",
    "New Zealand": "NZD",
    "South Africa": "ZAR",
    "Switzerland": "CHF",
    "Sweden": "SEK",
    "Norway": "NOK",
}

FX_USD_SPOT_YAHOO = {
    "USD": None,
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "NZD": "NZDUSD=X",
    "JPY": "JPY=X",
    "MXN": "MXN=X",
    "BRL": "BRL=X",
    "CNY": "CNY=X",
    "CAD": "CAD=X",
    "CHF": "CHF=X",
    "SEK": "SEK=X",
    "NOK": "NOK=X",
    "INR": "INR=X",
    "KRW": "KRW=X",
    "ZAR": "ZAR=X",
}


def parse_forex_factory_datetime(date_text: str, time_text: str) -> pd.Timestamp:
    date = pd.to_datetime(str(date_text or "").strip(), format="%m-%d-%Y", errors="coerce")
    if pd.isna(date):
        return pd.NaT
    time_raw = str(time_text or "").strip()
    if not time_raw or time_raw.lower() in {"all day", "day 1", "day 2", "tentative"}:
        return pd.Timestamp(date).normalize()
    parsed_time = pd.to_datetime(time_raw, format="%I:%M%p", errors="coerce")
    if pd.isna(parsed_time):
        parsed_time = pd.to_datetime(time_raw, errors="coerce")
    if pd.isna(parsed_time):
        return pd.Timestamp(date).normalize()
    return pd.Timestamp(date).normalize() + pd.Timedelta(hours=parsed_time.hour, minutes=parsed_time.minute)


def fetch_forex_factory_calendar(
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    payload = {"url": FOREX_FACTORY_CALENDAR_URL, "scope": "thisweek"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("forex_factory_calendar", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    rows = []
    try:
        xml_text = http_read_text(FOREX_FACTORY_CALENDAR_URL, timeout=12)
        root = ET.fromstring(xml_text)
        def txt(event_node, tag):
            val = event_node.findtext(tag)
            return str(val).strip() if val is not None else ""

        for event in root.findall(".//event"):
            date_text = txt(event, "date")
            time_text = txt(event, "time")
            impact = txt(event, "impact") or "Unknown"
            currency = txt(event, "country")
            event_dt = parse_forex_factory_datetime(date_text, time_text)
            central_time = pd.NaT
            if pd.notna(event_dt):
                try:
                    central_time = (
                        pd.Timestamp(event_dt)
                        .tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
                        .tz_convert(CENTRAL_TIMEZONE)
                        .tz_localize(None)
                    )
                except Exception:
                    central_time = pd.Timestamp(event_dt)
            rows.append(
                {
                    "Event_Time": event_dt,
                    "Central_Time": central_time,
                    "Timezone": CENTRAL_TIMEZONE,
                    "Date": pd.Timestamp(event_dt).date() if pd.notna(event_dt) else pd.NaT,
                    "Time": time_text,
                    "Currency": currency,
                    "Impact": impact,
                    "Impact_Weight": FOREX_FACTORY_IMPACT_WEIGHTS.get(impact, 0.0),
                    "Event": txt(event, "title"),
                    "Actual": txt(event, "actual"),
                    "Forecast": txt(event, "forecast"),
                    "Previous": txt(event, "previous"),
                    "URL": txt(event, "url"),
                    "Source": "ForexFactory/FairEconomy weekly XML",
                    "Availability_Date": pd.Timestamp.utcnow().tz_localize(None),
                }
            )
    except Exception:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Event_Time", "Currency", "Impact_Weight"], ascending=[True, True, False]).reset_index(drop=True)
        if use_cache:
            PERSISTENT_CACHE.set_df("forex_factory_calendar", payload, out)
    return out


def forex_factory_event_risk(calendar: pd.DataFrame, now: pd.Timestamp | None = None, horizon_days: int = 7) -> pd.DataFrame:
    if calendar is None or calendar.empty:
        return pd.DataFrame()
    now = pd.Timestamp(now or pd.Timestamp.utcnow()).tz_localize(None)
    out = calendar.copy()
    out["Event_Time"] = pd.to_datetime(out["Event_Time"], errors="coerce")
    out = out.dropna(subset=["Event_Time"])
    out["Hours_To_Event"] = (out["Event_Time"] - now).dt.total_seconds() / 3600.0
    future = out[(out["Hours_To_Event"] >= -6.0) & (out["Hours_To_Event"] <= horizon_days * 24.0)].copy()
    if future.empty:
        return pd.DataFrame()
    future["Time_Decay"] = np.exp(-future["Hours_To_Event"].clip(lower=0.0) / 48.0)
    future["EventRisk"] = pd.to_numeric(future["Impact_Weight"], errors="coerce").fillna(0.0) * future["Time_Decay"]
    summary = future.groupby("Currency").agg(
        Events=("Event", "count"),
        High_Impact=("Impact", lambda x: int((pd.Series(x).astype(str) == "High").sum())),
        Medium_Impact=("Impact", lambda x: int((pd.Series(x).astype(str) == "Medium").sum())),
        EventRiskScore=("EventRisk", "sum"),
        Next_Event_Time=("Event_Time", "min"),
    ).reset_index()
    summary["EventRiskLevel"] = np.select(
        [summary["EventRiskScore"] >= 1.50, summary["EventRiskScore"] >= 0.75, summary["EventRiskScore"] > 0],
        ["High", "Medium", "Low"],
        default="None",
    )
    summary["Suggested_Action"] = np.select(
        [summary["EventRiskLevel"].eq("High"), summary["EventRiskLevel"].eq("Medium")],
        ["Avoid rebalance / raise CVaR penalty", "Review execution timing"],
        default="Monitor",
    )
    return summary.sort_values("EventRiskScore", ascending=False).reset_index(drop=True)


def fetch_fx_usd_value_series(
    currencies: Iterable[str],
    period: str = "3y",
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    """Fetch zero-cost Yahoo FX spot proxies converted to USD value per currency unit."""
    currencies = tuple(dict.fromkeys([str(c).upper().strip() for c in currencies if str(c).strip()]))
    if not currencies:
        return pd.DataFrame()
    payload = {"version": 1, "currencies": currencies, "period": period}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("fx_usd_value_series", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached.sort_index().ffill()
    ticker_map = {ccy: FX_USD_SPOT_YAHOO.get(ccy) for ccy in currencies if FX_USD_SPOT_YAHOO.get(ccy)}
    out = pd.DataFrame()
    if ticker_map:
        try:
            raw = yf.download(
                tickers=list(dict.fromkeys(ticker_map.values())),
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if not raw.empty:
                close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": next(iter(ticker_map.values()))})
                close = close.sort_index().ffill()
                for ccy, ticker in ticker_map.items():
                    if ticker not in close:
                        continue
                    s = pd.to_numeric(close[ticker], errors="coerce")
                    if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                        out[ccy] = s
                    else:
                        out[ccy] = 1.0 / s.replace(0, np.nan)
        except Exception:
            out = pd.DataFrame()
    if "USD" in currencies:
        if out.empty:
            out = pd.DataFrame(index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252))
        out["USD"] = 1.0
    out = out.sort_index().ffill().dropna(axis=1, how="all")
    if use_cache and not out.empty:
        PERSISTENT_CACHE.set_df("fx_usd_value_series", payload, out)
    return out


def fx_pair_risk_metrics(long_currency: str, short_currency: str, fx_usd_values: pd.DataFrame) -> dict:
    long_currency = str(long_currency).upper()
    short_currency = str(short_currency).upper()
    if fx_usd_values is None or fx_usd_values.empty or long_currency not in fx_usd_values or short_currency not in fx_usd_values:
        return {
            "FX_Pair": f"{long_currency}/{short_currency}",
            "FX_Data_Status": "missing_fx_spot_proxy",
            "FX_Ann_Vol": np.nan,
            "FX_Max_Drawdown": np.nan,
            "FX_Trend_252D": np.nan,
        }
    cross = (pd.to_numeric(fx_usd_values[long_currency], errors="coerce") / pd.to_numeric(fx_usd_values[short_currency], errors="coerce")).dropna()
    if len(cross) < 60:
        return {
            "FX_Pair": f"{long_currency}/{short_currency}",
            "FX_Data_Status": "insufficient_fx_history",
            "FX_Ann_Vol": np.nan,
            "FX_Max_Drawdown": np.nan,
            "FX_Trend_252D": np.nan,
        }
    log_ret = np.log(cross).diff().dropna()
    equity = cross / cross.iloc[0]
    return {
        "FX_Pair": f"{long_currency}/{short_currency}",
        "FX_Data_Status": "ok_yahoo_spot_proxy",
        "FX_Ann_Vol": float(log_ret.std(ddof=1) * np.sqrt(252.0)),
        "FX_Max_Drawdown": float((equity / equity.cummax() - 1.0).min()),
        "FX_Trend_252D": float(cross.iloc[-1] / cross.iloc[max(0, len(cross) - 252)] - 1.0),
    }


def carry_trade_suggestions(global_rates: pd.DataFrame, event_risk: pd.DataFrame | None = None) -> pd.DataFrame:
    if global_rates is None or global_rates.empty:
        return pd.DataFrame()
    base = global_rates.copy()
    base["Currency"] = base["Country"].map(CURRENCY_BY_COUNTRY).fillna(base["Country"])
    base = base.dropna(subset=["Yield_10Y"])
    if base.empty:
        return pd.DataFrame()
    rows = []
    for _, long_row in base.iterrows():
        for _, short_row in base.iterrows():
            if long_row["Currency"] == short_row["Currency"]:
                continue
            carry = to_float(long_row.get("Yield_10Y")) - to_float(short_row.get("Yield_10Y"))
            if pd.isna(carry):
                continue
            long_curve = to_float(long_row.get("Curve_10Y_2Y"))
            short_curve = to_float(short_row.get("Curve_10Y_2Y"))
            long_term = to_float(long_row.get("Term_Premium_Proxy"))
            short_term = to_float(short_row.get("Term_Premium_Proxy"))
            inversion_penalty = max(0.0, -long_curve if pd.notna(long_curve) else 0.0)
            funding_stress = max(0.0, short_curve if pd.notna(short_curve) else 0.0) * 0.10
            event_penalty = 0.0
            if event_risk is not None and not event_risk.empty:
                er = event_risk.set_index("Currency")
                event_penalty += 0.15 * to_float(er["EventRiskScore"].get(long_row["Currency"], 0.0))
                event_penalty += 0.05 * to_float(er["EventRiskScore"].get(short_row["Currency"], 0.0))
            score = (
                carry
                + 0.20 * (long_curve if pd.notna(long_curve) else 0.0)
                + 0.10 * (long_term if pd.notna(long_term) else 0.0)
                - 0.10 * (short_term if pd.notna(short_term) else 0.0)
                - 0.35 * inversion_penalty
                - funding_stress
                - event_penalty
            )
            rows.append(
                {
                    "Long_Country": long_row["Country"],
                    "Long_Currency": long_row["Currency"],
                    "Short_Country": short_row["Country"],
                    "Short_Currency": short_row["Currency"],
                    "Carry_10Y_Spread": carry,
                    "Long_10Y": long_row.get("Yield_10Y"),
                    "Short_10Y": short_row.get("Yield_10Y"),
                    "Long_Curve_10Y_2Y": long_curve,
                    "Short_Curve_10Y_2Y": short_curve,
                    "Long_Regime": f"{long_row.get('Regime_Hawkish_Dovish')}/{long_row.get('Regime_Bull_Bear')}",
                    "Short_Regime": f"{short_row.get('Regime_Hawkish_Dovish')}/{short_row.get('Regime_Bull_Bear')}",
                    "Event_Risk_Penalty": event_penalty,
                    "Carry_Trade_Score": score,
                    "Signal": "Candidate" if score > 0 and carry > 0 else "Avoid",
                    "Risk_Note": "Uncovered carry screen; use FX-risk-adjusted validation before action.",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Carry_Trade_Score", ascending=False).reset_index(drop=True)


def validate_carry_trade_strategies(
    carry_trade: pd.DataFrame,
    global_rates: pd.DataFrame | None = None,
    fx_usd_values: pd.DataFrame | None = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
) -> pd.DataFrame:
    if carry_trade is None or carry_trade.empty:
        return pd.DataFrame()
    c = carry_trade.copy()
    currencies = sorted(set(c.get("Long_Currency", pd.Series(dtype=str)).dropna().astype(str)) | set(c.get("Short_Currency", pd.Series(dtype=str)).dropna().astype(str)))
    if fx_usd_values is None:
        fx_usd_values = fetch_fx_usd_value_series(currencies, period="3y", use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    fx_rows = []
    for _, row in c.iterrows():
        fx_rows.append(fx_pair_risk_metrics(row.get("Long_Currency"), row.get("Short_Currency"), fx_usd_values))
    fx_df = pd.DataFrame(fx_rows)
    if not fx_df.empty:
        c = pd.concat([c.reset_index(drop=True), fx_df.reset_index(drop=True)], axis=1)
    score = pd.to_numeric(c.get("Carry_Trade_Score"), errors="coerce")
    score_scale = float(iqr(score.dropna())) if score.notna().sum() > 3 else float(score.std(ddof=1) if score.notna().sum() > 1 else np.nan)
    score_scale = max(score_scale, 1e-9) if np.isfinite(score_scale) else 1.0
    c["Carry_Z_Proxy"] = (score - score.median()) / score_scale
    c["UIP_BreakEven_FX_Depreciation_Pct"] = pd.to_numeric(c.get("Carry_10Y_Spread"), errors="coerce")
    c["FX_Ann_Vol_Pct"] = 100.0 * pd.to_numeric(c.get("FX_Ann_Vol"), errors="coerce")
    c["FX_Max_Drawdown_Pct"] = 100.0 * pd.to_numeric(c.get("FX_Max_Drawdown"), errors="coerce")
    c["FX_Trend_252D_Pct"] = 100.0 * pd.to_numeric(c.get("FX_Trend_252D"), errors="coerce")
    c["FX_Vol_Carry_Coverage"] = c["UIP_BreakEven_FX_Depreciation_Pct"] / c["FX_Ann_Vol_Pct"].replace(0, np.nan)
    c["FX_Risk_Adjusted_Carry_Score"] = (
        pd.to_numeric(c.get("Carry_10Y_Spread"), errors="coerce")
        - 0.25 * c["FX_Ann_Vol_Pct"].fillna(20.0)
        - 0.10 * c["FX_Max_Drawdown_Pct"].abs().fillna(20.0)
        - pd.to_numeric(c.get("Event_Risk_Penalty"), errors="coerce").fillna(0.0)
    )
    c["Curve_Quality_Flag"] = np.select(
        [
            pd.to_numeric(c.get("Long_Curve_10Y_2Y"), errors="coerce") < -0.50,
            pd.to_numeric(c.get("Long_Curve_10Y_2Y"), errors="coerce") > 0.25,
        ],
        ["Long curve inverted: policy/recession risk", "Positive slope: better carry durability"],
        default="Neutral curve information",
    )
    c["Event_Risk_Flag"] = np.select(
        [
            pd.to_numeric(c.get("Event_Risk_Penalty"), errors="coerce").fillna(0.0) >= 0.35,
            pd.to_numeric(c.get("Event_Risk_Penalty"), errors="coerce").fillna(0.0) > 0.0,
        ],
        ["High event risk", "Moderate event risk"],
        default="No material event penalty",
    )
    carry_positive = pd.to_numeric(c.get("Carry_10Y_Spread"), errors="coerce") > 0
    score_positive = pd.to_numeric(c.get("Carry_Trade_Score"), errors="coerce") > 0
    event_ok = pd.to_numeric(c.get("Event_Risk_Penalty"), errors="coerce").fillna(0.0) < 0.50
    fx_ok = c["FX_Risk_Adjusted_Carry_Score"] > 0
    c["Mathematical_Admissibility"] = np.where(
        carry_positive & score_positive & event_ok & fx_ok,
        "Admissible research candidate after FX-vol/drawdown adjustment; not arbitrage",
        "Rejected/fragile after score, carry, event-risk, or FX-risk constraint",
    )
    c["FX_Hedge_Proxy"] = np.where(
        c["FX_Data_Status"].eq("ok_yahoo_spot_proxy"),
        "Spot-risk adjusted; forward hedge/basis unavailable in zero-cost data",
        "No reliable FX proxy; do not use as actionable carry",
    )
    c["No_Arbitrage_Check"] = (
        "Covered carry requires FX forwards and cross-currency basis. This zero-cost table ranks uncovered/spot-risk-adjusted carry premia, not CIP arbitrage."
    )
    c["Validation_Equation"] = (
        "E[rx] ~= i_long - i_short - E[Delta s]; require spread > FX-vol/drawdown + event/curve penalties."
    )
    keep = [
        "Long_Currency",
        "Short_Currency",
        "FX_Pair",
        "Carry_10Y_Spread",
        "Carry_Trade_Score",
        "FX_Risk_Adjusted_Carry_Score",
        "Carry_Z_Proxy",
        "UIP_BreakEven_FX_Depreciation_Pct",
        "FX_Ann_Vol_Pct",
        "FX_Max_Drawdown_Pct",
        "FX_Trend_252D_Pct",
        "FX_Vol_Carry_Coverage",
        "FX_Data_Status",
        "FX_Hedge_Proxy",
        "Curve_Quality_Flag",
        "Event_Risk_Flag",
        "Mathematical_Admissibility",
        "No_Arbitrage_Check",
        "Validation_Equation",
    ]
    return c[[col for col in keep if col in c.columns]].reset_index(drop=True)


def alternative_data_diagnostics(
    macro: pd.DataFrame,
    use_gdelt: bool = True,
    gdelt_query: str = "",
    use_cache: bool = True,
    cache_ttl_hours: int = 6,
    use_forex_factory: bool = True,
    forex_factory_cache_ttl_hours: int = 24,
) -> dict[str, pd.DataFrame]:
    rows = []
    if macro is not None and not macro.empty:
        latest = macro.dropna(how="all").iloc[-1]
        for col in ["FED_BALANCE_SHEET", "FED_REVERSE_REPO", "NFCI", "VIX", "HY_OAS", "EPU", "WTI", "IPMAN_YoY"]:
            if col in macro:
                s = pd.to_numeric(macro[col], errors="coerce").dropna()
                if s.empty:
                    continue
                z = (s.iloc[-1] - s.tail(252).mean()) / s.tail(252).std(ddof=1) if len(s.tail(252)) > 5 and s.tail(252).std(ddof=1) > 0 else np.nan
                rows.append({"Signal": col, "Latest": s.iloc[-1], "Z_252": z, "Source": "FRED"})
    gdelt = pd.DataFrame()
    if use_gdelt and gdelt_query:
        gdelt = fetch_gdelt_timeline(gdelt_query, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
        if not gdelt.empty:
            s = pd.to_numeric(gdelt["GDELT_Volume"], errors="coerce").dropna()
            if not s.empty:
                z = (s.iloc[-1] - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
                rows.append({"Signal": "GDELT_Geopolitical_Volume", "Latest": s.iloc[-1], "Z_252": z, "Source": "GDELT DOC 2.1"})
    ff_calendar = fetch_forex_factory_calendar(use_cache=use_cache, cache_ttl_hours=forex_factory_cache_ttl_hours) if use_forex_factory else pd.DataFrame()
    ff_event_risk = forex_factory_event_risk(ff_calendar) if not ff_calendar.empty else pd.DataFrame()
    if not ff_event_risk.empty:
        rows.append(
            {
                "Signal": "ForexFactory_EventRisk_Max",
                "Latest": ff_event_risk["EventRiskScore"].max(),
                "Z_252": np.nan,
                "Source": "ForexFactory/FairEconomy weekly XML",
            }
        )
    return {
        "summary": pd.DataFrame(rows),
        "gdelt_timeline": gdelt.reset_index(names="Date") if not gdelt.empty else pd.DataFrame(),
        "forex_factory_calendar": ff_calendar,
        "forex_factory_event_risk": ff_event_risk,
    }


def _causal_zscore_frame(features: pd.DataFrame, window: int = 252, min_periods: int = 60) -> pd.DataFrame:
    x = features.replace([np.inf, -np.inf], np.nan).astype(float)
    mu = x.rolling(window=window, min_periods=min_periods).mean()
    sig = x.rolling(window=window, min_periods=min_periods).std(ddof=1).replace(0.0, np.nan)
    return (x - mu) / sig


def market_sentiment_sem(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    forex_event_risk: pd.DataFrame | None = None,
    geopolitical_summary: pd.DataFrame | None = None,
    benchmark: str = "SPY",
    lookback: int = 756,
) -> dict[str, pd.DataFrame]:
    """Single-factor latent market sentiment SEM from zero-cost public inputs."""
    empty = {
        "timeline": pd.DataFrame(),
        "latest": pd.DataFrame(),
        "loadings": pd.DataFrame(),
        "structural_links": pd.DataFrame(),
        "diagnostics": pd.DataFrame(),
    }
    if prices is None or prices.empty:
        return empty
    px = prices.sort_index().ffill().dropna(axis=1, how="all").tail(max(int(lookback), 252) + 260)
    if px.empty:
        return empty
    ret = px.pct_change(fill_method=None)
    if benchmark in px.columns:
        bret = ret[benchmark]
        bpx = px[benchmark]
    else:
        bret = ret.mean(axis=1, skipna=True)
        bpx = px.mean(axis=1, skipna=True)

    raw = pd.DataFrame(index=px.index)
    raw["Equity_Momentum_21D"] = bpx.pct_change(21)
    raw["Equity_Momentum_63D"] = bpx.pct_change(63)
    raw["Cross_Asset_Breadth_21D"] = (px.pct_change(21) > 0).mean(axis=1)
    raw["Inverse_Realized_Vol_21D"] = -bret.rolling(21, min_periods=15).std(ddof=1) * np.sqrt(252.0)
    raw["Inverse_Downside_Vol_63D"] = -bret.clip(upper=0.0).rolling(63, min_periods=30).std(ddof=1) * np.sqrt(252.0)
    if macro is not None and not macro.empty:
        m = macro.sort_index().ffill().reindex(raw.index).ffill()
        macro_map = {
            "VIX": "Inverse_VIX",
            "HY_OAS": "Inverse_HY_OAS",
            "CREDIT_SPREAD": "Inverse_Credit_Spread",
            "NFCI": "Inverse_Financial_Stress",
            "EPU": "Inverse_Policy_Uncertainty",
        }
        for src, dst in macro_map.items():
            if src in m:
                raw[dst] = -pd.to_numeric(m[src], errors="coerce")
        if {"SOV_10Y", "SOV_2Y"}.issubset(m.columns):
            raw["Curve_Normality_10Y_2Y"] = pd.to_numeric(m["SOV_10Y"], errors="coerce") - pd.to_numeric(m["SOV_2Y"], errors="coerce")
        elif {"US10Y", "US2Y"}.issubset(m.columns):
            raw["Curve_Normality_10Y_2Y"] = pd.to_numeric(m["US10Y"], errors="coerce") - pd.to_numeric(m["US2Y"], errors="coerce")
        if "USD_BROAD" in m:
            raw["Inverse_USD_Shock_21D"] = -pd.to_numeric(m["USD_BROAD"], errors="coerce").pct_change(21)

    event_penalty = 0.0
    if forex_event_risk is not None and not forex_event_risk.empty and "EventRiskScore" in forex_event_risk:
        event_penalty = float(pd.to_numeric(forex_event_risk["EventRiskScore"], errors="coerce").fillna(0.0).sum())
    if event_penalty > 0:
        raw["Inverse_Event_Risk_Today"] = 0.0
        raw.iloc[-1, raw.columns.get_loc("Inverse_Event_Risk_Today")] = -event_penalty
    geo_penalty = 0.0
    if geopolitical_summary is not None and not geopolitical_summary.empty and "Positive_Shock_Score" in geopolitical_summary:
        geo_penalty = float(pd.to_numeric(geopolitical_summary["Positive_Shock_Score"], errors="coerce").fillna(0.0).sum())
    if geo_penalty > 0:
        raw["Inverse_Geopolitical_Shock_Today"] = 0.0
        raw.iloc[-1, raw.columns.get_loc("Inverse_Geopolitical_Shock_Today")] = -geo_penalty

    z = _causal_zscore_frame(raw, window=252, min_periods=60).tail(max(int(lookback), 120))
    z = z.dropna(axis=1, thresh=max(40, int(0.20 * len(z))))
    valid = z.dropna(how="all")
    if valid.empty or valid.shape[1] < 2:
        return empty
    x = valid.fillna(valid.median()).clip(-5.0, 5.0)
    x_dm = x - x.mean(axis=0)
    try:
        _, svals, vt = np.linalg.svd(x_dm.values, full_matrices=False)
        load = pd.Series(vt[0], index=x.columns, dtype=float)
        eta = pd.Series(x_dm.values @ load.values, index=x.index, dtype=float)
        anchor = x["Equity_Momentum_21D"] if "Equity_Momentum_21D" in x else x.iloc[:, 0]
        if eta.corr(anchor) < 0:
            load = -load
            eta = -eta
        eta_z = (eta - eta.rolling(252, min_periods=60).mean()) / eta.rolling(252, min_periods=60).std(ddof=1).replace(0.0, np.nan)
        explained = float((svals[0] ** 2) / np.sum(svals ** 2)) if np.sum(svals ** 2) > 0 else np.nan
    except Exception:
        return empty

    timeline = pd.DataFrame({"Date": eta_z.index, "Latent_Market_Sentiment_SEM": eta_z.values})
    timeline["Sentiment_Prob_Risk_On"] = 1.0 / (1.0 + np.exp(-timeline["Latent_Market_Sentiment_SEM"].clip(-6, 6)))
    timeline["Sentiment_State"] = pd.cut(
        timeline["Latent_Market_Sentiment_SEM"],
        bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
        labels=["Risk-off stress", "Defensive", "Neutral", "Constructive", "Risk-on"],
    ).astype(str)
    for col in x.columns:
        timeline[col] = x[col].reindex(eta_z.index).values

    links = []
    y = bret.reindex(eta_z.index).shift(-1).rename("Next_Benchmark_Return")
    reg = pd.concat([y, eta_z.rename("Latent_Market_Sentiment_SEM")], axis=1).dropna()
    if len(reg) >= 60 and reg["Latent_Market_Sentiment_SEM"].std(ddof=1) > 0:
        xmat = np.c_[np.ones(len(reg)), reg["Latent_Market_Sentiment_SEM"].values]
        beta = np.linalg.lstsq(xmat, reg["Next_Benchmark_Return"].values, rcond=None)[0]
        pred = xmat @ beta
        resid = reg["Next_Benchmark_Return"].values - pred
        ss_tot = np.sum((reg["Next_Benchmark_Return"].values - reg["Next_Benchmark_Return"].mean()) ** 2)
        r2 = 1.0 - np.sum(resid * resid) / ss_tot if ss_tot > 0 else np.nan
        links.append(
            {
                "Equation": "Next benchmark return ~ latent sentiment",
                "Beta_Latent_Sentiment": float(beta[1]),
                "Intercept": float(beta[0]),
                "R2": float(r2) if pd.notna(r2) else np.nan,
                "N": int(len(reg)),
                "Causal_Timing": "eta_t uses public information through t; return is t+1",
            }
        )

    loadings = (
        load.rename("Loading")
        .to_frame()
        .assign(Abs_Loading=lambda d: d["Loading"].abs(), Measurement_Equation="x_j,t = lambda_j eta_t + epsilon_j,t")
        .sort_values("Abs_Loading", ascending=False)
        .reset_index()
        .rename(columns={"index": "Indicator"})
    )
    latest = timeline.dropna(subset=["Latent_Market_Sentiment_SEM"]).tail(1)
    diagnostics = pd.DataFrame(
        [
            {"Metric": "Estimator", "Value": "Single latent factor PCA-SEM", "Interpretation": "Transparent zero-cost proxy for a one-factor measurement model."},
            {"Metric": "Explained_Variance_First_Factor", "Value": explained, "Interpretation": "Share of standardized indicator variance explained by latent sentiment."},
            {"Metric": "Indicator_Count", "Value": int(len(load)), "Interpretation": "Number of public measurement indicators entering SEM."},
            {"Metric": "Lookahead_Control", "Value": "Rolling causal z-scores", "Interpretation": "Historical sentiment path does not use future cross-sectional normalization."},
        ]
    )
    return {
        "timeline": timeline.reset_index(drop=True),
        "latest": latest.reset_index(drop=True),
        "loadings": loadings,
        "structural_links": pd.DataFrame(links),
        "diagnostics": diagnostics,
    }


def event_risk_exposure_multiplier(event_risk: pd.DataFrame, currency: str = "USD") -> float:
    if event_risk is None or event_risk.empty:
        return 1.0
    row = event_risk[event_risk["Currency"].astype(str).eq(str(currency).upper())]
    if row.empty:
        return 1.0
    score = to_float(row.iloc[0].get("EventRiskScore"))
    if pd.isna(score):
        return 1.0
    return float(np.clip(1.0 - 0.15 * score, 0.70, 1.0))


def score_cross_section(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    macro_row: pd.Series,
    asof_date,
    volumes: pd.DataFrame | None = None,
    use_garch: bool = True,
    garch_candidate_n: int | None = 30,
    crlb_penalty: float = 0.15,
    garch_penalty: float = 0.10,
    evt_penalty: float = 0.10,
    text_risk_penalty: float = 0.10,
    min_dollar_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    fund = fundamentals_asof(panel, prices, asof_date)
    if fund.empty:
        return pd.DataFrame()
    fund = add_pit_confidence(fund, asof_date=asof_date)
    ratio_cols = [c for c in CORE_RATIO_COLS if c in fund.columns and fund[c].notna().sum() > 0]
    cs = add_mahalanobis(fund, ratio_cols)
    tech = price_features(prices, asof_date)
    if tech.empty:
        return pd.DataFrame()
    cs = cs.merge(tech, on="Ticker", how="left")
    liq = liquidity_features(prices, volumes, asof_date) if volumes is not None else pd.DataFrame()
    if not liq.empty:
        cs = cs.merge(liq, on="Ticker", how="left")
    cs = robust_zscores(cs, ["Momentum_21", "Momentum_63", "Momentum_126", "Volatility_63", "Max_Drawdown_126", "Trend_50_200"])
    growth_cols = ["Revenue_Growth", "EPS_Growth", "Gross_Margin", "EBIT_Margin", "FCF_Margin", "Gross_Margin_Change"]
    if any(c in cs for c in growth_cols):
        cs = robust_zscores(cs, growth_cols)
    if "Dollar_Volume_63" in cs or "Amihud_ILLIQ_63" in cs:
        cs = robust_zscores(cs, ["Dollar_Volume_63", "Amihud_ILLIQ_63", "Spread_Proxy_63"])

    value_parts = []
    for c in ["EV_EBITDA_z", "Price_Book_z", "PE_Ratio_z", "NetDebt_EBITDA_z"]:
        if c in cs:
            value_parts.append(-cs[c])
    for c in ["FCF_Yield_z", "Earnings_Yield_z"]:
        if c in cs:
            value_parts.append(cs[c])
    cs["Value_Score"] = pd.concat(value_parts, axis=1).mean(axis=1, skipna=True) if value_parts else np.nan

    quality_parts = [cs[c] for c in ["ROIC_z", "ROE_z", "Interest_Coverage_z", "Asset_Turnover_z", "Piotroski_z", "Altman_Z_z", "Solvency_z"] if c in cs]
    cs["Quality_Score"] = pd.concat(quality_parts, axis=1).mean(axis=1, skipna=True) if quality_parts else np.nan

    growth_parts = [cs[c] for c in [f"{x}_z" for x in growth_cols] if c in cs]
    cs["Growth_Score"] = pd.concat(growth_parts, axis=1).mean(axis=1, skipna=True) if growth_parts else np.nan

    tech_parts = [cs[c] for c in ["Momentum_21_z", "Momentum_63_z", "Momentum_126_z", "Trend_50_200_z", "Max_Drawdown_126_z"] if c in cs]
    if "Volatility_63_z" in cs:
        tech_parts.append(-cs["Volatility_63_z"])
    cs["Technical_Score"] = pd.concat(tech_parts, axis=1).mean(axis=1, skipna=True) if tech_parts else np.nan
    liq_parts = []
    if "Dollar_Volume_63_z" in cs:
        liq_parts.append(cs["Dollar_Volume_63_z"])
    for c in ["Amihud_ILLIQ_63_z", "Spread_Proxy_63_z"]:
        if c in cs:
            liq_parts.append(-cs[c])
    cs["Liquidity_Score"] = pd.concat(liq_parts, axis=1).mean(axis=1, skipna=True) if liq_parts else np.nan
    cs["Anomaly_Penalty"] = -np.log1p(cs["Mahalanobis"].clip(lower=0))
    cs["Style_Value"] = cs["Value_Score"]
    cs["Style_Quality"] = cs["Quality_Score"]
    cs["Style_Growth"] = cs["Growth_Score"]
    cs["Style_Momentum"] = pd.concat([cs[c] for c in ["Momentum_63_z", "Momentum_126_z"] if c in cs], axis=1).mean(axis=1, skipna=True) if any(c in cs for c in ["Momentum_63_z", "Momentum_126_z"]) else np.nan
    cs["Style_LowVol"] = -cs["Volatility_63_z"] if "Volatility_63_z" in cs else np.nan
    cs["Style_Liquidity"] = cs["Liquidity_Score"]
    if "Market_Cap" in cs and cs["Market_Cap"].notna().sum() > 0:
        cs["Log_Market_Cap"] = np.log(cs["Market_Cap"].clip(lower=1.0))
        cs = robust_zscores(cs, ["Log_Market_Cap"])
        cs["Style_Size"] = -cs["Log_Market_Cap_z"]
    else:
        cs["Style_Size"] = np.nan
    style_cols = ["Style_Value", "Style_Quality", "Style_Growth", "Style_Momentum", "Style_LowVol", "Style_Size", "Style_Liquidity"]
    cs["Style_Composite"] = cs[[c for c in style_cols if c in cs]].mean(axis=1, skipna=True)

    hawkish = macro_row.get("Regime_Hawkish_Dovish", "Dovish")
    bullbear = macro_row.get("Regime_Bull_Bear", "Bull")
    if hawkish == "Hawkish" and bullbear == "Bear":
        wv, wq, wt, wl, wa = 0.13, 0.42, 0.23, 0.07, 0.15
    elif hawkish == "Hawkish":
        wv, wq, wt, wl, wa = 0.18, 0.33, 0.28, 0.06, 0.15
    elif bullbear == "Bear":
        wv, wq, wt, wl, wa = 0.18, 0.33, 0.28, 0.06, 0.15
    else:
        wv, wq, wt, wl, wa = 0.23, 0.24, 0.33, 0.05, 0.15
    cs["Composite_Score"] = (
        wv * cs["Value_Score"].fillna(0)
        + wq * cs["Quality_Score"].fillna(0)
        + wt * cs["Technical_Score"].fillna(0)
        + wl * cs["Liquidity_Score"].fillna(0)
        + wa * cs["Anomaly_Penalty"].fillna(0)
    )
    cs["Regime_Hawkish_Dovish"] = hawkish
    cs["Regime_Bull_Bear"] = bullbear
    for col in ["Markov_Next_State_Mode", "Markov_State_Persistence", "Markov_Stress_Prob", "Markov_Risk_On_Prob", "Markov_Transition_Entropy"]:
        if col in macro_row:
            cs[col] = macro_row.get(col)
    cs = add_post_phd_alpha_layer(
        cs,
        prices,
        macro_row,
        asof_date,
        use_garch=use_garch,
        garch_candidate_n=garch_candidate_n,
        crlb_penalty=crlb_penalty,
        garch_penalty=garch_penalty,
        evt_penalty=evt_penalty,
    )
    if "TextRisk_Score" in cs:
        tr = pd.to_numeric(cs["TextRisk_Score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        scale = tr[tr > 0].median()
        if not pd.notna(scale) or scale <= 0:
            scale = 1.0
        cs["TextRisk_Penalty"] = text_risk_penalty * (tr.fillna(0.0).clip(lower=0.0) / scale).clip(0.0, 5.0)
        cs["Composite_Score_Before_TextRisk"] = cs["Composite_Score"]
        cs["Composite_Score"] = cs["Composite_Score"] - cs["TextRisk_Penalty"].fillna(0.0)
    else:
        cs["TextRisk_Penalty"] = 0.0
    if "PIT_Confidence" in cs:
        pit = pd.to_numeric(cs["PIT_Confidence"], errors="coerce").fillna(0.25).clip(0.0, 1.0)
        cs["PIT_Confidence_Penalty"] = 0.10 * (1.0 - pit)
        cs["Composite_Score_Before_PIT_Confidence"] = cs["Composite_Score"]
        cs["Composite_Score"] = cs["Composite_Score"] - cs["PIT_Confidence_Penalty"]
    else:
        cs["PIT_Confidence_Penalty"] = 0.0
    return apply_fundamental_gate(cs, macro_row, min_dollar_volume=min_dollar_volume).sort_values("Composite_Score", ascending=False).reset_index(drop=True)


def apply_fundamental_gate(
    cs: pd.DataFrame,
    macro_row: pd.Series,
    min_valid: int = 6,
    min_dollar_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    out = cs.copy()
    hawkish = macro_row.get("Regime_Hawkish_Dovish", "Dovish")
    bear = macro_row.get("Regime_Bull_Bear", "Bull") == "Bear"
    out["Fundamental_Gate"] = out["Valid_Fundamental_Ratios"].fillna(0) >= min_valid
    out["Reject_Reasons"] = ""
    out.loc[~out["Fundamental_Gate"], "Reject_Reasons"] += "ratio_coverage;"
    if "Quality_Score" in out and out["Quality_Score"].notna().sum() >= 10:
        mask = out["Quality_Score"] >= out["Quality_Score"].quantile(0.30)
        out.loc[~mask, "Reject_Reasons"] += "quality_bottom_30pct;"
        out["Fundamental_Gate"] &= mask
    if "NetDebt_EBITDA" in out:
        mask = out["NetDebt_EBITDA"].isna() | (out["NetDebt_EBITDA"] <= (3.0 if hawkish == "Hawkish" else 4.5))
        out.loc[~mask, "Reject_Reasons"] += "leverage;"
        out["Fundamental_Gate"] &= mask
    if "Interest_Coverage" in out:
        mask = out["Interest_Coverage"].isna() | (out["Interest_Coverage"] >= (3.0 if hawkish == "Hawkish" or bear else 1.5))
        out.loc[~mask, "Reject_Reasons"] += "interest_coverage;"
        out["Fundamental_Gate"] &= mask
    if "Altman_Z" in out:
        mask = out["Altman_Z"].isna() | (out["Altman_Z"] >= (1.8 if hawkish == "Hawkish" or bear else 1.2))
        out.loc[~mask, "Reject_Reasons"] += "altman_distress;"
        out["Fundamental_Gate"] &= mask
    if "Dollar_Volume_63" in out:
        mask = out["Dollar_Volume_63"].isna() | (out["Dollar_Volume_63"] >= min_dollar_volume)
        out.loc[~mask, "Reject_Reasons"] += "liquidity_adv;"
        out["Fundamental_Gate"] &= mask
    out.loc[out["Fundamental_Gate"], "Reject_Reasons"] = ""
    return out


def rejection_diagnostics(cs: pd.DataFrame) -> pd.DataFrame:
    if cs.empty or "Fundamental_Gate" not in cs:
        return pd.DataFrame()
    rejected = cs[~cs["Fundamental_Gate"]].copy()
    if rejected.empty:
        return pd.DataFrame()
    cols = [
        "Ticker",
        "Sector",
        "Country",
        "Reject_Reasons",
        "Valid_Fundamental_Ratios",
        "Quality_Score",
        "NetDebt_EBITDA",
        "Interest_Coverage",
        "Altman_Z",
        "Dollar_Volume_63",
        "Composite_Score",
    ]
    return rejected[[c for c in cols if c in rejected.columns]].sort_values("Composite_Score", ascending=False)


def sortino_ratio(r: pd.Series, mar: float = 0.0) -> float:
    r = pd.Series(r).dropna().astype(float)
    if len(r) < 20:
        return np.nan
    downside = np.minimum(r - mar / 252.0, 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(252)
    return (r.mean() * 252 - mar) / downside_dev if downside_dev > 0 else np.nan


def sharpe_ratio(r: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(r) < 20:
        return np.nan
    vol = r.std(ddof=1) * np.sqrt(periods)
    return (r.mean() * periods - rf) / vol if vol > 0 else np.nan


def information_ratio(r: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    active = pd.Series(r).astype(float).sub(pd.Series(benchmark).astype(float), fill_value=np.nan).dropna()
    if len(active) < 20:
        return np.nan
    te = active.std(ddof=1) * np.sqrt(periods)
    return active.mean() * periods / te if te > 0 else np.nan


def beta_to_benchmark(r: pd.Series, benchmark: pd.Series) -> float:
    x = pd.concat([pd.Series(r).astype(float), pd.Series(benchmark).astype(float)], axis=1).dropna()
    if len(x) < 20:
        return np.nan
    bvar = x.iloc[:, 1].var(ddof=1)
    if bvar <= 0 or not np.isfinite(bvar):
        return np.nan
    return float(x.iloc[:, 0].cov(x.iloc[:, 1]) / bvar)


def treynor_ratio(r: pd.Series, benchmark: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    beta = beta_to_benchmark(r, benchmark)
    if pd.isna(beta) or abs(beta) <= 1e-8:
        return np.nan
    return (pd.Series(r).dropna().mean() * periods - rf) / abs(beta)


def max_drawdown(r: pd.Series) -> float:
    x = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if x.empty:
        return np.nan
    equity = (1.0 + x).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def objective_metric_name(objective: str) -> str:
    objective = str(objective or "sortino").lower()
    mapping = {
        "sortino": "Sortino",
        "sharpe": "Sharpe",
        "treynor": "Treynor",
        "information_ratio": "Information_Ratio",
        "mean_variance": "Mean_Variance_Score",
        "min_variance": "Neg_Ann_Vol",
        "cvar_min": "Neg_CVaR_95",
        "risk_parity": "Risk_Parity_Score",
        "hrp": "HRP_Score",
        "black_litterman": "Black_Litterman_Score",
        "max_return": "Ann_Return",
    }
    return mapping.get(objective, "Sortino")


def _standardize_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0.0, posinf=0.0, neginf=0.0)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 1e-12 else 1.0)


def black_litterman_posterior_alpha(
    selected_idx: pd.DataFrame,
    tickers: list[str],
    sigma: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    if not tickers:
        return pd.Series(dtype=float), pd.DataFrame()
    tau = float(np.clip(tau, 1e-4, 1.0))
    sigma = nearest_psd(np.asarray(sigma, dtype=float))
    n = len(tickers)
    caps = selected_idx.get("Market_Cap_AsOf", pd.Series(np.nan, index=tickers)).reindex(tickers).astype(float)
    if caps.notna().sum() == 0 or caps.clip(lower=0).sum() <= 0:
        w_mkt = np.ones(n) / n
        cap_source = "equal_weight_fallback"
    else:
        w_mkt = caps.clip(lower=0).fillna(0.0).values
        w_mkt = w_mkt / w_mkt.sum()
        cap_source = "market_cap"
    pi = float(max(risk_aversion, 1e-6)) * sigma @ w_mkt
    pi = _standardize_array(pi)
    if "Bayesian_Alpha_Mean" in selected_idx:
        q_raw = selected_idx["Bayesian_Alpha_Mean"].reindex(tickers).astype(float)
        view_source = "Bayesian_Alpha_Mean"
    else:
        q_raw = selected_idx.get("Composite_Score", pd.Series(0.0, index=tickers)).reindex(tickers).astype(float)
        view_source = "Composite_Score"
    q = _standardize_array(q_raw.fillna(q_raw.median() if q_raw.notna().any() else 0.0).values)
    bayes_std = selected_idx.get("Bayesian_Alpha_Std", pd.Series(1.0, index=tickers)).reindex(tickers).astype(float).fillna(1.0).values
    crlb = selected_idx.get("CRLB_Mu", pd.Series(0.0, index=tickers)).reindex(tickers).astype(float).fillna(0.0).values
    coverage = selected_idx.get("Valid_Fundamental_Ratios", pd.Series(5.0, index=tickers)).reindex(tickers).astype(float).fillna(5.0).values
    std_norm = np.square(_standardize_array(np.abs(bayes_std)) + 1.5)
    crlb_norm = np.abs(_standardize_array(crlb)) + 1.0
    coverage_penalty = 1.0 / np.clip(coverage, 1.0, None)
    omega_diag = np.clip(0.05 + 0.15 * std_norm + 0.10 * crlb_norm + coverage_penalty, 0.03, 10.0)
    omega_inv = np.diag(1.0 / omega_diag)
    tau_sigma_inv = np.linalg.pinv(tau * sigma + 1e-8 * np.eye(n))
    p = np.eye(n)
    lhs = tau_sigma_inv + p.T @ omega_inv @ p
    rhs = tau_sigma_inv @ pi + p.T @ omega_inv @ q
    posterior = np.linalg.pinv(lhs) @ rhs
    posterior = _standardize_array(posterior)
    diag = pd.DataFrame(
        {
            "Ticker": tickers,
            "BL_Prior_Equilibrium": pi,
            "BL_View": q,
            "BL_Omega_Diag": omega_diag,
            "BL_Posterior_Alpha": posterior,
            "BL_Market_Cap_Weight": w_mkt,
            "BL_View_Source": view_source,
            "BL_Cap_Source": cap_source,
            "BL_Tau": tau,
        }
    )
    return pd.Series(posterior, index=tickers), diag


def hierarchical_risk_parity_weights(cov: pd.DataFrame) -> tuple[pd.Series, dict]:
    if cov is None or cov.empty:
        return pd.Series(dtype=float), {"status": "empty_cov"}
    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tickers = list(cov.index)
    if len(tickers) == 1:
        return pd.Series(1.0, index=tickers), {"status": "single_asset", "hrp_order": tickers}
    diag = np.sqrt(np.clip(np.diag(cov.values), 1e-12, None))
    corr = cov.values / np.outer(diag, diag)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    if linkage is None or leaves_list is None or squareform is None:
        order = list(np.argsort(diag))
        status = "inverse_vol_order_fallback"
    else:
        try:
            link = linkage(squareform(dist, checks=False), method="single")
            order = list(leaves_list(link))
            status = "hrp"
        except Exception:
            order = list(np.argsort(diag))
            status = "inverse_vol_order_fallback"
    ordered = [tickers[i] for i in order]
    weights = pd.Series(1.0, index=ordered)

    def cluster_var(names: list[str]) -> float:
        sub = cov.reindex(index=names, columns=names).values
        ivp = 1.0 / np.clip(np.diag(sub), 1e-12, None)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub @ ivp)

    clusters = [ordered]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]
        left_var, right_var = cluster_var(left), cluster_var(right)
        alpha = 1.0 - left_var / max(left_var + right_var, 1e-12)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])
    weights = weights.reindex(tickers).fillna(0.0)
    weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(tickers), index=tickers)
    return weights, {"status": status, "hrp_order": ordered, "hrp_mean_distance": float(np.mean(dist[np.triu_indices_from(dist, 1)]))}


def historical_var_cvar(r: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    r = pd.Series(r).dropna().astype(float)
    if r.empty:
        return np.nan, np.nan
    losses = -r
    var = losses.quantile(alpha)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


def historical_cvar_loss(r: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    x = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if x.empty:
        return 0.0
    losses = -x
    var = losses.quantile(alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if not tail.empty else float(var)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.dropna(axis=1, how="all").fillna(0.0)
    if clean.empty:
        return pd.DataFrame()
    if clean.shape[0] < 5 or clean.shape[1] == 1:
        cov = clean.cov().values * 252
    else:
        cov = LedoitWolf().fit(clean.values).covariance_ * 252
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def nearest_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    a = np.asarray(matrix, dtype=float)
    a = np.nan_to_num((a + a.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigval, eigvec = np.linalg.eigh(a)
        eigval = np.clip(eigval, eps, None)
        out = eigvec @ np.diag(eigval) @ eigvec.T
        return np.nan_to_num((out + out.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        diag = np.clip(np.diag(a), eps, None)
        return np.diag(diag)


def factor_model_covariance_matrix(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if returns.empty or factors.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    idx = returns.index.intersection(factors.index)
    y = returns.reindex(idx).dropna(axis=1, how="all").fillna(0.0)
    x = factors.reindex(idx).dropna(axis=1, how="all").fillna(0.0)
    if len(idx) < 60 or y.empty or x.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    x_mat = np.column_stack([np.ones(len(x)), x.values])
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    beta_mat = xtx_inv @ x_mat.T @ y.values
    loadings = pd.DataFrame(beta_mat[1:, :].T, index=y.columns, columns=x.columns)
    fitted = x_mat @ beta_mat
    resid = pd.DataFrame(y.values - fitted, index=y.index, columns=y.columns)
    factor_cov = ledoit_wolf_cov(x)
    resid_var = resid.var(ddof=1).clip(lower=1e-10) * 252.0
    b = loadings.reindex(index=y.columns, columns=factor_cov.columns).fillna(0.0).values
    sigma = b @ factor_cov.values @ b.T + np.diag(resid_var.reindex(y.columns).fillna(resid_var.median()).values)
    sigma = nearest_psd(sigma)
    cov = pd.DataFrame(sigma, index=y.columns, columns=y.columns)
    return cov, loadings, resid_var


def construct_constrained_weights(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    asof_date,
    lookback: int,
    macro: pd.DataFrame | None = None,
    benchmark_ticker: str | None = None,
    max_weight: float = 0.25,
    sector_weight_cap: float = 0.40,
    risk_aversion: float = 4.0,
    alpha_weight: float = 1.0,
    objective: str = "sortino",
    entropy_penalty: float = 0.05,
    crlb_penalty: float = 0.15,
    garch_penalty: float = 0.10,
    evt_penalty: float = 0.10,
    cvar_penalty: float = 0.25,
    cvar_alpha: float = 0.95,
    robust_alpha_uncertainty: float = 0.35,
    robust_cov_uncertainty: float = 0.10,
    factor_cov_blend: float = 0.50,
    use_black_litterman: bool = False,
    black_litterman_tau: float = 0.05,
    portfolio_notional: float = 100_000.0,
    max_adv_participation: float = 0.05,
    target_vol: float | None = None,
    factor_caps: dict[str, float] | None = None,
    multistarts: int = 8,
) -> tuple[pd.Series, dict]:
    objective = str(objective or "sortino").lower()
    tickers = [t for t in selected["Ticker"].tolist() if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float), {"status": "no_tickers"}

    ret = prices.loc[:asof_date, tickers].tail(lookback + 1).pct_change(fill_method=None).dropna(how="all")
    if ret.empty:
        w = pd.Series(1.0 / len(tickers), index=tickers)
        return w, {"status": "equal_weight_no_returns"}

    ret_clean = ret.reindex(columns=tickers).fillna(0.0)
    cov = ledoit_wolf_cov(ret_clean)
    if cov.empty:
        w = pd.Series(1.0 / len(tickers), index=tickers)
        return w, {"status": "equal_weight_no_cov"}

    mu = selected.set_index("Ticker").reindex(tickers)["Composite_Score"].fillna(0.0)
    mu = (mu - mu.mean()) / (mu.std(ddof=0) if mu.std(ddof=0) > 0 else 1.0)
    selected_idx = selected.set_index("Ticker").reindex(tickers)
    crlb_vec = selected_idx.get("CRLB_Mu", pd.Series(0.0, index=tickers)).fillna(0.0).astype(float).values
    garch_vec = selected_idx.get("GARCH_Vol_Forecast", pd.Series(0.0, index=tickers)).fillna(0.0).astype(float).values
    evt_vec = selected_idx.get("EVT_CVaR_95", pd.Series(0.0, index=tickers)).fillna(0.0).astype(float).values
    crlb_scale = np.nanmedian(crlb_vec[crlb_vec > 0]) if np.any(crlb_vec > 0) else 1.0
    garch_scale = np.nanmedian(garch_vec[garch_vec > 0]) if np.any(garch_vec > 0) else 1.0
    evt_scale = np.nanmedian(evt_vec[evt_vec > 0]) if np.any(evt_vec > 0) else 1.0
    factor_betas = pd.DataFrame()
    factor_cov_used = 0.0
    if macro is not None and factor_cov_blend > 0:
        factors = build_factor_returns(prices, macro)
        fac_idx = ret_clean.index.intersection(factors.index) if not factors.empty else pd.Index([])
        if len(fac_idx) >= 60:
            fac_cov, factor_betas_tmp, _ = factor_model_covariance_matrix(
                ret_clean.reindex(fac_idx),
                factors.reindex(fac_idx),
            )
            if not fac_cov.empty:
                blend = float(np.clip(factor_cov_blend, 0.0, 1.0))
                cov = (1.0 - blend) * cov.reindex(index=tickers, columns=tickers).fillna(0.0) + blend * fac_cov.reindex(index=tickers, columns=tickers).fillna(0.0)
                factor_betas = factor_betas_tmp
                factor_cov_used = blend
    sigma = cov.reindex(index=tickers, columns=tickers).fillna(0.0).values
    n = len(tickers)
    bl_diag = pd.DataFrame()
    if use_black_litterman or objective == "black_litterman":
        bl_mu, bl_diag = black_litterman_posterior_alpha(
            selected_idx,
            tickers,
            sigma,
            risk_aversion=max(risk_aversion, 1e-6),
            tau=black_litterman_tau,
        )
        if not bl_mu.empty:
            mu = bl_mu.reindex(tickers).fillna(0.0)
    if benchmark_ticker and benchmark_ticker in prices.columns:
        bench_ret = prices.loc[:asof_date, benchmark_ticker].tail(lookback + 1).pct_change(fill_method=None).dropna()
        bench_ret = bench_ret.reindex(ret_clean.index).fillna(ret_clean.mean(axis=1))
    else:
        bench_ret = ret_clean.mean(axis=1)
    max_weight = max(1.0 / n, min(max_weight, 1.0))
    adv = selected_idx.get("Dollar_Volume_63", pd.Series(np.nan, index=tickers)).reindex(tickers).astype(float)
    adv_caps = (adv * max_adv_participation / max(float(portfolio_notional), 1.0)).replace([np.inf, -np.inf], np.nan)
    adv_caps = adv_caps.fillna(max_weight).clip(lower=0.0, upper=max_weight)
    if adv_caps.sum() < 1.0:
        adv_caps = pd.Series(max_weight, index=tickers)
        adv_binding = False
    else:
        adv_binding = bool((adv_caps < max_weight - 1e-12).any())
    bounds = [(0.0, float(adv_caps.loc[tk])) for tk in tickers]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    sectors = selected.set_index("Ticker").reindex(tickers)["Sector"].fillna("Unknown")
    for sector in sectors.unique():
        idx = np.where(sectors.values == sector)[0]
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx: sector_weight_cap - np.sum(w[idx])})

    if macro is not None and factor_caps:
        if factor_betas.empty:
            factor_betas = estimate_asset_factor_betas(prices, macro, tickers, asof_date, lookback)
        for factor, cap in factor_caps.items():
            if factor in factor_betas.columns and pd.notna(cap):
                b = factor_betas[factor].reindex(tickers).fillna(0.0).values.astype(float)
                constraints.append({"type": "ineq", "fun": lambda w, b=b, cap=cap: cap - float(w @ b)})
                constraints.append({"type": "ineq", "fun": lambda w, b=b, cap=cap: cap + float(w @ b)})

    def robust_penalties(w, port_ret, ann_var):
        alpha_uncertainty = float(np.sqrt(max(w @ np.diag(np.nan_to_num(crlb_vec / max(crlb_scale, 1e-12), nan=0.0)) @ w, 0.0)))
        cov_uncertainty = float(np.sum(np.square(w) * np.diag(sigma)))
        entropy = normalized_weight_entropy(pd.Series(w))
        crlb_port_penalty = float(w @ (crlb_vec / max(crlb_scale, 1e-12)))
        garch_port_penalty = float(w @ (garch_vec / max(garch_scale, 1e-12)))
        evt_port_penalty = float(w @ (evt_vec / max(evt_scale, 1e-12)))
        cvar_loss = historical_cvar_loss(port_ret, alpha=cvar_alpha)
        vol_penalty = 0.0
        if target_vol is not None and target_vol > 0:
            vol_penalty = max(np.sqrt(max(ann_var, 0.0)) - target_vol, 0.0) ** 2
        return (
            entropy_penalty * (1.0 - entropy)
            + crlb_penalty * crlb_port_penalty
            + garch_penalty * garch_port_penalty
            + evt_penalty * evt_port_penalty
            + robust_alpha_uncertainty * alpha_uncertainty
            + robust_cov_uncertainty * cov_uncertainty
            + cvar_penalty * cvar_loss
            + vol_penalty
        )

    def risk_parity_loss(w):
        port_var = float(w @ sigma @ w)
        if port_var <= 1e-12 or not np.isfinite(port_var):
            return 1e6
        marginal = sigma @ w
        risk_contrib = w * marginal / port_var
        target = np.ones_like(risk_contrib) / len(risk_contrib)
        return float(np.sum((risk_contrib - target) ** 2))

    def mv_objective(w):
        ann_var = float(w @ sigma @ w)
        alpha = float(w @ mu.values)
        port_ret = ret_clean.values @ w
        vol_penalty = 0.0
        if target_vol is not None and target_vol > 0:
            vol_penalty = 25.0 * max(np.sqrt(max(ann_var, 0.0)) - target_vol, 0.0) ** 2
        return risk_aversion * ann_var - alpha_weight * alpha + robust_penalties(w, port_ret, ann_var) + vol_penalty

    def utility_objective(w):
        port_ret = ret_clean.values @ w
        ann_var = float(w @ sigma @ w)
        ann_vol = float(np.sqrt(max(ann_var, 0.0)))
        ann_return = float(np.mean(port_ret) * 252.0)
        mean_excess = float(np.mean(port_ret) * 252.0)
        downside = np.minimum(port_ret, 0.0)
        downside_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252.0))
        sortino = mean_excess / downside_dev if downside_dev > 1e-10 and np.isfinite(downside_dev) else -1e6
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 and np.isfinite(ann_vol) else -1e6
        bench = bench_ret.reindex(ret_clean.index).fillna(0.0).values
        active = port_ret - bench
        tracking_error = float(np.std(active, ddof=1) * np.sqrt(252.0)) if len(active) > 2 else np.nan
        info = float(np.mean(active) * 252.0 / tracking_error) if pd.notna(tracking_error) and tracking_error > 1e-10 else -1e6
        beta = beta_to_benchmark(pd.Series(port_ret, index=ret_clean.index), bench_ret)
        treynor = ann_return / abs(beta) if pd.notna(beta) and abs(beta) > 1e-8 else -1e6
        cvar_loss_raw = historical_cvar_loss(port_ret, alpha=cvar_alpha)
        alpha_tilt = float(w @ mu.values)
        concentration_penalty = 0.01 * float(np.sum(np.square(w)))
        penalty = robust_penalties(w, port_ret, ann_var) + concentration_penalty
        if objective == "sortino":
            return -(sortino + 0.05 * alpha_weight * alpha_tilt) + penalty
        if objective == "sharpe":
            return -(sharpe + 0.05 * alpha_weight * alpha_tilt) + penalty
        if objective == "treynor":
            return -(treynor + 0.03 * alpha_weight * alpha_tilt) + penalty
        if objective == "information_ratio":
            return -(info + 0.03 * alpha_weight * alpha_tilt) + penalty
        if objective == "min_variance":
            return ann_var + penalty
        if objective == "cvar_min":
            return cvar_loss_raw + 0.10 * ann_var + penalty
        if objective == "risk_parity":
            return risk_parity_loss(w) + 0.10 * ann_var + penalty
        if objective == "hrp":
            return risk_parity_loss(w) + 0.05 * ann_var + penalty
        if objective == "black_litterman":
            return mv_objective(w)
        if objective == "max_return":
            return -ann_return + risk_aversion * ann_var + penalty
        return mv_objective(w)

    def random_feasible_start(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.random(n)
        x = x / x.sum()
        x = np.minimum(x, max_weight)
        if x.sum() == 0:
            x = np.ones(n) / n
        else:
            x = x / x.sum()
        # Project roughly through sector caps by shrinking crowded sectors.
        for _ in range(5):
            adjusted = False
            for sector in sectors.unique():
                idx = np.where(sectors.values == sector)[0]
                s = x[idx].sum()
                if s > sector_weight_cap:
                    x[idx] *= sector_weight_cap / s
                    adjusted = True
            if adjusted:
                deficit = 1.0 - x.sum()
                room = np.maximum(max_weight - x, 0.0)
                if room.sum() > 0:
                    x += deficit * room / room.sum()
            else:
                break
        return x / x.sum()

    starts = [np.ones(n) / n]
    if objective == "hrp":
        hrp_start, hrp_meta = hierarchical_risk_parity_weights(cov.reindex(index=tickers, columns=tickers))
        if not hrp_start.empty:
            starts[0] = hrp_start.reindex(tickers).fillna(0.0).values
            if starts[0].sum() > 0:
                starts[0] = starts[0] / starts[0].sum()
    else:
        hrp_meta = {}
    for seed in range(max(0, multistarts - 1)):
        starts.append(random_feasible_start(10_000 + seed))
    obj = mv_objective if objective == "mean_variance" else utility_objective
    results = []
    for x0 in starts:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 700, "ftol": 1e-9})
        if res.success and np.all(np.isfinite(res.x)):
            results.append(res)
    result = min(results, key=lambda r: r.fun) if results else minimize(obj, starts[0], method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 700, "ftol": 1e-9})
    if not result.success or np.any(~np.isfinite(result.x)):
        if objective in {"sortino", "sharpe", "treynor", "information_ratio", "max_return", "cvar_min", "risk_parity", "hrp", "black_litterman"}:
            result_mv = minimize(mv_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-9})
            if result_mv.success and np.all(np.isfinite(result_mv.x)):
                w = result_mv.x
                status = f"mean_variance_fallback_after_{objective}:{result.message}"
            else:
                inv_vol = 1.0 / np.sqrt(np.clip(np.diag(sigma), 1e-10, None))
                w = inv_vol / inv_vol.sum()
                w = np.minimum(w, max_weight)
                w = w / w.sum()
                status = f"inverse_vol_fallback:{result.message}"
        else:
            inv_vol = 1.0 / np.sqrt(np.clip(np.diag(sigma), 1e-10, None))
            w = inv_vol / inv_vol.sum()
            w = np.minimum(w, max_weight)
            w = w / w.sum()
            status = f"inverse_vol_fallback:{result.message}"
    else:
        w = result.x
        status = f"{objective}_optimized"

    weights = pd.Series(w, index=tickers)
    port_ret = ret_clean @ weights
    bench_for_metrics = bench_ret.reindex(port_ret.index).fillna(0.0)
    var95, cvar95 = historical_var_cvar(port_ret, 0.95)
    meta = {
        "status": status,
        "weight_objective": objective,
        "multistarts": multistarts,
        "realized_weight_sortino": sortino_ratio(port_ret),
        "realized_weight_sharpe": sharpe_ratio(port_ret),
        "realized_weight_information": information_ratio(port_ret, bench_for_metrics),
        "realized_weight_treynor": treynor_ratio(port_ret, bench_for_metrics),
        "realized_weight_beta_to_benchmark": beta_to_benchmark(port_ret, bench_for_metrics),
        "realized_weight_beta_to_internal_benchmark": beta_to_benchmark(port_ret, bench_for_metrics),
        "realized_weight_ann_return": float(port_ret.mean() * 252.0),
        "realized_weight_ann_vol": float(port_ret.std(ddof=1) * np.sqrt(252.0)),
        "realized_weight_max_drawdown": max_drawdown(port_ret),
        "weight_entropy": normalized_weight_entropy(weights),
        "portfolio_crlb_mu": float(weights.values @ crlb_vec),
        "portfolio_garch_vol": float(weights.values @ garch_vec),
        "portfolio_evt_cvar_95": float(weights.values @ evt_vec),
        "portfolio_optimized_cvar_alpha": cvar_alpha,
        "portfolio_optimized_hist_cvar_loss": historical_cvar_loss(port_ret, alpha=cvar_alpha),
        "cvar_penalty": cvar_penalty,
        "robust_alpha_uncertainty": robust_alpha_uncertainty,
        "robust_cov_uncertainty": robust_cov_uncertainty,
        "factor_cov_blend": factor_cov_used,
        "black_litterman_used": bool(use_black_litterman or objective == "black_litterman"),
        "black_litterman_tau": black_litterman_tau,
        "black_litterman_view_count": int(len(bl_diag)) if not bl_diag.empty else 0,
        "hrp_status": hrp_meta.get("status"),
        "hrp_mean_distance": hrp_meta.get("hrp_mean_distance"),
        "portfolio_notional": portfolio_notional,
        "max_adv_participation": max_adv_participation,
        "adv_participation_binding": adv_binding,
        "min_adv_weight_cap": float(adv_caps.min()) if len(adv_caps) else np.nan,
        "portfolio_alpha_uncertainty_norm": float(np.sqrt(max(weights.values @ np.diag(np.nan_to_num(crlb_vec / max(crlb_scale, 1e-12), nan=0.0)) @ weights.values, 0.0))),
        "portfolio_cov_uncertainty_norm": float(np.sum(np.square(weights.values) * np.diag(sigma))),
        "ex_ante_vol": float(np.sqrt(max(weights.values @ sigma @ weights.values, 0.0))),
        "hist_var_95_daily": var95,
        "hist_cvar_95_daily": cvar95,
        "weight_hhi": float(np.sum(np.square(weights.values))),
        "max_weight_actual": float(weights.max()),
    }
    if not factor_betas.empty:
        for factor in factor_betas.columns:
            meta[f"factor_exposure_{factor}"] = float(weights.reindex(factor_betas.index).fillna(0.0).values @ factor_betas[factor].reindex(weights.index).fillna(0.0).values)
    return weights, meta


def estimate_transaction_cost(
    weights: pd.Series,
    prev_weights: pd.Series,
    prices: pd.DataFrame,
    asof_date,
    fixed_tc_bps: float,
    impact_coefficient: float = 0.10,
) -> tuple[float, dict]:
    all_names = weights.index.union(prev_weights.index)
    delta = weights.reindex(all_names).fillna(0.0) - prev_weights.reindex(all_names).fillna(0.0)
    turnover = 0.5 * delta.abs().sum()
    fixed = turnover * fixed_tc_bps / 10000.0

    ret = prices.loc[:asof_date, [c for c in all_names if c in prices.columns]].tail(64).pct_change(fill_method=None)
    vol63 = ret.std().reindex(all_names).fillna(ret.stack().std() if not ret.empty else 0.02)
    impact = float((delta.abs() * impact_coefficient * vol63 * np.sqrt(delta.abs().clip(lower=0.0))).sum())
    total = float(fixed + impact)
    return total, {"turnover": float(turnover), "fixed_tc": float(fixed), "impact_tc": impact}


def bootstrap_chunk_stability(
    candidates: pd.DataFrame,
    ret: pd.DataFrame,
    min_chunk: int,
    max_chunk: int,
    max_names_per_sector: int,
    n_bootstrap: int = 64,
) -> pd.Series:
    tickers = [t for t in candidates["Ticker"] if t in ret.columns]
    if len(tickers) < min_chunk or ret.empty or n_bootstrap <= 0:
        return pd.Series(0.0, index=candidates["Ticker"])
    sectors = candidates.set_index("Ticker")["Sector"].to_dict()
    rng = np.random.default_rng(20260506)
    counts = pd.Series(0.0, index=tickers)
    values = ret[tickers].fillna(0.0)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(values), size=len(values))
        sampled = values.iloc[sample_idx]
        best_combo, best_score = None, -np.inf
        for k in range(min_chunk, min(max_chunk, len(tickers)) + 1):
            for combo in combinations(tickers, k):
                sec_counts = {}
                ok = True
                for tk in combo:
                    sec = sectors.get(tk, "Unknown")
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1
                    if sec_counts[sec] > max_names_per_sector:
                        ok = False
                        break
                if not ok:
                    continue
                score = sortino_ratio(sampled.loc[:, list(combo)].mean(axis=1))
                if pd.notna(score) and score > best_score:
                    best_score, best_combo = score, combo
        if best_combo:
            counts.loc[list(best_combo)] += 1.0
    return counts / max(n_bootstrap, 1)


def optimize_chunks(
    cs: pd.DataFrame,
    prices: pd.DataFrame,
    asof_date,
    macro: pd.DataFrame | None = None,
    benchmark_ticker: str | None = None,
    lookback: int = 126,
    min_chunk: int = 5,
    max_chunk: int = 10,
    preselect_n: int = 14,
    max_combos: int = 25_000,
    max_names_per_sector: int = 3,
    max_weight: float = 0.25,
    sector_weight_cap: float = 0.40,
    risk_aversion: float = 4.0,
    alpha_weight: float = 1.0,
    weight_objective: str = "sortino",
    entropy_penalty: float = 0.05,
    crlb_penalty: float = 0.15,
    garch_penalty: float = 0.10,
    evt_penalty: float = 0.10,
    cvar_penalty: float = 0.25,
    cvar_alpha: float = 0.95,
    robust_alpha_uncertainty: float = 0.35,
    robust_cov_uncertainty: float = 0.10,
    factor_cov_blend: float = 0.50,
    use_black_litterman: bool = False,
    black_litterman_tau: float = 0.05,
    portfolio_notional: float = 100_000.0,
    max_adv_participation: float = 0.05,
    target_vol: float | None = None,
    nested_validation_fraction: float = 0.35,
    purge_days: int = 5,
    bootstrap_samples: int = 64,
    factor_caps: dict[str, float] | None = None,
    multistarts: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = cs[cs["Fundamental_Gate"]].copy()
    if len(candidates) < min_chunk:
        candidates = cs.head(max(min_chunk, preselect_n)).copy()
    candidates = candidates[candidates["Ticker"].isin(prices.columns)].sort_values("Composite_Score", ascending=False)
    if len(candidates) < min_chunk:
        return pd.DataFrame(), pd.DataFrame()

    n_eff = min(preselect_n, len(candidates))
    while n_eff > min_chunk and sum(math.comb(n_eff, k) for k in range(min_chunk, min(max_chunk, n_eff) + 1)) > max_combos:
        n_eff -= 1
    candidates = candidates.head(n_eff)
    tickers = candidates["Ticker"].tolist()
    sectors = candidates.set_index("Ticker")["Sector"].to_dict()
    px_window = prices.loc[:asof_date, tickers].tail(lookback + 1)
    opt_start = px_window.index.min() if not px_window.empty else pd.NaT
    opt_end = px_window.index.max() if not px_window.empty else pd.NaT
    ret = px_window.pct_change(fill_method=None).dropna(how="all")
    if len(ret) < 40:
        return pd.DataFrame(), pd.DataFrame()

    val_n = int(np.floor(len(ret) * nested_validation_fraction))
    val_n = min(max(val_n, 20), max(len(ret) - 25, 20))
    train_end = max(len(ret) - val_n - purge_days, 10)
    train_ret = ret.iloc[:train_end]
    val_ret = ret.iloc[-val_n:]
    if train_ret.empty or val_ret.empty:
        train_ret = ret
        val_ret = ret
    chunk_objective_col = objective_metric_name(weight_objective)
    if benchmark_ticker and benchmark_ticker in prices.columns:
        bench_window = prices.loc[:asof_date, benchmark_ticker].tail(lookback + 1).pct_change(fill_method=None).dropna()
        val_benchmark = bench_window.reindex(val_ret.index).fillna(val_ret.reindex(columns=tickers).mean(axis=1))
    else:
        val_benchmark = val_ret.reindex(columns=tickers).mean(axis=1)

    records = []
    for k in range(min_chunk, min(max_chunk, len(tickers)) + 1):
        for combo in combinations(tickers, k):
            counts = {}
            ok = True
            for tk in combo:
                sec = sectors.get(tk, "Unknown")
                counts[sec] = counts.get(sec, 0) + 1
                if counts[sec] > max_names_per_sector:
                    ok = False
                    break
            if not ok:
                continue
            pr_train = train_ret.loc[:, list(combo)].mean(axis=1)
            pr_val = val_ret.loc[:, list(combo)].mean(axis=1)
            pr_full = ret.loc[:, list(combo)].mean(axis=1)
            eq = (1 + pr_val).cumprod()
            ann_return = pr_val.mean() * 252
            ann_vol = pr_val.std(ddof=1) * np.sqrt(252)
            var95, cvar95 = historical_var_cvar(pr_val, 0.95)
            active = pr_val.sub(val_benchmark, fill_value=np.nan).dropna()
            tracking_error = active.std(ddof=1) * np.sqrt(252) if len(active) > 2 else np.nan
            info = active.mean() * 252 / tracking_error if pd.notna(tracking_error) and tracking_error > 0 else np.nan
            treynor = treynor_ratio(pr_val, val_benchmark)
            mean_variance_score = ann_return - risk_aversion * (ann_vol ** 2 if pd.notna(ann_vol) else np.nan)
            combo_vols = val_ret.loc[:, list(combo)].std(ddof=1) * np.sqrt(252)
            risk_parity_score = -float(combo_vols.std(ddof=0)) if combo_vols.notna().any() else np.nan
            hrp_score = risk_parity_score - 0.05 * (ann_vol if pd.notna(ann_vol) else 0.0)
            bl_score = mean_variance_score + 0.25 * sortino_ratio(pr_val)
            records.append(
                {
                    "Tickers": ",".join(combo),
                    "N": k,
                    "Sortino": sortino_ratio(pr_val),
                    "Sharpe": sharpe_ratio(pr_val),
                    "Information_Ratio": info,
                    "Treynor": treynor,
                    "Mean_Variance_Score": mean_variance_score,
                    "Neg_Ann_Vol": -ann_vol if pd.notna(ann_vol) else np.nan,
                    "VaR_95": var95,
                    "CVaR_95": cvar95,
                    "Neg_CVaR_95": -cvar95 if pd.notna(cvar95) else np.nan,
                    "Risk_Parity_Score": risk_parity_score,
                    "HRP_Score": hrp_score,
                    "Black_Litterman_Score": bl_score,
                    "Train_Sortino": sortino_ratio(pr_train),
                    "Validation_Sortino": sortino_ratio(pr_val),
                    "Full_Window_Sortino": sortino_ratio(pr_full),
                    "Ann_Return": ann_return,
                    "Ann_Vol": ann_vol,
                    "Max_Drawdown": (eq / eq.cummax() - 1).min(),
                    "Chunk_Objective": weight_objective,
                    "Chunk_Objective_Metric": chunk_objective_col,
                    "Opt_Start": opt_start,
                    "Opt_End": opt_end,
                    "Train_End": train_ret.index.max(),
                    "Validation_Start": val_ret.index.min(),
                    "Purged_Days": purge_days,
                }
            )
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    options = pd.DataFrame(records)
    if chunk_objective_col not in options.columns:
        return pd.DataFrame(), options
    options = options.dropna(subset=[chunk_objective_col]).sort_values(chunk_objective_col, ascending=False)
    if options.empty:
        return pd.DataFrame(), options
    best = options.iloc[0]["Tickers"].split(",")
    portfolio = candidates[candidates["Ticker"].isin(best)].copy()
    stability = bootstrap_chunk_stability(candidates, val_ret, min_chunk, max_chunk, max_names_per_sector, bootstrap_samples)
    weights, risk_meta = construct_constrained_weights(
        portfolio,
        prices,
        asof_date,
        lookback,
        macro=macro,
        benchmark_ticker=benchmark_ticker,
        max_weight=max_weight,
        sector_weight_cap=sector_weight_cap,
        risk_aversion=risk_aversion,
        alpha_weight=alpha_weight,
        objective=weight_objective,
        entropy_penalty=entropy_penalty,
        crlb_penalty=crlb_penalty,
        garch_penalty=garch_penalty,
        evt_penalty=evt_penalty,
        cvar_penalty=cvar_penalty,
        cvar_alpha=cvar_alpha,
        robust_alpha_uncertainty=robust_alpha_uncertainty,
        robust_cov_uncertainty=robust_cov_uncertainty,
        use_black_litterman=use_black_litterman,
        black_litterman_tau=black_litterman_tau,
        factor_caps=factor_caps,
        factor_cov_blend=factor_cov_blend,
        portfolio_notional=portfolio_notional,
        max_adv_participation=max_adv_participation,
        multistarts=multistarts,
        target_vol=target_vol,
    )
    portfolio["Weight"] = portfolio["Ticker"].map(weights).fillna(0.0)
    portfolio["Optimization_Sortino"] = options.iloc[0]["Sortino"]
    portfolio["Optimization_Objective"] = weight_objective
    portfolio["Optimization_Objective_Metric"] = chunk_objective_col
    portfolio["Optimization_Objective_Value"] = options.iloc[0][chunk_objective_col]
    portfolio["Bootstrap_Stability"] = portfolio["Ticker"].map(stability).fillna(0.0)
    for key, value in risk_meta.items():
        portfolio[key] = value
    return portfolio.sort_values("Composite_Score", ascending=False), options


def backtest(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    volumes: pd.DataFrame,
    macro: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    px = prices.sort_index().ffill()
    rebal_dates = px.resample(config.rebalance_freq).last().index
    rebal_dates = [px.index[px.index.get_indexer([d], method="ffill")[0]] for d in rebal_dates if px.index.get_indexer([d], method="ffill")[0] >= 0]
    perf, holdings, opt_rows = [], [], []
    prev_w = pd.Series(dtype=float)
    cached_best_port = pd.DataFrame()
    cached_best_options = pd.DataFrame()
    cached_best_key = None
    cached_best_selection_score = np.nan
    last_reoptimization_key = None

    def reoptimization_key(date) -> tuple | int:
        freq = str(getattr(config, "reoptimization_freq", "YE")).upper()
        ts = pd.Timestamp(date)
        if freq in {"ME", "M", "MONTHLY"}:
            return (ts.year, ts.month)
        if freq in {"2QE", "2Q", "6M", "6ME", "SEMIANNUAL", "SEMESTRAL"}:
            return (ts.year, 1 if ts.quarter <= 2 else 2)
        return ts.year

    factor_caps = {
        "MKT": config.factor_mkt_cap,
        "RATES_10Y_D": config.factor_rates_cap,
        "CREDIT_D": config.factor_credit_cap,
        "OIL_R": config.factor_oil_cap,
        "USD_R": config.factor_usd_cap,
    }

    for i in range(len(rebal_dates) - 1):
        exec_date, next_date = rebal_dates[i], rebal_dates[i + 1]
        exec_pos = px.index.get_indexer([exec_date], method="ffill")[0]
        signal_date = px.index[max(0, exec_pos - config.embargo_days)]
        macro_hist = macro.loc[:signal_date].dropna(subset=["hawkish_score", "bullish_score"])
        macro_row = macro_hist.iloc[-1] if not macro_hist.empty else macro.loc[:signal_date].iloc[-1]
        cs = score_cross_section(
            panel,
            px,
            macro_row,
            signal_date,
            volumes=volumes,
            use_garch=config.use_garch,
            garch_candidate_n=config.garch_candidate_n,
            crlb_penalty=config.crlb_penalty,
            garch_penalty=config.garch_penalty,
            evt_penalty=config.evt_penalty,
            text_risk_penalty=config.text_risk_penalty,
            min_dollar_volume=config.min_dollar_volume,
        )
        if cs.empty:
            continue
        prior_opt_grid = pd.DataFrame(opt_rows)
        prior_persistence = trial_persistence_table(prior_opt_grid, min_obs=config.robust_selection_min_obs)
        persistence_map = prior_persistence.set_index("Trial_Key")["Persistence_Score"].to_dict() if not prior_persistence.empty else {}
        model_confidence, confidence_meta = model_confidence_from_history(
            prior_opt_grid,
            window=config.model_confidence_window,
            min_confidence=config.model_confidence_min,
        )
        effective_alpha_weight = config.alpha_weight * model_confidence
        effective_risk_aversion = config.risk_aversion / max(model_confidence, 1e-6)
        current_reoptimization_key = reoptimization_key(exec_date)
        reoptimized = cached_best_port.empty or current_reoptimization_key != last_reoptimization_key
        if reoptimized:
            best_port, best_options, best_key = pd.DataFrame(), pd.DataFrame(), None
            best_selection_score = -np.inf
            for lb in config.lookback_grid:
                for size in config.chunk_size_grid:
                    port, opts = optimize_chunks(
                        cs,
                        px,
                        signal_date,
                        macro=macro,
                        benchmark_ticker=config.benchmark_ticker,
                        lookback=lb,
                        min_chunk=min(size, config.top_n),
                        max_chunk=min(size, config.top_n),
                        preselect_n=config.preselect_n,
                        max_combos=config.max_combos,
                        max_names_per_sector=config.max_names_per_sector,
                        max_weight=config.max_weight,
                        sector_weight_cap=config.sector_weight_cap,
                        risk_aversion=effective_risk_aversion,
                        alpha_weight=effective_alpha_weight,
                        weight_objective=config.weight_objective,
                        entropy_penalty=config.entropy_penalty,
                        crlb_penalty=config.crlb_penalty,
                        garch_penalty=config.garch_penalty,
                        evt_penalty=config.evt_penalty,
                        cvar_penalty=config.cvar_penalty,
                        cvar_alpha=config.cvar_alpha,
                        robust_alpha_uncertainty=config.robust_alpha_uncertainty,
                        robust_cov_uncertainty=config.robust_cov_uncertainty,
                        factor_cov_blend=config.factor_cov_blend,
                        use_black_litterman=config.use_black_litterman,
                        black_litterman_tau=config.black_litterman_tau,
                        portfolio_notional=config.portfolio_notional,
                        max_adv_participation=config.max_adv_participation,
                        target_vol=config.target_vol,
                        nested_validation_fraction=config.nested_validation_fraction,
                        purge_days=config.purge_days,
                        bootstrap_samples=config.bootstrap_samples,
                        factor_caps=factor_caps,
                        multistarts=config.sortino_multistarts,
                    )
                    if not opts.empty:
                        opts_eval = opts.head(config.max_oos_trials_per_rebalance).copy()
                        for opt_rank, opt_row in opts_eval.reset_index(drop=True).iterrows():
                            combo_tickers = [t for t in str(opt_row.get("Tickers", "")).split(",") if t in px.columns]
                            if not combo_tickers:
                                continue
                            oos_asset = (px.loc[next_date, combo_tickers] / px.loc[exec_date, combo_tickers] - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
                            oos_eq = float(oos_asset.mean()) if len(oos_asset) else np.nan
                            oos_best_proxy = np.nan
                            if not port.empty:
                                w_proxy = port.set_index("Ticker")["Weight"].reindex(combo_tickers).fillna(0.0)
                                if w_proxy.sum() > 0:
                                    w_proxy = w_proxy / w_proxy.sum()
                                    oos_best_proxy = float((w_proxy * oos_asset.reindex(w_proxy.index).fillna(0)).sum())
                            trial = opt_row.to_dict()
                            trial.update(
                                {
                                    "Signal_Date": signal_date,
                                    "Rebalance_Date": exec_date,
                                    "OOS_Start": exec_date,
                                    "OOS_End": next_date,
                                    "Lookback": lb,
                                    "Chunk_Target": size,
                                    "Combos": len(opts),
                                    "Trial_Rank_IS": opt_rank + 1,
                                    "Trial_Key": f"lb={lb}|k={size}|{opt_row.get('Tickers')}",
                                    "OOS_Equal_Return": oos_eq,
                                    "OOS_ProxyWeighted_Return": oos_best_proxy,
                                    "Reoptimized": True,
                                    "Reoptimization_Frequency": config.reoptimization_freq,
                                }
                            )
                            opt_rows.append(trial)
                        row = opts.iloc[0].to_dict()
                        row.update(
                            {
                                "Signal_Date": signal_date,
                                "Rebalance_Date": exec_date,
                                "OOS_Start": exec_date,
                                "OOS_End": next_date,
                                "Lookback": lb,
                                "Chunk_Target": size,
                                "Combos": len(opts),
                                "Trial_Rank_IS": 1,
                                "Trial_Key": f"lb={lb}|k={size}|{opts.iloc[0].get('Tickers')}",
                            }
                        )
                        row_trial_key = row["Trial_Key"]
                        persistence_bonus = persistence_map.get(row_trial_key, 0.0)
                        selection_metric = row.get("Chunk_Objective_Metric", objective_metric_name(config.weight_objective))
                        base_selection_score = to_float(row.get(selection_metric))
                        if pd.isna(base_selection_score):
                            base_selection_score = to_float(row.get("Sortino"))
                        selection_score = base_selection_score + config.robust_selection_lambda * persistence_bonus
                        row["Persistence_Bonus"] = persistence_bonus
                        row["Robust_Selection_Metric"] = selection_metric
                        row["Robust_Selection_Score"] = selection_score
                        if selection_score > best_selection_score:
                            best_selection_score = selection_score
                            best_port, best_options, best_key = port, opts, (lb, size)
            if not best_port.empty:
                cached_best_port = best_port.copy()
                cached_best_options = best_options.copy()
                cached_best_key = best_key
                cached_best_selection_score = best_selection_score
                last_reoptimization_key = current_reoptimization_key
        else:
            best_port = cached_best_port.copy()
            best_options = cached_best_options.copy()
            best_key = cached_best_key
            best_selection_score = cached_best_selection_score
        if best_port.empty:
            continue
        best_trial_key = f"lb={best_key[0]}|k={best_key[1]}|{best_options.iloc[0].get('Tickers')}"
        best_persistence_bonus = persistence_map.get(best_trial_key, 0.0)
        tickers = best_port["Ticker"].tolist()
        weights = best_port.set_index("Ticker")["Weight"]
        weights = weights / weights.sum()
        latent_entropy = to_float(macro_row.get("Latent_State_Entropy"))
        latent_prob = to_float(macro_row.get("Latent_State_Prob"))
        markov_stress_prob = to_float(macro_row.get("Markov_Stress_Prob"))
        markov_transition_entropy = to_float(macro_row.get("Markov_Transition_Entropy"))
        entropy_drag = 1.0 - config.regime_entropy_exposure_penalty * (latent_entropy if pd.notna(latent_entropy) else 0.0)
        dynamic_exposure = model_confidence * max(entropy_drag, 0.0)
        if pd.notna(latent_prob):
            dynamic_exposure *= 0.50 + 0.50 * float(np.clip(latent_prob, 0.0, 1.0))
        if pd.notna(markov_stress_prob):
            dynamic_exposure *= max(0.0, 1.0 - config.markov_stress_exposure_penalty * float(np.clip(markov_stress_prob, 0.0, 1.0)))
        dynamic_exposure = float(np.clip(dynamic_exposure, config.min_dynamic_exposure, config.max_dynamic_exposure)) if config.use_dynamic_sizing else 1.0
        effective_weights = weights * dynamic_exposure
        asset_ret = (px.loc[next_date, tickers] / px.loc[exec_date, tickers] - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
        gross_unscaled = float((weights.reindex(asset_ret.index).fillna(0) * asset_ret).sum())
        gross = float((effective_weights.reindex(asset_ret.index).fillna(0) * asset_ret).sum())
        tc, tc_meta = estimate_transaction_cost(
            effective_weights,
            prev_w,
            px,
            signal_date,
            fixed_tc_bps=config.tc_bps,
            impact_coefficient=config.impact_coefficient,
        )
        perf.append(
            {
                "Signal_Date": signal_date,
                "Rebalance_Date": exec_date,
                "Period_End": next_date,
                "Opt_End": best_options.iloc[0].get("Opt_End", pd.NaT),
                "OOS_Start": exec_date,
                "OOS_End": next_date,
                "Gross_Return": gross,
                "Gross_Unscaled_Return": gross_unscaled,
                "Dynamic_Exposure": dynamic_exposure,
                "Cash_Weight": 1.0 - dynamic_exposure,
                "Transaction_Cost": tc,
                "Net_Return": gross - tc,
                "Turnover": tc_meta["turnover"],
                "Fixed_TC": tc_meta["fixed_tc"],
                "Impact_TC": tc_meta["impact_tc"],
                "N": len(tickers),
                "Best_Sortino_IS": best_options.iloc[0]["Sortino"],
                "Best_Objective": config.weight_objective,
                "Best_Objective_Metric": best_options.iloc[0].get("Chunk_Objective_Metric", objective_metric_name(config.weight_objective)),
                "Best_Objective_Value_IS": best_options.iloc[0].get(best_options.iloc[0].get("Chunk_Objective_Metric", objective_metric_name(config.weight_objective)), np.nan),
                "Best_Robust_Selection_Score": best_selection_score,
                "Best_Persistence_Bonus": best_persistence_bonus,
                "Best_Trial_Key": best_trial_key,
                "Reoptimized": bool(reoptimized),
                "Rebalance_Frequency": config.rebalance_freq,
                "Reoptimization_Frequency": config.reoptimization_freq,
                "Model_Confidence": model_confidence,
                "Effective_Alpha_Weight": effective_alpha_weight,
                "Effective_Risk_Aversion": effective_risk_aversion,
                **confidence_meta,
                "Best_Lookback": best_key[0],
                "Best_Chunk_Size": best_key[1],
                "ExAnte_Vol": best_port["ex_ante_vol"].iloc[0] if "ex_ante_vol" in best_port else np.nan,
                "Hist_VaR_95_Daily": best_port["hist_var_95_daily"].iloc[0] if "hist_var_95_daily" in best_port else np.nan,
                "Hist_CVaR_95_Daily": best_port["hist_cvar_95_daily"].iloc[0] if "hist_cvar_95_daily" in best_port else np.nan,
                "Weight_HHI": best_port["weight_hhi"].iloc[0] if "weight_hhi" in best_port else np.nan,
                "Regime": f"{macro_row.get('Regime_Hawkish_Dovish')}/{macro_row.get('Regime_Bull_Bear')}",
                "Latent_Regime_State": macro_row.get("Latent_State_Label", macro_row.get("Latent_Regime_State", np.nan)),
                "Latent_Regime_Prob": latent_prob,
                "Latent_Regime_Entropy": latent_entropy,
                "Markov_Next_State_Mode": macro_row.get("Markov_Next_State_Mode", np.nan),
                "Markov_State_Persistence": macro_row.get("Markov_State_Persistence", np.nan),
                "Markov_Stress_Prob": markov_stress_prob,
                "Markov_Risk_On_Prob": macro_row.get("Markov_Risk_On_Prob", np.nan),
                "Markov_Transition_Entropy": markov_transition_entropy,
            }
        )
        for _, r in best_port.iterrows():
            holdings.append(
                {
                    "Signal_Date": signal_date,
                    "Rebalance_Date": exec_date,
                    "Ticker": r["Ticker"],
                    "Sector": r.get("Sector"),
                    "Country": r.get("Country"),
                    "Weight": r["Weight"],
                    "Effective_Weight": r["Weight"] * dynamic_exposure,
                    "Dynamic_Exposure": dynamic_exposure,
                    "Reoptimized": bool(reoptimized),
                    "Composite_Score": r["Composite_Score"],
                    "OOS_Return": asset_ret.get(r["Ticker"], np.nan),
                    "Sortino_IS": r["Optimization_Sortino"],
                    "Bootstrap_Stability": r.get("Bootstrap_Stability", np.nan),
                }
            )
        prev_w = effective_weights.copy()
    return pd.DataFrame(perf), pd.DataFrame(holdings), pd.DataFrame(opt_rows)


def sector_diagnostics(prices: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    if cs.empty:
        return pd.DataFrame()
    sector_map = cs[["Ticker", "Sector"]].drop_duplicates().set_index("Ticker")["Sector"]
    valid = [c for c in prices.columns if c in sector_map.index]
    if not valid:
        return pd.DataFrame()
    ret = prices[valid].pct_change(fill_method=None).dropna(how="all")
    sec_ret = ret.T.groupby(sector_map.reindex(valid)).mean().T
    return pd.DataFrame(
        {
            "Momentum_21": sec_ret.iloc[-21:].add(1).prod() - 1,
            "Momentum_63": sec_ret.iloc[-63:].add(1).prod() - 1,
            "Vol_63": sec_ret.iloc[-63:].std() * np.sqrt(252),
            "Downside_Vol_63": sec_ret.iloc[-63:].clip(upper=0).std() * np.sqrt(252),
        }
    ).sort_values("Momentum_63", ascending=False)


def portfolio_vs_benchmark_curve(perf: pd.DataFrame, prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    out = perf.copy()
    if benchmark in prices.columns:
        b = []
        for _, row in out.iterrows():
            try:
                b.append(prices.loc[row["OOS_End"], benchmark] / prices.loc[row["OOS_Start"], benchmark] - 1)
            except Exception:
                b.append(np.nan)
        out["Benchmark_Return"] = b
    else:
        out["Benchmark_Return"] = np.nan
    out["Portfolio_Equity"] = (1 + out["Net_Return"].fillna(0.0)).cumprod()
    out["Benchmark_Equity"] = (1 + out["Benchmark_Return"].fillna(0.0)).cumprod()
    out["Active_Equity"] = out["Portfolio_Equity"] / out["Benchmark_Equity"].replace(0, np.nan)
    return out


def side_boom_selected_frame(prices: pd.DataFrame, tickers: Iterable[str], asof_date, lookback: int = 252) -> pd.DataFrame:
    tickers = [t for t in normalize_side_tickers(tickers) if t in prices.columns]
    if not tickers:
        return pd.DataFrame()
    ret = prices.loc[:asof_date, tickers].tail(lookback + 1).pct_change(fill_method=None)
    rows = []
    for ticker in tickers:
        r = ret[ticker].dropna()
        rows.append(
            {
                "Ticker": ticker,
                "Sector": "Side Alpha",
                "Country": "United States",
                "Composite_Score": sortino_ratio(r) if len(r) > 3 else 0.0,
                "Dollar_Volume_63": 1e12,
                "Data_Available": bool(len(r) > 1),
                "First_Price_Date": prices[ticker].dropna().index.min() if prices[ticker].notna().any() else pd.NaT,
                "Last_Price_Date": prices[ticker].dropna().index.max() if prices[ticker].notna().any() else pd.NaT,
                "Return_Obs": int(len(r)),
            }
        )
    out = pd.DataFrame(rows)
    if out["Composite_Score"].notna().sum() > 1:
        out["Composite_Score"] = out["Composite_Score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def optimize_side_boom_portfolio(
    prices: pd.DataFrame,
    config: RunConfig,
    macro: pd.DataFrame | None = None,
    lookback: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = normalize_side_tickers(config.side_boom_tickers)
    fixed_weights = side_boom_fixed_weight_map(config)
    fixed_tickers = set(fixed_weights)
    fixed_weight_total = float(sum(fixed_weights.values()))
    asof_date = prices.index.max() if not prices.empty else pd.Timestamp.today()
    selected = side_boom_selected_frame(prices, tickers, asof_date, lookback=lookback)
    all_rows = []
    selected_tickers = selected.get("Ticker", pd.Series(dtype=str)).astype(str).tolist()
    for fixed_ticker in fixed_tickers:
        if fixed_ticker and fixed_ticker not in selected_tickers:
            all_rows.append(
                {
                    "Ticker": fixed_ticker,
                    "Sector": "Side Alpha",
                    "Country": "United States",
                    "Composite_Score": 0.0,
                    "Dollar_Volume_63": 1e12,
                    "Data_Available": bool(fixed_ticker in prices.columns and prices[fixed_ticker].notna().sum() > 1),
                    "First_Price_Date": prices[fixed_ticker].dropna().index.min() if fixed_ticker in prices.columns and prices[fixed_ticker].notna().any() else pd.NaT,
                    "Last_Price_Date": prices[fixed_ticker].dropna().index.max() if fixed_ticker in prices.columns and prices[fixed_ticker].notna().any() else pd.NaT,
                    "Return_Obs": int(prices[fixed_ticker].pct_change(fill_method=None).dropna().shape[0]) if fixed_ticker in prices.columns else 0,
                }
            )
    if not selected.empty:
        all_rows.extend(selected.to_dict(orient="records"))
    selected_all = pd.DataFrame(all_rows).drop_duplicates("Ticker", keep="last") if all_rows else pd.DataFrame()
    available = selected_all[selected_all["Data_Available"].fillna(False)].copy() if not selected_all.empty else pd.DataFrame()
    variable = available[~available["Ticker"].isin(fixed_tickers)].copy() if not available.empty else pd.DataFrame()
    remaining = max(1.0 - fixed_weight_total, 0.0)
    weights = pd.Series(dtype=float)
    side_objective = "fixed_side_alpha" if fixed_weight_total >= 1.0 - 1e-8 else "sortino"
    meta = {"status": "no_variable_assets", "side_objective": side_objective, "fixed_tickers": ",".join(sorted(fixed_tickers)), "fixed_weight_total": fixed_weight_total}
    if remaining > 0 and not variable.empty:
        if len(variable) == 1:
            w_var = pd.Series(1.0, index=variable["Ticker"].tolist())
            meta.update({"status": "single_variable_asset"})
        else:
            w_var, meta = construct_constrained_weights(
                variable,
                prices,
                asof_date,
                lookback=min(lookback, max(20, len(prices.loc[:asof_date]) - 1)),
                macro=macro,
                benchmark_ticker=config.benchmark_ticker,
                max_weight=1.0,
                sector_weight_cap=1.0,
                risk_aversion=config.risk_aversion,
                alpha_weight=0.0,
                objective="sortino",
                entropy_penalty=0.0,
                crlb_penalty=0.0,
                garch_penalty=0.0,
                evt_penalty=0.0,
                cvar_penalty=config.cvar_penalty,
                cvar_alpha=config.cvar_alpha,
                robust_alpha_uncertainty=0.0,
                robust_cov_uncertainty=0.0,
                factor_cov_blend=0.0,
                portfolio_notional=config.portfolio_notional,
                max_adv_participation=1.0,
                target_vol=config.target_vol,
                multistarts=max(3, min(config.sortino_multistarts, 8)),
            )
        weights = w_var * remaining
    for fixed_ticker, fixed_weight in fixed_weights.items():
        if fixed_weight > 0:
            weights.loc[fixed_ticker] = fixed_weight
    weights = weights[weights > 1e-12]
    if weights.sum() > 0:
        weights = weights / weights.sum()
    portfolio = selected_all.set_index("Ticker").reindex(weights.index).reset_index() if not selected_all.empty else pd.DataFrame({"Ticker": weights.index})
    if "Ticker" not in portfolio.columns and len(portfolio.columns):
        portfolio = portfolio.rename(columns={portfolio.columns[0]: "Ticker"})
    portfolio["Weight"] = portfolio["Ticker"].map(weights).fillna(0.0)
    portfolio["Side_Objective"] = side_objective
    portfolio["Private_Sleeve"] = "Side Alpha"
    portfolio["Fixed_Weight_Constraint"] = portfolio["Ticker"].isin(fixed_tickers)
    portfolio["Fixed_Weight_Target"] = portfolio["Ticker"].map(fixed_weights).fillna(np.nan)
    portfolio["Compliance_Note"] = np.where(
        portfolio["Fixed_Weight_Constraint"],
        "User scenario constraint; do not use material non-public information.",
        "Sortino-optimized residual inside the private Side Alpha sleeve.",
    )
    for key, value in meta.items():
        if np.isscalar(value) or isinstance(value, str):
            portfolio[f"side_{key}"] = value
    curve = side_boom_equity_curve(prices, weights, benchmark=config.benchmark_ticker)
    diagnostics = side_boom_diagnostics(curve, portfolio, benchmark=config.benchmark_ticker)
    return portfolio.sort_values("Weight", ascending=False).reset_index(drop=True), curve, diagnostics


def side_asset_has_trade_pair(prices: pd.DataFrame, ticker: str, start_date, end_date) -> bool:
    if ticker not in prices.columns:
        return False
    try:
        return pd.notna(prices.loc[start_date, ticker]) and pd.notna(prices.loc[end_date, ticker])
    except Exception:
        return False


def side_asset_obs_count(prices: pd.DataFrame, ticker: str, asof_date) -> int:
    if ticker not in prices.columns:
        return 0
    return int(prices.loc[:asof_date, ticker].pct_change(fill_method=None).dropna().shape[0])


def optimize_side_boom_weights_asof(
    prices: pd.DataFrame,
    config: RunConfig,
    signal_date,
    exec_date,
    next_date,
    macro: pd.DataFrame | None = None,
    lookback: int = 252,
) -> tuple[pd.Series, dict]:
    tickers = normalize_side_tickers(config.side_boom_tickers)
    fixed_requested = side_boom_fixed_weight_map(config)
    fixed_tickers = set(fixed_requested)
    fixed_weight_requested = float(sum(fixed_requested.values()))
    cash_return = float(config.side_boom_cash_return)
    fixed_active = {
        ticker: weight
        for ticker, weight in fixed_requested.items()
        if side_asset_has_trade_pair(prices, ticker, exec_date, next_date)
    }
    fixed_weight_active = float(sum(fixed_active.values()))
    cash_weight = fixed_weight_requested - fixed_weight_active

    variable_rows = []
    for ticker in tickers:
        if ticker in fixed_tickers:
            continue
        obs = side_asset_obs_count(prices, ticker, signal_date)
        tradable = side_asset_has_trade_pair(prices, ticker, exec_date, next_date)
        if obs >= int(config.side_boom_min_obs) and tradable:
            r = prices.loc[:signal_date, ticker].tail(lookback + 1).pct_change(fill_method=None).dropna()
            variable_rows.append(
                {
                    "Ticker": ticker,
                    "Sector": "Side Alpha",
                    "Country": "United States",
                    "Composite_Score": sortino_ratio(r) if len(r) > 3 else 0.0,
                    "Dollar_Volume_63": 1e12,
                    "Return_Obs": obs,
                    "Data_Available": True,
                }
            )
    variable = pd.DataFrame(variable_rows)
    remaining = max(1.0 - fixed_weight_requested, 0.0)
    weights = pd.Series(dtype=float)
    status = "cash_only"
    if remaining > 0 and not variable.empty:
        if len(variable) == 1:
            w_var = pd.Series(1.0, index=variable["Ticker"].tolist())
            status = "single_variable_asset"
        else:
            w_var, meta = construct_constrained_weights(
                variable,
                prices,
                signal_date,
                lookback=min(lookback, max(20, len(prices.loc[:signal_date]) - 1)),
                macro=macro.loc[:signal_date] if macro is not None and not macro.empty else None,
                benchmark_ticker=config.benchmark_ticker,
                max_weight=1.0,
                sector_weight_cap=1.0,
                risk_aversion=config.risk_aversion,
                alpha_weight=0.0,
                objective="sortino",
                entropy_penalty=0.0,
                crlb_penalty=0.0,
                garch_penalty=0.0,
                evt_penalty=0.0,
                cvar_penalty=config.cvar_penalty,
                cvar_alpha=config.cvar_alpha,
                robust_alpha_uncertainty=0.0,
                robust_cov_uncertainty=0.0,
                factor_cov_blend=0.0,
                portfolio_notional=config.portfolio_notional,
                max_adv_participation=1.0,
                target_vol=config.target_vol,
                multistarts=max(3, min(config.sortino_multistarts, 8)),
            )
            status = str(meta.get("status", "sortino_optimized"))
        weights = w_var * remaining
    else:
        cash_weight += remaining
    for fixed_ticker, fixed_weight in fixed_active.items():
        if fixed_weight > 0:
            weights.loc[fixed_ticker] = fixed_weight
    if weights.sum() > 1.0:
        weights = weights / weights.sum()
        cash_weight = 0.0
    else:
        cash_weight += 1.0 - weights.sum() - cash_weight
    cash_weight = float(np.clip(cash_weight, 0.0, 1.0))
    meta = {
        "status": status,
        "fixed_ticker": config.side_boom_fixed_ticker,
        "fixed_tickers": ",".join(sorted(fixed_tickers)),
        "fixed_weight_requested": fixed_weight_requested,
        "fixed_weight_active": fixed_weight_active,
        "cash_weight": cash_weight,
        "cash_return": cash_return,
        "eligible_variable_assets": int(len(variable)),
        "private_firewall": config.side_boom_mode,
    }
    return weights.sort_values(ascending=False), meta


def side_boom_walk_forward(
    prices: pd.DataFrame,
    config: RunConfig,
    macro: pd.DataFrame | None = None,
    perf: pd.DataFrame | None = None,
    lookback: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if perf is not None and not perf.empty and {"Signal_Date", "OOS_Start", "OOS_End"}.issubset(perf.columns):
        schedule = perf[["Signal_Date", "OOS_Start", "OOS_End"]].dropna().copy()
    else:
        px = prices.sort_index().ffill()
        rebal_dates = px.resample(config.rebalance_freq).last().index
        rebal_dates = [px.index[px.index.get_indexer([d], method="ffill")[0]] for d in rebal_dates if px.index.get_indexer([d], method="ffill")[0] >= 0]
        rows = []
        for i in range(len(rebal_dates) - 1):
            exec_date, next_date = rebal_dates[i], rebal_dates[i + 1]
            exec_pos = px.index.get_indexer([exec_date], method="ffill")[0]
            rows.append({"Signal_Date": px.index[max(0, exec_pos - config.embargo_days)], "OOS_Start": exec_date, "OOS_End": next_date})
        schedule = pd.DataFrame(rows)
    perf_rows, holding_rows = [], []
    prev_w = pd.Series(dtype=float)
    for _, row in schedule.iterrows():
        signal_date = pd.Timestamp(row["Signal_Date"])
        exec_date = pd.Timestamp(row["OOS_Start"])
        next_date = pd.Timestamp(row["OOS_End"])
        weights, meta = optimize_side_boom_weights_asof(prices, config, signal_date, exec_date, next_date, macro=macro, lookback=lookback)
        asset_returns = {}
        gross = 0.0
        for ticker, weight in weights.items():
            if side_asset_has_trade_pair(prices, ticker, exec_date, next_date):
                r = float(prices.loc[next_date, ticker] / prices.loc[exec_date, ticker] - 1.0)
            else:
                r = float(config.side_boom_cash_return)
            asset_returns[ticker] = r
            gross += float(weight) * r
        gross += float(meta.get("cash_weight", 0.0)) * float(config.side_boom_cash_return)
        turnover = float((weights.subtract(prev_w, fill_value=0.0).abs().sum()) / 2.0) if not prev_w.empty else float(weights.abs().sum())
        perf_rows.append(
            {
                "Signal_Date": signal_date,
                "OOS_Start": exec_date,
                "OOS_End": next_date,
                "Side_Boom_Return": gross,
                "Side_Boom_Cash_Weight": meta.get("cash_weight", 0.0),
                "Side_Boom_Turnover": turnover,
                **meta,
            }
        )
        for ticker, weight in weights.items():
            holding_rows.append(
                {
                    "Signal_Date": signal_date,
                    "OOS_Start": exec_date,
                    "OOS_End": next_date,
                    "Ticker": ticker,
                    "Weight": weight,
                    "Asset_Return": asset_returns.get(ticker, np.nan),
                    "Contribution": weight * asset_returns.get(ticker, 0.0),
                    "Fixed_Weight_Constraint": ticker in set(str(meta.get("fixed_tickers", "")).split(",")),
                    "Private_Firewall": config.side_boom_mode,
                }
            )
        if meta.get("cash_weight", 0.0) > 1e-12:
            holding_rows.append(
                {
                    "Signal_Date": signal_date,
                    "OOS_Start": exec_date,
                    "OOS_End": next_date,
                    "Ticker": "CASH",
                    "Weight": meta.get("cash_weight", 0.0),
                    "Asset_Return": config.side_boom_cash_return,
                    "Contribution": meta.get("cash_weight", 0.0) * config.side_boom_cash_return,
                    "Fixed_Weight_Constraint": False,
                    "Private_Firewall": config.side_boom_mode,
                }
            )
        prev_w = weights.copy()
    side_perf = pd.DataFrame(perf_rows)
    side_holdings = pd.DataFrame(holding_rows)
    if side_perf.empty:
        return pd.DataFrame(), side_holdings, pd.DataFrame()
    side_perf["Side_Boom_Equity"] = (1.0 + side_perf["Side_Boom_Return"].fillna(0.0)).cumprod()
    if config.benchmark_ticker in prices.columns:
        b = []
        for _, row in side_perf.iterrows():
            if side_asset_has_trade_pair(prices, config.benchmark_ticker, row["OOS_Start"], row["OOS_End"]):
                b.append(float(prices.loc[row["OOS_End"], config.benchmark_ticker] / prices.loc[row["OOS_Start"], config.benchmark_ticker] - 1.0))
            else:
                b.append(np.nan)
        side_perf["Side_Benchmark_Return"] = b
        side_perf["Side_Benchmark_Equity"] = (1.0 + side_perf["Side_Benchmark_Return"].fillna(0.0)).cumprod()
    side_perf["Period_End"] = side_perf["OOS_End"]
    diagnostics = side_boom_diagnostics(side_perf, side_holdings, benchmark=config.benchmark_ticker)
    return side_perf, side_holdings, diagnostics


def side_boom_equity_curve(prices: pd.DataFrame, weights: pd.Series, benchmark: str = "SPY") -> pd.DataFrame:
    if prices.empty or weights.empty:
        return pd.DataFrame()
    tickers = list(weights.index)
    px = prices.reindex(columns=tickers).sort_index().ffill()
    ret = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    port_ret = ret @ weights.reindex(tickers).fillna(0.0)
    out = pd.DataFrame({"Period_End": port_ret.index, "Side_Boom_Return": port_ret.values})
    out["Side_Boom_Equity"] = (1.0 + out["Side_Boom_Return"].fillna(0.0)).cumprod()
    if benchmark in prices.columns:
        b = prices[benchmark].sort_index().ffill().pct_change(fill_method=None).reindex(port_ret.index).fillna(0.0)
        out["Side_Benchmark_Return"] = b.values
        out["Side_Benchmark_Equity"] = (1.0 + b.fillna(0.0)).cumprod().values
    for ticker in tickers:
        if ticker in ret.columns:
            out[f"{ticker}_Contribution"] = ret[ticker].values * float(weights.get(ticker, 0.0))
    return out


def side_boom_diagnostics(curve: pd.DataFrame, portfolio: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    r = pd.to_numeric(curve["Side_Boom_Return"], errors="coerce").dropna()
    b = pd.to_numeric(curve.get("Side_Benchmark_Return", pd.Series(dtype=float)), errors="coerce").reindex(r.index).fillna(0.0)
    equity = (1.0 + r.fillna(0.0)).cumprod()
    periods = 252
    rows = [
        {"Metric": "Side_Total_Return", "Value": equity.iloc[-1] - 1.0},
        {"Metric": "Side_Annualized_Return", "Value": equity.iloc[-1] ** (periods / max(len(r), 1)) - 1.0},
        {"Metric": "Side_Annualized_Vol", "Value": r.std(ddof=1) * np.sqrt(periods) if len(r) > 1 else np.nan},
        {"Metric": "Side_Sortino", "Value": sortino_ratio(r)},
        {"Metric": "Side_Sharpe", "Value": sharpe_ratio(r)},
        {"Metric": "Side_Max_Drawdown", "Value": (equity / equity.cummax() - 1.0).min()},
        {"Metric": "Side_Information_Ratio", "Value": information_ratio(r, b) if len(b) else np.nan},
        {"Metric": "Side_Benchmark", "Value": benchmark},
        {"Metric": "Side_N_Assets", "Value": len(portfolio) if portfolio is not None else 0},
    ]
    if portfolio is not None and not portfolio.empty and "Ticker" in portfolio:
        latest = portfolio.copy()
        if "OOS_End" in latest:
            latest_date = latest["OOS_End"].max()
            latest = latest[latest["OOS_End"].eq(latest_date)]
        if "Weight" in latest:
            rows.append({"Metric": "Side_Latest_Cash_Weight", "Value": float(latest.loc[latest["Ticker"].eq("CASH"), "Weight"].sum())})
            rows.append({"Metric": "Side_Latest_HHI", "Value": float(np.square(pd.to_numeric(latest["Weight"], errors="coerce").fillna(0.0)).sum())})
    return pd.DataFrame(rows)


def side_boom_pelt_diagnostics(curve: pd.DataFrame, source_label: str = "Private Side Alpha current allocation") -> dict[str, pd.DataFrame]:
    if curve is None or curve.empty or "Side_Boom_Return" not in curve:
        return {
            "side_boom_pelt_regime_segments": pd.DataFrame(),
            "side_boom_pelt_change_points": pd.DataFrame(),
            "side_boom_pelt_timeline": pd.DataFrame(),
        }
    date_col = "Period_End" if "Period_End" in curve else "OOS_End" if "OOS_End" in curve else None
    idx = pd.to_datetime(curve[date_col], errors="coerce") if date_col else pd.RangeIndex(len(curve))
    r = pd.Series(pd.to_numeric(curve["Side_Boom_Return"], errors="coerce").values, index=idx).dropna()
    r = r[~pd.isna(r.index)] if not isinstance(r.index, pd.RangeIndex) else r
    diag = pelt_change_point_analysis(r, min_size=21, max_obs=756)
    mapped = {
        "side_boom_pelt_regime_segments": diag.get("pelt_regime_segments", pd.DataFrame()).copy(),
        "side_boom_pelt_change_points": diag.get("pelt_change_points", pd.DataFrame()).copy(),
        "side_boom_pelt_timeline": diag.get("pelt_timeline", pd.DataFrame()).copy(),
    }
    for df in mapped.values():
        if df is not None and not df.empty:
            df.insert(0, "PELT_Source", source_label)
            df.insert(0, "Series", "SIDE_BOOM")
    return mapped


def merge_side_boom_into_equity_curve(equity_curve: pd.DataFrame, side_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve is None or equity_curve.empty or side_curve is None or side_curve.empty:
        return equity_curve
    left = equity_curve.copy()
    left["Period_End"] = pd.to_datetime(left["Period_End"])
    right_cols = [c for c in ["Period_End", "Side_Boom_Equity", "Side_Benchmark_Equity", "Side_Boom_Return", "Side_Boom_Cash_Weight"] if c in side_curve.columns]
    right = side_curve[right_cols].copy()
    right["Period_End"] = pd.to_datetime(right["Period_End"])
    merged = pd.merge_asof(
        left.sort_values("Period_End"),
        right.sort_values("Period_End"),
        on="Period_End",
        direction="backward",
    )
    return merged.sort_values("Period_End").reset_index(drop=True)


def summarize_backtest(perf: pd.DataFrame, prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    out = portfolio_vs_benchmark_curve(perf, prices, benchmark)

    r = out["Net_Return"].astype(float)
    b = out["Benchmark_Return"].astype(float)
    active = r - b
    equity = (1 + r.fillna(0.0)).cumprod()
    periods = 12
    var95, cvar95 = historical_var_cvar(r, 0.95)
    beta = np.nan
    if b.notna().sum() > 3 and b.var(ddof=1) > 0:
        beta = float(r.cov(b) / b.var(ddof=1))
    summary = {
        "Periods": len(out),
        "Total_Return": equity.iloc[-1] - 1,
        "Annualized_Return": (equity.iloc[-1] ** (periods / len(out))) - 1 if len(out) > 0 else np.nan,
        "Annualized_Vol": r.std(ddof=1) * np.sqrt(periods),
        "Sharpe": (r.mean() * periods) / (r.std(ddof=1) * np.sqrt(periods)) if r.std(ddof=1) > 0 else np.nan,
        "Sortino": sortino_ratio(r, mar=0.0) / np.sqrt(252 / periods) if len(r) > 2 else np.nan,
        "Max_Drawdown": (equity / equity.cummax() - 1).min(),
        "Hit_Rate": (r > 0).mean(),
        "Avg_Turnover": out["Turnover"].mean() if "Turnover" in out else np.nan,
        "Avg_TC": out["Transaction_Cost"].mean() if "Transaction_Cost" in out else np.nan,
        "VaR_95_Period": var95,
        "CVaR_95_Period": cvar95,
        "Benchmark_Annualized_Return": (1 + b.mean()) ** periods - 1 if b.notna().sum() else np.nan,
        "Active_Annualized_Return": active.mean() * periods if active.notna().sum() else np.nan,
        "Tracking_Error": active.std(ddof=1) * np.sqrt(periods) if active.notna().sum() > 1 else np.nan,
        "Information_Ratio": (active.mean() * periods) / (active.std(ddof=1) * np.sqrt(periods)) if active.std(ddof=1) > 0 else np.nan,
        "Beta_To_Benchmark": beta,
    }
    return pd.DataFrame(summary, index=["portfolio"]).T.reset_index().rename(columns={"index": "Metric", "portfolio": "Value"})


def portfolio_return_diagnostics(
    prices: pd.DataFrame,
    portfolio: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    asof_date=None,
    lookback: int = 252,
    initial_value: float = 100_000.0,
    forecast_horizon_days: int = 252,
    forecast_sims: int = 3000,
) -> dict[str, pd.DataFrame]:
    if portfolio.empty:
        return {"individual_returns": pd.DataFrame(), "covariance": pd.DataFrame(), "correlation": pd.DataFrame(), "portfolio_returns": pd.DataFrame()}
    tickers = [t for t in portfolio["Ticker"] if t in prices.columns]
    if not tickers:
        return {"individual_returns": pd.DataFrame(), "covariance": pd.DataFrame(), "correlation": pd.DataFrame(), "portfolio_returns": pd.DataFrame()}
    end_date = pd.Timestamp(asof_date) if asof_date is not None else prices.index.max()
    ret = prices.loc[:end_date, tickers].tail(lookback + 1).pct_change(fill_method=None).dropna(how="all")
    weights = portfolio.set_index("Ticker").reindex(tickers)["Weight"].fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()
    port_ret = ret.fillna(0.0) @ weights
    variance_rows = []
    port_models = fit_variance_architecture(port_ret)
    if not port_models.empty:
        pm = port_models.copy()
        pm.insert(0, "Series", "PORTFOLIO")
        variance_rows.append(pm)
    for tk in tickers:
        vm = fit_variance_architecture(ret[tk])
        if not vm.empty:
            vm = vm.copy()
            vm.insert(0, "Series", tk)
            variance_rows.append(vm)
    variance_selection = pd.concat(variance_rows, ignore_index=True) if variance_rows else pd.DataFrame()
    port_vol = port_ret.std(ddof=1)
    mall_rows = []
    if port_vol and port_vol > 0:
        worst = port_ret <= port_ret.quantile(0.05)
        for tk in tickers:
            asset = ret[tk].fillna(0.0)
            local_derivative = weights.get(tk, 0.0) * asset.std(ddof=1) / port_vol
            tail_derivative = weights.get(tk, 0.0) * asset[worst].mean() if worst.any() else np.nan
            mall_rows.append(
                {
                    "Ticker": tk,
                    "Discrete_Malliavin_Local_Sensitivity": local_derivative,
                    "Discrete_Malliavin_Tail_Sensitivity": tail_derivative,
                    "Weight": weights.get(tk, 0.0),
                }
            )
    factor_diag = factor_model_risk_decomposition(prices, macro, portfolio, asof_date=end_date, lookback=lookback) if macro is not None else {}
    out = {
        "individual_returns": ret.reset_index(names="Date"),
        "covariance": ret.cov() * 252,
        "correlation": ret.corr(),
        "portfolio_returns": pd.DataFrame({"Date": port_ret.index, "Portfolio_Return": port_ret.values}),
        "malliavin_sensitivity": pd.DataFrame(mall_rows),
        "variance_model_selection": variance_selection,
    }
    out.update(pelt_change_point_analysis(port_ret))
    out.update(portfolio_gbm_forecast(port_ret, initial_value=initial_value, horizon_days=forecast_horizon_days, n_sims=forecast_sims))
    out.update(factor_diag)
    return out


def portfolio_gbm_forecast(
    portfolio_returns: pd.Series,
    initial_value: float = 100_000.0,
    horizon_days: int = 252,
    n_sims: int = 3000,
    seed: int = 20260518,
) -> dict[str, pd.DataFrame]:
    """Monte Carlo GBM forecast for the optimized portfolio value process.

    The estimator uses only observed portfolio returns up to the current as-of
    date. The simulated law is:

        dV_t / V_t = mu dt + sigma dW_t

    with daily time step dt=1/252. Quantiles are stochastic confidence bands,
    not deterministic forecasts.
    """
    r = pd.Series(portfolio_returns).dropna().astype(float)
    if len(r) < 30:
        return {"gbm_forecast_path": pd.DataFrame(), "gbm_forecast_summary": pd.DataFrame()}
    initial_value = float(max(to_float(initial_value), 1.0))
    horizon_days = int(np.clip(to_float(horizon_days), 1, 2520))
    n_sims = int(np.clip(to_float(n_sims), 100, 20000))
    log_r = np.log1p(r.clip(lower=-0.95)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(log_r) < 30:
        return {"gbm_forecast_path": pd.DataFrame(), "gbm_forecast_summary": pd.DataFrame()}
    mu_ann = float(log_r.mean() * 252.0)
    sigma_ann = float(log_r.std(ddof=1) * np.sqrt(252.0))
    if not np.isfinite(sigma_ann) or sigma_ann <= 1e-8:
        return {"gbm_forecast_path": pd.DataFrame(), "gbm_forecast_summary": pd.DataFrame()}
    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((horizon_days, n_sims))
    increments = (mu_ann - 0.5 * sigma_ann * sigma_ann) * dt + sigma_ann * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(increments, axis=0)])
    values = initial_value * np.exp(log_paths)
    qs = np.nanquantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=1)
    path = pd.DataFrame(
        {
            "Step": np.arange(horizon_days + 1, dtype=int),
            "Q01": qs[0],
            "Q05": qs[1],
            "Q25": qs[2],
            "Q50": qs[3],
            "Q75": qs[4],
            "Q95": qs[5],
            "Q99": qs[6],
        }
    )
    terminal = values[-1, :]
    t1 = values[min(1, horizon_days), :]
    terminal_return = terminal / initial_value - 1.0
    t1_return = t1 / initial_value - 1.0
    var_95 = float(np.nanquantile(terminal_return, 0.05))
    cvar_95 = float(np.nanmean(terminal_return[terminal_return <= var_95])) if np.any(terminal_return <= var_95) else np.nan
    summary = pd.DataFrame(
        [
            {"Metric": "Initial_Value", "Value": initial_value},
            {"Metric": "Horizon_Days", "Value": horizon_days},
            {"Metric": "Simulations", "Value": n_sims},
            {"Metric": "Estimated_Log_Mu_Ann", "Value": mu_ann},
            {"Metric": "Estimated_Log_Sigma_Ann", "Value": sigma_ann},
            {"Metric": "TPlus1_Mean_Value", "Value": float(np.nanmean(t1))},
            {"Metric": "TPlus1_Q05_Value", "Value": float(np.nanquantile(t1, 0.05))},
            {"Metric": "TPlus1_Q50_Value", "Value": float(np.nanquantile(t1, 0.50))},
            {"Metric": "TPlus1_Q95_Value", "Value": float(np.nanquantile(t1, 0.95))},
            {"Metric": "TPlus1_Prob_Loss", "Value": float(np.nanmean(t1_return < 0.0))},
            {"Metric": "Terminal_Mean_Value", "Value": float(np.nanmean(terminal))},
            {"Metric": "Terminal_Q05_Value", "Value": float(np.nanquantile(terminal, 0.05))},
            {"Metric": "Terminal_Q50_Value", "Value": float(np.nanquantile(terminal, 0.50))},
            {"Metric": "Terminal_Q95_Value", "Value": float(np.nanquantile(terminal, 0.95))},
            {"Metric": "Terminal_Prob_Loss", "Value": float(np.nanmean(terminal_return < 0.0))},
            {"Metric": "Terminal_VaR_95_Return", "Value": var_95},
            {"Metric": "Terminal_CVaR_95_Return", "Value": cvar_95},
            {"Metric": "Model", "Value": "GBM Monte Carlo under empirical log-return estimators"},
        ]
    )
    return {"gbm_forecast_path": path, "gbm_forecast_summary": summary}


def overfit_diagnostics(opt_grid: pd.DataFrame) -> pd.DataFrame:
    if opt_grid.empty or "Sortino" not in opt_grid:
        return pd.DataFrame()
    g = opt_grid.copy()
    grouped = g.groupby("Rebalance_Date")["Sortino"]
    rows = []
    for date, x in grouped:
        x = x.dropna()
        if len(x) < 3:
            continue
        best = x.max()
        median = x.median()
        sd = x.std(ddof=1)
        rows.append(
            {
                "Rebalance_Date": date,
                "N_Trials": len(x),
                "Best_Sortino": best,
                "Median_Sortino": median,
                "Best_Minus_Median": best - median,
                "Selection_Z": (best - median) / sd if sd > 0 else np.nan,
                "Deflated_Sortino_Proxy": (best - median) / (sd * np.sqrt(1 + np.log(len(x)))) if sd > 0 else np.nan,
                "PBO_Proxy": float((x.rank(pct=True).loc[x.idxmax()] > 0.90) and ((best - median) / sd > 2.0)) if sd > 0 else np.nan,
                "Top_Decile_Threshold": x.quantile(0.90),
            }
        )
    diag = pd.DataFrame(rows)
    if diag.empty:
        return diag
    diag["Overfit_Risk_Flag"] = diag["Selection_Z"] > 2.0
    return diag


def deflated_sortino_diagnostics(r: pd.Series, n_trials: int = 1, periods_per_year: int = 12) -> pd.DataFrame:
    x = pd.Series(r).dropna().astype(float)
    if len(x) < 6:
        return pd.DataFrame()
    s = sortino_ratio(x, mar=0.0) / np.sqrt(252 / periods_per_year)
    if not np.isfinite(s):
        vol = x.std(ddof=1)
        s = (x.mean() * periods_per_year) / (vol * np.sqrt(periods_per_year)) if vol and vol > 0 else np.nan
    sk = skew(x, bias=False) if len(x) > 3 else 0.0
    ku = kurtosis(x, fisher=False, bias=False) if len(x) > 4 else 3.0
    n_trials = max(int(n_trials), 1)
    # Bailey-Lopez de Prado style deflation adapted to Sortino as a downside-risk score.
    sr_var = max((1.0 - sk * s + ((ku - 1.0) / 4.0) * s * s) / max(len(x) - 1, 1), 1e-12)
    expected_max_noise = norm.ppf(1.0 - 1.0 / max(n_trials, 2)) * np.sqrt(sr_var) if n_trials > 1 else 0.0
    dsr = (s - expected_max_noise) / np.sqrt(sr_var)
    return pd.DataFrame(
        [
            {
                "Metric": "Sortino",
                "Value": s,
            },
            {"Metric": "Deflated_Sortino", "Value": dsr},
            {"Metric": "Deflated_Sortino_PValue", "Value": 1.0 - norm.cdf(dsr)},
            {"Metric": "Trials_Adjustment", "Value": n_trials},
            {"Metric": "Sortino_Estimator_Var", "Value": sr_var},
            {"Metric": "Return_Skew", "Value": sk},
            {"Metric": "Return_Kurtosis", "Value": ku},
        ]
    )


def cpcv_pbo_diagnostics(opt_grid: pd.DataFrame, n_folds: int = 4) -> pd.DataFrame:
    if opt_grid is None or opt_grid.empty or not {"Rebalance_Date", "Tickers", "Sortino"}.issubset(opt_grid.columns):
        return pd.DataFrame()
    g = opt_grid.dropna(subset=["Rebalance_Date", "Tickers", "Sortino"]).copy()
    key_col = "Trial_Key" if "Trial_Key" in g.columns else "Tickers"
    oos_col = "OOS_Equal_Return" if "OOS_Equal_Return" in g.columns and g["OOS_Equal_Return"].notna().any() else "Sortino"
    g = g.dropna(subset=[oos_col])
    if g["Rebalance_Date"].nunique() < 4 or g[key_col].nunique() < 3:
        return pd.DataFrame()
    dates = np.array(sorted(pd.to_datetime(g["Rebalance_Date"]).unique()))
    n_folds = max(2, min(int(n_folds), len(dates)))
    folds = np.array_split(dates, n_folds)
    rows = []
    for fold_id, test_dates in enumerate(folds):
        test_dates = pd.to_datetime(test_dates)
        train = g[~pd.to_datetime(g["Rebalance_Date"]).isin(test_dates)]
        test = g[pd.to_datetime(g["Rebalance_Date"]).isin(test_dates)]
        if train.empty or test.empty:
            continue
        train_scores = train.groupby(key_col)["Sortino"].mean().dropna()
        test_scores = test.groupby(key_col)[oos_col].mean().dropna()
        common = train_scores.index.intersection(test_scores.index)
        if len(common) < 3:
            continue
        train_scores = train_scores.reindex(common)
        test_scores = test_scores.reindex(common)
        selected = train_scores.idxmax()
        test_rank_pct = test_scores.rank(pct=True).loc[selected]
        eps = 1e-6
        lambda_logit = np.log(np.clip(test_rank_pct, eps, 1 - eps) / np.clip(1.0 - test_rank_pct, eps, 1 - eps))
        rows.append(
            {
                "Fold": fold_id,
                "Train_Start": train["Rebalance_Date"].min(),
                "Train_End": train["Rebalance_Date"].max(),
                "Test_Start": test["Rebalance_Date"].min(),
                "Test_End": test["Rebalance_Date"].max(),
                "Selected_Trial": selected,
                "Train_Sortino": train_scores.loc[selected],
                "Test_OOS_Return": test_scores.loc[selected],
                "Test_Rank_Pct": test_rank_pct,
                "Lambda_Logit": lambda_logit,
                "PBO_Event": lambda_logit < 0,
                "OOS_Metric": oos_col,
                "N_Common_Trials": len(common),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["PBO"] = out["PBO_Event"].mean()
    return out


def white_reality_check_spa(
    opt_grid: pd.DataFrame,
    samples: int = 512,
    block_p: float = 0.35,
    seed: int = 20260509,
) -> pd.DataFrame:
    if opt_grid is None or opt_grid.empty or not {"Rebalance_Date", "Trial_Key", "OOS_Equal_Return"}.issubset(opt_grid.columns):
        return pd.DataFrame()
    g = opt_grid.dropna(subset=["Rebalance_Date", "Trial_Key", "OOS_Equal_Return"]).copy()
    if g["Rebalance_Date"].nunique() < 6 or g["Trial_Key"].nunique() < 3:
        return pd.DataFrame()
    pivot = g.pivot_table(index="Rebalance_Date", columns="Trial_Key", values="OOS_Equal_Return", aggfunc="mean").sort_index()
    pivot = pivot.dropna(axis=1, thresh=max(4, int(0.25 * len(pivot)))).fillna(0.0)
    if pivot.shape[0] < 6 or pivot.shape[1] < 3:
        return pd.DataFrame()
    x = pivot.values
    t, m = x.shape
    means = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    best_idx = int(np.nanargmax(means))
    best_trial = str(pivot.columns[best_idx])
    observed_wrc = float(np.nanmax(means))
    observed_spa = float(np.nanmax(np.sqrt(t) * means / np.maximum(std, 1e-12)))
    centered = x - means
    rng = np.random.default_rng(seed)
    wrc_stats, spa_stats = [], []
    for _ in range(max(int(samples), 1)):
        idx = []
        pos = int(rng.integers(0, t))
        for _j in range(t):
            if _j == 0 or rng.random() < block_p:
                pos = int(rng.integers(0, t))
            else:
                pos = (pos + 1) % t
            idx.append(pos)
        sample = centered[idx, :]
        sample_mean = sample.mean(axis=0)
        sample_std = sample.std(axis=0, ddof=1)
        wrc_stats.append(float(np.nanmax(sample_mean)))
        spa_stats.append(float(np.nanmax(np.sqrt(t) * sample_mean / np.maximum(sample_std, 1e-12))))
    wrc_stats = np.asarray(wrc_stats, dtype=float)
    spa_stats = np.asarray(spa_stats, dtype=float)
    return pd.DataFrame(
        [
            {"Metric": "Best_Trial", "Value": best_trial},
            {"Metric": "Observed_Best_Mean_OOS", "Value": observed_wrc},
            {"Metric": "White_Reality_Check_PValue", "Value": float((wrc_stats >= observed_wrc).mean())},
            {"Metric": "Observed_SPA_T", "Value": observed_spa},
            {"Metric": "Hansen_SPA_PValue", "Value": float((spa_stats >= observed_spa).mean())},
            {"Metric": "Reality_Check_Strategies", "Value": m},
            {"Metric": "Reality_Check_Periods", "Value": t},
        ]
    )


def trial_persistence_table(opt_grid: pd.DataFrame, min_obs: int = 4) -> pd.DataFrame:
    if opt_grid is None or opt_grid.empty or not {"Trial_Key", "Sortino", "OOS_Equal_Return"}.issubset(opt_grid.columns):
        return pd.DataFrame()
    g = opt_grid.dropna(subset=["Trial_Key", "Sortino", "OOS_Equal_Return"]).copy()
    if g.empty:
        return pd.DataFrame()
    rows = []
    for key, x in g.groupby("Trial_Key"):
        if len(x) < min_obs:
            continue
        is_rank = x["Sortino"].rank(pct=True)
        oos_rank = x["OOS_Equal_Return"].rank(pct=True)
        rank_corr = is_rank.corr(oos_rank, method="spearman") if len(x) >= 3 else np.nan
        rows.append(
            {
                "Trial_Key": key,
                "Obs": len(x),
                "Mean_IS_Sortino": x["Sortino"].mean(),
                "Mean_OOS_Return": x["OOS_Equal_Return"].mean(),
                "Median_OOS_Return": x["OOS_Equal_Return"].median(),
                "OOS_Hit_Rate": (x["OOS_Equal_Return"] > 0).mean(),
                "IS_OOS_Rank_Corr": rank_corr,
                "OOS_Return_Std": x["OOS_Equal_Return"].std(ddof=1),
                "Persistence_Score": x["OOS_Equal_Return"].mean() / (x["OOS_Equal_Return"].std(ddof=1) + 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values("Persistence_Score", ascending=False) if rows else pd.DataFrame()


def model_confidence_from_history(
    opt_grid: pd.DataFrame,
    window: int = 6,
    min_confidence: float = 0.25,
) -> tuple[float, dict]:
    if opt_grid is None or opt_grid.empty or not {"Rebalance_Date", "Sortino", "OOS_Equal_Return"}.issubset(opt_grid.columns):
        return 1.0, {"confidence_reason": "insufficient_history"}
    g = opt_grid.dropna(subset=["Rebalance_Date", "Sortino", "OOS_Equal_Return"]).copy()
    if g.empty or g["Rebalance_Date"].nunique() < 3:
        return 1.0, {"confidence_reason": "insufficient_history"}
    dates = sorted(pd.to_datetime(g["Rebalance_Date"]).unique())[-max(3, int(window)) :]
    recent = g[pd.to_datetime(g["Rebalance_Date"]).isin(dates)].copy()
    rank_rows = []
    for date, x in recent.groupby("Rebalance_Date"):
        if len(x) < 3:
            continue
        corr = x["Sortino"].rank(pct=True).corr(x["OOS_Equal_Return"].rank(pct=True), method="spearman")
        top = x.sort_values("Sortino", ascending=False).head(max(1, len(x) // 5))
        rank_rows.append(
            {
                "Rebalance_Date": date,
                "Rank_Corr": corr,
                "Top_IS_Mean_OOS": top["OOS_Equal_Return"].mean(),
                "OOS_Hit": (top["OOS_Equal_Return"] > 0).mean(),
            }
        )
    if not rank_rows:
        return 1.0, {"confidence_reason": "insufficient_rank_history"}
    d = pd.DataFrame(rank_rows)
    corr = d["Rank_Corr"].replace([np.inf, -np.inf], np.nan).mean()
    top_oos = d["Top_IS_Mean_OOS"].mean()
    hit = d["OOS_Hit"].mean()
    corr_score = 0.5 + 0.5 * np.nan_to_num(corr, nan=0.0)
    oos_score = 1.0 / (1.0 + np.exp(-25.0 * np.nan_to_num(top_oos, nan=0.0)))
    hit_score = np.nan_to_num(hit, nan=0.5)
    confidence = float(np.clip(0.40 * corr_score + 0.35 * oos_score + 0.25 * hit_score, min_confidence, 1.0))
    return confidence, {
        "confidence_reason": "rank_oos_history",
        "confidence_rank_corr": corr,
        "confidence_top_is_oos": top_oos,
        "confidence_oos_hit": hit,
        "confidence_observations": len(d),
    }


def moving_block_bootstrap(x: pd.Series, samples: int = 512, block: int = 3, seed: int = 20260507) -> pd.DataFrame:
    r = pd.Series(x).dropna().astype(float)
    if len(r) < max(6, block + 2) or samples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    vals = r.values
    rows = []
    for _ in range(samples):
        chunks = []
        while sum(len(c) for c in chunks) < len(vals):
            start = rng.integers(0, max(1, len(vals) - block + 1))
            chunks.append(vals[start : start + block])
        sample = pd.Series(np.concatenate(chunks)[: len(vals)])
        rows.append(
            {
                "Method": "moving_block",
                "Total_Return": (1 + sample).prod() - 1,
                "Mean_Return": sample.mean(),
                "Sortino": sortino_ratio(sample) / np.sqrt(252 / 12) if len(sample) > 2 else np.nan,
                "Max_Drawdown": ((1 + sample).cumprod() / (1 + sample).cumprod().cummax() - 1).min(),
            }
        )
    return pd.DataFrame(rows)


def stationary_bootstrap(x: pd.Series, samples: int = 512, p: float = 0.35, seed: int = 20260508) -> pd.DataFrame:
    r = pd.Series(x).dropna().astype(float)
    if len(r) < 6 or samples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    vals = r.values
    rows = []
    for _ in range(samples):
        idx = []
        j = rng.integers(0, len(vals))
        for _t in range(len(vals)):
            if rng.random() < p:
                j = rng.integers(0, len(vals))
            idx.append(j)
            j = (j + 1) % len(vals)
        sample = pd.Series(vals[idx])
        rows.append(
            {
                "Method": "stationary",
                "Total_Return": (1 + sample).prod() - 1,
                "Mean_Return": sample.mean(),
                "Sortino": sortino_ratio(sample) / np.sqrt(252 / 12) if len(sample) > 2 else np.nan,
                "Max_Drawdown": ((1 + sample).cumprod() / (1 + sample).cumprod().cummax() - 1).min(),
            }
        )
    return pd.DataFrame(rows)


def validation_diagnostics(
    perf: pd.DataFrame,
    holdings: pd.DataFrame,
    opt_grid: pd.DataFrame,
    samples: int = 512,
    cpcv_folds: int = 4,
    reality_check_samples: int = 512,
) -> dict[str, pd.DataFrame]:
    out = {
        "bootstrap": pd.DataFrame(),
        "ic": pd.DataFrame(),
        "rank_stability": pd.DataFrame(),
        "deflated_sortino": pd.DataFrame(),
        "cpcv_pbo": pd.DataFrame(),
        "reality_check_spa": pd.DataFrame(),
        "trial_persistence": pd.DataFrame(),
        "summary": pd.DataFrame(),
    }
    if perf is not None and not perf.empty and "Net_Return" in perf:
        block = moving_block_bootstrap(perf["Net_Return"], samples=samples)
        stat = stationary_bootstrap(perf["Net_Return"], samples=samples)
        out["bootstrap"] = pd.concat([block, stat], ignore_index=True) if not block.empty or not stat.empty else pd.DataFrame()
        out["deflated_sortino"] = deflated_sortino_diagnostics(perf["Net_Return"], n_trials=len(opt_grid) if opt_grid is not None else 1)
    if holdings is not None and not holdings.empty and {"Rebalance_Date", "Composite_Score", "OOS_Return"}.issubset(holdings.columns):
        rows = []
        for date, g in holdings.dropna(subset=["Composite_Score", "OOS_Return"]).groupby("Rebalance_Date"):
            if len(g) < 3:
                continue
            ic = g["Composite_Score"].corr(g["OOS_Return"], method="spearman")
            rows.append(
                {
                    "Rebalance_Date": date,
                    "N": len(g),
                    "IC_Spearman": ic,
                    "Mean_OOS_Return": g["OOS_Return"].mean(),
                    "Turnover_Adjusted_IC": ic / (1.0 + g["Weight"].abs().sum()) if pd.notna(ic) else np.nan,
                }
            )
        ic_df = pd.DataFrame(rows)
        if not ic_df.empty:
            ic_df["Rolling_IC_3"] = ic_df["IC_Spearman"].rolling(3, min_periods=1).mean()
            out["ic"] = ic_df
        stab = holdings.groupby("Ticker").agg(
            Selection_Count=("Ticker", "size"),
            Avg_Weight=("Weight", "mean"),
            Avg_Score=("Composite_Score", "mean"),
            Score_Std=("Composite_Score", "std"),
            Avg_OOS_Return=("OOS_Return", "mean"),
        ).reset_index()
        total_dates = max(holdings["Rebalance_Date"].nunique(), 1)
        stab["Selection_Frequency"] = stab["Selection_Count"] / total_dates
        stab["Rank_Stability"] = stab["Avg_Score"] / (1.0 + stab["Score_Std"].fillna(0.0).abs())
        out["rank_stability"] = stab.sort_values(["Selection_Frequency", "Rank_Stability"], ascending=False)
    summary_rows = []
    if not out["ic"].empty:
        ic = out["ic"]["IC_Spearman"].dropna()
        summary_rows.append({"Metric": "Mean_IC", "Value": ic.mean()})
        summary_rows.append({"Metric": "ICIR", "Value": ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else np.nan})
        summary_rows.append({"Metric": "Hit_Rate_IC_Positive", "Value": (ic > 0).mean()})
    if not out["bootstrap"].empty:
        for method, g in out["bootstrap"].groupby("Method"):
            summary_rows.append({"Metric": f"{method}_Total_Return_P05", "Value": g["Total_Return"].quantile(0.05)})
            summary_rows.append({"Metric": f"{method}_Total_Return_P50", "Value": g["Total_Return"].quantile(0.50)})
            summary_rows.append({"Metric": f"{method}_Total_Return_P95", "Value": g["Total_Return"].quantile(0.95)})
            summary_rows.append({"Metric": f"{method}_Sortino_P50", "Value": g["Sortino"].quantile(0.50)})
    if opt_grid is not None and not opt_grid.empty:
        out["cpcv_pbo"] = cpcv_pbo_diagnostics(opt_grid, n_folds=cpcv_folds)
        out["reality_check_spa"] = white_reality_check_spa(opt_grid, samples=reality_check_samples)
        out["trial_persistence"] = trial_persistence_table(opt_grid, min_obs=max(2, cpcv_folds))
        summary_rows.append({"Metric": "Rank_Trials_Total", "Value": len(opt_grid)})
        if "Sortino" in opt_grid:
            summary_rows.append({"Metric": "Rank_Trials_Sortino_IQR", "Value": opt_grid["Sortino"].quantile(0.75) - opt_grid["Sortino"].quantile(0.25)})
    if not out["deflated_sortino"].empty:
        for row in out["deflated_sortino"].to_dict(orient="records"):
            summary_rows.append({"Metric": row["Metric"], "Value": row["Value"]})
    if not out["cpcv_pbo"].empty:
        summary_rows.append({"Metric": "CPCV_PBO", "Value": out["cpcv_pbo"]["PBO"].iloc[-1]})
        summary_rows.append({"Metric": "CPCV_Mean_Test_Rank_Pct", "Value": out["cpcv_pbo"]["Test_Rank_Pct"].mean()})
    if not out["reality_check_spa"].empty:
        for row in out["reality_check_spa"].to_dict(orient="records"):
            summary_rows.append({"Metric": row["Metric"], "Value": row["Value"]})
    if not out["trial_persistence"].empty:
        summary_rows.append({"Metric": "Trial_Persistence_Top_Mean_OOS", "Value": out["trial_persistence"]["Mean_OOS_Return"].iloc[0]})
        summary_rows.append({"Metric": "Trial_Persistence_Top_Score", "Value": out["trial_persistence"]["Persistence_Score"].iloc[0]})
    if perf is not None and not perf.empty and "Model_Confidence" in perf:
        summary_rows.append({"Metric": "Model_Confidence_Last", "Value": perf["Model_Confidence"].dropna().iloc[-1] if perf["Model_Confidence"].notna().any() else np.nan})
        summary_rows.append({"Metric": "Model_Confidence_Mean", "Value": perf["Model_Confidence"].mean()})
    out["summary"] = pd.DataFrame(summary_rows)
    return out


def build_factor_returns(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    idx = prices.index.intersection(macro.index)
    if len(idx) == 0:
        return pd.DataFrame()
    factors = pd.DataFrame(index=idx)
    px = prices.reindex(idx).ffill()
    if "SPY" in px:
        factors["MKT"] = px["SPY"].pct_change(fill_method=None)
    else:
        factors["MKT"] = px.mean(axis=1).pct_change(fill_method=None)
    if "QQQ" in px and "SPY" in px:
        factors["GROWTH_MINUS_MKT"] = px["QQQ"].pct_change(fill_method=None) - px["SPY"].pct_change(fill_method=None)
    if "IWM" in px and "SPY" in px:
        factors["SIZE_MINUS_MKT"] = px["IWM"].pct_change(fill_method=None) - px["SPY"].pct_change(fill_method=None)
    if "US10Y" in macro:
        factors["RATES_10Y_D"] = macro.reindex(idx)["US10Y"].diff()
    if "Curve_10Y_2Y" in macro:
        factors["CURVE_D"] = macro.reindex(idx)["Curve_10Y_2Y"].diff()
    if "CREDIT_SPREAD" in macro:
        factors["CREDIT_D"] = macro.reindex(idx)["CREDIT_SPREAD"].diff()
    if "USD_BROAD" in macro:
        factors["USD_R"] = macro.reindex(idx)["USD_BROAD"].pct_change(fill_method=None)
    if "WTI" in macro:
        factors["OIL_R"] = macro.reindex(idx)["WTI"].pct_change(fill_method=None)
    return factors.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def estimate_asset_factor_betas(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list[str],
    asof_date,
    lookback: int = 252,
) -> pd.DataFrame:
    factors = build_factor_returns(prices, macro)
    if factors.empty:
        return pd.DataFrame(index=tickers)
    asset_ret = prices.loc[:asof_date, tickers].pct_change(fill_method=None)
    idx = asset_ret.index.intersection(factors.index)
    asset_ret = asset_ret.reindex(idx).tail(lookback)
    fac = factors.reindex(idx).tail(lookback).dropna(axis=1, how="all").fillna(0.0)
    if len(asset_ret) < 60 or fac.empty:
        return pd.DataFrame(index=tickers)
    x = np.column_stack([np.ones(len(fac)), fac.values])
    rows = {}
    for tk in tickers:
        y = asset_ret[tk].fillna(0.0).values
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        rows[tk] = dict(zip(fac.columns, beta[1:]))
    return pd.DataFrame.from_dict(rows, orient="index")


def factor_risk_attribution(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    portfolio: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame()
    tickers = [t for t in portfolio["Ticker"] if t in prices.columns]
    if not tickers:
        return pd.DataFrame()
    weights = portfolio.set_index("Ticker").reindex(tickers)["Weight"].fillna(0.0)
    weights = weights / weights.sum()
    factors = build_factor_returns(prices, macro)
    ret = prices[tickers].pct_change(fill_method=None)
    idx = ret.index.intersection(factors.index)
    y = (ret.reindex(idx).fillna(0.0) @ weights).tail(lookback)
    X = factors.reindex(y.index).tail(lookback).dropna(axis=1, how="all").fillna(0.0)
    if len(y) < 60 or X.empty:
        return pd.DataFrame()
    X_mat = np.column_stack([np.ones(len(X)), X.values])
    beta = np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T @ y.values
    fitted = X_mat @ beta
    residual = y.values - fitted
    rows = []
    for i, col in enumerate(X.columns, start=1):
        contrib = beta[i] * X[col]
        rows.append(
            {
                "Factor": col,
                "Beta": beta[i],
                "Ann_Contribution_Mean": contrib.mean() * 252,
                "Contribution_Vol": contrib.std(ddof=1) * np.sqrt(252),
            }
        )
    rows.append(
        {
            "Factor": "IDIOSYNCRATIC_RESIDUAL",
            "Beta": np.nan,
            "Ann_Contribution_Mean": residual.mean() * 252,
            "Contribution_Vol": residual.std(ddof=1) * np.sqrt(252),
        }
    )
    return pd.DataFrame(rows).sort_values("Contribution_Vol", ascending=False)


def factor_model_risk_decomposition(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    portfolio: pd.DataFrame,
    asof_date=None,
    lookback: int = 252,
) -> dict[str, pd.DataFrame]:
    empty = {
        "factor_model_asset_risk": pd.DataFrame(),
        "factor_model_factor_risk": pd.DataFrame(),
        "factor_model_covariance": pd.DataFrame(),
        "factor_model_betas": pd.DataFrame(),
        "factor_model_residual_variance": pd.DataFrame(),
    }
    if portfolio.empty or macro is None or macro.empty:
        return empty
    tickers = [t for t in portfolio["Ticker"] if t in prices.columns]
    if not tickers:
        return empty
    end_date = pd.Timestamp(asof_date) if asof_date is not None else prices.index.max()
    weights = portfolio.set_index("Ticker").reindex(tickers)["Weight"].fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()
    asset_ret = prices.loc[:end_date, tickers].tail(lookback + 1).pct_change(fill_method=None).dropna(how="all")
    factors = build_factor_returns(prices.loc[:end_date], macro.loc[:end_date])
    idx = asset_ret.index.intersection(factors.index)
    asset_ret = asset_ret.reindex(idx).tail(lookback).fillna(0.0)
    factors = factors.reindex(idx).tail(lookback).dropna(axis=1, how="all").fillna(0.0)
    if len(asset_ret) < 60 or factors.empty:
        return empty
    sigma_fm, betas, resid_var = factor_model_covariance_matrix(asset_ret, factors)
    if sigma_fm.empty:
        return empty
    weights = weights.reindex(sigma_fm.index).fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()
    sigma = sigma_fm.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    sigma_values = sigma.values
    w = weights.values
    port_var = float(w @ sigma_values @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sigma_w = sigma_values @ w
    asset_rows = []
    for i, tk in enumerate(weights.index):
        variance_contribution = float(w[i] * sigma_w[i])
        asset_rows.append(
            {
                "Ticker": tk,
                "Weight": float(w[i]),
                "Marginal_Contribution_To_Risk": float(sigma_w[i] / max(port_vol, 1e-12)),
                "Component_Risk": float(variance_contribution / max(port_vol, 1e-12)),
                "Pct_Total_Variance": float(variance_contribution / max(port_var, 1e-12)),
                "Standalone_Ann_Vol": float(np.sqrt(max(sigma_values[i, i], 0.0))),
                "Residual_Ann_Variance": float(resid_var.reindex(weights.index).fillna(0.0).iloc[i]),
            }
        )
    factor_cov = ledoit_wolf_cov(factors)
    beta_port = betas.reindex(index=weights.index, columns=factor_cov.columns).fillna(0.0).T @ weights
    factor_var = float(beta_port.values @ factor_cov.values @ beta_port.values)
    factor_sigma_beta = factor_cov.values @ beta_port.values
    factor_rows = []
    for j, factor in enumerate(factor_cov.columns):
        var_contribution = float(beta_port.iloc[j] * factor_sigma_beta[j])
        factor_rows.append(
            {
                "Risk_Block": "FACTOR",
                "Name": factor,
                "Portfolio_Beta": float(beta_port.iloc[j]),
                "Variance_Contribution": var_contribution,
                "Pct_Total_Variance": float(var_contribution / max(port_var, 1e-12)),
                "Pct_Factor_Variance": float(var_contribution / max(factor_var, 1e-12)),
            }
        )
    idio_var = float(np.sum(np.square(w) * resid_var.reindex(weights.index).fillna(0.0).values))
    factor_rows.append(
        {
            "Risk_Block": "IDIOSYNCRATIC",
            "Name": "SPECIFIC_RISK",
            "Portfolio_Beta": np.nan,
            "Variance_Contribution": idio_var,
            "Pct_Total_Variance": float(idio_var / max(port_var, 1e-12)),
            "Pct_Factor_Variance": np.nan,
        }
    )
    summary = pd.DataFrame(
        [
            {"Risk_Block": "TOTAL", "Name": "PORTFOLIO_VOL", "Portfolio_Beta": np.nan, "Variance_Contribution": port_var, "Pct_Total_Variance": 1.0, "Pct_Factor_Variance": np.nan},
            {"Risk_Block": "TOTAL", "Name": "FACTOR_SHARE", "Portfolio_Beta": np.nan, "Variance_Contribution": factor_var, "Pct_Total_Variance": factor_var / max(port_var, 1e-12), "Pct_Factor_Variance": np.nan},
            {"Risk_Block": "TOTAL", "Name": "SPECIFIC_SHARE", "Portfolio_Beta": np.nan, "Variance_Contribution": idio_var, "Pct_Total_Variance": idio_var / max(port_var, 1e-12), "Pct_Factor_Variance": np.nan},
        ]
    )
    factor_risk = pd.concat([summary, pd.DataFrame(factor_rows)], ignore_index=True)
    betas_out = betas.reset_index().rename(columns={"index": "Ticker"})
    resid_out = resid_var.rename("Residual_Ann_Variance").reset_index().rename(columns={"index": "Ticker"})
    return {
        "factor_model_asset_risk": pd.DataFrame(asset_rows).sort_values("Pct_Total_Variance", ascending=False),
        "factor_model_factor_risk": factor_risk.sort_values("Pct_Total_Variance", ascending=False),
        "factor_model_covariance": sigma.reset_index().rename(columns={"index": "Ticker"}),
        "factor_model_betas": betas_out,
        "factor_model_residual_variance": resid_out,
    }


def oos_factor_attribution(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    perf: pd.DataFrame,
    holdings: pd.DataFrame,
    benchmark: str = "SPY",
    lookback: int = 252,
) -> pd.DataFrame:
    if perf is None or perf.empty or holdings is None or holdings.empty or macro is None or macro.empty:
        return pd.DataFrame()
    factors = build_factor_returns(prices, macro)
    if factors.empty:
        return pd.DataFrame()
    rows = []
    for _, period in perf.iterrows():
        signal_date = pd.Timestamp(period.get("Signal_Date"))
        start = pd.Timestamp(period.get("OOS_Start"))
        end = pd.Timestamp(period.get("OOS_End"))
        h = holdings[pd.to_datetime(holdings["Rebalance_Date"]).eq(pd.Timestamp(period.get("Rebalance_Date")))].copy()
        if h.empty:
            continue
        tickers = [t for t in h["Ticker"] if t in prices.columns]
        if not tickers:
            continue
        weight_col = "Effective_Weight" if "Effective_Weight" in h.columns else "Weight"
        weights = h.set_index("Ticker").reindex(tickers)[weight_col].fillna(0.0)
        betas = estimate_asset_factor_betas(prices, macro, tickers, signal_date, lookback=lookback)
        factor_period = factors.loc[(factors.index > start) & (factors.index <= end)].fillna(0.0)
        if betas.empty or factor_period.empty:
            continue
        factor_returns = factor_period.sum()
        beta_port = betas.reindex(index=tickers, columns=factor_returns.index).fillna(0.0).T @ weights.reindex(tickers).fillna(0.0)
        factor_contrib = beta_port * factor_returns
        factor_total = float(factor_contrib.sum())
        gross = to_float(period.get("Gross_Return"))
        net = to_float(period.get("Net_Return"))
        cost = to_float(period.get("Transaction_Cost"))
        benchmark_ret = np.nan
        if benchmark in prices.columns:
            try:
                benchmark_ret = float(prices.loc[end, benchmark] / prices.loc[start, benchmark] - 1.0)
            except Exception:
                benchmark_ret = np.nan
        base = {
            "Rebalance_Date": period.get("Rebalance_Date"),
            "Period_End": end,
            "Regime": period.get("Regime"),
            "Latent_Regime_State": period.get("Latent_Regime_State"),
            "Dynamic_Exposure": period.get("Dynamic_Exposure"),
            "Gross_Return": gross,
            "Net_Return": net,
            "Benchmark_Return": benchmark_ret,
            "Active_Return": net - benchmark_ret if pd.notna(net) and pd.notna(benchmark_ret) else np.nan,
            "Factor_Return": factor_total,
            "Specific_Selection_Return": gross - factor_total if pd.notna(gross) else np.nan,
            "Cost_Return": -cost if pd.notna(cost) else np.nan,
        }
        rows.append({**base, "Component": "FACTOR_TOTAL", "Contribution": factor_total})
        rows.append({**base, "Component": "SPECIFIC_SELECTION", "Contribution": base["Specific_Selection_Return"]})
        rows.append({**base, "Component": "TRANSACTION_COST", "Contribution": base["Cost_Return"]})
        for factor, value in factor_contrib.items():
            rows.append({**base, "Component": f"FACTOR_{factor}", "Contribution": float(value), "Portfolio_Beta": float(beta_port.get(factor, np.nan)), "Factor_Period_Return": float(factor_returns.get(factor, np.nan))})
    return pd.DataFrame(rows)


def regime_conditioned_performance(perf: pd.DataFrame) -> pd.DataFrame:
    if perf is None or perf.empty or "Net_Return" not in perf:
        return pd.DataFrame()
    rows = []
    for col in ["Regime", "Latent_Regime_State"]:
        if col not in perf.columns:
            continue
        for state, g in perf.dropna(subset=[col]).groupby(col):
            r = g["Net_Return"].astype(float).dropna()
            if r.empty:
                continue
            equity = (1.0 + r).cumprod()
            var95, cvar95 = historical_var_cvar(r, 0.95)
            rows.append(
                {
                    "Regime_Type": col,
                    "State": state,
                    "Periods": len(r),
                    "Mean_Return": r.mean(),
                    "Total_Return": equity.iloc[-1] - 1.0,
                    "Ann_Return_Approx": r.mean() * 12.0,
                    "Ann_Vol_Approx": r.std(ddof=1) * np.sqrt(12.0) if len(r) > 1 else np.nan,
                    "Sortino_Approx": sortino_ratio(r, mar=0.0) / np.sqrt(252 / 12) if len(r) > 2 else np.nan,
                    "Max_Drawdown": (equity / equity.cummax() - 1.0).min(),
                    "Hit_Rate": (r > 0).mean(),
                    "Avg_Turnover": g["Turnover"].mean() if "Turnover" in g else np.nan,
                    "Avg_Dynamic_Exposure": g["Dynamic_Exposure"].mean() if "Dynamic_Exposure" in g else np.nan,
                    "VaR_95": var95,
                    "CVaR_95": cvar95,
                }
            )
    return pd.DataFrame(rows).sort_values(["Regime_Type", "Total_Return"], ascending=[True, False])


def event_driven_capital_ledger(perf: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.DataFrame:
    if perf is None or perf.empty:
        return pd.DataFrame()
    capital = float(initial_capital)
    rows = []
    for _, row in perf.sort_values("OOS_Start").iterrows():
        begin = capital
        exposure = to_float(row.get("Dynamic_Exposure"))
        exposure = 1.0 if pd.isna(exposure) else exposure
        invested = begin * exposure
        cash = begin - invested
        gross_pnl = begin * to_float(row.get("Gross_Return"))
        cost = begin * to_float(row.get("Transaction_Cost"))
        net_pnl = gross_pnl - cost
        capital = begin + net_pnl
        rows.append(
            {
                "Signal_Date": row.get("Signal_Date"),
                "Order_Date": row.get("Rebalance_Date"),
                "Fill_Date": row.get("OOS_Start"),
                "Period_End": row.get("OOS_End"),
                "Begin_Capital": begin,
                "Dynamic_Exposure": exposure,
                "Invested_Capital": invested,
                "Cash_Capital": cash,
                "Gross_PnL": gross_pnl,
                "Transaction_Cost_Cash": cost,
                "Net_PnL": net_pnl,
                "End_Capital": capital,
                "Capital_Return": capital / begin - 1.0 if begin else np.nan,
                "Drawdown": capital / max([initial_capital] + [r["End_Capital"] for r in rows]) - 1.0 if rows else min(capital / initial_capital - 1.0, 0.0),
                "Regime": row.get("Regime"),
                "Latent_Regime_State": row.get("Latent_Regime_State"),
            }
        )
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger["Peak_Capital"] = ledger["End_Capital"].cummax()
        ledger["Drawdown"] = ledger["End_Capital"] / ledger["Peak_Capital"] - 1.0
    return ledger


def portfolio_stress_tests(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    portfolio: pd.DataFrame,
    asof_date=None,
    lookback: int = 252,
) -> pd.DataFrame:
    if portfolio.empty or macro is None or macro.empty:
        return pd.DataFrame()
    tickers = [t for t in portfolio["Ticker"] if t in prices.columns]
    if not tickers:
        return pd.DataFrame()
    end_date = pd.Timestamp(asof_date) if asof_date is not None else prices.index.max()
    weights = portfolio.set_index("Ticker").reindex(tickers)["Weight"].fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()
    betas = estimate_asset_factor_betas(prices, macro, tickers, end_date, lookback=lookback)
    if betas.empty:
        return pd.DataFrame()
    beta_port = betas.reindex(index=tickers).fillna(0.0).T @ weights
    scenarios = {
        "Equity shock -2 sigma": {"MKT": -0.06, "GROWTH_MINUS_MKT": -0.02, "CREDIT_D": 0.25, "USD_R": 0.015},
        "Rates parallel up": {"RATES_10Y_D": 0.75, "CURVE_D": 0.00, "MKT": -0.025},
        "Bear steepener": {"RATES_10Y_D": 0.50, "CURVE_D": 0.60, "CREDIT_D": 0.15, "MKT": -0.020},
        "Credit widening": {"CREDIT_D": 0.75, "MKT": -0.050, "USD_R": 0.020},
        "Oil shock up": {"OIL_R": 0.12, "MKT": -0.010},
        "USD squeeze": {"USD_R": 0.05, "MKT": -0.020, "CREDIT_D": 0.20},
        "Risk-on melt-up": {"MKT": 0.050, "GROWTH_MINUS_MKT": 0.020, "CREDIT_D": -0.20, "USD_R": -0.010},
    }
    rows = []
    for scenario, shocks in scenarios.items():
        contribution = {}
        total = 0.0
        for factor, shock in shocks.items():
            beta = float(beta_port.get(factor, 0.0))
            value = beta * shock
            contribution[f"{factor}_Contribution"] = value
            total += value
        rows.append({"Scenario": scenario, "Estimated_Portfolio_Return": total, **contribution})
    return pd.DataFrame(rows).sort_values("Estimated_Portfolio_Return")


def portfolio_hedge_suggestions(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    portfolio: pd.DataFrame,
    benchmark: str = "SPY",
    portfolio_notional: float = 100_000.0,
    asof_date=None,
    lookback: int = 252,
) -> pd.DataFrame:
    if portfolio.empty or macro is None or macro.empty:
        return pd.DataFrame()
    tickers = [t for t in portfolio["Ticker"] if t in prices.columns]
    if not tickers:
        return pd.DataFrame()
    end_date = pd.Timestamp(asof_date) if asof_date is not None else prices.index.max()
    weights = portfolio.set_index("Ticker").reindex(tickers)["Weight"].fillna(0.0)
    if weights.sum() != 0:
        weights = weights / weights.sum()
    factors = build_factor_returns(prices.loc[:end_date], macro.loc[:end_date])
    asset_ret = prices.loc[:end_date, tickers].pct_change(fill_method=None)
    idx = asset_ret.index.intersection(factors.index)
    y = (asset_ret.reindex(idx).fillna(0.0) @ weights).tail(lookback)
    X = factors.reindex(y.index).tail(lookback).dropna(axis=1, how="all").fillna(0.0)
    if len(y) < 60 or X.empty:
        return pd.DataFrame()
    x_mat = np.column_stack([np.ones(len(X)), X.values])
    beta = np.linalg.pinv(x_mat.T @ x_mat) @ x_mat.T @ y.values
    beta_map = dict(zip(X.columns, beta[1:]))
    notional = float(max(to_float(portfolio_notional), 1.0))
    hedge_specs = {
        "MKT": {
            "Shock": -0.06,
            "Instrument": f"{benchmark} put spread / short {benchmark}",
            "Hedge_When": "beta > 0",
            "Action_Positive": "Buy protective index puts, put-spread collar, or reduce net equity beta.",
            "Action_Negative": "Short squeeze hedge: buy benchmark calls or reduce inverse exposure.",
        },
        "GROWTH_MINUS_MKT": {
            "Shock": -0.03,
            "Instrument": "QQQ put spread / QQQ-SPY relative hedge",
            "Hedge_When": "growth beta > 0",
            "Action_Positive": "Buy QQQ puts or short QQQ against SPY to neutralize growth duration.",
            "Action_Negative": "Use QQQ calls if portfolio is structurally short growth.",
        },
        "RATES_10Y_D": {
            "Shock": 0.75,
            "Instrument": "TBT / short TLT / payer swaption proxy",
            "Hedge_When": "rates-up loss contribution < 0",
            "Action_Positive": "Yields up help this portfolio; hedge opposite convexity with TLT calls only if needed.",
            "Action_Negative": "Use TBT, short TLT, or payer swaptions to hedge duration/rates-up losses.",
        },
        "CURVE_D": {
            "Shock": 0.60,
            "Instrument": "2s10s steepener/flattener proxy",
            "Hedge_When": "curve shock loss contribution < 0",
            "Action_Positive": "Curve steepening helps; hedge flattening with duration barbell if needed.",
            "Action_Negative": "Use flattener proxy or reduce long-duration equity exposure.",
        },
        "CREDIT_D": {
            "Shock": 0.75,
            "Instrument": "HYG/LQD puts or short high-yield proxy",
            "Hedge_When": "credit-widening loss contribution < 0",
            "Action_Positive": "Credit widening helps; verify this is not a statistical artifact.",
            "Action_Negative": "Buy HYG/LQD puts or reduce credit-beta equities.",
        },
        "OIL_R": {
            "Shock": 0.12,
            "Instrument": "XLE/USO options",
            "Hedge_When": "oil shock loss contribution < 0",
            "Action_Positive": "Oil up helps; hedge oil-down risk with XLE puts if energy concentration dominates.",
            "Action_Negative": "Buy XLE/USO calls or hold energy proxy to hedge oil-up inflation shock.",
        },
        "USD_R": {
            "Shock": 0.05,
            "Instrument": "UUP / FX hedge proxy",
            "Hedge_When": "USD squeeze loss contribution < 0",
            "Action_Positive": "USD up helps; hedge USD-down with currency diversification if required.",
            "Action_Negative": "Use UUP/currency hedge proxy for USD-squeeze risk.",
        },
    }
    rows = []
    for factor, spec in hedge_specs.items():
        if factor not in beta_map:
            continue
        b = float(beta_map.get(factor, np.nan))
        shock = float(spec["Shock"])
        contribution = b * shock
        loss = min(contribution, 0.0)
        hedge_notional = min(max(abs(loss) * notional / 0.05, 0.0), notional) if loss < 0 else 0.0
        if factor in {"MKT", "GROWTH_MINUS_MKT"}:
            hedge_notional = min(abs(b) * notional, notional) if contribution < 0 else 0.0
        action = spec["Action_Positive"] if b >= 0 else spec["Action_Negative"]
        priority = "High" if loss <= -0.05 else "Medium" if loss <= -0.02 else "Low"
        rows.append(
            {
                "Factor": factor,
                "Portfolio_Beta": b,
                "Stress_Shock": shock,
                "Estimated_Return_Contribution": contribution,
                "Estimated_Loss_USD": abs(loss) * notional,
                "Priority": priority,
                "Suggested_Hedge_Instrument": spec["Instrument"],
                "Indicative_Hedge_Notional_USD": hedge_notional,
                "Hedge_Action": action,
                "Hedge_Objective": "Reduce left-tail loss and stabilize CVaR, not maximize standalone hedge PnL.",
                "Implementation_Risk": "ETF/options proxies have basis risk, theta decay, liquidity constraints, and model beta instability.",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_Priority_Order"] = out["Priority"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(3)
    return out.sort_values(["_Priority_Order", "Estimated_Loss_USD"], ascending=[True, False]).drop(columns=["_Priority_Order"]).reset_index(drop=True)


def decision_attribution(cs: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    if cs is None or cs.empty:
        return pd.DataFrame()
    selected = set(portfolio["Ticker"].astype(str)) if portfolio is not None and not portfolio.empty else set()
    rows = []
    for _, row in cs.iterrows():
        reasons = []
        if bool(row.get("Fundamental_Gate", False)):
            reasons.append("fundamental_gate_pass")
        else:
            reasons.append("fundamental_gate_fail")
        if to_float(row.get("Dollar_Volume_63")) < 1_000_000:
            reasons.append("liquidity_reject")
        if to_float(row.get("Prob_Alpha_Positive")) >= 0.60:
            reasons.append("posterior_alpha_positive")
        elif pd.notna(to_float(row.get("Prob_Alpha_Positive"))) and to_float(row.get("Prob_Alpha_Positive")) < 0.40:
            reasons.append("posterior_alpha_weak")
        if to_float(row.get("EVT_CVaR_95")) > to_float(cs.get("EVT_CVaR_95", pd.Series(dtype=float)).median()):
            reasons.append("tail_risk_high")
        if to_float(row.get("GARCH_Vol_Forecast")) > to_float(cs.get("GARCH_Vol_Forecast", pd.Series(dtype=float)).median()):
            reasons.append("conditional_vol_high")
        if to_float(row.get("Mahalanobis")) > 3.0:
            reasons.append("sector_outlier")
        reject_text = row.get("Reject_Reasons")
        if isinstance(reject_text, str) and reject_text:
            reasons.extend([f"reject_{x}" for x in reject_text.split(";") if x])
        rows.append(
            {
                "Ticker": row.get("Ticker"),
                "Sector": row.get("Sector"),
                "Decision": "INCLUDED" if row.get("Ticker") in selected else "EXCLUDED",
                "Weight": portfolio.set_index("Ticker")["Weight"].get(row.get("Ticker"), 0.0) if portfolio is not None and not portfolio.empty else 0.0,
                "Composite_Score": row.get("Composite_Score"),
                "Prob_Alpha_Positive": row.get("Prob_Alpha_Positive"),
                "Bayesian_Posterior_Confidence": row.get("Bayesian_Posterior_Confidence"),
                "Fundamental_Gate": row.get("Fundamental_Gate"),
                "Reject_Reasons": row.get("Reject_Reasons"),
                "Decision_Reasons": ";".join(dict.fromkeys(reasons)),
            }
        )
    return pd.DataFrame(rows).sort_values(["Decision", "Composite_Score"], ascending=[False, False])


def population_stability_index(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) < bins or len(b) < bins:
        return np.nan
    cuts = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return np.nan
    a_counts = pd.cut(a, bins=cuts, include_lowest=True).value_counts(normalize=True).sort_index()
    b_counts = pd.cut(b, bins=cuts, include_lowest=True).value_counts(normalize=True).sort_index()
    eps = 1e-6
    return float(((b_counts + eps) - (a_counts + eps)).mul(np.log((b_counts + eps) / (a_counts + eps))).sum())


def monitoring_diagnostics(cs: pd.DataFrame, opt_grid: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not cs.empty:
        for col in ["Composite_Score", "Composite_Score_Raw", "Quality_Score", "Value_Score", "GARCH_Vol_Forecast", "CRLB_Mu"]:
            if col in cs and cs[col].notna().sum() > 20:
                x = cs[col].dropna()
                rows.append({"Metric": f"PSI_{col}_first_vs_second_half", "Value": population_stability_index(x.iloc[: len(x)//2], x.iloc[len(x)//2:])})
        if "Ranking_Shannon_Entropy" in cs:
            rows.append({"Metric": "Ranking_Shannon_Entropy", "Value": cs["Ranking_Shannon_Entropy"].dropna().iloc[0] if cs["Ranking_Shannon_Entropy"].notna().any() else np.nan})
    if not opt_grid.empty and "Sortino" in opt_grid:
        rows.append({"Metric": "Optimization_Trials", "Value": len(opt_grid)})
        rows.append({"Metric": "Optimization_Sortino_Median", "Value": opt_grid["Sortino"].median()})
        rows.append({"Metric": "Optimization_Sortino_Std", "Value": opt_grid["Sortino"].std(ddof=1)})
    if not perf.empty:
        rows.append({"Metric": "Turnover_Adjusted_Return", "Value": perf["Net_Return"].sum() / max(perf["Turnover"].sum(), 1e-9) if "Turnover" in perf else np.nan})
        if "Best_Sortino_IS" in perf:
            rows.append({"Metric": "IS_OOS_Return_Correlation", "Value": perf["Best_Sortino_IS"].corr(perf["Net_Return"])})
    return pd.DataFrame(rows)


KAIZEN_ACTIONS = [
    {"Action_ID": "sortino_balanced", "Objective": "sortino", "lambda_cvar": 0.35, "w_max": 0.20, "sector_cap": 0.35, "target_vol": 0.15, "N": 10},
    {"Action_ID": "sortino_defensive", "Objective": "sortino", "lambda_cvar": 0.65, "w_max": 0.12, "sector_cap": 0.25, "target_vol": 0.10, "N": 8},
    {"Action_ID": "cvar_min_tail", "Objective": "cvar_min", "lambda_cvar": 0.90, "w_max": 0.12, "sector_cap": 0.25, "target_vol": 0.09, "N": 8},
    {"Action_ID": "hrp_uncertain_alpha", "Objective": "hrp", "lambda_cvar": 0.45, "w_max": 0.18, "sector_cap": 0.35, "target_vol": 0.14, "N": 12},
    {"Action_ID": "risk_parity_stable", "Objective": "risk_parity", "lambda_cvar": 0.45, "w_max": 0.16, "sector_cap": 0.30, "target_vol": 0.13, "N": 12},
    {"Action_ID": "black_litterman_quality", "Objective": "black_litterman", "lambda_cvar": 0.30, "w_max": 0.22, "sector_cap": 0.40, "target_vol": 0.16, "N": 10},
    {"Action_ID": "information_ratio_relative", "Objective": "information_ratio", "lambda_cvar": 0.30, "w_max": 0.18, "sector_cap": 0.35, "target_vol": 0.15, "N": 10},
    {"Action_ID": "mean_variance_growth", "Objective": "mean_variance", "lambda_cvar": 0.20, "w_max": 0.30, "sector_cap": 0.50, "target_vol": 0.22, "N": 14},
    {"Action_ID": "min_variance_low_vol", "Objective": "min_variance", "lambda_cvar": 0.70, "w_max": 0.12, "sector_cap": 0.25, "target_vol": 0.08, "N": 8},
]


def _metric_value(df: pd.DataFrame | dict | None, metric: str) -> float:
    if df is None:
        return np.nan
    if isinstance(df, dict):
        df = df.get("summary", pd.DataFrame())
    if not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan
    if {"Metric", "Value"}.issubset(df.columns):
        s = df.loc[df["Metric"].astype(str).eq(metric), "Value"]
        return to_float(s.iloc[0]) if not s.empty else np.nan
    return np.nan


def kaizen_state_vector(config: RunConfig, latest_macro: pd.Series, perf: pd.DataFrame, validation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if perf is not None and not perf.empty:
        r = pd.to_numeric(perf.get("Net_Return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        equity = (1.0 + r).cumprod()
        dd = equity / equity.cummax() - 1.0 if not equity.empty else pd.Series(dtype=float)
        current_dd = float(dd.iloc[-1]) if not dd.empty else np.nan
        max_dd = float(dd.min()) if not dd.empty else np.nan
        var95, cvar95 = historical_var_cvar(r, 0.95)
        turnover = to_float(perf.get("Turnover", pd.Series(dtype=float)).tail(3).mean())
        model_conf = to_float(perf.get("Model_Confidence", pd.Series(dtype=float)).dropna().tail(1).mean())
    else:
        current_dd = max_dd = cvar95 = turnover = model_conf = np.nan
    state = {
        "hawkish_score": to_float(latest_macro.get("hawkish_score")),
        "bullish_score": to_float(latest_macro.get("bullish_score")),
        "latent_entropy": to_float(latest_macro.get("Latent_State_Entropy")),
        "markov_stress_prob": to_float(latest_macro.get("Markov_Stress_Prob")),
        "model_confidence": model_conf,
        "rolling_ic": _metric_value(validation, "Mean_IC"),
        "pbo": _metric_value(validation, "CPCV_PBO"),
        "current_drawdown": current_dd,
        "max_drawdown": max_dd,
        "cvar_95": cvar95,
        "turnover": turnover,
        "suitability_score": config.suitability_score,
        "profile_conservative": 1.0 if config.suitability_profile == "Conservador" else 0.0,
        "profile_balanced": 1.0 if config.suitability_profile == "Balanceado" else 0.0,
        "profile_aggressive": 1.0 if config.suitability_profile == "Agresivo" else 0.0,
        "profile_speculative": 1.0 if config.suitability_profile == "Especulativo" else 0.0,
    }
    return pd.DataFrame([state]).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def kaizen_reward_series(perf: pd.DataFrame, config: RunConfig, action_id: str | None = None, run_hash: str | None = None) -> pd.DataFrame:
    if perf is None or perf.empty:
        return pd.DataFrame()
    r = pd.to_numeric(perf.get("Net_Return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    downside = r.clip(upper=0.0).abs()
    rolling_cvar = downside.rolling(6, min_periods=1).mean()
    equity = (1.0 + r).cumprod()
    drawdown = (equity / equity.cummax() - 1.0).abs()
    turnover = pd.to_numeric(perf.get("Turnover", pd.Series(0.0, index=perf.index)), errors="coerce").fillna(0.0)
    cost = pd.to_numeric(perf.get("Transaction_Cost", pd.Series(0.0, index=perf.index)), errors="coerce").fillna(0.0)
    dd_limit = max(float(config.investor_max_drawdown), 1e-6)
    suitability_breach = (drawdown > dd_limit).astype(float)
    reward = (
        r
        - config.kaizen_reward_cvar_lambda * rolling_cvar
        - config.kaizen_reward_drawdown_lambda * np.maximum(drawdown - dd_limit, 0.0)
        - config.kaizen_reward_turnover_lambda * turnover
        - config.kaizen_reward_cost_lambda * cost
        - 0.25 * suitability_breach
    )
    out = pd.DataFrame(
        {
            "run_hash": run_hash,
            "Signal_Date": perf.get("Signal_Date"),
            "OOS_End": perf.get("OOS_End", perf.get("Period_End")),
            "Action_ID": action_id or f"{config.weight_objective}_current",
            "Objective": config.weight_objective,
            "Reward": reward,
            "Net_Return": r,
            "Rolling_CVaR_Proxy": rolling_cvar,
            "Drawdown_Abs": drawdown,
            "Turnover": turnover,
            "Cost": cost,
            "Suitability_Breach": suitability_breach,
            "Profile": config.suitability_profile,
        }
    )
    return out


def kaizen_promotion_gate(perf_summary: pd.DataFrame, validation: dict[str, pd.DataFrame], config: RunConfig) -> pd.DataFrame:
    dsr = _metric_value(validation, "Deflated_Sortino")
    pbo = _metric_value(validation, "CPCV_PBO")
    spa_p = _metric_value(validation, "Hansen_SPA_PValue")
    max_dd = _metric_value(perf_summary, "Max_Drawdown")
    gates = {
        "DSR_Positive": bool(pd.notna(dsr) and dsr > 0.0),
        "PBO_Below_Limit": bool(pd.isna(pbo) or pbo < 0.20),
        "SPA_Not_Rejected": bool(pd.isna(spa_p) or spa_p < 0.10),
        "Drawdown_Within_Suitability": bool(pd.isna(max_dd) or abs(max_dd) <= max(float(config.investor_max_drawdown), 1e-6)),
        "Suitability_Not_Blocked": not bool(config.suitability_hard_block),
    }
    row = {
        "Deflated_Sortino_Z": dsr,
        "PBO": pbo,
        "SPA_p_value": spa_p,
        "Max_Drawdown": max_dd,
        **gates,
        "Promotion_Gate_Passed": all(gates.values()),
    }
    return pd.DataFrame([row])


def load_kaizen_history(limit: int | None = None) -> pd.DataFrame:
    path = MODEL_REGISTRY_DIR / "rl_state_action_rewards.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        return df.tail(limit).reset_index(drop=True) if limit else df
    except Exception:
        return pd.DataFrame()


def persist_kaizen_rewards(rewards: pd.DataFrame) -> None:
    if rewards is None or rewards.empty:
        return
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_REGISTRY_DIR / "rl_state_action_rewards.parquet"
    old = load_kaizen_history()
    out = pd.concat([old, rewards], ignore_index=True) if not old.empty else rewards.copy()
    subset = [c for c in ["run_hash", "Signal_Date", "OOS_End", "Action_ID"] if c in out.columns]
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")
    out.to_parquet(path, index=False)


def _kaizen_feature_columns(history: pd.DataFrame, state: pd.DataFrame) -> list[str]:
    preferred = [
        "hawkish_score", "bullish_score", "latent_entropy", "markov_stress_prob", "model_confidence",
        "rolling_ic", "pbo", "current_drawdown", "max_drawdown", "cvar_95", "turnover", "suitability_score",
        "profile_conservative", "profile_balanced", "profile_aggressive", "profile_speculative",
    ]
    return [c for c in preferred if c in history.columns and c in state.columns]


def kaizen_contextual_bandit_diagnostics(
    config: RunConfig,
    latest_macro: pd.Series,
    perf: pd.DataFrame,
    validation: dict[str, pd.DataFrame],
    perf_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    state = kaizen_state_vector(config, latest_macro, perf, validation)
    current_rewards = kaizen_reward_series(perf, config)
    history = load_kaizen_history()
    matching_actions = [a["Action_ID"] for a in KAIZEN_ACTIONS if a["Objective"] == config.weight_objective]
    current_action = matching_actions[0] if matching_actions else f"{config.weight_objective}_current"
    if not current_rewards.empty:
        enriched = current_rewards.copy()
        for col, val in state.iloc[0].items():
            enriched[col] = val
        enriched["Action_ID"] = current_action
        history_fit = pd.concat([history, enriched], ignore_index=True) if not history.empty else enriched
        current_rewards = enriched
    else:
        history_fit = history
    rows = []
    state_x = state.iloc[0]
    base_conf = float(np.clip(state_x.get("model_confidence", 0.0), 0.0, 1.0))
    stress = float(np.clip(state_x.get("markov_stress_prob", 0.0), 0.0, 1.0))
    for action in KAIZEN_ACTIONS:
        action_id = action["Action_ID"]
        if not history_fit.empty and "Action_ID" in history_fit:
            h = history_fit[history_fit["Action_ID"].astype(str).eq(action_id)]
        else:
            h = pd.DataFrame()
        n = len(h)
        mean_reward = pd.to_numeric(h.get("Reward", pd.Series(dtype=float)), errors="coerce").mean() if n else np.nan
        lin_pred = np.nan
        ucb_bonus = config.kaizen_ucb_alpha / math.sqrt(n + 1.0)
        feature_cols = _kaizen_feature_columns(h, state)
        if n >= max(4, len(feature_cols) + 1) and feature_cols:
            X = h[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            y = pd.to_numeric(h["Reward"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            A = np.eye(X.shape[1]) + X.T @ X
            b = X.T @ y
            inv_a = np.linalg.pinv(A)
            theta = inv_a @ b
            x = state[feature_cols].to_numpy(dtype=float).reshape(-1)
            lin_pred = float(x @ theta)
            ucb_bonus = float(config.kaizen_ucb_alpha * math.sqrt(max(x @ inv_a @ x, 0.0)))
        heuristic = 0.0
        if config.suitability_profile == "Conservador" and action["Objective"] in {"min_variance", "cvar_min", "risk_parity", "hrp"}:
            heuristic += 0.015
        if config.suitability_profile == "Especulativo" and action["Objective"] in {"mean_variance", "black_litterman", "sortino"}:
            heuristic += 0.010
        if stress > 0.50 and action["Objective"] in {"cvar_min", "min_variance", "risk_parity", "hrp"}:
            heuristic += 0.020
        if base_conf < 0.40 and action["Objective"] in {"hrp", "risk_parity", "min_variance"}:
            heuristic += 0.020
        score = (lin_pred if pd.notna(lin_pred) else (mean_reward if pd.notna(mean_reward) else 0.0)) + ucb_bonus + heuristic
        rows.append({**action, "N_Obs": n, "Mean_Reward": mean_reward, "LinUCB_Prediction": lin_pred, "UCB_Bonus": ucb_bonus, "Heuristic": heuristic, "Bandit_Score": score})
    actions = pd.DataFrame(rows).sort_values("Bandit_Score", ascending=False).reset_index(drop=True)
    if not actions.empty:
        actions["Recommended"] = False
        actions.loc[0, "Recommended"] = True
    gate = kaizen_promotion_gate(perf_summary, validation, config)
    regime_matrix = pd.DataFrame()
    if not history_fit.empty and {"Profile", "Objective", "Reward"}.issubset(history_fit.columns):
        regime_matrix = (
            history_fit.pivot_table(index="Profile", columns="Objective", values="Reward", aggfunc="mean")
            .reset_index()
        )
    return {
        "state": state,
        "actions": actions,
        "current_rewards": current_rewards,
        "promotion_gate": gate,
        "history_tail": history_fit.tail(250).reset_index(drop=True) if not history_fit.empty else pd.DataFrame(),
        "regime_objective_matrix": regime_matrix,
    }


class QuantDataEngine:
    def __init__(self, config: RunConfig):
        self.config = config

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        invest_tickers = tuple(dict.fromkeys([t for t in self.config.tickers if t != self.config.benchmark_ticker]))
        side_tickers = normalize_side_tickers(self.config.side_boom_tickers) if self.config.use_side_boom_portfolio else tuple()
        price_tickers = tuple(dict.fromkeys(list(invest_tickers) + [self.config.benchmark_ticker] + list(side_tickers)))
        prices = download_prices(
            price_tickers,
            self.config.price_period,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        if prices.empty:
            raise ValueError("Prices could not be downloaded.")
        volumes = download_volume(
            price_tickers,
            self.config.price_period,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        valid_tickers = tuple([t for t in prices.columns if t in invest_tickers])
        panel = build_fundamental_panel(
            valid_tickers,
            self.config.accounting_lag_days,
            max_workers=self.config.max_workers,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        if self.config.use_sec_edgar:
            sec_panel = build_sec_companyfacts_panel(
                valid_tickers,
                self.config.accounting_lag_days,
                max_workers=self.config.max_workers,
                use_cache=self.config.use_persistent_cache,
                cache_ttl_hours=max(self.config.cache_ttl_hours, 168),
                user_agent=self.config.sec_user_agent,
            )
            panel = merge_fundamental_sources(panel, sec_panel)
            if self.config.use_sec_nlp:
                try:
                    sec_nlp = build_sec_nlp_panel(
                        valid_tickers,
                        max_tickers=self.config.sec_nlp_max_tickers,
                        max_filings=self.config.sec_nlp_max_filings,
                        max_workers=self.config.max_workers,
                        use_cache=self.config.use_persistent_cache,
                        cache_ttl_hours=max(self.config.cache_ttl_hours, 168),
                        user_agent=self.config.sec_user_agent,
                    )
                    panel = merge_sec_nlp_into_panel(panel, sec_nlp)
                except Exception:
                    pass
        if panel.empty:
            raise ValueError("The causal fundamental panel could not be built.")
        return prices, panel, volumes


class RegimeAnalyzer:
    def __init__(self, config: RunConfig):
        self.config = config

    def fit(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return market_regime(
            prices,
            country=self.config.rate_country,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
            use_latent_macro_regime=self.config.use_latent_macro_regime,
            latent_regime_states=self.config.latent_regime_states,
            latent_regime_min_train=self.config.latent_regime_min_train,
            latent_regime_refit_days=self.config.latent_regime_refit_days,
            markov_transition_min_obs=self.config.markov_transition_min_obs,
        )


class AlphaResearchEngine:
    def __init__(self, config: RunConfig):
        self.config = config

    def score(self, panel: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, latest_macro: pd.Series) -> pd.DataFrame:
        return score_cross_section(
            panel,
            prices,
            latest_macro,
            prices.index.max(),
            volumes=volumes,
            use_garch=self.config.use_garch,
            garch_candidate_n=self.config.garch_candidate_n,
            crlb_penalty=self.config.crlb_penalty,
            garch_penalty=self.config.garch_penalty,
            evt_penalty=self.config.evt_penalty,
            text_risk_penalty=self.config.text_risk_penalty,
            min_dollar_volume=self.config.min_dollar_volume,
        )


class PortfolioOptimizer:
    def __init__(self, config: RunConfig):
        self.config = config

    @property
    def factor_caps(self) -> dict[str, float]:
        return {
            "MKT": self.config.factor_mkt_cap,
            "RATES_10Y_D": self.config.factor_rates_cap,
            "CREDIT_D": self.config.factor_credit_cap,
            "OIL_R": self.config.factor_oil_cap,
            "USD_R": self.config.factor_usd_cap,
        }

    def current_portfolio(self, cs: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return optimize_chunks(
            cs,
            prices,
            prices.index.max(),
            macro=macro,
            lookback=126,
            min_chunk=self.config.min_chunk,
            max_chunk=self.config.max_chunk,
            preselect_n=self.config.preselect_n,
            max_combos=self.config.max_combos,
            max_names_per_sector=self.config.max_names_per_sector,
            max_weight=self.config.max_weight,
            sector_weight_cap=self.config.sector_weight_cap,
            risk_aversion=self.config.risk_aversion,
            alpha_weight=self.config.alpha_weight,
            weight_objective=self.config.weight_objective,
            entropy_penalty=self.config.entropy_penalty,
            crlb_penalty=self.config.crlb_penalty,
            garch_penalty=self.config.garch_penalty,
            evt_penalty=self.config.evt_penalty,
            cvar_penalty=self.config.cvar_penalty,
            cvar_alpha=self.config.cvar_alpha,
            robust_alpha_uncertainty=self.config.robust_alpha_uncertainty,
            robust_cov_uncertainty=self.config.robust_cov_uncertainty,
            factor_cov_blend=self.config.factor_cov_blend,
            use_black_litterman=self.config.use_black_litterman,
            black_litterman_tau=self.config.black_litterman_tau,
            portfolio_notional=self.config.portfolio_notional,
            max_adv_participation=self.config.max_adv_participation,
            target_vol=self.config.target_vol,
            nested_validation_fraction=self.config.nested_validation_fraction,
            purge_days=self.config.purge_days,
            bootstrap_samples=self.config.bootstrap_samples,
            factor_caps=self.factor_caps,
            multistarts=self.config.sortino_multistarts,
        )


class BacktestEngine:
    def __init__(self, config: RunConfig):
        self.config = config

    def run(self, prices: pd.DataFrame, panel: pd.DataFrame, volumes: pd.DataFrame, macro: pd.DataFrame):
        return backtest(prices, panel, volumes, macro, self.config)


class QuantStockPickerPipeline:
    def __init__(self, config: RunConfig):
        self.config = config
        self.data = QuantDataEngine(config)
        self.regime = RegimeAnalyzer(config)
        self.alpha = AlphaResearchEngine(config)
        self.optimizer = PortfolioOptimizer(config)
        self.backtester = BacktestEngine(config)

    def run(self) -> dict[str, pd.DataFrame | pd.Series | dict]:
        timings = {}
        t0 = time.perf_counter()
        prices, panel, volumes = self.data.load()
        timings["data_load_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        macro, latest_macro = self.regime.fit(prices)
        timings["macro_regime_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cs = self.alpha.score(panel, prices, volumes, latest_macro)
        timings["alpha_score_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        portfolio, options = self.optimizer.current_portfolio(cs, prices, macro)
        timings["current_optimization_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        perf, holdings, opt_grid = self.backtester.run(prices, panel, volumes, macro)
        timings["walk_forward_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        option_tickers = portfolio["Ticker"].tolist() if not portfolio.empty else cs.head(self.config.preselect_n)["Ticker"].tolist()
        options_chain = (
            fetch_options_snapshot(
                option_tickers,
                prices,
                max_expiries=self.config.option_expiries,
                max_workers=self.config.max_workers,
                use_cache=self.config.use_persistent_cache,
                cache_ttl_hours=self.config.options_cache_ttl_hours,
            )
            if self.config.use_options_snapshot
            else pd.DataFrame()
        )
        options_summary = summarize_options_snapshot(options_chain)
        portfolio_vol_surface = portfolio_implied_vol_surface(options_chain, portfolio)
        timings["options_snapshot_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        validation = validation_diagnostics(
            perf,
            holdings,
            opt_grid,
            samples=self.config.validation_bootstrap_samples,
            cpcv_folds=self.config.cpcv_folds,
            reality_check_samples=self.config.reality_check_samples,
        )
        timings["validation_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        alt_data = alternative_data_diagnostics(
            macro,
            use_gdelt=self.config.use_gdelt,
            gdelt_query=self.config.gdelt_query,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
            use_forex_factory=self.config.use_forex_factory_calendar,
            forex_factory_cache_ttl_hours=self.config.forex_factory_cache_ttl_hours,
        )
        timings["alternative_data_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        global_rates = global_yield_curve_snapshot(
            prices,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        global_rate_history = global_yield_curve_discrete_history(
            prices,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        interbank_reference_rates = fetch_interbank_reference_rates(
            prices.index.min() - pd.Timedelta(days=30),
            prices.index.max() + pd.Timedelta(days=5),
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        carry_trade = carry_trade_suggestions(
            global_rates,
            alt_data.get("forex_factory_event_risk", pd.DataFrame()) if isinstance(alt_data, dict) else pd.DataFrame(),
        )
        carry_trade_validation = validate_carry_trade_strategies(
            carry_trade,
            global_rates,
            use_cache=self.config.use_persistent_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
        )
        timings["global_yield_curves_sec"] = time.perf_counter() - t0
        performance_summary = summarize_backtest(perf, prices, benchmark=self.config.benchmark_ticker)
        equity_curve = portfolio_vs_benchmark_curve(perf, prices, benchmark=self.config.benchmark_ticker)
        t0 = time.perf_counter()
        if self.config.use_side_boom_portfolio:
            side_boom_portfolio, side_boom_curve, side_boom_diagnostics_df = optimize_side_boom_portfolio(
                prices,
                self.config,
                macro=macro,
                lookback=max(self.config.lookback_grid) if self.config.lookback_grid else 252,
            )
            side_boom_wf_curve, side_boom_wf_holdings, side_boom_wf_diagnostics = side_boom_walk_forward(
                prices,
                self.config,
                macro=macro,
                perf=perf,
                lookback=max(self.config.lookback_grid) if self.config.lookback_grid else 252,
            )
            if not side_boom_wf_diagnostics.empty:
                side_boom_diagnostics_df = side_boom_wf_diagnostics
            equity_curve = merge_side_boom_into_equity_curve(equity_curve, side_boom_curve)
            equity_curve = merge_side_boom_into_equity_curve(equity_curve, side_boom_wf_curve)
            side_boom_pelt = side_boom_pelt_diagnostics(
                side_boom_wf_curve if not side_boom_wf_curve.empty and len(side_boom_wf_curve) >= 63 else side_boom_curve,
                source_label="Private Side Alpha walk-forward" if not side_boom_wf_curve.empty and len(side_boom_wf_curve) >= 63 else "Private Side Alpha current allocation",
            )
        else:
            side_boom_portfolio, side_boom_curve, side_boom_diagnostics_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            side_boom_wf_curve, side_boom_wf_holdings = pd.DataFrame(), pd.DataFrame()
            side_boom_pelt = {
                "side_boom_pelt_regime_segments": pd.DataFrame(),
                "side_boom_pelt_change_points": pd.DataFrame(),
                "side_boom_pelt_timeline": pd.DataFrame(),
            }
        timings["side_boom_portfolio_sec"] = time.perf_counter() - t0
        oos_attr = oos_factor_attribution(prices, macro, perf, holdings, benchmark=self.config.benchmark_ticker)
        capital_ledger = event_driven_capital_ledger(perf, initial_capital=self.config.portfolio_notional)
        t0 = time.perf_counter()
        kaizen_diagnostics = (
            kaizen_contextual_bandit_diagnostics(self.config, latest_macro, perf, validation, performance_summary)
            if self.config.use_kaizen_bandit
            else {
                "state": pd.DataFrame(),
                "actions": pd.DataFrame(),
                "current_rewards": pd.DataFrame(),
                "promotion_gate": pd.DataFrame(),
                "history_tail": pd.DataFrame(),
                "regime_objective_matrix": pd.DataFrame(),
            }
        )
        timings["kaizen_bandit_sec"] = time.perf_counter() - t0
        timings["total_sec"] = sum(v for k, v in timings.items() if k != "total_sec")
        benchmark_governance = benchmark_governance_diagnostics(
            self.config.benchmark_ticker,
            self.config.benchmark_group,
            self.config.benchmark_mandate_type,
            self.config.rate_country,
            self.config.investor_objective,
            self.config.weight_objective,
            self.config.tickers,
            cross_section=cs,
        )
        suitability_diagnostics = pd.DataFrame(
            [
                {
                    "Suitability_Mode": self.config.suitability_mode,
                    "Suitability_Profile": self.config.suitability_profile,
                    "Suitability_Score": self.config.suitability_score,
                    "Horizon_Years": self.config.investor_horizon_years,
                    "Initial_Capital": self.config.investor_initial_capital,
                    "Monthly_Contribution": self.config.investor_monthly_contribution,
                    "Liquidity_Need": self.config.investor_liquidity_need,
                    "User_Max_Drawdown": self.config.investor_max_drawdown,
                    "Risk_Aversion_Score": self.config.investor_risk_aversion_score,
                    "Investor_Objective": self.config.investor_objective,
                    "Base_Currency": self.config.investor_base_currency,
                    "Hard_Block": self.config.suitability_hard_block,
                    "Warnings": " | ".join(self.config.suitability_warnings),
                    "Applied_Target_Vol": self.config.target_vol,
                    "Applied_Max_Weight": self.config.max_weight,
                    "Applied_Sector_Cap": self.config.sector_weight_cap,
                    "Applied_Top_N": self.config.top_n,
                    "Applied_Max_ADV_Participation": self.config.max_adv_participation,
                }
            ]
        )
        model_registry = build_model_registry_record(
            self.config,
            prices,
            panel,
            macro,
            cs,
            portfolio,
            perf,
            performance_summary,
            benchmark_governance,
            suitability_diagnostics,
            validation,
            timings,
        )
        try:
            persist_model_registry_record(model_registry)
        except Exception as exc:
            model_registry["registry_persist_error"] = str(exc)
        try:
            rewards = kaizen_diagnostics.get("current_rewards", pd.DataFrame()).copy()
            if not rewards.empty:
                rewards["run_hash"] = model_registry.get("run_hash")
                state = kaizen_diagnostics.get("state", pd.DataFrame())
                if not state.empty:
                    for col, val in state.iloc[0].items():
                        rewards[col] = val
                persist_kaizen_rewards(rewards)
                kaizen_diagnostics["current_rewards"] = rewards
        except Exception as exc:
            kaizen_diagnostics["persist_error"] = pd.DataFrame([{"Error": str(exc)}])
        return_diag = portfolio_return_diagnostics(
            prices,
            portfolio,
            macro=macro,
            initial_value=self.config.portfolio_notional,
            forecast_horizon_days=252,
            forecast_sims=3000,
        )
        stress_tests_df = portfolio_stress_tests(prices, macro, portfolio)
        hedge_suggestions = portfolio_hedge_suggestions(
            prices,
            macro,
            portfolio,
            benchmark=self.config.benchmark_ticker,
            portfolio_notional=self.config.portfolio_notional,
        )
        backtest_path_bundle = build_backtest_path_bundle(
            perf,
            holdings,
            prices,
            self.config.benchmark_ticker,
            equity_curve=equity_curve,
        )
        suitability_gate = evaluate_suitability_gate(self.config, portfolio, performance_summary, suitability_diagnostics)
        promotion_gate = evaluate_promotion_gate(performance_summary, validation, self.config)
        cache_inventory = PERSISTENT_CACHE.inventory()
        data_freshness_report = build_data_freshness_report(cache_inventory)
        result = {
            "prices": prices,
            "volumes": volumes,
            "fundamental_panel": panel,
            "macro": macro,
            "latest_macro": latest_macro,
            "cross_section": cs,
            "portfolio": portfolio,
            "side_boom_portfolio": side_boom_portfolio,
            "side_boom_curve": side_boom_curve,
            "side_boom_walk_forward": side_boom_wf_curve,
            "side_boom_holdings": side_boom_wf_holdings,
            "side_boom_diagnostics": side_boom_diagnostics_df,
            **side_boom_pelt,
            "portfolio_options": options,
            "options_chain": options_chain,
            "options_summary": options_summary,
            "portfolio_vol_surface": portfolio_vol_surface.get("portfolio_vol_surface", pd.DataFrame()),
            "portfolio_vol_surface_matrix": portfolio_vol_surface.get("portfolio_vol_surface_matrix", pd.DataFrame()),
            "portfolio_vol_surface_diagnostics": portfolio_vol_surface.get("portfolio_vol_surface_diagnostics", pd.DataFrame()),
            "backtest_perf": perf,
            "backtest_holdings": holdings,
            "optimization_grid": opt_grid,
            "performance_summary": performance_summary,
            "equity_curve": equity_curve,
            "return_diagnostics": return_diag,
            "oos_factor_attribution": oos_attr,
            "capital_ledger": capital_ledger,
            "regime_performance": regime_conditioned_performance(perf),
            "stress_tests": stress_tests_df,
            "hedge_suggestions": hedge_suggestions,
            "decision_attribution": decision_attribution(cs, portfolio),
            "overfit_diagnostics": overfit_diagnostics(opt_grid),
            "factor_attribution": factor_risk_attribution(prices, macro, portfolio),
            "monitoring_diagnostics": monitoring_diagnostics(cs, opt_grid, perf),
            "validation_diagnostics": validation,
            "latent_regime_diagnostics": latent_regime_diagnostics(macro),
            "alternative_data": alt_data,
            "global_yield_curves": global_rates,
            "global_rate_history": global_rate_history,
            "interbank_reference_rates": interbank_reference_rates,
            "carry_trade_suggestions": carry_trade,
            "carry_trade_validation": carry_trade_validation,
            "benchmark_governance": benchmark_governance,
            "suitability_diagnostics": suitability_diagnostics,
            "suitability_gate": suitability_gate,
            "promotion_gate": promotion_gate,
            "backtest_path_bundle": backtest_path_bundle,
            "model_registry": pd.DataFrame([registry_json_safe(model_registry)]),
            "kaizen_diagnostics": kaizen_diagnostics,
            "sector_diagnostics": sector_diagnostics(prices, cs),
            "rejection_diagnostics": rejection_diagnostics(cs),
            "cache_inventory": cache_inventory,
            "data_freshness_report": data_freshness_report,
            "timings": pd.DataFrame([{"Stage": k, "Seconds": v} for k, v in timings.items()]),
        }
        result["dashboard_payload"] = build_dashboard_payload(
            result,
            backtest_path_bundle,
            suitability_gate,
            promotion_gate,
            freshness_report=data_freshness_report,
        )
        return result


def run_pipeline(config: RunConfig) -> dict[str, pd.DataFrame | pd.Series | dict]:
    return QuantStockPickerPipeline(config).run()

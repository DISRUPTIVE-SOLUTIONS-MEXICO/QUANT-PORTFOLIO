from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional visualization fallback.
    px = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional degraded UI.
    TfidfVectorizer = None
    cosine_similarity = None

from quant_stockpicker_core import (
    DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT,
    DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS,
    DEFAULT_SIDE_ALPHA_TICKERS,
    RunConfig,
    benchmark_governance_diagnostics,
    build_suitability_constraints,
    download_prices,
    geopolitical_thermometer,
    global_yield_curve_snapshot,
    market_regime,
    market_sentiment_sem,
    fetch_forex_factory_calendar,
    fetch_interbank_reference_rates,
    forex_factory_event_risk,
    add_english_article_titles,
    geopolitical_country_heatmap,
    geopolitical_thermometer_model_audit,
    carry_trade_suggestions,
    global_yield_curve_discrete_history,
    normalize_side_tickers,
    optimize_side_boom_portfolio,
    suggest_benchmark,
    load_public_universe,
    run_pipeline,
)
from quant_dashboard_utils import (
    balanced_rate_history_sample,
    build_drawdown_frame,
    country_flag,
    latest_rate_observations,
    prepare_discrete_rate_plot_data,
    prepare_global_curve_matrix,
    spread_label_positions,
)

try:
    from supabase_store import (
        list_runs,
        list_user_portfolios,
        load_run_bundle,
        load_user_portfolio,
        save_run_to_supabase,
        supabase_available,
    )
except Exception:
    list_runs = list_user_portfolios = load_run_bundle = load_user_portfolio = save_run_to_supabase = None

    def supabase_available():
        return False

try:
    from cloud_jobs import latest_dashboard_artifact, latest_dashboard_artifacts
except Exception:
    latest_dashboard_artifact = None
    latest_dashboard_artifacts = None


DEFAULT_UNIVERSE = """
AAPL MSFT NVDA META GOOGL AMZN ORCL CRM AMD QCOM
JPM BAC WFC GS MS BLK SCHW C
XOM CVX COP SLB EOG MPC
LLY JNJ MRK ABBV AMGN TMO DHR ISRG
VRTX GEHC REGN
PG KO PEP WMT COST MDLZ
HD LOW MCD NKE SBUX BKNG
CAT DE HON GE RTX LMT UPS UNP
LIN APD SHW ECL NEM FCX
NEE DUK SO AEP XEL VST CEG SMR ED
PLD AMT EQIX DLR O WELL
SPY QQQ IWM DIA EEM VEA ACWI
"""

PROJECT_RESEARCH_ARTIFACT_DIR = Path(__file__).resolve().with_name("research_artifacts")
RESEARCH_ARTIFACT_DIR = Path(
    os.environ.get("QPK_RESEARCH_ARTIFACT_DIR", str(PROJECT_RESEARCH_ARTIFACT_DIR))
)
RESEARCH_PREFERRED_OBJECTIVES = (
    "enhanced_growth_anchor_dd_budget_policy",
    "fundamental_upside_convex_anchor_dd_budget_policy",
    "downside_preserving_growth_policy",
    "capital_preservation_policy",
)

DASHBOARD_UI_SCHEMA_VERSION = "2026.06.08-market-intelligence-v5"
APP_BUILD_ID = "2026.06.08-complete-dashboard-v5"

BENCHMARK_PRESETS = {
    "US Market": {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "DIA": "Dow Jones"},
    "US Sector": {
        "XLK": "Technology", "XLV": "Health Care", "XLU": "Utilities", "XLRE": "Real Estate",
        "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials", "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples", "XLB": "Materials",
    },
    "Country": {
        "EWW": "Mexico", "EWC": "Canada", "EWZ": "Brazil", "EWU": "United Kingdom",
        "EWG": "Germany", "EWQ": "France", "EWP": "Spain", "EWJ": "Japan",
        "MCHI": "China", "INDA": "India", "EWA": "Australia",
    },
    "International": {"ACWI": "Global ACWI", "VT": "Total World", "EFA": "Developed ex-US", "VEA": "Developed ex-US", "EEM": "Emerging Markets"},
}

BENCHMARK_COUNTRY = {
    "SPY": "United States", "QQQ": "United States", "IWM": "United States", "DIA": "United States",
    "XLK": "United States", "XLV": "United States", "XLU": "United States", "XLRE": "United States",
    "XLE": "United States", "XLF": "United States", "XLI": "United States", "XLY": "United States",
    "XLP": "United States", "XLB": "United States",
    "EWW": "Mexico", "EWC": "Canada", "EWZ": "Brazil", "EWU": "United Kingdom",
    "EWG": "Germany", "EWQ": "France", "EWP": "Spain", "EWJ": "Japan",
}

RISK_PRESETS = {
    "Conservador": dict(max_weight=0.12, sector_weight_cap=0.25, risk_aversion=9.0, alpha_weight=0.6, target_vol=0.10, entropy_penalty=0.18, crlb_penalty=0.30, garch_penalty=0.24, evt_penalty=0.28, max_sector=2, impact_coefficient=0.16, factor_cov_blend=0.75),
    "Balanceado": dict(max_weight=0.20, sector_weight_cap=0.35, risk_aversion=5.0, alpha_weight=1.0, target_vol=0.15, entropy_penalty=0.08, crlb_penalty=0.18, garch_penalty=0.12, evt_penalty=0.14, max_sector=3, impact_coefficient=0.10, factor_cov_blend=0.50),
    "Agresivo": dict(max_weight=0.30, sector_weight_cap=0.50, risk_aversion=2.5, alpha_weight=1.6, target_vol=0.22, entropy_penalty=0.04, crlb_penalty=0.10, garch_penalty=0.06, evt_penalty=0.08, max_sector=4, impact_coefficient=0.07, factor_cov_blend=0.35),
    "Especulativo": dict(max_weight=0.40, sector_weight_cap=0.65, risk_aversion=1.2, alpha_weight=2.2, target_vol=0.30, entropy_penalty=0.02, crlb_penalty=0.06, garch_penalty=0.04, evt_penalty=0.05, max_sector=5, impact_coefficient=0.05, factor_cov_blend=0.20),
}

PROFILE_EN = {
    "Conservador": "Conservative",
    "Balanceado": "Balanced",
    "Agresivo": "Aggressive",
    "Especulativo": "Speculative",
}

OBJECTIVE_LABELS = {
    "Sortino downside": "sortino",
    "Sharpe total risk": "sharpe",
    "Treynor beta-adjusted": "treynor",
    "Information ratio vs benchmark": "information_ratio",
    "Mean-variance utility": "mean_variance",
    "Minimum variance": "min_variance",
    "Minimum CVaR": "cvar_min",
    "Risk parity": "risk_parity",
    "Hierarchical Risk Parity": "hrp",
    "Black-Litterman posterior": "black_litterman",
    "Max return penalized": "max_return",
}

OBJECTIVE_HELP = {
    "sortino": "Maximizes return per downside deviation; default for non-financial users because it penalizes losses more than upside volatility.",
    "sharpe": "Maximizes return per total volatility; useful when upside and downside are treated symmetrically.",
    "treynor": "Maximizes return per benchmark beta; useful when systematic risk is explicitly accepted.",
    "information_ratio": "Maximizes active return per tracking error; useful for relative benchmark mandates.",
    "mean_variance": "Robust Markowitz utility: alpha minus variance aversion and risk penalties.",
    "min_variance": "Seeks the least volatile constrained portfolio; conservative profile.",
    "cvar_min": "Minimizes expected tail loss; defensive/tail-risk profile.",
    "risk_parity": "Balances risk contributions; robust when alpha is uncertain.",
    "hrp": "Uses hierarchical clustering and recursive bisection; robust when mean estimates are noisy.",
    "black_litterman": "Uses posterior alpha as Bayesian views shrunk toward an implied equilibrium prior.",
    "max_return": "Prioritizes penalized expected return; aggressive and less stable.",
}


st.set_page_config(
    page_title="Quant Portfolio-Kaizen",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hardened security headers (CSP, no-sniff, referrer policy, noindex).
from security import (
    inject_security_headers,
    enforce_pipeline_quota,
    sanitize_ticker_list,
    audit as audit_event,
)
inject_security_headers()

# Load typography first (separate call so a font CDN hiccup doesn't kill CSS).
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# IMPORTANT: keep this <style> block free of blank lines and angle-bracket comparisons
# (e.g. "<= 767px"). Streamlit's markdown engine sees a blank line as the end of an
# HTML block and "<= 767px" as a malformed tag — both cause the CSS to render as text.
_QPK_CSS = """
<style>
:root {
    --qpk-bg: #07080c;
    --qpk-panel: #0b101a;
    --qpk-panel-2: #0f1624;
    --qpk-line: rgba(148, 163, 184, 0.18);
    --qpk-line-strong: rgba(125, 211, 252, 0.34);
    --qpk-text: #eef3fb;
    --qpk-muted: #a8b3c7;
    --qpk-faint: #6b7689;
    --qpk-accent: #7dd3fc;
    --qpk-accent-2: #22d3ee;
    --qpk-positive: #4ade80;
    --qpk-warn: #fbbf24;
    --qpk-negative: #f87171;
    --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    --font-mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, monospace;
}
    /* Apply Inter only at the document root and let inheritance do the work.
       Critical: do NOT use `!important` on a wildcard `span`/`div` selector.
       Streamlit icon spans declare `font-family: 'Material Symbols Outlined'`
       and that rule MUST win; otherwise icons render as literal text such as
       `keyboard_arrow_right` or `check`. */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stMarkdown, p, label {
        font-family: var(--font-sans);
    }
    /* Defensive guard for Material Symbols / Material Icons spans. */
    .material-icons,
    .material-icons-outlined,
    .material-icons-rounded,
    .material-icons-sharp,
    .material-symbols-outlined,
    .material-symbols-rounded,
    .material-symbols-sharp,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: "Material Symbols Outlined", "Material Symbols Rounded", "Material Icons", "Material Icons Outlined" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        font-style: normal !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, var(--qpk-bg) 28%, #03050a 100%) !important;
        color: var(--qpk-text) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(5, 7, 13, 0.72) !important;
        backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(125, 211, 252, 0.14);
    }
    .block-container {
        padding-top: 1.05rem;
        padding-bottom: 2.4rem;
        padding-left: 1.15rem;
        padding-right: 1.15rem;
        max-width: 1760px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b12 0%, #090f1a 100%) !important;
        border-right: 1px solid var(--qpk-line);
        max-width: 360px !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--qpk-text);
    }
    h1, h2, h3 {
        letter-spacing: 0 !important;
        color: var(--qpk-text) !important;
        font-weight: 650 !important;
    }
    h1 {
        font-size: 2.05rem !important;
        margin-bottom: 0.15rem !important;
    }
    h2, h3 {
        margin-top: 0.95rem !important;
    }
    p, label, span, div {
        letter-spacing: 0 !important;
    }
    .qpk-hero {
        border-bottom: 1px solid rgba(125, 211, 252, 0.22);
        background: rgba(7, 11, 18, 0.34);
        padding: 12px 2px 14px 2px;
        margin: 0 0 14px 0;
        position: relative;
        overflow: hidden;
    }
    .qpk-hero:before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--qpk-accent), transparent);
    }
    .qpk-kicker {
        color: var(--qpk-accent);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em !important;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .qpk-title {
        font-size: 1.78rem;
        font-weight: 720;
        line-height: 1.08;
        color: var(--qpk-text);
    }
    .qpk-subtitle {
        color: var(--qpk-muted);
        max-width: 1080px;
        margin-top: 6px;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .qpk-meta {
        color: var(--qpk-faint);
        margin-top: 8px;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em !important;
    }
    .qpk-ops-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px 18px;
        padding: 10px 12px;
        margin: 0 0 12px 0;
        border: 1px solid var(--qpk-line);
        border-left: 3px solid var(--qpk-positive);
        background: rgba(11, 16, 26, 0.72);
    }
    .qpk-ops-title {
        color: var(--qpk-text);
        font-size: 0.86rem;
        font-weight: 650;
    }
    .qpk-ops-meta {
        color: var(--qpk-muted);
        font-size: 0.78rem;
        line-height: 1.4;
    }
    .qpk-login-brand {
        max-width: 520px;
        margin: clamp(36px, 9vh, 92px) auto 14px auto;
        padding: 0 2px;
    }
    .qpk-login-brand-title {
        color: var(--qpk-text);
        font-size: clamp(1.55rem, 4vw, 2rem);
        font-weight: 720;
        line-height: 1.1;
    }
    .qpk-login-brand-copy {
        color: var(--qpk-muted);
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 8px;
    }
    body:has(.qpk-login-brand) div[data-testid="stForm"] {
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }
    body:has(.qpk-login-brand) div[data-testid="stForm"] .stButton > button {
        width: 100%;
    }
    body:has(.qpk-login-brand) button[data-testid="stBaseButton-secondaryFormSubmit"] {
        width: 100% !important;
        min-height: 46px !important;
        border-color: rgba(125, 211, 252, 0.58) !important;
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.28), rgba(8, 47, 73, 0.34)) !important;
        color: var(--qpk-text) !important;
        font-weight: 650 !important;
    }
    body:has(.qpk-login-brand) div[data-testid="stFormSubmitButton"] {
        width: 100% !important;
    }
    body:has(.qpk-login-brand) div[data-testid="stElementContainer"]:has(div[data-testid="stFormSubmitButton"]),
    body:has(.qpk-login-brand) div[data-testid="stElementContainer"]:has(div[data-testid="stFormSubmitButton"]) > div {
        width: 100% !important;
    }
    body:has(.qpk-login-brand) div[data-testid="stAlert"] {
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(8, 13, 22, 0.90));
        border: 1px solid var(--qpk-line);
        border-left: 2px solid var(--qpk-accent);
        padding: 12px 14px;
        min-height: 92px;
    }
    div[data-testid="stMetric"] label {
        color: var(--qpk-muted) !important;
        font-size: 0.76rem !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: var(--qpk-text) !important;
        font-size: 1.42rem !important;
        line-height: 1.10 !important;
        font-weight: 650;
        white-space: normal !important;
        overflow-wrap: anywhere;
        word-break: normal;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid var(--qpk-line);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 10px 12px 10px;
        color: var(--qpk-muted);
        border-bottom: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--qpk-accent) !important;
        border-bottom-color: var(--qpk-accent) !important;
        background: rgba(125, 211, 252, 0.06);
    }
    div[data-testid="stPills"] {
        margin: 2px 0 16px 0;
    }
    div[data-testid="stPills"] [role="radiogroup"] {
        gap: 6px;
        flex-wrap: wrap;
    }
    div[data-testid="stPills"] button {
        min-height: 42px;
        border-radius: 4px !important;
        border: 1px solid var(--qpk-line) !important;
        background: rgba(7, 11, 18, 0.72) !important;
        color: var(--qpk-muted) !important;
        font-weight: 620 !important;
    }
    div[data-testid="stPills"] button[aria-checked="true"] {
        border-color: rgba(125, 211, 252, 0.62) !important;
        background: rgba(14, 165, 233, 0.13) !important;
        color: var(--qpk-text) !important;
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"], div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--qpk-line);
        background: rgba(7, 11, 18, 0.54);
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--qpk-line) !important;
        background: rgba(7, 11, 18, 0.64) !important;
    }
    .stButton > button {
        border: 1px solid rgba(125, 211, 252, 0.38);
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.20), rgba(8, 47, 73, 0.24));
        color: var(--qpk-text);
        font-weight: 650;
        border-radius: 4px;
    }
    .stButton > button:hover {
        border-color: var(--qpk-accent);
        color: white;
        box-shadow: 0 0 0 1px rgba(125, 211, 252, 0.20);
    }
    div[data-baseweb="select"] > div, textarea, input {
        background-color: #090f1a !important;
        border-color: var(--qpk-line) !important;
        color: var(--qpk-text) !important;
        border-radius: 4px !important;
    }
    .small-note {
        color: var(--qpk-muted);
        font-size: 0.86rem;
        line-height: 1.42;
    }
    .qpk-section {
        color: var(--qpk-accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.10em !important;
        text-transform: uppercase;
        margin: 22px 0 8px 0;
    }
    hr {
        border-color: var(--qpk-line) !important;
    }

    /* ============================================================
       Numeric / data display: tabular-nums for stable alignment
       ============================================================ */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"],
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stTable"] td {
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        letter-spacing: -0.01em !important;
    }

    /* ============================================================
       Touch targets >= 44px (Apple HIG / WCAG 2.5.5)
       ============================================================ */
    .stButton > button,
    .stDownloadButton > button,
    [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    [data-testid="stFileUploader"] button,
    [data-baseweb="tab"] {
        min-height: 44px !important;
    }
    .stCheckbox label,
    .stRadio label {
        min-height: 32px;
    }
    .stSlider [role="slider"] {
        height: 22px !important;
        width: 22px !important;
    }

    /* ============================================================
       Tabs: scroll-snap horizontal on small screens
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scroll-snap-type: x mandatory;
        scrollbar-width: thin;
        scrollbar-color: var(--qpk-line) transparent;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 4px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: var(--qpk-line);
        border-radius: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        scroll-snap-align: start;
        flex-shrink: 0;
        white-space: nowrap;
    }

    /* ============================================================
       Safe-area (iOS notch / Dynamic Island / home indicator)
       ============================================================ */
    .block-container {
        padding-left: max(1.15rem, env(safe-area-inset-left)) !important;
        padding-right: max(1.15rem, env(safe-area-inset-right)) !important;
        padding-bottom: max(2.4rem, env(safe-area-inset-bottom)) !important;
    }

    /* ============================================================
       Hero: fluid sizing to avoid overflow at 375px
       ============================================================ */
    .qpk-title {
        font-size: clamp(1.45rem, 4vw, 1.78rem) !important;
        line-height: 1.1 !important;
    }
    .qpk-subtitle {
        font-size: clamp(0.92rem, 2.4vw, 1rem) !important;
    }
    .qpk-hero {
        padding: clamp(10px, 2.4vw, 14px) 2px !important;
    }

    /* ============================================================
       RESPONSIVE: Tablet (<= 1023px)
       ============================================================ */
    @media (max-width: 1023px) {
        .block-container {
            padding-top: 0.75rem !important;
            max-width: 100% !important;
        }
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.05rem !important; }
        div[data-testid="stMetric"] {
            min-height: 78px !important;
            padding: 10px 12px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.12rem !important;
            line-height: 1.12 !important;
        }
    }

    /* ============================================================
       RESPONSIVE: Mobile (<= 767px)
       st.columns(N) -> 2-col grid (or 1-col on ultra-small)
       ============================================================ */
    @media (max-width: 767px) {
        .block-container {
            padding-left: max(0.75rem, env(safe-area-inset-left)) !important;
            padding-right: max(0.75rem, env(safe-area-inset-right)) !important;
        }
        /* Stack general content; only compact metric rows remain two-up. */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(div[data-testid="stMetric"]) {
            flex: 1 1 calc(50% - 4px) !important;
            min-width: calc(50% - 4px) !important;
            width: calc(50% - 4px) !important;
        }
        /* Metric cards: tighter */
        div[data-testid="stMetric"] {
            min-height: 70px !important;
            padding: 10px 11px !important;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.66rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
            line-height: 1.12 !important;
        }
        /* Avoid iOS auto-zoom on input focus (>=16px font) */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        [data-baseweb="select"] input {
            font-size: 16px !important;
        }
        /* Charts: prevent overflow */
        div[data-testid="stPlotlyChart"],
        div.stPlotlyChart,
        [data-testid="stImage"] img,
        [data-testid="stPyplot"] img {
            max-width: 100% !important;
            height: auto !important;
        }
        /* Tables: horizontal scroll inside container */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }
        /* Hero meta breaks line */
        .qpk-meta { font-size: 0.7rem !important; }
        .qpk-ops-strip {
            align-items: flex-start;
            padding: 10px !important;
        }
        /* Tab labels smaller */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.86rem !important;
            padding: 10px 8px !important;
        }
    }

    /* ============================================================
       RESPONSIVE: Ultra-small (<= 374px) — iPhone SE 1st gen, etc.
       Force single column for metric rows
       ============================================================ */
    @media (max-width: 374px) {
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(div[data-testid="stMetric"]) {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
    }

    /* ============================================================
       Landscape phones: keep some horizontal density
       ============================================================ */
    @media (max-width: 920px) and (orientation: landscape) {
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(div[data-testid="stMetric"]) {
            flex: 1 1 calc(33.333% - 6px) !important;
            min-width: calc(33.333% - 6px) !important;
        }
    }

    /* ============================================================
       Accessibility: prefers-reduced-motion
       ============================================================ */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }

    /* ============================================================
       Focus-visible: keyboard accessibility (don't kill rings)
       ============================================================ */
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-baseweb="tab"]:focus-visible,
    [data-baseweb="select"] > div:focus-visible,
    input:focus-visible,
    textarea:focus-visible {
        outline: 2px solid var(--qpk-accent) !important;
        outline-offset: 2px !important;
    }

    /* ============================================================
       Loading state for primary action button
       ============================================================ */
    .stButton > button:disabled {
        opacity: 0.55 !important;
        cursor: not-allowed !important;
    }

    /* ============================================================
       Sidebar: better mobile collapse handling
       ============================================================ */
    @media (max-width: 767px) {
        section[data-testid="stSidebar"] {
            width: 88vw !important;
            max-width: 360px !important;
            min-width: 0 !important;
            position: fixed !important;
            inset: 0 auto 0 0 !important;
            z-index: 1000000 !important;
            transition: transform 180ms ease !important;
            box-shadow: 18px 0 42px rgba(0, 0, 0, 0.42);
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(-100%) !important;
            pointer-events: none !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
            pointer-events: auto !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 0.75rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
            pointer-events: auto !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > div,
        [data-testid="stMain"],
        main {
            width: 100% !important;
            max-width: 100% !important;
            margin-left: 0 !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            top: max(0.6rem, env(safe-area-inset-top)) !important;
            left: max(0.55rem, env(safe-area-inset-left)) !important;
            z-index: 1000001 !important;
        }
    }
</style>
"""

# Sanitize: collapse blank lines (markdown treats them as HTML block boundary)
# and neutralize any leftover angle-bracket comparisons in comments.
import re as _re_css
_QPK_CSS = _re_css.sub(r"\n\s*\n+", "\n", _QPK_CSS)
_QPK_CSS = _QPK_CSS.replace("<=", "&lt;=").replace(" < ", " &lt; ")
st.markdown(_QPK_CSS, unsafe_allow_html=True)


# ============================================================
# Authentication gate
# ============================================================
# This must run BEFORE any widget that exposes data or backend triggers.
# `require_authentication()` either returns an AuthenticatedUser or
# renders the login form + st.stop()s the script.

from auth import require_authentication, filter_accessible_sections

current_user = require_authentication()

plt.rcParams.update(
    {
        "figure.facecolor": "#05070d",
        "axes.facecolor": "#080d16",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#dbeafe",
        "axes.titlecolor": "#e5edf7",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "text.color": "#e5edf7",
        "legend.facecolor": "#05070d",
        "legend.edgecolor": "#334155",
        "savefig.facecolor": "#05070d",
    }
)


def parse_tickers(text: str) -> tuple[str, ...]:
    raw = text.replace(",", " ").replace("\n", " ").split()
    return tuple(sanitize_ticker_list(raw))


def fmt_pct(value, default: str = "n/a") -> str:
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.0%}"
    except Exception:
        return default


def _fmt_pct(value, digits: int = 2, default: str = "n/a") -> str:
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{digits}%}"
    except Exception:
        return default


def _fmt_float(value, digits: int = 3, default: str = "n/a") -> str:
    try:
        if pd.isna(value):
            return default
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "passed"}


def _read_research_csv(base: Path, filename: str) -> pd.DataFrame:
    path = base / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def load_xcdr_research_artifacts(artifact_dir: str | None = None) -> dict:
    candidates: list[Path] = []
    if artifact_dir:
        candidates.append(Path(artifact_dir))
    elif os.environ.get("QPK_RESEARCH_ARTIFACT_DIR"):
        candidates.append(Path(os.environ["QPK_RESEARCH_ARTIFACT_DIR"]))
    else:
        candidates.extend(
            [
                Path(r"C:\Users\chris\Downloads"),
                PROJECT_RESEARCH_ARTIFACT_DIR,
            ]
        )
    base = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "xcdr_v3_parallel_research_summary.csv").exists()
        ),
        candidates[-1] if candidates else RESEARCH_ARTIFACT_DIR,
    )
    files = {
        "summary": "xcdr_v3_parallel_research_summary.csv",
        "windows": "xcdr_v3_parallel_research_windows.csv",
        "daily": "xcdr_v3_parallel_research_daily_oos.csv",
        "daily_summary": "xcdr_v3_parallel_research_daily_summary.csv",
        "holdout_summary": "xcdr_v3_parallel_research_holdout_summary.csv",
        "weights": "xcdr_v3_parallel_research_weights.csv",
        "red_team": "xcdr_v3_parallel_research_red_team.csv",
    }
    data = {key: _read_research_csv(base, filename) for key, filename in files.items()}
    report_path = base / "xcdr_v3_parallel_research_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    missing = [filename for filename in files.values() if not (base / filename).exists()]
    if not report_path.exists():
        missing.append(report_path.name)
    return {
        **data,
        "report": report,
        "artifact_dir": str(base),
        "missing": missing,
        "loaded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def select_research_objective(summary: pd.DataFrame) -> str | None:
    if not isinstance(summary, pd.DataFrame) or summary.empty or "objective" not in summary.columns:
        return None
    frame = summary.copy()
    if "research_gate_pass" in frame.columns:
        promoted = frame[frame["research_gate_pass"].map(_coerce_bool)]
        if not promoted.empty:
            promoted = promoted.sort_values(["daily_active_ann_return", "active_ann_return"], ascending=False, na_position="last")
            return str(promoted.iloc[0]["objective"])
    objectives = set(frame["objective"].astype(str))
    for objective in RESEARCH_PREFERRED_OBJECTIVES:
        if objective in objectives:
            return objective
    score_col = "daily_active_ann_return" if "daily_active_ann_return" in frame.columns else "active_ann_return"
    if score_col in frame.columns:
        ranked = frame.sort_values(score_col, ascending=False, na_position="last")
        if not ranked.empty:
            return str(ranked.iloc[0]["objective"])
    return str(frame.iloc[0]["objective"])


def _research_daily_frame(daily: pd.DataFrame, objective: str) -> pd.DataFrame:
    required = {"objective", "date", "portfolio_return", "xi_return"}
    if not isinstance(daily, pd.DataFrame) or daily.empty or not required.issubset(daily.columns):
        return pd.DataFrame()
    sub = daily[daily["objective"].astype(str) == str(objective)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["portfolio_return"] = pd.to_numeric(sub["portfolio_return"], errors="coerce").fillna(0.0)
    sub["xi_return"] = pd.to_numeric(sub["xi_return"], errors="coerce").fillna(0.0)
    sub = sub.dropna(subset=["date"]).sort_values("date")
    if sub.empty:
        return pd.DataFrame()
    sub["Research strategy NAV"] = (1.0 + sub["portfolio_return"]).cumprod()
    sub["Benchmark NAV"] = (1.0 + sub["xi_return"]).cumprod()
    sub["Active NAV"] = (1.0 + sub["portfolio_return"] - sub["xi_return"]).cumprod()
    sub["Research strategy drawdown"] = sub["Research strategy NAV"] / sub["Research strategy NAV"].cummax() - 1.0
    sub["Benchmark drawdown"] = sub["Benchmark NAV"] / sub["Benchmark NAV"].cummax() - 1.0
    return sub


def _research_chart_frames(artifacts: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = artifacts.get("summary", pd.DataFrame()) if isinstance(artifacts, dict) else pd.DataFrame()
    daily = artifacts.get("daily", pd.DataFrame()) if isinstance(artifacts, dict) else pd.DataFrame()
    objective = select_research_objective(summary)
    if objective is None:
        return pd.DataFrame(), pd.DataFrame(), {}
    nav = _research_daily_frame(daily, objective)
    if nav.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    selected_daily = daily[daily["objective"].astype(str) == str(objective)].copy()
    xi = "Benchmark"
    if "xi" in selected_daily.columns and selected_daily["xi"].notna().any():
        modes = selected_daily["xi"].dropna().astype(str).mode()
        xi = str(modes.iloc[0]) if not modes.empty else str(selected_daily["xi"].dropna().iloc[-1])
    strategy_label = "XCDR/XODR candidate"
    benchmark_label = f"{xi} benchmark"
    prices = nav[["date", "Research strategy NAV", "Benchmark NAV"]].rename(
        columns={
            "date": "Date",
            "Research strategy NAV": strategy_label,
            "Benchmark NAV": benchmark_label,
        }
    )
    prices[strategy_label] = pd.to_numeric(prices[strategy_label], errors="coerce") * 100.0
    prices[benchmark_label] = pd.to_numeric(prices[benchmark_label], errors="coerce") * 100.0
    drawdowns = nav[["date", "Research strategy drawdown", "Benchmark drawdown"]].rename(
        columns={
            "date": "Date",
            "Research strategy drawdown": strategy_label,
            "Benchmark drawdown": benchmark_label,
        }
    )
    return prices, drawdowns, {
        "objective": objective,
        "xi": xi,
        "strategy_label": strategy_label,
        "benchmark_label": benchmark_label,
        "research_only": True,
    }


def _ann_return_from_daily(r: pd.Series) -> float:
    r = pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.mean() * 252.0) if len(r) else np.nan


def _ann_vol_from_daily(r: pd.Series) -> float:
    r = pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 2 else np.nan


def _downside_from_daily(r: pd.Series) -> float:
    r = pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) * np.sqrt(252.0)) if len(r) else np.nan


def _cvar_loss_from_daily(r: pd.Series, alpha: float = 0.95) -> float:
    losses = -pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if losses.empty:
        return np.nan
    q = losses.quantile(alpha)
    tail = losses[losses >= q]
    return float(tail.mean()) if len(tail) else float(q)


def _maxdd_loss_from_daily(r: pd.Series) -> float:
    nav = (1.0 + pd.Series(r, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float(-(nav / nav.cummax() - 1.0).min())


def _latest_local_dashboard_artifact() -> dict:
    path = Path(__file__).resolve().with_name(".quant_cache") / "cloud" / "latest_dashboard_payload.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _payload_frame(value) -> pd.DataFrame:
    """Restore a DataFrame serialized through Supabase/Postgres JSON."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict) and value:
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])
    return pd.DataFrame()


def _minimal_results_from_dashboard_payload(payload: dict, *, benchmark: str) -> dict:
    safe_payload = payload if isinstance(payload, dict) else {}
    allocation = safe_payload.get("allocation", {}) if isinstance(safe_payload, dict) else {}
    charts = safe_payload.get("charts", {}) if isinstance(safe_payload, dict) else {}
    tables = safe_payload.get("tables", {}) if isinstance(safe_payload, dict) else {}
    status = safe_payload.get("status", {}) if isinstance(safe_payload, dict) else {}
    research = safe_payload.get("research", {}) if isinstance(safe_payload, dict) else {}
    diagnostics = safe_payload.get("diagnostics", {}) if isinstance(safe_payload, dict) else {}
    market_snapshot = safe_payload.get("market_snapshot", {}) if isinstance(safe_payload, dict) else {}
    market_intelligence = safe_payload.get("market_intelligence", {}) if isinstance(safe_payload, dict) else {}
    risk_table = _payload_frame(tables.get("risk")) if isinstance(tables, dict) else pd.DataFrame()
    portfolio = _payload_frame(allocation.get("recommended_portfolio")) if isinstance(allocation, dict) else pd.DataFrame()
    side_sleeve = _payload_frame(allocation.get("side_sleeve")) if isinstance(allocation, dict) else pd.DataFrame()
    freshness = _payload_frame(status.get("data_freshness")) if isinstance(status, dict) else pd.DataFrame()
    suitability = _payload_frame(status.get("suitability")) if isinstance(status, dict) else pd.DataFrame()
    suitability_breaches = _payload_frame(status.get("suitability_breaches")) if isinstance(status, dict) else pd.DataFrame()
    promotion = _payload_frame(status.get("promotion")) if isinstance(status, dict) else pd.DataFrame()
    promotion_tests = _payload_frame(status.get("promotion_tests")) if isinstance(status, dict) else pd.DataFrame()
    snapshot_meta = _payload_frame(status.get("snapshot_meta")) if isinstance(status, dict) else pd.DataFrame()
    market_context = _payload_frame(status.get("market_context")) if isinstance(status, dict) else pd.DataFrame()
    price_paths = _payload_frame(charts.get("price_paths")) if isinstance(charts, dict) else pd.DataFrame()
    drawdowns = _payload_frame(charts.get("drawdowns")) if isinstance(charts, dict) else pd.DataFrame()
    rate_curves = _payload_frame(charts.get("rate_curves")) if isinstance(charts, dict) else pd.DataFrame()
    options_surface = _payload_frame(charts.get("options_surface")) if isinstance(charts, dict) else pd.DataFrame()
    max_drawdown = _payload_frame(tables.get("max_drawdown")) if isinstance(tables, dict) else pd.DataFrame()
    rejections = _payload_frame(tables.get("rejections")) if isinstance(tables, dict) else pd.DataFrame()
    fundamentals = _payload_frame(tables.get("fundamentals")) if isinstance(tables, dict) else pd.DataFrame()
    validation = _payload_frame(tables.get("validation")) if isinstance(tables, dict) else pd.DataFrame()
    observed_selection = (
        _payload_frame(market_snapshot.get("observed_selection"))
        if isinstance(market_snapshot, dict)
        else pd.DataFrame()
    )
    restored_market_intelligence = {
        key: _payload_frame(value)
        for key, value in market_intelligence.items()
    } if isinstance(market_intelligence, dict) else {}
    normalized_payload = dict(safe_payload)
    normalized_payload["status"] = {
        "suitability": suitability,
        "suitability_breaches": suitability_breaches,
        "promotion": promotion,
        "promotion_tests": promotion_tests,
        "data_freshness": freshness,
        "snapshot_meta": snapshot_meta,
        "market_context": market_context,
    }
    normalized_payload["allocation"] = {
        "recommended_portfolio": portfolio,
        "side_sleeve": side_sleeve,
        "weights": _payload_frame(allocation.get("weights")) if isinstance(allocation, dict) else pd.DataFrame(),
    }
    normalized_payload["charts"] = {
        "price_paths": price_paths,
        "drawdowns": drawdowns,
        "forecast_cone": _payload_frame(charts.get("forecast_cone")) if isinstance(charts, dict) else pd.DataFrame(),
        "conditional_vol": _payload_frame(charts.get("conditional_vol")) if isinstance(charts, dict) else pd.DataFrame(),
        "rate_curves": rate_curves,
        "options_surface": options_surface,
    }
    normalized_payload["tables"] = {
        "fundamentals": fundamentals,
        "risk": risk_table,
        "validation": validation,
        "rejections": rejections,
        "max_drawdown": max_drawdown,
    }
    normalized_payload["research"] = {
        key: _payload_frame(value)
        for key, value in research.items()
    } if isinstance(research, dict) else {}
    normalized_payload["market_snapshot"] = {
        "observed_selection": observed_selection,
        "price_only": bool(market_snapshot.get("price_only", False)) if isinstance(market_snapshot, dict) else False,
        "context": _payload_frame(market_snapshot.get("context")) if isinstance(market_snapshot, dict) else pd.DataFrame(),
    }
    normalized_payload["market_intelligence"] = restored_market_intelligence
    restored_research = normalized_payload["research"]
    restored_diagnostics = {
        group: {
            key: _payload_frame(value)
            for key, value in values.items()
        }
        for group, values in diagnostics.items()
        if isinstance(values, dict)
    } if isinstance(diagnostics, dict) else {}
    normalized_payload["diagnostics"] = restored_diagnostics
    restored_return = restored_diagnostics.get("return", {})
    restored_validation = restored_diagnostics.get("validation", {})
    restored_latent = restored_diagnostics.get("latent_regime", {})
    restored_alternative = restored_diagnostics.get("alternative_data", {})
    restored_kaizen = restored_diagnostics.get("kaizen", {})
    if "summary" not in restored_validation:
        restored_validation["summary"] = validation
    latest_macro_frame = restored_market_intelligence.get("latest_macro", pd.DataFrame())
    macro_history = restored_market_intelligence.get("macro_history", pd.DataFrame())
    if latest_macro_frame.empty:
        latest_macro_frame = market_context
    if macro_history.empty:
        macro_history = market_context
    sentiment_sem = {
        "timeline": restored_market_intelligence.get("sentiment_timeline", pd.DataFrame()),
        "latest": restored_market_intelligence.get("sentiment_latest", pd.DataFrame()),
        "loadings": restored_market_intelligence.get("sentiment_loadings", pd.DataFrame()),
        "structural_links": restored_market_intelligence.get("sentiment_structural_links", pd.DataFrame()),
        "diagnostics": restored_market_intelligence.get("sentiment_diagnostics", pd.DataFrame()),
    }
    restored_alternative = dict(restored_alternative)
    for target, source in (
        ("forex_factory_calendar", "forex_factory_calendar"),
        ("forex_factory_event_risk", "forex_factory_event_risk"),
        ("summary", "geopolitical_summary"),
        ("gdelt_timeline", "geopolitical_timeline"),
    ):
        value = restored_market_intelligence.get(source, pd.DataFrame())
        if isinstance(value, pd.DataFrame) and not value.empty:
            restored_alternative[target] = value

    return {
        "dashboard_payload": normalized_payload,
        "artifact_created_at": safe_payload.get("_artifact_created_at"),
        "artifact_run_id": safe_payload.get("_artifact_run_id"),
        "latest_macro": latest_macro_frame.iloc[-1] if not latest_macro_frame.empty else pd.Series(dtype=object),
        "prices": pd.DataFrame(),
        "cross_section": pd.DataFrame(),
        "portfolio": portfolio,
        "side_boom_portfolio": side_sleeve,
        "side_boom_curve": pd.DataFrame(),
        "side_boom_walk_forward": pd.DataFrame(),
        "side_boom_holdings": pd.DataFrame(),
        "side_boom_diagnostics": pd.DataFrame(),
        "portfolio_options": pd.DataFrame(),
        "backtest_perf": pd.DataFrame(),
        "backtest_holdings": pd.DataFrame(),
        "price_snapshot_selection": observed_selection,
        "optimization_grid": restored_research.get("optimization_grid", pd.DataFrame()),
        "sector_diagnostics": restored_research.get("sector_diagnostics", pd.DataFrame()),
        "macro": macro_history,
        "equity_curve": pd.DataFrame(),
        "return_diagnostics": restored_return,
        "performance_summary": risk_table,
        "overfit_diagnostics": restored_research.get("overfit_diagnostics", pd.DataFrame()),
        "factor_attribution": restored_research.get("factor_attribution", pd.DataFrame()),
        "monitoring_diagnostics": restored_research.get("monitoring_diagnostics", pd.DataFrame()),
        "rejection_diagnostics": rejections,
        "cache_inventory": restored_research.get("cache_inventory", pd.DataFrame()),
        "timings": restored_research.get("timings", pd.DataFrame()),
        "options_chain": restored_research.get("options_chain", pd.DataFrame()),
        "options_summary": restored_research.get("options_summary", pd.DataFrame()),
        "portfolio_vol_surface": restored_research.get("vol_surface", pd.DataFrame()),
        "portfolio_vol_surface_matrix": options_surface,
        "portfolio_vol_surface_diagnostics": restored_research.get("vol_surface_diagnostics", pd.DataFrame()),
        "validation_diagnostics": restored_validation,
        "kaizen_diagnostics": restored_kaizen,
        "latent_regime_diagnostics": restored_latent,
        "alternative_data": restored_alternative,
        "market_sentiment_sem": sentiment_sem,
        "global_yield_curves": restored_market_intelligence.get("global_yield_curves", rate_curves),
        "global_rate_history": restored_market_intelligence.get(
            "global_rate_history",
            restored_research.get("global_rate_history", pd.DataFrame()),
        ),
        "interbank_reference_rates": restored_market_intelligence.get(
            "interbank_reference_rates",
            restored_research.get("interbank_reference_rates", pd.DataFrame()),
        ),
        "carry_trade_suggestions": restored_market_intelligence.get(
            "carry_trade_suggestions",
            restored_research.get("carry_trade_suggestions", pd.DataFrame()),
        ),
        "carry_trade_validation": restored_market_intelligence.get(
            "carry_trade_validation",
            restored_research.get("carry_trade_validation", pd.DataFrame()),
        ),
        "suitability_diagnostics": suitability,
        "suitability_gate": {"summary": suitability, "breaches": suitability_breaches},
        "promotion_gate": {"summary": promotion, "tests": promotion_tests},
        "backtest_path_bundle": {
            "price_paths": price_paths,
            "drawdowns": drawdowns,
            "max_drawdown_table": max_drawdown,
        },
        "benchmark_governance": restored_research.get("benchmark_governance", pd.DataFrame()),
        "model_registry": restored_research.get("model_registry", pd.DataFrame()),
        "oos_factor_attribution": restored_research.get("oos_factor_attribution", pd.DataFrame()),
        "regime_performance": restored_research.get("regime_performance", pd.DataFrame()),
        "stress_tests": restored_research.get("stress_tests", pd.DataFrame()),
        "hedge_suggestions": restored_research.get("hedge_suggestions", pd.DataFrame()),
        "decision_attribution": restored_research.get("decision_attribution", pd.DataFrame()),
        "capital_ledger": restored_research.get("capital_ledger", pd.DataFrame()),
        "side_boom_pelt_regime_segments": restored_research.get("side_pelt_regime_segments", pd.DataFrame()),
        "side_boom_pelt_change_points": restored_research.get("side_pelt_change_points", pd.DataFrame()),
        "side_boom_pelt_timeline": restored_research.get("side_pelt_timeline", pd.DataFrame()),
        "data_freshness_report": freshness,
        "fundamentals_snapshot": fundamentals,
        "benchmark_ticker": benchmark,
    }


def _load_precomputed_dashboard_results(benchmark: str) -> dict:
    if os.getenv("QPK_LOAD_LATEST_DASHBOARD_ON_START", "1") == "0":
        return {}
    artifact_bundle = {}
    if latest_dashboard_artifacts is not None:
        try:
            artifact_bundle = latest_dashboard_artifacts() or {}
        except Exception:
            artifact_bundle = {}
    artifact = (
        artifact_bundle.get("full_analysis")
        or artifact_bundle.get("daily_snapshot")
        or artifact_bundle.get("latest_any")
        or {}
    )
    if not artifact and latest_dashboard_artifact is not None:
        try:
            artifact = latest_dashboard_artifact() or {}
        except Exception:
            artifact = {}
    if not artifact:
        artifact = _latest_local_dashboard_artifact()
    payload = artifact.get("dashboard_payload") if isinstance(artifact, dict) else None
    if not isinstance(payload, dict) or not payload:
        return {}
    payload_with_meta = dict(payload)
    payload_with_meta["_artifact_created_at"] = artifact.get("created_at")
    payload_with_meta["_artifact_run_id"] = artifact.get("run_id")
    results = _minimal_results_from_dashboard_payload(payload_with_meta, benchmark=benchmark)

    daily_artifact = artifact_bundle.get("daily_snapshot") or {}
    daily_payload = daily_artifact.get("dashboard_payload")
    if isinstance(daily_payload, dict) and daily_payload:
        daily_payload_with_meta = dict(daily_payload)
        daily_payload_with_meta["_artifact_created_at"] = daily_artifact.get("created_at")
        daily_payload_with_meta["_artifact_run_id"] = daily_artifact.get("run_id")
        daily_results = _minimal_results_from_dashboard_payload(daily_payload_with_meta, benchmark=benchmark)
        results["daily_snapshot_payload"] = daily_results.get("dashboard_payload", {})
        results["daily_snapshot_created_at"] = daily_artifact.get("created_at")
        results["daily_snapshot_run_id"] = daily_artifact.get("run_id")
    results["full_analysis_created_at"] = artifact.get("created_at")
    results["full_analysis_run_id"] = artifact.get("run_id")
    return results


@st.cache_data(show_spinner=False, ttl=86400)
def cached_public_universe(source: str, asof_date, use_cache: bool, cache_ttl_hours: int, user_agent: str):
    return load_public_universe(
        source,
        asof_date=asof_date,
        use_cache=use_cache,
        cache_ttl_hours=cache_ttl_hours,
        user_agent=user_agent,
    )


@st.cache_data(show_spinner=False, ttl=86400)
def cached_preflight_market(
    benchmark_ticker: str,
    benchmark_group: str,
    rate_country: str,
    price_period: str,
    side_tickers: tuple[str, ...],
    side_fixed_ticker: str,
    side_fixed_weight: float,
    side_fixed_weights: tuple[tuple[str, float], ...],
    side_min_obs: int,
    use_cache: bool,
    cache_ttl_hours: int,
    include_global_rates: bool = False,
    include_geopolitical: bool = False,
    geopolitical_cache_ttl_hours: int = 24,
):
    group_tickers = list(BENCHMARK_PRESETS.get(benchmark_group, {}).keys())
    broad = ["SPY", "QQQ", "IWM", "ACWI", "VT", "EEM", "EWW", "XLK", "XLV", "XLU"]
    bench_tickers = tuple(dict.fromkeys([benchmark_ticker] + group_tickers[:8] + broad))
    side_tickers = tuple(side_tickers)
    all_tickers = tuple(dict.fromkeys(list(bench_tickers) + list(side_tickers)))
    prices = download_prices(all_tickers, price_period, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    bench_prices = prices[[c for c in bench_tickers if c in prices.columns]].copy() if not prices.empty else pd.DataFrame()
    macro, latest_macro = market_regime(
        bench_prices if not bench_prices.empty else prices,
        country=rate_country,
        use_cache=use_cache,
        cache_ttl_hours=cache_ttl_hours,
        use_latent_macro_regime=False,
    )
    global_rates = (
        global_yield_curve_snapshot(
            bench_prices if not bench_prices.empty else prices,
            use_cache=use_cache,
            cache_ttl_hours=cache_ttl_hours,
        )
        if include_global_rates
        else pd.DataFrame()
    )
    global_rate_history = (
        global_yield_curve_discrete_history(
            bench_prices if not bench_prices.empty else prices,
            use_cache=use_cache,
            cache_ttl_hours=cache_ttl_hours,
        )
        if include_global_rates
        else pd.DataFrame()
    )
    rate_index = bench_prices.index if not bench_prices.empty else prices.index
    interbank_reference_rates = (
        fetch_interbank_reference_rates(
            rate_index.min() - pd.Timedelta(days=30),
            rate_index.max() + pd.Timedelta(days=5),
            use_cache=use_cache,
            cache_ttl_hours=cache_ttl_hours,
        )
        if len(rate_index) > 0
        else pd.DataFrame()
    )
    ff_calendar = fetch_forex_factory_calendar(use_cache=use_cache, cache_ttl_hours=24)
    ff_event_risk = forex_factory_event_risk(ff_calendar)
    carry_suggestions = carry_trade_suggestions(global_rates, ff_event_risk)
    pre_cfg = RunConfig(
        tickers=tuple([t for t in side_tickers if t != benchmark_ticker]) or ("SPY",),
        benchmark_ticker=benchmark_ticker,
        price_period=price_period,
        use_persistent_cache=use_cache,
        cache_ttl_hours=cache_ttl_hours,
        rate_country=rate_country,
        use_side_boom_portfolio=True,
        side_boom_tickers=side_tickers,
        side_boom_fixed_ticker=side_fixed_ticker,
        side_boom_fixed_weight=side_fixed_weight,
        side_boom_fixed_weights=side_fixed_weights,
        side_boom_min_obs=side_min_obs,
        use_sec_edgar=False,
        use_sec_nlp=False,
        use_options_snapshot=False,
        use_gdelt=False,
        sortino_multistarts=3,
        max_workers=4,
    )
    side_portfolio, side_curve, side_diag = optimize_side_boom_portfolio(prices, pre_cfg, macro=macro, lookback=126)
    geo = (
        geopolitical_thermometer(use_cache=use_cache, cache_ttl_hours=geopolitical_cache_ttl_hours)
        if include_geopolitical
        else {"summary": pd.DataFrame(), "articles": pd.DataFrame(), "timeline": pd.DataFrame()}
    )
    sentiment_sem = market_sentiment_sem(
        bench_prices if not bench_prices.empty else prices,
        macro=macro,
        forex_event_risk=ff_event_risk,
        geopolitical_summary=geo.get("summary", pd.DataFrame()) if isinstance(geo, dict) else pd.DataFrame(),
        benchmark=benchmark_ticker,
        lookback=756,
    )
    return {
        "prices": prices,
        "benchmark_prices": bench_prices,
        "macro": macro,
        "latest_macro": latest_macro,
        "global_rates": global_rates,
        "global_rate_history": global_rate_history,
        "interbank_reference_rates": interbank_reference_rates,
        "forex_factory_calendar": ff_calendar,
        "forex_factory_event_risk": ff_event_risk,
        "carry_trade_suggestions": carry_suggestions,
        "side_portfolio": side_portfolio,
        "side_curve": side_curve,
        "side_diagnostics": side_diag,
        "geopolitical": geo,
        "market_sentiment_sem": sentiment_sem,
    }


def csv_download(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        st.caption(f"{label}: no data.")
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")


def polished_table(df: pd.DataFrame, metric_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if metric_col and metric_col in out:
        out[metric_col] = out[metric_col].astype(str).str.replace("_", " ", regex=False)
    labels = {
        "Latest_Volume": "Latest volume",
        "Baseline_Median": "Baseline median",
        "Robust_Z_Score": "Robust Z",
        "Positive_Shock_Score": "Positive shock",
        "Statistical_Admissibility": "Statistically admissible",
        "Risk_Overlay_Admissible": "Risk overlay",
        "Data_Source_Type": "Data source type",
        "Score_Type": "Score type",
        "Sample_Size": "Sample size",
        "Unique_Observations": "Unique observations",
        "Article_Count": "Articles",
        "Unique_Domains": "Unique domains",
        "Geo_News_Attention_Score": "Attention score",
        "Topic_Count": "Topics",
        "Weighted_Topic_Intensity": "Weighted intensity",
        "Risk_Overlay_Article_Count": "Risk-overlay articles",
        "Dominant_Topic": "Dominant topic",
        "Geo_Inference_Methods": "Inference methods",
        "Mean_Geo_Inference_Confidence": "Mean inference confidence",
        "Regex_Inferred_Article_Count": "Regex-inferred articles",
        "SourceCountry_Fallback_Count": "Source-country fallbacks",
        "Heat_Level": "Heat level",
        "Data_Source": "Data source",
        "SeenDate": "Seen date",
        "Translation_Status": "Translation status",
    }
    out = out.rename(columns={c: labels.get(c, c.replace("_", " ")) for c in out.columns})
    for col in ["Metric", "Data source type", "Score type", "Translation status", "Inference methods"]:
        if col in out:
            out[col] = out[col].astype(str).str.replace("_", " ", regex=False)
    return out


def formal_equation(title: str, equation: str, note: str | None = None):
    st.markdown(f"**{title}**")
    st.latex(equation)
    if note:
        st.caption(note)


def _aligned_benchmark_price(curve: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.Series:
    if curve is None or curve.empty or prices is None or prices.empty or benchmark_ticker not in prices:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(curve["Period_End"], errors="coerce")
    px = pd.to_numeric(prices[benchmark_ticker], errors="coerce").sort_index().ffill()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index()
    aligned = px.reindex(dates, method="ffill")
    aligned.index = curve.index
    return aligned


def _period_holdings(holdings: pd.DataFrame, period_row: pd.Series, date_col_candidates: tuple[str, ...]) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    for col in date_col_candidates:
        if col in holdings and col in period_row:
            target = pd.Timestamp(period_row[col])
            hdates = pd.to_datetime(holdings[col], errors="coerce")
            sub = holdings[hdates.eq(target)].copy()
            if not sub.empty:
                return sub
    return pd.DataFrame()


def _daily_synthetic_nav_from_holdings(
    schedule: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    holding_date_cols: tuple[str, ...],
    weight_col: str = "Effective_Weight",
    label: str = "Optimized synthetic NAV price",
    cost_cols: tuple[str, ...] = ("Fixed_TC", "Impact_TC"),
) -> pd.DataFrame:
    if schedule is None or schedule.empty or holdings is None or holdings.empty or prices is None or prices.empty:
        return pd.DataFrame()
    if benchmark_ticker not in prices:
        return pd.DataFrame()
    px = prices.copy()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index().ffill()
    rows = []
    nav = 1.0
    prior_date = None
    schedule = schedule.copy()
    for col in ["OOS_Start", "OOS_End", "Period_End", "Rebalance_Date"]:
        if col in schedule:
            schedule[col] = pd.to_datetime(schedule[col], errors="coerce")
    date_sort = "OOS_Start" if "OOS_Start" in schedule else "Period_End"
    for _, period in schedule.sort_values(date_sort).iterrows():
        start = pd.Timestamp(period.get("OOS_Start", period.get("Rebalance_Date", pd.NaT)))
        end = pd.Timestamp(period.get("OOS_End", period.get("Period_End", pd.NaT)))
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        h = _period_holdings(holdings, period, holding_date_cols)
        if h.empty or "Ticker" not in h:
            continue
        wc = weight_col if weight_col in h else "Weight" if "Weight" in h else None
        if wc is None:
            continue
        weights = (
            h[~h["Ticker"].astype(str).str.upper().eq("CASH")]
            .groupby("Ticker")[wc]
            .sum()
            .pipe(pd.to_numeric, errors="coerce")
            .dropna()
        )
        tickers = [t for t in weights.index if t in px.columns]
        if not tickers:
            continue
        interval = px.loc[(px.index >= start) & (px.index <= end), list(dict.fromkeys(tickers + [benchmark_ticker]))].copy()
        if len(interval) < 2:
            continue
        asset_ret = interval[tickers].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weights = weights.reindex(tickers).fillna(0.0).astype(float)
        daily_ret = asset_ret @ weights
        period_cost = 0.0
        for c in cost_cols:
            val = pd.to_numeric(pd.Series([period.get(c, 0.0)]), errors="coerce").iloc[0]
            if pd.notna(val) and np.isfinite(float(val)):
                period_cost += float(val)
        first_return_idx = daily_ret.index[1] if len(daily_ret) > 1 else daily_ret.index[0]
        if np.isfinite(period_cost) and abs(period_cost) > 0:
            daily_ret.loc[first_return_idx] = daily_ret.loc[first_return_idx] - period_cost
        bench = interval[benchmark_ticker]
        for dt in daily_ret.index:
            if prior_date is not None and pd.Timestamp(dt) <= prior_date:
                continue
            nav *= 1.0 + float(daily_ret.loc[dt])
            rows.append({"Date": pd.Timestamp(dt), "_Synthetic_NAV": nav, f"{benchmark_ticker} observed price": float(bench.loc[dt])})
            prior_date = pd.Timestamp(dt)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["Date", f"{benchmark_ticker} observed price"]).sort_values("Date")
    benchmark_first = out[f"{benchmark_ticker} observed price"].dropna()
    nav_first = out["_Synthetic_NAV"].dropna()
    if benchmark_first.empty or nav_first.empty:
        return pd.DataFrame()
    anchor = float(benchmark_first.iloc[0])
    out[label] = anchor * out["_Synthetic_NAV"] / float(nav_first.iloc[0])
    return out.drop(columns=["_Synthetic_NAV"])


def daily_backtest_price_frame(perf: pd.DataFrame, holdings: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    return _daily_synthetic_nav_from_holdings(
        perf,
        holdings,
        prices,
        benchmark_ticker,
        holding_date_cols=("Rebalance_Date", "OOS_Start", "Period_End"),
        weight_col="Effective_Weight",
        label="Optimized portfolio NAV",
    )


def daily_side_price_frame(side_perf: pd.DataFrame, side_holdings: pd.DataFrame, prices: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    return _daily_synthetic_nav_from_holdings(
        side_perf,
        side_holdings,
        prices,
        benchmark_ticker,
        holding_date_cols=("OOS_Start", "Rebalance_Date", "Period_End"),
        weight_col="Weight",
        label="Private Side Alpha synthetic NAV price",
        cost_cols=(),
    )


def backtest_price_frame(curve: pd.DataFrame, benchmark_ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame()
    benchmark_price = _aligned_benchmark_price(curve, prices, benchmark_ticker)
    if benchmark_price.empty or benchmark_price.dropna().empty:
        return pd.DataFrame()
    anchor_price = float(benchmark_price.dropna().iloc[0])
    if not np.isfinite(anchor_price) or anchor_price <= 0:
        return pd.DataFrame()
    out = pd.DataFrame({"Period_End": pd.to_datetime(curve["Period_End"], errors="coerce")})
    out[f"{benchmark_ticker} observed price"] = benchmark_price.values
    for src, label in [
        ("Portfolio_Equity", "Optimized portfolio NAV"),
        ("Side_Boom_Equity", "Private Side Alpha synthetic NAV price"),
    ]:
        if src in curve and pd.to_numeric(curve[src], errors="coerce").notna().any():
            equity = pd.to_numeric(curve[src], errors="coerce")
            first_equity = float(equity.dropna().iloc[0]) if not equity.dropna().empty else np.nan
            if np.isfinite(first_equity) and abs(first_equity) > 1e-12:
                out[label] = anchor_price * equity / first_equity
    return out.dropna(subset=["Period_End"]).dropna(axis=1, how="all")


def plot_price_path(price_frame: pd.DataFrame, benchmark_ticker: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if price_frame.empty:
        ax.text(0.5, 0.5, "No daily price path", ha="center", va="center")
        return fig
    date_col = "Date" if "Date" in price_frame else "Period_End"
    x = pd.to_datetime(price_frame[date_col], errors="coerce")
    colors = {
        "Optimized portfolio NAV": "#155eef",
        f"{benchmark_ticker} observed price": "#667085",
        "Private Side Alpha synthetic NAV price": "#d97706",
    }
    widths = {
        "Optimized portfolio NAV": 2.4,
        f"{benchmark_ticker} observed price": 1.9,
        "Private Side Alpha synthetic NAV price": 2.1,
    }
    for col in price_frame.columns:
        if col in {"Date", "Period_End"}:
            continue
        ax.plot(x, price_frame[col], label=col, linewidth=widths.get(col, 1.8), alpha=0.86, color=colors.get(col))
    if f"{benchmark_ticker} observed price" in price_frame and price_frame[f"{benchmark_ticker} observed price"].dropna().any():
        anchor = float(price_frame[f"{benchmark_ticker} observed price"].dropna().iloc[0])
        ax.axhline(anchor, color="#94a3b8", linewidth=0.8, alpha=0.45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.2f}"))
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=3)
    ax.grid(alpha=0.25)
    return fig


def price_drawdown_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    if price_frame is None or price_frame.empty:
        return pd.DataFrame()
    date_col = "Date" if "Date" in price_frame else "Period_End" if "Period_End" in price_frame else None
    if date_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({date_col: pd.to_datetime(price_frame[date_col], errors="coerce")})
    for col in price_frame.columns:
        if col == date_col:
            continue
        px = pd.to_numeric(price_frame[col], errors="coerce")
        if px.notna().sum() < 2:
            continue
        dd = px / px.cummax() - 1.0
        out[col] = dd
    return out.dropna(subset=[date_col]).dropna(axis=1, how="all")


def plot_price_drawdown(price_frame: pd.DataFrame, title: str = "Daily drawdown from price paths"):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    dd = price_drawdown_frame(price_frame)
    if dd.empty or len(dd.columns) <= 1:
        ax.text(0.5, 0.5, "No price drawdown data", ha="center", va="center")
        return fig
    date_col = "Date" if "Date" in dd else "Period_End"
    x = pd.to_datetime(dd[date_col], errors="coerce")
    colors = {
        "Optimized portfolio NAV": "#155eef",
        "Private Side Alpha synthetic NAV price": "#d97706",
    }
    global_min = 0.0
    for col in dd.columns:
        if col == date_col:
            continue
        y = pd.to_numeric(dd[col], errors="coerce").fillna(0.0)
        color = colors.get(col, "#667085")
        label = (
            "Optimized portfolio"
            if "Optimized portfolio" in col
            else "Private Side Alpha"
            if "Private Side Alpha" in col
            else col.replace(" observed price", "")
        )
        ax.fill_between(x, y, 0, color=color, alpha=0.14)
        ax.plot(x, y, color=color, linewidth=1.5, label=label, alpha=0.88)
        if y.min() < 0:
            pos = int(y.values.argmin())
            ax.axhline(y.min(), color=color, linewidth=0.8, linestyle=":", alpha=0.65)
            ax.annotate(f"{label} max DD {y.min():.1%}", xy=(x.iloc[pos], y.iloc[pos]), xytext=(8, -12), textcoords="offset points", fontsize=8, color=color)
        global_min = min(global_min, float(y.min()))
    ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.45)
    ax.set_ylim(min(-0.01, global_min * 1.25), 0.005)
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def plot_equity_vs_benchmark(curve: pd.DataFrame, benchmark_ticker: str, prices: pd.DataFrame):
    return plot_price_path(
        backtest_price_frame(curve, benchmark_ticker, prices),
        benchmark_ticker,
        "Backtest price path: optimized synthetic NAV vs benchmark observed price",
    )


def plot_covariance(cov: pd.DataFrame, compact: bool = True):
    n = int(max(cov.shape)) if cov is not None and not cov.empty else 0
    size = min(4.2, max(2.9, 0.16 * n + 2.2)) if compact else min(7.0, max(4.8, 0.34 * n + 2.8))
    fig, ax = plt.subplots(figsize=(size, size * 0.78))
    if cov is None or cov.empty:
        ax.text(0.5, 0.5, "No covariance matrix", ha="center", va="center")
        return fig
    annot = bool(n <= 6 and compact)
    sns.heatmap(
        cov,
        ax=ax,
        cmap="vlag",
        center=0,
        linewidths=0.15,
        annot=annot,
        fmt=".2g",
        cbar_kws={"shrink": 0.58},
    )
    ax.set_title("Annualized covariance matrix")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7 if compact else 8)
    ax.tick_params(axis="y", labelsize=7 if compact else 8)
    fig.tight_layout()
    return fig


def matrix_stability_diagnostics(cov: pd.DataFrame, corr: pd.DataFrame | None = None) -> pd.DataFrame:
    if cov is None or cov.empty:
        return pd.DataFrame()
    x = cov.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = (x + x.T) / 2.0
    try:
        eig = np.linalg.eigvalsh(x.values.astype(float))
    except Exception:
        eig = np.array([])
    eig_pos = eig[eig > 1e-12]
    total_var = float(np.trace(x.values)) if x.size else np.nan
    largest_share = float(eig_pos.max() / eig_pos.sum()) if eig_pos.size and eig_pos.sum() > 0 else np.nan
    condition = float(eig_pos.max() / eig_pos.min()) if eig_pos.size > 1 else np.nan
    if eig_pos.size and eig_pos.sum() > 0:
        p = eig_pos / eig_pos.sum()
        effective_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()))
    else:
        effective_rank = np.nan
    rows = [
        {"Metric": "Assets", "Value": int(x.shape[0]), "Interpretation": "Number of names in the risk matrix."},
        {"Metric": "Total annualized variance", "Value": total_var, "Interpretation": "Trace of the annualized covariance matrix."},
        {"Metric": "Largest eigenvalue share", "Value": largest_share, "Interpretation": "Spectral concentration; high values imply one dominant risk mode."},
        {"Metric": "Effective rank", "Value": effective_rank, "Interpretation": "Diversification dimension implied by eigenvalue entropy."},
        {"Metric": "Condition number", "Value": condition, "Interpretation": "Numerical stability of the covariance matrix; high values imply unstable inversion."},
    ]
    if corr is not None and not corr.empty:
        c = corr.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = ~np.eye(len(c), dtype=bool)
        vals = c.values[mask]
        vals = vals[np.isfinite(vals)]
        rows.extend(
            [
                {"Metric": "Mean pairwise correlation", "Value": float(np.nanmean(vals)) if vals.size else np.nan, "Interpretation": "Average off-diagonal correlation."},
                {"Metric": "Max absolute pairwise correlation", "Value": float(np.nanmax(np.abs(vals))) if vals.size else np.nan, "Interpretation": "Largest absolute pairwise dependency."},
            ]
        )
    return pd.DataFrame(rows)


def top_correlation_pairs(corr: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if corr is None or corr.empty:
        return pd.DataFrame()
    c = corr.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows = []
    cols = list(c.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = c.loc[a, b] if a in c.index and b in c.columns else np.nan
            if pd.notna(val):
                rows.append({"Asset 1": a, "Asset 2": b, "Correlation": float(val), "Abs correlation": abs(float(val))})
    return pd.DataFrame(rows).sort_values("Abs correlation", ascending=False).head(int(top_n)) if rows else pd.DataFrame()


def plot_overlap_returns(ret: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if ret is None or ret.empty:
        ax.text(0.5, 0.5, "No return distribution data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    if "Date" in ret.columns:
        ret = ret.drop(columns=["Date"])
    ret = ret.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    plotted = False
    for col in ret.columns:
        x = ret[col].dropna()
        if len(x) > 10:
            sns.kdeplot(x=x, ax=ax, label=col, alpha=0.35, linewidth=1.4)
            plotted = True
        elif len(x) >= 2:
            ax.hist(
                x,
                bins=min(8, max(2, len(x))),
                density=True,
                alpha=0.22,
                label=f"{col} (n={len(x)})",
                edgecolor="none",
            )
            y_level = 0.02 + 0.015 * (list(ret.columns).index(col) % 5)
            ax.scatter(x, np.full(len(x), y_level), s=18, alpha=0.75)
            plotted = True
        elif len(x) == 1:
            ax.axvline(float(x.iloc[0]), linewidth=1.8, alpha=0.75, label=f"{col} (n=1)")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "Not enough return observations", ha="center", va="center")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Periodic return")
    ax.set_ylabel("Density")
    if plotted:
        ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def side_boom_return_distribution_frame(side_curve: pd.DataFrame) -> pd.DataFrame:
    if side_curve is None or side_curve.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    if "Period_End" in side_curve:
        out["Date"] = pd.to_datetime(side_curve["Period_End"], errors="coerce")
    for src, dst in [
        ("Side_Boom_Return", "Private Side Alpha"),
        ("Side_Benchmark_Return", "Side benchmark"),
    ]:
        if src in side_curve:
            out[dst] = pd.to_numeric(side_curve[src], errors="coerce")
    for src, dst in [
        ("Side_Boom_Equity", "Private Side Alpha"),
        ("Side_Benchmark_Equity", "Side benchmark"),
    ]:
        if dst not in out and src in side_curve:
            out[dst] = pd.to_numeric(side_curve[src], errors="coerce").pct_change(fill_method=None)
    return out.dropna(how="all")


def benchmark_return_distribution_frame(prices: pd.DataFrame, max_cols: int = 10) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    px = prices.sort_index().ffill().dropna(axis=1, how="all")
    if px.empty:
        return pd.DataFrame()
    ret = px.iloc[:, :max_cols].pct_change(fill_method=None).dropna(how="all").reset_index()
    return ret.rename(columns={ret.columns[0]: "Date"})


def plot_drawdown(curve: pd.DataFrame, perf: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    dd_frame = build_drawdown_frame(curve, perf)
    if dd_frame.empty:
        ax.text(0.5, 0.5, "No drawdown data", ha="center", va="center")
        return fig
    data = curve.copy() if curve is not None and not curve.empty else dd_frame.copy()
    date_col = "Period_End"
    x = pd.to_datetime(data[date_col], errors="coerce")
    specs = [
        ("Portfolio_Equity", "Optimized portfolio", "#155eef", 0.24),
        ("Benchmark_Equity", "Benchmark", "#667085", 0.18),
        ("Side_Boom_Equity", "Private Side Alpha", "#d97706", 0.20),
    ]
    plotted = False
    global_min = 0.0
    for col, label, color, alpha in specs:
        if col not in data:
            continue
        eq = pd.to_numeric(data[col], errors="coerce")
        if eq.notna().sum() < 2:
            continue
        dd = (eq / eq.cummax() - 1.0).fillna(0.0)
        ax.fill_between(x, dd, 0, color=color, alpha=alpha)
        ax.plot(x, dd, color=color, linewidth=1.5, label=label)
        if dd.min() < 0:
            pos = int(dd.argmin())
            ax.axhline(dd.min(), color=color, linewidth=0.9, linestyle=":", alpha=0.85)
            ax.axvline(x.iloc[pos], color=color, linewidth=0.8, linestyle=":", alpha=0.72)
            ax.annotate(f"{label} max DD {dd.min():.1%}", xy=(x.iloc[pos], dd.min()), xytext=(8, -12), textcoords="offset points", fontsize=8, color=color)
            global_min = min(global_min, float(dd.min()))
        plotted = True
    if not plotted and "Drawdown" in dd_frame:
        x = pd.to_datetime(dd_frame["Period_End"])
        dd = pd.to_numeric(dd_frame["Drawdown"], errors="coerce").fillna(0.0)
        ax.fill_between(x, dd, 0, color="#155eef", alpha=0.24)
        ax.plot(x, dd, color="#155eef", linewidth=1.7, label="Optimized portfolio")
        global_min = float(dd.min())
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_ylabel("Drawdown")
    ax.set_ylim(min(-0.01, global_min * 1.25), 0.005)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    return fig


def plot_benchmark_panel(prices: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    if prices is None or prices.empty:
        ax.text(0.5, 0.5, "No benchmarks", ha="center", va="center")
        return fig
    normalized = prices.sort_index().ffill().dropna(axis=1, how="all")
    normalized = normalized / normalized.iloc[0] * 100.0
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, linewidth=1.6, alpha=0.9)
    ax.axhline(100.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Price index (base=100)")
    ax.set_title("Public benchmark price indices")
    ax.legend(frameon=False, ncols=4, fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def plot_global_yield_curves(curves: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax_heat, ax_slope = axes
    if curves is None or curves.empty:
        ax_heat.text(0.5, 0.5, "No global curves", ha="center", va="center")
        ax_slope.axis("off")
        return fig
    matrix = prepare_global_curve_matrix(curves)
    if matrix.empty:
        ax_heat.text(0.5, 0.5, "No complete curve levels", ha="center", va="center")
        ax_slope.axis("off")
        return fig
    sns.heatmap(
        matrix,
        ax=ax_heat,
        cmap="RdYlGn_r",
        center=0,
        linewidths=0.25,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Rate / spread (%)", "shrink": 0.75},
    )
    ax_heat.set_title("Global rates heatmap")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax_heat.tick_params(axis="y", labelsize=8)

    slope = curves.copy()
    slope["Yield_2Y"] = pd.to_numeric(slope.get("Yield_2Y"), errors="coerce")
    slope["Yield_10Y"] = pd.to_numeric(slope.get("Yield_10Y"), errors="coerce")
    slope["Curve_10Y_2Y"] = pd.to_numeric(slope.get("Curve_10Y_2Y"), errors="coerce")
    slope = slope.dropna(subset=["Yield_2Y", "Yield_10Y", "Curve_10Y_2Y"])
    if slope.empty:
        ax_slope.text(0.5, 0.5, "No 2Y/10Y pairs", ha="center", va="center")
    else:
        steep = slope.nlargest(4, "Curve_10Y_2Y")
        inverted = slope.nsmallest(4, "Curve_10Y_2Y")
        selected = pd.concat([steep, inverted]).drop_duplicates("Country").sort_values("Yield_10Y")
        label_y = spread_label_positions(selected["Yield_10Y"].astype(float).tolist(), min_gap=0.22)
        for _, row in selected.iterrows():
            color = "#b42318" if row["Curve_10Y_2Y"] < 0 else "#2563eb"
            ax_slope.plot([0, 1], [row["Yield_2Y"], row["Yield_10Y"]], marker="o", linewidth=1.8, alpha=0.85, color=color)
        for (_, row), y_text in zip(selected.iterrows(), label_y):
            color = "#b42318" if row["Curve_10Y_2Y"] < 0 else "#2563eb"
            ax_slope.plot([1.0, 1.08], [row["Yield_10Y"], y_text], color=color, linewidth=0.7, alpha=0.55)
            ax_slope.text(
                1.105,
                y_text,
                country_flag(row["Country"]),
                va="center",
                ha="left",
                fontsize=13,
                color=color,
            )
        ax_slope.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax_slope.set_xlim(-0.08, 1.25)
        ax_slope.set_xticks([0, 1], ["2Y", "10Y"])
        ax_slope.set_ylabel("Yield (%)")
        ax_slope.set_title("Most inverted / steep curves")
        ax_slope.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


RATE_TENOR_LABELS_APP = {
    "POLICY_RATE": "Policy / short policy rate",
    "SOV_2Y": "2Y sovereign or money-market proxy",
    "SOV_10Y": "10Y sovereign",
}


def plot_discrete_global_rate_evolution(history: pd.DataFrame, tenor_code: str = "SOV_10Y", normalize_frequency: str = "month_end"):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    if history is None or history.empty:
        ax.text(0.5, 0.5, "No discrete rate history", ha="center", va="center")
        return fig
    data = prepare_discrete_rate_plot_data(history, tenor_code, max_countries=10, lookback_days=365 * 3, normalize_frequency=normalize_frequency)
    if data.empty:
        ax.text(0.5, 0.5, f"No {tenor_code} history", ha="center", va="center")
        return fig
    for country, sub in data.groupby("Country"):
        sub = sub.sort_values("Observation_Date")
        ax.step(pd.to_datetime(sub["Observation_Date"]), sub["Rate"], where="post", linewidth=1.5, alpha=0.85, label=country)
        marker_stride = max(1, int(len(sub) / 28))
        marks = sub.iloc[::marker_stride]
        ax.scatter(pd.to_datetime(marks["Observation_Date"]), marks["Rate"], s=9, alpha=0.45)
    view_label = "monthly comparable view" if normalize_frequency == "month_end" else "native observation frequency"
    ax.set_title(f"Discrete sovereign rate evolution: {RATE_TENOR_LABELS_APP.get(tenor_code, tenor_code)} ({view_label})")
    ax.set_xlabel("Observation date")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    return fig


def plot_interbank_reference_rates(ref_rates: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if ref_rates is None or ref_rates.empty:
        ax.text(0.5, 0.5, "No global overnight reference-rate data", ha="center", va="center")
        return fig
    data = ref_rates.copy()
    data["Observation_Date"] = pd.to_datetime(data["Observation_Date"], errors="coerce")
    data["Rate"] = pd.to_numeric(data["Rate"], errors="coerce")
    latest_date = data["Observation_Date"].max()
    cutoff = latest_date - pd.Timedelta(days=365 * 3) if pd.notna(latest_date) else pd.NaT
    if pd.notna(cutoff):
        data = data[data["Observation_Date"].ge(cutoff)]
    for benchmark, sub in data.groupby("Benchmark"):
        sub = sub.sort_values("Observation_Date")
        currency = sub["Currency"].dropna().iloc[-1] if "Currency" in sub and sub["Currency"].dropna().size else ""
        label = f"{benchmark} ({currency})" if currency else benchmark
        ax.step(sub["Observation_Date"], sub["Rate"], where="post", linewidth=1.8, alpha=0.9, label=label)
    ax.set_title("Global overnight reference rates: SOFR, SONIA, ESTR, TONAR")
    ax.set_xlabel("Observation date")
    ax.set_ylabel("Rate (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.text(
        0.01,
        0.02,
        "Cross-currency level differences reflect different monetary regimes; they are not arbitrage spreads without FX basis and hedging costs.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.72,
    )
    fig.tight_layout()
    return fig


def plot_preflight_side(side_curve: pd.DataFrame, benchmark_ticker: str):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    if side_curve is None or side_curve.empty:
        ax.text(0.5, 0.5, "No Private Side Alpha", ha="center", va="center")
        return fig
    x = pd.to_datetime(side_curve["Period_End"])
    if "Side_Boom_Equity" in side_curve:
        ax.plot(x, pd.to_numeric(side_curve["Side_Boom_Equity"], errors="coerce") * 100.0, label="Private Side Alpha", linewidth=2.0)
    if "Side_Benchmark_Equity" in side_curve:
        ax.plot(x, pd.to_numeric(side_curve["Side_Benchmark_Equity"], errors="coerce") * 100.0, label=benchmark_ticker, linewidth=1.7, alpha=0.85)
    ax.axhline(100.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Price/value index (base=100)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return fig


def side_walk_forward_price_frame(side_curve: pd.DataFrame, benchmark_ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
    if side_curve is None or side_curve.empty or "Period_End" not in side_curve:
        return pd.DataFrame()
    benchmark_price = _aligned_benchmark_price(side_curve, prices, benchmark_ticker)
    if benchmark_price.empty or benchmark_price.dropna().empty:
        return pd.DataFrame()
    anchor_price = float(benchmark_price.dropna().iloc[0])
    if not np.isfinite(anchor_price) or anchor_price <= 0:
        return pd.DataFrame()
    out = pd.DataFrame({"Period_End": pd.to_datetime(side_curve["Period_End"], errors="coerce")})
    out[f"{benchmark_ticker} observed price"] = benchmark_price.values
    if "Side_Boom_Equity" in side_curve:
        equity = pd.to_numeric(side_curve["Side_Boom_Equity"], errors="coerce")
        first_equity = float(equity.dropna().iloc[0]) if not equity.dropna().empty else np.nan
        if np.isfinite(first_equity) and abs(first_equity) > 1e-12:
            out["Private Side Alpha synthetic NAV price"] = anchor_price * equity / first_equity
    return out.dropna(subset=["Period_End"]).dropna(axis=1, how="all")


def plot_side_walk_forward_price(side_curve: pd.DataFrame, benchmark_ticker: str, prices: pd.DataFrame):
    return plot_price_path(
        side_walk_forward_price_frame(side_curve, benchmark_ticker, prices),
        benchmark_ticker,
        "Private Side Alpha walk-forward price path",
    )


def plot_geopolitical_thermometer(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12.2, 3.8))
    if summary is None or summary.empty:
        ax.text(0.5, 0.5, "No public timeline signal", ha="center", va="center")
        return fig
    z = pd.to_numeric(summary.get("Robust_Z_Score", summary.get("Z_Score", pd.Series(dtype=float))), errors="coerce")
    if z.notna().any():
        data = summary.assign(_Score=z).dropna(subset=["_Score"]).sort_values("_Score", ascending=True).copy()
        if data.empty:
            ax.text(0.5, 0.5, "No statistically admissible GDELT timeline shocks", ha="center", va="center")
            return fig
        colors = [
            "#b42318" if v >= 2 else "#d97706" if v >= 1 else "#2563eb" if v >= 0 else "#64748b"
            for v in data["_Score"].fillna(0)
        ]
        ax.barh(data["Topic"], data["_Score"], color=colors, alpha=0.82)
        ax.axvline(-2, color="#64748b", linewidth=0.8, linestyle=":")
        ax.axvline(-1, color="#94a3b8", linewidth=0.8, linestyle=":")
        ax.axvline(1, color="#d97706", linewidth=0.9, linestyle="--")
        ax.axvline(2, color="#b42318", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Robust within-topic GDELT news-flow shock")
        ax.set_title("Global geopolitical risk monitor: abnormal attention")
    elif "News_Flow_Score" in summary:
        score = pd.to_numeric(summary["News_Flow_Score"], errors="coerce")
        data = summary.assign(_Score=score).sort_values("_Score", ascending=True).copy()
        colors = ["#d97706" if str(src).startswith("Google") else "#2563eb" for src in data.get("Source", pd.Series("", index=data.index)).astype(str)]
        ax.barh(data["Topic"], data["_Score"].fillna(0), color=colors, alpha=0.82)
        ax.set_xlabel("Article-flow proxy, not a Z-score")
        ax.set_title("Global geopolitical news flow: fallback proxy")
        ax.text(
            0.01,
            0.02,
            "Timeline unavailable: bars are article-count/diversity proxies and can saturate at the query cap.",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.75,
        )
    else:
        ax.text(0.5, 0.5, "No geopolitical score", ha="center", va="center")
        return fig
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_geo_news_heatmap(country_heatmap: pd.DataFrame):
    if px is None or country_heatmap is None or country_heatmap.empty:
        return None
    data = country_heatmap.copy()
    data["Geo_News_Attention_Score"] = pd.to_numeric(data["Geo_News_Attention_Score"], errors="coerce")
    data = data.dropna(subset=["Country", "Geo_News_Attention_Score"])
    if data.empty:
        return None
    fig = px.choropleth(
        data,
        locations="Country",
        locationmode="country names",
        color="Geo_News_Attention_Score",
        hover_name="Country",
        hover_data={
            "Geo_News_Attention_Score": ":.3f",
            "Article_Count": True,
            "Unique_Domains": True,
            "Topic_Count": True,
            "Dominant_Topic": True,
            "Heat_Level": True,
            "Regex_Inferred_Article_Count": True,
            "SourceCountry_Fallback_Count": True,
            "Mean_Geo_Inference_Confidence": ":.2f",
        },
        color_continuous_scale=["#111827", "#1d4ed8", "#f59e0b", "#dc2626"],
        projection="natural earth",
    )
    fig.update_layout(
        template="plotly_dark",
        title="Country-level event-attention heatmap",
        margin=dict(l=0, r=0, t=48, b=0),
        height=720,
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        coloraxis_colorbar=dict(title="Attention score"),
        geo=dict(
            bgcolor="#0b0f19",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#334155",
            projection_type="natural earth",
            landcolor="#111827",
            oceancolor="#020617",
            lakecolor="#020617",
            showocean=True,
            showlakes=True,
        ),
    )
    return fig


def plot_market_sentiment_sem(timeline: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    if timeline is None or timeline.empty or "Latent_Market_Sentiment_SEM" not in timeline:
        ax.text(0.5, 0.5, "No latent sentiment SEM", ha="center", va="center")
        return fig
    data = timeline.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Latent_Market_Sentiment_SEM"] = pd.to_numeric(data["Latent_Market_Sentiment_SEM"], errors="coerce")
    data = data.dropna(subset=["Date", "Latent_Market_Sentiment_SEM"]).tail(504)
    if data.empty:
        ax.text(0.5, 0.5, "No latent sentiment SEM", ha="center", va="center")
        return fig
    x = data["Date"]
    y = data["Latent_Market_Sentiment_SEM"]
    ax.fill_between(x, 0, y, where=y >= 0, color="#16a34a", alpha=0.20, interpolate=True)
    ax.fill_between(x, 0, y, where=y < 0, color="#b42318", alpha=0.20, interpolate=True)
    ax.plot(x, y, color="#155eef", linewidth=2.0, label="Latent sentiment SEM")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(1, color="#16a34a", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.axhline(-1, color="#b42318", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_title("Market sentiment SEM: latent risk-on / risk-off construct")
    ax.set_ylabel("Causal latent z-score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_market_sentiment_loadings(loadings: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if loadings is None or loadings.empty or "Loading" not in loadings:
        ax.text(0.5, 0.5, "No SEM loadings", ha="center", va="center")
        return fig
    data = loadings.copy().head(10).sort_values("Loading")
    colors = ["#b42318" if v < 0 else "#16a34a" for v in data["Loading"]]
    ax.barh(data["Indicator"], data["Loading"], color=colors, alpha=0.82)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("SEM measurement loadings")
    ax.set_xlabel("Loading on latent sentiment")
    ax.grid(axis="x", alpha=0.25)
    return fig


def render_preflight_market(preflight: dict, benchmark_ticker: str):
    st.subheader("Pre-Allocation Market State")
    st.caption("Public-data market state: benchmarks, rates, regimes, latent sentiment, event risk, and the segregated Private Side Alpha sleeve.")
    latest = preflight.get("latest_macro", pd.Series())
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Rates", str(latest.get("Regime_Hawkish_Dovish", "n/a")))
    p2.metric("Market", str(latest.get("Regime_Bull_Bear", "n/a")))
    p3.metric("10Y-2Y curve", f"{latest.get('Country_Curve_10Y_2Y', latest.get('Curve_10Y_2Y', float('nan'))):.2f}" if pd.notna(latest.get("Country_Curve_10Y_2Y", latest.get("Curve_10Y_2Y", float("nan")))) else "n/a")
    p4.metric("Credit spread", f"{latest.get('CREDIT_SPREAD', float('nan')):.2f}" if pd.notna(latest.get("CREDIT_SPREAD", float("nan"))) else "n/a")
    p5.metric("Rates source", str(latest.get("Country_Rate_Source", "public")))

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("**Public benchmarks**")
        st.pyplot(plot_benchmark_panel(preflight.get("benchmark_prices", pd.DataFrame())), clear_figure=True)
    with c2:
        st.markdown("**Country yield curve / regime**")
        macro = preflight.get("macro", pd.DataFrame())
        if not macro.empty:
            st.pyplot(plot_rates_curve(macro), clear_figure=True)
        else:
            st.caption("No macro data.")

    sentiment = preflight.get("market_sentiment_sem", {}) if isinstance(preflight, dict) else {}
    sentiment_timeline = sentiment.get("timeline", pd.DataFrame()) if isinstance(sentiment, dict) else pd.DataFrame()
    sentiment_latest = sentiment.get("latest", pd.DataFrame()) if isinstance(sentiment, dict) else pd.DataFrame()
    sentiment_loadings = sentiment.get("loadings", pd.DataFrame()) if isinstance(sentiment, dict) else pd.DataFrame()
    sentiment_links = sentiment.get("structural_links", pd.DataFrame()) if isinstance(sentiment, dict) else pd.DataFrame()
    sentiment_diag = sentiment.get("diagnostics", pd.DataFrame()) if isinstance(sentiment, dict) else pd.DataFrame()
    st.markdown("**Latent market sentiment SEM**")
    if not sentiment_timeline.empty:
        if not sentiment_latest.empty:
            srow = sentiment_latest.iloc[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("Latent sentiment", f"{float(srow.get('Latent_Market_Sentiment_SEM', float('nan'))):.2f}")
            m2.metric("Risk-on probability", f"{float(srow.get('Sentiment_Prob_Risk_On', float('nan'))):.0%}")
            m3.metric("SEM state", str(srow.get("Sentiment_State", "n/a")))
        sem_left, sem_right = st.columns([1.25, 0.75])
        with sem_left:
            st.pyplot(plot_market_sentiment_sem(sentiment_timeline), clear_figure=True)
        with sem_right:
            st.pyplot(plot_market_sentiment_loadings(sentiment_loadings), clear_figure=True)
        with st.expander("SEM measurement and structural diagnostics", expanded=False):
            st.latex(r"x_t=\Lambda\eta_t+\varepsilon_t,\qquad r_{b,t+1}=\alpha+\beta_{\eta}\eta_t+\beta_f^\top f_t+u_{t+1}")
            st.caption(
                "The latent construct is estimated from rolling causal z-scores of public indicators: benchmark momentum, breadth, realized volatility, VIX/credit/financial-stress proxies, rates, event risk, and geopolitical shocks when available."
            )
            if not sentiment_diag.empty:
                st.dataframe(sentiment_diag, use_container_width=True, hide_index=True)
            if not sentiment_links.empty:
                st.dataframe(sentiment_links, use_container_width=True, hide_index=True)
            if not sentiment_loadings.empty:
                st.dataframe(sentiment_loadings, use_container_width=True, hide_index=True)
    else:
        st.caption("No latent sentiment SEM. Requires enough benchmark and macro history.")

    global_rates = preflight.get("global_rates", pd.DataFrame())
    global_rate_history = preflight.get("global_rate_history", pd.DataFrame())
    st.markdown("**Comparative discrete global sovereign curves**")
    if not global_rates.empty:
        gr1, gr2 = st.columns([1.15, 0.85])
        with gr1:
            st.pyplot(plot_global_yield_curves(global_rates), clear_figure=True)
        with gr2:
            show_cols = [
                "Country",
                "Policy_Rate",
                "Yield_2Y",
                "Short_Rate_Tenor",
                "Yield_10Y",
                "Curve_10Y_2Y",
                "Term_Premium_Proxy",
                "Policy_Observation_Date",
                "Short_Observation_Date",
                "TenY_Observation_Date",
                "TenY_Observation_Frequency",
                "Regime_Hawkish_Dovish",
                "Regime_Bull_Bear",
                "Curve_Shape",
                "Rate_Source",
            ]
            st.dataframe(global_rates[[c for c in show_cols if c in global_rates.columns]], use_container_width=True, hide_index=True)
        if not global_rate_history.empty:
            st.pyplot(plot_discrete_global_rate_evolution(global_rate_history, "SOV_10Y", "month_end"), clear_figure=True)
            st.markdown("**Latest rate observations by country and tenor**")
            st.dataframe(latest_rate_observations(global_rate_history), use_container_width=True, hide_index=True)
            with st.expander("Balanced historical sample", expanded=False):
                st.dataframe(balanced_rate_history_sample(global_rate_history, rows_per_group=3), use_container_width=True, hide_index=True)
    else:
        st.info(
            "The global curve comparison is computed on demand because it queries several public sources. "
            "Press the button to load and cache it."
        )
        if st.button("Load global curves now", key="load_global_rates_inside", use_container_width=True):
            st.session_state["load_global_rates"] = True
            st.rerun()

    interbank_rates = preflight.get("interbank_reference_rates", pd.DataFrame())
    st.markdown("**Global overnight reference-rate governance**")
    st.caption(
        "Active public overnight reference rates: SOFR (USD), SONIA (GBP), ESTR (EUR), "
        "and TONAR/overnight call-rate proxy (JPY). Cross-currency levels are policy diagnostics, "
        "not arbitrage spreads without FX basis and hedging costs."
    )
    if not interbank_rates.empty:
        ib1, ib2 = st.columns([1.1, 0.9])
        latest_ib = (
            interbank_rates.sort_values("Observation_Date")
            .groupby("Benchmark", as_index=False)
            .tail(1)
            .sort_values("Rate", ascending=False)
        )
        with ib1:
            st.pyplot(plot_interbank_reference_rates(interbank_rates), clear_figure=True)
        with ib2:
            cols = [
                "Benchmark",
                "Jurisdiction",
                "Currency",
                "Tenor",
                "Rate",
                "Observation_Date",
                "Observation_Frequency",
                "Level_Diff_vs_SOFR_bps",
                "Data_Staleness_Days",
                "Comparable_To_Current_Funding",
                "Status",
                "Source",
            ]
            st.dataframe(latest_ib[[c for c in cols if c in latest_ib.columns]], use_container_width=True, hide_index=True)
    else:
        st.caption("No global overnight reference-rate data from public sources.")

    ff_calendar = preflight.get("forex_factory_calendar", pd.DataFrame())
    ff_event_risk = preflight.get("forex_factory_event_risk", pd.DataFrame())
    st.markdown("**ForexFactory macro calendar: operational event risk**")
    if not ff_event_risk.empty:
        er1, er2 = st.columns([0.8, 1.2])
        with er1:
            st.dataframe(ff_event_risk, use_container_width=True, hide_index=True)
        with er2:
            st.bar_chart(ff_event_risk.set_index("Currency")["EventRiskScore"])
    else:
        st.caption("No upcoming macro events from ForexFactory/FairEconomy.")
    if not ff_calendar.empty:
        upcoming = ff_calendar.copy()
        upcoming["Event_Time"] = pd.to_datetime(upcoming["Event_Time"], errors="coerce")
        upcoming = upcoming.sort_values(["Event_Time", "Impact_Weight"], ascending=[True, False])
        cols = ["Central_Time", "Timezone", "Event_Time", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous", "Source", "URL"]
        st.dataframe(upcoming[[c for c in cols if c in upcoming.columns]].head(60), use_container_width=True, hide_index=True)

    carry = preflight.get("carry_trade_suggestions", pd.DataFrame())
    st.markdown("**Carry candidates from global rates**")
    if not carry.empty:
        carry_cols = [
            "Long_Currency", "Short_Currency", "Long_Country", "Short_Country",
            "Carry_10Y_Spread", "Carry_Trade_Score", "Long_Curve_10Y_2Y",
            "Short_Curve_10Y_2Y", "Long_Regime", "Short_Regime",
            "Event_Risk_Penalty", "Signal", "Risk_Note",
        ]
        st.dataframe(carry[[c for c in carry_cols if c in carry.columns]].head(20), use_container_width=True, hide_index=True)
    else:
        st.caption("Load global yield curves to compute carry trade candidates.")

    s1, s2 = st.columns([1.1, 0.9])
    with s1:
        st.markdown("**Private Side Alpha vs benchmark**")
        st.pyplot(plot_preflight_side(preflight.get("side_curve", pd.DataFrame()), benchmark_ticker), clear_figure=True)
    with s2:
        st.markdown("**Current Private Side Alpha weights**")
        side_port = preflight.get("side_portfolio", pd.DataFrame())
        if not side_port.empty:
            cols = [c for c in ["Ticker", "Weight", "Fixed_Weight_Constraint", "Fixed_Weight_Target", "Return_Obs", "First_Price_Date", "Compliance_Note"] if c in side_port.columns]
            st.dataframe(side_port[cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No Private Side Alpha sleeve.")

    geo = preflight.get("geopolitical", {}) if isinstance(preflight, dict) else {}
    gsum = geo.get("summary", pd.DataFrame()) if isinstance(geo, dict) else pd.DataFrame()
    gart = geo.get("articles", pd.DataFrame()) if isinstance(geo, dict) else pd.DataFrame()
    if not gart.empty:
        gart = add_english_article_titles(gart, use_cache=True, cache_ttl_hours=168, max_workers=4)
    gcountry = geo.get("country_heatmap", pd.DataFrame()) if isinstance(geo, dict) else pd.DataFrame()
    if gcountry.empty and not gart.empty:
        gcountry = geopolitical_country_heatmap(gart, gsum)
    st.markdown("**Global geopolitical risk monitor**")
    if gsum.empty and not st.session_state.get("load_geopolitical_thermometer"):
        st.info(
            "The geopolitical monitor is on demand to keep the initial load fast. "
            "Press the button to query GDELT and cache it for 24 hours."
        )
        if st.button("Load geopolitical monitor now", key="load_geo_inside", use_container_width=True):
            st.session_state["load_geopolitical_thermometer"] = True
            st.rerun()
    elif gsum.empty:
        st.warning(
            "GDELT was requested, but it returned no usable data or the public source responded slowly. "
            "The app keeps benchmarks, rates, strategic sleeve, and optimization running without blocking."
        )
    else:
        g1, g2 = st.columns([1.05, 1.35])
        with g1:
            st.pyplot(plot_geopolitical_thermometer(gsum), clear_figure=True)
        with g2:
            geo_cols = [
                "Topic", "Latest_Volume", "Baseline_Median", "Robust_Z_Score",
                "Positive_Shock_Score", "Percentile", "Thermometer", "Statistical_Admissibility",
                "Risk_Overlay_Admissible", "Data_Source_Type", "Score_Type",
                "Sample_Size", "Unique_Observations", "Article_Count", "Unique_Domains", "Source",
            ]
            st.dataframe(polished_table(gsum[[c for c in geo_cols if c in gsum.columns]]), use_container_width=True, hide_index=True, height=278)
        audit = geopolitical_thermometer_model_audit(gsum)
        with st.expander("Model audit and robust scaling", expanded=False):
            if not audit.empty:
                st.dataframe(polished_table(audit, metric_col="Metric"), use_container_width=True, hide_index=True)
            st.caption(
                "Validation note: the monitor ranks within-topic abnormal attention, not raw cross-topic volume. "
                "Negative values mean news flow is below its own robust baseline. Fallback article counts are qualitative only."
            )
            st.latex(
                r"Z^{robust}_{k,t}=\frac{V_{k,t}-\operatorname{median}(V_{k,\tau})}{1.4826\,\operatorname{median}(|V_{k,\tau}-\operatorname{median}(V_{k,\tau})|)}"
            )
            st.markdown(
                """
                The constant `1.4826` is the Gaussian consistency correction for the median absolute deviation:
                """
            )
            st.latex(r"\operatorname{MAD}=\operatorname{median}(|X-\operatorname{median}(X)|)\approx 0.67449\sigma")
            st.latex(r"\widehat{\sigma}_{MAD}=\frac{\operatorname{MAD}}{0.67449}\approx 1.4826\,\operatorname{MAD}")
            st.markdown(
                """
                Null fields are intentional risk-control outputs. A robust Z-score is shown only when a topic has
                enough GDELT timeline observations and enough dispersion. If the topic falls back to capped article
                counts, RSS news, or a near-constant time series, `Robust_Z_Score`, `Positive_Shock_Score`, and
                `Percentile` remain null instead of manufacturing false precision.
                """
            )
    if not gsum.empty:
        st.markdown("**Country event-attention map**")
        geo_fig = plot_geo_news_heatmap(gcountry)
        if geo_fig is not None:
            st.plotly_chart(geo_fig, use_container_width=True)
            gcountry_cols = [
                "Country", "Geo_News_Attention_Score", "Article_Count", "Unique_Domains",
                "Topic_Count", "Weighted_Topic_Intensity", "Risk_Overlay_Article_Count",
                "Dominant_Topic", "Geo_Inference_Methods", "Mean_Geo_Inference_Confidence",
                "Regex_Inferred_Article_Count", "SourceCountry_Fallback_Count", "Percentile",
                "Heat_Level", "Data_Source",
            ]
            st.dataframe(polished_table(gcountry[[c for c in gcountry_cols if c in gcountry.columns]].head(30)), use_container_width=True, hide_index=True)
            st.caption(
                "Geo analytics uses regex event-country inference from headlines/query text. "
                "Source-country metadata is used only as a low-confidence fallback. "
                "Multi-country headlines contribute to each inferred event country."
            )
        else:
            st.caption("No country-level article metadata for the geopolitical heatmap.")
    if not gsum.empty:
        with st.container():
            if not gart.empty:
                display_articles = gart.copy()
                if "Title_EN" in display_articles:
                    display_articles["Title"] = display_articles["Title_EN"].fillna(display_articles.get("Title")).astype(str)
                article_cols = [c for c in ["Topic", "Title", "Domain", "SeenDate", "Translation_Status", "URL"] if c in display_articles.columns]
                st.markdown("**Public news feed**")
                st.dataframe(polished_table(display_articles[article_cols].head(40)), use_container_width=True, hide_index=True, height=360)
            else:
                st.caption("No recent GDELT articles.")


def plot_rates_curve(macro: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    latest = macro.dropna(how="all").iloc[-1]
    country = latest.get("Rate_Country", "Country")
    source = latest.get("Country_Rate_Source", "public source")
    points = {
        "Policy": latest.get("POLICY_RATE", latest.get("FEDFUNDS", float("nan"))),
        "2Y": latest.get("SOV_2Y", latest.get("US2Y", float("nan"))),
        "10Y": latest.get("SOV_10Y", latest.get("US10Y", float("nan"))),
    }
    ser = pd.Series(points).dropna()
    ax.plot(ser.index, ser.values, marker="o", linewidth=2.0)
    ax.set_ylabel("Yield / rate")
    ax.set_title(f"Yield curve: {country} | {source}")
    ax.grid(alpha=0.25)
    return fig


def plot_selected_fundamental_ratios(portfolio: pd.DataFrame):
    ratios = ["ROIC", "EV_EBITDA", "FCF_Yield", "NetDebt_EBITDA", "Altman_Z", "Interest_Coverage", "ROE", "Revenue_Growth", "EPS_Growth", "Gross_Margin"]
    cols = [c for c in ratios if c in portfolio.columns]
    fig, axes = plt.subplots(max(1, len(cols)), 1, figsize=(9, max(3, 2.1 * len(cols))))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.barplot(data=portfolio, x=col, y="Ticker", ax=ax, hue="Sector" if "Sector" in portfolio else None, dodge=False)
        if ax.legend_:
            ax.legend_.remove()
        ax.set_title(col)
        ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_option_bid_ask(chain: pd.DataFrame, ticker: str):
    df = chain[chain["Ticker"].eq(ticker)].copy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if df.empty:
        return fig
    expiry = df.sort_values("DTE")["Expiry"].dropna().iloc[0]
    d = df[df["Expiry"].eq(expiry)].copy()
    for opt_type, color in [("call", "#155eef"), ("put", "#b42318")]:
        sub = d[d["Option_Type"].eq(opt_type)].sort_values("Strike")
        if sub.empty:
            continue
        ax.plot(sub["Strike"], sub["Bid"], color=color, alpha=0.65, linewidth=1.4, label=f"{opt_type} bid")
        ax.plot(sub["Strike"], sub["Ask"], color=color, alpha=0.35, linewidth=1.4, linestyle="--", label=f"{opt_type} ask")
    if "Spot" in d and d["Spot"].notna().any():
        ax.axvline(d["Spot"].dropna().iloc[-1], color="black", alpha=0.45, linewidth=1.0)
    ax.set_title(f"{ticker} bid/ask options snapshot - {pd.Timestamp(expiry).date()}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Premium")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=4)
    return fig


def plot_option_iv_smile(chain: pd.DataFrame, ticker: str):
    df = chain[chain["Ticker"].eq(ticker)].copy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if df.empty:
        return fig
    expiry = df.sort_values("DTE")["Expiry"].dropna().iloc[0]
    d = df[df["Expiry"].eq(expiry)].copy()
    for opt_type, color in [("call", "#155eef"), ("put", "#b42318")]:
        sub = d[d["Option_Type"].eq(opt_type)].sort_values("Strike")
        if sub.empty:
            continue
        ax.plot(sub["Strike"], sub["Implied_Vol"], color=color, alpha=0.75, marker=".", linewidth=1.2, label=opt_type)
    ax.set_title(f"{ticker} IV smile snapshot - {pd.Timestamp(expiry).date()}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_gbm_forecast(path: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    if path is None or path.empty:
        ax.text(0.5, 0.5, "No GBM forecast", ha="center", va="center")
        return fig
    x = path["Step"]
    ax.fill_between(x, path["Q05"], path["Q95"], color="#155eef", alpha=0.16, label="5%-95% stochastic band")
    ax.fill_between(x, path["Q25"], path["Q75"], color="#155eef", alpha=0.25, label="25%-75% stochastic band")
    ax.plot(x, path["Q50"], color="#155eef", linewidth=2.0, label="Median path")
    ax.plot(x, path["Q01"], color="#667085", linewidth=0.9, linestyle="--", alpha=0.8, label="1% / 99% tails")
    ax.plot(x, path["Q99"], color="#667085", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_title("GBM stochastic portfolio value forecast")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    return fig


def plot_portfolio_vol_surface(surface_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    if surface_matrix is None or surface_matrix.empty or "Moneyness_Bucket" not in surface_matrix:
        ax.text(0.5, 0.5, "No portfolio volatility surface", ha="center", va="center")
        return fig
    mat = surface_matrix.set_index("Moneyness_Bucket")
    mat = mat.apply(pd.to_numeric, errors="coerce")
    mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if mat.empty:
        ax.text(0.5, 0.5, "No populated volatility buckets", ha="center", va="center")
        return fig
    values = mat.stack().dropna()
    vmin = float(values.quantile(0.05)) if not values.empty else None
    vmax = float(values.quantile(0.95)) if not values.empty else None
    sns.heatmap(
        mat,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".1%",
        linewidths=0.45,
        linecolor="#f1f5f9",
        cbar_kws={"label": "Weighted IV", "shrink": 0.78},
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Portfolio implied volatility surface - populated buckets")
    ax.set_xlabel("Days to expiry bucket")
    ax.set_ylabel("Moneyness bucket K/S")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def _variance_path_for_model(returns: pd.Series, model: str, params_text: str) -> pd.DataFrame:
    x = pd.Series(returns).dropna().astype(float)
    if len(x) < 30:
        return pd.DataFrame()
    y = (x - x.mean()).values
    params = [float(v) for v in str(params_text or "").split(",") if str(v).strip()]
    if not params:
        return pd.DataFrame()
    var0 = float(np.nanvar(y))
    h = [max(var0, 1e-12)]
    eps2 = y * y
    model = str(model)
    for t in range(1, len(y)):
        if model == "CONST" and len(params) >= 1:
            ht = params[0]
        elif model == "ARCH1" and len(params) >= 2:
            omega, a1 = params[:2]
            ht = omega + a1 * eps2[t - 1]
        elif model == "ARCH2" and len(params) >= 3:
            omega, a1, a2 = params[:3]
            ht = omega + a1 * eps2[t - 1] + a2 * (eps2[t - 2] if t >= 2 else var0)
        elif model == "GARCH11" and len(params) >= 3:
            omega, a1, b1 = params[:3]
            ht = omega + a1 * eps2[t - 1] + b1 * h[-1]
        elif model == "GARCH12" and len(params) >= 4:
            omega, a1, b1, b2 = params[:4]
            ht = omega + a1 * eps2[t - 1] + b1 * h[-1] + b2 * (h[-2] if len(h) >= 2 else var0)
        elif model == "GARCH21" and len(params) >= 4:
            omega, a1, a2, b1 = params[:4]
            ht = omega + a1 * eps2[t - 1] + a2 * (eps2[t - 2] if t >= 2 else var0) + b1 * h[-1]
        else:
            ht = var0
        h.append(max(float(ht), 1e-12) if pd.notna(ht) else np.nan)
    price = (1.0 + x).cumprod() * 100.0
    ann_vol = pd.Series(h, index=x.index).pow(0.5) * np.sqrt(252.0)
    one_day = ann_vol / np.sqrt(252.0)
    return pd.DataFrame(
        {
            "Date": x.index,
            "Price_Index": price.values,
            "Conditional_Variance": h,
            "Conditional_Daily_Vol": pd.Series(h, index=x.index).pow(0.5).values,
            "Ann_Conditional_Vol": ann_vol.values,
            "Upper_1Sigma_Price": (price * (1.0 + one_day)).values,
            "Lower_1Sigma_Price": (price * (1.0 - one_day)).values,
        }
    )


def variance_return_series(return_diag: dict, series_name: str) -> pd.Series:
    if not isinstance(return_diag, dict):
        return pd.Series(dtype=float)
    if str(series_name) == "PORTFOLIO":
        pr = return_diag.get("portfolio_returns", pd.DataFrame())
        if pr is None or pr.empty:
            return pd.Series(dtype=float)
        return pd.Series(pd.to_numeric(pr["Portfolio_Return"], errors="coerce").values, index=pd.to_datetime(pr["Date"], errors="coerce")).dropna()
    ind = return_diag.get("individual_returns", pd.DataFrame())
    if ind is None or ind.empty or str(series_name) not in ind:
        return pd.Series(dtype=float)
    return pd.Series(pd.to_numeric(ind[str(series_name)], errors="coerce").values, index=pd.to_datetime(ind["Date"], errors="coerce")).dropna()


def selected_variance_model_row(variance_models: pd.DataFrame, series_name: str) -> pd.Series | None:
    if variance_models is None or variance_models.empty:
        return None
    rows = variance_models[variance_models.get("Series", pd.Series(dtype=str)).astype(str).eq(str(series_name))].copy()
    if rows.empty:
        return None
    row = rows[rows["Best_BIC"].fillna(False).astype(bool)].head(1) if "Best_BIC" in rows else pd.DataFrame()
    if row.empty:
        row = rows.sort_values("BIC", na_position="last").head(1) if "BIC" in rows else rows.head(1)
    return row.iloc[0] if not row.empty else None


def arch_garch_forecast_cone(returns: pd.Series, model_row: pd.Series, horizon_days: int = 63) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.Series(returns).dropna().astype(float)
    if len(x) < 30 or model_row is None:
        return pd.DataFrame(), pd.DataFrame()
    model = str(model_row.get("Model"))
    params = [float(v) for v in str(model_row.get("Params", "") or "").split(",") if str(v).strip()]
    hist = _variance_path_for_model(x, model, model_row.get("Params"))
    if hist.empty or not params:
        return pd.DataFrame(), pd.DataFrame()
    log_r = np.log1p(x.clip(lower=-0.999))
    mu = float(log_r.mean())
    price0 = float(hist["Price_Index"].iloc[-1])
    h_last = float(hist["Conditional_Variance"].iloc[-1])
    eps = (x - x.mean()).values
    eps2_lags = [float(eps[-1] ** 2), float(eps[-2] ** 2) if len(eps) >= 2 else h_last]
    h_lags = [h_last, float(hist["Conditional_Variance"].iloc[-2]) if len(hist) >= 2 else h_last]
    h_fore = []
    for _ in range(int(horizon_days)):
        if model == "CONST" and len(params) >= 1:
            h_next = params[0]
        elif model == "ARCH1" and len(params) >= 2:
            omega, a1 = params[:2]
            h_next = omega + a1 * eps2_lags[0]
        elif model == "ARCH2" and len(params) >= 3:
            omega, a1, a2 = params[:3]
            h_next = omega + a1 * eps2_lags[0] + a2 * eps2_lags[1]
        elif model == "GARCH11" and len(params) >= 3:
            omega, a1, b1 = params[:3]
            h_next = omega + a1 * eps2_lags[0] + b1 * h_lags[0]
        elif model == "GARCH12" and len(params) >= 4:
            omega, a1, b1, b2 = params[:4]
            h_next = omega + a1 * eps2_lags[0] + b1 * h_lags[0] + b2 * h_lags[1]
        elif model == "GARCH21" and len(params) >= 4:
            omega, a1, a2, b1 = params[:4]
            h_next = omega + a1 * eps2_lags[0] + a2 * eps2_lags[1] + b1 * h_lags[0]
        else:
            h_next = h_last
        h_next = max(float(h_next), 1e-12) if np.isfinite(h_next) else max(h_last, 1e-12)
        h_fore.append(h_next)
        eps2_lags = [h_next, eps2_lags[0]]
        h_lags = [h_next, h_lags[0]]
    cum_var = np.cumsum(h_fore)
    steps = np.arange(1, int(horizon_days) + 1)
    qs = {"Q05": -1.6448536269514722, "Q25": -0.6744897501960817, "Q50": 0.0, "Q75": 0.6744897501960817, "Q95": 1.6448536269514722}
    cone = pd.DataFrame({"Step": steps, "Ann_Conditional_Vol": np.sqrt(h_fore) * np.sqrt(252.0)})
    for name, z in qs.items():
        cone[name] = price0 * np.exp(mu * steps + z * np.sqrt(cum_var))
    summary = pd.DataFrame(
        [
            {"Metric": "Selected_Model", "Value": model},
            {"Metric": "Horizon_Days", "Value": int(horizon_days)},
            {"Metric": "Start_Price_Index", "Value": price0},
            {"Metric": "Forecast_Ann_Vol_T1", "Value": float(cone["Ann_Conditional_Vol"].iloc[0])},
            {"Metric": "Forecast_Ann_Vol_TEnd", "Value": float(cone["Ann_Conditional_Vol"].iloc[-1])},
            {"Metric": "Terminal_Q05", "Value": float(cone["Q05"].iloc[-1])},
            {"Metric": "Terminal_Q50", "Value": float(cone["Q50"].iloc[-1])},
            {"Metric": "Terminal_Q95", "Value": float(cone["Q95"].iloc[-1])},
        ]
    )
    return cone, summary


def plot_arch_garch_price_bands(return_diag: dict, variance_models: pd.DataFrame, series_name: str):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax, ax_vol = axes
    if not isinstance(return_diag, dict) or variance_models is None or variance_models.empty:
        ax.text(0.5, 0.5, "No ARCH/GARCH price-band data", ha="center", va="center")
        ax_vol.axis("off")
        return fig
    r = variance_return_series(return_diag, series_name)
    row = selected_variance_model_row(variance_models, series_name)
    if r.empty or row is None:
        ax.text(0.5, 0.5, f"No variance architecture for {series_name}", ha="center", va="center")
        ax_vol.axis("off")
        return fig
    path = _variance_path_for_model(r, row.get("Model"), row.get("Params"))
    if path.empty:
        ax.text(0.5, 0.5, "No conditional variance path", ha="center", va="center")
        ax_vol.axis("off")
        return fig
    x = pd.to_datetime(path["Date"], errors="coerce")
    ax.fill_between(x, path["Lower_1Sigma_Price"], path["Upper_1Sigma_Price"], color="#155eef", alpha=0.16, label="One-day 68% conditional band")
    ax.plot(x, path["Price_Index"], color="#155eef", linewidth=2.0, label=f"{series_name} price index")
    ax.set_ylabel("Price/value index (base=100)")
    ax.set_title(f"Historical conditional variance band - {series_name} | selected {row.get('Model')}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax_vol.plot(x, path["Ann_Conditional_Vol"], color="#d97706", linewidth=1.4, alpha=0.86, label="Annualized conditional volatility")
    ax_vol.set_ylabel("Ann. vol")
    ax_vol.grid(alpha=0.25)
    ax_vol.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_arch_garch_forecast_cone(return_diag: dict, variance_models: pd.DataFrame, series_name: str, horizon_days: int = 63):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    r = variance_return_series(return_diag, series_name)
    row = selected_variance_model_row(variance_models, series_name)
    cone, _summary = arch_garch_forecast_cone(r, row, horizon_days=horizon_days)
    if cone.empty:
        ax.text(0.5, 0.5, "No ARCH/GARCH forecast cone", ha="center", va="center")
        return fig
    x = cone["Step"]
    ax.fill_between(x, cone["Q05"], cone["Q95"], color="#155eef", alpha=0.14, label="90% conditional cone")
    ax.fill_between(x, cone["Q25"], cone["Q75"], color="#155eef", alpha=0.26, label="50% conditional cone")
    ax.plot(x, cone["Q50"], color="#155eef", linewidth=2.0, label="Median forecast")
    ax.set_title(f"ARCH/GARCH forecast cone - {series_name} | selected {row.get('Model') if row is not None else 'n/a'}")
    ax.set_xlabel("Forecast trading days")
    ax.set_ylabel("Price/value index")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_variance_model_comparison(variance_models: pd.DataFrame, series_name: str):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    if variance_models is None or variance_models.empty:
        ax.text(0.5, 0.5, "No variance model selection", ha="center", va="center")
        return fig
    data = variance_models[variance_models["Series"].astype(str).eq(str(series_name))].copy() if "Series" in variance_models else variance_models.copy()
    if data.empty:
        ax.text(0.5, 0.5, f"No variance models for {series_name}", ha="center", va="center")
        return fig
    data["BIC"] = pd.to_numeric(data["BIC"], errors="coerce")
    data["AIC"] = pd.to_numeric(data["AIC"], errors="coerce")
    data = data.sort_values("BIC")
    x = range(len(data))
    ax.bar(x, data["BIC"], color="#155eef", alpha=0.75, label="BIC")
    ax.plot(x, data["AIC"], color="#d97706", marker="o", linewidth=1.5, label="AIC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["Model"].astype(str), rotation=0)
    ax.set_title(f"ARCH/GARCH architecture selection - {series_name}")
    ax.set_ylabel("Information criterion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_pelt_regime_timeline(timeline: pd.DataFrame, title: str = "PELT regime change analysis: portfolio realized volatility"):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    if timeline is None or timeline.empty:
        ax.text(0.5, 0.5, "No PELT regime analysis", ha="center", va="center")
        return fig
    data = timeline.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Rolling_21D_Ann_Vol"] = pd.to_numeric(data.get("Rolling_21D_Ann_Vol"), errors="coerce")
    for seg, sub in data.groupby("Segment"):
        ax.plot(sub["Date"], sub["Rolling_21D_Ann_Vol"], linewidth=1.8, alpha=0.9, label=f"Regime {seg}")
    cp_flags = data["Is_Change_Point"].fillna(False).astype(bool) if "Is_Change_Point" in data else pd.Series(False, index=data.index)
    cps = data[cp_flags]
    for _, row in cps.iterrows():
        ax.axvline(row["Date"], color="#b42318", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("21D annualized volatility")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    return fig


def build_rag_corpus(results: dict) -> list[dict]:
    corpus = []

    def add_frame(name: str, df: pd.DataFrame, max_rows: int = 80):
        if df is None or df.empty:
            return
        for row in df.head(max_rows).to_dict(orient="records"):
            text = f"{name}: " + "; ".join(f"{k}={v}" for k, v in row.items() if pd.notna(v))
            corpus.append({"source": name, "text": text})

    add_frame("selected portfolio", results.get("portfolio", pd.DataFrame()), 40)
    add_frame("performance summary", results.get("performance_summary", pd.DataFrame()), 20)
    add_frame("benchmark governance", results.get("benchmark_governance", pd.DataFrame()), 10)
    add_frame("suitability", results.get("suitability_diagnostics", pd.DataFrame()), 10)
    add_frame("options", results.get("options_summary", pd.DataFrame()), 60)
    add_frame("portfolio volatility surface", results.get("portfolio_vol_surface", pd.DataFrame()), 80)
    add_frame("portfolio volatility surface diagnostics", results.get("portfolio_vol_surface_diagnostics", pd.DataFrame()), 20)
    ret_diag = results.get("return_diagnostics", {})
    if isinstance(ret_diag, dict):
        add_frame("gbm forecast", ret_diag.get("gbm_forecast_summary", pd.DataFrame()), 40)
        add_frame("variance model selection", ret_diag.get("variance_model_selection", pd.DataFrame()), 80)
        add_frame("pelt regime segments", ret_diag.get("pelt_regime_segments", pd.DataFrame()), 20)
        add_frame("pelt change points", ret_diag.get("pelt_change_points", pd.DataFrame()), 20)
    add_frame("hedge suggestions", results.get("hedge_suggestions", pd.DataFrame()), 40)
    add_frame("carry validation", results.get("carry_trade_validation", pd.DataFrame()), 40)
    add_frame("global discrete rate history", results.get("global_rate_history", pd.DataFrame()), 80)
    add_frame("global overnight reference rates", results.get("interbank_reference_rates", pd.DataFrame()), 80)
    add_frame("private side alpha", results.get("side_boom_diagnostics", pd.DataFrame()), 20)
    add_frame("private side alpha pelt segments", results.get("side_boom_pelt_regime_segments", pd.DataFrame()), 20)
    add_frame("private side alpha pelt changes", results.get("side_boom_pelt_change_points", pd.DataFrame()), 20)
    add_frame("macro latest", pd.DataFrame([results.get("latest_macro", pd.Series()).to_dict()]), 1)
    cross = results.get("cross_section", pd.DataFrame())
    if not cross.empty:
        cols = [
            "Ticker", "Sector", "Country", "Composite_Score", "Value_Score", "Quality_Score",
            "Growth_Score", "Technical_Score", "Liquidity_Score", "Revenue_Growth", "EPS_Growth",
            "ROIC", "EV_EBITDA", "FCF_Yield", "NetDebt_EBITDA", "Altman_Z", "Mahalanobis",
            "Reject_Reasons",
        ]
        add_frame("ranking and fundamentals", cross[[c for c in cols if c in cross.columns]], 80)
    return corpus


def answer_rag_question(question: str, corpus: list[dict]) -> tuple[str, pd.DataFrame]:
    if not question.strip() or not corpus:
        return "I need a loaded run and a concrete question to retrieve evidence.", pd.DataFrame()
    if TfidfVectorizer is None or cosine_similarity is None:
        return "The local RAG assistant is unavailable because sklearn did not load in this environment.", pd.DataFrame()
    texts = [x["text"] for x in corpus]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts + [question])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    order = sims.argsort()[::-1][:6]
    evidence = pd.DataFrame(
        [{"score": float(sims[i]), "source": corpus[i]["source"], "evidence": corpus[i]["text"]} for i in order if sims[i] > 0]
    )
    if evidence.empty:
        return "I did not find enough evidence in the current run. Check that the pipeline has run or rephrase with tickers/metrics.", evidence
    answer = (
        "Answer based on the current run, not on a discretionary recommendation:\n\n"
        + "\n".join(f"- {row.source}: {row.evidence[:650]}" for row in evidence.itertuples(index=False))
    )
    return answer, evidence


st.markdown(
    """
    <div class="qpk-hero">
        <div class="qpk-kicker">Portfolio decision system</div>
        <div class="qpk-title">Quant Portfolio-Kaizen</div>
        <div class="qpk-subtitle">
            Auditable allocation, downside control, and benchmark-relative evidence from causal public-data pipelines.
        </div>
        <div class="qpk-meta">Central Time · daily data snapshot · semiannual rebalance · annual reoptimization</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Allocation Setup")
    price_period = st.selectbox("Base period", ["2y", "3y", "5y", "10y"], index=1)
    benchmark_group = st.selectbox("Benchmark type", ["US Market", "US Sector", "Country", "International", "Custom"], index=0)
    if benchmark_group == "Custom":
        benchmark_ticker = st.text_input("Benchmark custom", value="SPY").strip().upper()
    else:
        preset_benchmarks = BENCHMARK_PRESETS[benchmark_group]
        benchmark_label = st.selectbox("Benchmark", [f"{k} - {v}" for k, v in preset_benchmarks.items()])
        benchmark_ticker = benchmark_label.split(" - ")[0]
    benchmark_mandate_type = st.selectbox(
        "Benchmark mandate",
        ["Absolute", "Relative vs benchmark", "Country", "Sector", "International"],
        index={"US Market": 1, "US Sector": 3, "Country": 2, "International": 4, "Custom": 1}.get(benchmark_group, 1),
    )
    inferred_country = BENCHMARK_COUNTRY.get(benchmark_ticker, "United States")
    rate_countries = ["United States", "Mexico", "Canada", "United Kingdom", "Germany", "France", "Spain", "Japan", "Brazil"]
    rate_country = st.selectbox(
        "Country for sovereign curve",
        rate_countries,
        index=rate_countries.index(inferred_country) if inferred_country in rate_countries else 0,
    )
    compute_mode_label = st.radio("Compute profile", ["Fast", "Rigorous"], index=0, horizontal=True)
    compute_mode = {"Fast": "Rapido", "Rigorous": "Riguroso"}[compute_mode_label]

    st.divider()
    st.header("Suitability Profile")
    use_suitability_engine = True
    investor_horizon_label = st.selectbox("Horizon", ["6 months", "1 year", "3 years", "5 years", "10+ years"], index=2)
    horizon_map = {"6 months": 0.5, "1 year": 1.0, "3 years": 3.0, "5 years": 5.0, "10+ years": 10.0}
    investor_horizon_years = horizon_map[investor_horizon_label]
    investor_initial_capital = st.number_input("Initial capital", min_value=500.0, max_value=100_000_000.0, value=100_000.0, step=5_000.0)
    investor_monthly_contribution = st.number_input("Monthly contribution", min_value=0.0, max_value=1_000_000.0, value=0.0, step=500.0)
    investor_liquidity_need = st.selectbox("Liquidity need", ["Low", "Medium", "High"], index=1)
    investor_max_drawdown = st.slider("Maximum tolerated drawdown", 0.05, 0.60, 0.20, 0.01)
    investor_risk_aversion_score = st.slider("Risk aversion", 0.0, 10.0, 5.0, 0.5)
    investor_objective = st.selectbox(
        "Objective",
        ["Capital preservation", "Income", "Balanced growth", "Aggressive growth", "High conviction"],
        index=2,
    )
    investor_base_currency = st.selectbox("Base currency", ["USD", "MXN", "CAD", "EUR", "GBP", "JPY", "BRL"], index=0)
    suitability = build_suitability_constraints(
        investor_horizon_years,
        investor_initial_capital,
        investor_monthly_contribution,
        investor_liquidity_need,
        investor_max_drawdown,
        investor_risk_aversion_score,
        investor_objective,
        base_currency=investor_base_currency,
        parameter_mode="automatic",
    )
    profile_label = PROFILE_EN.get(suitability["Suitability_Profile"], suitability["Suitability_Profile"])
    st.caption(
        f"Profile: {profile_label} | score={suitability['Suitability_Score']:.2f} | "
        f"max vol={suitability['Vol_Max']:.0%} | daily CVaR={suitability['CVaR_Max_Daily']:.1%}"
    )
    for warning in suitability["Warnings"]:
        st.warning(warning)

    universe_text = DEFAULT_UNIVERSE.strip()
    use_persistent_cache = True
    cache_ttl_hours = 24
    universe_source = "Manual"
    universe_asof = datetime.now().date()
    universe_mode = "Merge with manual"
    max_public_names = 120
    sec_user_agent = "QuantStockPicker/1.0 contact@example.com"
    use_sec_edgar = True
    use_sec_nlp = True
    sec_nlp_max_tickers = 30
    sec_nlp_max_filings = 2
    accounting_lag_days = 90
    max_workers = 8
    benchmark_auto_select = False
    use_side_boom_portfolio = True
    side_boom_text = " ".join(DEFAULT_SIDE_ALPHA_TICKERS)
    side_boom_fixed_ticker = "CBRS"
    side_boom_fixed_weight = float(DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT)
    side_boom_extra_fixed_weights = DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS
    side_boom_min_obs = 60

    top_n = int(suitability["Top_N_Max"])
    preselect_n = max(8, min(24, max(14, top_n + 4)))
    min_chunk = int(suitability["Min_Chunk"])
    max_chunk = int(suitability["Max_Chunk"])
    max_combos = 5000 if compute_mode == "Rapido" else 25000
    parameter_mode = "Automatic by risk aversion"
    risk_profile = suitability["Suitability_Profile"]
    preset = RISK_PRESETS[risk_profile]
    max_sector = int(suitability["Max_Names_Per_Sector"])
    max_weight = float(suitability["Max_Weight"])
    sector_weight_cap = float(suitability["Sector_Weight_Cap"])
    risk_aversion = preset["risk_aversion"]
    alpha_weight = preset["alpha_weight"]
    target_vol = float(suitability["Target_Vol"])
    entropy_penalty = float(suitability["Entropy_Penalty"])
    crlb_penalty = preset["crlb_penalty"]
    garch_penalty = preset["garch_penalty"]
    evt_penalty = preset["evt_penalty"]
    cvar_penalty = float(suitability["CVaR_Penalty"])
    cvar_alpha = 0.95
    portfolio_notional = float(investor_initial_capital)
    max_adv_participation = float(suitability["Max_ADV_Participation"])
    impact_coefficient = preset["impact_coefficient"]
    factor_cov_blend = float(suitability["Factor_Cov_Blend"])
    objective_label = st.selectbox("Quantitative objective", list(OBJECTIVE_LABELS.keys()), index=0)
    weight_objective = OBJECTIVE_LABELS[objective_label]
    st.caption(OBJECTIVE_HELP.get(weight_objective, "Robust quantitative objective."))

    use_garch = compute_mode == "Riguroso"
    nested_validation_fraction = 0.35
    purge_days = 5
    sortino_multistarts = 3 if compute_mode == "Rapido" else 8
    bootstrap_samples = 16 if compute_mode == "Rapido" else 64
    garch_candidate_n = 30
    robust_alpha_uncertainty = 0.35
    robust_cov_uncertainty = 0.10
    text_risk_penalty = 0.10
    use_black_litterman = weight_objective == "black_litterman"
    black_litterman_tau = 0.05
    use_latent_macro_regime = True
    latent_regime_states = 4
    latent_regime_refit_days = 42 if compute_mode == "Rapido" else 21
    latent_regime_min_train = 252
    min_dollar_volume = int(suitability["Min_Dollar_Volume"])
    use_options_snapshot = True
    option_expiries = 3
    options_cache_ttl_hours = 24
    validation_bootstrap_samples = 256 if compute_mode == "Rapido" else 512
    reality_check_samples = 128 if compute_mode == "Rapido" else 512
    cpcv_folds = 4
    max_oos_trials_per_rebalance = 100 if compute_mode == "Rapido" else 200
    robust_selection_lambda = 0.50
    robust_selection_min_obs = 4
    model_confidence_window = 6
    model_confidence_min = 0.25
    use_kaizen_bandit = True
    kaizen_ucb_alpha = 0.75
    use_forex_factory_calendar = True
    use_dynamic_sizing = True
    min_dynamic_exposure = 0.25
    max_dynamic_exposure = 1.00
    regime_entropy_exposure_penalty = 0.50
    markov_stress_exposure_penalty = 0.50
    markov_transition_min_obs = 60
    use_gdelt = False
    gdelt_query = "(tariff OR sanctions OR geopolitical OR election OR regulation OR fiscal policy)"
    factor_mkt_cap = 1.25
    factor_rates_cap = 0.25
    factor_credit_cap = 0.25
    factor_oil_cap = 0.35
    factor_usd_cap = 0.35
    tc_bps = 10
    embargo_days = 5

    with st.expander("Advanced settings", expanded=False):
        st.caption("Research controls. Automatic mode derives defaults from suitability, liquidity, and mandate constraints.")
        universe_text = st.text_area("Universe", value=universe_text, height=180)
        use_persistent_cache = st.checkbox("Persistent parquet cache", value=use_persistent_cache)
        cache_ttl_hours = st.select_slider("Cache TTL hours", options=[1, 6, 12, 24, 72, 168], value=cache_ttl_hours)
        universe_source = st.selectbox(
            "Universe source",
            ["Manual", "Wikipedia S&P 500 as-of", "NasdaqTrader listed", "SEC company tickers", "Stooq local delisted CSV", "Kaggle local CSV"],
            index=0,
        )
        universe_asof = st.date_input("Universe as-of date", value=universe_asof)
        universe_mode = st.radio("Universe usage", ["Replace manual", "Merge with manual"], index=1, horizontal=True)
        max_public_names = st.slider("Max names from public source", 20, 800, max_public_names, 20)
        sec_user_agent = st.text_input("SEC User-Agent", value=sec_user_agent)
        use_sec_edgar = st.checkbox("SEC EDGAR companyfacts PIT", value=use_sec_edgar)
        use_sec_nlp = st.checkbox("SEC NLP 10-K/10-Q", value=use_sec_nlp)
        sec_nlp_max_tickers = st.slider("SEC NLP max tickers", 5, 120, sec_nlp_max_tickers, 5)
        sec_nlp_max_filings = st.slider("SEC NLP filings per ticker", 1, 4, sec_nlp_max_filings)
        accounting_lag_days = st.slider("Causal accounting lag days", 45, 180, accounting_lag_days, 15)
        max_workers = st.slider("Parallel workers", 1, 16, max_workers)
        benchmark_auto_select = st.checkbox("Auto benchmark suggested", value=benchmark_auto_select)

        st.markdown("**Private Side Alpha sleeve**")
        use_side_boom_portfolio = st.checkbox("Enable Private Side Alpha sleeve", value=use_side_boom_portfolio)
        side_boom_text = st.text_area("Side Alpha tickers", value=side_boom_text, height=80)
        side_boom_fixed_ticker = "CBRS"
        side_boom_fixed_weight = float(DEFAULT_SIDE_ALPHA_CEREBRAS_WEIGHT)
        side_boom_extra_fixed_weights = DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS
        fixed_side_alpha = pd.DataFrame(
            list(DEFAULT_SIDE_ALPHA_FIXED_WEIGHTS) + [(side_boom_fixed_ticker, side_boom_fixed_weight)],
            columns=["Ticker", "Fixed_Weight"],
        )
        st.caption(f"Fixed schedule: listed weights plus residual CBRS weight of {side_boom_fixed_weight:.8%}.")
        st.dataframe(fixed_side_alpha, use_container_width=True, hide_index=True)
        side_boom_min_obs = st.slider("Minimum observations for Private Side Alpha", 20, 252, side_boom_min_obs, 10)
        st.warning("Governance: Private Side Alpha is a segregated scenario. It must not use material non-public information in public scoring, RAG, or third-party recommendations.")

        st.markdown("**Selection and allocation**")
        top_n = st.slider("Maximum portfolio names", 3, 20, top_n)
        preselect_n = st.slider("Chunk preselection", 8, 24, preselect_n)
        min_chunk = st.slider("Minimum chunk", 3, 12, min(min_chunk, top_n))
        max_chunk = st.slider("Maximum chunk", min_chunk, 16, min(max_chunk, max(top_n, min_chunk)))
        max_combos = st.select_slider("Max combinations", options=[1000, 5000, 10000, 25000, 50000], value=max_combos)
        max_sector = st.slider("Max names per sector", 1, 6, max_sector)
        max_weight = st.slider("Maximum single-name weight", 0.05, 0.50, max_weight, 0.01)
        sector_weight_cap = st.slider("Maximum sector weight", 0.10, 0.80, sector_weight_cap, 0.05)
        risk_aversion = st.slider("Variance aversion", 0.0, 12.0, risk_aversion, 0.5)
        alpha_weight = st.slider("Alpha score weight", 0.0, 5.0, alpha_weight, 0.25)
        entropy_penalty = st.slider("Low-entropy penalty", 0.00, 0.50, entropy_penalty, 0.01)
        crlb_penalty = st.slider("CRLB penalty", 0.00, 0.75, crlb_penalty, 0.01)
        garch_penalty = st.slider("GARCH volatility penalty", 0.00, 0.75, garch_penalty, 0.01)
        evt_penalty = st.slider("EVT tail penalty", 0.00, 0.75, evt_penalty, 0.01)
        cvar_penalty = st.slider("CVaR penalty", 0.00, 1.50, cvar_penalty, 0.05)
        cvar_alpha = st.select_slider("CVaR alpha", options=[0.90, 0.95, 0.975, 0.99], value=cvar_alpha)
        portfolio_notional = st.number_input("Notional portfolio", min_value=1_000.0, max_value=100_000_000.0, value=portfolio_notional, step=10_000.0)
        max_adv_participation = st.slider("Max ADV participation", 0.001, 0.25, max_adv_participation, 0.005)
        impact_coefficient = st.slider("Volatility impact coefficient", 0.0, 0.50, impact_coefficient, 0.01)
        factor_cov_blend = st.slider("Blend covariance factor model", 0.00, 1.00, factor_cov_blend, 0.05)
        use_garch = st.checkbox("GARCH(1,1) on candidates", value=use_garch)
        nested_validation_fraction = st.slider("Nested validation fraction", 0.20, 0.50, nested_validation_fraction, 0.05)
        purge_days = st.slider("Purged days train-validation", 0, 20, purge_days)
        sortino_multistarts = st.slider("Multistarts Sortino", 1, 24, sortino_multistarts)
        bootstrap_samples = st.slider("Bootstrap stability samples", 0, 256, bootstrap_samples, 16)
        garch_candidate_n = st.slider("GARCH only top candidates", 5, 80, garch_candidate_n, 5)
        robust_alpha_uncertainty = st.slider("Robust alpha uncertainty", 0.0, 1.50, robust_alpha_uncertainty, 0.05)
        robust_cov_uncertainty = st.slider("Robust covariance uncertainty", 0.0, 0.75, robust_cov_uncertainty, 0.05)
        text_risk_penalty = st.slider("SEC TextRisk penalty", 0.00, 0.75, text_risk_penalty, 0.01)
        use_black_litterman = st.checkbox("Black-Litterman alpha posterior", value=use_black_litterman)
        black_litterman_tau = st.slider("Black-Litterman tau", 0.005, 0.250, black_litterman_tau, 0.005)
        use_latent_macro_regime = st.checkbox("Causal latent regime", value=use_latent_macro_regime)
        latent_regime_states = st.slider("Macro latent states", 2, 6, latent_regime_states)
        latent_regime_refit_days = st.select_slider("Latent regime refit", options=[10, 21, 42, 63], value=latent_regime_refit_days)
        latent_regime_min_train = st.select_slider("Minimum latent regime train", options=[126, 252, 504], value=latent_regime_min_train)
        adv_options = [0, 500_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000]
        min_dollar_volume = st.select_slider("Minimum dollar ADV", options=adv_options, value=min(adv_options, key=lambda x: abs(x - min_dollar_volume)))

        st.markdown("**Options, validation, and execution**")
        use_options_snapshot = st.checkbox("Public options snapshot", value=use_options_snapshot)
        option_expiries = st.slider("Expiries per ticker", 1, 6, option_expiries)
        options_cache_ttl_hours = st.select_slider("Options TTL hours", options=[1, 3, 6, 12, 24], value=options_cache_ttl_hours)
        validation_bootstrap_samples = st.select_slider("Validation bootstrap", options=[0, 128, 256, 512, 1024], value=validation_bootstrap_samples)
        reality_check_samples = st.select_slider("White/SPA bootstrap", options=[0, 128, 256, 512, 1024], value=reality_check_samples)
        cpcv_folds = st.slider("CPCV folds", 2, 8, cpcv_folds)
        max_oos_trials_per_rebalance = st.select_slider("OOS trials per rebalance", options=[25, 50, 100, 200, 500], value=max_oos_trials_per_rebalance)
        robust_selection_lambda = st.slider("Robust IS/OOS selection lambda", 0.0, 2.0, robust_selection_lambda, 0.05)
        robust_selection_min_obs = st.slider("Minimum trial-persistence observations", 2, 12, robust_selection_min_obs)
        model_confidence_window = st.slider("Model confidence window", 3, 18, model_confidence_window)
        model_confidence_min = st.slider("Minimum model confidence", 0.05, 1.00, model_confidence_min, 0.05)
        use_kaizen_bandit = st.checkbox("Kaizen contextual bandit", value=use_kaizen_bandit)
        kaizen_ucb_alpha = st.slider("Kaizen LinUCB exploration", 0.00, 2.00, kaizen_ucb_alpha, 0.05)
        use_forex_factory_calendar = st.checkbox("ForexFactory event-risk calendar", value=use_forex_factory_calendar)
        use_dynamic_sizing = st.checkbox("Dynamic sizing", value=use_dynamic_sizing)
        min_dynamic_exposure = st.slider("Minimum dynamic exposure", 0.00, 1.00, min_dynamic_exposure, 0.05)
        max_dynamic_exposure = st.slider("Maximum dynamic exposure", min_dynamic_exposure, 1.50, max_dynamic_exposure, 0.05)
        regime_entropy_exposure_penalty = st.slider("Regime-entropy exposure penalty", 0.00, 1.00, regime_entropy_exposure_penalty, 0.05)
        markov_stress_exposure_penalty = st.slider("Markov stress exposure penalty", 0.00, 1.00, markov_stress_exposure_penalty, 0.05)
        markov_transition_min_obs = st.select_slider("Minimum Markov transition observations", options=[20, 40, 60, 126, 252], value=markov_transition_min_obs)
        use_gdelt = st.checkbox("GDELT inside pipeline", value=use_gdelt)
        gdelt_query = st.text_input("Query GDELT", value=gdelt_query)
        factor_mkt_cap = st.slider("|Beta MKT| max", 0.10, 2.00, factor_mkt_cap, 0.05)
        factor_rates_cap = st.slider("|Beta Rates| max", 0.00, 1.00, factor_rates_cap, 0.05)
        factor_credit_cap = st.slider("|Beta Credit| max", 0.00, 1.00, factor_credit_cap, 0.05)
        factor_oil_cap = st.slider("|Beta Oil| max", 0.00, 1.50, factor_oil_cap, 0.05)
        factor_usd_cap = st.slider("|Beta USD| max", 0.00, 1.50, factor_usd_cap, 0.05)
        tc_bps = st.slider("Transaction cost bps", 0, 100, tc_bps)
        embargo_days = st.slider("Signal-execution embargo", 0, 15, embargo_days)

    suitability_block_run = bool(suitability["Hard_Block"])
    if suitability_block_run:
        st.error("Suitability blocks the run: adjust horizon, drawdown, objective, or risk aversion.")
        st.caption("Default policy: daily data/options refresh, semiannual rebalance, annual weight reoptimization. Private Side Alpha uses the fixed schedule with residual CBRS/Cerebras weight.")
    pipeline_running = bool(st.session_state.get("pipeline_running", False))
    portfolio_name = st.text_input(
        "Portfolio name",
        value=st.session_state.get("portfolio_name", f"{current_user.name} Portfolio"),
        max_chars=80,
        help="This label is used in My Portfolio after a successful optimization.",
    ).strip()
    st.session_state["portfolio_name"] = portfolio_name or f"{current_user.name} Portfolio"
    run_button_label = "Running pipeline..." if pipeline_running else "Run Allocation Engine"
    run_button = st.button(
        run_button_label,
        type="primary",
        use_container_width=True,
        disabled=suitability_block_run or pipeline_running,
        help="Disabled while the causal pipeline is running." if pipeline_running else "Launches the full causal allocation pipeline.",
    )


manual_tickers = parse_tickers(universe_text)
public_universe = pd.DataFrame()
if universe_source != "Manual":
    try:
        public_universe = cached_public_universe(universe_source, universe_asof, use_persistent_cache, cache_ttl_hours, sec_user_agent)
    except Exception as exc:
        st.sidebar.warning(f"Could not load public universe: {exc}")
public_tickers = tuple(public_universe["Ticker"].dropna().astype(str).str.upper().head(max_public_names).tolist()) if not public_universe.empty and "Ticker" in public_universe else tuple()
if universe_source == "Manual":
    tickers = manual_tickers
elif universe_mode == "Replace manual":
    tickers = public_tickers
else:
    tickers = tuple(dict.fromkeys(list(manual_tickers) + list(public_tickers)))
side_boom_tickers = normalize_side_tickers(parse_tickers(side_boom_text)) if use_side_boom_portfolio else DEFAULT_SIDE_ALPHA_TICKERS

pre_suggested_benchmark = suggest_benchmark(
    benchmark_mandate_type,
    rate_country,
    dominant_sector=None,
    benchmark_group=benchmark_group,
    investor_objective=investor_objective,
)
if benchmark_auto_select and benchmark_group != "Custom":
    benchmark_ticker = pre_suggested_benchmark
pre_benchmark_gov = benchmark_governance_diagnostics(
    benchmark_ticker,
    benchmark_group,
    benchmark_mandate_type,
    rate_country,
    investor_objective,
    weight_objective,
    tickers,
)

with st.sidebar:
    if universe_source != "Manual":
        st.caption(f"Public universe loaded: {len(public_tickers)} of {len(public_universe) if not public_universe.empty else 0} names.")
    gov_row = pre_benchmark_gov.iloc[0]
    st.caption(f"Final benchmark: {benchmark_ticker} | suggested: {gov_row.get('Suggested_Benchmark', pre_suggested_benchmark)}")
    if str(gov_row.get("Warnings", "")).strip():
        st.warning(str(gov_row.get("Warnings")))

if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

config = RunConfig(
    tickers=tickers,
    benchmark_ticker=benchmark_ticker,
    price_period=price_period,
    accounting_lag_days=accounting_lag_days,
    max_workers=max_workers,
    top_n=top_n,
    preselect_n=preselect_n,
    min_chunk=min_chunk,
    max_chunk=max_chunk,
    max_combos=max_combos,
    compute_mode=compute_mode.lower(),
    use_persistent_cache=use_persistent_cache,
    cache_ttl_hours=cache_ttl_hours,
    rate_country=rate_country,
    use_sec_edgar=use_sec_edgar,
    sec_user_agent=sec_user_agent,
    use_sec_nlp=use_sec_nlp,
    sec_nlp_max_tickers=sec_nlp_max_tickers,
    sec_nlp_max_filings=sec_nlp_max_filings,
    universe_source=universe_source,
    universe_asof=str(universe_asof),
    use_options_snapshot=use_options_snapshot,
    option_expiries=option_expiries,
    options_cache_ttl_hours=options_cache_ttl_hours,
    garch_candidate_n=garch_candidate_n,
    validation_bootstrap_samples=validation_bootstrap_samples,
    reality_check_samples=reality_check_samples,
    cpcv_folds=cpcv_folds,
    use_gdelt=use_gdelt,
    gdelt_query=gdelt_query,
    rebalance_freq="2QE",
    reoptimization_freq="YE",
    max_names_per_sector=max_sector,
    max_weight=max_weight,
    sector_weight_cap=sector_weight_cap,
    weight_objective=weight_objective,
    risk_aversion=risk_aversion,
    alpha_weight=alpha_weight,
    entropy_penalty=entropy_penalty,
    crlb_penalty=crlb_penalty,
    garch_penalty=garch_penalty,
    evt_penalty=evt_penalty,
    cvar_penalty=cvar_penalty,
    cvar_alpha=cvar_alpha,
    robust_alpha_uncertainty=robust_alpha_uncertainty,
    robust_cov_uncertainty=robust_cov_uncertainty,
    text_risk_penalty=text_risk_penalty,
    use_black_litterman=use_black_litterman,
    black_litterman_tau=black_litterman_tau,
    use_garch=use_garch,
    target_vol=target_vol,
    nested_validation_fraction=nested_validation_fraction,
    purge_days=purge_days,
    sortino_multistarts=sortino_multistarts,
    bootstrap_samples=bootstrap_samples,
    factor_mkt_cap=factor_mkt_cap,
    factor_rates_cap=factor_rates_cap,
    factor_credit_cap=factor_credit_cap,
    factor_oil_cap=factor_oil_cap,
    factor_usd_cap=factor_usd_cap,
    factor_cov_blend=factor_cov_blend,
    portfolio_notional=float(portfolio_notional),
    max_adv_participation=float(max_adv_participation),
    tc_bps=float(tc_bps),
    impact_coefficient=impact_coefficient,
    min_dollar_volume=float(min_dollar_volume),
    embargo_days=embargo_days,
    max_oos_trials_per_rebalance=max_oos_trials_per_rebalance,
    robust_selection_lambda=robust_selection_lambda,
    robust_selection_min_obs=robust_selection_min_obs,
    model_confidence_window=model_confidence_window,
    model_confidence_min=model_confidence_min,
    use_kaizen_bandit=use_kaizen_bandit,
    kaizen_ucb_alpha=kaizen_ucb_alpha,
    use_forex_factory_calendar=use_forex_factory_calendar,
    forex_factory_cache_ttl_hours=24,
    use_latent_macro_regime=use_latent_macro_regime,
    latent_regime_states=latent_regime_states,
    latent_regime_refit_days=latent_regime_refit_days,
    latent_regime_min_train=latent_regime_min_train,
    use_dynamic_sizing=use_dynamic_sizing,
    min_dynamic_exposure=min_dynamic_exposure,
    max_dynamic_exposure=max_dynamic_exposure,
    regime_entropy_exposure_penalty=regime_entropy_exposure_penalty,
    markov_stress_exposure_penalty=markov_stress_exposure_penalty,
    markov_transition_min_obs=markov_transition_min_obs,
    suitability_mode="automatic" if use_suitability_engine else "manual",
    investor_horizon_years=float(investor_horizon_years),
    investor_initial_capital=float(investor_initial_capital),
    investor_monthly_contribution=float(investor_monthly_contribution),
    investor_liquidity_need=investor_liquidity_need,
    investor_max_drawdown=float(investor_max_drawdown),
    investor_cvar_max_daily=float(suitability["CVaR_Max_Daily"]),
    investor_risk_aversion_score=float(investor_risk_aversion_score),
    investor_objective=investor_objective,
    investor_base_currency=investor_base_currency,
    suitability_profile=suitability["Suitability_Profile"],
    suitability_score=float(suitability["Suitability_Score"]),
    suitability_hard_block=bool(suitability["Hard_Block"]),
    suitability_warnings=tuple(suitability["Warnings"]),
    benchmark_group=benchmark_group,
    benchmark_mandate_type=benchmark_mandate_type,
    benchmark_auto_select=benchmark_auto_select,
    benchmark_suggested_ticker=str(pre_benchmark_gov.iloc[0].get("Suggested_Benchmark", pre_suggested_benchmark)),
    benchmark_governance_warnings=tuple(str(pre_benchmark_gov.iloc[0].get("Warnings", "")).split(" | ")) if str(pre_benchmark_gov.iloc[0].get("Warnings", "")).strip() else (),
    use_side_boom_portfolio=use_side_boom_portfolio,
    side_boom_tickers=side_boom_tickers,
    side_boom_fixed_ticker=side_boom_fixed_ticker,
    side_boom_fixed_weight=float(side_boom_fixed_weight),
    side_boom_fixed_weights=side_boom_extra_fixed_weights,
    side_boom_mode="private_side_alpha_firewall",
    side_boom_min_obs=int(side_boom_min_obs),
    side_boom_cash_return=0.0,
    lookback_grid=(63, 126) if compute_mode == "Rapido" else (63, 126, 252),
    chunk_size_grid=tuple(sorted(set([min_chunk, max_chunk]))) if compute_mode == "Rapido" else tuple(sorted(set([min_chunk, min(8, max_chunk), max_chunk]))),
)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_run(config: RunConfig):
    return run_pipeline(config)


# Hydrate the dashboard before any optional live-data work. This keeps the
# first useful paint bound to the persisted cloud artifact instead of Yahoo,
# FRED, GDELT, or options-network latency.
dashboard_schema_changed = (
    st.session_state.get("dashboard_ui_schema_version") != DASHBOARD_UI_SCHEMA_VERSION
)
if st.session_state.get("results") is None or dashboard_schema_changed:
    startup_results = _load_precomputed_dashboard_results(benchmark_ticker)
    if startup_results:
        st.session_state["results"] = startup_results
        st.session_state["last_run_at"] = (
            startup_results.get("artifact_created_at") or "persisted cloud artifact"
        )
    st.session_state["dashboard_ui_schema_version"] = DASHBOARD_UI_SCHEMA_VERSION

st.session_state.setdefault("load_geopolitical_thermometer", False)
st.session_state.setdefault("load_global_rates", False)
has_persisted_dashboard = isinstance(st.session_state.get("results"), dict)
has_full_analysis = bool(
    has_persisted_dashboard and st.session_state["results"].get("full_analysis_run_id")
)
has_daily_snapshot = bool(
    has_persisted_dashboard and st.session_state["results"].get("daily_snapshot_run_id")
)
if has_full_analysis and has_daily_snapshot:
    _ops_title = "Research workspace ready"
    _ops_detail = (
        "The latest full analytical run is loaded with a newer daily market pulse. "
        "Validation, fundamentals, risk and audit evidence remain intact."
    )
    _ops_scope = "Full research · daily market overlay · no automatic refetch"
elif has_persisted_dashboard:
    _ops_title = "Persisted dashboard ready"
    _ops_detail = (
        "The dashboard is served from the latest auditable cloud artifact. "
        "Live public-data modules remain available on demand."
    )
    _ops_scope = "Fast first paint · auditable timestamp · no automatic refetch"
else:
    _ops_title = ""
    _ops_detail = ""
    _ops_scope = ""
st.markdown(
    (
        '<div class="qpk-ops-strip" role="status" aria-live="polite">'
        f'<div><div class="qpk-ops-title">{_ops_title}</div>'
        f'<div class="qpk-ops-meta">{_ops_detail}</div></div>'
        f'<div class="qpk-ops-meta">{_ops_scope}</div>'
        '</div>'
    )
    if has_persisted_dashboard
    else (
        '<div class="qpk-ops-strip" role="status" aria-live="polite" style="border-left-color:var(--qpk-warn);">'
        '<div><div class="qpk-ops-title">No persisted dashboard found</div>'
        '<div class="qpk-ops-meta">A lightweight market panel will load while the allocation engine remains available.</div></div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)

with st.expander("Data operations", expanded=False):
    st.caption("Use these controls only when you need an intraday public-data refresh. The scheduled snapshot remains the default.")
    geo_cols = st.columns(2)
    with geo_cols[0]:
        if st.button("Load global curves", use_container_width=True):
            st.session_state["load_global_rates"] = True
    with geo_cols[1]:
        if st.button("Refresh geopolitical monitor", use_container_width=True):
            st.session_state["load_geopolitical_thermometer"] = True

live_preflight_requested = (
    not has_persisted_dashboard
    or bool(st.session_state.get("load_global_rates"))
    or bool(st.session_state.get("load_geopolitical_thermometer"))
)
if live_preflight_requested:
    try:
        with st.status("Updating the live market monitor...", expanded=False) as pre_status:
            preflight_market = cached_preflight_market(
                benchmark_ticker,
                benchmark_group,
                rate_country,
                price_period,
                side_boom_tickers,
                side_boom_fixed_ticker,
                float(side_boom_fixed_weight),
                side_boom_extra_fixed_weights,
                int(side_boom_min_obs),
                use_persistent_cache,
                cache_ttl_hours,
                bool(st.session_state.get("load_global_rates")),
                bool(st.session_state.get("load_geopolitical_thermometer")),
                24,
            )
            pre_status.update(label="Live market monitor ready", state="complete")
        render_preflight_market(preflight_market, benchmark_ticker)
    except Exception as exc:
        st.warning(f"The live market monitor is unavailable. The persisted dashboard remains usable: {exc}")


if run_button:
    # Per-user rate limit on the heavy backend call (security.RateLimiter).
    if not enforce_pipeline_quota(current_user.username):
        audit_event("pipeline.blocked_by_rate_limit", username=current_user.username)
        st.stop()

    st.session_state["last_run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["pipeline_running"] = True
    pipeline_ok = False
    _t_start = datetime.now()
    audit_event(
        "pipeline.start",
        username=current_user.username,
        role=current_user.role,
        objective=weight_objective,
        benchmark=benchmark_ticker,
    )
    try:
        with st.status("Running causal pipeline...", expanded=True) as status:
            st.write("Downloading prices and fundamentals.")
            st.write("Loading public universe as-of, SEC EDGAR, and SEC NLP if enabled.")
            st.write("Rebuilding fundamental panel with Availability_Date.")
            st.write("Estimating sovereign curves, macro regime, sectors, and cross-sectional ranking.")
            st.write(f"Optimizing chunks by {weight_objective}, running walk-forward, and evaluating Kaizen.")
            try:
                pipeline_results = cached_run(config)
                st.session_state["results"] = pipeline_results
                st.session_state["dashboard_ui_schema_version"] = DASHBOARD_UI_SCHEMA_VERSION
                if save_run_to_supabase is not None and supabase_available():
                    try:
                        saved_run_id = save_run_to_supabase(
                            pipeline_results,
                            config,
                            status="user_completed",
                            owner_username=current_user.username,
                            portfolio_name=st.session_state.get("portfolio_name"),
                        )
                        st.session_state["saved_user_run_id"] = saved_run_id
                        st.write("Personal portfolio saved to My Portfolio.")
                    except Exception as persist_exc:
                        st.warning(
                            "The analysis completed, but the personal portfolio could not be published "
                            f"to Supabase: {persist_exc}"
                        )
                        audit_event(
                            "portfolio.persist_error",
                            username=current_user.username,
                            error_type=type(persist_exc).__name__,
                            error_msg=str(persist_exc)[:240],
                        )
                status.update(label="Process completed", state="complete")
                pipeline_ok = True
            except Exception as exc:
                status.update(label="Execution failed", state="error")
                st.exception(exc)
                audit_event(
                    "pipeline.error",
                    username=current_user.username,
                    error_type=type(exc).__name__,
                    error_msg=str(exc)[:240],
                )
    finally:
        st.session_state["pipeline_running"] = False
        audit_event(
            "pipeline.end",
            username=current_user.username,
            ok=pipeline_ok,
            duration_seconds=round((datetime.now() - _t_start).total_seconds(), 2),
        )
    try:
        if pipeline_ok:
            st.toast("Allocation engine ready", icon="✅")
        else:
            st.toast("Pipeline failed — see details above", icon="⚠️")
    except Exception as exc:
        audit_event("ui.toast_error", username=current_user.username, error_type=type(exc).__name__)

results = st.session_state.get("results")

if results is None:
    st.info("Configure the mandate and press **Run Allocation Engine**. For a first test use 25-40 tickers.")
    st.markdown(
        '<p class="small-note">Causal note: optimization and backtest share the selected base period, '
        'but each rebalance optimizes only with data before the signal. The evaluated OOS window never enters optimization.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

# Streamlit sessions can survive a code redeploy. Heal any payload that was
# hydrated by an older frontend before JSON-to-DataFrame restoration existed.
raw_dashboard_payload = results.get("dashboard_payload", {}) if isinstance(results, dict) else {}
raw_allocation = raw_dashboard_payload.get("allocation", {}) if isinstance(raw_dashboard_payload, dict) else {}
raw_charts = raw_dashboard_payload.get("charts", {}) if isinstance(raw_dashboard_payload, dict) else {}
payload_requires_restore = any(
    isinstance(value, list)
    for value in (
        raw_allocation.get("recommended_portfolio") if isinstance(raw_allocation, dict) else None,
        raw_charts.get("price_paths") if isinstance(raw_charts, dict) else None,
        raw_charts.get("drawdowns") if isinstance(raw_charts, dict) else None,
    )
)
if payload_requires_restore:
    restored_results = _minimal_results_from_dashboard_payload(
        raw_dashboard_payload,
        benchmark=benchmark_ticker,
    )
    restored_results["artifact_created_at"] = results.get("artifact_created_at")
    restored_results["artifact_run_id"] = results.get("artifact_run_id")
    st.session_state["results"] = restored_results
    results = restored_results

latest_macro = results["latest_macro"]
prices = results["prices"]
cs = results["cross_section"]
portfolio = results["portfolio"]
side_boom_portfolio = results.get("side_boom_portfolio", pd.DataFrame())
side_boom_curve = results.get("side_boom_curve", pd.DataFrame())
side_boom_wf = results.get("side_boom_walk_forward", pd.DataFrame())
side_boom_holdings = results.get("side_boom_holdings", pd.DataFrame())
side_boom_diagnostics = results.get("side_boom_diagnostics", pd.DataFrame())
options = results["portfolio_options"]
perf = results["backtest_perf"]
holdings = results["backtest_holdings"]
opt_grid = results["optimization_grid"]
sector_diag = results["sector_diagnostics"]
macro = results["macro"]
equity_curve = results.get("equity_curve", pd.DataFrame())
return_diag = results.get("return_diagnostics", {})
perf_summary = results.get("performance_summary", pd.DataFrame())
overfit_diag = results.get("overfit_diagnostics", pd.DataFrame())
factor_attr = results.get("factor_attribution", pd.DataFrame())
monitoring_diag = results.get("monitoring_diagnostics", pd.DataFrame())
reject_diag = results.get("rejection_diagnostics", pd.DataFrame())
cache_inventory = results.get("cache_inventory", pd.DataFrame())
timings = results.get("timings", pd.DataFrame())
options_chain = results.get("options_chain", pd.DataFrame())
options_summary = results.get("options_summary", pd.DataFrame())
portfolio_vol_surface = results.get("portfolio_vol_surface", pd.DataFrame())
portfolio_vol_surface_matrix = results.get("portfolio_vol_surface_matrix", pd.DataFrame())
portfolio_vol_surface_diagnostics = results.get("portfolio_vol_surface_diagnostics", pd.DataFrame())
validation_diag = results.get("validation_diagnostics", {})
kaizen_diag = results.get("kaizen_diagnostics", {})
latent_regime_diag = results.get("latent_regime_diagnostics", {})
alternative_data = results.get("alternative_data", {})
market_sentiment_sem_results = results.get("market_sentiment_sem", {})
global_yield_curves = results.get("global_yield_curves", pd.DataFrame())
global_rate_history = results.get("global_rate_history", pd.DataFrame())
interbank_reference_rates = results.get("interbank_reference_rates", pd.DataFrame())
forex_factory_calendar = alternative_data.get("forex_factory_calendar", pd.DataFrame()) if isinstance(alternative_data, dict) else pd.DataFrame()
carry_trade = results.get("carry_trade_suggestions", pd.DataFrame())
carry_trade_validation = results.get("carry_trade_validation", pd.DataFrame())
suitability_diag = results.get("suitability_diagnostics", pd.DataFrame())
suitability_gate = results.get("suitability_gate", {})
promotion_gate = results.get("promotion_gate", {})
backtest_path_bundle = results.get("backtest_path_bundle", {})
dashboard_payload = results.get("dashboard_payload", {})
benchmark_gov = results.get("benchmark_governance", pd.DataFrame())
model_registry = results.get("model_registry", pd.DataFrame())
oos_attr = results.get("oos_factor_attribution", pd.DataFrame())
regime_perf = results.get("regime_performance", pd.DataFrame())
stress_tests = results.get("stress_tests", pd.DataFrame())
hedge_suggestions = results.get("hedge_suggestions", pd.DataFrame())
decision_attr = results.get("decision_attribution", pd.DataFrame())
capital_ledger = results.get("capital_ledger", pd.DataFrame())
gbm_path = return_diag.get("gbm_forecast_path", pd.DataFrame()) if isinstance(return_diag, dict) else pd.DataFrame()
gbm_summary = return_diag.get("gbm_forecast_summary", pd.DataFrame()) if isinstance(return_diag, dict) else pd.DataFrame()
pelt_segments = return_diag.get("pelt_regime_segments", pd.DataFrame()) if isinstance(return_diag, dict) else pd.DataFrame()
pelt_changes = return_diag.get("pelt_change_points", pd.DataFrame()) if isinstance(return_diag, dict) else pd.DataFrame()
pelt_timeline = return_diag.get("pelt_timeline", pd.DataFrame()) if isinstance(return_diag, dict) else pd.DataFrame()
side_boom_pelt_segments = results.get("side_boom_pelt_regime_segments", pd.DataFrame())
side_boom_pelt_changes = results.get("side_boom_pelt_change_points", pd.DataFrame())
side_boom_pelt_timeline = results.get("side_boom_pelt_timeline", pd.DataFrame())
daily_backtest_prices = backtest_path_bundle.get("price_paths", pd.DataFrame()) if isinstance(backtest_path_bundle, dict) else pd.DataFrame()
if daily_backtest_prices.empty:
    daily_backtest_prices = daily_backtest_price_frame(perf, holdings, prices, benchmark_ticker)
daily_side_prices = daily_side_price_frame(side_boom_wf, side_boom_holdings, prices, benchmark_ticker)
objective_metric_col = (
    options["Chunk_Objective_Metric"].dropna().iloc[0]
    if not options.empty and "Chunk_Objective_Metric" in options.columns and options["Chunk_Objective_Metric"].notna().any()
    else "Sortino"
)

display_ticker_count = int(results["prices"].shape[1]) if isinstance(results.get("prices"), pd.DataFrame) else 0
if display_ticker_count == 0 and isinstance(results.get("portfolio"), pd.DataFrame):
    display_ticker_count = int(results["portfolio"].shape[0])
st.caption(
    f"Last run: {st.session_state.get('last_run_at', 'n/a')} | "
    f"Portfolio names: {display_ticker_count} | Benchmark: {benchmark_ticker}"
)


# ============================================================
# UI = Renderer. Core = Source of truth.
# All sections below consume `payload = results["dashboard_payload"]`.
# Pre-computed `results[...]` objects are read only when the payload
# does not yet expose them. No financial math happens in this file.
# ============================================================

def _build_gate_state(payload_obj: dict) -> dict:
    """Pure function: derive an explicit gate-state struct from the dashboard payload.

    Returned keys are stable contract for the frontend renderers.
    """
    safe_payload = payload_obj if isinstance(payload_obj, dict) else {}
    status = safe_payload.get("status", {}) if isinstance(safe_payload, dict) else {}
    allocation = safe_payload.get("allocation", {}) if isinstance(safe_payload, dict) else {}
    charts = safe_payload.get("charts", {}) if isinstance(safe_payload, dict) else {}
    tables = safe_payload.get("tables", {}) if isinstance(safe_payload, dict) else {}
    explanations = safe_payload.get("explanations", {}) if isinstance(safe_payload, dict) else {}
    market_intelligence = safe_payload.get("market_intelligence", {}) if isinstance(safe_payload, dict) else {}

    s_suit = status.get("suitability", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()
    s_breaches = status.get("suitability_breaches", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()
    s_prom = status.get("promotion", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()
    s_prom_tests = status.get("promotion_tests", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()
    s_fresh = status.get("data_freshness", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()
    s_snapshot = status.get("snapshot_meta", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()
    s_market_context = status.get("market_context", pd.DataFrame()) if isinstance(status, dict) else pd.DataFrame()

    suit_status = "unknown"
    if isinstance(s_suit, pd.DataFrame) and not s_suit.empty:
        if "Gate_Status" in s_suit.columns:
            suit_status = str(s_suit.iloc[0]["Gate_Status"]).lower()
        elif {"Metric", "Value"}.issubset(s_suit.columns):
            lookup = s_suit.set_index("Metric")["Value"]
            if "Snapshot_Status" in lookup.index:
                suit_status = "snapshot"

    prom_status = "unknown"
    if isinstance(s_prom, pd.DataFrame) and not s_prom.empty:
        if "Promotion_Status" in s_prom.columns:
            prom_status = str(s_prom.iloc[0]["Promotion_Status"]).lower()
        elif {"Metric", "Value"}.issubset(s_prom.columns):
            lookup = s_prom.set_index("Metric")["Value"]
            if "Promotion_Status" in lookup.index:
                prom_status = str(lookup.loc["Promotion_Status"]).lower()

    if suit_status == "blocked":
        alloc_state = "blocked"
    elif prom_status != "promoted":
        alloc_state = "research_only"
    elif suit_status == "approved":
        alloc_state = "approved"
    else:
        alloc_state = "research_only"

    is_snapshot = suit_status == "snapshot" or "snapshot" in prom_status

    return {
        "payload": safe_payload,
        "allocation_state": alloc_state,
        "suitability_status": suit_status,
        "promotion_status": prom_status,
        "suit_summary": s_suit,
        "suit_breaches": s_breaches,
        "prom_summary": s_prom,
        "prom_tests": s_prom_tests,
        "freshness": s_fresh,
        "snapshot_meta": s_snapshot,
        "market_context": s_market_context,
        "performance": tables.get("risk", pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame(),
        "is_snapshot": is_snapshot,
        "allocation_block": allocation,
        "charts_block": charts,
        "tables_block": tables,
        "market_intelligence_block": market_intelligence,
        "explanations_block": explanations,
    }


payload = results.get("dashboard_payload", {}) or {}
if not isinstance(payload, dict):
    payload = {}

gate_state = _build_gate_state(payload)
daily_snapshot_payload = results.get("daily_snapshot_payload", {})
daily_snapshot_gate_state = (
    _build_gate_state(daily_snapshot_payload)
    if isinstance(daily_snapshot_payload, dict) and daily_snapshot_payload
    else None
)
allocation_state = gate_state["allocation_state"]
suitability_status = gate_state["suitability_status"]
promotion_status = gate_state["promotion_status"]
suit_summary_df = gate_state["suit_summary"]
suit_breaches_df = gate_state["suit_breaches"]
prom_summary_df = gate_state["prom_summary"]
prom_tests_df = gate_state["prom_tests"]
freshness_df = gate_state["freshness"]
allocation_block = gate_state["allocation_block"]
charts_block = gate_state["charts_block"]
tables_block = gate_state["tables_block"]
explanations_block = gate_state["explanations_block"]
status_block = payload.get("status", {}) if isinstance(payload, dict) else {}


# ------------------------------------------------------------
# Visual primitives (style-only helpers; zero financial math)
# ------------------------------------------------------------

def _fmt_pct(value, default: str = "—", digits: int = 1) -> str:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
            return default
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return default


def _fmt_num(value, default: str = "—", digits: int = 2) -> str:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
            return default
        return f"{float(value):,.{digits}f}"
    except Exception:
        return default


def _normalize_weight_fraction(df: pd.DataFrame, col: str = "Weight") -> pd.DataFrame:
    """Ensure the Weight column is expressed as a fraction in [0, 1].

    Backend may emit weights either as fractions (0-1) or as percentage values (0-100).
    This helper detects the convention by the column's max value and normalizes to
    a fraction so downstream `format=\"%.2f%%\"` renders correctly. Returns a copy.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return df
    out = df.copy()
    numeric = pd.to_numeric(out[col], errors="coerce")
    finite = numeric[numeric.notna() & np.isfinite(numeric)]
    if finite.empty:
        return out
    # Heuristic: if absolute max > 1.5, assume percentage-scale and divide by 100.
    if finite.abs().max() > 1.5:
        out[col] = numeric / 100.0
    else:
        out[col] = numeric
    return out


def _with_weight_percent(df: pd.DataFrame, col: str = "Weight") -> pd.DataFrame:
    """Add an explicit percentage column for Streamlit table rendering."""
    out = _normalize_weight_fraction(df, col)
    if out is None or not isinstance(out, pd.DataFrame) or out.empty or col not in out.columns:
        return out
    out = out.copy()
    out["Weight_Pct"] = pd.to_numeric(out[col], errors="coerce") * 100.0
    return out


def _status_pill(label: str, kind: str = "neutral", *, live: bool = False) -> str:
    import html as _html
    palette = {
        "approved": ("rgba(74,222,128,0.14)", "rgba(74,222,128,0.45)", "#4ade80"),
        "promoted": ("rgba(74,222,128,0.14)", "rgba(74,222,128,0.45)", "#4ade80"),
        "pass": ("rgba(74,222,128,0.14)", "rgba(74,222,128,0.45)", "#4ade80"),
        "valid": ("rgba(74,222,128,0.14)", "rgba(74,222,128,0.45)", "#4ade80"),
        "watchlist": ("rgba(251,191,36,0.14)", "rgba(251,191,36,0.5)", "#fbbf24"),
        "research_only": ("rgba(251,191,36,0.14)", "rgba(251,191,36,0.5)", "#fbbf24"),
        "missing": ("rgba(251,191,36,0.14)", "rgba(251,191,36,0.5)", "#fbbf24"),
        "fallback": ("rgba(251,191,36,0.14)", "rgba(251,191,36,0.5)", "#fbbf24"),
        "stale": ("rgba(251,191,36,0.14)", "rgba(251,191,36,0.5)", "#fbbf24"),
        "blocked": ("rgba(248,113,113,0.14)", "rgba(248,113,113,0.5)", "#f87171"),
        "rejected": ("rgba(248,113,113,0.14)", "rgba(248,113,113,0.5)", "#f87171"),
        "fail": ("rgba(248,113,113,0.14)", "rgba(248,113,113,0.5)", "#f87171"),
        "error": ("rgba(248,113,113,0.14)", "rgba(248,113,113,0.5)", "#f87171"),
        "fixed": ("rgba(129,140,248,0.14)", "rgba(129,140,248,0.5)", "#a5b4fc"),
        "info": ("rgba(125,211,252,0.10)", "rgba(125,211,252,0.40)", "#7dd3fc"),
        "neutral": ("rgba(148,163,184,0.10)", "rgba(148,163,184,0.35)", "#a8b3c7"),
    }
    bg, border, color = palette.get(kind.lower() if isinstance(kind, str) else "neutral", palette["neutral"])
    safe_label = _html.escape(str(label))
    role_attrs = ' role="status" aria-live="polite"' if live else ' role="img" aria-label="status: ' + safe_label + '"'
    return (
        f'<span{role_attrs} style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:4px 10px;border-radius:999px;background:{bg};'
        f'border:1px solid {border};color:{color};font-size:0.74rem;'
        f'font-weight:600;letter-spacing:0.04em;text-transform:uppercase;'
        f'font-family:var(--font-sans);white-space:nowrap;">{safe_label}</span>'
    )


def _empty_state(message: str, suggestion: str | None = None) -> None:
    import html as _html
    safe_msg = _html.escape(str(message))
    suggestion_html = (
        f'<div style="color:var(--qpk-faint);font-size:0.82rem;margin-top:6px;">{_html.escape(str(suggestion))}</div>'
        if suggestion else ""
    )
    st.markdown(
        f'<div role="status" aria-live="polite" style="border:1px dashed var(--qpk-line);border-radius:6px;'
        f'padding:16px 18px;background:rgba(11,16,26,0.6);">'
        f'<div style="color:var(--qpk-muted);font-size:0.92rem;">{safe_msg}</div>'
        f'{suggestion_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _section_header(title: str, subtitle: str | None = None) -> None:
    import html as _html
    safe_title = _html.escape(str(title))
    sub_html = (
        f'<div class="small-note" style="margin-top:4px;">{_html.escape(str(subtitle))}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin: 4px 0 12px 0;">'
        f'<div style="color:var(--qpk-text);font-size:1.05rem;font-weight:650;letter-spacing:0;">{safe_title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _banner(kind: str, title: str, body: str) -> None:
    """Render a colored banner (info/warning/error/success) with proper a11y."""
    import html as _html
    accents = {
        "error": ("#f87171", "rgba(248,113,113,0.4)", "rgba(248,113,113,0.06)", "alert"),
        "warning": ("#fbbf24", "rgba(251,191,36,0.4)", "rgba(251,191,36,0.06)", "status"),
        "success": ("#4ade80", "rgba(74,222,128,0.4)", "rgba(74,222,128,0.06)", "status"),
        "info": ("#a5b4fc", "rgba(129,140,248,0.32)", "rgba(129,140,248,0.05)", "status"),
        "neutral": ("#a8b3c7", "rgba(148,163,184,0.32)", "rgba(148,163,184,0.06)", "status"),
    }
    fg, border, bg, role = accents.get(kind, accents["neutral"])
    safe_title = _html.escape(str(title))
    safe_body = _html.escape(str(body))
    st.markdown(
        f'<div role="{role}" aria-live="polite" style="border:1px solid {border};border-left:3px solid {fg};'
        f'background:{bg};padding:12px 14px;border-radius:4px;color:var(--qpk-text);">'
        f'<strong style="color:{fg};">{safe_title}</strong> {safe_body}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _plotly_dark_layout(fig, height: int = 360, title: str | None = None):
    if fig is None:
        return None
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(7,8,12,0)",
        plot_bgcolor="rgba(11,16,26,0.55)",
        margin=dict(l=56, r=24, t=52 if title else 24, b=92),
        height=height,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#eef3fb"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0,
            title_text="",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor="#0b101a", bordercolor="rgba(125,211,252,0.4)", font=dict(family="JetBrains Mono", color="#eef3fb")),
        hovermode="x unified",
        xaxis=dict(
            title_text="",
            automargin=True,
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.18)",
        ),
        yaxis=dict(
            title_text="",
            automargin=True,
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.18)",
        ),
    )
    if title:
        layout["margin"]["t"] = 112
        layout["margin"]["b"] = 48
        layout["title"] = dict(
            text=title,
            x=0.0,
            y=0.99,
            xanchor="left",
            yanchor="top",
            font=dict(size=13, color="#a8b3c7"),
        )
    else:
        # Plotly/Streamlit can render a literal "undefined" when an explicit
        # null title object is serialized. Omit the property entirely.
        fig.update_layout(title_text="")
    fig.update_layout(**layout)
    return fig


def _line_chart(df: pd.DataFrame, *, x: str, ys: list[str], height: int = 360, title: str | None = None,
                y_format: str | None = None, percent: bool = False, fill: bool = False):
    if df is None or df.empty or px is None:
        return None
    sub = df[[x] + [c for c in ys if c in df.columns]].dropna(how="all", subset=[c for c in ys if c in df.columns])
    if sub.empty:
        return None
    melted = sub.melt(id_vars=[x], var_name="Series", value_name="Value")
    fig = px.line(melted, x=x, y="Value", color="Series")
    if fill:
        for idx, tr in enumerate(fig.data):
            tr.update(
                fill="tozeroy" if idx == 0 else None,
                fillcolor="rgba(21,94,239,0.16)" if idx == 0 else None,
                line=dict(width=2.0 if idx == 0 else 1.7, dash="solid" if idx == 0 else "dot"),
            )
    if percent:
        fig.update_yaxes(tickformat=".1%")
    elif y_format:
        fig.update_yaxes(tickformat=y_format)
    fig.update_layout(legend_title_text="")
    return _plotly_dark_layout(fig, height=height, title=title)


def _plotly_sovereign_curves(curves: pd.DataFrame):
    """Comparative sovereign curve snapshot (Plotly, replaces matplotlib `plot_global_yield_curves`)."""
    if px is None or curves is None or not isinstance(curves, pd.DataFrame) or curves.empty:
        return None
    tenor_cols = [c for c in ("Policy_Rate", "Yield_2Y", "Yield_10Y") if c in curves.columns]
    if not tenor_cols or "Country" not in curves.columns:
        return None
    sub = curves[["Country"] + tenor_cols].copy()
    sub = sub.dropna(how="all", subset=tenor_cols)
    if sub.empty:
        return None
    tenor_map = {"Policy_Rate": "Policy", "Yield_2Y": "2Y", "Yield_10Y": "10Y"}
    melted = sub.melt(id_vars="Country", var_name="Tenor", value_name="Yield")
    melted["Tenor"] = melted["Tenor"].map(tenor_map).fillna(melted["Tenor"])
    tenor_order = ["Policy", "2Y", "10Y"]
    fig = px.line(
        melted, x="Tenor", y="Yield", color="Country", markers=True,
        category_orders={"Tenor": tenor_order},
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=7))
    fig.update_yaxes(tickformat=".2f", ticksuffix="%")
    return _plotly_dark_layout(fig, height=360)


def _plotly_interbank_rates(ref_rates: pd.DataFrame):
    """Overnight reference-rate panel (Plotly, replaces matplotlib `plot_interbank_reference_rates`)."""
    if px is None or ref_rates is None or not isinstance(ref_rates, pd.DataFrame) or ref_rates.empty:
        return None
    if not {"Observation_Date", "Benchmark", "Rate"}.issubset(ref_rates.columns):
        return None
    sub = ref_rates.copy()
    sub["Observation_Date"] = pd.to_datetime(sub["Observation_Date"], errors="coerce")
    sub = sub.dropna(subset=["Observation_Date", "Rate"])
    if sub.empty:
        return None
    fig = px.line(sub, x="Observation_Date", y="Rate", color="Benchmark")
    fig.update_traces(line=dict(width=1.6))
    fig.update_yaxes(tickformat=".2f", ticksuffix="%")
    return _plotly_dark_layout(fig, height=320)


def _plotly_sentiment_sem(timeline: pd.DataFrame):
    if px is None or timeline is None or not isinstance(timeline, pd.DataFrame) or timeline.empty:
        return None
    required = {"Date", "Latent_Market_Sentiment_SEM"}
    if not required.issubset(timeline.columns):
        return None
    data = timeline[list(required)].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Latent_Market_Sentiment_SEM"] = pd.to_numeric(
        data["Latent_Market_Sentiment_SEM"], errors="coerce"
    )
    data = data.dropna().sort_values("Date").tail(756)
    if data.empty:
        return None
    fig = px.line(data, x="Date", y="Latent_Market_Sentiment_SEM")
    fig.update_traces(line=dict(color="#7dd3fc", width=2.1), name="Latent sentiment")
    fig.add_hrect(y0=-1.0, y1=1.0, fillcolor="rgba(148,163,184,0.06)", line_width=0)
    fig.add_hline(y=0.0, line_color="rgba(238,243,251,0.42)", line_width=1)
    fig.add_hline(y=1.0, line_color="rgba(74,222,128,0.55)", line_dash="dot")
    fig.add_hline(y=-1.0, line_color="rgba(248,113,113,0.55)", line_dash="dot")
    return _plotly_dark_layout(fig, height=350, title="Latent market sentiment")


def _plotly_global_rate_history(rate_history: pd.DataFrame, countries: list[str] | None = None):
    if px is None or rate_history is None or not isinstance(rate_history, pd.DataFrame) or rate_history.empty:
        return None
    required = {"Country", "Observation_Date", "Rate"}
    if not required.issubset(rate_history.columns):
        return None
    data = rate_history.copy()
    if "Tenor_Code" in data.columns:
        ten_year = data[data["Tenor_Code"].astype(str).eq("SOV_10Y")]
        if not ten_year.empty:
            data = ten_year
    data["Observation_Date"] = pd.to_datetime(data["Observation_Date"], errors="coerce")
    data["Rate"] = pd.to_numeric(data["Rate"], errors="coerce")
    data = data.dropna(subset=["Observation_Date", "Rate", "Country"])
    if countries:
        data = data[data["Country"].astype(str).isin(countries)]
    if data.empty:
        return None
    fig = px.line(data, x="Observation_Date", y="Rate", color="Country")
    fig.update_traces(line=dict(width=1.8))
    fig.update_yaxes(tickformat=".2f", ticksuffix="%")
    return _plotly_dark_layout(fig, height=390, title="Discrete sovereign 10Y evolution")


def _plotly_macro_history(macro: pd.DataFrame):
    if px is None or macro is None or not isinstance(macro, pd.DataFrame) or macro.empty:
        return None
    data = macro.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.set_index("Date")
    if not isinstance(data.index, pd.DatetimeIndex):
        return None
    candidates = [
        "VIX", "HY_OAS", "CREDIT_SPREAD", "NFCI", "EPU",
        "FED_BALANCE_SHEET", "FED_REVERSE_REPO", "WTI",
    ]
    selected = [c for c in candidates if c in data.columns]
    if not selected:
        return None
    normalized = pd.DataFrame(index=data.index)
    for col in selected:
        series = pd.to_numeric(data[col], errors="coerce")
        mean = series.rolling(252, min_periods=60).mean()
        std = series.rolling(252, min_periods=60).std(ddof=1).replace(0.0, np.nan)
        normalized[col] = ((series - mean) / std).clip(-5.0, 5.0)
    normalized = normalized.tail(756).reset_index(names="Date")
    long = normalized.melt(id_vars="Date", var_name="Indicator", value_name="Rolling z-score").dropna()
    if long.empty:
        return None
    fig = px.line(long, x="Date", y="Rolling z-score", color="Indicator")
    fig.update_traces(line=dict(width=1.5))
    fig.add_hline(y=0.0, line_color="rgba(238,243,251,0.35)", line_width=1)
    return _plotly_dark_layout(fig, height=390, title="Macro-financial stress indicators")


def _plotly_vol_surface(surface_matrix: pd.DataFrame):
    """Implied vol surface heatmap (Plotly, replaces matplotlib `plot_portfolio_vol_surface`)."""
    if px is None or surface_matrix is None or not isinstance(surface_matrix, pd.DataFrame) or surface_matrix.empty:
        return None
    # Heuristic: rows = DTE/Tenor, columns = moneyness/strike, values = IV.
    grid = surface_matrix.apply(pd.to_numeric, errors="coerce")
    if grid.dropna(how="all").empty:
        return None
    fig = px.imshow(
        grid.values,
        x=[str(c) for c in grid.columns],
        y=[str(i) for i in grid.index],
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(x="Strike / Moneyness", y="DTE / Tenor", color="Implied vol"),
    )
    fig.update_traces(hovertemplate="x=%{x}<br>y=%{y}<br>IV=%{z:.3f}<extra></extra>")
    return _plotly_dark_layout(fig, height=380)


# ------------------------------------------------------------
# Render: Executive Overview
# ------------------------------------------------------------

def _render_daily_market_pulse(
    gate: dict | None,
    *,
    benchmark_ticker: str,
    created_at: str | None,
) -> None:
    if not isinstance(gate, dict):
        return
    research_artifacts = load_xcdr_research_artifacts()
    research_prices, research_drawdowns, research_meta = _research_chart_frames(research_artifacts)
    research_daily = research_artifacts.get("daily", pd.DataFrame()) if isinstance(research_artifacts, dict) else pd.DataFrame()
    if research_meta and isinstance(research_daily, pd.DataFrame) and not research_daily.empty:
        objective = str(research_meta["objective"])
        selected = research_daily[research_daily["objective"].astype(str).eq(objective)].copy()
        selected["portfolio_return"] = pd.to_numeric(selected["portfolio_return"], errors="coerce")
        selected["xi_return"] = pd.to_numeric(selected["xi_return"], errors="coerce")
        selected = selected.dropna(subset=["portfolio_return", "xi_return"])
        if not selected.empty:
            portfolio_returns = selected["portfolio_return"]
            xi_returns = selected["xi_return"]
            active_return = _ann_return_from_daily(portfolio_returns - xi_returns)
            up_mask = xi_returns > 0
            down_mask = xi_returns < 0
            upside_capture = (
                float(portfolio_returns[up_mask].mean() / xi_returns[up_mask].mean())
                if up_mask.any() and abs(float(xi_returns[up_mask].mean())) > 1e-12
                else np.nan
            )
            downside_capture = (
                float(portfolio_returns[down_mask].mean() / xi_returns[down_mask].mean())
                if down_mask.any() and abs(float(xi_returns[down_mask].mean())) > 1e-12
                else np.nan
            )
            _section_header(
                "Governed research pulse",
                f"{objective} | optimal benchmark xi = {research_meta['xi']} | causal daily OOS evidence.",
            )
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("XCDR return", _fmt_pct(_ann_return_from_daily(portfolio_returns)))
            k2.metric("Active return", _fmt_pct(active_return))
            k3.metric("Annualized volatility", _fmt_pct(_ann_vol_from_daily(portfolio_returns)))
            k4.metric("Upside capture", _fmt_num(upside_capture, digits=2))
            k5.metric("Downside capture", _fmt_num(downside_capture, digits=2))
            _banner(
                "warning",
                "Research-only evidence.",
                "Capital remains governed by WRC, Hansen SPA, PBO, CVaR, drawdown and downside-preservation gates.",
            )
            if not research_prices.empty:
                price_fig = _line_chart(
                    research_prices,
                    x="Date",
                    ys=[c for c in research_prices.columns if c != "Date"],
                    height=430,
                    title="XCDR/XODR candidate and optimal benchmark xi",
                    y_format=".2f",
                )
                if price_fig is not None:
                    st.plotly_chart(price_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
            if not research_drawdowns.empty:
                drawdown_fig = _line_chart(
                    research_drawdowns,
                    x="Date",
                    ys=[c for c in research_drawdowns.columns if c != "Date"],
                    height=320,
                    title="OOS drawdown from running maximum",
                    percent=True,
                    fill=True,
                )
                if drawdown_fig is not None:
                    st.plotly_chart(drawdown_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
            return

    performance = gate.get("performance", pd.DataFrame())
    market_context = gate.get("market_context", pd.DataFrame())
    charts = gate.get("charts_block", {})
    if not isinstance(performance, pd.DataFrame) or performance.empty:
        return
    if not {"Metric", "Value"}.issubset(performance.columns):
        return

    metrics = dict(zip(performance["Metric"].astype(str), performance["Value"]))
    context_row = (
        market_context.iloc[0]
        if isinstance(market_context, pd.DataFrame) and not market_context.empty
        else pd.Series(dtype=object)
    )
    p_return = metrics.get("Annualized_Return", np.nan)
    b_return = metrics.get("Benchmark_Annualized_Return", np.nan)
    active_return = (
        float(p_return) - float(b_return)
        if pd.notna(p_return) and pd.notna(b_return)
        else np.nan
    )

    _section_header(
        "Daily market pulse",
        f"Price-derived OOS snapshot updated {created_at or 'n/a'}. The full research run remains the analytical source of truth.",
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Snapshot return", _fmt_pct(p_return))
    k2.metric("Active return", _fmt_pct(active_return), delta=f"{benchmark_ticker} {_fmt_pct(b_return)}")
    k3.metric("Annualized volatility", _fmt_pct(metrics.get("Annualized_Vol")))
    k4.metric("Maximum drawdown", _fmt_pct(metrics.get("Max_Drawdown")))
    k5.metric(
        "Market state",
        str(context_row.get("Trend_Regime", "Unavailable")),
        delta=str(context_row.get("Volatility_Regime", "Unavailable")),
        delta_color="off",
    )

    price_paths = charts.get("price_paths", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
    drawdowns = charts.get("drawdowns", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
    if isinstance(price_paths, pd.DataFrame) and not price_paths.empty:
        date_col = "Date" if "Date" in price_paths.columns else "Period_End"
        price_fig = _line_chart(
            price_paths,
            x=date_col,
            ys=[c for c in price_paths.columns if c != date_col],
            height=430,
            title="Causal OOS NAV path",
            y_format=".2f",
        )
        if price_fig is not None:
            st.plotly_chart(price_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        if isinstance(drawdowns, pd.DataFrame) and not drawdowns.empty:
            date_col = "Date" if "Date" in drawdowns.columns else "Period_End"
            dd_fig = _line_chart(
                drawdowns,
                x=date_col,
                ys=[c for c in drawdowns.columns if c != date_col],
                height=320,
                title="Drawdown from running maximum",
                percent=True,
                fill=True,
            )
            if dd_fig is not None:
                st.plotly_chart(dd_fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def render_executive_overview(
    gate: dict,
    *,
    benchmark_ticker: str,
    last_run_at: str,
    tickers_count: int,
    latest_macro_row: pd.Series,
    daily_snapshot_gate: dict | None = None,
    daily_snapshot_created_at: str | None = None,
) -> None:
    import html as _html
    alloc_state = gate["allocation_state"]
    suit_status = gate["suitability_status"]
    prom_status = gate["promotion_status"]
    suit_summary = gate["suit_summary"]
    suit_breaches = gate["suit_breaches"]
    prom_summary = gate["prom_summary"]
    explanations = gate["explanations_block"]
    is_snapshot = bool(gate.get("is_snapshot", False))
    performance = gate.get("performance", pd.DataFrame())
    market_context = gate.get("market_context", pd.DataFrame())
    freshness = gate.get("freshness", pd.DataFrame())

    def metric_lookup(frame: pd.DataFrame) -> dict:
        if not isinstance(frame, pd.DataFrame) or frame.empty or not {"Metric", "Value"}.issubset(frame.columns):
            return {}
        return dict(zip(frame["Metric"].astype(str), frame["Value"]))

    if is_snapshot:
        metrics = metric_lookup(performance)
        context_row = (
            market_context.iloc[0]
            if isinstance(market_context, pd.DataFrame) and not market_context.empty
            else pd.Series(dtype=object)
        )
        freshness_row = (
            freshness.iloc[0]
            if isinstance(freshness, pd.DataFrame) and not freshness.empty
            else pd.Series(dtype=object)
        )
        portfolio_return = metrics.get("Annualized_Return", np.nan)
        benchmark_return = metrics.get("Benchmark_Annualized_Return", np.nan)
        active_return = (
            float(portfolio_return) - float(benchmark_return)
            if pd.notna(portfolio_return) and pd.notna(benchmark_return)
            else np.nan
        )

        st.markdown(
            '<div role="region" aria-label="Daily snapshot status" '
            'style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">'
            '<span style="color:var(--qpk-faint);font-size:0.72rem;text-transform:uppercase;'
            'letter-spacing:0.12em;">Evidence scope</span>'
            f'{_status_pill("Daily market snapshot", "info", live=True)}'
            f'{_status_pill("Benchmark " + benchmark_ticker, "neutral")}'
            f'{_status_pill(str(tickers_count) + " selected names", "neutral")}'
            '</div>',
            unsafe_allow_html=True,
        )
        _banner(
            "info",
            "Precomputed daily snapshot.",
            "This view contains causal price analytics and an observed price-only selection. "
            "Suitability, fundamentals, options, sovereign rates, and promotion tests are evaluated only in a full user allocation run.",
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Portfolio return", _fmt_pct(portfolio_return), help="Annualized return of the causal OOS snapshot path.")
        k2.metric(
            "Active return",
            _fmt_pct(active_return),
            delta=f"{benchmark_ticker} {_fmt_pct(benchmark_return)}",
            delta_color="normal",
            help="Annualized portfolio return minus annualized benchmark return.",
        )
        k3.metric("Annualized volatility", _fmt_pct(metrics.get("Annualized_Vol")), help="Realized annualized volatility.")
        k4.metric("Sortino ratio", _fmt_num(metrics.get("Sortino")), help="Annualized return divided by annualized downside deviation.")
        k5.metric("Max drawdown", _fmt_pct(metrics.get("Max_Drawdown")), help="Maximum peak-to-trough loss on the daily price path.")

        _section_header("Risk and market state", "Price-derived, causal diagnostics as of the persisted snapshot.")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Daily CVaR 95%", _fmt_pct(metrics.get("CVaR_95")), help="Mean return in the worst five percent of daily observations.")
        r2.metric("Trend regime", str(context_row.get("Trend_Regime", "Unavailable")), help="Benchmark trend proxy using trailing return and moving average.")
        r3.metric("Volatility regime", str(context_row.get("Volatility_Regime", "Unavailable")), help="21-day realized volatility relative to its 126-day baseline.")
        r4.metric("Market breadth", _fmt_pct(context_row.get("Breadth_Above_126D_MA")), help="Share of investable assets above their trailing 126-day mean.")

        charts = gate.get("charts_block", {})
        allocation = gate.get("allocation_block", {})
        price_paths = charts.get("price_paths", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
        drawdowns = charts.get("drawdowns", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
        portfolio_df = allocation.get("recommended_portfolio", pd.DataFrame()) if isinstance(allocation, dict) else pd.DataFrame()

        _section_header(
            "Portfolio vs benchmark",
            "Daily out-of-sample price path and drawdown. Both panels use the same aligned observation dates.",
        )
        chart_left, chart_right = st.columns([1.35, 0.65])
        with chart_left:
            if isinstance(price_paths, pd.DataFrame) and not price_paths.empty:
                date_col = "Date" if "Date" in price_paths.columns else "Period_End"
                series_cols = [c for c in price_paths.columns if c != date_col]
                price_fig = _line_chart(
                    price_paths,
                    x=date_col,
                    ys=series_cols,
                    height=390,
                    title="Observed benchmark price vs synthetic portfolio price",
                    y_format=".2f",
                )
                if price_fig is not None:
                    st.plotly_chart(price_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
            else:
                _empty_state("Price path unavailable.")
        with chart_right:
            if isinstance(drawdowns, pd.DataFrame) and not drawdowns.empty:
                date_col = "Date" if "Date" in drawdowns.columns else "Period_End"
                series_cols = [c for c in drawdowns.columns if c != date_col]
                drawdown_fig = _line_chart(
                    drawdowns,
                    x=date_col,
                    ys=series_cols,
                    height=390,
                    title="Drawdown from running maximum",
                    percent=True,
                    fill=True,
                )
                if drawdown_fig is not None:
                    st.plotly_chart(drawdown_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
            else:
                _empty_state("Drawdown path unavailable.")

        _section_header(
            "Risk-return decomposition",
            "Portfolio and benchmark metrics are computed from the same causal daily OOS interval.",
        )
        comparison_rows = []
        comparison_specs = [
            ("Annualized return", "Annualized_Return", "Benchmark_Annualized_Return", "Higher is better."),
            ("Annualized volatility", "Annualized_Vol", "Benchmark_Annualized_Vol", "Total realized dispersion."),
            ("Downside deviation", "Downside_Deviation", "Benchmark_Downside_Deviation", "Lower partial deviation."),
            ("Sortino ratio", "Sortino", "Benchmark_Sortino", "Return per unit of downside deviation."),
            ("Maximum drawdown", "Max_Drawdown", "Benchmark_Max_Drawdown", "Peak-to-trough loss."),
            ("Daily CVaR 95%", "CVaR_95", "Benchmark_CVaR_95", "Mean return in the worst five percent of days."),
        ]
        for label, portfolio_key, benchmark_key, interpretation in comparison_specs:
            p_value = metrics.get(portfolio_key, np.nan)
            b_value = metrics.get(benchmark_key, np.nan)
            difference = (
                float(p_value) - float(b_value)
                if pd.notna(p_value) and pd.notna(b_value)
                else np.nan
            )
            comparison_rows.append(
                {
                    "Metric": label,
                    "Portfolio": p_value,
                    benchmark_ticker: b_value,
                    "Difference": difference,
                    "Interpretation": interpretation,
                }
            )
        comparison = pd.DataFrame(comparison_rows)

        detail_left, detail_right = st.columns([1.25, 0.75])
        with detail_left:
            st.dataframe(
                comparison,
                width="stretch",
                hide_index=True,
                column_config={
                    "Portfolio": st.column_config.NumberColumn("Portfolio", format="%.4f"),
                    benchmark_ticker: st.column_config.NumberColumn(benchmark_ticker, format="%.4f"),
                    "Difference": st.column_config.NumberColumn("Active difference", format="%+.4f"),
                },
            )
        with detail_right:
            if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
                weights = _with_weight_percent(portfolio_df, "Weight")
                weight_cols = [c for c in ["Ticker", "Weight_Pct", "Optimization_Sortino"] if c in weights.columns]
                st.dataframe(
                    weights[weight_cols].sort_values("Weight_Pct", ascending=False),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Weight_Pct": st.column_config.ProgressColumn(
                            "Weight", format="%.2f%%", min_value=0.0, max_value=100.0
                        ),
                        "Optimization_Sortino": st.column_config.NumberColumn("Selection Sortino", format="%.3f"),
                    },
                )
            else:
                _empty_state("Portfolio weights unavailable.")

        with st.expander("Formal definitions and evidence scope", expanded=False):
            st.latex(r"R_{\mathrm{ann}}=\left(\prod_{t=1}^{T}(1+r_t)\right)^{252/T}-1")
            st.latex(r"\sigma_{\mathrm{ann}}=\sqrt{252}\,\widehat{\sigma}(r_t)")
            st.latex(r"DD_t=\frac{P_t}{\max_{\tau\le t}P_\tau}-1")
            st.latex(r"\operatorname{CVaR}_{0.95}=\mathbb{E}[r_t\mid r_t\le q_{0.05}(r)]")
            st.caption(
                "Snapshot evidence: adjusted public prices, causal rolling selection, monthly execution windows, "
                "daily OOS path, and a benchmark-aligned risk comparison. Fundamentals, options, sovereign curves, "
                "PBO, WRC, SPA and suitability remain unavailable until a full allocation run computes them."
            )

        as_of = freshness_row.get("As_Of", context_row.get("As_Of", "n/a"))
        st.caption(
            f"Market data as of {as_of} • Artifact created {last_run_at} • "
            "Full analytics are never inferred from missing snapshot fields."
        )
        return

    state_label = {
        "approved": ("Allocation Approved", "approved"),
        "research_only": ("Research Mode — Not Yet Promoted", "research_only"),
        "blocked": ("Allocation Blocked", "blocked"),
    }[alloc_state]

    pill_html = (
        f'<div role="region" aria-label="Allocation gate status" '
        f'style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">'
        f'  <span style="color:var(--qpk-faint);font-size:0.72rem;text-transform:uppercase;letter-spacing:0.12em;">Allocation status</span>'
        f'  {_status_pill(state_label[0], state_label[1], live=True)}'
        f'  {_status_pill("Suitability " + suit_status, suit_status)}'
        f'  {_status_pill("Promotion " + prom_status, prom_status)}'
        f'</div>'
    )
    st.markdown(pill_html, unsafe_allow_html=True)

    if alloc_state == "blocked":
        _banner(
            "error",
            "Allocation blocked by suitability.",
            "Risk limits are breached. The portfolio is not presented as a recommendation. "
            "Adjust horizon, drawdown tolerance, single-name weight, or risk aversion to retry.",
        )
    elif alloc_state == "research_only":
        _banner(
            "warning",
            "Research mode.",
            "Out-of-sample evidence (Deflated Sortino, PBO, Hansen SPA, ICIR) is insufficient to promote the strategy. "
            "Use the candidate weights as research signal only.",
        )

    # KPI strip — sourced from payload + pre-loaded macro row (already computed by backend)
    k1, k2, k3, k4, k5 = st.columns(5)
    suit_row = suit_summary.iloc[0] if isinstance(suit_summary, pd.DataFrame) and not suit_summary.empty else pd.Series(dtype=object)
    prom_row = prom_summary.iloc[0] if isinstance(prom_summary, pd.DataFrame) and not prom_summary.empty else pd.Series(dtype=object)

    k1.metric(
        "Suitability gate",
        suit_status.upper() if suit_status != "unknown" else "—",
        help="Hard risk-limit gate evaluated by the backend.",
    )
    k2.metric(
        "Promotion gate",
        prom_status.upper() if prom_status != "unknown" else "—",
        help="OOS evidence gate (Deflated Sortino, PBO, Hansen SPA, ICIR).",
    )
    k3.metric(
        "Benchmark",
        benchmark_ticker,
        help="Active benchmark for tracking-error and beta-adjusted diagnostics.",
    )
    val_score = prom_row.get("Validation_Score", float("nan")) if not prom_row.empty else float("nan")
    k4.metric(
        "Validation score",
        _fmt_pct(val_score, digits=0) if pd.notna(val_score) else "—",
        help="Share of promotion-gate tests passed.",
    )
    k5.metric(
        "Tickers in run",
        str(tickers_count),
        help="Number of valid price series after data sanity filtering.",
    )

    # Risk warning summary (limits vs observed)
    if isinstance(suit_summary, pd.DataFrame) and not suit_summary.empty:
        cols = st.columns(4)
        for i, (label, obs_key, lim_key) in enumerate([
            ("Vol vs limit", "Observed_Volatility", "Vol_Max"),
            ("Daily CVaR vs limit", "Observed_CVaR", "CVaR_Max_Daily"),
            ("Max drawdown vs limit", "Observed_Max_Drawdown", "DD_Max"),
            ("Single-name vs cap", "Observed_Max_Weight", "Max_Weight"),
        ]):
            obs = suit_row.get(obs_key, float("nan")) if obs_key else float("nan")
            lim = suit_row.get(lim_key, float("nan"))
            cols[i].metric(label, _fmt_pct(obs), delta=f"limit {_fmt_pct(lim)}", delta_color="off")

    if alloc_state == "blocked" and isinstance(suit_breaches, pd.DataFrame) and not suit_breaches.empty:
        _section_header("Breaches", "Hard limits violated by the candidate portfolio.")
        st.dataframe(suit_breaches, use_container_width=True, hide_index=True)

    rates_regime = latest_macro_row.get("Regime_Hawkish_Dovish", np.nan)
    market_regime = latest_macro_row.get("Regime_Bull_Bear", np.nan)
    curve_val = latest_macro_row.get("Country_Curve_10Y_2Y", latest_macro_row.get("Curve_10Y_2Y", float("nan")))
    stress = latest_macro_row.get("Markov_Stress_Prob", float("nan"))
    macro_values = [rates_regime, market_regime, curve_val, stress]
    has_macro_evidence = any(
        pd.notna(value) and str(value).strip().lower() not in {"", "—", "nan", "none"}
        for value in macro_values
    )
    if has_macro_evidence:
        _section_header("Macro context", "Pre-computed by the backend — read-only.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rates regime", str(rates_regime) if pd.notna(rates_regime) else "—")
        m2.metric("Market regime", str(market_regime) if pd.notna(market_regime) else "—")
        m3.metric("10Y–2Y curve (bp)", _fmt_num(curve_val, digits=2))
        m4.metric("Markov stress", _fmt_pct(stress, digits=0) if pd.notna(stress) else "—")

    user_summary = ""
    if isinstance(explanations, dict):
        user_summary = str(explanations.get("user_safe_summary", "") or "")
    if user_summary:
        st.markdown(
            f'<div class="small-note" style="margin-top:14px;">{_html.escape(user_summary)}</div>',
            unsafe_allow_html=True,
        )

    _render_daily_market_pulse(
        daily_snapshot_gate,
        benchmark_ticker=benchmark_ticker,
        created_at=daily_snapshot_created_at,
    )

    st.caption(
        f"Last run: {last_run_at} • Benchmark: {benchmark_ticker} • UI = render-only contract"
    )


# ------------------------------------------------------------
# Render: Recommended Allocation
# ------------------------------------------------------------

def render_allocation(gate: dict) -> None:
    alloc_state = gate["allocation_state"]
    alloc_block = gate["allocation_block"]

    if alloc_state == "blocked":
        _banner(
            "error",
            "Allocation Blocked",
            "Recommendation withheld: suitability constraints are breached. Review limits in Overview → Risk limits.",
        )
        return

    if alloc_state == "research_only":
        _banner(
            "warning",
            "Research-only signal.",
            "Weights below are a research signal pending out-of-sample evidence. They are not a recommended allocation.",
        )

    portfolio_df = alloc_block.get("recommended_portfolio", pd.DataFrame()) if isinstance(alloc_block, dict) else pd.DataFrame()
    if not isinstance(portfolio_df, pd.DataFrame) or portfolio_df.empty:
        _empty_state(
            "No allocation available yet.",
            suggestion="Run the allocation engine from the sidebar to populate this view.",
        )
        return

    portfolio_df = _with_weight_percent(portfolio_df, "Weight")

    display_cols = [c for c in [
        "Ticker", "Sector", "Weight_Pct", "Composite_Score", "PIT_Confidence",
        "Quality_Score", "Value_Score", "Growth_Score", "Technical_Score", "Liquidity_Score",
        "Bayesian_Alpha_Shrink", "Prob_Alpha_Positive", "Risk_Contribution",
        "Latent_Regime_State", "Inclusion_Reason",
    ] if c in portfolio_df.columns]

    table = portfolio_df[display_cols] if display_cols else portfolio_df
    column_config = {}
    if "Weight_Pct" in table.columns:
        column_config["Weight_Pct"] = st.column_config.ProgressColumn(
            "Weight", format="%.2f%%", min_value=0.0, max_value=100.0,
            help="Portfolio weight (fraction of NAV).",
        )
    if "PIT_Confidence" in table.columns:
        column_config["PIT_Confidence"] = st.column_config.NumberColumn(
            "PIT confidence", format="%.0f%%",
            help="Confidence that fundamentals are point-in-time accurate.",
        )
    if "Composite_Score" in table.columns:
        column_config["Composite_Score"] = st.column_config.NumberColumn("Composite", format="%.2f")
    if "Prob_Alpha_Positive" in table.columns:
        column_config["Prob_Alpha_Positive"] = st.column_config.NumberColumn(
            "P(α>0)", format="%.0f%%",
            help="Posterior probability that alpha is positive (Bayesian shrinkage).",
        )

    weight_source = (
        portfolio_df["Weight"]
        if "Weight" in portfolio_df.columns
        else pd.Series(0.0, index=portfolio_df.index)
    )
    weights_numeric = pd.to_numeric(weight_source, errors="coerce").fillna(0.0)
    weight_sum = float(weights_numeric.sum())
    effective_n = float(1.0 / np.square(weights_numeric).sum()) if np.square(weights_numeric).sum() > 0 else np.nan
    sector_series = (
        portfolio_df["Sector"].astype(str).str.strip()
        if "Sector" in portfolio_df.columns
        else pd.Series(dtype=str)
    )
    invalid_sector_labels = {"price-only snapshot", "undefined", "unknown", "none", "nan", ""}
    verified_sector_count = int(
        sector_series[~sector_series.str.lower().isin(invalid_sector_labels)].nunique()
    ) if not sector_series.empty else 0

    integrity_cols = st.columns(4)
    integrity_cols[0].metric("Positions", f"{len(portfolio_df):d}")
    integrity_cols[1].metric("Weight total", f"{weight_sum * 100:.2f}%")
    integrity_cols[2].metric("Effective positions", _fmt_num(effective_n, digits=2))
    integrity_cols[3].metric("Verified sectors", f"{verified_sector_count:d}")
    if not np.isclose(weight_sum, 1.0, atol=0.005):
        _banner(
            "error",
            "Allocation integrity failure.",
            f"Portfolio weights sum to {weight_sum:.6f}; the allocation is withheld until the backend returns a normalized simplex.",
        )
        return

    allocation_title = "Candidate Allocation" if alloc_state == "research_only" else "Recommended Allocation"
    allocation_detail = (
        "Research weights produced by the backend; promotion gates remain binding."
        if alloc_state == "research_only"
        else "Weights produced by the promoted objective; read-only from backend."
    )
    a1, a2 = st.columns([1.35, 0.65])
    with a1:
        _section_header(allocation_title, allocation_detail)
        st.dataframe(table, use_container_width=True, hide_index=True, column_config=column_config)
    with a2:
        _section_header("Sector exposure", "Aggregated only from verified full-analysis classifications.")
        if px is not None and "Sector" in portfolio_df.columns and "Weight" in portfolio_df.columns:
            sectors = portfolio_df.groupby("Sector", dropna=False)["Weight"].sum().reset_index().sort_values("Weight", ascending=True)
            sector_labels = set(sectors["Sector"].astype(str).str.strip().str.lower())
            if not sectors.empty and not sector_labels.issubset(invalid_sector_labels):
                fig = px.bar(sectors, x="Weight", y="Sector", orientation="h", text="Weight")
                fig.update_traces(texttemplate="%{x:.1%}", textposition="outside",
                                  marker_color="#7dd3fc", marker_line_color="rgba(125,211,252,0.5)")
                fig.update_xaxes(tickformat=".0%")
                st.plotly_chart(_plotly_dark_layout(fig, height=340), use_container_width=True, config={"displayModeBar": False})
            else:
                _empty_state(
                    "Verified sector classifications are unavailable.",
                    suggestion="Sector exposure is shown only for a full fundamental analysis, never from a price-only snapshot.",
                )
        else:
            _empty_state("Sector breakdown unavailable.")


# ------------------------------------------------------------
# Render: My Portfolio
# ------------------------------------------------------------

def _artifact_payload(bundle: dict[str, pd.DataFrame], name: str) -> dict:
    artifacts = bundle.get("artifacts", pd.DataFrame())
    if not isinstance(artifacts, pd.DataFrame) or artifacts.empty:
        return {}
    selected = artifacts[artifacts.get("artifact_name", pd.Series(dtype=str)).astype(str) == name]
    if selected.empty:
        return {}
    value = selected.iloc[0].get("artifact_json")
    return value if isinstance(value, dict) else {}


def render_my_portfolio(username: str) -> None:
    _section_header(
        "My Portfolio",
        "Versioned allocations saved after each successful optimization. Every view is reconstructed from the immutable Supabase run.",
    )
    if list_user_portfolios is None or load_user_portfolio is None or not supabase_available():
        _empty_state(
            "Personal portfolio storage is unavailable.",
            suggestion="Verify the server-side Supabase configuration. Existing global analytics remain available.",
        )
        return
    try:
        portfolios = list_user_portfolios(username, limit=50)
    except Exception as exc:
        _empty_state("Personal portfolios could not be loaded.", suggestion=str(exc))
        return
    if portfolios.empty:
        _empty_state(
            "No saved portfolios yet.",
            suggestion="Configure the mandate in the sidebar and run the allocation engine. The completed result will appear here.",
        )
        return

    portfolios = portfolios.copy()
    portfolios["created_at"] = pd.to_datetime(portfolios["created_at"], errors="coerce", utc=True)
    portfolios["Display"] = portfolios.apply(
        lambda row: (
            f"{row.get('portfolio_name', 'My Portfolio')} · "
            f"{row.get('benchmark_ticker', '—')} · "
            f"{row['created_at'].strftime('%Y-%m-%d %H:%M UTC') if pd.notna(row['created_at']) else 'date unavailable'}"
        ),
        axis=1,
    )
    selected_label = st.selectbox(
        "Saved version",
        portfolios["Display"].tolist(),
        index=0,
        help="Select any prior immutable optimization run.",
    )
    selected_meta = portfolios.loc[portfolios["Display"] == selected_label].iloc[0]
    try:
        bundle = load_user_portfolio(str(selected_meta["run_id"]), username)
    except Exception as exc:
        _empty_state("The selected portfolio could not be loaded.", suggestion=str(exc))
        return
    run_df = bundle.get("run", pd.DataFrame())
    portfolio_df = bundle.get("portfolio", pd.DataFrame()).copy()
    risk_df = bundle.get("risk", pd.DataFrame()).copy()
    backtest_df = bundle.get("backtest", pd.DataFrame()).copy()
    dashboard = _artifact_payload(bundle, "dashboard_payload")
    if run_df.empty or portfolio_df.empty:
        _empty_state("The saved run is incomplete and remains withheld from analysis.")
        return

    risk_map = {}
    if not risk_df.empty and {"metric", "value"}.issubset(risk_df.columns):
        risk_map = dict(zip(risk_df["metric"].astype(str), pd.to_numeric(risk_df["value"], errors="coerce")))

    metrics = st.columns(6)
    metrics[0].metric("Annual return", _fmt_pct(risk_map.get("Annualized_Return")))
    metrics[1].metric("Annual volatility", _fmt_pct(risk_map.get("Annualized_Vol")))
    metrics[2].metric("Sortino", _fmt_num(risk_map.get("Sortino"), digits=2))
    metrics[3].metric("Max drawdown", _fmt_pct(risk_map.get("Max_Drawdown")))
    metrics[4].metric("Daily CVaR 95%", _fmt_pct(risk_map.get("CVaR_95")))
    metrics[5].metric("Benchmark", str(selected_meta.get("benchmark_ticker") or "—"))

    weight_col = "weight" if "weight" in portfolio_df.columns else "Weight"
    ticker_col = "ticker" if "ticker" in portfolio_df.columns else "Ticker"
    sector_col = "sector" if "sector" in portfolio_df.columns else "Sector"
    portfolio_df[weight_col] = pd.to_numeric(portfolio_df[weight_col], errors="coerce").fillna(0.0)
    weight_sum = float(portfolio_df[weight_col].sum())
    if not np.isclose(weight_sum, 1.0, atol=0.005):
        _banner(
            "error",
            "Saved allocation failed the simplex check.",
            f"Weights total {weight_sum:.6f}. This version is visible for audit but is not actionable.",
        )

    left, right = st.columns([1.35, 0.65])
    with left:
        _section_header(
            "Saved allocation",
            f"{len(portfolio_df)} positions · objective {selected_meta.get('objective') or 'not recorded'} · period {selected_meta.get('price_period') or '—'}",
        )
        display = portfolio_df.rename(
            columns={
                ticker_col: "Ticker",
                sector_col: "Sector",
                weight_col: "Weight",
                "composite_score": "Composite",
                "optimization_sortino": "Optimization Sortino",
            }
        )
        display["Weight_Pct"] = pd.to_numeric(display["Weight"], errors="coerce").fillna(0.0) * 100.0
        visible = [c for c in ["Ticker", "Sector", "Weight_Pct", "Composite", "Optimization Sortino"] if c in display]
        st.dataframe(
            display[visible],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Weight_Pct": st.column_config.ProgressColumn(
                    "Weight", format="%.2f%%", min_value=0.0, max_value=100.0
                )
            },
        )
    with right:
        _section_header("Sector exposure", "Aggregated from the saved point-in-time allocation.")
        if px is not None and sector_col in portfolio_df:
            sectors = (
                portfolio_df.assign(**{sector_col: portfolio_df[sector_col].fillna("Unclassified")})
                .groupby(sector_col, dropna=False)[weight_col]
                .sum()
                .reset_index()
                .sort_values(weight_col)
            )
            fig = px.bar(sectors, x=weight_col, y=sector_col, orientation="h", text=weight_col)
            fig.update_traces(texttemplate="%{x:.1%}", marker_color="#7dd3fc")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(
                _plotly_dark_layout(fig, height=max(300, 34 * len(sectors) + 90)),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    charts = dashboard.get("charts", {}) if isinstance(dashboard, dict) else {}
    price_paths = pd.DataFrame(charts.get("price_paths", [])) if isinstance(charts, dict) else pd.DataFrame()
    drawdowns = pd.DataFrame(charts.get("drawdowns", [])) if isinstance(charts, dict) else pd.DataFrame()
    if price_paths.empty and not backtest_df.empty:
        price_paths = backtest_df.copy()

    path_col, dd_col = st.columns(2)
    with path_col:
        _section_header("Portfolio vs benchmark", "Out-of-sample price or NAV path saved with this run.")
        if not price_paths.empty:
            date_col = next((c for c in ["Date", "period_end", "rebalance_date"] if c in price_paths), None)
            value_cols = [c for c in price_paths.columns if c != date_col and pd.api.types.is_numeric_dtype(price_paths[c])]
            if date_col and value_cols:
                price_paths[date_col] = pd.to_datetime(price_paths[date_col], errors="coerce")
                long = price_paths.melt(id_vars=date_col, value_vars=value_cols, var_name="Series", value_name="Value")
                fig = px.line(long.dropna(), x=date_col, y="Value", color="Series")
                st.plotly_chart(
                    _plotly_dark_layout(fig, height=360),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                _empty_state("A chartable price path is not present in this saved version.")
        else:
            _empty_state("No saved price path is available for this version.")
    with dd_col:
        _section_header("Drawdown", "Peak-to-trough loss path computed from the saved daily series.")
        if not drawdowns.empty and "Date" in drawdowns:
            drawdowns["Date"] = pd.to_datetime(drawdowns["Date"], errors="coerce")
            value_cols = [c for c in drawdowns.columns if c != "Date"]
            long = drawdowns.melt(id_vars="Date", value_vars=value_cols, var_name="Series", value_name="Drawdown")
            fig = px.area(long.dropna(), x="Date", y="Drawdown", color="Series")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(
                _plotly_dark_layout(fig, height=360),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            _empty_state("No saved daily drawdown path is available for this version.")

    st.markdown("#### Mathematical mandate")
    st.latex(
        r"w_t\in\Delta_N,\qquad "
        r"R_{p,t+1}=w_t^\top r_{t+1}-TC_t,\qquad "
        r"DD_t=\frac{NAV_t}{\max_{\tau\leq t}NAV_\tau}-1"
    )
    with st.expander("Saved configuration and audit metadata"):
        run_config = run_df.iloc[0].get("config", {}) if not run_df.empty else {}
        st.json(run_config if isinstance(run_config, dict) else {})
        st.dataframe(portfolios.drop(columns=["Display"]), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Render: Research Strategy
# ------------------------------------------------------------

def render_research_strategy(gate: dict | None = None) -> None:
    artifacts = load_xcdr_research_artifacts()
    summary = artifacts.get("summary", pd.DataFrame())
    daily = artifacts.get("daily", pd.DataFrame())
    daily_summary = artifacts.get("daily_summary", pd.DataFrame())
    weights = artifacts.get("weights", pd.DataFrame())
    windows = artifacts.get("windows", pd.DataFrame())
    report = artifacts.get("report", {})
    objective = select_research_objective(summary)

    _section_header(
        "Research Strategy",
        "XCDR/XODR research artifact replacing the former side portfolio view. Read-only; promotion gates remain binding.",
    )

    if objective is None:
        _empty_state(
            "No XCDR/XODR research artifact found.",
            f"Run C:\\Users\\chris\\FINANZAS\\run_xcdr_v3_parallel_research.py. Artifact directory: {artifacts.get('artifact_dir')}",
        )
        return

    summary = summary.copy()
    selected = summary[summary["objective"].astype(str) == str(objective)].copy()
    if selected.empty:
        _empty_state("Selected research objective is unavailable in the summary artifact.")
        return
    selected_row = selected.iloc[0]

    daily_metric_row = pd.Series(dtype=object)
    if isinstance(daily_summary, pd.DataFrame) and not daily_summary.empty and "objective" in daily_summary.columns:
        ds = daily_summary[daily_summary["objective"].astype(str) == str(objective)]
        if not ds.empty:
            daily_metric_row = ds.iloc[0]

    nav = _research_daily_frame(daily, objective)
    xi = "n/a"
    if isinstance(windows, pd.DataFrame) and not windows.empty and {"objective", "xi"}.issubset(windows.columns):
        wx = windows[windows["objective"].astype(str) == str(objective)]
        if not wx.empty:
            xi = str(wx["xi"].mode().iloc[0]) if not wx["xi"].mode().empty else str(wx["xi"].iloc[-1])
    elif isinstance(daily, pd.DataFrame) and not daily.empty and {"objective", "xi"}.issubset(daily.columns):
        dx = daily[daily["objective"].astype(str) == str(objective)]
        if not dx.empty:
            xi = str(dx["xi"].mode().iloc[0]) if not dx["xi"].mode().empty else str(dx["xi"].iloc[-1])

    def _first_available(*keys, source=None, fallback=np.nan):
        source = selected_row if source is None else source
        for key in keys:
            if key in source.index and not pd.isna(source[key]):
                return source[key]
        return fallback

    if not nav.empty:
        p = nav["portfolio_return"]
        x = nav["xi_return"]
        computed = {
            "strategy_return": _ann_return_from_daily(p),
            "benchmark_return": _ann_return_from_daily(x),
            "active_return": _ann_return_from_daily(p - x),
            "strategy_vol": _ann_vol_from_daily(p),
            "benchmark_vol": _ann_vol_from_daily(x),
            "strategy_downside": _downside_from_daily(p),
            "benchmark_downside": _downside_from_daily(x),
            "strategy_cvar": _cvar_loss_from_daily(p),
            "benchmark_cvar": _cvar_loss_from_daily(x),
            "strategy_maxdd": _maxdd_loss_from_daily(p),
            "benchmark_maxdd": _maxdd_loss_from_daily(x),
        }
    else:
        computed = {}

    strategy_return = _first_available("daily_ann_return", source=daily_metric_row, fallback=_first_available("daily_ann_return", "ann_return", fallback=computed.get("strategy_return", np.nan)))
    benchmark_return = _first_available("daily_xi_ann_return", source=daily_metric_row, fallback=_first_available("daily_xi_ann_return", "xi_ann_return", fallback=computed.get("benchmark_return", np.nan)))
    active_return = _first_available("daily_active_ann_return", source=daily_metric_row, fallback=_first_available("daily_active_ann_return", "active_ann_return", fallback=computed.get("active_return", np.nan)))
    strategy_vol = _first_available("daily_ann_vol", source=daily_metric_row, fallback=_first_available("ann_vol", fallback=computed.get("strategy_vol", np.nan)))
    benchmark_vol = _first_available("daily_xi_ann_vol", source=daily_metric_row, fallback=_first_available("xi_ann_vol", fallback=computed.get("benchmark_vol", np.nan)))
    strategy_downside = _first_available("daily_downside", source=daily_metric_row, fallback=_first_available("downside", fallback=computed.get("strategy_downside", np.nan)))
    benchmark_downside = _first_available("daily_xi_downside", source=daily_metric_row, fallback=_first_available("xi_downside", fallback=computed.get("benchmark_downside", np.nan)))
    strategy_cvar = _first_available("daily_cvar_loss", source=daily_metric_row, fallback=_first_available("cvar_loss", fallback=computed.get("strategy_cvar", np.nan)))
    benchmark_cvar = _first_available("daily_xi_cvar_loss", source=daily_metric_row, fallback=_first_available("xi_cvar_loss", fallback=computed.get("benchmark_cvar", np.nan)))
    strategy_maxdd = _first_available("daily_maxdd_loss", source=daily_metric_row, fallback=_first_available("maxdd_loss", fallback=computed.get("strategy_maxdd", np.nan)))
    benchmark_maxdd = _first_available("daily_xi_maxdd_loss", source=daily_metric_row, fallback=_first_available("xi_maxdd_loss", fallback=computed.get("benchmark_maxdd", np.nan)))
    upside_capture = _first_available("daily_upside_capture", source=daily_metric_row, fallback=_first_available("upside_capture"))
    downside_capture = _first_available("daily_downside_capture", source=daily_metric_row, fallback=_first_available("downside_capture"))
    wrc_p = _first_available("WRC_p")
    spa_p = _first_available("SPA_p")
    pbo = _first_available("PBO_proxy")
    gate_pass = _coerce_bool(_first_available("research_gate_pass", fallback=False))
    windows_n = int(float(_first_available("windows", fallback=0) or 0))
    min_windows = int(float(_first_available("Promotion_Min_Windows", fallback=report.get("config", {}).get("min_promotion_windows", 12)) or 12))

    if gate_pass:
        _banner("success", "Promotion gate passed.", "This objective satisfies the stored research promotion diagnostics.")
    else:
        _banner(
            "warning",
            "Research-only candidate.",
            "The dashboard can display this strategy, but it is not promoted unless WRC, SPA, PBO, upside/downside and preservation gates pass.",
        )
    if windows_n < min_windows:
        _banner(
            "warning",
            "Insufficient walk-forward depth.",
            f"Current artifact has {windows_n} window(s); strict promotion requires at least {min_windows}.",
        )
    if artifacts.get("missing"):
        _banner("neutral", "Missing optional artifacts.", ", ".join(artifacts["missing"][:6]))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Research return", _fmt_pct(strategy_return), delta=f"{_fmt_pct(active_return)} active")
    k2.metric(f"{xi} return", _fmt_pct(benchmark_return))
    k3.metric("Research volatility", _fmt_pct(strategy_vol), delta=f"{_fmt_pct(strategy_vol - benchmark_vol)} vs benchmark" if not pd.isna(strategy_vol) and not pd.isna(benchmark_vol) else None)
    k4.metric("Max drawdown", _fmt_pct(strategy_maxdd), delta=f"{_fmt_pct(strategy_maxdd - benchmark_maxdd)} vs benchmark" if not pd.isna(strategy_maxdd) and not pd.isna(benchmark_maxdd) else None)

    comparison = pd.DataFrame(
        [
            {"Metric": "Annual return", "Research strategy": strategy_return, "Benchmark": benchmark_return, "Interpretation": "Mean daily OOS return annualized."},
            {"Metric": "Annual volatility", "Research strategy": strategy_vol, "Benchmark": benchmark_vol, "Interpretation": "Realized daily OOS volatility annualized."},
            {"Metric": "Downside deviation", "Research strategy": strategy_downside, "Benchmark": benchmark_downside, "Interpretation": "Annualized lower-partial deviation."},
            {"Metric": "Daily CVaR 95% loss", "Research strategy": strategy_cvar, "Benchmark": benchmark_cvar, "Interpretation": "Expected tail loss beyond the 95% loss quantile."},
            {"Metric": "Max drawdown loss", "Research strategy": strategy_maxdd, "Benchmark": benchmark_maxdd, "Interpretation": "Largest OOS peak-to-trough loss."},
            {"Metric": "Upside capture", "Research strategy": upside_capture, "Benchmark": 1.0, "Interpretation": "Target: > 1 versus optimal benchmark xi."},
            {"Metric": "Downside capture", "Research strategy": downside_capture, "Benchmark": 1.0, "Interpretation": "Target: < 1 versus optimal benchmark xi."},
            {"Metric": "WRC p-value", "Research strategy": wrc_p, "Benchmark": np.nan, "Interpretation": "Strict promotion target: < 0.05."},
            {"Metric": "SPA p-value", "Research strategy": spa_p, "Benchmark": np.nan, "Interpretation": "Strict promotion target: < 0.05."},
            {"Metric": "PBO proxy", "Research strategy": pbo, "Benchmark": np.nan, "Interpretation": "Strict promotion target: < 0.10."},
        ]
    )

    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        _section_header("Out-of-sample value path", f"Selected objective: {objective} | optimal benchmark xi: {xi}")
        if not nav.empty:
            plot = nav.rename(columns={"date": "Date"})
            fig = _line_chart(
                plot,
                x="Date",
                ys=["Research strategy NAV", "Benchmark NAV", "Active NAV"],
                height=390,
                title="Research strategy vs benchmark",
                y_format=".2f",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        else:
            _empty_state("Daily OOS path unavailable.", "Run the research script to generate daily OOS artifacts.")
    with c2:
        _section_header("Drawdown path", "Computed from daily OOS NAV paths.")
        if not nav.empty:
            fig = _line_chart(
                nav.rename(columns={"date": "Date"}),
                x="Date",
                ys=["Research strategy drawdown", "Benchmark drawdown"],
                height=390,
                title="Drawdown comparison",
                percent=True,
                fill=True,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        else:
            _empty_state("Drawdown path unavailable.")

    _section_header("Risk / return diagnostics", "Strategy and benchmark metrics use the same OOS daily path when available.")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Research strategy": st.column_config.NumberColumn("Research strategy", format="%.4f"),
            "Benchmark": st.column_config.NumberColumn("Benchmark", format="%.4f"),
        },
    )

    _section_header("Research weights", "Latest selected walk-forward weights by ticker. Requires the weights artifact generated by the research script.")
    if isinstance(weights, pd.DataFrame) and not weights.empty and {"objective", "ticker", "weight"}.issubset(weights.columns):
        w = weights[weights["objective"].astype(str) == str(objective)].copy()
        if "test_start" in w.columns:
            w["test_start_dt"] = pd.to_datetime(w["test_start"], errors="coerce")
            latest = w["test_start_dt"].max()
            if pd.notna(latest):
                w = w[w["test_start_dt"] == latest]
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
        w = w.sort_values("weight", ascending=False)
        w["Weight_%"] = w["weight"] * 100.0
        view_cols = [c for c in ["ticker", "Weight_%", "xi", "state_label", "val_growth_mass", "val_alpha_signal_mass", "universe_selected_count"] if c in w.columns]
        st.dataframe(
            w[view_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "Weight_%": st.column_config.ProgressColumn("Weight", format="%.2f%%", min_value=0.0, max_value=100.0),
                "val_growth_mass": st.column_config.NumberColumn("Growth mass", format="%.3f"),
                "val_alpha_signal_mass": st.column_config.NumberColumn("Alpha mass", format="%.3f"),
            },
        )
    else:
        _empty_state(
            "Weights artifact unavailable.",
            "Re-run the research script after this update to create xcdr_v3_parallel_research_weights.csv.",
        )

    _section_header("Candidate table", "All research candidates in the latest summary artifact.")
    keep = [
        "objective", "windows", "ann_return", "xi_ann_return", "active_ann_return", "ann_vol",
        "downside", "cvar_loss", "maxdd_loss", "upside_capture", "downside_capture",
        "WRC_p", "SPA_p", "PBO_proxy", "research_gate_pass",
    ]
    keep = [c for c in keep if c in summary.columns]
    st.dataframe(summary[keep], use_container_width=True, hide_index=True)


def render_private_sleeve(gate: dict, *, side_diag: pd.DataFrame, fixed_tickers: tuple[str, ...] = ()) -> None:
    alloc_block = gate["allocation_block"]
    side_df = alloc_block.get("side_sleeve", pd.DataFrame()) if isinstance(alloc_block, dict) else pd.DataFrame()
    _banner(
        "info",
        "Private Side Alpha Sleeve.",
        "Private scenario: user-specified fixed-weight sleeve with residual allocation in CBRS/Cerebras. "
        "Not part of the public-data recommendation engine. Disclosed separately by design.",
    )

    if not isinstance(side_df, pd.DataFrame) or side_df.empty:
        _empty_state("No Private Side Alpha available.", "The private sleeve is optional and depends on configuration.")
        return

    side_df = _normalize_weight_fraction(side_df, "Weight")

    # Tag fixed-weight tickers with a synthetic Constraint column (R8).
    fixed_set = {str(t).upper() for t in fixed_tickers}
    if "Ticker" in side_df.columns:
        upper = side_df["Ticker"].astype(str).str.upper()
        side_df = side_df.copy()
        side_df.insert(min(2, side_df.shape[1]), "Constraint",
                       np.where(upper.isin(fixed_set), "FIXED", "OPTIMIZED"))

    display_cols = [c for c in [
        "Ticker", "Sector", "Constraint", "Weight", "Composite_Score", "PIT_Confidence", "Inclusion_Reason"
    ] if c in side_df.columns]
    table = side_df[display_cols] if display_cols else side_df
    column_config = {}
    if "Weight" in table.columns:
        column_config["Weight"] = st.column_config.ProgressColumn(
            "Weight", format="%.2f%%", min_value=0.0, max_value=1.0,
        )
    if "Constraint" in table.columns:
        column_config["Constraint"] = st.column_config.TextColumn(
            "Constraint",
            help="FIXED = mandated weight; OPTIMIZED = Sortino-optimized remainder.",
        )

    s1, s2 = st.columns([1.35, 0.65])
    with s1:
        _section_header(
            "Private Side Alpha composition",
            "FIXED rows reflect mandated weights; OPTIMIZED rows are solver output.",
        )
        st.dataframe(table, use_container_width=True, hide_index=True, column_config=column_config)
    with s2:
        _section_header("Diagnostics")
        if isinstance(side_diag, pd.DataFrame) and not side_diag.empty:
            st.dataframe(side_diag, use_container_width=True, hide_index=True)
        else:
            _empty_state("No diagnostics available.")


# ------------------------------------------------------------
# Render: Price Path & Drawdown
# ------------------------------------------------------------

def render_price_paths(gate: dict) -> None:
    charts = gate["charts_block"]
    tables = gate["tables_block"]
    price_paths = charts.get("price_paths", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
    drawdowns = charts.get("drawdowns", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
    max_dd_table = tables.get("max_drawdown", pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame()

    if not isinstance(price_paths, pd.DataFrame) or price_paths.empty:
        _empty_state(
            "Price path unavailable.",
            suggestion="Backtest requires at least one out-of-sample rebalance window with valid holdings.",
        )
        return

    date_col = "Date" if "Date" in price_paths.columns else "Period_End"
    series_cols = [c for c in price_paths.columns if c != date_col]

    _section_header(
        "Price Path",
        "Observed benchmark price vs optimized synthetic NAV reconstructed from out-of-sample holdings."
    )
    fig = _line_chart(price_paths, x=date_col, ys=series_cols, height=400, y_format=".2f")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        _empty_state("Chart engine unavailable. Install plotly to view interactive price paths.")

    _section_header(
        "Drawdown",
        "Pointwise drawdown from running maximum (formula reported by backend: P_t / running_max(P_t) − 1)."
    )
    if isinstance(drawdowns, pd.DataFrame) and not drawdowns.empty:
        dd_date_col = "Date" if "Date" in drawdowns.columns else "Period_End"
        dd_series_cols = [c for c in drawdowns.columns if c != dd_date_col]
        dd_fig = _line_chart(drawdowns, x=dd_date_col, ys=dd_series_cols, height=300, percent=True, fill=True)
        if dd_fig is not None:
            st.plotly_chart(dd_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    else:
        _empty_state("Drawdown series not produced for this run.")

    if isinstance(max_dd_table, pd.DataFrame) and not max_dd_table.empty:
        _section_header("Max drawdown markers")
        view = max_dd_table.copy()
        if "Max_Drawdown" in view.columns:
            view["Max_Drawdown"] = pd.to_numeric(view["Max_Drawdown"], errors="coerce")
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Max_Drawdown": st.column_config.NumberColumn("Max drawdown", format="%.2f%%"),
                "Max_Drawdown_Date": st.column_config.DateColumn("Date"),
            } if "Max_Drawdown" in view.columns else None,
        )


# ------------------------------------------------------------
# Render: Risk Diagnostics
# ------------------------------------------------------------

def render_risk_diagnostics(gate: dict, *, return_diagnostics: dict, perf_summary_df: pd.DataFrame,
                            forecast_path: pd.DataFrame, forecast_summary: pd.DataFrame) -> None:
    tables = gate["tables_block"]
    charts = gate["charts_block"]
    risk_table = tables.get("risk", pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame()

    # Top-N risk metric cards (R5): promote core institutional metrics from the
    # backend's performance_summary into KPI cards; leave the raw table behind an expander.
    _section_header("Risk summary", "Top institutional metrics, computed by the backend.")

    headline_specs = [
        ("Sharpe", ("Sharpe", "Sharpe_Ratio", "Annualized_Sharpe"), "{:.2f}", "Risk-adjusted return (total risk)."),
        ("Sortino", ("Sortino", "Sortino_Ratio", "Annualized_Sortino"), "{:.2f}", "Risk-adjusted return (downside risk)."),
        ("Ann. volatility", ("Annualized_Vol", "Ann_Vol", "Volatility"), "{:.1%}", "Annualized realized volatility."),
        ("Max drawdown", ("Max_Drawdown", "Maximum_Drawdown"), "{:.1%}", "Peak-to-trough drawdown of the synthetic NAV."),
        ("Daily CVaR 95%", ("CVaR_95", "Hist_CVaR_95_Daily", "CVaR"), "{:.2%}", "Expected loss beyond the 5th percentile (1d)."),
    ]

    metric_lookup: dict[str, float] = {}
    if isinstance(risk_table, pd.DataFrame) and not risk_table.empty and {"Metric", "Value"}.issubset(risk_table.columns):
        for _, row in risk_table.iterrows():
            try:
                metric_lookup[str(row["Metric"])] = float(row["Value"])
            except (TypeError, ValueError):
                continue

    headline_cols = st.columns(5)
    for i, (label, keys, fmt, help_text) in enumerate(headline_specs):
        val = float("nan")
        for k in keys:
            if k in metric_lookup and pd.notna(metric_lookup[k]):
                val = metric_lookup[k]
                break
        if pd.isna(val):
            display = "—"
        else:
            try:
                display = fmt.format(val)
            except Exception:
                display = _fmt_num(val)
        headline_cols[i].metric(label, display, help=help_text)

    if isinstance(risk_table, pd.DataFrame) and not risk_table.empty:
        with st.expander("Full performance summary", expanded=False):
            st.dataframe(risk_table, use_container_width=True, hide_index=True)
    else:
        _empty_state("No risk summary table available.")

    cone_df = charts.get("forecast_cone", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
    cond_vol_df = charts.get("conditional_vol", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()

    rc1, rc2 = st.columns(2)
    with rc1:
        _section_header("Forecast cone", "GBM/conditional forecast envelope provided by backend.")
        if isinstance(cone_df, pd.DataFrame) and not cone_df.empty:
            date_col = "Date" if "Date" in cone_df.columns else cone_df.columns[0]
            cone_series = [c for c in cone_df.columns if c != date_col]
            fig = _line_chart(cone_df, x=date_col, ys=cone_series, height=320)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        elif isinstance(forecast_path, pd.DataFrame) and not forecast_path.empty:
            date_col = "Date" if "Date" in forecast_path.columns else forecast_path.columns[0]
            cols = [c for c in forecast_path.columns if c != date_col]
            fig = _line_chart(forecast_path, x=date_col, ys=cols, height=320)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        else:
            _empty_state("No forecast cone available.")

    with rc2:
        _section_header("Conditional volatility", "ARCH/GARCH conditional paths produced by backend.")
        if isinstance(cond_vol_df, pd.DataFrame) and not cond_vol_df.empty:
            date_col = "Date" if "Date" in cond_vol_df.columns else cond_vol_df.columns[0]
            cols = [c for c in cond_vol_df.columns if c != date_col]
            fig = _line_chart(cond_vol_df, x=date_col, ys=cols, height=320, percent=True)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        else:
            _empty_state("Conditional vol path not produced.")

    if isinstance(return_diagnostics, dict):
        pelt_timeline_df = return_diagnostics.get("pelt_timeline", pd.DataFrame())
        if isinstance(pelt_timeline_df, pd.DataFrame) and not pelt_timeline_df.empty:
            _section_header("PELT regime timeline", "Change-point segmentation of realized volatility.")
            try:
                st.pyplot(plot_pelt_regime_timeline(pelt_timeline_df), clear_figure=True)
            except Exception:
                _empty_state("PELT chart could not be rendered.")

    if isinstance(forecast_summary, pd.DataFrame) and not forecast_summary.empty:
        with st.expander("GBM forecast summary table", expanded=False):
            st.dataframe(forecast_summary, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Render: Validation
# ------------------------------------------------------------

def render_validation(gate: dict) -> None:
    tests_df = gate["prom_tests"]
    prom_summary = gate["prom_summary"]
    tables = gate["tables_block"]
    if not isinstance(tests_df, pd.DataFrame) or tests_df.empty:
        _empty_state("Promotion gate not yet evaluated.")
        return

    _section_header("Promotion gate tests", "Each test must pass for the strategy to be promoted.")
    cards = st.columns(4)
    for i, (_, row) in enumerate(tests_df.iterrows()):
        col = cards[i % 4]
        passed = bool(row.get("Passed", False))
        observed = row.get("Observed", float("nan"))
        threshold = row.get("Threshold", float("nan"))
        direction = str(row.get("Direction", ""))
        name = str(row.get("Test", "—"))
        if pd.isna(observed):
            kind, label = "watchlist", "MISSING"
        elif passed:
            kind, label = "promoted", "PASS"
        else:
            kind, label = "rejected", "FAIL"
        op = "≥" if direction == "higher" else "<"
        observed_text = _fmt_num(observed, digits=3) if abs(threshold) < 5 else _fmt_pct(observed)
        threshold_text = _fmt_num(threshold, digits=3) if abs(threshold) < 5 else _fmt_pct(threshold)
        col.markdown(
            f'<div style="border:1px solid var(--qpk-line);border-radius:6px;padding:14px;background:rgba(11,16,26,0.6);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'  <span style="color:var(--qpk-muted);font-size:0.74rem;text-transform:uppercase;letter-spacing:0.08em;">{name}</span>'
            f'  {_status_pill(label, kind)}'
            f'</div>'
            f'<div style="margin-top:10px;font-family:var(--font-mono);font-size:1.25rem;color:var(--qpk-text);font-feature-settings:\'tnum\';">{observed_text}</div>'
            f'<div class="small-note" style="margin-top:2px;">required: observed {op} {threshold_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    val_table = tables.get("validation", pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame()
    if isinstance(val_table, pd.DataFrame) and not val_table.empty:
        with st.expander("Full validation summary (IC, ICIR, deflated Sortino, CPCV/PBO, Hansen SPA)", expanded=False):
            st.dataframe(val_table, use_container_width=True, hide_index=True)

    if isinstance(prom_summary, pd.DataFrame) and not prom_summary.empty:
        with st.expander("Promotion summary row", expanded=False):
            st.dataframe(prom_summary, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Render: Market Regime
# ------------------------------------------------------------

def render_market_regime(
    gate: dict,
    *,
    latest_macro_row: pd.Series,
    macro_history_df: pd.DataFrame,
    sentiment_sem: dict,
    global_rates_df: pd.DataFrame,
    global_rate_history_df: pd.DataFrame,
    interbank_df: pd.DataFrame,
    alternative_data_dict: dict,
    carry_df: pd.DataFrame,
    carry_validation_df: pd.DataFrame,
) -> None:
    charts = gate["charts_block"]
    intelligence = gate.get("market_intelligence_block", {})

    def intelligence_frame(key: str, fallback: pd.DataFrame) -> pd.DataFrame:
        value = intelligence.get(key, pd.DataFrame()) if isinstance(intelligence, dict) else pd.DataFrame()
        return value if isinstance(value, pd.DataFrame) and not value.empty else fallback

    latest_frame = intelligence_frame("latest_macro", pd.DataFrame())
    if not latest_frame.empty:
        latest_macro_row = latest_frame.iloc[-1]
    macro_history_df = intelligence_frame("macro_history", macro_history_df)
    rate_curves_df = intelligence_frame(
        "global_yield_curves",
        charts.get("rate_curves", global_rates_df) if isinstance(charts, dict) else global_rates_df,
    )
    global_rate_history_df = intelligence_frame("global_rate_history", global_rate_history_df)
    interbank_df = intelligence_frame("interbank_reference_rates", interbank_df)
    carry_df = intelligence_frame("carry_trade_suggestions", carry_df)
    carry_validation_df = intelligence_frame("carry_trade_validation", carry_validation_df)

    sentiment = dict(sentiment_sem) if isinstance(sentiment_sem, dict) else {}
    for target, source in (
        ("timeline", "sentiment_timeline"),
        ("latest", "sentiment_latest"),
        ("loadings", "sentiment_loadings"),
        ("structural_links", "sentiment_structural_links"),
        ("diagnostics", "sentiment_diagnostics"),
    ):
        value = intelligence_frame(source, sentiment.get(target, pd.DataFrame()))
        sentiment[target] = value

    alternative = dict(alternative_data_dict) if isinstance(alternative_data_dict, dict) else {}
    for target, source in (
        ("forex_factory_calendar", "forex_factory_calendar"),
        ("forex_factory_event_risk", "forex_factory_event_risk"),
        ("summary", "geopolitical_summary"),
        ("gdelt_timeline", "geopolitical_timeline"),
    ):
        alternative[target] = intelligence_frame(source, alternative.get(target, pd.DataFrame()))

    _section_header(
        "Market intelligence",
        "Persisted public-data state: latent sentiment, macro-financial conditions, sovereign curves and event risk.",
    )
    curve_val = latest_macro_row.get(
        "Country_Curve_10Y_2Y",
        latest_macro_row.get("Curve_10Y_2Y", float("nan")),
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rates regime", str(latest_macro_row.get("Regime_Hawkish_Dovish", "—")))
    m2.metric("Market regime", str(latest_macro_row.get("Regime_Bull_Bear", "—")))
    m3.metric("10Y–2Y curve", _fmt_num(curve_val, digits=2))
    m4.metric("Credit spread", _fmt_num(latest_macro_row.get("CREDIT_SPREAD", float("nan")), digits=2))
    m5.metric(
        "Stress probability",
        _fmt_pct(latest_macro_row.get("Markov_Stress_Prob", float("nan")), digits=0),
    )

    sentiment_timeline = sentiment.get("timeline", pd.DataFrame())
    sentiment_latest = sentiment.get("latest", pd.DataFrame())
    sentiment_loadings = sentiment.get("loadings", pd.DataFrame())
    sentiment_links = sentiment.get("structural_links", pd.DataFrame())
    sentiment_diagnostics = sentiment.get("diagnostics", pd.DataFrame())
    _section_header(
        "Latent market sentiment",
        r"Single-factor SEM estimated causally from momentum, breadth, volatility, credit, rates and event-risk indicators.",
    )
    if isinstance(sentiment_timeline, pd.DataFrame) and not sentiment_timeline.empty:
        if isinstance(sentiment_latest, pd.DataFrame) and not sentiment_latest.empty:
            row = sentiment_latest.iloc[-1]
            s1, s2, s3 = st.columns(3)
            s1.metric("Latent state", _fmt_num(row.get("Latent_Market_Sentiment_SEM"), digits=2))
            s2.metric("Risk-on probability", _fmt_pct(row.get("Sentiment_Prob_Risk_On")))
            s3.metric("Classification", str(row.get("Sentiment_State", "—")))
        sem_fig = _plotly_sentiment_sem(sentiment_timeline)
        if sem_fig is not None:
            st.plotly_chart(sem_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        with st.expander("SEM loadings, structural link and formal specification", expanded=False):
            st.latex(r"x_t=\Lambda\eta_t+\varepsilon_t")
            st.latex(r"r_{\xi,t+1}=\alpha+\beta_{\eta}\eta_t+\beta_f^\top f_t+u_{t+1}")
            if isinstance(sentiment_loadings, pd.DataFrame) and not sentiment_loadings.empty:
                st.dataframe(sentiment_loadings, width="stretch", hide_index=True)
            if isinstance(sentiment_links, pd.DataFrame) and not sentiment_links.empty:
                st.dataframe(sentiment_links, width="stretch", hide_index=True)
            if isinstance(sentiment_diagnostics, pd.DataFrame) and not sentiment_diagnostics.empty:
                st.dataframe(sentiment_diagnostics, width="stretch", hide_index=True)
    else:
        _empty_state(
            "Latent sentiment is unavailable in the current artifact.",
            "The next rigorous daily refresh will persist the SEM timeline and diagnostics.",
        )

    _section_header(
        "Macro-financial state",
        "Rolling causal z-scores preserve scale comparability without treating levels as commensurate.",
    )
    macro_fig = _plotly_macro_history(macro_history_df)
    if macro_fig is not None:
        st.plotly_chart(macro_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
    elif isinstance(macro_history_df, pd.DataFrame) and not macro_history_df.empty:
        st.dataframe(macro_history_df.tail(24), width="stretch", hide_index=False)
    else:
        _empty_state("Macro history is unavailable in the current artifact.")

    _section_header("Global sovereign curves", "Policy, short-end and 10Y observations retain source-specific discrete timing.")
    if isinstance(rate_curves_df, pd.DataFrame) and not rate_curves_df.empty:
        fig = _plotly_sovereign_curves(rate_curves_df)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        show_cols = [c for c in [
            "Country", "Policy_Rate", "Yield_2Y", "Yield_10Y", "Curve_10Y_2Y",
            "Term_Premium_Proxy", "Regime_Hawkish_Dovish", "Curve_Shape", "Rate_Source",
        ] if c in rate_curves_df.columns]
        if show_cols:
            st.dataframe(rate_curves_df[show_cols], width="stretch", hide_index=True)
    else:
        _empty_state("Sovereign curve snapshot not loaded.")

    if isinstance(global_rate_history_df, pd.DataFrame) and not global_rate_history_df.empty:
        available_countries = sorted(global_rate_history_df["Country"].dropna().astype(str).unique().tolist())
        preferred = [
            country for country in
            ["United States", "Mexico", "Canada", "Brazil", "United Kingdom", "Germany", "Japan", "China"]
            if country in available_countries
        ]
        selected_countries = st.multiselect(
            "Countries shown in discrete 10Y history",
            available_countries,
            default=preferred or available_countries[:6],
            max_selections=10,
        )
        history_fig = _plotly_global_rate_history(global_rate_history_df, selected_countries)
        if history_fig is not None:
            st.plotly_chart(history_fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        with st.expander("Latest observations by country and tenor", expanded=False):
            st.dataframe(
                latest_rate_observations(global_rate_history_df),
                width="stretch",
                hide_index=True,
            )

    _section_header("Overnight reference rates", "SOFR, SONIA, ESTR and TONAR replace discontinued LIBOR references.")
    if isinstance(interbank_df, pd.DataFrame) and not interbank_df.empty:
        fig = _plotly_interbank_rates(interbank_df)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        latest_interbank = (
            interbank_df.assign(
                Observation_Date=pd.to_datetime(interbank_df["Observation_Date"], errors="coerce")
            )
            .sort_values("Observation_Date")
            .groupby("Benchmark", as_index=False)
            .tail(1)
        )
        st.dataframe(latest_interbank, width="stretch", hide_index=True)
    else:
        _empty_state("Overnight reference rates not available.")

    event_risk = alternative.get("forex_factory_event_risk", pd.DataFrame())
    calendar = alternative.get("forex_factory_calendar", pd.DataFrame())
    _section_header("Scheduled macro event risk", "Public calendar evidence in Central Time; event risk is not directional alpha.")
    if isinstance(event_risk, pd.DataFrame) and not event_risk.empty:
        st.dataframe(event_risk, width="stretch", hide_index=True)
    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        st.dataframe(calendar.head(60), width="stretch", hide_index=True)
    if (
        not isinstance(event_risk, pd.DataFrame) or event_risk.empty
    ) and (
        not isinstance(calendar, pd.DataFrame) or calendar.empty
    ):
        _empty_state("No scheduled macro-event calendar was persisted.")

    _section_header("Carry research", "Rate differentials are filtered by event risk; FX basis, hedge cost and funding liquidity remain explicit caveats.")
    if isinstance(carry_df, pd.DataFrame) and not carry_df.empty:
        st.dataframe(carry_df.head(30), width="stretch", hide_index=True)
        if isinstance(carry_validation_df, pd.DataFrame) and not carry_validation_df.empty:
            with st.expander("Carry validation and no-arbitrage checks", expanded=False):
                st.dataframe(carry_validation_df, width="stretch", hide_index=True)
    else:
        _empty_state("No admissible carry candidates in the persisted state.")

    geopolitical_summary = alternative.get("summary", pd.DataFrame())
    geopolitical_timeline = alternative.get("gdelt_timeline", pd.DataFrame())
    _section_header(
        "Geopolitical attention monitor",
        "Within-topic abnormal attention only; raw cross-topic news counts are not interpreted as comparable probabilities.",
    )
    if isinstance(geopolitical_summary, pd.DataFrame) and not geopolitical_summary.empty:
        st.dataframe(geopolitical_summary, width="stretch", hide_index=True)
        if isinstance(geopolitical_timeline, pd.DataFrame) and not geopolitical_timeline.empty:
            with st.expander("GDELT timeline evidence", expanded=False):
                st.dataframe(geopolitical_timeline.tail(200), width="stretch", hide_index=True)
    else:
        _empty_state("No statistically admissible geopolitical signal was persisted.")


# ------------------------------------------------------------
# Render: Options Market
# ------------------------------------------------------------

def render_options(gate: dict, *, options_summary_df: pd.DataFrame, options_chain_df: pd.DataFrame,
                   vol_surface_diag_df: pd.DataFrame) -> None:
    charts = gate["charts_block"]
    surface_df = charts.get("options_surface", pd.DataFrame()) if isinstance(charts, dict) else pd.DataFrame()
    _banner(
        "info",
        "Current snapshot only —",
        "not historical backtest evidence. Options Greeks and IV come from the latest public Yahoo snapshot.",
    )

    _section_header("Portfolio implied vol surface")
    if isinstance(surface_df, pd.DataFrame) and not surface_df.empty:
        fig = _plotly_vol_surface(surface_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
        else:
            _empty_state("Vol surface chart could not be rendered.")
    else:
        _empty_state("No implied vol surface for current snapshot.")

    if isinstance(options_summary_df, pd.DataFrame) and not options_summary_df.empty:
        _section_header("Options summary by ticker")
        show_cols = [c for c in [
            "Ticker", "Expiry", "DTE", "ATM_IV", "Skew", "Put_OI", "Call_OI", "Bid", "Ask", "Mid"
        ] if c in options_summary_df.columns]
        st.dataframe(options_summary_df[show_cols] if show_cols else options_summary_df,
                     use_container_width=True, hide_index=True)

    if isinstance(options_chain_df, pd.DataFrame) and not options_chain_df.empty:
        with st.expander("Options chain snapshot", expanded=False):
            st.dataframe(options_chain_df.head(500), use_container_width=True, hide_index=True)

    if isinstance(vol_surface_diag_df, pd.DataFrame) and not vol_surface_diag_df.empty:
        with st.expander("Surface diagnostics", expanded=False):
            st.dataframe(vol_surface_diag_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Render: Fundamentals
# ------------------------------------------------------------

def render_fundamentals(gate: dict) -> None:
    tables = gate["tables_block"]
    fundamentals_df = tables.get("fundamentals", pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame()
    if not isinstance(fundamentals_df, pd.DataFrame) or fundamentals_df.empty:
        _empty_state("No fundamentals exposed for this run.")
        return

    columns = [c for c in [
        "Ticker", "Sector", "Weight", "PIT_Confidence",
        "Revenue_Growth", "EPS_Growth", "Gross_Margin", "EBIT_Margin", "FCF_Margin",
        "ROE", "ROIC", "Net_Debt_EBITDA", "Interest_Coverage",
        "Sector_Zscore", "Mahalanobis", "SEC_Period_Type", "SEC_Accepted_At",
        "PIT_Source_Class", "Availability_Date", "Missing_Reason",
    ] if c in fundamentals_df.columns]

    column_config = {}
    if "PIT_Confidence" in columns:
        column_config["PIT_Confidence"] = st.column_config.NumberColumn("PIT confidence", format="%.0f%%")
    if "Weight" in columns:
        column_config["Weight"] = st.column_config.NumberColumn("Weight", format="%.2f%%")
    for ratio in ["Revenue_Growth", "EPS_Growth", "Gross_Margin", "EBIT_Margin", "FCF_Margin", "ROE", "ROIC"]:
        if ratio in columns:
            column_config[ratio] = st.column_config.NumberColumn(ratio.replace("_", " "), format="%.2f")
    if "Mahalanobis" in columns:
        column_config["Mahalanobis"] = st.column_config.NumberColumn("Mahalanobis", format="%.2f")

    _section_header(
        "Fundamentals — portfolio selection",
        "Backend-computed point-in-time ratios, sector-relative z-scores, and Mahalanobis distance."
    )
    st.dataframe(fundamentals_df[columns] if columns else fundamentals_df,
                 use_container_width=True, hide_index=True, column_config=column_config)

    if "PIT_Source_Class" in fundamentals_df.columns:
        with st.expander("Source class distribution", expanded=False):
            st.dataframe(
                fundamentals_df["PIT_Source_Class"].value_counts(dropna=False).rename_axis("Source class").reset_index(name="Count"),
                use_container_width=True, hide_index=True,
            )


# ------------------------------------------------------------
# Render: Data Freshness
# ------------------------------------------------------------

def render_data_freshness(gate: dict, *, timings_df: pd.DataFrame) -> None:
    fresh = gate["freshness"]
    if isinstance(fresh, pd.DataFrame) and not fresh.empty:
        _section_header("Data freshness", "Cache age, TTL and status by data source. Central Time.")

        show_cols = [c for c in [
            "Source", "Dataset", "Last_Update_Central", "Age_Hours", "TTL_Hours",
            "Status", "Rows", "Errors", "Fallback"
        ] if c in fresh.columns]
        view = (fresh[show_cols] if show_cols else fresh).copy()

        # Status summary chips (R7)
        if "Status" in view.columns:
            status_counts = view["Status"].astype(str).str.lower().value_counts()
            chips = []
            for key in ("valid", "stale", "fallback", "error"):
                if key in status_counts.index:
                    chips.append(_status_pill(f"{status_counts[key]} {key}", key))
            if chips:
                st.markdown(
                    '<div role="region" aria-label="Data freshness summary" '
                    'style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;">'
                    + "".join(chips)
                    + "</div>",
                    unsafe_allow_html=True,
                )

        column_config = {}
        if "Age_Hours" in view.columns:
            column_config["Age_Hours"] = st.column_config.NumberColumn("Age (h)", format="%.1f")
        if "TTL_Hours" in view.columns:
            column_config["TTL_Hours"] = st.column_config.NumberColumn("TTL (h)", format="%.0f")

        # Styler-driven row coloring by Status (R7)
        try:
            def _row_style(row: pd.Series) -> list[str]:
                status_val = str(row.get("Status", "")).lower() if "Status" in row.index else ""
                bg = {
                    "valid": "rgba(74,222,128,0.10)",
                    "stale": "rgba(251,191,36,0.10)",
                    "fallback": "rgba(251,191,36,0.10)",
                    "error": "rgba(248,113,113,0.10)",
                }.get(status_val, "")
                if not bg:
                    return [""] * len(row)
                return [f"background-color: {bg};"] * len(row)
            styler = view.style.apply(_row_style, axis=1)
            st.dataframe(styler, use_container_width=True, hide_index=True, column_config=column_config)
        except Exception:
            st.dataframe(view, use_container_width=True, hide_index=True, column_config=column_config)
    else:
        _empty_state("No data-freshness report exposed for this run.")

    if isinstance(timings_df, pd.DataFrame) and not timings_df.empty:
        _section_header("Pipeline timings", "Per-stage duration of the last run.")
        sorted_t = timings_df.sort_values("Seconds", ascending=False)
        if px is not None:
            fig = px.bar(sorted_t, x="Seconds", y="Stage", orientation="h", text="Seconds")
            fig.update_traces(texttemplate="%{x:.1f}s", textposition="outside",
                              marker_color="#7dd3fc", marker_line_color="rgba(125,211,252,0.5)")
            st.plotly_chart(_plotly_dark_layout(fig, height=max(280, 22 * len(sorted_t) + 80)),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.dataframe(sorted_t, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Render: Advanced Research (hidden by default, behind expanders)
# ------------------------------------------------------------

def render_advanced_research(gate: dict, *, results_dict: dict) -> None:
    tables = gate["tables_block"]
    st.markdown(
        '<div class="small-note" style="margin-bottom:10px;">'
        'Advanced research artefacts are exposed as expanders. They are reproducible audit trails for the current run.'
        '</div>',
        unsafe_allow_html=True,
    )

    opt_grid_df = results_dict.get("optimization_grid", pd.DataFrame())
    with st.expander("Optimization grid (raw)", expanded=False):
        if isinstance(opt_grid_df, pd.DataFrame) and not opt_grid_df.empty:
            st.dataframe(opt_grid_df.head(500), use_container_width=True, hide_index=True)
        else:
            _empty_state("Empty optimization grid.")

    validation_diag_dict = results_dict.get("validation_diagnostics", {}) or {}
    with st.expander("Validation diagnostics (raw)", expanded=False):
        if isinstance(validation_diag_dict, dict) and validation_diag_dict:
            for k, v in validation_diag_dict.items():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    st.markdown(f"**{k}**")
                    st.dataframe(v.head(300), use_container_width=True, hide_index=True)
        else:
            _empty_state("Empty validation diagnostics.")

    factor_attr_df = results_dict.get("factor_attribution", pd.DataFrame())
    oos_attr_df = results_dict.get("oos_factor_attribution", pd.DataFrame())
    with st.expander("Factor attribution (raw)", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**In-sample**")
            if isinstance(factor_attr_df, pd.DataFrame) and not factor_attr_df.empty:
                st.dataframe(factor_attr_df, use_container_width=True, hide_index=True)
            else:
                _empty_state("No in-sample attribution.")
        with a2:
            st.markdown("**Out-of-sample**")
            if isinstance(oos_attr_df, pd.DataFrame) and not oos_attr_df.empty:
                st.dataframe(oos_attr_df, use_container_width=True, hide_index=True)
            else:
                _empty_state("No OOS attribution.")

    return_diag_dict = results_dict.get("return_diagnostics", {}) or {}
    with st.expander("GARCH / variance model selection (raw)", expanded=False):
        var_models_df = return_diag_dict.get("variance_model_selection", pd.DataFrame()) if isinstance(return_diag_dict, dict) else pd.DataFrame()
        if isinstance(var_models_df, pd.DataFrame) and not var_models_df.empty:
            st.dataframe(var_models_df, use_container_width=True, hide_index=True)
        else:
            _empty_state("No variance model selection table.")

    with st.expander("PELT regime tables (raw)", expanded=False):
        for key in ("pelt_regime_segments", "pelt_change_points", "pelt_timeline"):
            df_pelt = return_diag_dict.get(key, pd.DataFrame()) if isinstance(return_diag_dict, dict) else pd.DataFrame()
            if isinstance(df_pelt, pd.DataFrame) and not df_pelt.empty:
                st.markdown(f"**{key}**")
                st.dataframe(df_pelt.head(300), use_container_width=True, hide_index=True)

    cache_inv_df = results_dict.get("cache_inventory", pd.DataFrame())
    with st.expander("Cache inventory (raw)", expanded=False):
        if isinstance(cache_inv_df, pd.DataFrame) and not cache_inv_df.empty:
            st.dataframe(cache_inv_df, use_container_width=True, hide_index=True)
        else:
            _empty_state("Cache empty or disabled.")

    model_registry_df = results_dict.get("model_registry", pd.DataFrame())
    with st.expander("Model registry (current run)", expanded=False):
        if isinstance(model_registry_df, pd.DataFrame) and not model_registry_df.empty:
            st.dataframe(model_registry_df, use_container_width=True, hide_index=True)
        else:
            _empty_state("Model registry not produced.")

    kaizen_diag_dict = results_dict.get("kaizen_diagnostics", {}) or {}
    with st.expander("Kaizen diagnostics (raw)", expanded=False):
        if isinstance(kaizen_diag_dict, dict) and kaizen_diag_dict:
            for k, v in kaizen_diag_dict.items():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    st.markdown(f"**{k}**")
                    st.dataframe(v.head(200), use_container_width=True, hide_index=True)
        else:
            _empty_state("No Kaizen diagnostics for this run.")

    rejection_df = tables.get("rejections", pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame()
    with st.expander("Rejection list (filtered-out assets)", expanded=False):
        if isinstance(rejection_df, pd.DataFrame) and not rejection_df.empty:
            st.dataframe(rejection_df, use_container_width=True, hide_index=True)
        else:
            _empty_state("No rejections registered.")


# ============================================================
# Dispatch
# ============================================================

valid_tickers_count = int(results["prices"].shape[1]) if isinstance(results.get("prices"), pd.DataFrame) else 0
if valid_tickers_count == 0 and isinstance(results.get("portfolio"), pd.DataFrame):
    valid_tickers_count = int(results["portfolio"].shape[0])
last_run_at_str = str(st.session_state.get("last_run_at", "n/a"))

# Build fixed-weight ticker list (used to badge FIXED rows in private sleeve)
_fixed_tickers_list: list[str] = []
try:
    if side_boom_fixed_ticker:
        _fixed_tickers_list.append(str(side_boom_fixed_ticker).upper())
    for tk, _w in (side_boom_extra_fixed_weights or ()):
        if tk:
            _fixed_tickers_list.append(str(tk).upper())
except NameError:
    pass
fixed_tickers_tuple = tuple(dict.fromkeys(_fixed_tickers_list))

# Deep-link: read the requested section from URL query params (R9)
ALL_SECTION_LABELS = [
    "Overview", "Allocation", "My Portfolio", "Research", "Performance", "Risk",
    "Validation", "Market Intelligence", "Options", "Fundamentals",
    "Data Freshness", "Advanced",
]
ALL_SECTION_SLUGS = [
    "overview", "allocation", "my-portfolio", "private-alpha", "price-path", "risk",
    "validation", "market-regime", "options", "fundamentals",
    "data-freshness", "advanced",
]

# RBAC: filter sections by the authenticated user's role.
_accessible_slugs = set(filter_accessible_sections(current_user, ALL_SECTION_SLUGS))
SECTION_LABELS = [lbl for lbl, slug in zip(ALL_SECTION_LABELS, ALL_SECTION_SLUGS) if slug in _accessible_slugs]
SECTION_SLUGS = [slug for slug in ALL_SECTION_SLUGS if slug in _accessible_slugs]

if not SECTION_SLUGS:
    st.error("Your role has no accessible sections. Contact the administrator.")
    audit_event("rbac.empty_access", username=current_user.username, role=current_user.role)
    st.stop()

_qp = {}
try:
    _qp = dict(st.query_params)
except Exception:
    try:
        _qp = {k: v[0] if isinstance(v, list) else v for k, v in st.experimental_get_query_params().items()}
    except Exception:
        _qp = {}
requested_section = str(_qp.get("section", SECTION_SLUGS[0])).lower().strip()
if requested_section not in SECTION_SLUGS:
    requested_section = SECTION_SLUGS[0]
st.session_state.setdefault("ui_active_section", requested_section)
if st.session_state["ui_active_section"] not in SECTION_SLUGS:
    st.session_state["ui_active_section"] = SECTION_SLUGS[0]

# Visible navigation with lazy rendering. Unlike st.tabs, only the selected
# workspace executes, while all available analytical surfaces remain visible.
section_label_default = SECTION_LABELS[SECTION_SLUGS.index(st.session_state["ui_active_section"])]
_picked_label = st.pills(
    "Workspace",
    SECTION_LABELS,
    default=section_label_default,
    help="Choose an analytical workspace. Only the selected surface is computed.",
    key="ui_workspace_pills",
    label_visibility="collapsed",
    width="stretch",
)
if _picked_label is None:
    _picked_label = section_label_default
_picked_slug = SECTION_SLUGS[SECTION_LABELS.index(_picked_label)]
if _picked_slug != st.session_state.get("ui_active_section"):
    st.session_state["ui_active_section"] = _picked_slug

# Map slug -> renderer thunk so RBAC removes both the navigation item and the
# renderer call.
def _render_overview():
    render_executive_overview(
        gate_state,
        benchmark_ticker=benchmark_ticker,
        last_run_at=last_run_at_str,
        tickers_count=valid_tickers_count,
        latest_macro_row=latest_macro if isinstance(latest_macro, pd.Series) else pd.Series(dtype=object),
        daily_snapshot_gate=daily_snapshot_gate_state,
        daily_snapshot_created_at=results.get("daily_snapshot_created_at"),
    )

def _render_allocation():
    render_allocation(gate_state)

def _render_my_portfolio():
    render_my_portfolio(current_user.username)

def _render_private():
    render_research_strategy(gate_state)

def _render_price_path():
    render_price_paths(gate_state)

def _render_risk():
    render_risk_diagnostics(
        gate_state,
        return_diagnostics=return_diag if isinstance(return_diag, dict) else {},
        perf_summary_df=perf_summary if isinstance(perf_summary, pd.DataFrame) else pd.DataFrame(),
        forecast_path=gbm_path if isinstance(gbm_path, pd.DataFrame) else pd.DataFrame(),
        forecast_summary=gbm_summary if isinstance(gbm_summary, pd.DataFrame) else pd.DataFrame(),
    )

def _render_validation():
    render_validation(gate_state)

def _render_market_regime():
    render_market_regime(
        gate_state,
        latest_macro_row=latest_macro if isinstance(latest_macro, pd.Series) else pd.Series(dtype=object),
        macro_history_df=macro if isinstance(macro, pd.DataFrame) else pd.DataFrame(),
        sentiment_sem=market_sentiment_sem_results if isinstance(market_sentiment_sem_results, dict) else {},
        global_rates_df=global_yield_curves if isinstance(global_yield_curves, pd.DataFrame) else pd.DataFrame(),
        global_rate_history_df=global_rate_history if isinstance(global_rate_history, pd.DataFrame) else pd.DataFrame(),
        interbank_df=interbank_reference_rates if isinstance(interbank_reference_rates, pd.DataFrame) else pd.DataFrame(),
        alternative_data_dict=alternative_data if isinstance(alternative_data, dict) else {},
        carry_df=carry_trade if isinstance(carry_trade, pd.DataFrame) else pd.DataFrame(),
        carry_validation_df=carry_trade_validation if isinstance(carry_trade_validation, pd.DataFrame) else pd.DataFrame(),
    )

def _render_options():
    render_options(
        gate_state,
        options_summary_df=options_summary if isinstance(options_summary, pd.DataFrame) else pd.DataFrame(),
        options_chain_df=options_chain if isinstance(options_chain, pd.DataFrame) else pd.DataFrame(),
        vol_surface_diag_df=portfolio_vol_surface_diagnostics if isinstance(portfolio_vol_surface_diagnostics, pd.DataFrame) else pd.DataFrame(),
    )

def _render_fundamentals():
    render_fundamentals(gate_state)

def _render_freshness():
    render_data_freshness(gate_state, timings_df=timings if isinstance(timings, pd.DataFrame) else pd.DataFrame())

def _render_advanced():
    render_advanced_research(gate_state, results_dict=results)

_RENDERERS_BY_SLUG = {
    "overview": _render_overview,
    "allocation": _render_allocation,
    "my-portfolio": _render_my_portfolio,
    "private-alpha": _render_private,
    "price-path": _render_price_path,
    "risk": _render_risk,
    "validation": _render_validation,
    "market-regime": _render_market_regime,
    "options": _render_options,
    "fundamentals": _render_fundamentals,
    "data-freshness": _render_freshness,
    "advanced": _render_advanced,
}

active_renderer = _RENDERERS_BY_SLUG.get(_picked_slug)
if active_renderer is not None:
    active_renderer()


st.markdown(
    '<p class="small-note" style="margin-top:24px;">'
    f'Build {APP_BUILD_ID} · '
    'Implementation note: this frontend is render-only. All analytics, validation tests, gates and risk metrics '
    'are produced by the backend pipeline. Yahoo Finance fundamentals are a public approximation; vendor-grade '
    'point-in-time data is recommended for production deployments.'
    '</p>',
    unsafe_allow_html=True,
)

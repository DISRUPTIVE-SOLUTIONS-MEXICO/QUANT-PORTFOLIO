import gzip
import json
from pathlib import Path

APP_SOURCE = Path(__file__).resolve().parents[1] / "stockpicker_app.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return APP_SOURCE.read_text(encoding="utf-8")


def test_workspace_renders_only_active_section():
    source = _source()
    assert "section = st.tabs(" not in source
    assert "_picked_label = st.pills(" in source
    assert "active_renderer = _RENDERERS_BY_SLUG.get(_picked_slug)" in source
    assert "active_renderer()" in source


def test_persisted_dashboard_hydrates_before_live_preflight():
    source = _source()
    hydrate = source.index("startup_results = _load_precomputed_dashboard_results(benchmark_ticker)")
    live_request = source.index("live_preflight_requested = (")
    preflight_call = source.index("preflight_market = cached_preflight_market(", live_request)
    assert hydrate < live_request < preflight_call


def test_mobile_layout_stacks_non_metric_columns():
    source = _source()
    assert '[data-testid="stHorizontalBlock"] > [data-testid="column"]:has(div[data-testid="stMetric"])' in source
    assert "flex: 1 1 100% !important;" in source


def test_collapsed_sidebar_releases_desktop_layout_width():
    source = _source()
    selector = 'section[data-testid="stSidebar"][aria-expanded="false"]'
    collapsed_start = source.index(selector)
    collapsed_end = source.index("}", collapsed_start)
    collapsed_rule = source[collapsed_start:collapsed_end]
    assert "width: 0 !important;" in collapsed_rule
    assert "flex: 0 0 0 !important;" in collapsed_rule
    assert '[data-testid="stMain"] {' in source
    assert "flex: 1 1 auto !important;" in source


def test_overview_leads_with_research_strategy_vs_its_own_benchmark():
    source = _source()
    # First-instance contract: the dashboard headline is the governed research
    # strategy measured against the benchmark xi the research selected (e.g.
    # USMV), never the daily SPY proxy.
    assert "def render_research_headline" in source
    headline_def = source.index("def render_research_headline")
    overview_def = source.index("def render_executive_overview")
    assert headline_def < overview_def
    overview_body = source[overview_def:]
    call_pos = overview_body.index("render_research_headline()")
    snapshot_branch = overview_body.index("if is_snapshot:")
    assert call_pos < snapshot_branch, "research headline must render before the SPY snapshot proxy"
    # The SPY-relative snapshot is explicitly labeled as a proxy market monitor.
    assert "Daily market snapshot (proxy)" in source
    assert "not the research strategy" in source
    # Headline metrics are xi-relative.
    assert "Benchmark ξ: {xi}" in source or "Benchmark ξ" in source
    # The SPY proxy never renders in the open: both the snapshot block and the
    # market-pulse fallback live inside collapsed expanders. Legacy
    # Sortino-branded series from stale persisted payloads are preserved as
    # evidence but canonicalized in the presentation layer.
    assert source.count("Market monitor - daily proxy vs") == 2
    assert "def _strip_legacy_proxy_series" in source
    assert "def _canonical_series_label" in source
    assert 'return "XCDR research portfolio price"' in source


def test_supabase_json_tables_are_restored_for_rendering():
    source = _source()
    assert "def _payload_frame(value) -> pd.DataFrame:" in source
    # Formatter-resilient: assert the restore calls exist without pinning
    # the exact line wrapping around the assignment.
    assert '_payload_frame(charts.get("price_paths"))' in source
    assert '_payload_frame(allocation.get("recommended_portfolio"))' in source
    assert "payload_requires_restore = any(" in source
    assert 'st.session_state["results"] = restored_results' in source


def test_daily_snapshot_uses_real_metrics_instead_of_empty_full_run_cards():
    source = _source()
    assert "if is_snapshot:" in source
    assert '"Proxy return"' in source
    assert '"Active return"' in source
    assert '"Daily CVaR 95%"' in source
    assert '"Market breadth"' in source
    assert '"Full analytics are never inferred from missing snapshot fields."' in source


def test_workspace_change_does_not_trigger_query_parameter_rerun():
    source = _source()
    workspace_start = source.index("_picked_label = st.pills(")
    renderer_start = source.index("# Map slug -> renderer thunk", workspace_start)
    workspace_block = source[workspace_start:renderer_start]
    assert 'st.query_params["section"]' not in workspace_block
    assert "st.experimental_set_query_params" not in workspace_block


def test_snapshot_overview_is_an_analytical_command_center():
    source = _source()
    assert '"Portfolio vs benchmark"' in source
    assert '"Risk-return decomposition"' in source
    assert "Formal definitions and evidence scope" in source
    assert "snapshot_slugs =" not in source
    assert '"Rates, Macro & Geo"' in source
    assert "qpk-command-grid" in source
    assert 'initial_sidebar_state="collapsed"' in source


def test_personal_portfolio_workspace_is_versioned_and_accessible():
    source = _source()
    assert '"My Portfolio"' in source
    assert '"my-portfolio": _render_my_portfolio' in source
    assert "save_run_to_supabase(" in source
    assert 'status="user_completed"' in source


def test_research_candidate_replaces_sortino_series_in_market_pulse():
    source = _source()
    assert "def _research_chart_frames(" in source
    assert '"XCDR/XODR synthetic strategy price"' in source
    assert '"Governed research pulse"' in source
    assert '"Sortino optimized synthetic NAV price"' not in source


def test_chart_layout_reserves_separate_legend_and_axis_space():
    source = _source()
    assert "bottom_margin = 74" in source
    assert "top_margin = int(min(168, 86 + 18 * legend_rows + (18 if title else 0)))" in source
    assert "legend_layout = dict(" in source
    assert "yanchor=\"bottom\"" in source
    assert "y=1.02" in source
    assert "margin=dict(l=72, r=34, t=top_margin, b=bottom_margin)" in source
    assert 'itemwidth=42' not in source
    assert 'y=-0.30' not in source
    assert 'y=-0.18' not in source
    assert "compact_legend" not in source
    assert 'fig.update_xaxes(title_text="")' in source
    assert 'fig.update_yaxes(title_text="")' in source
    assert "nticks=6" in source
    assert 'title_text=""' in source
    assert 'hovermode="x unified"' in source
    assert 'title="XCDR/XODR candidate and optimal benchmark xi"' in source
    assert 'title="OOS drawdown from running maximum"' in source


def test_market_intelligence_restores_full_persisted_contract():
    source = _source()
    assert 'normalized_payload["market_intelligence"] = restored_market_intelligence' in source
    assert 'normalized_payload["strategy_lab"] = restored_strategy_lab' in source
    assert 'normalized_payload["fixed_income_intelligence"] = restored_fixed_income' in source
    assert 'results["daily_strategy_lab"] = daily_results.get("strategy_lab", {})' in source
    assert "def _render_strategy_lab_payload(" in source
    assert "def _plotly_selected_curve_from_snapshot(" in source
    assert "def _plotly_sentiment_sem(" in source
    assert "def _plotly_global_rate_history(" in source
    assert '"Latent market sentiment"' in source
    assert '"Global sovereign comparison"' in source
    assert '"Scheduled macro event risk"' in source
    assert 'APP_BUILD_ID = "2026.06.12-institutional-terminal-ux-v12"' in source
    assert "persisted_market_intelligence_missing = bool(" in source
    assert "Backfill only missing analytical surfaces" in source
    assert '"Repair missing intelligence"' in source
    assert "Overlay daily market intelligence onto the latest full analysis" in source
    assert "never replace the full portfolio, fundamentals, validation or gates" in source


def test_missing_market_intelligence_never_blocks_the_persisted_dashboard():
    source = _source()
    live_start = source.index("live_preflight_requested = (")
    live_end = source.index(")", live_start)
    live_block = source[live_start:live_end]
    assert "persisted_market_intelligence_missing" not in live_block
    assert "The dashboard will not block on public APIs" in source


def test_xcdr_research_gate_overrides_stale_objective_promotion():
    source = _source()
    assert 'config.weight_objective == "xcdr_v3"' in source
    assert 'gate_state["allocation_state"] = "research_only"' in source
    assert 'gate_state["promotion_status"] = "research-only"' in source
    assert "WRC, Hansen SPA, PBO, ICIR and downside-preservation tests" in source


def test_public_dashboard_uses_governed_research_and_observed_prices():
    source = _source()
    assert "Governed research strategy vs optimal benchmark" in source
    assert '"Observed adjusted price"' in source
    assert '"Price index (base=100)"' not in source
    preflight_start = source.index("def render_preflight_market(")
    preflight_end = source.index("def render_market_regime(", preflight_start)
    preflight = source[preflight_start:preflight_end]
    assert "Private Side Alpha vs benchmark" not in preflight


def test_minimum_three_year_history_is_a_hard_dashboard_contract():
    source = _source()
    assert "MIN_PORTFOLIO_HISTORY_YEARS = 3" in source
    assert "MIN_PORTFOLIO_HISTORY_OBS = 720" in source
    assert '"Three-year OOS requirement not met."' in source


def test_us_yield_curve_requires_multiple_real_tenors():
    source = _source()
    for tenor in ["US3M", "US6M", "US1Y", "US5Y", "US7Y", "US20Y", "US30Y"]:
        assert tenor in source
    assert "a yield curve requires at least two maturities" in source
    assert "def _plotly_selected_sovereign_curve" in source
    assert "Formal curve construction" in source
    assert "does not manufacture missing yields" in source
    assert "def _fmt_bps" in source


def test_public_seed_dashboard_prevents_empty_hosted_first_paint():
    source = _source()
    assert "PUBLIC_DASHBOARD_ARTIFACT_DIR" in source
    assert "def _latest_public_seed_dashboard_artifacts" in source
    assert "def _seed_dashboard_results" in source
    assert "return _seed_dashboard_results(benchmark, seed_bundle)" in source
    assert "latest_full_dashboard_payload.seed.json.gz" in source
    assert "latest_daily_dashboard_payload.seed.json.gz" in source
    assert "public_seed_artifact" in source
    assert "def _artifact_has_full_research_contract" in source

    full_seed = PROJECT_ROOT / "public_artifacts" / "latest_full_dashboard_payload.seed.json.gz"
    daily_seed = PROJECT_ROOT / "public_artifacts" / "latest_daily_dashboard_payload.seed.json.gz"
    assert full_seed.exists()
    assert daily_seed.exists()

    with gzip.open(full_seed, "rt", encoding="utf-8") as fh:
        artifact = json.load(fh)
    payload = artifact.get("dashboard_payload", {})
    assert artifact.get("public_seed") is True
    assert artifact.get("scope") == "full_analysis"
    assert payload.get("fixed_income_intelligence", {}).get("country_metrics")
    assert payload.get("market_intelligence", {}).get("sentiment_timeline")
    assert payload.get("allocation", {}).get("side_sleeve") is None
    assert "private side alpha" not in json.dumps(artifact).lower()
    assert "service_role" not in json.dumps(artifact).lower()

    with gzip.open(daily_seed, "rt", encoding="utf-8") as fh:
        daily_artifact = json.load(fh)
    daily_payload = daily_artifact.get("dashboard_payload", {})
    assert daily_artifact.get("scope") == "daily_snapshot"
    assert daily_payload.get("strategy_lab", {}).get("oos_price_paths")
    assert daily_payload.get("strategy_lab", {}).get("weights")

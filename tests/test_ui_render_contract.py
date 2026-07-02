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
    assert "WORKSPACE_COMMAND_DECK" in source
    assert "def _render_workspace_command_deck(" in source
    assert "_render_workspace_command_deck()" in source
    assert 'initial_sidebar_state="collapsed"' in source
    assert source.count('<div class="qpk-hero">') == 1
    assert 'div[data-testid="stElementContainer"]:has(.qpk-hero)' in source
    assert "overflow: hidden !important;" in source


def test_product_hero_paints_before_sidebar_and_heavy_preflight():
    source = _source()
    hero_call = source.index("render_product_hero()")
    sidebar = source.index("with st.sidebar:")
    hydrate = source.index("startup_results = _load_precomputed_dashboard_results(benchmark_ticker)")
    preflight = source.index("live_preflight_requested = (")
    assert hero_call < sidebar < hydrate < preflight


def test_live_preflight_refresh_flags_are_one_shot():
    source = _source()
    start = source.index("_live_global_rates_requested =")
    end = source.index("if run_button:", start)
    block = source[start:end]
    assert "_live_global_rates_requested = bool(st.session_state.get(\"load_global_rates\"))" in block
    assert "_live_geopolitical_requested = bool(st.session_state.get(\"load_geopolitical_thermometer\"))" in block
    assert "finally:" in block
    assert 'st.session_state["load_global_rates"] = False' in block
    assert 'st.session_state["load_geopolitical_thermometer"] = False' in block


def test_command_deck_exposes_all_core_workspaces_without_manual_duplication():
    source = _source()
    start = source.index("WORKSPACE_COMMAND_DECK")
    end = source.index("def _render_workspace_command_deck", start)
    deck = source[start:end]
    for slug in [
        "allocation",
        "my-portfolio",
        "private-alpha",
        "price-path",
        "risk",
        "validation",
        "market-regime",
        "options",
        "fundamentals",
        "data-freshness",
    ]:
        assert f'"slug": "{slug}"' in deck
    render_start = source.index("def _render_workspace_command_deck(")
    render_end = source.index("def _canonical_series_label", render_start)
    render_block = source[render_start:render_end]
    assert "SECTION_SLUGS" in render_block
    assert 'with st.expander("Workspace map", expanded=False):' in render_block
    assert "does not duplicate" in render_block
    assert "aria-label=\"Institutional terminal workspaces\"" in render_block


def test_command_center_exposes_institutional_capability_map():
    source = _source()
    assert "FEATURE_PRESERVATION_MANIFEST" in source
    assert "feature_preservation_manifest_frame" in source
    assert "def _institutional_module_items(" in source
    assert "def _render_institutional_module_map(" in source
    assert "_render_institutional_module_map(gate, results)" in source
    assert "qpk-terminal-map" in source
    assert "qpk-module-tile" in source
    assert "data-capability-id" in source
    for field in [
        "owner_layer",
        "canonical_calculation",
        "canonical_artifact",
        "primary_view",
        "secondary_views",
        "freshness_requirement",
        "validation_status",
    ]:
        assert field in source
    for capability_id in [
        "market_intelligence",
        "rates_fixed_income",
        "equity_fundamentals",
        "options_volatility",
        "portfolio_construction",
        "xcdr_research",
        "risk_laboratory",
        "validation_governance",
    ]:
        assert capability_id in source
    for label in [
        "Market Intelligence",
        "Rates & Fixed Income",
        "Equity Fundamentals",
        "Options & Volatility",
        "Portfolio Construction",
        "XCDR Research",
        "Risk Laboratory",
        "Validation & Governance",
    ]:
        assert label in source
    assert "Feature preservation contract" in source
    assert "Missing modules show as missing rather than being hidden" in source


def test_data_freshness_exposes_feature_preservation_manifest():
    source = _source()
    start = source.index("def render_data_freshness(")
    end = source.index("# ------------------------------------------------------------", start)
    block = source[start:end]
    assert "Feature preservation manifest" in block
    assert "Canonical product contract" in block
    assert "feature_preservation_manifest_frame()" in block
    assert "daily overlay or a thin live fallback" in block


def test_command_center_translates_evidence_into_decision_brief():
    source = _source()
    assert "def _render_decision_brief(" in source
    assert "_render_decision_brief(gate, results, benchmark_ticker)" in source
    assert "Investment command brief" in source
    assert "What the evidence currently supports" in source
    assert "qpk-decision-panel" in source
    assert "Decision intelligence matrix" in source
    assert "Research edge is visible, but promotion gates still dominate the decision." in source
    assert "w_t = pi(F_t); R_(p,t+1)=w_t^T r_(t+1)-TC_t" in source
    assert "Next best operational step" in source
    assert "Render-only · no recommendation · gates remain binding" in source
    for label in ["Posture", "XCDR vs ξ", "Risk brake", "Macro overlay", "Coverage"]:
        assert label in source
    assert "qpk-insight-grid" in source
    assert "qpk-insight-card" in source
    assert "board-level translation of persisted research artifacts" in source
    assert "Benchmark contract: ξ = {xi}" in source
    assert "never overrides suitability, liquidity, WRC, SPA, PBO, CVaR or drawdown gates" in source


def test_streamlit_stale_frames_are_hidden_for_first_fold_surfaces():
    source = _source()
    for selector in [
        ".stale:has(.qpk-hero)",
        ".stale:has(.qpk-ops-strip)",
        ".stale:has(.qpk-decision-panel)",
        ".stale:has(.qpk-insight-grid)",
        ".stale:has(.qpk-market-ribbon)",
        ".stale:has(.qpk-terminal-map)",
    ]:
        assert selector in source
    assert 'div[data-testid="stElementContainer"][style*="opacity"]:has(.qpk-hero)' in source
    assert 'div[data-testid="stElementContainer"][style*="opacity"]:has(.qpk-market-ribbon)' in source
    assert "div[data-testid=\"stElementContainer\"]:has(.qpk-decision-panel)" in source
    assert "display: none !important;" in source


def test_compact_research_headline_uses_lightweight_artifact_scope():
    source = _source()
    start = source.index("def _render_strategy_lab_payload(")
    end = source.index('with st.expander("Formal construction of benchmark', start)
    block = source[start:end]
    compact_start = block.index("if compact:")
    plot_start = block.index("if not price_paths.empty:")
    assert compact_start < plot_start
    compact_block = block[compact_start:plot_start]
    assert "Research artifact scope" in compact_block
    assert "Open Price & Drawdown or XCDR Research for full charts" in compact_block
    assert "st.plotly_chart" not in compact_block
    assert "not a Sortino proxy or private side sleeve" in compact_block


def test_command_center_orders_decision_before_inventory_surfaces():
    source = _source()
    start = source.index("def render_executive_overview(")
    end = source.index("# ------------------------------------------------------------", start)
    block = source[start:end]
    assert block.index("render_research_headline()") < block.index("_render_decision_brief(")
    assert block.index("_render_decision_brief(") < block.index("_render_market_intelligence_tape(")
    assert block.index("_render_market_intelligence_tape(") < block.index("_render_workspace_command_deck()")
    assert block.index("_render_workspace_command_deck()") < block.index("_render_institutional_module_map(")


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


def test_weight_renderer_tolerates_artifact_weight_aliases():
    source = _source()
    helper_start = source.index("def _with_weight_percent(")
    helper_end = source.index("def _status_pill", helper_start)
    helper = source[helper_start:helper_end]
    for alias in [
        '"Weight_Pct"',
        '"weight_pct"',
        '"portfolio_weight"',
        '"Portfolio_Weight"',
        '"allocation"',
        '"Allocation"',
    ]:
        assert alias in helper
    overview_start = source.index("def render_executive_overview(")
    overview_end = source.index("# ------------------------------------------------------------", overview_start)
    overview = source[overview_start:overview_end]
    assert 'if "Weight_Pct" in weights.columns and weight_cols:' in overview
    assert 'sort_values("Weight_Pct", ascending=False)' in overview


def test_chart_layout_reserves_separate_legend_and_axis_space():
    source = _source()
    assert "bottom_margin = 58" in source
    assert "top_margin = int(min(220, 96 + 24 * legend_rows + (18 if title else 0)))" in source
    assert "legend_layout = dict(" in source
    assert "yanchor=\"bottom\"" in source
    assert "y=1.08" in source
    assert "margin=dict(l=74, r=34, t=top_margin, b=bottom_margin)" in source
    assert "height=max(int(height), 320)" in source
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


def test_terminal_navigation_is_sticky_and_auto_fitting():
    source = _source()
    assert "repeat(auto-fit, minmax(210px, 1fr))" in source
    assert ".qpk-command-grid { grid-template-columns: repeat(3" not in source
    assert ".qpk-command-grid { grid-template-columns: repeat(2" not in source
    assert "position: sticky;" in source
    assert "top: 2.75rem;" in source
    assert "backdrop-filter: blur(14px);" in source
    assert ".qpk-command-link:active" in source


def test_streamlit_stale_frames_are_hidden_to_prevent_duplicate_dashboard_flash():
    source = _source()
    assert '[data-stale="true"]' in source
    assert '[stale-data="true"]' in source
    assert '[data-testid="staleElement"]' in source
    assert '[class*="stale"]' in source
    assert "duplicated dashboards" in source
    assert "contain: layout paint;" in source
    assert "min-height: 116px;" in source


def test_sidebar_uses_progressive_disclosure_for_investor_profile():
    source = _source()
    start = source.index("with st.sidebar:")
    end = source.index("manual_tickers = parse_tickers", start)
    sidebar = source[start:end]
    assert 'with st.expander("Investor risk profile", expanded=False):' in sidebar
    assert "Suitability maps horizon, capital, liquidity and loss tolerance" in sidebar
    assert "qpk-sidebar-summary" in source
    assert "aria-label=\"Suitability summary\"" in sidebar
    assert "Start with the mandate" in sidebar
    for label in [
        '"Horizon"',
        '"Initial capital"',
        '"Monthly contribution"',
        '"Liquidity need"',
        '"Maximum tolerated drawdown"',
        '"Risk aversion"',
        '"Base currency"',
    ]:
        assert label in sidebar


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
    assert 'APP_BUILD_ID = "2026.06.29-bloomberg-zero-cost-terminal-v18"' in source
    assert "persisted_market_intelligence_missing = bool(" in source
    assert "Backfill only missing analytical surfaces" in source
    assert '"Repair missing intelligence"' in source
    assert "partial modules are flagged explicitly" in source
    assert "Validation, fundamentals, risk and audit evidence remain intact" not in source


def test_full_research_contract_rejects_market_only_overlay():
    source = _source()
    start = source.index("def _artifact_has_full_research_contract")
    end = source.index("def _payload_frame", start)
    contract_body = source[start:end]
    assert "strategy_lab = payload.get(\"strategy_lab\", {})" in contract_body
    assert "if _strategy_lab_has_oos_evidence(strategy_lab):" in contract_body
    assert "full_surfaces = [" in contract_body
    assert "return sum(bool(x) for x in full_surfaces) >= 2" in contract_body
    assert "market_intelligence" not in contract_body
    assert "fixed_income_intelligence" not in contract_body
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
    assert "def _preflight_plot_sovereign_curve" in source
    assert "def _curve_10y_2y_from_row" in source
    assert source.index("def _fmt_bps") < source.index("def render_preflight_market(")
    assert source.index("def _curve_10y_2y_from_row") < source.index("def render_preflight_market(")
    assert source.index("def _preflight_plot_sovereign_curve") < source.index("def render_preflight_market(")
    assert "Formal curve construction" in source
    assert "does not manufacture missing yields" in source
    assert "def _fmt_bps" in source


def test_public_seed_dashboard_prevents_empty_hosted_first_paint():
    source = _source()
    assert "No persisted dashboard found" not in source
    assert "PUBLIC_DASHBOARD_ARTIFACT_DIR" in source
    assert "def _public_dashboard_artifact_dirs" in source
    assert "def _latest_public_seed_dashboard_artifacts" in source
    assert "def _seed_dashboard_results" in source
    assert "def _research_artifact_bootstrap_results" in source
    assert "return _seed_dashboard_results(benchmark, seed_bundle)" in source
    assert "return _research_artifact_bootstrap_results(benchmark)" in source
    assert "latest_full_dashboard_payload.seed.json.gz" in source
    assert "latest_daily_dashboard_payload.seed.json.gz" in source
    assert "public_seed_artifact" in source
    assert "def _artifact_has_full_research_contract" in source
    assert "def _artifact_research_score" in source
    assert "def _richest_artifact" in source
    assert "def _strategy_lab_has_oos_evidence" in source
    assert "def _latest_strategy_weights_for_allocation" in source
    assert "def _xcdr_artifacts_to_strategy_lab" in source
    assert "QPK_DASHBOARD_SEED_FIRST" in source
    assert "QPK_DASHBOARD_REMOTE_ON_START" in source
    assert "QPK_LIVE_PREFLIGHT_ON_EMPTY_START" in source
    assert "remote_first or remote_on_start" in source
    assert "First paint must be deterministic and fast" in source
    assert source.count('<div class="qpk-hero">') == 1
    assert source.count("Portfolio decision system") == 1
    assert "repo_lab = _xcdr_artifacts_to_strategy_lab(load_xcdr_research_artifacts())" in source
    assert "def _render_market_intelligence_tape" in source
    assert "Market intelligence tape" in source
    assert "qpk-market-ribbon" in source
    assert "qpk-market-chip" in source
    assert "Compact status only" in source
    assert "repository_research_artifact_fallback" in source
    assert "Institutional artifact not hydrated" in source
    assert "richer_artifact = _richest_artifact(" in source
    assert "2026.06.29-institutional-terminal-full-artifact-v18" in source
    assert "2026.06.29-bloomberg-zero-cost-terminal-v18" in source


def test_public_seed_builder_injects_repo_xcdr_when_cloud_payload_is_thin():
    source = _source()
    script = (PROJECT_ROOT / "scripts" / "build_public_dashboard_seed.py").read_text(encoding="utf-8")
    assert "def _build_xcdr_strategy_lab" in script
    assert "def _publication_completeness" in script
    assert "PUBLIC_DASHBOARD_UI_SCHEMA_VERSION" in script
    assert "PUBLIC_DASHBOARD_BUILD_ID" in script
    assert "def _inject_xcdr_research_if_missing" in script
    assert "public_seed_repo_xcdr_v3" in script
    assert "xcdr_v3_parallel_research_daily_oos.csv" in script
    assert "XCDR/XODR synthetic strategy price" in script
    assert "recommended_portfolio" in script
    assert "Research strategy" in script
    assert "portfolio = _latest_strategy_weights_for_allocation(restored_strategy_lab)" in source

    full_seed = PROJECT_ROOT / "public_artifacts" / "latest_full_dashboard_payload.seed.json.gz"
    daily_seed = PROJECT_ROOT / "public_artifacts" / "latest_daily_dashboard_payload.seed.json.gz"
    assert full_seed.exists()
    assert daily_seed.exists()

    with gzip.open(full_seed, "rt", encoding="utf-8") as fh:
        artifact = json.load(fh)
    payload = artifact.get("dashboard_payload", {})
    strategy_lab = payload.get("strategy_lab", {})
    completeness = payload.get("publication_completeness", {})
    assert artifact.get("public_seed") is True
    assert artifact.get("scope") == "full_analysis"
    assert payload.get("schema_version") == "2026.06.29-institutional-terminal-full-artifact-v18"
    assert payload.get("app_build_id") == "2026.06.29-bloomberg-zero-cost-terminal-v18"
    assert payload.get("contract", {}).get("schema_version") == "2026.06.29-institutional-terminal-full-artifact-v18"
    assert payload.get("contract", {}).get("app_build_id") == "2026.06.29-bloomberg-zero-cost-terminal-v18"
    assert completeness.get("ratio", 0) >= 0.8
    assert completeness.get("checks", {}).get("xcdr_weights") is True
    assert completeness.get("checks", {}).get("global_yield_curves") is True
    assert completeness.get("checks", {}).get("latent_sentiment") is True
    assert strategy_lab.get("generation") == "public_seed_repo_xcdr_v3"
    assert strategy_lab.get("benchmark_xi") == "USMV"
    assert strategy_lab.get("frozen_candidate") == "enhanced_growth_anchor_dd_budget_policy"
    assert len(strategy_lab.get("weights", [])) >= 20
    assert len(strategy_lab.get("oos_price_paths", [])) >= 40
    assert len(payload.get("allocation", {}).get("recommended_portfolio", [])) >= 20
    assert payload.get("fixed_income_intelligence", {}).get("country_metrics")
    market = payload.get("market_intelligence", {})
    assert market.get("sentiment_timeline")
    assert len(market.get("global_yield_curves", [])) >= 10
    assert len(market.get("global_rate_history", [])) >= 500
    assert len(market.get("interbank_reference_rates", [])) >= 100
    assert len(market.get("carry_trade_suggestions", [])) >= 5
    assert len(market.get("geopolitical_articles", [])) >= 10
    assert len(market.get("fundamentals_snapshot", [])) >= 8
    assert payload.get("allocation", {}).get("side_sleeve") is None
    assert "private side alpha" not in json.dumps(artifact).lower()
    assert "sortino optimized synthetic nav" not in json.dumps(artifact).lower()
    assert "service_role" not in json.dumps(artifact).lower()

    with gzip.open(daily_seed, "rt", encoding="utf-8") as fh:
        daily_artifact = json.load(fh)
    daily_payload = daily_artifact.get("dashboard_payload", {})
    assert daily_artifact.get("scope") == "daily_snapshot"
    assert daily_payload.get("strategy_lab", {}).get("oos_price_paths")
    assert daily_payload.get("strategy_lab", {}).get("weights")
    assert daily_payload.get("market_intelligence", {}).get("global_yield_curves")
    assert daily_payload.get("market_intelligence", {}).get("sentiment_timeline")
    assert len(daily_payload.get("research", {}).get("options_summary", [])) >= 1
    assert len(daily_payload.get("research", {}).get("options_chain", [])) >= 100
    assert len(daily_payload.get("charts", {}).get("options_surface", [])) >= 1

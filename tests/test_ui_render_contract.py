from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "stockpicker_app.py"


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
    hydrate = source.index('startup_results = _load_precomputed_dashboard_results(benchmark_ticker)')
    live_request = source.index("live_preflight_requested = (")
    preflight_call = source.index("preflight_market = cached_preflight_market(", live_request)
    assert hydrate < live_request < preflight_call


def test_mobile_layout_stacks_non_metric_columns():
    source = _source()
    assert '[data-testid="stHorizontalBlock"] > [data-testid="column"]:has(div[data-testid="stMetric"])' in source
    assert "flex: 1 1 100% !important;" in source


def test_supabase_json_tables_are_restored_for_rendering():
    source = _source()
    assert "def _payload_frame(value) -> pd.DataFrame:" in source
    assert 'price_paths = _payload_frame(charts.get("price_paths"))' in source
    assert 'portfolio = _payload_frame(allocation.get("recommended_portfolio"))' in source
    assert "payload_requires_restore = any(" in source
    assert 'st.session_state["results"] = restored_results' in source


def test_daily_snapshot_uses_real_metrics_instead_of_empty_full_run_cards():
    source = _source()
    assert 'if is_snapshot:' in source
    assert '"Portfolio return"' in source
    assert '"Active return"' in source
    assert '"Daily CVaR 95%"' in source
    assert '"Market breadth"' in source
    assert '"Full analytics are never inferred from missing snapshot fields."' in source


def test_workspace_change_does_not_trigger_query_parameter_rerun():
    source = _source()
    workspace_start = source.index('_picked_label = st.pills(')
    renderer_start = source.index('# Map slug -> renderer thunk', workspace_start)
    workspace_block = source[workspace_start:renderer_start]
    assert 'st.query_params["section"]' not in workspace_block
    assert "st.experimental_set_query_params" not in workspace_block


def test_snapshot_overview_is_an_analytical_command_center():
    source = _source()
    assert '"Portfolio vs benchmark"' in source
    assert '"Risk-return decomposition"' in source
    assert '"Formal definitions and evidence scope"' in source
    assert 'snapshot_slugs = {"overview", "price-path", "risk", "data-freshness"}' in source
    assert 'initial_sidebar_state="collapsed"' in source

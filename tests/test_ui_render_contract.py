from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "stockpicker_app.py"


def _source() -> str:
    return APP_SOURCE.read_text(encoding="utf-8")


def test_workspace_renders_only_active_section():
    source = _source()
    assert "section = st.tabs(" not in source
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

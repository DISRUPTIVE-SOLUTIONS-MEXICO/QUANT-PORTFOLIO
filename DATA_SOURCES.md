# Data Sources — Zero-Cost Catalog

Every input to Quant Portfolio-Kaizen is free. Robustness comes from
**redundancy + cross-validation + provenance**, not from paying a vendor:
overlapping sources are reconciled (`quant_core/data/reconcile.py`), every
fetch records a content hash (`quant_core/data/provenance.py`), and gaps or
discrepancies surface in the Data Freshness / Data Provenance section of the
app instead of silently propagating into research.

## Prices and volumes

| Source | Role | Endpoint | Cost / license notes |
|---|---|---|---|
| Yahoo Finance (yfinance) | Primary daily OHLCV, fundamentals, options | yfinance library | Free; Yahoo ToS technically limits to personal use — disclosed honestly for any due diligence |
| Stooq | Redundancy check + partial **delisted** coverage + long history backfill | `https://stooq.com/q/d/l/?s={sym}.us&i=d` | Free EOD CSVs; research use. Endpoint availability varies by network/region — the chain degrades gracefully |
| Tiingo | Third redundancy leg (adjusted closes) | `api.tiingo.com/tiingo/daily/...` | Free EOD tier with registered token (`TIINGO_TOKEN`); no-ops without it |

Fallback chain: `download_prices(..., fallback_chain=("yfinance", "stooq", "tiingo"))`.
Cross-validation: `reconcile_price_frames` flags tickers where >1% of
overlapping closes differ by >50 bps. Ingest anomarly scan:
`quant_core/data/quality.py` (8-sigma jumps, stale feeds, calendar gaps).

## Point-in-time universe and survivorship control

| Source | Role | Notes |
|---|---|---|
| Wikipedia S&P 500 page | Current constituents + historical changes table (~2000→today) | `load_sp500_wikipedia_asof` |
| Wayback Machine availability API | PIT membership for as-of dates beyond changes coverage | Free, no key (`quant_core/data/universe.py`) |
| NasdaqTrader symbol directory | Current listings | Free FTP/HTTP files |
| Delisting registry | `data/universes/delisted_registry.parquet` built by `scripts/build_delisted_registry.py` | Wiki removals × current listings × Stooq recoverability |

The backtest applies `RunConfig.delisting_return_assumption` (default −30%,
cf. Shumway 1997) when a held ticker shows a stale price **and** zero volume
across a holding window. Survivorship inflation is bounded explicitly by
`scripts/survivorship_sensitivity.py` (PIT vs current-constituents batch).

## Fundamentals (point-in-time)

| Source | Role | Notes |
|---|---|---|
| SEC EDGAR companyfacts | Primary PIT fundamentals (filing-dated) | Public domain; respectful User-Agent required |
| SEC EDGAR filings (10-K/10-Q text) | NLP risk features | Public domain |
| Yahoo Finance | Secondary fundamentals | Reconciled against SEC with confidence scoring |

## Macro and rates — Mexico (free tokens)

| Source | Series | Token env |
|---|---|---|
| Banxico SIE | Tasa objetivo, TIIE 28/91/182, CETES 28–364, USD/MXN FIX, UDIS, INPC | `BANXICO_TOKEN` (free) |
| INEGI BIE | IGAE, inflación quincenal, desocupación | `INEGI_TOKEN` (free) |

Module: `quant_core/data/macro_mx.py`. Without a token the provider returns
an empty frame and the freshness report shows the gap (honest degradation).

## Macro and rates — global (no keys)

| Source | Series | Module |
|---|---|---|
| FRED public CSV | US + OECD country rates, macro panel | `quant_core/data/macro_global.py` |
| US Treasury fiscal data | Daily yield curve | core (`fetch_us_treasury_yield_curve`) |
| ECB SDW | Euro-area yield curve | `macro_global.fetch_ecb_yield_curve` |
| Bank of Canada Valet | Policy + 2Y/10Y | `macro_global.fetch_bank_of_canada_rates` |
| BCB SGS (Brazil) | SELIC | `macro_global.fetch_bcb_sgs_rates` |
| Bank of England IADB | Bank Rate et al. | `macro_global.fetch_boe_iadb_series` |
| SNB data portal | Swiss policy rate cubes | `macro_global.fetch_snb_series` |
| World Bank API | Annual macro fallback | `macro_global.fetch_worldbank_indicator` |
| IMF SDMX | Quarterly/monthly macro fallback | `macro_global.fetch_imf_sdmx_series` |

## News / event risk

| Source | Role |
|---|---|
| GDELT 2.0 | Geopolitical timeline + articles |
| Google News RSS | Fallback articles |
| ForexFactory / FairEconomy | Macro event calendar |

## Governed scraping (official sources without APIs)

`quant_core/data/scraping.py` — robots.txt checked per domain, per-domain
throttling, exponential backoff with jitter, and the raw HTML snapshot is
persisted hash-keyed so every parsed dataset traces back to exact bytes.
Current parsers: Banxico monetary-policy announcements (fixture-validated).

## OCR (last resort)

`quant_core/data/ocr.py` — extraction chain `pdfplumber` (born-digital) →
`pytesseract` + OpenCV (image-only). Optional dependencies in
`requirements-ocr.txt`; never loaded by the Streamlit runtime. Every OCR
table must pass `validate_ocr_frame` (range/date plausibility) before it can
enter the cache.

## Caching and provenance

- `PersistentCache` (`quant_core/data/base.py`): hash-keyed parquet+json with
  TTL per namespace (see `quant_core/data_freshness.py` for TTLs).
- `Provenance` records: source, URL, retrieval timestamp, content sha256,
  license note.
- The app's **Data Freshness** section shows freshness, cache inventory,
  source catalog and the anomaly scan.
